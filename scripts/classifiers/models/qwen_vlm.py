"""
Qwen2.5-VL model wrapper for robot state classification.

This module provides a clean interface for using Qwen2.5-VL for:
- Zero-shot robot state classification
- Few-shot learning with example images
- Batch prediction

Usage:
    from models.qwen_vlm import Qwen2_5Model
    
    model = Qwen2_5Model(cache_dir="/path/to/cache")
    result = model.predict("image.png", "What is the robot doing?")
"""

import os
import gc
import base64
from typing import List, Dict, Union, Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Add qwen-vl-utils to path if available
import sys
QWEN_UTILS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "flow_label", "v2label", "qwen-vl-utils", "src")
if os.path.exists(QWEN_UTILS_PATH):
    sys.path.append(QWEN_UTILS_PATH)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen_vl_utils not found. Please install it:\n"
        "git clone https://github.com/QwenLM/Qwen2-VL.git\n"
        "pip install -e Qwen2-VL/qwen-vl-utils"
    )


class Qwen2_5Model:
    """Qwen2.5-VL model wrapper for robot state classification."""
    
    # Robot state labels
    LABELS = {
        "i": "initialize",
        "n": "near",
        "t": "attempt to grab",
        "c": "pushing",
        "s": "holding",
        "g": "goal state",
    }
    
    DEFAULT_SYSTEM_MESSAGE = (
        "You are an expert robot observer. "
        "You will be shown an image and must respond with "
        "**exactly one** of these labels: i, n, t, c, s, g. "
        "i=initialize, n=near, t=attempt to grab, c=pushing, s=holding, g=goal state. "
        "Do **not** add any other text."
    )
    
    DEFAULT_PROMPT = "Label the robot arm state in the image. Answer with a single character."
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir: str = "/mnt/l1/huggingface",
        max_memory: Optional[Dict[int, str]] = None,
        hf_token: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize Qwen2.5-VL model.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model weights
            max_memory: GPU memory allocation, e.g., {0: "24GB", 1: "24GB"}
            hf_token: HuggingFace API token
            device: Device to use for inference
        """
        self.device = device
        self.max_memory = max_memory or {0: "24GB"}
        
        print(f"Loading Qwen2.5-VL from {model_name}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=self.max_memory,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
        
        print("Model loaded successfully.")
    
    def load_image_as_base64(self, image_path: str) -> str:
        """Load an image file and encode as base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def predict(
        self,
        image_path: str,
        prompt: str = None,
        system_message: str = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Predict robot state from a single image.
        
        Args:
            image_path: Path to input image
            prompt: Question/prompt for the model
            system_message: System message to guide behavior
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's prediction as string
        """
        self.model.eval()
        
        prompt = prompt or self.DEFAULT_PROMPT
        system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        
        img_b64 = self.load_image_as_base64(image_path)
        
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output[0] if output else ""
    
    def predict_with_examples(
        self,
        example_images: List[Dict[str, str]],
        target_image: str,
        prompt: str = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Few-shot prediction with example images (zero-shot with in-context examples).
        
        Args:
            example_images: List of {"image_path": str, "label": str} or 
                           {"image_path": str, "text": str} dicts
            target_image: Path to image to classify
            prompt: Question for target image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's prediction
        """
        self.model.eval()
        
        prompt = prompt or self.DEFAULT_PROMPT
        
        # Load example images and build content
        examples_content = []
        for ex in example_images:
            img_b64 = self.load_image_as_base64(ex["image_path"])
            examples_content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{img_b64}"
            })
            # Support both "label" and "text" keys
            label_text = ex.get("text") or f"Label: {ex.get('label', '?')}"
            examples_content.append({
                "type": "text",
                "text": label_text
            })
        
        # Load target image
        target_b64 = self.load_image_as_base64(target_image)
        
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": examples_content},
            {"role": "assistant", "content": "I understand the examples."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{target_b64}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # Clean up
        del generated_ids, inputs, generated_ids_trimmed, image_inputs
        gc.collect()
        
        return output[0] if output else ""
    
    def predict_v2(
        self,
        example_images: List[Dict[str, str]],
        target_image: str,
        prompt: str = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Alias for predict_with_examples for compatibility.
        
        This matches the original qwen2_5_model.py interface.
        """
        return self.predict_with_examples(example_images, target_image, prompt, max_new_tokens)
    
    def predict_from_base64(
        self,
        image_base64: str,
        prompt: str = None,
        max_new_tokens: int = 4,
    ) -> str:
        """
        Predict from base64-encoded image (for API use).
        
        Args:
            image_base64: Base64-encoded image string
            prompt: Question for the model
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Single character label prediction
        """
        self.model.eval()
        
        prompt = prompt or self.DEFAULT_PROMPT
        
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to GPU and convert to fp16
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
                if inputs[k].dtype.is_floating_point:
                    inputs[k] = inputs[k].half()
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output[0] if output else ""


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL model")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--cache_dir", type=str, default="/mnt/l1/huggingface")
    args = parser.parse_args()
    
    model = Qwen2_5Model(cache_dir=args.cache_dir)
    result = model.predict(args.image)
    print(f"Prediction: {result}")

