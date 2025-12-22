"""
LLaVA model wrappers for robot state classification.

This module provides clean interfaces for using LLaVA models:
- LLaVA 1.5 (7B)
- LLaVA-NeXT (Mistral 7B)

Usage:
    from models.llava_vlm import LlavaModel, LlavaNextModel
    
    model = LlavaModel(cache_dir="/path/to/cache")
    result = model.predict("image.png", "What is the robot doing?")
"""

import torch
from typing import List, Dict, Union, Optional
from PIL import Image

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)


class LlavaModel:
    """LLaVA 1.5 model wrapper for robot state classification."""
    
    DEFAULT_PROMPT = "Label the robot arm state in the image."
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        cache_dir: str = "/mnt/l1/huggingface",
        max_memory: Optional[Dict[int, str]] = None,
        hf_token: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize LLaVA 1.5 model.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model weights
            max_memory: GPU memory allocation
            hf_token: HuggingFace API token
            device: Device for inference
        """
        self.device = device
        self.max_memory = max_memory or {0: "24GB", 1: "24GB"}
        
        print(f"Loading LLaVA from {model_name}...")
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            max_memory=self.max_memory,
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
    
    def load_image(
        self,
        image_path: Union[str, List[str]],
        target_size: tuple = (224, 224),
    ) -> List[Image.Image]:
        """Load and preprocess image(s)."""
        if isinstance(image_path, str):
            image_path = [image_path]
        
        images = []
        for path in image_path:
            img = Image.open(path).convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
        
        return images
    
    def predict(
        self,
        image_path: str,
        prompt: str = None,
        max_new_tokens: int = 30,
    ) -> str:
        """
        Predict robot state from a single image.
        
        Args:
            image_path: Path to input image
            prompt: Question/prompt for the model
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's prediction as string
        """
        self.model.eval()
        
        prompt = prompt or self.DEFAULT_PROMPT
        images = self.load_image(image_path)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        output = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        
        # Extract assistant response
        for i, response in enumerate(output):
            if "ASSISTANT:" in response:
                output[i] = response.split("ASSISTANT:")[1].strip()
        
        return output[0] if output else ""
    
    def predict_multiple_images(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Predict using multiple images (few-shot).
        
        Args:
            image_paths: List of image paths
            prompt: Question/prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's prediction
        """
        self.model.eval()
        
        images = self.load_image(image_paths)
        
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        output = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        
        for i, response in enumerate(output):
            if "ASSISTANT:" in response:
                output[i] = response.split("ASSISTANT:")[1].strip()
        
        return output[0] if output else ""


class LlavaNextModel:
    """LLaVA-NeXT (Mistral 7B) model wrapper for robot state classification."""
    
    DEFAULT_SYSTEM_MESSAGE = (
        "You are an expert robot observer. "
        "You will be shown an image and must respond with "
        "**exactly one** of these labels: initialize, near, attempt to grab, pushing, holding, goal state. "
        "Do **not** add any other text."
    )
    
    DEFAULT_PROMPT = "Label the robot arm state in the image."
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        cache_dir: str = "/mnt/l1/huggingface",
        max_memory: Optional[Dict[int, str]] = None,
        hf_token: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize LLaVA-NeXT model.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model weights
            max_memory: GPU memory allocation
            hf_token: HuggingFace API token
            device: Device for inference
        """
        self.device = device
        self.max_memory = max_memory or {0: "24GB", 1: "24GB"}
        
        print(f"Loading LLaVA-NeXT from {model_name}...")
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=hf_token,
        ).eval()
        
        self.processor = LlavaNextProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
        
        print("Model loaded successfully.")
    
    def load_image(
        self,
        image_path: Union[str, List[str]],
    ) -> List[Image.Image]:
        """Load image(s) without resizing."""
        if isinstance(image_path, str):
            image_path = [image_path]
        
        images = []
        for path in image_path:
            img = Image.open(path).convert("RGB")
            images.append(img)
        
        return images
    
    def predict(
        self,
        image_path: str,
        prompt: str = None,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Predict robot state from a single image.
        
        Args:
            image_path: Path to input image
            prompt: Question/prompt for the model
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's prediction as string
        """
        self.model.eval()
        
        prompt = prompt or self.DEFAULT_PROMPT
        images = self.load_image(image_path)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images, text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        output = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response after [/INST] tag
        if "[/INST]" in output:
            output = output.split("[/INST]")[-1].strip()
        
        return output
    
    def predict_with_examples(
        self,
        example_images: List[Dict[str, str]],
        target_image: str,
        prompt: str = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Few-shot prediction with example images.
        
        Args:
            example_images: List of {"image_path": str, "label": str} dicts
            target_image: Path to image to classify
            prompt: Question for target image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's prediction
        """
        self.model.eval()
        
        prompt = prompt or self.DEFAULT_PROMPT
        
        # Build conversation with examples
        examples_content = []
        for ex in example_images:
            examples_content.append({"type": "image", "image": ex["image_path"]})
            examples_content.append({"type": "text", "text": ex["label"]})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.DEFAULT_SYSTEM_MESSAGE}]},
            {"role": "user", "content": examples_content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": target_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        output = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        
        if isinstance(output, list):
            output = [d.split("[/INST]")[-1].strip() for d in output]
            return output[0]
        
        return output.split("[/INST]")[-1].strip()


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLaVA models")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--model", type=str, choices=["llava", "llava-next"], default="llava")
    parser.add_argument("--cache_dir", type=str, default="/mnt/l1/huggingface")
    args = parser.parse_args()
    
    if args.model == "llava":
        model = LlavaModel(cache_dir=args.cache_dir)
    else:
        model = LlavaNextModel(cache_dir=args.cache_dir)
    
    result = model.predict(args.image)
    print(f"Prediction: {result}")

