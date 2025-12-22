#!/usr/bin/env python3
"""
Microsoft Phi-4 Vision-Language Model Wrapper.

Original: flow_label/v2label/microsoft_phi_4_vlm.py
See ../MODIFICATIONS.md for changes made.

Changes from original:
- Removed hardcoded HuggingFace token (use HF_TOKEN env var)
- Made cache_dir and device configurable
"""

import os
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Model config
MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"

# System message for classification
SYSTEM_MESSAGE = (
    "You are an expert robot observer. "
    "You will be shown a image and must respond with "
    "**exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state. "
    "Do **not** add any other text."
)


class PHI_4_Model:
    """Microsoft Phi-4 VLM for robot state classification."""
    
    def __init__(
        self,
        device: str = None,
        cache_dir: str = None,
        hf_token: str = None,
        max_memory: dict = None,
    ):
        """
        Initialize Phi-4 model.
        
        Args:
            device: Device to use (default: cuda:0)
            cache_dir: HuggingFace cache directory
            hf_token: HuggingFace token (defaults to HF_TOKEN env var)
            max_memory: GPU memory config per device
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        token = hf_token or os.environ.get("HF_TOKEN")
        
        if max_memory is None:
            max_memory = {0: "24GB", 1: "24GB"}
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            max_memory=max_memory,
            cache_dir=cache_dir,
            torch_dtype="auto",
            trust_remote_code=True,
            token=token,
            _attn_implementation="flash_attention_2",
        )
        
        self.generation_config = GenerationConfig.from_pretrained(
            MODEL_PATH,
            cache_dir=cache_dir,
        )

    def clean_response(self, response: str, instruction_keywords: List[str]) -> str:
        """Remove prompt text from response."""
        for keyword in instruction_keywords:
            if response.lower().startswith(keyword.lower()):
                response = response[len(keyword):].strip()
        return response

    def predict(self, image_path: str, question: str) -> str:
        """Single image prediction."""
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        
        prompt = f"{user_prompt}<|image_1|>{question}{prompt_suffix}{assistant_prompt}"
        image = Image.open(image_path)
        
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
        
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True
        )[0]
        
        return self.clean_response(response, [question])

    def predict_multiple_images(self, files: List[str], question: str) -> str:
        """Multiple images in one prompt."""
        user_p = "<|user|>"
        assistant_p = "<|assistant|>"
        end_p = "<|end|>"
        
        prompt = user_p
        images = []
        for idx, fpath in enumerate(files, start=1):
            prompt += f"<|image_{idx}|>"
            images.append(Image.open(fpath))
        prompt += f"{question}{end_p}{assistant_p}"
        
        inputs = self.processor(
            text=prompt, images=images, return_tensors="pt"
        ).to(self.device)
        
        input_len = inputs["input_ids"].shape[-1]
        max_new = 50
        num_logits_to_keep = input_len + max_new
        
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_logits_to_keep=num_logits_to_keep,
        )
        
        raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return self.clean_response(raw, [question])

    def predict_v2(
        self,
        samples: List[dict],
        target_image: str,
        prompt: str,
        use_prompt: bool = True,
    ) -> str:
        """
        Few-shot prediction - samples shown as examples.
        
        Args:
            samples: List of {"image_path": str, "text": str}
            target_image: Path to image to classify
            prompt: Classification prompt
            use_prompt: Whether to include text with each sample
        """
        user_p = "<|user|>"
        assistant_p = "<|assistant|>"
        system_p = "<|system|>"
        end_p = "<|end|>"
        
        message = f"{system_p}{SYSTEM_MESSAGE}{end_p}"
        images = []
        
        # Add sample images
        for idx, sample in enumerate(samples, start=1):
            if use_prompt:
                message += f"{user_p}<|image_{idx}|>{sample['text']}{end_p}"
            else:
                message += f"{user_p}<|image_{idx}|>{end_p}"
            images.append(Image.open(sample["image_path"]))
        
        # Add target image
        target_idx = len(samples) + 1
        if use_prompt:
            message += f"{user_p}<|image_{target_idx}|>{prompt}{end_p}{assistant_p}"
        else:
            message += f"{user_p}<|image_{target_idx}|>{end_p}{assistant_p}"
        images.append(Image.open(target_image))
        
        inputs = self.processor(
            text=message, images=images, return_tensors="pt"
        ).to(self.device)
        
        input_len = inputs["input_ids"].shape[-1]
        max_new = 50
        num_logits_to_keep = input_len + max_new
        
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_logits_to_keep=num_logits_to_keep,
        )
        
        raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return self.clean_response(raw, [prompt])

    def predict_v3(
        self,
        samples: List[dict],
        target_image: str,
        prompt: str,
        use_prompt: bool = True,
    ) -> str:
        """
        Few-shot prediction with assistant responses for each sample.
        
        Each sample gets a user message and assistant response,
        creating a multi-turn conversation.
        """
        user_p = "<|user|>"
        assistant_p = "<|assistant|>"
        system_p = "<|system|>"
        end_p = "<|end|>"
        
        message = f"{system_p}{SYSTEM_MESSAGE}{end_p}"
        images = []
        
        # Add samples with assistant responses
        for idx, sample in enumerate(samples, start=1):
            if use_prompt:
                message += f"{user_p}<|image_{idx}|>{sample['text']}{end_p}"
            else:
                message += f"{user_p}<|image_{idx}|>{end_p}"
            images.append(Image.open(sample["image_path"]))
            message += f"{assistant_p}{sample['text']}{end_p}"
        
        # Add target image
        target_idx = len(samples) + 1
        if use_prompt:
            message += f"{user_p}<|image_{target_idx}|>{prompt}{end_p}{assistant_p}"
        else:
            message += f"{user_p}<|image_{target_idx}|>{end_p}{assistant_p}"
        images.append(Image.open(target_image))
        
        inputs = self.processor(
            text=message, images=images, return_tensors="pt"
        ).to(self.device)
        
        input_len = inputs["input_ids"].shape[-1]
        max_new = 50
        num_logits_to_keep = input_len + max_new
        
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_logits_to_keep=num_logits_to_keep,
        )
        
        raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return self.clean_response(raw, [prompt])
