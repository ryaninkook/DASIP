#!/usr/bin/env python3
"""
Gemma3 Vision-Language Model Wrapper.

Original: flow_label/v2label/gemma3_model.py
See ../MODIFICATIONS.md for changes made.

Changes from original:
- Removed hardcoded HuggingFace token (use HF_TOKEN env var)
- Made cache_dir configurable
- Made device configurable
"""

import os
from typing import List

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Model config
MODEL_NAME = "google/gemma-3-4b-it"
LORA_RANK = 16


class Gemma3Model:
    """Gemma3 VLM for robot state classification."""
    
    def __init__(
        self,
        device_idx: int = 0,
        cache_dir: str = None,
        hf_token: str = None,
    ):
        """
        Initialize Gemma3 model.
        
        Args:
            device_idx: CUDA device index
            cache_dir: HuggingFace cache directory
            hf_token: HuggingFace token (defaults to HF_TOKEN env var)
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get token from env if not provided
        token = hf_token or os.environ.get("HF_TOKEN")
        
        # Quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # LoRA config
        peft_config = LoraConfig(
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )
        
        # Device map
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            token=token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, peft_config)
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.config.text_config.use_cache = False

    def load_local_image(self, img_path, target_size=(224, 224)):
        """Load and resize a local image."""
        if img_path is None:
            raise ValueError("Image path cannot be None")
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            return img

    def predict(self, image_path: str, question: str) -> str:
        """Single image prediction."""
        self.model.eval()
        model_device = next(self.model.parameters()).device
        
        system_message = (
            "You are an expert robomimic simulation robot observer. "
            "You will be shown a image and must respond with "
            "**exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state. "
            "Do **not** add any other text."
        )
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image_url": None}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, return_tensors="pt")
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(model_device, dtype=torch.float16)
                else:
                    inputs[k] = v.to(model_device)
        
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        if gen_ids.shape[1] <= prompt_len:
            return ""
        
        assistant_tokens = gen_ids[0, prompt_len:]
        reply = self.tokenizer.decode(assistant_tokens, skip_special_tokens=True).strip()
        
        if reply.lower().startswith("assistant"):
            reply = reply.split('\n', 1)[1].strip() if "\n" in reply else reply
        
        return reply

    def predict_v2(
        self,
        sample_images_with_prompts: List[dict],
        target_image: str,
        prompt: str
    ) -> str:
        """
        Few-shot prediction with samples in single message.
        
        Args:
            sample_images_with_prompts: List of {"image_path": str, "text": str}
            target_image: Path to image to classify
            prompt: Classification prompt
        """
        self.model.eval()
        model_device = next(self.model.parameters()).device
        
        system_message = (
            "You are an expert robot observer. "
            "You will be shown a image and must respond with "
            "**exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state. "
            "Do **not** add any other text."
        )
        
        # Build conversation
        samples = []
        for img_data in sample_images_with_prompts:
            samples.append({"type": "image", "image_url": img_data['image_path']})
            samples.append({"type": "text", "text": img_data["text"]})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": samples},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": target_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt_text, return_tensors="pt")
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(model_device, dtype=torch.float16)
                else:
                    inputs[k] = v.to(model_device)
        
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        if gen_ids.shape[1] <= prompt_len:
            return ""
        
        assistant_tokens = gen_ids[0, prompt_len:]
        reply = self.tokenizer.decode(assistant_tokens, skip_special_tokens=True).strip()
        
        if reply.lower().startswith("assistant"):
            reply = reply.split('\n', 1)[1].strip() if "\n" in reply else reply
        
        return reply

    def predict_v3(
        self,
        sample_images_with_prompts: List[dict],
        target_image: str,
        prompt: str
    ) -> str:
        """
        Few-shot prediction with multi-turn conversation.
        
        Each sample becomes a user/assistant exchange.
        """
        self.model.eval()
        model_device = next(self.model.parameters()).device
        
        system_message = (
            "You are an expert robomimic simulation robot observer. "
            "You will be shown a image and must respond with "
            "**exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state. "
            "Do **not** add any other text."
        )
        
        # Build multi-turn conversation
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]}
        ]
        
        for img_data in sample_images_with_prompts:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "image", "image_url": img_data['image_path']},
                    {"type": "text", "text": img_data["text"]}
                ]
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": img_data["text"]}]
            })
        
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image_url": target_image},
                {"type": "text", "text": prompt}
            ]
        })
        
        prompt_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt_text, return_tensors="pt")
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(model_device, dtype=torch.float16)
                else:
                    inputs[k] = v.to(model_device)
        
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        if gen_ids.shape[1] <= prompt_len:
            return ""
        
        assistant_tokens = gen_ids[0, prompt_len:]
        reply = self.tokenizer.decode(assistant_tokens, skip_special_tokens=True).strip()
        
        if reply.lower().startswith("assistant"):
            reply = reply.split('\n', 1)[1].strip() if "\n" in reply else reply
        
        return reply
