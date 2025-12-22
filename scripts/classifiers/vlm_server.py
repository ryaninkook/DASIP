#!/usr/bin/env python3
"""
FastAPI server for VLM-based robot state classification.

This server provides an API endpoint for real-time robot state classification
using a finetuned Qwen2.5-VL model. It automatically selects the best model
version for each environment.

Usage:
    # Start server for a specific environment
    python vlm_server.py --env square --port 8236
    
    # Start with custom model path
    python vlm_server.py --env can --model_path /path/to/model --port 8237

API Endpoints:
    POST /predict - Classify robot state from image
        - image_file: Image file upload
        - prompt: Optional custom prompt
        Returns: {"prediction": "i", "status": 200}
    
    GET / - Health check
"""

import argparse
import base64
import os
import random

import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, set_seed
from peft import PeftModel, prepare_model_for_kbit_training

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    raise ImportError("Please install transformers>=4.37.0")

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "flow_label", "v2label", "qwen-vl-utils", "src"))
    from qwen_vl_utils import process_vision_info

from fastapi import FastAPI, File, UploadFile, Form
import uvicorn


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Best model version per environment (from evaluation)
# Update these based on your finetuning results
BEST_MODELS = {
    "can": "version_1",
    "lift": "version_3",
    "square": "version_2",
    "tool_hang": "version_2",
    "transport": "version_3",
    "pusht": "version_2",
}

# System message for inference
SYSTEM_MESSAGE = (
    "You are an expert robot observer. "
    "You will be shown an image and must respond with "
    "*exactly one* of these labels: i, n, t, c, s, g. "
    "i means the robot is in the initial position. "
    "n means the robot is near the object. "
    "t means the robot is attempting to grab the object. "
    "c means the robot is pushing the object. "
    "s means the robot is holding the object. "
    "g means the robot is in the goal state."
)

DEFAULT_PROMPT = (
    "Label the robot in the image whatever you think closest. "
    "Do not explain answer only give one character."
)


# =============================================================================
# Model Loading
# =============================================================================

class VLMClassifier:
    """VLM-based classifier for robot state."""
    
    def __init__(
        self,
        env: str,
        base_model_dir: str = "vlm_finetuned",
        cache_dir: str = "/mnt/l1/huggingface",
        use_8bit: bool = True,
        max_memory: dict = None,
        hf_token: str = None,
        device: str = "cuda:0",
    ):
        """
        Initialize VLM classifier.
        
        Args:
            env: Environment name (for model selection)
            base_model_dir: Directory containing finetuned models
            cache_dir: HuggingFace cache directory
            use_8bit: Use 8-bit quantization
            max_memory: GPU memory allocation
            hf_token: HuggingFace token
            device: Device for inference
        """
        self.env = env
        self.device = device
        max_memory = max_memory or {0: "24GB"}
        
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        
        # Quantization config
        quant_config = None
        if use_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4",
            )
        
        # Load base model
        print(f"Loading base model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            max_memory=max_memory,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            token=hf_token,
        )
        
        if use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Load finetuned weights
        model_version = BEST_MODELS.get(env, "version_1")
        finetune_path = os.path.join(base_model_dir, model_version)
        
        if os.path.exists(finetune_path):
            print(f"Loading finetuned model for {env}: {finetune_path}")
            self.model = PeftModel.from_pretrained(
                self.model, finetune_path, torch_dtype=torch.float16
            )
        else:
            print(f"Warning: Finetuned model not found at {finetune_path}")
            print("Using base model (zero-shot)")
        
        torch.cuda.empty_cache()
        print("Model loaded successfully.")
    
    def predict(self, image_base64: str, prompt: str = None) -> str:
        """
        Predict robot state from base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image
            prompt: Optional custom prompt
            
        Returns:
            Predicted label (single character)
        """
        self.model.eval()
        
        prompt = prompt or DEFAULT_PROMPT
        
        # Build conversation
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
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
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device and convert types
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
                if inputs[k].dtype.is_floating_point:
                    inputs[k] = inputs[k].half()
        
        # Ensure visual module is in fp16
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "to"):
            self.model.visual = self.model.visual.half()
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=4)
        
        output_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        
        output = self.processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output[0] if output else ""


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app(classifier: VLMClassifier) -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="Robot State Classifier API",
        description="VLM-based robot state classification",
        version="1.0.0",
    )
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "message": "Robot State Classifier API is running",
            "environment": classifier.env,
        }
    
    @app.post("/predict")
    async def predict(
        image_file: UploadFile = File(...),
        prompt: str = Form(None),
    ):
        """
        Classify robot state from uploaded image.
        
        Args:
            image_file: Image file to classify
            prompt: Optional custom prompt
            
        Returns:
            {"prediction": str, "status": int}
        """
        try:
            # Read and encode image
            content = await image_file.read()
            image_b64 = base64.b64encode(content).decode("utf-8")
            
            # Get prediction
            prediction = classifier.predict(image_b64, prompt)
            
            # Normalize prediction
            if isinstance(prediction, list):
                prediction = prediction[0]
            
            print(f"Prediction: {prediction}")
            
            return {"prediction": prediction, "status": 200}
            
        except Exception as e:
            return {"error": str(e), "status": 500}
    
    return app


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Start VLM classification server"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default="square",
        choices=["can", "lift", "square", "tool_hang", "transport", "pusht"],
        help="Environment (for model selection)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="vlm_finetuned",
        help="Path to finetuned model directory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/l1/huggingface",
        help="HuggingFace cache directory",
    )
    parser.add_argument("--port", type=int, default=8236)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize classifier
    classifier = VLMClassifier(
        env=args.env,
        base_model_dir=args.model_path,
        cache_dir=args.cache_dir,
    )
    
    # Create and run app
    app = create_app(classifier)
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Environment: {args.env}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

