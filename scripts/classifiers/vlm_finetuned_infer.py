#!/usr/bin/env python3
"""
VLM Inference with Fine-Tuned Model.

Uses a LoRA fine-tuned Qwen2.5-VL model to classify robot states.

Original: flow_label/v2label/a8_2_use_finetune_model_zeeshan_qwen.py
See MODIFICATIONS.md for changes made.

Usage:
    python vlm_finetuned_infer.py --env lift --model_path ../../flow_label/qwen_finetuned/version_1
    python vlm_finetuned_infer.py --env can --num_samples 20
"""

import argparse
import base64
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)
from qwen_vl_utils import process_vision_info

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Default paths (relative to this script)
DEFAULT_DATA_PATH = "../../flow_label/data/good_4_user_filtered.csv"
DEFAULT_IMAGE_DIR = "../../SIP/data"
DEFAULT_MODEL_PATH = "../../flow_label/qwen_finetuned/version_1"

# All environments
ALL_ENVS = ["can", "lift", "square", "tool_hang", "transport", "pusht"]

# System message for the model
SYSTEM_MESSAGE = (
    "You are an expert robot observer. "
    "You will be shown a image and must respond with "
    "**exactly one** of these labels: i, n, t, c, s, g. "
    "i means the robot is in the initial position. "
    "n means the robot is near the object. "
    "t means the robot is attempting to grab the object. "
    "c means the robot is pushing the object. "
    "s means the robot is holding the object. "
    "g means the robot is in the goal state."
)

PROMPT = "Label the robot in the image whatever you think closest. Do not explain answer only give one character."


# =============================================================================
# Helper Functions
# =============================================================================

def load_local_image(img_path: str) -> str:
    """Load image as base64 string."""
    if not img_path:
        raise ValueError("Image path cannot be None")
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def add_image_path(df, base_path):
    """Add image paths to dataframe."""
    images_paths = []
    for _, row in df.iterrows():
        image_path = os.path.join(
            base_path,
            f"{row['env_choice']}/images/",
            f"episode_{int(row['episode_idx'])}/",
            f"image_{int(row['timestep_idx'])}.png"
        )
        if os.path.exists(image_path):
            images_paths.append(image_path)
        else:
            images_paths.append(None)
    df["image_path"] = images_paths
    return df


def sample_data(df, num_rows: int = 100):
    """Sample data balanced by environment and interaction type."""
    df = df.sort_values(["env_choice", "episode_idx", "timestep_idx"])
    df = (
        df.groupby(["env_choice", "interaction_type"])
        .apply(lambda x: x.head(num_rows))
        .reset_index(drop=True)
    )
    return df


def normalize_prediction(y_hat):
    """Normalize VLM output to single character label."""
    if isinstance(y_hat, list):
        y_hat = y_hat[0]
    if "Label: " in y_hat:
        y_hat = y_hat.split("Label: ")[1].strip()
    
    y_hat_lower = y_hat.lower()
    if "near" in y_hat_lower:
        return "n"
    elif "initialize" in y_hat_lower:
        return "i"
    elif "attempt to grab" in y_hat_lower:
        return "t"
    elif "pushing" in y_hat_lower:
        return "c"
    elif "hanger" in y_hat_lower:
        return "s"
    elif "goal state" in y_hat_lower:
        return "g"
    return y_hat


# =============================================================================
# Model Loading
# =============================================================================

def load_finetuned_model(model_path, use_8bit=True):
    """Load fine-tuned Qwen model with LoRA weights."""
    print(f"Loading fine-tuned model from: {model_path}")
    
    # Use HF_TOKEN from environment if available
    hf_token = os.environ.get("HF_TOKEN", None)
    
    # GPU memory config
    max_memory = {0: "24GB"}
    
    # Quantization config
    quantization_config = None
    if use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
        )
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        max_memory=max_memory,
        trust_remote_code=True,
        token=hf_token,
    )
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
        token=hf_token,
    )
    
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.float16)
    model.print_trainable_parameters()
    
    return model, processor


# =============================================================================
# Prediction
# =============================================================================

def predict(image_path, model, processor):
    """Run prediction on a single image."""
    model.eval()
    img = load_local_image(image_path)
    
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{img}"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to GPU and convert to fp16
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda:0")
            if v.dtype.is_floating_point:
                inputs[k] = inputs[k].half()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    return output_text[0]


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_env(model, processor, df, env_choice):
    """Evaluate fine-tuned model on one environment."""
    print(f"\n{'='*60}")
    print(f"Environment: {env_choice}")
    print("=" * 60)
    
    env_df = df[df["env_choice"] == env_choice].reset_index(drop=True)
    
    if len(env_df) == 0:
        print(f"No data for {env_choice}")
        return None
    
    print(f"Testing on {len(env_df)} samples")
    
    y_hats = []
    y_trues = []
    times = []
    
    for idx in range(len(env_df)):
        image_path = env_df.iloc[idx]["image_path"]
        y_true = env_df.iloc[idx]["interaction_type"]
        
        start = time.time()
        y_hat = predict(image_path, model, processor)
        elapsed = time.time() - start
        
        y_hat = normalize_prediction(y_hat)
        
        y_hats.append(y_hat)
        y_trues.append(y_true)
        times.append(elapsed)
        
        print(f"  [{idx+1}/{len(env_df)}] true={y_true} pred={y_hat} ({elapsed:.2f}s)")
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(y_trues, y_hats) if t == p)
    accuracy = correct / len(env_df)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(env_df)})")
    print(f"  Avg time: {np.mean(times):.2f}s")
    print(f"  Predictions: {pd.Series(y_hats).value_counts().to_dict()}")
    
    return {
        "env": env_choice,
        "accuracy": accuracy,
        "num_samples": len(env_df),
        "avg_time": np.mean(times),
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned VLM inference")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--env", type=str, default=None, help="Environment (default: all)")
    parser.add_argument("--num_samples", type=int, default=10, help="Samples per class")
    parser.add_argument("--no_8bit", action="store_true", help="Disable 8-bit quantization")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path, index_col=0)
    df = df.reset_index()
    df = add_image_path(df, args.image_dir)
    df = df[df["interaction_type"].isin(["g", "i", "n", "c", "s", "t"])]
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    df = sample_data(df, num_rows=args.num_samples)
    
    print(f"Loaded {len(df)} samples with valid images")
    
    # Load model
    model, processor = load_finetuned_model(
        args.model_path, use_8bit=not args.no_8bit
    )
    
    # Determine environments
    envs = [args.env] if args.env else ALL_ENVS
    envs = [e for e in envs if e in df["env_choice"].unique()]
    
    # Evaluate
    results = []
    for env in envs:
        result = evaluate_env(model, processor, df, env)
        if result:
            results.append(result)
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['env']}: {r['accuracy']:.2%}")
    
    if results:
        avg = np.mean([r["accuracy"] for r in results])
        print(f"\nAverage: {avg:.2%}")


if __name__ == "__main__":
    main()
