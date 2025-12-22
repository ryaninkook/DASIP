#!/usr/bin/env python3
"""
Finetune Qwen2.5-VL for robot state classification.

This script finetunes a Vision-Language Model (VLM) on labeled robot interaction
images to predict the robot's state (initialize, near, attempting, pushing, 
holding, goal).

Features:
- 8-bit quantization for memory efficiency
- LoRA for parameter-efficient finetuning
- Leave-one-out environment evaluation
- Automatic checkpoint saving

Usage:
    # Train on all envs except 'lift', evaluate on 'lift'
    python vlm_finetune.py --train_envs can square tool_hang transport --test_env lift
    
    # Train with custom settings
    python vlm_finetune.py --train_envs square can --test_env lift --epochs 10 --batch_size 4
"""

import argparse
import base64
import copy
import io
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score

# Transformers and training
from datasets import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    default_data_collator,
)
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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


# =============================================================================
# Constants
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Label definitions
LABELS = ["i", "n", "t", "c", "s", "g"]
LABEL_NAMES = {
    "i": "initialize",
    "n": "near",
    "t": "attempt to grab",
    "c": "pushing",
    "s": "holding",
    "g": "goal state",
}

# Default system message and prompt
SYSTEM_MESSAGE = (
    "You are a helpful robot label assistant. "
    "You will be shown an image and must respond with the label i, n, t, c, s, g."
)
FINETUNE_PROMPT = (
    "Label the robot arm in the image whatever you think closest. "
    "Do not explain answer only give one character."
)


# =============================================================================
# Data Processing
# =============================================================================

def add_image_paths(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Add image paths to dataframe based on env/episode/timestep."""
    image_paths = []
    
    for _, row in df.iterrows():
        path = os.path.join(
            data_dir,
            f"{row['env_choice']}/images/",
            f"episode_{row['episode_idx']}/",
            f"image_{row['timestep_idx']}.png"
        )
        image_paths.append(path if os.path.exists(path) else None)
    
    df["image_path"] = image_paths
    return df


def load_data(
    data_path: str,
    data_dir: str,
    train_envs: list,
    test_env: str,
) -> tuple:
    """
    Load and split data for training and testing.
    
    Args:
        data_path: Path to labeled data CSV
        data_dir: Directory containing images
        train_envs: List of environments for training
        test_env: Environment for testing
        
    Returns:
        (train_df, test_df)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)
    
    # Split by environment
    train_df = df[df["env_choice"].isin(train_envs)].copy()
    test_df = df[df["env_choice"] == test_env].copy()
    
    # Add image paths
    train_df = add_image_paths(train_df, data_dir)
    test_df = add_image_paths(test_df, data_dir)
    
    # Filter valid labels and existing images
    valid_labels = set(LABELS)
    train_df = train_df[train_df["interaction_type"].isin(valid_labels)]
    test_df = test_df[test_df["interaction_type"].isin(valid_labels)]
    train_df = train_df.dropna(subset=["image_path"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["image_path"]).reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)} from {train_envs}")
    print(f"Test samples: {len(test_df)} from {test_env}")
    
    return train_df, test_df


def sample_data(
    df: pd.DataFrame,
    max_per_class: int = 1000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Sample data balanced by environment and interaction type."""
    df = df.sort_values(["env_choice", "episode_idx", "timestep_idx"])
    
    sampled = (
        df.groupby(["env_choice", "object", "interaction_type"])
        .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=seed))
        .reset_index(drop=True)
    )
    
    return sampled


# =============================================================================
# Model Setup
# =============================================================================

def load_model_and_processor(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir: str = "/mnt/l1/huggingface",
    use_8bit: bool = True,
    max_memory: dict = None,
    hf_token: str = None,
):
    """
    Load Qwen2.5-VL model with quantization and LoRA.
    
    Returns:
        (model, processor, tokenizer, lora_config)
    """
    max_memory = max_memory or {0: "24GB", 1: "24GB"}
    
    # Quantization config
    quant_config = None
    if use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
        )
    
    print(f"Loading model: {model_name}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
        max_memory=max_memory,
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=hf_token,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = processor.tokenizer
    
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    language_targets = [
        name for name, module in model.model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(p in name for p in ("q_proj", "k_proj", "v_proj", "o_proj", 
                                      "gate_proj", "up_proj", "down_proj"))
    ]
    
    visual_targets = [
        name for name, module in model.visual.named_modules()
        if isinstance(module, torch.nn.Linear) and "proj" in name
    ]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=language_targets + visual_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor, tokenizer, lora_config


# =============================================================================
# Training
# =============================================================================

def create_preprocess_fn(processor, max_seq_length: int = 128):
    """Create preprocessing function for training data."""
    
    def preprocess(example):
        # Load and resize image
        img = Image.open(example["image_path"]).convert("RGB")
        img = img.resize((224, 224), resample=Image.BILINEAR)
        
        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Build conversation
        conv = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{img_b64}"},
                    {"type": "text", "text": FINETUNE_PROMPT},
                ],
            },
            {"role": "assistant", "content": example["interaction_type"]},
        ]
        
        # Process
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conv)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        
        # Flatten and add labels
        tensor_inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        tensor_inputs["labels"] = tensor_inputs["input_ids"].clone()
        
        return tensor_inputs
    
    return preprocess


class SaveStepsCallback(TrainerCallback):
    """Callback to save model at regular intervals."""
    
    def __init__(self, save_every: int = 100):
        self.save_every = save_every
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_every == 0 and state.global_step > 0:
            trainer = kwargs.get("trainer")
            if trainer:
                step_dir = os.path.join(args.output_dir, f"step_{state.global_step}")
                os.makedirs(step_dir, exist_ok=True)
                trainer.save_model(step_dir)
                print(f"Model saved at step {state.global_step}")
        return control


def train(
    model,
    processor,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    lora_config,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
):
    """Run finetuning."""
    
    # Sample data
    train_df = sample_data(train_df, max_per_class=1000)
    eval_df = sample_data(eval_df, max_per_class=200)
    
    print(f"Training on {len(train_df)} samples")
    print(f"Evaluating on {len(eval_df)} samples")
    
    # Create datasets
    preprocess_fn = create_preprocess_fn(processor, max_seq_length)
    
    train_dataset = Dataset.from_pandas(train_df).map(
        preprocess_fn, remove_columns=train_df.columns.tolist()
    )
    eval_dataset = Dataset.from_pandas(eval_df).map(
        preprocess_fn, remove_columns=eval_df.columns.tolist()
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        prediction_loss_only=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        fp16=True,
        label_names=["labels"],
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        peft_config=lora_config,
        callbacks=[SaveStepsCallback(save_every=100)],
    )
    
    # Train
    print("\nStarting training...")
    result = trainer.train()
    
    # Save final model
    model_path = f"{output_dir}/model"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Training metrics: {result.metrics}")
    
    return result


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    model,
    processor,
    test_df: pd.DataFrame,
    device: str = "cuda:0",
):
    """Evaluate finetuned model on test set."""
    
    model.eval()
    
    # Sample test data
    test_df = sample_data(test_df, max_per_class=1000)
    
    y_true = []
    y_pred = []
    
    print(f"\nEvaluating on {len(test_df)} samples...")
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        image_path = row["image_path"]
        true_label = row["interaction_type"]
        
        # Load image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Build message
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{img_b64}"},
                    {"type": "text", "text": FINETUNE_PROMPT},
                ],
            },
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device and convert types
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
                if inputs[k].dtype.is_floating_point:
                    inputs[k] = inputs[k].half()
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=4)
        
        output_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        
        pred = processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Normalize prediction
        pred = pred.strip().lower()
        if len(pred) > 1:
            # Try to extract single character
            for label in LABELS:
                if label in pred:
                    pred = label
                    break
        
        y_true.append(true_label)
        y_pred.append(pred)
        
        if (idx + 1) % 50 == 0:
            current_acc = accuracy_score(y_true, y_pred)
            print(f"  Progress: {idx + 1}/{len(test_df)}, Current Accuracy: {current_acc:.4f}")
    
    # Final accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    # Per-label accuracy
    for label in LABELS:
        indices = [i for i, y in enumerate(y_true) if y == label]
        if indices:
            label_acc = sum(1 for i in indices if y_pred[i] == label) / len(indices)
            print(f"  {label} ({LABEL_NAMES[label]}): {label_acc:.4f} ({len(indices)} samples)")
    
    return accuracy


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Qwen2.5-VL for robot state classification"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="../flow_label/v3_combined_df/good_4_user_filtered.csv",
        help="Path to labeled data CSV",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Directory containing image data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vlm_finetuned",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--train_envs",
        type=str,
        nargs="+",
        default=["can", "square", "tool_hang", "transport"],
        help="Environments to train on",
    )
    parser.add_argument(
        "--test_env",
        type=str,
        default="lift",
        help="Environment to test on",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/l1/huggingface",
        help="HuggingFace cache directory",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model_path", type=str, help="Path to finetuned model for eval")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    
    # Load data
    train_df, test_df = load_data(
        args.data_path,
        args.data_dir,
        args.train_envs,
        args.test_env,
    )
    
    # Load model
    model, processor, tokenizer, lora_config = load_model_and_processor(
        cache_dir=args.cache_dir,
    )
    
    torch.cuda.empty_cache()
    
    if not args.eval_only:
        # Train
        train(
            model=model,
            processor=processor,
            train_df=train_df,
            eval_df=test_df,
            lora_config=lora_config,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    
    # Evaluate
    accuracy = evaluate(model, processor, test_df)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Environment: {args.test_env}")
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

