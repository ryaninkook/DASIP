#!/usr/bin/env python3
"""
VLM Zero-Shot Inference with Few-Shot Examples.

This script evaluates VLM classification using sample images as in-context
examples (no fine-tuning required). Shows example images with labels,
then asks model to classify new images.

Original: flow_label/v2label/a7_1_zeeshan_focused_samples.py
See MODIFICATIONS.md for changes made.

Usage:
    python vlm_zeroshot_fewshot.py --env square --num_examples 2 --method v2
    python vlm_zeroshot_fewshot.py --env can --num_examples 3 --method v3
"""

import argparse
import copy
import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

# =============================================================================
# Configuration - Modify these paths as needed
# =============================================================================

# Default paths (relative to this script's location)
DEFAULT_DATA_PATH = "../../flow_label/data/good_4_user_filtered.csv"
DEFAULT_IMAGE_DIR = "../../SIP/data"
DEFAULT_SAMPLE_DIR = "../../flow_label/sample_images"

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Label mapping
MAP_LABELS = {
    "i": "initialize",
    "n": "near",
    "t": "attempt to grab",
    "c": "pushing",
    "s": "hanger",
    "g": "goal state"
}

# All environments
ALL_ENVS = ['can', 'lift', 'square', 'tool_hang', 'transport', 'pusht']


# =============================================================================
# Helper Functions
# =============================================================================

def sample_data(df, num_rows: int = 100):
    """Sample data balanced by environment and interaction type."""
    df = df.sort_values(['env_choice', 'episode_idx', 'timestep_idx'])
    df = df.groupby(['env_choice', 'interaction_type']).apply(
        lambda x: x.head(num_rows)
    ).reset_index(drop=True)
    return df


def add_image_path(df, base_path):
    """Add image paths to dataframe."""
    images_paths = []
    for index, row in df.iterrows():
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


def normalize_prediction(y_hat):
    """Normalize VLM output to single character label."""
    if isinstance(y_hat, list):
        y_hat = y_hat[0]
    if "Label: " in y_hat:
        y_hat = y_hat.split("Label: ")[1].strip()
    
    y_hat_lower = y_hat.lower()
    if 'near' in y_hat_lower:
        return 'n'
    elif 'initialize' in y_hat_lower:
        return 'i'
    elif 'attempt to grab' in y_hat_lower:
        return 't'
    elif 'pushing' in y_hat_lower:
        return 'c'
    elif 'hanger' in y_hat_lower:
        return 's'
    elif 'goal state' in y_hat_lower:
        return 'g'
    return y_hat


def load_sample_images(env_choice, unique_interactions, sample_dir, num_per_class):
    """Load sample images for each interaction type."""
    sample_images = []
    image_paths = []
    
    for interaction in unique_interactions:
        sample_save_path = os.path.join(sample_dir, env_choice, interaction)
        if not os.path.exists(sample_save_path):
            print(f"Warning: Sample path not found: {sample_save_path}")
            continue
        
        img_list = os.listdir(sample_save_path)
        img_list = img_list[:num_per_class]
        
        for img in img_list:
            full_path = os.path.join(sample_save_path, img)
            sample_images.append({
                "image_path": full_path,
                "text": MAP_LABELS[interaction]
            })
            image_paths.append((full_path, interaction))
    
    return sample_images, image_paths


def load_model(model_type: str):
    """Load VLM model."""
    if model_type == "qwen":
        from models.qwen_vlm_original import Qwen2_5Model
        return Qwen2_5Model()
    elif model_type == "llava":
        from models.llava_vlm import LlavaNextModel
        return LlavaNextModel()
    elif model_type == "gemma":
        from models.gemma3_vlm import Gemma3Model
        return Gemma3Model()
    else:
        raise ValueError(f"Unknown model: {model_type}")


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_env(
    model,
    df,
    env_choice,
    sample_dir,
    num_examples=2,
    num_test=50,
    method="v2",
):
    """
    Evaluate zero-shot few-shot on one environment.
    
    Args:
        model: VLM model with predict_v2/predict_v3 methods
        df: DataFrame with image paths
        env_choice: Environment to evaluate
        sample_dir: Directory with sample images
        num_examples: Number of example images per class
        num_test: Number of test samples
        method: "v2" or "v3" (different prompting styles)
    
    Returns:
        dict with accuracy, times, predictions
    """
    print(f"\n{'='*60}")
    print(f"Environment: {env_choice}")
    print(f"Method: predict_{method}")
    print(f"Examples per class: {num_examples}")
    print("=" * 60)
    
    # Filter data
    env_df = df[df["env_choice"] == env_choice].reset_index(drop=True)
    env_df = env_df[env_df["interaction_type"].isin(['g', 'i', 'n', 's', 't'])]
    
    if len(env_df) == 0:
        print(f"No data for {env_choice}")
        return None
    
    unique_interactions = env_df["interaction_type"].unique()
    print(f"Interactions: {list(unique_interactions)}")
    
    # Load sample images
    sample_images, _ = load_sample_images(
        env_choice, unique_interactions, sample_dir, num_examples
    )
    
    if not sample_images:
        print(f"No sample images found for {env_choice}")
        return None
    
    print(f"Loaded {len(sample_images)} sample images")
    
    # Shuffle and limit test data
    shuffled_df = env_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    run_df = shuffled_df[:num_test]
    
    print(f"Testing on {len(run_df)} samples")
    
    # Prompt
    prompt = "Simply pick the label from the sample images you see and tell me which one is closest to what you see."
    
    # Run predictions
    y_hats = []
    y_trues = []
    times = []
    
    predict_fn = model.predict_v2 if method == "v2" else model.predict_v3
    
    for idx in range(len(run_df)):
        images_path = run_df.iloc[idx]["image_path"]
        y_true = run_df.iloc[idx]["interaction_type"]
        
        # Copy sample images (avoid mutation)
        imgs = copy.deepcopy(sample_images)
        
        # Predict
        start = time.time()
        y_hat = predict_fn(imgs, target_image=images_path, prompt=prompt)
        elapsed = time.time() - start
        
        # Normalize
        y_hat = normalize_prediction(y_hat)
        
        y_hats.append(y_hat)
        y_trues.append(y_true)
        times.append(elapsed)
        
        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for t, p in zip(y_trues, y_hats) if t == p) / len(y_trues)
            print(f"  Progress: {idx+1}/{len(run_df)}, Accuracy: {current_acc:.2%}")
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(y_trues, y_hats) if t == p)
    accuracy = correct / len(run_df)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(run_df)})")
    print(f"  Avg time: {np.mean(times):.2f}s")
    print(f"  Prediction distribution: {pd.Series(y_hats).value_counts().to_dict()}")
    
    return {
        "env": env_choice,
        "method": method,
        "accuracy": accuracy,
        "num_examples": num_examples,
        "num_test": len(run_df),
        "avg_time": np.mean(times),
        "y_true": y_trues,
        "y_pred": y_hats,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM zero-shot evaluation with few-shot examples"
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--env", type=str, default=None, help="Environment (default: all)")
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "llava", "gemma"])
    parser.add_argument("--num_examples", type=int, default=2, help="Sample images per class")
    parser.add_argument("--num_test", type=int, default=50, help="Test samples per env")
    parser.add_argument("--method", type=str, default="v2", choices=["v2", "v3"])
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path, index_col=0)
    df = add_image_path(df, args.image_dir)
    df = df.dropna(subset=["image_path"])
    
    print(f"Loaded {len(df)} samples with valid images")
    
    # Load model
    print(f"\nLoading {args.model} model...")
    model = load_model(args.model)
    
    # Determine environments
    envs = [args.env] if args.env else ALL_ENVS
    envs = [e for e in envs if e in df["env_choice"].unique()]
    
    # Evaluate
    results = []
    for env in envs:
        result = evaluate_env(
            model=model,
            df=df,
            env_choice=env,
            sample_dir=args.sample_dir,
            num_examples=args.num_examples,
            num_test=args.num_test,
            method=args.method,
        )
        if result:
            results.append(result)
    
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
