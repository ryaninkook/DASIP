#!/usr/bin/env python3
"""
Select Sample Images for Few-Shot VLM Prompting.

Interactive script to browse and select representative images 
for each environment/interaction type combination.

Original: flow_label/v2label/filter_best_samples_zeeshan.py
See MODIFICATIONS.md for changes made.

Usage:
    python select_sample_images.py --env can --interaction i
    python select_sample_images.py --env square --interaction g
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Default paths
DEFAULT_DATA_PATH = "../../flow_label/data/good_4_user_filtered.csv"
DEFAULT_IMAGE_DIR = "../../SIP/data"
DEFAULT_SAMPLE_DIR = "../../flow_label/sample_images"

# All environments and interactions
ALL_ENVS = ['can', 'lift', 'square', 'tool_hang', 'transport', 'pusht']
ALL_INTERACTIONS = ['i', 'n', 't', 'c', 's', 'g']


# =============================================================================
# Helper Functions
# =============================================================================

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


def display_images(df, start_idx=0, num_images=5):
    """Display a grid of images from the dataframe."""
    end_idx = min(start_idx + num_images, len(df))
    num_show = end_idx - start_idx
    
    if num_show <= 0:
        print("No more images to display")
        return
    
    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
    if num_show == 1:
        axes = [axes]
    
    for i, idx in enumerate(range(start_idx, end_idx)):
        img_path = df.iloc[idx]['image_path']
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Index: {idx}')
    
    plt.tight_layout()
    plt.show()
    
    return start_idx, end_idx


def save_sample(df, idx, save_dir, sample_num):
    """Save a selected image as a sample."""
    os.makedirs(save_dir, exist_ok=True)
    img_path = df.iloc[idx]['image_path']
    img = Image.open(img_path)
    save_path = os.path.join(save_dir, f'sample_{sample_num}.png')
    img.save(save_path)
    print(f"Saved: {save_path}")


# =============================================================================
# Interactive Selection
# =============================================================================

def interactive_select(df, env_choice, interaction, sample_dir):
    """Interactively select and save sample images."""
    # Filter data
    filtered = df[(df['env_choice'] == env_choice) & 
                  (df['interaction_type'] == interaction)]
    filtered = filtered.reset_index(drop=True)
    
    if len(filtered) == 0:
        print(f"No images found for {env_choice}/{interaction}")
        return
    
    print(f"\nFound {len(filtered)} images for {env_choice}/{interaction}")
    print("Commands: n=next, p=prev, s=save, q=quit")
    
    save_dir = os.path.join(sample_dir, env_choice, interaction)
    current_idx = 0
    sample_num = 1
    
    # Check existing samples
    if os.path.exists(save_dir):
        existing = [f for f in os.listdir(save_dir) if f.startswith('sample_')]
        sample_num = len(existing) + 1
        print(f"Existing samples: {len(existing)}")
    
    while True:
        display_images(filtered, current_idx, num_images=5)
        
        cmd = input("\nCommand (n/p/s <idx>/q): ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'n':
            current_idx = min(current_idx + 5, len(filtered) - 1)
        elif cmd == 'p':
            current_idx = max(current_idx - 5, 0)
        elif cmd.startswith('s '):
            try:
                idx = int(cmd.split()[1])
                if 0 <= idx < len(filtered):
                    save_sample(filtered, idx, save_dir, sample_num)
                    sample_num += 1
                else:
                    print(f"Invalid index. Must be 0-{len(filtered)-1}")
            except (ValueError, IndexError):
                print("Usage: s <index>")
        else:
            print("Unknown command")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Select sample images for few-shot")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--env", type=str, required=True, choices=ALL_ENVS)
    parser.add_argument("--interaction", type=str, required=True, choices=ALL_INTERACTIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path, index_col=0)
    df = add_image_path(df, args.image_dir)
    df = df.dropna(subset=["image_path"])
    
    print(f"Loaded {len(df)} samples with valid images")
    
    # Show available interactions for the environment
    env_df = df[df['env_choice'] == args.env]
    available = env_df['interaction_type'].unique()
    print(f"Available interactions for {args.env}: {list(available)}")
    
    # Interactive selection
    interactive_select(df, args.env, args.interaction, args.sample_dir)


if __name__ == "__main__":
    main()
