#!/usr/bin/env python3
"""
Download script for SIP datasets and pretrained models.

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --datasets
    python scripts/download_data.py --classifiers
    python scripts/download_data.py --checkpoints
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

SI_policy_URL = "https://diffusion-policy.cs.columbia.edu/data"

# Dataset URLs (from original Diffusion Policy)
DATASETS = {
    "pusht": f"{SI_policy_URL}/training/pusht.zip",
    "robomimic": f"{SI_policy_URL}/training/robomimic.zip",
}

# Flow label data (classifiers, etc.) - hosted on Dropbox
FLOW_LABEL_URL = "https://www.dropbox.com/scl/fi/y72ccuu6lv3sdztqau0a3/data-flow-label.zip?rlkey=buxwsuhs7n1ccm27h9uhc46qv&dl=1"

# Pretrained checkpoints (add URLs when available)
# These are trained SIP models ready for inference
CHECKPOINTS = {
    # Add URLs when hosting is set up
    # "square_linear": "https://your-host.com/checkpoints/square_linear.zip",
    # "can_linear": "https://your-host.com/checkpoints/can_linear.zip",
}

# Pre-trained classifier models (XGBoost)
# These are used for DASIP adaptive inference
CLASSIFIER_URLS = {
    # Add URLs when hosting is set up
    # "square": "https://your-host.com/classifiers/square_xgboost.pkl",
}


# =============================================================================
# Download Functions
# =============================================================================

def run_command(cmd: list, cwd: str = None) -> bool:
    """Run a shell command."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False


def download_file(url: str, output_path: str) -> bool:
    """Download a file using wget."""
    print(f"Downloading: {url}")
    print(f"       To: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try wget first, then curl
    if run_command(["wget", "-O", output_path, url]):
        return True
    elif run_command(["curl", "-L", "-o", output_path, url]):
        return True
    else:
        print(f"Failed to download: {url}")
        return False


def download_and_extract(url: str, extract_dir: str) -> bool:
    """Download and extract a zip file."""
    zip_path = os.path.join(extract_dir, os.path.basename(url))
    
    if not download_file(url, zip_path):
        return False
    
    print(f"Extracting: {zip_path}")
    if run_command(["unzip", "-o", zip_path, "-d", extract_dir]):
        os.remove(zip_path)
        return True
    return False


def download_datasets(data_dir: str, tasks: list = None):
    """Download training datasets."""
    print("\n" + "="*60)
    print("Downloading Datasets")
    print("="*60)
    
    os.makedirs(data_dir, exist_ok=True)
    
    for name, url in DATASETS.items():
        if tasks and name not in tasks:
            continue
        
        print(f"\n[{name}]")
        download_and_extract(url, data_dir)
    
    print("\nDatasets downloaded to:", data_dir)


def download_classifiers(data_dir: str):
    """Download flow label data (classifiers, etc.)."""
    print("\n" + "="*60)
    print("Downloading Flow Label Data (Classifiers)")
    print("="*60)
    
    zip_path = os.path.join(data_dir, "data-flow-label.zip")
    
    if download_file(FLOW_LABEL_URL, zip_path):
        print(f"Extracting: {zip_path}")
        # Extract to data_dir (the zip contains flow_label/ folder)
        if run_command(["unzip", "-o", zip_path, "-d", data_dir]):
            os.remove(zip_path)
            print("Flow label data extracted successfully!")
        else:
            print("Failed to extract. Please extract manually.")
    else:
        print("\n⚠️  Failed to download flow label data.")
        print("   Manual download: https://www.dropbox.com/scl/fi/y72ccuu6lv3sdztqau0a3/data-flow-label.zip")
        print(f"   Extract to: {data_dir}/")


def download_checkpoints(data_dir: str):
    """Download pretrained checkpoints."""
    print("\n" + "="*60)
    print("Downloading Pretrained Checkpoints")
    print("="*60)
    
    checkpoint_dir = os.path.join(data_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not CHECKPOINTS:
        print("\n⚠️  No pretrained checkpoints configured yet.")
        return
    
    # TODO: Update URL after hosting
    print("\n⚠️  Checkpoint download URLs not yet configured.")
    print("   Please manually download checkpoints to:", checkpoint_dir)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SIP datasets and models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_data.py --all
    python scripts/download_data.py --datasets
    python scripts/download_data.py --datasets --tasks pusht square
    python scripts/download_data.py --classifiers
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download everything (datasets, classifiers, checkpoints)",
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Download training datasets",
    )
    parser.add_argument(
        "--classifiers",
        action="store_true",
        help="Download classifier models",
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Download pretrained checkpoints",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to download (e.g., pusht square)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="SIP/data",
        help="Directory to download data to",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Default to --all if no specific option given
    if not (args.all or args.datasets or args.classifiers or args.checkpoints):
        args.all = True
    
    print("="*60)
    print("SIP Data Download Script")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    
    if args.all or args.datasets:
        download_datasets(args.data_dir, args.tasks)
    
    if args.all or args.classifiers:
        download_classifiers(args.data_dir)
    
    if args.all or args.checkpoints:
        download_checkpoints(args.data_dir)
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)


if __name__ == "__main__":
    main()

