#!/usr/bin/env python3
"""
Prepare and filter labeled data for classifier training.

This script processes raw labeled data by:
1. Filtering to trusted labelers (4 "good" users)
2. Computing majority votes for overlapping labels
3. Adding image paths for CNN training
4. Outputting clean CSV files for training

Usage:
    # Process labels from Excel file
    python prepare_labels.py --input labels.xlsx --output clean_labels.csv
    
    # Process from MongoDB export
    python prepare_labels.py --input labels_v2.csv --output clean_labels.csv --from_mongo
"""

import argparse
import ast
import os
from collections import Counter
from pathlib import Path

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

# Trusted labelers (filtered for quality)
GOOD_USERS = ["fbdht901", "zxwmp123", "ydgfv456", "lqwst012"]

# Valid interaction types
VALID_INTERACTIONS = ["i", "n", "t", "c", "s", "g"]

# Interaction type mapping
INTERACTION_MAP = {
    "i": 0,  # initialize
    "n": 1,  # near
    "t": 2,  # attempt to grab
    "s": 3,  # holding
    "c": 4,  # pushing
    "g": 5,  # goal state
}


# =============================================================================
# Data Processing Functions
# =============================================================================

def parse_list_str(list_str):
    """Parse a string representation of a list."""
    try:
        return ast.literal_eval(list_str)
    except:
        return []


def load_from_mongo_export(path: str) -> pd.DataFrame:
    """
    Load and process data from MongoDB CSV export.
    
    This format has 'objects' and 'interaction_types' as list columns
    that need to be exploded and voted on.
    """
    print(f"Loading MongoDB export from {path}...")
    df = pd.read_csv(path)
    
    # Parse list columns
    df["objects"] = df["objects"].apply(parse_list_str)
    df["interaction_types"] = df["interaction_types"].apply(parse_list_str)
    
    # Explode lists to get individual votes
    df_exploded = df.explode("objects").explode("interaction_types")
    
    # Group by unique (env, episode, timestep)
    group_cols = ["env_choice", "episode_idx", "timestep_idx"]
    
    # Most frequent object
    most_freq_objects = (
        df_exploded.groupby(group_cols + ["objects"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
        .rename(columns={"objects": "object"})
    )
    
    # Most frequent interaction
    most_freq_interactions = (
        df_exploded.groupby(group_cols + ["interaction_types"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
        .rename(columns={"interaction_types": "interaction_type"})
    )
    
    # Merge votes
    result = pd.merge(
        most_freq_objects[group_cols + ["object", "count"]],
        most_freq_interactions[group_cols + ["interaction_type", "count"]],
        on=group_cols,
        suffixes=("_obj", "_int"),
    )
    
    # Get state observations (last timestamp for each group)
    if "state_obs" in df.columns:
        df_sorted = df.sort_values("timestamp")
        final_state = df_sorted.groupby(group_cols, as_index=False).last()
        final_state = final_state[group_cols + ["state_obs"]]
        result = pd.merge(result, final_state, on=group_cols, how="left")
    
    return result


def load_from_excel(path: str, filter_users: bool = True) -> pd.DataFrame:
    """
    Load and process data from Excel file with individual user labels.
    
    Computes majority vote across users for each (env, episode, timestep).
    """
    print(f"Loading Excel file from {path}...")
    df = pd.read_excel(path)
    
    # Filter to good users if requested
    if filter_users and "username" in df.columns:
        original_len = len(df)
        df = df[df["username"].isin(GOOD_USERS)]
        print(f"Filtered to {len(df)}/{original_len} samples from trusted users")
    
    # Drop missing values
    df = df.dropna(subset=["interaction_type"]).reset_index(drop=True)
    
    # Compute majority votes
    result = compute_majority_votes(df)
    
    return result


def compute_majority_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute majority vote for interaction_type across overlapping labels.
    
    Args:
        df: DataFrame with columns [env_choice, episode_idx, timestep_idx, interaction_type, ...]
        
    Returns:
        DataFrame with one row per (env, episode, timestep) with majority label
    """
    group_cols = ["env_choice", "episode_idx", "timestep_idx"]
    grouped = df.groupby(group_cols)
    
    results = []
    
    for name, group in grouped:
        env_choice, episode_idx, timestep_idx = name
        
        # Get majority vote
        counter = Counter(group["interaction_type"])
        majority_type, vote_count = counter.most_common(1)[0]
        
        # Build result row
        row = {
            "env_choice": env_choice,
            "episode_idx": episode_idx,
            "timestep_idx": timestep_idx,
            "interaction_type": majority_type,
            "vote_count": vote_count,
            "total_votes": len(group),
        }
        
        # Copy other columns from first row
        first = group.iloc[0]
        for col in df.columns:
            if col not in row and col != "username":
                row[col] = first[col]
        
        results.append(row)
    
    return pd.DataFrame(results)


def add_image_paths(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Add image paths based on env/episode/timestep."""
    paths = []
    
    for _, row in df.iterrows():
        path = os.path.join(
            data_dir,
            f"{row['env_choice']}/images/",
            f"episode_{int(row['episode_idx'])}/",
            f"image_{int(row['timestep_idx'])}.png"
        )
        paths.append(path if os.path.exists(path) else None)
    
    df["image_path"] = paths
    return df


def add_category_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer category codes for interactions."""
    df["interaction_category"] = df["interaction_type"].map(INTERACTION_MAP)
    return df


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with valid interaction types."""
    original_len = len(df)
    
    # Filter empty/invalid
    df = df[df["interaction_type"].isin(VALID_INTERACTIONS)]
    
    # Remove compound labels (e.g., "i,n")
    if df["interaction_type"].str.contains(",").any():
        df = df[~df["interaction_type"].str.contains(",", na=False)]
    
    print(f"Filtered to {len(df)}/{original_len} valid samples")
    return df


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare labeled data for classifier training"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (CSV or Excel)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clean_labels.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Directory containing images (for adding paths)",
    )
    parser.add_argument(
        "--from_mongo",
        action="store_true",
        help="Input is MongoDB CSV export format",
    )
    parser.add_argument(
        "--no_filter_users",
        action="store_true",
        help="Don't filter to trusted users",
    )
    parser.add_argument(
        "--add_images",
        action="store_true",
        help="Add image paths to output",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    input_path = args.input
    
    if args.from_mongo or input_path.endswith(".csv"):
        df = load_from_mongo_export(input_path)
    else:
        df = load_from_excel(input_path, filter_users=not args.no_filter_users)
    
    print(f"Loaded {len(df)} samples")
    
    # Filter valid labels
    df = filter_valid_labels(df)
    
    # Add category codes
    df = add_category_codes(df)
    
    # Add image paths if requested
    if args.add_images:
        df = add_image_paths(df, args.data_dir)
        valid_images = df["image_path"].notna().sum()
        print(f"Found {valid_images}/{len(df)} valid image paths")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df)} samples to {output_path}")
    
    # Print summary
    print("\nLabel distribution:")
    print(df["interaction_type"].value_counts())
    
    print("\nPer-environment counts:")
    print(df.groupby("env_choice").size())


if __name__ == "__main__":
    main()

