#!/usr/bin/env python3
"""
Train XGBoost/RandomForest classifiers for robot state classification.

This script trains lightweight classifiers that can be used for fast inference
during policy evaluation. The classifiers are trained on state observations
from labeled robot interaction data.

Labels:
    i = initialize (robot in starting position)
    n = near (robot near object)
    t = attempt to grab (robot trying to grasp)
    c = pushing (robot pushing object)
    s = holding (robot holding object)
    g = goal state (task completed)

Usage:
    # Train on all environments
    python train_xgboost.py --data_path total_labels_data_v2.csv
    
    # Train on specific environments
    python train_xgboost.py --data_path data.csv --envs square can lift
    
    # Use grid search for hyperparameter tuning
    python train_xgboost.py --data_path data.csv --grid_search
"""

import argparse
import ast
import copy
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    xgb = None

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None


# =============================================================================
# Constants
# =============================================================================

# Interaction type mapping: label -> integer
INTERACTION_MAP = {
    "i": 0,  # initialize
    "n": 1,  # near
    "t": 2,  # attempt to grab
    "s": 3,  # holding
    "c": 4,  # pushing
    "g": 5,  # goal state
}

# All supported environments
ALL_ENVS = ["pusht", "transport", "square", "tool_hang", "lift", "can"]

# Default grid search parameters
DEFAULT_PARAM_GRID = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
}


# =============================================================================
# Data Processing
# =============================================================================

def parse_list_string(s):
    """Parse a string representation of a list."""
    try:
        return ast.literal_eval(s)
    except:
        return []


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the labeled data.
    
    Args:
        data_path: Path to CSV file with labels
        
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Parse list columns if they're strings
    if df["objects"].dtype == object:
        df["objects"] = df["objects"].apply(parse_list_string)
    if df["interaction_types"].dtype == object:
        df["interaction_types"] = df["interaction_types"].apply(parse_list_string)
    
    # Explode lists to get individual votes
    df_exploded = df.explode("objects").explode("interaction_types")
    
    # Group by unique (env, episode, timestep) and find most frequent labels
    group_cols = ["env_choice", "episode_idx", "timestep_idx"]
    
    # Most frequent object
    most_freq_objects = (
        df_exploded.groupby(group_cols + ["objects"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
        .rename(columns={"objects": "most_frequent_object"})
    )
    
    # Most frequent interaction
    most_freq_interactions = (
        df_exploded.groupby(group_cols + ["interaction_types"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
        .rename(columns={"interaction_types": "most_frequent_interaction"})
    )
    
    # Merge votes
    votes = pd.merge(
        most_freq_objects[group_cols + ["most_frequent_object", "count"]],
        most_freq_interactions[group_cols + ["most_frequent_interaction", "count"]],
        on=group_cols,
        suffixes=("_obj", "_int"),
    )
    
    # Get final state observation (last timestamp for each group)
    df_sorted = df.sort_values("timestamp")
    final_state = df_sorted.groupby(group_cols, as_index=False).last()
    final_state = final_state[group_cols + ["state_obs"]]
    
    # Merge
    result = pd.merge(votes, final_state, on=group_cols, how="left")
    
    # Filter invalid entries
    result = result[
        (result["most_frequent_object"] != "") & 
        (result["most_frequent_interaction"] != "")
    ]
    result = result[~result["most_frequent_interaction"].str.contains(",", na=False)]
    
    # Parse state_obs if it's a string
    if result["state_obs"].dtype == object:
        result["state_obs"] = result["state_obs"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    print(f"Loaded {len(result)} samples")
    return result


def prepare_features(df: pd.DataFrame, env: str, min_vote_count: int = 0):
    """
    Prepare features and labels for training.
    
    Args:
        df: Preprocessed DataFrame
        env: Environment name to filter
        min_vote_count: Minimum vote count to include sample
        
    Returns:
        X_train, X_test, y_train, y_test numpy arrays
    """
    # Filter by environment
    env_df = df[df["env_choice"] == env].copy()
    
    # Map interactions to integers
    env_df["label"] = env_df["most_frequent_interaction"].map(INTERACTION_MAP)
    
    # Filter by vote count
    if min_vote_count > 0:
        env_df = env_df[env_df["count_int"] > min_vote_count]
    
    if len(env_df) == 0:
        return None, None, None, None
    
    # Prepare features
    X_list = env_df["state_obs"].tolist()
    X = np.array(X_list, dtype=np.float32)
    y = env_df["label"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def add_missing_classes(X_train, y_train, n_classes=6):
    """Add dummy samples for missing classes to ensure all classes are represented."""
    expected = set(range(n_classes))
    present = set(np.unique(y_train))
    missing = expected - present
    
    if missing:
        print(f"  Adding dummy samples for missing classes: {missing}")
        dummy_X = np.zeros((len(missing), X_train.shape[1]), dtype=X_train.dtype)
        dummy_y = np.array(list(missing), dtype=y_train.dtype)
        X_train = np.vstack([X_train, dummy_X])
        y_train = np.concatenate([y_train, dummy_y])
    
    return X_train, y_train


# =============================================================================
# Training Functions
# =============================================================================

def train_xgboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    grid_search: bool = True,
    param_grid: dict = None,
) -> tuple:
    """
    Train XGBoost classifier.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        grid_search: Whether to use grid search for hyperparameters
        param_grid: Parameter grid for grid search
        
    Returns:
        (model, accuracy, best_params)
    """
    if xgb is None:
        raise ImportError("XGBoost not installed")
    
    # Add dummy samples for missing classes
    X_train, y_train = add_missing_classes(X_train, y_train)
    
    if grid_search:
        param_grid = param_grid or DEFAULT_PARAM_GRID
        
        xgb_clf = xgb.XGBClassifier(random_state=42, num_class=6)
        
        grid = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring="accuracy",
            cv=3,
            verbose=1,
            n_jobs=-1,
        )
        
        grid.fit(X_train, y_train)
        
        model = grid.best_estimator_
        best_params = grid.best_params_
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Best CV Score: {grid.best_score_:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Best Params: {best_params}")
        
    else:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            num_class=6,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        best_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        
        print(f"  Test Accuracy: {accuracy:.4f}")
    
    return model, accuracy, best_params


def train_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
) -> tuple:
    """
    Train Random Forest classifier.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        n_estimators: Number of trees
        
    Returns:
        (model, accuracy)
    """
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn not installed")
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  Random Forest Accuracy: {accuracy:.4f}")
    
    return model, accuracy


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost/RF classifiers for robot state classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../flow_label/data/good_4_user_filtered.csv",
        help="Path to labeled data CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../SIP/data/classifiers",
        help="Directory to save trained models (default: SIP/data/classifiers)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="Environments to train on (default: all)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["xgboost", "rf", "both"],
        default="xgboost",
        help="Model type to train",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Use grid search for XGBoost hyperparameters",
    )
    parser.add_argument(
        "--min_vote_count",
        type=int,
        default=0,
        help="Minimum vote count to include sample",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_preprocess_data(args.data_path)
    
    # Determine environments to train on
    envs = args.envs or ALL_ENVS
    available_envs = df["env_choice"].unique().tolist()
    envs = [e for e in envs if e in available_envs]
    
    print(f"\nTraining on environments: {envs}")
    
    results = []
    
    for env in envs:
        print(f"\n{'='*60}")
        print(f"Training classifier for: {env}")
        print("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_features(
            df, env, args.min_vote_count
        )
        
        if X_train is None:
            print(f"  No data available for {env}, skipping...")
            continue
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Train XGBoost
        if args.model_type in ["xgboost", "both"]:
            print("\n  Training XGBoost...")
            try:
                model, acc, params = train_xgboost(
                    X_train, X_test, y_train, y_test,
                    grid_search=args.grid_search,
                )
                
                # Save model with detailed name
                acc_pct = int(acc * 100)
                params_str = "_".join(f"{k}{v}" for k, v in params.items())
                filename = f"{env}_xg_{acc_pct}_{params_str}.pkl"
                filepath = output_dir / filename
                joblib.dump(model, filepath)
                print(f"  Saved: {filepath}")
                
                # Also save with simple name for DASIP compatibility
                simple_path = output_dir / f"{env}_xgboost.pkl"
                joblib.dump(model, simple_path)
                print(f"  Saved: {simple_path} (for DASIP)")
                
                results.append({
                    "env": env,
                    "model": "xgboost",
                    "accuracy": acc,
                    "file": str(filepath),
                })
                
            except Exception as e:
                print(f"  XGBoost training failed: {e}")
        
        # Train Random Forest
        if args.model_type in ["rf", "both"]:
            print("\n  Training Random Forest...")
            try:
                model, acc = train_random_forest(X_train, X_test, y_train, y_test)
                
                # Save model
                acc_pct = int(acc * 100)
                filename = f"{env}_rf_{acc_pct}.pkl"
                filepath = output_dir / filename
                
                joblib.dump(model, filepath)
                print(f"  Saved: {filepath}")
                
                results.append({
                    "env": env,
                    "model": "rf",
                    "accuracy": acc,
                    "file": str(filepath),
                })
                
            except Exception as e:
                print(f"  Random Forest training failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Save summary
        summary_path = output_dir / "training_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("No models were trained.")


if __name__ == "__main__":
    main()

