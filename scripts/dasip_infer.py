#!/usr/bin/env python3
"""
DASIP: Diffusion-Adaptive State-Informed Policy Inference.

This script implements the complete DASIP inference pipeline:
1. Load a trained diffusion policy checkpoint
2. Load a trained state classifier (XGBoost/CNN)
3. Run simulation with adaptive diffusion steps based on classifier output
4. Measure and report speedup vs baseline

The key insight of DASIP is that different robot states require different
amounts of diffusion computation:
- Early states (initialize, near): Fewer steps needed
- Complex states (attempt, pushing): More steps needed  
- Final states (holding, goal): Fewer steps needed

Usage:
    # Basic usage
    python scripts/dasip_infer.py \
        --env square \
        --checkpoint_path v11_square_train_Linear \
        --infer_from 2000

    # Compare with baseline (fixed steps)
    python scripts/dasip_infer.py \
        --env square \
        --checkpoint_path v11_square_train_Linear \
        --infer_from 2000 \
        --compare_baseline

    # Custom step schedule
    python scripts/dasip_infer.py \
        --env square \
        --checkpoint_path v11_square_train_Linear \
        --steps_easy 2 \
        --steps_medium 5 \
        --steps_hard 10

Example output:
    ============================================================
    DASIP Inference Results
    ============================================================
    Environment: square
    Episodes: 300
    
    DASIP (Adaptive):
      Success Rate: 94.3%
      Avg Time/Episode: 1.23s
      Total Time: 369.0s
    
    Baseline (Fixed 10 steps):
      Success Rate: 95.0%
      Avg Time/Episode: 2.45s
      Total Time: 735.0s
    
    Speedup: 1.99x (with 0.7% success rate reduction)
    ============================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import joblib

# Add SIP to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
SIP_DIR = PROJECT_ROOT / "SIP"
sys.path.insert(0, str(SIP_DIR))

import hydra
from omegaconf import OmegaConf

# =============================================================================
# Configuration
# =============================================================================

# Classifier paths (relative to SIP/)
CLASSIFIER_PATHS = {
    "square": "data/classifiers/square_xgboost.pkl",
    "can": "data/classifiers/can_xgboost.pkl",
    "lift": "data/classifiers/lift_xgboost.pkl",
    "pusht": "data/classifiers/pusht_xgboost.pkl",
    "tool_hang": "data/classifiers/tool_hang_xgboost.pkl",
    "transport": "data/classifiers/transport_xgboost.pkl",
}

# State-to-difficulty mapping
# i=initialize, n=near, t=attempt, c=pushing, s=holding, g=goal
STATE_DIFFICULTY = {
    "i": "easy",      # Initialize - simple, just starting
    "n": "medium",    # Near - approaching, some precision needed
    "t": "hard",      # Attempt to grab - complex manipulation
    "c": "hard",      # Pushing - complex control
    "s": "medium",    # Holding - maintaining grip
    "g": "easy",      # Goal - near completion
}

# Default step counts for each difficulty
DEFAULT_STEPS = {
    "easy": 2,
    "medium": 5,
    "hard": 10,
}

# Baseline fixed steps for comparison
BASELINE_STEPS = 10


# =============================================================================
# Utility Functions
# =============================================================================

def get_task_name(env: str) -> str:
    """Get task name from environment."""
    env_to_task = {
        "square": "square",
        "can": "can",
        "lift": "lift",
        "pusht": "pusht",
        "tool_hang": "tool_hang",
        "transport": "transport",
    }
    return env_to_task.get(env, env)


def load_classifier(env: str) -> Optional[Any]:
    """Load the XGBoost classifier for the environment."""
    classifier_path = CLASSIFIER_PATHS.get(env)
    if not classifier_path:
        print(f"Warning: No classifier configured for {env}")
        return None
    
    full_path = SIP_DIR / classifier_path
    if not full_path.exists():
        print(f"Warning: Classifier not found at {full_path}")
        print("  Train one with: python scripts/classifiers/train_xgboost.py")
        return None
    
    print(f"Loading classifier from: {full_path}")
    return joblib.load(full_path)


def classify_state(classifier, obs: np.ndarray) -> str:
    """Classify the current robot state from observation."""
    if classifier is None:
        return "t"  # Default to hard if no classifier
    
    # XGBoost expects 2D array
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    
    # Get prediction
    pred = classifier.predict(obs)[0]
    
    # Map numeric to label if needed
    label_map = {0: "i", 1: "n", 2: "t", 3: "c", 4: "s", 5: "g"}
    if isinstance(pred, (int, np.integer)):
        return label_map.get(pred, "t")
    return pred


def get_steps_for_state(state: str, step_config: Dict[str, int]) -> int:
    """Get number of diffusion steps based on state difficulty."""
    difficulty = STATE_DIFFICULTY.get(state, "hard")
    return step_config.get(difficulty, DEFAULT_STEPS["hard"])


# =============================================================================
# Checkpoint Loading
# =============================================================================

def find_checkpoint(checkpoint_path: str, infer_from: int) -> Tuple[Path, OmegaConf]:
    """Find checkpoint file and config."""
    # Try different path formats
    candidates = [
        SIP_DIR / "outputs" / checkpoint_path,
        SIP_DIR / checkpoint_path,
        Path(checkpoint_path),
    ]
    
    checkpoint_dir = None
    for candidate in candidates:
        if candidate.exists():
            checkpoint_dir = candidate
            break
    
    if checkpoint_dir is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Find checkpoint file
    ckpt_file = checkpoint_dir / "checkpoints" / f"epoch_{infer_from:04d}.ckpt"
    if not ckpt_file.exists():
        # Try finding latest
        ckpt_dir = checkpoint_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt"))
            if ckpts:
                ckpt_file = ckpts[-1]
                print(f"Using latest checkpoint: {ckpt_file.name}")
    
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    
    # Load config
    config_file = checkpoint_dir / ".hydra" / "config.yaml"
    if not config_file.exists():
        config_file = checkpoint_dir / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found in {checkpoint_dir}")
    
    cfg = OmegaConf.load(config_file)
    
    return ckpt_file, cfg


def load_policy(ckpt_file: Path, cfg: OmegaConf, device: str = "cuda:0"):
    """Load the policy from checkpoint."""
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    
    # Load checkpoint
    payload = torch.load(ckpt_file, map_location=device)
    
    # Instantiate workspace
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy
    policy = workspace.model
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    return policy, workspace, cfg


# =============================================================================
# DASIP Inference Engine
# =============================================================================

class DASIPInference:
    """DASIP adaptive inference engine."""
    
    def __init__(
        self,
        policy,
        classifier,
        cfg: OmegaConf,
        step_config: Dict[str, int],
        device: str = "cuda:0",
    ):
        self.policy = policy
        self.classifier = classifier
        self.cfg = cfg
        self.step_config = step_config
        self.device = device
        
        # Statistics
        self.state_counts = defaultdict(int)
        self.step_history = []
    
    def get_action(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get action with adaptive diffusion steps.
        
        Returns:
            action: The action to take
            info: Dict with state, steps, timing info
        """
        start_time = time.time()
        
        # Extract observation for classification
        if "obs" in obs:
            obs_flat = obs["obs"]
        elif "agent_pos" in obs:
            obs_flat = obs["agent_pos"]
        else:
            # Try to flatten
            obs_flat = np.concatenate([v.flatten() for v in obs.values()])
        
        # Classify state
        state = classify_state(self.classifier, obs_flat)
        self.state_counts[state] += 1
        
        # Get adaptive step count
        num_steps = get_steps_for_state(state, self.step_config)
        self.step_history.append(num_steps)
        
        # Set diffusion steps in policy
        if hasattr(self.policy, 'num_inference_steps'):
            self.policy.num_inference_steps = num_steps
        elif hasattr(self.policy, 'n_action_steps'):
            # Some policies use different attribute
            pass
        
        # Get action from policy
        with torch.no_grad():
            obs_dict = {k: torch.tensor(v, device=self.device).unsqueeze(0) 
                       for k, v in obs.items() if isinstance(v, np.ndarray)}
            action = self.policy.predict_action(obs_dict)
            if isinstance(action, dict):
                action = action['action']
            action = action.cpu().numpy().squeeze()
        
        elapsed = time.time() - start_time
        
        return action, {
            "state": state,
            "steps": num_steps,
            "time": elapsed,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "state_counts": dict(self.state_counts),
            "avg_steps": np.mean(self.step_history) if self.step_history else 0,
            "total_steps": sum(self.step_history),
            "step_history": self.step_history,
        }


# =============================================================================
# Evaluation Runner
# =============================================================================

def run_evaluation(
    policy,
    classifier,
    cfg: OmegaConf,
    step_config: Dict[str, int],
    num_episodes: int = 300,
    device: str = "cuda:0",
    max_steps: int = 400,
) -> Dict[str, Any]:
    """
    Run DASIP evaluation on environment.
    
    Returns dict with success_rate, avg_time, total_time, etc.
    """
    from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
    
    # Create environment runner
    runner_cfg = cfg.task.env_runner
    runner = hydra.utils.instantiate(runner_cfg)
    
    # Create DASIP inference engine
    dasip = DASIPInference(policy, classifier, cfg, step_config, device)
    
    results = {
        "successes": 0,
        "episodes": 0,
        "total_time": 0,
        "episode_times": [],
        "episode_rewards": [],
    }
    
    print(f"\nRunning {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs = runner.reset()
        episode_reward = 0
        episode_start = time.time()
        done = False
        step = 0
        
        while not done and step < max_steps:
            action, info = dasip.get_action(obs)
            obs, reward, done, env_info = runner.step(action)
            episode_reward += reward
            step += 1
        
        episode_time = time.time() - episode_start
        success = episode_reward > 0.5 or env_info.get("success", False)
        
        results["successes"] += int(success)
        results["episodes"] += 1
        results["total_time"] += episode_time
        results["episode_times"].append(episode_time)
        results["episode_rewards"].append(episode_reward)
        
        if (ep + 1) % 50 == 0:
            current_sr = results["successes"] / results["episodes"]
            print(f"  Episode {ep+1}/{num_episodes}: SR={current_sr:.1%}")
    
    results["success_rate"] = results["successes"] / results["episodes"]
    results["avg_time"] = np.mean(results["episode_times"])
    results["stats"] = dasip.get_stats()
    
    return results


def run_baseline(
    policy,
    cfg: OmegaConf,
    num_episodes: int = 300,
    fixed_steps: int = BASELINE_STEPS,
    device: str = "cuda:0",
    max_steps: int = 400,
) -> Dict[str, Any]:
    """Run baseline evaluation with fixed diffusion steps."""
    from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
    
    # Set fixed steps
    if hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = fixed_steps
    
    runner_cfg = cfg.task.env_runner
    runner = hydra.utils.instantiate(runner_cfg)
    
    results = {
        "successes": 0,
        "episodes": 0,
        "total_time": 0,
        "episode_times": [],
    }
    
    print(f"\nRunning baseline ({fixed_steps} steps) - {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs = runner.reset()
        episode_reward = 0
        episode_start = time.time()
        done = False
        step = 0
        
        while not done and step < max_steps:
            with torch.no_grad():
                obs_dict = {k: torch.tensor(v, device=device).unsqueeze(0) 
                           for k, v in obs.items() if isinstance(v, np.ndarray)}
                action = policy.predict_action(obs_dict)
                if isinstance(action, dict):
                    action = action['action']
                action = action.cpu().numpy().squeeze()
            
            obs, reward, done, env_info = runner.step(action)
            episode_reward += reward
            step += 1
        
        episode_time = time.time() - episode_start
        success = episode_reward > 0.5 or env_info.get("success", False)
        
        results["successes"] += int(success)
        results["episodes"] += 1
        results["total_time"] += episode_time
        results["episode_times"].append(episode_time)
        
        if (ep + 1) % 50 == 0:
            current_sr = results["successes"] / results["episodes"]
            print(f"  Episode {ep+1}/{num_episodes}: SR={current_sr:.1%}")
    
    results["success_rate"] = results["successes"] / results["episodes"]
    results["avg_time"] = np.mean(results["episode_times"])
    
    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DASIP: Adaptive State-Informed Policy Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument("--env", type=str, required=True,
                       choices=["square", "can", "lift", "pusht", "tool_hang", "transport"],
                       help="Environment name")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory")
    
    # Checkpoint options
    parser.add_argument("--infer_from", type=int, default=2000,
                       help="Epoch number to load (default: 2000)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    # Evaluation options
    parser.add_argument("--num_episodes", type=int, default=300,
                       help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=400,
                       help="Max steps per episode")
    
    # DASIP step configuration
    parser.add_argument("--steps_easy", type=int, default=2,
                       help="Diffusion steps for easy states (i, g)")
    parser.add_argument("--steps_medium", type=int, default=5,
                       help="Diffusion steps for medium states (n, s)")
    parser.add_argument("--steps_hard", type=int, default=10,
                       help="Diffusion steps for hard states (t, c)")
    
    # Comparison options
    parser.add_argument("--compare_baseline", action="store_true",
                       help="Also run baseline for comparison")
    parser.add_argument("--baseline_steps", type=int, default=10,
                       help="Fixed steps for baseline comparison")
    
    # Output options
    parser.add_argument("--output_json", type=str, default=None,
                       help="Save results to JSON file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("DASIP: Diffusion-Adaptive State-Informed Policy Inference")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Step config: easy={args.steps_easy}, medium={args.steps_medium}, hard={args.steps_hard}")
    
    # Load classifier
    print("\n" + "-" * 40)
    print("Loading classifier...")
    classifier = load_classifier(args.env)
    
    # Load checkpoint
    print("\n" + "-" * 40)
    print("Loading policy checkpoint...")
    ckpt_file, cfg = find_checkpoint(args.checkpoint_path, args.infer_from)
    print(f"Found checkpoint: {ckpt_file}")
    
    policy, workspace, cfg = load_policy(ckpt_file, cfg, args.device)
    print("Policy loaded successfully!")
    
    # Configure step counts
    step_config = {
        "easy": args.steps_easy,
        "medium": args.steps_medium,
        "hard": args.steps_hard,
    }
    
    # Run DASIP evaluation
    print("\n" + "-" * 40)
    print("Running DASIP evaluation...")
    dasip_results = run_evaluation(
        policy, classifier, cfg, step_config,
        num_episodes=args.num_episodes,
        device=args.device,
        max_steps=args.max_steps,
    )
    
    # Run baseline if requested
    baseline_results = None
    if args.compare_baseline:
        print("\n" + "-" * 40)
        baseline_results = run_baseline(
            policy, cfg,
            num_episodes=args.num_episodes,
            fixed_steps=args.baseline_steps,
            device=args.device,
            max_steps=args.max_steps,
        )
    
    # Print results
    print("\n" + "=" * 60)
    print("DASIP INFERENCE RESULTS")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.num_episodes}")
    
    print(f"\nDASIP (Adaptive):")
    print(f"  Success Rate: {dasip_results['success_rate']:.1%}")
    print(f"  Avg Time/Episode: {dasip_results['avg_time']:.2f}s")
    print(f"  Total Time: {dasip_results['total_time']:.1f}s")
    print(f"  Avg Diffusion Steps: {dasip_results['stats']['avg_steps']:.1f}")
    print(f"  State Distribution: {dasip_results['stats']['state_counts']}")
    
    if baseline_results:
        print(f"\nBaseline (Fixed {args.baseline_steps} steps):")
        print(f"  Success Rate: {baseline_results['success_rate']:.1%}")
        print(f"  Avg Time/Episode: {baseline_results['avg_time']:.2f}s")
        print(f"  Total Time: {baseline_results['total_time']:.1f}s")
        
        # Calculate speedup
        speedup = baseline_results['total_time'] / dasip_results['total_time']
        sr_diff = dasip_results['success_rate'] - baseline_results['success_rate']
        
        print(f"\nSpeedup: {speedup:.2f}x", end="")
        if sr_diff >= 0:
            print(f" (with +{sr_diff:.1%} success rate improvement)")
        else:
            print(f" (with {sr_diff:.1%} success rate reduction)")
    
    print("=" * 60)
    
    # Save results if requested
    if args.output_json:
        output = {
            "env": args.env,
            "checkpoint": args.checkpoint_path,
            "num_episodes": args.num_episodes,
            "step_config": step_config,
            "dasip": {
                "success_rate": dasip_results['success_rate'],
                "avg_time": dasip_results['avg_time'],
                "total_time": dasip_results['total_time'],
                "avg_steps": dasip_results['stats']['avg_steps'],
                "state_counts": dasip_results['stats']['state_counts'],
            },
        }
        if baseline_results:
            output["baseline"] = {
                "success_rate": baseline_results['success_rate'],
                "avg_time": baseline_results['avg_time'],
                "total_time": baseline_results['total_time'],
                "fixed_steps": args.baseline_steps,
            }
            output["speedup"] = speedup
        
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

