#!/usr/bin/env python3
"""
Evaluation script for Diffusion Policy with SIP (Score-based Interpolant Policy).

This script runs policy evaluation on robomimic environments (square, can, lift, 
tool_hang, transport) or pusht environment using trained diffusion policy checkpoints.

Usage:
    python evaluate.py --env square --checkpoint_path /path/to/checkpoint.ckpt
    
Example:
    python evaluate.py \
        --env square \
        --device 0 \
        --num_steps 2 \
        --solver Euler \
        --classifier_type cnn \
        --checkpoint_path /path/to/model.ckpt
"""

import os
import sys
import argparse
import pathlib
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import hydra
import joblib
import numpy as np
from omegaconf import OmegaConf

# Setup paths
ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "SIP")
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from process_results import ExperimentResultsProcessor


# =============================================================================
# Configuration
# =============================================================================

# Classifier paths - relative to SIP/ directory
# These files are created by scripts/classifiers/train_xgboost.py
# or can be downloaded via scripts/download_data.py --classifiers
CLASSIFIER_PATHS = {
    "square": "data/classifiers/square_xgboost.pkl",
    "can": "data/classifiers/can_xgboost.pkl",
    "lift": "data/classifiers/lift_xgboost.pkl",
    "pusht": "data/classifiers/pusht_xgboost.pkl",
    "tool_hang": "data/classifiers/tool_hang_xgboost.pkl",
    "transport": "data/classifiers/transport_xgboost.pkl",
}

DEFAULT_TS_LIST = [2, 2, 10, 50, 100, 2]


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Diffusion Policy on robotic manipulation tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment settings
    parser.add_argument(
        "--env", 
        type=str, 
        default="square",
        choices=["square", "can", "lift", "pusht", "tool_hang", "transport"],
        help="Environment name for evaluation",
    )
    
    # Device settings
    parser.add_argument(
        "--device", 
        type=int, 
        default=0,
        help="CUDA device index",
    )
    
    # Diffusion settings
    parser.add_argument(
        "--num_steps", 
        type=int, 
        default=2,
        dest="TS",
        help="Number of diffusion sampling steps (TS)",
    )
    parser.add_argument(
        "--solver", 
        type=str, 
        default="Euler",
        choices=["Euler", "Heun"],
        help="ODE/SDE solver type",
    )
    parser.add_argument(
        "--last_step_solver", 
        type=str, 
        default="Euler",
        choices=["Euler", "Tweedie"],
        help="Solver for the last diffusion step",
    )
    parser.add_argument(
        "--ode_sde", 
        type=str, 
        default="ODE",
        dest="ODE_SDE",
        choices=["ODE", "SDE"],
        help="Use ODE or SDE formulation",
    )
    parser.add_argument(
        "--path_type", 
        type=str, 
        default="Linear",
        choices=["Linear", "GVP", "VP", "DDPM", "DDIM"],
        help="Diffusion path type",
    )
    parser.add_argument(
        "--diffusion_form", 
        type=str, 
        default="",
        help="Diffusion form specification",
    )
    parser.add_argument(
        "--ts_list", 
        type=str, 
        default=str(DEFAULT_TS_LIST),
        help="List of timestep configurations (as string)",
    )
    
    # Classifier settings
    parser.add_argument(
        "--classifier_type", 
        type=str, 
        default=None,
        choices=["cnn", "vlm", None],
        help="Classifier type for guided sampling (None for no classifier)",
    )
    parser.add_argument(
        "--url", 
        type=str, 
        default="",
        help="URL for VLM classifier API",
    )
    
    # Checkpoint settings
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--seed", 
        type=int, 
        default=2000000,
        dest="infer_seed",
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=300,
        help="Number of evaluation episodes",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/eval",
        help="Base directory for evaluation outputs",
    )
    
    return parser.parse_args()


# =============================================================================
# Configuration Builder
# =============================================================================

class ConfigBuilder:
    """Builds OmegaConf configuration for evaluation."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def build(self) -> OmegaConf:
        """Build the complete configuration."""
        # Register resolvers
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        
        # Load base configuration
        cfg = OmegaConf.load("SI_policy/config/train_diffusion_unet_lowdim_workspace.yaml")
        
        # Set task configuration
        cfg.task = {"name": f"{self.args.env}_lowdim_abs"}
        task_cfg = OmegaConf.load(f"SI_policy/config/task/{cfg.task.name}.yaml")
        cfg = OmegaConf.merge(cfg, {"task": task_cfg})
        
        # Apply settings
        self._configure_logging(cfg)
        self._configure_training(cfg)
        self._configure_policy(cfg)
        self._configure_env_runner(cfg)
        
        return cfg
    
    def _configure_logging(self, cfg: OmegaConf) -> None:
        """Configure logging settings."""
        cfg.logging = {
            "name": f"{self.args.env}_lowdim_abs",
            "project": f"eval_{self.args.env}_{self.args.path_type}",
        }
    
    def _configure_training(self, cfg: OmegaConf) -> None:
        """Configure training/evaluation settings."""
        cfg.training.architecture = "unet"
        cfg.training.device = self.args.device
        cfg.training.seed = self.args.infer_seed
        cfg.training.training = True
        cfg.dataloader.num_workers = 0
        cfg.training.save_50 = True
        cfg.training.save_from = 0.0
        cfg.training.checkpoint_every = 10
        cfg.training.num_epochs = 10
        cfg.training.num_video = 3
        cfg.training.infer_from = 0
        cfg.training.resume = False
    
    def _configure_policy(self, cfg: OmegaConf) -> None:
        """Configure policy settings based on path type."""
        args = self.args
        
        # Set T and Tdiscrete based on environment and path type
        if args.env == "square" and args.path_type == "Linear":
            T, Tdiscrete = 1000, "Tconti"
        else:
            T, Tdiscrete = 100, "Tdiscrete"
        
        # Set policy target based on path type
        if args.path_type == "DDPM":
            cfg.policy._target_ = "SI_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy"
            cfg.policy.model._target_ = "SI_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D"
        elif args.path_type == "DDIM":
            cfg.policy._target_ = "SI_policy.policy.diffusion_unet_lowdim_policy_ddim.DiffusionUnetLowdimPolicy_ddim"
            cfg.policy.model._target_ = "SI_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D"
        else:
            cfg.policy._target_ = "SI_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler"
            cfg.policy.model._target_ = "SI_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow"
        
        # Apply policy parameters
        cfg.policy.path_type = args.path_type
        cfg.policy.ODE_SDE = args.ODE_SDE
        cfg.policy.DDPM = args.solver
        cfg.policy.diffusion_form = args.diffusion_form
        cfg.policy.WSCALE = 100
        cfg.policy.T = T
        cfg.policy.model.T = T
        cfg.policy.T_CHANGED = True
        cfg.policy.model.T_CHANGED = True
        cfg.policy.TS = args.TS
        cfg.policy.last_step = args.last_step_solver
        cfg.policy.last_step_size = 0.01
        cfg.policy.train_eps = 0.001
        cfg.policy.sample_eps = 0.001
        cfg.policy.prediction = "velocity"
        cfg.policy.Tsam = "uniform"
        cfg.policy.Ssam = "uniform"
        cfg.policy.Tdiscrete = Tdiscrete
        cfg.policy.model.t_minus_one = "t_minus_one"
        cfg.policy.beta_params = "_"
        cfg.policy.beta_scaling = "_"
        
        # Set TS list for non-DDPM/DDIM methods
        if args.path_type not in ["DDPM", "DDIM"]:
            cfg.policy.TS_list = eval(args.ts_list)
    
    def _configure_env_runner(self, cfg: OmegaConf) -> None:
        """Configure environment runner."""
        args = self.args
        output_dir = os.path.join(args.output_dir, args.env, self.timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set runner target based on environment
        if args.env == "pusht":
            cfg.task.env_runner._target_ = "SI_policy.env_runner.pusht_keypoints_runner_classifier_total.PushTKeypointsRunner"
        else:
            cfg.task.env_runner.dataset_path = f"./data/robomimic/datasets/{args.env}/ph/low_dim_abs.hdf5"
            cfg.task.env_runner._target_ = "SI_policy.env_runner.robomimic_lowdim_runner_classifier_total.RobomimicLowdimRunner"
        
        cfg.task.env_runner.output_dir = output_dir
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_train_vis = 0
        cfg.task.env_runner.n_test = 1
        cfg.task.env_runner.n_test_vis = 1
        cfg.task.env_runner.n_envs = 1
        cfg.task.env_runner.test_start_seed = args.infer_seed


# =============================================================================
# Model Loader
# =============================================================================

class ModelLoader:
    """Handles loading of policy models from checkpoints."""
    
    @staticmethod
    def load_policy(cfg: OmegaConf, checkpoint_path: str, device: int) -> Any:
        """Load policy from checkpoint."""
        # Instantiate model and policy
        model = hydra.utils.instantiate(cfg.policy.model)
        policy = hydra.utils.instantiate(cfg.policy)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{device}")
        
        if "state_dicts" in checkpoint:
            state_dict_keys = list(checkpoint["state_dicts"].keys())
            print(f"Available keys in checkpoint: {state_dict_keys}")
            
            if state_dict_keys:
                model_state_dict = checkpoint["state_dicts"][state_dict_keys[0]]
                policy.load_state_dict(model_state_dict, strict=False)
        else:
            policy.load_state_dict(checkpoint, strict=False)
        
        policy = policy.to(f"cuda:{device}")
        return policy
    
    @staticmethod
    def load_classifier(env: str, classifier_type: Optional[str]) -> Optional[Any]:
        """Load classifier for guided sampling."""
        if classifier_type not in ["cnn", "vlm"]:
            return None
        
        classifier_path = CLASSIFIER_PATHS.get(env)
        if classifier_path and os.path.exists(classifier_path):
            return joblib.load(classifier_path)
        
        print(f"Warning: Classifier path not found for {env}")
        return None


# =============================================================================
# Evaluator
# =============================================================================

class PolicyEvaluator:
    """Runs policy evaluation on environments."""
    
    def __init__(
        self,
        cfg: OmegaConf,
        policy: Any,
        classifier: Optional[Any],
        args: argparse.Namespace,
    ):
        self.cfg = cfg
        self.policy = policy
        self.classifier = classifier
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run(self) -> str:
        """Run evaluation and return output folder path."""
        classifier_bool = self.args.classifier_type in ["cnn", "vlm"]
        param_folder = self._get_param_folder()
        output_folder = os.path.join(
            self.args.output_dir,
            self.args.env,
            self.timestamp,
            param_folder,
        )
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting evaluation")
        print(f"  Environment: {self.args.env}")
        print(f"  Classifier: {self.args.classifier_type}")
        print(f"  Sampling steps: {self.args.TS}")
        print(f"  Solver: {self.args.solver}/{self.args.last_step_solver}")
        print(f"  Output: {output_folder}")
        print(f"{'='*60}\n")
        
        for episode in range(self.args.num_episodes):
            try:
                env_runner = hydra.utils.instantiate(
                    self.cfg.task.env_runner,
                    output_dir=self.cfg.task.env_runner.output_dir,
                )
                
                seed_number = self.cfg.task.env_runner.test_start_seed
                env_runner.run(
                    self.policy,
                    self.classifier,
                    classifier_bool,
                    self.args.TS,
                    self.timestamp,
                    seed_number,
                    output_folder,
                    self.args.env,
                    self.args.url,
                    self.args.classifier_type,
                )
                
                self.cfg.task.env_runner.test_start_seed += 1
                
                if (episode + 1) % 10 == 0:
                    print(f"Completed episode {episode + 1}/{self.args.num_episodes}")
                    
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                break
        
        return output_folder
    
    def _get_param_folder(self) -> str:
        """Generate parameter folder name."""
        return (
            f"{self.args.classifier_type or 'none'}_"
            f"TS{self.args.TS}_"
            f"{self.args.ODE_SDE}_"
            f"{self.args.solver}_"
            f"{self.args.last_step_solver}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("Diffusion Policy Evaluation")
    print("="*60)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Build configuration
    config_builder = ConfigBuilder(args)
    cfg = config_builder.build()
    
    # Load model
    print("Loading policy from checkpoint...")
    policy = ModelLoader.load_policy(cfg, args.checkpoint_path, args.device)
    print("Policy loaded successfully.")
    
    # Load classifier if needed
    classifier = ModelLoader.load_classifier(args.env, args.classifier_type)
    if classifier:
        print(f"Loaded {args.classifier_type} classifier.")
    
    # Run evaluation
    evaluator = PolicyEvaluator(cfg, policy, classifier, args)
    output_folder = evaluator.run()
    
    # Process and save results
    print("\nProcessing results...")
    processor = ExperimentResultsProcessor()
    processor.process_file(os.path.join(output_folder, "time_taken.txt"))
    
    print(f"\nEvaluation complete. Results saved to: {output_folder}")


if __name__ == "__main__":
    main()

