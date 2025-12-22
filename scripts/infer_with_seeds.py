#!/usr/bin/env python3
"""
Multi-seed inference script for Diffusion Policy evaluation.

This script runs policy inference across multiple seeds, saves videos,
observations, and logs results to Weights & Biases.

Usage:
    python infer_with_seeds.py \
        --env square \
        --checkpoint_path /path/to/checkpoints/ \
        --infer_from 2000 \
        --num_seeds 300

Example with full options:
    python infer_with_seeds.py \
        --env square \
        --device 0 \
        --num_steps 10 \
        --solver Euler \
        --path_type Linear \
        --checkpoint_path /path/to/experiment/checkpoints/ \
        --infer_from 2000 \
        --num_seeds 300 \
        --use_classifier
"""

import os
import sys
import copy
import json
import time
import shutil
import random
import argparse
import pathlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import cv2
import torch
import hydra
import joblib
import numpy as np
import wandb
from omegaconf import OmegaConf
from pytz import timezone

# Setup paths
ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "SIP")
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from torch.utils.data import DataLoader
from SI_policy.common.pytorch_util import dict_apply, optimizer_to
from SI_policy.workspace.base_workspace import BaseWorkspace
from SI_policy.policy.diffusion_unet_lowdim_policy_flow_euler import DiffusionUnetLowdimPolicy_flow_euler
from SI_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from SI_policy.policy.diffusion_unet_lowdim_policy_ddim import DiffusionUnetLowdimPolicy_ddim
from SI_policy.dataset.base_dataset import BaseLowdimDataset
from SI_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from SI_policy.common.checkpoint_util import TopKCheckpointManager
from SI_policy.common.json_logger import JsonLogger
from SI_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel


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

DEFAULT_TS_LIST = [10, 10, 10, 10, 10, 10, 10]


# =============================================================================
# Utility Functions
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now(timezone('US/Eastern')).strftime("%Y%m%d_%H%M%S")


def find_empty_gpu() -> torch.device:
    """Find an available GPU with most free memory."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    # Get GPU with most free memory
    max_free = 0
    best_gpu = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free = props.total_memory - torch.cuda.memory_allocated(i)
        if free > max_free:
            max_free = free
            best_gpu = i
    
    return torch.device(f'cuda:{best_gpu}')


class AverageTracker:
    """Track running average of values."""
    
    def __init__(self, max_size: int = 50000):
        self.values: List[float] = []
        self.max_size = max_size
    
    def update(self, value: float) -> None:
        """Add a new value."""
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values.pop(0)
    
    @property
    def value(self) -> float:
        """Get current average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


# =============================================================================
# Video and Observation Saving
# =============================================================================

class MediaSaver:
    """Handles saving of videos and observations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.media_dir = os.path.join(output_dir, 'media')
        self.log_dir = os.path.join(self.media_dir, 'log')
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def save_episode(self, runner_log: Dict, episode_idx: int) -> None:
        """Save video, observations, and log for an episode."""
        # Save videos
        for i in range(6):
            self._save_video(runner_log, i, episode_idx)
        
        # Save log
        log_path = os.path.join(self.log_dir, f'{episode_idx}_log.json')
        with open(log_path, 'w') as f:
            json.dump(runner_log, f, default=str)
    
    def _save_video(self, runner_log: Dict, index: int, episode_idx: int) -> None:
        """Save a single video and its observations."""
        try:
            # Determine keys based on train/test
            if index < 2:
                video_key = f'train/sim_video_{index}'
                obs_key = f'train/sim_video_{index}_obs'
                reward_key = f'train/sim_max_reward_{index}'
            else:
                video_key = f'test/sim_video_10000{index-2}'
                obs_key = f'train/sim_video_10000{index-2}_obs'
                reward_key = f'test/sim_max_reward_10000{index-2}'
            
            if video_key not in runner_log:
                return
            
            # Get data
            max_reward = round(runner_log[reward_key], 2)
            video_path = runner_log[video_key]._path
            obs_torch = runner_log[obs_key]
            
            # Create new filename with reward
            base_name = os.path.basename(video_path).replace('.mp4', '')
            new_name = f'{episode_idx}_{base_name}_{max_reward}.mp4'
            new_path = os.path.join(self.media_dir, new_name)
            
            # Move video
            shutil.move(video_path, new_path)
            
            # Save observations
            obs_path = new_path.replace('.mp4', '.pt')
            torch.save(obs_torch, obs_path)
            
            # Save observation frames
            self._save_observation_frames(new_path)
            
        except Exception as e:
            print(f'Error saving video {index}: {e}')
    
    def _save_observation_frames(self, video_path: str) -> None:
        """Extract and save observation frames from video."""
        pt_path = video_path.replace('.mp4', '.pt')
        if not os.path.exists(pt_path):
            return
        
        unique_name = os.path.basename(video_path).replace('.mp4', '')
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        frames = frames[..., ::-1]  # BGR to RGB
        
        # Read observations
        pt = torch.load(pt_path)
        
        if len(frames) == 0 or len(pt) == 0:
            return
        
        frames_per_obs = len(frames) // len(pt)
        if frames_per_obs == 0:
            return
        
        # Create save directories
        save_dir = os.path.join(self.media_dir, 'obs', unique_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save frames and observations
        for i in range(len(pt)):
            obs_i = pt[i].numpy()
            start_idx = i * frames_per_obs
            end_idx = (i + 1) * frames_per_obs
            step_frames = frames[start_idx:end_idx]
            
            if len(step_frames) == 0:
                continue
            
            # Save last frame of chunk
            rep_frame = step_frames[-1]
            rep_frame_bgr = rep_frame[..., ::-1]
            
            frame_path = os.path.join(save_dir, f"obs_{i:03d}.png")
            cv2.imwrite(frame_path, rep_frame_bgr)
            
            obs_path = os.path.join(save_dir, f"obs_{i:03d}.npy")
            np.save(obs_path, obs_i)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-seed inference for Diffusion Policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment settings
    parser.add_argument(
        "--env",
        type=str,
        default="square",
        choices=["square", "can", "lift", "pusht", "tool_hang", "transport"],
        help="Environment name",
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device index (auto-select if not specified)",
    )
    
    # Diffusion settings
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        dest="TS",
        help="Number of diffusion sampling steps",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="Euler",
        choices=["Euler", "Heun"],
        help="ODE solver type",
    )
    parser.add_argument(
        "--path_type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "GVP2", "VP", "DDPM", "DDIM"],
        help="Diffusion path type",
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
        "--ts_list",
        type=str,
        default=str(DEFAULT_TS_LIST),
        help="List of timestep configurations",
    )
    
    # Checkpoint settings
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory or save_path name",
    )
    parser.add_argument(
        "--infer_from",
        type=int,
        default=2000,
        help="Epoch number to load checkpoint from",
    )
    
    # Inference settings
    parser.add_argument(
        "--seed",
        type=int,
        default=2000000,
        help="Starting random seed",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=300,
        help="Number of seeds/episodes to run",
    )
    parser.add_argument(
        "--use_classifier",
        action="store_true",
        help="Use classifier for guided sampling",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Use EMA model for inference",
    )
    
    # Logging settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    
    return parser.parse_args()


# =============================================================================
# Inference Workspace
# =============================================================================

class InferenceWorkspace(BaseWorkspace):
    """Workspace for running multi-seed inference."""
    
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, args: argparse.Namespace):
        self.args = args
        self.timestamp = get_timestamp()
        
        # Setup output directories
        self._setup_directories(cfg)
        
        super().__init__(cfg, output_dir=self.output_dir)
        
        # Set random seeds
        self._set_seeds(cfg.training.seed)
        
        # Initialize model
        self._init_model(cfg)
        
        # Load classifier if needed
        self.classifier = self._load_classifier(cfg)
        
        # Initialize media saver
        self.media_saver = MediaSaver(self.output_dir)
    
    def _setup_directories(self, cfg: OmegaConf) -> None:
        """Setup output directories."""
        # Build logging name
        if cfg.policy.path_type in ['DDPM', 'DDIM']:
            log_name = (
                f"_TS{cfg.policy.TS}_{cfg.logging.name}_"
                f"{cfg.training.architecture}_{cfg.policy.DDPM}_"
                f"{cfg.policy.path_type}_{cfg.policy.prediction}_"
                f"{cfg.policy.ODE_SDE}_TS{cfg.policy.TS}"
            )
        else:
            log_name = (
                f"_TS{cfg.policy.TS}_{cfg.policy.TS_list}_{cfg.logging.name}_"
                f"{cfg.training.architecture}_{cfg.policy.DDPM}_"
                f"{cfg.policy.path_type}_{cfg.policy.prediction}_"
                f"{cfg.policy.ODE_SDE}_Teps{cfg.policy.train_eps}_"
                f"Seps{cfg.policy.sample_eps}_Ssam{cfg.policy.Ssam}_"
                f"TS{cfg.policy.TS}_{cfg.training.infer_from}"
            )
        
        cfg.logging.name = log_name
        
        # Setup paths
        base_dir = os.path.join('data', 'output4_train', cfg.task.name, cfg.training.save_path)
        self.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        self.output_dir = os.path.join(base_dir, 'media', f'{self.timestamp}_{log_name}')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'media'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'media', 'log'), exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _init_model(self, cfg: OmegaConf) -> None:
        """Initialize the policy model."""
        if cfg.policy.path_type == 'DDPM':
            self.model: DiffusionUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        elif cfg.policy.path_type == 'DDIM':
            # Configure DDIM scheduler
            OmegaConf.set_struct(cfg.policy.noise_scheduler, False)
            if 'variance_type' in cfg.policy.noise_scheduler:
                cfg.policy.noise_scheduler.pop('variance_type')
            OmegaConf.set_struct(cfg.policy.noise_scheduler, True)
            
            cfg.policy.noise_scheduler._target_ = 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'
            cfg.policy.noise_scheduler.num_train_timesteps = 100
            cfg.policy.noise_scheduler.beta_start = 0.0001
            cfg.policy.noise_scheduler.beta_end = 0.02
            cfg.policy.noise_scheduler.beta_schedule = 'squaredcos_cap_v2'
            cfg.policy.noise_scheduler.clip_sample = True
            
            self.model: DiffusionUnetLowdimPolicy_ddim = hydra.utils.instantiate(cfg.policy)
            self.model.num_inference_steps = cfg.policy.TS
        else:
            self.model: DiffusionUnetLowdimPolicy_flow_euler = hydra.utils.instantiate(cfg.policy)
        
        # Setup EMA model
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        
        # Setup optimizer (needed for checkpoint loading)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0
    
    def _load_classifier(self, cfg: OmegaConf) -> Optional[Any]:
        """Load the classifier for guided sampling."""
        if not self.args.use_classifier:
            return None
        
        task_name = cfg.task.name.split('_')[0]
        classifier_path = CLASSIFIER_PATHS.get(task_name)
        
        if classifier_path and os.path.exists(classifier_path):
            print(f"Loading classifier from: {classifier_path}")
            return joblib.load(classifier_path)
        
        print(f"Warning: Classifier not found for {task_name}")
        return None
    
    def _find_checkpoint(self, epoch: int) -> str:
        """Find checkpoint file for given epoch."""
        checkpoints = os.listdir(self.checkpoint_dir)
        checkpoints = sorted(checkpoints)
        
        for ckpt in checkpoints:
            if '.ckpt' in ckpt and '=' in ckpt:
                ckpt_epoch = int(ckpt.split('=')[1].split('-')[0])
                if ckpt_epoch == epoch:
                    return os.path.join(self.checkpoint_dir, ckpt)
        
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found in {self.checkpoint_dir}")
    
    def run(self) -> None:
        """Run multi-seed inference."""
        cfg = copy.deepcopy(self.cfg)
        
        # Configure dataset and normalizer
        dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        
        # Setup device
        if self.args.device is not None:
            device = torch.device(f'cuda:{self.args.device}')
        else:
            device = find_empty_gpu()
        print(f"Using device: {device}")
        
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        
        # Load checkpoint
        ckpt_path = self._find_checkpoint(cfg.training.infer_from)
        print(f"Loading checkpoint: {ckpt_path}")
        self.load_checkpoint(path=ckpt_path)
        
        # Configure env runner
        cfg.task.env_runner.test_start_seed = cfg.training.seed
        if 'push' in cfg.task.name:
            cfg.task.env_runner._target_ = 'SI_policy.env_runner.pusht_keypoints_runner_classifier.PushTKeypointsRunner'
        else:
            cfg.task.env_runner._target_ = 'SI_policy.env_runner.robomimic_lowdim_runner_classifier.RobomimicLowdimRunner'
        
        # Initialize wandb
        if not self.args.no_wandb:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                project=self.args.wandb_project or cfg.logging.project,
                name=cfg.logging.name,
            )
            wandb.config.update({"output_dir": self.output_dir})
        
        # Initialize trackers
        test_success_tracker = AverageTracker()
        train_success_tracker = AverageTracker()
        time_tracker = AverageTracker()
        
        # Setup logging
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        
        print(f"\n{'='*60}")
        print(f"Starting inference")
        print(f"  Environment: {cfg.task.name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Seeds: {self.args.num_seeds}")
        print(f"  Classifier: {self.args.use_classifier}")
        print(f"{'='*60}\n")
        
        with JsonLogger(log_path) as json_logger:
            for seed_idx in range(self.args.num_seeds):
                start_time = time.time()
                
                # Configure environment runner for this seed
                cfg.task.env_runner.n_envs = 1
                cfg.task.env_runner.n_train = 0
                cfg.task.env_runner.n_test = 1
                
                env_runner: BaseLowdimRunner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir
                )
                cfg.task.env_runner.test_start_seed += 1
                
                # Get policy for evaluation
                policy = self.ema_model if cfg.training.use_ema and self.ema_model else self.model
                policy.eval()
                
                # Run episode
                env_start = time.time()
                try:
                    if 'pusht' in cfg.task.name:
                        runner_log = env_runner.run(
                            policy, self.classifier, self.args.use_classifier,
                            cfg.policy.TS, getattr(cfg.training, 'pusht_heun', False)
                        )
                    else:
                        runner_log = env_runner.run(
                            policy, self.classifier, self.args.use_classifier,
                            cfg.policy.TS
                        )
                except Exception as e:
                    print(f"Error in episode {seed_idx}: {e}")
                    continue
                
                env_time = time.time() - env_start
                
                # Save media
                self.media_saver.save_episode(runner_log, seed_idx)
                
                # Extract results
                test_score = runner_log.get('test/mean_score', 0)
                train_score = runner_log.get('train/mean_score', 0)
                
                # Update trackers
                test_success_tracker.update(test_score)
                train_success_tracker.update(train_score)
                time_tracker.update(env_time)
                
                # Log
                step_log = {
                    'test_mean_score': round(test_score, 2),
                    'train_mean_score': round(train_score, 2),
                    'env_time_took': round(env_time, 2),
                    'test_mean_score_avg': test_success_tracker.value,
                    'train_mean_score_avg': train_success_tracker.value,
                    'time_avg': time_tracker.value,
                }
                
                if not self.args.no_wandb:
                    wandb.log(step_log, step=seed_idx)
                json_logger.log(step_log)
                
                # Print progress
                elapsed = time.time() - start_time
                if seed_idx % 10 == 0:
                    print(f"Episode {seed_idx}/{self.args.num_seeds}")
                    print(f"  Test score: {test_score:.2f}, Avg: {test_success_tracker.value:.2f}")
                    print(f"  Time: {elapsed:.1f}s")
        
        print(f"\n{'='*60}")
        print(f"Inference complete")
        print(f"  Final average test score: {test_success_tracker.value:.3f}")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")


# =============================================================================
# Configuration Builder
# =============================================================================

def build_config(args: argparse.Namespace) -> OmegaConf:
    """Build OmegaConf configuration from arguments."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # Load base config
    cfg = OmegaConf.load("SI_policy/config/train_diffusion_unet_lowdim_workspace.yaml")
    
    # Set task
    cfg.task = {"name": f"{args.env}_lowdim_abs"}
    task_cfg = OmegaConf.load(f"SI_policy/config/task/{cfg.task.name}.yaml")
    cfg = OmegaConf.merge(cfg, {"task": task_cfg})
    
    # Logging
    cfg.logging = {
        "name": f"{args.env}_lowdim_abs",
        "project": args.wandb_project or f"infer_{args.env}_{args.path_type}",
    }
    
    # Training settings
    cfg.training.architecture = "unet"
    cfg.training.seed = args.seed
    cfg.training.num_epochs = args.num_seeds
    cfg.training.use_ema = args.use_ema
    cfg.training.training = False
    cfg.training.infer_from = args.infer_from
    cfg.training.save_path = args.checkpoint_path
    cfg.dataloader.num_workers = 0
    
    # Policy target based on path type
    if args.path_type == "DDPM":
        cfg.policy._target_ = "SI_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy"
        cfg.policy.model._target_ = "SI_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D"
    elif args.path_type == "DDIM":
        cfg.policy._target_ = "SI_policy.policy.diffusion_unet_lowdim_policy_ddim.DiffusionUnetLowdimPolicy_ddim"
        cfg.policy.model._target_ = "SI_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D"
    else:
        cfg.policy._target_ = "SI_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler"
        cfg.policy.model._target_ = "SI_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow"
    
    # Policy parameters
    cfg.policy.path_type = args.path_type
    cfg.policy.ODE_SDE = args.ODE_SDE
    cfg.policy.DDPM = args.solver
    cfg.policy.diffusion_form = "."
    cfg.policy.WSCALE = 1000
    cfg.policy.T = 100
    cfg.policy.model.T = 100
    cfg.policy.T_CHANGED = True
    cfg.policy.model.T_CHANGED = True
    cfg.policy.TS = args.TS
    cfg.policy.last_step = "."
    cfg.policy.last_step_size = "."
    cfg.policy.train_eps = 0
    cfg.policy.sample_eps = 0
    cfg.policy.prediction = "velocity"
    cfg.policy.Tsam = "uniform"
    cfg.policy.Ssam = "uniform"
    cfg.policy.Tdiscrete = "Tdiscrete"
    cfg.policy.model.t_minus_one = "t_minus_one"
    cfg.policy.beta_params = "_"
    cfg.policy.beta_scaling = "_"
    
    if args.path_type not in ["DDPM", "DDIM"]:
        cfg.policy.TS_list = eval(args.ts_list)
    
    return cfg


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("Multi-Seed Inference for Diffusion Policy")
    print("="*60)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Build configuration
    cfg = build_config(args)
    
    # Create and run workspace
    workspace = InferenceWorkspace(cfg, args)
    workspace.run()


if __name__ == "__main__":
    main()

