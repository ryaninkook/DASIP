#!/usr/bin/env python3
"""
Training and Inference script for Diffusion Policy.

This script provides a unified workspace for training diffusion policies
and running inference/evaluation with video saving and wandb logging.

Usage:
    # Training
    python train.py \
        --env square \
        --save_path my_experiment \
        --path_type Linear \
        --num_epochs 3000

    # Inference
    python train.py \
        --env square \
        --save_path my_experiment \
        --mode infer \
        --infer_from 2000

Example training command:
    python train.py \
        --env square \
        --device 0 \
        --path_type Linear \
        --save_path v11_square_train_Linear \
        --seed 1000000 \
        --num_epochs 3000

Example inference command:
    python train.py \
        --env square \
        --save_path v11_square_train_Linear \
        --mode infer \
        --infer_from 2000 \
        --num_steps 10 \
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
from typing import Optional, Dict, Any, List, Union

import cv2
import torch
import hydra
import joblib
import numpy as np
import wandb
import tqdm
from omegaconf import OmegaConf
from pytz import timezone
from torch.utils.data import DataLoader

# Setup paths
ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "SIP")
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler import DiffusionUnetLowdimPolicy_flow_euler
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy_ddim import DiffusionUnetLowdimPolicy_ddim
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
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
    
    def __init__(self, window_size: int = 10):
        self.values: List[float] = []
        self.window_size = window_size
    
    def update(self, value: float) -> None:
        """Add a new value."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    @property
    def value(self) -> float:
        """Get current average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


# =============================================================================
# Media Saving
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
        for i in range(6):
            self._save_video(runner_log, i, episode_idx)
        
        log_path = os.path.join(self.log_dir, f'{episode_idx}_log.json')
        with open(log_path, 'w') as f:
            json.dump(runner_log, f, default=str)
    
    def _save_video(self, runner_log: Dict, index: int, episode_idx: int) -> None:
        """Save a single video and its observations."""
        try:
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
            
            max_reward = round(runner_log[reward_key], 2)
            video_path = runner_log[video_key]._path
            obs_torch = runner_log[obs_key]
            
            base_name = os.path.basename(video_path).replace('.mp4', '')
            new_name = f'{episode_idx}_{base_name}_{max_reward}.mp4'
            new_path = os.path.join(self.media_dir, new_name)
            
            shutil.move(video_path, new_path)
            
            obs_path = new_path.replace('.mp4', '.pt')
            torch.save(obs_torch, obs_path)
            
            self._save_observation_frames(new_path)
            
        except Exception as e:
            print(f'Error saving video {index}: {e}')
    
    def _save_observation_frames(self, video_path: str) -> None:
        """Extract and save observation frames from video."""
        pt_path = video_path.replace('.mp4', '.pt')
        if not os.path.exists(pt_path):
            return
        
        unique_name = os.path.basename(video_path).replace('.mp4', '')
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return
        
        frames = np.array(frames)
        frames = frames[..., ::-1]
        
        pt = torch.load(pt_path)
        
        if len(pt) == 0:
            return
        
        frames_per_obs = len(frames) // len(pt)
        if frames_per_obs == 0:
            return
        
        save_dir = os.path.join(self.media_dir, 'obs', unique_name)
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(len(pt)):
            obs_i = pt[i].numpy()
            start_idx = i * frames_per_obs
            end_idx = (i + 1) * frames_per_obs
            step_frames = frames[start_idx:end_idx]
            
            if len(step_frames) == 0:
                continue
            
            rep_frame = step_frames[-1]
            rep_frame_bgr = rep_frame[..., ::-1]
            
            frame_path = os.path.join(save_dir, f"obs_{i:03d}.png")
            cv2.imwrite(frame_path, rep_frame_bgr)
            
            np.save(os.path.join(save_dir, f"obs_{i:03d}.npy"), obs_i)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or run inference for Diffusion Policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer"],
        help="Run mode: train or infer",
    )
    
    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="square",
        choices=["square", "can", "lift", "pusht", "tool_hang", "transport"],
        help="Environment name",
    )
    
    # Device
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
    parser.add_argument(
        "--T",
        type=int,
        default=100,
        help="Total diffusion timesteps",
    )
    parser.add_argument(
        "--Tdiscrete",
        type=str,
        default="Tdiscrete",
        choices=["Tdiscrete", "Tconti"],
        help="Discrete or continuous time",
    )
    
    # Training settings
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Save path name for experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2000000,
        help="Random seed",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Use EMA model",
    )
    
    # Inference settings
    parser.add_argument(
        "--infer_from",
        type=int,
        default=2000,
        help="Epoch to start inference from",
    )
    parser.add_argument(
        "--use_classifier",
        action="store_true",
        help="Use classifier for guided sampling",
    )
    
    # Logging
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
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (short runs)",
    )
    
    return parser.parse_args()


# =============================================================================
# Training Workspace
# =============================================================================

class DiffusionPolicyWorkspace(BaseWorkspace):
    """Unified workspace for training and inference."""
    
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, args: argparse.Namespace):
        self.args = args
        self.is_training = args.mode == "train"
        self.timestamp = get_timestamp()
        
        # Setup directories and logging name
        self._setup_directories(cfg)
        
        super().__init__(cfg, output_dir=self.output_dir)
        
        # Set seeds
        self._set_seeds(cfg.training.seed)
        
        # Initialize model
        self._init_model(cfg)
        
        # Load classifier
        self.classifier = self._load_classifier(cfg)
        
        # Media saver for inference
        self.media_saver = MediaSaver(self.output_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def _setup_directories(self, cfg: OmegaConf) -> None:
        """Setup output directories and logging name."""
        # Build logging name based on mode and settings
        if self.is_training:
            if cfg.policy.path_type in ['DDPM', 'DDIM']:
                log_name = (
                    f"{cfg.logging.name}_{cfg.training.architecture}_"
                    f"{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}"
                )
            else:
                log_name = (
                    f"{cfg.logging.name}_{cfg.training.architecture}_"
                    f"{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_"
                    f"Teps{cfg.policy.train_eps}_Tsam{cfg.policy.Tsam}_T{cfg.policy.T}"
                )
        else:
            if cfg.policy.path_type in ['DDPM', 'DDIM']:
                log_name = (
                    f"{cfg.logging.name}_{cfg.training.architecture}_"
                    f"{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_"
                    f"{cfg.policy.ODE_SDE}_TS{cfg.policy.TS}"
                )
            else:
                log_name = (
                    f"_TS{cfg.policy.TS}_{cfg.policy.TS_list}_{cfg.logging.name}_"
                    f"{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_"
                    f"{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_"
                    f"Teps{cfg.policy.train_eps}_Seps{cfg.policy.sample_eps}_"
                    f"Ssam{cfg.policy.Ssam}_TS{cfg.policy.TS}"
                )
        
        cfg.logging.name = log_name
        
        # Setup paths
        base_dir = os.path.join('data', 'output4_train', cfg.task.name, cfg.training.save_path)
        
        if self.is_training:
            self.output_dir = base_dir
            self.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        else:
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
        
        # Setup EMA
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        
        # Optimizer
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
    
    def _load_classifier(self, cfg: OmegaConf) -> Optional[Any]:
        """Load classifier for guided sampling."""
        task_name = cfg.task.name.split('_')[0]
        classifier_path = CLASSIFIER_PATHS.get(task_name)
        
        if classifier_path and os.path.exists(classifier_path):
            print(f"Loading classifier from: {classifier_path}")
            return joblib.load(classifier_path)
        
        print(f"Warning: Classifier not found for {task_name}")
        return None
    
    def _find_checkpoints(self) -> Dict[int, str]:
        """Find all checkpoints and return epoch -> path mapping."""
        load_dict = {}
        if os.path.exists(self.checkpoint_dir):
            for item in sorted(os.listdir(self.checkpoint_dir)):
                fullpath = os.path.join(self.checkpoint_dir, item)
                if os.path.isfile(fullpath) and '.ckpt' in item and '=' in item:
                    epoch_num = int(item.split('=')[1].split('-')[0])
                    load_dict[epoch_num] = fullpath
        return load_dict
    
    def run(self) -> None:
        """Run training or inference."""
        cfg = copy.deepcopy(self.cfg)
        
        # Setup dataset
        dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        
        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) 
                // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )
        
        # EMA
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        
        # Device
        if self.args.device is not None:
            device = torch.device(f'cuda:{self.args.device}')
        else:
            device = find_empty_gpu()
        print(f"Using device: {device}")
        
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # Resume if needed
        if cfg.training.resume:
            ckpt_path = self.get_checkpoint_path()
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)
        
        # Setup env runner
        cfg.task.env_runner.test_start_seed = cfg.training.seed
        if 'push' in cfg.task.name:
            cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.pusht_keypoints_runner_classifier.PushTKeypointsRunner'
        else:
            cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.robomimic_lowdim_runner_classifier.RobomimicLowdimRunner'
        
        cfg.task.env_runner.n_envs = 1
        cfg.task.env_runner.n_train = 1 if self.is_training else 0
        cfg.task.env_runner.n_test = 2 if self.is_training else 1
        
        env_runner: BaseLowdimRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        
        # Find checkpoints for inference
        load_dict = self._find_checkpoints()
        
        # Wandb
        if not self.args.no_wandb:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                project=self.args.wandb_project or cfg.logging.project,
                name=cfg.logging.name,
            )
            wandb.config.update({"output_dir": self.output_dir})
        
        # Checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        
        # Debug mode
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
        
        # Trackers
        test_tracker = AverageTracker(10)
        train_tracker = AverageTracker(10)
        time_tracker = AverageTracker(10)
        
        # Logging
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        train_sampling_batch = None
        
        print(f"\n{'='*60}")
        print(f"Starting {'Training' if self.is_training else 'Inference'}")
        print(f"  Environment: {cfg.task.name}")
        print(f"  Epochs: {cfg.training.num_epochs}")
        print(f"  Path Type: {cfg.policy.path_type}")
        print(f"{'='*60}\n")
        
        with JsonLogger(log_path) as json_logger:
            for epoch_idx in range(cfg.training.num_epochs):
                start_time = time.time()
                step_log = {}
                
                if self.is_training:
                    # Skip if resuming
                    if cfg.training.resume and epoch_idx < self.epoch:
                        continue
                    
                    # Training loop
                    train_losses = []
                    with tqdm.tqdm(train_dataloader, desc=f"Epoch {self.epoch}", 
                                   leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch
                            
                            # Compute loss
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()
                            
                            # Optimizer step
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                            
                            # EMA update
                            if cfg.training.use_ema:
                                ema.step(self.model)
                            
                            # Logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu)
                            train_losses.append(raw_loss_cpu)
                            
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            
                            if batch_idx < len(train_dataloader) - 1:
                                json_logger.log(step_log)
                                self.global_step += 1
                            
                            if cfg.training.max_train_steps and batch_idx >= cfg.training.max_train_steps - 1:
                                break
                    
                    # Epoch average loss
                    step_log['train_loss'] = np.mean(train_losses)
                    step_log['test_mean_score'] = 0
                    step_log['train_mean_score'] = 0
                    
                    # Checkpointing
                    if epoch_idx % cfg.training.checkpoint_every == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()
                        
                        metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                        
                        if cfg.training.save_50 and self.epoch >= int(cfg.training.save_from):
                            topk_path = topk_manager.get_ckpt_path(metric_dict, True)
                            self.save_checkpoint(path=topk_path)
                        else:
                            topk_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_path is not None:
                                self.save_checkpoint(path=topk_path)
                
                else:
                    # Inference mode
                    if epoch_idx % cfg.training.checkpoint_every != 0 or epoch_idx < cfg.training.infer_from:
                        if not self.args.no_wandb:
                            wandb.log(step_log, step=epoch_idx)
                        continue
                    
                    # Load checkpoint
                    try:
                        ckpt_path = load_dict[epoch_idx]
                        print(f"Loading checkpoint: {ckpt_path}")
                        self.load_checkpoint(path=ckpt_path)
                    except KeyError:
                        print(f"Checkpoint for epoch {epoch_idx} not found, skipping")
                        continue
                    
                    # Evaluation
                    policy = self.ema_model if cfg.training.use_ema and self.ema_model else self.model
                    policy.eval()
                    
                    # Rollout
                    if epoch_idx % cfg.training.rollout_every == 0:
                        env_start = time.time()
                        
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
                        
                        env_time = time.time() - env_start
                        
                        self.media_saver.save_episode(runner_log, epoch_idx)
                        
                        test_score = round(runner_log.get('test/mean_score', 0), 2)
                        train_score = round(runner_log.get('train/mean_score', 0), 2)
                        
                        step_log['test_mean_score'] = test_score
                        step_log['train_mean_score'] = train_score
                        step_log['env_time_took'] = round(env_time, 2)
                        
                        test_tracker.update(test_score)
                        train_tracker.update(train_score)
                        time_tracker.update(env_time)
                    
                    # Validation
                    if epoch_idx % cfg.training.val_every == 0:
                        with torch.no_grad():
                            val_losses = []
                            for batch_idx, batch in enumerate(val_dataloader):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if cfg.training.max_val_steps and batch_idx >= cfg.training.max_val_steps - 1:
                                    break
                            if val_losses:
                                step_log['val_loss'] = torch.mean(torch.tensor(val_losses)).item()
                
                # End of epoch logging
                step_log['test_mean_score_last10'] = test_tracker.value
                step_log['train_mean_score_last10'] = train_tracker.value
                step_log['time_avg'] = time_tracker.value
                
                if not self.args.no_wandb:
                    wandb.log(step_log, step=epoch_idx)
                json_logger.log(step_log)
                
                self.global_step += 1
                self.epoch += 1
                
                elapsed = time.time() - start_time
                
                if epoch_idx % 10 == 0:
                    print(f"Epoch {epoch_idx}/{cfg.training.num_epochs}")
                    print(f"  Test: {step_log.get('test_mean_score', 0):.2f}, "
                          f"Avg: {test_tracker.value:.2f}")
                    print(f"  Time: {elapsed:.1f}s")
        
        print(f"\n{'='*60}")
        print(f"{'Training' if self.is_training else 'Inference'} Complete")
        print(f"  Final test avg: {test_tracker.value:.3f}")
        print(f"  Results: {self.output_dir}")
        print(f"{'='*60}\n")


# =============================================================================
# Configuration Builder
# =============================================================================

def build_config(args: argparse.Namespace) -> OmegaConf:
    """Build OmegaConf configuration from arguments."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # Load base config
    cfg = OmegaConf.load("diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml")
    
    # Task
    cfg.task = {"name": f"{args.env}_lowdim_abs"}
    task_cfg = OmegaConf.load(f"diffusion_policy/config/task/{cfg.task.name}.yaml")
    cfg = OmegaConf.merge(cfg, {"task": task_cfg})
    
    # Logging
    cfg.logging = {
        "name": f"{args.env}_lowdim_abs",
        "project": args.wandb_project or f"{'train' if args.mode == 'train' else 'infer'}_{args.env}_{args.path_type}",
    }
    
    # Training
    cfg.training.architecture = "unet"
    cfg.training.seed = args.seed
    cfg.training.num_epochs = args.num_epochs
    cfg.training.use_ema = args.use_ema
    cfg.training.training = args.mode == "train"
    cfg.training.infer_from = args.infer_from
    cfg.training.save_path = args.save_path
    cfg.training.resume = args.resume
    cfg.training.checkpoint_every = args.checkpoint_every
    cfg.training.save_50 = True
    cfg.training.save_from = 0.0
    cfg.training.debug = args.debug
    cfg.dataloader.num_workers = 0
    
    # Policy target
    if args.path_type == "DDPM":
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy"
        cfg.policy.model._target_ = "diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D"
    elif args.path_type == "DDIM":
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_lowdim_policy_ddim.DiffusionUnetLowdimPolicy_ddim"
        cfg.policy.model._target_ = "diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D"
    else:
        cfg.policy._target_ = "diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler"
        cfg.policy.model._target_ = "diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow"
    
    # Policy parameters
    cfg.policy.path_type = args.path_type
    cfg.policy.ODE_SDE = args.ODE_SDE
    cfg.policy.DDPM = args.solver
    cfg.policy.diffusion_form = "."
    cfg.policy.WSCALE = 1000
    cfg.policy.T = args.T
    cfg.policy.model.T = args.T
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
    cfg.policy.Tdiscrete = args.Tdiscrete
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
    print(f"Diffusion Policy {'Training' if args.mode == 'train' else 'Inference'}")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    cfg = build_config(args)
    workspace = DiffusionPolicyWorkspace(cfg, args)
    workspace.run()


if __name__ == "__main__":
    main()

