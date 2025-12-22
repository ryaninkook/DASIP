# SIP: Stochastic Interpolant Policy

This is the core implementation of Stochastic Interpolant Policy (SIP) for robotic manipulation tasks.

## Overview

SIP extends diffusion-based policies with Stochastic interpolant methods, enabling:
- **Flexible path types**: Linear, VP (Variance Preserving), GVP (Generalized VP), DDPM, DDIM
- **Fast inference**: Achieve strong performance with as few as 2 sampling steps
- **Classifier guidance**: Optional guided sampling for improved success rates

## Project Structure

```
SIP/
├── SI_policy/
│   ├── config/              # Hydra configuration files
│   │   └── task/            # Task-specific configs
│   ├── dataset/             # Dataset implementations
│   ├── env/                 # Environment implementations
│   │   ├── pusht/           # Push-T task
│   │   ├── robomimic/       # Robomimic tasks (square, can, lift, etc.)
│   │   ├── kitchen/         # Kitchen task
│   │   └── block_pushing/   # Block pushing task
│   ├── env_runner/          # Environment runners for evaluation
│   ├── model/               # Neural network architectures
│   │   └── diffusion/       # Diffusion/flow models
│   ├── policy/              # Policy implementations
│   │   ├── diffusion_unet_lowdim_policy_flow_euler.py  # SIP policy
│   │   ├── diffusion_unet_lowdim_policy.py             # DDPM policy
│   │   └── diffusion_unet_lowdim_policy_ddim.py        # DDIM policy
│   └── workspace/           # Training workspace classes
├── data/                    # Datasets and checkpoints
├── import_true_basics.py    # Utility imports
├── train.py                 # Original training entry point
└── eval.py                  # Original evaluation entry point
```

## Installation

### Prerequisites

Ubuntu 20.04 with Nvidia GPU. Install system dependencies:

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

### Conda Environment

Using [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (recommended):

```bash
mamba env create -f conda_environment.yaml
conda activate robodiff
```

Or with conda:

```bash
conda env create -f conda_environment.yaml
conda activate robodiff
```

### Install Package

```bash
pip install -e .
```

## Data

### Download Training Data

Create data directory:

```bash
mkdir -p data && cd data
```

Download datasets from [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/data/training/):

```bash
# Push-T
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm pusht.zip

# Robomimic tasks
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic.zip
unzip robomimic.zip && rm robomimic.zip
```

### Supported Tasks

| Task | Type | Description |
|------|------|-------------|
| `pusht` | Low-dim | Push T-block to target pose |
| `square` | Low-dim | Square nut assembly |
| `can` | Low-dim | Can picking |
| `lift` | Low-dim | Object lifting |
| `tool_hang` | Low-dim | Tool hanging |
| `transport` | Low-dim | Two-arm transport |

## Quick Start

See the main project README at `../README.md` for usage of the clean scripts in `../scripts/`.

### Training

```bash
cd ..  # Go to DASIP root
python scripts/train.py \
    --mode train \
    --env square \
    --path_type Linear \
    --save_path my_experiment \
    --num_epochs 3000
```

### Inference

```bash
python scripts/train.py \
    --mode infer \
    --env square \
    --save_path my_experiment \
    --infer_from 2000 \
    --use_classifier
```

### Evaluation

```bash
python scripts/evaluate.py \
    --env square \
    --checkpoint_path /path/to/checkpoint.ckpt \
    --num_episodes 300
```

## Configuration

### Path Types

| Path Type | Description |
|-----------|-------------|
| `Linear` | Linear interpolation (flow matching) |
| `VP` | Variance Preserving path |
| `GVP` | Generalized Variance Preserving path |
| `DDPM` | Denoising Diffusion Probabilistic Model |
| `DDIM` | Denoising Diffusion Implicit Model |

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `T` | Total diffusion timesteps | 100 |
| `TS` | Sampling steps | 10 |
| `prediction` | Prediction type (velocity, noise) | velocity |
| `ODE_SDE` | ODE or SDE formulation | ODE |
| `solver` | ODE solver (Euler, Heun) | Euler |

## Architecture

### Policy Interface

Policies follow the standard interface:

```python
class BaseLowdimPolicy:
    def predict_action(self, obs_dict) -> dict:
        """
        Args:
            obs_dict: {"obs": Tensor of shape (B, To, Do)}
        Returns:
            {"action": Tensor of shape (B, Ta, Da)}
        """
        pass
    
    def compute_loss(self, batch) -> Tensor:
        """Compute training loss."""
        pass
```

### Observation/Action Dimensions

- `To`: Observation horizon (number of past observations)
- `Ta`: Action horizon (number of actions to execute)
- `T`: Prediction horizon (total sequence length predicted)

## Classifier Guidance

SIP supports classifier-guided sampling for improved success rates:

```python
# Classifiers are stored in data/flow_label/v2/
# Format: {task}_xg_{accuracy}_c{class}_{params}.pkl
```

Enable with `--use_classifier` flag during inference.

## Citation

```bibtex
@article{sip2025,
  title={Stochastic Interpolant Policy for Robotic Manipulation},
  author={...},
  journal={...},
  year={2025}
}
```

## Acknowledgement

This codebase builds upon:
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) by Chi et al.
- [Robomimic](https://github.com/ARISE-Initiative/robomimic) benchmark
- [Flow Matching](https://arxiv.org/abs/2210.02747) methods

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
