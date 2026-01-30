# DASIP: Difficulty-Aware Stochastic Interpolant Policy

**Dynamic Test-Time Compute Scaling in Control Policy**

Fast diffusion/flow policy inference through adaptive computation based on difficulty classification.

[![arXiv](https://img.shields.io/badge/arXiv-2511.20906-b31b1b.svg)](https://arxiv.org/abs/2511.20906)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Inkook Chun†, Seungjae Lee‡, Michael S. Albergo⋄, Saining Xie†, Eric Vanden-Eijnden†,§

†New York University, ‡University of Maryland, ⋄Harvard University, §Capital Fund Management

## Overview

![DASIP Overview](assets/intro.png)

Diffusion and flow policies deliver state-of-the-art performance on long-horizon robotic manipulation and imitation learning tasks. However, these controllers employ a fixed inference budget at every control step, regardless of task complexity—wasting computation on simple subtasks while potentially underperforming on challenging ones.

**DASIP** addresses this by enabling robotic controllers to adaptively adjust their integration horizon in real time based on task difficulty. Our approach employs a difficulty classifier that analyzes observations to dynamically select an inference configuration triple:

**⟨step count, solver type, ODE/SDE integration⟩**

This is analogous to "System 1 vs. System 2" thinking in cognitive science—fast, intuitive processing for easy states and deliberate, effortful computation for hard states.

## Key Idea

Standard diffusion policies use a fixed number of denoising steps for every state. DASIP uses a learned difficulty classifier to identify the current robot state and adapts the inference configuration accordingly:

| State | Description | Difficulty | Diffusion Steps |
|-------|-------------|------------|-----------------|
| Initialize | Robot at starting position | Easy | 2 |
| Near | Robot approaching object | Medium | 5 |
| Attempt | Robot grabbing object | Hard | 10 |
| Pushing | Robot manipulating object | Hard | 10 |
| Holding | Robot holding object | Medium | 5 |
| Goal | Task completed | Easy | 2 |

This achieves **2.6–4.4× speedup** while maintaining task success rates comparable to fixed maximum-computation baselines.

## Stochastic Interpolant Framework

DASIP is grounded in the **stochastic interpolant (SI) framework**, which unifies flow-based (ODE) and diffusion-based (SDE) generative modeling. This formulation enables:

- **Flexible training configurations:** Choose prediction target (noise, score, velocity) and interpolant type (Linear, VP, GVP, DDPM)
- **Adaptive inference:** Translate difficulty levels into optimized inference configurations without retraining
- **ODE/SDE switching:** Dynamically select deterministic or stochastic integration based on task requirements

### Supported Configurations

| Training | Options |
|----------|---------|
| Prediction Target | Noise, Score, Velocity |
| Interpolant Type | Linear, VP (Variance Preserving), GVP (Generalized VP), DDPM |

| Inference | Options |
|-----------|---------|
| Integration Mode | ODE (deterministic), SDE (stochastic) |
| Solver Type | Euler, Heun, DPM++, etc. |
| Step Count | 1–20 (adaptive based on difficulty) |

## Difficulty Classifiers

DASIP supports three complementary classifier architectures:

| Classifier | Description | Use Case |
|------------|-------------|----------|
| **XGBoost** | Lightweight, fast inference on state observations | Default, production |
| **CNN** | Image-based classification | Visual observations |
| **VLM (Few-shot)** | Vision-language model with exemplar prompts | Zero-shot transfer |
| **VLM (Fine-tuned)** | VLM fine-tuned on difficulty annotations | Highest accuracy |

## Quick Start

```bash
# 1. Install
conda env create -f SIP/conda_environment.yaml
conda activate SI_policy
cd SIP && pip install -e . && cd ..

# 2. Download data
python scripts/download_data.py --all

# 3. Train classifier
python scripts/classifiers/train_xgboost.py --envs square can lift

# 4. Train stochastic interpolant policy
./scripts/run_train.sh --env square --save_path my_square_policy --num_epochs 3000

# 5. Run DASIP inference
./scripts/run_dasip.sh --env square --checkpoint my_square_policy --compare_baseline
```

## Project Structure

```
DASIP/
├── README.md
├── assets/                     # Documentation assets
│   ├── intro.png               # Overview figure
│   ├── DASIP poster.pdf        # Project poster
│   └── overview.pdf            # Architecture diagram
├── scripts/
│   ├── download_data.py        # Download datasets and models
│   ├── train.py                # Train stochastic interpolant policy
│   ├── dasip_infer.py          # DASIP adaptive inference
│   ├── evaluate.py             # Standard evaluation
│   ├── infer_with_seeds.py     # Multi-seed inference
│   ├── process_results.py      # Result processing
│   ├── run_train.sh            # Training shell wrapper
│   ├── run_dasip.sh            # DASIP inference wrapper
│   └── classifiers/            # Classifier training
│       ├── train_xgboost.py    # XGBoost/Random Forest
│       ├── train_cnn.py        # CNN classifier
│       ├── vlm_finetune.py     # VLM fine-tuning
│       └── models/             # VLM wrappers
├── SIP/                        # Core implementation
│   ├── SI_policy/              # Policy models
│   │   ├── config/             # Hydra configs
│   │   ├── model/              # Neural networks
│   │   ├── policy/             # Policy classes
│   │   └── env_runner/         # Environment runners
│   └── data/                   # Data directory
│       ├── pusht/              # PushT dataset
│       ├── robomimic/          # Robomimic datasets
│       ├── checkpoints/        # Trained SIP models
│       └── classifiers/        # Trained classifiers
└── flow_label/                 # Classifier training data
    └── data/
```

## Complete Pipeline

### Step 1: Download Data

```bash
# Download all data (datasets + classifiers)
python scripts/download_data.py --all

# Or download selectively
python scripts/download_data.py --datasets              # Training datasets
python scripts/download_data.py --datasets --tasks pusht square  # Specific tasks
python scripts/download_data.py --classifiers           # Classifier models
```

### Step 2: Train Difficulty Classifier

#### XGBoost (Recommended)

Fast, lightweight, works well with state observations:

```bash
python scripts/classifiers/train_xgboost.py \
    --data_path flow_label/data/good_4_user_filtered.csv \
    --output_dir SIP/data/classifiers \
    --grid_search
```

#### CNN

For image-based difficulty classification:

```bash
python scripts/classifiers/train_cnn.py \
    --data_path flow_label/data/good_4_user_filtered.csv \
    --output_dir SIP/data/classifiers \
    --env square
```

#### VLM Fine-tuning

For high-accuracy vision-language classification:

```bash
python scripts/classifiers/vlm_finetune.py \
    --train_envs can square tool_hang transport \
    --test_env lift \
    --epochs 5
```

See `scripts/classifiers/README.md` for details.

### Step 3: Train Stochastic Interpolant Policy

```bash
# Train with Linear interpolant (fastest)
./scripts/run_train.sh \
    --env square \
    --save_path v11_square_Linear \
    --path_type Linear \
    --num_epochs 3000

# Train with VP (Variance Preserving) interpolant
./scripts/run_train.sh \
    --env square \
    --save_path v11_square_VP \
    --path_type VP \
    --num_epochs 3000

# Train with GVP (Generalized VP) interpolant
./scripts/run_train.sh \
    --env square \
    --save_path v11_square_GVP \
    --path_type GVP \
    --num_epochs 3000

# Training options
./scripts/run_train.sh --help
```

### Step 4: Run DASIP Inference

```bash
# Basic DASIP inference
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v11_square_Linear

# Compare with fixed-step baseline
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v11_square_Linear \
    --compare_baseline

# Custom step schedule
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v11_square_Linear \
    --steps_easy 1 \
    --steps_medium 3 \
    --steps_hard 8

# Save results to JSON
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v11_square_Linear \
    --compare_baseline \
    --output results.json
```

### Step 5: Process Results

```bash
# Generate CSV summary from evaluation logs
python scripts/process_results.py \
    --input_dir SIP/outputs \
    --output results.csv
```

## Configuration Options

### Stochastic Interpolant Policy Training

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | `square` |
| `--path_type` | Interpolant type (Linear, GVP, VP, DDPM) | `Linear` |
| `--prediction` | Prediction target (velocity, noise, score) | `velocity` |
| `--num_epochs` | Training epochs | `3000` |
| `--checkpoint_every` | Save frequency | `10` |
| `--seed` | Random seed | `2000000` |

### DASIP Inference

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | required |
| `--checkpoint` | Checkpoint path | required |
| `--steps_easy` | Steps for easy states (initialize, goal) | `2` |
| `--steps_medium` | Steps for medium states (near, holding) | `5` |
| `--steps_hard` | Steps for hard states (attempt, pushing) | `10` |
| `--integration` | Integration mode (ode, sde) | `ode` |
| `--compare_baseline` | Run baseline comparison | `false` |
| `--num_episodes` | Evaluation episodes | `300` |

### Supported Environments

| Environment | Task | Difficulty States |
|-------------|------|-------------------|
| `square` | Square nut assembly | i, n, t, c, s, g |
| `can` | Can picking | i, n, t, c, s, g |
| `lift` | Object lifting | i, n, t, c, s, g |
| `pusht` | Push-T | i, n, t, c, s, g |
| `tool_hang` | Tool hanging | i, n, t, c, s, g |
| `transport` | Two-arm transport | i, n, t, c, s, g |

**Difficulty state codes:** i=Initialize, n=Near, t=aTtempt, c=Catching/pushing, s=Steady/holding, g=Goal

## Results

DASIP achieves significant computational efficiency gains while maintaining task success rates:

| Environment | Method | Success Rate | Avg Steps | Speedup |
|-------------|--------|-------------|-----------|---------|
| Square | Baseline (10 steps) | 95.0% | 10.0 | 1.0× |
| Square | DASIP (adaptive) | 94.3% | 4.2 | **2.4×** |
| Can | Baseline (10 steps) | 92.0% | 10.0 | 1.0× |
| Can | DASIP (adaptive) | 91.5% | 2.8 | **3.6×** |
| Lift | Baseline (10 steps) | 98.0% | 10.0 | 1.0× |
| Lift | DASIP (adaptive) | 97.5% | 2.3 | **4.4×** |

**Overall:** 2.6–4.4× reduction in total computation time while maintaining comparable success rates.

## Manual Data Download

If automatic download fails:

| Data | URL | Destination |
|------|-----|-------------|
| PushT | [download](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip) | `SIP/data/` |
| Robomimic | [download](https://diffusion-policy.cs.columbia.edu/data/training/robomimic.zip) | `SIP/data/` |
| Flow Label | [Dropbox](https://www.dropbox.com/scl/fi/y72ccuu6lv3sdztqau0a3/data-flow-label.zip?rlkey=buxwsuhs7n1ccm27h9uhc46qv&dl=0) | `flow_label/` |

```bash
# Example manual download
wget -O pusht.zip "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
unzip pusht.zip -d SIP/data/

wget -O flow_label.zip "https://www.dropbox.com/scl/fi/y72ccuu6lv3sdztqau0a3/data-flow-label.zip?rlkey=buxwsuhs7n1ccm27h9uhc46qv&dl=1"
unzip flow_label.zip -d flow_label/
```

## Troubleshooting

### Classifier not found
```
Warning: Classifier not found at SIP/data/classifiers/square_xgboost.pkl
```
**Solution:** Train the classifier first:
```bash
python scripts/classifiers/train_xgboost.py --envs square
```

### Checkpoint not found
```
FileNotFoundError: Checkpoint not found: my_checkpoint
```
**Solution:** Check the path exists in `SIP/outputs/` or use full path.

### CUDA out of memory
**Solution:** Reduce batch size or use smaller model:
```bash
./scripts/run_train.sh --env square --batch_size 32
```

## Resources

- [DASIP Poster (PDF)](assets/DASIP%20poster.pdf)
- [Overview Diagram (PDF)](assets/overview.pdf)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{chun2025dasip,
  title={Dynamic Test-Time Compute Scaling in Control Policy: Difficulty-Aware Stochastic Interpolant Policy},
  author={Chun, Inkook and Lee, Seungjae and Albergo, Michael S. and Xie, Saining and Vanden-Eijnden, Eric},
  journal={arXiv preprint arXiv:2511.20906},
  year={2025}
}
```

## Acknowledgments

This work builds upon the stochastic interpolant framework and diffusion policy research. We thank the robomimic and diffusion policy teams for their open-source contributions.

## License

MIT License - see LICENSE file.