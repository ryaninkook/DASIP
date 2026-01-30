# DASIP: Difficulty-Aware Stochastic Interpolant Policy

**Dynamic Test-Time Compute Scaling in Control Policy via Adaptive Solver Configuration.**

[NeurIPS 2025] Official implementation of **DASIP**, a framework that unifies diffusion and flow-matching policies to enable adaptive computation.

[![arXiv](https://img.shields.io/badge/arXiv-2511.20906-b31b1b.svg)](https://arxiv.org/abs/2511.20906)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Inkook Chun†, Seungjae Lee‡, Michael S. Albergo⋄, Saining Xie†, Eric Vanden-Eijnden†,§

†New York University, ‡University of Maryland, ⋄Harvard University, §Capital Fund Management

## Overview

![DASIP Overview](assets/intro.png)

Standard generative policies (diffusion or flow-based) employ a fixed inference budget for every control step, wasting resources on simple subtasks while potentially underperforming on hard ones.

**DASIP** introduces a **Stochastic Interpolant (SI)** framework that unifies ODE (Flow) and SDE (Diffusion) formulations. It uses a lightweight **difficulty classifier** to analyze observations in real-time and dynamically selects an optimal **Inference Configuration Triple**:

$$\langle \text{Step Count}, \text{Solver Type}, \text{ODE/SDE Integration} \rangle$$

This is analogous to "System 1 vs. System 2" thinking in cognitive science—fast, intuitive processing for easy states and deliberate, effortful computation for hard states.

## Key Idea

| Difficulty Category | Context Example | Compute Strategy | Configuration (Steps, Solver, Mode) |
|:---|:---|:---|:---|
| **Initial (I)** | Robot positioned away from targets | Minimal | 1, Euler, ODE |
| **Near (N)** | Approaching near object | Low | 5, Euler, ODE |
| **Grabbing (G)** | Grasping and placing objects | Medium | 10, Euler, ODE |
| **Stochastic (S)** | Precision alignment (e.g., Tool Hang) | High (Exploration) | 50, Euler, SDE |
| **Continuous (C)** | Sustained precise manipulation (e.g., Push) | Max (Precision) | 100, Heun, SDE |
| **End (E)** | Task objectives achieved | Minimal | 1, Euler, ODE |

This approach achieves **2.6–4.4× reduction in computation time** while maintaining or exceeding state-of-the-art success rates.

## Stochastic Interpolant Framework

DASIP is grounded in the **stochastic interpolant (SI) framework**, which unifies flow-based (ODE) and diffusion-based (SDE) generative modeling. This formulation enables:

- **Flexible training configurations:** Choose prediction target (noise, score, velocity) and interpolant type (Linear, VP, GVP)
- **Adaptive inference:** Translate difficulty levels into optimized inference configurations without retraining
- **ODE/SDE switching:** Dynamically select deterministic or stochastic integration based on task requirements

### Supported Configurations

| Training | Options |
|----------|---------|
| Prediction Target | Noise, Score, Velocity |
| Interpolant Type | Linear, VP (Variance Preserving), GVP (Generalized VP) |

| Inference | Options |
|-----------|---------|
| Integration Mode | ODE (deterministic), SDE (stochastic) |
| Solver Type | Euler, Heun |
| Step Count | 1–100 (adaptive based on difficulty) |

## Difficulty Classifiers

DASIP supports three classifier architectures:

| Classifier | Description | Use Case |
|------------|-------------|----------|
| **Lightweight CNN** | Trained on 32×32 images (~300 samples/task), 0.023s inference | Default, production |
| **Few-shot VLM** | Vision-language model with exemplar prompts | Zero-shot transfer |
| **Fine-tuned VLM** | Qwen-VL fine-tuned on difficulty annotations | Highest accuracy |

## Quick Start

```bash
# 1. Install
conda env create -f SIP/conda_environment.yaml
conda activate SI_policy
cd SIP && pip install -e . && cd ..

# 2. Download data (Annotated Difficulty Data + Robomimic/PushT)
python scripts/download_data.py --all

# 3. Train Difficulty Classifier (Lightweight CNN)
python scripts/classifiers/train_cnn.py --env square --data_path flow_label/data/

# 4. Train Stochastic Interpolant (SI) Policy
./scripts/run_train.sh --env square --save_path my_si_policy --path_type Linear --num_epochs 5000

# 5. Run DASIP Adaptive Inference
./scripts/run_dasip.sh --env square --checkpoint my_si_policy --compare_baseline
```

## Project Structure

```
DASIP/
├── README.md
├── assets/                     # Documentation assets
│   ├── intro.png               # Overview figure
│   ├── DASIP_poster.pdf       # Project poster
│   └── overview.pdf            # Architecture diagram
├── scripts/
│   ├── download_data.py        # Download datasets and models
│   ├── train.py                # Train SI policy
│   ├── dasip_infer.py          # DASIP adaptive inference
│   ├── evaluate.py             # Standard evaluation
│   ├── infer_with_seeds.py     # Multi-seed inference
│   ├── process_results.py      # Result processing
│   ├── run_train.sh            # Training shell wrapper
│   ├── run_dasip.sh            # DASIP inference wrapper
│   └── classifiers/            # Difficulty Classifiers
│       ├── train_cnn.py        # Lightweight CNN (Recommended)
│       ├── vlm_finetune.py     # Fine-tuned VLM (High Accuracy)
│       ├── vlm_fewshot.py      # Few-shot VLM
│       └── models/             # VLM wrappers
├── SIP/                        # Core Implementation
│   ├── SI_policy/              # Stochastic Interpolant Models
│   │   ├── config/             # Hydra configs
│   │   ├── model/              # Neural networks (U-Net transformer)
│   │   ├── policy/             # Unified ODE/SDE policies
│   │   └── env_runner/         # Environment runners
│   └── data/                   # Data directory
│       ├── pusht/              # PushT dataset
│       ├── robomimic/          # Robomimic datasets
│       ├── checkpoints/        # Trained SIP models
│       └── classifiers/        # Trained classifiers
└── flow_label/                 # Difficulty annotation data
    └── data/
```

## Complete Pipeline

### Step 1: Download Data

Download the annotated difficulty datasets (~20k labeled states from 8 annotators) and standard benchmarks.

```bash
# Download all data (datasets + classifiers)
python scripts/download_data.py --all

# Or download selectively
python scripts/download_data.py --datasets              # Training datasets
python scripts/download_data.py --datasets --tasks pusht square  # Specific tasks
python scripts/download_data.py --classifiers           # Classifier models
```

### Step 2: Train Difficulty Classifier

We support three difficulty classifier variants:

1. **Lightweight CNN:** (Recommended) Trained on ~300 images per task (32×32 input). Fast inference (~23ms).
2. **Fine-tuned VLM:** (High Accuracy) Qwen2.5-VL fine-tuned for 8 epochs.
3. **Few-shot VLM:** Multi-image prompting with 1-3 exemplars per category.

#### Lightweight CNN (Recommended)

```bash
python scripts/classifiers/train_cnn.py \
    --data_path flow_label/data/annotations.csv \
    --output_dir SIP/data/classifiers \
    --env square
```

#### VLM Fine-tuning

For high-accuracy vision-language classification:

```bash
python scripts/classifiers/vlm_finetune.py \
    --train_envs can square tool_hang transport \
    --test_env lift \
    --epochs 8
```

See `scripts/classifiers/README.md` for details.

### Step 3: Train Stochastic Interpolant Policy

Train a single generative policy capable of both ODE and SDE integration. The **Stochastic Interpolant** framework allows selecting the path type (Linear, VP, GVP) at training time.

```bash
# Train with Linear Interpolant (Flow Matching equivalent)
./scripts/run_train.sh \
    --env square \
    --save_path v1_square_Linear \
    --path_type Linear \
    --num_epochs 5000

# Train with VP Interpolant (Diffusion equivalent)
./scripts/run_train.sh \
    --env square \
    --save_path v1_square_VP \
    --path_type VP \
    --num_epochs 5000

# Train with GVP (Generalized VP) interpolant
./scripts/run_train.sh \
    --env square \
    --save_path v1_square_GVP \
    --path_type GVP \
    --num_epochs 5000

# Training options
./scripts/run_train.sh --help
```

### Step 4: Run DASIP Inference

During inference, the policy adapts the **Step Count**, **Solver**, and **Integration Mode** based on the predicted difficulty.

```bash
# Run adaptive inference (Auto-configures steps based on difficulty)
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v1_square_Linear \
    --classifier_path SIP/data/classifiers/square_cnn.pth

# Compare with fixed-budget baseline (Max Compute)
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v1_square_Linear \
    --compare_baseline

# Save results to JSON
./scripts/run_dasip.sh \
    --env square \
    --checkpoint v1_square_Linear \
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

### Stochastic Interpolant Training (`run_train.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | `square` |
| `--path_type` | Interpolant type: `Linear` (Flow), `VP` (Diffusion), `GVP` | `Linear` |
| `--prediction` | Prediction target: `velocity`, `score`, or `noise` | `velocity` |
| `--num_epochs` | Training epochs | `5000` |
| `--batch_size` | Batch size | `256` |
| `--lr` | Learning rate | `1e-4` |
| `--checkpoint_every` | Save frequency | `50` |
| `--seed` | Random seed | `42` |

### Adaptive Inference (`run_dasip.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | required |
| `--checkpoint` | Checkpoint path | required |
| `--classifier` | Path to trained classifier (CNN/VLM) | auto-detect |
| `--solver_config` | JSON map of Difficulty → (Steps, Solver, Mode) | default schedule |
| `--use_sde` | Enable stochastic integration (SDE) for hard states | `true` |
| `--compare_baseline` | Run baseline comparison | `false` |
| `--num_episodes` | Evaluation episodes | `50` |

### Supported Environments

| Environment | Task | Difficulty States |
|-------------|------|-------------------|
| `square` | Square nut assembly | I, N, G, S, C, E |
| `can` | Can picking | I, N, G, S, C, E |
| `lift` | Object lifting | I, N, G, S, C, E |
| `pusht` | Push-T | I, N, G, S, C, E |
| `tool_hang` | Tool hanging | I, N, G, S, C, E |
| `transport` | Two-arm transport | I, N, G, S, C, E |

**Difficulty state codes:** I=Initial, N=Near, G=Grabbing, S=Stochastic, C=Continuous, E=End

## Results

### Success Rate Comparison (from Table 5)

| Environment | Min Compute | Max Compute | DASIP (CNN) | Few-shot VLM | Fine-tuned VLM |
|-------------|-------------|-------------|--------------|--------------|----------------|
| Tool Hang | 25% | 25% | 27% (+2%) | 33% (+8%) | 33% (+8%) |
| Square | 88% | 90% | 92% (+2%) | 88% (-2%) | 89% (-1%) |
| Transport | 62% | 80% | 77% (-3%) | 62% (-18%) | 79% (-1%) |
| Push-T | 81% | 86% | 87% (+1%) | 86% (0%) | 87% (+1%) |
| Lift | 100% | 100% | 100% (0%) | 100% (0%) | 100% (0%) |
| Can | 98% | 99% | 99% (0%) | 99% (0%) | 99% (0%) |

### Computational Efficiency - Speedup vs. Max Compute (from Table 4)

| Environment | CNN Classifier | Few-shot VLM | Fine-tuned VLM |
|-------------|----------------|--------------|----------------|
| Tool Hang | **3.22×** | 2.28× | 2.45× |
| Square | **2.61×** | 1.65× | 1.35× |
| Transport | **4.43×** | 1.92× | 2.10× |
| Push-T | 2.69× | **5.55×** | 2.69× |
| Lift | **4.09×** | 2.61× | 2.94× |
| Can | **2.89×** | 2.26× | 2.49× |

### Time Comparison in Seconds (from Table 4)

| Environment | Min Compute | Max Compute | DASIP (CNN) |
|-------------|-------------|-------------|--------------|
| Tool Hang | 91.02 | 342.13 | 106.37 |
| Square | 25.86 | 146.68 | 56.20 |
| Transport | 73.04 | 342.54 | 77.36 |
| Push-T | 1.44 | 121.89 | 45.35 |
| Lift | 57.04 | 263.28 | 64.31 |
| Can | 63.70 | 214.81 | 74.32 |

**Key findings:**
- CNN classifier achieves best balance of speed (avg **3.3×** speedup) and accuracy (avg **0.3%** difference from max compute)
- Fine-tuned VLM offers **2.3×** speedup with **1.8%** performance difference
- Simple tasks (Lift, Can) achieve near-perfect success even with minimal compute (1 step)

### Classifier Accuracy (from Table 2)

| Environment | CNN | Few-shot VLM (Qwen2.5) | Fine-tuned VLM |
|-------------|-----|------------------------|----------------|
| Can | 88.9% | 37.5% | 47.5% |
| Lift | 92.8% | 83.3% | 86.7% |
| Push-T | 94.6% | 42.0% | 56.7% |
| Square | 87.4% | 26.0% | 52.0% |
| Tool Hang | 81.3% | 50.0% | 40.0% |
| Transport | 91.5% | 30.0% | 40.0% |

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
Warning: Classifier not found at SIP/data/classifiers/square_cnn.pth
```
**Solution:** Train the classifier first:
```bash
python scripts/classifiers/train_cnn.py --env square
```

### Checkpoint not found
```
FileNotFoundError: Checkpoint not found: my_checkpoint
```
**Solution:** Check the path exists in `SIP/outputs/` or use full path.

### CUDA out of memory
**Solution:** Reduce batch size:
```bash
./scripts/run_train.sh --env square --batch_size 128
```

## Resources

- [DASIP Poster (PDF)](assets/DASIP_poster.pdf)
- [Overview Diagram (PDF)](assets/overview.pdf)

## Citation

If you use DASIP in your research, please cite our NeurIPS 2025 paper:

```bibtex
@inproceedings{chun2025dasip,
  title={Dynamic Test-Time Compute Scaling in Control Policy: Difficulty-Aware Stochastic Interpolant Policy},
  author={Chun, Inkook and Lee, Seungjae and Albergo, Michael S. and Xie, Saining and Vanden-Eijnden, Eric},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgments

We thank the authors of Stochastic Interpolant and SiT for their paper and codebase, which served as the foundation of our research. We are also grateful for the authors of the Diffusion Policy codebase, which was instrumental for our implementation.

We thank Sanghyun Woo for valuable early discussions on research methodology and comparative analysis, and Nur Muhammad Mahi Shafiullah for insightful discussions on robotics experiment design and ensuring fair comparisons with baseline methods. We thank Lerrel Pinto and Mark Goldstein for helpful feedback on the draft and experiments.

This work was supported by compute resources provided by NYU IT High Performance Computing resources and services.

## License

MIT License - see LICENSE file.