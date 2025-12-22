# DASIP: Diffusion-Adaptive State-Informed Policy

Fast diffusion policy inference through adaptive computation based on robot state classification.

## Key Idea

Standard diffusion policies use a fixed number of denoising steps for every state. **DASIP** uses a lightweight classifier to identify the current robot state and adapts the number of diffusion steps accordingly:

| State | Description | Diffusion Steps |
|-------|-------------|-----------------|
| Initialize | Robot at starting position | 2 (easy) |
| Near | Robot approaching object | 5 (medium) |
| Attempt | Robot grabbing object | 10 (hard) |
| Pushing | Robot manipulating object | 10 (hard) |
| Holding | Robot holding object | 5 (medium) |
| Goal | Task completed | 2 (easy) |

This achieves **~2x speedup** while maintaining high success rates.

## Quick Start

```bash
# 1. Install
conda env create -f SIP/conda_environment.yaml
conda activate diffusion_policy
cd SIP && pip install -e . && cd ..

# 2. Download data
python scripts/download_data.py --all

# 3. Train classifier
python scripts/classifiers/train_xgboost.py --envs square can lift

# 4. Train diffusion policy
./scripts/run_train.sh --env square --save_path my_square_policy --num_epochs 3000

# 5. Run DASIP inference
./scripts/run_dasip.sh --env square --checkpoint my_square_policy --compare_baseline
```

## Project Structure

```
DASIP/
├── README.md
├── scripts/
│   ├── download_data.py        # Download datasets and models
│   ├── train.py                # Train diffusion policy (SIP)
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
│   ├── diffusion_policy/       # Policy models
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

### Step 2: Train State Classifier

Train an XGBoost classifier to identify robot states:

```bash
cd scripts/classifiers

# Train on all environments
python train_xgboost.py \
    --data_path ../../flow_label/data/good_4_user_filtered.csv \
    --output_dir ../../SIP/data/classifiers \
    --grid_search

# Train on specific environments
python train_xgboost.py \
    --envs square can lift \
    --output_dir ../../SIP/data/classifiers
```

The trained classifiers are saved to `SIP/data/classifiers/`.

### Step 3: Train Diffusion Policy (SIP)

```bash
# Train with Linear path (fastest)
./scripts/run_train.sh \
    --env square \
    --save_path v11_square_Linear \
    --path_type Linear \
    --num_epochs 3000

# Train with VP path
./scripts/run_train.sh \
    --env square \
    --save_path v11_square_VP \
    --path_type VP \
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

### Diffusion Policy Training

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | `square` |
| `--path_type` | Diffusion path (Linear, GVP, VP, DDPM) | `Linear` |
| `--num_epochs` | Training epochs | `3000` |
| `--checkpoint_every` | Save frequency | `10` |
| `--seed` | Random seed | `2000000` |

### DASIP Inference

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | required |
| `--checkpoint` | Checkpoint path | required |
| `--steps_easy` | Steps for easy states (i, g) | `2` |
| `--steps_medium` | Steps for medium states (n, s) | `5` |
| `--steps_hard` | Steps for hard states (t, c) | `10` |
| `--compare_baseline` | Run baseline comparison | `false` |
| `--num_episodes` | Evaluation episodes | `300` |

### Supported Environments

| Environment | Task | Classifier States |
|-------------|------|-------------------|
| `square` | Square nut assembly | i, n, t, c, s, g |
| `can` | Can picking | i, n, t, c, s, g |
| `lift` | Object lifting | i, n, t, c, s, g |
| `pusht` | Push-T | i, n, t, c, s, g |
| `tool_hang` | Tool hanging | i, n, t, c, s, g |
| `transport` | Two-arm transport | i, n, t, c, s, g |

## Classifier Training

### XGBoost (Recommended)

Fast, lightweight, works well with state observations:

```bash
python scripts/classifiers/train_xgboost.py \
    --data_path flow_label/data/good_4_user_filtered.csv \
    --output_dir SIP/data/classifiers \
    --grid_search
```

### CNN

For image-based state classification:

```bash
python scripts/classifiers/train_cnn.py \
    --data_path flow_label/data/good_4_user_filtered.csv \
    --output_dir SIP/data/classifiers \
    --env square
```

### VLM Fine-tuning

For high-accuracy vision-based classification:

```bash
python scripts/classifiers/vlm_finetune.py \
    --train_envs can square tool_hang transport \
    --test_env lift \
    --epochs 5
```

See `scripts/classifiers/README.md` for details.

## Results

Expected performance on Square environment:

| Method | Success Rate | Avg Steps | Time/Episode | Speedup |
|--------|-------------|-----------|--------------|---------|
| Baseline (10 steps) | 95.0% | 10.0 | 2.45s | 1.0x |
| DASIP (adaptive) | 94.3% | 4.2 | 1.23s | **2.0x** |

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

## Citation

```bibtex
@article{dasip2025,
  title={DASIP: Fast Diffusion Policy via Adaptive State-Informed Computation},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License - see LICENSE file.
