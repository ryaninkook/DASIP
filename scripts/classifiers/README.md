# Robot State Classifiers

This directory contains scripts for training and deploying robot state classifiers. These classifiers predict the current state of a robot arm during manipulation tasks.

## Labels

| Label | Description |
|-------|-------------|
| `i` | **Initialize** - Robot in starting position |
| `n` | **Near** - Robot near the object |
| `t` | **Attempt** - Robot attempting to grab |
| `c` | **Pushing** - Robot pushing object |
| `s` | **Holding** - Robot holding object |
| `g` | **Goal** - Task completed |

## Available Classifiers

### 1. XGBoost/Random Forest (State-based, ~90-97% accuracy)

Train lightweight classifiers on state observations for fast inference.

```bash
python train_xgboost.py --data_path labels.csv --grid_search
```

### 2. CNN (Image-based, ~65-95% accuracy)

Train a lightweight CNN on robot images.

```bash
python train_cnn.py --data_path labels.csv --data_dir ../data --test_noise
```

### 3. VLM Zero-Shot with Few-Shot Examples (~20-83% accuracy)

Use sample images as in-context examples without fine-tuning.

**Scripts from experiments:**
- `vlm_zeroshot_fewshot.py` - Zero-shot evaluation with sample images (uses `predict_v2`/`predict_v3`)
- `select_sample_images.py` - Tool to select best sample images per environment

**How it works:**
1. First, select sample images for each interaction type per environment
2. At inference, show sample images + labels in the prompt
3. Ask model to classify a new image based on examples

```bash
# Select sample images first (interactive)
python select_sample_images.py

# Run zero-shot evaluation
python vlm_zeroshot_fewshot.py
```

**Inference methods:**
- `predict_v2`: All sample images in one user message → assistant understands → classify target
- `predict_v3`: Each sample as user/assistant turn (multi-turn conversation)

### 4. VLM Fine-tuned (~40-73% average accuracy)

Fine-tune Qwen2.5-VL on labeled data, then use for inference.

**Scripts:**
- `vlm_finetune.py` - Fine-tune VLM on labeled images
- `vlm_finetuned_infer.py` - Inference using fine-tuned model
- `vlm_server.py` - FastAPI server for fine-tuned model

```bash
# Fine-tune
python vlm_finetune.py --train_envs can square tool_hang transport --test_env lift

# Inference with fine-tuned model
python vlm_finetuned_infer.py
```

## VLM Models

All models in `models/` folder with consistent interface:

| Model | File | Description |
|-------|------|-------------|
| Qwen2.5-VL | `qwen_vlm_original.py` | Best overall (7B, supports fine-tuning) |
| LLaVA-NeXT | `llava_vlm.py` | Good alternative (Mistral 7B base) |
| Gemma3 | `gemma3_vlm.py` | Google's 4B model |
| Phi-4 | `phi4_vlm.py` | Microsoft's 5.6B model |

**Common interface:**
```python
from models.qwen_vlm_original import Qwen2_5Model
model = Qwen2_5Model()

# Single image (zero-shot)
result = model.predict(image_path, prompt)

# Few-shot with examples (v2 - all in one message)
examples = [{"image_path": "img.png", "text": "Label: initialize"}, ...]
result = model.predict_v2(examples, target_image, prompt)

# Few-shot with examples (v3 - multi-turn conversation)
result = model.predict_v3(examples, target_image, prompt)
```

## Experiment Results Summary

From experiments (04/28 - 05/13/2025):

| Method | Model | Best Accuracy | Notes |
|--------|-------|---------------|-------|
| CNN | - | 94.64% (PushT) | Fast inference |
| Random Forest | - | 97.32% (PushT) | Best state-based |
| Zero-shot + few-shot | Qwen2.5 | 83% (Lift) | Uses predict_v2 |
| Fine-tuned | Qwen2.5 | 73% (Lift, Can) | Zero-shot after training |

## Directory Structure

```
classifiers/
├── models/
│   ├── qwen_vlm_original.py   # Qwen2.5-VL (original from experiments)
│   ├── qwen_vlm.py            # Qwen2.5-VL (cleaned version)
│   ├── llava_vlm.py           # LLaVA / LLaVA-NeXT
│   ├── gemma3_vlm.py          # Gemma3
│   └── phi4_vlm.py            # Phi-4
├── train_xgboost.py           # XGBoost/RF training
├── train_cnn.py               # CNN training
├── vlm_zeroshot_fewshot.py    # Zero-shot + few-shot examples
├── vlm_finetuned_infer.py     # Fine-tuned model inference
├── vlm_finetune.py            # VLM fine-tuning
├── vlm_server.py              # FastAPI server
├── select_sample_images.py    # Sample image selection tool
├── prepare_labels.py          # Data preprocessing
└── run_classifier_training.sh # Shell wrapper
```

## Sample Images

For zero-shot few-shot inference, you need sample images at:
```
../sample_images/{env_choice}/{interaction_type}/sample_1.png
```

Use `select_sample_images.py` to create these from the labeled dataset.

## Requirements

```bash
# Core
pip install pandas numpy scikit-learn joblib matplotlib

# XGBoost
pip install xgboost

# CNN
pip install torch torchvision opencv-python

# VLM
pip install transformers>=4.37.0 peft trl datasets bitsandbytes
pip install qwen-vl-utils  # For Qwen

# Server
pip install fastapi uvicorn python-multipart
```
