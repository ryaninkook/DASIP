# Classifier Scripts Modifications Log

This file documents all modifications made to scripts copied from `flow_label/v2label/` and `flow_label/v3_combined_df/`.

**Last Updated:** Dec 22, 2025

## Overview

| Script | Original | Status | Key Changes |
|--------|----------|--------|-------------|
| `vlm_zeroshot_fewshot.py` | `a7_1_zeeshan_focused_samples.py` | Refactored | CLI, modularity, no #%% |
| `vlm_finetuned_infer.py` | `a8_2_use_finetune_model_zeeshan_qwen.py` | Refactored | CLI, no hardcoded token |
| `select_sample_images.py` | `filter_best_samples_zeeshan.py` | Refactored | CLI, interactive mode |
| `models/qwen_vlm.py` | `qwen2_5_model1_ryan.py` | Refactored | Clean API, env var token |
| `models/qwen_vlm_original.py` | `qwen2_5_model.py` | Kept for reference | Original version |
| `models/gemma3_vlm.py` | `gemma3_model.py` | Refactored | No hardcoded paths/tokens |
| `models/phi4_vlm.py` | `microsoft_phi_4_vlm.py` | Refactored | No hardcoded paths/tokens |
| `models/llava_vlm.py` | `llava_models.py` | Refactored | Clean API |

---

## Detailed Changes

### vlm_zeroshot_fewshot.py

**Original:** `flow_label/v2label/a7_1_zeeshan_focused_samples.py`

**Changes Made:**
1. Removed `#%%` cell markers (Jupyter notebook style)
2. Added argparse CLI: `--env`, `--model`, `--num_examples`, `--method`
3. Made data paths configurable via CLI args
4. Modularized into functions: `sample_data()`, `load_sample_images()`, `evaluate_env()`
5. Added `if __name__ == "__main__"` block
6. Added progress logging

**Preserved:**
- `predict_v2` and `predict_v3` method calls (both supported via `--method`)
- Sample image loading logic
- Accuracy calculation
- Label mapping

---

### vlm_finetuned_infer.py

**Original:** `flow_label/v2label/a8_2_use_finetune_model_zeeshan_qwen.py`

**Changes Made:**
1. Removed `#%%` cell markers
2. Added argparse CLI: `--model_path`, `--env`, `--num_samples`
3. HuggingFace token via `HF_TOKEN` env var (removed hardcoded)
4. Modularized into functions: `load_finetuned_model()`, `evaluate_env()`
5. Added main block and summary output

**Preserved:**
- Model loading with PeftModel
- Prediction logic and prompts
- Quantization configuration

---

### select_sample_images.py

**Original:** `flow_label/v2label/filter_best_samples_zeeshan.py`

**Changes Made:**
1. Removed `#%%` cell markers
2. Added argparse CLI: `--env`, `--interaction`
3. Made interactive with command loop (n/p/s/q)
4. Made paths configurable

**Preserved:**
- Image display logic
- Save path structure

---

### models/gemma3_vlm.py

**Original:** `flow_label/v2label/gemma3_model.py`

**Changes Made:**
1. `cache_dir` now configurable (was hardcoded `/mnt/l1/huggingface`)
2. `device_idx` configurable
3. HuggingFace token via `HF_TOKEN` env var or constructor arg
4. Added docstrings

**Preserved:**
- Quantization config (4-bit NF4)
- LoRA config
- predict, predict_v2, predict_v3 method logic

---

### models/phi4_vlm.py

**Original:** `flow_label/v2label/microsoft_phi_4_vlm.py`

**Changes Made:**
1. `cache_dir`, `device`, `max_memory` configurable
2. HuggingFace token via `HF_TOKEN` env var
3. Added docstrings
4. Moved SYSTEM_MESSAGE to module-level constant

**Preserved:**
- Flash attention 2 usage
- predict, predict_v2, predict_v3 method logic

---

### models/qwen_vlm.py

**Original:** `flow_label/v2label/qwen2_5_model1_ryan.py` (refactored)

**Changes Made:**
1. Clean class structure with docstrings
2. Configurable constructor: `model_name`, `cache_dir`, `max_memory`, `hf_token`
3. Methods: `predict()`, `predict_with_examples()`, `predict_v2()`, `predict_from_base64()`
4. Works with environment variable `HF_TOKEN`

**Note:** `models/qwen_vlm_original.py` kept as-is for reference/fallback.

---

## How to Revert

If any modification causes issues, the original files are still at:

| Modified | Original |
|----------|----------|
| `vlm_zeroshot_fewshot.py` | `flow_label/v2label/a7_1_zeeshan_focused_samples.py` |
| `vlm_finetuned_infer.py` | `flow_label/v2label/a8_2_use_finetune_model_zeeshan_qwen.py` |
| `select_sample_images.py` | `flow_label/v2label/filter_best_samples_zeeshan.py` |
| `models/gemma3_vlm.py` | `flow_label/v2label/gemma3_model.py` |
| `models/phi4_vlm.py` | `flow_label/v2label/microsoft_phi_4_vlm.py` |
| `models/llava_vlm.py` | `flow_label/v2label/llava_models.py` |

To revert any file:
```bash
cp flow_label/v2label/<original_file> scripts/classifiers/<target_file>
```

---

## Testing After Modifications

To verify modifications don't introduce bugs:

```bash
cd DASIP

# 1. Test zero-shot few-shot with Qwen
python scripts/classifiers/vlm_zeroshot_fewshot.py --env lift --model qwen --method v2

# 2. Test fine-tuned inference (requires fine-tuned model)
python scripts/classifiers/vlm_finetuned_infer.py --env lift --model_path flow_label/qwen_finetuned/version_1

# 3. Test XGBoost training
python scripts/classifiers/train_xgboost.py --data_path flow_label/data/good_4_user_filtered.csv --test_size 0.2
```

Expected results should match the experiment tables:

| Model | Method | Accuracy |
|-------|--------|----------|
| Qwen 2.5 VL | Zero-shot v2 | ~45-55% |
| Qwen 2.5 VL | Fine-tuned | ~55-60% |
| XGBoost | State-based | ~85-95% |

