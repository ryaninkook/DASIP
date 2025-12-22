#!/bin/bash
#
# Robot State Classifier Training & Inference Scripts
#
# This script provides easy access to all classifier-related commands.
#
# Usage:
#   ./run_classifier_training.sh <command> [options]
#
# Commands:
#   xgboost       Train XGBoost/Random Forest classifier
#   cnn           Train CNN classifier
#   vlm-finetune  Fine-tune VLM (Qwen2.5-VL)
#   vlm-zeroshot  Run zero-shot/few-shot VLM evaluation
#   vlm-infer     Run fine-tuned VLM inference
#   server        Start VLM API server
#   prepare       Prepare labels with majority voting
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    echo "Robot State Classifier Scripts"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  xgboost       Train XGBoost/Random Forest classifier"
    echo "  cnn           Train CNN classifier"
    echo "  vlm-finetune  Fine-tune VLM (Qwen2.5-VL) with LoRA"
    echo "  vlm-zeroshot  Zero-shot/few-shot VLM evaluation"
    echo "  vlm-infer     Inference with fine-tuned VLM"
    echo "  server        Start VLM prediction API server"
    echo "  prepare       Prepare labels with majority voting"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Train XGBoost with grid search"
    echo "  $0 xgboost --grid_search"
    echo ""
    echo "  # Train CNN with noise robustness"
    echo "  $0 cnn --env lift --noise_test"
    echo ""
    echo "  # VLM zero-shot with 2 examples per class"
    echo "  $0 vlm-zeroshot --env square --num_examples 2 --method v2"
    echo ""
    echo "  # VLM fine-tuned inference"
    echo "  $0 vlm-infer --env lift"
    echo ""
    echo "  # Start API server on port 8001"
    echo "  $0 server --port 8001"
    echo ""
    echo "For detailed options, run: $0 <command> --help"
}

# =============================================================================
# Command Handlers
# =============================================================================

run_xgboost() {
    echo "Training XGBoost/Random Forest classifier..."
    python train_xgboost.py "$@"
}

run_cnn() {
    echo "Training CNN classifier..."
    python train_cnn.py "$@"
}

run_vlm_finetune() {
    echo "Fine-tuning VLM..."
    python vlm_finetune.py "$@"
}

run_vlm_zeroshot() {
    echo "Running VLM zero-shot/few-shot evaluation..."
    python vlm_zeroshot_fewshot.py "$@"
}

run_vlm_infer() {
    echo "Running fine-tuned VLM inference..."
    python vlm_finetuned_infer.py "$@"
}

run_server() {
    echo "Starting VLM prediction API server..."
    python vlm_server.py "$@"
}

run_prepare() {
    echo "Preparing labels with majority voting..."
    python prepare_labels.py "$@"
}

# =============================================================================
# Main
# =============================================================================

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    xgboost)
        run_xgboost "$@"
        ;;
    cnn)
        run_cnn "$@"
        ;;
    vlm-finetune)
        run_vlm_finetune "$@"
        ;;
    vlm-zeroshot)
        run_vlm_zeroshot "$@"
        ;;
    vlm-infer)
        run_vlm_infer "$@"
        ;;
    server)
        run_server "$@"
        ;;
    prepare)
        run_prepare "$@"
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        show_help
        exit 1
        ;;
esac
