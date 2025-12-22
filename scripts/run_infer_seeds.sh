#!/bin/bash
# =============================================================================
# Multi-Seed Inference Script for Diffusion Policy
# =============================================================================
#
# This script runs policy inference across multiple seeds with video saving
# and wandb logging.
#
# Usage:
#   ./run_infer_seeds.sh --env square --checkpoint v11_square_train_Linear --infer_from 2000
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Default Configuration
# =============================================================================

ENV="square"
DEVICE=""
NUM_STEPS=10
SOLVER="Euler"
PATH_TYPE="Linear"
ODE_SDE="ODE"
TS_LIST="[10,10,10,10,10,10,10]"
CHECKPOINT_PATH=""
INFER_FROM=2000
SEED=2000000
NUM_SEEDS=300
USE_CLASSIFIER=false
USE_EMA=true
WANDB_PROJECT=""
NO_WANDB=false

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --solver)
            SOLVER="$2"
            shift 2
            ;;
        --path_type)
            PATH_TYPE="$2"
            shift 2
            ;;
        --ode_sde)
            ODE_SDE="$2"
            shift 2
            ;;
        --ts_list)
            TS_LIST="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --infer_from)
            INFER_FROM="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --use_classifier)
            USE_CLASSIFIER=true
            shift
            ;;
        --no_ema)
            USE_EMA=false
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --no_wandb)
            NO_WANDB=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --env             Environment (square, can, lift, pusht, tool_hang, transport)"
            echo "  --device          CUDA device index (auto-select if not specified)"
            echo "  --num_steps       Number of diffusion steps (default: 10)"
            echo "  --solver          Solver type: Euler, Heun (default: Euler)"
            echo "  --path_type       Path type: Linear, GVP, GVP2, VP, DDPM, DDIM (default: Linear)"
            echo "  --ode_sde         ODE or SDE (default: ODE)"
            echo "  --ts_list         TS list configuration (default: [10,10,10,10,10,10,10])"
            echo "  --checkpoint      Checkpoint path or save_path name (required)"
            echo "  --infer_from      Epoch to load checkpoint from (default: 2000)"
            echo "  --seed            Starting random seed (default: 2000000)"
            echo "  --num_seeds       Number of seeds/episodes to run (default: 300)"
            echo "  --use_classifier  Enable classifier guidance"
            echo "  --no_ema          Disable EMA model (use base model)"
            echo "  --wandb_project   Weights & Biases project name"
            echo "  --no_wandb        Disable wandb logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validation
# =============================================================================

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --checkpoint is required"
    echo "Run with --help for usage information"
    exit 1
fi

# =============================================================================
# Environment Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIP_DIR="$PROJECT_DIR/SIP"

echo "============================================================"
echo "Multi-Seed Inference for Diffusion Policy"
echo "============================================================"
echo "Environment:       $ENV"
echo "Device:            ${DEVICE:-auto}"
echo "Sampling Steps:    $NUM_STEPS"
echo "Solver:            $SOLVER"
echo "Path Type:         $PATH_TYPE"
echo "ODE/SDE:           $ODE_SDE"
echo "TS List:           $TS_LIST"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "Infer From:        $INFER_FROM"
echo "Seed:              $SEED"
echo "Num Seeds:         $NUM_SEEDS"
echo "Classifier:        $USE_CLASSIFIER"
echo "Use EMA:           $USE_EMA"
echo "Wandb:             $([ "$NO_WANDB" = true ] && echo 'disabled' || echo 'enabled')"
echo "============================================================"

# =============================================================================
# Run Inference
# =============================================================================

cd "$SIP_DIR"

# Build command
CMD="python $SCRIPT_DIR/infer_with_seeds.py \
    --env $ENV \
    --num_steps $NUM_STEPS \
    --solver $SOLVER \
    --path_type $PATH_TYPE \
    --ode_sde $ODE_SDE \
    --ts_list '$TS_LIST' \
    --checkpoint_path $CHECKPOINT_PATH \
    --infer_from $INFER_FROM \
    --seed $SEED \
    --num_seeds $NUM_SEEDS"

# Add optional arguments
if [[ -n "$DEVICE" ]]; then
    CMD="$CMD --device $DEVICE"
fi

if [[ "$USE_CLASSIFIER" = true ]]; then
    CMD="$CMD --use_classifier"
fi

if [[ "$USE_EMA" = false ]]; then
    CMD="$CMD"  # Default is true, no flag needed
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    CMD="$CMD --wandb_project $WANDB_PROJECT"
fi

if [[ "$NO_WANDB" = true ]]; then
    CMD="$CMD --no_wandb"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "Inference Complete"
echo "============================================================"

