#!/bin/bash
# =============================================================================
# Training and Inference Script for Diffusion Policy
# =============================================================================
#
# This script runs training or inference for diffusion policies.
#
# Usage:
#   ./run_train.sh --env square --save_path my_experiment
#   ./run_train.sh --env square --save_path my_experiment --mode infer --infer_from 2000
#
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================

MODE="train"
ENV="square"
DEVICE=""
NUM_STEPS=10
SOLVER="Euler"
PATH_TYPE="Linear"
ODE_SDE="ODE"
TS_LIST="[10,10,10,10,10,10,10]"
T=100
TDISCRETE="Tdiscrete"
SAVE_PATH=""
SEED=2000000
NUM_EPOCHS=3000
CHECKPOINT_EVERY=10
RESUME=false
USE_EMA=true
INFER_FROM=2000
USE_CLASSIFIER=false
WANDB_PROJECT=""
NO_WANDB=false
DEBUG=false

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
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
        --T)
            T="$2"
            shift 2
            ;;
        --Tdiscrete)
            TDISCRETE="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --checkpoint_every)
            CHECKPOINT_EVERY="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no_ema)
            USE_EMA=false
            shift
            ;;
        --infer_from)
            INFER_FROM="$2"
            shift 2
            ;;
        --use_classifier)
            USE_CLASSIFIER=true
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
        --debug)
            DEBUG=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --mode            train or infer (default: train)"
            echo "  --env             Environment (square, can, lift, pusht, tool_hang, transport)"
            echo "  --device          CUDA device index (auto-select if not specified)"
            echo "  --num_steps       Number of diffusion steps (default: 10)"
            echo "  --solver          Solver type: Euler, Heun (default: Euler)"
            echo "  --path_type       Path type: Linear, GVP, GVP2, VP, DDPM, DDIM (default: Linear)"
            echo "  --ode_sde         ODE or SDE (default: ODE)"
            echo "  --ts_list         TS list configuration"
            echo "  --T               Total diffusion timesteps (default: 100)"
            echo "  --Tdiscrete       Tdiscrete or Tconti (default: Tdiscrete)"
            echo "  --save_path       Save path name (required)"
            echo "  --seed            Random seed (default: 2000000)"
            echo "  --num_epochs      Number of epochs (default: 3000)"
            echo "  --checkpoint_every Checkpoint frequency (default: 10)"
            echo "  --resume          Resume training from checkpoint"
            echo "  --no_ema          Disable EMA model"
            echo "  --infer_from      Epoch to start inference from (default: 2000)"
            echo "  --use_classifier  Enable classifier guidance"
            echo "  --wandb_project   W&B project name"
            echo "  --no_wandb        Disable wandb logging"
            echo "  --debug           Enable debug mode"
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

if [[ -z "$SAVE_PATH" ]]; then
    echo "Error: --save_path is required"
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
echo "Diffusion Policy ${MODE^}"
echo "============================================================"
echo "Mode:              $MODE"
echo "Environment:       $ENV"
echo "Device:            ${DEVICE:-auto}"
echo "Path Type:         $PATH_TYPE"
echo "Solver:            $SOLVER"
echo "ODE/SDE:           $ODE_SDE"
echo "T:                 $T"
echo "TS:                $NUM_STEPS"
echo "Save Path:         $SAVE_PATH"
echo "Seed:              $SEED"
echo "Epochs:            $NUM_EPOCHS"
if [[ "$MODE" == "infer" ]]; then
    echo "Infer From:        $INFER_FROM"
    echo "Classifier:        $USE_CLASSIFIER"
fi
echo "============================================================"

# =============================================================================
# Run
# =============================================================================

cd "$SIP_DIR"

# Build command
CMD="python $SCRIPT_DIR/train.py \
    --mode $MODE \
    --env $ENV \
    --num_steps $NUM_STEPS \
    --solver $SOLVER \
    --path_type $PATH_TYPE \
    --ode_sde $ODE_SDE \
    --ts_list '$TS_LIST' \
    --T $T \
    --Tdiscrete $TDISCRETE \
    --save_path $SAVE_PATH \
    --seed $SEED \
    --num_epochs $NUM_EPOCHS \
    --checkpoint_every $CHECKPOINT_EVERY \
    --infer_from $INFER_FROM"

# Add optional flags
if [[ -n "$DEVICE" ]]; then
    CMD="$CMD --device $DEVICE"
fi

if [[ "$RESUME" = true ]]; then
    CMD="$CMD --resume"
fi

if [[ "$USE_EMA" = true ]]; then
    CMD="$CMD --use_ema"
fi

if [[ "$USE_CLASSIFIER" = true ]]; then
    CMD="$CMD --use_classifier"
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    CMD="$CMD --wandb_project $WANDB_PROJECT"
fi

if [[ "$NO_WANDB" = true ]]; then
    CMD="$CMD --no_wandb"
fi

if [[ "$DEBUG" = true ]]; then
    CMD="$CMD --debug"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "${MODE^} Complete"
echo "============================================================"

