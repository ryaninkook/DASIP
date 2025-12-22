#!/bin/bash
# =============================================================================
# Evaluation Script for Diffusion Policy with SIP
# =============================================================================
#
# This script runs policy evaluation on robotic manipulation environments.
#
# Usage:
#   ./run_evaluation.sh                    # Run with default settings
#   ./run_evaluation.sh --env can          # Specify environment
#   ./run_evaluation.sh --device 1         # Use GPU 1
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Default Configuration
# =============================================================================

ENV="square"
DEVICE=0
NUM_STEPS=2
SOLVER="Euler"
LAST_STEP_SOLVER="Euler"
ODE_SDE="ODE"
PATH_TYPE="Linear"
CLASSIFIER_TYPE=""
SEED=2000000
NUM_EPISODES=300
CHECKPOINT_PATH=""
OUTPUT_DIR="outputs/eval"

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
        --last_step_solver)
            LAST_STEP_SOLVER="$2"
            shift 2
            ;;
        --ode_sde)
            ODE_SDE="$2"
            shift 2
            ;;
        --path_type)
            PATH_TYPE="$2"
            shift 2
            ;;
        --classifier_type)
            CLASSIFIER_TYPE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --env             Environment (square, can, lift, pusht, tool_hang, transport)"
            echo "  --device          CUDA device index (default: 0)"
            echo "  --num_steps       Number of diffusion steps (default: 2)"
            echo "  --solver          Solver type: Euler, Heun (default: Euler)"
            echo "  --last_step_solver Last step solver: Euler, Tweedie (default: Euler)"
            echo "  --ode_sde         ODE or SDE (default: ODE)"
            echo "  --path_type       Path type: Linear, GVP, VP, DDPM, DDIM (default: Linear)"
            echo "  --classifier_type Classifier: cnn, vlm, or empty (default: none)"
            echo "  --seed            Random seed (default: 2000000)"
            echo "  --num_episodes    Number of evaluation episodes (default: 300)"
            echo "  --checkpoint      Path to model checkpoint (required)"
            echo "  --output_dir      Output directory (default: outputs/eval)"
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

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# =============================================================================
# Environment Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIP_DIR="$PROJECT_DIR/SIP"

echo "============================================================"
echo "Diffusion Policy Evaluation"
echo "============================================================"
echo "Environment:       $ENV"
echo "Device:            $DEVICE"
echo "Sampling Steps:    $NUM_STEPS"
echo "Solver:            $SOLVER / $LAST_STEP_SOLVER"
echo "ODE/SDE:           $ODE_SDE"
echo "Path Type:         $PATH_TYPE"
echo "Classifier:        ${CLASSIFIER_TYPE:-none}"
echo "Seed:              $SEED"
echo "Episodes:          $NUM_EPISODES"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "Output:            $OUTPUT_DIR"
echo "============================================================"

# =============================================================================
# Run Evaluation
# =============================================================================

cd "$SIP_DIR"

# Build command
CMD="python $SCRIPT_DIR/evaluate.py \
    --env $ENV \
    --device $DEVICE \
    --num_steps $NUM_STEPS \
    --solver $SOLVER \
    --last_step_solver $LAST_STEP_SOLVER \
    --ode_sde $ODE_SDE \
    --path_type $PATH_TYPE \
    --seed $SEED \
    --num_episodes $NUM_EPISODES \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR"

# Add optional arguments
if [[ -n "$CLASSIFIER_TYPE" ]]; then
    CMD="$CMD --classifier_type $CLASSIFIER_TYPE"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "Evaluation Complete"
echo "============================================================"

