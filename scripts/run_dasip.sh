#!/bin/bash
#
# DASIP Inference Script
#
# Run diffusion policy with adaptive state-informed inference.
# Uses classifier to dynamically adjust diffusion steps based on robot state.
#
# Usage:
#   ./scripts/run_dasip.sh --env square --checkpoint my_checkpoint
#   ./scripts/run_dasip.sh --env can --checkpoint my_checkpoint --compare_baseline
#

set -e

# =============================================================================
# Default Configuration
# =============================================================================

ENV=""
CHECKPOINT=""
INFER_FROM=2000
DEVICE=0
NUM_EPISODES=300
STEPS_EASY=2
STEPS_MEDIUM=5
STEPS_HARD=10
COMPARE_BASELINE=false
BASELINE_STEPS=10
OUTPUT_JSON=""

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --infer_from)
            INFER_FROM="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --steps_easy)
            STEPS_EASY="$2"
            shift 2
            ;;
        --steps_medium)
            STEPS_MEDIUM="$2"
            shift 2
            ;;
        --steps_hard)
            STEPS_HARD="$2"
            shift 2
            ;;
        --compare_baseline)
            COMPARE_BASELINE=true
            shift
            ;;
        --baseline_steps)
            BASELINE_STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        -h|--help)
            echo "DASIP: Diffusion-Adaptive State-Informed Policy Inference"
            echo ""
            echo "Usage: $0 --env ENV --checkpoint PATH [options]"
            echo ""
            echo "Required:"
            echo "  --env ENV              Environment (square, can, lift, pusht, tool_hang, transport)"
            echo "  --checkpoint PATH      Checkpoint directory name or path"
            echo ""
            echo "Options:"
            echo "  --infer_from N         Epoch to load (default: 2000)"
            echo "  --device N             CUDA device (default: 0)"
            echo "  --num_episodes N       Evaluation episodes (default: 300)"
            echo "  --steps_easy N         Steps for easy states (default: 2)"
            echo "  --steps_medium N       Steps for medium states (default: 5)"
            echo "  --steps_hard N         Steps for hard states (default: 10)"
            echo "  --compare_baseline     Also run baseline comparison"
            echo "  --baseline_steps N     Fixed steps for baseline (default: 10)"
            echo "  --output FILE          Save results to JSON file"
            echo ""
            echo "Examples:"
            echo "  # Basic DASIP inference"
            echo "  $0 --env square --checkpoint v11_square_train_Linear"
            echo ""
            echo "  # Compare with baseline"
            echo "  $0 --env square --checkpoint v11_square_train_Linear --compare_baseline"
            echo ""
            echo "  # Custom step schedule"
            echo "  $0 --env can --checkpoint my_can_model --steps_easy 1 --steps_hard 15"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ENV" ]; then
    echo "Error: --env is required"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# =============================================================================
# Run DASIP
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/SIP"

CMD="python ../scripts/dasip_infer.py"
CMD="$CMD --env $ENV"
CMD="$CMD --checkpoint_path $CHECKPOINT"
CMD="$CMD --infer_from $INFER_FROM"
CMD="$CMD --device cuda:$DEVICE"
CMD="$CMD --num_episodes $NUM_EPISODES"
CMD="$CMD --steps_easy $STEPS_EASY"
CMD="$CMD --steps_medium $STEPS_MEDIUM"
CMD="$CMD --steps_hard $STEPS_HARD"

if [ "$COMPARE_BASELINE" = true ]; then
    CMD="$CMD --compare_baseline --baseline_steps $BASELINE_STEPS"
fi

if [ -n "$OUTPUT_JSON" ]; then
    CMD="$CMD --output_json $OUTPUT_JSON"
fi

echo "============================================================"
echo "DASIP Inference"
echo "============================================================"
echo "Environment: $ENV"
echo "Checkpoint: $CHECKPOINT"
echo "Steps: easy=$STEPS_EASY, medium=$STEPS_MEDIUM, hard=$STEPS_HARD"
echo ""
echo "Running: $CMD"
echo "============================================================"
echo ""

eval $CMD

