#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro/data_teleop/pick_place_object_v2/test_datasets/pick_place_object_dataset"

CHECKPOINTS_ROOT="$PROJECT_ROOT/training_results/pick_place_object_v2/act/v1/train/20260901_172324/checkpoints"

OUTPUT_ROOT="$PROJECT_ROOT/training_results/pick_place_object_v2/act/v1/evaluation"

EVALUATOR="$SCRIPT_DIR/evaluate_act_checkpoints_v2.py"

if [[ ! -f "$EVALUATOR" ]]; then
    echo "[FAIL] Missing evaluator: $EVALUATOR" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "/home/robot/.conda/envs/lerobot/bin/python" ]]; then
    PYTHON_BIN="/home/robot/.conda/envs/lerobot/bin/python"
else
    echo "[FAIL] LeRobot Conda environment 'lerobot' was not found." >&2
    echo "Activate it first: conda activate lerobot" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/tb6_act_offline_evaluation_hf_cache}"

exec "$PYTHON_BIN" "$EVALUATOR" \
    --dataset-root "$DATASET_ROOT" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoints 020000 040000 060000 080000 100000 120000 \
    "$@"
