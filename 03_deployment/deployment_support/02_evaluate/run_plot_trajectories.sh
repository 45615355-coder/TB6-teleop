#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="/home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v4/data_teleop/pick_place_object_v1/test_datasets/pick_place_object_v1_test"
CHECKPOINTS_ROOT="$PROJECT_ROOT/training_results/pick_place_object_v1/act/v1/train/20260827_113046/checkpoints"
OUTPUT_ROOT="$PROJECT_ROOT/training_results/pick_place_object_v1/act/v1/evaluation_plots"
PLOTTER="$SCRIPT_DIR/plot_act_trajectories.py"

if [[ ! -f "$PLOTTER" ]]; then
    echo "[FAIL] Missing trajectory plotter: $PLOTTER" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "/home/zzw/miniconda3/envs/lerobot/bin/python" ]]; then
    PYTHON_BIN="/home/zzw/miniconda3/envs/lerobot/bin/python"
else
    echo "[FAIL] LeRobot Conda environment 'lerobot' was not found." >&2
    echo "Activate it first: conda activate lerobot" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/tb6_act_offline_evaluation_hf_cache}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tb6_act_plot_mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/tb6_act_plot_cache}"

exec "$PYTHON_BIN" "$PLOTTER" \
    --dataset-root "$DATASET_ROOT" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint 080000 \
    --episodes 0 5 7 \
    "$@"
