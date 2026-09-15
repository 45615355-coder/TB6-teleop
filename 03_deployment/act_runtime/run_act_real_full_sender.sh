#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON_BIN="/home/zzw/miniconda3/envs/lerobot/bin/python"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/tb6_act_offline_evaluation_hf_cache}"

echo "[ACT-FULL-SENDER] checkpoint=080000 | LeRobot=0.6.1 | speed_scale is configurable by --act-speed-scale"
echo "[ACT-FULL-SENDER] infinite real mode; stop with Ctrl+C"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/act_real_full_sender.py" \
    --act-speed-scale 0.03 \
    "$@"
