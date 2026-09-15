#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON_BIN="/home/robot/.conda/envs/lerobot/bin/python"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/tb6_act_offline_evaluation_hf_cache}"

echo "[ACT-FULL-SENDER-V2] task=pick_place_object_v2 | checkpoint=120000 | LeRobot=0.6.1"
echo "[ACT-FULL-SENDER-V2] default speed_scale=0.03; override with --act-speed-scale <value>"
echo "[ACT-FULL-SENDER-V2] infinite real mode; stop with Ctrl+C"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/act_real_full_sender_v2.py" \
    --act-speed-scale 0.3 \
    "$@"
