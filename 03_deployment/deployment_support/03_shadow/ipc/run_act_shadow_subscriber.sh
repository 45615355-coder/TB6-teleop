#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBSCRIBER="$SCRIPT_DIR/act_shadow_subscriber.py"

echo "[SHADOW] ACT SUBSCRIBER: OBSERVATION ONLY"
echo "[SHADOW] NO PUB/REP/REQ SOCKET"
echo "[SHADOW] NO ROBOT / RPC / JOGANY / GRIPPER / CAMERA / TOPIC ACCESS"

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "/home/robot/.conda/envs/lerobot/bin/python" ]]; then
    PYTHON_BIN="/home/robot/.conda/envs/lerobot/bin/python"
else
    echo "[FAIL] LeRobot Conda environment 'lerobot' was not found." >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/tb6_act_offline_evaluation_hf_cache}"

exec "$PYTHON_BIN" "$SUBSCRIBER" "$@"
