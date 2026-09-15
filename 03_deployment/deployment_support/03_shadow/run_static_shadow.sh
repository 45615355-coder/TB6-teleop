#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHADOW_SCRIPT="$SCRIPT_DIR/act_static_shadow.py"

echo "[SHADOW]"
echo "READ ONLY"
echo "NO RPC COMMAND"
echo "NO JOGANY"
echo "NO GRIPPER COMMAND"
echo "ROBOT WILL NOT BE MOVED BY ACT"

if [[ ! -f "$SHADOW_SCRIPT" ]]; then
    echo "[FAIL] Missing static Shadow script: $SHADOW_SCRIPT" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "/home/zzw/miniconda3/envs/lerobot/bin/python" ]]; then
    PYTHON_BIN="/home/zzw/miniconda3/envs/lerobot/bin/python"
else
    echo "[FAIL] LeRobot Conda environment 'lerobot' was not found." >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "$PYTHON_BIN" "$SHADOW_SCRIPT" "$@"
