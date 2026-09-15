#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMIC_SHADOW="$SCRIPT_DIR/act_dynamic_shadow.py"
LERO_PYTHON="/home/robot/.conda/envs/lerobot/bin/python"
XR_PYTHON="/home/robot/.conda/envs/teleop_act_xr/bin/python"
V5_FAKE_PUBLISHER="/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro/collector/shadow_observation_publisher.py"

echo "[DYNAMIC-SHADOW] ACT SUBSCRIBER: OBSERVATION ONLY"
echo "[DYNAMIC-SHADOW] SUB ONLY | NO PUB/PUSH/REQ/REP"
echo "[DYNAMIC-SHADOW] NO ROBOT / RPC / JOGANY / GRIPPER / CAMERA / TOPIC ACCESS"

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "$LERO_PYTHON" ]]; then
    PYTHON_BIN="$LERO_PYTHON"
else
    echo "[FAIL] LeRobot Conda environment 'lerobot' was not found." >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/tb6_act_offline_evaluation_hf_cache}"

if [[ "${1:-}" == "--fake-test" ]]; then
    shift
    [[ -x "$XR_PYTHON" ]] || { echo "[FAIL] xrteleop Python missing" >&2; exit 1; }
    [[ -f "$V5_FAKE_PUBLISHER" ]] || { echo "[FAIL] Product v5 fake publisher missing" >&2; exit 1; }
    output_root="${DYNAMIC_SHADOW_FAKE_OUTPUT_ROOT:-/tmp/tb6_act_dynamic_shadow_fake_$(date +%Y%m%d_%H%M%S)}"
    echo "[DYNAMIC-SHADOW/FAKE] Synthetic PUB only; no hardware imports or initialization."
    PYTHONDONTWRITEBYTECODE=1 "$XR_PYTHON" "$V5_FAKE_PUBLISHER" \
        --fake --fake-count 600 --fake-hz 10 --startup-delay-s 2.0 &
    publisher_pid=$!
    trap 'kill "$publisher_pid" 2>/dev/null || true; wait "$publisher_pid" 2>/dev/null || true' EXIT
    sleep 0.5
    "$PYTHON_BIN" "$DYNAMIC_SHADOW" "$@" \
        --inference-period-s 2.0 --max-inferences 3 --output-root "$output_root"
    kill "$publisher_pid" 2>/dev/null || true
    wait "$publisher_pid" 2>/dev/null || true
    trap - EXIT
    echo "[PASS] Dynamic fake Shadow output: $output_root"
    exit 0
fi

exec "$PYTHON_BIN" "$DYNAMIC_SHADOW" "$@"
