#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SENDER="${SCRIPT_DIR}/act_real_once_sender.py"
readonly LERO_PYTHON="/home/robot/.conda/envs/lerobot/bin/python"
readonly XR_PYTHON="/home/robot/.conda/envs/teleop_act_xr/bin/python"
readonly PRODUCT_ROOT="/home/robot/teleop_act/03_deployment/quest3_tb6_r5_omnipicker_data_product_v6"
readonly PRODUCT_COLLECTOR="${PRODUCT_ROOT}/collector"
readonly V5_FAKE_PUBLISHER="${PRODUCT_COLLECTOR}/shadow_observation_publisher.py"
readonly V5_DRY_EXECUTOR="${PRODUCT_COLLECTOR}/act_control_once.py"
readonly TOPIC_DIR="${PRODUCT_COLLECTOR}/vendor/topic"
readonly RPC_SO_DIR="${PRODUCT_COLLECTOR}/vendor/rpc/lib/linux/x86/2004"

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
    test_root="${ACT_REAL_ONCE_FAKE_ROOT:-/tmp/tb6_act_real_once_fake_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$test_root"
    dry_report="${test_root}/product_dry_run.json"
    echo "[ACT-REAL/FAKE] NO HARDWARE: fake observation + mock action PUSH/PULL + dry-run resampling"
    LD_LIBRARY_PATH="${TOPIC_DIR}:${RPC_SO_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${PRODUCT_COLLECTOR}:${TOPIC_DIR}:${RPC_SO_DIR}" \
    PYTHONDONTWRITEBYTECODE=1 "$XR_PYTHON" "$V5_DRY_EXECUTOR" \
        --dry-run-action-ipc --act-speed-scale 0.4 --action-timeout-s 30 \
        --dry-run-output "$dry_report" &
    product_pid=$!
    PYTHONDONTWRITEBYTECODE=1 "$XR_PYTHON" "$V5_FAKE_PUBLISHER" \
        --fake --fake-count 300 --fake-hz 10 --startup-delay-s 1.0 &
    publisher_pid=$!
    trap 'kill "$product_pid" "$publisher_pid" 2>/dev/null || true; wait "$product_pid" "$publisher_pid" 2>/dev/null || true' EXIT
    sleep 0.5
    "$PYTHON_BIN" "$SENDER" --mock-safe-action --observation-timeout-s 20 \
        --output-root "${test_root}/sender" "$@"
    wait "$product_pid"
    kill "$publisher_pid" 2>/dev/null || true
    wait "$publisher_pid" 2>/dev/null || true
    trap - EXIT
    echo "[PASS] ACT real-once fake IPC/dry-run: $test_root"
    exit 0
fi

echo "[ACT-REAL-SENDER] ONE [100,7] ACTION CHUNK WILL BE SENT"
echo "[ACT-REAL-SENDER] ACTION CHANNEL: PUSH/connect tcp://127.0.0.1:5558"
echo "[ACT-REAL-SENDER] PRODUCT v6 MUST ALREADY BE WAITING"
exec "$PYTHON_BIN" "$SENDER" "$@"
