#!/usr/bin/env bash
set -euo pipefail

readonly PRODUCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COLLECTOR_ROOT="${PRODUCT_ROOT}/collector"
readonly TOPIC_DIR="${COLLECTOR_ROOT}/vendor/topic"
readonly RPC_SO_DIR="${COLLECTOR_ROOT}/vendor/rpc/lib/linux/x86/2004"
readonly PYTHON_BIN="/home/robot/.conda/envs/teleop_act_xr/bin/python"

echo "[ACT-FULL] REAL infinite arm + OmniPicker execution"
echo "[ACT-FULL] speed_scale is configurable by --act-speed-scale | stop with Ctrl+C / Web STOP"
LD_LIBRARY_PATH="${TOPIC_DIR}:${RPC_SO_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONPATH="${COLLECTOR_ROOT}:${TOPIC_DIR}:${RPC_SO_DIR}" \
PYTHONDONTWRITEBYTECODE=1 \
exec "${PYTHON_BIN}" "${COLLECTOR_ROOT}/act_control_full.py" \
    --execute-act-motion \
    --confirm-arm-home \
    --confirm-gripper-open \
    --confirm-web-stop-ready \
    --confirm-emergency-stop-ready \
    --act-speed-scale 0.3 \
    "$@"
