#!/usr/bin/env bash
set -euo pipefail

readonly PRODUCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COLLECTOR_ROOT="${PRODUCT_ROOT}/collector"
readonly TOPIC_DIR="${COLLECTOR_ROOT}/vendor/topic"
readonly RPC_SO_DIR="${COLLECTOR_ROOT}/vendor/rpc/lib/linux/x86/2004"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================"
echo "ACT REAL ROBOT CONTROL"
echo "REAL MOTION WILL OCCUR"
echo "ONE CHUNK ONLY"
echo "ARM ONLY"
echo "GRIPPER WILL NOT BE CONTROLLED BY ACT"
echo "SPEED SCALE = 0.1"
echo "WEB STOP / EMERGENCY STOP MUST BE READY"
echo "========================================"

LD_LIBRARY_PATH="${TOPIC_DIR}:${RPC_SO_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONPATH="${COLLECTOR_ROOT}:${TOPIC_DIR}:${RPC_SO_DIR}" \
exec "${PYTHON_BIN}" "${COLLECTOR_ROOT}/act_control_once.py" \
    "$@" \
    --execute-act-motion \
    --confirm-arm-home \
    --confirm-gripper-open \
    --confirm-web-stop-ready \
    --confirm-emergency-stop-ready \
    --act-speed-scale 0.1
