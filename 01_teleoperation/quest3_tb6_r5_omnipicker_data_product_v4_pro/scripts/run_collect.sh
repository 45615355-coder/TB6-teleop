#!/usr/bin/env bash
set -euo pipefail

readonly PRODUCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COLLECTOR_ROOT="${PRODUCT_ROOT}/collector"
readonly TOPIC_DIR="${COLLECTOR_ROOT}/vendor/topic"
readonly RPC_SO_DIR="${COLLECTOR_ROOT}/vendor/rpc/lib/linux/x86/2004"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"

readonly THIRD_PARTY_DIR="${PRODUCT_ROOT}/../third_party"

echo "TB6-R5 + Quest3 + OmniPicker + 3xD435 DATA COLLECTOR V4"
echo "STRICT INCREMENTAL / NO-BACKLOG"
echo "REAL HARDWARE CONTROL"
echo "Grip=teleop | Trigger=gripper | A=HOME | B=record | right stick=discard"

LD_LIBRARY_PATH="${TOPIC_DIR}:${RPC_SO_DIR}:${THIRD_PARTY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONPATH="${COLLECTOR_ROOT}:${TOPIC_DIR}:${RPC_SO_DIR}:${THIRD_PARTY_DIR}" \
exec "${PYTHON_BIN}" "${COLLECTOR_ROOT}/teleop.py" \
    --execute-motion \
    --confirm-controller-ready \
    --confirm-robot-enabled \
    --confirm-robot-stationary \
    --record-data \
    "$@"
