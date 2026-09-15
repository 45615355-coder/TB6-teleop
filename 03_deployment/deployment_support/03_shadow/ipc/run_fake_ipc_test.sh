#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PRODUCT_PUBLISHER="/home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v4/collector/shadow_observation_publisher.py"
XR_PYTHON="/home/zzw/miniconda3/envs/xrteleop/bin/python"
LERO_PYTHON="/home/zzw/miniconda3/envs/lerobot/bin/python"
ENDPOINT="tcp://127.0.0.1:5557"
OUTPUT_ROOT="${1:-/tmp/tb6_act_shadow_fake_ipc}"

[[ -x "$XR_PYTHON" ]] || { echo "[FAIL] xrteleop Python missing" >&2; exit 1; }
[[ -x "$LERO_PYTHON" ]] || { echo "[FAIL] lerobot Python missing" >&2; exit 1; }
[[ -f "$PRODUCT_PUBLISHER" ]] || { echo "[FAIL] Product fake publisher missing" >&2; exit 1; }

echo "[FAKE-IPC] No hardware imports, no hardware initialization, no motion commands."
PYTHONDONTWRITEBYTECODE=1 "$XR_PYTHON" "$PRODUCT_PUBLISHER" \
    --fake --endpoint "$ENDPOINT" --fake-count 100 --fake-hz 10 --startup-delay-s 1.0 &
PUBLISHER_PID=$!
trap 'kill "$PUBLISHER_PID" 2>/dev/null || true' EXIT
sleep 0.3
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "$LERO_PYTHON" "$SCRIPT_DIR/act_shadow_subscriber.py" \
    --endpoint "$ENDPOINT" --validate-only --timeout-s 10 --output-root "$OUTPUT_ROOT"
wait "$PUBLISHER_PID" || true
trap - EXIT
echo "[PASS] Fake IPC validation output: $OUTPUT_ROOT"
