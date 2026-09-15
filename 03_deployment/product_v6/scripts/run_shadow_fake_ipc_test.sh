#!/usr/bin/env bash
set -euo pipefail

# This test imports no hardware library and initializes no robot, camera, Topic,
# or OmniPicker object. It validates only Product v5's local PUB -> ACT SUB wire format.
readonly PRODUCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PRODUCT_PUBLISHER="${PRODUCT_ROOT}/collector/shadow_observation_publisher.py"
readonly ACT_SUBSCRIBER="/home/zzw/projects/TB6_R5_ACT/training_package/03_shadow/ipc/act_shadow_subscriber.py"
readonly XR_PYTHON="/home/zzw/miniconda3/envs/xrteleop/bin/python"
readonly LEROBOT_PYTHON="/home/zzw/miniconda3/envs/lerobot/bin/python"
readonly ENDPOINT="tcp://127.0.0.1:5557"
readonly OUTPUT_ROOT="${1:-/tmp/tb6_act_shadow_v5_fake_ipc}"

[[ -x "$XR_PYTHON" ]] || { echo "[FAIL] xrteleop Python missing" >&2; exit 1; }
[[ -x "$LEROBOT_PYTHON" ]] || { echo "[FAIL] lerobot Python missing" >&2; exit 1; }
[[ -f "$PRODUCT_PUBLISHER" ]] || { echo "[FAIL] Product v5 publisher missing" >&2; exit 1; }
[[ -f "$ACT_SUBSCRIBER" ]] || { echo "[FAIL] ACT subscriber missing" >&2; exit 1; }

echo "[FAKE-IPC/V5] No hardware imports, no hardware initialization, no motion commands."
PYTHONDONTWRITEBYTECODE=1 "$XR_PYTHON" "$PRODUCT_PUBLISHER" \
    --fake --endpoint "$ENDPOINT" --fake-count 100 --fake-hz 10 --startup-delay-s 1.0 &
publisher_pid=$!
trap 'kill "$publisher_pid" 2>/dev/null || true' EXIT
sleep 0.3
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "$LEROBOT_PYTHON" "$ACT_SUBSCRIBER" \
    --endpoint "$ENDPOINT" --validate-only --timeout-s 10 --output-root "$OUTPUT_ROOT"
wait "$publisher_pid" || true
trap - EXIT
echo "[PASS] Product v5 fake IPC validation output: $OUTPUT_ROOT"
