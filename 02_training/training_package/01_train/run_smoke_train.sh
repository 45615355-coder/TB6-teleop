#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage:"
    echo "  $0 /absolute/path/to/dataset [config_file]"
    exit 2
fi

DATASET_ROOT="$(realpath "$1")"
CONFIG_FILE="${2:-$PACKAGE_ROOT/config/pick_place_object_v2/act/v1.sh}"
CONFIG_FILE="$(realpath "$CONFIG_FILE")"

tb6_require_dir "$DATASET_ROOT"
tb6_require_file "$DATASET_ROOT/meta/info.json"
tb6_require_file "$CONFIG_FILE"
command -v lerobot-train >/dev/null 2>&1 || tb6_die "lerobot-train not found. Activate the lerobot environment."

source "$CONFIG_FILE"
tb6_validate_config

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$PROJECT_ROOT/training_results/$TASK_NAME/$POLICY_NAME/$TRAIN_VERSION/smoke/$STAMP"
JOB_NAME="${TASK_NAME}_${POLICY_NAME}_${TRAIN_VERSION}_smoke"

SMOKE_STEPS=20
SMOKE_BATCH_SIZE=1
SMOKE_NUM_WORKERS=1
SMOKE_SAVE_FREQ=0
SMOKE_LOG_FREQ=1

echo "=== TB6-R5 ACT smoke training ==="
echo "Dataset       : $DATASET_ROOT"
echo "Config        : $CONFIG_FILE"
echo "Output        : $OUTPUT_DIR"
echo
tb6_print_config
echo
echo "Smoke override:"
echo "  steps=20, batch_size=1, num_workers=1, save_freq=0, log_freq=1"
echo

tb6_train_args \
    "$DATASET_ROOT" "$OUTPUT_DIR" "$JOB_NAME" \
    "$SMOKE_STEPS" "$SMOKE_BATCH_SIZE" "$SMOKE_NUM_WORKERS" \
    "$SMOKE_SAVE_FREQ" "$SMOKE_LOG_FREQ"

lerobot-train "${TB6_ARGS[@]}"

cp "$CONFIG_FILE" "$OUTPUT_DIR/experiment_config.sh"

echo
echo "[PASS] Smoke training completed."
echo "[OUTPUT] $OUTPUT_DIR"
