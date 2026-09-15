#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage:"
    echo "  $0 /absolute/path/to/dataset [config_file]"
    echo
    echo "Examples:"
    echo "  $0 /data/pick_place_object_v1_train"
    echo "  $0 /data/pick_place_object_v1_train ../config/pick_place_object_v1/act/v2.sh"
    exit 2
fi

DATASET_ROOT="$(realpath "$1")"
CONFIG_FILE="${2:-$PACKAGE_ROOT/config/pick_place_object_v1/act/v1.sh}"
CONFIG_FILE="$(realpath "$CONFIG_FILE")"

tb6_require_dir "$DATASET_ROOT"
tb6_require_file "$DATASET_ROOT/meta/info.json"
tb6_require_file "$CONFIG_FILE"
command -v lerobot-train >/dev/null 2>&1 || tb6_die "lerobot-train not found. Activate the lerobot environment."

source "$CONFIG_FILE"
tb6_validate_config

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$PROJECT_ROOT/training_results/$TASK_NAME/$POLICY_NAME/$TRAIN_VERSION/train/$STAMP"
JOB_NAME="${TASK_NAME}_${POLICY_NAME}_${TRAIN_VERSION}"

LOG_FILE="$PROJECT_ROOT/training_results/$TASK_NAME/$POLICY_NAME/$TRAIN_VERSION/train/${STAMP}_train.log"
mkdir -p "$(dirname "$LOG_FILE")"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== TB6-R5 ACT formal training ==="
echo "Dataset       : $DATASET_ROOT"
echo "Config        : $CONFIG_FILE"
echo "Output        : $OUTPUT_DIR"
echo
tb6_print_config
echo

tb6_train_args \
    "$DATASET_ROOT" "$OUTPUT_DIR" "$JOB_NAME" \
    "$STEPS" "$BATCH_SIZE" "$NUM_WORKERS" "$SAVE_FREQ" "$LOG_FREQ"

lerobot-train "${TB6_ARGS[@]}"

cp "$CONFIG_FILE" "$OUTPUT_DIR/experiment_config.sh"

mv "$LOG_FILE" "$OUTPUT_DIR/train.log"

echo
echo "[PASS] Formal training completed."
echo "[OUTPUT] $OUTPUT_DIR"
echo "[LOG] $OUTPUT_DIR/train.log"
