#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <dataset_name> \"<task description>\"" >&2
    exit 2
fi

readonly DATASET_NAME="$1"
readonly TASK_DESCRIPTION="$2"
readonly PRODUCT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"

mapfile -t CONFIG_PATHS < <(
    PYTHONPATH="${PRODUCT_ROOT}/collector${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -c \
        'import config; print(config.RAW_DATA_DIR.resolve()); print(config.LEROBOT_DATASETS_DIR.resolve())'
)

if [[ ${#CONFIG_PATHS[@]} -ne 2 ]]; then
    echo "Could not read data paths from collector/config.py" >&2
    exit 1
fi

readonly INPUT_ROOT="${CONFIG_PATHS[0]}"
readonly OUTPUT_ROOT="${CONFIG_PATHS[1]}/${DATASET_NAME}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Output dataset already exists: ${OUTPUT_ROOT}" >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${PRODUCT_ROOT}/converter/convert_tb6_omnipicker_to_lerobot.py" \
    --data-dir "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --repo-id "local/${DATASET_NAME}" \
    --task "${TASK_DESCRIPTION}"
