#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage:"
    echo "  $0 /absolute/path/to/pick_place_object_v2_train"
    exit 2
fi

DATASET_ROOT="$(realpath "$1")"
MANIFEST="$PACKAGE_ROOT/dataset_manifests/pick_place_object_v2_train.sha256"

echo "=== 1/3 Environment ==="
python "$SCRIPT_DIR/check_environment.py"

echo
echo "=== 2/3 Dataset SHA256 ==="
if [[ -f "$MANIFEST" ]]; then
    python "$PACKAGE_ROOT/tools/verify_dataset_manifest.py" \
        --dataset-root "$DATASET_ROOT" \
        --manifest "$MANIFEST"
else
    echo "[FAIL] Manifest not found: $MANIFEST"
    exit 1
fi

echo
echo "=== 3/3 LeRobot dataset load/decode ==="
python "$SCRIPT_DIR/check_dataset_v2.py" --dataset-root "$DATASET_ROOT"

echo
echo "[PASS] Company-computer preflight completed."
