#!/usr/bin/env bash
# Activate teleop_act_xr before running this launcher.
(
    set -euo pipefail
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd -- "$script_dir"
    exec "$script_dir/../../run_python.sh" calibrate_wrist_pivot.py "$@"
)
