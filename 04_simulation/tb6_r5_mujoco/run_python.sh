#!/usr/bin/env bash
# Project-local Python entry point. Activate teleop_act_xr first.
# Preserve the caller's working directory, environment, and shell options.
(
    set -euo pipefail

    project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    # Resolve the shared SDK within teleop_act without changing or copying it.
    sdk_dir="$(cd -- "$project_dir/../.." && pwd)/01_teleoperation/third_party"
    if [[ ! -r "$sdk_dir/libPXREARobotSDK.so" ]]; then
        printf 'SDK library not found or unreadable: %s\n' "$sdk_dir/libPXREARobotSDK.so" >&2
        exit 1
    fi
    if ! command -v python >/dev/null 2>&1; then
        printf 'Python not found. Run: conda activate teleop_act_xr\n' >&2
        exit 1
    fi

    exec env LD_LIBRARY_PATH="$sdk_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        python "$@"
)
