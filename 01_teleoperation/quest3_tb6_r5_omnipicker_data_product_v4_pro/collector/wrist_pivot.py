"""Runtime-only Quest wrist-pivot profile loading and position correction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class WristPivotProfileError(RuntimeError):
    """Raised when an enabled wrist-pivot profile is unsafe to use."""


def load_wrist_pivot_profile(profile_path: Path | str, max_offset_m: float) -> np.ndarray:
    path = Path(profile_path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WristPivotProfileError(f"cannot load wrist-pivot profile {path}: {error}") from error

    if not isinstance(profile, dict) or profile.get("valid") is not True:
        raise WristPivotProfileError(f"wrist-pivot profile is not valid: {path}")
    raw_offset: Any = profile.get("offset_m")
    if (not isinstance(raw_offset, list) or len(raw_offset) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   for value in raw_offset)):
        raise WristPivotProfileError(f"wrist-pivot offset_m must contain exactly three numbers: {path}")

    offset = np.asarray(raw_offset, dtype=float)
    limit = float(max_offset_m)
    if not np.all(np.isfinite(offset)):
        raise WristPivotProfileError(f"wrist-pivot offset_m must be finite: {path}")
    if not math.isfinite(limit) or limit <= 0.0:
        raise WristPivotProfileError("wrist-pivot maximum offset must be finite and positive")
    if float(np.linalg.norm(offset)) > limit:
        raise WristPivotProfileError(
            f"wrist-pivot offset norm exceeds {limit:.6f} m: {path}"
        )
    return offset


def apply_wrist_pivot(raw_position: Any, raw_rotation: Any, offset_m: Any,
                      gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(raw_position, dtype=float)
    rotation = np.asarray(raw_rotation, dtype=float)
    offset = np.asarray(offset_m, dtype=float)
    correction_gain = float(gain)
    if position.shape != (3,) or rotation.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("wrist-pivot position, rotation, and offset shapes must be (3,), (3,3), and (3,)")
    if (not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(offset)) or not math.isfinite(correction_gain)):
        raise ValueError("wrist-pivot inputs must be finite")
    return position + correction_gain * (rotation @ offset), rotation
