"""Pure wrist-pivot calibration, correction, and profile validation math."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class WristPivotProfileError(ValueError):
    """A requested correction profile is missing or unsafe to use."""


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result.copy()


def _rotation3(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    return result.copy()


def apply_wrist_pivot(position: Any, rotation: Any, offset: Any, gain: float = 1.0) -> np.ndarray:
    """Return p_controller + gain * R_controller @ offset without changing R."""
    position_array = _vector3(position, "position")
    rotation_array = _rotation3(rotation)
    offset_array = _vector3(offset, "offset")
    if not math.isfinite(float(gain)):
        raise ValueError("gain must be finite")
    return position_array + float(gain) * (rotation_array @ offset_array)


def corrected_wrist_pose(position: Any, rotation: Any, offset: Any, gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return corrected position and an elementwise unchanged rotation copy."""
    rotation_array = _rotation3(rotation)
    return apply_wrist_pivot(position, rotation_array, offset, gain), rotation_array


def solve_wrist_pivot(positions: Any, rotations: Any, *, max_offset_m: float, max_rms_error_m: float) -> dict[str, Any]:
    """Fit c ~= p_i + R_i @ offset using mean-centred least squares."""
    position_array = np.asarray(positions, dtype=float)
    rotation_array = np.asarray(rotations, dtype=float)
    if position_array.ndim != 2 or position_array.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if rotation_array.shape != (position_array.shape[0], 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if position_array.shape[0] < 2 or not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(rotation_array)):
        raise ValueError("calibration samples must be finite and contain at least two poses")
    if not math.isfinite(max_offset_m) or not math.isfinite(max_rms_error_m) or max_offset_m <= 0.0 or max_rms_error_m <= 0.0:
        raise ValueError("calibration bounds must be finite and positive")

    position_mean = np.mean(position_array, axis=0)
    rotation_mean = np.mean(rotation_array, axis=0)
    matrix = (rotation_array - rotation_mean).reshape(-1, 3)
    target = (-(position_array - position_mean)).reshape(-1)
    offset, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    offset = _vector3(offset, "estimated offset")
    offset_norm = float(np.linalg.norm(offset))
    if singular_values.size == 0 or singular_values[-1] <= 1e-12:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    pivots = position_array + np.einsum("nij,j->ni", rotation_array, offset)
    pivot_center = np.mean(pivots, axis=0)
    residuals = np.linalg.norm(pivots - pivot_center, axis=1)
    rms_error = float(math.sqrt(float(np.mean(np.square(residuals)))))
    valid = bool(
        int(rank) >= 3
        and np.all(np.isfinite(offset))
        and math.isfinite(offset_norm)
        and math.isfinite(rms_error)
        and offset_norm <= max_offset_m
        and rms_error <= max_rms_error_m
    )
    return {
        "valid": valid,
        "offset_m": offset,
        "offset_norm_m": offset_norm,
        "rms_error_m": rms_error,
        "rank": int(rank),
        "condition": condition,
        "samples": int(position_array.shape[0]),
    }


def load_wrist_pivot_profile(path: Path | str, *, max_offset_m: float) -> dict[str, Any]:
    """Load only a valid bounded profile; never silently fall back to raw pose."""
    profile_path = Path(path)
    if not profile_path.is_file():
        raise WristPivotProfileError(f"wrist-pivot profile is missing: {profile_path}")
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WristPivotProfileError(f"cannot read wrist-pivot profile: {profile_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        raise WristPivotProfileError(f"wrist-pivot profile is not valid: {profile_path}")
    try:
        offset = _vector3(payload["offset_m"], "profile offset_m")
    except (KeyError, ValueError) as exc:
        raise WristPivotProfileError(f"wrist-pivot profile offset is invalid: {profile_path}") from exc
    norm = float(np.linalg.norm(offset))
    if not math.isfinite(norm) or norm > max_offset_m:
        raise WristPivotProfileError(f"wrist-pivot profile offset norm exceeds {max_offset_m:.3f} m: {profile_path}")
    return {"path": profile_path, "offset_m": offset, "payload": payload}


def solution_json(solution: dict[str, Any], *, samples_yaw: int, samples_pitch: int, samples_roll: int) -> dict[str, Any]:
    """Convert a solver result to the on-disk calibration profile schema."""
    condition = float(solution["condition"])
    return {
        "valid": bool(solution["valid"]),
        "offset_m": np.asarray(solution["offset_m"], dtype=float).tolist(),
        "offset_norm_m": float(solution["offset_norm_m"]),
        "rms_error_m": float(solution["rms_error_m"]),
        "samples_total": int(solution["samples"]),
        "samples_yaw": int(samples_yaw),
        "samples_pitch": int(samples_pitch),
        "samples_roll": int(samples_roll),
        "rank": int(solution["rank"]),
        "condition": condition if math.isfinite(condition) else None,
    }
