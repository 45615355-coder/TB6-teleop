#!/usr/bin/env python3
"""Quest-only collection of Yaw, Pitch, and Roll wrist-pivot samples."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from typing import Any

import numpy as np

import config
from wrist_pivot import solution_json, solve_wrist_pivot

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))), dtype=float)


class FreshMonitor:
    def __init__(self) -> None:
        self.timestamp: int | None = None
        self.changed_at = time.perf_counter()

    def fresh(self, timestamp: int, now: float, valid: bool) -> bool:
        if timestamp == 0:
            return False
        if timestamp != self.timestamp:
            self.timestamp, self.changed_at = timestamp, now
        return valid and now - self.changed_at <= config.XR_STALE_S


def read_pose() -> tuple[np.ndarray | None, np.ndarray | None, float, int]:
    if xrt is None:
        raise RuntimeError("xrobotoolkit_sdk is unavailable; activate xrteleop")
    raw = np.asarray(list(xrt.get_right_controller_pose()), dtype=float)
    grip, timestamp = float(xrt.get_right_grip()), int(xrt.get_time_stamp_ns())
    if raw.shape != (7,) or not np.all(np.isfinite(raw)) or np.all(raw[:3] == 0.0):
        return None, None, grip, timestamp
    magnitude = float(np.linalg.norm(raw[3:]))
    if not config.QUATERNION_NORM_MIN <= magnitude <= config.QUATERNION_NORM_MAX:
        return None, None, grip, timestamp
    return raw[:3].copy(), _rotation_from_quaternion(raw[3:] / magnitude), grip, timestamp


def wait_for_initial_release(monitor: FreshMonitor) -> None:
    printed = False
    while True:
        now = time.perf_counter()
        try:
            position, rotation, grip, timestamp = read_pose()
            if monitor.fresh(timestamp, now, position is not None and rotation is not None) and grip <= config.GRIP_THRESHOLD:
                return
        except Exception:
            pass
        if not printed:
            print("[CAL] Release Grip once to arm calibration.", flush=True)
            printed = True
        time.sleep(config.DT)


def capture_direction(name: str, index: int, monitor: FreshMonitor) -> tuple[list[np.ndarray], list[np.ndarray]]:
    movement = {"YAW": "left/right", "PITCH": "up/down", "ROLL": "around the controller axis"}[name]
    while True:
        print(f"\n[CAL] {index}/3 {name}")
        print("Keep wrist center approximately still.")
        print(f"Hold Grip and rotate {movement}; release Grip when finished.")
        positions: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        pressed = False
        last_captured_timestamp: int | None = None
        while True:
            now = time.perf_counter()
            try:
                position, rotation, grip, timestamp = read_pose()
                fresh = monitor.fresh(timestamp, now, position is not None and rotation is not None)
            except Exception:
                position = rotation = None
                grip, fresh = 0.0, False
            if not pressed:
                if fresh and grip > config.GRIP_THRESHOLD:
                    pressed = True
                else:
                    time.sleep(config.DT)
                    continue
            if grip <= config.GRIP_THRESHOLD:
                break
            if (fresh and position is not None and rotation is not None
                    and timestamp != last_captured_timestamp):
                positions.append(position)
                rotations.append(rotation)
                last_captured_timestamp = timestamp
            time.sleep(config.DT)
        if len(positions) >= config.WRIST_PIVOT_MIN_SAMPLES_PER_DIRECTION:
            print(f"[CAL] {name} captured: {len(positions)} samples", flush=True)
            return positions, rotations
        print(f"[CAL] {name} needs at least {config.WRIST_PIVOT_MIN_SAMPLES_PER_DIRECTION} fresh samples; capture again.", flush=True)


def save_result(payload: dict[str, Any]) -> tuple[object, object | None]:
    calibration_directory = config.WRIST_PIVOT_PROFILE.parent
    calibration_directory.mkdir(parents=True, exist_ok=True)
    stamped = calibration_directory / f"wrist_pivot_{datetime.now():%Y%m%d_%H%M%S}.json"
    stamped.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    current = None
    if payload["valid"]:
        current = config.WRIST_PIVOT_PROFILE
        current.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stamped, current


def print_result(payload: dict[str, Any], profile: object | None) -> None:
    offset = np.asarray(payload["offset_m"], dtype=float) * 1000.0
    print("\n========== WRIST PIVOT CALIBRATION ==========")
    print("offset:")
    print(f"X = {offset[0]:.1f} mm")
    print(f"Y = {offset[1]:.1f} mm")
    print(f"Z = {offset[2]:.1f} mm")
    print(f"\noffset norm = {payload['offset_norm_m'] * 1000.0:.1f} mm")
    print(f"RMS residual = {payload['rms_error_m'] * 1000.0:.1f} mm")
    print(f"samples = {payload['samples_total']}")
    print(f"condition = {payload['condition']}")
    print("\nVALID" if payload["valid"] else "\nINVALID (current_user.json was not changed)")
    if profile is not None:
        print(f"profile:\n{profile}")
    print("=============================================")


def main() -> int:
    if xrt is None:
        raise RuntimeError("xrobotoolkit_sdk is unavailable; run in the xrteleop environment")
    print("[CAL] Wrist-pivot calibration")
    print("[CAL] Quest-only: no MuJoCo, real TB6-R5, Topic, RPC, JogAny, OmniPicker, or RealSense.")
    monitor = FreshMonitor()
    started = False
    try:
        xrt.init()
        started = True
        wait_for_initial_release(monitor)
        yaw_p, yaw_r = capture_direction("YAW", 1, monitor)
        pitch_p, pitch_r = capture_direction("PITCH", 2, monitor)
        roll_p, roll_r = capture_direction("ROLL", 3, monitor)
        print("\n[CAL] solving...", flush=True)
        solution = solve_wrist_pivot(yaw_p + pitch_p + roll_p, yaw_r + pitch_r + roll_r,
                                     max_offset_m=config.WRIST_PIVOT_MAX_OFFSET_M,
                                     max_rms_error_m=config.WRIST_PIVOT_MAX_RMS_ERROR_M)
        payload = solution_json(solution, samples_yaw=len(yaw_p), samples_pitch=len(pitch_p), samples_roll=len(roll_p))
        _, profile = save_result(payload)
        print_result(payload, profile)
        return 0 if payload["valid"] else 1
    except KeyboardInterrupt:
        print("[CAL] cancelled", flush=True)
        return 130
    finally:
        if started:
            try:
                xrt.close()
            except Exception as exc:
                print(f"[CAL] XR close error: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
