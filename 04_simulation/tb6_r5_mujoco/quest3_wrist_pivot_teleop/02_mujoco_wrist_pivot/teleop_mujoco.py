#!/usr/bin/env python3
"""Standalone Quest 3 strict-no-backlog teleoperation visualized in MuJoCo."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mujoco
import mujoco.viewer
import numpy as np

import config
from placo_ik import IncrementalMujocoPlacoPlanner, orthonormalize_rotation
from wrist_pivot import WristPivotProfileError, corrected_wrist_pose, load_wrist_pivot_profile

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

HOLD, ACTIVE = "HOLD", "ACTIVE"
FRESH, STALE, WAITING = "FRESH", "STALE", "WAITING"


@dataclass(frozen=True)
class XrPose:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class FrameEvent:
    delta_position: np.ndarray
    delta_rotation: np.ndarray
    delta_rotation_deg: float
    xr_new_frame: bool
    state_changed: bool
    transition: str


class XrMonitor:
    def __init__(self) -> None:
        self.last_timestamp_ns: Optional[int] = None
        self.last_change_s = time.perf_counter()

    def observe(self, timestamp_ns: int, now_s: float, valid: bool) -> str:
        if timestamp_ns == 0:
            return WAITING
        if timestamp_ns != self.last_timestamp_ns:
            self.last_timestamp_ns, self.last_change_s = timestamp_ns, now_s
        return FRESH if valid and now_s - self.last_change_s <= config.XR_STALE_S else STALE


def quaternion_rotation(quaternion: Any) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    return orthonormalize_rotation(np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                                             (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                                             (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))), dtype=float))


def validate_xr_pose(raw: Any) -> Optional[XrPose]:
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != (7,) or not np.all(np.isfinite(values)) or np.all(values[:3] == 0.0):
        return None
    magnitude = float(np.linalg.norm(values[3:]))
    if not config.QUATERNION_NORM_MIN <= magnitude <= config.QUATERNION_NORM_MAX:
        return None
    return XrPose(values[:3].copy(), quaternion_rotation(values[3:] / magnitude))


def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    error = orthonormalize_rotation(rotation)
    angle = math.acos(float(np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-12:
        return np.array((1.0, 0.0, 0.0)), 0.0
    axis = np.array((error[2, 1] - error[1, 2], error[0, 2] - error[2, 0], error[1, 0] - error[0, 1]))
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        # A 180-degree rotation has no usable skew component; choose a stable eigen-axis.
        values, vectors = np.linalg.eig(error)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        norm = float(np.linalg.norm(axis))
    return axis / norm, angle


def _scaled_rotation(rotation: np.ndarray) -> np.ndarray:
    axis, angle = _axis_angle(rotation)
    x, y, z = axis
    theta = angle * config.ROTATION_SCALE
    c, s, d = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return orthonormalize_rotation(np.array(((c + x * x * d, x * y * d - z * s, x * z * d + y * s),
                                             (y * x * d + z * s, c + y * y * d, y * z * d - x * s),
                                             (z * x * d - y * s, z * y * d + x * s, c + z * z * d)), dtype=float))


def frame_delta(previous: XrPose, current: XrPose) -> tuple[np.ndarray, np.ndarray, float]:
    position = config.TRANSLATION_SCALE * (config.R_XR_TO_ROBOT @ (current.position - previous.position))
    relative = config.R_XR_TO_ROBOT @ (current.rotation @ previous.rotation.T) @ config.R_XR_TO_ROBOT.T
    _, angle = _axis_angle(relative)
    return position, _scaled_rotation(relative), math.degrees(angle)


class MujocoTeleopSession:
    """Exact v2 frame-consumption clutch semantics, without robot transport."""

    def __init__(self) -> None:
        self.state = HOLD
        self.previous_pose: Optional[XrPose] = None
        self.previous_timestamp: Optional[int] = None

    def _hold(self, transition: str) -> FrameEvent:
        self.state, self.previous_pose, self.previous_timestamp = HOLD, None, None
        return FrameEvent(np.zeros(3), np.eye(3), 0.0, False, True, transition)

    def consume(self, grip: float, pose: Optional[XrPose], timestamp_ns: int, xr_status: str) -> FrameEvent:
        if grip <= config.GRIP_THRESHOLD:
            return self._hold("ACTIVE -> HOLD (RELEASE)") if self.state == ACTIVE else FrameEvent(np.zeros(3), np.eye(3), 0.0, False, False, "")
        if self.state == ACTIVE and xr_status != FRESH:
            return self._hold("ACTIVE -> HOLD (XR_STALE)")
        if pose is None or xr_status != FRESH:
            return FrameEvent(np.zeros(3), np.eye(3), 0.0, False, False, "")
        if self.state == HOLD:
            self.state, self.previous_pose, self.previous_timestamp = ACTIVE, pose, timestamp_ns
            return FrameEvent(np.zeros(3), np.eye(3), 0.0, True, True, "HOLD -> ACTIVE")
        if timestamp_ns == self.previous_timestamp:
            return FrameEvent(np.zeros(3), np.eye(3), 0.0, False, False, "")
        assert self.previous_pose is not None
        position, rotation, angle = frame_delta(self.previous_pose, pose)
        # Must precede PlaCo: rejected frame intent is permanently discarded.
        self.previous_pose, self.previous_timestamp = pose, timestamp_ns
        return FrameEvent(position, rotation, angle, True, False, "")


def get_joint_qpos_indices(model: mujoco.MjModel) -> list[int]:
    indices = []
    for name in (f"joint{index}" for index in range(1, 7)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint is absent: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def set_joint_positions(data: mujoco.MjData, qpos_indices: list[int], joints_rad: np.ndarray) -> None:
    values = np.asarray(joints_rad, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise RuntimeError("MuJoCo joint command must contain six finite radians")
    for qpos_index, value in zip(qpos_indices, values):
        data.qpos[qpos_index] = value


def get_ee_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_Link")
    if body_id < 0:
        raise RuntimeError("MuJoCo body is absent: ee_Link")
    return data.xpos[body_id].copy(), orthonormalize_rotation(data.xmat[body_id].reshape(3, 3).copy())


def pose_jump_reason(current: XrPose, previous: Optional[XrPose]) -> str:
    if previous is None:
        return ""
    position_delta = float(np.linalg.norm(current.position - previous.position))
    _, rotation_delta = _axis_angle(current.rotation @ previous.rotation.T)
    if position_delta > config.XR_POSE_JUMP_POSITION_LIMIT_M:
        return f"position={position_delta:.3f}m"
    if math.degrees(rotation_delta) > config.XR_POSE_JUMP_ROTATION_LIMIT_DEG:
        return f"rotation={math.degrees(rotation_delta):.1f}deg"
    return ""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _event_record(event: FrameEvent) -> dict[str, Any]:
    result = asdict(event)
    return {key: _json_value(value) for key, value in result.items()}


def read_xr() -> tuple[Optional[XrPose], float, int]:
    if xrt is None:
        raise RuntimeError("xrobotoolkit_sdk is unavailable; activate xrteleop")
    raw = list(xrt.get_right_controller_pose())
    return validate_xr_pose(raw), float(xrt.get_right_grip()), int(xrt.get_time_stamp_ns())


def active_wrist_pivot_offset() -> np.ndarray:
    """Return the required calibrated offset, or fail before viewer startup."""
    if not config.WRIST_PIVOT_ENABLED:
        return np.zeros(3, dtype=float)
    try:
        profile = load_wrist_pivot_profile(
            config.WRIST_PIVOT_PROFILE, max_offset_m=config.WRIST_PIVOT_MAX_OFFSET_M
        )
    except WristPivotProfileError as exc:
        raise RuntimeError(f"wrist-pivot correction is enabled but unavailable: {exc}") from exc
    return profile["offset_m"]


def main() -> int:
    if xrt is None:
        raise RuntimeError("xrobotoolkit_sdk is unavailable; run in the xrteleop environment")
    wrist_pivot_offset = active_wrist_pivot_offset()
    model = mujoco.MjModel.from_xml_path(str(config.MUJOCO_URDF_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    qpos_indices = get_joint_qpos_indices(model)
    q_initial = np.deg2rad(config.INITIAL_JOINTS_DEG)
    set_joint_positions(data, qpos_indices, q_initial)
    mujoco.mj_forward(model, data)
    planner = IncrementalMujocoPlacoPlanner(q_initial)
    session, monitor = MujocoTeleopSession(), XrMonitor()
    config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIRECTORY / f"mujoco_incremental_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print("[SYS] phase=02_mujoco_wrist_pivot")
    print(f"[SYS] MuJoCo URDF={config.MUJOCO_URDF_PATH}")
    print(f"[SYS] translation_scale={config.TRANSLATION_SCALE}")
    print(f"[SYS] rotation_scale={config.ROTATION_SCALE}")
    print(f"[SYS] joint_speed_limit={config.JOINT_SPEED_LIMIT_RAD_S} rad/s")
    print(f"[SYS] wrist_pivot={'enabled' if config.WRIST_PIVOT_ENABLED else 'disabled'} gain={config.WRIST_PIVOT_CORRECTION_GAIN}")
    print("[SYS] local Quest + MuJoCo only; no real TB6-R5 connection.")
    xrt_started, last_warning, last_observed_pose = False, 0.0, None
    # The strict v2 runtime requires one observed low Grip after startup before
    # it will arm the first ACTIVE session.
    grip_release_seen = False
    deadline = time.perf_counter() + config.DT
    try:
        xrt.init()
        xrt_started = True
        with log_path.open("a", encoding="utf-8") as log, mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                now = time.perf_counter()
                if now < deadline:
                    time.sleep(deadline - now)
                started = time.perf_counter()
                try:
                    raw_pose, grip, stamp = read_xr()
                    jump = pose_jump_reason(raw_pose, last_observed_pose) if raw_pose is not None else ""
                    if raw_pose is not None and not jump:
                        last_observed_pose = raw_pose
                    xr_status = monitor.observe(stamp, started, raw_pose is not None and not jump)
                    if jump:
                        raw_pose, xr_status = None, STALE
                        if started - last_warning >= config.WARNING_INTERVAL_S:
                            print(f"[XR] jump rejected ({jump}); session held", flush=True)
                            last_warning = started
                    if raw_pose is None:
                        pose = None
                    else:
                        corrected_position, unchanged_rotation = corrected_wrist_pose(
                            raw_pose.position, raw_pose.rotation, wrist_pivot_offset,
                            config.WRIST_PIVOT_CORRECTION_GAIN,
                        )
                        pose = XrPose(corrected_position, unchanged_rotation)
                except Exception as exc:
                    raw_pose, pose, grip, stamp, xr_status = None, None, 0.0, 0, STALE
                    if started - last_warning >= config.WARNING_INTERVAL_S:
                        print(f"[XR] read failure: {type(exc).__name__}: {exc}", flush=True)
                        last_warning = started
                if session.state == HOLD and grip <= config.GRIP_THRESHOLD:
                    grip_release_seen = True
                if session.state == HOLD and not grip_release_seen:
                    event = FrameEvent(np.zeros(3), np.eye(3), 0.0, False, False, "")
                else:
                    event = session.consume(grip, pose, stamp, xr_status)
                if event.state_changed:
                    print(f"[ARM] {event.transition}", flush=True)
                pending = None
                if event.xr_new_frame and session.state == ACTIVE and not event.state_changed:
                    pending = planner.propose(event.delta_position, event.delta_rotation)
                    if pending is None:
                        planner.rollback()
                        if started - last_warning >= config.WARNING_INTERVAL_S:
                            print(f"[PLACO] rejected: {planner.last_failure}", flush=True)
                            last_warning = started
                    else:
                        planner.commit(pending)
                        set_joint_positions(data, qpos_indices, planner.q_commit)
                        mujoco.mj_forward(model, data)
                viewer.sync()
                tcp_position, tcp_rotation = get_ee_pose(model, data)
                record = {"timestamp": started, "xr_status": xr_status, "grip": grip,
                          "xr_position": None if raw_pose is None else raw_pose.position,
                          "xr_rotation": None if raw_pose is None else raw_pose.rotation,
                          "xr_corrected_position": None if pose is None else pose.position,
                          "wrist_pivot_offset_m": wrist_pivot_offset,
                          "wrist_pivot_gain": config.WRIST_PIVOT_CORRECTION_GAIN,
                          "xr_raw_to_corrected_delta_m": None if raw_pose is None or pose is None else pose.position - raw_pose.position,
                          "q_before": None if pending is None else pending["q_before"],
                          "q_commit": planner.q_commit, "accepted": pending is not None,
                          "placo_failure": planner.last_failure, "mujoco_tcp_position": tcp_position,
                          "mujoco_tcp_rotation": tcp_rotation, **_event_record(event)}
                log.write(json.dumps(record, default=_json_value, ensure_ascii=False) + "\n")
                if config.LOG_FLUSH_EVERY_FRAME:
                    log.flush()
                deadline += config.DT
                if time.perf_counter() >= deadline:
                    deadline = time.perf_counter() + config.DT
    except KeyboardInterrupt:
        print("[EXIT] local operator exit", flush=True)
    finally:
        if xrt_started:
            try:
                xrt.close()
            except Exception as exc:
                print(f"[XR] close error: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
