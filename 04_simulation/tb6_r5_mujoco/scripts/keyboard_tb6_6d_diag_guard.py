#!/usr/bin/env python3
"""Continuous keyboard 6D diagnostics for TB6 R5 MuJoCo only.

This program does not connect to Quest, PC Service, or a physical robot.
"""

from __future__ import annotations

from pathlib import Path
import math
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

IK_SERVER_PATH = str(PROJECT_ROOT / "ik_test" / "build" / "ik_server")

INITIAL_JOINTS_DEG = np.array(
    [
        -17.999469,
        -46.798620,
        75.438946,
        -118.796496,
        -89.997346,
        -75.597770,
    ],
    dtype=float,
)

CONTROL_HZ = 50.0
KEY_LINEAR_SPEED_M_S = 0.05
KEY_ANGULAR_SPEED_DEG_S = 20.0

MAX_TCP_SPEED_M_S = 0.20
MAX_ANGULAR_SPEED_DEG_S = 60.0
MAX_JOINT_SPEED_DEG_S = 180.0
JOINT_STEP_MARGIN_DEG = 1.0

MAX_DT_S = 0.05
POSITION_OFFSET_LIMIT_M = np.array([1.30, 1.30, 1.25], dtype=float)

MOTION_KEYS = set("wsadqr123456")


@dataclass(frozen=True)
class KeyboardSnapshot:
    pressed_keys: set[str]
    space_down: bool
    exit_requested: bool
    listener_error: str | None


@dataclass
class KeyboardState:
    pressed_keys: set[str] = field(default_factory=set)
    space_down: bool = False
    exit_requested: bool = False
    listener_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> KeyboardSnapshot:
        with self.lock:
            return KeyboardSnapshot(
                pressed_keys=set(self.pressed_keys),
                space_down=self.space_down,
                exit_requested=self.exit_requested,
                listener_error=self.listener_error,
            )

    def record_listener_error(self, error: Exception) -> None:
        with self.lock:
            if self.listener_error is None:
                self.listener_error = event_reason(error)
            self.exit_requested = True


@dataclass
class PerformanceWindow:
    loop_period_s: list[float] = field(default_factory=list)
    ik_duration_s: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.loop_period_s.clear()
        self.ik_duration_s.clear()


def rotation_matrix_to_pe321(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """Convert a rotation matrix to rz, ry, rx in radians."""
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    singular = sy < 1e-9

    if not singular:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        ry = math.atan2(-rotation[2, 0], sy)
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        ry = math.atan2(-rotation[2, 0], sy)
        rz = 0.0

    return rz, ry, rx


def axis_angle_to_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=float)

    unit_axis = axis / axis_norm
    x, y, z = unit_axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=float,
    )


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    fixed = u @ vt
    if np.linalg.det(fixed) < 0.0:
        u[:, -1] *= -1.0
        fixed = u @ vt
    return fixed


def rotation_error_axis_angle(
    target: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, float]:
    error = orthonormalize_rotation(target @ current.T)
    cos_angle = (float(np.trace(error)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.acos(cos_angle)

    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    if math.pi - angle < 1e-5:
        axis = np.array(
            [
                math.sqrt(max(0.0, (error[0, 0] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (error[1, 1] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (error[2, 2] + 1.0) * 0.5)),
            ],
            dtype=float,
        )
        axis[0] = math.copysign(axis[0], error[2, 1] - error[1, 2])
        axis[1] = math.copysign(axis[1], error[0, 2] - error[2, 0])
        axis[2] = math.copysign(axis[2], error[1, 0] - error[0, 1])
    else:
        axis = np.array(
            [
                error[2, 1] - error[1, 2],
                error[0, 2] - error[2, 0],
                error[1, 0] - error[0, 1],
            ],
            dtype=float,
        )
        axis /= 2.0 * math.sin(angle)

    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    return axis / norm, angle


def rotation_distance_rad(
    rotation_a: np.ndarray,
    rotation_b: np.ndarray,
) -> float:
    _, angle = rotation_error_axis_angle(rotation_a, rotation_b)
    return angle


def wrapped_joint_delta(
    current: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    raw_delta = target - current
    return np.arctan2(np.sin(raw_delta), np.cos(raw_delta))


def check_joint_step_limit(
    current_joints_rad: np.ndarray,
    new_joints_rad: np.ndarray,
    dt: float,
) -> tuple[bool, int, np.ndarray, float]:
    joint_delta_rad = wrapped_joint_delta(
        current_joints_rad,
        new_joints_rad,
    )
    joint_delta_deg = np.rad2deg(joint_delta_rad)
    allowed_joint_step_deg = (
        MAX_JOINT_SPEED_DEG_S * dt + JOINT_STEP_MARGIN_DEG
    )
    largest_joint_index = int(np.argmax(np.abs(joint_delta_deg)))
    accepted = bool(
        np.all(np.abs(joint_delta_deg) <= allowed_joint_step_deg)
    )
    return (
        accepted,
        largest_joint_index,
        joint_delta_deg,
        allowed_joint_step_deg,
    )


def pose_to_pe321(
    position: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    rz, ry, rx = rotation_matrix_to_pe321(rotation)
    return np.array(
        [position[0], position[1], position[2], rz, ry, rx],
        dtype=float,
    )


def get_joint_qpos_indices(model: mujoco.MjModel) -> list[int]:
    indices: list[int] = []
    for index in range(6):
        joint_name = f"joint{index + 1}"
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise RuntimeError(f"Cannot find {joint_name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def set_joint_positions(
    data: mujoco.MjData,
    qpos_indices: list[int],
    joints_rad: np.ndarray,
) -> None:
    if joints_rad.size != 6 or not np.all(np.isfinite(joints_rad)):
        raise RuntimeError("Invalid MuJoCo joint positions")
    for index, qpos_index in enumerate(qpos_indices):
        data.qpos[qpos_index] = joints_rad[index]


def get_ee_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    ee_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ee_Link",
    )
    if ee_body_id < 0:
        raise RuntimeError("Cannot find ee_Link")
    position = data.xpos[ee_body_id].copy()
    rotation = data.xmat[ee_body_id].reshape(3, 3).copy()
    return position, orthonormalize_rotation(rotation)


def start_ik_server() -> subprocess.Popen[str]:
    command = (
        "source /opt/ros/humble/setup.bash && "
        f"export TB6_IK_URDF={shlex.quote(URDF_PATH)} && "
        f"exec {shlex.quote(IK_SERVER_PATH)}"
    )
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Cannot read IK server output")
    if process.stdin is None:
        raise RuntimeError("Cannot write IK server input")

    first_line = process.stdout.readline().strip()
    if first_line != "READY":
        process.terminate()
        raise RuntimeError(f"IK server startup failed: {first_line}")
    return process


def solve_ik(
    process: subprocess.Popen[str],
    target_pe321: np.ndarray,
    seed_joints_rad: np.ndarray,
) -> np.ndarray:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("IK server streams are unavailable")
    if process.poll() is not None:
        raise RuntimeError(
            f"IK server exited with code {process.returncode}"
        )

    seed_joints_deg = np.rad2deg(seed_joints_rad)
    values = np.concatenate([target_pe321, seed_joints_deg])
    if values.size != 12 or not np.all(np.isfinite(values)):
        raise RuntimeError("IK request contains invalid values")

    request = " ".join(f"{value:.12f}" for value in values)
    process.stdin.write(request + "\n")
    process.stdin.flush()

    response = process.stdout.readline().strip()
    if response == "":
        raise RuntimeError("IK server returned no output")
    if response.startswith("ERR"):
        raise RuntimeError(response)
    if not response.startswith("OK "):
        raise RuntimeError(f"Unexpected IK response: {response}")

    joints_rad = np.array(
        [float(value) for value in response.split()[1:]],
        dtype=float,
    )
    if joints_rad.size != 6:
        raise RuntimeError(
            f"IK returned {joints_rad.size} joints instead of 6"
        )
    if not np.all(np.isfinite(joints_rad)):
        raise RuntimeError("IK returned NaN or Inf")
    return joints_rad


def stop_ik_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    try:
        if process.stdin is not None and process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
        process.wait(timeout=2.0)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except Exception:
            process.kill()


def event_reason(error: Exception) -> str:
    reason = "_".join(str(error).split())
    return reason if reason else type(error).__name__


def key_to_char(key: Any) -> str | None:
    char = getattr(key, "char", None)
    if not isinstance(char, str) or not char:
        return None
    return char.lower()


def create_keyboard_listener(state: KeyboardState) -> keyboard.Listener:
    def on_press(key: Any) -> bool | None:
        try:
            with state.lock:
                if key == keyboard.Key.esc:
                    state.exit_requested = True
                    return False
                if key == keyboard.Key.space:
                    state.space_down = True
                    return None
                char = key_to_char(key)
                if char in MOTION_KEYS:
                    state.pressed_keys.add(char)
        except Exception as error:
            state.record_listener_error(error)
            return False
        return None

    def on_release(key: Any) -> bool | None:
        try:
            with state.lock:
                if key == keyboard.Key.space:
                    state.space_down = False
                    return None
                char = key_to_char(key)
                if char is not None:
                    state.pressed_keys.discard(char)
        except Exception as error:
            state.record_listener_error(error)
            return False
        return None

    return keyboard.Listener(on_press=on_press, on_release=on_release)


def integrate_desired_position(
    desired_position: np.ndarray,
    pressed_keys: set[str],
    initial_position: np.ndarray,
    dt: float,
) -> np.ndarray:
    translation_direction = np.array(
        [
            float("w" in pressed_keys) - float("s" in pressed_keys),
            float("a" in pressed_keys) - float("d" in pressed_keys),
            float("q" in pressed_keys) - float("r" in pressed_keys),
        ],
        dtype=float,
    )
    direction_norm = float(np.linalg.norm(translation_direction))
    if direction_norm > 1e-12:
        normalized_direction = translation_direction / direction_norm
        desired_position = (
            desired_position
            + normalized_direction * KEY_LINEAR_SPEED_M_S * dt
        )

    lower = initial_position - POSITION_OFFSET_LIMIT_M
    upper = initial_position + POSITION_OFFSET_LIMIT_M
    return np.clip(desired_position, lower, upper)


def integrate_desired_rotation(
    desired_rotation: np.ndarray,
    pressed_keys: set[str],
    dt: float,
) -> np.ndarray:
    rotation_direction_world = np.array(
        [
            float("1" in pressed_keys) - float("2" in pressed_keys),
            float("3" in pressed_keys) - float("4" in pressed_keys),
            float("5" in pressed_keys) - float("6" in pressed_keys),
        ],
        dtype=float,
    )
    direction_norm = float(np.linalg.norm(rotation_direction_world))
    if direction_norm <= 1e-12:
        # Translation-only input must not alter the desired orientation.
        return desired_rotation

    rotation_axis_world = rotation_direction_world / direction_norm
    angular_delta_rad = math.radians(KEY_ANGULAR_SPEED_DEG_S) * dt
    delta_rotation_world = axis_angle_to_rotation(
        rotation_axis_world,
        angular_delta_rad,
    )
    return orthonormalize_rotation(
        delta_rotation_world @ desired_rotation
    )


def step_position_toward(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
) -> np.ndarray:
    delta = target - current
    distance = float(np.linalg.norm(delta))
    max_step = MAX_TCP_SPEED_M_S * dt
    if distance <= max_step or distance < 1e-12:
        return target.copy()
    return current + delta / distance * max_step


def step_rotation_toward(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
) -> np.ndarray:
    axis, angle = rotation_error_axis_angle(target, current)
    max_step = math.radians(MAX_ANGULAR_SPEED_DEG_S) * dt
    if angle <= max_step or angle < 1e-12:
        return orthonormalize_rotation(target)
    delta = axis_angle_to_rotation(axis, max_step)
    return orthonormalize_rotation(delta @ current)


def clamp_dt(dt: float) -> float:
    if dt < 0.0:
        return 0.0
    if dt > MAX_DT_S:
        return MAX_DT_S
    return dt


def percentile_ms_or_na(samples: list[float]) -> str:
    if len(samples) < 2:
        return "n/a"
    percentile_ms = float(np.percentile(samples, 95)) * 1000.0
    return f"{percentile_ms:.1f}ms"


def print_performance_line(
    window: PerformanceWindow,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
    commanded_position: np.ndarray,
    commanded_rotation: np.ndarray,
    actual_position: np.ndarray,
    actual_rotation: np.ndarray,
    enabled: bool,
) -> None:
    position_lag_mm = float(
        np.linalg.norm(desired_position - commanded_position)
    ) * 1000.0
    rotation_lag_deg = math.degrees(
        rotation_distance_rad(desired_rotation, commanded_rotation)
    )
    tcp_position_error_mm = float(
        np.linalg.norm(actual_position - commanded_position)
    ) * 1000.0
    tcp_rotation_error_deg = math.degrees(
        rotation_distance_rad(actual_rotation, commanded_rotation)
    )
    print(
        "PERF "
        f"loop_p95={percentile_ms_or_na(window.loop_period_s)} "
        f"ik_p95={percentile_ms_or_na(window.ik_duration_s)} "
        f"lag={position_lag_mm:.1f}mm/{rotation_lag_deg:.1f}deg "
        f"tcp_err={tcp_position_error_mm:.2f}mm/"
        f"{tcp_rotation_error_deg:.2f}deg "
        f"enabled={str(enabled).lower()}",
        flush=True,
    )


def print_startup_info() -> None:
    print("TB6 R5 MuJoCo continuous keyboard 6D diagnostic")
    print("")
    print("Hold SPACE to enable.")
    print("")
    print("SPACE+W/S : world X +/-")
    print("SPACE+A/D : world Y +/-")
    print("SPACE+Q/R : world Z +/-")
    print("")
    print("SPACE+1/2 : world X rotation +/-")
    print("SPACE+3/4 : world Y rotation +/-")
    print("SPACE+5/6 : world Z rotation +/-")
    print("")
    print("Esc: exit")
    print("")
    print(f"linear_speed={KEY_LINEAR_SPEED_M_S * 1000.0:.1f}mm/s")
    print(f"angular_speed={KEY_ANGULAR_SPEED_DEG_S:.1f}deg/s")
    print(f"control_hz={CONTROL_HZ:.1f}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0

    qpos_indices = get_joint_qpos_indices(model)
    current_joints_rad = np.deg2rad(INITIAL_JOINTS_DEG)
    set_joint_positions(data, qpos_indices, current_joints_rad)
    mujoco.mj_forward(model, data)

    initial_position, initial_rotation = get_ee_pose(model, data)
    desired_position = initial_position.copy()
    desired_rotation = initial_rotation.copy()
    commanded_position = initial_position.copy()
    commanded_rotation = initial_rotation.copy()
    actual_position = initial_position.copy()
    actual_rotation = initial_rotation.copy()

    last_valid_position = commanded_position.copy()
    last_valid_rotation = commanded_rotation.copy()
    last_valid_joints_rad = current_joints_rad.copy()

    keyboard_state = KeyboardState()
    listener: keyboard.Listener | None = None
    listener_started = False
    listener_error_reported = False
    ik_process: subprocess.Popen[str] | None = None

    performance = PerformanceWindow()
    period = 1.0 / CONTROL_HZ
    previous_loop_start: float | None = None
    last_performance_time = 0.0
    previous_enabled = False
    require_space_release = False
    ik_failure_latched = False
    joint_jump_latched = False

    print_startup_info()

    try:
        try:
            listener = create_keyboard_listener(keyboard_state)
            listener.start()
            listener_started = True
        except Exception as error:
            print(
                f"KEYBOARD_LISTENER_ERROR reason={event_reason(error)}",
                flush=True,
            )
            listener_error_reported = True
            return

        ik_process = start_ik_server()
        last_performance_time = time.perf_counter()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_start = time.perf_counter()
                if previous_loop_start is None:
                    dt = period
                else:
                    loop_period = loop_start - previous_loop_start
                    performance.loop_period_s.append(loop_period)
                    dt = clamp_dt(loop_period)
                previous_loop_start = loop_start

                snapshot = keyboard_state.snapshot()
                if snapshot.listener_error is not None:
                    if not listener_error_reported:
                        print(
                            "KEYBOARD_LISTENER_ERROR "
                            f"reason={snapshot.listener_error}",
                            flush=True,
                        )
                        listener_error_reported = True
                    break
                if snapshot.exit_requested:
                    break
                if (
                    listener_started
                    and listener is not None
                    and not listener.is_alive()
                ):
                    if not listener_error_reported:
                        print(
                            "KEYBOARD_LISTENER_ERROR "
                            "reason=listener_stopped_unexpectedly",
                            flush=True,
                        )
                        listener_error_reported = True
                    break

                if require_space_release and not snapshot.space_down:
                    require_space_release = False

                enabled = snapshot.space_down and not require_space_release
                disable_event: tuple[float, float, float] | None = None

                if enabled and not previous_enabled:
                    joint_jump_latched = False
                    print("ENABLED", flush=True)

                normal_disable = (
                    previous_enabled
                    and not snapshot.space_down
                    and not require_space_release
                )
                if normal_disable:
                    disable_detect_time = time.perf_counter()
                    position_backlog_m = float(
                        np.linalg.norm(
                            desired_position - commanded_position
                        )
                    )
                    rotation_backlog_rad = rotation_distance_rad(
                        desired_rotation,
                        commanded_rotation,
                    )
                    desired_position = commanded_position.copy()
                    desired_rotation = commanded_rotation.copy()
                    disable_event = (
                        disable_detect_time,
                        position_backlog_m,
                        rotation_backlog_rad,
                    )

                active_keys = snapshot.pressed_keys if enabled else set()
                if enabled:
                    desired_position = integrate_desired_position(
                        desired_position,
                        active_keys,
                        initial_position,
                        dt,
                    )
                    desired_rotation = integrate_desired_rotation(
                        desired_rotation,
                        active_keys,
                        dt,
                    )

                commanded_position = step_position_toward(
                    commanded_position,
                    desired_position,
                    dt,
                )
                commanded_rotation = step_rotation_toward(
                    commanded_rotation,
                    desired_rotation,
                    dt,
                )

                # KDL always receives the complete 6D TCP target.
                target_pe321 = pose_to_pe321(
                    commanded_position,
                    commanded_rotation,
                )

                ik_error: Exception | None = None
                new_joints_rad: np.ndarray | None = None
                ik_start = time.perf_counter()
                try:
                    new_joints_rad = solve_ik(
                        ik_process,
                        target_pe321,
                        current_joints_rad,
                    )
                except Exception as error:
                    ik_error = error
                finally:
                    performance.ik_duration_s.append(
                        time.perf_counter() - ik_start
                    )

                ik_accepted = False
                if ik_error is not None:
                    desired_position = last_valid_position.copy()
                    desired_rotation = last_valid_rotation.copy()
                    commanded_position = last_valid_position.copy()
                    commanded_rotation = last_valid_rotation.copy()
                    current_joints_rad = last_valid_joints_rad.copy()
                    require_space_release = True
                    enabled = False
                    if not ik_failure_latched:
                        print(
                            f"IK_FAIL reason={event_reason(ik_error)}",
                            flush=True,
                        )
                    ik_failure_latched = True
                elif new_joints_rad is not None:
                    ik_failure_latched = False
                    (
                        joint_step_ok,
                        largest_joint_index,
                        joint_delta_deg,
                        allowed_joint_step_deg,
                    ) = check_joint_step_limit(
                        current_joints_rad,
                        new_joints_rad,
                        dt,
                    )

                    if not joint_step_ok:
                        desired_position = last_valid_position.copy()
                        desired_rotation = last_valid_rotation.copy()
                        commanded_position = last_valid_position.copy()
                        commanded_rotation = last_valid_rotation.copy()
                        current_joints_rad = last_valid_joints_rad.copy()
                        require_space_release = True
                        enabled = False
                        if not joint_jump_latched:
                            print(
                                "JOINT_JUMP_REJECT "
                                f"joint=J{largest_joint_index + 1} "
                                f"delta={joint_delta_deg[largest_joint_index]:.1f}deg "
                                f"limit={allowed_joint_step_deg:.1f}deg "
                                f"dt={dt * 1000.0:.1f}ms",
                                flush=True,
                            )
                        joint_jump_latched = True
                    else:
                        current_joints_rad = new_joints_rad
                        ik_accepted = True

                set_joint_positions(
                    data,
                    qpos_indices,
                    current_joints_rad,
                )
                actual_position, actual_rotation = get_ee_pose(model, data)
                viewer.sync()

                if ik_accepted:
                    last_valid_position = commanded_position.copy()
                    last_valid_rotation = commanded_rotation.copy()
                    last_valid_joints_rad = current_joints_rad.copy()

                if disable_event is not None:
                    (
                        disable_detect_time,
                        position_backlog_m,
                        rotation_backlog_rad,
                    ) = disable_event
                    # This host-side tail starts when the control loop observes
                    # SPACE release and ends after this frame's viewer sync.
                    disable_stop_tail_ms = (
                        time.perf_counter() - disable_detect_time
                    ) * 1000.0
                    print(
                        "DISABLED "
                        f"backlog={position_backlog_m * 1000.0:.1f}mm/"
                        f"{math.degrees(rotation_backlog_rad):.1f}deg "
                        f"stop_tail={disable_stop_tail_ms:.1f}ms",
                        flush=True,
                    )

                if loop_start - last_performance_time >= 1.0:
                    print_performance_line(
                        performance,
                        desired_position,
                        desired_rotation,
                        commanded_position,
                        commanded_rotation,
                        actual_position,
                        actual_rotation,
                        enabled,
                    )
                    performance.clear()
                    last_performance_time = loop_start

                previous_enabled = enabled

                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Exiting.")
    finally:
        try:
            if listener_started and listener is not None:
                listener.stop()
                try:
                    listener.join(timeout=2.0)
                    if listener.is_alive() and not listener_error_reported:
                        print(
                            "KEYBOARD_LISTENER_ERROR "
                            "reason=listener_join_timeout",
                            flush=True,
                        )
                except Exception as error:
                    if not listener_error_reported:
                        print(
                            "KEYBOARD_LISTENER_ERROR "
                            f"reason={event_reason(error)}",
                            flush=True,
                        )
        finally:
            stop_ik_server(ik_process)

    print("\nProgram exited")


if __name__ == "__main__":
    main()
