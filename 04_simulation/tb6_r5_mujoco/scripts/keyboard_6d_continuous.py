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

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

IK_SERVER_PATH = str(PROJECT_ROOT / "ik_test" / "build" / "ik_server")

INITIAL_JOINTS_DEG = np.array(
    [0.0, 20.0, -30.0, 0.0, 10.0, 0.0],
    dtype=float,
)

CONTROL_HZ = 50.0
LINEAR_SPEED_M_S = 0.05
ANGULAR_SPEED_DEG_S = 20.0
POSITION_OFFSET_LIMIT_M = np.array([0.30, 0.30, 0.25], dtype=float)

MAX_DT_S = 0.05
IK_WARNING_INTERVAL_S = 1.0
STATUS_INTERVAL_S = 1.0

TRANSLATION_KEYS = {
    "l": np.array([1.0, 0.0, 0.0], dtype=float),
    "j": np.array([-1.0, 0.0, 0.0], dtype=float),
    "i": np.array([0.0, 1.0, 0.0], dtype=float),
    "k": np.array([0.0, -1.0, 0.0], dtype=float),
    "u": np.array([0.0, 0.0, 1.0], dtype=float),
    "o": np.array([0.0, 0.0, -1.0], dtype=float),
}

ROTATION_KEYS = {
    "h": np.array([1.0, 0.0, 0.0], dtype=float),
    "f": np.array([-1.0, 0.0, 0.0], dtype=float),
    "t": np.array([0.0, 1.0, 0.0], dtype=float),
    "g": np.array([0.0, -1.0, 0.0], dtype=float),
    "b": np.array([0.0, 0.0, 1.0], dtype=float),
    "v": np.array([0.0, 0.0, -1.0], dtype=float),
}

MOTION_KEYS = set(TRANSLATION_KEYS) | set(ROTATION_KEYS)


@dataclass
class KeyboardSnapshot:
    pressed_keys: set[str]
    right_shift_down: bool
    reset_requested: bool
    print_requested: bool
    exit_requested: bool


@dataclass
class KeyboardState:
    pressed_keys: set[str] = field(default_factory=set)
    right_shift_down: bool = False
    reset_requested: bool = False
    print_requested: bool = False
    exit_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> KeyboardSnapshot:
        with self.lock:
            snapshot = KeyboardSnapshot(
                pressed_keys=set(self.pressed_keys),
                right_shift_down=self.right_shift_down,
                reset_requested=self.reset_requested,
                print_requested=self.print_requested,
                exit_requested=self.exit_requested,
            )
            self.reset_requested = False
            self.print_requested = False
            return snapshot

    def clear_motion_keys(self) -> None:
        with self.lock:
            self.pressed_keys.clear()


def rotation_matrix_to_pe321(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """旋转矩阵转换为 rz、ry、rx，单位为弧度。"""
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


def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


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


def get_joint_qpos_indices(
    model: mujoco.MjModel,
) -> list[int]:
    indices: list[int] = []

    for index in range(6):
        joint_name = f"joint{index + 1}"
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"找不到 {joint_name}")

        indices.append(int(model.jnt_qposadr[joint_id]))

    return indices


def set_joint_positions(
    data: mujoco.MjData,
    qpos_indices: list[int],
    joints_rad: np.ndarray,
) -> None:
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
        raise RuntimeError("找不到 ee_Link")

    position = data.xpos[ee_body_id].copy()
    rotation = data.xmat[ee_body_id].reshape(3, 3).copy()

    return position, orthonormalize_rotation(rotation)


def pose_to_pe321(
    position: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    rz, ry, rx = rotation_matrix_to_pe321(rotation)
    return np.array(
        [
            position[0],
            position[1],
            position[2],
            rz,
            ry,
            rx,
        ],
        dtype=float,
    )


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
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("无法读取 IK server 输出")
    if process.stdin is None:
        raise RuntimeError("无法写入 IK server 输入")

    first_line = process.stdout.readline().strip()

    if first_line != "READY":
        process.terminate()
        raise RuntimeError(
            f"IK server 启动失败，收到：{first_line}"
        )

    return process


def solve_ik(
    process: subprocess.Popen[str],
    target_pe321: np.ndarray,
    seed_joints_rad: np.ndarray,
) -> np.ndarray:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("IK server 输入输出不可用")
    if process.poll() is not None:
        raise RuntimeError(
            f"IK server 已退出，返回码：{process.returncode}"
        )

    seed_joints_deg = np.rad2deg(seed_joints_rad)

    values = np.concatenate(
        [
            target_pe321,
            seed_joints_deg,
        ]
    )

    if values.size != 12 or not np.all(np.isfinite(values)):
        raise RuntimeError("IK 请求包含非法数值")

    request = " ".join(f"{value:.12f}" for value in values)

    process.stdin.write(request + "\n")
    process.stdin.flush()

    response = process.stdout.readline().strip()
    if response == "":
        raise RuntimeError("IK server 无输出")

    if response.startswith("ERR"):
        raise RuntimeError(response)

    if not response.startswith("OK "):
        raise RuntimeError(
            f"IK server 返回格式异常：{response}"
        )

    joints_rad = np.array(
        [float(value) for value in response.split()[1:]],
        dtype=float,
    )

    if joints_rad.size != 6:
        raise RuntimeError(
            f"IK 返回关节数量错误：{joints_rad.size}"
        )

    if not np.all(np.isfinite(joints_rad)):
        raise RuntimeError("IK 返回 NaN 或 Inf")

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


def key_to_char(key: Any) -> str | None:
    char = getattr(key, "char", None)
    if not isinstance(char, str) or len(char) == 0:
        return None
    return char.lower()


def create_keyboard_listener(state: KeyboardState) -> Any:
    if pynput_keyboard is None:
        raise RuntimeError("pynput 未安装")

    def on_press(key: Any) -> bool | None:
        with state.lock:
            if key == pynput_keyboard.Key.shift_r:
                state.right_shift_down = True
                return None

            if key == pynput_keyboard.Key.esc:
                state.exit_requested = True
                return False

            if not state.right_shift_down:
                return None

            if key == pynput_keyboard.Key.enter:
                state.reset_requested = True
                state.pressed_keys.clear()
                return None

            char = key_to_char(key)
            if char is None:
                return None

            if char == "p":
                state.print_requested = True
            elif char in MOTION_KEYS:
                state.pressed_keys.add(char)

        return None

    def on_release(key: Any) -> bool | None:
        with state.lock:
            if key == pynput_keyboard.Key.shift_r:
                state.right_shift_down = False
                state.pressed_keys.clear()
                return None

            char = key_to_char(key)
            if char is not None:
                state.pressed_keys.discard(char)

        return None

    return pynput_keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )


def integrate_desired_position(
    desired_position: np.ndarray,
    pressed_keys: set[str],
    initial_position: np.ndarray,
    dt: float,
) -> np.ndarray:
    direction = np.zeros(3, dtype=float)

    for key in pressed_keys:
        direction += TRANSLATION_KEYS.get(
            key,
            np.zeros(3, dtype=float),
        )

    norm = float(np.linalg.norm(direction))
    if norm > 1e-12:
        direction /= norm
        desired_position = (
            desired_position
            + direction * LINEAR_SPEED_M_S * dt
        )

    lower = initial_position - POSITION_OFFSET_LIMIT_M
    upper = initial_position + POSITION_OFFSET_LIMIT_M
    return np.clip(desired_position, lower, upper)


def integrate_desired_rotation(
    desired_rotation: np.ndarray,
    pressed_keys: set[str],
    dt: float,
) -> np.ndarray:
    direction = np.zeros(3, dtype=float)

    for key in pressed_keys:
        direction += ROTATION_KEYS.get(
            key,
            np.zeros(3, dtype=float),
        )

    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return desired_rotation

    direction /= norm
    angle = math.radians(ANGULAR_SPEED_DEG_S) * dt

    # 姿态控制基于 base/world 坐标系，因此增量旋转左乘。
    delta_rotation = axis_angle_to_rotation(direction, angle)
    return orthonormalize_rotation(delta_rotation @ desired_rotation)


def step_position_toward(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
) -> np.ndarray:
    delta = target - current
    distance = float(np.linalg.norm(delta))
    max_step = LINEAR_SPEED_M_S * dt

    if distance <= max_step or distance < 1e-12:
        return target.copy()

    return current + delta / distance * max_step


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


def step_rotation_toward(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
) -> np.ndarray:
    axis, angle = rotation_error_axis_angle(target, current)
    max_step = math.radians(ANGULAR_SPEED_DEG_S) * dt

    if angle <= max_step or angle < 1e-12:
        return orthonormalize_rotation(target)

    delta = axis_angle_to_rotation(axis, max_step)
    return orthonormalize_rotation(delta @ current)


def format_vector(values: np.ndarray, precision: int = 1) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def pe321_with_degrees(
    position: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    rz, ry, rx = rotation_matrix_to_pe321(rotation)
    return np.array(
        [
            position[0],
            position[1],
            position[2],
            math.degrees(rz),
            math.degrees(ry),
            math.degrees(rx),
        ],
        dtype=float,
    )


def print_instructions() -> None:
    print("TB6 R5 末端六自由度连续键盘控制已启动")
    print("")
    print("必须按住右 Shift，以下运动按键才有效。松开右 Shift 立即停止目标积分。")
    print("位置控制（base/world 坐标系）：")
    print("  L / J：X 正方向 / 负方向")
    print("  I / K：Y 正方向 / 负方向")
    print("  U / O：Z 正方向 / 负方向")
    print("姿态控制（绕 base/world 坐标系轴旋转）：")
    print("  H / F：RX 正方向 / 负方向")
    print("  T / G：RY 正方向 / 负方向")
    print("  B / V：RZ 正方向 / 负方向")
    print("其他：")
    print("  右 Shift + Enter：平滑复位到启动时初始位姿")
    print("  右 Shift + P：打印当前状态")
    print("  Esc：安全退出程序")
    print("")
    print(f"线速度：{LINEAR_SPEED_M_S * 1000.0:.1f} mm/s")
    print(f"角速度：{ANGULAR_SPEED_DEG_S:.1f} deg/s")
    print("")


def print_status(
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
    actual_position: np.ndarray,
    actual_rotation: np.ndarray,
    current_joints_rad: np.ndarray,
    right_shift_down: bool,
) -> None:
    desired_angles_deg = np.rad2deg(
        np.array(rotation_matrix_to_pe321(desired_rotation), dtype=float)
    )
    actual_angles_deg = np.rad2deg(
        np.array(rotation_matrix_to_pe321(actual_rotation), dtype=float)
    )
    joints_deg = np.rad2deg(current_joints_rad)
    desired_pe321 = pe321_with_degrees(
        desired_position,
        desired_rotation,
    )
    actual_pe321 = pe321_with_degrees(
        actual_position,
        actual_rotation,
    )

    print("")
    print("当前状态")
    print(
        "  目标 PE321 [m, m, m, deg, deg, deg]："
        f"{format_vector(desired_pe321, precision=3)}"
    )
    print(
        "  实际 PE321 [m, m, m, deg, deg, deg]："
        f"{format_vector(actual_pe321, precision=3)}"
    )
    print(f"  目标位置 mm：{format_vector(desired_position * 1000.0)}")
    print(f"  实际位置 mm：{format_vector(actual_position * 1000.0)}")
    print(
        "  目标姿态 deg [rz, ry, rx]："
        f"{format_vector(desired_angles_deg)}"
    )
    print(
        "  实际姿态 deg [rz, ry, rx]："
        f"{format_vector(actual_angles_deg)}"
    )
    print(f"  当前关节角 deg：{format_vector(joints_deg)}")
    print(f"  右 Shift：{'按下' if right_shift_down else '未按下'}")


def clamp_dt(dt: float) -> float:
    if dt < 0.0:
        return 0.0
    if dt > MAX_DT_S:
        return MAX_DT_S
    return dt


def main() -> None:
    if pynput_keyboard is None:
        print("缺少依赖 pynput，请执行：")
        print("conda activate xrteleop")
        print("python -m pip install pynput")
        return

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

    last_valid_position = commanded_position.copy()
    last_valid_rotation = commanded_rotation.copy()
    last_valid_joints_rad = current_joints_rad.copy()

    keyboard_state = KeyboardState()
    listener = create_keyboard_listener(keyboard_state)
    listener_started = False
    ik_process: subprocess.Popen[str] | None = None

    period = 1.0 / CONTROL_HZ
    last_time = time.perf_counter()
    last_status_time = 0.0
    last_ik_warning_time = 0.0

    print_instructions()

    try:
        ik_process = start_ik_server()
        listener.start()
        listener_started = True

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                now = time.perf_counter()
                dt = clamp_dt(now - last_time)
                last_time = now

                snapshot = keyboard_state.snapshot()
                if snapshot.exit_requested:
                    break

                if snapshot.reset_requested:
                    desired_position = initial_position.copy()
                    desired_rotation = initial_rotation.copy()
                    keyboard_state.clear_motion_keys()

                active_keys = (
                    snapshot.pressed_keys
                    if snapshot.right_shift_down
                    else set()
                )

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

                target_pe321 = pose_to_pe321(
                    commanded_position,
                    commanded_rotation,
                )

                try:
                    new_joints_rad = solve_ik(
                        ik_process,
                        target_pe321,
                        current_joints_rad,
                    )
                except RuntimeError as error:
                    warning_now = time.perf_counter()
                    if (
                        warning_now - last_ik_warning_time
                        >= IK_WARNING_INTERVAL_S
                    ):
                        print(f"\nIK 求解失败，已回退：{error}")
                        last_ik_warning_time = warning_now

                    desired_position = last_valid_position.copy()
                    desired_rotation = last_valid_rotation.copy()
                    commanded_position = last_valid_position.copy()
                    commanded_rotation = last_valid_rotation.copy()
                    current_joints_rad = last_valid_joints_rad.copy()
                    keyboard_state.clear_motion_keys()

                    set_joint_positions(
                        data,
                        qpos_indices,
                        current_joints_rad,
                    )
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(period)
                    continue

                current_joints_rad = new_joints_rad
                set_joint_positions(data, qpos_indices, current_joints_rad)
                mujoco.mj_forward(model, data)
                viewer.sync()

                last_valid_position = commanded_position.copy()
                last_valid_rotation = commanded_rotation.copy()
                last_valid_joints_rad = current_joints_rad.copy()

                actual_position, actual_rotation = get_ee_pose(model, data)
                if (
                    snapshot.print_requested
                    and now - last_status_time >= STATUS_INTERVAL_S
                ):
                    print_status(
                        desired_position,
                        desired_rotation,
                        actual_position,
                        actual_rotation,
                        current_joints_rad,
                        snapshot.right_shift_down,
                    )
                    last_status_time = time.perf_counter()
                elif now - last_status_time >= STATUS_INTERVAL_S:
                    print_status(
                        desired_position,
                        desired_rotation,
                        actual_position,
                        actual_rotation,
                        current_joints_rad,
                        snapshot.right_shift_down,
                    )
                    last_status_time = now

                sleep_time = period - (time.perf_counter() - now)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    finally:
        try:
            if listener_started:
                listener.stop()
        finally:
            stop_ik_server(ik_process)

    print("\n程序已结束")


if __name__ == "__main__":
    main()
