#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

IK_SERVER_PATH = str(PROJECT_ROOT / "ik_test" / "build" / "ik_server")

INITIAL_JOINTS_DEG = np.array(
    [0.0, 20.0, -30.0, 0.0, 10.0, 0.0],
    dtype=float,
)

CONTROL_HZ = 50.0
GRIP_THRESHOLD = 0.9

XR_TRANSLATION_SCALE = 1.0
XR_ROTATION_SCALE = 1.0

MAX_TCP_SPEED_M_S = 0.20
MAX_ANGULAR_SPEED_DEG_S = 60.0

XR_STALE_TIMEOUT_S = 0.5
MAX_DT_S = 0.05
STATUS_INTERVAL_S = 1.0
WARNING_INTERVAL_S = 1.0
XR_INITIAL_WAIT_PRINT_INTERVAL_S = 1.0

# 按当前实测输入尺度判定 Quest 位姿单帧位置跳变。
XR_POSE_JUMP_POSITION_LIMIT = 0.20
XR_POSE_JUMP_ROTATION_LIMIT_DEG = 45.0

QUATERNION_NORM_MIN = 1e-6
QUATERNION_NORM_MAX = 10.0

POSITION_OFFSET_LIMIT_M = np.array([0.30, 0.30, 0.25], dtype=float)

R_XR_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

WAITING = "WAITING"
FRESH = "FRESH"
STALE = "STALE"

DISARMED = "DISARMED"
ARMED = "ARMED"


@dataclass
class XrSample:
    pose: np.ndarray
    grip: float
    timestamp_ns: int


@dataclass
class ValidControllerPose:
    position: np.ndarray
    rotation: np.ndarray
    quaternion_norm: float


@dataclass
class XrMonitor:
    last_timestamp_ns: int | None = None
    last_timestamp_change_time: float = 0.0
    status: str = WAITING


@dataclass
class GripClutch:
    state: str = DISARMED
    require_grip_release: bool = False
    controller_position_reference: np.ndarray | None = None
    controller_rotation_reference: np.ndarray | None = None
    robot_position_reference: np.ndarray | None = None
    robot_rotation_reference: np.ndarray | None = None


@dataclass
class LastAcceptedXrPose:
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None


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


def quaternion_xyzw_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion_xyzw

    return orthonormalize_rotation(
        np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=float,
        )
    )


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
    if joints_rad.size != 6 or not np.all(np.isfinite(joints_rad)):
        raise RuntimeError("写入 MuJoCo 的关节角非法")

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

        # 不创建未读取的 stderr 管道。
        # KDL/ROS 警告直接显示在当前终端，避免长时间运行后管道堵塞。
        stderr=None,

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
    values = np.concatenate([target_pe321, seed_joints_deg])

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


def initialize_xr_sdk() -> None:
    if xrt is None:
        print("缺少依赖 xrobotoolkit_sdk。请在 xrteleop 环境中运行：")
        print("conda activate xrteleop")
        raise RuntimeError("xrobotoolkit_sdk 未安装或不可导入")

    xrt.init()
    print("XRoboToolkit SDK initialized.")


def close_xr_sdk() -> None:
    if xrt is None:
        return

    try:
        xrt.close()
        print("XR SDK closed")
    except Exception as error:
        print(
            f"ERROR while closing XR SDK: {type(error).__name__}: {error}",
            file=sys.stderr,
        )


def read_xr_sample() -> XrSample:
    if xrt is None:
        raise RuntimeError("xrobotoolkit_sdk 未安装或不可导入")

    pose = np.asarray(
        list(xrt.get_right_controller_pose()),
        dtype=float,
    )
    grip = float(xrt.get_right_grip())
    timestamp_ns = int(xrt.get_time_stamp_ns())

    return XrSample(
        pose=pose,
        grip=grip,
        timestamp_ns=timestamp_ns,
    )


def validate_controller_pose(
    pose: np.ndarray,
) -> tuple[ValidControllerPose | None, str]:
    if pose.size != 7:
        return None, f"invalid right_controller pose length: {pose.size}"
    if not np.all(np.isfinite(pose)):
        return None, "right_controller pose contains NaN or Inf"
    if np.all(pose == 0.0) or np.all(pose[:3] == 0.0):
        return None, "right_controller pose is all zero or position is all zero"

    quaternion_xyzw = pose[3:7].copy()
    quaternion_norm = float(np.linalg.norm(quaternion_xyzw))
    if (
        not math.isfinite(quaternion_norm)
        or quaternion_norm < QUATERNION_NORM_MIN
        or quaternion_norm > QUATERNION_NORM_MAX
    ):
        return None, f"invalid quaternion norm: {quaternion_norm:.6f}"

    quaternion_xyzw /= quaternion_norm
    rotation = quaternion_xyzw_to_rotation(quaternion_xyzw)

    return (
        ValidControllerPose(
            position=pose[:3].copy(),
            rotation=rotation,
            quaternion_norm=quaternion_norm,
        ),
        "",
    )


def update_xr_monitor(
    monitor: XrMonitor,
    timestamp_ns: int,
    now: float,
    pose_valid: bool,
) -> str:
    # FRESH 只表示 XR 数据流 timestamp 正在更新，不证明手柄跟踪一定准确。
    if timestamp_ns == 0:
        monitor.status = WAITING
        return monitor.status

    if timestamp_ns != monitor.last_timestamp_ns:
        monitor.last_timestamp_change_time = now
        monitor.last_timestamp_ns = timestamp_ns
        monitor.status = FRESH if pose_valid else WAITING
        return monitor.status

    if now - monitor.last_timestamp_change_time > XR_STALE_TIMEOUT_S:
        monitor.status = STALE
    elif not pose_valid:
        monitor.status = WAITING
    else:
        monitor.status = FRESH

    return monitor.status


def disarm_clutch(
    clutch: GripClutch,
    require_grip_release: bool,
) -> None:
    clutch.state = DISARMED
    clutch.controller_position_reference = None
    clutch.controller_rotation_reference = None
    clutch.robot_position_reference = None
    clutch.robot_rotation_reference = None
    if require_grip_release:
        clutch.require_grip_release = True


def detect_xr_pose_jump(
    current_pose: ValidControllerPose,
    last_pose: LastAcceptedXrPose,
) -> str | None:
    if last_pose.position is None or last_pose.rotation is None:
        return None

    position_delta = float(
        np.linalg.norm(current_pose.position - last_pose.position)
    )
    _, rotation_delta_rad = rotation_error_axis_angle(
        current_pose.rotation,
        last_pose.rotation,
    )
    rotation_delta_deg = math.degrees(rotation_delta_rad)

    if position_delta > XR_POSE_JUMP_POSITION_LIMIT:
        return (
            "XR pose jump rejected: "
            f"position_delta={position_delta:.6f}"
        )
    if rotation_delta_deg > XR_POSE_JUMP_ROTATION_LIMIT_DEG:
        return (
            "XR pose jump rejected: "
            f"rotation_delta_deg={rotation_delta_deg:.3f}"
        )

    return None


def scale_rotation(
    rotation: np.ndarray,
    scale: float,
) -> np.ndarray:
    if abs(scale - 1.0) < 1e-12:
        return orthonormalize_rotation(rotation)

    axis, angle = rotation_error_axis_angle(
        rotation,
        np.eye(3, dtype=float),
    )
    return orthonormalize_rotation(
        axis_angle_to_rotation(axis, angle * scale)
    )


def update_grip_clutch(
    clutch: GripClutch,
    grip: float,
    controller_pose: ValidControllerPose,
    commanded_position: np.ndarray,
    commanded_rotation: np.ndarray,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
    initial_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grip_active = grip > GRIP_THRESHOLD

    if not grip_active:
        if clutch.state == ARMED or clutch.require_grip_release:
            disarm_clutch(clutch, require_grip_release=False)
        clutch.require_grip_release = False
        return desired_position, desired_rotation

    if clutch.require_grip_release:
        return desired_position, desired_rotation

    if clutch.state == DISARMED:
        clutch.controller_position_reference = controller_pose.position.copy()
        clutch.controller_rotation_reference = controller_pose.rotation.copy()
        clutch.robot_position_reference = commanded_position.copy()
        clutch.robot_rotation_reference = commanded_rotation.copy()
        clutch.state = ARMED
        print("ARMED: Grip references captured")
        return desired_position, desired_rotation

    if (
        clutch.controller_position_reference is None
        or clutch.controller_rotation_reference is None
        or clutch.robot_position_reference is None
        or clutch.robot_rotation_reference is None
    ):
        disarm_clutch(clutch, require_grip_release=True)
        return desired_position, desired_rotation

    controller_delta_xr = (
        controller_pose.position
        - clutch.controller_position_reference
    )
    controller_delta_robot = R_XR_TO_ROBOT @ controller_delta_xr
    next_desired_position = (
        clutch.robot_position_reference
        + XR_TRANSLATION_SCALE * controller_delta_robot
    )

    lower = initial_position - POSITION_OFFSET_LIMIT_M
    upper = initial_position + POSITION_OFFSET_LIMIT_M
    next_desired_position = np.clip(next_desired_position, lower, upper)

    controller_delta_rotation_xr = (
        controller_pose.rotation
        @ clutch.controller_rotation_reference.T
    )
    controller_delta_rotation_robot = (
        R_XR_TO_ROBOT
        @ controller_delta_rotation_xr
        @ R_XR_TO_ROBOT.T
    )
    controller_delta_rotation_robot = scale_rotation(
        controller_delta_rotation_robot,
        XR_ROTATION_SCALE,
    )
    next_desired_rotation = orthonormalize_rotation(
        controller_delta_rotation_robot
        @ clutch.robot_rotation_reference
    )

    return next_desired_position, next_desired_rotation


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


def format_vector(values: np.ndarray, precision: int = 3) -> str:
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


def print_startup_info(
    xr_status: str,
    grip: float,
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
) -> None:
    initial_pe321 = pe321_with_degrees(
        initial_position,
        initial_rotation,
    )

    print("\n========== Quest 3 Right Controller TB6 R5 MuJoCo ==========")
    print("当前仅做 MuJoCo 运动学仿真，不连接真实机器人，不发送真实机器人指令。")
    print(f"Quest连接状态: {xr_status}")
    print(f"Grip状态: {grip:.3f}")
    print(f"控制频率: {CONTROL_HZ:.1f} Hz")
    print(f"位置缩放 XR_TRANSLATION_SCALE: {XR_TRANSLATION_SCALE:.3f}")
    print(f"姿态缩放 XR_ROTATION_SCALE: {XR_ROTATION_SCALE:.3f}")
    print(f"最大平移速度: {MAX_TCP_SPEED_M_S * 1000.0:.1f} mm/s")
    print(f"最大角速度: {MAX_ANGULAR_SPEED_DEG_S:.1f} deg/s")
    print("坐标映射矩阵 R_XR_TO_ROBOT:")
    print(R_XR_TO_ROBOT)
    print(f"当前TB6初始关节角 deg: {format_vector(INITIAL_JOINTS_DEG)}")
    print(
        "当前TB6初始PE321 [m, m, m, deg, deg, deg]: "
        f"{format_vector(initial_pe321)}"
    )
    print("按住右 Grip 才建立离合并跟随右手柄；松开 Grip 保持目标。")
    print("关闭 MuJoCo 窗口或 Ctrl+C 退出。")
    print("============================================================\n")


def print_status(
    xr_status: str,
    grip: float,
    clutch: GripClutch,
    raw_controller_position: np.ndarray | None,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
    actual_position: np.ndarray,
    actual_rotation: np.ndarray,
    current_joints_rad: np.ndarray,
    ik_status: str,
) -> None:
    raw_text = (
        "None"
        if raw_controller_position is None
        else format_vector(raw_controller_position)
    )
    desired_pe321 = pe321_with_degrees(
        desired_position,
        desired_rotation,
    )
    actual_pe321 = pe321_with_degrees(
        actual_position,
        actual_rotation,
    )

    print(
        "status "
        f"xr={xr_status} "
        f"grip={grip:.3f} "
        f"clutch={clutch.state} "
        f"lockout={clutch.require_grip_release} "
        f"quest_pos={raw_text} "
        f"robot_target_pos={format_vector(desired_position)} "
        f"mujoco_pos={format_vector(actual_position)} "
        f"target_pe321_deg={format_vector(desired_pe321)} "
        f"actual_pe321_deg={format_vector(actual_pe321)} "
        f"joints_deg={format_vector(np.rad2deg(current_joints_rad))} "
        f"ik={ik_status}",
        flush=True,
    )


def print_warning_throttled(
    message: str,
    now: float,
    last_warning_time: float,
) -> float:
    if now - last_warning_time >= WARNING_INTERVAL_S:
        print(message, flush=True)
        return now
    return last_warning_time


def wait_for_initial_xr_data(monitor: XrMonitor) -> XrSample:
    last_print_time = 0.0

    while True:
        now = time.perf_counter()
        sample = read_xr_sample()
        controller_pose, _ = validate_controller_pose(sample.pose)
        update_xr_monitor(
            monitor,
            sample.timestamp_ns,
            now,
            controller_pose is not None,
        )

        if monitor.status == FRESH and controller_pose is not None:
            print("Quest XR data stream is FRESH.")
            return sample

        if now - last_print_time >= XR_INITIAL_WAIT_PRINT_INTERVAL_S:
            print(
                "Waiting for valid Quest right_controller data... "
                f"status={monitor.status} "
                f"timestamp_ns={sample.timestamp_ns} "
                f"grip={sample.grip:.3f}",
                flush=True,
            )
            last_print_time = now

        time.sleep(1.0 / CONTROL_HZ)


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

    last_valid_position = commanded_position.copy()
    last_valid_rotation = commanded_rotation.copy()
    last_valid_joints_rad = current_joints_rad.copy()

    clutch = GripClutch()
    monitor = XrMonitor(
        last_timestamp_change_time=time.perf_counter(),
    )
    last_accepted_xr_pose = LastAcceptedXrPose()

    ik_process: subprocess.Popen[str] | None = None
    xr_initialized = False
    period = 1.0 / CONTROL_HZ
    last_time = time.perf_counter()
    last_status_time = 0.0
    last_xr_warning_time = 0.0
    last_ik_warning_time = 0.0
    ik_status = "OK"
    last_sample: XrSample | None = None
    last_controller_position: np.ndarray | None = None

    try:
        initialize_xr_sdk()
        xr_initialized = True
        first_sample = wait_for_initial_xr_data(monitor)
        last_sample = first_sample

        print_startup_info(
            monitor.status,
            first_sample.grip,
            initial_position,
            initial_rotation,
        )

        ik_process = start_ik_server()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_start = time.perf_counter()
                dt = clamp_dt(loop_start - last_time)
                last_time = loop_start

                try:
                    sample = read_xr_sample()
                    last_sample = sample
                    grip = sample.grip
                    controller_pose, invalid_reason = validate_controller_pose(
                        sample.pose
                    )
                except Exception as error:
                    grip = 0.0 if last_sample is None else last_sample.grip
                    controller_pose = None
                    invalid_reason = (
                        f"XR read failed: {type(error).__name__}: {error}"
                    )

                xr_status = update_xr_monitor(
                    monitor,
                    0 if last_sample is None else last_sample.timestamp_ns,
                    loop_start,
                    controller_pose is not None,
                )

                if xr_status == STALE:
                    if clutch.state == ARMED or not clutch.require_grip_release:
                        disarm_clutch(clutch, require_grip_release=True)
                    desired_position = last_valid_position.copy()
                    desired_rotation = last_valid_rotation.copy()
                    commanded_position = last_valid_position.copy()
                    commanded_rotation = last_valid_rotation.copy()
                    current_joints_rad = last_valid_joints_rad.copy()
                    last_xr_warning_time = print_warning_throttled(
                        "XR data STALE: disarmed; release Grip before re-arming.",
                        loop_start,
                        last_xr_warning_time,
                    )
                    controller_pose = None

                if controller_pose is None:
                    if invalid_reason:
                        last_xr_warning_time = print_warning_throttled(
                            f"Invalid XR pose: {invalid_reason}. Target held.",
                            loop_start,
                            last_xr_warning_time,
                        )
                    if clutch.state == ARMED:
                        disarm_clutch(clutch, require_grip_release=True)
                        desired_position = last_valid_position.copy()
                        desired_rotation = last_valid_rotation.copy()
                        commanded_position = last_valid_position.copy()
                        commanded_rotation = last_valid_rotation.copy()
                        current_joints_rad = last_valid_joints_rad.copy()
                    elif grip <= GRIP_THRESHOLD:
                        disarm_clutch(clutch, require_grip_release=False)
                    raw_position = None
                else:
                    raw_position = controller_pose.position.copy()
                    grip_active = grip > GRIP_THRESHOLD
                    jump_reason = (
                        detect_xr_pose_jump(
                            controller_pose,
                            last_accepted_xr_pose,
                        )
                        if grip_active or clutch.state == ARMED
                        else None
                    )
                    if jump_reason is not None:
                        disarm_clutch(clutch, require_grip_release=True)
                        desired_position = last_valid_position.copy()
                        desired_rotation = last_valid_rotation.copy()
                        commanded_position = last_valid_position.copy()
                        commanded_rotation = last_valid_rotation.copy()
                        current_joints_rad = last_valid_joints_rad.copy()
                        last_xr_warning_time = print_warning_throttled(
                            (
                                f"{jump_reason}; disarmed; "
                                "release Grip before re-arming."
                            ),
                            loop_start,
                            last_xr_warning_time,
                        )
                    else:
                        last_accepted_xr_pose.position = (
                            controller_pose.position.copy()
                        )
                        last_accepted_xr_pose.rotation = (
                            controller_pose.rotation.copy()
                        )
                        desired_position, desired_rotation = update_grip_clutch(
                            clutch,
                            grip,
                            controller_pose,
                            commanded_position,
                            commanded_rotation,
                            desired_position,
                            desired_rotation,
                            initial_position,
                        )

                last_controller_position = raw_position

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
                    current_joints_rad = new_joints_rad
                    set_joint_positions(
                        data,
                        qpos_indices,
                        current_joints_rad,
                    )
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    last_valid_position = commanded_position.copy()
                    last_valid_rotation = commanded_rotation.copy()
                    last_valid_joints_rad = current_joints_rad.copy()
                    ik_status = "OK"

                except RuntimeError as error:
                    desired_position = last_valid_position.copy()
                    desired_rotation = last_valid_rotation.copy()
                    commanded_position = last_valid_position.copy()
                    commanded_rotation = last_valid_rotation.copy()
                    current_joints_rad = last_valid_joints_rad.copy()
                    disarm_clutch(clutch, require_grip_release=True)
                    ik_status = "FAIL"

                    set_joint_positions(
                        data,
                        qpos_indices,
                        current_joints_rad,
                    )
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    last_ik_warning_time = print_warning_throttled(
                        (
                            f"IK failed and rolled back: {error}; "
                            "release Grip before re-arming."
                        ),
                        loop_start,
                        last_ik_warning_time,
                    )

                actual_position, actual_rotation = get_ee_pose(model, data)

                if loop_start - last_status_time >= STATUS_INTERVAL_S:
                    print_status(
                        monitor.status,
                        grip,
                        clutch,
                        last_controller_position,
                        desired_position,
                        desired_rotation,
                        actual_position,
                        actual_rotation,
                        current_joints_rad,
                        ik_status,
                    )
                    last_status_time = loop_start

                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Exiting.")
    finally:
        stop_ik_server(ik_process)
        if xr_initialized:
            close_xr_sdk()

    print("\n程序已结束")


if __name__ == "__main__":
    main()
