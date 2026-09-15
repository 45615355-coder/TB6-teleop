from pathlib import Path
import math
import shlex
import subprocess
import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

IK_SERVER_PATH = str(PROJECT_ROOT / "ik_test" / "build" / "ik_server")

# 当前已经验证过的初始姿态，单位为度
INITIAL_JOINTS_DEG = np.array(
    [0.0, 20.0, -30.0, 0.0, 10.0, 0.0],
    dtype=float,
)

CONTROL_HZ = 50.0

# 每按一次按键，目标移动 1 cm
KEY_STEP_M = 0.01

# 末端相对初始位置最多移动 ±15 cm
MAX_OFFSET_M = 1.0

# 仿真中末端目标移动速度
MAX_TCP_SPEED_M_S = 0.08


def rotation_matrix_to_pe321(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """旋转矩阵转换为 rz、ry、rx。"""
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


def get_ee_pe321(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
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


def start_ik_server() -> subprocess.Popen:
    command = (
        "source /opt/ros/humble/setup.bash && "
        f"export TB6_IK_URDF={shlex.quote(URDF_PATH)} && "
        f"exec {shlex.quote(IK_SERVER_PATH)}"
    )

    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("无法读取 IK server 输出")

    first_line = process.stdout.readline().strip()

    if first_line != "READY":
        process.terminate()
        raise RuntimeError(
            f"IK server 启动失败，收到：{first_line}"
        )

    return process


def solve_ik(
    process: subprocess.Popen,
    target_pe321: np.ndarray,
    seed_joints_rad: np.ndarray,
) -> np.ndarray:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("IK server 输入输出不可用")

    seed_joints_deg = np.rad2deg(seed_joints_rad)

    values = np.concatenate(
        [
            target_pe321,
            seed_joints_deg,
        ]
    )

    request = " ".join(
        f"{value:.12f}"
        for value in values
    )

    process.stdin.write(request + "\n")
    process.stdin.flush()

    response = process.stdout.readline().strip()

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


def stop_ik_server(process: subprocess.Popen) -> None:
    try:
        if process.stdin is not None:
            process.stdin.write("quit\n")
            process.stdin.flush()

        process.wait(timeout=2.0)
    except Exception:
        process.terminate()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)

    # 当前做纯运动学调试，暂时关闭重力
    model.opt.gravity[:] = 0.0

    qpos_indices = get_joint_qpos_indices(model)

    current_joints_rad = np.deg2rad(
        INITIAL_JOINTS_DEG
    )

    set_joint_positions(
        data,
        qpos_indices,
        current_joints_rad,
    )

    initial_pe321 = get_ee_pe321(model, data)
    initial_position = initial_pe321[:3].copy()

    commanded_position = initial_position.copy()

    shared_state = {
        "desired_position": initial_position.copy(),
        "exit_requested": False,
    }

    state_lock = threading.Lock()

    def print_target(position: np.ndarray) -> None:
        offset_mm = (position - initial_position) * 1000.0

        print(
            "\n目标位置："
            f"[{position[0]:.6f}, "
            f"{position[1]:.6f}, "
            f"{position[2]:.6f}] m"
        )

        print(
            "相对初始位置："
            f"[{offset_mm[0]:.1f}, "
            f"{offset_mm[1]:.1f}, "
            f"{offset_mm[2]:.1f}] mm"
        )

    def key_callback(keycode: int) -> None:
        delta = np.zeros(3, dtype=float)

        if keycode == glfw.KEY_W:
            delta[0] = KEY_STEP_M
        elif keycode == glfw.KEY_S:
            delta[0] = -KEY_STEP_M
        elif keycode == glfw.KEY_A:
            delta[1] = KEY_STEP_M
        elif keycode == glfw.KEY_D:
            delta[1] = -KEY_STEP_M
        elif keycode == glfw.KEY_Q:
            delta[2] = KEY_STEP_M
        elif keycode == glfw.KEY_E:
            delta[2] = -KEY_STEP_M
        elif keycode == glfw.KEY_R:
            with state_lock:
                shared_state["desired_position"] = (
                    initial_position.copy()
                )

            print("\n目标位置已复位")
            print_target(initial_position)
            return
        elif keycode == glfw.KEY_P:
            with state_lock:
                target = shared_state[
                    "desired_position"
                ].copy()

            print_target(target)
            return
        elif keycode == glfw.KEY_X:
            with state_lock:
                shared_state["exit_requested"] = True

            print("\n收到退出指令")
            return
        else:
            return

        with state_lock:
            new_target = (
                shared_state["desired_position"] + delta
            )

            lower = initial_position - MAX_OFFSET_M
            upper = initial_position + MAX_OFFSET_M

            new_target = np.clip(
                new_target,
                lower,
                upper,
            )

            shared_state["desired_position"] = new_target

        print_target(new_target)

    ik_process = start_ik_server()

    period = 1.0 / CONTROL_HZ
    max_position_step = (
        MAX_TCP_SPEED_M_S / CONTROL_HZ
    )

    print("TB6 R5 键盘末端控制已启动")
    print("")
    print("W / S：X 正方向 / 负方向")
    print("A / D：Y 正方向 / 负方向")
    print("Q / E：Z 正方向 / 负方向")
    print("R：回到初始目标")
    print("P：打印当前目标")
    print("X：退出程序")
    print("")
    print(f"每次按键移动：{KEY_STEP_M * 1000:.0f} mm")
    print("末端姿态保持不变")
    print_target(initial_position)

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
        ) as viewer:
            while viewer.is_running():
                with state_lock:
                    desired_position = shared_state[
                        "desired_position"
                    ].copy()

                    exit_requested = bool(
                        shared_state["exit_requested"]
                    )

                if exit_requested:
                    break

                position_delta = (
                    desired_position - commanded_position
                )

                distance = float(
                    np.linalg.norm(position_delta)
                )

                if distance > max_position_step:
                    commanded_position += (
                        position_delta
                        / distance
                        * max_position_step
                    )
                else:
                    commanded_position = (
                        desired_position.copy()
                    )

                target_pe321 = initial_pe321.copy()
                target_pe321[:3] = commanded_position

                try:
                    new_joints_rad = solve_ik(
                        ik_process,
                        target_pe321,
                        current_joints_rad,
                    )
                except RuntimeError as error:
                    print(f"\nIK 求解失败：{error}")
                    print("目标已停在当前可达位置")

                    with state_lock:
                        shared_state[
                            "desired_position"
                        ] = commanded_position.copy()

                    time.sleep(period)
                    continue

                current_joints_rad = new_joints_rad

                set_joint_positions(
                    data,
                    qpos_indices,
                    current_joints_rad,
                )

                mujoco.mj_forward(model, data)
                viewer.sync()

                time.sleep(period)

    finally:
        stop_ik_server(ik_process)

    final_pe321 = get_ee_pe321(model, data)

    print("\n程序已结束")
    print("最终末端位置：")
    print(final_pe321[:3])


if __name__ == "__main__":
    main()
