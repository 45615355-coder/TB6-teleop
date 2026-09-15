from pathlib import Path
import math
import shlex
import subprocess
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

IK_SERVER_PATH = str(PROJECT_ROOT / "ik_test" / "build" / "ik_server")

INITIAL_JOINTS_DEG = np.array(
    [0.0, 20.0, -30.0, 0.0, 10.0, 0.0]
)

MOVE_DISTANCE_X = 0.10  # 末端沿 X 方向移动 10 cm
MOVE_TIME = 3.0         # 运动时间 3 秒
CONTROL_HZ = 50         # 每秒计算 50 次


def rotation_matrix_to_pe321(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """旋转矩阵转为 rz、ry、rx。"""
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
    indices = []

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

    return np.array([
        position[0],
        position[1],
        position[2],
        rz,
        ry,
        rx,
    ])


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

    values = np.concatenate([
        target_pe321,
        seed_joints_deg,
    ])

    request = " ".join(
        f"{value:.12f}"
        for value in values
    )

    process.stdin.write(request + "\n")
    process.stdin.flush()

    response = process.stdout.readline().strip()

    if response.startswith("ERR"):
        raise RuntimeError(f"IK 求解失败：{response}")

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

    # 当前先做纯运动学动画，不考虑重力
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

    final_target_pe321 = initial_pe321.copy()
    final_target_pe321[0] += MOVE_DISTANCE_X

    ik_process = start_ik_server()

    step_count = int(MOVE_TIME * CONTROL_HZ)
    period = 1.0 / CONTROL_HZ

    print("初始末端位置：")
    print(initial_pe321[:3])

    print("\n目标末端位置：")
    print(final_target_pe321[:3])

    print(
        f"\n开始运动：X +{MOVE_DISTANCE_X * 1000:.1f} mm，"
        f"时间 {MOVE_TIME:.1f} s"
    )

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:
            for step in range(step_count + 1):
                if not viewer.is_running():
                    break

                alpha = step / step_count

                target_pe321 = initial_pe321.copy()
                target_pe321[0] += (
                    MOVE_DISTANCE_X * alpha
                )

                current_joints_rad = solve_ik(
                    ik_process,
                    target_pe321,
                    current_joints_rad,
                )

                set_joint_positions(
                    data,
                    qpos_indices,
                    current_joints_rad,
                )

                mujoco.mj_forward(model, data)
                viewer.sync()

                time.sleep(period)

            actual_pe321 = get_ee_pe321(
                model,
                data,
            )

            position_error_mm = (
                np.linalg.norm(
                    actual_pe321[:3]
                    - final_target_pe321[:3]
                )
                * 1000.0
            )

            print("\n运动完成")
            print("最终实际位置：")
            print(actual_pe321[:3])

            print(
                f"末端位置误差："
                f"{position_error_mm:.6f} mm"
            )

            print("\n关闭窗口即可结束程序")

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

    finally:
        stop_ik_server(ik_process)


if __name__ == "__main__":
    main()
