from pathlib import Path
import argparse
import math
import shlex
import subprocess

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

IK_PROGRAM = str(PROJECT_ROOT / "ik_test" / "build" / "ik_cli")

# 暂时使用已经验证过的初始姿态
SEED_JOINTS_DEG = [0.0, 20.0, -30.0, 0.0, 10.0, 0.0]


def rotation_matrix_to_pe321(rotation: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵转为 [rz, ry, rx]。"""
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


def set_joint_angles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_deg: list[float],
) -> None:
    for index, angle_deg in enumerate(joints_deg):
        joint_name = f"joint{index + 1}"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"没有找到 {joint_name}")

        qpos_index = model.jnt_qposadr[joint_id]
        data.qpos[qpos_index] = np.deg2rad(angle_deg)


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
        raise RuntimeError("没有找到 ee_Link")

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


def solve_ik(
    target_pe321: np.ndarray,
    seed_joints_deg: list[float],
) -> list[float]:
    arguments = [
        *target_pe321.tolist(),
        *seed_joints_deg,
    ]

    argument_text = " ".join(
        shlex.quote(f"{value:.12f}")
        for value in arguments
    )

    command = (
        "source /opt/ros/humble/setup.bash >/dev/null 2>&1 && "
        f"TB6_IK_URDF={shlex.quote(URDF_PATH)} "
        f"{shlex.quote(IK_PROGRAM)} "
        f"{argument_text}"
    )

    result = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "IK 求解失败：\n"
            + result.stdout
            + result.stderr
        )

    for line in result.stdout.splitlines():
        if line.startswith("q_target_deg:"):
            values = line.split(":")[1].strip().split()
            return [float(value) for value in values]

    raise RuntimeError(
        "没有在 IK 输出中找到 q_target_deg：\n"
        + result.stdout
        + result.stderr
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="让 TB6 R5 末端移动一个指定增量"
    )

    parser.add_argument("dx", type=float, help="X 增量，单位米")
    parser.add_argument("dy", type=float, help="Y 增量，单位米")
    parser.add_argument("dz", type=float, help="Z 增量，单位米")

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)

    # 暂时关闭重力
    model.opt.gravity[:] = 0.0

    set_joint_angles(model, data, SEED_JOINTS_DEG)

    initial_pe321 = get_ee_pe321(model, data)

    target_pe321 = initial_pe321.copy()
    target_pe321[0] += args.dx
    target_pe321[1] += args.dy
    target_pe321[2] += args.dz

    q_target_deg = solve_ik(
        target_pe321,
        SEED_JOINTS_DEG,
    )

    set_joint_angles(model, data, q_target_deg)

    actual_pe321 = get_ee_pe321(model, data)

    position_error_mm = (
        np.linalg.norm(
            actual_pe321[:3] - target_pe321[:3]
        )
        * 1000.0
    )

    print("初始 PE321：")
    print(initial_pe321)

    print("\n目标 PE321：")
    print(target_pe321)

    print("\nIK 关节角（度）：")
    print(np.array(q_target_deg))

    print("\nMuJoCo 实际 PE321：")
    print(actual_pe321)

    print(
        f"\n末端位置误差："
        f"{position_error_mm:.6f} mm"
    )

    print("\n关闭 MuJoCo 窗口即可结束程序")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()