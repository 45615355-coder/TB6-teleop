from pathlib import Path
import math
import os
import subprocess

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

KDL_PROGRAM = str(PROJECT_ROOT / "ik_test" / "build" / "fk_kdl_test")


def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """KDL RPY: R = Rz(rz) @ Ry(ry) @ Rx(rx)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ])

    ry_matrix = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])

    rz_matrix = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return rz_matrix @ ry_matrix @ rx_matrix


def get_mujoco_pose(
    model: mujoco.MjModel,
    angles_deg: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)

    for index, angle_deg in enumerate(angles_deg):
        joint_name = f"joint{index + 1}"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"MuJoCo 中找不到 {joint_name}")

        qpos_index = model.jnt_qposadr[joint_id]
        data.qpos[qpos_index] = np.deg2rad(angle_deg)

    mujoco.mj_forward(model, data)

    ee_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ee_Link",
    )

    if ee_body_id < 0:
        raise RuntimeError("MuJoCo 中找不到 ee_Link")

    position = data.xpos[ee_body_id].copy()
    rotation = data.xmat[ee_body_id].reshape(3, 3).copy()

    return position, rotation


def get_kdl_pose(
    angles_deg: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    angle_text = " ".join(str(value) for value in angles_deg)

    command = (
        "source /opt/ros/humble/setup.bash >/dev/null 2>&1 && "
        f'TB6_IK_URDF="{URDF_PATH}" '
        f'"{KDL_PROGRAM}" {angle_text}'
    )

    result = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        capture_output=True,
        check=True,
    )

    values = None

    for line in reversed(result.stdout.splitlines()):
        parts = line.strip().split()

        if len(parts) != 6:
            continue

        try:
            values = [float(value) for value in parts]
            break
        except ValueError:
            continue

    if values is None:
        raise RuntimeError(
            "无法从 KDL 输出中读取 PE321：\n"
            + result.stdout
            + result.stderr
        )

    x, y, z, rz, ry, rx = values

    position = np.array([x, y, z])
    rotation = rpy_to_matrix(rx, ry, rz)

    return position, rotation


def rotation_error_deg(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    relative = first.T @ second

    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return math.degrees(math.acos(cosine))


def main() -> None:
    if not os.path.isfile(KDL_PROGRAM):
        raise FileNotFoundError(
            f"找不到 KDL 测试程序：{KDL_PROGRAM}"
        )

    model = mujoco.MjModel.from_xml_path(URDF_PATH)

    cases = [
        ("全部零位", [0, 0, 0, 0, 0, 0]),
        ("joint1 +20°", [20, 0, 0, 0, 0, 0]),
        ("joint2 +20°", [0, 20, 0, 0, 0, 0]),
        ("joint3 +20°", [0, 0, 20, 0, 0, 0]),
        ("joint4 +20°", [0, 0, 0, 20, 0, 0]),
        ("joint5 +20°", [0, 0, 0, 0, 20, 0]),
        ("joint6 +20°", [0, 0, 0, 0, 0, 20]),
        ("组合姿态A", [0, 20, -30, 0, 10, 0]),
	("组合姿态B", [10, -15, 25, 5, -20, 30]),
    ]

    print(
        f"{'测试姿态':<16}"
        f"{'位置误差(mm)':>16}"
        f"{'姿态误差(°)':>16}"
    )

    print("-" * 48)

    for name, angles in cases:
        mujoco_position, mujoco_rotation = get_mujoco_pose(
            model,
            angles,
        )

        kdl_position, kdl_rotation = get_kdl_pose(angles)

        position_error_mm = (
            np.linalg.norm(
                mujoco_position - kdl_position
            )
            * 1000.0
        )

        orientation_error_deg = rotation_error_deg(
            mujoco_rotation,
            kdl_rotation,
        )

        print(
            f"{name:<16}"
            f"{position_error_mm:>16.3f}"
            f"{orientation_error_deg:>16.3f}"
        )


if __name__ == "__main__":
    main()
