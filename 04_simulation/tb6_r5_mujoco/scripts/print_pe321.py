from pathlib import Path
import argparse
import math

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")


def rotation_matrix_to_pe321(rotation: np.ndarray):
    """
    将旋转矩阵转换为：
    RZ、RY、RX，也就是 Z-Y-X 欧拉角。
    返回顺序为 rz, ry, rx，单位为弧度。
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="输出 TB6 R5 末端完整 PE321 位姿"
    )
    parser.add_argument(
        "angles",
        nargs=6,
        type=float,
        help="joint1 到 joint6 的角度，单位为度",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)

    model.opt.gravity[:] = 0.0

    for index, angle_deg in enumerate(args.angles):
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

    pe321 = np.array([
        position[0],
        position[1],
        position[2],
        rz,
        ry,
        rx,
    ])

    print("输入关节角（度）：")
    print(np.array(args.angles))

    print("\nPE321，顺序 [x, y, z, rz, ry, rx]：")
    print(pe321)

    print("\n位置单位：米")
    print("姿态单位：弧度")

    print("\n姿态角（度），顺序 [rz, ry, rx]：")
    print(np.rad2deg([rz, ry, rx]))
    


if __name__ == "__main__":
    main()
