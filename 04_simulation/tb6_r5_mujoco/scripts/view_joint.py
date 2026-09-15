from pathlib import Path
import argparse

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查看 TB6 R5 单个关节转动效果"
    )
    parser.add_argument(
        "joint",
        choices=[f"joint{i}" for i in range(1, 7)],
        help="要设置的关节，例如 joint1",
    )
    parser.add_argument(
        "angle_deg",
        type=float,
        help="目标角度，单位为度，例如 20",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)

    # 暂时关闭重力，让模型保持设定姿态
    model.opt.gravity[:] = 0.0

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        args.joint,
    )

    if joint_id < 0:
        raise RuntimeError(f"没有找到关节：{args.joint}")

    angle_rad = np.deg2rad(args.angle_deg)

    # 检查是否超过 URDF 中的关节限位
    if model.jnt_limited[joint_id]:
        lower, upper = model.jnt_range[joint_id]
        if not lower <= angle_rad <= upper:
            raise ValueError(
                f"{args.joint} 的 {args.angle_deg}° 超过限位："
                f"{np.rad2deg(lower):.1f}° 到 "
                f"{np.rad2deg(upper):.1f}°"
            )

    qpos_index = model.jnt_qposadr[joint_id]
    data.qpos[qpos_index] = angle_rad

    mujoco.mj_forward(model, data)

    print(f"{args.joint} 已设置为 {args.angle_deg:.1f}°")
    print("关闭 MuJoCo 窗口即可结束程序")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()