from pathlib import Path
import argparse

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="设置 TB6 R5 六个关节角，并查看末端位置"
    )

    parser.add_argument(
        "angles",
        nargs=6,
        type=float,
        help="joint1～joint6 的角度，单位为度",
    )

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)

    # 暂时关闭重力，避免关节下垂
    model.opt.gravity[:] = 0.0

    for index in range(6):
        joint_name = f"joint{index + 1}"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"没有找到 {joint_name}")

        angle_rad = np.deg2rad(args.angles[index])

        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]

            if not lower <= angle_rad <= upper:
                raise ValueError(
                    f"{joint_name} 的 {args.angles[index]}° 超过限位："
                    f"{np.rad2deg(lower):.1f}° 到 "
                    f"{np.rad2deg(upper):.1f}°"
                )

        qpos_index = model.jnt_qposadr[joint_id]
        data.qpos[qpos_index] = angle_rad

    # 根据六个关节角计算所有连杆位置
    mujoco.mj_forward(model, data)

    ee_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ee_Link",
    )

    if ee_body_id < 0:
        raise RuntimeError("没有找到 ee_Link")

    ee_position = data.xpos[ee_body_id]

    print("六个关节角（度）：")
    print(np.array(args.angles))

    print("\nee_Link 位置（米）：")
    print(
        f"x = {ee_position[0]:.6f}\n"
        f"y = {ee_position[1]:.6f}\n"
        f"z = {ee_position[2]:.6f}"
    )

    print("\n关闭 MuJoCo 窗口即可结束")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()