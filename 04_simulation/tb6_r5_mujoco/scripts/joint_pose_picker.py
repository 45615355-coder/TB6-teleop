from __future__ import annotations

from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = str(PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf")

# 姿态选择器启动时的初始值，单位：度
START_JOINTS_DEG = np.array(
    [0.0, 20.0, -30.0, 0.0, 10.0, 0.0],
    dtype=float,
)


def get_joint_qpos_indices(
    model: mujoco.MjModel,
) -> list[int]:
    indices: list[int] = []

    for number in range(1, 7):
        joint_name = f"joint{number}"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"找不到关节：{joint_name}")

        indices.append(int(model.jnt_qposadr[joint_id]))

    return indices


def read_joint_degrees(
    data: mujoco.MjData,
    qpos_indices: list[int],
) -> np.ndarray:
    joints_rad = np.array(
        [data.qpos[index] for index in qpos_indices],
        dtype=float,
    )
    return np.rad2deg(joints_rad)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)

    # 只找姿态，不做动力学测试。
    model.opt.gravity[:] = 0.0

    qpos_indices = get_joint_qpos_indices(model)

    start_joints_rad = np.deg2rad(START_JOINTS_DEG)

    for joint_index, qpos_index in enumerate(qpos_indices):
        data.qpos[qpos_index] = start_joints_rad[joint_index]

    mujoco.mj_forward(model, data)

    print("========== TB6 R5 初始姿态选择器 ==========")
    print("1. 在MuJoCo窗口右上角展开 Joint")
    print("2. 保持 Simulation 为 Pause，不要点击 Run")
    print("3. 拖动 joint1～joint6 的滑块")
    print("4. 找到合适姿态后，直接关闭MuJoCo窗口")
    print("5. 关闭后终端会打印最终六个关节角")
    print()
    print(
        "启动角度 deg:",
        np.array2string(
            START_JOINTS_DEG,
            precision=3,
            separator=", ",
        ),
    )
    print("===========================================")

    # 使用MuJoCo完整交互式查看器。
    # 右侧Joint面板可以直接修改data.qpos。
    mujoco.viewer.launch(model, data)

    final_joints_deg = read_joint_degrees(
        data,
        qpos_indices,
    )

    print()
    print("最终选择的关节角 [J1,J2,J3,J4,J5,J6] deg:")
    print(
        np.array2string(
            final_joints_deg,
            precision=6,
            separator=", ",
            suppress_small=True,
        )
    )

    print()
    print("可直接复制到遥操作程序：")
    print("INITIAL_JOINTS_DEG = np.array(")
    print(
        "    "
        + np.array2string(
            final_joints_deg,
            precision=6,
            separator=", ",
            suppress_small=True,
        )
        + ","
    )
    print("    dtype=float,")
    print(")")


if __name__ == "__main__":
    main()
