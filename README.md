# TB6-teleop

基于 **Meta Quest 3、TB6-R5 机械臂、OmniPicker 夹爪和三路 Intel RealSense D435** 的遥操作与模仿学习项目，覆盖 MuJoCo 仿真、手腕标定、示教采集、数据转换、ACT 策略训练及实机部署。

当前任务围绕物体抓取与放置（`pick_place_object_v1` / `pick_place_object_v2`）展开。本仓库保存迁移后的项目源码、机器人模型、项目专用运行库、现有测试数据、训练产物及运行记录。

## 工作流程

```mermaid
flowchart LR
    A[Quest 3 手柄示教] --> B[TB6-R5 与 OmniPicker]
    B --> C[三路图像与状态同步采集]
    C --> D[原始 PKL 数据]
    D --> E[LeRobot 数据集]
    E --> F[ACT 策略训练]
    F --> G[模型推理与实机执行]
```

## 主要功能

- **遥操作控制**：按 Quest 手柄相邻帧的位姿增量控制机械臂，丢弃未执行增量，避免历史动作积压。
- **手腕补偿**：v4 Pro 在 v4 基础上增加 wrist-pivot 标定补偿，配置保存在产品目录内。
- **MuJoCo 仿真**：提供 Quest 3 增量遥操作基线、手腕补偿版本、键盘控制、关节姿态查看和 IK 工具。
- **示教采集**：记录机械臂实际状态、已接受的控制指令、夹爪状态以及 front / top / wrist 三路图像。
- **数据转换**：将原始 PKL 转为 LeRobot 数据集，供 ACT 训练使用。
- **训练管理**：提供环境与数据预检查、短程 smoke 训练、正式训练及按任务/版本归档的配置。
- **部署链路**：包括离线评估、Shadow 辅助模块、ACT 推理进程和机械臂/夹爪执行进程。

## 目录结构

```text
TB6-teleop/
├── 00_environment/                 # LeRobot 环境导出清单，不含安装后的环境
├── 01_teleoperation/
│   ├── quest3_tb6_r5_omnipicker_data_product_v4/
│   ├── quest3_tb6_r5_omnipicker_data_product_v4_pro/
│   │   ├── calibration/           # 手腕标定
│   │   ├── collector/             # 控制、相机、夹爪与采集
│   │   ├── converter/             # PKL → LeRobot
│   │   ├── model/                 # URDF 与 meshes
│   │   ├── scripts/               # 采集和转换入口
│   │   └── data_teleop/           # 当前已有的测试数据
│   └── third_party/               # 项目专用依赖与运行库
├── 02_training/
│   ├── training_package/          # 检查、训练脚本与任务配置
│   └── training_results/          # 当前已有的训练记录和 checkpoint
├── 03_deployment/
│   ├── act_runtime/               # ACT 推理发送端
│   ├── deployment_support/        # 评估、Shadow、单次执行辅助代码
│   ├── product_v6/                # 实机执行端与运行记录
│   └── logs/
├── 04_simulation/tb6_r5_mujoco/
│   ├── quest3_wrist_pivot_teleop/  # 基础仿真、手腕补偿与标定
│   ├── scripts/                   # Quest、键盘控制及姿态查看工具
│   ├── ik_test/                   # C++ IK / FK 工具及现有构建产物
│   ├── urdf/                      # 仿真机器人模型
│   ├── meshes/                    # 模型网格
│   └── run_python.sh              # 自动设置 SDK 库路径的启动入口
├── 07_docs/environment/            # 遥操作环境导出清单
└── README-CN.md                    # 原有迁移说明，原样保留
```

## 硬件与环境

| 部分 | 配置 |
| --- | --- |
| 机械臂 | TB6-R5，当前配置 IP 为 `192.168.11.11` |
| XR 输入 | Meta Quest 3 右手柄、XRoboToolkit |
| 夹爪 | OmniPicker，USB-CAN |
| 相机 | 3 × RealSense D435：front、top、wrist |
| 遥操作环境 | 当前迁移说明使用 `teleop_act_xr`；旧文档中也称 `xrteleop` |
| 转换、训练、推理环境 | `lerobot`，项目脚本按 LeRobot 0.6.1 配置 |

环境清单：

- [LeRobot Conda 环境](00_environment/lerobot_environment.yml)
- [LeRobot 软件包清单](00_environment/lerobot_packages.txt)
- [遥操作 Conda 环境](07_docs/environment/xrteleop_environment.yml)
- [遥操作软件包清单](07_docs/environment/xrteleop_packages.txt)

安装后的 `lerobot/`、`lerobot_broken/` 环境目录不纳入备份。环境导出文件保留了原机器的信息，新机器需要按实际安装位置处理其中的路径，并配置 XRoboToolkit PC Service、Quest 客户端、设备驱动及网络。原有配置记录见 [README-CN.md](README-CN.md)。

## 使用入口

以下以仓库根目录为起点。先将 `TB6_ROOT` 设为自己的实际路径：

```bash
export TB6_ROOT=/path/to/TB6-teleop
```

### 1. 遥操作与采集

**该入口会连接真实设备并发送运动指令。** 使用前需按产品说明完成机械臂、夹爪、相机和手腕标定检查，并确保停止控制可用。

```bash
conda activate teleop_act_xr
cd "$TB6_ROOT/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro"
bash scripts/run_collect.sh
```

| 手柄操作 | 功能 |
| --- | --- |
| Grip | 遥操作启停（HOLD / ACTIVE） |
| Trigger | 夹爪开合 |
| A | HOME / HOME_SYNC |
| B | 开始录制，再按停止并保存 |
| 右摇杆按下 | 丢弃当前 episode |

当前 v4 Pro 的 `collector/config.py` 将数据写入 `data_teleop/pick_place_object_v2/test_raw_data/`，转换结果放入 `test_datasets/`。这与旧文档中的 `raw_data/`、`datasets/` 示例不同，以代码配置为准。完整操作流程见 [v4 Pro 说明](01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro/README.md)。

### 2. 转换为 LeRobot 数据集

```bash
conda activate lerobot
cd "$TB6_ROOT/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro"
bash scripts/run_convert.sh pick_place_object_dataset "pick and place the object"
```

转换脚本从采集配置读取输入路径，输出目录已存在时会拒绝覆盖。当前备份已有同名测试数据集；转换新数据时需使用新的数据集名称。

数据接口：

```text
observation.state = [6 维关节实际位置, 夹爪实际位置]
action            = [6 维已接受关节指令, 夹爪指令]
observation.images.front / top / wrist
采样频率：50 Hz
```

状态和动作均为 7 维；夹爪值沿用 0～100 的表示，0 为闭合，100 为张开。

### 3. ACT 训练

先准备目标数据集，核对环境和与该数据集匹配的 SHA256 清单。仓库提供 `run_preflight.sh` 和 `run_preflight_v2.sh`；已有清单不能直接视为任意新数据集的校验清单。

```bash
conda activate lerobot
cd "$TB6_ROOT/02_training"
export TB6_DATASET=/absolute/path/to/your/lerobot_dataset
export TB6_CONFIG="$TB6_ROOT/02_training/training_package/config/pick_place_object_v2/act/v1.sh"

# 先进行 20 步短程训练，确认数据加载和训练链路
bash training_package/01_train/run_smoke_train.sh "$TB6_DATASET" "$TB6_CONFIG"

# 短程训练通过后，再单独执行正式训练
bash training_package/01_train/run_train.sh "$TB6_DATASET" "$TB6_CONFIG"
```

这里显式传入 v2 配置，因为当前正式训练脚本默认使用 v1 任务配置，而 smoke 脚本默认使用 v2。

v2 当前配置：120000 步、batch size 8、ResNet18、动作块长度 100、学习率 `1e-5`；默认关闭 Hub 上传和 W&B。结果保存到 `02_training/training_results/<任务>/act/<版本>/<smoke或train>/<时间戳>/`。

### 4. 实机部署

部署由两个进程配合：

| 进程 | 入口 |
| --- | --- |
| ACT 推理发送端 | `03_deployment/act_runtime/run_act_real_full_sender_v2.sh` |
| 机械臂与夹爪执行端 | `03_deployment/product_v6/scripts/run_act_control_full.sh` |

启动前需核对以下当前配置：

- 两个启动脚本分别固定使用 `/home/robot/.conda/envs/lerobot/bin/python` 和 `/home/robot/.conda/envs/teleop_act_xr/bin/python`。
- v2 推理代码引用项目目录外的 `20260901_172324/checkpoints/120000/pretrained_model`；**本次备份不包含该外部模型**。
- 两个启动脚本实际传入的 `--act-speed-scale` 都是 `0.3`；部分提示文本和 Python 默认值仍是 `0.03`。
- 部分日志、评估和辅助模块保留迁移前的绝对路径，需要在实际部署副本中核对。

这些入口会执行真实运动。备份过程未启动机器人、采集、训练或部署程序，也未修改上述配置。

### 5. MuJoCo 仿真与手腕标定

仿真使用 `teleop_act_xr` 环境。新增环境记录列出的版本为 MuJoCo `3.13.0`、glfw `2.10.2`、pynput `1.8.2`；完整依赖和验证范围见 [仿真环境记录](04_simulation/tb6_r5_mujoco/README_environment.md)。

基础 Quest 3 增量遥操作仿真：

```bash
conda activate teleop_act_xr
cd "$TB6_ROOT/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/01_mujoco_incremental_baseline"
./run_mujoco.sh
```

手腕补偿版本在 `02_mujoco_wrist_pivot` 目录中。需要重新标定时先运行 `./run_calibration.sh`，完成后使用 `./run_mujoco.sh` 启动仿真。有效标定写入该目录的 `calibration/current_user.json`；补偿启用时，缺失或无效的标定会阻止启动。

其他脚本统一从仿真根目录启动，例如：

```bash
cd "$TB6_ROOT/04_simulation/tb6_r5_mujoco"
./run_python.sh scripts/keyboard_tcp_control.py
# 或查看指定关节角对应的姿态（单位：度）
./run_python.sh scripts/view_pose.py 0 20 -30 0 10 0
```

统一入口为当前进程添加 `01_teleoperation/third_party` 的 SDK 动态库路径。仿真产生的有效手腕标定可按 [仿真使用说明](04_simulation/tb6_r5_mujoco/README-CN.md) 应用到实机 v4 Pro；本次备份不会自动替换实机标定。

现有环境记录报告基础版 7 项、补偿版 14 项离线测试通过，但未验证该次迁移后的窗口交互和 Quest 端到端运行。这是项目已有记录，本次同步没有重新执行仿真测试。`ik_test/build` 保留旧机器的 CMake 缓存，重新编译时需要重新生成构建目录。

## 当前备份范围与状态

- 保留原项目文件内容，包括源码、原有文档、标定、URDF、项目运行库、测试数据、日志、视频及已有训练结果。
- 排除 `00_environment` 下安装后的环境目录，保留环境导出清单。
- 本目录内的训练 checkpoint 是 `20260915_092721` 的 **20 步 smoke 训练产物**，不能等同于部署脚本引用的 120000 步正式模型。
- 保留项目现有验证记录；本次只做备份完整性核对，不代表迁移后的实机运行验证。
- 视频、模型等大文件使用 Git LFS 保存。下载完整文件需要安装 Git LFS，并在克隆后执行 `git lfs pull`。
- 新增的主页说明与 Git LFS 配置在备份副本中生成，原始工作目录保持不动。

## 进一步阅读

- [原始迁移与环境说明](README-CN.md)
- [v4 遥操作说明](01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4/README.md)
- [v4 Pro 遥操作与手腕补偿说明](01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro/README.md)
- [训练流程说明](02_training/training_package/README_CN.md)
- [训练配置与预检查说明](02_training/training_package/COMPANY_TRAINING_README.md)
- [原有部署说明](03_deployment/README-CN.md)

旧文档作为历史记录原样保留；其中的旧目录、环境名称和默认参数需结合当前代码阅读。
