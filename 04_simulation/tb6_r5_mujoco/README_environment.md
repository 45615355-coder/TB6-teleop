# 新电脑仿真环境记录

记录日期：2026-09-15。

## 环境位置

- Conda 环境：`teleop_act_xr`（旧电脑为 `xrteleop`）。
- 环境目录：`/home/robot/.conda/envs/teleop_act_xr`。
- Python：3.10.21。
- 项目目录：`/home/robot/teleop_act/04_simulation/tb6_r5_mujoco`。

```bash
conda activate teleop_act_xr
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco
python -c "import sys; print(sys.executable)"
```

Python 路径应为 `/home/robot/.conda/envs/teleop_act_xr/bin/python`。
源码中模型及 IK 程序路径根据 `__file__` 定位项目，不依赖 Conda 环境目录或启动时的工作目录。

## 整个项目的统一启动入口

激活 `teleop_act_xr` 后，用项目根目录的 `run_python.sh` 代替直接调用 `python`。
它为本次 Python 进程添加 SDK 动态库搜索目录，保留原有 `LD_LIBRARY_PATH`，不改变调用终端的环境和工作目录。
SDK 目录按项目位置定位到 `teleop_act/01_teleoperation/third_party`，只读取，不复制或修改动态库。

在项目根目录启动任意脚本，例如：

```bash
./run_python.sh scripts/quest3_tb6_6d_mujoco.py
./run_python.sh scripts/view_pose.py 0 20 -30 0 10 0
./run_python.sh -c 'import xrobotoolkit_sdk; print("SDK import OK")'
```

在 01 或 02 子目录内，仿真继续使用 `./run_mujoco.sh`。
在 02 子目录内，手腕标定使用 `./run_calibration.sh`。
这些入口统一调用根目录的 `run_python.sh`。
其他 Python 命令也可以在这两个子目录内使用 `../../run_python.sh`，例如：

```bash
../../run_python.sh -m unittest discover -s tests -v
```

直接执行 `python xxx.py` 不会自动获得这里设置的动态库路径。
本配置不写入 `.bashrc`、Conda、系统库或实机工程，也不修改 Python 控制源码及标定结果。

## 本次安装

| 包 | 安装版本 | 用途 |
|---|---|---|
| mujoco | 3.13.0 | 仿真与模型加载 |
| glfw | 2.10.2 | 图形窗口 |
| pynput | 1.8.2 | 部分键盘控制脚本 |

实际执行的命令：

```bash
/home/robot/.conda/envs/teleop_act_xr/bin/python -m pip install mujoco glfw pynput
```

如需复现这三个包的版本，在目标环境激活后执行：

```bash
python -m pip install mujoco==3.13.0 glfw==2.10.2 pynput==1.8.2
```

这不是完整环境锁定文件。安装也添加了所需的传递依赖，原有 NumPy 2.2.6 未被升级。

## 验证结果

- `mujoco`、`mujoco.viewer`、`glfw` 导入成功。
- `pynput.keyboard` 在可连接桌面的进程中导入成功；最初沙箱检查因无法访问 X 显示服务失败。
- `urdf/tb6_r5_mujoco.urdf` 加载和 `mj_forward` 成功：6 个自由度、`joint1` 至 `joint6`、7 个网格，末端 `ee_Link` 可查询。
- MuJoCo 默认导入时会将固定基座合并到 world，不应以必须保留独立 `base_link` body 作为加载成功条件。
- 01 离线测试：7 项通过；02 离线测试：14 项通过，包含 PlaCo 初始化、正运动学和腕部补偿测试。
- 本次没有打开交互式仿真窗口，没有初始化 Quest 或连接实机；窗口交互和手柄端到端运行尚未验证。
- URDF、网格及控制算法未因本次安装而修改。

在项目根目录复查离线测试：

```bash
python -B -m unittest discover -s quest3_wrist_pivot_teleop/01_mujoco_incremental_baseline/tests -v
python -B -m unittest discover -s quest3_wrist_pivot_teleop/02_mujoco_wrist_pivot/tests -v
```

## XR SDK 与其他环境事项

用户已确认手柄实机遥操作可用。本次没有重装或修改 `xrobotoolkit_sdk`、ROS 或相关动态库配置。
先前直接调用环境 Python 导入 SDK 时曾报 `libPXREARobotSDK.so` 无法找到；这只能说明该检查进程的动态库查找环境不足，不能据此认定实机启动流程不可用。继续沿用已验证的实机启动配置。

`python -m pip check` 另报告：

- `jinja2 3.1.6` 缺少 `markupsafe`。
- `ultralytics-thop 2.0.20` 缺少 `torch`。

本次未处理这两项；上述仿真离线测试仍全部通过。

`ik_test/build` 仍包含旧电脑的 CMake 路径缓存，本次 pip 安装不会更新 C++ 构建缓存；需要重新编译时应重新生成构建目录。
