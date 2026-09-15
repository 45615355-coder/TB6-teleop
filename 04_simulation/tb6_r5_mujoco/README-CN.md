### Mujoco仿真启动：
先激活环境：
```bash
conda activate teleop_act_xr
```

### 1. 基础 Quest3 遥操作仿真（01）

```bash
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/01_mujoco_incremental_baseline
./run_mujoco.sh
```

### 2. 手腕补偿遥操作仿真（02）

```bash
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/02_mujoco_wrist_pivot
./run_mujoco.sh
```

重新标定手腕时，在同一目录执行：

```bash
./run_calibration.sh
```

### 3. 其他仿真脚本

先进入项目根目录：

```bash
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco
```

然后按需**选择一条**运行：

```bash
# Quest3 六维遥操作
./run_python.sh scripts/quest3_tb6_6d_mujoco.py

# Quest3 六维遥操作诊断版本
./run_python.sh scripts/quest3_tb6_6d_mujoco_diag_guard.py

# Quest3 位姿追踪版本
./run_python.sh scripts/quest3_tb6_6d_pose_trace.py

# 键盘 TCP 控制
./run_python.sh scripts/keyboard_tcp_control.py

# 键盘六维连续控制
./run_python.sh scripts/keyboard_6d_continuous.py

# 关节姿态选择工具
./run_python.sh scripts/joint_pose_picker.py

# 查看指定关节角对应的姿态（单位：度）
./run_python.sh scripts/view_pose.py 0 20 -30 0 10 0
```

**记住：01、02 用 `./run_mujoco.sh`；其他脚本在根目录用 `./run_python.sh 脚本路径`。** 这些入口会自动添加 SDK 动态库路径；其他脚本的完整运行兼容性尚未逐个验证。




###完成手腕标定后会生成一个/home/robot/teleop_act/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/02_mujoco_wrist_pivot/calibration/current_user.json的json文件，把这个文件复制到/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro/calibration去替换同名文件即可完成标定后的数值补偿
