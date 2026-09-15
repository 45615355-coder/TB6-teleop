# TB6-R5 + Quest3 + OmniPicker + 3×D435
# Strict Incremental Teleoperation & Imitation Learning Data Product v4 Pro

## A. Product v4 是什么

这是当前完整、独立维护的 TB6-R5 模仿学习数据采集产品。控制核心从旧版 Grip reference +
persistent TCP target 升级为：

- Quest unique frame-to-frame ΔSE(3)
- strict no-backlog，未执行增量永久 DROP
- 连续 absolute accepted q_command
- PlaCo success-only propose/commit，PlaCo 是 ACTIVE 唯一 runtime joint-speed limiter
- Grip release 使用固定最后 accepted command 的 controller-side deceleration
- normal release 没有 host BRAKE
- mid-decel re-Grip 可立即抢占，第一张 fresh XR frame 只重新 anchor

产品同时集成 OmniPicker、A HOME/HOME_SYNC、B episode recording、right-stick discard、
3×D435 单 owner 采集、20 Hz preview 和 50 Hz PKL。

Product v4 包含项目专用 Python、native runtime、URDF 和 meshes，运行时不依赖任何历史开发目录。

Product v4 Pro 只在成熟 v4 控制链路前增加 Quest3 wrist-pivot 位置补偿：

~~~text
p_corrected = p_raw + R_raw @ offset_m
R_corrected = R_raw
~~~

补偿使用 Product-local `calibration/current_user.json`，当前 gain 为 1.0。启动时会先校验
profile 的 `valid`、三维 finite offset 和 0.20 m norm 上限；启用补偿时 profile 缺失或非法会在
任何 Topic/RPC/XR/CAN/相机初始化之前报错退出，不会静默退回 raw Quest。raw pose validation 与
freshness 判定仍先于补偿，补偿后的 pose 仅进入原 frame-to-frame ΔSE(3) 控制链路。训练数据
schema 与 q_command action 语义不变。

## B. 目录结构

~~~text
quest3_tb6_r5_omnipicker_data_product_v4_pro/
├── calibration/     # Product-local validated wrist-pivot profile
├── collector/       # 控制、Topic/RPC、夹爪、相机、recording
│   └── vendor/      # product-local RPC/Topic/OmniPicker native runtime
├── converter/       # 冻结的 PKL → LeRobot converter
├── model/           # tb6_r5_correct.urdf 与 meshes
├── scripts/         # 正式采集与转换入口
├── tests/           # offline semantic/packaging tests
├── logs/            # product-v4 JSONL diagnostics
├── data_teleop/
│   └── pick_place_object_v1/
│       ├── raw_data/
│       └── datasets/
├── MANIFEST.sha256
└── README.md
~~~

## C. 环境

采集环境：

~~~bash
conda activate xrteleop
~~~

转换环境：

~~~bash
conda activate lerobot
~~~

Product 打包项目专用源码/native/model，但不打包完整 conda 环境。目标电脑仍需预装 Quest3 SDK、
NumPy、PlaCo/Pinocchio、OpenCV、pyrealsense2 和相应系统 USB/网络支持。

## D. 硬件配置

- Robot：TB6-R5，IP 192.168.11.11
- XR：Meta Quest 3 右 Touch Plus controller
- Gripper：OmniPicker USB-CAN
- D435 front：261322073567
- D435 top：261322076396
- D435 wrist：254622071455
- Camera：640×480、60 FPS、JPEG quality 85
- Logger：50 Hz
- Preview：20 Hz、426×320

## E. 正式启动

> 以下命令会连接真实机器人、Quest、OmniPicker CAN 和 3×D435，并发送 JogAny；机器人会真实运动。

~~~bash
cd /home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v4_pro
conda activate xrteleop
./scripts/run_collect.sh
~~~

run_collect.sh 自动使用 product-local collector、Topic、RPC 和 native library，并自动带上：

~~~text
--execute-motion
--confirm-controller-ready
--confirm-robot-enabled
--confirm-robot-stationary
--record-data
~~~

采集时禁止同时打开 realsense-viewer，否则可能抢占 D435。

## F. 手柄映射

- Grip：机械臂 HOLD/ACTIVE clutch
- Trigger：OmniPicker absolute target，target_pos = 100 × (1-trigger)
- A：HOME gate / HOME_SYNC / HOME
- B：第一次 rising START，下一次 rising STOP+SAVE
- right-stick-click：DISCARD 当前 episode

OmniPicker 定义：0 = fully closed，100 = fully open；控制为 latest-target-wins。

## G. 正式 episode SOP

~~~text
启动 Product
→ A HOME
→ HOME PASS
→ B START
→ Grip/Trigger 执行完整任务
→ 手动回到 HOME 附近
→ 松 Grip
→ 等机器人真正静止
→ A HOME
→ HOME_SYNC
→ HOME PASS
→ B STOP+SAVE
~~~

HOME PASS/FAILED 都不会自动结束 program 或 recording；episode 只由 B STOP+SAVE 或
right-stick DISCARD 控制。

## H. HOME / HOME_SYNC

固定 HOME：

~~~text
[90, -90, 90, -90, -90, 0] deg
~~~

- HOME_SYNC_SPEED_RAD_S = 0.02
- HOME_SPEED_RAD_S = 0.05
- HOME_SYNC bypass gap：0.2°
- HOME_SYNC max initial gap：5.0°
- HOME_SYNC ready gap：0.6°
- HOME_SYNC max passes：2（第一次 + 最多一次自动retry）

流程：

~~~text
HOME gate
→ HOME_SYNC
→ HOME_SYNC_DECEL / target+actual settle
→ HOME MOTION
→ HOME-specific host BRAKE
→ HOME SETTLE
→ PASS / FAILED
~~~

HOME_SYNC 是 continuous q_command 架构的 HOME-only baseline reconciliation。它把 q_command
低速同步到 A rising 瞬间一次性抓取的 stationary q_actual snapshot，不追逐实时 q_actual，也不是固定 offset
补偿。它不影响 normal teleop。

HOME_SYNC 使用最多2-pass的bounded reconciliation。第1 pass在SYNC_DECEL后重新确认q_target和q_actual
均已静止；若残差仍大于0.6°，才从当前最后accepted q_command出发，并重新抓取此时最新stationary
q_actual作为第2 pass的fixed snapshot target。第2 pass过程中同样不追踪实时q_actual，且绝不创建第3 pass。

原 HOME 末端仍保留成熟的 segmented host BRAKE；normal Grip release 仍只使用 controller-side deceleration。

## I. q 定义

- q_command：本程序最后成功提交的连续 absolute command
- q_target：机器人 controller Topic target
- q_actual：机器人真实反馈位置

训练 action 始终使用 transport.last_accepted_command_q。HOME_SYNC 期间真实 accepted commands
也会原样进入 episode。

## J. PKL 数据 schema

每个 timestep：

~~~text
timestamp
step_index
q_actual
q_command
gripper_actual_pos
gripper_command_pos
image
topic_global_count
~~~

训练字段：

~~~text
observation.state = [q_actual(6), gripper_actual]       shape=(7,)
action            = [q_command(6), gripper_command]    shape=(7,)
observation.images.front
observation.images.top
observation.images.wrist
fps = 50
~~~

Gripper 不归一化为 0～1；raw PKL 是 source of truth。

## K. 数据路径

~~~text
data_teleop/pick_place_object_v1/raw_data/
data_teleop/pick_place_object_v1/datasets/
logs/
~~~

切换正式任务时，只修改 collector/config.py 中 DATA_COLLECTION_DIR 的 task 子目录。
所有路径都从当前 PRODUCT_ROOT 计算，不需要修改绝对工程路径。

## L. 数据转换

~~~bash
cd /home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v4
conda activate lerobot
./scripts/run_convert.sh pick_place_object_dataset "pick and place the object"
~~~

输入：

~~~text
data_teleop/pick_place_object_v1/raw_data/*.pkl
~~~

输出：

~~~text
data_teleop/pick_place_object_v1/datasets/pick_place_object_dataset/
~~~

输出目录已存在时脚本拒绝覆盖。转换不会删除 raw PKL，不会边采边转，也不会自动上传 Hub。

## M. Safety

- 启动前确认 controller ready、robot enabled、robot stationary。
- Web/robot-side STOP 优先；已观察到 robot-side STOP 时不重复发 Stop。
- HOME gate 要求 HOLD、Grip low、Topic safe、机器人 stationary、距离 HOME 每轴不超过 15°。
- HOME_SYNC 初始 command/actual gap 大于 5°时非 fatal 拒绝，不发送 SYNC。
- HOME position-only FAILED 不 Stop、不退出，保持 HOLD。
- HOME serious local control fault 在 RPC 可用且 robot-side 未 STOP 时，best-effort 使用同一 client 发送
  exact synchronous {Stop}，timeout 10000 ms，且只发送一次。
- Preview failure 不得触发机器人 fault。
- 禁止同时运行 realsense-viewer。

## N. Development provenance / History

- Product packaging 来源：
  /home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v3
- 控制来源：
  /home/zzw/projects/Teleoperation_TB6R5_Quest3/teleop_regrip/incremental_no_backlog/04_feature_integration/v2
- 创建/封装完成时间：2026-08-25T17:42:29+08:00

开发来源此前已经实机验证：

- strict incremental teleop / strict no-backlog
- controller-side release 与 mid-decel re-Grip
- HOME_SYNC 与 HOME PASS
- 3×D435 recording
- 两个连续 PKL episode
- 50 Hz recording

上述验证属于控制开发来源。Product v4 完成封装后，尚未在 v4 正式目录执行新的 real-hardware smoke test。

## O. 当前状态

~~~text
quest3_tb6_r5_omnipicker_data_product_v4
PACKAGED
OFFLINE TESTED
READY FOR PRODUCT V4 REAL-HARDWARE SMOKE TEST
~~~

当前不得标记为 PRODUCT V4 REAL HARDWARE PASS。


### 数据转换代码：
conda activate lerobot
cd /home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v4_pro
conda activate lerobot
./scripts/run_convert.sh pick_place_object_dataset "pick and place the object"
- *pick_place_object_dataset* 这个是指存放在datasets里面文件夹的名字
- *pick and place the object*这个是指对这个任务的说明，不影响数据存储之类的
存放的位置在config里面改

### 机器人运行代码：
conda activate teleop_act_xr
cd /home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro
./scripts/run_collect.sh
