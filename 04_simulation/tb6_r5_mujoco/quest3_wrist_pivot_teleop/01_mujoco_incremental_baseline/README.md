# 01 MuJoCo incremental baseline

This standalone phase reproduces the verified strict-no-backlog Quest 3 arm-control semantics in local MuJoCo before any wrist-pivot work. It initializes Quest/XRoboToolkit and a local MuJoCo viewer only. It never connects to TB6-R5, creates RPC, subscribes to Topics, sends JogAnyJ3/Stop/CAN, or connects OmniPicker or RealSense.

The MuJoCo/viewer and XR lifecycle are adapted from `scripts/quest3_tb6_6d_mujoco.py` (read only). The incremental control kernel and PlaCo propose/commit transaction are migrated from `Teleoperation_TB6R5_Quest3/teleop_regrip/incremental_no_backlog/04_feature_integration/v2/{teleop.py,placo_ik.py,config.py}` (read only).

The planner deliberately loads `/home/robot/teleop_act/04_simulation/tb6_r5_mujoco/urdf/tb6_r5_mujoco.urdf` and its own relative simulation meshes. It does not use the real-robot `tb6_r5_correct.urdf`. Both versions use Pinocchio/PlaCo, the same frame-to-frame XR delta mapping, scales, 50 Hz control rate, and PlaCo velocity limit; they differ in robot model and execution boundary.

On every fresh active frame, `MujocoTeleopSession.consume()` computes only the current XR delta and advances the previous-XR baseline before PlaCo runs. `IncrementalMujocoPlacoPlanner.propose()` creates one temporary TCP candidate from `q_commit`; `commit()` replaces `q_commit` only for a valid candidate. A failed candidate is rolled back and its XR intent is not replayed. There is no persistent Cartesian target, residual queue, target interpolation, or backlog recovery.

Grip must first be observed released after program startup, matching the strict-v2 safety gate. Grip release then clears the XR anchor and holds `q_commit`. The first fresh re-Grip frame creates a new anchor with zero delta, so the robot does not jump; the second fresh frame can move it. MuJoCo qpos is a direct visualization of accepted `q_commit`, not an emulation of JogAnyJ3 controller dynamics.

## Start

```bash
conda activate teleop_act_xr
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/01_mujoco_incremental_baseline
./run_mujoco.sh
```

Detailed JSONL records are written under `logs/`; terminal output is transition/error/status oriented rather than frame-by-frame.

## Test

```bash
conda activate teleop_act_xr
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/01_mujoco_incremental_baseline
python -m unittest discover -s tests -v
```

This phase contains no wrist-pivot calibration or control code.

## Offline wrist-rotation analysis

The JSONL control log is unchanged. Analyze only its independent ACTIVE sessions with a manual trial label; `yaw`, `pitch`, and `roll` do not alter, filter, or rotate source data.

```bash
python analyze_wrist_rotation.py logs/mujoco_incremental_YYYYMMDD_HHMMSS.jsonl --mode yaw
```

The script opens no Quest, MuJoCo viewer, or robot connection. It writes frame CSV and per-session/overall JSON summary to `analysis/`.
