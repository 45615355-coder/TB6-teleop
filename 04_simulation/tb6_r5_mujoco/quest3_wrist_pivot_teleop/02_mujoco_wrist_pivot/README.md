# 02 MuJoCo wrist-pivot

02 is a copy of the verified 01 MuJoCo incremental baseline plus one control-input change: before the unchanged frame-to-frame delta computation, raw Quest controller position is corrected as `p + gain * R @ offset`. Controller rotation is passed through unchanged. Pinocchio/PlaCo propose/commit, Grip/re-Grip, XR frame consumption, strict no-backlog, URDF, scales, initial joints, and joint-speed limit are copied from 01 unchanged.

`offset` is obtained only from the Quest-only calibration program; it is not hard-coded. `gain` defaults to `1.0`; use `0.0` only for an explicit no-correction experiment, never as an automatic fallback. When `WRIST_PIVOT_ENABLED=True`, teleop refuses to start unless `calibration/current_user.json` exists, has `valid: true`, and contains a finite in-bound offset.

## Calibrate first

```bash
conda activate teleop_act_xr
cd /home/robot/teleop_act/04_simulation/tb6_r5_mujoco/quest3_wrist_pivot_teleop/02_mujoco_wrist_pivot
./run_calibration.sh
```

The program opens neither MuJoCo nor any robot connection. First release Grip once; then capture distinct YAW, PITCH, and ROLL segments. Each capture begins on Grip press and ends on release. All three segments jointly fit one offset with mean-centred least squares. It saves a timestamped diagnostic JSON every time, but overwrites `calibration/current_user.json` only for a valid solution.

## Run corrected teleoperation

```bash
./run_mujoco.sh
```

This opens Quest XR and a local MuJoCo viewer only. The JSONL keeps `xr_position` and `xr_rotation` as raw Quest data, and adds `xr_corrected_position`, `wrist_pivot_offset_m`, `wrist_pivot_gain`, and `xr_raw_to_corrected_delta_m`.

## Analyze a corrected trial

```bash
python analyze_wrist_rotation.py logs/mujoco_incremental_YYYYMMDD_HHMMSS.jsonl --mode yaw
```

The mode is a manual label only. Per-session CSV and JSON output in `analysis/` includes raw Quest, wrist-pivot-corrected Quest, MuJoCo TCP, and SO(3) relative-rotation metrics. No session paths are joined.

## Test

```bash
python -m unittest discover -s tests -v
```
