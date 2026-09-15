"""Parameters for the standalone Quest 3 -> MuJoCo incremental baseline."""

from pathlib import Path

import numpy as np

PHASE_ROOT = Path(__file__).resolve().parent

# This is deliberately the simulation URDF, not the real-robot URDF.
PROJECT_ROOT = PHASE_ROOT.parents[1]
MUJOCO_URDF_PATH = PROJECT_ROOT / "urdf" / "tb6_r5_mujoco.urdf"

# Migrated from 04_feature_integration/v2/config.py.
CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ
GRIP_THRESHOLD = 0.9
TRANSLATION_SCALE = 0.8
ROTATION_SCALE = 0.3
JOINT_SPEED_LIMIT_RAD_S = 0.4
MAX_ALLOWED_JOINT_SPEED_RAD_S = 0.45

INITIAL_JOINTS_DEG = np.array((90.0, -120.0, 110.0, -110.0, -90.0, 0.0), dtype=float)

R_XR_TO_ROBOT = np.array(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=float
)

# XR safety and diagnostic settings retained from the old MuJoCo reference.
XR_STALE_S = 0.5
XR_POSE_JUMP_POSITION_LIMIT_M = 0.20
XR_POSE_JUMP_ROTATION_LIMIT_DEG = 45.0
QUATERNION_NORM_MIN = 1e-6
QUATERNION_NORM_MAX = 10.0
STATUS_INTERVAL_S = 1.0
WARNING_INTERVAL_S = 1.0
INITIAL_WAIT_PRINT_INTERVAL_S = 1.0

LOG_DIRECTORY = PHASE_ROOT / "logs"
LOG_FLUSH_EVERY_FRAME = True

# 02-only wrist-pivot calibration and correction.  The values above remain the
# copied, user-verified 01 baseline parameters.
WRIST_PIVOT_ENABLED = True
WRIST_PIVOT_PROFILE = PHASE_ROOT / "calibration" / "current_user.json"
WRIST_PIVOT_CORRECTION_GAIN = 1.0
WRIST_PIVOT_MIN_SAMPLES_PER_DIRECTION = 30
WRIST_PIVOT_MAX_OFFSET_M = 0.20
WRIST_PIVOT_MAX_RMS_ERROR_M = 0.025

if not 0.0 < JOINT_SPEED_LIMIT_RAD_S <= MAX_ALLOWED_JOINT_SPEED_RAD_S:
    raise ValueError("JOINT_SPEED_LIMIT_RAD_S must be in (0, MAX_ALLOWED_JOINT_SPEED_RAD_S]")
