"""Standalone product v4 configuration for strict incremental data collection."""

from pathlib import Path

PRODUCT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_IP = "192.168.11.11"
CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ
TOPIC_CONTROLLER_PERIOD_S = 0.002
GRIP_THRESHOLD = 0.9
XR_STALE_S = 0.5
JOINT_SPEED_LIMIT_RAD_S = 0.30
MAX_ALLOWED_JOINT_SPEED_RAD_S = 0.31
PAYLOAD_ACCELERATION_SCALE = 4.0
RELEASE_DECEL_RAD_S2 = 0.8
TRANSLATION_SCALE = 0.6
ROTATION_SCALE = 0.15
TOPIC_STALE_MS = 100.0
ROBOT_STATIONARY_LIMIT_RAD_S = 0.0005
BRAKE_SETTLE_VELOCITY_RAD_S = 0.0005
BRAKE_SETTLE_UNIQUE_SAMPLES = 3
BRAKE_TOTAL_TIMEOUT_S = 3.0
BRAKE_DECEL_REFERENCE_RAD_S2 = 0.0447520321024188
BRAKE_SEGMENT_TIME_LIMIT_S = 0.05

# Product-local correct robot model.
URDF_PATH = PRODUCT_ROOT / "model" / "tb6_r5_correct.urdf"
MESH_DIRECTORY = PRODUCT_ROOT / "model" / "meshes"

# ============================================================
# HOME button motion (only used inside the existing control loop)
# ============================================================
HOME_JOINT_DEG = (
    90.0,
    -120.0,
    110.0,
    -110.0,
    -90.0,
    0.0,
)
HOME_SPEED_RAD_S = 0.12
HOME_RAMP_DURATION_S = 0.60
HOME_MAX_DELTA_DEG = 20.0
HOME_FINAL_TOLERANCE_DEG = 0.60
HOME_SETTLE_HOLD_S = 1.0
HOME_SETTLE_TIMEOUT_S = 3.0

# HOME-only command/actual baseline reconciliation.
HOME_SYNC_SPEED_RAD_S = 0.06
HOME_SYNC_BYPASS_GAP_DEG = 0.2
HOME_SYNC_MAX_INITIAL_GAP_DEG = 5.0
HOME_SYNC_READY_GAP_DEG = 0.6
HOME_SYNC_MAX_PASSES = 2

# ============================================================
# OmniPicker
# ============================================================
GRIPPER_VEL = 60.0
GRIPPER_FORCE = 70.0
GRIPPER_ACC = 40.0
GRIPPER_DEC = 40.0

# ============================================================
# RealSense episode recording
# ============================================================
CAM_SERIAL_DICT = {
    "front": "261322073567",
    "top": "261322076396",
    "wrist": "254622071455",
}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60
CAMERA_JPG_QUALITY = 85
DATA_LOG_HZ = 50.0

# ============================================================
# Non-critical latest-frame camera preview
# ============================================================
CAMERA_PREVIEW_ENABLED = True
CAMERA_PREVIEW_HZ = 20.0
CAMERA_PREVIEW_WIDTH = 426
CAMERA_PREVIEW_HEIGHT = 320
CAMERA_PREVIEW_WINDOW_NAME = "TB6-R5 Camera Preview"

# ============================================================
# Training data storage
# Change this path for each collection/task.
# ============================================================
DATA_COLLECTION_DIR = PRODUCT_ROOT / "data_teleop" / "pick_place_object_v2"
RAW_DATA_DIR = DATA_COLLECTION_DIR / "raw_data"
LEROBOT_DATASETS_DIR = DATA_COLLECTION_DIR / "datasets"
LOG_DIRECTORY = PRODUCT_ROOT / "logs"

if not 0.0 < JOINT_SPEED_LIMIT_RAD_S <= MAX_ALLOWED_JOINT_SPEED_RAD_S:
    raise ValueError("JOINT_SPEED_LIMIT_RAD_S must be in (0, MAX_ALLOWED_JOINT_SPEED_RAD_S]")
