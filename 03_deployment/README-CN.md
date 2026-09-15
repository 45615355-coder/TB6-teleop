1、启动前需修改训练模型路径：
/home/robot/teleop_act/03_deployment/act_runtime/act_real_full_sender_v2.py里面的

#模型路径
###  CHECKPOINT_PATH = Path(
    "/home/robot/data_collect/training_results/pick_place_object_v2/act/v1/train/"
    "20260901_172324/checkpoints/120000/pretrained_model"
)
#输出日志路径
###   DEFAULT_OUTPUT_ROOT = Path(
    "/home/zzw/projects/TB6_R5_ACT/training_results/pick_place_object_v2/act/v1/"
    "real_robot/full_sender"
)

#任务名称
TASK_NAME = "pick_place_object_v2"

#以及各项训练参数：
TEMPORAL_ENSEMBLE_COEFF = 0.01
CHUNK_SIZE = 100
ACTION_DIM = 7
MODEL_HZ = 50.0
DEFAULT_SPEED_SCALE = 0.03
MAX_PLANNING_LEAD_S = 0.20
PLANNING_LEAD_STEP_FRACTION = 0.8

2、以及/home/robot/teleop_act/03_deployment/deployment_support/03_shadow/ipc/act_shadow_subscriber.py里面的：
CHECKPOINT = Path(
    "/home/robot/data_collect/training_results/pick_place_object_v2/act/v1/train/"
    "20260901_172324/checkpoints/120000/pretrained_model"
)

3、启动：
启动机械臂：
conda activate teleop_act_xr
cd /home/robot/teleop_act/03_deployment/quest3_tb6_r5_omnipicker_data_product_v6
bash scripts/run_act_control_full.sh

启动ACT
conda activate lerobot
cd /home/robot/teleop_act/03_deployment/act_runtime
bash run_act_real_full_sender_v2.sh
