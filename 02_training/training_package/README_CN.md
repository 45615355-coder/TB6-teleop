## 训练步骤：
1、启动环境：
conda activate lerobot

2、进入训练工作空间：
cd /home/robot/teleop_act/02_training

3、转换后数据的初Smoke训练：
bash training_package/01_train/run_smoke_train.sh \
/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4/data_teleop/pick_place_object_v2/datasets/pick_place_object_dataset
#第一行是训练指令的sh文件，训练源代码在lerobot环境里
#第二三行是你的转换后代码的位置

4、正式训练：
bash training_package/01_train/run_train.sh \
/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4/data_teleop/pick_place_object_v2/datasets/pick_place_object_dataset

5、训练参数在
/home/robot/teleop_act/02_training/training_package/config进行修改
通过train.sh文件（smoke文件同理）里面的：
CONFIG_FILE="${2:-$PACKAGE_ROOT/config/pick_place_object_v1/act/v1.sh}"
进行索引
