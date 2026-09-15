这个版本是用来部署pick and place v2采集数据的新模型
目前部署的是120k的模型

启动方法：

A终端启动ACT：

conda activate lerobot
cd /home/zzw/projects/TB6_R5_ACT/training_package/06_real_robot_full
bash run_act_real_full_sender_v2.sh


B终端启动xrteleop：

conda activate xrteleop
cd /home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v6
bash scripts/run_act_control_full.sh
