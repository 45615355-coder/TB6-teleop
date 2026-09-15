# zzw_teleop 项目结构说明

1、遥操作XR项目配置：https://github.com/XR-Robotics（均已配置，新电脑需要重新配置）
XRoboToolkit-PC-Service：Ubuntu电脑端服务            已安装：/opt/apps/roboticsservice/runService.sh
XRoboToolkit-PC-Service-Pybind：Python绑定
XRoboToolkit-Teleop-Sample-Python：Python遥操作示例
XRoboToolkit-Unity-Client-Quest：Quest端客户端

2、环境配置：（均已配置，新电脑需要重新配置）
环境名称:
teleop_act_xr

环境类型:
Conda environment

环境位置:
/home/robot/.conda/envs/teleop_act_xr

Python版本:
Python 3.10.21

3、XRoboToolkit环境
Python SDK
文件：
xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so
位置：
/home/robot/.conda/envs/teleop_act_xr/lib/python3.10/site-packages/

依赖：
libPXREARobotSDK.so
位置：
/home/robot/teleop_act/01_teleoperation/third_party/

4、通讯配置：
机械臂：192.168.11.11
主机：192.168.11.102（直接在路由器上插网线自动分配）
Quest3：192.168.11.100

5、遥操作成品代码：

名称:
quest3_tb6_r5_omnipicker_data_product_v4_pro

迁移位置:
/home/robot/teleop_act/01_teleoperation/

包含:
- Quest3遥操作
- TB6-R5控制接口
- OmniPicker接口
- RealSense采集
- 数据采集流程
- wrist-pivot手腕标定相关配置

6、python依赖：
placo==0.9.23
pin==3.8.0
ibpinocchio==3.8.0
pyrealsense2==2.58.3.10794
numpy==2.2.6
opencv-python==5.0.0.93
xrobotoolkit-sdk==1.0.2
xrobotoolkit-teleop==1.0.3
LeRobot==0.6.1
PyTorch==2.11.0
Torchvision==0.26.0
Datasets==4.8.5
Torchcodec==0.11.1

