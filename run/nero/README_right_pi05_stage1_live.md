# Nero 右单臂 Stage 1 checkpoint 推理

本入口使用单卡 4090 上的 RLinf 模型和 AgileX 现有 native bridge。模型输入是场景相机、右腕相机、右臂 8D 状态；输出是 50×8 右臂动作。发送给桥之前，把本次观测的左臂位置复制为 50 个保持目标，组成既有的 50×16 协议动作。桥仍负责 TOPPRA、ROS 发布和已有安全守卫；第三路左腕相机仍由桥采集，但不进入模型。真机 BGR 图像在模型入口转为训练所用的 RGB。

模型权重与归一化文件须同在 `/home/rcir/Data_Disk/fjh/checkpoints/nero_right_rlt_stage1_14000/`。推理使用训练配置中的 `pi05_nero_right`、`use_rlt=True`、50 步 horizon 和 5 步采样；语言条件必须为 `Stack the red square on top of the green square.`。

## 先在单卡机离线验证

```bash
cd ~/Data_Disk/fjh/workbench/RLinf
bash run/nero/run_right_pi05_container.sh --mode offline --max-generations 3
```

这一步不连接 AgileX。首次加载和首个采样会较慢；后续打印每次推理时间、`(50, 16)` 动作形状和右臂关节相对当前状态的最大偏差。

## 真机 shadow

机器人现场先在一个 AgileX 终端启动 CAN 和 ROS2 节点，确认左右臂都报告 firmware、gripper initialized 和 control enabled：

```zsh
cd ~/nero_aloha
bash can_config.sh
source /opt/ros/humble/setup.zsh
source ~/nero_aloha/install/setup.zsh
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py \
  mode:=0 auto_enable:=true fast_mode:=false
```

在 AgileX 的另一个终端启动原生桥；此桥直接打开 Orbbec，不要同时启动相机 ROS 脚本或 LookCameras。

```zsh
source /opt/ros/humble/setup.zsh
source ~/nero_aloha/install/setup.zsh
cd ~/fjh/workbench/pi05
TASK='Stack the red square on top of the green square.'
./bridge/run_native_nero_bridge.sh \
  --prompt "$TASK" --bind tcp://0.0.0.0:5555 \
  --action-timeout 180 \
  --max-joint-speed 0.05 --max-joint-accel 0.2 \
  --monitor
```

单卡 4090 上运行：

```bash
cd ~/Data_Disk/fjh/workbench/RLinf
bash run/nero/run_right_pi05_container.sh --mode shadow --max-generations 3
```

shadow 会从真机读取相机和关节反馈并提交动作，但桥不向 ROS2 发布控制命令。先查看桥和推理端是否连续产生三次 generation，核对画面、关节输入和预测动作。

## 小范围发布验证

停止 shadow 桥，在现场人员可立即急停的条件下，以相同参数重启桥并增加 `--publish`；按桥的交互提示确认。单卡机先执行一个 generation：

```bash
bash run/nero/run_right_pi05_container.sh --mode publish --max-generations 1
```

确认动作方向和幅度后再增加 generation 数；`--max-generations 0` 表示持续运行直到 Ctrl+C。结束时也在 AgileX 桥终端按 Ctrl+C。`--mode` 与桥的 shadow/publish 模式必须一致，否则客户端会拒绝运行。

本入口只使用 14k 权重。换 checkpoint 时设置主机环境变量 `NERO_RIGHT_CHECKPOINT_DIR=/path/to/checkpoint_dir`，该目录需同时包含 `full_weights.pt` 与对应的 `norm_stats.json`。
