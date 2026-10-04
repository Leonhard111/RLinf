# Nero 采集数据转 RLinf/OpenPI LeRobot 数据集

该工具把 AgileX `data_tools` 生成的：

```text
<capture-root>/episode<N>/episode<N>.hdf5
```

转换成 RLinf 的 `pi05_nero` 数据配置可直接读取的 LeRobot v2.1 数据集。
转换过程完全离线，不连接 ROS、CAN 或机械臂。

## 数据契约

- 频率：30Hz。
- `observation.state`：16D，顺序为左从臂 7 关节与夹爪，再接右从臂。
- `action`：默认采用左右主臂位置，16D，每个数据行只保存一个绝对动作。
- 图像：`front -> cam_high`、`left -> cam_left_wrist`、
  `right -> cam_right_wrist`，默认转成 RGB 640×480 视频。
- 训练时 OpenPI/LeRobot 根据 30Hz 时间轴自动取得未来 50 个 `action`，
  组成 π0.5 的 `[50, 16]` action horizon。转换器不会预先把 action 写成
  `[50,16]` 或 `[200,16]`，否则加载器会重复分块。

## 1. 激活 AgileX 转换环境

```bash
cd ~/fjh/workbench/RLinf
conda activate lerobot
export PYTHONPATH=$PWD:${PYTHONPATH}
```

## 2. 先只验证原始数据

当前 `/home/agilex/data_new` 的 instruction 是 `null`，因此必须显式提供
训练 prompt：

```bash
python toolkits/lerobot/convert_nero_capture_to_lerobot.py \
  --source /home/agilex/data_new \
  --prompt "First, place the red square on the green square, and then put the blue square on the red square." \
  --validate-only
```

验证会检查 HDF5 键、长度、NaN/Inf、时间戳、30Hz 频率和每个 episode 的
第一张图像。缺少 HDF5 的未完成 episode 会明确报警并跳过；添加 `--strict`
可以改为立即失败。

## 3. 小规模 smoke test

输出目录必须不存在，工具不会覆盖已有数据集：

```bash
python toolkits/lerobot/convert_nero_capture_to_lerobot.py \
  --source /home/agilex/data_new \
  --output /home/agilex/fjh/datasets/nero_smoke_lerobot \
  --repo-id local/nero_smoke_lerobot \
  --prompt "First, place the red square on the green square, and then put the blue square on the red square." \
  --episodes 0 \
  --max-frames-per-episode 30
```

`--max-frames-per-episode` 仅用于调试，正式转换不要设置。

## 4. 正式转换全部数据

```bash
python toolkits/lerobot/convert_nero_capture_to_lerobot.py \
  --source /home/agilex/data_new \
  --output /home/agilex/fjh/datasets/nero_data_new_lerobot \
  --repo-id local/nero_data_new_lerobot \
  --prompt "First, place the red square on the green square, and then put the blue square on the red square."
```

完成后应存在：

```text
nero_data_new_lerobot/
├── data/
├── meta/
├── videos/
└── conversion_report.json
```

`conversion_report.json` 会记录 episode 映射、帧数、prompt、测得频率、
跳过的未完成 episode，以及 state/action 的确切来源。

## 5. 在 4090 上用于 π0.5 SFT/RLT Stage 1

将转换后的整个目录复制到 4090，例如：

```text
~/Data_Disk/fjh/datasets/nero_data_new_lerobot
```

训练配置中需要保持：

```yaml
data:
  train_data_paths: /home/rcir/Data_Disk/fjh/datasets/nero_data_new_lerobot

actor:
  model:
    num_action_chunks: 50
    action_dim: 16
    openpi_data:
      norm_stats_path: /home/rcir/Data_Disk/fjh/datasets/nero_data_new_lerobot
    openpi:
      config_name: pi05_nero
      num_images_in_input: 3
      action_horizon: 50
      action_chunk: 50
      action_env_dim: 16
      model_action_dim: 32
```

在开始训练前，使用同一个数据路径和 `pi05_nero` 配置计算新的 norm stats：

```bash
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_nero \
  --repo-id /home/rcir/Data_Disk/fjh/datasets/nero_data_new_lerobot
```

norm stats 必须与本次转换出的 state/action 数据匹配，不能复用其他数据集的
统计量。上面的统计命令会在数据集根目录生成 `norm_stats.json`；训练配置的
`openpi_data.norm_stats_path` 应指向这个数据集根目录。

## 可选参数

- `--action-source master`：默认值，和现有 Nero 遥操作训练约定一致。
- `--action-source puppet`：以真实从臂位置作为监督动作，用于对照实验。
- `--episodes 0 1 5`：只转换指定的原始 episode。
- `--limit-episodes N`：按 episode 编号排序后只取前 N 条。
- `--strict`：遇到没有 HDF5 的 episode 目录时失败，而不是报警后跳过。
