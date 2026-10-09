# Nero 右单臂 RLT Stage 1 GPU2 训练交接

本文记录 2026 年 10 月 4 日多卡 4090 主机上的实际训练准备状态。当前结论是：RLinf、官方 π0.5 基础权重、RLToken/OpenPI 镜像、候选训练集、NAS 输出目录和 GPU2 单卡参数已经准备完成；正式训练前仍应确认 GPU2 上其他任务已释放足够显存，并了解当前 99 条数据只通过了结构检查，尚未逐条确认示范质量。

## 已部署资源

- RLinf：`/home/ubuntu/fjh/workbench/RLinf`，分支 `nero_pi05`。
- 官方 π0.5 base：`/home/ubuntu/lxd_shared/pi_origin/pi05_base`。
- 原始归一化候选集：`/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass`。
- 训练默认数据集：`/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass_norm_wide20`。
- 训练语言条件：`Stack the red square on top of the green square.`；真机推理应使用相同文本。
- Docker：`rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`。
- 训练输出：`/home/ubuntu/fjh/workbench/RLinf/logs`，对应 NAS `//192.168.1.114/robots/fjh/workbench/RLinf/logs`。

语言修正后的未拓宽数据集有 303 个文件、107,101,692 字节，目录聚合 SHA-256 为 `aaf5522ea5b154f6a13164badd62f93ae7b91400b605d7e153869dae8615d0c6`；拓宽版本有 304 个文件、107,102,378 字节，目录聚合 SHA-256 为 `8bbbd66b93b8ea22958316c8fa9309a1237eca888177ede969138d33b4f46ec1`。两者都包含 99 条、45,980 帧、两路 224×224 视频和右臂 8D 状态/动作。

## NAS 挂载

当前会话已经完成挂载和写入验证。重启后使用以下命令重新挂载，命令会交互式询问 SMB 密码，不把密码写入 shell 历史：

```bash
mkdir -p /home/ubuntu/fjh/workbench/RLinf/logs
sudo mount -t cifs \
  //192.168.1.114/robots/fjh/workbench/RLinf/logs \
  /home/ubuntu/fjh/workbench/RLinf/logs \
  -o username=liuxing,uid=1000,gid=1000,file_mode=0775,dir_mode=0775

mountpoint -q /home/ubuntu/fjh/workbench/RLinf/logs
```

启动脚本在挂载点缺失时会拒绝训练，避免 checkpoint 误写入根分区。

## GPU2 与 batch 参数

训练使用物理 GPU2：`cluster.component_placement.actor=2-2`。仍让容器看到全部 GPU，由 RLinf placement 选择 GPU2；不要把容器限制成只看到一张卡后仍使用 `2-2`，否则容器中的 GPU 编号会改变。

关键参数如下：

| 参数 | 值 | 说明 |
|---|---:|---|
| micro batch | 1 | 单步显存最小，不能再降低 |
| global batch | 16 | 保持常见的等效 batch |
| 梯度累计 | 16 | `16 / 1 / 1 GPU`，由 RLinf 自动计算 |
| gradient checkpointing | 开启 | 用计算换激活显存 |
| precision | FP32 配置，FSDP 参数 BF16 | reduce 和 buffer 保持 FP32 |
| horizon | 50 | 右臂 8D action chunk |
| max steps | 2000 | 每 500 step 保存一次 |
| learning rate | 1e-5 | 100 step warmup，cosine 到 2.5e-6 |
| dropout | wrist 0.1，state 0.1 | 两者独立；前视和 action 不丢弃 |
| train expert only | True | 冻结 VLM，AE、投影和 RLT 部分训练 |

GPU2 的共享占用会波动：准备过程中一度被其他进程占用约 9–10 GiB，最终检查时仅使用 216 MiB、空闲 48,304 MiB。虽然 gradient checkpointing 已开启，每次正式启动前仍应确认 GPU2 空闲显存达到约 40 GiB 且利用率较低。不要终止不属于本任务的进程。

## 归一化调整

`pi05_nero_right` 实际启用 quantile normalization，因此训练与推理使用 `q01/q99`，不是 `mean/std`。原始统计完整保留在原候选集；训练默认使用独立的 `norm_wide20` 变体。

调整规则为：状态和 action 的第 0–6 维围绕原区间中点，把 `q99-q01` 和 `std` 乘以 1.20，即区间两侧各增加原跨度的 10%。第 7 维夹爪保持原值，因为它是有明确物理边界的绝对量。该调整减少长尾样本产生的过大归一化值，但会使相同归一化输出对应略大的物理动作，因此训练和部署必须使用同一份调整后统计，不能混用。

原始统计 SHA-256 为 `59c5c98c64e426ad0a029d272c156f5ed860159c3a16118ae7fc9eb68ae445c35`，调整后为 `e387b81c83ed213867d3c70c68190718289f62791ac84550393cc4369a722330a`。新数据目录用硬链接复用视频和 Parquet，实际新增磁盘块约 16 KiB，而不是重新复制约 103 MiB 数据。

调整工具为：

```bash
python toolkits/lerobot/widen_norm_stats.py \
  /path/to/original/norm_stats.json \
  /path/to/wide/norm_stats.json \
  --scale 1.20
```

工具不会覆盖源文件，并在输出目录生成 `norm_stats_adjustment.json`，记录输入输出哈希、缩放比例和保留维度。

## 启动与验证

先检查 GPU2：

```bash
nvidia-smi --id=2
```

只跑一个优化步并保存结果：

```bash
cd /home/ubuntu/fjh/workbench/RLinf
bash examples/sft/run_nero_right_rlt_stage1_gpu2_docker.sh \
  runner.max_steps=1 runner.save_interval=1
```

确认日志中出现数据加载、冻结参数、梯度累计和 checkpoint 保存，并检查 GPU2 峰值显存。通过后再正式训练：

```bash
cd /home/ubuntu/fjh/workbench/RLinf
bash examples/sft/run_nero_right_rlt_stage1_gpu2_docker.sh
```

输出位于挂载后的 `logs/<时间戳>-nero_right_rlt_stage1_sft_openpi_pi05/`。用 `Ctrl+C` 中止时，最近一次按 `save_interval` 保存的 checkpoint 仍保留，但不会自动补存当前未到保存点的状态。

## 未完成事项

- 尚未在 GPU2 上启动训练或单步 smoke test，以免干扰当前占用 GPU 的其他任务。
- 99 条候选数据尚未逐条做成功率和动作质量筛选；结构通过不等于专家示范优质。
- 新模型是两相机、右臂 8D 接口，当前三相机、双臂真机部署桥仍需后续适配。

## GPU 3–4 训练入口

`examples/sft/run_nero_right_rlt_stage1_gpu34_docker.sh` 默认使用同一份
`0929_red_on_green_right_lerobot_structural_pass_norm_wide20` 数据集及其
delta action 统计量。训练配置默认 global batch 32、每卡 micro batch 16；
显存不足时可在命令末尾覆盖 `actor.micro_batch_size=4`，保持 global batch 32。
使用 `NERO_RIGHT_DATASET_HOST` 覆盖数据路径时，须确认其中的
`norm_stats.json` 对应前 7 维关节增量。
