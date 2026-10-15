# Nero 右臂 π0.5：两张 48GB GPU 的 full-shard 全量训练

## 已验证的范围

在 GPU 3、4 上用 RLT Stage 1、完整 VLA 与 RLT 模块参与训练、`full_shard`、每卡 micro batch 16、global batch 32，完成了一个优化步骤并保存了完整模型权重。训练容器限制为 8 CPU、38 GiB 内存，训练没有显存 OOM。单步测试不能证明 30,000 步长跑没有数据、稳定性或磁盘问题。

当前机器的 SSH/LXC 环境显示约 90 GiB 可用内存，但嵌套 Docker 和 Ray 实际看到的物理主机在测试时仅有约 38–47 GiB 可用。无限制启动时 Ray 曾在载入模型阶段因主机内存达到阈值而中止。使用 8 CPU 减少 Ray 空闲进程，并设置明确的 Docker 内存上限，使测试可复现。

额外做过一次容器内存上限 **100 GiB** 的一步完整 checkpoint 探测。独立守护进程每秒记录物理主机可用内存、父 cgroup 和训练容器用量，物理主机可用内存低于 12 GiB 或父 cgroup 使用量高于 88 GB 时立即杀停训练容器。这次探测在模型加载阶段被守护进程停止；最后一次采样显示训练容器约 **37.03 GiB**、物理主机可用内存约 **10.75 GiB**，尚未到 checkpoint 保存。容器停止后主机内存和 GPU 显存均已回收。因此 **100 GiB 容器额度也不能证明当前物理主机有 100 GiB 可用内存**，这次探测没有测出完整 checkpoint 的峰值。原始采样在远端 `/tmp/nero_ckpt_probe_100g.csv`，守护日志在 `/tmp/nero_ckpt_probe_100g_guard.log`。

若物理主机以后释放了足够内存，可用同一个守护进程复测完整 checkpoint。此探测只运行一步，仍需先确认两卡空闲：

```bash
cd /home/ubuntu/fjh/workbench/RLinf
PROBE_ID="nero-ckpt-probe-$(date +%Y%m%d_%H%M%S)"
nohup bash examples/sft/nero_ckpt_memory_guard.sh \
  "$PROBE_ID" "/tmp/${PROBE_ID}.csv" \
  > "/tmp/${PROBE_ID}.guard.log" 2>&1 < /dev/null &

NERO_DOCKER_NAME="$PROBE_ID" NERO_DOCKER_CPUS=8 NERO_DOCKER_MEMORY=100g \
bash examples/sft/run_nero_right_rlt_stage1_gpu34_docker.sh \
  runner.max_steps=1 runner.save_interval=1 \
  actor.micro_batch_size=16 actor.global_batch_size=32 \
  actor.model.openpi.train_expert_only=false \
  +actor.model.load_base_on_rank0_only=true \
  actor.fsdp_config.sharding_strategy=full_shard \
  +data.num_workers=0 \
  runner.logger.log_path="/workspace/RLinf/logs/${PROBE_ID}"
```

该探测保留 `save_training_state=true` 默认值，因此若成功，才可观察完整训练状态和部署权重两个保存阶段的峰值。可从 CSV 计算容器使用量最大值，以及物理主机可用内存最小值；被守护进程停掉的测试不能用作完整 checkpoint 内存上限的证据。

## 启动 30,000 步

先确认 GPU 3、4 空闲，并确认 `logs` 已挂载 NAS；启动脚本会再检查数据集、基础权重与挂载。以下命令沿用 **相对关节动作** 数据和 norm。

```bash
cd /home/ubuntu/fjh/workbench/RLinf
RUN_ID="$(date +%Y%m%d_%H%M%S)-nero_right_full_rlt_fsdp_g34"

NERO_DOCKER_CPUS=8 NERO_DOCKER_MEMORY=80g \
bash examples/sft/run_nero_right_rlt_stage1_gpu34_docker.sh \
  runner.max_steps=20000 \
  actor.micro_batch_size=32 \
  actor.global_batch_size=64 \
  actor.model.openpi.train_expert_only=false \
  +actor.model.load_base_on_rank0_only=true \
  actor.fsdp_config.sharding_strategy=full_shard \
  +actor.fsdp_config.save_training_state=false \
  +data.num_workers=0 \
  runner.save_interval=2000 \
  actor.optim.lr_warmup_steps=500 \
  runner.logger.log_path="/workspace/RLinf/logs/${RUN_ID}"
```
查看docker内存上限设置
```
docker inspect -f 'Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}' a3be905679e1

docker stats --no-stream a3be905679e1
```

`actor.optim.total_training_steps` 在 YAML 中引用 `runner.max_steps`，无需另设。`load_base_on_rank0_only=true` 只在 rank 0 读取完整的基础 safetensors，然后由 FSDP 同步权重，减少模型加载时双份 CPU 峰值。其默认值是 `false`，不改变其他训练入口。

## 保存内容与续训限制

RLinf 原保存流程会先通过 Distributed Checkpoint 保存分片的模型、Adam 优化器和学习率调度器状态，随后额外导出部署用 `model_state_dict/full_weights.pt`。测试中的 CPU OOM 发生在前一个阶段，两名 worker 的常驻内存各从约 2.6–2.7 GiB 上升到约 15.7 GiB；完整权重导出还没有开始。因此单独关闭 `save_full_model_weights` 不能解决此次 OOM。

`save_training_state=false` 跳过前一个阶段，只导出完整模型权重；这条路径已经在 38 GiB 容器上完成一步训练和保存，生成约 16.4 GB 的 `full_weights.pt`。它可用于部署或作为新一轮训练的模型初值，**不能原样恢复 Adam 动量、学习率进度、数据迭代器与随机数状态**。发生中断后不能把这种目录当作完整 checkpoint 直接 `resume`。每 2,000 步保存一次权重可以保住模型参数，但长跑若要求精确续训，需要另行实现低峰值的完整训练状态保存，或在训练时取得更多主机内存。

## 显存理解与观测

基础模型约 3.35B 参数，RLT 模块约 0.75B 参数。按 FP32 参数、梯度、Adam 两份状态估算，`full_shard` 两卡均分后，每卡仅这些持久状态的理论下界约为 30.5 GiB；前向/反向激活、FSDP 临时 all-gather、CUDA 工作区和碎片仍需额外显存。单步测试在两卡上观测到约 42.5/44.3 GiB，原方案尝试 GPU 端完整保存时达到约 46 GiB，额外申请 256–512 MiB 即失败。micro batch 16 可以完成训练步，但**没有很大的显存提升余量**；减小 micro batch 也未能消除完整保存的额外峰值。

单卡只训练动作专家时，冻结的 VLM 无需梯度和 Adam 状态；与此处全量 VLA+RLT 的多卡显存和主机内存占用不能直接对比。

## 变更位置

- `rlinf/models/embodiment/openpi_rlinf/__init__.py`：可选的基础权重只由 rank 0 加载。
- `rlinf/hybrid_engines/fsdp/strategy/base.py`、`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`：可选的仅模型权重保存。
- `rlinf/workers/sft/fsdp_vla_sft_worker.py`：仅模型权重模式下跳过续训用的数据迭代器和随机数状态文件。
- `examples/sft/run_nero_right_rlt_stage1_gpu34_docker.sh`：可选 Docker CPU/内存限制。
- `examples/sft/nero_ckpt_memory_guard.sh`：高上限探测时使用的独立内存安全守护进程；普通训练命令不会自动启动它。

所有新选项默认关闭，原有训练命令的保存行为不变。
