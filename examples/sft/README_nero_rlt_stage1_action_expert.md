# Nero π0.5 RLT Stage 1：冻结 VLM、训练 action expert

本配置只影响 RLT Stage 1 的示范数据训练。设置 `actor.model.openpi.train_expert_only: True` 后，SigLIP 与 Gemma VLM（expert-0）冻结；action expert（expert-1）、动作/时间投影层和 RLT token 模块继续训练。损失仍为 `rlt_loss + rlt_alpha * vla_loss`。Stage 2 保持原来的冻结整个特征模型、训练 actor/critic 的流程。

参考配置：`examples/sft/config/nero_rlt_stage1_sft_openpi_pi05.yaml`。它不会改变原有 SFT/RLT 配置的默认行为；将开关设为 `False` 即恢复 VLA 全量训练。

## 启动前

1. 将采集数据转换为 LeRobot 数据集，并将数据及 `norm_stats.json` 放到 4090。数据为 30 Hz、三相机、16D 状态和 16D 绝对动作；50 个 action 对应约 1.67 秒。
2. 在参考配置中填写 `data.train_data_paths`、`actor.model.model_path`、`actor.model.openpi_data.repo_id`、`actor.model.openpi_data.norm_stats_path`。`repo_id` 和 norm stats 必须与实际数据及后续 Stage 2 一致。
3. 先用 `runner.max_steps=1` 做一次离线训练 smoke test，确认日志出现 `openpi_rlinf[sft]: train_expert_only=True`、`vla_loss` 与 `rlt_loss` 均为有限值，并确认 checkpoint 能重载。再设置正式步数。单张 4090 建议从 `micro_batch_size: 1` 起测显存。

```bash
cd /home/rcir/Data_Disk/fjh/workbench/RLinf
bash examples/sft/run_vla_sft.sh nero_rlt_stage1_sft_openpi_pi05
```

这个开关必须用在 `openpi_rlinf` 的 `task: sft` 上；旧 `openpi` 模型的同名开关不是 RLT Stage 1 实现。参考配置采用 `use_orig_params: True`，以支持同一 Gemma block 内冻结 VLM、训练 action expert 的 FSDP 参数组合。

Stage 1 产出的完整 `actor` checkpoint（包含 RLT token）应填入 Stage 2 的 `rollout.rlt_feature_model.model_path`，不要只复制裸 `model.safetensors`。
