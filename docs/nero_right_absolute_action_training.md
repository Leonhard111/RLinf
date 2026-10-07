# Nero 右臂 π0.5：绝对关节动作训练

本文对应多卡 4090 上的 `/home/ubuntu/fjh/workbench/RLinf`。新动作空间指 LeRobot `action` 前 7 维直接作为右臂绝对关节目标；第 8 维夹爪值仍为绝对值。图像、`observation.state`、任务文本、50 步 action chunk 和 episode 末端填充方式均不变。

## 实现及数据

- `rlinf/models/embodiment/openpi/dataconfig/nero_right_dataconfig.py` 增加 `use_delta_joint_actions`。默认 `true`，保持旧 checkpoint 行为；设为 `false` 时，训练输入不再执行 `DeltaActions`，推理输出也不再执行 `AbsoluteActions`。
- `toolkits/lerobot/recompute_nero_right_norm_stats.py` 从已有 LeRobot Parquet 重算 norm，无需重做视频。它逐帧取 50 步 chunk，超出 episode 时重复最后一个动作，再汇总所有时间步，按 8 个维度计算 `mean/std/q01/q99`。`--action-space delta` 复现旧算法；`absolute` 不减当前状态。
- `toolkits/lerobot/widen_norm_stats.py` 仍用于 1.20 倍拓宽，默认只拓宽前 7 维，不动夹爪。
- `examples/sft/config/nero_right_rlt_stage1_sft_openpi_pi05.yaml` 默认使用 GPU 3–4、global batch 32、每卡 micro batch 16 和绝对动作。`examples/sft/run_nero_right_rlt_stage1_gpu34_docker.sh` 是对应的启动入口；旧的 `gpu2` 脚本仍显式使用 GPU 2、batch 16 和增量动作。

**norm 与模式必须配对：**绝对动作 checkpoint 使用绝对动作 norm 与 `use_delta_joint_actions=false`；旧增量 checkpoint 使用旧 norm 与默认的 `true`。

本次已在多卡机准备好：

| 用途 | 路径 |
| --- | --- |
| 原结构化 LeRobot 数据 | `/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass` |
| 绝对动作训练副本 | `/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass_abs_wide20` |
| 未拓宽的绝对动作统计量 | `/home/ubuntu/fjh/workbench/data/nero_right_abs_norm_profiles/norm_stats_abs_raw.json` |
| 训练使用的统计量 | 上述训练副本根目录的 `norm_stats.json` |

数据共 99 条 episode、45,980 帧、30 Hz。原始 Parquet 的 `action` 本来就是绝对关节目标。新副本保留原有 Parquet、视频与元数据，只换 norm。最终 norm 的 SHA-256 为 `d7f9cb6ce74951dd175ae81996ab472c1077c43f12b8df21c4cf300249b2c9bc`。

## 从头重建副本

已有上表目录时跳过本节。如需重新实验，先为 `ABS` 和 `PROFILE_DIR` 选择**新的、不存在的路径**；统计工具和拓宽脚本默认拒绝覆盖输出。

```bash
cd /home/ubuntu/fjh/workbench/RLinf
SRC=/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass
ABS=/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass_abs_wide20
PROFILE_DIR=/home/ubuntu/fjh/workbench/data/nero_right_abs_norm_profiles
IMAGE=rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

mkdir -p "$ABS" "$PROFILE_DIR"
rsync -a --exclude='/norm_stats.json' "$SRC/" "$ABS/"

docker run --rm \
  -v "$PWD:/workspace/RLinf:ro" \
  -v "$SRC:/data:ro" \
  -v "$PROFILE_DIR:/profiles" \
  -w /workspace/RLinf \
  --entrypoint bash "$IMAGE" \
  -lc 'source switch_env openpi && python toolkits/lerobot/recompute_nero_right_norm_stats.py --dataset /data --output /profiles/norm_stats_abs_raw.json --action-space absolute --horizon 50'

python3 toolkits/lerobot/widen_norm_stats.py \
  "$PROFILE_DIR/norm_stats_abs_raw.json" \
  "$ABS/norm_stats.json" \
  --scale 1.20
```

新 norm 的 `state` 仍对应绝对关节状态；`actions` 对应绝对目标。π0.5 使用每维 `q01/q99` 分位数归一化，50 个 chunk 时间位置共用一套 8 维统计量。新旧拓宽数据的 state 统计量和夹爪动作统计量完全相同；变化的是前 7 个动作维度。

## 启动前检查

训练脚本要求仓库 `logs` 是 NAS 挂载点。运行前确认挂载；未挂载时启动脚本会拒绝训练。

```bash
cd /home/ubuntu/fjh/workbench/RLinf
mountpoint -q logs && echo "logs mounted"
test -f /home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass_abs_wide20/norm_stats.json
test -f /home/ubuntu/lxd_shared/pi_origin/pi05_base/model.safetensors
```

以下检查新旧 norm 没有混用：

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path('/home/ubuntu/fjh/workbench/data')
old = json.loads((base / '0929_red_on_green_right_lerobot_structural_pass_norm_wide20/norm_stats.json').read_text())['norm_stats']
new = json.loads((base / '0929_red_on_green_right_lerobot_structural_pass_abs_wide20/norm_stats.json').read_text())['norm_stats']
assert old['state'] == new['state']
assert all(old['actions'][key][-1] == new['actions'][key][-1] for key in ('mean', 'std', 'q01', 'q99'))
assert len(new['actions']['q01']) == len(new['actions']['q99']) == 8
assert all(high > low for low, high in zip(new['actions']['q01'], new['actions']['q99']))
assert old['actions']['q01'][:7] != new['actions']['q01'][:7]
print('absolute-action norm: OK')
PY
```

## 训练

先用下面的命令把 `runner.max_steps=20000` 和 `runner.save_interval=2000` 临时改为 `2`，并使用单独的 `RUN_ID` 做启动检查。确认训练、保存正常后再执行完整训练。当前默认设置为 GPU 3–4、global batch 32、每卡 micro batch 16、梯度累积 1、500 步 warmup。相比本地 `Aloha/nero_agilex/docs/data_prepare_traning.md` 中的旧单卡设置，批量和卡位已更新。

```bash
cd /home/ubuntu/fjh/workbench/RLinf
RUN_ID="$(date +%Y%m%d_%H%M%S)-nero_right_abs_stage1"

bash examples/sft/run_nero_right_rlt_stage1_gpu34_docker.sh \
  runner.max_steps=20000 \
  runner.save_interval=2000 \
  actor.optim.lr_warmup_steps=500 \
  runner.logger.log_path="/workspace/RLinf/logs/${RUN_ID}"
```

模型仍从 `pi05_base` 开始，`train_expert_only=True` 和梯度检查点等设置不变。不要用旧 delta 训练的 `global_step_14000` 直接续训本实验，否则无法单独比较动作空间。若 GPU 3–4 上有其他程序，先清空这些卡；如遇显存不足，可在命令末尾加 `actor.micro_batch_size=8`，此时每卡累计 2 个 micro batch。

### micro batch 实测依据

使用上述绝对动作数据、两张 48 GiB 4090、global batch 32 和相同模型设置短跑。下表 `time/step` 是 TensorBoard 中去掉首次启动步后的平均值；1/2/4 运行 5 步，8/16 运行 20 步。

| 每卡 micro batch | 每卡梯度累积 | 稳态秒/步 | 约样本/秒 |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 7.96 | 4.0 |
| 2 | 8 | 5.56 | 5.8 |
| 4 | 4 | 4.11 | 7.8 |
| 8 | 2 | 3.56 | 9.0 |
| **16** | **1** | **3.36** | **9.5** |

16 在 20 步内稳定且最快，因此作为默认值。监控中 GPU 3/4 的显存峰值接近 48 GiB，余量很小；这只证明当前固定图像尺寸、任务文本和独占显卡条件下的短程可运行性，不保证与其他 GPU 进程并行时仍可运行。

## 训练后验证与部署

1. 检查日志中的数据路径、`use_delta_joint_actions=false`、有限的 loss 和 checkpoint。抽一帧确认监督目标前 7 维等于 Parquet 的 `action`，而非 `action − observation.state`。
2. 把本次训练的 `full_weights.pt` **和本次的** `norm_stats.json` 一起复制到单卡 4090 的新 checkpoint 目录，勿沿用旧 delta norm。
3. 单卡推理 RLinf 也须包含相同的 DataConfig 修改，并在加载新模型时设置 `use_delta_joint_actions=false`。目前单卡实时入口使用固定的旧训练配置；只换权重和 norm 会令输出端再次加当前关节状态。应先修改配置并做离线／shadow 核对，再上真机。
4. 关闭 `AbsoluteActions` 后，反归一化的前 7 维已经是绝对右臂目标；同步 chunk 执行与 PCHIP 插值不应再额外加 state。

绝对动作是否改善抓取，应在相同起始位姿和部署设置下与旧增量模型对照验证。
