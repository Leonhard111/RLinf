# Nero 右单臂双相机数据 → π0.5 RLT Stage 1

本方案只新增一条训练数据路径；原有 `pi05_nero`（三相机、双臂 16D）和原有部署桥不变。原始采集目录不会被修改或删除。新模型的接口为前视 RGB + 右腕 RGB + 右从臂 8D 状态，输出右臂 8D 动作（7 关节 + 夹爪）。50 个动作的训练 horizon 保持不变，标称采样率 30 Hz。

## 本次核验与清洗边界

原始数据在 AgileX 的 `/home/agilex/fjh/workbench/data/0929_stack2item/`，目录为 `episode0` 至 `episode120`。默认只审计 `episode15` 起的 106 条。2026-10-02 的完整解码审计结果：99 条结构通过（合计 45,980 个 30 Hz 对齐帧）、7 条因对齐后不足 50 帧被拒绝，分别为 `episode17`、`23`、`55`、`65`、`69`、`92`、`110`。完整 CSV 和每条的六图接触表位于 AgileX：`/home/agilex/fjh/workbench/data/0929_stack2item_audit_deep_final/`。

这些检查能发现缺帧、明显时间轴异常、图像解码失败、无效 8D 关节值、严重曝光或过低清晰度；`--deep` 会逐个解码相关 JPG/JSON，而默认快速审计只抽检 3 个时间点。**不能判定任务是否成功、示范是否熟练**。7 条被拒绝的记录只有 0、10 或 31 个对齐帧；99 条通过项最短为 354 帧（约 11.8 秒），不存在介于 50–353 帧的可疑短片段。原始 `selection.csv` 中通过项的 `decision` 仍为空，须审阅接触表或原始视频后再填 `accept` / `reject`。转换器只读取 `accept`，并会在写入前重新做深度结构检查。

为便于先传输和验证训练流程，已经另建 `selection_structural_pass_unreviewed.csv`，将 99 条结构通过项标为 `accept`，**不覆盖原始筛选表，也不代表人工确认的优质专家集**。使用该表生成的候选数据位于 AgileX `/home/agilex/fjh/workbench/data/0929_stack2item_right_lerobot_structural_pass/`：99 条、45,980 帧、198 个视频、99 个 Parquet 文件，LeRobot 首/中/末帧均可读。文件字节数为 107,151,596 B（102.19 MiB）；原始目录为 19,919,338,470 B（18.55 GiB），按文件字节数缩小约 186 倍。`du -sh` 的磁盘占用可能高于这些文件字节数。上传训练机前建议完成人工语义筛选；若先用候选集跑单步 smoke test，不能据其评估最终策略质量。

## AgileX：审计、预览、转换

```bash
cd ~/fjh/workbench/RLinf
conda run -n lerobot python toolkits/lerobot/prepare_nero_right_dataset.py audit \
  --source ~/fjh/workbench/data/0929_stack2item \
  --report-dir ~/fjh/workbench/data/0929_stack2item_audit_new \
  --contact-sheets --deep

# 单条预览：每列一个时间点，上排前视，下排右腕；均为最终 224×224 的尺寸。
conda run -n lerobot python toolkits/lerobot/prepare_nero_right_dataset.py preview \
  --source ~/fjh/workbench/data/0929_stack2item --episode 15 \
  --output ~/fjh/workbench/data/nero_right_preview15.jpg
```

审阅 `selection.csv` 和 `contact_sheets/`，只在最后一列 `decision` 填 `accept` 或 `reject`；不要改列名。已有 `/home/agilex/fjh/workbench/data/0929_stack2item_audit_deep_final/selection.csv` 可直接审阅，不必重复生成。`instructions.json` 的指令为 `null`，所以转换时必须显式给出真实任务文本：

```bash
conda run -n lerobot python toolkits/lerobot/prepare_nero_right_dataset.py convert \
  --source ~/fjh/workbench/data/0929_stack2item \
  --selection ~/fjh/workbench/data/0929_stack2item_audit_deep_final/selection.csv \
  --output ~/fjh/workbench/data/0929_stack2item_right_lerobot \
  --repo-id local/nero_right_stack2item \
  --task 'Place the red square on the green square, then put the blue square on the red square.'
```

请先确认该英文任务描述与实际采集指令一致。输出目录必须是新目录且位于原始目录之外；程序不会覆盖已有输出。转换依据均匀 30 Hz 时间轴，就近对齐前视、右腕、`puppetRight.position`（观察）和 `masterRight.position`（专家动作）。图像按比例双线性缩放并居中补黑到 224×224，编码为 LeRobot MP4；OpenPI 的 224 resize 随后基本是恒等变换。保留 8D 绝对目标，OpenPI 数据变换再将前 7 维转为相对当前关节的 delta，夹爪保持绝对值。`norm_stats.json` 按 50-step delta action 和状态一起写出。每条 episode 的采样长度与源映射记录在 `conversion_manifest.json`。

单条 `episode15` 的试转换为 412 帧，LeRobot 两路视频约 1 MB，原始该条约 195 MB；LeRobot 读取验证得到 `(8,)` state、`(50,8)` action 和两路 `(3,224,224)` 图像。解码视频第一帧与预处理图的平均绝对像素差约 1.33/255。人工筛选后的正式数据体积将取决于最终保留条数。

## 传到 4090 并训练

只传转换后的目录（包括 `meta/`、`data/`、`videos/` 和 `norm_stats.json`），不要传原始目录。以下命令传输的是**未经人工语义筛选的候选集**；若完成筛选后另行转换正式数据，请同步改为正式目录名：

```bash
# AgileX
rsync -a --info=progress2 \
  ~/fjh/workbench/data/0929_stack2item_right_lerobot_structural_pass/ \
  rcir@192.168.50.102:/home/rcir/Data_Disk/fjh/workbench/data/0929_stack2item_right_lerobot_structural_pass/
```

在已挂载 RLinf 仓库、数据目录和 checkpoint 的 4090 训练容器内，先做单步验证：

```bash
cd /workspace/RLinf
source switch_env openpi
export NERO_RIGHT_DATASET=/workspace/data/0929_stack2item_right_lerobot_structural_pass
export NERO_PI05_CHECKPOINT=/workspace/fjh/checkpoints/nero_pi05_30000_openpi_rlinf
bash examples/sft/train_nero_right_rlt_stage1.sh runner.max_steps=1

# 检查 loss / checkpoint 后，正式训练：
bash examples/sft/train_nero_right_rlt_stage1.sh
```

容器中的这两个路径必须与你实际的挂载点一致。配置 `examples/sft/config/nero_right_rlt_stage1_sft_openpi_pi05.yaml` 使用 `pi05_nero_right`、`action_dim: 8`、两相机、50-step horizon、RLT token，且 `train_expert_only: True`：冻结 SigLIP 和 Gemma VLM，action expert、投影和 RLT 模块继续训练。训练端实测 OpenPI loader 输出两相机、`[1,32]` 的 padded state、`[1,200]` 的 tokenized prompt 和 `[1,50,32]` 的 padded action；物理输出仍裁回右臂 8D。

显存须单步验证，不能按“仅 3 亿参数 AE”估算。现有基础 checkpoint 的 safetensors 元信息为 3,353,433,872 个 FP32 参数；按当前冻结规则，其中约 430,098,464 个基础参数仍训练。默认 RLT encoder+decoder 实测另有 745,715,712 个可训练参数，因此总可训练量约 1,175,814,176。若按 FP32 参数、梯度和 Adam 两份动量粗估，静态部分约 28.41 GiB，**尚未计入**激活、FSDP/bf16 临时副本、CUDA 工作区等；实际驻留精度也可能改变这个数，所以它不是实测峰值或严格下界。4090 现场显示总显存 49,140 MiB、当前空闲 35,938 MiB，另一个进程占用约 12,398 MiB；在不干扰该进程的前提下，是否能完成一个优化步仍未知，务必先跑 `runner.max_steps=1` 并观察 `nvidia-smi`。`global_batch_size=16`、`micro_batch_size=1` 使用梯度累积，不是同时向 GPU 放 16 个样本。

两个 dropout 完全独立、逐样本抽取，只用于 SFT 训练：右腕图像置黑并将 mask 设为 false；归一化后的 8D 状态在 π0.5 离散 token 化**之前**置零。前视图像与 action target 永不 dropout，验证集与无 action 的推理输入也不会 dropout。让关键视角常驻、随机丢其他视角的做法可参考 [Octo 官方数据代码](https://github.com/octo-models/octo/blob/main/octo/data/dataset.py)；[MA-VLA 论文](https://arxiv.org/abs/2608.25864) 也采用 view dropout，但都不能直接给出 Nero 的最优概率。默认 `wrist_dropout_prob=0.1`、`state_dropout_prob=0.1` 是本项目的保守起点，不是论文定值；此时约 19% 的样本至少缺一项、1% 同时缺两项。先比较 `0/0` 基线；若模型过度依赖手眼或关节，可小幅提高相应概率。`0.2/0.3` 会使 44% 样本至少缺一项、6% 同时缺两项，对仅有两相机的小数据集偏激进，建议在验证集证实收益后再用。

当前 3 相机/16D 真机 bridge **尚未改为**新 2 相机/8D 接口；本交付覆盖数据处理与 RLT Stage 1 训练，不包含真机部署或控制器修改。

## 代码边界与验证

- `toolkits/lerobot/prepare_nero_right_dataset.py`：只读原始数据，审计、预览、转换、统计。
- `rlinf/models/embodiment/openpi/policies/nero_right_policy.py` 和 `dataconfig/nero_right_dataconfig.py`：新的右单臂输入输出与训练 dropout；原 `nero_policy.py` 不改。
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`：只增加 `pi05_nero_right` 注册。
- `rlinf/models/embodiment/openpi_rlinf/pi0_model/model.py`：按实际存在的相机键做预处理，保留原三相机路径。
- `rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py`：新配置的验证集关闭 dropout。
- `examples/sft/config/nero_right_rlt_stage1_sft_openpi_pi05.yaml`、`examples/sft/train_nero_right_rlt_stage1.sh`：独立训练入口，不改旧配置。
- `tests/unit_tests/test_nero_right_rlt_pipeline.py`：resize、30 Hz 对齐/统计、两个独立 dropout、双相机模型预处理测试。

已验证：AgileX LeRobot 0.3.4 上完整候选集转换完成且首/中/末帧可读；4090 的 RLinf OpenPI 容器可加载新配置和单条真实 LeRobot 样本；单元测试 4/4 通过。尚未对人工筛选后的完整数据启动正式训练，也未做真机部署测试。
