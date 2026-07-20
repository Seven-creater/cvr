# Audio-CVR 审核后冻结、训练集与反向增强路线

日期：2026-07-20

## 1. 数量口径

```text
旧 640 + fresh 485 = 1125 条输入
过滤和去重后约 1005 条 = combined_pool_deduplicated
430 条 = 从完整去重池中抽出的 benchmark 审核 shortlist
```

430 不是全部可用数据。测试集和验证集冻结后，训练集必须回到完整去重池中构造。

## 2. 冻结设置

当前 speech 候选因 `full_av_not_required` 和 `audio_only_solvability_high` 全部退出主 benchmark，不降低门槛补数。

```text
test_main:
  sound_event = 120
  music = 30
  speech = 0
  total = 150

validation:
  sound_event = 22
  music = 8
  speech = 0
  total = 30
```

所有被判为 ASR-only、transcript-like、generic talking-head 或 audio-only-solvable 的 speech 记录保留到 diagnostic，不进入主测试集。

数据集来源比例是覆盖度诊断，不应覆盖三阶段内容质量判断。若严格 dataset cap 导致只能冻结一个仍由单一数据集主导的小集合，本轮允许显式设置 `--max-dataset-ratio 1.0 --relaxed-dataset-ratio 1.0`，冻结 150 条 source-unique 高质量样本。论文必须报告完整 dataset 分布、按 dataset 的结果和来源偏斜限制；不得把该设置描述为跨数据集均衡。另从主测试集构造 dominant-source 与 non-dominant-source 数量匹配的诊断子集，仅作 robustness 分析，不替代 150 条主测试集。

## 3. 训练集来源

```text
combined_pool_deduplicated
- test sample/pair/source
- validation sample/pair/source
- diagnostic sample
= clean forward training pool
```

冻结器按照 `raw_source_id/source_disjoint_group_id`、`pair_group_id/inverse_pair_group_id` 和 `sample_id` 同时排除 holdout。训练前必须运行 `audit-training-splits`；任何 source 或 pair 泄漏都应立即停止。

```bash
bash scripts/finalize_audio_cvr_review_and_prepare_training.sh \
  --work-root <WORK_ROOT>
```

主要输出：

```text
<WORK_ROOT>/benchmark_v1/test_main_150.jsonl
<WORK_ROOT>/benchmark_v1/val.jsonl
<WORK_ROOT>/benchmark_v1/train.jsonl
<WORK_ROOT>/benchmark_v1/test_asr_diagnostic.jsonl
<WORK_ROOT>/benchmark_v1/audit/training_split_audit.json
<WORK_ROOT>/benchmark_v1/audit/training_split_audit.md
```

## 4. Train-only inverse augmentation

数据切分完成并通过审计后，才允许交换 reference 和 target，并生成反向 edit：

```text
forward: A + edit(A->B) -> B
inverse: B + edit(B->A) -> A
```

反向记录必须重新通过 audio-only、muted-video 和 full-AV 验证。不能机械交换路径后直接接受。

```bash
python3 -m app.audio_lines_single_source augment-b-inverse \
  --run-root <INVERSE_RUN_ROOT> \
  --input-path <WORK_ROOT>/benchmark_v1/train.jsonl \
  --root <MEDIA_ROOT> \
  --base-url http://127.0.0.1:8093/v1 \
  --api-key EMPTY \
  --model <OMNI_MODEL> \
  --omni-retries 2 \
  --resume
```

输出的 `b_train_bidirectional_triplets.jsonl` 包含完整 forward train 和通过复核的 inverse train。forward/inverse 共用 pair group，只能位于 train；test-main 每个 pair 仍只保留一个方向。

该策略有直接的 CIR 研究依据：WACV 2024 的 [Bi-Directional Training for Composed Image Retrieval](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) 同时训练 forward 和 reversed queries；[Scale Up Composed Image Retrieval Learning via Modification Text Generation](https://arxiv.org/abs/2504.05316) 也生成 target-to-reference 的 reverse modification text。

## 5. 训练与正式评估

训练数量必须分开报告：

```text
train_forward_pair_count
train_inverse_accepted_count
train_directional_record_count
train_unique_source_count
```

inverse 增加方向监督，但不增加独立 source 数量。

训练继续使用冻结的 E5-Omni-7B 和轻量 adapter。超参数只在 30 条 validation 上选择；配置冻结后，使用 5 个 seed 对 150 条 test-main 做一次正式评估。主实验至少包含：

```text
with-reference vs without-reference
V+T vs V+A+T
R@1 / R@5 / R@10
target_beats_reference
target-reference score margin
reference-induced R@1 drop
```

test 结果不得参与 step、learning rate、batch size、inverse 接受规则或数据配额选择。
