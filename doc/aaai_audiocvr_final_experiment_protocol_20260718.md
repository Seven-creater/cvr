# Audio-CVR AAAI 最终实验协议

## 1. 目的

本协议替代“固定 600 步跑一次 test”的做法。目标不是继续试出一个更好看的单次结果，而是产出可以写入论文、能回答审稿人复现问题的最终证据：

1. 所有超参数只在 validation split 上选择；
2. 配置锁定后才运行 test-main；
3. 最终结果使用 5 个独立随机种子，报告 mean/std；
4. `V+A+T` 与 `V+T` 使用相同 query、相同 gallery 做成对比较；
5. 同时报告置信区间、随机化检验和逐 seed McNemar 检验；
6. 七种模态消融和传统 late-fusion baseline 使用同一测试协议；
7. 保留每条 query 的 rank、score gap 和 top-k 错例，支持人工复核。

本轮模型仍为：

```text
frozen E5-Omni-7B + lightweight projection adapter
training profile = e5_omni_recipe
LoRA = off
AudioDelta task-specific losses = off
```

## 2. 前人经验与本项目对应关系

### 2.1 AAAI 的实验要求

[AAAI Reproducibility Checklist](https://aaai.org/conference/aaai/aaai-23/reproducibility-checklist/) 明确要求论文说明：随机运行次数、方差或置信区间、统计检验、最终超参数，以及开发期间尝试的超参数范围和选择标准。因此本协议固定记录：

```text
coarse grid 的 steps / learning rate / batch size
validation-only selection rule
refinement 使用的 seeds
final test 使用的 seeds
每个 seed 的完整结果和 loss curve
```

### 2.2 CoVR 的数据与人工评估经验

[CoVR](https://ojs.aaai.org/index.php/AAAI/article/view/28334) 使用可扩展的自动 triplet 构造方法，同时另外建立人工标注 evaluation set。对应到本项目：训练数据可以由多阶段自动质检产生，但正式 test-main 必须导出人工核验清单，不能把 `manual_review_required=0` 等价为人工验证完成。

### 2.3 EgoCVR 的 global/local gallery

[EgoCVR](https://github.com/ExplainableML/EgoCVR) 同时报告 global gallery 和限制为同一视频序列的 local gallery。对应到 Audio-CVR：

```text
global: target + reference + typed hard negatives + random distractors
local: target + reference + strict same-source clips
```

当前 pilot 的 strict local 覆盖不足，因此本轮 `typed_hardneg` 是可执行主协议，local 结果只能在有真实同源候选时补报，不能用跨源 fallback 冒充 strict local。

### 2.4 CoVA 的 audio-aware benchmark 与融合基线

[CoVA](https://perceptualai-lab.github.io/CoVA/) 直接指出传统 CoVR 忽略 audio，并使用 audio/visual/text fusion 与 hard negatives 评估 audio-visual composed retrieval。对应到本项目：

1. 必须报告七种 audio necessity 模式；
2. 核心比较是 `V+A+T` 对 `V+T`；
3. 增加一个 validation 选权重的 `V+T` 与 `A+T` late-fusion baseline；
4. late fusion 只称为传统分数融合，不称为 CoVA/AVT 复现。

CoVA 官方代码依赖 CLIP、AST 和其数据格式。未完成输入与 checkpoint 对齐前，不能把本项目的 late fusion 标为 CoVA 方法结果。

### 2.5 e5-omni 的 adapter recipe

[e5-omni](https://arxiv.org/abs/2601.03666) 使用 modality-aware temperature、negative curriculum/debiasing、batch whitening/covariance regularization处理 omni-modal embedding 对齐。当前 `e5_omni_recipe` 保留这些组件，额外 AudioDelta loss 不进入本轮方法贡献。

### 2.6 检索显著性检验

AAAI checklist 建议使用合适的成对统计检验。IR 研究还表明，随机化、bootstrap、Wilcoxon 等检验在不同样本规模下有不同误差与功效特性，不能只给单个平均值。参考：

- [Agreement Among Statistical Significance Tests for Information Retrieval Evaluation](https://ciir-publications.cs.umass.edu/getabs.php?id=885)
- [Using score distributions to compare statistical significance tests for information retrieval evaluation](https://arxiv.org/abs/1901.10696)
- [Statistical Significance Testing in Information Retrieval](https://arxiv.org/abs/1905.11096)

本协议同时输出：

```text
5-seed mean/std
query-level paired bootstrap 95% CI
query-level sign-flip randomization p-value
per-seed exact McNemar p-value + Holm correction
```

seed 与 query 是两类不同的不确定性来源，不把 `5 x query_count` 粗暴当成独立样本。

## 3. 数据拆分

输入使用已有 source-disjoint split，不重新随机切分：

```text
train.jsonl
val.jsonl
test_main.jsonl
```

`prepare-splits` 执行：

1. 原样保留 train/val/test 的 source assignment；
2. 检查 source、pair 和 inverse pair 是否跨 split；
3. 从原 test 文件中过滤 `split_tier=main` 且非 inverse 的正式 `test_main`；
4. 同时保留未过滤的 `test_all` 作为审计文件；
5. 生成 `test_main_human_review.jsonl`。

如果发现任何 source/pair leakage，实验立即失败，不继续编码。

## 4. Validation-only 超参数选择

### 4.1 Coarse grid

固定 seed 13，搜索：

```text
steps         = 60, 120, 240, 450, 700, 1000
learning rate = 1e-4, 3e-4, 1e-3
batch size    = 4, 8, 16
```

共 54 个配置。搜索 batch size 的动机来自对比学习对 in-batch negatives 的敏感性；搜索更宽的 steps 是为了避免把 600 步事后当成最优。

### 4.2 Three-seed refinement

coarse grid 前 6 个配置使用：

```text
seeds = 13, 23, 42
```

重新独立训练并计算 validation mean/std。

### 4.3 预注册选择规则

选择顺序固定为：

```text
1. 最大 mean validation R@1
2. 最大 mean target_beats_reference
3. 最小 R@1 std
4. 更少 steps
```

选择程序不会读取 test summary。最终配置写入：

```text
validation/final_selection/validation_model_selection.json
validation/final_selection/selected_config.tsv
```

## 5. Final test

配置锁定后，在 train split 上使用 5 个 seeds 独立训练：

```text
13, 23, 42, 71, 101
```

test-main 使用同一套 1000-item typed-hardneg gallery：

```text
positive targets
reference negatives
visual_hard
audio_hard
asr_hard
random distractors
```

如果当前 test-main 没有经过 false-negative guard 的 strict local clip，则不生成假的 local_same_source 结论。

## 6. 七种 Audio Necessity 消融

| 模式 | Query | Gallery | Audio 状态 |
|---|---|---|---|
| T-only-fullAV | edit text | full AV | gallery on |
| V-only | reference video，无 edit text | muted video | 两侧 off |
| A-only | reference audio，无 edit text | audio | 两侧 audio-only |
| V+T | reference muted video + edit | muted video | 两侧 off |
| A+T | reference audio + edit | audio | 两侧 audio-only |
| V+A | reference full AV，无 edit text | full AV | 两侧 on |
| V+A+T | reference full AV + edit | full AV | 两侧 on |

注意：`video_only` 在当前代码中表示“不含 edit text 的视频 query”；`V+T` 使用 `composed + video_audio_mode=off`。所有模式必须共享 sample IDs、gallery count、positive index 和 reference index。

## 7. Late-fusion baseline

定义：

```text
score = alpha * score(V+T) + (1-alpha) * score(A+T)
```

`alpha` 只在 validation 上从 `0.0, 0.1, ..., 1.0` 选择。每个 final seed 使用其 adapter 在 validation 上选择 alpha，再将该 alpha 固定用于 test。

这个对照回答：

> 完整 `V+A+T` 联合编码是否优于简单地把视觉文本分数和音频文本分数相加？

## 8. 统计与输出

主要指标：

```text
R@1 / R@5 / R@10
target_beats_reference
target-reference score gap
reference / visual / audio / ASR hard-negative breakdown
subtype breakdown
top-k error type
```

主统计比较：

```text
V+A+T - V+T
```

输出：

```text
paper_splits/split_verification.json
validation/*/validation_model_selection.json
statistics/per_seed_results.json
statistics/test_main_mean_std.json
statistics/test_main_comparison.md
statistics/audio_gain_summary.md
statistics/error_breakdown.json
statistics/audit.json
paper_splits/test_main_human_review.jsonl
```

## 9. 论文可用判定

只有同时满足以下条件，`audio gain` 才进入主结论：

1. `V+A+T - V+T` 的 R@1 mean 为正；
2. paired bootstrap 95% CI 不跨 0，或随机化检验在预注册显著性水平下成立；
3. 5 个 seeds 的方向基本一致；
4. target-beats-reference 同时改善；
5. audit 无 split、sample、gallery 或 NaN 违规；
6. test-main 的关键样本完成最小人工核验。

如果 R@1 提升但 CI 跨 0，应写成趋势而不是显著提升。如果 late fusion 与 V+A+T 相当，论文应承认当前 adapter 的联合融合优势尚未建立。

## 10. 运行方式

服务器应在干净的 GitHub `main` clone 中运行：

```bash
setsid nohup bash scripts/run_audio_cvr_aaai_final_experiment.sh \
  --run-root <RUN_ROOT> \
  --split-root <RUN_ROOT>/b_splits \
  --output-dir <OUTPUT_DIR> \
  --gpu-ids 1,2,3,4,5,6,7 \
  > <MASTER_LOG> 2>&1 < /dev/null &
```

脚本支持断点复用：已有的 `eval_embeddings.npz`、`adapter.pt` 和 `summary.json` 不重复生成。`cache-embeddings --skip-train` 只跳过七模态 test cache 中重复的训练集编码，不影响唯一的 V+A+T train cache。

## 11. 尚未覆盖的外部基线

本脚本完整覆盖项目内部的正式实验，但不声称自动完成外部模型复现。投稿前仍建议单独完成：

```text
CoVA/AVT（官方 CLIP + AST 实现）
LanguageBind 或 ImageBind composed baseline
BLIP-CoVR / TF-CVR visual composed baseline
```

这些模型的输入格式、训练数据和 checkpoint 必须严格对齐后才能进入对比表；不能把 E5 late fusion 改名为其中任何一种方法。
