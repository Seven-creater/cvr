# Audio-CVR AAAI 论文通俗逻辑

> 更新日期：2026-07-21  
> 用途：向合作者快速解释整篇论文讲什么、证据是什么、哪些话可以说、哪些话不能说。  
> 权威关系：科学主线以 `aaai_audiocvr_paper_storyline_canonical.md` 为准，实验数字以 `aaai_audiocvr_submission_dossier_20260717.md` 及最终冻结结果为准。本文保留 Core150 阶段的通俗解释。

## 一句话说明

这篇论文不是在讲“我们发明了一个复杂的新模型”，而是在研究一个现有整体指标容易掩盖的问题：模型能够排除大量无关视频，却经常分不清尚未发生声音修改的 reference 与已经满足修改要求的 target。我们为此构造 Audio-CVR 数据、显式测量 source-target confusion，并证明在视觉语境基本不变时，audio 能显著帮助模型判断修改方向。

## 1. 从什么问题出发

组合视频检索的输入是：

```text
reference video/audio + modification text -> target video/audio
```

例如：

```text
reference：一个男人，背景有钢琴声
edit text：把钢琴声换成吉他声
target：画面仍是这个男人，但背景变成吉他声
```

真正困难的地方不是找到“一个长得像的视频”，而是理解有方向的变化：

```text
reference 仍是钢琴声，不满足 edit
target 已变为吉他声，满足 edit
```

reference 与 target 的画面和场景可能高度相似。只依赖整体相似度的模型，很容易把尚未修改的 reference 排在 target 前面。

## 2. 我们发现了什么

如果 gallery 主要由随机无关视频组成，模型很容易获得较高 Recall。这只能说明模型可以排除无关内容，不能证明它理解了 edit direction。

在冻结的 Audio-CVR test150 上，我们只移除当前 query 对应的 reference，其他候选、query、模型和 embedding cache 全部保持不变：

| 设置 | V+A+T Adapter R@1 |
|---|---:|
| With reference | 22.93% |
| Without reference | 99.47% |
| 移除 reference 后的增幅 | **+76.53pp** |

五个 seed 合计出现578次 top-1 错误，错误项全部是 `reference_negative`。这说明模型通常已经找到了正确的视频语境，却没有判断出“修改前”和“修改后”的方向。

这就是论文的核心研究对象：

> **source-target confusion，即模型把未修改的 reference 排在满足 edit 的 target 前面。**

## 3. 为什么需要重新构造数据

普通视频数据可能允许模型走捷径：

1. 只看画面就能定位 target。
2. 只匹配字幕、说话主题或关键词就能定位 target。
3. reference 与 target 的视觉差异太大，不需要听声音。
4. gallery 中大多是随机视频，检索任务过于简单。

因此，我们不是让 Omni 给出一次 `accept` 就直接收录，而是执行多阶段多模态自动构造与审核：

```text
自然视频切片与 source-aware pair mining
-> audio-only difference analysis
-> 仅根据声音证据生成 edit text
-> directional audio verification
-> muted-video shortcut rejection
-> full audiovisual consistency verification
-> ASR-shortcut screening
-> sampled repeat audit
-> source-disjoint train/val/test split
```

该流程分别确认：

```text
reference 不满足 edit
target 满足 edit
edit text 描述的是声音变化
数据经静音审核筛选，使画面本身不足以稳定确定 target
完整视频中该声音变化仍然成立
样本没有退化成 transcript/ASR matching
```

冻结主测试集包含150条 query：

| Subtype | Count |
|---|---:|
| Sound event | 120 |
| Music | 30 |
| Speech | 0 |

Speech 候选主要因 `full_av_not_required`、`audio_only_solvability_high` 或 ASR 风险被隔离到 diagnostic，不进入主结论。该 benchmark 是 `automatically curated and model-verified`，不是 human-validated。

## 4. 评测方法有什么不同

OmniCVR 已经把 source video 放入 gallery，并加入 source-local distractors。因此，我们不能声称“首次将 reference 放进候选库”。

我们的贡献是把 reference 从普通 gallery item 变成独立的诊断对象。Aggregate R@K 只告诉我们 target 排名，不能直接回答错误是否由 reference 获胜导致。为此，我们执行严格配对的两种评估：

```text
With-reference：保留当前 query 的 reference
Without-reference：只 mask 当前 query 的 reference，其他候选身份完全不变
```

除 R@1/R@5/R@10 外，还报告：

```text
target_beats_reference
target-reference score margin
reference rank
reference-induced R@1 drop
top-1 error type
```

这样可以区分两种完全不同的失败：

```text
模型根本找不到正确视频语境
模型找到正确语境，但把修改前的 reference 排在修改后的 target 前面
```

准确的 novelty 是 **reference-specific diagnosis and controlled counterfactual evaluation**，不是 reference inclusion 本身。

## 5. 为什么 audio 重要

我们在相同 query、相同 gallery identities、相同 adapter 和五个 seed 下比较：

```text
V+T：reference video + edit text，query/gallery 两侧均关闭 audio
V+A+T：reference video/audio + edit text，query/gallery 两侧均启用 audio
```

正式结果：

| Mode | R@1 | Target beats reference |
|---|---:|---:|
| V+T | 11.33% | 11.33% |
| V+A+T | 22.93% | 23.20% |
| Audio gain | **+11.60pp** | **+11.87pp** |

R@1 配对增益的95%置信区间为 `[+6.27,+17.07]pp`，Holm 校正后 `p<0.001`。target-reference score gap 也显著改善。

因此论文的准确主张不是“audio 对所有视频检索都必要”，而是：

> **当视觉语境基本保持、target 主要由方向性声音变化区分时，audio 为 reference-to-target 方向判别提供显著的额外证据。**

## 6. Adapter 在论文中是什么角色

实验使用：

```text
frozen E5-Omni-7B
identity-initialized low-rank residual adapter
validation-only model selection
5-seed final evaluation
```

adapter 只是 baseline，用来说明：

1. 原始 E5-Omni embedding 不擅长 Audio-CVR 的方向排序。
2. 少量任务数据可以改善任务适配。
3. 适配后的模型仍然严重受到 reference confusion 影响。

我们不把 adapter architecture、E5 recipe 或 auxiliary loss 写成论文主要创新。

## 7. OmniCVR 跨 Benchmark 实验说明什么

OmniCVR 实验已经完成，不再是待运行计划。我们从1,000条 audio-centered query 中保留995条有效 query，在2,000-item official gallery 中统一 mask 6 个解码失败视频，得到1,994个有效 gallery item。五个 seed、四种模式的 sample IDs 完全一致，审计违规为0。

### Reference removal

| Mode | Model | With reference R@1 | Without reference R@1 | 增幅 |
|---|---|---:|---:|---:|
| V+A+T | Adapter | 0.12% | 14.21% | **+14.09pp** |
| V+A+T | Base E5 | 0.00% | 17.39% | **+17.39pp** |
| V+T | Adapter | 0.00% | 12.86% | **+12.86pp** |

Adapter 的两项 reference-removal R@1 增幅均经过配对 bootstrap 和 randomization test，Holm 校正后 `p<0.001`。

### Audio 与迁移边界

OmniCVR 上 `V+A+T - V+T` 的 R@1 增益只有 `+0.12pp`，且不显著；audio 对 target-reference score gap 的改善为 `+0.0281`，Holm `p<0.001`。在 no-reference 条件下，adapter R@1 为14.21%，低于 Base E5 的17.39%。

因此，OmniCVR 实验只支持以下结论：

1. source-target confusion 不只存在于我们构造的数据，在公开 benchmark 上也存在。
2. 仅把 source 放入 gallery 不足以解释模型为什么失败，必须报告 reference-specific metrics。
3. Audio-CVR adapter 没有表现出跨 benchmark 性能优势。
4. Audio necessity 的主要显著证据来自 Audio-CVR test150，而不是 OmniCVR。

## 8. 整篇论文的三个贡献

### 贡献一：Audio-primary 数据构造方法

从自然视频中自动构造视觉语境保持、声音变化有方向、排除视觉与 ASR 捷径的 Audio-CVR triplets，并通过多阶段跨模态审核和重复复核控制质量。

### 贡献二：Reference-specific 诊断协议

把 unchanged reference 形式化为 query-specific counterfactual，报告 target-over-reference、score margin 和精确 paired reference removal，从 aggregate recall 中单独识别 source-target confusion。

### 贡献三：方向性 audio 证据

通过严格配对的 audio-on/off 与 reference-on/off 实验，证明 audio 在 Audio-CVR 中显著改善 reference-to-target 方向判断，并在 OmniCVR 上验证 reference confusion 的跨 benchmark 存在性。

## 9. 论文正文的逻辑顺序

```text
早期 CVR 主要研究视觉修改
-> CoVA 与 OmniCVR 已经把 audio 纳入 composed retrieval
-> OmniCVR 也已经把 source 放入 gallery
-> 但 aggregate R@K 不能单独揭示 source-target confusion
-> 我们构造 audio-primary、视觉语境保持的方向性数据
-> 显式定义 reference-specific metrics 与 paired removal
-> Audio-CVR 中移除 reference：R@1 22.93% -> 99.47%
-> Audio-CVR 中加入 audio：R@1 11.33% -> 22.93%
-> OmniCVR 中移除 reference：R@1 0.12% -> 14.21%
-> reference confusion 是跨 benchmark 的方向性失败，audio 能在专门控制的数据中显著缓解它
```

## 10. 最终可直接使用的论文故事

现有音视频组合检索模型能够排除大量无关视频，却经常无法区分尚未发生声音修改的 reference 与已经满足修改要求的 target。虽然 OmniCVR 等近期 benchmark 已把 source 放入 gallery，但 aggregate recall 不能单独揭示这种 source-target confusion。为研究这一问题，我们定义 Audio-CVR，并提出多阶段多模态自动构造流程，从自然视频中筛选视觉语境保持、声音变化有方向且不依赖视觉或 ASR 捷径的 triplets。我们进一步将 unchanged reference 形式化为 query-specific counterfactual，通过 target-over-reference、方向性分差和精确 paired reference removal 显式测量编辑方向理解。在冻结的150-query Audio-CVR 测试集上，加入 audio 将 R@1 从11.33%显著提升到22.93%，而移除 reference 则使 R@1 升至99.47%。在995-query OmniCVR 跨 benchmark 诊断中，移除 reference 同样使 R@1 从0.12%升至14.21%。这些结果表明，reference confusion 是 aggregate retrieval metrics 容易掩盖的核心失败模式，而 audio 能在视觉语境保持的 Audio-CVR 条件下显著帮助模型判断 reference-to-target 的修改方向。

## 11. 不能越界的表述

不得声称：

1. 我们首次把 reference/source 放入 composed video retrieval gallery。
2. Audio 对所有视频检索任务都必要。
3. Audio-CVR adapter 可以跨 benchmark 泛化。
4. Benchmark 已经过完整人工验证。
5. 当前 typed hard negatives 或 strict local negatives 已经足够全面。

必须写清：

1. OmniCVR 已有 source/reference 与 source-local distractors。
2. 我们的 novelty 是 reference-specific diagnosis，而不是 reference inclusion。
3. 主 benchmark 是150条 non-speech、model-verified query，且90%来自 avatar。
4. OmniCVR 的 audio R@1 增益不显著，只验证 reference confusion 的跨 benchmark 存在。
5. Adapter 是 baseline，不是方法贡献。
