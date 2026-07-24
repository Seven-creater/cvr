# Audio-CVR：AAAI 最终实验数据与论文证据总览

> 更新日期：2026-07-23  
> 状态：最终 Test1000、E5-Omni、ImageBind 和 OmniCVR 实验均已完成。  
> 证据规则：本文只保留当前论文可使用的最终结果；已被替代的 pilot、旧 gallery 结果和旧训练实验不再列入。

## 1. 论文主线

Audio-CVR 研究的不是一般视频相似性检索，而是：

```text
reference video/audio + directional audio edit text -> target video/audio
```

其中，reference 表示修改前状态，target 表示满足 edit 后的状态。核心问题是：

> 模型能否在视觉语境基本保持时，根据声音修改方向，把 target 排在 unchanged reference 前面？

本文围绕三项贡献组织：

1. **Audio-primary directional CVR**：构造以声音变化决定 target、同时筛查视觉和 ASR 捷径的检索任务。
2. **多阶段自动构造方法**：通过 audio-only 分析与验证、muted-video shortcut screening、full-AV consistency、ASR-shortcut screening、去重和 source-disjoint 审计构造数据。
3. **Reference-specific diagnosis**：把 unchanged reference 视为 query-specific pre-edit counterfactual，使用 target-over-reference、score margin、错误归因和 exact own-reference masking 显式测量 source-target confusion。

E5 adapter 是任务适配 baseline，ImageBind 是独立零样本 baseline。二者都不是论文的主要方法贡献。

## 2. 最终 Audio-CVR Test1000

### 2.1 构造流程

```text
异构原始视频
-> 6-9 秒 source-aware clips
-> 同源候选配对
-> audio-only change analysis / edit generation
-> directional audio verification
-> muted-video shortcut screening
-> full-AV consistency
-> ASR-shortcut screening
-> sampled repeat audit
-> sample/source/pair 去重与泄漏审计
-> 冻结 Test1000
```

该流程的目标是降低视觉和 transcript 捷径风险，而不是声称所有捷径已被人工彻底排除。

### 2.2 冻结规模

| 项目 | 数量 |
|---|---:|
| 最终 query | **1,000** |
| Sound event | **829** |
| Music | **171** |
| Speech | **0** |
| 固定既有集合 | 516 |
| 最终新增样本 | 484 |
| Target gallery items | 1,000 |
| Unchanged reference items | 1,000 |
| With-reference gallery | **2,000** |
| Without-reference effective gallery | **1,999 / query** |

`without-reference` 不重新采样 gallery，只在同一 score matrix 中 mask 当前 query 自己的 reference；其他 1,999 个候选完全不变。

最终补齐漏斗如下：

| 阶段 | 数量 | Sound event | Music |
|---|---:|---:|---:|
| 固定既有集合 | 516 | 414 | 102 |
| 待审核候选池 | 1,519 | - | - |
| 当前策略下 eligible | 518 | 448 | 70 |
| 选入的新增样本 | 484 | 415 | 69 |
| **冻结 Test1000** | **1,000** | **829** | **171** |
| Shortfall | **0** | - | - |

### 2.3 数据来源

| 数据来源族 | Query 数 | 占比 |
|---|---:|---:|
| AVATAR family | 510 | 51.0% |
| VGGSound family | 275 | 27.5% |
| AVE | 196 | 19.6% |
| WorldSense | 10 | 1.0% |
| VGG-MonoAudio | 9 | 0.9% |
| **总计** | **1,000** | **100%** |

其中 AVATAR family 合并内部 `avatar` 与 `avqa_videos` 标签，VGGSound family 合并新旧 VGGSound 标签。

### 2.4 数据审计

| 审计项 | 结果 |
|---|---:|
| Duplicate sample | 0 |
| Duplicate source | 0 |
| Duplicate canonical pair | 0 |
| Leakage violations | 0 |
| Missing media | 0 |
| Selection uses retrieval scores | false |
| Repeat review requested | 78 |
| Repeat review completed | 78 |
| Exact decision agreement | 79.49% |
| Field-level agreement | 85.64% |
| Speech-role agreement | 57.69% |
| Missing repeat reviews | 0 |

准确口径是：

> Full Test1000 is automatically curated and model-verified, not fully human-validated. All 78 requested sampled repeats were completed after freezing as an observational stability audit; they did not change test membership or selection.

Core150 中随机抽取的 10 条由两位作者共同人工查看，并被定性判断为质量较高；这只是抽查，不是 150 条全量审核、独立盲审或正式人工一致性实验。本文的最终成绩统一以 Test1000 为准。

### 2.5 冻结标识

```text
test:
runs/audio_cvr_test1000_unified_auditonly_20260723_142000/
final_test1000/test_main_1000.jsonl

SHA256:
70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e
```

## 3. 最终实验设置

### 3.1 模型

| 项目 | E5-Omni baseline | ImageBind baseline |
|---|---|---|
| Backbone | Frozen E5-Omni-7B | ImageBind-Huge |
| Embedding dim | 3,584 | 1,024 |
| 训练方式 | Low-rank residual adapter | Zero-shot，无训练 |
| Adapter rank | 32 | - |
| Trainable parameters | 688,131 | 0 |
| 训练数据 | 65 forward + 24 verified inverse = 89 directional instances | 不使用训练集 |
| 独立 source pairs | 65 | - |
| Validation queries | 28 | - |
| Steps / LR / batch | 400 / 1e-3 / 8 | - |
| Final seeds | 13, 23, 42, 71, 101 | 单次确定性推理 |
| 配置选择 | 仅使用 validation；one-standard-error rule | 固定等权模态组合 |

E5 的 AudioDelta、reference、hard-negative 和 edit-type 专用附加损失权重均为 0；最终贡献不能写成新 loss。ImageBind 的 V/A/T 向量分别归一化后等权相加，再次归一化；没有在 Test1000 上调融合权重。

### 3.2 评估模式

| 模式 | Query | Gallery document |
|---|---|---|
| T-only-fullAV | Edit text | Full audiovisual candidate |
| V-only | Reference vision | Candidate vision |
| A-only | Reference audio | Candidate audio |
| V+T | Reference vision + edit text | Candidate vision |
| A+T | Reference audio + edit text | Candidate audio |
| V+A | Reference vision + audio | Candidate vision + audio |
| V+A+T | Reference vision + audio + edit text | Candidate vision + audio |

### 3.3 指标与统计

主指标为：

```text
R@1 / R@5 / R@10 / MRR
target-over-reference accuracy
target-reference cosine margin
reference-induced R@1 gain after exact masking
top-1 own-reference error attribution
```

配对检验使用 20,000 次 bootstrap、20,000 次 randomization test、McNemar 检验和 Holm 多重比较校正。E5 与 ImageBind 在相同的 1,000 个 query 上评估，NaN/Inf 和审计违规均为 0。

## 4. E5-Omni 最终结果

Adapter 结果为五个 seeds 的 mean ± std；Base E5 为确定性结果。R@K、MRR 和 target-over-reference 使用百分数，margin 使用 cosine score 原值。

### 4.1 七模态检索结果

| Mode | Model | R@1 | R@5 | R@10 | MRR | Mean rank | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | Adapter | 1.22 ± 0.35 | 4.12 ± 1.70 | 6.62 ± 2.56 | 3.38 ± 1.04 | 574.72 ± 42.53 | 333.90 ± 71.92 |
| T-only-fullAV | Base E5 | 2.40 | 10.40 | 17.10 | 7.35 | 327.54 | 75.50 |
| V-only | Adapter | 2.22 ± 1.59 | 99.82 ± 0.16 | 99.88 ± 0.04 | 50.89 ± 0.65 | 2.03 ± 0.01 | 2.00 |
| V-only | Base E5 | 0.10 | 99.90 | 99.90 | 49.96 | 2.04 | 2.00 |
| A-only | Adapter | 12.88 ± 1.37 | 73.30 ± 8.54 | 82.16 ± 6.94 | 41.50 ± 4.07 | 18.90 ± 6.70 | 2.20 ± 0.40 |
| A-only | Base E5 | 0.00 | 96.00 | 96.90 | 47.43 | 7.32 | 2.00 |
| V+T | Adapter | 5.94 ± 0.99 | 99.38 ± 0.45 | 99.62 ± 0.31 | 52.50 ± 0.38 | 2.10 ± 0.12 | 2.00 |
| V+T | Base E5 | 0.70 | 99.90 | 100.00 | 50.27 | 2.00 | 2.00 |
| A+T | Adapter | 13.54 ± 0.49 | 72.84 ± 8.21 | 81.14 ± 6.66 | 41.46 ± 3.95 | 17.50 ± 6.52 | 2.20 ± 0.40 |
| A+T | Base E5 | 2.10 | 96.80 | 97.70 | 48.90 | 4.86 | 2.00 |
| V+A | Adapter | 9.36 ± 4.33 | 99.30 ± 1.05 | 99.62 ± 0.51 | 54.08 ± 1.58 | 2.02 ± 0.13 | 2.00 |
| V+A | Base E5 | 0.00 | 99.90 | 99.90 | 49.92 | 2.01 | 2.00 |
| **V+A+T** | **Adapter** | **12.78 ± 2.55** | **99.16 ± 0.84** | **99.56 ± 0.53** | **55.69 ± 0.95** | **2.14 ± 0.32** | **2.00** |
| V+A+T | Base E5 | 0.30 | 100.00 | 100.00 | 50.09 | 2.00 | 2.00 |
| V+T, masked ref | Adapter | 98.28 ± 0.98 | 99.46 ± 0.40 | 99.68 ± 0.26 | 98.80 ± 0.74 | 1.16 ± 0.13 | 1.00 |
| V+T, masked ref | Base E5 | 99.70 | 100.00 | 100.00 | 99.80 | 1.01 | 1.00 |
| V+A+T, masked ref | Adapter | 97.26 ± 1.91 | 99.30 ± 0.75 | 99.60 ± 0.50 | 98.14 ± 1.37 | 1.27 ± 0.34 | 1.00 |
| V+A+T, masked ref | Base E5 | 99.70 | 100.00 | 100.00 | 99.83 | 1.00 | 1.00 |

### 4.2 方向性指标

| Mode | Model | Target over reference | Target-reference margin |
|---|---|---:|---:|
| T-only-fullAV | Adapter | 61.82 ± 0.26 | +0.0066 ± 0.0007 |
| T-only-fullAV | Base E5 | 61.80 | +0.0047 |
| V-only | Adapter | 2.18 ± 1.71 | -0.0328 ± 0.0007 |
| V-only | Base E5 | 0.00 | -0.0394 |
| A-only | Adapter | 18.28 ± 3.94 | -0.0163 ± 0.0013 |
| A-only | Base E5 | 0.00 | -0.0306 |
| V+T | Adapter | 6.06 ± 1.07 | -0.0282 ± 0.0011 |
| V+T | Base E5 | 0.60 | -0.0350 |
| A+T | Adapter | 22.44 ± 3.10 | -0.0136 ± 0.0013 |
| A+T | Base E5 | 2.10 | -0.0260 |
| V+A | Adapter | 9.52 ± 4.60 | -0.0233 ± 0.0014 |
| V+A | Base E5 | 0.00 | -0.0343 |
| **V+A+T** | **Adapter** | **13.20 ± 2.84** | **-0.0208 ± 0.0012** |
| V+A+T | Base E5 | 0.30 | -0.0308 |

R@5/R@10 接近饱和，而 median rank 为 2，说明 target 通常已经被定位到极小候选范围，但 unchanged reference 排在它前面。因此 R@1 与 reference-specific metrics 才是本任务的决定性指标。

### 4.3 预设配对比较

| Comparison | Metric | Difference | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|
| V+A+T - V+T | R@1 | **+6.84pp** | **[+5.18,+8.54]pp** | **0.000250** |
| V+A+T - V+T | R@5 | -0.22pp | [-0.52,+0.10]pp | 0.640 |
| V+A+T - V+T | R@10 | -0.06pp | [-0.28,+0.16]pp | 1.000 |
| V+A+T - V+T | Target over reference | **+7.14pp** | **[+5.48,+8.86]pp** | **0.000250** |
| V+A+T - V+T | Target-reference margin | **+0.00739** | **[+0.00640,+0.00841]** | **0.000250** |
| V+A+T - V+T | Reciprocal rank | **+0.03188** | **[+0.02335,+0.04070]** | **0.000250** |
| Masked-ref V+T - with-ref V+T | R@1 | **+92.34pp** | **[+91.06,+93.56]pp** | **0.000250** |
| Masked-ref V+A+T - with-ref V+A+T | R@1 | **+84.48pp** | **[+82.60,+86.24]pp** | **0.000250** |
| V+A+T - V+A | R@1 | +3.42pp | [+2.34,+4.54]pp | 0.000250 |
| A+T - A-only | R@1 | +0.66pp | [-0.50,+1.84]pp | 0.290 |

### 4.4 Reference 错误归因

| 项目 | 数量 / 比例 |
|---|---:|
| 五 seeds 的 V+A+T top-1 errors | 4,361 |
| Top-1 被 own reference 占据 | 4,317 |
| Own-reference error share | **98.99%** |
| 其他 top-1 errors | 44 |

结论：E5 并不是普遍找不到 target，而是几乎总在最后一步把修改前的 reference 排在修改后的 target 前面。

## 5. ImageBind-Huge 最终结果

ImageBind 不训练 adapter，使用预先固定的零样本等权模态算术。下表 R@5/R@10 为 with-reference 条件。

### 5.1 七模态与 reference masking

| Mode | With-ref R@1 | Masked-ref R@1 | Masking gain | R@5 | R@10 | Target over reference | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | 2.10 | 3.20 | +1.10pp | 9.60 | 15.20 | 52.20 | +0.00433 |
| V-only | 0.90 | 99.60 | +98.70pp | 99.90 | 100.00 | 0.00 | -0.01323 |
| A-only | 0.00 | 94.20 | +94.20pp | 97.30 | 98.30 | 0.00 | -0.07324 |
| **V+T** | **11.70** | **99.30** | **+87.60pp** | **99.80** | **99.80** | **10.80** | **-0.00767** |
| A+T | 4.10 | 92.90 | +88.80pp | 97.60 | 98.50 | 4.10 | -0.04377 |
| V+A | 0.00 | 98.10 | +98.10pp | 99.20 | 99.60 | 0.00 | -0.03267 |
| **V+A+T** | **2.50** | **98.50** | **+96.00pp** | **99.40** | **99.60** | **2.50** | **-0.02334** |

### 5.2 配对统计

| Comparison | Metric | Difference | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|
| V+A+T - V+T | R@1 | **-9.20pp** | **[-11.30,-7.10]pp** | **0.000200** |
| V+A+T - V+T | Target-reference margin | **-0.01567** | **[-0.01765,-0.01383]** | **0.000200** |
| Masked-ref V+T - with-ref V+T | R@1 | **+87.60pp** | **[+85.50,+89.60]pp** | **0.000200** |
| Masked-ref V+A+T - with-ref V+A+T | R@1 | **+96.00pp** | **[+94.70,+97.20]pp** | **0.000200** |

ImageBind 支持 reference confusion 的跨模型结论，但不支持“直接加入 audio 一定提高 Recall”。在零样本等权融合下，audio 造成显著模态干扰；任务适配后的 E5 才能把音频转化为正向的方向性增益。

## 6. 跨模型核心对照

| Model | Mode | With-ref R@1 | Masked-ref R@1 | Masking gain | Audio gain |
|---|---|---:|---:|---:|---:|
| Base E5 | V+T | 0.70 | 99.70 | +99.00pp | - |
| Base E5 | V+A+T | 0.30 | 99.70 | +99.40pp | -0.40pp |
| E5 Adapter | V+T | 5.94 ± 0.99 | 98.28 ± 0.98 | +92.34pp | - |
| E5 Adapter | V+A+T | 12.78 ± 2.55 | 97.26 ± 1.91 | +84.48pp | **+6.84pp** |
| ImageBind zero-shot | V+T | 11.70 | 99.30 | +87.60pp | - |
| ImageBind zero-shot | V+A+T | 2.50 | 98.50 | +96.00pp | **-9.20pp** |

最稳的论文结论是：

1. **Reference confusion 跨模型成立。** Base E5、E5 Adapter 和 ImageBind 在保留 own reference 时均显著失败，mask 后均接近饱和。
2. **Audio 收益依赖表示与任务适配。** E5 Adapter 的 audio 增益显著；ImageBind 的简单零样本融合反而下降。
3. **高 R@5/R@10 不能替代方向性评估。** 模型通常找到了正确视觉语境，却没有理解修改前后状态。

## 7. OmniCVR 跨 Benchmark 诊断

OmniCVR 已经把 source 放入 gallery。该实验的贡献不是“加入 source”，而是在外部 benchmark 上用相同 cache 精确 mask 当前 query 的 source，检验 source-target confusion 是否只存在于 Audio-CVR。

### 7.1 审计设置

| 项目 | 数值 |
|---|---:|
| 原始 audio-centered queries | 1,000 |
| 有效 queries | 995 |
| 原始 gallery | 2,000 |
| 统一排除的解码失败视频 | 6 |
| Effective gallery | 1,994 |
| 统一排除的受影响 queries | 5 |
| Seeds | 5 |
| Sample IDs identical | true |
| Audit violations | 0 |

### 7.2 E5 跨 benchmark 结果

| Mode | Model | R@1 | R@5 | R@10 | MRR | Target over reference | Margin |
|---|---|---:|---:|---:|---:|---:|---:|
| V+A+T with-ref | Adapter | 0.12 ± 0.04 | 30.71 ± 2.30 | 44.56 ± 2.60 | 15.29 ± 0.76 | 0.12 ± 0.04 | -0.2304 |
| V+A+T with-ref | Base E5 | 0.00 | 41.71 | 57.89 | 19.16 | 0.00 | -0.2366 |
| V+A+T masked-ref | Adapter | 14.21 ± 0.85 | 34.35 ± 2.40 | 46.41 ± 2.73 | 24.52 ± 1.27 | - | - |
| V+A+T masked-ref | Base E5 | 17.39 | 45.33 | 60.50 | 31.00 | - | - |
| V+T with-ref | Adapter | 0.00 | 29.17 ± 1.91 | 43.84 ± 2.07 | 14.68 ± 0.78 | 0.04 ± 0.05 | -0.2584 |
| V+T with-ref | Base E5 | 0.00 | 36.58 | 54.57 | 17.54 | 0.00 | -0.2633 |
| V+T masked-ref | Adapter | 12.86 ± 1.16 | 33.47 ± 2.40 | 45.95 ± 1.96 | 23.33 ± 1.43 | - | - |
| V+T masked-ref | Base E5 | 15.18 | 41.21 | 56.68 | 28.01 | - | - |

### 7.3 OmniCVR 配对检验

| Comparison | Metric | Difference | 95% bootstrap CI | Holm-adjusted p | 结论 |
|---|---|---:|---:|---:|---|
| Masked-ref V+A+T - with-ref | R@1 | **+14.09pp** | **[+12.20,+16.02]pp** | **<0.001** | Reference effect 显著 |
| Masked-ref V+T - with-ref | R@1 | **+12.86pp** | **[+11.10,+14.71]pp** | **<0.001** | Reference effect 显著 |
| V+A+T - V+T | R@1 | +0.12pp | [0.00,+0.32]pp | 0.2449 | 不显著 |
| V+A+T - V+T | R@5 | +1.55pp | [-0.24,+3.34]pp | 0.0904 | 不显著 |
| V+A+T - V+T | Target over reference | +0.08pp | [-0.08,+0.28]pp | 1.0000 | 不显著 |
| V+A+T - V+T | Target-reference margin | **+0.0281** | **[+0.0252,+0.0309]** | **<0.001** | 分差显著改善 |
| V+A+T - V+T | Reciprocal rank | +0.0061 | [+0.00004,+0.0122] | 0.0477 | 边界显著 |

OmniCVR 只支持两个有边界的结论：

1. source/reference confusion 在独立 benchmark 上同样存在；
2. audio 改善平均 target-reference margin，但没有带来显著 R@1 增益。

Audio-CVR adapter 在 OmniCVR 的 masked-reference V+A+T R@1 为 14.21%，低于 Base E5 的 17.39%，因此不能声称 adapter 具有跨 benchmark 性能优势。

## 8. 论文可写结论与边界

### 8.1 可以主张

| 主张 | 直接证据 |
|---|---|
| Aggregate Recall 会掩盖 source-target directional failure | Test1000 中 E5 与 ImageBind mask own reference 后 R@1 提升 84.48-99.40pp |
| Reference confusion 不是单一 adapter 的偶然现象 | Base E5、E5 Adapter、ImageBind 和 OmniCVR 均复现 |
| 任务适配后的 E5 能利用 audio | V+A+T 比 V+T 提升 6.84pp R@1，CI 不含 0，Holm p=0.000250 |
| Audio 改善 E5 的方向判别 | Target-over-reference +7.14pp；margin +0.00739 |
| 简单多模态融合不保证获益 | ImageBind V+A+T 比 V+T 下降 9.20pp |
| 自动构造流程可扩展到 1,000-query benchmark | Test1000 冻结、去重/泄漏/missing media 均为 0 |

### 8.2 必须披露

| 限制 | 准确表述 |
|---|---|
| 人工核验 | Full Test1000 不是 fully human-validated gold set；Core150 仅有两位作者共同完成的随机 10 条定性抽查 |
| 重复审核 | 78/78 完成；exact-decision 79.49%，field-level 85.64%；属于 post-freeze、audit-only 的同模型稳定性检查 |
| 类型范围 | 主集为 829 sound event + 171 music，不覆盖 speech |
| 数据分布 | AVATAR family 占 51.0%，数据源仍不均衡 |
| 训练规模 | Adapter 仅使用 65 个独立 source pair 和 24 个 verified inverse |
| 模型结论 | Audio 的 top-1 收益只在 task-adapted E5 上成立，不可外推到所有模型 |
| OmniCVR | Audio 的 R@1 增益不显著；该实验主要验证 reference-specific diagnosis |
| Gallery | 主 gallery 为 target + reference；本轮不声称完整验证 strict local/typed hard-negative benchmark |

### 8.3 不得声称

```text
我们首次把 source/reference 放入 gallery
audio 对所有 composed video retrieval 模型都提高 Recall
adapter 已经解决 reference confusion
Full Test1000 是完整人工 gold annotation
ImageBind 的下降说明 audio 不重要
OmniCVR 证明 adapter 能跨 benchmark 泛化
```

## 9. 最终论文结果摘要

> On the frozen 1,000-query Audio-CVR benchmark, a low-rank adapter on frozen E5-Omni obtains 12.78% R@1 with visual, audio, and text input, compared with 5.94% without audio. The paired gain is 6.84 points (95% CI: 5.18-8.54; Holm-adjusted p<0.001), while target-over-reference accuracy improves by 7.14 points. However, masking only the query-specific unchanged reference raises R@1 to 97.26%. Across five seeds, 4,317 of 4,361 top-1 errors are attributable to the own reference. ImageBind independently reproduces the reference effect, although its zero-shot equal-weight audio fusion reduces R@1, showing that audio gains depend on representation and task adaptation. On 995 valid OmniCVR queries, exact source masking also raises R@1 by 14.09 points for the adapter, confirming that source-target confusion extends beyond our benchmark.

## 10. 最终证据路径

### 10.1 Test1000 与最终 E5/ImageBind

```text
paper/evidence/server_final1000/final_status.json
paper/evidence/server_final1000/dataset/frozen_benchmark_manifest.json
paper/evidence/server_final1000/dataset/frozen_benchmark.sha256
paper/evidence/server_final1000/dataset/dedup_audit.json
paper/evidence/server_final1000/dataset/leakage_audit.json
paper/evidence/server_final1000/adapter_training/adapter_config_seed13.json
paper/evidence/server_final1000/adapter_training/validation_selection.json
paper/evidence/server_final1000/adapter_training/inverse_summary.json
paper/evidence/server_final1000/e5_statistics/test_main_comparison.md
paper/evidence/server_final1000/e5_statistics/paired_comparisons.json
paper/evidence/server_final1000/e5_statistics/error_breakdown.json
paper/evidence/server_final1000/imagebind_statistics/paper_results.md
paper/evidence/server_final1000/imagebind_statistics/paired_comparisons.json
paper/evidence/server_final1000/audits/common_query_audit.json
```

服务器最终 run：

```text
runs/audio_cvr_e5_imagebind_final1000_20260723_143500
Git HEAD = 6ede6560e8a6c6af24746ee4c6729933bca97ad8
```

### 10.2 OmniCVR

```text
doc/omnicvr_reference_cross_benchmark_results_20260721.md
C:/Users/29785/Desktop/research/runs/
omnicvr_reference_diagnostics_paper_results_20260721/
```
