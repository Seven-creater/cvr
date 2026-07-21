# Audio-CVR：AAAI 可用实验结果与证据总览

> 更新日期：2026-07-21
>
> 本文只记录可以进入论文的冻结实验。早期 30/37/68-query pilot 不进入主表。
>
> 论文定位：Audio-CVR 任务与自动构造方法、reference-aware benchmark，以及 adapter-only baseline。

## 1. 论文核心问题

Audio-CVR 的输入和目标是：

```text
query  = reference video/audio + audio edit text
target = target video/audio
```

任务要求模型在视觉语境保持时，根据有方向的声音修改找到 target：

```text
reference 不满足 edit
target 满足 edit
静音画面不能稳定确定 target
样本不能退化为 transcript / ASR matching
```

核心困难不是从无关视频中召回相似内容，而是把已经满足 edit 的 target 排在尚未发生 edit 的 reference 前面。reference 因而是天然的反事实负例。

## 2. 论文三项贡献

1. **Audio-primary directional CVR**：定义视觉语境保持条件下的方向性声音组合检索，并控制视觉与 ASR 捷径。
2. **多阶段多模态自动构造方法**：从自然视频执行 source-aware 配对、audio-only edit 构造与验证、muted-video shortcut rejection、full-AV consistency、anti-ASR、重复审核和 source-disjoint 切分。
3. **Reference-aware benchmark**：强制加入 unchanged reference，并通过有无 reference、七模态消融、类型化困难负例和方向性指标测量模型是否真正使用 audio edit。

E5-Omni、低秩 adapter 和 e5-compatible recipe 都是 baseline，不包装成原创贡献。

## 3. 冻结 Benchmark

冻结路径：

```text
runs/audiocvr_benchmark150_auto_20260720_164327/benchmark_v1_final150_val28
test SHA256 = f4b22e25e1f1262d488ff5474fdae9511301919611b42b9cc89f55c3aa633fd6
```

### 3.1 Split

| Split | Records | Sound event | Music | Speech |
|---|---:|---:|---:|---:|
| Train pool | 606 | 48 | 17 | 541 |
| Primary train subset | **65** | **48** | **17** | **0** |
| Validation | **28** | **20** | **8** | **0** |
| Test | **150** | **120** | **30** | **0** |

主训练只使用 65 条与 test subtype 对齐的 non-speech forward pair。train/val/test 的 source、pair 和 inverse group 泄漏均为 0。

### 3.2 自动审核

- Benchmark 为 `automatically curated and model-verified`，不是 human-validated。
- 430 条候选经过独立 Omni benchmark audit；46 条执行重复审核。
- 重复审核 exact decision agreement 为 97.8%。
- Speech 候选因 `full_av_not_required`、`audio_only_solvability_high` 和 ASR 风险被隔离，不进入主 benchmark。
- 冻结 test 的来源为 avatar 135、vggsound 7、worldsense 8。

### 3.3 Gallery

正式测试使用固定 1,000-item gallery，包含：

```text
target positives
mandatory reference negatives
visual hard negatives
audio hard negatives
ASR hard negatives
random distractors
```

有无 reference 的评估复用同一 cache，只精确 mask `reference_negative`。20 个 variant/seed/mode 审计项全部通过，`violation_count=0`。

当前正式 test 没有 strict `local_same_source` negative。论文不得声称已完成 strict-local 主实验。

## 4. Baseline 与训练协议

```text
backbone: frozen E5-Omni-7B
embedding dimension: 3584
adapter: identity-initialized low-rank residual projections
adapter rank: 32
trainable parameters: 688,131
steps: 400
learning rate: 1e-3
batch size: 8
final seeds: 13, 23, 42, 71, 101
LoRA: disabled
AudioDelta-specific auxiliary losses: disabled
```

模型选择只读取 val28。候选 rank、steps 和 learning rate 经过 coarse search，前四项使用 3 个 validation seeds 复核；one-standard-error 规则最终选择 `rank=32, steps=400, lr=1e-3`。test150 在配置冻结后才读取。

共核验 38 条训练 loss curve，`NaN/Inf=0`。

## 5. 正式主结果：Forward-only Adapter

### 5.1 七模态结果

| Mode | R@1 mean ± std | R@5 | R@10 | Target beats reference | Gap |
|---|---:|---:|---:|---:|---:|
| T-only-fullAV | 2.27 ± 1.00 | 6.13 | 11.87 | 62.00 | +0.0059 |
| V-only | 8.80 ± 3.36 | 100.00 | 100.00 | 8.93 | -0.0198 |
| A-only | 18.40 ± 3.28 | 84.27 | 92.53 | 22.00 | -0.0167 |
| **V+T** | **11.33 ± 2.83** | **100.00** | **100.00** | **11.33 ± 2.83** | **-0.0172** |
| A+T | 23.20 ± 1.95 | 86.27 | 93.87 | 28.40 | -0.0117 |
| V+A | 19.20 ± 4.45 | 100.00 | 100.00 | 19.73 | -0.0160 |
| **V+A+T** | **22.93 ± 5.31** | **100.00** | **100.00** | **23.20 ± 5.12** | **-0.0128** |
| Late fusion | 21.33 ± 4.20 | 92.40 | 96.13 | 24.26 | -0.0129 |

百分数表中 gap 保持 cosine score 原值。

### 5.2 Audio necessity

预设主比较 `V+A+T - V+T`：

| Metric | Difference | 95% paired bootstrap CI | Randomization p | Holm p |
|---|---:|---:|---:|---:|
| R@1 | **+11.60pp** | **[+6.27,+17.07]pp** | 0.000050 | **0.000300** |
| Target beats reference | **+11.87pp** | **[+6.53,+17.47]pp** | 0.000100 | **0.000600** |
| Target-reference gap | **+0.004394** | **[+0.002110,+0.006701]** | 0.000250 | **0.001000** |
| MRR | +0.058111 | [+0.031331,+0.085556] | 0.000100 | 0.000400 |

可用于摘要的结论：在相同 query、相同 gallery 和五种子下，audio 显著改善 top-rank retrieval 与 reference-target 方向判别。

### 5.3 Base E5 与 Adapter

| Method | Mode | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| Base E5 | V+A+T | 1.33 | 100.00 | 100.00 |
| Low-rank adapter | V+A+T | **22.93 ± 5.31** | 100.00 | 100.00 |

Base E5 已能把 target 放入很小候选范围，但几乎总把 reference 排在前面。adapter 提升的是任务适配和方向排序，而不是普通召回。

## 6. Reference Counterfactual 实验

| Mode | With reference R@1 | Without reference R@1 | Inflation after removal |
|---|---:|---:|---:|
| V+T | 11.33 | 99.60 | **+88.27pp** |
| V+A+T | 22.93 | 99.47 | **+76.53pp** |

`V+A+T_no_ref - V+A+T` 的 R@1 置信区间为 `[+70.80,+82.00]pp`，Holm p=0.000300。forward-only 的 578 个跨种子 top-1 错误全部由 `reference_negative` 引起。

这是论文最有辨识度的实验：如果删除 reference，接近 100% 的 R@1 会掩盖模型对 edit direction 的失败。

## 7. Hard-negative 诊断

Forward-only、V+A+T 下：

| Negative | Positive beats negative |
|---|---:|
| reference_negative | 23.20%（五种子均值） |
| visual_hard | 100% |
| audio_hard | 100% |
| asr_hard | 100% |

这说明现有 typed hard negatives 偏容易，reference 才是主要难例。论文不能把 visual/audio/asr hard negative 的饱和结果写成模型已经解决复杂困难负例，而应把它作为 benchmark 后续增强方向。

## 8. Verified Bidirectional Augmentation：负消融

```text
65 forward pairs
24 Omni-accepted inverse records
= 89 directional training instances
inverse acceptance rate = 36.9%
```

| Variant | V+A+T R@1 | Target beats ref | Gap |
|---|---:|---:|---:|
| Forward-only | **22.93 ± 5.31** | **23.20 ± 5.12** | -0.0128 ± 0.0015 |
| Forward+Bidir | 18.93 ± 2.72 | 19.47 ± 3.30 | **-0.0094 ± 0.0013** |

`Forward+Bidir - Forward-only`：

| Metric | Difference | 95% CI | Holm p |
|---|---:|---:|---:|
| R@1 | **-4.00pp** | **[-7.60,-0.53]pp** | **0.0373** |
| Target beats reference | -3.73pp | [-7.33,-0.13]pp | 0.0470 |
| Gap | +0.003433 | [+0.001721,+0.005076] | 0.000150 |

Inverse augmentation 改善平均分差，却显著降低 R@1 和 target-beats-reference。论文主方法因此使用 forward-only；inverse 只作为负消融，说明在极少样本下增加相关方向实例不能替代新的独立 source diversity。

## 9. 可以写入论文的核心发现

1. 自动多阶段流程可以构造满足 audio-primary 条件的自然视频 triplets，并以模型重复审核量化一致性。
2. 原始 E5-Omni embedding 几乎不能解决 target-reference 方向排序；轻量 adapter 将 V+A+T R@1 从 1.33% 提升到 22.93%。
3. Audio 在严格 reference-aware protocol 下带来显著的 +11.60pp R@1 和 +11.87pp target-over-reference 增益。
4. 删除 reference 会把 R@1 从 22.93% 虚高到 99.47%；reference 是决定 benchmark 难度的核心反事实负例。
5. Typed visual/audio/asr negatives 目前过容易，不能替代 mandatory reference。
6. Verified inverse augmentation 没有改善最终排序，独立 source diversity 比机械增加方向实例更重要。

## 10. 必须披露的限制

- 独立训练 pair 只有 65 条；这是 few-shot baseline，不是大规模训练。
- Test 150 中 avatar 占 90%，跨数据集泛化证据有限。
- 当前 benchmark 是自动构造和模型复核，不是 human-validated。
- 正式 test 不含 speech，只覆盖 sound event 和 music。
- 正式 gallery 没有 strict local_same_source negatives。
- R@5/R@10 接近饱和，主分析应聚焦 R@1、MRR、target-over-reference 和 reference-induced drop。
- 当前导出的 subtype 评估字段把测试记录统一归为 `audio_event`，因此论文暂不报告 sound-event/music 的分项模型性能。

## 11. 论文结果段落候选

> On the frozen 150-query source-disjoint test set, the forward-only low-rank adapter obtains 22.9% R@1 with full audiovisual input, compared with 11.3% when audio is removed. The paired 11.6-point gain is statistically significant (95% CI: 6.3--17.1; Holm-adjusted p<0.001), and target-over-reference accuracy improves by 11.9 points. In contrast, removing the unchanged reference raises R@1 to 99.5%. These results show that easy galleries substantially overestimate composed retrieval performance and that the unchanged reference is the decisive counterfactual for measuring directional audio understanding.

## 12. 最终证据路径

本地：

```text
C:/Users/29785/Desktop/research/runs/fewshot_bidir_results_final_20260721/fewshot_bidir_results
```

核心文件：

```text
status.json
paper_results.md
validation_selection.json
loss_audit.json
reference_exclusion_audit.json
statistics_forward_only/test_main_comparison.md
statistics_forward_only/paired_comparisons.md
statistics_variant_comparison/paired_comparisons.md
```
