# OmniCVR 跨 Benchmark Reference 诊断结果

> 状态：`COMPLETE`  
> 日期：2026-07-21  
> 用途：AAAI 论文的 cross-benchmark diagnostic，验证 reference-induced difficulty 是否超出 Audio-CVR 自建 benchmark。  
> 非用途：不作为 Audio-CVR audio necessity 主结果，不作为 adapter 跨域性能提升证据。

## 1. 研究问题

该实验只回答一个问题：

```text
当 query 对应的 reference/source video 也进入候选库时，
模型是否会因为 reference 与 query 高度相似，
而无法把已经满足修改文本的 target 排在 reference 前面？
```

我们在 OmniCVR 上进行精确控制：query、模型、embedding cache 和所有其他候选保持不变，只在 no-reference 条件下 mask 当前 query 对应的 reference。

OmniCVR 原协议已经把 source video 放入 gallery，并包含 source-local distractors。本实验的贡献不是加入 reference，而是把 source-target confusion 单独隔离出来，补充 aggregate R@K 无法提供的 reference-specific 诊断。

## 2. 数据与审计

| 项目 | 数值 |
|---|---:|
| 原始 query | 1,000 |
| 有效 query | 995 |
| 官方 gallery | 2,000 |
| 统一 mask 的解码失败视频 | 6 |
| 有效 gallery | 1,994 |
| Final seeds | 5 |
| Modes | 4 |
| Sample IDs identical | true |
| Audit violations | 0 |

五个 query 因其 reference 或 target 使用了解码失败视频而在所有模式中统一排除。六个失败视频在 V+A+T 和 V+T gallery 中使用相同排除集合，因此模态比较和 reference removal 没有候选身份偏差。

## 3. 五种子结果

| Mode | Model | R@1 | R@5 | R@10 | MRR | Target beats reference | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| V+A+T with-ref | Adapter | 0.12 ± 0.04 | 30.71 ± 2.30 | 44.56 ± 2.60 | 15.29 ± 0.76 | 0.12 ± 0.04 | -0.2304 |
| V+A+T with-ref | Base E5 | 0.00 | 41.71 | 57.89 | 19.16 | 0.00 | -0.2366 |
| V+A+T no-ref | Adapter | 14.21 ± 0.85 | 34.35 ± 2.40 | 46.41 ± 2.73 | 24.52 ± 1.27 | - | - |
| V+A+T no-ref | Base E5 | 17.39 | 45.33 | 60.50 | 31.00 | - | - |
| V+T with-ref | Adapter | 0.00 | 29.17 ± 1.91 | 43.84 ± 2.07 | 14.68 ± 0.78 | 0.04 ± 0.05 | -0.2584 |
| V+T with-ref | Base E5 | 0.00 | 36.58 | 54.57 | 17.54 | 0.00 | -0.2633 |
| V+T no-ref | Adapter | 12.86 ± 1.16 | 33.47 ± 2.40 | 45.95 ± 1.96 | 23.33 ± 1.43 | - | - |
| V+T no-ref | Base E5 | 15.18 | 41.21 | 56.68 | 28.01 | - | - |

表中 R@K、MRR 和 target-beats-reference 使用百分数；gap 使用 cosine score 原值。no-reference 条件不报告 target-beats-reference 和 gap，因为 reference 已从有效检索库中移除。

## 4. Reference Removal 统计检验

| Comparison | Metric | Difference | 95% paired bootstrap CI | Holm p |
|---|---|---:|---:|---:|
| V+A+T no-ref - with-ref | R@1 | **+14.09pp** | **[+12.20,+16.02]pp** | **<0.001** |
| V+A+T no-ref - with-ref | R@5 | +3.64pp | [+3.04,+4.28]pp | <0.001 |
| V+A+T no-ref - with-ref | MRR | +9.23pp | [+8.25,+10.22]pp | <0.001 |
| V+T no-ref - with-ref | R@1 | **+12.86pp** | **[+11.10,+14.71]pp** | **<0.001** |
| V+T no-ref - with-ref | R@5 | +4.30pp | [+3.58,+5.05]pp | <0.001 |
| V+T no-ref - with-ref | MRR | +8.65pp | [+7.73,+9.58]pp | <0.001 |

reference removal 对五个 seed 的 R@1 McNemar 检验也全部显著。它不是重新采样 gallery，而是对同一 cache 的 query-specific reference 做精确 mask。

## 5. Audio 比较

`V+A+T - V+T` 的配对结果：

| Metric | Difference | 95% CI | Holm p | 结论 |
|---|---:|---:|---:|---|
| R@1 | +0.12pp | [0.00,+0.32]pp | 0.2449 | 不显著 |
| R@5 | +1.55pp | [-0.24,+3.34]pp | 0.0904 | 不显著 |
| R@10 | +0.72pp | [-1.23,+2.67]pp | 0.4825 | 不显著 |
| Target beats reference | +0.08pp | [-0.08,+0.28]pp | 1.0000 | 不显著 |
| Target-reference gap | **+0.0281** | **[+0.0252,+0.0309]** | **<0.001** | 显著改善分差 |
| Reciprocal rank | +0.0061 | [+0.00004,+0.0122] | 0.0477 | 边界显著 |

因此 OmniCVR 不能用于声称 audio 显著提高 R@1。准确表述是：audio 稳定改善 target 相对 reference 的平均 score margin，但该变化没有转化为显著的 top-1 命中提升。

## 6. Adapter 跨域结果

在 no-reference 的 V+A+T 条件下：

```text
Audio-CVR adapter R@1 = 14.21%
Base E5 R@1           = 17.39%
```

adapter 没有超过 Base E5，说明由少量 Audio-CVR pair 训练的适配器具有任务特定性。论文不能把该实验写成模型迁移成功；它反而说明本文贡献应落在任务、构造方法和 reference-aware protocol，而不是 adapter architecture。

## 7. 论文可用结论

可以写：

1. reference-induced difficulty 不只存在于 Audio-CVR 自建数据，在 OmniCVR 上同样显著。
2. 去掉 reference 会系统性抬高 R@1，改变 benchmark 实际测量的问题。
3. Base E5 和 adapter 都受到该现象影响，因此它不是某个 adapter 的偶然失败。
4. 外部 benchmark 结果支持把 unchanged reference 作为独立 counterfactual 对象持续报告，而不是只把它混在 aggregate R@K 中。

不能写：

1. Audio 在 OmniCVR 上显著提高 R@1。
2. Audio-CVR adapter 能跨 benchmark 泛化。
3. 14.21% no-reference R@1 是本文模型优于既有工作的成绩。
4. 解码失败视频被删除后没有任何评估影响；准确说法是六个视频被所有模式统一 mask，五个相关 query 被统一排除。

## 8. 与 Audio-CVR 主实验的关系

| Evidence | Audio-CVR test150 | OmniCVR test995 |
|---|---:|---:|
| Audio R@1 gain | +11.60pp，显著 | +0.12pp，不显著 |
| Reference removal R@1 gain | +76.53pp | +14.09pp |
| Adapter exceeds Base E5 | 是 | 否 |
| 论文角色 | 主实验 | 外部诊断 |

两项实验共同证明 reference-aware evaluation 的必要性；只有 Audio-CVR test150 支持 audio-primary task 中的 audio necessity。

## 9. 证据路径

压缩包：

```text
C:/Users/29785/Desktop/research/runs/omnicvr_reference_diagnostics_paper_results_20260721.tar.gz
SHA256: 8AAEBC6A7986BE5D6ADD169F67A4ECAE416F3BC5D77417CD7AE3297AB0919018
```

解压目录：

```text
C:/Users/29785/Desktop/research/runs/omnicvr_reference_diagnostics_paper_results_20260721
```

核心文件：

```text
status.json
statistics/audit.json
statistics/test_main_comparison.md
statistics/test_main_mean_std.json
statistics/paired_comparisons.json
statistics/audio_gain_summary.md
statistics/per_seed_results.json
cache_V_A_T/summary.json
cache_V_T/summary.json
```
