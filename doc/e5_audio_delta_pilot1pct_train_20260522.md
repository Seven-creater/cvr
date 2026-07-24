# E5 AudioDelta 1% Pilot 训练与诊断记录

日期：2026-05-22

## 1. 实验目的

本次实验不是为了得到最终模型结果，而是为了用 1% B-line Audio-CVR 数据把训练链路和评估协议跑通，并查清楚当前 E5-Omni embedding 在 AudioDelta-CVR 任务上的真实问题。

核心问题是：

```text
query = reference video/audio + audio edit_text
target = target video/audio
```

模型不仅要找到视觉/语境相近的视频，还要理解 target 相比 reference 是否按照 `edit_text` 发生了正确的音频变化。

本次 pilot 重点验证三件事：

1. B-line 数据能否支持 E5 训练流程。
2. `e5_omni_recipe` adapter 是否能稳定训练。
3. 加入 reference negative 后，模型是否真的能区分 `reference` 和 `target`。

最终发现非常关键：普通随机 gallery 会严重高估 Base E5；只有把 reference clip 加入 gallery，才能暴露 AudioDelta-CVR 的方向性难点。

---

## 2. 数据来源

### 2.1 B-line 1% Pilot 数据构造

数据来自 B-line Audio-CVR 构造流程，使用 6-9 秒切片，默认 8 秒。B-line 当前定义为：

```text
视觉上下文尽量保持，声音内容发生变化；
edit_text 只描述声音变化；
样本通过 audio-only blind review + video-only shortcut check + full AV consistency。
```

1% pilot 的数据构造结果如下：

| 指标 | 数值 |
|---|---:|
| annotation clips | 7,180 |
| B candidates | 2,777 |
| B ranked | 1,546 |
| B accepted/exported | 287 |
| accepted rate | 18.6% |

Accepted 子类型分布：

| 子类型 | 数量 | 占比 |
|---|---:|---:|
| sound_event | 180 | 62.7% |
| music | 54 | 18.8% |
| speech_topic_in_video_context | 53 | 18.5% |

Tier 分布：

| tier | 数量 |
|---|---:|
| main | 286 |
| extended | 1 |
| diagnostic | 0 |

这个 tier 分布说明当前 blind review 很严格，但 tier assignment 可能仍偏宽：几乎所有 accepted 都进入 main。后续全量构造后需要继续复查 `B-main / B-extended / B-diagnostic` 的阈值和诊断标签。

### 2.2 训练与评估样本

本次训练没有使用全部 287 条，而是做小规模 smoke/pilot：

| 用途 | 数量 |
|---|---:|
| train records | 64 |
| eval queries | 30 |
| gallery positive targets | 30 |
| gallery reference negatives | 30 |
| random distractors | 940 |
| final gallery size | 1,000 |

注意：这里的 `gallery=1000` 是 pilot-only 诊断设置。它不是最终 benchmark protocol。后续全量数据完成后，评估应使用正式 split 和真实大 gallery。

---

## 3. 训练 Recipe

本轮使用 `e5_omni_recipe`，目标是先对齐 E5-Omni 风格的通用 embedding 训练，不加入 AudioDelta 专属 loss。

### 3.1 开启项

`e5_omni_recipe` 当前默认开启：

1. Modality-aware temperature calibration
2. Negative curriculum + false-negative debiasing
3. Batch whitening
4. Query-target covariance / CORAL alignment
5. Masked DCL contrastive objective

### 3.2 默认关闭项

为了建立干净基线，当前默认关闭以下 AudioDelta 扩展：

| 模块 | 状态 |
|---|---|
| L_delta | off |
| L_ref | off |
| L_hn margin hard-negative loss | off |
| edit-type-aware delta loss | off |
| local segment matching | off |
| multi-positive | off |
| memory bank | off |
| LoRA | off |

因此，本轮 adapter 的目标不是专门学习 `target_audio - reference_audio ≈ edit_text`，而是检验 E5-Omni recipe 在当前任务上能否带来基础改进。

### 3.3 训练配置

| 参数 | 值 |
|---|---|
| model embedding dim | 3584 |
| train records | 64 |
| eval records | 30 |
| steps | 80 |
| batch size | 8 |
| learning rate | 1e-3 |
| profile | e5_omni_recipe |
| device | CUDA |

训练链路已完整跑通：

```text
prepare -> cache-embeddings -> train-adapter -> eval
```

---

## 4. 评估协议演进

### 4.1 初始随机 gallery 评估

最开始的 pilot 使用：

```text
30 positive targets + 970 random distractors
```

结果是：

| 方法 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Base E5 | 100.00% | 100.00% | 100.00% |
| Adapter | 76.67% | 100.00% | 100.00% |

这个结果一开始看起来说明 Base E5 已经非常强。但后续诊断证明：这个协议缺少最关键的负例，也就是 query 自己的 reference video。

### 4.2 为什么必须加入 reference negative

在 CVR 中，reference 是“尚未发生 edit 的原视频”，target 才是“发生 edit 后的视频”。如果 gallery 里没有 reference，模型只要找到同源/同场景/同语境的视频，就可能得到很高分。

但真正的 AudioDelta-CVR 需要满足：

```text
score(query, target) > score(query, reference)
```

因此新增 pilot-only reference-negative gallery：

```text
30 positive targets
+ 30 reference negatives
+ 940 random distractors
= 1000 gallery videos
```

这个协议可以直接测试模型是否理解 edit 的方向性。

### 4.3 reuse-cache bug 与修复

第一次使用 `--reuse-cache-from` 构建 reference-negative gallery 时，出现了异常结果：

```text
Base E5 R@1/R@5/R@10 = 0
Adapter R@1/R@5/R@10 = 0
```

这个结果不合理。排查后发现是代码 bug：

```text
positive_gallery_index / reference_gallery_index
被按 gallery shuffle 后的出现顺序记录，
而不是按 eval query 顺序记录。
```

也就是说，query 0 的正确 target 可能被错误地指向 query 17 的 target。修复 commit：

```text
eb0180b Fix reused gallery index alignment
```

修复后才得到可信结果。

---

## 5. 修复后的核心结果

最终可信 run：

```text
TRAIN_RUN = runs/e5_pilot1pct_refneg_reuse_fixed_20260522_233802
```

Gallery 设置：

```text
30 target positives + 30 reference negatives + 940 random distractors
```

整体检索结果：

| 方法 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Base E5 | 6.67% | 100.00% | 100.00% |
| Adapter | 40.00% | 100.00% | 100.00% |

这个结果非常重要：

1. 加入 reference negative 后，Base E5 从随机 gallery 的 R@1=100% 掉到 6.67%。
2. Adapter 把 R@1 从 6.67% 提升到 40.00%。
3. R@5/R@10 仍为 100%，说明 target 通常在前几名，问题主要是 target 和 reference 的第一名排序。

这说明当前任务不是“找不到 target”，而是“target 与 reference 太接近，模型常把未编辑的 reference 排在 target 前面”。

---

## 6. 子类型结果

| 子类型 | count | Base R@1 | Adapter R@1 |
|---|---:|---:|---:|
| audio_event | 25 | 8.00% | 36.00% |
| speech | 5 | 0.00% | 60.00% |

解读：

- `audio_event` 是主要样本类型，也是当前最稳定的分析对象。
- `speech` 只有 5 条，Adapter R@1=60% 只能作为初步信号，不能下强结论。
- Adapter 在两类上都优于 Base E5，说明 `e5_omni_recipe` adapter 不是无效的，但提升还不够。

---

## 7. Reference vs Target 诊断

这是本次最重要的诊断。

| 指标 | Base E5 | Adapter |
|---|---:|---:|
| reference rank median | 1 | 2 |
| reference rank <= 1 | 93.3% | 46.7% |
| target beats reference | 6.67% | 46.67% |
| target-ref score gap mean | -0.015 | -0.002 |
| target-ref score gap max | +0.002 | +0.014 |

### 7.1 Base E5 的问题

Base E5 几乎总把 reference 排在 target 前面：

```text
reference rank <= 1 = 93.3%
target beats reference = 6.67%
```

这说明 Base E5 对同源 reference/target 的视觉相似性非常敏感，但对 edit_text 指定的音频变化方向不敏感。

换句话说，Base E5 更像是在回答：

```text
哪个视频和 reference 最像？
```

而不是：

```text
哪个视频是 reference 按音频 edit 后得到的 target？
```

### 7.2 Adapter 的改进

Adapter 把：

```text
target_beats_reference: 6.67% -> 46.67%
reference rank median: 1 -> 2
R@1: 6.67% -> 40.00%
```

说明 adapter 已经学到了一部分方向性，把大量 reference 从 rank 1 挤下去了。

但它还没有完全解决问题，因为仍有超过一半样本中 reference 分数高于 target。

### 7.3 Score gap 的意义

Base E5 的 target-ref gap 均值为 `-0.015`，Adapter 为 `-0.002`。这说明 Adapter 让 target 和 reference 的分数差距明显接近 0，但还没有稳定转正。

这正是后续引入 `L_ref` 的直接动机：

```text
L_ref: score(query, target) > score(query, reference)
```

---

## 8. Hard Negative 诊断

| 负例类型 | positive beats negative rate |
|---|---:|
| asr_hard | 100.00% |
| audio_hard | 100.00% |
| visual_hard | 100.00% |
| reference_negative | 46.67% |

前三类 hard negatives 在当前 pilot 中并不是真正难点。真正困难的是 reference negative。

这说明当前 hard-negative 设计需要调整优先级：

```text
reference_negative 应该成为每条 query 的必备核心负例。
```

后续训练不应只依赖 `asr_hard / audio_hard / visual_hard`，而应显式优化 target 相对 reference 的排序。

---

## 9. 当前 Adapter 的局限

### 9.1 e5_omni_recipe 有效，但不是 AudioDelta 专属解法

Adapter 从 `6.67%` 提升到 `40.00%`，说明 E5-Omni recipe 的三类机制有帮助：

1. modality-aware temperature
2. negative curriculum + false-negative debiasing
3. covariance/CORAL alignment

但这些机制仍是通用 embedding recipe，并没有显式建模：

```text
target_audio - reference_audio ≈ edit_text
```

因此它只能部分缓解 reference-vs-target 问题。

### 9.2 当前最大错误不是随机 distractor

加入 reference negative 后，随机 distractor 已经不是主要问题。R@5/R@10 仍是 100%，说明 target 和 reference 都通常排在前几名。

真正的问题是：

```text
target 和 reference 谁排第 1？
```

这个问题必须用方向性训练目标解决。

### 9.3 不能再用“无 reference 的随机 gallery”证明模型有效

无 reference 的随机 gallery 会让 Base E5 R@1=100%，这会掩盖 AudioDelta 任务的核心难度。

后续所有 pilot 诊断都应至少包含：

```text
target positives
reference negatives
random distractors
```

正式 benchmark 还应加入更强的同源/同场景/同音频类型 hard gallery。

---

## 10. 本次实验暴露出的关键问题

### 10.1 原始 E5 是 video/context-centric

Base E5 在随机 gallery 上几乎完美，但 reference negative 一加入就崩到 R@1=6.67%。这支持我们之前的判断：

```text
现有 E5/Omni embedding 更擅长找视觉和语境相似，
但没有稳定建模音频 edit 的方向性。
```

### 10.2 当前评估必须报告 reference-aware 指标

仅报告 R@1/R@5/R@10 不够。必须额外报告：

1. target beats reference
2. reference rank median
3. target-reference score gap
4. reference_negative recall
5. per-query top-k 错例

这些指标比普通随机 gallery R@1 更能说明 AudioDelta-CVR 的真实难度。

### 10.3 Stage-1 recipe 是有效 baseline，不是最终方法

`e5_omni_recipe` adapter 的提升说明当前训练链路有用：

```text
Base E5 R@1 = 6.67%
Adapter R@1 = 40.00%
```

但 40% 仍远远不够。下一步必须进入 AudioDelta Stage-2。

---

## 11. 下一步实验计划

下一轮不要立刻加复杂模块。建议复用同一份 cache，做最小 Stage-2 网格：

| 实验 | 目的 |
|---|---|
| e5_omni_recipe | 当前 baseline |
| + L_ref | 专门解决 target vs reference 排序 |
| + L_ref + L_delta | 学习 target_audio - reference_audio ≈ edit_text |
| + L_ref + L_delta + edit_type | 区分 add/remove/replace 等编辑方向 |

### 11.1 主要指标

每个实验都必须报告：

1. R@1 / R@5 / R@10
2. target beats reference
3. reference rank median
4. target-reference score gap
5. audio_event R@1
6. speech R@1
7. per-query top-k 错误样本

### 11.2 判断标准

如果 `+ L_ref` 明显提升：

```text
target_beats_reference
reference rank median
target-reference score gap
```

说明方向性问题主要来自 reference negative 未被显式训练。

如果 `+ L_delta` 继续提升 audio_event/speech：

```text
audio_event R@1
speech R@1
target-reference score gap
```

说明 AudioDelta loss 确实是任务核心贡献。

如果 `edit_type` 有进一步收益，则说明 add/remove/replace 的方向建模值得保留。

---

## 12. 结论

本次 1% pilot 的最大价值不是 adapter 最终 R@1 有多高，而是证明了评估协议和方法设计中的核心事实：

1. 随机 gallery 会高估 Base E5。
2. reference negative 是 AudioDelta-CVR 最关键的 hard case。
3. Base E5 主要按视觉/语境相似度排序，缺少音频 edit 方向性。
4. `e5_omni_recipe` adapter 能把 R@1 从 6.67% 提升到 40.00%，说明训练链路有效。
5. 但当前 Stage-1 recipe 仍不足以解决 target/reference 排序。
6. 下一步必须引入 AudioDelta 专属的 `L_ref` 和 `L_delta`。

一句话总结：

```text
AudioDelta-E5 的核心难点不是“在随机视频里找到相似 target”，
而是“在 reference 和 target 高度相似时，判断哪个视频真的发生了 edit_text 指定的音频变化”。
```

