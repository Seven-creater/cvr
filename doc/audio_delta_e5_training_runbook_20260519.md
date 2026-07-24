# AudioDelta-E5 训练说明（研究版）

日期：2026-05-19  
适用范围：`app/e5_audio_delta_train.py`（V1 / V2.1）

## 1. 我们在训练什么

AudioDelta-E5 训练的核心不是做 ASR 转写检索，也不是普通视频语义检索，而是学习一个**有方向的相对编辑检索**：

```text
reference + edit_text -> target
```

目标是让模型判断：`target` 相比 `reference` 是否按 `edit_text` 发生了正确的音频变化，同时仍在合理的视频上下文里。

---

## 2. 损失是怎么构成的（目的 + 依据）

总体目标函数（简化）：

```text
L_total
= L_cvr
+ λ_delta * L_delta
+ λ_hn * L_hn
+ λ_ref * L_ref
+ λ_edit * L_edit_type
+ λ_visual * L_visual
+ λ_hw * L_hardness_weighted
+ λ_coral * L_coral_align
+ λ_white * L_batch_whitening
```

其中相似度统一记为（向量先 `L2` 归一化）：

```text
s(x,y) = x^T y
```

### 2.1 主检索损失 `L_cvr`（InfoNCE）
- 目的：把正确 target 拉近，把错误候选推远，形成基础检索空间。
- 依据：对比学习 InfoNCE/CPC 是检索训练的通用基础。  
  参考：https://arxiv.org/abs/1807.03748

### 2.2 音频差分损失 `L_delta`
- 目的：显式学习“reference 到 target 的音频变化方向”，避免模型只学主题相似。
- 形式：约束 `sim(a_t, e) - sim(a_r, e)`，让 target 对 edit 更匹配、reference 更不匹配。
- 价值：这是 AudioDelta 任务最关键的任务特化约束。

```text
Δ(q,d) = s(a_t, e) - s(a_r, e)
L_delta = max(0, m_delta - Δ(q,d+) + Δ(q,d-))
```

### 2.3 难负样本损失 `L_hn`
- 目的：针对 `visual_hard / audio_hard / asr_hard / reference_negative` 强化区分，抑制捷径。
- 价值：避免模型只靠视觉、关键词或 reference 自身“投机”。

```text
L_hn = Σ_k w_k * max(0, m_k - s(q, d+) + s(q, d_k-))
```

### 2.4 `L_ref`（Reference-as-Negative）
- 目的：强制模型知道“reference 不是答案”，答案是“被 edit 后的 target”。
- 价值：显式编码“方向性编辑”的任务本质。

```text
L_ref = max(0, m_ref - s(q, d+) + s(q, d_ref))
```

### 2.5 `L_edit_type`（按编辑类型建模）
- 目的：`add/remove/replace/increase/decrease` 使用不同约束，避免一套公式覆盖所有编辑。
- 价值：对齐真实编辑语义，提升可解释性与泛化。

```text
Add/Increase:
L_add = max(0, m_add - s(a_t, e) + s(a_r, e))

Remove/Decrease:
L_remove = max(0, m_remove - s(a_r, e) + s(a_t, e))

Replace:
L_replace =
  max(0, m_rep - s(a_t, e_new) + s(a_t, e_old))
+ max(0, m_rep - s(a_r, e_old) + s(a_r, e_new))
```

### 2.6 `L_visual`（上下文保留）
- 目的：保持视频上下文一致性，避免纯音频检索退化。
- 价值：确保任务仍是 CVR，而不是音频单模态检索。

```text
L_visual = max(0, m_vis - s(v_r, v_t+))
```

### 2.7 Hardness-weighted negatives（V2.1）
- 目的：给更难负样本更高权重，减少 easy negatives 主导训练。
- 价值：提升困难样本下的判别能力，尤其是混淆场景。

```text
w_k^hard = clip( softmax(s(q, d_k-) / τ_hard) * K, w_min, w_max )
```

### 2.8 Modality-aware temperature（V2.1）
- 目的：不同模态组合（T/A/V）相似度分布不同，用单一温度会失配；引入模态温度校准 logits。
- 依据：e5-omni / MMEB-V3 强调多模态检索中的温度与模态感知优化。  
  参考：https://arxiv.org/abs/2601.03666

```text
τ(x) = mean({τ_m | m ∈ modalities(x)})
τ_pair(x,y) = 0.5 * (τ(x) + τ(y))
logits(x,y) = s(x,y) / τ_pair(x,y)
```

### 2.9 `L_coral_align`（协方差对齐，V2.1）
- 目的：对齐 doc/edit（及 delta/edit）二阶统计，缓解模态分布不一致。
- 依据：Deep CORAL 协方差对齐。  
  参考：https://arxiv.org/abs/1607.01719

```text
Cov(Z) = (Z - μ)^T (Z - μ) / (B - 1)

L_coral_doc_edit =
  ||Cov(Z_doc) - Cov(Z_edit)||_F^2 / (4 d^2)

L_coral_delta_edit =
  ||Cov(Z_delta) - Cov(Z_edit)||_F^2 / (4 d^2)
```

### 2.10 `L_batch_whitening`（V2.1）
- 目的：约束 batch 表示的均值和协方差，降低塌缩与异常偏移风险。
- 价值：改善训练稳定性与跨 batch 一致性。

```text
L_whiten(Z) =
  ||mean(Z)||_2^2 + ||Cov(Z) - I||_F^2 / d^2
```

### 2.11 局部片段匹配（Local Segment Mix）

用于处理“变化只发生在局部时间片段”的情况：

```text
s_mix(q,d) = (1 - α) * s_global(q,d) + α * s_local(q,d)
```

其中 `s_local` 来自局部 segment/patch 聚合匹配分数。

---

## 3. e5-omni 借鉴对比表

| 组件 | e5-omni / MMEB-V3 思路 | AudioDelta-E5 落地方式 | 作用 |
|---|---|---|---|
| 主对比检索 | 多模态对比学习主干 | `L_cvr`（InfoNCE） | 建立基础检索空间 |
| 模态温度 | 模态感知 temperature 校准 | `enable_modality_temperature` + `tau_text/audio/video` | 修正不同模态 logits sharpness |
| 负样本学习 | negative-aware / hard-negative 强化 | `L_hn` + hardness weighting + quantile curriculum | 提升混淆样本区分 |
| 去偏 | 避免 false negatives 干扰 | false-negative debiasing（同组置零/降权） | 减少错误梯度 |
| 统计对齐 | 分布/协方差层面的跨模态对齐 | `L_coral_align`（doc-edit、delta-edit） | 缓解模态间分布错位 |
| 稳定化 | 训练稳定性增强 | `L_batch_whitening` | 降低表示塌缩与震荡 |
| 任务特化 |（e5-omni偏通用检索） | `L_delta + L_ref + L_edit_type + L_visual` | 对齐 Audio-CVR 的“方向性编辑” |

> 说明：AudioDelta-E5 不是照搬 e5-omni，而是在其多模态检索思想上，增加了面向 Audio-CVR 的方向性损失与反捷径约束。

---

## 3.1 e5-omni 损失构造（补充）

根据 e5-omni / MMEB-V3 的公开描述，可归纳为“对比学习主损失 + 负样本强化 + 模态/统计校准”三层：

1. **主对比损失（InfoNCE）**

```text
L_contrast = -log( exp(s(q, d+) / τ) / Σ_j exp(s(q, d_j) / τ) )
```

- 其中 `s(·,·)` 为归一化向量相似度（通常点积/余弦）。
- 核心作用：拉近正样本，推远负样本。

2. **Negative-aware / Hard Negative 强化**

```text
L_neg-aware = Σ_k w_k * max(0, m_k - s(q, d+) + s(q, d_k-))
```

- `w_k` 可由负样本难度决定（越难权重越高）。
- 核心作用：提升混淆样本区分能力，减少 easy negative 主导。

3. **Modality-aware temperature 校准**

```text
logits(q,d) = s(q,d) / τ_modality(q,d)
```

- 不同模态组合使用不同有效温度（而非单一全局 `τ`）。
- 核心作用：修正多模态相似度分布 sharpness 不一致问题。

4. **分布/协方差对齐正则（可对应 CORAL）**

```text
L_align = || Cov(Z_a) - Cov(Z_b) ||_F^2
```

- 对齐不同模态或不同表示空间的二阶统计。
- 核心作用：缓解模态 gap，提升共享检索空间一致性。

在本项目中，上述四类思想分别映射到：
- `L_cvr`（主对比）
- `L_hn + hardness/quantile`（负样本强化）
- `modality-aware temperature`（模态温度）
- `L_coral_align + L_batch_whitening`（统计对齐与稳定化）

---

## 4. 结果怎么看（最小分析框架）

至少观察四类结果：

1. **主指标**：R@1 / R@5 / R@10  
2. **分组指标**：`by_split_tier`、`by_audio_delta_type`、`by_shortcut_label`  
3. **方向性指标**：`delta_score_pos_mean` vs `delta_score_neg_mean`、reference rank  
4. **训练稳定性**：loss 是否 NaN、温度是否越界、effective negatives 是否过低

如果以下现象出现，说明设计有效：

- `without_delta` 明显下降（方向性损失有效）  
- `without_modality_temperature` 在混合模态检索上下降（温度校准有效）  
- `without_coral_align` 或 `without_batch_whitening` 稳定性变差（对齐/稳定项有效）  
- `without_false_negative_debiasing` 误检上升（去偏有效）

---

## 5. 一句话结论

AudioDelta-E5 的损失设计是“**通用多模态检索框架 + Audio-CVR 任务特化约束**”：  
通用部分借鉴 e5-omni（模态温度、负样本强化、统计对齐），任务部分强调音频编辑方向、reference 非答案与反捷径，从而让模型学到真正可用的 audio-delta 检索能力。
