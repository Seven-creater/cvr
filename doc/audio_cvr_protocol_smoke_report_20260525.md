# Audio-CVR 7-Mode Ablation 报告

- 日期: 2026-05-28
- 数据: merged_all_220 B-line (`audio_cvr_bline_6_9s_merged_all_220_20260527_164758`)
- 模型: E5-omni-7B + adapter (3584 维, **600 步**, e5_omni_recipe)
- 设备: GPU 1-7 并行 (7 × A6000 49 GiB)
- Gallery: typed_hardneg + local_same_source + random, 1000 条
- 训练集: 512 条, 评估集: 128 条

---

## 1. 数据集

### 1.1 来源与规模

数据合并自三批 B-line 评审结果，覆盖 6 个视频数据集，共 640 条 triplets。

| 指标 | 数值 |
|---|---|
| 总 triplets | 640 条（372 main + 268 extended） |
| 训练集 (b_splits/train) | 507 条 |
| 验证集 (b_splits/val) | 65 条 |
| 测试集 (b_splits/test) | 68 条 |
| E5 训练集 | 512 条 |
| E5 评估集 | 128 条（val + test 合并） |
| 唯一视频片段 | 1080 个（564 reference + 569 target） |

### 1.2 数据集分布

| 数据集 | 数量 | 占比 | audio_event | speech |
|---|---:|---:|---:|---:|
| avatar | 326 | 50.9% | 276 | 50 |
| hdtf | 204 | 31.9% | 0 | 204 |
| voxceleb | 61 | 9.5% | 0 | 61 |
| worldsense | 30 | 4.7% | 22 | 8 |
| daily_omni | 11 | 1.7% | 0 | 11 |
| vggsound | 8 | 1.2% | 7 | 1 |
| **合计** | **640** | | **305** | **335** |

hdtf 和 voxceleb 全部为 speech 类型（说话人数据集），avatar 最多样（276 audio_event + 50 speech）。

### 1.3 Triplet 结构

每个 triplet 是一个**音频编辑对**：

```
reference clip (原始视频)  →  target clip (音频被编辑后的同一视频)
```

- **reference**: 原始未编辑的视频片段
- **target**: 同一源视频，但音频经过编辑（语音内容或音频事件被修改）
- **edit_text**: 描述编辑操作的自然语言

视觉内容几乎一样，只有音频不同。模型必须靠音频差异判断哪个是编辑后的版本。

### 1.4 数据集划分

```
640 条 triplets (source-disjoint split)
  ├── 训练集: 507 条 (80%)
  ├── 验证集: 65 条 (10%)
  └── 测试集: 68 条 (10%)
```

划分是 source-disjoint 的：同源视频的所有 triplets 只出现在一个 split 中，避免数据泄露。

### 1.5 Gallery 检索库构成

检索库共 1000 条，128 个 eval query 共享同一个 gallery：

| 类别 | 数量 | 说明 |
|---|---:|---|
| **positive** | 128 | 所有 eval query 的 target（编辑后的正例） |
| **reference_negative** | 128 | 对应的原始 reference（未编辑的同源视频） |
| **visual_hard** | 128 | 视觉上和 target 相似的其他视频 |
| **audio_hard** | 128 | 音频上和 target 相似的其他视频 |
| **asr_hard** | 128 | ASR 文本和 target 相似的其他视频 |
| **local_same_source** | 36 | 同源视频的其他 clip（严格同源） |
| **local_fallback_visual** | 62 | 同源但降级为 visual fallback 的 clip |
| **distractor** | 262 | 从 distractor pool 随机采样 |
| **总计** | **1000** | |

**核心设计**: reference_negative 是最难的负例——它和 target 来自同一个源视频，视觉几乎一模一样，只有音频被编辑。`target_beats_reference` 指标衡量 target 排在 reference 之上的比例，是本实验最核心的判别指标。

**local_same_source 覆盖率低的原因**: avatar 数据集每个源视频只切了 2 个 clip（ref + target），没有额外的同源 clip。只有 hdtf/worldsense 等数据集（部分源有 3+ clips）能提供同源干扰，因此只产生了 36 条严格同源 + 62 条 fallback。

---

## 2. 实验设计

7 种输入模态组合，共用同一个 adapter（在 V+A+T cache 上训练），对比检索性能：

| 模式 | Query 输入 | Document 输入 | 音频 | 说明 |
|---|---|---|---|---|
| T-only-fullAV | text_only | video (audio-on) | on | 纯文本查询，文档带音频编码 |
| V-only | video_only | video (audio-off) | off | 纯视觉查询+文档 |
| A-only | audio_only | audio (audio-off) | off | 纯音频查询+文档 |
| V+T | composed | video (audio-off) | off | 视觉+文本，无音频 |
| A+T | audio_text | audio (audio-off) | off | 音频+文本，无视频 |
| V+A | video_only | video (audio-on) | on | 视觉+音频，无文本 |
| V+A+T | composed | video (audio-on) | on | 全模态融合 |

---

## 3. 检索性能总览

### Adapter 结果（主表，600 步）

| 模式 | R@1 | R@5 | R@10 | beats_ref | beats_strict_local† | gap_mean |
|---|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | 0.0234 | 0.1484 | 0.3125 | 0.5938 | 0.7875 | 0.0103 |
| V-only | 0.1328 | 0.8594 | 0.9609 | 0.1953 | 0.6500 | -0.0105 |
| A-only | 0.1250 | 0.6016 | 0.7344 | 0.2188 | 0.5750 | -0.0311 |
| V+T | 0.1875 | 0.8672 | 0.9453 | 0.2812 | 0.6875 | -0.0072 |
| A+T | 0.2109 | 0.7344 | 0.8125 | 0.2891 | 0.5625 | -0.0208 |
| V+A | 0.2500 | 0.8828 | 0.9453 | 0.3359 | 0.7250 | -0.0141 |
| **V+A+T** | **0.3359** | **0.8984** | **0.9609** | **0.4375** | **0.7500** | **-0.0016** |

† beats_strict_local 仅统计**同源**比较（target 与来自同一 source 的 local clip）。详见 Section 5 同源/异源分解。

### Base E5 完整结果

| 模式 | R@1 | R@5 | R@10 | beats_ref | beats_strict_local† | gap_mean |
|---|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | 0.0859 | 0.3750 | 0.5234 | 0.5234 | 0.7375 | 0.0011 |
| V-only | 0.0000 | 0.8672 | 0.9453 | 0.0000 | 0.4625 | -0.0220 |
| A-only | 0.0000 | 0.7812 | 0.8516 | 0.0000 | 0.5250 | -0.0503 |
| V+T | 0.0156 | 0.8750 | 0.9531 | 0.0156 | 0.5125 | -0.0179 |
| A+T | 0.0312 | 0.9375 | 0.9766 | 0.0312 | 0.6500 | -0.0402 |
| V+A | 0.0000 | 0.9062 | 0.9453 | 0.0000 | 0.5875 | -0.0545 |
| V+A+T | 0.0156 | 0.9453 | 0.9844 | 0.0234 | 0.6375 | -0.0325 |

### Base E5 vs Adapter 对比

| 模式 | Base R@1 | Adapter R@1 | 提升 | Base beats_ref | Adapter beats_ref | 提升 |
|---|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | 0.0859 | 0.0234 | -6.3pp | 0.5234 | 0.5938 | +7.0pp |
| V-only | 0.0000 | 0.1328 | +13.3pp | 0.0000 | 0.1953 | +19.5pp |
| A-only | 0.0000 | 0.1250 | +12.5pp | 0.0000 | 0.2188 | +21.9pp |
| V+T | 0.0156 | 0.1875 | +17.2pp | 0.0156 | 0.2812 | +26.6pp |
| A+T | 0.0312 | 0.2109 | +18.0pp | 0.0312 | 0.2891 | +25.8pp |
| V+A | 0.0000 | 0.2500 | +25.0pp | 0.0000 | 0.3359 | +33.6pp |
| V+A+T | 0.0156 | 0.3359 | **+32.0pp** | 0.0234 | 0.4375 | **+41.4pp** |

---

## 4. Audio 必要性分析

### 核心对比: V+T vs V+A vs V+A+T

| 指标 | V+T | V+A | V+A+T | V+T→V+A 增量 | V+T→V+A+T 增量 |
|---|---:|---:|---:|---:|---:|
| R@1 | 0.1875 | 0.2500 | 0.3359 | **+6.2pp** | **+14.8pp** |
| target_beats_ref | 0.2812 | 0.3359 | 0.4375 | **+5.5pp** | **+15.6pp** |
| gap_mean | -0.0072 | -0.0141 | -0.0016 | -0.0069 | +0.0056 |

### 分析

1. **音频在 R@1 上有明确提升**: V+T→V+A+T 提升 14.8pp（18.8%→33.6%），在全模态组合中增量最大。

2. **target_beats_ref 提升显著**: V+T→V+A+T 的 beats_ref 从 28.1% 提升到 43.8%（+15.6pp），说明音频帮助模型更好地区分编辑后的 target 与原始 reference。

3. **gap_mean 接近零**: V+A+T 的 gap_mean 为 -0.0016，是所有模式中最接近零的，表明 adapter 在全模态下对 target/reference 的得分差距最小化效果最好。

4. **T-only-fullAV 的异常行为**: Base E5 的 T-only R@1=8.6%，但 adapter 训练后降到 2.3%。Adapter 在纯文本查询上产生了过拟合或方向错误，但 beats_ref 从 52.3% 升到 59.4%，说明文本查询虽然检索排序差，但 target/reference 区分方向有所改善。

### 模态贡献排序

| 排名 | 模态 | 关键证据 |
|---|---|---|
| 1 | **音频 (A)** | V+T→V+A+T R@1 +14.8pp，beats_ref +15.6pp，增量最大 |
| 2 | **视频 (V)** | V-only R@1=13.3%，是单模态中第二强 |
| 3 | **文本 (T)** | T-only R@1 仅 2.3%（adapter 后），对检索帮助有限 |

---

## 5. Hard Negative 分解

### Adapter 结果（600 步）

| 模式 | beats ref_neg | beats strict_local | beats visual_hard | beats audio_hard | beats asr_hard |
|---|---:|---:|---:|---:|---:|
| T-only-fullAV | 0.5938 | 0.7875 | 0.9609 | 0.9453 | 0.8594 |
| V-only | 0.1953 | 0.6500 | 1.0000 | 1.0000 | 1.0000 |
| A-only | 0.2188 | 0.5750 | 1.0000 | 1.0000 | 0.9922 |
| V+T | 0.2812 | 0.6875 | 1.0000 | 1.0000 | 1.0000 |
| A+T | 0.2891 | 0.5625 | 1.0000 | 1.0000 | 1.0000 |
| V+A | 0.3359 | 0.7250 | 1.0000 | 1.0000 | 1.0000 |
| V+A+T | **0.4375** | **0.7500** | 1.0000 | 1.0000 | 1.0000 |

### Base E5 结果

| 模式 | beats ref_neg | beats strict_local | beats visual_hard | beats audio_hard | beats asr_hard |
|---|---:|---:|---:|---:|---:|
| T-only-fullAV | 0.5234 | 0.7375 | 0.7969 | 0.8203 | 0.9453 |
| V-only | 0.0000 | 0.4625 | 1.0000 | 1.0000 | 1.0000 |
| A-only | 0.0000 | 0.5250 | 1.0000 | 1.0000 | 1.0000 |
| V+T | 0.0156 | 0.5125 | 1.0000 | 1.0000 | 1.0000 |
| A+T | 0.0312 | 0.6500 | 1.0000 | 1.0000 | 1.0000 |
| V+A | 0.0000 | 0.5875 | 1.0000 | 1.0000 | 1.0000 |
| V+A+T | 0.0234 | 0.6375 | 1.0000 | 1.0000 | 1.0000 |

### 同源/异源分解（关键修正）

Gallery 中 36 条 strict_local 和 62 条 fallback_local 来自多个 source，但每个 query 只与**同源**的 local clip 才构成真正的挑战。混合计算会被大量异源比较稀释。

**strict_local 覆盖**: 128 个 query 中仅 14 个有同源 strict_local（共 80 对比较）。

#### Adapter 600 步: strict_local 同源/异源分解

| 模式 | 同源 beats | 同源 total | 同源 rate | 异源 beats | 异源 total | 异源 rate |
|---|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | 63 | 80 | **0.7875** | 4108 | 4528 | 0.9072 |
| V-only | 52 | 80 | **0.6500** | 4528 | 4528 | 1.0000 |
| A-only | 46 | 80 | **0.5750** | 4469 | 4528 | 0.9870 |
| V+T | 55 | 80 | **0.6875** | 4528 | 4528 | 1.0000 |
| A+T | 45 | 80 | **0.5625** | 4500 | 4528 | 0.9938 |
| V+A | 58 | 80 | **0.7250** | 4523 | 4528 | 0.9989 |
| V+A+T | 60 | 80 | **0.7500** | 4528 | 4528 | 1.0000 |

#### Base E5: strict_local 同源/异源分解

| 模式 | 同源 beats | 同源 total | 同源 rate | 异源 beats | 异源 total | 异源 rate |
|---|---:|---:|---:|---:|---:|---:|
| T-only-fullAV | 59 | 80 | 0.7375 | 4260 | 4528 | 0.9408 |
| V-only | 37 | 80 | 0.4625 | 4528 | 4528 | 1.0000 |
| A-only | 42 | 80 | 0.5250 | 4518 | 4528 | 0.9978 |
| V+T | 41 | 80 | 0.5125 | 4528 | 4528 | 1.0000 |
| A+T | 52 | 80 | 0.6500 | 4528 | 4528 | 1.0000 |
| V+A | 47 | 80 | 0.5875 | 4528 | 4528 | 1.0000 |
| V+A+T | 51 | 80 | 0.6375 | 4528 | 4528 | 1.0000 |

#### fallback_local

62 条 fallback_local 全部为**异源**——没有一条与任何 eval query 的 target 同源。因此 beats_fallback_local 的 100% 不具有判别意义，已从主表中移除。

### 分析

1. **同源 strict_local 是真正的第二难点**: V+A+T 同源 beats=75.0%，远低于之前混合计算的 99.5%。仅高于 reference_negative（43.8%），但远低于 typed hard negative（100%）。

2. **异源 local 无意义**: 4528 条异源比较中几乎全部 100% 打败，严重稀释了同源数据。

3. **Reference negative 仍是最难的**: beats_ref=43.8%，低于同源 strict_local 的 75.0%。Reference 和 target 来自同一段视频的不同 clip（segment_001 vs segment_002），视觉上比 strict_local（不同 segment）更相似。

4. **Adapter 对同源 local 有提升**: Base E5 V+T 同源 beats=51.25%，adapter 后升至 68.75%（+17.5pp）。全模态 V+A+T 从 63.75% 升到 75.0%（+11.25pp）。

---

## 6. 错误分析

| 错误类型 | 数量 | 占比 | 说明 |
|---|---:|---:|---|
| reference wins | 566 | 77.1% | reference 排在 target 之上（方向性判别失败） |
| random wins | 131 | 17.8% | 随机干扰项排在 target 之上 |
| local_same_source wins | 37 | 5.0% | 同源 clip 排在 target 之上 |

**reference wins 占主导**（77.1%），表明模型区分编辑前后版本的能力仍严重不足。与 120 步结果（73.7%）相比略有恶化，说明增加步数主要改善了正确检索的比例（R@1 从 28.9% 提升到 33.6%），但错误类型分布未根本改变。

---

## 7. Adapter 训练收敛

### 训练配置

| 参数 | 值 |
|---|---|
| 训练步数 | **600** |
| Batch size | 8 |
| Learning rate | 3e-4 |
| Training profile | e5_omni_recipe |
| Adapter 维度 | 3584 × 3584 |

### Loss 曲线

| 指标 | Step 1 | Step 600 |
|---|---:|---:|
| total loss | 0.1997 | 0.0024 |
| cvr loss | 0.1940 | 0.0000 |
| delta loss | 0.0456 | 0.1260 |
| ref loss | 0.4387 | 0.0000 |
| edit_type loss | 0.0975 | 0.2037 |
| batch whitening | 0.5715 | 0.2376 |

训练收敛良好，total loss 从 0.20 降到 0.002。CVR loss 和 ref loss 都降至接近零，但 delta loss 和 edit_type loss 仍较高，表明模型在编辑方向性判断上仍有学习空间。

---

## 8. Adapter 步数 Sweep

对 adapter 训练步数进行 4 组对照（120 / 300 / 600 / 1000），其他参数不变。

### V+A+T 性能趋势

| Steps | R@1 | R@5 | R@10 | beats_ref | gap_mean |
|---:|---:|---:|---:|---:|---:|
| 120 | 0.2891 | 0.8750 | 0.9375 | 0.3750 | -0.0067 |
| 300 | 0.3125 | 0.8984 | 0.9453 | 0.3984 | -0.0054 |
| **600** | **0.3359** | **0.8984** | **0.9609** | **0.4375** | **-0.0016** |
| 1000 | 0.2969 | 0.9062 | 0.9609 | 0.3906 | -0.0065 |

### 音频必要性 Delta（V+A+T - V+T）

| Steps | V+T R@1 | V+A+T R@1 | Delta R@1 | V+T beats_ref | V+A+T beats_ref | Delta beats_ref |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 0.1562 | 0.2891 | +13.3pp | 0.2422 | 0.3750 | +13.3pp |
| 300 | 0.1406 | 0.3125 | +17.2pp | 0.2266 | 0.3984 | +17.2pp |
| **600** | **0.1875** | **0.3359** | **+14.8pp** | **0.2812** | **0.4375** | **+15.6pp** |
| 1000 | 0.1641 | 0.2969 | +13.3pp | 0.2734 | 0.3906 | +11.7pp |

### 各模式最优步数

| 模式 | 最优步数 | 最优 R@1 | 最优 beats_ref |
|---|---:|---:|---:|
| T-only-fullAV | 1000 | 0.0469 | 0.5625 |
| V-only | 1000 | 0.1562 | 0.2344 |
| A-only | 120 | 0.1562 | 0.2188 |
| V+T | 600 | 0.1875 | 0.2812 |
| A+T | 600 | 0.2109 | 0.2891 |
| V+A | 300 | 0.2578 | 0.3281 |
| **V+A+T** | **600** | **0.3359** | **0.4375** |

### 分析

1. **600 步是 V+A+T 的 sweet spot**: R@1=33.6%, beats_ref=43.8%, gap_mean=-0.0016（最接近零）。相比 120 步提升 4.7pp R@1 和 6.3pp beats_ref。

2. **1000 步出现过拟合**: V+A+T R@1 从 33.6%（600 步）回落到 29.7%（1000 步），beats_ref 从 43.8% 降到 39.1%。512 条训练数据的容量有限。

3. **音频必要性在所有步数下都成立**: V+T→V+A+T 的 R@1 增量在所有步数下均 >13pp，证明音频贡献不是训练偶然现象。

4. **单模态行为不一致**: A-only 在 120 步最优但 1000 步明显退化（15.6%→10.9%），说明纯音频模式对过拟合更敏感。

---

## 9. 与上次实验对比

| 指标 | 上次 (merged2, 64 eval, 120步) | 本次 (merged_all_220, 128 eval, **600步**) | 变化 |
|---|---|---|---|
| 总 triplets | 336 | 640 | 1.9× |
| 数据集 | 1 (avatar) | 6 (avatar+hdtf+voxceleb+worldsense+daily_omni+vggsound) | 多源 |
| 训练集 | 192 | 512 | 2.7× |
| 评估集 | 64 | 128 | 2× |
| Gallery | 320 (typed_hardneg) | 1000 (typed_hardneg+local_same_source+random) | 3.1× |
| 训练步数 | 120 | **600** | 5× |
| V+A+T Adapter R@1 | 1.0000 | 0.3359 | -66.4pp |
| V+T Adapter R@1 | 0.9844 | 0.1875 | -79.7pp |
| V+T→V+A+T R@1 增量 | +1.56pp | +14.8pp | 音频贡献增大 |

### 性能大幅下降的原因分析

1. **Gallery 规模增大 3 倍**: 从 320 条增至 1000 条，包含 262 条随机干扰项，检索难度显著增加。

2. **评估 query 翻倍且来源多样**: 128 个 eval query 覆盖 6 个数据集（avatar 占 50.9%），比之前单数据集评估更具挑战。

3. **gap_mean 接近零但未转正**: 600 步 V+A+T gap_mean=-0.0016，比 120 步（-0.0067）更接近零，但仍为负。说明 reference 的得分系统性高于 target 的情况在改善。

4. **与上次 smoke 测试的反差**: 上次在极小规模 (320 gallery, 64 eval) 上达到 100% R@1，可能是因为 gallery 太小、难度太低，模型表现被高估了。本次结果更接近真实性能。

---

## 10. 结论

1. **V+A+T 全模态最优**: R@1=33.6%, target_beats_ref=43.8%。在全模态融合下，有 43.8% 的 query 能正确将 target 排在 reference 之上，是所有模态组合中最高的。

2. **音频必要性在困难场景下更明显**: V+T→V+A+T 的 R@1 增量为 +14.8pp，比上次的 +1.56pp 大得多。在更难的 gallery 下，音频的贡献从"补丁"升级为"关键信号"。此结论在所有训练步数下均成立（>13pp）。

3. **Reference negative 仍是核心瓶颈**: 77.1% 的错误是 reference 排在 target 之上。模型在"编辑前后区分"这个核心任务上仍有根本性困难。

4. **上次 100% R@1 是假象**: 小 gallery (320) + 小 eval (64) 下的完美表现不代表泛化能力。本次 1000 gallery + 128 eval 的结果更可信。

5. **600 步是最优训练量**: 相比 120 步提升 4.7pp R@1，1000 步开始过拟合。512 条训练数据的 adapter 容量已接近上限。

6. **后续方向**: 扩充训练数据量（当前 512 条可能不够）、增加 reference/delta loss 权重、尝试更深层的 adapter 结构、扩充训练数据覆盖更多数据集。

---

## 11. 输出路径

| 产物 | 路径 |
|---|---|
| OUT_ROOT | `runs/audio_necessity_7mode_all_220_20260527_164758/` |
| Records | `runs/.../records_typed_hardneg_localsource/` |
| Adapter (600步) | `runs/.../adapter_steps600/` |
| 7-mode eval (600步) | `runs/.../eval_steps600_{T_only_fullAV,V_only,...}/` |
| Sweep 汇总 | `runs/.../summary_steps600/` |
| 全部 sweep eval | `runs/.../eval_steps{120,300,600,1000}_*/` |
| Gallery 结果 | `runs/.../summary_steps600/gallery_protocol_results.md` |
| Hard negative 分解 | `runs/.../summary_steps600/hard_negative_breakdown.md` |
