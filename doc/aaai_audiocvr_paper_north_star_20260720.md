# Audio-CVR AAAI 论文北极星（历史版本）

> 状态：已由 `doc/aaai_audiocvr_paper_storyline_canonical.md` 取代。  
> 当前最高优先级写作基准：`doc/aaai_audiocvr_paper_storyline_canonical.md`。  
> 本文保留 2026-07-20 至 2026-07-21 阶段的推理过程和 Core150 证据，不再作为最终 Test1000 叙事入口。  
> 建立日期：2026-07-20。  
> 最新证据更新：2026-07-21，加入 OmniCVR 995-query 跨 benchmark reference 诊断。  
> 通俗版全文逻辑：`doc/aaai_audiocvr_plain_language_storyline_20260721.md`。  
> 适用范围：标题、摘要、Introduction、Related Work、Dataset、Experiments、Conclusion、答辩材料与对外介绍。  
> 规则：若本文与 canonical 总纲冲突，以 canonical 总纲为准。

## 1. 论文定位

本文是：

```text
Audio-CVR 任务
+ 多阶段多模态自动数据构造方法
+ reference-aware benchmark 与 audio necessity evaluation
```

本文不是：

```text
adapter 架构论文
新 loss 论文
E5-Omni recipe 论文
普通大规模视频检索论文
只靠 random gallery 提高或比较 R@K 的论文
```

## 2. 核心科学问题

标准 CoVR 通常回答：

```text
模型能否从大量视频中找到语义相关的 target？
```

Audio-CVR 要回答：

```text
在视觉语境基本保持时，
模型能否根据有方向的声音修改，
把已经发生修改的 target 排在尚未修改的 reference 前面？
```

reference 不是普通负样本，而是天然的编辑前反事实负例：

```text
视觉语境正确
内容高度相似
但尚未满足 edit
```

这是全文最有辨识度的中心。

## 3. 三项核心贡献

### 3.1 Audio-primary CVR 任务定义

核心定义：

> Directional audio edit under preserved visual context.

主测试样本必须满足：

```text
reference 不满足声音 edit
target 满足声音 edit
视觉语境保持
不能仅靠静音画面确定 target
不能退化成 transcript / ASR matching
```

与最接近的 CoVA 的边界必须写清：

```text
CoVA      -> 同时研究视觉和声音的 cross-modal changes
Audio-CVR -> 主动隔离 audio-primary change
Audio-CVR -> 检验 reference-to-target 的编辑方向
Audio-CVR -> 显式控制 visual 与 ASR shortcuts
```

不得声称 Audio-CVR 是首个 audio-visual composed retrieval 工作。

### 3.2 多阶段多模态自动数据构造方法

这是方法贡献，不是简单生成数据：

```text
自然视频切片
-> source-aware pair mining
-> audio-only delta analysis
-> audio-only edit generation
-> directional audio verification
-> muted-video shortcut rejection
-> full-AV consistency verification
-> ASR-shortcut screening
-> sampled repeat audit
-> source-disjoint split
```

相较 caption pair + LLM modification，流程额外验证：

```text
edit 是否真的来自声音
reference/target 方向是否成立
仅凭画面是否可以解题
speech 是否退化成 ASR
声音变化在完整视频中是否仍然成立
```

论文统一使用：

> multi-stage multimodal automatic curation pipeline

`agent` 可以描述工程实现，但不作为核心科学主张。核心是跨模态、反事实、分阶段审核，而不是 prompt orchestration。

### 3.3 Reference-specific diagnosis 与 audio necessity evidence

OmniCVR 已经把 source/reference 放入 gallery，并包含同源时间片 hard distractors。因此不得声称“首次把 reference 加入 gallery”，也不得把 mandatory inclusion 本身作为 novelty。

我们的区别是：即使 source 已在 gallery 中，aggregate R@K 也不会直接说明错误是否来自 source-target confusion。Audio-CVR 将 reference 单独形式化为 query-specific counterfactual，并显式报告：

```text
target_beats_reference
target-reference score margin
reference-induced R@1 drop
```

正式候选结构：

```text
target
+ unchanged reference
+ strict source-local clips
+ typed hard negatives
+ random distractors
```

其中：

```text
reference              -> 检验 edit direction
strict local clips     -> 检验同源细粒度区分
typed hard negatives   -> 检验 visual/audio/ASR shortcuts
random distractors     -> 扩大 gallery，只作基础检索压力
```

关键控制实验：

```text
With-reference vs Without-reference
V+A+T vs V+T
Global vs strict local
不同 gallery size
七种 modality combinations
```

`Without-reference` 只从 gallery 中移除当前 query 对应的 reference；query 输入中的 reference 始终保留。

## 4. 不是论文贡献的内容

| 内容 | 正确定位 |
|---|---|
| E5-Omni-7B | backbone |
| e5-omni recipe | 已有训练方法，必须归因原论文 |
| projection adapter | baseline |
| 有无 reference | 验证 reference-aware protocol 的关键实验 |
| 七种模态消融 | 验证 audio necessity 的实验 |
| 1000 random distractors | gallery 扩容手段 |
| typed hard negatives | benchmark protocol 组件 |
| 缓存、shard、resume、setsid/nohup | 可复现工程实现 |
| 历史 AudioDelta auxiliary losses | 不进入当前论文主方法或贡献 |

adapter 即使提升很大，也只写为：

> We establish an adapter-based baseline on top of a frozen E5-Omni backbone.

## 5. 全文唯一主故事线

```text
现有 CVR 难以严格证明 audio 被真正使用
↓
visual similarity、ASR 和 random-gallery shortcuts 会掩盖问题
↓
我们定义 audio-primary directional CVR
↓
设计多阶段跨模态自动审核流程构造数据
↓
将 unchanged reference 作为独立反事实对象进行显式诊断
↓
用 With/Without-reference 证明 reference 是核心难例
↓
用 V+A+T/V+T 证明 audio 对方向判别的额外价值
↓
用 local、typed negatives 和 gallery scaling 解释模型到底在哪里失败
```

任何章节如果不能服务这条链，应删减、下放附录或标记为历史探索。

### 5.1 两层实验证据

论文必须把两类结果分开：

```text
Audio-CVR 内部主实验
-> 证明 audio 在严格 audio-primary protocol 下提供显著增益
-> 证明 reference removal 会把 R@1 从 22.93% 虚高到 99.47%

OmniCVR 跨 benchmark 诊断
-> 证明 reference-induced difficulty 不只存在于我们构造的数据中
-> V+A+T adapter 的 R@1 从 with-reference 0.12% 升至 no-reference 14.21%
-> 不用于证明 audio 的 R@1 增益，也不用于证明 adapter 跨域泛化
```

这两层证据共同服务同一个结论：检索库是否保留 unchanged reference，会实质改变 composed retrieval 测量的问题。

## 6. 固定贡献表述

```text
Our contributions are threefold:

1. We formulate Audio-CVR, an audio-primary composed video
   retrieval task that requires a directional sound modification
   under preserved visual context, while explicitly controlling
   visual and transcript shortcuts.

2. We introduce a scalable multi-stage curation pipeline that
   constructs natural-video triplets through audio-only directional
   verification, muted-video shortcut rejection, full audiovisual
   consistency checking, ASR-shortcut screening, and sampled repeat
   auditing.

3. We introduce a reference-specific diagnostic protocol that isolates
   source-target confusion through target-over-reference accuracy,
   directional score margins, and exact paired reference removal.
   Controlled modality ablations further quantify the additional
   retrieval evidence provided by audio.
```

## 7. 标题基准

推荐标题：

> Audio-CVR: Automatic Curation and Reference-Aware Evaluation of Directional Audio Changes in Composed Video Retrieval

标题中的三个信号必须在正文有直接证据：

```text
Automatic Curation -> 各阶段产出、拒绝原因、重复审核一致率
Reference-Aware    -> reference-specific metrics 与精确配对 removal
Directional Audio  -> V+A+T/V+T、audio-only gate 与 ASR-shortcut screening
```

## 8. 证据完成条件

### 自动构造可信

```text
报告候选到冻结 benchmark 的完整漏斗
报告每阶段拒绝原因
报告重复审核一致率
报告 source/pair leakage = 0
明确 model-verified，不冒充 human-validated
```

### Reference 是核心难例

```text
同一 query、同一模型、同一候选身份
只改变当前 reference 是否进入 gallery
报告 R@1 drop、reference rank、target-reference margin
```

这里的 novelty 是显式隔离和量化 source-target confusion，而不是 source/reference 出现在 gallery 中。Related Work 必须承认 OmniCVR 已经包含 source video 和 source-local distractors。

### Audio 提供额外价值

```text
V+A+T 与 V+T 使用相同 query 和 gallery identities
query/gallery 两侧同步 audio-on/off
至少五个随机种子
报告 effect size、置信区间和配对显著性检验
```

### Benchmark 揭示真实缺陷

```text
random vs reference-aware
global vs strict local
typed negative breakdown
gallery-size scaling
sound_event/music/contextual speech 分项
```

### 跨 Benchmark 外部诊断

OmniCVR 诊断固定使用以下口径：

```text
995 个有效 query
2,000-item official gallery
6 个解码失败视频在所有模式中统一 mask
1,994 个有效 gallery item
5 个相同 seeds
with-reference 与 no-reference 复用同一 embedding cache
```

可写入论文的结果：

```text
V+A+T adapter: 0.12% -> 14.21% R@1 after reference removal (+14.09pp)
V+T adapter:     0.00% -> 12.86% R@1 after reference removal (+12.86pp)
Base E5 V+A+T:   0.00% -> 17.39% R@1 after reference removal
```

两个 reference-induced R@1 inflation 的 Holm 校正配对检验均 `p<0.001`。这项实验只能支持 reference-aware evaluation 的跨 benchmark 必要性。OmniCVR 上 audio 的 R@1 增益仅 `+0.12pp` 且不显著，adapter no-reference R@1 低于 Base E5，因此不得写成 audio necessity 或跨域模型提升证据。

## 9. 写作硬规则

1. 不把 adapter、LoRA、auxiliary loss 或 e5-omni recipe 写成原创贡献。
2. 不声称 audio 对所有 CVR 都必要，只限定于 Audio-CVR protocol 和对应 split。
3. 不以 random-gallery 高 R@K 作为核心成绩。
4. 不隐藏 reference-aware 性能下降；它是 benchmark 有效性的证据。
5. 不把 ASR-like speech 混入主结论；主集只保留 contextual speech，ASR-like 进入 diagnostic。
6. 不把 unverified local candidate 写成 strict local negative。
7. 不把 model-verified 写成 human-validated。
8. 新实验数字只能在 split 冻结、validation-only 选模和多 seed 完成后进入摘要与主表。

## 10. 最终判断标准

论文竞争力不押在 adapter 上，而取决于：

```text
自动构造是否可信
reference 是否被证明是核心难例
audio 是否在严格控制下提供增益
benchmark 是否揭示现有模型的真实缺陷
```

后续所有论文工作均以这四项为优先级。
