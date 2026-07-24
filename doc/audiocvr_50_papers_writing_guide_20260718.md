# Audio-CVR 相关 50 篇论文：写作精华与借鉴指南

> 用途：辅助 Audio-CVR AAAI 论文的 Introduction、Related Work、Dataset、Method 和 Experiments 写作。  
> 整理日期：2026-07-18。  
> 文献来源：[Awesome Composed Multi-modal Retrieval](https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval)、论文官方页面、arXiv/CVF/AAAI Proceedings，以及本地 `paper/` 中的五篇重点论文。

## 1. 如何使用这份文档

这 50 篇论文不是同等重要，建议分三层阅读：

### A 级：必须精读并在正文重点对话

1. CoVA
2. CoVR
3. EgoCVR
4. CIRR
5. e5-omni

这五篇分别对应本项目最关键的五个问题：

```text
CoVA      -> 最接近的 audio-visual composed retrieval 工作
CoVR      -> 大规模自动 triplet 构造与人工评估集
EgoCVR    -> global/local gallery 与细粒度视频变化
CIRR      -> 真实场景 composed benchmark 与 false-negative-aware 评估
e5-omni   -> 当前 embedding backbone 和 adapter 训练 recipe
```

### B 级：正文应引用并吸收实验设计

TIRG、FashionIQ、LaSCo、Visual Delta Generator、Bi-directional Training、CIReVL、VSE++、ImageBind、LanguageBind、Everything at Once、AVLnet、VALOR、VAST、VLM2Vec、VLM2Vec-V2。

### C 级：Related Work 或补充材料使用

其余论文主要用于建立技术谱系、说明自动数据构造、组合机制和 zero-shot 方法的发展。

## 2. 五十篇论文全景总结

### 2.1 任务定义、Benchmark 与数据构造

| # | 论文 | 核心精华 | 对 Audio-CVR 的直接借鉴 |
|---:|---|---|---|
| 1 | **Composed Multi-modal Retrieval: A Survey of Approaches and Applications** (2025) | 将 CMR 分成 supervised、zero-shot、semi-supervised，并把问题归纳为数据构造、组合架构和优化约束。 | Related Work 可以沿“任务与数据、组合检索、omni-modal embedding”三条线组织，而不是逐篇罗列。 |
| 2 | **Composing Text and Image for Image Retrieval: An Empirical Odyssey (TIRG)**, CVPR 2019 | 奠定 `reference + modification -> target` 范式；用 gated residual 表示“保留 reference 并施加变化”。 | 用于定义 Audio-CVR 的相对编辑本质；强调不是重新生成目标，而是在检索空间理解 modification。 |
| 3 | **FashionIQ: A New Dataset Towards Retrieving Images by Natural Language Feedback**, CVPR 2021 | 用自然语言描述目标相对 reference 的变化，说明真实 modification text 比固定属性更灵活。 | 支撑 edit text 必须表达“相对变化”而不是独立 target caption。 |
| 4 | **Image Retrieval on Real-Life Images with Pre-Trained Vision-and-Language Models (CIRR)**, ICCV 2021 | 将 CIR 扩展到真实自然图像；使用 reference-target group 和 subset 指标降低无关 gallery 与 false negatives 的影响。 | 借鉴 group-aware 评估思想；Audio-CVR 的 reference/local gallery 应与 global gallery分开报告。 |
| 5 | **CoVR: Learning Composed Video Retrieval from Web Video Captions**, AAAI 2024 | 从 WebVid caption 中挖相似视频对，再用 LLM 生成 modification，构造 160 万 triplets；另建人工评估集。 | 最直接的数据构造范式。需要学习其“自动大规模训练集 + 高质量独立测试集”的双层设计。 |
| 6 | **EgoCVR: An Egocentric Benchmark for Fine-Grained Composed Video Retrieval**, ECCV 2024 | 构建 2,295 个细粒度视频 query，并同时研究 global 与同序列 local 检索。 | 直接支撑 strict local_same_source protocol；论文必须同时报告 coverage 与 local difficulty。 |
| 7 | **CoVA: Text-Guided Composed Video Retrieval for Audio-Visual Content**, 2026 | 提出 audio-visual composed retrieval 和 AV-Comp，并设计 AVT fusion；明确指出传统 CoVR 忽略声音变化。 | 最接近的竞争工作。不能再声称“首个 audio-aware CoVR”；必须突出 audio-primary、preserved visual context、reference-aware 和 anti-ASR。 |
| 8 | **Composed Video Retrieval via Enriched Context and Discriminative Embeddings**, CVPR 2024 | 用更丰富的语言上下文和判别性 embedding 改善 CoVR，说明 modification 表达质量与候选区分性同样重要。 | 支撑保存 audio evidence、生成具体 edit text，并用 hard gallery 检查 edit 是否具有区分性。 |
| 9 | **Localizing Events in Videos with Multimodal Queries**, 2024 | 使用多模态 query 在长视频中定位事件，强调局部时序语义而非整段全局相似度。 | 支撑 6-9 秒切片、同源局部片段和未来 temporal/local embedding 分析。 |
| 10 | **Data Roaming and Quality Assessment for Composed Image Retrieval**, AAAI 2024 | 同时研究数据扩展和自动质量评估，发布 LaSCo，说明规模扩张必须配套质量评分。 | 对应本项目的 blind gates、tier 分层和 Audio-CVR-640 quality summary；不能只报告生成数量。 |
| 11 | **Zero-Shot Composed Text-Image Retrieval**, BMVC 2023 | 从通用 image-text 数据生成/利用 pseudo triplets，在没有人工 triplet 的情况下训练 CIR。 | 支撑自动构造 B-extended，但其 noisy pseudo-label 风险也说明 B-main 需要更强审核。 |
| 12 | **Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval**, CVPR 2024 | 用大模型生成图像对之间的 visual delta，并用半监督数据提高 CIR。 | 与 audio delta analysis 最相似。可借鉴“先提取差异，再生成 modification”的两阶段叙事。 |
| 13 | **Multimodal Composition Example Mining for Composed Query Image Retrieval**, TIP 2024 | 从未标注数据中挖有价值的 composition examples，而不是随机产生 triplets。 | 支撑 Audio-CVR 的 source-aware pair mining 和候选质量排序。 |
| 14 | **Bi-directional Training for Composed Image Retrieval via Text Prompt Learning**, WACV 2024 | 同时学习 forward 和 reversed query，增强模型对 modification direction 的理解。 | 证明 inverse augmentation 有研究依据；但 test 中同组正反方向必须隔离，避免泄漏。 |
| 15 | **CompoDiff: Versatile Composed Image Retrieval with Latent Diffusion**, 2023 | 用生成模型/扩散先验表达组合目标并扩展合成训练数据。 | 可放在自动数据生成 Related Work；本项目采用自然视频片段而非生成 target，这是重要区别。 |

### 2.2 Supervised 组合建模

| # | 论文 | 核心精华 | 对 Audio-CVR 的直接借鉴 |
|---:|---|---|---|
| 16 | **Effective Conditioned and Composed Image Retrieval Combining CLIP-Based Features**, CVPR 2022 | 在预训练 CLIP 表征上学习轻量 combiner，证明冻结强 backbone 加小模块是有效路线。 | 直接支持 frozen E5-Omni + projection adapter 的 baseline 定位。 |
| 17 | **ARTEMIS: Attention-Based Retrieval with Text-Explicit Matching and Implicit Similarity**, ICLR 2022 | 同时建模文本显式要求和 reference-target 隐式相似性。 | Audio-CVR 也需要“edit 满足度 + preserved context”双重条件。 |
| 18 | **SAC: Semantic Attention Composition for Text-Conditioned Image Retrieval**, WACV 2022 | 用语义注意力选择 reference 中应保留或修改的内容。 | 启发 future model 对声音事件和视觉上下文进行选择性组合，但不需要当前就加入复杂模块。 |
| 19 | **Image Search with Text Feedback by Visiolinguistic Attention Learning**, CVPR 2020 | 通过局部视觉-语言注意力识别 modification 相关区域。 | 支撑局部 segment/patch 诊断，尤其是 edit 只对应短时声音事件时。 |
| 20 | **Dual Compositional Learning in Interactive Image Retrieval**, AAAI 2021 | 用双向/循环约束强化组合关系和可逆性。 | 说明 direction consistency 是 composed retrieval 的结构性问题，可用于讨论 reference negative。 |
| 21 | **TRACE: Transform, Aggregate and Compose Visiolinguistic Representations for Image Search with Text Feedback**, AAAI 2021 | 将组合拆成变换、聚合、组合三个步骤，增强解释性。 | 写方法时可借鉴清晰的阶段命名；本项目的数据 pipeline 也应按阶段说明各自解决的错误。 |
| 22 | **FashionVLP: Vision Language Transformer for Fashion Retrieval with Feedback**, CVPR 2022 | 用领域预训练和多任务学习提升 feedback retrieval。 | 表明领域适配很重要；adapter 的作用应写成 task adaptation，而非新 backbone。 |
| 23 | **Target-Guided Composed Image Retrieval**, ACM MM 2023 | 在训练中利用 target 信息指导 query composition 和困难样本学习。 | 支撑训练时使用 target/reference 相对关系，但评估时不能泄露 target。 |
| 24 | **Decompose Semantic Shifts for Composed Image Retrieval**, 2023 | 把 modification 分解成需改变语义和需保留语义。 | 与 Audio-CVR 的“audio changes, visual context preserved”高度一致，可用于 formal task definition。 |
| 25 | **Sentence-Level Prompts Benefit Composed Image Retrieval**, ICLR 2024 | 用更自然的句级 prompt 改善预训练模型对组合 query 的理解。 | edit text 格式应自然、具体、可验证，避免标签式或空泛短语。 |

### 2.3 Zero-shot 与 Training-free CIR

| # | 论文 | 核心精华 | 对 Audio-CVR 的直接借鉴 |
|---:|---|---|---|
| 26 | **Pic2Word: Mapping Pictures to Words for Zero-Shot Composed Image Retrieval**, CVPR 2023 | 将 reference image 映射为伪词，再与 modification text 通过 CLIP 文本空间组合。 | 展示了把非文本模态转换到语言组合空间的可行性，可作为 adapter 之外的 future baseline。 |
| 27 | **Zero-Shot Composed Image Retrieval with Textual Inversion (SEARLE)**, ICCV 2023 | 用 textual inversion 将图像概念表示为可与 edit 组合的文本 token。 | 可用于 Related Work 中“reference-to-language”路线，并与本项目的 audio evidence text 化区分。 |
| 28 | **iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image Retrieval**, 2024 | 提高 textual inversion 的效率和泛化，减少逐样本优化成本。 | 提醒论文讨论可扩展性：数据构造和检索都应避免逐 query 昂贵优化。 |
| 29 | **Context-I2W: Mapping Images to Context-Dependent Words for Accurate Zero-Shot Composed Image Retrieval**, AAAI 2024 | 让伪词表示依赖 modification context，而不是固定描述 reference。 | 对 Audio-CVR 的启示是 audio representation 应受 edit 条件影响，而不是仅编码全局音频。 |
| 30 | **Denoise-I2W: Mapping Images to Denoising Words for Accurate Zero-Shot Composed Image Retrieval**, CVPR 2024 | 过滤 reference 中与 modification 无关的内容，减少伪词噪声。 | 对应 Audio-CVR 中抑制视觉主导和无关背景音。 |
| 31 | **Image2Sentence-Based Asymmetrical Zero-Shot Composed Image Retrieval**, ICLR 2024 | 将 reference 转成句子后在语言域完成组合，采用非对称 query/target 建模。 | 与 agent/caption 路线相近；可作为不训练 adapter 的语言中介 baseline。 |
| 32 | **Language-Only Efficient Training of Zero-Shot Composed Image Retrieval**, CVPR 2024 | 用语言数据模拟组合训练，降低对视觉 triplets 的依赖。 | 可用于说明为什么 text-only baseline 必须报告，以及它不能替代真实 audio necessity。 |
| 33 | **MoTaDual: Modality-Task Dual Alignment for Enhanced Zero-Shot Composed Image Retrieval**, CVPR 2024 | 同时对齐模态差异和任务差异，提高 zero-shot 泛化。 | 与 E5-Omni 的 cross-modal alignment 形成呼应，可用于 embedding Related Work。 |
| 34 | **Vision-by-Language for Training-Free Compositional Image Retrieval (CIReVL)**, ICLR 2024 | 先 caption reference，再用 LLM 按 edit 改写 caption，最后做文本检索；模块化且可解释。 | 可作为 future agent baseline；同时提醒 full-AV 阶段不能反过来污染 audio-only edit text。 |
| 35 | **Knowledge-Enhanced Dual-Stream Zero-Shot Composed Image Retrieval**, CVPR 2024 | 用外部知识和双流建模补足 reference/edit 中缺失的信息。 | 可用于 Related Work，但当前 Audio-CVR 应优先证明数据和协议，而不是堆知识模块。 |
| 36 | **HyCIR: Boosting Zero-Shot Composed Image Retrieval with Synthetic Labels**, 2024 | 将 synthetic labels 与真实/预训练数据混合，兼顾规模和噪声控制。 | 对应 B-extended 用于训练、B-main 用于干净评估的分层思想。 |

### 2.4 数据扩展、伪标签与噪声控制

| # | 论文 | 核心精华 | 对 Audio-CVR 的直接借鉴 |
|---:|---|---|---|
| 37 | **Pseudo Triplet Guided Few-Shot Composed Image Retrieval**, 2024 | 用少量真实 triplets 引导大量 pseudo triplets，缓解标注稀缺。 | Audio-CVR-640 可用少量人工抽检校准自动 gate；论文必须如实说明当前尚无正式人工验证。 |
| 38 | **good4cir: Generating Detailed Synthetic Captions for Composed Image Retrieval**, CVPR 2025 | 重点提升 synthetic modification caption 的细节和可判别性。 | 直接支持禁止空泛 edit text，并把 edit specificity 作为数据质量指标。 |
| 39 | **Scale Up Composed Image Retrieval Learning via Modification Text Generation**, CVPR 2025 | 通过大规模 modification text generation 扩展 CIR 训练。 | 可借鉴规模化文本生成，但 Audio-CVR 必须坚持 edit 只来自 audio-only evidence。 |
| 40 | **CoLLM: A Large Language Model for Composed Image Retrieval**, CVPR 2025 | 利用 LLM 进行组合推理和检索表示学习，增强复杂 modification 理解。 | 可放在 future model；当前论文不应让 LLM agent 抢走数据构造与 protocol 主线。 |

### 2.5 Hard Negatives、Audio-Visual 与 Omni-modal Embeddings

| # | 论文 | 核心精华 | 对 Audio-CVR 的直接借鉴 |
|---:|---|---|---|
| 41 | **VSE++: Improving Visual-Semantic Embeddings with Hard Negatives**, BMVC 2018 | 证明 hardest in-batch negatives 对跨模态检索至关重要，但也会放大 false negative 风险。 | 支撑 reference/local/typed hard gallery，同时要求 false-negative guard。 |
| 42 | **ImageBind: One Embedding Space to Bind Them All**, CVPR 2023 | 将 image、text、audio 等六种模态对齐到统一空间，并展示跨模态 embedding arithmetic。 | 是 Audio-CVR 的重要通用 baseline；也提醒以 image 为桥可能让视觉主导声音。 |
| 43 | **LanguageBind: Extending Video-Language Pretraining to N-Modality by Language-Based Semantic Alignment**, 2023 | 以 language 为共享锚点对齐 video、audio、depth 等模态，并构建 VIDAL-10M。 | 支撑统一音频、视频、文本空间；可用于比较 language-anchored 与 omni-VLM embedding。 |
| 44 | **Everything at Once: Multi-Modal Fusion Transformer for Video Retrieval**, 2022 | 对单模态、模态对和多模态组合进行 combinatorial training，使模型测试时处理任意模态子集。 | 七种 audio necessity 消融的理论参照；不同模态组合应在同一评估协议中比较。 |
| 45 | **AVLnet: Learning Audio-Visual Language Representations from Instructional Videos**, ECCV 2020 | 从 raw audio、video 和 text 学共享表征，并分析 speech 与自然声音的贡献。 | 直接支撑“音频不等于 ASR”；Audio-CVR 应按 speech/music/sound-event 分项报告。 |
| 46 | **VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset**, 2023 | 联合建模 vision、audio、language，并构造三模态预训练/评测数据。 | 可作为 V+A+T baseline 和数据构造参照，尤其是 audio-on/off 的两侧同步。 |
| 47 | **VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset**, NeurIPS 2023 | 用 VAST-27M 自动生成大规模 vision-audio-subtitle-text 数据，统一多种 video-language 任务。 | 展示自动大规模 omni-modal annotation 的可能性，也提醒 subtitle/ASR 与真实非语音音频要分开。 |
| 48 | **VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks**, 2024 | 提出 MMEB 和将 VLM 转成统一向量模型的 contrastive 框架。 | 为“冻结强 VLM/Omni backbone，再训练 embedding adapter”提供直接依据。 |
| 49 | **VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents**, 2025 | 将 MMEB 扩展到视频、时序定位和文档，强调统一模型需覆盖不同输入形态。 | 可作为正式视频 embedding baseline；其 video tasks 比只支持 image 的 embedding 更公平。 |
| 50 | **e5-omni: Explicit Cross-Modal Alignment for Omni-Modal Embeddings**, 2026 | 用 modality-aware temperature、negative curriculum/debiasing、batch whitening 和 covariance regularization显式对齐多模态 embedding。 | 当前 backbone 与训练 recipe 的核心来源。论文必须说明这些组件来自 e5-omni，不作为 Audio-CVR 原创贡献。 |

## 3. 五篇核心论文精读与写作借鉴

### 3.1 CoVA：最接近的竞争工作

本地 PDF：[`01_CoVA_Text_Guided_Composed_Video_Retrieval_for_Audio_Visual_Content.pdf`](../paper/01_CoVA_Text_Guided_Composed_Video_Retrieval_for_Audio_Visual_Content.pdf)

### 它是怎么写问题的

CoVA 的 Introduction 很直接：

```text
现有 CoVR 只关注 visual modifications
-> 真实视频还包含具有判别力的 audio variations
-> 因此定义 audio-visual composed retrieval
-> 构建 AV-Comp
-> 提出 AVT fusion baseline
```

这种写法的优点是 gap、task、dataset、model 四步紧密相连，没有先讲大量泛化背景。

### 它最值得学习的地方

1. 用一张 teaser 直观对比“视觉相似但声音不同”的视频。
2. Task、dataset、method 三条贡献互相支撑。
3. 不只给 overall R@K，还通过 audio/visual aspect 证明多模态价值。
4. 使用 human-verified test，提高 benchmark 可信度。

### Audio-CVR 应如何区别

不能只写“我们也加入 audio”。应明确：

- CoVA 处理 audio 和 visual 的 cross-modal changes；Audio-CVR 的 B-main 是 audio-primary edit；
- Audio-CVR 强制 preserved visual context；
- edit text 必须来自 audio-only evidence；
- video-only shortcut 是独立拒绝阶段；
- reference negative 是正式 gallery 必选项；
- speech 样本具有 anti-ASR tiering；
- 评价包含 target-beats-reference 和 target-reference gap。

### 不应照搬的地方

Audio-CVR-640 当前没有正式人工验证，因此不能用与 CoVA 相同强度的“high-quality benchmark”措辞。必须写成 small-scale automatically curated diagnostic dataset，并将自动门控与人工验证明确区分。

### 3.2 CoVR：自动数据构造的主要模板

本地 PDF：[`02_CoVR_A_Benchmark_for_Composed_Video_Retrieval.pdf`](../paper/02_CoVR_A_Benchmark_for_Composed_Video_Retrieval.pdf)

### 论文结构精华

CoVR 将昂贵的 triplet 标注拆成：

```text
已有 video-caption pairs
-> 按 caption 语义挖相似视频对
-> LLM 生成 target 相对 reference 的 modification
-> 大规模自动训练集
-> 独立人工评估集
```

它没有把 LLM 本身包装成核心算法，而是强调它解决了 triplet 标注不可扩展的问题。

### Audio-CVR 应吸收的写法

1. 先说明人工标 audio-delta triplets 为什么昂贵。
2. 用 pipeline diagram 展示规模化构造。
3. 把训练集和测试集的质量标准分开写。
4. 报告每阶段候选数、通过率和拒绝原因。
5. 通过公开 benchmark transfer 或 strong baseline 证明数据有用。

### Audio-CVR 比 CoVR 多出的关键步骤

CoVR 的 caption similarity + LLM modification 不足以保证声音必要。Audio-CVR 必须突出：

```text
audio delta first
audio-only edit generation
audio-only reference/target verification
video-only shortcut rejection
full-AV consistency
```

这五步应成为 Dataset Construction 的主图和主要方法贡献。

### 3.3 EgoCVR：Local/Global 评估的模板

本地 PDF：[`03_EgoCVR_An_Egocentric_Benchmark_for_Composed_Video_Retrieval.pdf`](../paper/03_EgoCVR_An_Egocentric_Benchmark_for_Composed_Video_Retrieval.pdf)

### 它解决了什么问题

EgoCVR 认为普通 global gallery 不能充分检测细粒度时序变化，因此从同一 egocentric sequence 构建 local candidates。其 2,295 个 query 专门强调 temporal understanding。

### 最值得借鉴的实验写法

1. global 与 local 分表报告，而不是混成一个 overall。
2. 明确 local gallery 的平均/最大候选规模。
3. 给出细粒度 query 类别和失败案例。
4. 先证明现有模型在新 benchmark 上失败，再提出 reranking baseline。

### 对 Audio-CVR 的直接修改

Audio-CVR 的 local 必须更严格：

- 同一 raw source；
- 排除 reference 和 target；
- 候选不满足 edit；
- `uncertain` 不进入正式负例；
- strict local 和跨 source fallback 分开统计。

同时必须报告 local coverage。Audio-CVR-640 只有 10.75% 的 B-main query 拥有严格同源局部候选，不能仅报告 local recall。

### 3.4 CIRR：真实场景与 False Negative 处理模板

本地 PDF：[`04_CIRR_Image_Retrieval_on_Real_Life_Images_with_Pretrained_Vision_Language_Models.pdf`](../paper/04_CIRR_Image_Retrieval_on_Real_Life_Images_with_Pretrained_Vision_Language_Models.pdf)

### 它的核心价值

CIRR 将 CIR 从 fashion/synthetic 数据带到真实自然图像，并注意到大 gallery 中可能存在多个合理 target。它通过 candidate groups 和 subset metrics 更精细地分析模型。

### Audio-CVR 应借鉴的点

1. 明确“唯一标注 target”不等于“唯一语义正确 target”。
2. hard negative 必须做 false-negative guard。
3. global R@K 之外增加 group/reference-aware 指标。
4. 测试 query 的语言质量与 gallery 构成都要人工或高质量审核。

### 可直接改写进论文的方法学句子

```text
Because multiple clips may satisfy a broad audio modification, candidate negatives are admitted only when the audio-only verifier confirms that they do not satisfy the edit; uncertain candidates are excluded from the main evaluation.
```

### 3.5 e5-omni：Baseline 和训练描述模板

本地 PDF：[`05_e5_omni_Explicit_Cross_modal_Alignment_for_Omni_modal_Embeddings.pdf`](../paper/05_e5_omni_Explicit_Cross_modal_Alignment_for_Omni_modal_Embeddings.pdf)

### 它的核心问题分解

e5-omni 把 omni-modal embedding 的问题归纳为：

1. 不同模态 similarity sharpness 不一致；
2. mixed-modality negatives 难度失衡；
3. 不同模态 embedding 的统计几何不匹配。

对应三类训练机制：

```text
modality-aware temperature calibration
controllable negative curriculum with debiasing
batch whitening and covariance regularization
```

### Audio-CVR 应如何写

当前论文只需写：

> We freeze the E5-Omni backbone and train a lightweight projection adapter using its compatible embedding recipe.

这三类训练机制属于 e5-omni recipe，不是 Audio-CVR 原创贡献。Audio-CVR 的贡献应落在数据构造、reference-aware protocol 和 audio necessity evidence。

### 实验写法可借鉴之处

- 报告不同 modality combinations，而不是只报 full input；
- 检查 embedding score calibration；
- 使用 grouped metrics；
- 对每个 recipe component 做清楚 ablation，但 Audio-CVR 第一版不必把这些 ablation 当主表。

## 4. 从 50 篇论文中提炼的写作方法

### 4.1 小规模数据集的投稿策略

Audio-CVR-640 不应与 CoVR、WebVid 等工作竞争数据规模。现有证据更适合将它定位为：

> A small-scale, automatically curated diagnostic dataset for testing directional audio edits under preserved visual context.

写作时遵守以下边界：

1. **不声称大规模 benchmark**：标题、摘要和贡献中直接写 640 triplets，并说明 B-main 为 372 条。
2. **强调诊断价值**：1000-item gallery、mandatory reference negative、strict local candidates 和 typed hard negatives 比单纯样本数更重要。
3. **强调构造方法**：audio-only edit generation、video-only shortcut rejection 和 full-AV consistency verification 是数据方法贡献。
4. **如实描述验证等级**：数据是 automatically curated，尚未完成正式人工验证；`manual_review_required=0` 不等于 human-validated。
5. **严格描述评估集**：正式结果使用 source-disjoint 的 train507/val65/test68；旧 128-query val+test 合并池只作为开发诊断。
6. **报告重复实验**：正式 V+T 与 V+A+T 使用 seeds 13/23/42，报告 mean ± std，不再用单 seed 充当主结果。
7. **不争 SOTA**：主要结论是现有 embedding 在 reference directionality 上失败，以及 audio-on 相比 audio-off 的受控收益。
8. **用限制增强可信度**：主动报告来源不均衡、正式 test 无 strict local、test 含 5 条 extended、没有 music，以及 typed negatives 偏易。

这一路线借鉴 CIRR、EgoCVR 和 CoVA 对评估难度与诊断性的重视，但不照搬它们对规模或人工验证的主张。

### 4.2 Abstract 的五句结构

推荐按照以下顺序写：

1. **背景**：Composed video retrieval uses a reference video and modification text to retrieve a target.
2. **缺口**：Existing benchmarks either focus on visual modifications or fail to test whether audio is actually needed.
3. **方法/数据**：We introduce Audio-CVR and an audio-first multi-stage construction pipeline.
4. **协议**：We use reference-aware, same-source and typed-hard-negative galleries plus seven modality ablations.
5. **结果**：直接报告正式证据：在 source-disjoint 68-query、1000-item test 上，三种子 V+A+T 平均 R@1 为 24.51 ± 1.83%，高于 V+T 的 17.16 ± 1.83%；`target_beats_reference` 从 25.98 ± 5.41% 提升到 34.80 ± 3.02%。

不要在摘要中堆所有工程模块，也不要把 Audio-CVR-640 写成大规模或人工验证 benchmark。数字可以使用，但必须和 small-scale、automatically curated、diagnostic 的限定同时出现。

### 4.3 Introduction 的六段结构

### 第 1 段：任务价值

介绍 composed retrieval 比 text-only/video-only search 更自然，因为用户可以用 reference 表达保留内容，用 edit 表达变化。

### 第 2 段：现有缺口

引用 CoVR、EgoCVR、CoVA：现有 CVR 逐渐关注视频和细粒度变化，但 audio contribution、ASR shortcut 和 reference directionality 仍需更严格隔离。

### 第 3 段：为什么这个问题难

强调三种捷径：

```text
visual similarity shortcut
ASR/topic shortcut
audio-only presence shortcut
```

再引出 reference negative：reference 最相似，但尚未满足 edit。

### 第 4 段：我们的解决方案

用一段话概括 audio-first pipeline 和五阶段 blind review，不在 Introduction 展开阈值细节。

### 第 5 段：Benchmark protocol

说明 reference-aware、strict local、typed hard negatives、source-disjoint split 和七模态消融。

### 第 6 段：贡献

只列三条：数据构造、评估协议、adapter baseline 与 audio necessity evidence。

### 4.4 Related Work 的组织方式

不要按年份流水账，建议四个小节：

```text
Composed Image and Video Retrieval
Automatic Triplet Construction and Quality Control
Audio-Visual Retrieval and Omni-Modal Embeddings
Hard-Negative and Reference-Aware Evaluation
```

每个小节最后必须写一句与本项目的差异，而不是只总结别人。

示例：

> Unlike prior automatic triplet construction that derives modifications from visual captions, our edit text is generated exclusively from audio-only evidence and is subsequently checked against muted video and full audiovisual context.

### 4.5 Dataset/Method 的写作方式

从 CoVR、Visual Delta Generator、LaSCo 和 CoVA 中可提炼出一条共同原则：**数据论文必须展示证据链，而不是只给最终数量。**

Audio-CVR 应报告：

```text
raw videos
eligible clips
source groups
candidate pairs
audio-delta pass
audio-only verification pass
video-only shortcut rejection
full-AV pass
B-main/B-extended/B-diagnostic
human pass/uncertain/fail
```

并给出每阶段代表性正例和拒绝案例。

### 4.6 Experiments 的写作方式

推荐主表顺序：

1. Dataset statistics and human verification；
2. Base E5/ImageBind/LanguageBind/VLM2Vec-type baselines；
3. Adapter baseline；
4. Seven-mode audio necessity；
5. Random vs reference-aware；
6. Global vs strict local；
7. Hard-negative breakdown；
8. speech/music/sound-event breakdown。

指标优先级：

```text
R@1 / R@5 / R@10
target_beats_reference
reference_rank_median
target-reference gap
positive beats each negative type
```

### 4.7 结果解释的写法

### Random 高、reference-aware 低

不要写“模型失败”。应写：random gallery 高估性能，reference-aware protocol 暴露 edit directionality 的真实难点。

### V+A+T 高于 V+T

只能说明 audio 在该 protocol 和该 split 中提供额外信息，不能泛化为所有 CVR 都需要 audio。

### A-only 接近 V+A+T

说明数据可能退化成 audio-only 或 ASR retrieval，需要降级或按 shortcut label 单独报告。

### Typed negatives 接近 100%

说明负例不够难，不能包装成模型很强。应继续挖 nearest-neighbor negatives，并把 reference negative 作为核心诊断。

## 5. 对 Audio-CVR 论文各章节的引用映射

| 论文章节 | 重点引用 | 目的 |
|---|---|---|
| Introduction | CoVR、EgoCVR、CoVA | 建立 video composed retrieval 与 audio gap |
| Task Definition | TIRG、FashionIQ、CIRR | 解释 reference + modification 的相对检索定义 |
| Dataset Construction | CoVR、Visual Delta Generator、LaSCo、good4cir | 支撑自动 triplet、delta generation 和质量控制 |
| Anti-ASR/Audio | CoVA、AVLnet、VALOR、VAST | 区分真实 audio cues、speech 和 subtitle/ASR |
| Evaluation Protocol | CIRR、EgoCVR、VSE++ | 支撑 group/local、hard negative 和 false-negative guard |
| Baselines | ImageBind、LanguageBind、VLM2Vec-V2、e5-omni | 建立公平的 omni-modal embedding 对照 |
| Agent/Training-free Baseline | CIReVL、Image2Sentence | 提供 caption/LLM recomposition 路线 |
| Limitations | CoVA、CoVR、CIRR | 对照人工验证规模与 benchmark 可信度 |

## 6. 最值得从五篇核心论文“拿走”的东西

### 从 CoVA 拿走

- 用 teaser 一眼展示声音决定 target；
- 正面承认它是最接近工作；
- 用任务边界而非“首个”争夺新颖性；
- 把 audio/visual modality analysis 放进主实验。

### 从 CoVR 拿走

- 自动训练数据与人工 test 分离；
- 把构造 pipeline 写成核心方法；
- 用规模、过滤率和 transfer/baseline 证明数据价值。

### 从 EgoCVR 拿走

- local/global 双协议；
- 报告 local gallery 规模和 coverage；
- 使用同源难例证明模型理解细粒度变化。

### 从 CIRR 拿走

- 防止把合理 target 当 negative；
- global R@K 与 group-aware 指标并列；
- 强调真实自然数据中的语义歧义。

### 从 e5-omni 拿走

- baseline 训练 recipe 要准确归因；
- 分模态报告和 score calibration；
- 简洁、可解释的 adapter baseline 比堆很多未验证模块更可信。

## 7. 最终写作原则

1. **不要争错误的 first claim**：CoVA 已经研究 audio-visual composed retrieval。
2. **把差异落到 protocol**：audio-only edit、preserved visual context、anti-ASR、mandatory reference negative。
3. **把数据构造当方法写**：五阶段 blind review 是论文核心，不是工程附录。
4. **测试集质量比训练集规模更重要**：Audio-CVR-640 必须清楚披露自动审核边界，不能暗示已完成正式人工验证。
5. **Reference 是主难例**：这是当前最独特且证据最强的发现，应进入标题、Introduction 和主实验。
6. **Audio necessity 要做控制变量**：核心比较是 V+A+T vs V+T，而不是只展示 full model 最高。
7. **不要隐藏负结果**：typed negatives 过易、local coverage 低，应作为 protocol 改进动机。
8. **adapter 是强 baseline，不是唯一贡献**：论文竞争力主要来自数据构造和 reference-aware benchmark。

## 8. 核心论文链接

- [Awesome Composed Multi-modal Retrieval](https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval)
- [CMR Survey](https://arxiv.org/abs/2503.01334)
- [TIRG](https://openaccess.thecvf.com/content_CVPR_2019/html/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.html)
- [CoVR](https://ojs.aaai.org/index.php/AAAI/article/view/28334)
- [EgoCVR](https://arxiv.org/abs/2407.16658)
- [CoVA](https://arxiv.org/abs/2601.22508)
- [CIRR](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html)
- [Data Roaming and Quality Assessment](https://ojs.aaai.org/index.php/AAAI/article/view/28081)
- [Visual Delta Generator](https://openaccess.thecvf.com/content/CVPR2024/html/Jang_Visual_Delta_Generator_with_Large_Multi-modal_Models_for_Semi-supervised_Composed_CVPR_2024_paper.html)
- [CIReVL](https://proceedings.iclr.cc/paper_files/paper/2024/hash/48fd58527b29c5c0ef2cae43065636e6-Abstract-Conference.html)
- [ImageBind](https://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper)
- [LanguageBind](https://arxiv.org/abs/2310.01852)
- [Everything at Once](https://arxiv.org/abs/2112.04446)
- [AVLnet](https://arxiv.org/abs/2006.09199)
- [VALOR](https://arxiv.org/abs/2304.08345)
- [VAST](https://arxiv.org/abs/2305.18500)
- [VLM2Vec](https://arxiv.org/abs/2410.05160)
- [VLM2Vec-V2](https://arxiv.org/abs/2507.04590)
- [e5-omni](https://arxiv.org/abs/2601.03666)
