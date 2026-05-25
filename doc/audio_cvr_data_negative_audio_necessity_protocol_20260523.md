# Audio-CVR 数据筛选、难负例构造与 Audio 必要性证明规范

日期：2026-05-24

## 0. 核心目标

本文档规定 B-line Audio-CVR 数据集应该如何规范筛数据、构造难负例，以及如何用实验严谨证明 audio 模态是必要的。

任务定义：

```text
query = reference video/audio + edit_text
target = target video/audio
```

模型需要判断 target 是否是 reference 按照 edit_text 发生声音变化后的结果。

B-line 的一句话定义：

```text
B-line 不是 audio determines target，
而是 audio edit under preserved video context determines target。
```

因此，数据集不能退化成以下任务：

1. 纯视觉检索：不听声音也能找到 target。
2. 纯 ASR 检索：只靠逐字 transcript 就能找到 target。
3. 普通相似视频检索：只找和 reference 最像的视频。
4. 随机 gallery 检索：target 与 distractor 太容易区分，无法暴露 audio edit 难点。

## 1. 相关工作对齐

Composed retrieval 的通用形式是 reference input + modification/edit text -> target retrieval。`Awesome-Composed-Multi-modal-Retrieval` 汇总的 CMR 文献说明，该方向的核心问题包括数据构造、组合语义建模、hard negative 和评估协议。

与本项目直接相关的规范依据：

1. **CoVR**：通过相似视频/文本对和 LLM 生成 modification，说明自动构造 video composed retrieval triplets 是可行路线。
2. **EgoCVR**：同时使用 global gallery 与 local/same-video gallery，说明只用随机 gallery 不足以证明模型理解 composed edit。
3. **COVA**：明确把 audio aspect 纳入 composed audio-visual retrieval，并通过 hard negatives 分析音频和视觉的组合需求。

对 Audio-CVR 的启发：

```text
正式 benchmark 不能只报告 random gallery 的 R@K。
必须报告 reference-aware、local same-source、typed hard-negative 和 audio necessity 消融结果。
```

## 2. 怎么规范筛数据

### 2.1 B-line 样本必须满足的条件

每条 B-line 样本必须同时满足：

1. reference 和 target 都有可识别的视频语境。
2. reference 和 target 的视觉上下文大体保留。
3. 声音内容存在明确、可人工听辨的变化。
4. edit_text 只描述声音变化，不能描述视觉变化。
5. reference 不满足 edit_text。
6. target 明确满足 edit_text。
7. 不听声音时，不能稳定确定 target。

推荐保存字段：

```json
{
  "reference_video": "...",
  "target_video": "...",
  "edit_text": "...",
  "b_subtype": "speech_topic_in_video_context | music | sound_event",
  "audio_delta_strength": 0.0,
  "video_context_strength": 0.0,
  "visual_shortcut_risk": 0.0,
  "asr_degeneracy_risk": 0.0,
  "audio_only_evidence": {},
  "video_only_shortcut": {},
  "full_av_consistency": {},
  "reference_satisfies_edit": false,
  "target_satisfies_edit": true
}
```

### 2.2 三类有效 B-line 样本

**speech_topic_in_video_context**

有效形式：

```text
同一新闻、比赛、教程、访谈、直播、产品演示等视频语境中，
说话主题、讲解步骤、解说内容发生变化。
```

好例子：

```text
change the commentary from introducing the players to describing the goal
change the tutorial narration from explaining ingredients to explaining the cooking step
change the interview topic from career history to future plans
```

坏例子：

```text
change the voice from saying "sentence A" to saying "sentence B"
speech content changed
change from unintelligible speech to another speech
```

原则：

```text
speech 可以存在，但必须是视频语境中的 speech，
不能变成孤立 transcript matching。
```

**music**

有效形式：

```text
同类视觉语境下，音乐风格、乐器、节奏、旋律、演唱内容发生变化。
```

好例子：

```text
replace soft piano music with upbeat pop music
change the background music from acoustic guitar to orchestral strings
add a short drum rhythm to the performance audio
```

坏例子：

```text
replace guitar music with similar guitar music
add music to the audio
change the tone of the sound
```

原则：

```text
music edit 必须具体到乐器、风格、节奏或可听事件，
不能只写“音乐变了”。
```

**sound_event**

有效形式：

```text
同一或相似视觉场景中，出现、消失或替换明显声音事件。
```

好例子：

```text
add crowd cheering to the match audio
replace quiet room ambience with machine noise
add water splashing sounds to the river scene
remove applause from the performance audio
```

坏例子：

```text
add target audio to the audio
add a sound
replace noise with noise
change the audio
```

原则：

```text
sound_event 必须可人工听辨，
不能是模型猜测的 vague hum/click/tone。
```

### 2.3 edit_text 规范

edit_text 必须回答：

```text
声音从什么变成什么？
target 中有什么声音，而 reference 中没有？
reference 中有什么声音，在 target 中被移除或替换？
```

允许：

```text
change the commentary from introducing the players to describing the goal
replace soft piano music with upbeat orchestral music
add crowd cheering to the match audio
remove applause from the performance audio
```

拒绝：

```text
speech content has been altered
change the speech from discussing A to discussing B
unintelligible speech
not transcribed
add target audio to the audio
replace sound with sound
```

禁止把视觉信息写成 B-line edit_text：

```text
smile
gesture
walking
button
screen
card front/back
close-up / wide shot
person appears/disappears
camera movement
subtitle / visible text
```

### 2.4 分层筛选

不要把所有 accepted 样本混成一个集合。应分成：

```text
B-main
B-extended
B-diagnostic
```

**B-main**

用于主 benchmark，要求最干净。

建议门槛：

```text
audio_delta_strength >= 0.70
video_context_strength >= 0.45
asr_degeneracy_risk <= 0.55
visual_shortcut_risk <= 0.35
audio_only_verification.accept = true
video_only_shortcut.can_identify_target_without_audio = false
reference_satisfies_edit = false
target_satisfies_edit = true
```

额外比例约束：

```text
speech_topic_in_video_context <= 35%~40%
music + sound_event >= 60%
```

原因：speech 样本过多时，容易被质疑为 ASR retrieval。

**B-extended**

用于训练，可以比 B-main 宽。

建议门槛：

```text
audio_delta_strength >= 0.60
video_context_strength >= 0.35
asr_degeneracy_risk <= 0.70
无明显视觉捷径
edit_text 不空洞
reference 不满足 edit
target 满足 edit
```

用途：

```text
训练 AudioDelta-E5，
提供更多 audio-language supervision，
不直接作为主 benchmark 结果。
```

**B-diagnostic**

诊断集，不进入主表。

包含：

```text
ASR-like speech
generic talking head
visual shortcut risk
audio-only solvability very high
ambiguous audio edit
transcript-like edit_text
```

用途：

```text
分析模型是否走 ASR shortcut，
分析 audio-only retrieval 是否已经足够，
作为附录或 failure mode。
```

### 2.5 Human Verification for B-main Test

B-main test 不能完全依赖 LLM 自动标注。CoVR 提供人工评估集，COVA 也强调 benchmark 质量和诊断作用；Audio-CVR 如果没有人工核验，容易被质疑为“LLM 生成的样本不可靠”。

人工核验不要求全量重标注，目标是“小规模但关键”：

```text
优先核验 B-main test；
优先核验 local_same_source 和 typed hard negatives；
争议样本进入 manual_review_required，不直接进主 benchmark。
```

B-main test 的人工核验至少检查：

1. `edit_text` 是否只描述音频变化。
2. reference 是否不满足 `edit_text`。
3. target 是否明确满足 `edit_text`。
4. 不听声音时是否难以确定 target。
5. local_same_source / typed hard negative 是否确实不满足 `edit_text`。

建议保存字段：

```json
{
  "manual_review_required": false,
  "manual_review_status": "not_needed | pending | passed | failed | uncertain",
  "manual_review_reason": null,
  "human_audio_edit_valid": true,
  "human_reference_satisfies_edit": false,
  "human_target_satisfies_edit": true,
  "human_video_only_can_identify_target": false,
  "human_negative_false_negative_risk": false
}
```

处理规则：

```text
manual_review_status = failed -> 不进入 B-main；
manual_review_status = uncertain -> 不进入正式主表，可进入复核池或 diagnostic；
hard negative 被人工判定满足 edit_text -> 从 negative gallery 移除。
```

## 3. 怎么规范构造难负例

Audio-CVR 不能只靠随机 distractor。随机 distractor 太容易，会让 Base E5 看起来虚高。

每条 query 至少应考虑以下负例类型。

### 3.1 reference_negative

定义：

```text
reference video 本身。
```

作用：

```text
测试模型是否理解 edit direction。
```

必须满足：

```text
reference 不应满足 edit_text；
target 应满足 edit_text；
模型应满足 score(query, target) > score(query, reference)。
```

这是当前 1% pilot 暴露出的最重要 hard case，训练和评估都应固定加入。

### 3.2 local_same_source

定义：

```text
同一个 raw source video 中，除 reference/target 外的其他片段。
```

作用：

```text
构造最接近真实困难度的 gallery。
这些片段和 reference/target 视觉语境接近，
模型必须依赖 audio edit 和 edit direction 排序。
```

注意：

```text
local_same_source 片段必须经过 false-negative guard。
如果该片段也满足 edit_text，就不能作为 negative。
```

采样优先级：

```text
local_same_source:
Level 1: 与 reference/target 时间相邻的同源 clip；
Level 2: 同源但非相邻 clip；
Level 3: 同一 source group / 同一视频事件下的 clip；

local_fallback_visual:
Level 4: 同 visual context 的跨 source clip；
Level 5: visual_hard 替代。
```

每个 local_same_source negative 必须保存：

```json
{
  "negative_type": "local_same_source",
  "temporal_relation": "adjacent_before | adjacent_after | same_source_non_adjacent | same_group",
  "satisfies_edit": false,
  "verification_status": "auto_verified | human_verified | uncertain",
  "missing_reason": null
}
```

如果使用跨 source fallback，则必须显式标记为 `negative_type=local_fallback_visual`，不能混写成严格的 `local_same_source`：

```json
{
  "negative_type": "local_fallback_visual",
  "temporal_relation": "cross_source_same_context | visual_hard_fallback",
  "satisfies_edit": false,
  "verification_status": "auto_verified | human_verified | uncertain",
  "missing_reason": "no_strict_local_same_source_candidate"
}
```

如果 `verification_status=uncertain`，该片段不能进入正式 B-main hard gallery，只能进入人工复核池或 diagnostic。这样可以避免把真正满足 edit 的同源片段误当成负例。

### 3.3 visual_hard

定义：

```text
视觉上下文和 target/reference 很像，
但音频 edit 不成立。
```

例子：

```text
同一比赛画面，但没有 crowd cheering；
同一厨房画面，但没有 boiling sound；
同一表演场景，但音乐不是目标风格。
```

作用：

```text
防止模型只靠画面相似度选答案。
```

### 3.4 audio_hard

定义：

```text
声音内容和 edit_text 相关，
但视频上下文不对。
```

例子：

```text
也有人群欢呼，但不是同一类比赛/场景；
也有机器噪声，但不是同一工作场景；
也有钢琴音乐，但不是同一表演/视觉语境。
```

作用：

```text
防止模型只靠 target audio 找答案。
```

### 3.5 asr_hard

定义：

```text
speech 关键词或主题相似，
但不是正确 target。
```

作用：

```text
防止模型退化成 ASR / transcript matching。
```

### 3.6 random_distractor

定义：

```text
从其他源视频随机采样的无关 clip。
```

作用：

```text
扩大 gallery，测试大候选池下的排序稳定性。
```

注意：

```text
random distractor 不能替代 hard negatives。
```

### 3.7 False-Negative Guard

任何 negative 候选都必须先确认“不满足 edit_text”。

拒绝作为 negative 的情况：

```text
候选片段也满足 edit_text
audio-only verifier 显示 verification_accept=true
metadata 中 satisfies_edit=true
无法确认是否满足 edit_text 且没有人工复核
```

如果挖不到某类 hard negative，不要伪造，应记录 missing reason。

### 3.8 难负例记录格式

建议每条样本保存：

```json
{
  "audio_delta_hard_negatives": [
    {
      "type": "reference_negative",
      "video": "...",
      "source_id": "...",
      "reason": "reference does not satisfy edit_text",
      "verification_accept": false,
      "satisfies_edit": false
    },
    {
      "type": "local_same_source",
      "video": "...",
      "source_id": "...",
      "temporal_relation": "adjacent_before | adjacent_after | same_source_non_adjacent | same_group",
      "reason": "same source but does not satisfy edit_text",
      "verification_accept": false,
      "satisfies_edit": false
    },
    {
      "type": "local_fallback_visual",
      "video": "...",
      "source_id": "...",
      "temporal_relation": "cross_source_same_context | visual_hard_fallback",
      "reason": "fallback because no strict local_same_source candidate exists",
      "verification_accept": false,
      "satisfies_edit": false
    },
    {
      "type": "visual_hard",
      "video": "...",
      "source_id": "...",
      "reason": "visual context similar but audio edit absent",
      "verification_accept": false,
      "satisfies_edit": false
    },
    {
      "type": "audio_hard",
      "video": "...",
      "source_id": "...",
      "reason": "audio cue similar but video context different",
      "verification_accept": false,
      "satisfies_edit": false
    },
    {
      "type": "asr_hard",
      "video": "...",
      "source_id": "...",
      "reason": "speech keywords similar but not target",
      "verification_accept": false,
      "satisfies_edit": false
    }
  ],
  "hard_negative_missing_reasons": {
    "visual_hard": null,
    "audio_hard": "not found in same source group",
    "asr_hard": "sample is non-speech"
  }
}
```

### 3.9 正式测试 Gallery

正式 benchmark 至少保留以下三套 gallery：

```text
global:
  target positive + reference negative + typed hard negatives + random distractors

local_same_source:
  target positive + reference negative + 同 raw source video 的其他片段
  若同源不足，退化到 visual_hard 或 same visual context candidates

typed_hardneg:
  target positive + reference negative + visual_hard + audio_hard + asr_hard
```

当前代码产物：

```text
b_main_eval_gallery_global.jsonl
b_main_eval_gallery_local_same_source.jsonl
b_main_eval_gallery_hardneg.jsonl
benchmark_quality_summary.json
audio_necessity_eval_manifest.json
```

## 4. 怎么规范证明 Audio 模态必要性

### 4.1 必须做的输入消融

至少报告以下输入模式：

| 模式 | query | gallery |
|---|---|---|
| V-only | reference video muted + edit_text | target videos muted |
| T-only-fullAV | edit_text only | 与主实验相同的 full AV gallery，作为文本先验 baseline |
| A-only | reference audio + edit_text | target audios |
| V+T | reference video + edit_text | target videos |
| A+T | reference audio + edit_text | target audios |
| V+A | reference video/audio without edit_text | target video/audio |
| V+A+T | reference video/audio + edit_text | target video/audio |

关键原则：

```text
如果测试 audio-on，query 和 gallery 两侧都必须 audio-on；
如果测试 audio-off，query 和 gallery 两侧都必须 audio-off。
```

不能只关 query 的 audio，也不能只关 gallery 的 audio。

### 4.2 B-main 的理想趋势

对 B-main，理想结果不是 audio-only 一定最高，而是：

```text
V-only 低
A-only 中等
V+A+T 最高
```

解释：

```text
V-only 低：证明只看画面不够。
A-only 中等：证明声音有用。
V+A+T 最高：证明任务需要 audio edit under video context。
```

如果出现：

```text
A-only ≈ V+A+T
```

则说明样本可能退化成 audio-only / ASR retrieval，应降级到 `B-diagnostic` 或 `B-extended`。

### 4.3 Audio Necessity 成立条件

证明 audio 模态必要，不能只看 `V+A+T` 是否最高。最关键比较是：

```text
V+T vs V+A+T
```

因为导师关心的是：audio 相比已有 video-text composed retrieval 是否提供额外价值。

Audio necessity 成立的建议条件：

1. `V+A+T` 显著高于 `V+T`。
2. `A+T` 显著高于 `T-only-fullAV` 或 random baseline。
3. `V-only` 在 `B-main / local_same_source / typed_hardneg` 上明显低。
4. `V+A+T` 的 `target_beats_reference` 最高。
5. audio-off 后 `target-reference score gap` 明显下降。
6. 如果 `A-only ≈ V+A+T`，该样本可能退化为 audio-only / ASR retrieval，应降级到 `B-diagnostic`。

正式报告时应至少给出：

```text
R@K(V+A+T) - R@K(V+T)
target_beats_reference(V+A+T) - target_beats_reference(V+T)
target_reference_score_gap(V+A+T) - target_reference_score_gap(V+T)
```

如果这些差值在 `local_same_source` 和 `typed_hardneg` 上仍然成立，才说明 audio 确实提供了 video-text 之外的有效信息。

### 4.4 必须加入 reference negative

评估 gallery 至少应包含：

```text
target positive
reference negative
random distractors
```

更强版本还应包含：

```text
local_same_source
visual_hard
audio_hard
asr_hard
```

核心指标：

```text
target_beats_reference
reference_rank_median
target_reference_score_gap
reference_negative_recall
```

如果模型在 random gallery 上 R@1 很高，但 target beat reference 很低，则说明模型仍未掌握 edit direction。

### 4.5 样本级诊断

每条样本应能回答以下问题：

1. 不听声音能否找到 target？
2. 只听声音是否就能找到 target？
3. reference 是否也满足 edit_text？
4. target 是否明确满足 edit_text？

对应处理：

```text
visual_shortcut_risk = true -> 不能进 B-main
audio_only_solvability = high -> 降级到 B-diagnostic 或 B-extended
reference_satisfies_edit = true -> 拒绝
target_satisfies_edit = false -> 拒绝或进入人工复核池
```

### 4.6 必须报告的指标

主结果不应只报告 overall R@K。至少应报告：

```text
R@1
R@5
R@10
Median Rank
Mean Rank
target_beats_reference
reference_rank_median
reference_rank <= 1
target-reference score gap mean
positive beats reference_negative
positive beats visual_hard
positive beats audio_hard
positive beats asr_hard
```

还必须按以下维度拆分：

```text
speech_topic_in_video_context
music
sound_event
clean_audio_delta
ASR-like
visual-shortcut
audio-only-solvable
ambiguous
```

### 4.7 Audio Necessity Manifest

每次正式评估都应保存 `audio_necessity_eval_manifest.json`，明确列出：

```text
T-only-fullAV
V-only
A-only
V+T
A+T
V+A
V+A+T
```

以及每种模式对应的 query/gallery 输入开关。

该 manifest 的作用是避免实验时只关闭 query audio 或只关闭 gallery audio，导致 audio 消融不规范。

推荐报告结构：

```text
主表：B-main / local_same_source / typed_hardneg
附表：global random distractor
分项：speech_topic_in_video_context / music / sound_event
诊断：B-diagnostic / ASR-like / visual-shortcut / audio-only-solvable
```

`random` 或 `pilot_only_random_distractor_gallery` 只能作为 smoke/pilot 结果，不应作为论文主 benchmark。

## 5. 当前 1% Pilot 给出的经验

1% pilot 加入 reference negative 后，结果为：

| 方法 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Base E5 | 6.67% | 100.00% | 100.00% |
| Adapter | 40.00% | 100.00% | 100.00% |

这个结果说明：

1. 随机 gallery 会严重高估 Base E5。
2. reference negative 是当前最关键 hard case。
3. Base E5 主要按视觉/语境相似度排序。
4. Adapter 能部分修复 directionality，但还不够。
5. 后续训练必须显式优化 `score(query, target) > score(query, reference)`。

因此，后续数据和评估都应把 reference negative 作为必备项。

## 6. 当前实现状态

当前 `prepare` 支持以下 gallery protocol：

```text
random
reference
local_same_source
typed_hardneg
audio_necessity
```

其中：

```text
reference
local_same_source
typed_hardneg
audio_necessity
```

都会默认加入 reference negative。

`eval --save-topk` 需要输出：

```text
negative_type
same_source
satisfies_edit
```

用于检查 top-k 错例到底是 reference、同源片段、typed hard negative，还是普通 random distractor。

## 7. 最终原则

Audio-CVR 的可信性来自三个方面：

1. **数据可信**：edit_text 只描述声音变化，reference 不满足，target 满足。
2. **负例可信**：reference、local_same_source、visual_hard、audio_hard、asr_hard 都被系统构造和报告。
3. **实验可信**：audio-on/off、reference-aware metrics、shortcut diagnosis 都能证明模型不是靠视觉或 ASR 捷径。

最终要证明的不是：

```text
模型能在随机视频里找到 target。
```

而是：

```text
当 reference 和 target 高度相似时，
模型必须利用 audio edit 才能判断哪个视频是真正的 target。
```
