# Audio-CVR 数据筛选、难负例构造与 Audio 必要性证明规范

日期：2026-05-23

## 1. 目标

这份文档规定当前 Audio-CVR，尤其是 B-line Audio-Primary CVR，应该如何规范筛数据、构造难负例，以及如何证明 audio 模态在任务中是必要的。

核心任务定义：

```text
query = reference video/audio + edit_text
target = target video/audio
```

模型需要判断：target 是否是 reference 按照 edit_text 发生音频变化后的结果。

因此，数据集不能退化成以下几类任务：

1. 纯视觉检索：不听声音也能找到 target。
2. 纯 ASR 检索：只靠逐字 transcript 就能找到 target。
3. 普通相似视频检索：只找和 reference 最像的视频。
4. 随机 gallery 检索：target 与 distractor 太容易区分，无法暴露 audio edit 难点。

一句话定义：

```text
B-line 不是 audio determines target，
而是 audio edit under preserved video context determines target。
```

### 1.1 与相关工作的规范对齐

本协议对齐 composed retrieval 的通用设置：给定 reference input 和 modification/edit text，在 gallery 中检索 target。`Awesome-Composed-Multi-modal-Retrieval` 汇总的 CMR 文献说明，该方向的关键问题包括数据构造、组合语义建模、hard negative 与评估协议。

与本项目直接相关的设计依据：

1. **CoVR**：通过相似视频/文本对和 LLM 生成 modification，说明自动构造 video composed retrieval triplets 是可行路线。
2. **EgoCVR**：同时使用 global gallery 与 local/same-video gallery，说明只用随机 gallery 不足以证明模型理解 composed edit。
3. **COVA**：明确将 audio aspect 纳入 composed audio-visual retrieval，并通过 hard negatives 分析音频和视觉的组合需求。

因此，Audio-CVR 的正式 benchmark 不能只报告 random gallery 的 R@K，而必须报告 reference-aware、local same-source、typed hard-negative 和 audio necessity 消融结果。

---

## 2. 数据筛选规范

### 2.1 B-line 样本基本条件

每条 B-line 样本必须满足：

1. reference 和 target 有可识别的视频语境。
2. 视觉上下文大体保留。
3. 声音内容存在明确变化。
4. edit_text 只描述声音变化。
5. reference 不满足 edit_text。
6. target 满足 edit_text。
7. 不听声音时，不能稳定确定 target。

推荐记录字段：

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
  "full_av_consistency": {}
}
```

### 2.2 三类有效 B-line 样本

#### 2.2.1 speech_topic_in_video_context

有效形式：

```text
同一新闻/比赛/教程/访谈/直播/产品演示上下文中，
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

#### 2.2.2 music

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

#### 2.2.3 sound_event

有效形式：

```text
同一或相似视觉场景中，出现/消失/替换明显声音事件。
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
sound_event 必须可人工听辨，不能是模型猜测的 vague hum/click/tone。
```

---

## 3. 分层筛选规范

不要把所有 accepted 样本混成一个集合。应分成：

1. `B-main`
2. `B-extended`
3. `B-diagnostic`

### 3.1 B-main

主 benchmark 使用，要求最干净。

建议门槛：

```text
audio_delta_strength >= 0.70
video_context_strength >= 0.45
asr_degeneracy_risk <= 0.55
visual_shortcut_risk = false 或 <= 0.35
audio_only_verification.accept = true
video_only_shortcut.can_identify_target_without_audio = false
reference_satisfies_edit = false
target_satisfies_edit = true
```

额外建议：

```text
speech_topic_in_video_context <= 35%~40%
music + sound_event >= 60%
```

原因：speech 样本过多时，容易被质疑为 ASR retrieval。

### 3.2 B-extended

训练集使用，可以比 B-main 宽。

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
训练 AudioDelta-E5；
提供更多 audio-language supervision；
不直接作为主 benchmark 结果。
```

### 3.3 B-diagnostic

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
分析模型是否走 ASR shortcut；
分析 audio-only retrieval 是否已经足够；
作为附录或 failure mode。
```

---

## 4. edit_text 规范

### 4.1 必须具体

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

### 4.2 禁止视觉描述混入 B-line edit_text

拒绝包含以下主导信息的 edit_text：

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

B-line 的 edit_text 必须回答：

```text
声音从什么变成什么？
target 中有什么声音，而 reference 中没有？
reference 中有什么声音，在 target 中被移除或替换？
```

---

## 5. 难负例构造规范

Audio-CVR 不能只靠随机 distractor。随机 distractor 太容易，会让 Base E5 看起来虚高。

每条 query 至少应考虑以下负例类型。

### 5.1 reference_negative

定义：

```text
reference video 本身。
```

作用：

```text
测试模型是否理解 edit direction。
```

这是当前 pilot 发现的最重要 hard case。

必须满足：

```text
reference 不应满足 edit_text；
target 应满足 edit_text；
模型应满足 score(query, target) > score(query, reference)。
```

训练和评估都应固定加入 reference negative。

### 5.2 visual_hard

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

### 5.3 audio_hard

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

### 5.4 asr_hard

定义：

```text
speech 关键词或主题相似，
但不是正确 target。
```

作用：

```text
防止模型退化成 ASR / transcript matching。
```

### 5.5 random_distractor

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

### 5.6 正式测试 gallery 类型

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

`local_same_source` 是最关键的难度来源，因为 reference、target 和同源 distractors 视觉/语境相近，模型必须依赖 edit direction 和 audio delta 才能排序。

同源 negative 必须经过 false-negative guard：

```text
如果候选片段也满足 edit_text，不能作为 negative。
如果 audio-only verifier / metadata 显示 verification_accept=true，不能作为 negative。
如果无法确认，则进入人工复核或记录 missing reason，不要伪造 negative。
```

当前代码产物：

```text
b_main_eval_gallery_global.jsonl
b_main_eval_gallery_local_same_source.jsonl
b_main_eval_gallery_hardneg.jsonl
benchmark_quality_summary.json
audio_necessity_eval_manifest.json
```

---

## 6. 难负例记录格式

建议每条样本保存：

```json
{
  "audio_delta_hard_negatives": [
    {
      "type": "reference_negative",
      "video": "...",
      "reason": "reference does not satisfy edit_text"
    },
    {
      "type": "visual_hard",
      "video": "...",
      "reason": "visual context similar but audio edit absent"
    },
    {
      "type": "audio_hard",
      "video": "...",
      "reason": "audio cue similar but video context different"
    },
    {
      "type": "asr_hard",
      "video": "...",
      "reason": "speech keywords similar but not target"
    }
  ],
  "hard_negative_missing_reasons": {
    "visual_hard": null,
    "audio_hard": "not found in same source group",
    "asr_hard": "sample is non-speech"
  }
}
```

如果挖不到某类 hard negative，不要伪造，应记录 missing reason。

---

## 7. 证明 Audio 模态必要性的评估协议

### 7.1 必须做的对照

至少报告以下输入模式：

| 模式 | query | gallery |
|---|---|---|
| V-only | reference video muted + edit_text | target videos muted |
| A-only | reference audio + edit_text | target audios |
| V+T | reference video + edit_text | target videos |
| A+T | reference audio + edit_text | target audios |
| V+A | reference video/audio without edit_text | target video/audio |
| V+A+T | reference video/audio + edit_text | target video/audio |

关键原则：

```text
如果测试 audio-on，就 query 和 gallery 两侧都必须 audio-on；
如果测试 audio-off，就 query 和 gallery 两侧都必须 audio-off。
```

不能只关 query 的 audio，也不能只关 gallery 的 audio。

### 7.2 B-main 的理想趋势

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

### 7.3 必须加入 reference negative

评估 gallery 至少应包含：

```text
target positive
reference negative
random distractors
```

更强版本还应包含：

```text
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

### 7.4 Audio necessity manifest

每次正式评估都应保存 `audio_necessity_eval_manifest.json`，明确列出：

```text
V-only
A-only
V+T
A+T
V+A
V+A+T
```

以及每种模式对应的 query/gallery 输入开关。该 manifest 的作用是避免实验时只关闭 query audio 或只关闭 gallery audio，导致 audio 消融不规范。

推荐报告结构：

```text
主表：B-main / local_same_source / typed_hardneg
附表：global random distractor
分项：speech_topic_in_video_context / music / sound_event
诊断：B-diagnostic / ASR-like / visual-shortcut / audio-only-solvable
```

`random` 或 `pilot_only_random_distractor_gallery` 只能作为 smoke/pilot 结果，不应作为论文主 benchmark。

---

## 8. 证明 Audio 必要性的样本级诊断

每条样本应能回答以下问题。

### 8.1 不听声音能否找到 target？

如果只看画面就能确定 target：

```text
visual_shortcut_risk = true
```

该样本不能进入 B-main。

### 8.2 只听声音是否就能找到 target？

如果只靠 target audio/transcript 就能找到 target，且视频上下文没有贡献：

```text
asr_degeneracy_risk = high
audio_only_solvability = high
```

该样本应进入 B-diagnostic 或 B-extended，而不是 B-main。

### 8.3 reference 是否也满足 edit_text？

如果 reference 也满足 edit_text：

```text
reference_satisfies_edit = true
```

样本应拒绝。

### 8.4 target 是否明确满足 edit_text？

如果 target 只弱满足或模型无法说明证据：

```text
target_satisfies_edit = false
```

样本应拒绝或进入人工复核池。

---

## 9. 推荐报告指标

主结果不应只报告 overall R@K。至少应报告：

### 9.1 Overall Retrieval

```text
R@1
R@5
R@10
Median Rank
Mean Rank
```

### 9.2 Reference Directionality

```text
target_beats_reference
reference_rank_median
reference_rank <= 1
target-reference score gap mean
target-reference score gap distribution
```

### 9.3 Subtype Breakdown

```text
speech_topic_in_video_context
music
sound_event
```

### 9.4 Shortcut Breakdown

```text
clean_audio_delta
ASR-like
visual-shortcut
audio-only-solvable
ambiguous
```

### 9.5 Hard Negative Recall

```text
positive beats reference_negative
positive beats visual_hard
positive beats audio_hard
positive beats asr_hard
```

---

## 10. 当前 1% Pilot 给出的经验

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

---

## 11. 后续训练与数据构造建议

### 11.1 数据构造

继续全量构造 B-line，但不要过早删除高风险样本。应做：

```text
先收集 -> 打标签 -> 分层 -> 再决定用途
```

对应用途：

```text
B-main: 主 benchmark
B-extended: 训练
B-diagnostic: shortcut / ASR-risk 分析
```

### 11.2 训练

第一版 baseline 使用 e5-omni recipe。

下一步最小 Stage-2 网格：

```text
e5_omni_recipe
+ L_ref
+ L_ref + L_delta
+ L_ref + L_delta + edit_type
```

判断标准：

```text
R@1
target_beats_reference
reference_rank_median
target-reference score gap
audio_event R@1
speech R@1
```

### 11.3 评估

所有 pilot 和正式评估都应保留：

```text
reference negative
hard negative by type
audio-on/off 双侧对照
per-query top-k 错例
```

当前实现中，`prepare` 支持以下 gallery protocol：

```text
random
reference
local_same_source
typed_hardneg
audio_necessity
```

其中 `reference`、`local_same_source`、`typed_hardneg`、`audio_necessity` 都会默认加入 reference negative。`eval --save-topk` 需要输出 `negative_type`、`same_source`、`satisfies_edit`，用于检查 top-k 错例到底是 reference、同源片段、typed hard negative，还是普通 random distractor。

---

## 12. 最终原则

Audio-CVR 的可信性来自三个方面：

1. **数据可信**：edit_text 只描述声音变化，reference 不满足，target 满足。
2. **负例可信**：reference、visual_hard、audio_hard、asr_hard 都被系统构造和报告。
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
