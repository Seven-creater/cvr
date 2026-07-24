# Omni Composed Video Retrieval 数据构造项目总结

更新时间：2026-04-23

本文档总结当前项目已经完成的工作、遇到的关键难题、对应解决方案，以及数据从原始数据集到中间数据集，再到最终目标数据集的构造路径。

这份文档只写最新状态，不复述所有零散实验日志。

## 1. 项目目标

我们现在的目标不是继续做普通的 video-text retrieval，而是构造一个适合 Omni 全模态组合视频检索的数据集。

目标任务形式是：

```text
reference video + edit text + visual/audio cues -> target video
```

也就是说，系统输入一段参考视频和一段编辑文本，例如：

```text
change one cat into two cats
add dog barking in the background
change the person from standing still to dancing
remove the lower-third speaker label
replace quiet ambient hum with electronic music
```

然后模型需要从候选视频库里找到目标视频。

这个任务的关键不是单独理解一个视频，也不是单独理解一句文本，而是判断：

- reference video 原本有什么。
- edit text 要改变什么。
- target video 是否真正体现了这个改变。
- reference 和 target 是否仍然处在相似上下文里。
- 这个改变是否需要视觉、音频、语音、屏幕文字等多模态证据支撑。

最终我们真正需要的不是普通的：

```text
video -> caption
text -> video
```

而是高质量三元组：

```text
reference_video
edit_text
target_video
```

## 2. 我们已经做了什么

### 2.1 早期 MSRVTT / AVIGATE 实验

项目一开始沿着 MSRVTT 和 AVIGATE 的普通检索路线推进。

我们做过：

- V2T：video to text retrieval。
- T2V：text to video retrieval。
- 使用 Qwen2.5-Omni 对初检结果做描述、理解和 rerank。
- 在 20 条、48 条等小规模样本上看 Omni rerank 对 recall 的影响。
- 尝试并发调用 Omni，提高实验效率。

早期结果说明：

- V2T 里 Omni 有时可以提升 R@1。
- 但 R@5 / R@10 不稳定，甚至会下降。
- T2V 里 query rewrite 和 video description 有一定作用，但在 MSRVTT 上提升有限。
- MSRVTT 本身并不是为 `reference video + edit text -> target video` 设计的。

这一步的结论很重要：继续在 MSRVTT 上堆 agent 不是最优方向。

### 2.2 从“改 agent”转向“构造新数据集”

我们后来明确了一个方向：

如果任务是 Omni 全模态 composed video retrieval，那么最先要解决的是数据问题。

我们不应该只依赖 MSRVTT 这种普通 caption 数据集，而应该构造新数据：

```text
同一上下文里的两个短视频片段
+ 它们之间的明确差异
+ 根据差异生成的 edit_text
+ 多模态证据
= composed retrieval 样本
```

这也是你前面说的关键判断：如果数据构造得足够合理，agent 不需要特别复杂。我们只需要证明 agentic retrieval 方法能利用这些数据和证据即可。

### 2.3 下载和准备原始数据源

我们准备了两个主要原始数据源：

```text
Daily-Omni
WorldSense
```

服务器统一数据根目录：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
```

已经归一化出的统计：

```text
source rows: 4368
unique clips: 2858
Daily-Omni: 1196 rows / 1196 clips
WorldSense: 3172 rows / 1662 clips
WorldSense archives extracted: 13
```

生成的核心文件：

```text
metadata/source_rows.jsonl
metadata/source_clips_all.jsonl
metadata/source_clips_pilot50.jsonl
reports/source_dataset_prepare_summary.md
```

### 2.4 下载和使用的 Omni 模型

服务器上已有模型包括：

```text
Qwen2.5-Omni-7B
Qwen2-VL-7B-Instruct
Qwen2-Audio-7B-Instruct
Qwen3-Omni-30B-A3B-Instruct
Qwen3-Omni-30B-A3B-Captioner
Qwen3-Omni-30B-A3B-Thinking
```

主要路径：

```text
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-captioner
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-thinking
```

当前实际跑通的主流程主要使用：

```text
Qwen3-Omni-30B-A3B-Instruct
```

服务端口：

```text
http://127.0.0.1:8093/v1
```

GPU 资源策略已经确定：

- 当前优先只用 2 张空闲 GPU。
- 即使以后资源充裕，单次实验也尽量不超过 6 张 GPU。
- 不同时加载 Instruct、Captioner、Thinking 三个 Omni 服务。
- 用完一个模型服务再关掉，再启动下一个阶段需要的模型。

## 3. 遇到的核心难题和解决过程

### 难题 1：MSRVTT 不适合 composed retrieval

MSRVTT 的样本大多是普通视频和普通 caption 的对应关系。

它不能自然提供：

- reference video。
- edit text。
- target video。
- reference 和 target 的局部差异。
- 音频、语音、屏幕文字等多模态编辑线索。

所以在 MSRVTT 上继续跑 Omni rerank，只能证明 Omni 对普通检索有一点帮助，不能支撑我们的核心任务。

解决方式：

我们转向构造新数据集，从 Daily-Omni 和 WorldSense 这类更丰富的视频数据源里构造 composed retrieval 样本。

### 难题 2：全局随机配对导致上下文不一致

早期 pilot50 里，我们把不同视频随机配对，再让模型生成 edit_text。

问题是 reference 和 target 往往不是同一个上下文。

例如：

```text
reference: 一个人在演讲
target: 乐队在舞台演奏
edit_text: 改变场景或人物
```

这种样本看起来有差异，但不像真实 composed retrieval。因为 reference + edit 并不能自然指向 target。

解决方式：

我们改成 Omni-Detective 风格：

```text
长视频 / 原始视频
-> 切成短事件 clip
-> 同源视频片段组成 group
-> 只在 group 内构造 pair
```

这样 reference 和 target 更可能来自同一个视频、同一主题、同一场景或相邻事件。

### 难题 3：reference 和 target 太像，甚至一模一样，也会被接受

人工抽查时发现，有些 pair 的 reference 和 target 几乎没有区别。

这类样本不需要 edit_text，也能匹配 target，因此不合格。

解决方式：

我们加入了三重 verification：

1. `caption_delta`
   - 判断 reference caption 和 target caption 是否等价。
   - 如果两者等价，直接 reject。
   - 必须存在具体差异。
   - 差异必须和 edit_text 对应。

2. `edit_projection`
   - 输入 reference caption + edit_text。
   - 生成理论上的 projected target caption。
   - 判断真实 target caption 是否符合这个 projected caption。

3. `edit_necessity`
   - 判断 edit 是否真的必要。
   - reference 不能已经满足 edit。
   - target 必须满足 edit。

最终只有满足下面条件的 pair 才能通过：

```text
reference != target
reference 不满足 edit
target 满足 edit
edit 是必要条件
差异具体且能被证据支撑
```

### 难题 4：旧 judge 分数会误杀好样本

我们发现一些样本通过了 verification，但模型 judge 的 `edit_match_score` 偏低。

如果完全相信 judge 分数，会错杀一些实际合理的样本。

解决方式：

引入 verification override：

如果三重 verification 全部通过，并且 quality 的其它关键约束达标，那么允许覆盖旧 judge 的低 `edit_match_score`。

这样不是放松质量，而是用更细的验证逻辑纠正粗粒度 judge。

### 难题 5：same_context 只看语义相似，不理解同源相邻片段

两个同源视频片段可能文字描述不完全相似，但它们在时间上相邻，确实属于同一上下文。

解决方式：

加入 temporal source context：

- 同一 source video 的相邻片段获得更高 source context。
- pair selection 优先选择同源、相邻、上下文连续的片段。

这一步显著提高了 same_context 的可靠性。

### 难题 6：类型覆盖不够，accepted 样本偏向 object_presence

早期 accepted 样本容易集中在：

```text
object_presence
```

比如“某个物体出现/消失”。

这类样本有价值，但我们最终需要覆盖：

```text
object_count
object_presence
attribute
action
scene
audio_event
speech
visible_text
```

解决方式：

我们加入 difference type bucket selection：

- 先保留原生 primary difference 候选。
- 再为缺少的类型补充 retargeted candidates。
- 避免为了多样性牺牲原本高质量候选。

最新小实验已经从只接受 object_presence，推进到：

```text
object_presence: 3
speech: 1
attribute: 1
```

但 action 类型仍然不足，这是下一步重点。

### 难题 7：target uniqueness 定义不适合 composed retrieval

之前 `target_uniqueness_score` 的逻辑是：

```text
target 和其它 clip 越像，target uniqueness 越低
```

这对普通检索合理，但对 composed retrieval 不合理。

因为 hard negative 本来就应该很像 target。只要 hard negative 不满足 edit，它就是好负例。

解决方式：

我们把 target uniqueness 改成 difference-aware：

- 如果其它 clip 很像 target，但不满足同一个 edit，只弱惩罚。
- 如果其它 clip 也满足同一个 edit，才强惩罚。

这次修改后，实验结果明显改善：

```text
accepted_count: 1 -> 5
accepted_and_verification_passed: 5
accepted_but_verification_failed: 0
target_uniqueness_avg: 0.778
```

这说明这个方向是对的。

## 4. 原始数据集长什么样

### 4.1 Daily-Omni

Daily-Omni 原始形式主要是 parquet 行数据。

一行大致包含：

```json
{
  "video_id": "Ec_lQgZ9wlg",
  "video": "<embedded video bytes>",
  "audio": "<embedded audio bytes>",
  "question": "Which audio event occurred simultaneously with ...?",
  "candidates": [
    "A. Sharp aerodynamic swoosh",
    "B. Foreign-language command phrases",
    "C. Numerical countdown sequence",
    "D. Sustained gunfire bursts"
  ],
  "answer": "D. Sustained gunfire bursts"
}
```

特点：

- 视频和音频通常嵌在 parquet 里。
- 任务形式偏向多模态问答。
- 很适合提取音频事件、语音和视觉同步关系。
- 但它原本不是 composed retrieval 数据，需要重新构造 pair。

归一化后，Daily-Omni 的 video/audio 被物化到文件系统：

```text
raw/daily_omni/video/...
raw/daily_omni/audio/...
```

### 4.2 WorldSense

WorldSense 原始形式包含：

```text
videos zip archives
subtitles
video_caption
question
candidates
answer
```

一条样本大致类似：

```json
{
  "video": "videos/BUCEfyAF.mp4",
  "subtitle_path": "./subtitles/BUCEfyAF.srt",
  "video_caption": "A woman sits at a piano ... another person plays trumpet ...",
  "question": "What instrument is played by the woman in the red checkered skirt?",
  "candidates": [
    "A. A French horn.",
    "B. A piano.",
    "C. A trumpet.",
    "D. A tuba."
  ],
  "answer": "C"
}
```

特点：

- 视频以 zip 分片方式存储。
- 有较长的 video caption。
- 有字幕和问答信息。
- 很适合找同类事件、音乐、演讲、舞台、教学等场景。

归一化时已经自动解压 13 个 WorldSense zip：

```text
raw_datasets/worldsense/_extracted/videos_chunk_*/videos/*.mp4
```

## 5. 现在的数据集长什么样

现在的数据不是最终训练/评测数据，而是一组中间层文件。它们记录了从原始视频到 composed retrieval pair 的每一步。

### 5.1 `source_rows.jsonl`

路径：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/metadata/source_rows.jsonl
```

作用：保存原始数据集的每一行，以及对应的视频、音频、文本字段。

简化结构：

```json
{
  "source_row_id": "daily_omni_8ed045ab7b786292",
  "dataset": "daily_omni",
  "split": "train",
  "row_index": 1,
  "source_file": ".../test-00000-of-00010.parquet",
  "video_id": "Ec_lQgZ9wlg",
  "video_path": ".../raw/daily_omni/video/test-00000-of-00010_1_video.mp4",
  "audio_path": ".../raw/daily_omni/audio/test-00000-of-00010_1_audio.wav",
  "text_fields": {
    "question": "...",
    "candidates": ["A. ...", "B. ..."],
    "answer": "B. ..."
  },
  "raw_columns": ["answer", "audio", "candidates", "question", "video", "video_id"]
}
```

### 5.2 `source_clips_all.jsonl`

路径：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/metadata/source_clips_all.jsonl
```

作用：把所有可用视频整理成统一的 source clip manifest。

简化结构：

```json
{
  "clip_id": "daily_omni_Ec_lQgZ9wlg",
  "source_path": ".../raw/daily_omni/video/test-00000-of-00010_1_video.mp4",
  "output_path": "raw/daily_omni/video/test-00000-of-00010_1_video.mp4",
  "start_seconds": 0.0,
  "end_seconds": 0.0,
  "duration_seconds": 0.0,
  "role": "source_clip",
  "dataset": "daily_omni",
  "source_row_ids": ["daily_omni_8ed045ab7b786292"],
  "text_fields": {
    "question": "...",
    "candidates": ["A. ...", "B. ..."],
    "answer": "B. ..."
  }
}
```

注意：这里很多还是 whole video，不是最终短事件片段。

### 5.3 `clip_plan_detective.jsonl`

作用：根据 source clips 生成事件切片计划。

简化结构：

```json
{
  "clip_id": "daily_omni_xxx__seg_001",
  "source_clip_id": "daily_omni_xxx",
  "source_path": ".../raw/daily_omni/video/xxx.mp4",
  "output_path": "clips/detective/daily_omni/daily_omni_xxx__seg_001.mp4",
  "start_seconds": 0.0,
  "end_seconds": 8.0,
  "duration_seconds": 8.0,
  "dataset": "daily_omni",
  "group_id": "group_daily_omni_xxx",
  "group_reason": "same_source_video"
}
```

默认切片逻辑：

- 4 到 12 秒左右事件片段。
- 硬上限 15 秒。
- 小于 3 秒的片段丢弃。
- 保留音频。

### 5.4 `clip_groups.jsonl`

作用：记录哪些短 clip 属于同一个上下文组。

简化结构：

```json
{
  "group_id": "group_daily_omni_xxx",
  "dataset": "daily_omni",
  "group_reason": "same_source_video",
  "source_clip_ids": ["daily_omni_xxx"],
  "candidate_clip_ids": [
    "daily_omni_xxx__seg_001",
    "daily_omni_xxx__seg_002",
    "daily_omni_xxx__seg_003"
  ],
  "group_tags": ["speech", "indoor", "object"]
}
```

目前 pair 主要只在 group 内构造，避免跨数据集随机配对。

### 5.5 `extracted_event_clips.jsonl`

作用：记录真正切出来的短视频片段。

简化结构：

```json
{
  "clip_id": "daily_omni_xxx__seg_001",
  "source_path": ".../raw/daily_omni/video/xxx.mp4",
  "output_path": "clips/detective/daily_omni/daily_omni_xxx__seg_001.mp4",
  "start_seconds": 0.0,
  "end_seconds": 8.0,
  "duration_seconds": 8.0,
  "dataset": "daily_omni",
  "group_id": "group_daily_omni_xxx"
}
```

### 5.6 `detective_annotations.jsonl`

作用：对每个短 clip 做 Omni-Detective 风格的细粒度多模态标注。

简化结构：

```json
{
  "clip_id": "daily_omni_xxx__seg_001",
  "output_path": "clips/detective/daily_omni/daily_omni_xxx__seg_001.mp4",
  "summary": "A man speaks to the camera while holding a rope.",
  "storyline": [
    {
      "start": 0.0,
      "end": 4.0,
      "visual": "The man faces the camera.",
      "audio": "Speech is heard.",
      "objects": ["man"],
      "actions": ["speaking"]
    }
  ],
  "events": [
    {
      "visual": "A rope is raised into view.",
      "audio": "The man continues speaking.",
      "objects": ["man", "rope"],
      "actions": ["holding", "showing"]
    }
  ],
  "visible_text": [],
  "speakers_and_transcript": [],
  "audio_events": ["speech"],
  "subjects": ["man"],
  "object_counts": {
    "man": 1,
    "rope": 1
  },
  "scene": "indoor room",
  "attributes": ["white wall"],
  "modalities": ["visual", "audio"],
  "uncertainties": [],
  "detective_trajectory": [
    {"tool": "media_probe", "observation": "..."},
    {"tool": "visual_observer", "observation": "..."},
    {"tool": "audio_observer", "observation": "..."},
    {"tool": "detective_agent", "decision": "..."}
  ],
  "fallback_used": false
}
```

这一步是后续 pair construction 的基础。

### 5.7 `judged_pair_proposals.jsonl`

作用：保存候选 pair、模型 judge、verification 和最终是否 accepted。

简化结构：

```json
{
  "proposal_id": "proposal__xxxx",
  "group_id": "group_daily_omni_xxx",
  "reference_video": "clips/detective/...seg_001.mp4",
  "target_video": "clips/detective/...seg_002.mp4",
  "edit_text": "A mushroom appears in the man's hand.",
  "modalities": ["visual", "audio"],
  "reference_caption": "...",
  "target_caption": "...",
  "difference": {
    "type": "object_presence",
    "from": "no mushroom",
    "to": "1 mushroom",
    "description": "mushroom appears in the target clip"
  },
  "quality": {
    "same_context_score": 0.9,
    "edit_match_score": 0.85,
    "target_uniqueness_score": 0.862,
    "difference_strength_score": 0.67
  },
  "verification": {
    "passed": true,
    "failures": [],
    "caption_delta": {
      "caption_equivalent": false,
      "has_concrete_difference": true,
      "difference_matches_edit": true
    },
    "edit_projection": {
      "target_matches_projection": true,
      "score": 0.9
    },
    "edit_necessity": {
      "edit_needed": true,
      "reference_satisfies_edit": false,
      "target_satisfies_edit": true,
      "score": 0.95
    }
  },
  "accepted": true
}
```

### 5.8 `accepted_pairs.jsonl`

作用：最终 pilot 数据样本。只有通过 judge / verification / quality gates 的样本会进入这里。

这一步已经接近我们真正需要的数据格式。

## 6. 我们真正需要的数据集长什么样

最终样本应该是这样的：

```json
{
  "sample_id": "covr_omni_000001",
  "reference_video": "clips/detective/daily_omni/xxx__seg_001.mp4",
  "target_video": "clips/detective/daily_omni/xxx__seg_002.mp4",
  "edit_text": "A mushroom appears in the man's hand.",
  "modalities": ["visual", "audio"],
  "reference_caption": "A man speaks to the camera in a forest setting.",
  "target_caption": "A man speaks to the camera while holding a mushroom.",
  "difference": {
    "type": "object_presence",
    "from": "no mushroom",
    "to": "1 mushroom",
    "description": "a mushroom appears in the target clip"
  },
  "hard_negatives": [
    "clips/detective/daily_omni/xxx__seg_003.mp4",
    "clips/detective/daily_omni/xxx__seg_004.mp4"
  ],
  "quality": {
    "same_context_score": 0.9,
    "edit_match_score": 0.85,
    "target_uniqueness_score": 0.862,
    "difference_strength_score": 0.67
  },
  "verification": {
    "passed": true,
    "failures": [],
    "caption_delta": {
      "caption_equivalent": false,
      "has_concrete_difference": true,
      "difference_matches_edit": true
    },
    "edit_projection": {
      "projected_target_caption": "...",
      "target_matches_projection": true,
      "score": 0.9
    },
    "edit_necessity": {
      "edit_needed": true,
      "reference_satisfies_edit": false,
      "target_satisfies_edit": true,
      "score": 0.95
    }
  },
  "evidence": {
    "difference_evidence": {
      "difference_type": "object_presence",
      "supporting_evidence": [
        "object_counts: mushroom 0 -> 1",
        "events: man speaking -> man holding mushroom"
      ]
    },
    "reference_storyline": ["..."],
    "target_storyline": ["..."],
    "audio_change": "...",
    "visible_text_change": "..."
  }
}
```

最终数据集必须满足：

- reference 和 target 要像，但不能一样。
- edit_text 必须是必要条件。
- reference 不能满足 edit。
- target 必须满足 edit。
- 差异必须具体，最好是单一主差异。
- 差异必须能在 annotation 证据中找到。
- hard negatives 必须接近，但不能满足同一个 edit。

## 7. 当前实验结论

### 7.1 pipeline 已经跑通

目前完整链路已经跑通：

```text
source_clips_all.jsonl
-> clip_plan_detective.jsonl
-> clip_groups.jsonl
-> extracted_event_clips.jsonl
-> detective_annotations.jsonl
-> judged_pair_proposals.jsonl
-> accepted_pairs.jsonl
-> pilot_review.md
```

这说明工程链路是可用的。

### 7.2 verification 生效

之前人工发现 reference / target 几乎一样也会被接受。

现在通过：

```text
caption_delta
edit_projection
edit_necessity
```

可以有效过滤这类样本。

最新结果里：

```text
accepted_and_verification_passed: 5
accepted_but_verification_failed: 0
```

说明 accepted 样本都通过了 verification。

### 7.3 target uniqueness 修正有效

之前 accepted_count 只有 1，主要是 target uniqueness 把很多 verification 通过的样本挡掉。

原因是旧 uniqueness 把相似 hard negative 当坏事。

修正后：

```text
accepted_count: 1 -> 5
target_uniqueness_avg: 0.778
accepted_difference_types:
  object_presence: 3
  speech: 1
  attribute: 1
```

这说明新的 difference-aware target uniqueness 更适合 composed retrieval。

### 7.4 当前还没有完全解决的问题

最新 pilot review 里仍然有一个失败项：

```text
action_samples_at_least_1: FAIL
```

也就是说，当前方法已经能稳定得到 object_presence、speech、attribute，但 action 类型还不够。

另外，accepted 中出现了两条类似：

```text
A mushroom appears in the man's hand.
```

这提示我们下一步可能需要加入：

- edit_text 去重。
- pair-level 去重。
- 同一目标差异的样本合并或只保留最优一条。

## 8. 为什么现在不应该继续重复跑实验

现在的实验结果已经说明：

- pipeline 能跑。
- verification 有效。
- target uniqueness 修正有效。
- accepted_count 已经从 1 提到 5。
- 但类型覆盖仍有短板，尤其 action。

所以如果不改方法，只继续重复跑同一套 smoke test，收益会很低。

接下来应该每次只验证一个明确方法假设。

例如：

```text
假设 1：action evidence 约束能提升 action 样本通过率。
假设 2：edit_text 去重能减少重复 mushroom 样本。
假设 3：Captioner 独立 audio observer 能提升 audio_event / speech 样本质量。
假设 4：Thinking verifier 只复核 accepted / borderline pair，能提高人工通过率。
```

不要为了“看起来有结果”盲目扩大跑量。现在更重要的是让每一轮实验回答一个具体问题。

## 9. 下一步方法优化方向

### 9.1 专门优化 action 类型

当前 action 类型没有 accepted。

建议下一步加入 action 专门约束：

```text
reference action != target action
reference storyline 支撑原动作
target storyline 支撑新动作
edit_text 必须描述动作变化，而不是完整 caption
```

可以新增 action-specific verification：

```json
{
  "reference_action_evidence": "...",
  "target_action_evidence": "...",
  "action_changed": true,
  "action_change_matches_edit": true
}
```

### 9.2 加入 edit_text 去重

如果多个 accepted pair 的 edit_text 几乎一样，例如：

```text
A mushroom appears in the man's hand.
```

可以只保留质量最高的一条。

去重维度：

- edit_text 语义。
- difference.type。
- difference.to。
- source group。

### 9.3 Captioner 单独做 audio observer

当前主要用 Qwen3-Omni Instruct 跑全流程。

后续可以阶段式使用 Captioner：

```text
启动 Qwen3-Omni Captioner
-> 只做 audio/video evidence annotation
-> 关闭 Captioner
-> 启动 Instruct 做 pair mining 和 judge
```

这符合 GPU 资源约束，也能提升 audio_event / speech 样本质量。

### 9.4 Thinking 只做最终复核

Qwen3-Omni Thinking 不应该全量跑。

建议只用于：

- accepted pairs。
- borderline pairs。
- 人工不确定样本。

它的任务是回答：

```text
reference 和 target 是否几乎一样？
edit 是否必要？
target 是否确实体现 edit？
是否存在多个主差异？
```

### 9.5 每次小实验只验证一个假设

推荐实验规模：

```text
20-30 个 source videos
40-80 个 short clips
5-10 条 accepted samples
```

每次只改一个方法点。

验收标准：

- 不出现 reference / target 一模一样。
- 每条 accepted 都有 verification.passed = true。
- 至少 5 条 accepted。
- 至少 2 条 object/object_count。
- 至少 1 条 action。
- 至少 1 条 audio_event 或 speech。
- 人工 review 通过率目标 >= 80%。

## 10. 当前状态一句话总结

我们已经从“在 MSRVTT 上尝试 Omni rerank”推进到了“能从 Daily-Omni / WorldSense 中自动切片、标注、分组、构造、验证 composed retrieval pilot 样本”的阶段。

当前最重要的进展是：

```text
reference 和 target 不再只要求像；
它们必须像但不等价，
必须存在明确差异，
这个差异必须由 edit_text 描述，
并且 target 必须唯一地满足这个 edit。
```

下一步不应该盲目扩大跑量，而应该围绕 action 样本、audio evidence、edit 去重和 Thinking 复核继续优化数据质量。
