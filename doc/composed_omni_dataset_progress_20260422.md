# Omni Composed Video Retrieval 数据构造进展

Last updated: 2026-04-22

## 1. 当前目标

本项目现在的核心目标已经从 MSRVTT 常规 video-text retrieval，转向构造一个更适合 **Omni 全模态组合视频检索** 的新数据集。

目标任务形式是：

```text
reference video + edit text + visual/audio cues -> target video
```

也就是说，系统输入一个参考视频和一段编辑文本，例如：

```text
change one cat into two cats
replace quiet background with dog barking
change the person from standing still to dancing
```

然后需要从候选视频库中找出满足该编辑要求的 target video。

这个任务需要同时覆盖：

- 视觉信息：主体、数量、动作、场景、属性、颜色、可见文字。
- 音频信息：音乐、说话声、掌声、动物叫声、机器声、环境声。
- 编辑文本：描述 reference 到 target 的关键变化。

我们前面已经确认，MSRVTT 更适合普通 video-text retrieval，不适合天然构造 `reference-target-edit` 三元组。因此现在的重点是构造新数据，而不是继续在 MSRVTT 上堆 agent 逻辑。

## 2. 参考方法：Omni-Captioner / Omni-Detective

参考仓库：

```text
https://github.com/ddlBoJack/Omni-Captioner
```

该项目的关键思想不是简单做一次视频 caption，而是用 **Omni-Detective** 做 agentic data generation。

Omni-Detective 的方法要点：

- 通过多轮 `Query-Observation` 迭代获取细粒度音视频信息。
- Detective Agent 主动规划需要观察什么。
- Tool Box 提供多种工具，用于从音频、视频、文字等模态提取证据。
- Independent Observers 面向原始音视频流回答局部问题。
- 最终综合成低幻觉、细粒度的 audio-visual annotation。

Omni-Captioner 推荐的结构化音视频描述至少包含三块：

```text
Storyline
Visible Text
Speakers and Transcript
```

这对我们非常重要，因为 composed retrieval 数据构造依赖的不是泛泛 summary，而是：

- 某个时间段发生了什么。
- 屏幕上出现了什么文字。
- 谁说了什么，语气和时间位置如何。
- 哪些音频事件和画面同步发生。
- reference 和 target 之间是否只有一个主要差异。

当前我们的代码已经开始模拟这个方向，但还没有真正完全复刻 Omni-Detective。现在只是一个早期的两步版本：

```text
observer pass -> detective final pass
```

它比单轮 caption 多了 `detective_trajectory / storyline / visible_text / speakers_and_transcript`，但还缺少真正的多工具、多观察者、多轮决策。

## 3. 服务器模型与数据状态

### 3.1 已准备模型

服务器已有模型目录：

```text
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-captioner
```

当前跑通数据构造链路的是 Instruct 服务：

```text
http://127.0.0.1:8093/v1
```

另一个 Thinking 模型已经安排从 ModelScope 下载：

```text
Qwen/Qwen3-Omni-30B-A3B-Thinking
```

计划用途：

- Instruct：数据构造总控、pair proposal、judge。
- Captioner：偏音频细粒度 caption，可用于补充 audio events / speech / acoustic scene。
- Thinking：更适合做 Detective Agent、pair judge、失败样本分析。

### 3.2 已准备原始数据

当前使用两个原始数据源：

```text
Daily-Omni
WorldSense
```

统一数据根目录：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
```

归一化后的统计：

```text
source rows: 4368
unique clips: 2858
pilot clips: 50
Daily-Omni: 1196 rows / 1196 clips
WorldSense: 3172 rows / 1662 clips
WorldSense archives extracted: 13
```

`source_clips_pilot50.jsonl` 已做数据源平衡：

```text
Daily-Omni: 25
WorldSense: 25
```

## 4. 已完成工程链路

已经新增并跑通的数据处理代码：

```text
app/composed_sources.py
app/composed_data.py
app/composed_omni.py
scripts/prepare_composed_sources.sh
scripts/run_composed_pilot50.sh
```

主要功能：

- `composed_sources.py`
  - 解析 Daily-Omni parquet。
  - 物化 Daily-Omni 内嵌视频和音频。
  - 解压 WorldSense zip。
  - 生成统一 source row 和 clip manifest。

- `composed_omni.py`
  - 调用 OpenAI-compatible Qwen3-Omni 服务。
  - 支持 single-pass clip annotation。
  - 支持 detective-style clip annotation。
  - 支持 pair proposal。
  - 对模型不稳定字段做归一化。

- `composed_data.py`
  - `annotate-clips`
  - `detective-annotate-clips`
  - `propose-pairs`
  - `validate-pilot`
  - gallery 构建和质量统计。

- `run_composed_pilot50.sh`
  - 一键跑 pilot50 标注、pair proposal、pilot 选择、校验和报告。

最近一次相关提交：

```text
b4e38e321cdc26546da1f5c1ce0e9236bb38d0e9
```

单元测试状态：

```text
Ran 43 tests
OK
```

## 5. 当前实验结果

### 5.1 single-pass / min-context 版本

早期单轮标注版本跑通后，主要结果是：

```text
clip_count: 50
annotated_count: 50
fallback_count: 1
proposal_count: 40
high_context_pool_count: 12
pilot_count: 10
same_context_min: 0.128
same_context_avg: 0.168
same_context_max: 0.233
```

优点：

- fallback 很少。
- proposal 数量较多。
- 上下文分数比后续 detective 初版更高。

问题：

- 标注偏单轮 summary，不够 Omni-Captioner。
- 缺少 trajectory、timeline、visible text、transcript 等证据链。
- pair 仍有不少只是“同数据集但语义不够相近”。

### 5.2 detective 初版

切到 `detective-annotate-clips` 后，标注明显更细：

```text
annotation_mode: detective
clip_annotations size: 605K
detective_trajectory: present
storyline: present
visible_text: present
speakers_and_transcript: present
```

但初版有明显失败：

```text
fallback_count: 9
proposal_count: 23
high_context_pool_count: 2
same_context_avg: 0.108
```

结论：

```text
detective 方向是对的，但两步 JSON 更容易失败，一失败就掉到空 fallback，导致候选池变小。
```

### 5.3 detective fallback 修复版

之后加入了 fallback 机制：

```text
detective failed -> single-pass annotation fallback
```

结果：

```text
fallback_count: 0
detective_to_single_pass_count: 9
proposal_count: 33
same_context_avg: 0.118
```

改进：

- 原本 9 条失败样本被 single-pass 救回来。
- 候选数从 23 回升到 33。

仍然不足：

- same_context_avg 仍然偏低。
- 真正高上下文候选不够。

### 5.4 high-context 选择版

最近一次高上下文筛选结果：

```text
git HEAD: b4e38e321cdc26546da1f5c1ce0e9236bb38d0e9
annotation_mode: detective
fallback_count: 0
detective_to_single_pass_count: 9
proposal_count: 33
high_context_pool_count: 9
selected_context_threshold: 0.1
pilot_count: 9
audio_count: 9
object_change_count: 4
action_count: 2
scene_count: 3
same_context_min: 0.109
same_context_avg: 0.141
same_context_max: 0.295
```

自动验收：

```text
sample_count_between_5_and_10: PASS
audio_samples_at_least_2: PASS
object_change_samples_at_least_2: PASS
action_samples_at_least_1: PASS
```

这说明工程链路已经比较稳，但效果仍然没有本质进展。

## 6. 当前判断

现在不建议继续重复跑 pilot50。

原因：

```text
同一批 50 条 pilot clips 的上限已经暴露出来。
重复跑只会在 0.11-0.17 的 same_context_avg 附近波动。
它不能解决数据构造方法本身的问题。
继续跑只是在浪费 Qwen3-Omni 服务时间和 GPU 资源。
```

当前真正的问题不是：

```text
模型没跑通
数据没读出来
schema 没写好
```

而是：

```text
我们还没有完全按 Omni-Captioner / Omni-Detective 的方式构造数据。
```

更具体地说，现在的不足是：

- clip 还是 whole source video，很多太长，事件混杂。
- pair 仍然主要来自全局候选筛选，不是同一上下文内局部差异。
- detective 只是两步 prompt，不是真正多轮 Query-Observation。
- 没有把 audio captioner、OCR、ASR、frame sampler、scene splitter 做成 Tool Box。
- 没有先做 temporal segmentation。
- 没有严格做 pair judge：reference 不满足 edit，target 满足 edit，hard negatives 接近但不满足。

所以接下来应该停止“重复跑 pilot50”，转向方法重构。

## 7. 当前数据结构

### 7.1 source_rows.jsonl

表示原始数据集中的一行：

```json
{
  "source_row_id": "daily_omni_xxx",
  "dataset": "daily_omni",
  "split": "train",
  "row_index": 1,
  "source_file": "...parquet",
  "video_id": "...",
  "video_path": "...mp4",
  "audio_path": "...wav",
  "text_fields": {
    "question": "...",
    "candidates": ["..."],
    "answer": "..."
  },
  "raw_columns": ["answer", "audio", "candidates", "question", "video", "video_id"]
}
```

### 7.2 source_clips_all.jsonl / source_clips_pilot50.jsonl

表示可用 clip。当前多数还是 whole source video：

```json
{
  "clip_id": "worldsense_ALfOUzDH",
  "source_path": "/data02/.../ALfOUzDH.mp4",
  "output_path": "raw_datasets/worldsense/_extracted/videos_chunk_003/videos/ALfOUzDH.mp4",
  "start_seconds": 0.0,
  "end_seconds": 0.0,
  "duration_seconds": 0.0,
  "role": "source_clip",
  "notes": "whole source video; run manual clipping before final pilot if this video is too long",
  "dataset": "worldsense",
  "source_row_ids": ["..."],
  "text_fields": {"question": "...", "answer": "..."}
}
```

### 7.3 clip_annotations_pilot50.jsonl

detective 版本的 annotation 大致是：

```json
{
  "clip_id": "...",
  "summary": "...",
  "subjects": ["..."],
  "object_counts": {"person": 1},
  "actions": ["..."],
  "scene": "...",
  "attributes": ["..."],
  "on_screen_text": ["..."],
  "speech": ["..."],
  "audio_events": ["..."],
  "modalities": ["visual", "audio"],
  "storyline": ["..."],
  "visible_text": ["..."],
  "speakers_and_transcript": ["..."],
  "detective_notes": ["..."],
  "detective_trajectory": [
    {"stage": "observer", "payload": "..."},
    {"stage": "detective_final", "payload": "..."}
  ],
  "detective_fallback_used": false
}
```

这一步已经开始接近 Omni-Captioner，但还不够。

### 7.4 pilot_10.jsonl

当前 pilot 样本格式：

```json
{
  "sample_id": "covr_pilot_0001",
  "reference_video": "...",
  "target_video": "...",
  "edit_text": "...",
  "modalities": ["visual", "audio"],
  "reference_caption": "...",
  "target_caption": "...",
  "difference": {
    "type": "object_presence",
    "from": "...",
    "to": "...",
    "description": "..."
  },
  "hard_negatives": ["...", "...", "..."],
  "quality": {
    "same_context_score": 0.17,
    "edit_match_score": 0.259,
    "target_uniqueness_score": 0.891
  },
  "source_context": {
    "relation": "same_dataset",
    "score": 0.289
  }
}
```

当前它适合作为 smoke test，不适合作为正式训练集或 benchmark。

## 8. 与目标数据集的差距

目标数据应该长这样：

```json
{
  "sample_id": "covr_000001",
  "reference_video": "clips/group_x/ref.mp4",
  "target_video": "clips/group_x/target.mp4",
  "edit_text": "change one cat into two cats",
  "modalities": ["visual"],
  "reference_caption": "A mouse stands beside one cat in the same cartoon room.",
  "target_caption": "A mouse stands beside two cats in the same cartoon room.",
  "difference": {
    "type": "object_count",
    "from": "one cat",
    "to": "two cats",
    "description": "the number of cats increases from one to two"
  },
  "hard_negatives": [
    "clips/group_x/one_cat_wrong_action.mp4",
    "clips/group_x/two_dogs_same_room.mp4",
    "clips/group_x/two_cats_different_scene.mp4"
  ],
  "quality": {
    "same_context_score": 0.85,
    "edit_match_score": 0.9,
    "target_uniqueness_score": 0.8
  },
  "evidence": {
    "reference_storyline": ["..."],
    "target_storyline": ["..."],
    "visible_text_change": "...",
    "audio_change": "..."
  }
}
```

关键要求：

- reference 和 target 背景相似。
- 变化尽量单一。
- edit text 只描述变化，不重写完整视频。
- target 在 gallery 中唯一满足 edit。
- hard negatives 是近邻干扰项，而不是随机视频。
- 至少覆盖视觉、音频、语音/音乐、动作等类型。

## 9. 下一步不再重复跑，而是改方法

### 9.1 第一阶段：Omni-Detective annotation 重构

当前两步 detective：

```text
observer -> detective_final
```

需要升级为更接近 Omni-Captioner 的多轮流程：

```text
Detective Agent
  -> plan questions
  -> call observers/tools
  -> collect evidence
  -> ask follow-up questions
  -> synthesize structured caption
```

建议输出字段：

```json
{
  "storyline": [
    {
      "start": 0.0,
      "end": 3.2,
      "visual": "...",
      "audio": "...",
      "actions": ["..."],
      "objects": ["..."]
    }
  ],
  "visible_text": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "...",
      "appearance": "..."
    }
  ],
  "speakers_and_transcript": [
    {
      "start": 1.0,
      "end": 3.0,
      "speaker": "narrator",
      "content": "...",
      "state": "calm female voice"
    }
  ],
  "audio_events": [
    {
      "start": 2.0,
      "end": 4.0,
      "event": "applause"
    }
  ],
  "uncertainties": ["..."],
  "detective_trajectory": ["..."]
}
```

### 9.2 第二阶段：Tool Box

需要逐步加入工具，而不是只靠一个大模型一次看完整视频。

建议工具：

- `ffprobe`：获取时长、分辨率、音轨。
- `ffmpeg`：切片、抽帧、抽音频。
- scene split：按视觉变化切成短片段。
- audio captioner：用 Qwen3-Omni Captioner 或 Qwen2-Audio 做音频细粒度描述。
- OCR：提取可见文字。
- ASR：提取语音文字。
- frame observer：对关键帧做视觉主体、动作、场景描述。

这些工具输出证据，Detective Agent 只负责规划和综合。

### 9.3 第三阶段：先切片，再配对

当前最大的结构性问题是 whole source video 太长。

下一步应该先切成：

```text
3-15 秒短 clip
每个 clip 尽量只有一个主要事件
保留音频
```

然后只在同一上下文组内配对：

```text
same source video
same video_id
same source_row
same series / same account
same scene cluster
same subject cluster
```

不要再从全局 2858 clips 随机配对。

### 9.4 第四阶段：Pair Judge

对每个候选 pair，要增加 judge：

```text
reference 是否不满足 edit
target 是否满足 edit
hard negatives 是否接近但不满足 edit
edit text 是否只描述一个变化
是否必须依赖音频
是否存在明显幻觉
```

这一步应该优先用 Thinking 模型或 Instruct 模型做。

## 10. 当前结论

这阶段已经完成：

- 原始数据准备。
- Daily-Omni 媒体物化。
- WorldSense 解压。
- Qwen3-Omni 服务接入。
- single-pass 标注。
- detective-style 标注。
- pair proposal。
- pilot validation。
- 多轮 smoke test。

也发现了核心问题：

```text
工程链路已经跑通，但当前自动 pair 质量没有明显进展。
继续重复跑 pilot50 没有意义。
```

现在最重要的结论是：

```text
不要继续刷同一批 pilot50。
要把方法转向 Omni-Captioner / Omni-Detective：
多工具观察、时间线证据、短片段切分、组内配对、pair judge。
```

下一步建议：

```text
1. 暂停重复 pilot50 实验。
2. 实现 clip segmentation + grouped manifests。
3. 实现真正的 detective tool box。
4. 在 5-10 个同上下文视频组上构造少量高质量样本。
5. 人工 review 后再决定是否扩大规模。
```

一句话总结：

```text
我们已经证明 pipeline 能跑；下一步要证明方法能产生高质量 composed retrieval 数据。
```

## 11. Omni-Detective 新链路实现入口

新的实现不再复用旧的 `run_composed_pilot50.sh` 做重复实验，而是新增一条更接近 Omni-Captioner 的链路：

```text
source_clips_all.jsonl
-> clip_plan_detective.jsonl
-> clip_groups.jsonl
-> extracted_event_clips.jsonl
-> detective_annotations.jsonl
-> judged_pair_proposals.jsonl
-> accepted_pairs.jsonl
-> gallery.jsonl / pilot_review.md
```

服务器入口脚本：

```text
scripts/run_omni_detective_pilot.sh
```

默认行为：

- 对 50-100 个 source videos 做 4-12 秒事件切片，硬上限 15 秒。
- 只在同源或同数据集语义组内配对，不再跨数据集随机配对。
- detective annotation 会写入 tool-box trajectory，包括 `media_probe`、`frame_sampler`、`audio_observer`、`ocr_asr_observer`。
- group-level pair 会经过 pair judge，只有满足阈值的样本进入 `accepted_pairs.jsonl`。

服务器可用命令：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr
mkdir -p /data02/usr/wangqihao/Demo/test/cvr/runs/omni_detective_pilot_20260422
nohup bash scripts/run_omni_detective_pilot.sh \
  > /data02/usr/wangqihao/Demo/test/cvr/runs/omni_detective_pilot_20260422/omni_detective_pilot.log 2>&1 &
```

第一轮只看：

```text
accepted_pairs.jsonl 是否至少 5 条
pilot_review.md 中 same_context_avg 是否明显高于 0.141
人工 review 后是否能保留 60% 以上
```

## 12. 2026-04-22 晚间结果更新

基于 Omni-Detective 新链路，服务器已经实际跑出一版更可信的 pilot：

```text
git HEAD (server workspace at run time): 8b4dd8d55ae37b3e84bcc7fe9177eca314ecd131
detective annotation clip_count: 160
annotated_count: 160
fallback_count: 1
detective_to_single_pass_count: 2

judged proposals: 47
accepted pairs: 5
rejected pairs: 42

pilot sample_count: 5
gallery_count: 13
same_context_min: 0.80
same_context_avg: 0.84
same_context_max: 0.90
```

这是一个重要分水岭。

前面 single-pass / early detective 版本的核心问题一直是：

```text
same_context 太低
reference 和 target 虽然“像是能改写”，但上下文并不真的近
```

这次新链路下，accepted 样本全部来自：

```text
same_source_video
```

也就是说，**同上下文约束已经真正起作用了**。  
从数据质量角度看，这比前面单纯刷更多 pilot50 更有意义。

### 12.1 当前 accepted 样本结构

当前 accepted 的 5 条样本中：

```text
difference_type:
- object_presence: 3
- action: 1
- attribute: 1

modalities:
- visual: 5
- audio: 1
```

当前自动验收：

```text
sample_count_between_5_and_10: PASS
object_change_samples_at_least_2: PASS
action_samples_at_least_1: PASS
audio_samples_at_least_2: FAIL
```

这说明我们已经解决了“配不出高上下文 pair”的主问题，但还没有解决“音频相关样本进入 accepted 集合的比例太低”这个新问题。

### 12.2 当前瓶颈从 context 转移到了 audio / speech / visible_text

服务器分析结果显示：

```text
total judged rows: 47
accepted rows: 5
rejected rows: 42

difference_type all:
- object_presence: 15
- action: 12
- attribute: 8
- scene: 7
- speech: 4
- audio_event: 1

judge.audio_required = True: 28
audio-related rejected_count: 29
```

而且过去很烦人的一个问题也暴露得很清楚：

```text
很多 rejected rows 的 reject_reason 是空字符串
```

这会让后续分析非常低效，因为我们只能看到“被拒了”，但不知道是：

- same_context 不够
- edit_match 不够
- uniqueness 不够
- reference/target 判定不对
- 还是 judge 自己虽然拒绝了，但没把原因写出来

### 12.3 今天本地代码补的两类修复

为了让下一轮结果更可解释，也更贴近 Omni-Captioner 的“证据驱动 pair 构造”，今天本地代码补了两类修复：

#### A. Judge 诊断补强

即使模型本身返回空的 `reject_reason`，代码现在也会在最终写盘前补上结构化失败原因，例如：

```text
same_context_score 0.211 is below 0.55
edit_match_score 0.405 is below 0.75
target_uniqueness_score 0.603 is below 0.70
the model judge did not accept the pair
```

这意味着下一轮再看 `judged_pair_proposals.jsonl`，可以直接按失败门槛做分桶分析，而不是再人工猜。

#### B. Proposal / Judge 的单差异约束补强

本地 prompt 现在更明确要求：

- `difference.type`
- `edit_text`
- `modalities`
- `difference.from / difference.to`

必须共同指向 **同一个主差异**。

同时新增了更贴近本任务的偏好：

- 如果 pair 来自 `same_source_video` 或高度相似上下文，
- 且主变化是 `speech / audio_event / visible_text`，
- 就优先把它当成真正的 composed edit，
- 不要再被泛化成 `attribute`、`scene` 或带次要变化的混合 edit。

#### C. 高上下文 pair 的差异优先级微调

在高上下文 pair 中，本地启发式排序会更优先考虑：

```text
speech
audio_event
visible_text
```

避免它们总是被 `object_presence / action / attribute` 这些更泛的差异压掉。

### 12.4 当前阶段的结论

截至 2026-04-22 晚间，项目的判断可以更新成：

```text
1. “高上下文 pair 难以构造” 这个问题，已经基本被同源切片 + 组内配对解决。
2. 当前真正的短板，已经收缩成 audio/speech/visible_text 类型进入 accepted 集的比例不够高。
3. 所以下一步不应该回头继续刷旧 pilot50，而应该继续沿 Omni-Detective 方向，把 audio observer / speech / visible text 这条证据链做扎实。
```

换句话说：

```text
我们现在已经从“链路通不通”阶段，进入到“accepted 样本的模态覆盖够不够好”阶段了。
```

这其实是进展，不是原地踏步。

## 13. 当前实际在用的 Omni 模型

这一节专门澄清一个很容易混淆的问题：

```text
“模型已经下载好”
不等于
“模型已经接入当前正在跑的 nohup 实验”
```

### 13.1 当前 Omni-Detective 数据构造链路里真正用上的模型

截至 2026-04-22 晚间，当前 **实际接入并正在用于数据构造实验** 的 Omni 模型只有一个：

```text
Qwen3-Omni-30B-A3B-Instruct
路径：
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
服务：
http://127.0.0.1:8093/v1
```

它当前负责：

- `detective-annotate-clips`
- `propose-group-pairs`
- `pair judge`

也就是说，当前这条 Omni-Detective pilot 链路里的：

- detective annotation
- pair proposal
- pair judge

都还是同一个 `Qwen3-Omni Instruct` 在做。

### 13.2 已下载但还没有真正接入当前 nohup 实验的模型

#### A. Qwen3-Omni Captioner

模型路径：

```text
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-captioner
```

状态：

```text
已下载
尚未接入当前 Omni-Detective nohup 实验
```

这意味着当前 pipeline 还没有单独把 Captioner 当作独立 `audio observer` 使用。  
虽然我们在设计上已经把它定位为：

- audio events
- speech
- acoustic scene

的优先补强模型，但它现在还没有真正串进主链路。

#### B. Qwen3-Omni Thinking

模型路径：

```text
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-thinking
```

状态：

```text
已下载
尚未接入当前正在运行的 nohup 实验
```

原因很直接：

- 当前实验脚本和手工命令仍然把 `MODEL` 指向 `qwen3-omni-30b-a3b-instruct`
- `BASE_URL` 仍然指向 `8093`

所以即使 Thinking 已经下载好了，只要服务入口和命令没有切换，它就还只是“已准备”，不是“已使用”。

### 13.3 历史上用过、但属于另一条实验线的 Omni 模型

在更早的 AVIGATE / MSRVTT agent 检索实验中，我们还用过：

```text
Qwen2.5-Omni
```

那条线主要是：

- V2T / T2V rerank
- Omni checker
- official retrieval 上层 agent

它和当前这条 **新数据集构造 / Omni-Detective** 链路不是同一条实验线。

所以当前项目里可以把 Omni 模型的使用状态理解成：

```text
历史旧线：
- Qwen2.5-Omni（已用过）

当前新线：
- Qwen3-Omni Instruct（正在用）
- Qwen3-Omni Captioner（已下载，未接入）
- Qwen3-Omni Thinking（已下载，未接入）
```

### 13.4 当前最合理的角色分工

从方法设计上，后续推荐的分工是：

#### Instruct

- 继续做基线 detective annotation
- pair proposal
- baseline pair judge
- 基础流程控制

#### Captioner

- 强化 `audio_events`
- 强化 `speech`
- 强化 `acoustic scene`
- 作为真正的独立 `audio_observer`

#### Thinking

- 更严格的 pair judge
- 失败样本分析
- detective planning / follow-up reasoning
- 人工 review 辅助总结

当前的核心事实不是“模型不够”，而是：

```text
模型已经准备得比当前 pipeline 用到的更多，
但 Captioner / Thinking 还没有真正接到实验链路里。
```

## 14. 服务器工作区状态与后续运行约束

### 14.1 旧工作区必须冻结

服务器旧工作区：

```text
/data02/usr/wangqihao/Demo/test/cvr
```

当前状态不是干净仓库，而是：

- 本地 HEAD 停在旧提交 `8b4dd8d`
- 暂存区里叠有本地 hotfix
- 存在若干未跟踪文件

这意味着它不适合继续作为正式实验目录。

当前旧工作区应被视为：

```text
历史实验现场
只保留
不继续在其中跑新实验
```

### 14.2 干净工作区已经建立

服务器已建立新的干净 worktree：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main
```

对应提交：

```text
1bf72f99547945669a2ae7d19a4d75b9638fba60
improve omni detective pair diagnostics
```

后续所有新的 Omni-Detective 实验，应该默认在这个目录下继续。

### 14.3 当前脚本的一个实际限制

虽然 clean worktree 已经建立，但当前仓库中的：

```text
scripts/run_omni_detective_pilot.sh
```

仍然把运行目录写成了：

```text
/data02/usr/wangqihao/Demo/test/cvr
```

也就是说，**如果直接运行这份脚本，它仍然会跳回旧的脏工作区**。

因此在 clean worktree 阶段，推荐的做法是：

```text
先用手工命令串行执行 pipeline
不要直接复用旧脚本
```

等我们后续把 clean worktree 路径方案固定好，再决定是否更新脚本。

### 14.4 clean worktree 的一次环境性测试问题

在 clean worktree 上第一次运行 unittest 时，出现过 1 个错误：

```text
FileNotFoundError: ... /cvr_clean_main/runs/... does not exist
```

这个错误的性质是：

```text
运行时目录不存在
不是核心代码逻辑错误
```

也就是说，它更像是环境准备问题，而不是当前 `Omni-Detective pair diagnostics` 这次提交本身的功能 bug。

## 15. 当前阶段的一句话更新

截至 2026-04-22 当前时点，项目状态可以概括为：

```text
我们已经把“高上下文组内配对”跑通，并确认当前真正在线使用的是 Qwen3-Omni Instruct；
Captioner 和 Thinking 都已经就位，但还没有正式接入当前实验链路。
下一步的关键不是再刷旧 pilot，而是把 Captioner / Thinking 有计划地接进 clean worktree 下的 Omni-Detective pipeline。
```
