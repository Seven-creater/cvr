# Omni 组合视频检索数据构造最新总结

Last updated: 2026-04-22

## 1. 这份文档只讲当前最新状态

这份文档不再复述早期的长过程，只保留当前阶段最重要的信息：

- 我们现在在做什么
- 已经做了什么
- 遇到了哪些难题
- 分别是怎么解决的
- 原始数据集长什么样
- 当前中间数据集长什么样
- 最终想要的数据集长什么样
- 当前离目标还差什么

当前任务已经明确从早期的 MSRVTT 常规 video-text retrieval，转向一个新的目标：

```text
reference video + edit text + visual/audio cues -> target video
```

也就是：

- 输入一个参考视频 `reference_video`
- 输入一段“变化描述” `edit_text`
- 系统要从候选库里找到满足这个变化后的 `target_video`

这套任务需要同时覆盖：

- 视觉变化：主体、数量、动作、场景、属性、可见文字
- 音频变化：音乐、说话、环境声、音效、乐器声
- 编辑文本：只描述 reference 到 target 的关键变化

一句话说，**我们现在的重点已经不是继续在旧检索数据上堆 agent，而是构造一套真正适合 Omni / agentic retrieval 的新数据集。**

## 2. 我已经做了什么

### 2.1 把原始数据源整理成统一格式

当前已经接入并处理了两个原始数据源：

- `Daily-Omni`
- `WorldSense`

统一数据根目录：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
```

已经完成的原始数据准备工作：

- 解析 `Daily-Omni` 的 parquet 数据
- 物化其中内嵌的视频和音频
- 解压 `WorldSense` 的 zip 包
- 把两个数据源统一整理成可下游处理的 `source_rows.jsonl` 和 `source_clips*.jsonl`

目前得到的归一化统计是：

```text
source rows: 4368
unique clips: 2858
pilot clips: 50

Daily-Omni: 1196 rows / 1196 clips
WorldSense: 3172 rows / 1662 clips
```

而且 `pilot50` 已经做过数据源均衡：

```text
Daily-Omni: 25
WorldSense: 25
```

### 2.2 跑通了第一版 Omni 数据构造链路

我已经把一套从原始视频到候选 pair 的链路写出来并跑通了，核心代码和脚本包括：

- `app/composed_sources.py`
- `app/composed_data.py`
- `app/composed_omni.py`
- `scripts/prepare_composed_sources.sh`
- `scripts/run_omni_detective_pilot.sh`

这条链路已经可以完成：

1. 原始数据发现和整理
2. source clip manifest 生成
3. detective event clip 规划
4. clip 抽取
5. Omni 标注
6. pair proposal
7. pair judge
8. accepted pairs 导出
9. gallery 与 pilot review 报告生成

### 2.3 从“整段视频乱配对”切到了“同上下文组内配对”

这是当前阶段最重要的改动。

早期版本的问题是：

- 还是在 whole video 级别配对
- 就算都来自同一个数据集，也不代表上下文真的接近
- 很容易配出“看起来能写 edit_text，但其实 reference 和 target 并不在同一个语境里”的假样本

我后来把方法改成了更接近 Omni-Captioner / Omni-Detective 的方式：

```text
source_clips_all.jsonl
-> clip_plan_detective.jsonl
-> clip_groups.jsonl
-> extracted_event_clips.jsonl
-> detective_annotations.jsonl
-> judged_pair_proposals.jsonl
-> accepted_pairs.jsonl
```

核心变化是：

- 先切事件片段，不再直接用整段视频
- 先做同源分组，再在 group 内配对
- 优先使用 `same_source_video`
- 不再允许跨数据集随机乱配

这个改动是当前最有效的一步，因为它直接把样本质量的核心问题从“上下文不相关”拉回到“同一语境下的局部差异”。

### 2.4 跑通了 Omni-Detective 风格的标注

当前的 detective 标注还不是完全体，但已经不是单轮 caption 了。

现在的标注流程已经包含：

- `media_probe`
- `frame_sampler`
- `audio_observer`
- `ocr_asr_observer`
- detective final synthesis

现在输出的 annotation 已经包括这些关键字段：

- `summary`
- `storyline`
- `visible_text`
- `speakers_and_transcript`
- `audio_events`
- `detective_trajectory`
- `uncertainties`

这一步的意义是：**我们不再只是得到一句泛泛 summary，而是开始积累构造 composed retrieval 样本所需要的证据链。**

### 2.5 做了 pair judge，而且把诊断信号补强了

我现在已经把 pair judge 加进去了。每条候选 pair 不再是“模型觉得差不多就留下”，而是会经过更严格的判断，包括：

- `reference_satisfies_edit`
- `target_satisfies_edit`
- `single_main_difference`
- `same_context_score`
- `edit_match_score`
- `target_uniqueness_score`
- `audio_required`
- `hard_negative_quality`
- `accept`
- `reject_reason`

而且我最近又补了一层很关键的诊断：

即使模型本身返回空的 `reject_reason`，代码也会根据最终门槛自动补出失败原因，比如：

```text
same_context_score 0.211 is below 0.55
edit_match_score 0.405 is below 0.75
target_uniqueness_score 0.603 is below 0.70
the model judge did not accept the pair
```

这意味着后面再分析失败样本时，我们终于不用靠猜了。

### 2.6 清理了一个服务器仓库状态问题

服务器上原来的工作区：

```text
/data02/usr/wangqihao/Demo/test/cvr
```

已经变成了一个“旧提交 + staged 本地热修 + 若干未跟踪文件”的脏工作区。  
我没有继续在这个目录上强推实验，而是明确把它冻结，当作旧现场保留。

然后新建了一个干净工作树：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main
```

这个新目录已经切到最新提交：

```text
1bf72f99547945669a2ae7d19a4d75b9638fba60
```

这样后面的实验终于可以在一个干净状态里跑，不会再把服务器 AI 的临时热修和正式代码混在一起。

## 3. 我遇到的主要难题，以及我是怎么解决的

### 难题 1：原始 whole video 太长，事件混杂，pair 质量很差

#### 问题

原始视频很多都很长，一个视频里同时包含多个事件、多个镜头、多个音频变化。  
如果直接拿 whole video 做 `reference -> target` 配对，会出现两个问题：

- 上下文其实不够接近
- edit_text 很容易变成“整段视频重写”，而不是单一差异

#### 解决方法

我把流程改成了：

- 先切片
- 再分组
- 最后只在组内配对

目前优先使用的是：

- `same_source_video`
- 其次才是语义组

这个改动带来的最直接结果是：

```text
accepted 样本的 same_context_avg 从早期低水平，提升到了 0.84
```

这说明“上下文相似性”这个核心问题，已经不再是主矛盾了。

### 难题 2：Detective 风格标注更细，但更容易失败

#### 问题

detective 两阶段 JSON 输出比 single-pass 更复杂，早期很容易因为模型返回不稳定而失败。  
一旦失败，整条样本会直接丢掉，候选池会明显缩小。

#### 解决方法

我加了 fallback：

```text
detective annotation failed -> single-pass annotation fallback
```

而且 fallback 不会让整批中断，只会把该样本带上：

- `detective_fallback_used`
- `detective_fallback_reason`
- `detective_trajectory`

这样我们既保住了候选数量，也保住了失败样本的可诊断性。

### 难题 3：服务器脚本里出现了 CLI 参数路由错误

#### 问题

之前出现过：

```text
AttributeError: 'Namespace' object has no attribute 'max_accepted_pairs'
```

根因是：

- `detective-annotate-clips` 的 parser 没有这个参数
- 但 main() 里错误地把 `args.max_accepted_pairs` 传给了它

#### 解决方法

我修正了 CLI 路由：

- `detective-annotate-clips` 不再读取 `max_accepted_pairs`
- `propose-group-pairs` 才真正接收 `max_accepted_pairs`

同时加了对应测试，防止这个问题再回头。

### 难题 4：同上下文已经解决，但 audio 样本还是太少

#### 问题

当前最新一轮 Omni-Detective pilot 的 accepted 结果是：

```text
sample_count: 5
gallery_count: 13
same_context_avg: 0.84
```

但模态覆盖仍然不够好：

```text
audio: 1
visual: 5
```

自动验收里：

```text
audio_samples_at_least_2: FAIL
```

这说明我们现在已经不是“样本配不出来”，而是“音频相关差异进入 accepted 集的比例不够高”。

#### 已经做的处理

我做了两件事：

1. 在高上下文 pair 中，提高了 `speech / audio_event / visible_text` 的优先级
2. 收紧 proposal / judge 的 prompt，要求 `difference.type / edit_text / modalities / from-to` 必须描述同一个主差异

#### 还没彻底解决

这个问题目前还没有彻底解决。  
下一步需要真正把 `Captioner` 和更强的音频观察链路接进来，而不是只靠 `Instruct` 一把梭。

### 难题 5：服务器工作区脏，实验结果容易掺杂本地热修

#### 问题

服务器原工作区里出现了：

- staged 修改
- 未跟踪文件
- 本地 `origin/main` 落后于远端

这种状态下继续实验，结果会越来越难解释。

#### 解决方法

我没有强行清理这个目录，而是：

- 保留旧现场
- 只做只读检查
- 新建干净 worktree `cvr_clean_main`

这是更稳的做法，因为它不会误伤旧实验现场，也不会把仓库状态问题继续滚大。

## 4. 原始数据集长什么样

### 4.1 Daily-Omni 原始样子

Daily-Omni 的原始行本质上更接近“多模态问答数据”，而不是 composed retrieval 数据。

原始一行大概是这种结构：

```json
{
  "video_id": "Ec_lQgZ9wlg",
  "video": "...(内嵌视频字节或路径)...",
  "audio": "...(内嵌音频字节或路径)...",
  "question": "What visual elements were displayed immediately after ...?",
  "candidates": [
    "A. ...",
    "B. ...",
    "C. ...",
    "D. ..."
  ],
  "answer": "B. ..."
}
```

问题在于：

- 它原本是 QA 任务
- 不是 reference-target-edit 三元组
- 不能直接拿来做 composed retrieval

### 4.2 WorldSense 原始样子

WorldSense 更像视频文件 + 字幕/描述/问答混合资源。

原始资源大概包括：

- zip 压缩的视频文件
- 字幕文件
- video caption
- question / candidates / answer

它的原始重点仍然不是：

```text
reference video + edit text -> target video
```

所以它也必须经过重构。

## 5. 现在的数据集长什么样

### 5.1 第一步：统一 source rows

原始数据先被统一成 `source_rows.jsonl`，每一行代表一个可追踪的原始样本：

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

### 5.2 第二步：统一 source clips

然后变成 `source_clips_all.jsonl` / `source_clips_pilot50.jsonl`：

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
  "text_fields": {
    "question": "...",
    "answer": "..."
  }
}
```

这一步还是“原始候选池”的形态，还不是最终任务样本。

### 5.3 第三步：Detective 事件片段和标注

之后又生成：

- `clip_plan_detective.jsonl`
- `clip_groups.jsonl`
- `extracted_event_clips.jsonl`
- `detective_annotations.jsonl`

其中 `detective_annotations.jsonl` 现在大概长这样：

```json
{
  "clip_id": "...",
  "output_path": "clips/detective/...",
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
  "uncertainties": ["..."]
}
```

### 5.4 第四步：judged pair proposals

再往后进入 pair proposal 和 judge：

```json
{
  "proposal_id": "proposal__...",
  "group_id": "group_daily_omni_xxx",
  "reference_video": "clips/detective/...",
  "target_video": "clips/detective/...",
  "edit_text": "...",
  "modalities": ["visual", "audio"],
  "reference_caption": "...",
  "target_caption": "...",
  "difference": {
    "type": "action",
    "from": "...",
    "to": "...",
    "description": "..."
  },
  "hard_negatives": ["...", "...", "..."],
  "quality": {
    "same_context_score": 0.84,
    "edit_match_score": 0.90,
    "target_uniqueness_score": 0.75
  },
  "judge": {
    "reference_satisfies_edit": false,
    "target_satisfies_edit": true,
    "single_main_difference": true,
    "audio_required": false,
    "hard_negative_quality": "good",
    "accept": true,
    "reject_reason": ""
  }
}
```

这一步已经很接近最终 wanted dataset 了。

## 6. 我需要的数据集长什么样

最终真正要交付或扩展的目标数据集，应该是这种结构：

```json
{
  "sample_id": "covr_omni_pilot_0001",
  "reference_video": "clips/group_x/ref.mp4",
  "target_video": "clips/group_x/target.mp4",
  "edit_text": "change one cat into two cats",
  "modalities": ["visual", "audio"],
  "reference_caption": "A mouse stands beside one cat in the same cartoon room.",
  "target_caption": "A mouse stands beside two cats in the same cartoon room.",
  "difference": {
    "type": "object_count",
    "from": "one cat",
    "to": "two cats",
    "description": "the number of cats increases from one to two"
  },
  "hard_negatives": [
    "clips/group_x/wrong_1.mp4",
    "clips/group_x/wrong_2.mp4",
    "clips/group_x/wrong_3.mp4"
  ],
  "quality": {
    "same_context_score": 0.85,
    "edit_match_score": 0.90,
    "target_uniqueness_score": 0.80
  },
  "evidence": {
    "reference_storyline": ["..."],
    "target_storyline": ["..."],
    "audio_change": "...",
    "visible_text_change": "..."
  },
  "judge": {
    "reference_satisfies_edit": false,
    "target_satisfies_edit": true,
    "single_main_difference": true,
    "audio_required": true,
    "hard_negative_quality": "good",
    "accept": true,
    "reject_reason": ""
  }
}
```

这个最终目标数据集必须满足几条硬要求：

- reference 和 target 的上下文足够近
- edit_text 只描述变化，不重写整段视频
- target 真正满足 edit
- reference 不满足 edit
- hard negatives 要接近，但不能满足 edit
- 样本要覆盖视觉、音频、语音、可见文字等不同差异类型

## 7. 目前最新结果是什么

当前最新、最可信的一轮 Omni-Detective pilot 结果是：

```text
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

当前 accepted 样本的特点：

```text
difference_type:
- object_presence: 3
- action: 1
- attribute: 1

modalities:
- visual: 5
- audio: 1
```

这组结果说明：

### 已经解决的事

- 我们已经不再停留在“链路能不能跑通”
- 我们已经能自动构造出高上下文的 pair
- `same_context` 这件事已经从主问题退下来了

### 还没解决的事

- audio/speech/visible_text 类型样本占比仍然太低
- `audio_samples_at_least_2` 目前仍然失败

所以当前阶段的主瓶颈已经非常明确：

```text
不是 pair 配不出来
而是“音频相关差异”还不够容易进入 accepted 集
```

## 8. 当前实际用到了哪些 Omni 模型

### 已经真正用上的

当前这条 Omni-Detective 数据构造链路里，真正跑起来的核心模型是：

- `Qwen3-Omni-30B-A3B-Instruct`

当前服务端口：

```text
http://127.0.0.1:8093/v1
```

它现在负责：

- detective annotation
- pair proposal
- pair judge

### 已下载但还没正式接入当前链路的

- `Qwen3-Omni-30B-A3B-Captioner`
- `Qwen3-Omni-30B-A3B-Thinking`

当前它们的状态是：

- 模型已经在服务器上
- 但还没有真正接入当前 nohup 实验主链路

所以当前 pipeline 的真实状态是：

```text
Instruct：在用
Captioner：已下载，待接入
Thinking：已下载，待接入
```

## 9. 我接下来建议怎么做

当前最合理的下一步，不是继续重复跑旧的 pilot50，也不是在脏工作区里继续叠补丁，而是：

### 9.1 只在干净 worktree 里继续实验

新的实验目录应该固定为：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main
```

旧目录保留，不再作为主实验目录。

### 9.2 做 audio-focused pilot

因为当前主瓶颈已经明确是 audio 样本不足，所以最值得做的不是继续随机抽 clip，而是：

- 从 `source_clips_all.jsonl` 中优先筛出音频相关 clip
- 再跑新一轮 detective pipeline
- 看 accepted 集里 audio / speech / visible_text 是否上升

### 9.3 正式把 Captioner 接进 audio observer

这一步非常关键。

当前虽然有 `audio_observer` 这个逻辑位置，但真正的强音频能力还没接上。  
接下来应该优先把：

- `Qwen3-Omni Captioner`

用于：

- `audio_events`
- `speech`
- `acoustic scene`

这样才有机会真正提高 audio 类型样本进入 accepted 集的概率。

### 9.4 再把 Thinking 接到更严格的 judge / failure analysis

`Thinking` 更适合做：

- 更严格的 pair judge
- 多轮 detective planning
- 失败样本分析

这一步更像是质量拔高，而不是链路打底。

## 10. 当前一句话结论

当前阶段最准确的总结是：

```text
我们已经把原始 Daily-Omni / WorldSense 数据整理成统一可处理的数据格式，
也已经把 Omni-Detective 风格的切片、标注、组内配对和 pair judge 跑通了。

现在最大的难点已经不再是“上下文相似性不够”，
而是“如何让音频 / 语音 / 可见文字差异更稳定地进入 accepted 数据集”。
```

如果只用一句更短的话来概括：

```text
链路已经跑通，当前真正要攻克的是多模态差异覆盖率，尤其是音频相关样本。
```
