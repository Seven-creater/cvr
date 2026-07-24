# Omni 构造 CVR 音视频融合数据集方法说明

> 当前日期：2026-05-10  
> 当前代码主分支：`main`  
> 当前关键 commit：`fb77b64 Include quality-passed single-source pairs beyond cap`  
> 本文目标：完整说明目前如何使用 Omni/Qwen3-Omni 构造 CVR（Composed Video Retrieval）数据集，包含数据来源、切片策略、pair 构造、Omni 标注、Omni 对比、最终核验、质量门槛、输出结构和已知边界。

---

## 1. 我们现在到底在构造什么数据集

当前数据集是 **Composed Video Retrieval / CVR** 风格的数据集。每条样本由三部分组成：

```text
reference video + edit_text -> target video
```

也就是：

- `reference.mp4`：参考视频片段。
- `edit_text.txt`：一句短编辑文本，描述从 reference 到 target 的变化。
- `target.mp4`：目标视频片段。

模型训练或评测时，输入是：

```text
reference video + edit_text
```

希望从候选库里检索到正确的：

```text
target video
```

一个合格样本需要满足：

1. reference 和 target 有明显关系，不是完全无关的两个视频。
2. reference 和 target 有清楚、可见、可描述的差异，不是几乎一样。
3. `edit_text` 能准确描述主差异。
4. 这个差异是人打开两个视频后一眼能确认的。
5. 差异不能主要来自字幕、标题卡、OCR、产品包装文字或 speech transcript。
6. Omni 的详细描述、pair-level evidence 和 final verification 都支持这个样本。

---

## 2. 为什么最终选择“同源单视频内部 pair”

我们尝试过三条路线：

### 2.1 生成式视频编辑路线

曾经尝试：

- plain masked VACE
- composite-first-frame VACE
- deterministic background composite
- mask 增强
- 背景替换、物体替换、衣服替换等

主要问题：

- VACE 对大面积结构替换不稳定。
- talking-head 场景常出现源背景残留或只是颜色风格变化。
- mask 质量、生成质量、语义核验都很重。
- 大规模构造数据时成本高、失败率高、人工审核压力大。

结论：

```text
生成式视频编辑可以保留为补充，不作为当前大规模数据集主线。
```

### 2.2 跨视频 pair 路线

也尝试过从不同视频中找相似 pair，例如：

- 女人演讲 -> 男人演讲
- 一个场景 -> 另一个场景
- 一个主体 -> 另一个主体

这种路线的问题是：

- 两个视频往往同时有主体、场景、衣服、动作、背景、构图、音频等多重差异。
- Omni judge 经常认为存在 competing differences。
- edit_text 很难只描述一个主差异。
- 人工看时会觉得“差异太大，不像一次编辑”。

结论：

```text
跨视频 pair 暂时不作为主线。
```

### 2.3 当前采用：同一个源视频内部切片 pair

当前主线是：

```text
从同一个原始视频中取一个 30s 窗口
-> 切成 5 个 6s segment
-> 在同一个源视频内部构造 segment pair
```

这样做的好处：

- reference 和 target 自然有同源上下文。
- 差异不会完全失控。
- 不需要生成视频。
- 每个源视频可以产生多个 pair。
- Omni 可以结合整个 30s 视频上下文理解片段变化。
- 人工审核更容易判断：同一视频内从前一段变到后一段。

这条路线的核心原则是：

```text
宁可从同源视频里找清楚的自然变化，也不要让模型凭空生成目标视频。
```

---

## 3. 数据来源

服务器数据根目录：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
```

当前使用两份原始数据：

```text
daily_omni
worldsense
```

详细结构可参考：

```text
/Users/Admin/Desktop/omni-runs/STRUCTURE.md
```

### 3.1 Daily-Omni

原始数据目录：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/daily_omni
```

已提取出的 30s 左右原始视频：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw/daily_omni/video
```

规模：

```text
1196 个 mp4
```

特点：

- 每条视频通常约 30 秒。
- 适合直接作为一个 single-source unit。
- 每条视频切 5 个 6s segment。
- 每条源视频最多可产生 10 个 chronological pair。

### 3.2 WorldSense

WorldSense 的 parquet 只包含 metadata，不直接包含视频二进制。实际视频已解压在：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense/_extracted/videos_chunk_001/videos
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense/_extracted/videos_chunk_002/videos
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense/_extracted/videos_chunk_003/videos
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense/_extracted/videos_chunk_004/videos
```

已知规模：

| chunk | 文件数 | 大小 |
|---|---:|---:|
| `videos_chunk_001` | 459 | 5.0 GB |
| `videos_chunk_002` | 490 | 5.0 GB |
| `videos_chunk_003` | 497 | 5.0 GB |
| `videos_chunk_004` | 216 | 2.1 GB |
| 合计 | 1662 | 17.1 GB |

特点：

- 视频名是 `{video_id}.mp4`，例如 `AAWgrzYx.mp4`。
- 视频长度不固定，很多明显超过 30 秒。
- 当前不会把整条长视频全部切完，而是从单个长视频中取一个 30s 窗口。

WorldSense 的窗口策略：

```text
source_window_duration_seconds = 30
source_window_start_seconds = min(30s, max(0, duration * 0.25))
```

并保证：

```text
window_start + 30 <= duration
```

这样可以避开长视频开头可能存在的片头、黑屏、标题卡，同时不会取到结尾不足 30 秒的区域。

---

## 4. 当前不用哪些已有 clip

数据目录里有历史切片：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/omni_stable
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/detective
```

但当前 CVR 构造主线不以它们为入口。

原因：

- `clips/omni_stable` 多数约 4 秒，太短，不符合当前 6s 切片方案。
- `clips/detective` 多数约 12 秒，是旧 detective pipeline 产物，不是当前统一的 30s -> 6s 单源切片逻辑。

当前入口固定为：

```text
daily_omni: raw/daily_omni/video/*.mp4
worldsense: raw_datasets/worldsense/_extracted/videos_chunk_*/videos/*.mp4
```

---

## 5. 服务器和代码环境

服务器代码路径：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main
```

GitHub 仓库：

```text
git@github.com:Seven-creater/cvr.git
```

当前工作分支：

```text
main
```

当前关键 commit：

```text
fb77b64 Include quality-passed single-source pairs beyond cap
```

更新代码：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main || exit 1
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
```

Omni 模型 checkpoint：

```text
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
```

API 服务地址：

```text
http://127.0.0.1:8093/v1
```

注意：调用 API 时模型名必须是 vLLM 注册名：

```text
qwen3-omni
```

不要使用完整 checkpoint 路径作为 `--model`。之前出过一次事故：pipeline 用完整路径，vLLM 注册名是 `qwen3-omni`，导致大量 annotation fallback，最终 0 输出。

Conda 环境：

```bash
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src
python --version
```

已知 Python：

```text
Python 3.10.20
```

检查 Omni 服务：

```bash
curl -s http://127.0.0.1:8093/v1/models | python -m json.tool
```

期望返回的模型 id 包含：

```text
qwen3-omni
```

---

## 6. 代码入口和核心文件

代码仓库路径：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main
```

核心代码：

```text
app/composed_data.py
app/composed_omni.py
```

核心脚本：

```text
scripts/run_single_source_omni_pair_pilot.sh
scripts/run_single_source_omni_batch.sh
```

核心测试：

```text
tests/test_composed_data.py
tests/test_composed_omni.py
tests/test_scripts.py
```

主要函数：

| 函数 | 作用 |
|---|---|
| `select_single_source_video` | 从候选源视频中选择一个 30s 源视频。 |
| `plan_single_source_clips` | 把源视频或 30s 窗口规划成 5 个 6s segment。 |
| `annotate_clips` | 调 Omni 对 whole window 或 segment 做结构化 annotation。 |
| `mine_single_source_pairs` | 枚举同源 segment 的 chronological pair。 |
| `propose_single_source_pairs` | 调 Omni 做 pair-level 对比、生成 edit_text，并执行 final verification。 |
| `build_single_source_review_bundle` | 生成人工核验 bundle。 |
| `_single_source_pair_acceptance_issues` | 本地 pair-level hard/soft gate。 |
| `_single_source_final_verification_issues` | final Omni verifier 结果转拒绝原因。 |
| `_apply_single_source_delta_uniqueness` | 同一源内相同 delta family 去重。 |
| `_select_single_source_quality_passed_records` | 收集所有质量通过的 single-source pair。 |

当前 final quality 阈值写在：

```python
MIN_SINGLE_SOURCE_FINAL_OMNI_QUALITY_SCORE = 0.70
```

禁用的最终差异类型：

```python
FINAL_DISABLED_DIFFERENCE_TYPES = {"speech", "visible_text"}
```

---

## 7. 单个源视频的完整处理流程

### 7.1 选择源视频或 30s 窗口

Daily-Omni：

```text
直接选择 28-32 秒、有音频、有视频、可解码的 raw/daily_omni/video/*.mp4
```

WorldSense：

```text
从 long video 中取 30 秒 window
```

每个 selected source 会写入：

```text
selected_source_video.json
```

其中关键字段包括：

```json
{
  "source_clip_id": "...",
  "dataset": "daily_omni 或 worldsense",
  "source_path": "/absolute/path/to/video.mp4",
  "duration_seconds": 30.0,
  "source_window_start_seconds": 0.0,
  "source_window_duration_seconds": 30.0,
  "media_probe": {...}
}
```

### 7.2 规划 6s 切片

当前固定：

```text
segment_seconds = 6
source_window_duration_seconds = 30
```

因此一般得到：

```text
5 个 segment
```

如果 segment 数小于 4，则认为这个 source 不适合本轮构造。

输出：

```text
single_source_clip_plan.jsonl
single_source_clip_groups.jsonl
selected_source_manifest.jsonl
```

每个 segment record 包含：

```json
{
  "clip_id": "...__single_001",
  "source_path": "/absolute/source.mp4",
  "output_path": "clips/single_source/{source_id}/{clip_id}.mp4",
  "start_seconds": 0.0,
  "end_seconds": 6.0,
  "duration_seconds": 6.0,
  "role": "single_source_segment",
  "dataset": "daily_omni",
  "source_clip_id": "...",
  "source_window_start_seconds": 0.0,
  "source_window_duration_seconds": 30.0,
  "relative_start_seconds": 0.0,
  "relative_end_seconds": 6.0
}
```

### 7.3 实际切出视频

通过 ffmpeg 从原视频切出：

```text
extracted_single_source_whole.jsonl
extracted_single_source_clips.jsonl
```

其中：

- whole clip 是整个 30s window。
- segment clips 是 5 个 6s 视频。

### 7.4 Omni 对 whole window 做全局描述

whole window 用于给 pair-level 对比提供全局上下文。

输出：

```text
single_source_whole_annotation.jsonl
```

它描述整个 30s 的：

- 总体场景。
- 主要人物/主体。
- 时间线。
- 关键物体。
- 动作变化。
- 音频事件。
- 可能出现的文字。

### 7.5 Omni 对每个 6s segment 做细粒度 annotation

输出：

```text
single_source_annotations.jsonl
```

每个 segment 的 annotation 包含：

```text
summary
subjects
object_counts
actions
scene
attributes
visible_text / on_screen_text
speech
speakers_and_transcript
audio_events
storyline
detective_notes
detective_trajectory
uncertainties
raw_model_output
fallback_used
```

这个阶段非常重要，因为后续 pair 对比会把 reference annotation、target annotation 和 whole annotation 一起交给 Omni。

### 7.6 枚举同源 chronological pair

如果有 5 个 segment：

```text
segment_1, segment_2, segment_3, segment_4, segment_5
```

则构造：

```text
C(5, 2) = 10 个 pair
```

只构造时间顺序 pair：

```text
reference = earlier segment
target = later segment
```

例如：

```text
1 -> 2
1 -> 3
1 -> 4
1 -> 5
2 -> 3
2 -> 4
2 -> 5
3 -> 4
3 -> 5
4 -> 5
```

不构造反向 pair。

输出：

```text
single_source_pair_candidates.jsonl
single_source_pair_report.md
```

candidate 阶段只是枚举，不再信任本地粗差异词直接生成最终 edit_text。最终 edit_text 必须由 Omni 在 pair-level 直接看两个视频后生成。

---

## 8. Omni 在当前方法里的角色

Omni 不是单纯 captioner，也不是完全黑盒裁判。当前把 Omni 拆成三层角色：

```text
1. Observer：看 whole video 和每个 segment，做细粒度 annotation。
2. Pair Comparator / Editor：直接比较 reference 和 target，生成 edit_text 和 dominant_delta。
3. Final Verifier：拿着视频、proposal、local gate report 再做最终打分。
```

### 8.1 第一层：segment detailed annotation

目标：

```text
让每个 6s segment 有足够细的结构化描述。
```

关注字段：

- 画面主体是谁。
- 主体在做什么。
- 有什么物体。
- 有无 product / overlay / PIP / inset。
- 场景和构图是什么。
- 是否有明显动作变化。
- 是否有非语音音频事件。
- 有无 visible text，但 visible text 默认不作为最终主差异。

### 8.2 第二层：pair-level direct video comparison

输入：

- reference video
- target video
- reference annotation
- target annotation
- whole video annotation
- candidate metadata

Omni 必须输出：

```json
{
  "edit_text": "...",
  "modalities": ["visual"],
  "reference_caption": "...",
  "target_caption": "...",
  "difference": {
    "type": "object_presence",
    "from": "...",
    "to": "...",
    "description": "..."
  },
  "dominant_delta": {
    "type": "object_presence",
    "from": "...",
    "to": "...",
    "reason": "..."
  },
  "reference_state": {
    "main_speaker": "...",
    "inset_subjects": [],
    "product_overlay": "",
    "composition": "...",
    "internal_transitions": []
  },
  "target_state": {
    "main_speaker": "...",
    "inset_subjects": [],
    "product_overlay": "...",
    "composition": "...",
    "internal_transitions": []
  },
  "delta_temporal_extent": {
    "reference": "...",
    "target": "...",
    "target_coverage": 0.9,
    "evidence": "..."
  },
  "subject_roles": {
    "main_speaker": "...",
    "inset_subjects": [],
    "product_overlay": "..."
  },
  "is_segment_wide_delta": true,
  "discarded_deltas": ["..."],
  "evidence": ["..."],
  "confidence": 0.9,
  "accept": true,
  "reject_reason": ""
}
```

这一层的 prompt 明确要求：

- 优先选择具体视觉变化。
- 优先 object / product / overlay / action / composition / scene。
- 不要选很弱的衣服措辞差异，例如 `blouse -> shirt`。
- 不要把 PIP/inset 里的男人/女人写成主 speaker。
- 不要把 `speaker + product overlay` 误写成 `product close-up`。
- 不要把短暂末尾变化当成整段主差异。
- speech 和 visible text 只能作为辅助证据。

### 8.3 第三层：final Omni verification

输入：

- reference video
- target video
- pair proposal JSON
- local gate report
- reference annotation
- target annotation
- whole annotation

Final verifier 输出：

```json
{
  "accept": true,
  "confidence": 0.9,
  "quality_score": 0.9,
  "reference_satisfies_edit": false,
  "target_satisfies_edit": true,
  "observable_delta": true,
  "single_primary_delta": true,
  "text_or_ocr_driven": false,
  "segment_wide": true,
  "edit_text_accurate": true,
  "main_reject_reason": "",
  "evidence": ["..."],
  "recommended_edit_text": ""
}
```

Final verifier 需要检查：

- reference 是否不满足 edit。
- target 是否满足 edit。
- 是否有真实可见差异。
- 是否单一主差异。
- 是否文字/OCR 驱动。
- 差异是否覆盖 target 大部分片段。
- edit_text 是否准确。
- 是否有更强但未描述的差异。
- 是否把 inset/PIP 主体误当成主视频主体。
- 是否误写 `product close-up`、`full-screen product presentation`、`speaker replacement` 等。

当前质量阈值：

```text
quality_score >= 0.70
```

0.70 的含义：

```text
borderline but acceptable for human review
```

低于 0.70 必须拒绝。

---

## 9. 允许和拒绝的差异类型

### 9.1 允许进入最终数据集的主差异

| 类型 | 说明 | 示例 edit_text |
|---|---|---|
| `scene` | 场景、背景、构图明显变化。 | `change the scene from a close-up of hands playing a guitar to a medium shot of two people playing guitars together` |
| `object_presence` | 某个物体、产品、overlay、PIP 出现或消失。 | `add a picture-in-picture demonstration overlay` |
| `object_count` | 物体数量变化。 | `add a second trading card featuring Yao Ming and Tracy McGrady` |
| `action` | 主体动作发生清楚变化。 | `change the action from speaking to applying mascara` |
| `attribute` | 明显属性变化，但不能只是弱措辞。 | `change the presenter from wearing no visible makeup to wearing visible makeup` |
| `audio_event` | 非语音声音变化，且视觉变化不能更强。 | `add a clicking sound` |

实际已完成数据集中，主要是：

```text
scene
object_presence
action
attribute
object_count
少量 audio_event
```

### 9.2 禁止作为最终主差异

| 类型 | 原因 |
|---|---|
| `visible_text` | 容易变成 OCR/字幕/标题卡数据，不符合目标。 |
| `speech` | 争议大，人工快速核验难，容易把 transcript 当主差异。 |
| OCR / title card / lower-third | 文字变化不是我们想要的音视频融合主差异。 |
| product packaging text | 产品包装文字变化不能当作物体变化。 |
| near-duplicate attribute wording | 例如 blouse vs shirt、long brown hair vs long hair。 |

代码中最终禁用：

```python
FINAL_DISABLED_DIFFERENCE_TYPES = {"speech", "visible_text"}
```

---

## 10. 本地 hard gate

Omni pair-level proposal 不是直接 accepted。它会先经过本地 gate。

本地 gate 的主要职责：

```text
拦掉明确不能进入数据集的样本。
```

典型 hard reject：

| issue | 说明 |
|---|---|
| `fallback_pair_proposal` | Omni 失败后用了 fallback 输出，不能入库。 |
| `reference_video_missing` | reference 文件不存在。 |
| `target_video_missing` | target 文件不存在。 |
| `visible_text is diagnostic-only...` | visible_text 是最终禁用类型。 |
| `speech is diagnostic-only...` | speech 是最终禁用类型。 |
| `low_pair_video_confidence` | pair-level confidence 低于当前 profile 阈值。 |
| `bad_edit_text_quality` | edit_text 质量差。 |
| `weak_attribute_wording` | 属性差异只是措辞差异。 |
| `text_driven_product_overlay_change` | 产品变化实际由包装文字/OCR 驱动。 |

典型 review-required：

| issue | 说明 |
|---|---|
| `transient_delta_not_segment_wide` | 差异只在片段末尾短暂出现。 |
| `segment_internal_transition` | reference 或 target 内部自己发生多阶段变化。 |
| `composition_label_mismatch` | edit_text 对构图描述夸大或错误。 |
| `subject_role_mismatch` | PIP/inset 主体和 main speaker 混淆。 |

review-required 不是马上硬拒，但如果 final Omni verifier 也发现问题，就会拒绝。

---

## 11. Final Omni verification gate

Final verifier 返回后，本地会再转换成 issues。

拒绝条件包括：

| issue | 说明 |
|---|---|
| `final_omni_reject` | final verifier 明确拒绝。 |
| `final_omni_low_confidence` | final confidence 低于 profile 阈值。 |
| `final_omni_quality_score_below_threshold` | `quality_score < 0.70`。 |
| `final_omni_reference_satisfies_edit` | reference 已经满足 edit，不需要编辑。 |
| `final_omni_target_missing_edit` | target 不满足 edit。 |
| `final_omni_missing_observable_delta` | 无可见主差异。 |
| `final_omni_not_single_primary_delta` | 不是单一主差异。 |
| `final_omni_text_or_ocr_driven` | 文字/OCR 驱动。 |
| `final_omni_delta_not_segment_wide` | 主差异不稳定覆盖 target。 |
| `final_omni_edit_text_inaccurate` | edit_text 不准确。 |

只有没有 blocking issues 时，pair 才能 accepted。

---

## 12. Accepted 的精确定义

当前 accepted 逻辑可以概括为：

```python
accepted =
    pair_level_omni_accept
    and local_hard_gate_passed
    and final_omni_accept
    and final_omni_quality_score >= 0.70
    and not duplicate_delta_family
```

更具体地说：

1. pair-level Omni 必须 `accept=true`。
2. 本地 hard gate 不能有 hard reject。
3. final Omni verifier 必须 `accept=true`。
4. final verifier 的 `quality_score >= 0.70`。
5. final verifier 必须认为：
   - reference 不满足 edit。
   - target 满足 edit。
   - 有 observable delta。
   - 是 single primary delta。
   - 不是 text/OCR driven。
   - segment-wide。
   - edit_text accurate。
6. 同一个源视频中相同 `delta_family` 仍然只保留更靠前/更高质量的一条。

### 12.1 关于 cap 的最新修复

以前有一个限制：

```text
每个源视频最多 accepted 3 条
```

这导致质量过关样本被标成：

```text
single_source_accept_cap_exceeded
```

用户明确要求：

```text
质量过关就可以作为数据集，不要因为 cap 截掉。
```

最新代码已经修复：

```text
commit: fb77b64
```

现在：

- `max_accepted_pairs_per_source` 参数仍保留兼容，但不再把质量过关样本踢出 `accepted_pairs.jsonl`。
- `single_source_accept_cap_exceeded` 不再产生。
- 但 `duplicate_delta_family` 仍然会拒绝，因为这是去重，不是数量 cap。

---

## 13. Delta family 去重

同一个源视频中，多个 pair 可能描述本质相同的变化。例如：

```text
add product overlay
add product overlay
add static product image overlay
```

这些不应该全部进入最终 accepted，否则数据会重复。

代码会给 pair 生成：

```text
single_source_delta_family
```

常见 family：

```text
add_pip_demo
pip_demo_to_product_overlay
pip_subject_change
add_product_overlay
product_overlay_change
product_closeup_claim
action:{hash}
scene:{hash}
attribute:{hash}
```

同一个 family 只保留一条 accepted，其余标为：

```text
duplicate_delta_family:{family}
```

---

## 14. 当前已完成的数据集

数据集路径：

```text
/data02/usr/wangqihao/Demo/test/data/
```

这是已经导出的最终样本目录，每个子目录就是一个可用于人工检查或后续训练/评测的数据样本。

本轮已完成数据集对应的原始 overnight run 目录是：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/
```

这个 run 目录保存了导出最终数据集之前的全部中间产物，包括每个 source 的 selected source、切片计划、Omni annotation、pair candidates、ranked pairs、accepted pairs、review bundle 和日志。

详细 schema：

```text
/Users/Admin/Desktop/omni-runs/DATASET_SCHEMA.md
```

当前数据集统计：

| 项目 | 数量 |
|---|---:|
| 样本数 | 943 |
| 源视频数 | 193 |
| daily_omni 源视频 | 94 |
| worldsense 源视频 | 99 |
| daily_omni 样本 | 419 |
| worldsense 样本 | 524 |

历史导出中还有：

| 类别 | 数量 |
|---|---:|
| 原 accepted | 497 |
| 原 cap_exceeded 但质量过关 | 446 |

注意：这是在 `cf79b61` 跑出来的 overnight 数据。当时 cap 还没修，所以质量过关样本一部分被截成 `cap_exceeded`。现在 `fb77b64` 已修掉，后续再跑会直接收进 accepted。

差异类型分布：

| 类型 | 数量 |
|---|---:|
| `scene` | 513 |
| `object_presence` | 276 |
| `action` | 101 |
| `attribute` | 45 |
| `object_count` | 7 |
| `audio_event` | 1 |

---

## 15. 最终样本文件结构

每个样本一个文件夹：

```text
00001_daily_omni_daily_omni_-BAFzpKigw/
├── reference.mp4
├── target.mp4
├── edit_text.txt
├── info.json
├── reference_annotation.json
├── target_annotation.json
├── reference_omni_description.txt
└── target_omni_description.txt
```

### 15.1 `reference.mp4`

6 秒 reference segment。

### 15.2 `target.mp4`

6 秒 target segment。

### 15.3 `edit_text.txt`

一行短文本，描述 reference 到 target 的变化。

例子：

```text
add a second trading card featuring Yao Ming and Tracy McGrady
```

```text
change the scene from a close-up of hands playing a guitar to a medium shot of a young man and woman playing guitars together
```

### 15.4 `info.json`

核心字段：

| 字段 | 说明 |
|---|---|
| `edit_text` | 编辑文本。 |
| `reference_clip_id` | reference segment id。 |
| `target_clip_id` | target segment id。 |
| `source` | 源视频 id/name。 |
| `accepted` | 老数据中可能受 cap 影响；新代码下表示质量通过。 |
| `final_omni_accept` | final Omni 是否接受。 |
| `final_omni_quality_score` | final Omni 分数。 |
| `difference_type` | 主差异类型。 |
| `reference_caption` | reference 描述。 |
| `target_caption` | target 描述。 |
| `dominant_delta` | 主差异对象。 |
| `discarded_deltas` | 被忽略的次要差异。 |
| `pair_video_evidence` | Omni 对差异的证据。 |
| `issues` | 问题标签。 |
| `reference_annotation` | reference Omni annotation 摘要。 |
| `target_annotation` | target Omni annotation 摘要。 |

### 15.5 annotation 文件

```text
reference_annotation.json
target_annotation.json
```

包含 segment 的完整结构化 Omni 标注。

### 15.6 Omni description 文本

```text
reference_omni_description.txt
target_omni_description.txt
```

主要保存更完整的人类可读 Omni 原始观察，如：

- visual observations
- audio observations
- timeline
- speaker identification
- uncertainty

---

## 16. Batch run 输出结构

批量跑数脚本：

```text
scripts/run_single_source_omni_batch.sh
```

典型 run root：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_{HEAD}_{timestamp}
```

当前已经完成并用于导出 943 条样本的 run root：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050
```

主要输出：

```text
batch_source_manifest.jsonl
batch_status.jsonl
batch_ranked_pairs.jsonl
batch_accepted_pairs.jsonl
batch_summary.md
manual_review/accepted/
manual_review/diagnostic/
sources/{dataset}_{source_id}/
```

对应到服务器绝对路径是：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/batch_source_manifest.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/batch_status.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/batch_ranked_pairs.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/batch_accepted_pairs.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/batch_summary.md
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/manual_review/accepted/
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/manual_review/diagnostic/
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/sources/
```

`composed_omni_retrieval` 数据根目录里也有一个 runs 软链接：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/runs
```

它指向：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs
```

所以同一个历史 run 也可通过下面这个路径访问：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/runs/single_source_batch6_overnight_cf79b61_20260508_014050
```

其中每个 source 独立目录：

```text
sources/{dataset}_{source_id}/
├── selected_source_video.json
├── selected_source_manifest.jsonl
├── single_source_clip_plan.jsonl
├── single_source_clip_groups.jsonl
├── extracted_single_source_whole.jsonl
├── extracted_single_source_clips.jsonl
├── single_source_whole_annotation.jsonl
├── single_source_annotations.jsonl
├── single_source_pair_candidates.jsonl
├── single_source_pair_report.md
├── ranked_single_source_pairs.jsonl
├── accepted_pairs.jsonl
├── single_source_review_bundle/
├── logs/
└── status.json
```

对应到服务器绝对路径模板：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/selected_source_video.json
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/selected_source_manifest.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_clip_plan.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_clip_groups.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/extracted_single_source_whole.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/extracted_single_source_clips.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_whole_annotation.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_annotations.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_pair_candidates.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_pair_report.md
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/ranked_single_source_pairs.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/accepted_pairs.jsonl
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/single_source_review_bundle/
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/logs/
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/<RUN_NAME>/sources/<dataset>_<source_id>/status.json
```

如果要找某一个具体源视频的中间产物，可以先列 source 目录：

```bash
RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050
find "$RUN_ROOT/sources" -maxdepth 1 -mindepth 1 -type d | head
```

再进入某个 source 目录查看：

```bash
SOURCE_RUN=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/sources/某个source目录名
ls -lh "$SOURCE_RUN"
```

### 16.0 中间产物逐层说明

下面按流水线顺序说明每个中间产物放在哪里，以及它的用途。`<RUN_ROOT>` 指：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050
```

`<SOURCE_RUN>` 指：

```text
<RUN_ROOT>/sources/<dataset>_<source_id>
```

#### 16.0.1 Batch source manifest

绝对路径：

```text
<RUN_ROOT>/batch_source_manifest.jsonl
```

真实历史路径：

```text
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_cf79b61_20260508_014050/batch_source_manifest.jsonl
```

用途：

```text
记录本次 batch 选中的所有 source video / source window。
```

每行包含：

- `job_id`
- `source_clip_id`
- `dataset`
- `source_path`
- `duration_seconds`
- `source_window_start_seconds`
- `source_window_duration_seconds`
- `media_probe`
- `selection_notes`

#### 16.0.2 Batch status

绝对路径：

```text
<RUN_ROOT>/batch_status.jsonl
```

用途：

```text
记录每个 source job 成功、失败、accepted 数、ranked 数、失败原因。
```

排查 overnight 是否有 source 失败时先看它。

#### 16.0.3 Batch ranked pairs

绝对路径：

```text
<RUN_ROOT>/batch_ranked_pairs.jsonl
```

用途：

```text
汇总所有 source 的 ranked_single_source_pairs.jsonl。
```

它包含 accepted 和 rejected，是排查模型/规则问题最重要的总文件。

#### 16.0.4 Batch accepted pairs

绝对路径：

```text
<RUN_ROOT>/batch_accepted_pairs.jsonl
```

用途：

```text
汇总所有 source 最终质量通过的 accepted pair。
```

从 `fb77b64` 开始，它不再被每源 3 条 cap 截断。

#### 16.0.5 Batch summary

绝对路径：

```text
<RUN_ROOT>/batch_summary.md
```

用途：

```text
汇总本次 batch 的 source 数、成功失败数、ranked pair 数、accepted pair 数、拒绝原因分布等。
```

#### 16.0.6 Manual review

绝对路径：

```text
<RUN_ROOT>/manual_review/accepted/
<RUN_ROOT>/manual_review/diagnostic/
```

用途：

- `accepted/`：人工优先审核的样本。
- `diagnostic/`：拒绝样本和问题样本，用于修 prompt / gate。

#### 16.0.7 Selected source

绝对路径：

```text
<SOURCE_RUN>/selected_source_video.json
```

用途：

```text
记录这个 source job 使用的是哪个原始视频，以及取的是哪个 30s window。
```

对 Daily-Omni，通常：

```text
source_window_start_seconds = 0
source_window_duration_seconds = 30
```

对 WorldSense，通常：

```text
source_window_start_seconds = min(30s, duration*0.25 后修正)
source_window_duration_seconds = 30
```

#### 16.0.8 Whole window manifest

绝对路径：

```text
<SOURCE_RUN>/selected_source_manifest.jsonl
```

用途：

```text
记录要切出的 30s whole window。
```

这个 whole window 后续会给 Omni 做全局描述。

#### 16.0.9 Segment plan

绝对路径：

```text
<SOURCE_RUN>/single_source_clip_plan.jsonl
```

用途：

```text
记录 5 个 6s segment 的 start/end/output_path。
```

典型数量：

```text
5 行
```

#### 16.0.10 Clip groups

绝对路径：

```text
<SOURCE_RUN>/single_source_clip_groups.jsonl
```

用途：

```text
记录这 5 个 segment 属于同一个 single_source group。
```

后续 pair mining 只在这个 group 内部做。

#### 16.0.11 Extracted whole video manifest

绝对路径：

```text
<SOURCE_RUN>/extracted_single_source_whole.jsonl
```

用途：

```text
记录 ffmpeg 实际切出的 30s whole window 文件。
```

对应视频通常在数据根目录下：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/single_source/<source_id>/<source_id>__whole_window.mp4
```

#### 16.0.12 Extracted segment manifest

绝对路径：

```text
<SOURCE_RUN>/extracted_single_source_clips.jsonl
```

用途：

```text
记录 ffmpeg 实际切出的 5 个 6s segment 文件。
```

对应视频通常在：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/single_source/<source_id>/<source_id>__single_001.mp4
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/single_source/<source_id>/<source_id>__single_002.mp4
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/single_source/<source_id>/<source_id>__single_003.mp4
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/single_source/<source_id>/<source_id>__single_004.mp4
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/single_source/<source_id>/<source_id>__single_005.mp4
```

#### 16.0.13 Whole annotation

绝对路径：

```text
<SOURCE_RUN>/single_source_whole_annotation.jsonl
```

用途：

```text
Omni 对 30s whole window 的全局结构化理解。
```

后续 pair-level 对比会把它作为上下文传给 Omni。

#### 16.0.14 Segment annotations

绝对路径：

```text
<SOURCE_RUN>/single_source_annotations.jsonl
```

用途：

```text
Omni 对每个 6s segment 的细粒度结构化描述。
```

典型数量：

```text
5 行
```

每行对应一个 segment。

#### 16.0.15 Pair candidates

绝对路径：

```text
<SOURCE_RUN>/single_source_pair_candidates.jsonl
```

用途：

```text
同源 5 个 segment 两两组合得到的 chronological pair 候选。
```

典型数量：

```text
10 行
```

这些只是候选，不是最终样本。

#### 16.0.16 Pair report

绝对路径：

```text
<SOURCE_RUN>/single_source_pair_report.md
```

用途：

```text
记录本 source 的 segment 数、candidate 数、候选类型分布、fallback 情况。
```

#### 16.0.17 Ranked single-source pairs

绝对路径：

```text
<SOURCE_RUN>/ranked_single_source_pairs.jsonl
```

用途：

```text
保存 pair-level Omni 对比 + 本地 gate + final Omni verification 后的完整结果。
```

这是单个 source 最重要的调试文件。

里面每行包含：

- `edit_text`
- `difference`
- `dominant_delta`
- `reference_state`
- `target_state`
- `delta_temporal_extent`
- `subject_roles`
- `discarded_deltas`
- `pair_video_evidence`
- `local_gate_report`
- `final_omni_verification`
- `single_source_pair_acceptance_issues`
- `accepted`
- `raw_model_output`
- `raw_final_omni_output`

#### 16.0.18 Accepted pairs

绝对路径：

```text
<SOURCE_RUN>/accepted_pairs.jsonl
```

用途：

```text
保存这个 source 最终质量通过的 pair。
```

从 `fb77b64` 开始，它包括所有质量过关样本，不再因为 per-source cap 截掉。

#### 16.0.19 Source review bundle

绝对路径：

```text
<SOURCE_RUN>/single_source_review_bundle/
```

用途：

```text
单个 source 的人工审核材料。
```

通常包含：

- `source_30s.mp4`
- `segments/`
- `segment_descriptions.md`
- `all_pair_ranking.md`
- `top_pairs/`
- `pair_review/accepted/`
- `pair_review/diagnostic/`

#### 16.0.20 Logs

绝对路径：

```text
<SOURCE_RUN>/logs/
```

用途：

```text
保存这个 source 的 select / extract / annotate / propose / review 阶段日志。
```

#### 16.0.21 Status

绝对路径：

```text
<SOURCE_RUN>/status.json
```

用途：

```text
记录该 source job 的最终状态，成功/失败、accepted 数、ranked 数、错误原因等。
```

### 16.1 `ranked_single_source_pairs.jsonl`

保存全部 pair 的详细记录，无论 accepted 还是 rejected。

它是最重要的调试文件。

每行包含：

```text
proposal_id
candidate_id
reference_clip_id
target_clip_id
reference_video
target_video
edit_text
modalities
reference_caption
target_caption
difference
dominant_delta
reference_state
target_state
delta_temporal_extent
subject_roles
is_segment_wide_delta
discarded_deltas
pair_video_evidence
confidence
model_accepted
final_omni_accept
final_omni_verification
local_gate_report
single_source_delta_family
single_source_pair_acceptance_issues
quality
heuristic_quality
judge
verification
observable_difference
accepted
raw_model_output
raw_final_omni_output
```

### 16.2 `accepted_pairs.jsonl`

保存当前 source 质量通过的 pair。

从 `fb77b64` 开始，它不再受每源 cap 截断。

### 16.3 `batch_accepted_pairs.jsonl`

汇总所有 source 的 accepted pairs。

### 16.4 `manual_review/accepted`

给人工审核看的目录。

### 16.5 `manual_review/diagnostic`

保存 rejected 或 diagnostic 样本，用来分析 gate 和 prompt 的问题。

---

## 17. 人工审核时怎么看

人工审核优先看：

```text
manual_review/accepted/
```

每条样本需要打开：

```text
reference.mp4
target.mp4
edit_text.txt
description.md 或 info.json
```

人工判断顺序：

1. 先看 `edit_text`。
2. 打开 `reference.mp4`，确认 reference 不满足 edit。
3. 打开 `target.mp4`，确认 target 满足 edit。
4. 判断差异是否一眼可见。
5. 判断 edit_text 有没有夸大或写错主体。
6. 判断是否被文字/OCR/speech 偷偷驱动。
7. 判断是否只是末尾一瞬变化。
8. 判断是否有更强未描述差异。

人工应拒绝：

- reference 和 target 没明显差异。
- edit_text 描述的是画面文字变化。
- target 只是最后一帧出现变化。
- edit_text 写 `product close-up`，但其实还是 speaker + product overlay。
- edit_text 写主 speaker 变化，但变化人物其实在 PIP/inset 里。
- speech/transcript 是主差异。
- visible text/OCR 是主差异。

---

## 18. 常见好样本类型

### 18.1 场景/构图变化

reference：

```text
close-up of hands playing guitar
```

target：

```text
medium shot of two people playing guitars
```

edit_text：

```text
change the scene from a close-up of hands playing a guitar to a medium shot of two people playing guitars together
```

### 18.2 物体出现

reference：

```text
speaker talks without a product overlay
```

target：

```text
speaker talks with a product image overlay on the left
```

edit_text：

```text
add a static product image overlay on the left
```

### 18.3 PIP / inset 出现

reference：

```text
face-only talking-head segment
```

target：

```text
talking-head segment with a picture-in-picture brow treatment demonstration
```

edit_text：

```text
add a picture-in-picture brow treatment demonstration
```

### 18.4 动作变化

reference：

```text
person speaking to camera
```

target：

```text
person applying mascara
```

edit_text：

```text
change the action from speaking to applying mascara
```

---

## 19. 常见坏样本类型

### 19.1 文字/OCR 驱动

坏：

```text
change the product image from "Revision Skincare Brow-Lift Roller" to "Revlon Professional Smooth Line Reliever Pen"
```

如果这个变化主要来自包装文字或屏幕文字，就不能收。

### 19.2 没有明显差异

reference 和 target 几乎一样，即使 Omni 写了 edit_text，也不能收。

### 19.3 弱属性措辞

坏：

```text
change the blouse to a shirt
change long brown hair to long hair
```

这种通常不是可验证主差异。

### 19.4 夸大构图

坏：

```text
change the shot to a product close-up
```

但 target 里 speaker 仍然占主画面，只是旁边多了 product overlay。

正确应该写：

```text
add a static product image overlay beside the speaker
```

### 19.5 PIP 主体误判

坏：

```text
change the speaker from a woman to a man
```

但 target 里的 man 只是 inset/PIP 里的小画面人物，主 speaker 仍然是 woman。

正确应该写：

```text
add an inset video showing a man speaking
```

前提是 inset man 持续足够长，不是末尾一瞬。

---

## 20. 跑数命令

### 20.1 单 source smoke

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main || exit 1
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

git fetch origin
git checkout main
git pull --ff-only origin main
HEAD=$(git rev-parse --short HEAD)
echo "HEAD=$HEAD"

ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
WORLDSENSE_ROOT=$ROOT/raw_datasets/worldsense/_extracted
RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_cap_smoke_${HEAD}_$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN_ROOT/logs"

timeout 7200 bash scripts/run_single_source_omni_batch.sh \
  --root "$ROOT" \
  --run-root "$RUN_ROOT" \
  --model qwen3-omni \
  --base-url http://127.0.0.1:8093/v1 \
  --segment-seconds 6 \
  --daily-source-count 0 \
  --worldsense-source-count 1 \
  --worldsense-root "$WORLDSENSE_ROOT" \
  --max-parallel-jobs 1 \
  --max-accepted-pairs-per-source 3 \
  --acceptance-profile exploration \
  > "$RUN_ROOT/logs/batch_one_source.log" 2>&1

tail -80 "$RUN_ROOT/logs/batch_one_source.log"
wc -l "$RUN_ROOT/batch_ranked_pairs.jsonl" "$RUN_ROOT/batch_accepted_pairs.jsonl" 2>/dev/null
```

注意：`--max-accepted-pairs-per-source 3` 现在只是兼容参数，不会截掉质量通过样本。

### 20.2 Overnight batch

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main || exit 1
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

git fetch origin
git checkout main
git pull --ff-only origin main
HEAD=$(git rev-parse --short HEAD)
echo "HEAD=$HEAD"

ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
WORLDSENSE_ROOT=$ROOT/raw_datasets/worldsense/_extracted
RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/single_source_batch6_overnight_${HEAD}_$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN_ROOT/logs"

nohup bash scripts/run_single_source_omni_batch.sh \
  --root "$ROOT" \
  --run-root "$RUN_ROOT" \
  --model qwen3-omni \
  --base-url http://127.0.0.1:8093/v1 \
  --segment-seconds 6 \
  --daily-source-count 100 \
  --worldsense-source-count 100 \
  --worldsense-root "$WORLDSENSE_ROOT" \
  --max-parallel-jobs 4 \
  --max-accepted-pairs-per-source 3 \
  --acceptance-profile exploration \
  > "$RUN_ROOT.batch.log" 2>&1 &

echo "started pid=$!"
echo "RUN_ROOT=$RUN_ROOT"
echo "tail -f $RUN_ROOT.batch.log"
```

监控：

```bash
tail -f "$RUN_ROOT.batch.log"
wc -l "$RUN_ROOT/batch_status.jsonl" "$RUN_ROOT/batch_ranked_pairs.jsonl" "$RUN_ROOT/batch_accepted_pairs.jsonl" 2>/dev/null
```

日志中每个 pair 会输出：

```text
[propose-single-source-pairs] wrote proposal_count=... accepted_current=... final_omni_quality_score=... difference_type=... issues=... edit_text=...
```

用户要求过：每生成一个样本都要输出。因此这些逐 pair 日志不要删。

---

## 21. 并发策略

当前 batch 并发方式：

```text
每个 source 是一个独立 job
job 内 concurrency = 1
多个 source 并行，让 vLLM 自己 batch
```

推荐：

```text
smoke: max_parallel_jobs = 1 或 2
overnight: max_parallel_jobs = 4
```

当前服务器 Omni 服务在 GPU 0/1 A6000 上跑。不要在脚本里启动或关闭 Omni，只作为 client 调用：

```text
http://127.0.0.1:8093/v1
```

---

## 22. 测试

本地或服务器更新代码后先跑：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main || exit 1
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src
python -m unittest tests.test_scripts tests.test_composed_data tests.test_composed_omni -v
```

当前最新状态：

```text
250 tests OK
```

与当前方法强相关的测试包括：

- 30s + 6s 会得到 5 segments / 10 pairs。
- WorldSense 长视频会生成 30s window metadata。
- single-source pair 必须用 video comparison，不允许 attribute fallback。
- final Omni score 低于 0.70 会拒绝。
- text-driven product label change 会拒绝。
- duplicate delta family 会去重。
- 质量通过样本不会因为 source cap 被截掉。
- review bundle 必须包含 segment descriptions、ranked pairs、contact sheet、local gate report、final verifier report。

---

## 23. 当前方法与 Omni-Captioner 思想的关系

当前方法吸收了 Omni-Captioner / Omni-Detective 的核心思想：

```text
先细粒度观察，再基于证据生成描述。
```

但我们没有让 Omni 直接黑盒全量搜索 pair，而是拆成：

```text
固定切片
-> 固定同源 pair 枚举
-> Omni 细粒度 annotation
-> Omni pair-level 对比
-> 本地 gate
-> Omni final verification
```

这样做的好处：

- 可诊断。
- 可复跑。
- 每个阶段都有 JSONL 输出。
- 每个拒绝都有 issue。
- 不会卡在“Omni 全量搜索 pair”这种黑盒阶段。
- 人工审核可以追溯到 segment annotation 和 final verification evidence。

---

## 24. 这套方法的优点

1. 不需要生成视频，避免 VACE 质量不稳定。
2. reference/target 来自同一个源视频，上下文天然相近。
3. 6s segment 足够短，人工审核快。
4. 30s window 足够长，一个源视频可产生多个 pair。
5. Omni annotation 提供完整语义证据。
6. final verifier 提供第二次模型审查。
7. 本地 hard gate 控制禁区。
8. 所有中间结果可追踪。
9. batch 可并发扩展到 hundreds of sources。

---

## 25. 这套方法的局限

1. 同源视频内部不一定总有足够明显变化。
2. talking-head 视频可能差异偏弱，需要依赖 PIP、产品、动作、构图变化。
3. WorldSense 长视频取 30s window 有随机性，可能选到变化不足的窗口。
4. Omni 仍可能误判主体、PIP、构图，需要人工抽查。
5. 当前禁用了 speech / visible_text，牺牲了部分潜在样本数量。
6. duplicate delta family 去重可能会保守地丢掉一些可用但相似的样本。

---

## 26. 后续可优化方向

### 26.1 更智能的 30s window selection

当前 WorldSense 只取一个固定策略窗口。后续可让本地视频特征或 Omni 粗扫选择变化最丰富的 30s。

### 26.2 更细的 delta family

当前 delta family 对 product overlay / PIP 有规则，但对 scene/action 主要用 hash。后续可细化：

```text
scene_closeup_to_medium
object_added_card
action_speaking_to_demonstrating
composition_full_body_to_closeup
```

### 26.3 更强的人工审核工具

可以生成网页 gallery：

- 同屏播放 reference/target。
- 展示 edit_text。
- 展示 final score。
- 一键 accept/reject。
- 回写人工标签。

### 26.4 引入 hard negatives

当前样本保留了一些 hard negative 路径，但最终导出中没有重点使用。后续 CVR 训练/评测可以加入：

```text
reference + edit_text -> target
reference + edit_text -> close but wrong negatives
```

### 26.5 可控放开 audio_event

现在 audio_event 很少。后续可以只在视觉几乎一致、非语音声音清楚时增强 audio_event 样本。

---

## 27. 最重要的质量边界

如果只能记住几条，就是：

1. 当前数据集是 `reference video + edit_text -> target video`。
2. 当前主线是同一个源视频内部的 6s segment pair。
3. Omni 不是一次性决定一切，而是 annotation + pair comparison + final verification。
4. `final_omni_quality_score >= 0.70` 才能进。
5. visible_text、OCR、speech 不作为最终主差异。
6. 质量通过样本不再因每源 cap 被截掉。
7. duplicate delta family 仍然去重。
8. 人工审核以 `manual_review/accepted` 为主，`diagnostic` 用来修规则。
9. 新电脑/服务器直接拉 `main@fb77b64` 或更新版本。
10. 不要再把 VACE 当当前大规模数据构造主线。
