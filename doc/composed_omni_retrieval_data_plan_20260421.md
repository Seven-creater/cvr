# Omni Composed Video Retrieval 数据构造计划

Last updated: 2026-04-21

## 1. 目标

我们要构造一个新的 **Omni 全模态组合视频检索** 数据集，任务形式是：

```text
reference video + edit text + audio/visual cues -> target video
```

这个任务不再把 MSRVTT 当主数据集。MSRVTT 适合验证检索链路，但不适合证明 agentic composed retrieval，因为它缺少天然的 reference-target-edit 三元组，也缺少细粒度音频/视觉差异标注。

当前阶段的目标是：

- 先构造 5-10 条高质量 pilot 样本。
- 先证明数据流程可行，而不是直接做大规模训练集。
- 让 agent 只需要证明“能用”：能利用 reference video、edit text 和音频/视觉细节找到 target video。

## 2. 服务器模型准备

所有新模型都从 **ModelScope** 下载，不使用 Hugging Face fallback。

ModelScope 模型源：

- `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- `Qwen/Qwen3-Omni-30B-A3B-Captioner`

目标路径：

```text
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-captioner
```

模型分工：

- `qwen3-omni-30b-a3b-captioner`: 细粒度视频/音频描述。
- `qwen3-omni-30b-a3b-instruct`: 差异归纳、edit text 生成、样本质检。
- 现有 `qwen2.5-omni`, `qwen2-vl`, `qwen2-audio`: fallback 和对照。

服务器 AI 可直接运行：

```bash
mkdir -p /data02/usr/wangqihao/Demo/test/cvr/runs/model_download_20260421

nohup bash /data02/usr/wangqihao/Demo/test/cvr/scripts/download_qwen3_omni_modelscope.sh \
  > /data02/usr/wangqihao/Demo/test/cvr/runs/model_download_20260421/modelscope_qwen3_omni_download.log 2>&1 &
```

下载完成后回传：

- `modelscope_qwen3_omni_download.log` 最后 120 行
- 两个模型目录的 `du -sh`
- 两个模型目录是否存在 `config.json`
- 两个模型目录各有多少个 `.safetensors`
- 两个模型目录顶层文件列表

## 3. 原始视频采集策略

第一批原始材料先从国内网站人工找，目标是熟悉数据流程。先不追求版权可发布，只做内部研究 pilot；正式发布数据集时再换成授权或开放来源。

优先找这些视频：

1. **同一账号/同一系列**
   - 同一宠物账号、同一厨艺账号、同一手工账号、同一动画剪辑账号。
   - 好处是背景、主体、风格天然相似。

2. **固定机位**
   - 厨房、桌面手工、宠物房间、健身、乐器演奏。
   - 好处是 reference 和 target 只差一个关键编辑点。

3. **重复角色或重复场景**
   - 一只猫 vs 两只猫。
   - 一个人跳舞 vs 两个人跳舞。
   - 有球 vs 没球。

4. **明确音频事件**
   - 狗叫、猫叫、掌声、音乐开始/停止、有人说话、机器声、车辆声。
   - 用于构造 audio-required 样本。

5. **动画/游戏/影视二创片段**
   - 很容易出现相似场景差异。
   - 只建议先做内部 pilot，正式公开时要谨慎处理版权。

第一批人工采集规模：

```text
20-30 个原始视频
切出 30-50 个 3-15 秒 clip
筛出 5-10 条 composed retrieval pilot 样本
```

## 4. 数据目录

统一放到：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
```

目录结构：

```text
raw/
clips/
metadata/
captions/
pairs/
splits/
reports/
caches/
```

建议含义：

- `raw/`: 原始下载视频。
- `clips/`: 裁剪后的短 clip。
- `metadata/`: URL、平台、标题、作者、下载时间、授权备注。
- `captions/`: 模型生成的结构化描述。
- `pairs/`: reference-target-edit 样本。
- `splits/`: pilot/dev/test 划分。
- `reports/`: 人工检查报告。
- `caches/`: 中间模型输出缓存。

## 5. 样本格式

每条 pilot 样本固定为 JSONL 一行：

```json
{
  "sample_id": "covr_pilot_0001",
  "reference_video": "clips/xxx_ref.mp4",
  "target_video": "clips/xxx_target.mp4",
  "edit_text": "change one cat into two cats",
  "modalities": ["visual", "audio"],
  "reference_caption": "...",
  "target_caption": "...",
  "difference": {
    "type": "object_count",
    "from": "one cat",
    "to": "two cats"
  },
  "hard_negatives": [
    "clips/xxx_neg1.mp4",
    "clips/xxx_neg2.mp4",
    "clips/xxx_neg3.mp4"
  ],
  "quality": {
    "same_context_score": 0.0,
    "edit_match_score": 0.0,
    "target_uniqueness_score": 0.0
  },
  "source": {
    "platform": "bilibili",
    "url": "...",
    "license_note": "internal research pilot only"
  }
}
```

第一版重点差异类型：

- `object_count`: 数量变化。
- `object_presence`: 对象出现/消失。
- `attribute`: 颜色/大小/状态变化。
- `action`: 动作变化。
- `scene`: 场景变化。
- `audio_event`: 音频事件变化。
- `speech`: 语音有无或内容变化。

## 6. 构造流程

1. **采集 raw 视频**
   - 人工记录平台、URL、标题、作者、下载时间。
   - 不采集私密内容，不绕过 DRM。

2. **切 clip**
   - 每个 clip 3-15 秒。
   - 保留原音频。
   - 一个 clip 尽量只有一个主要事件。

3. **细粒度描述**
   - Captioner 输出：主体、数量、动作、场景、物体属性、可见文字、语音、音频事件。
   - Instruct 复核是否过宽或幻觉。

4. **候选配对**
   - 找背景/主体/风格相似，但关键差异明确的 clip pair。
   - 优先选单一差异，避免一个 pair 同时变了太多东西。

5. **生成 edit text**
   - edit text 只描述变化，不复述完整视频。
   - 例子：
     - `change one cat into two cats`
     - `replace the quiet background with dog barking`
     - `change the person from standing still to dancing`

6. **质量过滤**
   - reference 不满足 edit。
   - target 满足 edit。
   - 背景/主体足够相似。
   - 差异清楚、可验证。
   - 涉及音频时，Omni 或 Audio 模型必须确认音频差异。

7. **构造 hard negatives**
   - 与 reference 很像但不满足 edit。
   - 满足 edit 但背景/主体不像。
   - 视觉像但音频不对，或音频像但视觉不对。

## 7. Pilot 验收

Pilot 输出：

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/pilot_10/pilot_10.jsonl
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/pilot_10/gallery.jsonl
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/pilot_10/reports/pilot_review.md
```

验收标准：

- 5-10 条里至少 70% 人工认为合理。
- 每条都有清楚的 reference/target/edit 三元关系。
- 至少 2 条包含音频差异。
- 至少 2 条是数量或对象变化。
- 至少 1 条是动作变化。
- agent 能在小 gallery 中跑通 `reference video + edit text -> target video`。

## 8. 给服务器 AI 的第一条任务

```text
不要改代码。现在只做 ModelScope 模型下载。

1. 进入仓库：
cd /data02/usr/wangqihao/Demo/test/cvr
git pull
git rev-parse HEAD

2. 确认脚本存在：
ls -lh scripts/download_qwen3_omni_modelscope.sh

3. 用 nohup 启动下载：
mkdir -p /data02/usr/wangqihao/Demo/test/cvr/runs/model_download_20260421
nohup bash /data02/usr/wangqihao/Demo/test/cvr/scripts/download_qwen3_omni_modelscope.sh > /data02/usr/wangqihao/Demo/test/cvr/runs/model_download_20260421/modelscope_qwen3_omni_download.log 2>&1 &

4. 下载完成后回传：
- modelscope_qwen3_omni_download.log 最后 120 行
- 两个模型目录的 du -sh
- 两个模型目录是否存在 config.json
- 两个模型目录各有多少个 .safetensors
- 两个模型目录顶层文件列表

不要使用 Hugging Face，不要重启任何 vLLM/Omni 服务。
```

## 9. 下载完成后的数据归一化任务

Daily-Omni 和 WorldSense 下载完成后，先不要直接大规模调用 Qwen3-Omni。先把两个原始数据源统一整理成 source rows 和 source clips。

服务器 AI 可直接运行：

```text
不要改代码。现在只做 Daily-Omni 和 WorldSense 的数据归一化，不调用 Qwen3-Omni 推理。

1. 进入仓库并同步：
cd /data02/usr/wangqihao/Demo/test/cvr
git pull
git rev-parse HEAD

2. 安装 parquet 读取依赖：
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src
python -m pip install -U pyarrow

3. 用 nohup 运行 source prepare：
mkdir -p /data02/usr/wangqihao/Demo/test/cvr/runs/composed_source_prepare_20260422
nohup bash /data02/usr/wangqihao/Demo/test/cvr/scripts/prepare_composed_sources.sh \
  > /data02/usr/wangqihao/Demo/test/cvr/runs/composed_source_prepare_20260422/source_prepare.log 2>&1 &
```

完成后回传：

- `source_prepare.log` 最后 120 行
- `source_dataset_prepare_summary.md` 全文
- `source_rows.jsonl` 前 5 行
- `source_clips_all.jsonl` 前 5 行
- `source_clips_pilot*.jsonl` 前 10 行

这一步的输出是后续 `annotate-clips -> propose-pairs -> validate-pilot` 的输入。
