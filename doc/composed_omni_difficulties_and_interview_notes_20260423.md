# Omni 组合视频检索数据构造：难点、解决方案与面试表达

日期：2026-04-24

## 1. 这份文档是干什么的

这份文档不写流水账实验记录，而是专门回答四个问题：

1. 我们到底在做什么。
2. 这个项目真正难在哪里。
3. 每个难点后来是怎么被拆开、定位、修掉的。
4. 如果拿去面试，应该怎么把技术价值讲清楚。

它适合作为：

- 项目内部复盘文档
- 论文/报告前的技术整理
- 面试与求职时的项目讲稿底稿

---

## 2. 一句话讲清项目

我们做的不是普通的 video-text retrieval，而是 **Omni 全模态 composed video retrieval 数据构造**。

普通检索通常是：

```text
text query -> target video
```

我们真正想要的数据形式是：

```text
reference video + edit text + visual/audio cues -> target video
```

也就是说，给模型一个参考视频，再给一条“应该发生什么变化”的编辑指令，让它去候选视频里找出“参考视频经过这次变化之后”的目标视频。

这个任务的关键不在于“目标视频和参考视频要不同”，而在于：

- 它们必须很像，共享大部分上下文
- 但又不能一样
- edit_text 必须是必要条件
- target 必须满足 edit
- reference 必须不满足 edit

如果这几点做不好，模型学到的就不是 composed retrieval，而只是杂乱的相似性检索。

---

## 3. 当前最新状态

截至 2026-04-24，当前稳定版本已经达到一个比较像样的 pilot 质量。

最新代码版本：

```text
646b5d89ca830db2ccddbb53b616c4d6f79352cc
```

本地测试：

```text
Ran 88 tests
OK
```

最新一轮服务器 smoke test 的关键结果：

- Sample count: 10
- Difference types:
  - action: 1
  - audio_event: 2
  - object_count: 1
  - object_presence: 3
  - speech: 2
  - visible_text: 1
- 7 个 automated acceptance checks 全部 PASS

也就是说，当前版本已经不只是“能跑通”，而是已经满足了我们事先定义的结构性质量目标：

- 样本数量达标
- audio 样本达标
- 非语言 audio_event 达标
- speech 样本全部有证据
- speech 样本全部 transcript-backed
- object change 样本达标
- action 样本达标

这说明 pipeline 已经从“实验性脚本”进化成了一套有明确门控、可调试、可解释的数据构造系统。

---

## 4. 我们到底做了什么

整个项目是从“直接复用现成 retrieval 数据集”一步步转成“自己构造 Omni composed retrieval 数据”的。

已经完成的工作包括：

- 做过早期 AVIGATE / MSRVTT agent 检索实验
- 尝试过 V2T / T2V Omni rerank
- 判断出 MSRVTT 不适合直接构造成 agentic composed retrieval 数据
- 下载并归一化 Daily-Omni / WorldSense
- 下载并接入 Qwen3-Omni 系列模型
- 实现 source prepare
- 实现 whole video 到 event clip 的切分
- 实现 Omni-Detective 风格 annotation
- 实现同源 group pair proposal
- 实现 pair judge / verification
- 实现 verification override
- 实现 temporal source context
- 实现 visual near-duplicate gate
- 实现 action evidence gate
- 实现 speech / audio_event 分离门控
- 实现文件存在性检查、intra-clip conflict 检查、accepted 去重与类型平衡

一句更工程化的话来概括，就是：

> 我们不是在“调一个 prompt 生成 pair”，而是在搭一套能持续发现坏样本、定位坏样本、修复坏样本、并用测试和 smoke 验证修复是否真的生效的多模态数据构造流水线。

---

## 5. 原始数据集、中间数据集、目标数据集分别长什么样

### 5.1 原始数据集长什么样

我们主要处理了两类数据源：

#### Daily-Omni

- 数据形态：parquet 行
- 每行带视频、音频和问答信息
- 更像多模态问答/感知数据，而不是 retrieval 数据

简化示意：

```json
{
  "video_id": "XUWxQYmiBQY",
  "video": "<embedded bytes>",
  "audio": "<embedded bytes>",
  "question": "...",
  "candidates": ["A", "B", "C", "D"],
  "answer": "A"
}
```

#### WorldSense

- 数据形态：视频文件 + caption/subtitle/question/candidates
- 有更明显的文件化结构
- 但同样不是天然的 composed retrieval 三元组

简化示意：

```json
{
  "video": "videos/xxx.mp4",
  "subtitle_path": "subtitles/xxx.srt",
  "video_caption": "...",
  "question": "...",
  "candidates": ["A", "B", "C", "D"],
  "answer": "C"
}
```

归一化后的统计：

```text
source rows: 4368
unique clips: 2858
Daily-Omni: 1196 rows / 1196 clips
WorldSense: 3172 rows / 1662 clips
```

### 5.2 现在的中间数据集长什么样

原始数据不会直接拿来训练 composed retrieval。我们会先把它们变成一层层更结构化的中间产物：

```text
source_rows.jsonl
source_clips_all.jsonl
-> clip_plan_detective.jsonl
-> clip_groups.jsonl
-> extracted_event_clips.jsonl
-> detective_annotations.jsonl
-> judged_pair_proposals.jsonl
-> accepted_pairs.jsonl
-> gallery.jsonl / pilot_review.md
```

#### source_rows.jsonl

保留原始数据行级信息。

#### source_clips_all.jsonl

把所有原始视频标准化成 source clip。

#### clip_plan_detective.jsonl

把 whole video 规划成可以切出来的事件片段。

#### detective_annotations.jsonl

这是很关键的一层，里面是每个短 clip 的细粒度证据：

```json
{
  "clip_id": "daily_omni_xxx__seg_003",
  "summary": "A robot action figure rotates on a platform.",
  "object_counts": {"robot action figure": 1},
  "actions": ["rotating display"],
  "visible_text": [],
  "speech": [],
  "speakers_and_transcript": [],
  "audio_events": ["electronic hum"],
  "events": [
    {
      "visual": "robot rotates",
      "audio": "ambient electronic hum"
    }
  ],
  "modalities": ["visual", "audio"]
}
```

#### judged_pair_proposals.jsonl

这是候选 pair 层，保存 judge、verification、reject reason、quality 分数。

#### accepted_pairs.jsonl

这是最终 pilot dataset，结构已经非常接近我们想要的训练/评测数据。

### 5.3 我们真正需要的数据集长什么样

最终目标样本大概应该长这样：

```json
{
  "sample_id": "covr_omni_000001",
  "reference_video": "clips/...",
  "target_video": "clips/...",
  "edit_text": "change one cat into two cats",
  "modalities": ["visual", "audio"],
  "reference_caption": "...",
  "target_caption": "...",
  "difference": {
    "type": "object_count",
    "from": "one cat",
    "to": "two cats"
  },
  "hard_negatives": ["...", "..."],
  "quality": {
    "same_context_score": 0.85,
    "edit_match_score": 0.9,
    "target_uniqueness_score": 0.8,
    "difference_strength_score": 0.78
  },
  "verification": {
    "passed": true
  },
  "evidence": {
    "reference_storyline": ["..."],
    "target_storyline": ["..."],
    "difference_evidence": {
      "supporting_evidence": ["..."]
    }
  }
}
```

这个目标 schema 的核心约束是：

- reference / target 必须像，但不能一样
- target 必须满足 edit
- reference 必须不满足 edit
- edit_text 必须是必要条件
- hard negatives 必须接近，但不能也满足同一个 edit

---

## 6. 项目最难的地方，到底难在哪

这个项目真正的难点，不在“调用一个大模型写 caption”，而在于 **构造可信样本**。

一个高质量样本必须同时满足：

1. context 相似
2. difference 明确
3. difference 必须依赖 edit_text
4. target 满足 edit
5. reference 不满足 edit
6. 证据链可以回看
7. negative 足够难但不作弊

只要其中一条做得不严，就会出现这些坏情况：

- reference 和 target 实际一样
- 两个视频根本不是一个上下文
- edit 写得很漂亮，但其实不需要 edit 也能区分
- 把讲话内容变化误当成非语言音频变化
- 把单个视频内部自己的变化误当成跨视频 difference
- 文件路径都坏了，样本还被 accepted

所以这个项目最像的不是“写 prompt”，而是“做数据系统质量工程”。

---

## 7. 核心难点与解决过程

### 难点 1：MSRVTT 不适合 composed retrieval

MSRVTT 更适合普通 video-text retrieval，它没有天然的：

```text
reference video + edit text + target video
```

三元结构。

如果硬凑，很容易变成：

- 上下文不一致
- edit 太宽泛
- 样本不需要真正的组合理解

**解决方法：**

- 不再强行复用 MSRVTT 做主数据源
- 转向 Daily-Omni / WorldSense
- 自己搭从原始视频到 composed pair 的构造流水线

**面试说法：**

> 我不是先假设现成数据集一定能用，而是先看任务目标和数据结构是否匹配。发现 MSRVTT 缺少 composed retrieval 需要的局部差异结构后，我选择改做数据构造，而不是继续在不匹配的数据上打补丁。

### 难点 2：全局随机配对导致 reference / target 上下文不一致

早期如果在全局视频池里找 pair，会出现：

- 主体不同
- 场景不同
- 拍摄风格不同
- edit 变成“换整个世界”

**解决方法：**

- whole video 先切成短 event clips
- 只在同源视频或同组里配对
- 引入 temporal source context，让相邻片段得到更高上下文分

**本质收益：**

reference 和 target 不再只是“语义上有点像”，而是真正在拍摄上下文上连续。

### 难点 3：reference 和 target 几乎一样，却还能被接受

这是非常真实、也非常危险的问题。模型可能只是换了种说法写 caption，就让 verification 误以为有差异。

**解决方法：**

1. 三重 verification：
   - `caption_delta`
   - `edit_projection`
   - `edit_necessity`
2. 增加视频级 `visual_near_duplicate_score`
3. 后续又加了 `intra_clip_conflict` 检查，防止把单个视频内部自身变化当成跨视频差异

**一句人话：**

> 只看 caption 不够，必须看视频本身是不是几乎一模一样。

### 难点 4：judge 分数不稳，会误杀好样本

有些样本 judge 的 `edit_match_score` 很低，但 verification 明确表明：

- reference 不满足 edit
- target 满足 edit
- reference + edit 可以投影到 target

这种样本如果直接丢掉，会误杀。

**解决方法：**

引入 `verification override`。

前提不是“verification 过了就全放”，而是：

- verification 三项都过
- same_context 达标
- target_uniqueness 达标
- difference_strength 达标
- 各 difference-specific gate 达标

这样才允许 override judge 的低 edit_match。

### 难点 5：hard negatives 太远或太近都不行

如果 negative 太远，任务太简单；如果太近，又可能和 target 一样满足 edit。

**解决方法：**

- 不只按“像不像”选 negative
- 而是做 **edit-aware uniqueness**
- 如果候选也满足同一个 edit，就降低 target uniqueness

**关键思想：**

hard negative 的核心不是“视觉接近”，而是“接近但不满足同一个 edit”。

### 难点 6：action 样本少，而且很容易误标

模型很容易把：

- 物体出现
- 属性变化
- 讲话内容变化

误写成 action。

**解决方法：**

- 给 action 单独 bucket
- 提取 `actions` / `events` / `storyline` 里的动作证据
- 引入 `action_evidence_score`
- 只有真正有动作证据支撑的 pair 才能进 action

后来又进一步做了 accepted 结果的类型平衡，避免 action 永远被别的类型挤掉。

### 难点 7：speech 样本非常有争议

这个问题是用户人工抽查明确指出来的，也是一个真正重要的质量问题。

问题在于：

- “同一个人在同一个场景里讲话”
- 只是讲话主题变了
- 如果没有 transcript 或足够具体的 speech evidence
- 这类样本非常容易引发争议

**解决方法：**

把 `speech` 和 `audio_event` 强行拆开。

#### speech

只表示语言内容、说话主题、说话人、语气等变化，要求：

- modalities 必须包含 audio
- 必须依赖听音频
- 必须有 `speech` / `speakers_and_transcript` 证据
- `speech_evidence_score >= 0.75`
- `speech_specificity_score >= 0.70`
- 最终 accepted 的 speech 必须 transcript-backed

#### audio_event

只表示非语言声音变化，例如：

- hum
- whoosh
- wind
- machinery
- applause
- animal sounds
- scratching sound

不能把下面这些算作高质量 `audio_event`：

- only speech
- narration
- talking
- no background music
- no ambient noise

**这是后来专门修掉的一类伪 audio_event。**

### 难点 8：audio_event 不是“出不来”，而是容易长歪

后来我们遇到过两个阶段性问题：

1. 一开始 `audio_event` 根本出不来
2. 后来 `audio_event` 一下子太多，又把 object/action 类型挤掉了

**解决方法分两步：**

第一步，扩证据来源：

- 从 `audio_events`
- `events[].audio`
- `detective_notes`
- `summary`

一起挖非语言声音证据

第二步，做 accepted 去重和平衡：

- 去掉同一组、同一差异方向的重复样本
- 避免大量 hum add/remove 刷满 pilot
- 把接受策略调回类型平衡

### 难点 9：文件不存在、路径错误，也会污染数据

这类问题不是模型问题，是工程问题，但非常致命。

我们后来人工审计时就发现过：

- target 视频文件根本不存在
- 但样本已经进入 accepted

**解决方法：**

- `reference_video` / `target_video` / `hard_negatives` 全部做存在性检查
- 缺文件直接 reject

### 难点 10：单个 clip 内部自己的变化会伪装成跨视频 difference

例如一个 clip 自己从 hum 变 scratching sound，如果处理不严，就会被误当成：

```text
reference: hum
target: scratching sound
```

但其实这是一个视频内部的变化，不是两个视频之间的差异。

**解决方法：**

- 增加 `intra_clip_conflict` 检查
- 如果 `from` 和 `to` 同时能在同一个 clip 的 annotation / caption 里找到，就拒绝

### 难点 11：GPU 资源很紧，不能靠“多模型一起堆”

用户明确给了约束：

- 当前优先只用 2 张空闲 GPU
- 就算未来资源宽裕，也尽量不超过 6 张 GPU
- 不允许 Qwen3-Omni Instruct / Captioner / Thinking 三个服务同时常驻

**解决方法：**

做阶段式单模型加载：

```text
Captioner -> 关
Instruct -> 关
Thinking -> 只复核 accepted / borderline
```

这不是最省时间的方案，但对显存和服务稳定性最友好。

---

## 8. 这套方法的技术亮点

### 亮点 1：不是在“生成数据”，而是在“治理数据”

项目价值不只是调模型，而是建立了一套：

```text
发现坏样本 -> 抽象问题类型 -> 加门控 -> 加测试 -> 小实验复核
```

的可迭代数据质量工程流程。

### 亮点 2：Omni-Detective 风格细粒度证据链

不是只生成一句 caption，而是拆成：

- summary
- object counts
- actions
- visible text
- speech / transcript
- audio events
- timeline events
- detective notes

后续 pair proposal、verification、人工审计都可以回看证据来源。

### 亮点 3：difference-specific gate

不同差异类型，不是同一个阈值通吃，而是分类型做门控：

- visual difference -> `visual_near_duplicate_score`
- action -> `action_evidence_score`
- speech -> `speech_evidence_score` + `speech_specificity_score`
- audio_event -> `non_speech_audio_event_score`

### 亮点 4：verification override 不是拍脑袋放行

override 的前提是多层结构化证据同时一致，而不是“模型说可以就可以”。

### 亮点 5：把人工审计反馈真正编进系统

这个项目的一个真实优点是：用户随手抽查出来的坏样本，不是停留在口头批评，而是会变成：

- 一个明确失效模式
- 一条新门控
- 一条新测试

这就是工程上很硬的一点。

### 亮点 6：accepted 结果不是简单按分数排序

后来 accepted 结果做了：

- 去重复
- 去近重复
- 类型平衡
- 质量门控

所以最终 pilot 更像一个可用的小型 benchmark，而不只是“分数高的前 10 条”。

---

## 9. 当前阶段可以怎么评价

当前这套 pipeline 已经完成了三个层次的目标：

### 第一层：能跑通

已经不是半成品脚本，而是完整链路：

```text
raw -> normalized -> clips -> annotations -> proposals -> judged -> accepted -> pilot review
```

### 第二层：能发现并修掉明显坏样本

已经实际修过的问题包括：

- reference / target 实际一样
- caption 幻觉
- missing file
- intra-clip fake difference
- speech / audio_event 混淆
- 伪 non-speech 文本
- accepted 重复刷屏

### 第三层：结构性指标达标

最新 smoke test 已经实现：

- 7 个 automated checks 全 PASS
- 样本类型覆盖比较均衡
- audio_event 终于既存在，又不再长歪
- action / object / speech / visible_text 也没有被挤死

换句话说，当前版本已经具备“进入人工精选和进一步整理 benchmark”的基础。

---

## 10. 面试时怎么讲这个项目

### 10.1 简历版一句话

可以写成：

> 设计并实现了一套面向 Omni composed video retrieval 的多模态数据构造与质量控制 pipeline，将 Daily-Omni / WorldSense 等原始多模态数据转化为 reference video + edit text + target video 样本；系统包含事件切片、细粒度多模态证据标注、同源 pair proposal、三重 verification、near-duplicate 检测、speech/audio_event 分离门控、hard negative 选择与自动化质量审计，最终在小规模 pilot 上实现 7 项核心验收指标全部通过。

### 10.2 面试官问：你解决了什么问题

回答可以这样说：

> 我解决的不是一个模型精度问题，而是 Omni composed retrieval 数据缺失的问题。现成数据集大多只有 text-to-video 结构，没有 reference video + edit text + target video 这样的三元结构。我搭了一套数据构造和质量门控流水线，把原始多模态视频数据转成 composed retrieval 样本，并且尽量保证这些样本是真的需要 edit 才能区分。

### 10.3 面试官问：为什么不能直接用 MSRVTT

> 因为 MSRVTT 更像普通检索数据，不天然包含“同一上下文里的局部变化”这件事。composed retrieval 要求 reference 和 target 很像，但只差一个必要变化。MSRVTT 没有这种结构，硬构造会得到很多上下文不一致或 edit 过泛的坏样本。

### 10.4 面试官问：这个项目最难的点是什么

> 最难的是构造可信样本，而不是调模型。你要同时保证 reference 和 target 共享上下文、差异明确、edit 是必要条件、negative 够难但不作弊、并且最后还能解释为什么这个样本应该被接受。这个难点更像数据系统质量工程，而不是普通 prompt 工程。

### 10.5 面试官问：你怎么判断一个 pair 有效

> 我用了多层判断。第一层是 same_context_score，保证上下文相似。第二层是 difference_strength_score，保证差异不是特别弱。第三层是三重 verification：caption_delta 判断两者是否真的不同，edit_projection 判断 reference 加 edit 能不能推到 target，edit_necessity 判断 edit 是否真的是必要条件。对视觉差异，我还额外加了视频帧级 near-duplicate 检测，防止 caption 幻觉。

### 10.6 面试官问：为什么要做 verification override

> 因为单个 judge score 不稳定，会误杀一些本来是好样本的 pair。我的做法不是直接放开阈值，而是在 same_context、target_uniqueness、difference_strength 和类型门控都达标的前提下，只让多项 verification 都一致通过的样本 override 低 edit_match_score。

### 10.7 面试官问：speech 为什么难

> 因为 speech 的争议在于“讲话主题变了”不一定就是高质量 audio pair。如果没有 transcript 或足够具体的 speech evidence，人很难确认 edit 是否可靠。所以我把 speech 和 audio_event 拆开，speech 必须 transcript-backed，audio_event 只允许非语言声音变化，这样就把一个模糊问题拆成了两个可门控的问题。

### 10.8 面试官问：这个项目最有工程味的一点是什么

> 不是我用了多少模型，而是我把人工抽查发现的问题都系统化了。每发现一种坏样本，我都会把它抽象成一个失效模式，再落实成一条 gate、一条测试和一次小规模 smoke 验证。这样系统会越来越稳，而不是每次只靠换 prompt 碰运气。

### 10.9 面试官问：如果继续做，你下一步怎么优化

> 下一步我会把 Captioner 更系统地接成独立 audio observer，进一步提高 speech 和 non-speech audio evidence 的可靠性；再把 Thinking 只用于 accepted 和 borderline pair 的低频复核，控制成本。同时我会继续做人工精选和误差分析，把自动达标的小型 pilot 逐步推成更稳定的 benchmark 子集。

---

## 11. 八股文版项目亮点

如果要写得更像求职时常见的“亮点总结”，可以直接用下面这版：

### 项目背景

现有视频检索数据集大多面向 text-to-video retrieval，缺少参考视频驱动的 composed retrieval 样本，难以支撑“reference video + edit text -> target video”的全模态检索研究。

### 我的工作

1. 设计了从原始多模态视频数据到 composed retrieval 样本的全链路数据构造流程。
2. 引入 Omni-Detective 风格细粒度标注，抽取 visual / audio / text / transcript / event-level evidence。
3. 基于同源片段和 temporal context 构造 pair，替代全局随机配对。
4. 设计三重 verification、near-duplicate 检测、difference-specific gates、verification override 等质量控制机制。
5. 通过单元测试和小规模 smoke test 持续迭代，修复 speech/audio_event 混淆、伪音频事件、文件缺失、intra-clip 冲突、accepted 重复等问题。

### 技术价值

1. 提升了多模态 composed retrieval 数据的可用性和可信度。
2. 把“人工抽查经验”沉淀为可复用的质量门控规则。
3. 形成了一套可解释、可调试、可持续扩展的数据治理框架。

### 当前结果

当前 pilot 版本已在 10 条样本规模上实现 7 项核心自动验收指标全部通过，并覆盖 object、action、speech、audio_event、visible_text 等多种差异类型。

---

## 12. 当前结论

这件事走到现在，最重要的结论不是“终于跑出几个样本了”，而是：

> 我们已经把这个问题从“靠 prompt 撞运气生成 pair”，推进成了“有证据、有门控、有测试、有审计”的数据系统工程。

当前版本还不是最终 benchmark，但已经具备三个很重要的特征：

1. 能稳定产出结构合理的 pilot 样本
2. 能对已知坏样本类型做系统性防御
3. 能把人工发现的问题快速沉淀成下一轮规则和测试

所以这份工作的价值，不只是构造了几条数据，而是搭出了一条 **可持续迭代的 Omni 组合检索数据构造方法论**。
