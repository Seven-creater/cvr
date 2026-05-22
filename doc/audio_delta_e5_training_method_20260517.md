# AudioDelta-E5 训练方法记录

日期：2026-05-17

## 1. 目标

AudioDelta-E5 的目标是解决当前 Omni-CVR / composed video retrieval 中音频模态利用不足的问题。现有方法往往以 video-text retrieval 为主，audio 要么没有进入主检索模型，要么被额外模型转写成文本后再拼进 edit text。这样会导致模型没有真正学习：

```text
reference video/audio + edit_text -> target video/audio
```

特别是 B 线 Audio-CVR 中，目标不是做 ASR retrieval，而是让模型学习：

```text
target_audio 相比 reference_audio 是否按照 edit_text 发生了正确变化，
同时这个变化发生在保留的视频语境中。
```

因此训练框架从普通 E5 embedding adapter 升级为 AudioDelta-E5 training framework。

---

## 2. 训练数据输入

训练入口读取 B-line 构造出的结构化样本，而不是旧的 B-line 产物。默认读取以下新格式文件：

```text
b_splits/train.jsonl
b_train_bidirectional_triplets.jsonl
b_main_audio_cvr_triplets.jsonl
b_extended_audio_cvr_triplets.jsonl
b_all_audio_cvr_triplets.jsonl
```

旧的 `audio_ab_fresh800_omni_first_*` 不再自动参与训练，因为旧 B-line 没有经过最新 blind review v2、anti-ASR 分层、hard negative、AudioDelta 字段补全，容易污染训练。

每条训练样本需要尽量包含：

```json
{
  "reference_video": "...",
  "target_video": "...",
  "edit_text": "change the speech from discussing X to discussing Y",
  "edit_type": "replace",
  "audio_delta_type": "speech_topic | music | sound_event | ambient",
  "old_audio": "discussing X",
  "new_audio": "discussing Y",
  "direction": "forward | inverse",
  "split_tier": "main | extended | diagnostic",
  "raw_source_id": "...",
  "pair_group_id": "...",
  "inverse_pair_group_id": "...",
  "shortcut_label": "clean_audio_delta | asr_like | visual_shortcut | ...",
  "audio_delta_strength": 0.0,
  "video_context_strength": 0.0,
  "asr_degeneracy_risk": 0.0,
  "visual_shortcut_risk": 0.0,
  "audio_delta_hard_negatives": [
    {"type": "reference_negative", "video": "..."},
    {"type": "visual_hard", "video": "..."},
    {"type": "audio_hard", "video": "..."},
    {"type": "asr_hard", "video": "..."}
  ]
}
```

这些字段直接服务训练目标，而不是只为了保存 triplet。

---

## 3. 基础训练流程

当前训练链路为：

```text
prepare
  -> cache-embeddings
  -> train-adapter
  -> eval
```

### 3.1 prepare

`prepare` 负责从 B-line run 中读取训练记录，去重，并输出：

```text
records/train.jsonl
records/eval.jsonl
records/summary.json
```

它只接受新 B-line tier/split 输出。旧 B-line subtype 文件不再作为默认训练输入。

### 3.2 cache-embeddings

`cache-embeddings` 使用 e5-omni-7B 编码：

```text
query          = reference_video + edit_text
target         = target_video
reference      = reference_video
edit           = edit_text
old_audio      = old_audio / reference audio content
new_audio      = new_audio / target audio content
negative       = hard negative videos
```

一个关键修复是：视频不能作为裸字符串传给 `encode_document`。裸 mp4 字符串会被 sentence-transformers / Qwen2.5-Omni processor 误送进 audio feature extractor，导致错误。因此现在所有视频输入统一包装为：

```python
{"video": video_path}
```

query 则包装为：

```python
{"video": reference_video, "text": "Edit the reference video so that: ..."}
```

### 3.3 train-adapter

`train-adapter` 在冻结 E5-Omni embedding 的前提下训练轻量 projection adapter：

```text
query_proj
 doc_proj
 edit_proj
```

它相当于 RET-token / retrieval pooling 的工程版第一阶段：先不改 tokenizer、不动 E5 主干，只学习检索空间中的 query/doc/edit 投影。

### 3.4 eval

`eval` 输出：

```text
summary.json
comparison.md
```

报告内容包括：

```text
base_e5_global
audio_delta_adapter_global
base_e5_local
base_e5_global_local
audio_delta_adapter_local
audio_delta_adapter_global_local
```

并按以下字段分组：

```text
split_tier
audio_delta_type
shortcut_label
```

---

## 4. 八个核心设计如何落地

### 4.1 Audio-delta loss

目的：让模型学习 target audio 相比 reference audio 是否按照 edit_text 发生了变化。

基础形式：

```text
delta(q,d) = sim(a_t, e) - sim(a_r, e)
L_delta = max(0, m - delta)
```

对 add / increase：

```text
target 更接近 edit sound
reference 更远离 edit sound
```

对 remove / decrease：

```text
reference 更接近 edit sound
target 更远离 edit sound
```

对 replace：

```text
reference -> old_audio
target    -> new_audio
```

---

### 4.2 Hard negative curriculum

训练样本中支持四类负样本：

```text
reference_negative: reference 本身
visual_hard:        画面像，但声音不满足 edit
audio_hard:         声音像，但视频上下文不对
asr_hard:           speech 关键词像，但不是正确 target
```

代码中设置 curriculum stage：

```text
stage 1: reference_negative
stage 2: + visual_hard
stage 3: + audio_hard
stage 4: + asr_hard
```

这样训练可以从容易负样本逐步过渡到更难的 shortcut negative。

---

### 4.3 Reference-as-negative

reference 是“尚未发生 edit 的视频”，因此不能是答案。

训练目标：

```text
s(query, target) > s(query, reference)
```

对应 loss：

```text
L_ref = max(0, m_ref - s(q, target) + s(q, reference))
```

这个 loss 对 CVR 特别重要，因为 reference 和 target 通常非常接近，模型容易把 reference 当成候选答案。

---

### 4.4 Edit-type-aware delta

不同 edit 类型不能使用同一个训练逻辑。

当前支持：

```text
add
remove
increase
decrease
replace
```

replace 是最重要的情况，例如：

```text
change the speech from discussing X to discussing Y
```

训练约束为：

```text
reference 接近 old_audio
target 接近 new_audio
target 不应接近 old_audio
reference 不应接近 new_audio
```

这直接对应 B-line 中 speech topic / music / sound event 的方向性变化。

---

### 4.5 Local temporal segment matching

这是本轮重点补充的部分。

问题：全局视频向量可能丢失短暂事件。例如 edit_text 描述的是：

```text
某一句话
一小段掌声
短暂蜂鸣声
音乐中某个瞬间变化
```

如果把整段视频压成一个全局向量，短时事件可能被平均掉。

当前实现支持 `--local-segments N`，例如：

```bash
--local-segments 2
```

缓存时会得到：

```text
target_segments:    [num_samples, num_segments, dim]
reference_segments: [num_samples, num_segments, dim]
negative_segments:  [num_samples, num_negatives, num_segments, dim]
```

当前支持两种 local segment 模式：

```text
prompt: 对同一个视频加入 temporal focus instruction，适合 smoke 和快速调试。
ffmpeg: 在 cache 阶段把视频切成真实子片段，再分别编码，适合正式 1k+ 训练。
```

`ffmpeg` 模式只写入当前训练 run 的 cache，例如：

```text
embedding_cache/local_media_cache/
```

不修改原视频，也不覆盖 B-line 数据集产物。

检索分数：

```text
s_global(q, d) = q · d_global
s_local(q, d)  = max_t q · d_segment_t
s_mix(q, d)    = (1 - alpha) * s_global + alpha * s_local
```

正式训练建议使用 `--local-segment-mode ffmpeg --local-segments 2/3`。如果只是验证代码链路，仍建议使用 `prompt`，避免缓存阶段过慢。

---

### 4.6 RET-token / latent pooling 的工程版

真正修改 E5 tokenizer 加 `[RET]` token 成本较高，也会影响模型兼容性。

当前先实现轻量工程版：

```text
query_proj
 doc_proj
 edit_proj
```

也就是在 E5 生成 embedding 之后学习检索空间投影。

这一步的作用：

```text
先验证 AudioDelta loss 和 hard negatives 是否有收益，
再决定是否进入 LoRA / RET-token 级别微调。
```

当前 `train-lora` 仍是 dry-run plan，不阻塞 adapter 框架。

---

### 4.7 Source-disjoint split

新增 `build-splits` 命令，按 source / pair group 分组切分，避免泄漏。

硬规则：

```text
同一个 raw_source_id 不能跨 train / val / test
同一个 pair_group_id 的正向和反向不能跨 split
同一个 inverse_pair_group_id 不能跨 split
test_main 每个 pair group 只保留一个方向
inverse 样本进入 train 或 test_inverse_diagnostic
```

输出：

```text
b_splits/train.jsonl
b_splits/val.jsonl
b_splits/test_main.jsonl
b_splits/test_inverse_diagnostic.jsonl
b_splits/diagnostic.jsonl
b_splits/split_summary.json
```

---

### 4.8 Shortcut diagnosis

评估阶段不只报告 overall R@K，而是分组报告：

```text
by_split_tier
by_audio_delta_type
by_shortcut_label
```

这样可以区分：

```text
B-main
B-extended
B-diagnostic
speech_topic
music
sound_event
clean_audio_delta
ASR-like
visual-shortcut
```

这一步很重要，因为 B-line 很容易退化成 ASR retrieval。分组报告能证明主结果不是由 ASR-like 样本虚高造成。

---

## 5. V2 Research Profile

V1 已经跑通端到端训练；V2 不是替换 V1，而是新增显式 profile：

```bash
--training-profile v2_research
```

默认仍是：

```bash
--training-profile v1
```

这样做是为了控制复杂度：所有新 loss 都必须能单独关闭，避免“模块很多但贡献说不清楚”。

### 5.1 Hardness-weighted negatives

固定 margin hard negative loss 对所有 negative 一视同仁。V2 新增按难度加权：

```text
w_j = clip(softmax(s(q,d_j-) / tau_hard) * num_neg, w_min, w_max)
```

默认只对：

```text
visual_hard
audio_hard
asr_hard
```

加权；`reference_negative` 仍走固定 margin。这样避免 reference 这个必需负样本被过度放大。

新增日志项：

```text
loss_hw_hn
effective_negative_count
```

### 5.2 Multi-positive / Multi-view contrastive

训练阶段用：

```text
positive_group_id = inverse_pair_group_id or pair_group_id
```

同组正向 / 反向样本可互为 multi-positive：

```text
L_multi_pos = -log sum_{p in P+} exp(sim(q,p)/tau)
              / sum_{x in batch} exp(sim(q,x)/tau)
```

硬规则：

```text
train 可以启用 multi-positive
val/test 不启用 multi-positive
test_main 每个 pair_group 只保留一个方向
```

这避免反向样本把评测指标虚高。

### 5.3 CORAL modality alignment

V2 先实现轻量分布对齐，不直接引入真实 audio-only processor：

```text
doc side  = target/reference projection
edit side = edit/old_audio/new_audio projection
L_coral = ||Cov(doc) - Cov(edit)||_F^2 / (4*d*d)
```

它的目的不是替代 audio-delta loss，而是缓解 doc/edit 投影空间的分布偏移。

风险：alignment 过强可能伤害 clean_audio_delta。因此默认权重很小：

```text
lambda_coral_align = 0.05
```

后续必须在 `B-main clean_audio_delta` 上确认没有副作用。

### 5.4 Memory bank / larger negatives

V2 支持一个 detached target embedding memory bank：

```text
--memory-bank-size 4096
```

训练前 `warmup_ratio` 内不启用 memory bank，避免早期 embedding 还不稳定时污染训练。

它只用于 train，不参与 eval。显存压力可通过降低：

```text
memory_bank_size
batch_size
local_segments
```

控制。

### 5.5 False-negative filtering

V2 不删除原始 negative metadata，而是在训练时赋予有效权重：

```text
同 pair_group / inverse_pair_group: weight = 0
高相似疑似 false negative: weight = false_negative_soft_weight
普通 negative: weight = 1
```

默认：

```text
false_negative_sim_threshold = 0.92
false_negative_soft_weight = 0.15
```

这部分必须谨慎调参。阈值过低会把真正有用的 hard negative 软化，阈值过高又过滤不掉假负样本。

### 5.6 Schedule

V2 新增：

```text
warmup + cosine learning rate
temperature annealing
```

每步日志写入：

```text
lr
temperature
memory_bank_size
全部 loss 项
```

这让后续调参时能判断到底是 loss 本身失效，还是优化 schedule 不稳定。

---

## 6. V2.1：严格对齐 e5-omni recipe 的 Stage-1

V2.1 当前阶段定义为 **Stage-1**。这一阶段的目标不是追求 AudioDelta-CVR 最优效果，而是先建立一个和 e5-omni / MMEB-V3 训练 recipe 对齐的干净基线。默认只保留三类训练思想：

```text
modality-aware temperature
negative-aware contrastive learning
batch-wise covariance regularization
```

对应论文依据包括：

```text
MMEB-V3 / e5-omni: modality-aware temperature、negative-aware contrastive loss、batch-wise covariance regularization
Deep CORAL: covariance-level distribution alignment
```

本次实现原则：

```text
不改 prepare/cache/eval 主流程
不改 E5 视频输入包装
不启用 LoRA
不重写训练脚本
Stage-1 默认只开 e5-omni 三类核心思想
AudioDelta 专属 loss 和其他探索项只保留代码和开关，不混入默认训练
```

### 6.1 Modality-aware temperature calibration

原先训练只使用全局 temperature：

```text
logits = scores / temperature
```

但 query / target / edit / audio delta 的模态组合不同，similarity logits 的 sharpness 不一定一致。因此 V2.1 新增可学习 temperature 模块：

```text
ModalityAwareTemperature
  log_tau_text
  log_tau_audio
  log_tau_video
```

三类 temperature 通过 `exp(log_tau)` 得到，并 clamp 到：

```text
tau_min = 0.005
tau_max = 0.2
```

模态组合定义为：

```text
query              = text + audio + video
target/reference   = audio + video
edit               = text
audio delta        = audio vs text
local segment      = audio + video
```

任意输入的 temperature：

```text
tau(x) = average(tau_m for m in modalities(x))
```

任意 pair 的 temperature：

```text
tau_pair(x, y) = 0.5 * (tau(x) + tau(y))
```

启用参数：

```bash
--enable-modality-temperature
--modality-temperature-init 0.05
--modality-temperature-min 0.005
--modality-temperature-max 0.2
```

关闭时仍使用旧的全局 temperature，保证 V1/V2 旧实验可复现。

训练日志新增：

```text
tau_text
tau_audio
tau_video
tau_query
tau_target
tau_audio_text
effective_temperature_cvr
effective_temperature_delta
```

### 6.2 Masked DCL + negative curriculum + debiasing

原先 V2 已有 negative type curriculum：

```text
reference_negative -> visual_hard -> audio_hard -> asr_hard
```

V2.1 不删除这个阶段逻辑，而是在已启用的 negative type 内部继续做可控加权：训练前期保留所有 negatives，后期逐步聚焦更容易混淆的 negatives，避免 easy negatives 长期占据 denominator。主 contrastive objective 从普通 CE 切到 masked DCL：

```text
contrastive_objective = masked_dcl
```

默认 schedule：

```text
negative_keep_ratio: 1.0 -> 0.5
negative_curriculum_warmup_ratio: 0.1
easy_negative_weight: 0.1
```

进入 masked DCL denominator 的 negative weight：

```text
negative_weight =
  type_weight
  * quantile_mask_weight
  * false_negative_weight
```

其中 false-negative debiasing 优先级最高：

```text
同 pair_group / inverse_pair_group: weight = 0
疑似 false negative: weight = false_negative_soft_weight
普通 negative: weight = 1
```

注意：`hardness_weighting` 已保留为后续实验项，但 Stage-1 默认关闭。因此当前默认路径不包含 `hardness_weight`，只使用 negative type curriculum、quantile mask 和 false-negative debiasing。

启用参数：

```bash
--contrastive-objective masked_dcl
--dcl-debias-prob 0.1
--dcl-negative-floor 1e-6
--enable-quantile-negative-curriculum
--negative-keep-ratio-start 1.0
--negative-keep-ratio-end 0.5
--negative-curriculum-warmup-ratio 0.1
--easy-negative-weight 0.1
```

训练日志新增：

```text
effective_negative_count
kept_negative_count
masked_easy_negative_count
suspected_false_negative_count
avg_negative_weight
avg_hard_negative_score
avg_easy_negative_score
loss_masked_dcl
```

### 6.3 Batch whitening + query-target CORAL

V2.1 Stage-1 的对齐对象改为 e5 recipe 更直接的 query-target 表示空间：

```text
loss_coral_query_target:
  Cov(query projection) 对齐 Cov(target projection)
```

CORAL 形式：

```text
L_coral = || Cov(query) - Cov(target) ||_F^2 / (4 * d * d)
```

同时 Stage-1 默认启用 batch whitening 正则。whitening 统计从 `concat(query, target)` 计算，只作用于训练 loss，不改变 cache embedding，也不改变 eval 输入格式：

```text
L_whiten = || mean(z) ||_2^2 + || Cov(z) - I ||_F^2 / d^2
z = concat(query, target)
```

默认辅助权重为：

```text
lambda_coral_align = 0.05
lambda_batch_whitening = 0.01
```

启用参数：

```bash
--enable-coral-align
--lambda-coral-align 0.05
--enable-batch-whitening
--lambda-batch-whitening 0.01
```

训练日志新增：

```text
loss_coral_align
loss_coral_query_target
loss_batch_whitening
cov_query_trace
cov_target_trace
cov_query_target_gap
cov_doc_trace
cov_edit_trace
cov_delta_trace
```

### 6.4 V2.1 profile 默认行为

`v1` 保持保守：

```text
enable_modality_temperature = false
enable_quantile_negative_curriculum = false
enable_batch_whitening = false
enable_coral_align = false
```

`e5_omni_recipe` 与当前 `v2_research` 默认行为一致。默认开启：

```text
contrastive_objective = masked_dcl
enable_modality_temperature = true
enable_quantile_negative_curriculum = true
enable_false_negative_filtering = true
enable_coral_align = true
enable_batch_whitening = true
lambda_coral_align = 0.05
lambda_batch_whitening = 0.01
```

先不默认开启的探索项：

```text
enable_hardness_weighting = false
enable_multi_positive = false
enable_memory_bank = false
lambda_hw_hn = 0.0
lambda_multi_positive = 0.0
lambda_memory_bank = 0.0
```

同时默认关闭 AudioDelta 专属 loss：

```text
lambda_delta = 0.0
lambda_hn = 0.0
lambda_ref = 0.0
lambda_edit_type = 0.0
lambda_visual = 0.0
disable_local_segments = true
disable_global_local_mix = true
```

这样默认 profile 只保留 e5-omni 明确对应的训练思想：

```text
modality-aware temperature
negative-aware contrastive learning
batch whitening + query-target covariance / CORAL
```

`hardness_weighting`、`multi_positive`、`memory_bank`、以及 AudioDelta 专属 loss 仍然保留代码和开关，但先作为后续可控实验项，不混入 Stage-1 默认训练。需要时可以显式打开：

```bash
--enable-hardness-weighting --lambda-hw-hn 0.5
--enable-multi-positive --lambda-multi-positive 0.5
--enable-memory-bank --lambda-memory-bank 0.25
```

---

## 7. Ablation 设计

新增 `run-ablations` 命令，自动运行：

```text
full_v2
without_modality_temperature
without_quantile_negative_curriculum
without_false_negative_debiasing
without_hardness_weighting
without_multi_positive
without_coral_align
without_batch_whitening
without_memory_bank
without_false_negative_filtering
without_local_segments
without_delta
without_reference_negative
without_hard_negatives
v1_loss_only
```

其中 Stage-1 最关键的有效 ablation 是：

```text
without_modality_temperature
without_quantile_negative_curriculum
without_false_negative_debiasing
without_coral_align
without_batch_whitening
v1_loss_only
```

下面这些 ablation 是为后续探索项预留的；在 Stage-1 默认配置下，它们可能接近 no-op，只有显式打开对应模块后才有解释价值：

```text
without_hardness_weighting
without_multi_positive
without_memory_bank
```

输出：

```text
ablations/summary.json
ablations/comparison.md
每个 ablation 子目录下的 adapter/loss_curve.jsonl
每个 ablation 子目录下的 eval/summary.json
```

对应论文实验中的 strong ablation：每个模块是否真的贡献效果，都可以单独关闭验证。

`comparison.md` 除了 R@1/R@5/R@10，还会记录：

```text
reference_negative_average_rank
delta_score_pos_mean
delta_score_neg_mean
effective_negative_count
tau_text / tau_audio / tau_video
```

如果只想复现 V1 ablation，可以使用：

```bash
python -m app.e5_audio_delta_train run-ablations --training-profile v1 ...
```

---

## 8. 复杂度与风险控制

V2 模块明显比 V1 复杂，因此必须遵守以下顺序：

```text
1. 先跑 v1，确认数据、缓存、adapter、eval 正常。
2. 再跑 v2_research，但只用 50 条样本。
3. 看 loss_curve.jsonl：loss_masked_dcl / loss_batch_whitening / loss_coral_query_target / modality temperature / effective negatives 是否稳定。
4. 再跑 ablation，而不是直接宣称 full_v2 有效。
5. 通过 50 -> 200 -> 1k 三档后，才考虑 LoRA。
```

特别注意：

```text
Modality temperature: 必须检查 tau 是否始终在 clamp 范围内。
Quantile curriculum: 需要观察 effective_negative_count 是否过低。
Hardness weighting: 默认关闭，后续如果打开，需要调 tau_hard 和 clip 上下界。
Multi-positive: 只允许 train，必须依赖 source/pair disjoint split。
CORAL: 默认开启，权重必须小，重点检查 clean_audio_delta 是否下降。
Batch whitening: 默认开启，必须检查 loss 是否稳定、是否出现 NaN。
Memory bank: 容易增加显存和过时 negative 风险，warmup 后再启用。
False-negative filtering: 阈值要通过 diagnostic split 调。
LoRA: 必须在 frozen adapter + projection head 稳定后再开。
```

换句话说，V2 是研究 profile，不是盲目全开的大规模默认训练配置。

---

## 8.1 Pilot 评估扩池说明

为了在小样本阶段更真实地观察 recipe 训练后的检索变化，当前训练脚本支持一个 **pilot-only** 的评估扩池模式：

```text
query 数量保持很小
eval gallery 额外加入随机 distractor videos
典型用法：30 个 query + 约 970 个随机干扰视频
```

这个模式的目的只有一个：避免在 `32 query / 32 target` 这类极小候选池上得到虚高指标，便于早期判断 recipe 是否真的有泛化趋势。

必须注意：

```text
这不是最终 benchmark protocol
这不是全量数据集完成后的正式 test 设计
这不是训练集构造逻辑
```

推荐把它理解成：

```text
small-data pilot evaluation mode
```

当前实现约束：

```text
1. 只在 prepare/eval 阶段扩 gallery，不改 train records。
2. distractor 默认从当前 run 的 annotation / segment manifest 中随机抽。
3. 自动避开 train/eval 已使用的 reference/target/hard negative。
4. 默认尽量避开当前 eval query 的 raw_source_id，减少同源泄漏。
5. summary.json 中会显式写出:
   eval_protocol = pilot_only_random_distractor_gallery
```

等全量 Audio-CVR 数据集完成后，应回到正式 split 与正式 gallery：

```text
train / val / test_main / diagnostic
真实大 gallery
不再依赖随机 distractor 扩池
```

---

## 9. 已经解决的工程难点

### 6.1 CUDA / PyTorch / Transformers 兼容

服务器 driver 是 CUDA 12.2 级别，不能使用 CUDA 13 wheel。最开始安装最新 torch 时，真实 E5 加载失败。

解决方式：固定 PyTorch 三件套到 CUDA 12.1 wheel：

```text
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
index-url=https://download.pytorch.org/whl/cu121
```

并在安装 sentence-transformers 等依赖后，再强制重装一次三件套，避免 pip 把 torchvision / torchaudio 换成不匹配版本。

---

### 6.2 视频路径被误当成音频数组

真实 E5 smoke 中曾出现：

```text
ValueError: could not convert string to float: '...mp4'
```

原因是裸 mp4 路径字符串被传给了 audio feature extractor。

解决方式：所有视频输入必须包装成：

```python
{"video": video_path}
```

不能直接传：

```python
"video.mp4"
```

query 则必须是：

```python
{"video": reference_video, "text": edit_text}
```

---

### 6.3 旧 B-line 数据污染

旧 B-line 质量不够，且缺少新字段。如果训练脚本自动选择旧 run，会导致训练目标不干净。

解决方式：训练脚本只自动识别新的 Audio-CVR B-line 输出：

```text
audio_cvr_bline_6_9s_full_*
audio_cvr_ab_6_9s_minimal_*
```

并且只读取新格式文件。

---

### 6.4 Local segment 的实现成本

直接用 ffmpeg 把每个视频切成多个子片段，训练缓存会变慢，文件数量也会膨胀。

当前解决方式：V1 先使用“局部时间视图编码”，即对同一视频加不同 temporal segment focus instruction，得到局部向量。这样接口先跑通，后续可替换成真实子片段编码。

---

## 10. 当前验证状态

已通过：

```text
synthetic smoke: CPU + mock encoder
real E5 smoke: GPU + e5-omni-7B + local_segments=2
cache-embeddings: global + local embeddings
train-adapter: full loss suite
eval: global / local / global+local
```

本地测试：

```text
python -m unittest discover -v
399 tests OK
```

真实服务器 smoke 结果：

```text
prepare: OK
cache-embeddings: OK
train-adapter: loss 正常收敛
eval: global/local/global+local 全链路通过
```

---

## 11. 后续建议

短期：

```text
1. 等 B-line 大规模数据构造完成。
2. 先用 50-100 条跑 adapter。
3. 跑 run-ablations，确认各模块开关正常。
4. 再用 `v2_research` 跑 50 -> 200 -> 1k。
5. 每一档都先看 grouped recall 和 diagnostics，再决定是否放大。
```

中期：

```text
1. 正式训练启用 `--local-segment-mode ffmpeg`。
2. 对 B-main / B-extended / B-diagnostic 分别训练与评估。
3. 增加 embedding 可视化，例如 UMAP / t-SNE，检查 CORAL 是否真的改善模态分布。
```

长期：

```text
1. 从 adapter 进入 LoRA 微调。
2. 尝试真正的 RET-token / latent pooling。
3. 训练 AudioDelta-E5 full model。
4. 和 agent 路线结合：agent 负责音频证据解释和 rerank。
```

---

## 12. 一句话总结

AudioDelta-E5 训练框架的核心不是“让 E5 多看一点音频”，而是让模型显式学习：

```text
target audio/video 相比 reference audio/video，
是否按照 edit_text 发生了正确的、有方向的音频变化，
并且这种变化发生在保留的视频语境中。
```

当前 V1 已经实现了训练所需的主要结构：Audio-delta loss、hard negatives、reference negative、edit-type-aware delta、local temporal matching、source-disjoint split、shortcut diagnosis 和 ablation。
