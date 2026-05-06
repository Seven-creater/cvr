# VACE Pipeline 当前状态与问题记录

Last updated: 2026-05-06

## 0. 当前总览

### 0.1 我们现在处在什么阶段

当前项目还没有进入“稳定批量生成视觉目标视频”的阶段，而是在做 **VACE capability map**：

```text
目的不是立刻多跑，而是弄清楚：
哪些编辑类型可以生产；
哪些只能小规模实验；
哪些应自动拒绝，避免浪费 GPU。
```

已经明确的一点是：数据集最终可以有很多种类和数量，但 accepted synthetic pairs 的门槛不能靠放低来凑数量。探索阶段可以记录失败，生产阶段必须严格：

```text
mask gate passed
duration gate passed
semantic gate passed
review bundle complete
```

### 0.2 当前最重要的结论

| 问题 | 当前结论 |
|---|---|
| VACE 输入契约 | 已基本修好：`src_video + src_mask + src_ref_images + target prompt`，81f@16fps exact-frame |
| 普通 masked VACE 做完整背景替换 | 不稳定，常变成 blue overlay / style wash，不作为 production 默认路线 |
| 固定优先视觉路由 | 新默认策略：能确定性固定/贴参考图就不让 VACE 猜；VACE 只做必须生成、修边、补遮挡或复杂 inpaint |
| deterministic foreground/background composite | 背景替换 production 默认路线：用前景 mask 保留人物，把选中的 16:9 背景 plate 固定到底层，完全跳过 Wan/VACE |
| composite-first-frame / guided VACE | 保留为 fallback/refine：`mannul7` 和 `mannul8` 证明可行，但现在排在 deterministic composite 后面 |
| mask 生成 | 已开始改成 adaptive diagnostic mask：尽量生成全长稀疏 mask，但是否进 VACE 仍由质量 tier 决定 |
| 当前 35 个 plan | 大部分被正确拒绝；不是“没跑成”，而是发现原始候选质量不适合 VACE |
| 下一步 | 先用 adaptive mask 路线复跑历史失败 mask；对 `usable_for_vace=true` 的样本先走 deterministic/paste route，只有必须生成时才进入 VACE smoke |

### 0.3 当前能力地图

| 编辑类型 | 当前状态 | 原因 / 备注 | 下一步 |
|---|---|---|---|
| deterministic audio route | 已验证可生产 | 画面不变，只改非语言音频事件，已通过 10/10 | 可作为稳定数据来源继续扩大 |
| talking-head background replacement + fixed composite | 新 production 默认候选 | 背景可直接由 src_ref plate 固定铺底，mask 只负责保留人物；不再让 VACE 整块重猜背景 | 先用 `ef8f2818` 和 2-3 条单主体 clip 跑 deterministic smoke，不跑 VACE |
| talking-head background replacement + composite-first-frame | fallback / refine | `mannul7` 成功，`mannul8` 通过脚本化 route 复现；证明 VACE 可在强首帧锚点下工作 | deterministic 边缘差但语义正确时，再用 VACE 小 mask 修边 |
| plain masked background replacement | 默认禁用 | `mannul5/mannul6` 只生成蓝色叠加，原房间结构残留 | 只保留为 experiment / restyle |
| background restyle / soft repaint | 接近可行但未稳定 | 可能适合“风格化原背景”，不适合“房间换成实验室” | 需要单独定义轻量目标和验收 |
| deterministic masked reference paste | 新增候选 | 图标、平面物体、静态区域、简单贴图类编辑不应交给 VACE 猜 | 先挑平面/近静态样本做 1 条 paste smoke |
| existing large object / robot / vehicle attribute edit | 值得继续找样本 | 理论上较适合 VACE，但当前 plan 里很多没有真实 vehicle 或有可见文字 | 重新从 stable clips 中筛更大、更清晰、无文字目标 |
| clothing color / material | 高风险 | mask 经常覆盖乐器/手/身体，VACE 容易变成 vest/polo/dark shirt | 只保留低风险“已有衣物颜色/材质”实验 |
| structural clothing / try-on | 不走默认 VACE | black jacket/coat/blazer 属于虚拟试衣级别，已多次失败 | 未来单独做 try-on-first-frame route |
| small object replacement/removal | 生成诊断 mask，不默认进 VACE | cup/mug/chair 等覆盖太小或多实例，full-frame VACE 不稳定 | adaptive/tiled/high-res mask 先做诊断，达不到质量 tier 仍不进 VACE |
| seated support edit | 自动拒绝 | chair/stool/seat 与人物承重关系冲突 | 不进生产 |
| text/logo/screen edit | 自动拒绝 | OCR/屏幕文字风险高，VACE 容易生成乱码 | 不进生产 |
| multi-scene / montage clip | 生成全长稀疏诊断 mask | Florence-2/SAM2 只能在约 25% 帧检测目标；目标不可见帧应全黑保留 | 记录 `visible_spans`，默认 `usable_for_vace=false`，除非分段质量达标 |

### 0.4 当前最直接的下一步

1. 服务器拉取最新 `codex/vace-pipeline-hardening`，确认包含 `adaptive_repair_v1` mask 改造和 fixed-first route。
2. 先复跑 10 条历史 mask 失败样本，只生成 mask，不跑 VACE。
3. 汇报每条的 `mask_quality_tier`、`usable_for_vace`、`visible_spans`、`reinit_count`、`failure_reasons`。
4. 对 `usable_for_vace=true` 的 background replacement，先跑 `deterministic_foreground_background_composite`；不要直接跑 Wan/VACE。
5. 只有 deterministic/paste 不能表达，或者只需要修边/补遮挡时，才进入 `guided_composite_refine_vace` / `vace_full_generative`。
6. accepted pairs 只收同时通过 duration / semantic / bundle / mask gate 的结果；诊断 mask 继续进入 capability report。

## 1. 当前结论

当前视觉合成路线还没有稳定进入“批量生成目标视频”的阶段。真正的瓶颈已经从 VACE 本身前移到 **mask 生成、route 选择与诚实验收**：

```text
Omni 计划生成完成 -> 候选 plan 筛选完成 -> mask 生成/验收失败 -> 不进入 VACE
```

这不是坏事。现在的 gate 能及时拦截坏 mask，避免继续烧 VACE GPU 生成假阳性样本。当前应该继续先把 mask 路线跑通，而不是降低阈值硬跑 VACE。

官方契约仍然是：

```text
src_video + src_mask + optional src_ref_images + target prompt -> VACE target
```

其中 `src_mask` 白区是生成区域，黑区是保留区域；`src_video` 的编辑区域需要置灰为 127；prompt 应描述目标视频，而不是写操作指令。

截至 `fcbf033 Repair background replacement prompt conflicts` 之后，又根据 `mannul6` 复跑结果确认：即使 prompt / mask / src_ref / frame 对齐全部正确，plain masked VACE 对 talking-head full background replacement 仍然只产生 blue overlay / style wash，不是真正的空间背景重构。随后 `98e0070 Disable plain background replacement VACE route` 禁用了 full background replacement 的 plain masked production route，并切到 composite-first-frame。最新 fixed-first 改造进一步把默认 production route 前移到 deterministic foreground/background composite：能固定背景图时先固定，VACE 只作为 fallback/refine。

`mannul7` 结果证明：`vace_bg_replace_composite_first_frame_mv2v` 能把同一 woman 前景保留下来，并真实生成 futuristic laboratory 背景，而不是蓝色叠加。随后 `mannul8` 用脚本化 composite route 对同一 `ef8f2818` plan 做了复现，review bundle 完整，duration gate 与 semantic gate 均通过。

但 `mannul8` 也暴露了另一层问题：即使 mask 很好，生成式编辑仍可能在不该猜的地方猜错。背景替换、图标/平面贴图、静态区域替换这类任务本质上更适合“固定参考图 + mask 合成”。因此最新路线已经从“有好 mask 就交给 VACE”改成：

```text
能固定的，先固定。
能直接贴参考图的，先贴参考图。
只有必须生成隐藏内容、修边、补遮挡或做复杂 inpaint 时，才用 VACE。
```

Omni 的角色也相应调整：Omni planner 需要在生成 plan 时判断任务是否可 deterministic composite / masked reference paste，并把推荐路线写进 plan；本地 route policy 再做硬兜底，防止背景替换这类任务重新滑回 plain masked VACE。

当前已经完成九类关键修复：

1. **输入契约修复**：VACE 输入强制 exact-frame，`reference/src_video/src_mask` 对齐到 `81f@16fps`，避免 5 秒输入生成 9 秒 target。
2. **诚实验收修复**：duration gate、mask provenance gate、semantic gate、review bundle completeness gate 已经能拦住假阳性。
3. **background replacement prompt 修复**：planner 不再把 source 背景词和 preserve locks 带进 VACE prompt，避免 “换成 lab” 和 “保留 sunlit room/window/door/layout/lighting” 互相打架。
4. **background replacement route 降级**：full background replacement 不再允许作为 plain masked VACE production route；当前默认推荐 `deterministic_foreground_background_composite`，`vace_bg_replace_composite_first_frame_mv2v` 作为 fallback/refine。
5. **composite-first-frame 初步验证**：`mannul7` 的 9/9 review bundle 完整，语义门通过；`mannul8` 通过脚本化 route 复现同 plan，preflight / duration / semantic gate 均通过。
6. **background plan 前移拒绝**：`plan_video_edits` 现在会在规划阶段直接拒绝多场景和多主体的 background scene edit，避免这类样本继续走到 mask 阶段白跑。
7. **composite-first-frame 脚本化**：`run_vace_visual_synthetic_smoke.sh` 仍能根据 fallback route 自动构造 `composite_frame0`、composite `src_video`、composite `src_mask` 和对应 contact sheet，不再依赖服务器手工拼接。
8. **adaptive sparse mask 改造**：`plan-video-masks` 不再因 tiny/multi-shot/multi-instance 等问题直接不产 mask；mask 脚本会尽量生成全长稀疏诊断 mask，并用 `mask_quality_tier` / `usable_for_vace` 控制是否进入 VACE。
9. **fixed-first 路由改造**：full background replacement 的 production 默认路线改为 `deterministic_foreground_background_composite`；`vace_bg_replace_composite_first_frame_mv2v` 变成 fallback，`guided_composite_refine_vace` 只负责修边/融合，`vace_full_generative` 只保留给必须生成的任务。

当前只证明了一条 talking-head background replacement 可通过 composite-first-frame route 成功；下一轮要优先验证 deterministic fixed composite 是否能更稳定、更便宜地产出同类目标。接下来不是大规模跑批，而是用 fixed-first route 小批量验证：哪些任务能确定性完成，哪些需要 VACE 修边，哪些必须交给生成模型，哪些应自动拒绝。

参考：

- VACE User Guide: https://github.com/ali-vilab/VACE/blob/main/UserGuide.md
- Wan2.1 VACE: https://github.com/Wan-Video/Wan2.1
- SAM2 video predictor example: https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb
- Grounded-SAM-2: https://github.com/IDEA-Research/Grounded-SAM-2

## 2. 服务器与分支状态

主线要求：

```text
GitHub repo: Seven-creater/cvr
主工作分支: codex/vace-pipeline-hardening
不要再基于旧 codex/synthetic-dual-route-planner 开新工作
```

服务器关键路径：

```text
repo: /data02/usr/wangqihao/Demo/test/cvr_clean_main
data root: /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
model root: /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone
PLAN_RUN: /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/omni_stable_all_cache_20260428/omni_video_plan10_4gpu_20260429_234351
```

本地代码状态：

```text
branch: codex/vace-pipeline-hardening
最近已推送参考 HEAD: 8e827ab Support composite first-frame VACE smoke
后续新增: adaptive_repair_v1 mask 诊断生成与 VACE preflight 硬门控
本轮新增: fixed-first visual route, deterministic foreground/background composite helper, Omni route-selection prompt
```

服务器执行前必须先 `git fetch` / `git pull --ff-only`，并确认：

```text
git rev-parse --short HEAD
最新短 SHA
```

如果没有包含 `98e0070` 之后的 background route 降级修复，不要继续跑 background VACE。
如果没有包含 `adaptive_repair_v1`，不要复跑历史失败 mask。
如果没有包含 fixed-first route，不要把 background replacement 直接交给 VACE。

当前服务器下一次执行应确认：

```text
git rev-parse --short HEAD
# 期望: 包含 adaptive_repair_v1 的最新 HEAD
```

## 3. 数据与 plan 进展

Omni stable annotations 已完成：

```text
stable clips: 2853
annotations: 2853/2853
reference understanding cache: 约 2847
```

4-GPU / 10-worker 的 plan 任务已完成：

```text
seed inputs: 2847
有效 video edit plans: 35
失败 worker: 0
```

从 35 条 plan 里多轮筛选后，当前可尝试的真实视频候选非常少。大量 plan 已被正确拒绝：

- tiny object：cup / mug 太小，不适合 full-frame VACE。
- visible text：laptop / vehicle 类样本含屏幕文字或字幕。
- seated support：chair/stool/seat 和人物承重关系冲突。
- structural clothing：black jacket / coat / blazer 属于虚拟试衣级别，不再走普通 VACE。
- multi-subject background：多个主体或多场景，前景反选背景 mask 不稳定。
- low contrast clothing：black shirt -> navy shirt 这种差异太弱。

最近剩余的 4 个 real/animation mask smoke 候选是：

| plan | edit | mask query | 当前判断 |
|---|---|---|---|
| ef8f2818 | background -> futuristic laboratory | woman | 可试，但依赖 foreground inverse mask |
| aab85616 | background -> futuristic laboratory | man | 可试，但依赖 foreground inverse mask |
| 8e8069c9 | woman's blouse -> red blouse | blouse | 可试，局部衣物属性 |
| 2babdf1c | character robe red -> blue | robe | 可试，动画局部属性 |

## 4. 已经做过的主要修复

### 4.1 VACE 输入与验收硬化

已完成：

- 强制 VACE 输入 exact-frame：`81f@16fps`。
- 修复 VACE 输出 145 帧 / 9 秒的问题，黑衣服 smoke 曾从 `9.063s` 修到约 `5.063s`。
- 增加 duration drift gate，超过 0.5 秒直接失败。
- manual review bundle 要包含 reference、src_video_for_vace、src_mask、raw target、remux target、prompt、metrics、verdict。
- post-VACE semantic gate 从 black jacket 特例扩展到 clothing/background/object replacement/removal。

相关提交：

```text
83ab919 Force exact-frame VACE inputs
688eb49 Harden VACE mask provenance gates
```

### 4.2 plan lint 与 maskability gate

已完成：

- 拒绝 structural clothing：black jacket / coat / blazer 不再走普通 VACE。
- 拒绝 tiny object full-frame replacement：cup/mug。
- 拒绝 visible text / screen object replacement。
- 拒绝 seated support replacement/removal：chair/stool/seat/bench/sofa 与人物 sitting/seated 冲突。
- 拒绝 multi-subject background inverse mask。
- 拒绝 low-contrast dark clothing edit。

相关提交：

```text
24b4195 Reject structural clothing VACE edits
5da9a53 Reject unsafe VACE replacement plans
56f56b1 Reject multi-subject background VACE masks
```

### 4.3 mask provenance 与 query 修复

已完成：

- mask manifest 增加 `mask_semantics_version`、`mask_polarity`、`mask_query`、`mask_mode`、sampled frames、keyframe、generator commit。
- `mask_semantics_version` 升到 3，防止复用旧 mask。
- background inverse 增加 foreground raw mask 指标：
  - `foreground_subject_coverage_ratio_avg`
  - `foreground_subject_temporal_stability`
  - `foreground_subject_nonempty_frame_ratio`
- 不再只用 bbox 判断 subject overlap，避免 bbox 包含大量背景导致误判。
- 支持 `mask_query_candidates`，mask 脚本会逐个 query 尝试，并记录 `mask_attempts`。
- 修复 `robe/blouse` 这类实际局部衣物编辑但 family 被写成 `background_change` 的 plan：现在走局部 mask，不走背景反选。

相关提交：

```text
2346636 Retry VACE mask queries with foreground metrics
```

### 4.4 adaptive sparse mask 改造

当前最大的工程调整是把 mask 阶段拆成“生成诊断件”和“允许进 VACE”两层：

```text
mask 探索层:
尽量生成 full-length mask，目标不可见帧写全黑，记录失败原因和可见区间。

VACE 生产层:
只允许 status=generated 且 usable_for_vace=true 且 mask_quality_tier 不是 diagnostic/failed 的 mask 进入。
```

已完成：

- `plan-video-masks` 不再因为 `small_object_too_tiny_for_fullframe_vace`、`multi_shot_mask_route_unsupported`、`ambiguous_multi_instance_mask_query` 等问题直接不产 mask。
- 这些样本仍会写入 mask plan / initial manifest，但标记：

```text
mask_generation_strategy=adaptive_repair_v1
generate_diagnostic_mask=true
usable_for_vace_default=false
maskability_issue=<具体问题>
```

- `generate_grounded_sam2_video_masks.py` 增加 dense frame sampling，不再只抽 5 帧。
- 检测阶段会记录 `detection_attempts`，并从多个候选帧中选择最多 3 个 anchor frames。
- SAM2 video predictor 会对多个 anchor 重新 prompt / reinit，最后合并成全长 mask。
- 如果 gate 失败但已经生成可诊断 mask，输出 `status=diagnostic_generated`，保留 mask 视频，不再删除。
- 如果 detector 完全失败，仍会写全黑 full-length mask，方便下游和 review bundle 保持结构一致。
- manifest 新增字段：

```text
mask_generation_strategy
sparse_full_length
visible_spans
detector_cascade
detection_attempts
anchor_frame_indices
prompt_type
reinit_count
repair_rounds
mask_quality_tier
usable_for_vace
failure_reasons
```

质量 tier 语义：

| tier | 含义 | 是否可进 VACE |
|---|---|---|
| `excellent` | gate 全过，nonempty/stability 很高 | 是 |
| `usable_for_vace` | gate 全过，但质量不是 excellent | 是 |
| `diagnostic_only` | 产出了 mask，但 gate 未过或默认仅诊断 | 否 |
| `failed` | detector 失败或基本全空 | 否 |

VACE smoke preflight 现在会拒绝：

```text
mask_generation_strategy != adaptive_repair_v1
status != generated
usable_for_vace=false
mask_quality_tier in {diagnostic_only, failed}
```

这点很关键：我们不是降低验收门槛，而是把“多产 mask 用于诊断”和“只收高质量样本”分开。历史失败样本现在可以产生全长稀疏 mask 和更细的 failure report，但不会自动烧 VACE GPU。

### 4.5 GroundingDINO checkpoint 问题修复

服务器当前 GroundingDINO 目录是 HuggingFace 格式：

```text
model.safetensors
pytorch_model.bin
config.json
tokenizer.json
```

但官方 GroundingDINO `load_model()` 需要的是 torch checkpoint dict，并会读取：

```python
checkpoint["model"]
```

因此：

- `model.safetensors` 不能用 `torch.load()` 读取。
- 服务器的 `pytorch_model.bin` 是 HuggingFace state_dict 格式，没有 `"model"` key。
- 当前服务器没有正确的 `groundingdino_swint_ogc.pth`。

已推送修复：

```text
ffb8a3d Prefer torch GroundingDINO checkpoints
2d1fc41 Fallback from HF GroundingDINO checkpoints
```

`2d1fc41` 的作用是：`--grounder auto` 遇到 HF 格式 GroundingDINO checkpoint 时自动退到 Florence-2。服务器拉到 `fcbf033` 后已经包含这个 fallback。注意：这不是让 GroundingDINO 变可用，而是避免 `auto` 因错误 checkpoint 格式直接崩溃。

### 4.6 background replacement prompt 冲突修复

`mannul5` 暴露的背景替换失败不是 mask / 帧率 / remux 问题，而是 planner 把 source 背景锁进了 VACE prompt 与 preserve 约束：

```text
错误 target_prompt:
A woman ... speaks to the camera in a sunlit room with a futuristic laboratory background.

错误 preserve / locks:
sunlit room, window, door, preserve lighting exactly, preserve layout exactly
```

这会让 VACE 同时收到互相冲突的信号：

```text
把背景换成 futuristic laboratory
保留原 sunlit room / window / door / lighting / layout
```

因此现在把 `background_replace` 写成 deterministic repair rule，而不是继续让 Omni planner 自由发挥：

- `target_prompt` 必须是最终状态描述，只描述目标画面。
- VACE prompt 中不能出现 source 背景词：`sunlit room`、`window`、`door`、`original room`、`source background`。
- `preserve_tokens` 只保留前景主体、身份、脸、头发、眼镜、动作、嘴型、姿态、时序、camera framing。
- `preserve_regions` 只保留前景主体区域，禁止保留 `window/door/room/wall/background`。
- `negative_prompt` 只防人物坏、闪烁、伪影、额外人物；不再写 `preserve lighting/layout`。
- `visual_edit_risk.locks` 只保留 foreground identity / pose / timing / camera framing，并显式禁止保留 source background layout or lighting。

修复后的 VACE prompt 形态应类似：

```text
A woman with curly red hair and glasses speaks to the camera in a clean blue-white futuristic laboratory interior, with smooth illuminated wall panels and lab benches in the background, stable frontal medium-close-up framing.
```

旧 plan 或旧 review bundle 只要仍包含 `sunlit room/window/door/layout/lighting` 这类 source 背景锁，就必须被 plan lint / smoke preflight 拒绝，不能进入 VACE。

`mannul6` 已证明：即使上述 prompt 修复生效，plain masked VACE 仍然只做蓝色 overlay。因此现在进一步要求：

- `background_replace_policy.plain_masked_vace_production=false`
- `background_replace_policy.recommended_route=deterministic_foreground_background_composite`
- `background_replace_policy.fallback_route=vace_bg_replace_composite_first_frame_mv2v`
- `background_replace_policy.refine_route=guided_composite_refine_vace`
- `route_suitability.production_allowed=true` 只表示 deterministic fixed route 可进入生产候选，不表示 plain VACE 可进入生产
- normal `run_vace_visual_synthetic_smoke.sh` 默认拒绝 full background replacement，除非显式设置 `ALLOW_PLAIN_BACKGROUND_REPLACE=1` 做实验。
- accepted synthetic pairs 拒绝 `model_route=vace_controlled` 的 full background replacement，除非 generation 明确记录：
  - `background_replace_route=vace_bg_replace_composite_first_frame_mv2v`，或
  - `background_replace_route=deterministic_foreground_background_composite`

这会阻止服务器继续把同类 blue overlay 当成可收样本。

这次修复落在：

- `app/composed_data.py`
  - 新增 background replacement 判定与 deterministic repair。
  - 重写 target prompt、preserve tokens、preserve regions、negative prompt、risk locks。
  - plan lint 拒绝 source 背景词和 layout/lighting preserve lock。
  - 给 background replacement 写入 route policy，并在 accepted-pair gate 拒绝 plain masked VACE。
- `scripts/run_vace_visual_synthetic_smoke.sh`
  - 对旧 plan 增加 smoke preflight lint，防止复用冲突 plan。
  - 默认禁止 full background replacement 继续走 plain masked VACE。
- `tests/test_composed_data.py`
  - 增加 background prompt 冲突回归测试。
- `tests/test_scripts.py`
  - 确认 smoke script 包含 background prompt 冲突拦截。

本地验证结果：

```text
git diff --check: OK
bash -n scripts/run_vace_visual_synthetic_smoke.sh: OK
python -m unittest tests.test_scripts tests.test_composed_data -v: 188 OK, 2 skipped
python -m unittest discover -s tests -v: 214 OK, 2 skipped, 1 error
```

全量测试唯一 error 是本地 `.venv` 缺少 `torch`，失败点在 `tests/test_avigate_official.py`，与 VACE hardening 修改无关。

### 4.7 fixed-first visual route 修复

`mannul8` 之后进一步确认：高质量 mask 只是必要条件，不是充分条件。background replacement 的目标背景如果已经有 16:9 `src_ref` plate，最稳的生产路线不是让 VACE 重新生成整块背景，而是直接固定参考背景，再用 mask 保留前景主体。

当前新增三层视觉路线：

| route | 用途 | 是否默认用 VACE |
|---|---|---|
| `deterministic_foreground_background_composite` | talking-head / static-camera 背景替换；固定 `src_ref` plate，保留原前景 | 否 |
| `deterministic_masked_reference_paste` | 图标、平面物体、静态区域、简单参考图贴合 | 否 |
| `guided_composite_refine_vace` | deterministic 结构正确但边缘/融合/阴影需要小范围修复 | 是，但只给小 repair mask |
| `vace_full_generative` | object removal / hidden-content inpaint / 复杂非刚性替换 | 是 |

这次新增了脚本：

```text
scripts/build_deterministic_masked_composite.py
```

它按 VACE mask 语义复用现有 mask：

```text
mask white = 背景 / 可替换区域 -> 使用固定 src_ref plate
mask black = 前景 / 保留区域 -> 使用原 reference foreground
```

输出内容包括：

```text
target_with_ref_audio.mp4
metadata/deterministic_composite_metrics.json
metadata/deterministic_composite_command.json
review_inputs/src_ref_plate.png
review_inputs/alpha_contact.jpg
review_inputs/composite_target_contact.jpg
metadata/post_vace_or_composite_verdict.json
```

`run_vace_visual_synthetic_smoke.sh` 现在遇到 `background_replace_route=deterministic_foreground_background_composite` 会直接调用这个脚本并跳过 Wan/VACE。review bundle 不再要求 VACE log，但必须要求 deterministic composite metadata、alpha contact、composite target contact 和 semantic verdict。

Omni 侧也做了 prompt 调整：plan 生成时会提醒模型优先判断能否 deterministic composite / masked reference paste，并把 VACE 留给 seam repair、harmonization、occlusion/inpaint 或必须生成的内容。最终执行仍由本地 route policy 兜底，避免 planner 偶然把可固定任务送去 full generative VACE。

新的 accepted-pair 语义是：

```text
generation_route=deterministic_foreground_background_composite
requires_vace=false
duration_gate.passed=true
semantic_gate.passed=true
review_bundle_complete=true
```

这条路线的意义是把“能确定的视觉变化”从生成模型里拿出来。它不会解决所有编辑类型，但会让 background replacement 这类可固定任务更便宜、更稳定，也更适合批量生产。

## 5. 当前遇到的问题

### 5.1 Florence-2 + SAM2.1 mask 质量不稳定

最新 mask smoke 中 4 个候选只有 1 个通过 mask gate：

| plan | edit | status | 关键指标 / 原因 |
|---|---|---|---|
| ef8f2818 | woman background -> futuristic lab | mask passed | avg coverage 0.5207, temporal stability 0.9969, nonempty 1.0, foreground subject coverage 0.4793 |
| 8e8069c9 | blouse -> red blouse | failed | temporal stability 0.2476, nonempty 0.2533, ukulele protected overlap 0.99 |
| aab85616 | man background -> futuristic lab | failed | avg coverage 0.1819, temporal stability 0.2492, foreground nonempty 0.2511 |
| 2babdf1c | robe red -> blue | failed | avg coverage 0.0023, temporal stability 0.1444, nonempty 0.16 |

56f56b1 下曾经 4 个候选全部 mask gate 失败：

| 类别 | 失败表现 |
|---|---|
| background inverse | foreground subject mask 要么覆盖过大，要么跨帧不稳定 |
| clothing / robe / blouse | temporal stability 低于 0.75，或者存在空帧 |

典型失败：

```text
woman/man background inverse:
subject overlap 或 foreground stability 不达标

blouse/robe:
temporal_stability < 0.75
nonempty_frame_ratio 不稳定
coverage 极低或空帧
```

这是当前最核心的问题。不要用降低阈值来绕过，因为坏 mask 进入 VACE 后通常会产生：

- 背景没变，人物被糊。
- 衣物变成 vest/polo/暗色上衣，而不是目标衣物。
- target 变成 near-duplicate 或主体漂移。

最新策略不是“mask 失败就不做”，而是：

```text
尽量生成 mask -> 标质量 tier -> 写 visible spans / failure reasons -> 决定是否进 VACE
```

所以未来看到 `diagnostic_generated` 不应理解成成功，也不应理解成脚本失败。它表示：系统已经生成了可审查的全长 mask，但当前质量不足以进入 VACE。多场景 clip 中目标只出现后 25% 帧时，前 75% 帧应该是全黑 mask；这有助于后续分析和分段路线，但默认 `usable_for_vace=false`。

### 5.2 ef8f2818 VACE smoke 状态

`ef8f2818` 在补齐 16:9 futuristic laboratory background src_ref 后已经启动过一次 VACE smoke。结果：

```text
VACE 推理: 成功
raw target: 5.063s, 81 frames @ 16fps
audio-remux target: 5.109s
reference_for_vace: 5.086s
duration drift: 0.023s
preflight_report.json: passed=true
```

这说明 VACE 输入链路、exact-frame 策略、mask、src_ref、音频 remux 基本正常。

但这条样本不能标为 accepted。第一层原因是后处理脚本在生成 `duration_metrics.json` / `post_vace_verdict.json` 之前崩溃：

```text
TypeError: Python 3.8 does not support list[str] runtime annotations
```

根因是 `scripts/run_vace_visual_synthetic_smoke.sh` 的内联 Python 里有：

```python
def semantic_requirements_for_family(family: str) -> list[str]:
```

服务器系统 `python3` 是 3.8，不支持这种写法。修复方式是改成：

```python
def semantic_requirements_for_family(family: str) -> list:
```

本地已修复并通过测试。随后检查 `/Users/Admin/Desktop/mannul5` 的 review bundle，视觉结论是：

- reference/src_video/mask 极性正确，白区是背景，黑区保留 woman。
- target 不是真正的 futuristic laboratory；原来的 room/window/door 仍可见。
- VACE 只生成了蓝色科幻 overlay / tint，背景语义没有被替换。
- woman 基本保留，但这属于 `subject_preserved_but_edit_failed`。

因此 `mannul5` 必须归档为 `failed_semantic_gate`，错误标签应包含：

```text
target_background_missing
original_background_retained
background_not_replaced_original_room_still_visible
futuristic_lab_only_blue_overlay
subject_preserved_but_edit_failed
```

不要把这条样本写入 `accepted_synthetic_pairs.jsonl`。

### 5.3 mannul7 / mannul8 composite-first-frame 状态

`mannul7` 是同一类 background replacement 的第一条正向结果。review bundle 已完整下载到本地：

```text
/Users/Admin/Desktop/mannul7
```

bundle 9/9 项齐全：

| item | file |
|---|---|
| reference | reference_contact.jpg |
| composite frame0 | composite_frame0.png |
| src video contact | composite_src_video_contact.jpg |
| src mask contact | composite_src_mask_contact.jpg |
| src ref | src_ref_candidate_001.png |
| raw target | raw_output.mp4 |
| target | target_with_ref_audio.mp4 |
| duration metrics | duration_metrics.json |
| post-VACE verdict | post_vace_verdict.json |

语义结论：

- `post_vace_verdict.json` 标记 `passed_semantic_gate`。
- route 是 `vace_bg_replace_composite_first_frame_mv2v`。
- 背景真实替换为 futuristic laboratory，有发光墙板、走廊/实验台等目标语义。
- 原始 room / window / wall 结构消失，不再是 `mannul5` / `mannul6` 那种 blue overlay。
- foreground woman 的红卷发、眼镜、姿态和说话动作基本保留。

早期 `duration_metrics.json` 有一个重要 bug：它把约 3 秒 target 与约 15 秒原始 reference 比较，导致 `duration_drift_seconds=11.961`。正确做法是比较同一 VACE 输入段：

```text
reference_for_vace / composite src_video / src_mask / raw target / remux target
```

当前服务器复核结果已经修正为：

```text
duration drift = 0
duration_gate.passed = true
```

因此当前验收策略是：

- `mannul7` 可以作为 `composite-first-frame background replacement` 的语义成功样本。
- 但在 accepted synthetic pairs 中，必须要求 `generation.duration_metrics.duration_gate.passed=true`。
- 如果只有 `duration_drift_seconds`、没有 `duration_gate` 结构化字段，必须拒绝。
- duration gate 的参考必须是实际 VACE 输入段，而不是原始完整 reference clip。

新增代码已把 visual synthetic accepted-pair gate 改成强制要求：

```text
generation.duration_metrics.duration_gate.passed=true
```

这会防止类似早期 `mannul7` 的“语义成功但元数据参照错误”的样本直接进入 accepted pairs。

最新脚本化要求：

- `run_vace_visual_synthetic_smoke.sh` 直接支持 `vace_bg_replace_composite_first_frame_mv2v`。
- composite route 必须有 `src_mask` 和至少 1 张 selected `src_ref_image`。
- review bundle 必须写入 `composite_frame0.png`、`composite_src_video_contact.jpg`、`composite_src_mask_contact.jpg`。
- frame 0 使用 `composite_frame0 + all-black mask` 作为目标空间 anchor；后续帧继续使用原前景 + 灰色背景 + 白色背景 mask。

`mannul8` 是脚本化 route 的复现结果，已下载到本地：

```text
/Users/Admin/Desktop/mannul8
```

它对应同一 plan：

```text
plan_id tail: ef8f2818
route: vace_bg_replace_composite_first_frame_mv2v
commit: 8e827ab
```

bundle 完整项包括：

| item | file |
|---|---|
| reference | reference_contact.jpg |
| composite frame0 | composite_frame0.png |
| composite src video | composite_src_video_contact.jpg |
| composite src mask | composite_src_mask_contact.jpg |
| src ref | src_ref_images/001_candidate_001.png |
| raw target | raw_output / raw target contact |
| target | target_with_ref_audio / target contact |
| duration metrics | duration_metrics.json |
| post-VACE verdict | post_vace_verdict.json |

`mannul8` 的关键元数据：

```text
preflight_report.passed=true
reference_for_vace=81f@16fps, 5.086s
src_video_for_vace=81f@16fps, 5.063s
src_mask=81f@16fps, 5.063s
raw target=81f@16fps, 5.063s
audio-remux target=81f@16fps, 5.109s
duration_drift=0.023s
duration_gate.passed=true
post_vace_verdict.verdict=passed_semantic_gate
```

视觉检查结论：

- 原始房间、窗户、门、墙面结构已经消失，不再是 `mannul5/mannul6` 的 blue overlay。
- 目标背景有 clean blue-white futuristic laboratory 的空间语义。
- 前景 woman 的身份、红卷发、眼镜、位置和说话动作基本保留。
- target 背景比 composite frame0 更柔和、更虚化，但作为 background replacement route smoke 是正向结果。

因此当前可以把 `mannul7/mannul8` 合并判断为：

```text
composite-first-frame background replacement: route-level viable, needs more clip-level replication
plain masked background replacement: still disabled for full background replacement
```

### 5.4 adaptive mask 后还没解决的问题

adaptive mask v1 解决的是“不要过早放弃诊断”，不是一次性解决所有 mask 质量问题。仍然存在：

- Florence-2 对泛词 `man/woman/clothing` 的检测不稳定，尤其是多场景视频。
- 小目标在 640x360 下仍然覆盖太小，需要后续 high-res / tiled detection 才可能改善。
- 衣物与手、乐器、麦克风重叠时，mask 即使生成也常常 `diagnostic_only`。
- 多实例 `chair/table/person` 仍然需要 target instance alignment，否则只能做诊断 mask。
- 全长稀疏 mask 对 VACE 不一定有用；如果目标只在 25% 帧出现，默认仍不能进整段 VACE。

下一轮 mask 优化重点：

1. 给小目标接 high-res / tiled detection。
2. 给衣物接 protected-object negative prompt / point refinement。
3. 给多场景 clip 做 scene-span segmentation，而不是整段硬传播。
4. 正确接入本地 GroundingDINO `.pth` 或其它 open-vocabulary detector 作为 Florence-2 fallback。

### 5.5 mannul9 adaptive mask 诊断结果

`/Users/Admin/Desktop/mannul9` 是第一批 adaptive mask-only 诊断包。它的结论不是“又失败了”，而是 mask 阶段已经开始产出可审查证据：

```text
10 条 mask plan
1 条 excellent / usable_for_vace=true
9 条 diagnostic_generated / usable_for_vace=false
```

通过项：

| plan | edit | mask query | tier | 结论 |
|---|---|---|---|---|
| ef8f2818 | background -> futuristic laboratory | woman | excellent | 可进入 fixed-first 背景替换路线；优先 deterministic composite，不要直接跑 VACE |

失败项按原因分层：

| 类型 | 样本 | 现象 | 下一步 |
|---|---|---|---|
| 小目标但有局部检测 | 23ab74e5 cup, 6bc2cc4b cup | coverage 约 0.26%-0.44%，nonempty 约 58%-75% | 不进整段 VACE；后续尝试 high-res/tiled detector + visible-span clip |
| 动画小局部 | 2babdf1c robe | coverage 0.95%，nonempty 83%，接近但未过 | 可作为 visible-span / local recolor 诊断候选，不进 full VACE |
| 结构性衣物 | 2d9c7b5a jacket | protected overlap=1.0，且 black jacket 本身是 try-on 级别 | 继续拒绝，不用普通 VACE 救 |
| 衣物与保护物重叠 | 8e8069c9 blouse | nonempty 2%，protected overlap 0.9865 | 需要更准 clothing/person parsing；当前不进 VACE |
| 多场景/短暂主体 | aab85616, c708d4d5, e6d4f1fc | nonempty / temporal stability 约 25% | 不该整段跑；应裁 visible span 后再评估 |
| 多实例/不成立 removal | ebd86ec9 chair | almost empty，多实例/场景语义不稳定 | 保留拒绝样本 |

这批结果说明下一层 mask 优化方向应该是：

```text
full-length sparse mask 继续保留用于诊断；
但如果 visible span 本身足够长，就生成 span-level edit candidate；
对 span 内重新计算 nonempty / stability / coverage；
只有 span-level gate 通过，才进入 deterministic/paste/VACE route。
```

这能避免一个典型误判：目标只在后 25% 帧出现时，整段 `nonempty_frame_ratio` 必然失败；但把 clip 裁到可见区间后，它可能变成可用样本。也就是说，下一步不应降低整段 gate，而应新增 `visible_span_reroute`。

### 5.6 GroundingDINO 当前不可用

当前服务器没有官方格式 GroundingDINO `.pth` checkpoint。因此：

```text
--grounder groundingdino 不能用
--grounder auto 在 fcbf033 上会遇到 HF checkpoint 格式时 fallback 到 Florence-2
```

如果服务器没有拉到 `fcbf033`，仍应显式使用：

```text
--grounder florence2
```

### 5.7 复杂编辑路线暂时不应该继续烧 VACE

这些任务已经明确不适合当前默认 VACE route：

- short sleeve / patterned shirt -> open black long-sleeved jacket
- chair -> stool，尤其有人坐着时
- cup/mug 小物体 replacement
- laptop/tablet 且屏幕有文字
- 多主体或多镜头背景替换
- Qwen-Image 生成的 1:1 方形背景参考图用于 16:9 background change

### 5.8 capability map 新结论

最新 capability map 的意义不是“又失败了一轮”，而是把当前默认 VACE route 的主要瓶颈定位清楚了：

- 当前最大瓶颈不是 VACE 推理本身，而是 source clip suitability。
- 多场景 / 主体只在后半段短暂出现的 clip，会让 Florence-2 + SAM2.1 只能在约 25% 帧上检测到目标，因此天然过不了 `nonempty_frame_ratio >= 0.9` 和 `temporal_stability >= 0.75`。
- 这类问题对 full background replacement 最致命，所以现在已经把 background scene edit 的多场景 / 多主体参考前移到 `plan_video_edits` 拒绝，而不是等到 `plan_video_masks` 再失败。
- 其他类别暂时还保留在 maskability gate 探索，因为我们还不想过早把所有非 background 类型都一刀切掉。

## 6. 当前推荐下一步

### 6.0 总体策略

现在不要把“数量”理解成“所有 plan 都硬跑 VACE”。更合理的策略是分两层：

```text
探索层：多试类别，记录失败原因，完善 capability map。
生产层：只收 gate 全通过的样本，宁缺毋滥。
```

短期目标不是直接生成上千条视觉样本，而是先找到 2-3 个可重复的高产类别。当前最有希望的是：

1. fixed deterministic talking-head background replacement。
2. deterministic audio synthetic。
3. 更干净 source clips 上的 existing object / large attribute edit。
4. 静态或近静态视频中的局部颜色 / 材质变化。
5. visible-span reroute：对 sparse mask 中可见区间足够长的片段，裁成短 clip 后重新 gate。

当前不建议继续在这批 35 条 plan 里反复调阈值。更应该把新筛选规则前移到 clip/plan 阶段：

```text
单镜头
主体/目标全程可见
目标覆盖面积足够
无可见文字
无承重/接触关系
无多实例歧义
```

### 6.1 服务器下一步执行方向

先不要扩大跑批，也不要再跑 plain masked background replacement。服务器下一步应拉取最新 `codex/vace-pipeline-hardening`，先复跑 adaptive mask 诊断任务；如果出现新的 `usable_for_vace=true` 样本，先让 route policy 选择 deterministic / paste / guided refine / full VACE，不要默认把它送进 Wan/VACE。

第一优先级：adaptive mask 诊断复跑：

```text
从历史失败 mask 里选 10 条，只跑 mask，不跑 VACE。
目标: 观察 adaptive_repair_v1 是否能产出更完整的 visible_spans / diagnostic mask / usable_for_vace=true 候选。
```

第二优先级：visible-span reroute：

```text
输入: mannul9 这类 diagnostic_generated manifest
目标: 对 visible_spans 中足够长的区间裁短 clip，重新计算 span 内 mask gate
规则: 不降低整段阈值；只允许 span-level gate 通过的片段进入 deterministic/paste/VACE route
优先样本: aab85616/c708d4d5/e6d4f1fc 的 background span，23ab74e5/6bc2cc4b 的 cup span
```

第三优先级：验证 fixed deterministic background composite：

```text
plan: ef8f2818 或其它单主体全程可见 talking-head
task: background -> futuristic laboratory / clean target background
当前状态: helper 已实现，smoke script 会在 deterministic route 下跳过 Wan/VACE
下一步: 先跑 deterministic_foreground_background_composite，并把结果与 mannul8 VACE candidate 并列 review
```

第四优先级：继续验证 composite-first-frame / guided refine route：

```text
plan: ef8f2818
task: woman background -> futuristic laboratory
当前状态: mannul8 已用脚本化 route 复现成功
下一步: 只在 deterministic composite 边缘/融合不够时，用 seam/repair mask 做 guided VACE；不要整块背景交给 VACE
```

执行前必须：

```text
git fetch origin
git checkout codex/vace-pipeline-hardening
git pull --ff-only origin codex/vace-pipeline-hardening
git rev-parse --short HEAD
```

期望 HEAD 至少包含：

```text
98e0070
以及后续 composite-first-frame smoke 脚本化修复
```

然后跑：

```text
python -m unittest tests.test_scripts tests.test_composed_data -v
```

如果测试通过，再重新生成 ef8f2818 或其它单场景 talking-head background_replace 的 deterministic smoke。不要复用 mannul5/mannul6 的旧 plan，也不要复用旧 review bundle。

建议服务器先做这一条最小闭环：

```text
1. 拉最新 HEAD。
2. 确认 ef8f2818 的 v3 mask、selected src_ref_image、prompt、policy 都存在。
3. 直接跑 run_vace_visual_synthetic_smoke.sh，让脚本按 `background_replace_route` 决定是否跳过 Wan/VACE。
4. 如果是 deterministic route，检查 review bundle 是否包含 `src_ref_plate.png`、`alpha_contact.jpg`、`composite_target_contact.jpg`、`deterministic_composite_metrics.json`、`post_vace_or_composite_verdict.json`。
5. 如果 semantic gate 通过，再把该样本标记为 deterministic background composite 可生产候选。
```

新 plan 必须人工/脚本确认：

```text
target_prompt 不含 sunlit room / window / door / original room / source background
preserve_tokens 不含 sunlit room / lighting / layout / room / window / door
preserve_regions 不含 window / door / room / wall / background
VACE prompt 是最终状态描述，不是 “in a sunlit room with lab background”
```

只有这些都通过，也不要直接走 plain VACE。background replacement 默认先做 deterministic fixed composite：

```text
background plate = selected src_ref_image resize/crop 到视频尺寸
foreground = reference 中 mask 黑区
background = fixed plate 中 mask 白区
VACE = 不调用
```

如果 deterministic 结构正确但边缘需要修复，再生成 composite first frame / guided refine VACE：

```text
frame0 = same woman foreground + target futuristic lab background composite
frame0 mask = all black / retain full composite first frame
frames 1..N src_video = original woman foreground + gray 127 background
frames 1..N src_mask = foreground black preserve, background white generate
```

然后 generation metadata 必须记录：

```text
background_replace_route=deterministic_foreground_background_composite
requires_vace=false
duration_metrics.duration_gate.passed=true
```

否则 accepted-pair gate 会拒绝。

### 6.2 mask smoke 验收字段

每次 mask smoke 完成后，服务器必须汇报：

```text
plan_id_tail
edit_text
selected mask_query
mask_query_candidates
status
mask_attempts
mask_coverage_ratio_avg
mask_temporal_stability
mask_nonempty_frame_ratio
foreground_subject_coverage_ratio_avg
foreground_subject_temporal_stability
foreground_subject_nonempty_frame_ratio
failure_reason
```

只有 mask status 为 `generated` 且 gate passed 的 plan，才允许进入 VACE。

### 6.3 不要做的事

当前不要做：

- 不要降低 temporal stability / nonempty / coverage 阈值硬过。
- 不要复用旧 `mask_semantics_version < 3` 的 mask。
- 不要继续跑 black jacket。
- 不要继续跑 chair/stool。
- 不要给 GroundingDINO 下载或硬塞 HF checkpoint。
- 不要在 mask 全失败时跑 VACE。
- 不要复用 mannul5 的旧 plan 或旧 bundle。
- 不要把 blue overlay / original room retained 的背景样本标成成功。
- 不要再用普通 `run_vace_visual_synthetic_smoke.sh` 默认路线跑 full background replacement；它现在应被 preflight 拒绝。
- 不要在背景可固定时调用 VACE 重新生成整块背景。

## 7. 当前判断

现在 pipeline 的主要进步是：

```text
它开始诚实地失败。
```

这比之前假通过更重要。前面有多个样本“看起来通过”，但实际是：

- target 没有完成编辑。
- target 时长漂移。
- mask 极性错误。
- review bundle 不完整。
- semantic gate 没拦住 dark shirt / vest / blurred subject。

现在这些问题大部分已经被 gate 拦截。下一步的关键不是继续盲跑 VACE，而是：

1. background replacement prompt 冲突已经修复，但 plain masked VACE 仍然失败。
2. full background replacement 已从 plain masked VACE production route 降级。
3. fixed-first route 已经落地：background replacement 默认用 deterministic foreground/background composite，VACE 只作为 fallback/refine/full-generative route。
4. `mannul7/mannul8` 已经证明 `vace_bg_replace_composite_first_frame_mv2v` 能解决 blue overlay，并且脚本化 route 能复现；但它现在不是第一优先级。
5. 在现有 35 个 plan 和历史失败样本里继续生成 adaptive diagnostic mask，寻找新的 `usable_for_vace=true` 候选。
6. 如果 Florence-2 + SAM2.1 对真实视频仍然全失败，切换到更简单的 synthetic route：
   - deterministic audio route
   - deterministic fixed background composite
   - deterministic masked reference paste
   - existing large object / robot / vehicle attribute edit
   - animation / static-like video local color edit
7. 等 mask route 有稳定通过样本，再按 route selector 恢复非背景替换类 VACE smoke。
