# VACE Pipeline 当前状态与问题记录

Last updated: 2026-05-06

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

截至 `fcbf033 Repair background replacement prompt conflicts` 之后，又根据 `mannul6` 复跑结果确认：即使 prompt / mask / src_ref / frame 对齐全部正确，plain masked VACE 对 talking-head full background replacement 仍然只产生 blue overlay / style wash，不是真正的空间背景重构。随后 `98e0070 Disable plain background replacement VACE route` 禁用了 full background replacement 的 plain masked production route，并要求切到 composite-first-frame。

最新 `mannul7` 结果证明：`vace_bg_replace_composite_first_frame_mv2v` 能把同一 woman 前景保留下来，并真实生成 futuristic laboratory 背景，而不是蓝色叠加。这是第一条 background replacement route 层面的正向 smoke。不过当前还不能直接收进 accepted synthetic pairs，因为 `duration_metrics.json` 错把 target 与 15 秒原始 reference 比较，而不是与实际 VACE 输入段 / composite source 比较。下一步必须先修 duration gate 参照，再扩 2-3 条同 route smoke。

当前已经完成五类关键修复：

1. **输入契约修复**：VACE 输入强制 exact-frame，`reference/src_video/src_mask` 对齐到 `81f@16fps`，避免 5 秒输入生成 9 秒 target。
2. **诚实验收修复**：duration gate、mask provenance gate、semantic gate、review bundle completeness gate 已经能拦住假阳性。
3. **background replacement prompt 修复**：planner 不再把 source 背景词和 preserve locks 带进 VACE prompt，避免 “换成 lab” 和 “保留 sunlit room/window/door/layout/lighting” 互相打架。
4. **background replacement route 降级**：full background replacement 不再允许作为 plain masked VACE production route；默认推荐切到 `vace_bg_replace_composite_first_frame_mv2v`，否则保留为 experiment-only。
5. **composite-first-frame 初步验证**：`mannul7` 的 9/9 review bundle 完整，语义门通过；但 duration gate 元数据必须修正后才能作为正式 accepted 样本。

当前只证明了一条 talking-head background replacement 可通过 composite-first-frame route 成功。下一步不是大规模跑批，而是修正 duration gate 与 accepted-pair metadata，然后小批量验证同 route 是否可复现。

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
PLAN_RUN: /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/omni_stable_all_cache_20260428/omni_video_plan10_4gpu_20260429_234351
```

本地代码状态：

```text
branch: codex/vace-pipeline-hardening
远端 HEAD: 仍需以最新 codex/vace-pipeline-hardening 为准
状态: 本地与 origin/codex/vace-pipeline-hardening 已同步
```

服务器执行前必须先 `git fetch` / `git pull --ff-only`，并确认：

```text
git rev-parse --short HEAD
最新短 SHA
```

如果没有包含 `98e0070` 之后的 background route 降级修复，不要继续跑 background VACE。

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

### 4.4 GroundingDINO checkpoint 问题修复

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

### 4.5 background replacement prompt 冲突修复

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
- `background_replace_policy.recommended_route=vace_bg_replace_composite_first_frame_mv2v`
- `route_suitability.production_allowed=false`
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
python -m unittest tests.test_scripts tests.test_composed_data -v: 181 OK, 2 skipped
python -m unittest discover -s tests -v: 214 OK, 2 skipped, 1 error
```

全量测试唯一 error 是本地 `.venv` 缺少 `torch`，失败点在 `tests/test_avigate_official.py`，与 VACE hardening 修改无关。

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

### 5.3 mannul7 composite-first-frame 状态

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

但 `duration_metrics.json` 当前有一个重要 bug：它把约 3 秒 target 与约 15 秒原始 reference 比较，导致 `duration_drift_seconds=11.961`。正确做法应该比较同一 VACE 输入段：

```text
reference_for_vace / composite src_video / src_mask / raw target / remux target
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

这会防止类似 `mannul7` 的“语义成功但元数据参照错误”的样本直接进入 accepted pairs。

### 5.4 GroundingDINO 当前不可用

当前服务器没有官方格式 GroundingDINO `.pth` checkpoint。因此：

```text
--grounder groundingdino 不能用
--grounder auto 在 fcbf033 上会遇到 HF checkpoint 格式时 fallback 到 Florence-2
```

如果服务器没有拉到 `fcbf033`，仍应显式使用：

```text
--grounder florence2
```

### 5.5 复杂编辑路线暂时不应该继续烧 VACE

这些任务已经明确不适合当前默认 VACE route：

- short sleeve / patterned shirt -> open black long-sleeved jacket
- chair -> stool，尤其有人坐着时
- cup/mug 小物体 replacement
- laptop/tablet 且屏幕有文字
- 多主体或多镜头背景替换
- Qwen-Image 生成的 1:1 方形背景参考图用于 16:9 background change

## 6. 当前推荐下一步

### 6.1 服务器下一步执行方向

先不要扩大跑批，也不要再跑 plain masked background replacement。服务器下一步应把 `mannul7` 的后处理元数据修正为可验收格式，然后只做 2-3 条 composite-first-frame 小批量复现。

第一优先级：

```text
修正 duration_metrics：
reference_for_vace / composite src_video / src_mask / raw target / remux target 必须比较同一 VACE 输入段
写入 generation.duration_metrics.duration_gate.passed=true
不要用原始 15 秒 reference 与 3 秒 target 比
```

只有这个修正完成后，`mannul7` 才能作为 accepted synthetic pair 的候选。

第二优先级：继续验证 composite-first-frame route：

```text
plan: ef8f2818
task: woman background -> futuristic laboratory
目的: 复现 mannul7 的真实背景替换，不再回退到 plain masked blue overlay
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
以及后续 duration gate 强制验收修复
```

然后跑：

```text
python -m unittest tests.test_scripts tests.test_composed_data -v
```

如果测试通过，再重新生成 ef8f2818 的 plan。不要复用 mannul5/mannul6 的旧 plan，也不要复用旧 review bundle。

新 plan 必须人工/脚本确认：

```text
target_prompt 不含 sunlit room / window / door / original room / source background
preserve_tokens 不含 sunlit room / lighting / layout / room / window / door
preserve_regions 不含 window / door / room / wall / background
VACE prompt 是最终状态描述，不是 “in a sunlit room with lab background”
```

只有这些都通过，也不要直接走 plain VACE。background replacement 必须生成 composite first frame：

```text
frame0 = same woman foreground + target futuristic lab background composite
frame0 mask = all black / retain full composite first frame
frames 1..N src_video = original woman foreground + gray 127 background
frames 1..N src_mask = foreground black preserve, background white generate
```

然后 VACE generation metadata 必须记录：

```text
background_replace_route=vace_bg_replace_composite_first_frame_mv2v
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
3. `mannul7` 已经证明 `vace_bg_replace_composite_first_frame_mv2v` 能解决 blue overlay；现在要修 duration gate 元数据并做 2-3 条复现。
4. 在现有 35 个 plan 里继续寻找能稳定生成 mask 的少数样本。
5. 如果 Florence-2 + SAM2.1 对真实视频仍然全失败，切换到更简单的 synthetic route：
   - deterministic audio route
   - existing large object / robot / vehicle attribute edit
   - animation / static-like video local color edit
6. 等 mask route 有稳定通过样本，再恢复非背景替换类 VACE smoke。
