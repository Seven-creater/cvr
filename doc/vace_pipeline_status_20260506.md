# VACE Pipeline 当前状态与问题记录

Last updated: 2026-05-06

## 1. 当前结论

当前视觉合成路线还没有稳定进入“批量生成目标视频”的阶段。真正的瓶颈已经从 VACE 本身前移到 **mask 生成与 mask 验收**：

```text
Omni 计划生成完成 -> 候选 plan 筛选完成 -> mask 生成/验收失败 -> 不进入 VACE
```

这不是坏事。现在的 gate 能及时拦截坏 mask，避免继续烧 VACE GPU 生成假阳性样本。当前应该继续先把 mask 路线跑通，而不是降低阈值硬跑 VACE。

官方契约仍然是：

```text
src_video + src_mask + optional src_ref_images + target prompt -> VACE target
```

其中 `src_mask` 白区是生成区域，黑区是保留区域；`src_video` 的编辑区域需要置灰为 127；prompt 应描述目标视频，而不是写操作指令。

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
GitHub 已推送 HEAD: ffb8a3d Prefer torch GroundingDINO checkpoints
本地额外 commit: 2d1fc41 Fallback from HF GroundingDINO checkpoints
状态: 本地 ahead 1，因本机 DNS 无法解析 github.com，2d1fc41 暂未推送
```

也就是说，服务器目前能通过 GitHub 拉到 `ffb8a3d`，但还拉不到本地最新的 `2d1fc41`。

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
```

本地未推送修复：

```text
2d1fc41 Fallback from HF GroundingDINO checkpoints
```

`2d1fc41` 的作用是：`--grounder auto` 遇到 HF 格式 GroundingDINO checkpoint 时自动退到 Florence-2。但因为本机暂时无法解析 GitHub，这个 commit 还没有推到远端。

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

但这条样本还不能标为 accepted，原因是后处理脚本在生成 `duration_metrics.json` / `post_vace_verdict.json` 之前崩溃：

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

本地已修复并通过测试，但该 commit 尚未推送到 GitHub。视觉上是否成功仍需要人工查看 `*_with_ref_audio.mp4`：

- 背景是否真变成 futuristic laboratory。
- woman 身份、脸、动作是否保留。
- 是否有糊脸、换人、边缘蓝化、背景无变化等问题。

### 5.3 GroundingDINO 当前不可用

当前服务器没有官方格式 GroundingDINO `.pth` checkpoint。因此：

```text
--grounder groundingdino 不能用
--grounder auto 在 ffb8a3d 上仍可能碰到 HF checkpoint 格式问题
```

在 `2d1fc41` 推送前，服务器应显式使用：

```text
--grounder florence2
```

### 5.4 复杂编辑路线暂时不应该继续烧 VACE

这些任务已经明确不适合当前默认 VACE route：

- short sleeve / patterned shirt -> open black long-sleeved jacket
- chair -> stool，尤其有人坐着时
- cup/mug 小物体 replacement
- laptop/tablet 且屏幕有文字
- 多主体或多镜头背景替换
- Qwen-Image 生成的 1:1 方形背景参考图用于 16:9 background change

## 6. 当前推荐下一步

### 6.1 服务器今晚应执行的方向

不要跑 VACE，只重跑 mask smoke。

如果服务器只能拉到 `ffb8a3d`：

```text
显式使用 --grounder florence2
不要使用 --grounder auto
不要使用 --grounder groundingdino
```

如果之后本地 `2d1fc41` 成功推送到 GitHub，服务器再拉最新后可以恢复：

```text
--grounder auto
```

因为那时 auto 会自动识别 HF checkpoint 并退到 Florence-2。

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

1. 在现有 35 个 plan 里找到能稳定生成 mask 的少数样本。
2. 如果 Florence-2 + SAM2.1 对真实视频仍然全失败，切换到更简单的 synthetic route：
   - deterministic audio route
   - existing large object / robot / vehicle attribute edit
   - animation / static-like video local color edit
3. 等 mask route 有稳定通过样本，再恢复 VACE smoke。
