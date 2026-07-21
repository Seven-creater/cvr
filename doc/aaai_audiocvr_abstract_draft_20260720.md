# Audio-CVR AAAI 摘要定稿候选

> 论文主线：Audio-CVR 任务、自动多模态构造方法与 reference-aware benchmark；adapter 仅作为基线。
>
> 结果更新：2026-07-21；正式证据来自冻结的 150-query test、1,000-video gallery 和 5 个随机种子。

## 暂定标题

**Audio-CVR: Automatic Curation and Reference-Aware Evaluation for Directional Audio Composed Video Retrieval**

中文：**Audio-CVR：面向方向性声音组合视频检索的自动构造与参考感知评测**

## 中文摘要

组合视频检索将参考视频与修改文本组合起来检索目标视频，但常规评测可能混淆模型对修改方向的理解与视觉相似性、语音转写或简单候选库带来的捷径。本文提出 Audio-CVR，研究视觉语境保持条件下的方向性声音修改：reference 不满足声音 edit，target 必须满足 edit，且仅凭静音画面不能确定 target。我们提出多阶段多模态自动构造流程，包括 source-aware 片段配对、仅音频差异分析与 edit 生成、方向验证、静音视频捷径拒绝、完整音视频一致性检查、anti-ASR 过滤、重复审核和 source-disjoint 切分。进一步地，我们建立 reference-aware 评测协议，将未修改的 reference 作为强制反事实负例，并结合类型化困难负例和随机干扰项，通过有无 reference 与七种模态组合测量检索召回、target-over-reference 准确率、方向性得分差和 reference-induced R@1 drop。我们在自动构造、模型复核的非语音测试集上评估冻结 E5-Omni-7B 与轻量低秩适配器。对于 150 个 source-disjoint query、固定 1,000-video gallery 和 5 个随机种子，启用 audio 将 R@1 从 11.3% 提升到 22.9%，配对增益为 11.6 个百分点（95% CI：6.3--17.1，Holm 校正后 p<0.001），并将 target-over-reference 准确率从 11.3% 提升到 23.2%。移除 unchanged reference 后 R@1 达到 99.5%，表明普通候选库会显著掩盖编辑方向判别的难度。Audio-CVR 因而把方向性声音利用而非一般视频相似性转化为可显式测量的检索要求。

## English Abstract

Composed video retrieval combines a reference video and modification text to retrieve a target, yet conventional evaluation can conflate edit-direction understanding with shortcuts from visual similarity, speech transcripts, or easy candidate pools. We introduce **Audio-CVR**, which studies directional sound modifications under preserved visual context: the reference must not satisfy the edit, the target must satisfy it, and muted video alone must not identify the target. We develop a multi-stage multimodal curation pipeline comprising source-aware clip pairing, audio-only difference analysis and edit generation, directional verification, muted-video shortcut rejection, full audiovisual consistency checking, anti-ASR filtering, repeat auditing, and source-disjoint splitting. We further establish a reference-aware protocol that treats the unchanged reference as a mandatory counterfactual negative and combines it with typed hard negatives and random distractors. Controlled reference and seven-mode modality ablations measure recall, target-over-reference accuracy, directional score margins, and reference-induced R@1 drop. We evaluate a frozen E5-Omni-7B backbone with a lightweight low-rank adapter on an automatically curated and model-verified non-speech benchmark. Across 150 source-disjoint queries, fixed 1,000-video galleries, and five seeds, enabling audio raises R@1 from 11.3% to 22.9%, a paired gain of 11.6 points (95% CI: 6.3--17.1; Holm-adjusted p<0.001), and improves target-over-reference accuracy from 11.3% to 23.2%. Removing the unchanged reference raises R@1 to 99.5%, showing that conventional galleries can conceal the difficulty of edit-direction discrimination. Audio-CVR therefore makes directional audio use, rather than generic video similarity, an explicit and measurable retrieval requirement.

## 数字使用规则

- 主结果只使用 `Forward-only` adapter；verified inverse augmentation 是负消融。
- 正式测试：150 条，`sound_event=120`、`music=30`，不包含 speech。
- 数据是 `automatically curated and model-verified`，不是 `human-validated`。
- 测试集 90% 来自 avatar，当前没有 strict local negative；这两点必须在 limitations 中披露。
- 旧 30-query、37-query、68-query 结果只保留为开发历史，不进入摘要或论文主表。

## 证据路径

```text
C:/Users/29785/Desktop/research/runs/fewshot_bidir_results_final_20260721/fewshot_bidir_results
statistics_forward_only/test_main_comparison.md
statistics_forward_only/paired_comparisons.md
statistics_variant_comparison/paired_comparisons.md
```
