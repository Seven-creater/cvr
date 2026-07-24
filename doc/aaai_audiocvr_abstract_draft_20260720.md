# Audio-CVR AAAI 摘要定稿候选

> 论文主线：Audio-CVR 任务、自动多模态构造方法与 reference-aware benchmark；adapter 仅作为基线。
>
> 结果更新：2026-07-21；主证据来自冻结的 Audio-CVR 150-query test，外部诊断来自 OmniCVR 995-query test。

## 暂定标题

**Audio-CVR: Automatic Curation and Reference-Aware Evaluation for Directional Audio Composed Video Retrieval**

中文：**Audio-CVR：面向方向性声音组合视频检索的自动构造与参考感知评测**

## 中文摘要

组合视频检索将参考视频与修改文本组合起来，从候选库中检索目标视频。OmniCVR 已将 source video 纳入检索候选，但 aggregate recall 仍无法单独揭示模型是否把未修改的 source 排在已编辑 target 前面。本文提出 Audio-CVR，一个研究视觉语境保持条件下方向性声音变化的 audio-primary 诊断 benchmark。数据对经过专门筛选，使 source 不满足 edit、target 满足 edit，并使静音视频本身不足以确定 target。我们提出多阶段自动构造流程，包括 source-aware 配对、仅音频变化分析与 edit 生成、方向验证、静音视频捷径拒绝、音视频一致性检查、ASR 捷径筛查、抽样重复审核和 source-disjoint 切分。我们将 source 形式化为 query-specific counterfactual，并通过 target-over-source 准确率、target-source 得分差、精确 source masking 和 reference-specific 错误归因诊断 source-target confusion。在包含150个 query、固定1,000-item gallery 的模型复核非语音测试集上，冻结 E5-Omni backbone 上的轻量适配器在加入 audio 后，五个 seed 的平均 R@1 从11.3%提升到22.9%（增益11.6个百分点；95% CI：6.3--17.1；Holm 校正后 p<0.001），target-over-source 准确率也从11.3%提升到23.2%。只 mask source，R@1 就从22.9%升至99.5%。在995个 OmniCVR audio-centered query 上，同一干预使 Adapter 的 R@1 提升14.1个百分点，使冻结 E5-Omni 的 R@1 提升17.4个百分点。结果表明，aggregate retrieval metrics 会掩盖 source-target 的方向性失败，因此需要显式的 source-specific diagnostics。

## English Abstract

Composed video retrieval combines a reference video with modification text to retrieve a target from a gallery. OmniCVR includes the source video among retrieval candidates, but aggregate recall does not isolate whether a model ranks the unchanged source above the edited target. We introduce **Audio-CVR**, an audio-primary diagnostic benchmark for directional sound changes under preserved visual context. Its pairs are curated so that the source does not satisfy the edit, the target does, and muted video alone is insufficient to identify the target. We develop a multi-stage automatic curation pipeline with source-aware pairing, audio-only change analysis and edit generation, directional verification, muted-video shortcut rejection, audiovisual consistency checking, ASR-shortcut screening, sampled repeat auditing, and source-disjoint splitting. We formalize the source as a query-specific counterfactual and diagnose source-target confusion using target-over-source accuracy, target-source score margins, exact source masking, and reference-specific error attribution. On a model-verified, non-speech test set of 150 queries with a fixed 1,000-item gallery, adding audio to a frozen E5-Omni backbone with a lightweight adapter improves mean R@1 across five seeds from 11.3% to 22.9% (difference: 11.6 points; 95% CI: 6.3--17.1; Holm-adjusted p<0.001) and target-over-source accuracy from 11.3% to 23.2%. Masking only the source raises R@1 from 22.9% to 99.5%. On 995 OmniCVR audio-centered queries, the same intervention raises R@1 by 14.1 points for the adapter and 17.4 points for frozen E5-Omni. These results show that aggregate retrieval metrics can hide source-target directional failures and that explicit source-specific diagnostics are needed.

## 数字使用规则

- 主结果只使用 `Forward-only` adapter；verified inverse augmentation 是负消融。
- 正式测试：150 条，`sound_event=120`、`music=30`，不包含 speech。
- 数据是 `automatically curated and model-verified`，不是 `human-validated`。
- 测试集 90% 来自 avatar，当前没有 strict local negative；这两点必须在 limitations 中披露。
- 旧 30-query、37-query、68-query 结果只保留为开发历史，不进入摘要或论文主表。
- OmniCVR 只作为 reference-aware protocol 的跨 benchmark 诊断，不用于声称 audio R@1 显著提升或 adapter 跨域泛化。

## 证据路径

```text
C:/Users/29785/Desktop/research/runs/fewshot_bidir_results_final_20260721/fewshot_bidir_results
statistics_forward_only/test_main_comparison.md
statistics_forward_only/paired_comparisons.md
statistics_variant_comparison/paired_comparisons.md

C:/Users/29785/Desktop/research/runs/omnicvr_reference_diagnostics_paper_results_20260721
statistics/test_main_comparison.md
statistics/paired_comparisons.json
statistics/audit.json
```
