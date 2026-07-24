# Frontier Landscape: Audio-Aware Composed Video Retrieval

## 1. Scope

Composed video retrieval (CVR) retrieves a target video from a gallery using a
reference video and a text instruction that describes the desired change.
The emerging audio-aware branch asks whether speech, music, and environmental
sound should also influence the composition. The most relevant frontier now
contains three related but distinct problems:

1. General CVR over visual changes.
2. Audio-visual CVR in which audio and vision are both first-class evidence.
3. Diagnostic evaluation of whether a model distinguishes the pre-edit source
   from a post-edit target, especially when visual context is preserved.

Results from different benchmarks are not directly comparable because gallery
size, source inclusion, query type, and human-validation policy differ.

## 2. Key Methods and Benchmarks

### CoVR and CoVR-2

CoVR established web-scale composed video retrieval and automatic triplet
construction; CoVR-2 expanded the construction methodology. They remain the
main predecessors for scalable query-reference-target mining, but their task
focus is predominantly visual modification.

Primary sources:

- [CoVR, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28334)
- [CoVR-2, TPAMI 2024](https://ieeexplore.ieee.org/document/10685001)

### EgoCVR

EgoCVR emphasizes temporally fine-grained egocentric changes and evaluates
both global and local retrieval conditions. Its important methodological
lesson is that aggregate gallery recall should be supplemented with
difficulty-controlled diagnostics.

Primary source:

- [EgoCVR, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5363_ECCV_2024_paper.php)

### OmniCVR and AudioVLM2Vec

OmniCVR is the closest large-scale competitor. It treats vision, audio, and
text as first-class modalities, includes source videos and source-local
distractors in the candidate gallery, and uses automated construction followed
by model and human validation. The benchmark contains about 50,000 triplets
and a 5,000-query human-validated test set. On its audio-centric split,
AudioVLM2Vec reports R@1 of 77.2, compared with 12.4 for VLM2Vec and 13.6 for
OmniEmbed. Removing the source video from the query drops AudioVLM2Vec from
77.2 to 28.1 R@1, showing that the source is useful query context. This
ablation is different from masking the source as a gallery candidate.

Primary sources:

- [OmniCVR, ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010075)
- [OmniCVR paper and AudioVLM2Vec tables](https://openreview.net/pdf?id=KxxR7emO5K)

### CoVA

CoVA introduces AV-Comp and AVT Compositional Fusion for audio-visual changes.
Its central lesson is that direct trimodal fusion is not automatically better:
the paper reports R@1 of 25.9 for direct trimodal fusion, below 28.8 for V+T,
while selective fusion reaches 31.4. This makes modality selection, rather than
modality presence alone, a current design frontier.

Primary source:

- [CoVA, ICASSP 2026](https://arxiv.org/abs/2601.22508)

### AVIGATE and SAVE

AVIGATE demonstrates in video-text retrieval that blindly adding audio can
degrade representations and introduces gated fusion to suppress irrelevant
sound. SAVE extends this trend with a speech-aware branch and early
vision-audio alignment, improving SumR over AVIGATE on five video-text
retrieval benchmarks. These methods establish that negative audio gains are
not surprising; the important question is whether a model can identify when
audio is useful.

Primary sources:

- [AVIGATE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Learning_Audio-guided_Video_Representation_with_Gated_Attention_for_Video-Text_Retrieval_CVPR_2025_paper.html)
- [SAVE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_SAVE_Speech-Aware_Video_Representation_Learning_for_Video-Text_Retrieval_CVPR_2026_paper.html)

### Reasoning-First and Training-Free CVR

The 2026 frontier is moving beyond a single fused embedding toward explicit
candidate reasoning and reranking. CoVR-R introduces a reason-aware benchmark
with implicit after-effects and challenging distractors. A training-free
DINOv3 plus Qwen3-VL pipeline reports 48.78 R@1 on the CoVR-R challenge test
set, with DINO retrieval at 25.34, instruction reranking at 35.27, and
thinking-model refinement at 48.78. R3 similarly combines reasoning-guided
recall with pairwise reranking.

Primary sources:

- [CoVR-R](https://arxiv.org/abs/2603.20190)
- [Training-Free CVR with Video-LLM Reasoning](https://arxiv.org/abs/2606.02321)
- [R3: Reasoning-Guided Recalling and Reranking](https://arxiv.org/abs/2606.01113)

## 3. Current Pareto Frontier

There is no single comparable leaderboard across these tasks. The current
frontier is better described by separate operating points:

| Setting | Representative result | What it establishes |
|---|---:|---|
| OmniCVR audio-centric retrieval | AudioVLM2Vec 77.2 R@1 | Strong task-specific audio-as-text retrieval |
| CoVR-R challenge | Training-free reasoning 48.78 R@1 | Strong top-rank performance without task training |
| CoVA audio-visual composition | Selective fusion 31.4 R@1 | Audio routing beats direct trimodal fusion |
| Efficient generic embedding | E5-Omni, ImageBind, VLM2Vec families | Scalable baselines, but not necessarily task-specialized |

The practical Pareto trade-off is therefore among task-specific accuracy,
training-free transfer, scalable single-vector retrieval, and expensive
pairwise reasoning.

## 4. Emerging Trends

1. Audio is treated as conditional evidence. Gating, routing, or textualization
   increasingly replaces unconditional embedding addition.
2. Candidate reranking is becoming more important. Strong visual recall can
   find a plausible neighborhood, while a multimodal reasoner resolves the
   final ordering.
3. Benchmark papers are expected to document human validation, provenance,
   source overlap, and construction yield, not only final sample counts.
4. Aggregate recall is giving way to targeted diagnostics, including local
   galleries, hard-negative subsets, modality ablations, and pairwise
   source-target decisions.
5. Exact source inclusion creates a new evaluation tension: it is a useful
   counterfactual negative, but it may also trigger identity matching that is
   separable from semantic edit reasoning.

## 5. Lightweight Baselines That Set a High Bar

ImageBind arithmetic and frozen omni-modal embeddings are inexpensive
controls, but their negative audio gains should not be interpreted as the
frontier. More demanding lightweight baselines include validation-selected
gating, audio-as-text representations, and frozen visual recall followed by
limited top-K multimodal reranking. A new diagnostic benchmark should show
that its finding survives at least one strong task-specific or reasoning-based
system, or explicitly bound its claim to the evaluated model families.

## 6. Open Problems

1. Separating exact identity anchoring from true pre-edit versus post-edit
   directional reasoning.
2. Constructing verified same-source or near-duplicate negatives at scale.
3. Measuring automatic-review reliability with independent human raters.
4. Preventing speech and transcript shortcuts without removing realistic
   speech-centered use cases.
5. Comparing modality utility under fixed galleries and fixed score matrices.
6. Designing fusion that improves target-over-source ordering without harming
   global gallery ranking.
7. Standardizing reference-specific metrics so results transfer across
   Audio-CVR, OmniCVR, CoVA, and future benchmarks.

