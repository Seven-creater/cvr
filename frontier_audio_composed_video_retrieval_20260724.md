# Frontier: Audio-Aware Composed Video Retrieval

Verified: 2026-07-24

## Scope

Audio-aware composed video retrieval combines a source video and a modification
instruction to retrieve a target whose visual context and audiovisual state
satisfy the requested change. The closest research lies at the intersection of
composed video retrieval, omni-modal embedding, audio-guided video retrieval,
and diagnostic benchmark design.

## Current Methods And Benchmarks

| Work | Venue | Core idea | Frontier signal |
|---|---|---|---|
| CoVR | AAAI 2024 | Web-video triplets generated from captions and language-model modifications | Established large-scale composed video retrieval |
| EgoCVR | ECCV 2024 | Egocentric global/local retrieval with fine-grained temporal changes | Established local-gallery diagnostic evaluation |
| ReTrack | AAAI 2026 | Directional anchor calibration reduces reference-oriented bias in composed features | Closest algorithmic treatment of reference bias; distinct from a paired diagnostic protocol |
| Dense-WebVid-CoVR | 2025 preprint | 1.6 million triplets with dense modification text and a manually verified test set | Reports 71.3 R@1 and sets a high annotation-quality bar |
| AVIGATE | CVPR 2025 | Gated audio fusion suppresses irrelevant audio in video-text retrieval | Shows that indiscriminate audio fusion can be harmful |
| OmniCVR / AudioVLM2Vec | ICLR 2026 | Vision, audio, and text composed retrieval; audio is converted into explicit semantics | Reports 77.2 R@1 on audio-centric queries; removing source visual context reduces it to 28.1 |
| CoVA | ICASSP 2026 | Audio-visual composed retrieval with learned audiovisual fusion | Direct evidence that task-trained selective fusion outperforms blind audio injection |
| OmniRet | CVPR 2026 | Universal text/vision/audio retriever with resampling and Attention Sliced Wasserstein Pooling | Trained on about 6 million pairs from 30 datasets; introduces a human-evaluated audio-centric composed benchmark |
| Training-Free CoVR Reasoning | CVPR 2026 challenge report | DINOv3 recall followed by Qwen3-VL reranking and reasoning | Reaches 48.78 R@1 without task training, showing the strength of explicit pairwise reasoning |

Primary sources:

- [OmniCVR, ICLR 2026](https://openreview.net/pdf?id=KxxR7emO5K)
- [ReTrack, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39507)
- [OmniRet, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huynh_Efficient_and_High-Fidelity_Omni_Modality_Retrieval_CVPR_2026_paper.html)
- [AVIGATE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Learning_Audio-guided_Video_Representation_with_Gated_Attention_for_Video-Text_Retrieval_CVPR_2025_paper.html)
- [Dense-WebVid-CoVR](https://arxiv.org/abs/2508.14039)
- [Training-Free Composed Video Retrieval](https://arxiv.org/abs/2606.02321)

## Current Pareto Frontier

The field is moving in two directions. Large task-trained embedding systems
such as AudioVLM2Vec and OmniRet pursue high recall through explicit audio
semantics, selective fusion, and millions of training pairs. Training-free
systems instead use strong visual recall followed by candidate-level VLM
reasoning, reaching competitive first-rank performance without benchmark
training. A diagnostic benchmark must therefore test both embedding and
pairwise reasoning systems; generic zero-shot embeddings alone no longer span
the frontier.

## Emerging Trends

First, audio is increasingly treated as selectively useful rather than appended
unconditionally. AVIGATE, CoVA, and AudioVLM2Vec all motivate filtering,
semantic conversion, or task adaptation.

Second, candidate-level reasoning is becoming a credible alternative to a
single composed embedding. This matters when the main error is ordering two
near-identical candidates rather than recalling the correct semantic region.

Third, benchmark papers are expected to document human quality control and
hard-gallery construction. Dense-WebVid-CoVR uses a fully manually verified
test set, while OmniRet reports human evaluation for its audio-centric
benchmark.

## Open Problems

The central unresolved tension is how to separate semantic edit-direction
reasoning from source-instance identity. Including the exact source in the
gallery is diagnostically useful, but may create a self-match shortcut. A
convincing protocol needs exact masking, identity perturbations, and
same-source yet non-identical pre-edit controls.

Another open problem is audio utility. Audio can provide the only evidence for
a requested change, yet it can also reinforce the source identity or act as
noise. Model comparisons should distinguish task-trained selective fusion from
fixed additive fusion.

Finally, high-quality audiovisual triplets remain expensive. Automatically
curated benchmarks need enough blinded human auditing to estimate validity,
not only same-model repeat stability.
