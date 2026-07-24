# Audio-CVR Final Experiment Audit

Verified: 2026-07-24

This is the single source of truth for the paper. It inventories every completed
experiment that affects a manuscript claim and separates final evidence from
superseded development results.

## Frozen Evaluation Set

| Item | Final value |
|---|---:|
| Full1000 queries | 1,000 |
| Gallery | 2,000 items: 1,000 targets + 1,000 unchanged references |
| Full1000 SHA256 | `70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e` |
| Duplicate / leakage / missing-media violations | 0 / 0 / 0 |
| Scope | Non-speech sound events and music |
| Automatic repeat review | 78/78 completed; 79.5% decision agreement |
| Blinded human audit | 50 unique queries; 34 valid (68.0%) |

The human audit is partial and single-rater. Core150 contributed 38 audited
queries with 24 valid (63.2%); 10 of 12 audited supplemental queries were valid
(83.3%). The benchmark must be described as automatically curated and
model-verified, with a partial blinded human audit, not as human validated.

## Primary Full1000 Results

### Exact Reference And Masking

| Model | Mode | With-ref R@1 | Masked-ref R@1 | Reference drop | Target beats ref |
|---|---|---:|---:|---:|---:|
| Frozen E5-Omni | V+T | 0.70 | 99.70 | 99.00 | 0.60 |
| Frozen E5-Omni | V+A+T | 0.30 | 99.70 | 99.40 | 0.30 |
| E5 adapter, 5 seeds | V+T | 5.94+/-0.99 | 98.28+/-0.98 | 92.34 | 6.06 |
| E5 adapter, 5 seeds | V+A+T | 12.78+/-2.55 | 97.26+/-1.91 | 84.48 | 13.20 |
| ImageBind, zero-shot | V+T | 11.70 | 99.30 | 87.60 | 10.80 |
| ImageBind, zero-shot | V+A+T | 2.50 | 98.50 | 96.00 | 2.50 |
| OmniEmbed-MultiVent, zero-shot | V+T | 1.00 | 99.20 | 98.20 | 0.10 |
| OmniEmbed-MultiVent, zero-shot | V+A+T | 0.00 | 99.20 | 99.20 | 0.00 |

For the adapted E5 model, 4,317 of 4,361 exact-condition top-1 errors across
five seeds select the query's own reference (98.99%). This is the primary
reference-specific error-attribution result.

### Audio Effects

| Model | V+T R@1 | V+A+T R@1 | Difference | 95% paired CI | Holm p |
|---|---:|---:|---:|---:|---:|
| E5 adapter | 5.94 | 12.78 | +6.84 | [5.16, 8.52] | 0.00180 |
| ImageBind | 11.70 | 2.50 | -9.20 | [-11.30, -7.10] | 0.00090 |
| OmniEmbed-MultiVent | 1.00 | 0.00 | -1.00 | [-1.70, -0.40] | 0.00405 |

The supported claim is conditional audio utility. Task adaptation lets E5 use
audio for directional discrimination, whereas fixed or zero-shot audiovisual
fusion can treat audio as interference. The paper must not claim universal
audio benefit.

## Reference Identity Perturbation

Only the current query's own-reference gallery item changes. The target,
remaining 1,999 candidates, query input, index, and evaluation code remain
fixed.

| Model / mode | Exact drop | Transcoded drop | Temporal drop | Spatial drop |
|---|---:|---:|---:|---:|
| Frozen E5 V+T | 99.00 | 94.30 | 26.10 | 44.90 |
| Frozen E5 V+A+T | 99.40 | 97.30 | 30.80 | 60.90 |
| E5 adapter V+T | 92.34 | 84.14 | 32.62 | 41.52 |
| E5 adapter V+A+T | 84.48 | 78.94 | 42.90 | 51.04 |
| ImageBind V+T | 87.60 | 80.60 | 58.50 | 27.00 |
| ImageBind V+A+T | 96.00 | 94.00 | 84.00 | 81.50 |
| OmniEmbed V+T | 98.20 | 95.10 | 60.50 | 45.60 |
| OmniEmbed V+A+T | 99.20 | 97.10 | 60.00 | 45.80 |

Exact identity explains a substantial part of the collapse, but perturbations
that reduce frame-level identity still leave 26.1--84.0 point drops. The final
mechanistic wording is therefore **identity-sensitive pre-edit source
anchoring**, not pure semantic edit-direction failure.

## Cross-Benchmark Evidence

On 1,000 OmniCVR audio-centered queries, OmniEmbed-MultiVent obtains:

| Mode | With-source R@1 | Masked-source R@1 | Difference |
|---|---:|---:|---:|
| V+T | 0.10 | 14.50 | +14.40 |
| V+A+T | 0.00 | 14.00 | +14.00 |

One unrelated OmniCVR random-gallery video fails media encoding in both modes.
No target or own reference is affected. The shared effective gallery has 1,999
items. This experiment supports cross-benchmark source anchoring, not an audio
gain claim.

## Sensitivity And Diagnostic Experiments

### Human-Valid Subset

The following is descriptive because only 34 audited queries are valid:

| Model / mode | With-ref R@1 | Masked-ref R@1 |
|---|---:|---:|
| Frozen E5 V+T | 2.94 | 100.00 |
| Frozen E5 V+A+T | 2.94 | 100.00 |
| E5 adapter V+T | 11.18 | 97.06 |
| E5 adapter V+A+T | 17.06 | 93.53 |
| ImageBind V+T | 23.53 | 100.00 |
| ImageBind V+A+T | 5.88 | 94.12 |
| OmniEmbed V+T | 8.82 | 100.00 |
| OmniEmbed V+A+T | 0.00 | 100.00 |

The reference effect survives the partial human-valid subset, but the sample is
too small and selectively observed for a primary inferential claim.

### Seven-Mode Ablation

The adapted E5 R@1 values are: T-only 1.22, V-only 2.22, A-only 12.88,
V+T 5.94, A+T 13.54, V+A 9.36, and V+A+T 12.78. ImageBind has the
corresponding fixed-fusion seven-mode evaluation in the supplementary evidence
package. The main paper uses only V+T and V+A+T to keep the causal comparison
clear.

### Bidirectional Training

On Core150, forward-only training reaches 22.93+/-5.31 R@1 and verified
forward-plus-inverse training reaches 18.93+/-2.72. The difference is -4.00
points, 95% CI [-7.60, -0.53], Holm p=0.0373. The score margin improves by
0.00343, but inverse augmentation does not improve top-1 retrieval. It is a
negative ablation, not a contribution claim.

### Historical Hard-Negative Control

In a historical 128-query mixed gallery, the own reference outranks all three
typed hard negatives in 100% of comparisons, same-source local candidates in
83.3% of 36 comparisons, and random distractors in 99.94% of 33,536
comparisons. Local coverage is only nine queries, so this is diagnostic support
rather than a final benchmark result.

## Supplementary-Only Experiments

- Core150 is retained as a manually inspected sensitivity subset and historical
  development benchmark, not the primary result.
- The Audio-as-Text VLM2Vec experiment is a non-official reproduction and must
  not be presented as AudioVLM2Vec. OmniEmbed-MultiVent supersedes it as the
  strong trained retrieval baseline.
- The earlier 995-query E5 OmniCVR diagnostic is valid but superseded in the
  main narrative by the final 1,000-query OmniEmbed cross-benchmark result.
- Pre516 and intermediate Test1000 construction runs are provenance records,
  not separate evaluation datasets.

## Superseded Values

Do not use these as final Full1000 headline numbers:

- Core150 E5 adapter V+T 11.33 and V+A+T 22.93.
- Core150 masked-reference R@1 99.47.
- Earlier E5 OmniCVR reference drop 14.1 points on 995 queries.
- Earlier Holm-adjusted p=0.000250 computed before the expanded planned family.
- The two-author 10-query qualitative spot-check description.

## Final Claim Boundary

The paper can claim:

1. exact own-reference inclusion exposes severe source anchoring across frozen,
   adapted, independent, and retrieval-trained models;
2. identity perturbations reduce but do not eliminate the effect;
3. audio improves directional retrieval after task-specific E5 adaptation but
   can hurt zero-shot fusion;
4. the automatically curated benchmark has complete machine audit and a
   transparent, partial single-rater human audit.

The paper cannot claim a fully human-validated gold set, universal audio
benefit, pure semantic direction reasoning, state-of-the-art retrieval, or a
new model architecture.
