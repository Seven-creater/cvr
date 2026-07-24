# Audio-CVR Final Statistical Audit

Verified: 2026-07-24

## Design

- Full1000 contains 1,000 fixed queries and a shared 2,000-item gallery.
- Adapted E5 uses seeds 13, 23, 42, 71, and 101.
- ImageBind and OmniEmbed are deterministic zero-shot evaluations.
- Exact and masked-reference results share one score matrix; masking changes
  only the own-reference score.
- The independent inferential unit is the query. E5 differences are averaged
  across seeds within query before paired resampling.
- Confidence intervals use 20,000 paired bootstrap samples. Randomization tests
  use 20,000 paired sign permutations.
- Holm correction uses the final expanded planned comparison family, including
  the reference-perturbation conditions. Earlier adjusted values such as
  `0.000250` are superseded.

## Exact-Condition Audio Effects

| Model | V+T R@1 | V+A+T R@1 | Difference | 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|
| E5 adapter | 5.94+/-0.99 | 12.78+/-2.55 | +6.84 | [5.16, 8.52] | 0.001800 |
| ImageBind | 11.70 | 2.50 | -9.20 | [-11.30, -7.10] | 0.000900 |
| OmniEmbed | 1.00 | 0.00 | -1.00 | [-1.70, -0.40] | 0.004050 |

The E5 result supports a model-specific gain. The ImageBind and OmniEmbed
results reject a universal audio-benefit claim.

## Reference Interventions

| Model/mode | Exact drop | Transcoded | Temporal | Spatial |
|---|---:|---:|---:|---:|
| Frozen E5 V+T | 99.00 | 94.30 | 26.10 | 44.90 |
| Frozen E5 V+A+T | 99.40 | 97.30 | 30.80 | 60.90 |
| E5 adapter V+T | 92.34 | 84.14 | 32.62 | 41.52 |
| E5 adapter V+A+T | 84.48 | 78.94 | 42.90 | 51.04 |
| ImageBind V+T | 87.60 | 80.60 | 58.50 | 27.00 |
| ImageBind V+A+T | 96.00 | 94.00 | 84.00 | 81.50 |
| OmniEmbed V+T | 98.20 | 95.10 | 60.50 | 45.60 |
| OmniEmbed V+A+T | 99.20 | 97.10 | 60.00 | 45.80 |

All drops are absolute R@1 percentage points. Perturbation comparisons are
query paired. They establish a residual effect after identity reduction, but
do not identify a purely semantic mechanism.

## Error Attribution

- Adapted E5 sends 4,317 of 4,361 exact-condition top-1 errors to the own
  reference (98.99%).
- The adapted E5 target beats each designated visual, audio, and ASR hard
  negative in 100% of comparisons, but beats its own reference in only 13.2%.
- OmniEmbed source masking increases OmniCVR R@1 by 14.4 points for V+T and
  14.0 points for V+A+T.

## Human Audit

The human audit is descriptive:

| Quantity | Value |
|---|---:|
| Displayed items | 51 |
| Unique queries | 50 |
| Overall valid | 34 (68.0%) |
| Core150 valid | 24/38 (63.2%), Wilson 95% CI [47.3, 76.6] |
| Supplemental valid | 10/12 (83.3%), Wilson 95% CI [55.2, 95.3] |
| Completed paired repeats | 0 |

Because the planned audit was stopped early, its 68.0% value is not presented
as an unbiased Full1000 prevalence estimate. Human-valid-subset retrieval
results are sensitivity analyses only.

## Negative Bidirectional Ablation

Core150 forward-only R@1 is 22.93+/-5.31; forward-plus-verified-inverse R@1 is
18.93+/-2.72. The -4.00 point difference has 95% CI [-7.60, -0.53] and Holm
p=0.0373. The target-reference margin improves by 0.00343, but top-1 retrieval
does not. No claim of bidirectional-training improvement is permitted.

## Integrity

- Full1000 SHA256 is
  `70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e`.
- Membership and order are unchanged by the repair experiments.
- All E5 and ImageBind Full1000 evaluations contain 1,000 aligned queries.
- OmniEmbed covers all 1,000 Audio-CVR queries and all 1,000 OmniCVR queries.
- One unrelated OmniCVR gallery video is excluded identically from both
  OmniEmbed modes; no target or own reference is affected.
- NaN/Inf, benchmark leakage, duplicate, and final-audit violation counts are
  zero.

Detailed numerical provenance is recorded in `EXPERIMENT_AUDIT_FINAL.md` and
`evidence/weak_accept_repair/`.
