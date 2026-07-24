# Audio-CVR Claim-Evidence Matrix

Status: final contract for the frozen Full1000 paper.

## Fixed Thesis

Audio-CVR is a diagnostic benchmark and automatic curation method for testing
whether a composed video retriever can rank an audio-edited target above its
pre-edit reference under preserved visual context. The principal failure is
identity-sensitive pre-edit source anchoring; audio utility is conditional on
the representation and fusion method.

## Claims

| ID | Permitted claim | Final evidence | Paper location |
|---|---|---|---|
| C1 | Aggregate recall hides severe own-reference anchoring. | Exact one-score masking on Full1000: R@1 increases by 84.5 points for adapted E5 V+A+T, 96.0 for ImageBind, and 99.2 for OmniEmbed. Adapted E5 sends 98.99% of top-1 errors to the own reference. | Abstract, evaluation protocol, Table 2 |
| C2 | The effect is identity-sensitive but not limited to exact file or frame matching. | Transcoding preserves most of the drop; temporal and spatial variants still leave 26.1--84.0 point drops across models and modes. | Reference perturbation method, Table 3, discussion |
| C3 | Audio can help directional retrieval after adaptation, but can also act as interference. | E5 adapter gains +6.84 R@1 points, ImageBind loses 9.20, and OmniEmbed loses 1.00 under fixed zero-shot fusion. | Table 4, discussion |
| C4 | The failure is not specific to one model or one benchmark. | Frozen E5, adapted E5, ImageBind, and OmniEmbed agree on Full1000. OmniEmbed source masking adds 14.4/14.0 points on 1,000 OmniCVR queries. | Tables 2--4 |
| C5 | The data pipeline is auditable but not equivalent to a human gold set. | Full1000 SHA frozen; duplicate/leakage/missing media zero; 78/78 automatic repeats; partial blinded human audit 34/50 valid. | Table 1, limitations |
| C6 | Own-reference difficulty exceeds ordinary hard-negative difficulty. | Final typed-negative checks and the historical mixed-gallery control show that the own reference dominates typed, local, and random negatives. Local control coverage is limited. | Error analysis, supplement |

## Contribution Contract

1. **Audio-primary task and automatic curation.** Directional audio edits are
   screened for visual and ASR shortcuts with provenance-preserving audits.
2. **Reference-aware diagnosis.** Exact masking, reference-specific metrics,
   and controlled identity perturbations isolate source anchoring.
3. **Cross-model evidence.** Frozen and adapted E5, ImageBind, and
   retrieval-trained OmniEmbed show that anchoring is broad while audio effects
   are model dependent.

The adapter, distributed curation machinery, and Audio-as-Text VLM2Vec
reproduction are baselines or engineering support, not headline innovations.

## Evidence Levels

| Evidence | Permitted wording |
|---|---|
| Full1000 | Automatically curated and model-verified; frozen 1,000-query diagnostic benchmark. |
| Human audit | Partial blinded single-rater audit: 50 unique queries, 68.0% valid; no completed paired repeat. |
| Core150 | Historical sensitivity subset; not the primary benchmark. |
| OmniCVR | Independent cross-benchmark source-masking diagnostic; not evidence that audio improves R@1. |
| VLM2Vec reproduction | Supplementary non-official reproduction only. |

## Forbidden Claims

- Audio is universally necessary or beneficial.
- Audio-CVR is the first audiovisual CVR benchmark or the first to include a
  source in the gallery.
- Full1000 is human validated, consensus labeled, or a multi-rater gold set.
- Exact masking proves purely semantic edit-direction reasoning.
- Identity perturbation eliminates all identity information.
- The adapter is a new retrieval architecture or solves the task.
- The non-official VLM2Vec reproduction is AudioVLM2Vec.
- Full1000 is an open-world retrieval leaderboard rather than a controlled
  source-target diagnostic.

The detailed numerical ledger is `EXPERIMENT_AUDIT_FINAL.md`.
