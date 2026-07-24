# Audio-CVR Pre-Submission Review

## Review Setup

- **Input scope:** Complete six-page anonymous AAAI-27 manuscript, two figures,
  three tables, the locked result manifest, claim-evidence matrix, statistical
  audit, and reference-verification report.
- **Assessment boundary:** This is a reviewer-style manuscript assessment. Raw
  videos, the full JSONL review history, executable release package, and the
  completed AAAI reproducibility checklist were not independently re-run as
  part of this review.
- **Shared manuscript claim:** Existing composed video retrieval systems often
  retrieve the correct scene neighborhood but rank the unchanged reference
  above the post-edit target. Audio-CVR studies this directional failure with
  audio-primary curation, exact own-reference masking, and
  reference-specific metrics.
- **Visible evidence base:** A frozen non-speech Full1000 benchmark; a
  2,000-item target/reference gallery; five-seed E5-Omni adapter results;
  frozen E5 and zero-shot ImageBind controls; a Core150 sensitivity analysis;
  and an external 995-query OmniCVR diagnostic.
- **Missing materials affecting confidence:** Public release documentation,
  per-source review-profile counts, formal human-annotation records, and an
  independently executable reproduction of the data-construction pipeline.

## Reviewer 1

**Emphasis:** Technical validity and technical failings.

### Overall Assessment

The paper isolates a real evaluation failure with an unusually clean paired
intervention: the score matrix is held fixed while only the current query's
reference is masked. The near-complete attribution of top-1 failures to the
own reference, together with frozen E5 and ImageBind controls, makes the
central diagnosis technically persuasive. The main remaining risks concern
the evidentiary status of the automatically curated resource and the precision
of the statistical-method description.

### Who Would Be Interested, and Why

Researchers in composed retrieval, multimodal representation learning, and
benchmark design should care because the protocol separates scene relevance
from edit-direction discrimination. The result also matters to practitioners
who currently interpret aggregate recall as evidence of successful
composition.

### Major Strengths

- Exact own-reference masking changes one query-specific score and leaves the
  other 1,999 candidates fixed.
- Five-seed adapted E5 results, frozen E5, zero-shot ImageBind, Core150, and
  OmniCVR provide complementary controls.
- The paper reports effect sizes, confidence intervals, paired randomization,
  McNemar tests, and Holm-adjusted p values rather than significance alone.
- Negative ImageBind audio-fusion results are retained and correctly used to
  bound the claim.

### Major Concerns

**R1-M1 - [data-resource-quality] Heterogeneous verification and moderate repeat agreement**

- **Claim pointer:** Full1000 is presented as automatically curated and
  model-verified.
- **Evidence pointer:** "Audio-CVR Task and Benchmark / Composition and audit
  scope"; "Staged freeze and provenance"; "Limitations and Conclusion."
- **Concern:** All 78 requested sampled repeat reviews are now complete, with
  79.5% exact-decision and 85.6% field-level agreement. This resolves the
  coverage gap but not the validation boundary: the repeat is a post-freeze
  same-model stability audit, and the fixed and added pools retain explicitly
  heterogeneous review profiles. Core150 has only a qualitative 10-query spot
  check by both authors, not a full independent human annotation study.
- **Resolution test:** Publish a per-query provenance field and a compact
  count table mapping final queries to review profile, completed gates, repeat
  status, and source dataset. Keep "model-verified" explicitly qualified and
  do not imply uniform multi-pass verification.

**R1-M2 - [statistical-rigor] Resampling unit is not explicit enough**

- **Claim pointer:** Paired bootstrap confidence intervals and paired
  randomization tests support the E5 audio-gain claims.
- **Evidence pointer:** "Baselines and Experiments / Statistics"; Table 2;
  `STATISTICAL_AUDIT.md`.
- **Concern:** The manuscript names the procedures and iteration counts but
  does not state in the main text whether resampling is over queries, how the
  five seeds are aggregated inside each resample, or which planned
  comparisons form each Holm family. Those choices affect the interpretation
  of uncertainty.
- **Resolution test:** Add one methods sentence defining the independent unit,
  seed aggregation, paired resampling operation, and correction family.

**R1-M3 - [reproducibility] Release contract is not yet visible**

- **Claim pointer:** The pipeline is scalable, restartable, and preserves
  provenance through the frozen split.
- **Evidence pointer:** Figure 2; "Automatic Curation / Deduplication, review
  audit, and persistence."
- **Concern:** The manuscript describes persistence and provenance, but the
  supplied package does not yet show the final public data schema, media
  redistribution boundary, prompts/model versions, or an executable
  reconstruction recipe.
- **Resolution test:** Provide an anonymized release manifest containing
  stable IDs, media-source metadata, review fields, split hashes, model and
  prompt versions, and commands that regenerate all paper tables from the
  frozen manifest.

### Technical Failings to Address Before the Case Is Established

R1-M1 and R1-M2 should be addressed before submission. R1-M3 can be closed by
a concrete anonymized artifact plan if the full release cannot accompany the
initial manuscript.

### Assessment Against High-Impact Review Criteria

- **Originality:** The exact reference-specific intervention is distinctive.
- **Scientific importance:** Strong within composed retrieval and benchmark
  methodology; broader importance depends on artifact reuse.
- **Interdisciplinary readership:** Primarily multimodal learning and
  information retrieval.
- **Technical soundness:** Central score-masking evidence is strong; dataset
  verification reporting needs greater granularity.
- **Readability for nonspecialists:** The piano-to-guitar example and Figure 1
  make the problem accessible.

### Recommendation Posture

Supportive if verification provenance and statistical-unit reporting are made
fully explicit.

## Reviewer 2

**Emphasis:** Originality and scientific importance.

### Overall Assessment

The strongest contribution is not putting the source in the gallery, which
OmniCVR already does, but converting that source into an explicit
query-specific counterfactual and measuring the resulting directional error.
The manuscript now states this distinction honestly. Its significance will
depend on convincing readers that Audio-CVR is a reusable diagnostic benchmark
rather than a narrowly engineered test that exposes one predictable nearest
neighbor failure.

### Who Would Be Interested, and Why

Composed image/video retrieval researchers, multimodal embedding researchers,
and dataset curators would be interested in the distinction between finding
the right context and satisfying the requested edit. The result may also
influence evaluation design for other pre-state/post-state retrieval tasks.

### Major Strengths

- The paper does not claim to be the first audiovisual CVR benchmark or the
  first gallery containing a source video.
- The contribution identity is coherent: audio-primary curation,
  reference-aware evaluation, and controlled evidence.
- Cross-model and cross-benchmark analyses reduce the chance that the finding
  is merely an artifact of the few-shot adapter.
- The negative ImageBind result prevents the paper from collapsing into a
  simplistic "more modalities always help" claim.

### Major Concerns

**R2-M1 - [novelty-significance] The distinction from neighboring benchmarks
must remain concrete**

- **Claim pointer:** Audio-CVR contributes a narrower reference-specific
  diagnostic formulation beyond CoVA and OmniCVR.
- **Evidence pointer:** "Related Work / Audio-visual composed retrieval";
  Introduction contribution paragraph; Table 3.
- **Concern:** The current prose explains the distinction, but a skeptical
  reader may still view exact source masking and new metrics as post-hoc
  analysis on a familiar source-target gallery. The automatic curation method
  is therefore essential to the novelty case, not secondary implementation
  detail.
- **Resolution test:** Add a compact comparison in the supplement or related
  work covering task focus, source in gallery, own-source intervention,
  muted-video screening, ASR screening, and reference-specific error
  attribution for CoVR, EgoCVR, CoVA, OmniCVR, and Audio-CVR.

**R2-M2 - [experimental-design] Baseline coverage is narrow for a benchmark
paper**

- **Claim pointer:** The benchmark exposes a systematic failure in composed
  retrieval models.
- **Evidence pointer:** "Baselines and Experiments / Models"; Table 2;
  cross-benchmark diagnostic.
- **Concern:** E5-Omni and ImageBind are meaningfully different representation
  families, but neither is a purpose-built contemporary composed video
  retrieval system evaluated end-to-end on Full1000. Reviewers may ask whether
  a CoVA- or OmniCVR-style model exhibits the same failure under the proposed
  protocol.
- **Resolution test:** If a compatible public model can be run without
  changing the frozen test, add it. Otherwise, state the compatibility
  limitation explicitly, publish the evaluation adapter, and frame the
  cross-model conclusion as applying to the two evaluated embedding families.

**R2-M3 - [claim-moderation] Benchmark scope limits broad generality**

- **Claim pointer:** Audio-CVR is a benchmark for directional audio composed
  video retrieval.
- **Evidence pointer:** Table 1; "Composition and audit scope"; "Limitations
  and Conclusion."
- **Concern:** Full1000 is non-speech, 51% Avatar, and lacks benchmark-scale
  strict same-source local negatives. These are disclosed, but they limit a
  claim of broad audiovisual CVR coverage.
- **Resolution test:** Retain "non-speech, audio-primary diagnostic benchmark"
  in the abstract, dataset card, and conclusion; report source-specific
  results in the supplement; avoid using Full1000 as a proxy for general
  audiovisual reasoning.

### Technical Failings to Address Before the Case Is Established

R2-M1 is primarily a positioning requirement. R2-M2 is the largest empirical
risk; if no additional model can be evaluated, the scope of the cross-model
claim must remain exact.

### Assessment Against High-Impact Review Criteria

- **Originality:** Credible when defined as counterfactual diagnosis plus
  curation, not source inclusion.
- **Scientific importance:** Potentially influential for CVR evaluation;
  adoption will determine longer-term importance.
- **Interdisciplinary readership:** Moderate, concentrated in multimodal
  learning, retrieval, and benchmark design.
- **Technical soundness:** Strong central intervention; model coverage is
  limited.
- **Readability for nonspecialists:** The causal contrast is intuitive, though
  the benchmark landscape is specialized.

### Recommendation Posture

Promising, with the novelty case established only if the curation and
diagnostic protocol remain central and the model-scope boundary is explicit.

## Reviewer 3

**Emphasis:** Interdisciplinary reach and readability.

### Overall Assessment

The manuscript has a clear and memorable failure mode: a model recognizes the
scene but chooses the pre-edit state. Figure 1 is an effective summary, and
the abstract now presents both the positive E5 result and the negative
ImageBind fusion result. The remaining communication challenge is to keep
"source," "reference," "audio-primary," and "audiovisual" from implying
broader claims than the task actually tests.

### Who Would Be Interested, and Why

Readers studying multimodal reasoning, retrieval evaluation, and automated
dataset construction can understand the diagnostic without detailed knowledge
of a particular architecture. The pre-state/post-state distinction may also
be useful outside video retrieval.

### Major Strengths

- The opening example explains the task without model-specific jargon.
- Figures 1 and 2 map directly to the two scientific contributions.
- The discussion distinguishes retrieval relevance from edit direction and
  audio evidence from successful audio-text composition.
- Limitations are unusually direct for a benchmark submission.

### Major Concerns

**R3-M1 - [writing-clarity] Source and reference terminology can still drift**

- **Claim pointer:** The paper diagnoses reference/source confusion through
  exact own-reference masking and exact source masking.
- **Evidence pointer:** Abstract; Figure 1; "Reference-Aware Evaluation";
  "Cross-benchmark diagnostic."
- **Concern:** "Reference" is the Audio-CVR task input, while "source" follows
  OmniCVR terminology. Alternating the terms without a single explicit
  equivalence statement may make readers wonder whether two different gallery
  elements are being discussed.
- **Resolution test:** Define once that the unchanged query reference is called
  the source in OmniCVR, use "reference" throughout Audio-CVR, and reserve
  "source masking" for the OmniCVR experiment.

**R3-M2 - [data-resource-quality] The construction yield is not visible**

- **Claim pointer:** Multi-stage curation produces a controlled Full1000
  diagnostic set.
- **Evidence pointer:** Figure 2; "Staged freeze and provenance"; Table 1.
- **Concern:** The reader sees the final composition but not the funnel from
  candidate pairs through rejection stages to the frozen benchmark. Without a
  yield table, it is difficult to judge which gates dominate or how readily
  the procedure can be replicated on another source collection.
- **Resolution test:** Add a supplementary flow table with candidate count,
  pass/reject count by gate, final count by source dataset, and missing/error
  counts. Keep the main figure conceptual.

**R3-M3 - [claim-moderation] Audio-primary should not be read as joint-modality
necessity**

- **Claim pointer:** The benchmark tests directional sound changes under
  preserved visual context.
- **Evidence pointer:** "Audio-CVR Task and Benchmark / Operational validity";
  seven-mode ablation discussion.
- **Concern:** Audio-only and A+T results show that some queries may be solved
  largely from sound evidence. The paper should not imply that every query
  requires synergistic use of both audio and video.
- **Resolution test:** Preserve the current definition: muted video alone is
  screened as insufficient, while audio supplies the primary changing
  evidence. Explicitly state that the benchmark tests audio-grounded
  direction under controlled visual context, not universal audiovisual
  synergy.

### Technical Failings to Address Before the Case Is Established

No readability issue invalidates the central result. R3-M2 matters for trust
and reuse, while R3-M1 and R3-M3 can be resolved by local terminology edits.

### Assessment Against High-Impact Review Criteria

- **Originality:** The diagnostic viewpoint is easy to distinguish once the
  terminology is stable.
- **Scientific importance:** Clear field-level relevance; broader reach is
  plausible but not yet demonstrated.
- **Interdisciplinary readership:** The core idea travels well beyond the
  architecture details.
- **Technical soundness:** Understandable from the manuscript, subject to the
  provenance caveats raised above.
- **Readability for nonspecialists:** Strong overall; terminology and dataset
  funnel reporting are the remaining barriers.

### Recommendation Posture

Supportive after targeted clarification and a more transparent resource-yield
account.

## Cross-Review Synthesis

### Consensus Strengths

- Exact own-reference masking is a clean, interpretable intervention.
- The central source/reference confusion finding is supported across adapted
  E5, frozen E5, ImageBind, and OmniCVR.
- The manuscript appropriately separates the cross-model confusion result from
  the model-dependent benefit of audio.
- The paper is visually clear, concise, and unusually candid about its
  limitations.

### Consensus Technical Risks

Two reviewers independently identify **verification provenance and resource
transparency** as the largest trust risk (R1-M1 and R3-M2). Repeat-review
coverage is complete, but moderate agreement and heterogeneous review profiles
remain material boundaries. Two reviewers also
identify **scope control** as essential: Full1000 is a non-speech diagnostic
benchmark, not a complete test of audiovisual reasoning (R2-M3 and R3-M3).

### Where Emphasis Differs Across Reviewers

- Reviewer 1 places greatest weight on review-profile transparency, the
  statistical resampling unit, and executable provenance.
- Reviewer 2 places greatest weight on novelty relative to CoVA/OmniCVR and
  the absence of a purpose-built CVR model on Full1000.
- Reviewer 3 places greatest weight on terminology, construction-yield
  visibility, and preventing "audio-primary" from being read as mandatory
  joint-modality synergy.

### Broad-Interest and Significance Readout

The paper presents a strong and memorable field-level result: high aggregate
retrieval performance can hide a nearly deterministic failure at the exact
pre-edit/post-edit decision boundary. Its broadest reusable idea is the
query-specific counterfactual intervention, not the particular adapter or
source datasets. Claims of general audiovisual reasoning should remain
bounded.

### Most Important Issues to Resolve Before Submission

1. Publish or summarize per-query review provenance and the construction
   funnel.
2. Define the statistical resampling unit and Holm comparison families.
3. Add a compact prior-benchmark capability comparison.
4. Either add a compatible purpose-built CVR model or explicitly narrow the
   cross-model claim to E5-Omni and ImageBind.
5. Complete the anonymized artifact and reproducibility-checklist plan.

## Risk / Unsupported Claims

- A fully human-validated or multi-annotator gold-set claim is unsupported and
  must not appear.
- A claim that audio universally improves composed retrieval is contradicted
  by ImageBind and must not appear.
- A claim that every query requires joint audio-video reasoning is not
  established.
- A claim that strict same-source local-negative behavior is validated at
  Full1000 scale is unsupported.
- Generalization to speech-centered CVR is not established.
- The public reproducibility and licensing posture remains unverified until the
  release package and AAAI checklist are completed.
