# Audio-CVR Paper Build Status

Last verified: 2026-07-24

## Current State

```text
paper assets                PASS
strict result gate          PASS
Draw.io source/export       PASS
editable SVG text           PASS
600-dpi TIFF bundle         PASS
embedded Type 1/TrueType    PASS
Type 3 fonts                0
AAAI-27 compile             PASS
citation-key resolution     PASS
undefined references        0
missing result fields       0
manifest null values        0
LaTeX errors                0
figure-related warnings     0
overfull boxes              0
underfull paragraphs        2
repeat-review completion    78 / 78
repeat exact agreement      79.49%
repeat field agreement      85.64%
partial blinded audit       34 / 50 valid
PDF page size               US Letter
PDF pages                   7
page-by-page visual QA      PASS
figure direct QA            PASS
submission source state     READY FOR AUTHOR REVIEW
```

`python paper/build_paper_assets.py --strict --skip-figures` completes
successfully. The `--skip-figures` flag preserves the Draw.io exports instead
of invoking the legacy Matplotlib figure functions. The paper contains no
experimental placeholders: all Full1000, E5, ImageBind, Core150, and OmniCVR
values used in the manuscript are populated from the locked result manifest.

The two-page Draw.io source is self-contained and retains real Test1000 frames
and waveforms. Figure 1 uses an Avatar sound-event example (`add a dog bark to
the audio`) to connect the directional edit, the one-score masking
intervention, and three-model Full1000 results. Figure 2 is an icon-led
clip-pair-verify-freeze schematic whose audio-only, muted-video, and
full-audiovisual checks remain independent parallel gates. Long explanatory
text is kept in the manuscript caption rather than inside the artwork. The
latest SVG/PDF/PNG/TIFF exports are synchronized. SVG text remains editable
and media is embedded; TIFFs are 4320 x 1560 pixels at 600 dpi. Detailed
checks are recorded in
`paper/figures/QA_REPORT_20260724.md`.

## Locked Evidence

```text
server code commit
  6ede6560e8a6c6af24746ee4c6729933bca97ad8

server result run
  runs/audio_cvr_e5_imagebind_final1000_20260723_143500

frozen Test1000 SHA256
  70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e

post-freeze repeat audit
  paper/evidence/server_final1000/dataset/postfreeze_repeat_audit.json

five-seed adapter training audit
  paper/evidence/server_final1000/adapter_training/five_seed_training_audit.json

local evidence bundle
  paper/evidence/server_final1000/
```

The canonical storyline and claim-evidence matrix remain authoritative. The
paper presents reference/source confusion as the central failure, automatic
audio-primary curation as the data contribution, and exact own-reference
masking as the diagnostic intervention. The E5 adapter is a baseline rather
than a method contribution. ImageBind supports the cross-model reference
confusion claim while delimiting, rather than supporting, a universal audio
gain claim.

## Verified Artifacts

```text
paper/build_submission/main.pdf
paper/build_main/main_redrawn_figures.pdf
paper/main.tex
paper/results_manifest.json
paper/CLAIM_EVIDENCE_MATRIX.md
paper/STATISTICAL_AUDIT.md
paper/REFERENCE_VERIFICATION.md
paper/PRE_SUBMISSION_REVIEW.md
paper/evidence/core150_manual_spot_check.md
paper/figures/drawio/audio_cvr_figures.drawio
paper/figures/generated/figure1_reference_confusion.pdf
paper/figures/generated/figure1_reference_confusion.svg
paper/figures/generated/figure1_reference_confusion.png
paper/figures/generated/figure1_reference_confusion.tiff
paper/figures/generated/figure2_curation_pipeline.pdf
paper/figures/generated/figure2_curation_pipeline.svg
paper/figures/generated/figure2_curation_pipeline.png
paper/figures/generated/figure2_curation_pipeline.tiff
paper/figures/QA_REPORT_20260724.md
paper/generated/asset_build_summary.json
paper/generated/results_snapshot.json
```

The seven-page anonymous `build_submission/main.pdf` was rendered, with figure
pages 2 and 4 inspected at publication width. The older
`build_main/main.pdf` was open in another Windows process during final
compilation and was therefore left untouched rather than force-overwritten.
The title,
two-column text, equations, two figures, three tables, captions, and references
show no clipping, overlap, or unreadably small labels. Technical content remains
within the AAAI seven-page limit; references begin on page 6.

## Remaining Author Actions

The manuscript engineering gate is clear, but the PDF is not yet declared
submission-final. Authors must still:

1. Read and approve every claim, number, caption, and limitation.
2. Complete the official AAAI reproducibility checklist.
3. Confirm current AAAI-27 AI-assistance, data, and code-disclosure policies.
4. Add final author metadata only in the non-anonymous accepted/final version.
5. Finalize code/data availability, source-dataset licenses, and release scope.
6. Perform one final upload-preview check in OpenReview.
