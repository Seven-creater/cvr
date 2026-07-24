# Audio-CVR Figure Design Audit

This audit records the visual principles extracted from 20 closely related
papers before redesigning the two main Audio-CVR figures. It is a design audit,
not a claim or citation source; technical statements in the manuscript remain
grounded in the original papers and the locked evidence manifest.

## Papers Inspected

1. CoVA: Text-Guided Composed Video Retrieval for Audio-Visual Content
2. CoVR / CoVR-2: Automatic Data Construction for Composed Video Retrieval
3. EgoCVR: An Egocentric Benchmark for Composed Video Retrieval
4. CIRR: Image Retrieval on Real-Life Images
5. e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings
6. CoVR-2: Automatic Data Construction for Composed Video Retrieval
7. Composed Video Retrieval via Enriched Context and Discriminative Embeddings
8. Beyond Simple Edits: Composed Video Retrieval with Dense Modifications
9. CoVR-R: Reason-Aware Composed Video Retrieval
10. Training-Free Composed Video Retrieval via Video-LLM Reasoning
11. ImageBind: One Embedding Space To Bind Them All
12. Few-Shot Composition Learning for Image Retrieval
13. Bi-Directional Training for Composed Image Retrieval
14. WavCaps
15. AudioSetCaps
16. ACAV100M
17. VGGSound
18. Audio-Visual Event Localization in Unconstrained Videos
19. PinPoint: Composed Retrieval with Explicit Negatives
20. FlatSounds

## Recurring Visual Patterns

### 1. Start from an observable example

The strongest retrieval figures show a reference, an instruction, and ranked
positive or negative candidates. CoVA, CoVR, EgoCVR, CIRR, and PinPoint use
actual or stylized media examples as the primary visual anchor instead of
describing the task through abstract boxes.

**Audio-CVR implication:** Figure 1 should begin with two visually matched video
strips and distinct audio waveforms. The unchanged reference and edited target
must be recognizable before metrics are introduced.

### 2. Encode positive and negative roles directly

Green and red outlines, correct and incorrect rankings, and explicit negative
rows are common in EgoCVR, CIRR, and PinPoint. The color is attached to the
semantic role, not used decoratively.

**Audio-CVR implication:** keep reference blue, target green, rejected or masked
states red, and audio edits amber throughout both figures.

### 3. Make the main result a first-order visual signal

Several papers combine a task schematic with the central failure or improvement.
The reader should not need to decode four metric cards to discover the result.

**Audio-CVR implication:** Figure 1 uses a direct before/after R@1 slope plot.
The 84.5--96.0 point recovery after masking one reference candidate is the hero
evidence; individual diagnostics remain in the text and tables.

### 4. Use modality-native marks

WavCaps and AVE use waveforms; CoVA and EgoCVR use video strips; ImageBind uses
modality-specific examples; AudioSetCaps uses small modality icons and short
stage labels.

**Audio-CVR implication:** Figure 2 uses timelines, video strips, waveforms, and
review lanes. Generic rounded rectangles are retained only as quiet containers.

### 5. Show why each curation gate exists

WavCaps, AudioSetCaps, ACAV100M, and e5-omni contrast failure modes with their
correction or filtering stage. A pipeline is more convincing when the visual
shows what a stage observes and rejects.

**Audio-CVR implication:** audio-only, muted-video, and full-audiovisual review
are separate lanes with different inputs and acceptance questions. A pass at
one lane cannot visually overwrite a failure at another.

### 6. Keep the hierarchy asymmetric

Successful method figures rarely give every step the same size. A dominant task
example or mechanism is supported by smaller validation and audit elements.
Oversized headings inside the artwork are uncommon because the caption already
provides the figure title.

**Audio-CVR implication:** the query and cross-model masking result dominate
Figure 1; the three modality-isolated review lanes dominate Figure 2. Titles,
metadata, and provenance are visually quieter.

## Redesign Contract

### Figure 1

- **Core conclusion:** the unchanged reference is a query-specific pre-edit
  negative that dominates top-rank failure across E5-Omni and ImageBind.
- **Archetype:** asymmetric schematic plus quantitative validation.
- **Panels:** directional query; exact masking intervention; cross-model R@1
  recovery.
- **Risk controlled:** masking is not presented as a replacement benchmark; the
  figure states that all other scores remain fixed.

### Figure 2

- **Core conclusion:** independent modality-isolated gates make automatic
  audio-primary curation auditable and resistant to visual and ASR shortcuts.
- **Archetype:** schematic-led pipeline.
- **Panels:** source-aware clips; audio-first pairing; isolated review lanes;
  audit, deduplication, and frozen benchmark.
- **Risk controlled:** the artwork does not imply human validation or universal
  shortcut elimination.

## Export Contract

- Final width: full AAAI two-column width.
- Backend: Python / Matplotlib.
- Deliverables: editable-text SVG, vector PDF, 600 dpi TIFF, and 600 dpi PNG preview.
- Color: color-blind-safe semantic palette with grayscale-readable structure.
- Typography: 5.0--7.2 pt at final size, no slide-scale headings.
- Source: `build_paper_assets.py`; numerical values come from
  `results_manifest.json`.

## Final QA

- `nature-figure` static preflight: 14 pass, 0 warnings, 0 failures.
- AAAI compilation: TeX Live completed successfully; final manuscript is six
  pages including references.
- Visual inspection: pages 2 and 4 were rendered at 160 dpi and checked for
  overlap, clipping, disconnected arrows, illegible labels, and caption mismatch.
- Editable text: the exported Figure 1 and Figure 2 SVG files contain 50 and 44
  text elements, respectively.
- Draw.io handoff: the SVG files are the editable interchange artifacts. The
  globally installed Draw.io MCP can import or trace them in a newly loaded
  Codex task without changing the scientific layout or values.
