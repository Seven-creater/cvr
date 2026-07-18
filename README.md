# Audio-CVR / AudioDelta-E5

This repository is now centered on **Audio-CVR**: composed video retrieval where
the query is a reference video/audio plus an audio edit, and the target is the
video/audio that satisfies that edit under a preserved visual context.

```text
query  = reference video/audio + edit_text
target = target video/audio
```

The core B-line definition is:

```text
Audio edit under preserved video context determines the target.
```

This means the benchmark should not collapse into pure visual retrieval, pure
ASR/transcript retrieval, ordinary similar-video retrieval, or a trivial random
gallery test.

## Current Focus

The active work is split into three connected parts:

1. **B-line Audio-CVR dataset construction**
   - Build 6-9 second clips, default 8 seconds.
   - Generate audio-primary CVR triplets with `b_audio_blind_review_v2`.
   - Export tiered B-line records: `B-main`, `B-extended`, and `B-diagnostic`.

2. **Protocol-quality evaluation**
   - Build reference-aware, typed-hard-negative, and local/same-source galleries.
   - Verify that hard negatives do not satisfy the edit text.
   - Run audio necessity ablations, especially `V+T` vs `V+A+T`.

3. **AudioDelta-E5 training**
   - Train a lightweight adapter on frozen E5-Omni embeddings.
   - Current first-stage recipe follows E5-Omni-style ideas:
     modality-aware temperature, negative curriculum/debiasing, and covariance alignment.
   - AudioDelta-specific losses remain available for later controlled ablations.

The older AVIGATE wrapper code is still present for historical experiments, but
it is no longer the main project entry point.

## Repository Map

Important application modules:

```text
app/audio_cvr_clips.py            # Build 6-9s / 8-12s Audio-CVR clips.
app/audio_lines_single_source.py  # B-line candidate mining, proposal merge, tier export.
app/audio_cvr_protocol_eval.py    # Protocol summaries, local same-source mining, eval tables.
app/e5_audio_delta_train.py       # Prepare/cache/train/eval AudioDelta-E5 adapter.
app/composed_omni.py              # Omni prompt/client utilities.
app/composed_data.py              # Shared composed retrieval data helpers.
```

Important scripts:

```text
scripts/build_audio_cvr_6_9s_clips.sh
scripts/run_audio_cvr_bline_6_9s_full_4gpu.sh
scripts/run_audio_cvr_protocol_eval.sh
scripts/run_audio_necessity_7mode_parallel.sh
scripts/run_e5_audio_delta_smoke.sh
scripts/setup_e5_train_env.sh
```

Important docs:

```text
doc/linux_data_structure.md
doc/audio_lines_ab_flow_20260511.md
doc/audio_cvr_large_scale_handoff_20260514.md
doc/audio_delta_e5_training_method_20260517.md
doc/audio_delta_e5_training_runbook_20260519.md
doc/audio_cvr_data_negative_audio_necessity_protocol_20260523.md
doc/audio_cvr_protocol_smoke_report_20260525.md
```

## Data Layout

The server data root is expected to look like:

```text
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/
  raw/
    avatar/
    daily_omni/
    hdtf/
    vgg_monoaudio/
    vggsound/
    voxceleb/
    worldsense/
  clips/
    audio_cvr_6_9s/
  runs -> /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs
```

Current large-scale construction should avoid over-relying on `avatar`, because
its short clips often yield only reference/target pairs and weak local
same-source negatives. Longer or richer sources such as `daily_omni`,
`worldsense`, `hdtf`, `vggsound`, `vgg_monoaudio`, and selected `voxceleb`
should be balanced more carefully.

## B-Line Dataset Construction

The main construction script is:

```bash
bash scripts/run_audio_cvr_bline_6_9s_full_4gpu.sh \
  --run-root runs/audio_cvr_bline_6_9s_full_$(date +%Y%m%d_%H%M%S) \
  --gpu-ids 0,1,2,3 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --clip-seconds 8 \
  --min-clip-seconds 6 \
  --max-clip-seconds 9 \
  --propose-shards 64 \
  --propose-parallel-jobs 8 \
  --concurrency 4 \
  --request-timeout-seconds 240
```

This script:

1. Builds or reuses 6-9s clips.
2. Runs B-line annotation/proposal with Qwen3-Omni through vLLM.
3. Uses `b_audio_blind_review_v2`.
4. Writes tiered triplet outputs and review metadata.

Primary expected outputs under a run directory:

```text
b_all_audio_cvr_triplets.jsonl
b_main_audio_cvr_triplets.jsonl
b_extended_audio_cvr_triplets.jsonl
b_diagnostic_audio_cvr_triplets.jsonl
b_speech_audio_content_triplets.jsonl
benchmark_quality_summary.json
audio_necessity_eval_manifest.json
manual_review/B/
```

`B-main` is for the clean benchmark. `B-extended` is for training. `B-diagnostic`
keeps ASR-like, visual-shortcut, audio-only-solvable, or ambiguous cases out of
the main table while preserving them for analysis.

## Protocol Evaluation

The protocol runner prepares galleries, caches E5 embeddings, evaluates top-k
results, and writes diagnostics:

```bash
bash scripts/run_audio_cvr_protocol_eval.sh \
  --run-root <RUN_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --adapter-dir <ADAPTER_DIR> \
  --gallery-size 1000 \
  --protocols random,reference,local_same_source,typed_hardneg \
  --mine-local-same-source \
  --max-eval-records 64
```

Gallery protocols:

```text
random              # sanity only; not a main benchmark result
reference           # includes reference_negative to test edit direction
local_same_source   # uses same raw source clips when available
typed_hardneg       # reference + visual_hard + audio_hard + asr_hard
```

Hard negative types:

```text
reference_negative      # the unedited reference video itself
local_same_source       # same raw source, excluding reference/target
local_fallback_visual   # cross-source visual fallback when strict local is missing
visual_hard             # visually similar but audio edit does not hold
audio_hard              # audio-related but wrong visual context
asr_hard                # speech keywords/topics similar but not the target
random_distractor       # ordinary distractor, useful only for scale
```

Every hard negative should pass a false-negative guard. If a candidate also
satisfies the edit text, or its status is uncertain without review, it should
not be used as a formal negative.

## Audio Necessity Ablation

The 7-mode runner executes the required ablations in parallel across seven GPUs:

```bash
bash scripts/run_audio_necessity_7mode_parallel.sh \
  --run-root <RUN_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --gpu-ids 1,2,3,4,5,6,7 \
  --gallery-protocol typed_hardneg \
  --gallery-size 1000 \
  --max-train-records 192 \
  --max-eval-records 64 \
  --steps 120 \
  --batch-size 8 \
  --learning-rate 0.0003 \
  --local-segments 0
```

Modes:

```text
T-only-fullAV  # edit_text only -> full AV gallery, text-prior baseline
V-only         # muted reference video + edit_text -> muted gallery videos
A-only         # reference audio + edit_text -> gallery audios
V+T            # reference video + edit_text -> gallery videos
A+T            # reference audio + edit_text -> gallery audios
V+A            # reference video/audio without edit_text -> gallery video/audio
V+A+T          # full Audio-CVR query -> full gallery video/audio
```

The key proof is not just that `V+A+T` is highest. The main comparisons are:

```text
V+A+T vs V+T
A+T vs T-only-fullAV
target_beats_reference with audio on vs audio off
target-reference score gap with audio on vs audio off
```

If `A-only` is almost the same as `V+A+T`, that sample or split may be drifting
toward audio-only or ASR-style retrieval and should be treated cautiously.

## AudioDelta-E5 Training

Smoke test:

```bash
bash scripts/run_e5_audio_delta_smoke.sh \
  --dataset-run-root <RUN_ROOT> \
  --run-root runs/e5_audio_delta_smoke_$(date +%Y%m%d_%H%M%S) \
  --gpu-ids 0 \
  --max-train-records 8 \
  --max-eval-records 4 \
  --train-steps 20 \
  --training-profile e5_omni_recipe
```

Manual pipeline:

```bash
python3 -m app.e5_audio_delta_train prepare \
  --dataset-run-root <RUN_ROOT> \
  --output-dir <TRAIN_RUN>/records \
  --max-train-records 192 \
  --max-eval-records 64 \
  --eval-gallery-size 1000 \
  --eval-gallery-protocol typed_hardneg

python3 -m app.e5_audio_delta_train cache-embeddings \
  --records-dir <TRAIN_RUN>/records \
  --output-dir <TRAIN_RUN>/cache

python3 -m app.e5_audio_delta_train train-adapter \
  --cache-dir <TRAIN_RUN>/cache \
  --output-dir <TRAIN_RUN>/adapter \
  --training-profile e5_omni_recipe \
  --steps 120 \
  --batch-size 8 \
  --learning-rate 0.0003

python3 -m app.e5_audio_delta_train eval \
  --cache-dir <TRAIN_RUN>/cache \
  --adapter-dir <TRAIN_RUN>/adapter \
  --output-dir <TRAIN_RUN>/eval \
  --save-topk 10
```

Current default training profile:

```text
e5_omni_recipe / v2_research
  on:  modality-aware temperature
  on:  quantile negative curriculum
  on:  false-negative debiasing
  on:  CORAL / covariance alignment
  off: AudioDelta-specific delta/ref/hard-negative/edit-type/visual losses
  off: multi-positive, memory bank, LoRA
```

The current training code intentionally uses the S1/e5-omni recipe only.
Legacy AudioDelta lambda flags remain parseable for command compatibility, but
are forced to zero and cannot create a C1 run. The former loss schedule is
kept as a compatibility command that evaluates S1 only.

## Evaluation Metrics

### Paper-grade Audio-CVR experiment

Use `scripts/run_audio_cvr_aaai_final_experiment.sh` for the final adapter
experiment. It preserves the existing source-disjoint assignment, filters the
formal test set to forward B-main records, selects steps/LR/batch size on the
validation split, and then runs five final seeds across all seven audio
necessity modes. The launcher also evaluates a validation-tuned V+T/A+T late
fusion baseline and writes paired bootstrap, randomization, and McNemar
statistics.

```bash
setsid nohup bash scripts/run_audio_cvr_aaai_final_experiment.sh \
  --run-root /path/to/audio_cvr_run \
  --split-root /path/to/audio_cvr_run/b_splits \
  --output-dir runs/aaai_audiocvr_final_$(date +%Y%m%d_%H%M%S) \
  --gpu-ids 1,2,3,4,5,6,7 \
  > logs/aaai_audiocvr_final.log 2>&1 < /dev/null &
```

The experiment protocol and literature rationale are documented in
`doc/aaai_audiocvr_final_experiment_protocol_20260718.md`.

Report more than `R@K`:

```text
R@1 / R@5 / R@10
Median Rank / Mean Rank
target_beats_reference
reference_rank_median
reference_rank <= 1
target-reference score gap mean
positive beats reference_negative
positive beats local_same_source
positive beats visual_hard
positive beats audio_hard
positive beats asr_hard
subtype breakdown: speech_topic_in_video_context / music / sound_event
shortcut breakdown: clean_audio_delta / ASR-like / visual-shortcut / audio-only-solvable / ambiguous
```

If random-gallery results are high but reference/local/typed-hard-negative
results are low, that is useful evidence: the protocol is exposing the real
Audio-CVR difficulty rather than measuring trivial distractor separation.

## Testing

Run focused tests after code changes:

```bash
python -m unittest tests.test_audio_lines_single_source -v
python -m unittest tests.test_audio_cvr_clips -v
python -m unittest tests.test_e5_audio_delta_train -v
python -m unittest tests.test_scripts -v
```

Run all tests when the change crosses data construction, protocol evaluation,
and training:

```bash
python -m unittest discover -v
```

## Practical Notes

- Do not train on old B-line runs unless they have been converted into the new
  tiered format. The old loose B-line data is useful for history, not for the
  current Audio-CVR benchmark.
- After Omni construction finishes, stop the vLLM process before starting E5
  cache/training. E5 needs the GPUs and otherwise will OOM.
- `local_same_source=0` usually means the source only produced reference and
  target clips, not that local negatives are conceptually invalid.
- For final reporting, use reference-aware, local/same-source, typed-hardneg,
  and audio necessity tables. Treat random gallery as a smoke sanity check.
