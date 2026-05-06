# Natural Pair Dataset Route Status - 2026-05-06

## Current Decision

The main dataset route is now natural video pair construction:

```text
source videos -> event clips -> Omni-Detective annotations -> group pairs -> judge -> video verification -> accepted_pairs.jsonl
```

Synthetic video editing remains available for controlled experiments, but it is no longer the default route for large-scale dataset production. The VACE route exposed too many quality bottlenecks: mask quality, target video drift, and weak semantic control even when masks were good.

## What Changed

- `natural_pair` is the production default. `synthetic_edit` is a supplement only.
- `run_omni_detective_pilot.sh` now exposes `--concurrency`, `--max-accepted-pairs`, `--max-proposals`, `--annotation-max-passes`, stage timeouts, and `--start-stage`, and writes a manual review bundle after accepted pairs are produced.
- Pair priority now favors audio/video fusion signals first: `audio_event`, `speech`, `visible_text`, then object/action/attribute/scene.
- Natural pair gates now record explicit failure buckets:
  - `bad_imperative_edit_text`
  - `too_similar_without_observable_delta`
  - `too_broad_or_loose_pair`
  - `ocr_template_risk`
  - `audio_event_too_similar`
  - `visible_text_fragment_edit`
- Accepted natural pairs must still pass Omni judge, video-level verification, and local structured gates.

## Latest Smoke

`cf4489a` completed end-to-end with 236/236 unique clip annotations, 40 judged
pair proposals, and 3 accepted pairs. The run proved the staged resume and
fail-fast script behavior, but the accepted set is still too small and two
accepted examples exposed weak-delta risks:

- `audio_event`: `replace the low-frequency electronic hum with a low electronic hum`
  is nearly a same-event rewrite rather than a useful edit.
- `visible_text`: `Singapore's Manufacturing -> Singapore` keeps only a
  fragment of the source OCR text.

The current fix keeps those cases out of `accepted_pairs.jsonl` by requiring
audio event from/to values to be semantically distinct and visible-text targets
not to be simple fragments of the source text. The next smoke should increase
`--max-proposals` after pulling the new commit, because the previous 40
proposals produced too few diverse accepted pairs.

`c434c13` exposed one more operational issue: with a larger `--max-proposals`,
the first proposal can spend too long inside the three Omni calls
`propose -> judge -> video verification`, so the outer stage timeout can kill
the job before any row is written. The fix is to write per-proposal heartbeat
logs, use a shorter pair-request timeout, and skip expensive video verification
for candidates already rejected by judge/local precheck. Rejected rows should now
be persisted quickly instead of leaving 0-byte proposal files.

`2a923e5` then showed the process could still time out before the first proposal
heartbeat. That means the stall was earlier, inside local group/candidate
construction. The latest fix moves heartbeats to CLI/function/group entry,
caps local pair comparisons, and removes video near-duplicate probing from
candidate construction. Near-duplicate/video checks should happen only after
cheap annotation gates and Omni judge have narrowed the candidate set.

## Mannul Boundary Table

| Pair | Example edit | Route decision | Main reason |
| --- | --- | --- | --- |
| `pair_00` | `Turn their marriage into a nice holiday` | Reject / diagnostic | Broad semantic shift; likely requires many visual changes |
| `pair_01` | `make the plane flying through turbulent air` | Reject / diagnostic | Loose scene/action shift with modest video similarity |
| `pair_02` | `make the fields green` | Candidate | Clear attribute/scene-color change with related context |
| `pair_03` | `make it like a cancun beach` | Reject / diagnostic | Broad scene restyle; edit text is loose |
| `pair_04` | `put a landmark symbol` | Candidate with caution | Template/icon pair; needs concrete object evidence |
| `pair_05` | `make the school` | Reject | Bad imperative edit text |
| `pair_06` | `make it a touristy country in Canada` | Reject / diagnostic | Country/template shift; broad and text/logo prone |
| `pair_07` | `change it to a cloud` | Candidate with caution | Template/icon change, acceptable if target uniqueness is real |
| `pair_08` | `made in macedonia` | Reject unless repaired | Visible-text edit must include from/to OCR evidence |
| `pair_09` | `make it a bell` | Reject if no frame-backed delta | Near-duplicate risk; needs observable icon delta |

## Next Smoke

Run only the natural pair pipeline. Do not run VACE, masks, or src_ref generation.

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main || exit 1
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/natural_pair_omni_detective_$(date +%Y%m%d_%H%M%S)
ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
MODEL=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
BASE_URL=http://127.0.0.1:8093/v1

mkdir -p "$RUN_ROOT/logs"
nohup bash scripts/run_omni_detective_pilot.sh \
  --root "$ROOT" \
  --run-root "$RUN_ROOT" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --max-source-videos 120 \
  --segment-seconds 8 \
  --concurrency 1 \
  --max-accepted-pairs 20 \
  --max-proposals 120 \
  --annotation-max-passes 5 \
  --annotation-pass-timeout-seconds 900 \
  --propose-timeout-seconds 600 \
  --pair-request-timeout-seconds 90 \
  --model-stage instruct \
  > "$RUN_ROOT/logs/omni_detective_pair.log" 2>&1 &
```

If annotation stops partway through, rerun the same command with the same
`RUN_ROOT`. `detective_annotations.jsonl` is treated as a resume cache, and the
script now retries annotation passes until the unique `clip_id` count matches
`extracted_event_clips.jsonl` or `--annotation-max-passes` is exhausted.

If planning and extraction are already done, resume directly from a later
stage. For example:

```bash
nohup bash scripts/run_omni_detective_pilot.sh \
  --root "$ROOT" \
  --run-root "$RUN_ROOT" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --concurrency 1 \
  --max-accepted-pairs 20 \
  --max-proposals 120 \
  --annotation-max-passes 5 \
  --annotation-pass-timeout-seconds 900 \
  --propose-timeout-seconds 600 \
  --pair-request-timeout-seconds 90 \
  --start-stage annotate \
  --model-stage instruct \
  > "$RUN_ROOT/logs/omni_detective_resume.log" 2>&1 &
```

Report back:

- `git rev-parse --short HEAD`
- test summary
- `accepted_pairs.jsonl` count
- `pilot_review.md`
- top rejection buckets from `judged_pair_proposals.jsonl`
- `manual_review_bundle` file list

## Acceptance Target

- At least 10 accepted natural pairs in the smoke run.
- Manual review pass rate at least 60%.
- `same_context_avg >= 0.75`.
- At least 2 audio-related samples.
- At least 2 object/action/attribute samples.
