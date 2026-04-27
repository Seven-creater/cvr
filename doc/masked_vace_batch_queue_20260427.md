# Mask-Guided VACE Batch Queue

## Why

Single-sample operation is too slow because it serializes four expensive phases:

1. Omni understands the reference video and writes an edit plan.
2. Florence-2/SAM2.1 generates the mask.
3. VACE-14B generates the edited target.
4. Omni annotates and verifies the result.

The efficient route is to keep Omni loaded on GPU 0,1 and treat it as a planning and validation service, while GPU 6 handles masks and GPU 2,3,4,5 handle VACE. Omni should understand a batch of short clips once, cache the resulting plans, and then VACE should consume the queue.

## Fixed Resource Split

| Stage | GPU | Notes |
|---|---:|---|
| Omni planning / annotation / validation | 0,1 | Keep service running on port 8093. Do not restart between samples. |
| Mask generation | 6 | Florence-2 + SAM2.1. No VACE here. |
| VACE-14B generation | 2,3,4,5 | `torchrun`, `ulysses_size=4`, no CPU offload. |
| Manual review | CPU/file copy | No model required. |

## Queue Stages

The new orchestrator is:

```bash
scripts/run_masked_vace_pipeline_queue.sh
```

It does not start or stop Omni. It expects the Qwen3-Omni service to already be available at `http://127.0.0.1:8093/v1`.

Stages:

```text
plan      Omni creates video_edit_plan.jsonl and video_mask_plan.jsonl
mask      Florence-2/SAM2.1 creates mask videos
vace      VACE-14B creates edited targets and remuxes reference audio
annotate  Omni annotates generated targets
validate  Omni validates known pairs
bundle    Creates manual_review_bundle
all       Runs all stages sequentially
```

## Recommended Usage

For maximum throughput, run one large planning job first:

```bash
bash scripts/run_masked_vace_pipeline_queue.sh \
  --stage plan \
  --source-run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/omni_detective_prompt_gate_fix_20260424 \
  --run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/masked_vace_queue_20260427 \
  --max-plans 30
```

Then generate masks on GPU 6:

```bash
bash scripts/run_masked_vace_pipeline_queue.sh \
  --stage mask \
  --run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/masked_vace_queue_20260427 \
  --mask-gpu-ids 6
```

Then run VACE-14B on GPU 2,3,4,5:

```bash
bash scripts/run_masked_vace_pipeline_queue.sh \
  --stage vace \
  --run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/masked_vace_queue_20260427 \
  --vace-gpu-ids 2,3,4,5 \
  --vace-top-k 5
```

Then annotate, validate, and bundle:

```bash
bash scripts/run_masked_vace_pipeline_queue.sh \
  --stage annotate \
  --run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/masked_vace_queue_20260427

bash scripts/run_masked_vace_pipeline_queue.sh \
  --stage validate \
  --run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/masked_vace_queue_20260427

bash scripts/run_masked_vace_pipeline_queue.sh \
  --stage bundle \
  --run-root /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/masked_vace_queue_20260427
```

## Parallel Pattern

Once the first queue is in VACE stage, Omni can plan the next shard in a different `RUN_ROOT` because VACE uses GPU 2-5 and Omni stays on GPU 0,1:

```text
Terminal A: VACE consumes queue_001 on GPU 2,3,4,5.
Terminal B: Omni plans queue_002 on GPU 0,1.
Terminal C: mask generation prepares queue_002 on GPU 6.
```

This is the high-throughput pattern. It avoids restarting Omni and keeps VACE busy.

## Current Best Visual Edit Type

The proven route is:

```text
existing subject attribute/color/material change
```

Examples:

```text
robot body black/gold -> bright yellow
tote bag red -> dark navy blue
dump truck yellow -> bright orange
robot black/gray -> metallic silver
```

Do not spend VACE time on naked object insertion yet. For structure-level edits such as cup-to-bottle, phone-to-tablet, remove glasses, or background replacement, require strong masks and treat them as the next benchmark, not the current production path.
