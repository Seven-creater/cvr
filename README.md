# AVIGATE Step 1: Official Retrieval Wrapper

This repository now focuses on one thing:

- reuse the **official AVIGATE retrieval path**
- expose it as simple per-case functions
- keep the CLI small enough to validate against the paper reproduction on the server

The old `feature_dir + exported embeddings + cosine` route has been removed from the main code path because it did not reproduce the paper method.

## What is in the repo

- `app/avigate_official.py`
  - loads the AVIGATE runtime
  - caches the benchmark split
  - exposes:
    - `retrieve_videos_from_text_official(...)`
    - `retrieve_texts_from_video_official(...)`
    - `evaluate_avigate_official(...)`
- `app/avigate_vendor/`
  - the minimum vendored runtime pieces needed from the AVIGATE reproduction
  - runtime no longer depends on `tests/AVIGATE-CVPR2025/`
- `app/omni_checker.py`
  - kept for the next step, where an agent will inspect official retrieval candidates with Qwen2.5-Omni

## CLI

Overall metrics:

```bash
python -m app.eval avigate-baseline \
  --model-dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.bin \
  --data-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --audio-root /path/to/audio \
  --clip-weight /path/to/ViT-B-32.pt \
  --device cuda \
  --topk 1,5,10
```

Single text-to-video case:

```bash
python -m app.eval avigate-t2v-case \
  --model-dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.bin \
  --data-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --audio-root /path/to/audio \
  --clip-weight /path/to/ViT-B-32.pt \
  --device cuda \
  --query-text "a man is talking" \
  --topk-value 10
```

Single video-to-text case:

```bash
python -m app.eval avigate-v2t-case \
  --model-dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.bin \
  --data-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --audio-root /path/to/audio \
  --clip-weight /path/to/ViT-B-32.pt \
  --device cuda \
  --query-video-id video6513 \
  --topk-value 10
```

Single text-to-video official-agent case:

```bash
python -m app.eval avigate-t2v-agent-case \
  --model-dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.bin \
  --data-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --audio-root /path/to/audio \
  --clip-weight /path/to/ViT-B-32.pt \
  --device cuda \
  --query-text "a person is cooking" \
  --topk-value 10 \
  --checker-base-url http://127.0.0.1:8092/v1 \
  --checker-api-key EMPTY \
  --checker-model /path/to/qwen2.5-omni \
  --max-iter 3
```

Single video-to-text official-agent case:

```bash
python -m app.eval avigate-v2t-agent-case \
  --model-dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.bin \
  --data-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --audio-root /path/to/audio \
  --clip-weight /path/to/ViT-B-32.pt \
  --device cuda \
  --query-video-id video6513 \
  --topk-value 10 \
  --checker-base-url http://127.0.0.1:8092/v1 \
  --checker-api-key EMPTY \
  --checker-model /path/to/qwen2.5-omni \
  --max-iter 3
```

## Validation goal

The acceptance standard for Step 1 is:

- the per-case top-k results come from the same retrieval path as the paper reproduction
- `avigate-baseline` matches the reproduction's `R@1 / R@5 / R@10` on the server, up to small floating-point differences

## What is intentionally not in Step 1

- no exported `.npy` feature baseline
- no batch agent evaluation

Step 2 now adds only the smallest possible single-case agent loop on top of the official retrieval wrapper. Batch evaluation and broader planner logic still stay out of scope until the single-case path is validated on the server.

Small official-agent partial eval:

```bash
python -m app.eval avigate-agent-partial-eval \
  --model-dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.bin \
  --data-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --audio-root /path/to/audio \
  --clip-weight /path/to/ViT-B-32.pt \
  --device cuda \
  --mode v2t \
  --sample-size 20 \
  --topk-value 10 \
  --topk 1,5,10 \
  --checker-base-url http://127.0.0.1:8092/v1 \
  --checker-api-key EMPTY \
  --checker-model /path/to/qwen2.5-omni \
  --max-iter 3 \
  --output-dir /tmp/avigate_agent_partial_eval
```

The command appends one JSON trace per finished sample to `traces.jsonl`, rewrites `summary.json` after every sample, and prints progress lines while running.
