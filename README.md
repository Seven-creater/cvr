# Minimal MSRVTT Retriever + Agent

This repository is intentionally rebuilt from scratch.

Current scope:

- frozen feature-based retriever
- `text -> top-k videos`
- `video -> top-k texts`
- loop agent for both `T2V` and `V2T`
- standard `MSRVTT` retrieval metrics: `R@1 / R@5 / R@10`

## Commands

Frozen retriever baseline:

```bash
python -m app.eval baseline \
  --feature-dir /path/to/features \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --topk 1,5,10
```

Single T2V agent case:

```bash
python -m app.eval agent-case \
  --mode t2v \
  --feature-dir /path/to/features \
  --query-text "a man is playing guitar on stage" \
  --text-encoder-kind hashing \
  --checker-kind mock
```

Single V2T agent case:

```bash
python -m app.eval agent-case \
  --mode v2t \
  --feature-dir /path/to/features \
  --query-video-id video7010 \
  --checker-kind mock
```

Batch agent evaluation:

```bash
python -m app.eval agent-batch \
  --mode t2v \
  --feature-dir /path/to/features \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --text-encoder-kind hashing \
  --checker-kind mock \
  --output-dir runs/t2v-agent
```

For real `T2V` rewrite on the server, replace `--text-encoder-kind hashing` with
`--text-encoder-kind subprocess --text-encoder-command "..."` so the agent uses the frozen retriever's text encoder.

Build a feature-dir from MSRVTT metadata plus external embeddings:

```bash
python -m app.build_feature_dir \
  --msrvtt-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-root /path/to/videos \
  --output-dir /path/to/features \
  --text-embeddings-in /path/to/text_embeddings.npy \
  --video-visual-embeddings-in /path/to/video_visual_embeddings.npy
```

## Feature Directory

`feature-dir` should contain:

- `text_embeddings.npy`
- `video_visual_embeddings.npy`
- `video_audio_embeddings.npy` (optional)
- `text_rows.jsonl`
- `video_rows.jsonl`
