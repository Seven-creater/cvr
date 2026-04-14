# Minimal MSRVTT Retrieval

This repository is intentionally rebuilt from scratch.

Current scope:

- `text -> top-k videos`
- `video -> top-k texts`
- standard `MSRVTT` retrieval metrics: `R@1 / R@5 / R@10`

## Commands

Text to video:

```bash
python -m app.msrvtt_retrieval text2video \
  --msrvtt-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --text "a man is playing guitar on stage" \
  --topk 10
```

Video to text:

```bash
python -m app.msrvtt_retrieval video2text \
  --msrvtt-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --video-id video7010 \
  --topk 10
```

Evaluate:

```bash
python -m app.msrvtt_retrieval evaluate \
  --msrvtt-json /path/to/MSRVTT_data.json \
  --split-csv /path/to/MSRVTT_JSFUSION_test.csv \
  --topk 1,5,10
```
