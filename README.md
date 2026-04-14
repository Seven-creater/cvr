# Reflective CVR Agent Demo

This project is a minimal, training-free demo for composed video retrieval with a tool-calling agent.

## Quick start

Mock demo:

```bash
python -m app.demo --mode mock
```

Replay with file-backed sample data:

```bash
python -m app.demo --mode real --config configs/real.yaml
```

Build a pilot replay pack from raw MSR-VTT annotations:

```bash
python -m app.prepare_msrvtt_replay --msrvtt-json /path/to/MSRVTT_data.json --split-csv /path/to/MSRVTT_train.9k.csv --output-dir data/msrvtt_pilot --max-queries 90
```

If the generated pilot set is too ambiguous, tighten it with:

```bash
python -m app.prepare_msrvtt_replay --msrvtt-json /path/to/MSRVTT_data.json --split-csv /path/to/MSRVTT_train.9k.csv --output-dir data/msrvtt_pilot --max-queries 90 --min-target-margin 0.06 --max-strong-matches 1
```

For a more balanced filtered set, use the built-in preset:

```bash
python -m app.prepare_msrvtt_replay --msrvtt-json /path/to/MSRVTT_data.json --split-csv /path/to/MSRVTT_train.9k.csv --output-dir data/msrvtt_pilot_medium --max-queries 90 --difficulty-preset medium-hard
```

The replay generator now uses a decoupled generation-time discriminability scorer for query filtering, while runtime comparison still uses the backend compare logic. This keeps filtered packs from being selected by the exact same comparison function used at evaluation time.

Validate a file-backed dataset before replay:

```bash
python -m app.validate_data --candidates data/real_sample/candidates.json --queries data/real_sample/queries.json --scores data/real_sample/retrieval_scores.json
```

Run local tests:

```bash
python -m unittest discover -s tests -v
```

Use OpenAI function calling by setting `OPENAI_API_KEY` and selecting the OpenAI planner:

```bash
set OPENAI_API_KEY=...
python -m app.demo --mode mock --planner openai --model gpt-4.1-mini
```

Evaluate one or more batch logs:

```bash
python -m app.eval --input "runs/*.jsonl"
```

Artifacts are written to `runs/`.
