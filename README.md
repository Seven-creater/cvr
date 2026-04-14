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

For a harder replay pack that favors retry-relevant and less repetitive cases:

```bash
python -m app.prepare_msrvtt_replay --msrvtt-json /path/to/MSRVTT_data.json --split-csv /path/to/MSRVTT_train.9k.csv --output-dir data/msrvtt_pilot_hard --max-queries 90 --difficulty-preset hard
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

Export RL-ready contextual bandit samples from scripted rollouts:

```bash
python -m app.export_bandit_data --mode real --config data/msrvtt_pilot_replay_filtered/real.yaml --controller-profile adaptive --output runs/msrvtt_bandit_adaptive.jsonl
```

Run a strict MSR-VTT replay evaluation with T2V-style Recall@1/5/10:

```bash
python -m app.alignment_suite --config data/msrvtt_pilot_replay_filtered/real.yaml --profiles adaptive,fixed,single-round-fixed --fixed-topk 10 --recall-k 1,5,10 --output-dir runs/msrvtt_t2v_alignment
```

Generate a Markdown/JSON comparison against AVIGATE paper results:

```bash
python -m app.compare_msrvtt --summary runs/msrvtt_t2v_alignment/summary.json --profiles adaptive --paper-reference avigate_paper --include-reproduction --method-label "Ours" --output-md runs/msrvtt_t2v_alignment/compare.md --output-json runs/msrvtt_t2v_alignment/compare.json
```

Artifacts are written to `runs/`.
