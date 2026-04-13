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
