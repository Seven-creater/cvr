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

Run local tests:

```bash
python -m unittest discover -s tests -v
```

Use OpenAI function calling by setting `OPENAI_API_KEY` and selecting the OpenAI planner:

```bash
set OPENAI_API_KEY=...
python -m app.demo --mode mock --planner openai --model gpt-4.1-mini
```

Artifacts are written to `runs/`.

