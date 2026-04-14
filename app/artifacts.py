from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.backends.base import PROJECT_ROOT
from app.schemas import RunTrace


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def ensure_runs_dir(path: str | Path = "runs") -> Path:
    runs_dir = Path(path)
    if not runs_dir.is_absolute():
        runs_dir = (PROJECT_ROOT / runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def write_run_artifacts(trace: RunTrace, prefix: str | None = None) -> dict[str, Path]:
    runs_dir = ensure_runs_dir()
    stem = prefix or f"{trace.query.query_id}-{_stamp()}"
    json_path = runs_dir / f"{stem}.json"
    md_path = runs_dir / f"{stem}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(trace.to_dict(), handle, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(trace))

    return {"json": json_path, "md": md_path}


def append_jsonl(traces: Iterable[RunTrace], path: str | Path | None = None) -> Path:
    runs_dir = ensure_runs_dir()
    jsonl_path = Path(path) if path else runs_dir / f"batch-{_stamp()}.jsonl"
    if not jsonl_path.is_absolute():
        jsonl_path = (PROJECT_ROOT / jsonl_path).resolve()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
    return jsonl_path


def render_markdown(trace: RunTrace) -> str:
    lines = [
        f"# Run {trace.query.query_id}",
        "",
        f"- Planner: `{trace.planner_name}`",
        f"- Planner metadata: `{trace.planner_metadata}`",
        f"- Source: `{trace.query.source_video_id}`",
        f"- Target: `{trace.query.target_video_id}`",
        f"- Final candidate: `{trace.final_candidate_id}`",
        f"- Success: `{trace.success}`",
        "",
        "## Instruction",
        "",
        trace.query.edit_instruction,
        "",
        "## Rounds",
        "",
    ]
    for item in trace.rounds:
        lines.extend(
            [
                f"### Round {item.round_index}",
                "",
                f"- Params: `{item.retrieval_params.to_dict() if item.retrieval_params else {}}`",
                f"- Retrieved: `{item.retrieved_candidates}`",
                f"- Inspected: `{item.inspected_candidates}`",
                f"- Decision: `{item.decision}`",
                f"- Notes: {item.notes or 'n/a'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Explanation",
            "",
            trace.final_explanation or "n/a",
            "",
        ]
    )
    return "\n".join(lines)
