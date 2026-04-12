from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved run traces")
    parser.add_argument("--input", required=True, help="Path to a run jsonl file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    avg_rounds = sum(len(row.get("rounds", [])) for row in rows) / max(1, total)
    avg_tools = sum(len(row.get("tool_history", [])) for row in rows) / max(1, total)
    first_round_hits = 0
    for row in rows:
        target = row.get("query", {}).get("target_video_id")
        first_round = (row.get("rounds") or [{}])[0]
        retrieved = first_round.get("retrieved_candidates", [])
        if target and retrieved and retrieved[0] == target:
            first_round_hits += 1

    print(f"runs={total}")
    print(f"success_rate={successes / max(1, total):.3f}")
    print(f"avg_rounds={avg_rounds:.2f}")
    print(f"avg_tool_calls={avg_tools:.2f}")
    print(f"first_round_top1={first_round_hits / max(1, total):.3f}")


if __name__ == "__main__":
    main()

