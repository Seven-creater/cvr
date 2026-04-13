from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved run traces")
    parser.add_argument("--input", required=True, help="Path to a run jsonl file")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def query_type(row: dict[str, Any]) -> str:
    query = row.get("query", {})
    has_audio = bool(query.get("required_audio_tags"))
    has_object = bool(query.get("required_objects"))
    has_temporal = bool(query.get("required_temporal"))

    active = [
        name
        for name, flag in (
            ("audio", has_audio),
            ("object", has_object),
            ("temporal", has_temporal),
        )
        if flag
    ]
    if not active:
        return "other"
    if len(active) == 1:
        return active[0]
    return "+".join(active)


def final_comparison(row: dict[str, Any]) -> dict[str, Any]:
    final_candidate_id = row.get("final_candidate_id")
    if not final_candidate_id:
        return {}

    for round_row in reversed(row.get("rounds", [])):
        comparisons = round_row.get("comparisons", {}) or {}
        if final_candidate_id in comparisons:
            return comparisons[final_candidate_id] or {}
    return {}


def error_labels(row: dict[str, Any]) -> list[str]:
    if row.get("success") is True:
        return []

    labels: list[str] = []
    comparison = final_comparison(row)
    missing = set(comparison.get("missing", []))
    conflicts = set(comparison.get("conflicts", []))

    mapping = {
        "required-audio": "audio",
        "required-object": "object",
        "required-temporal": "temporal",
        "preserve-scene": "preserve",
    }
    for key, label in mapping.items():
        if key in missing:
            labels.append(f"missing_{label}")

    for conflict in sorted(conflicts):
        labels.append(f"conflict_{conflict}")

    if not labels:
        predicted = row.get("final_candidate_id")
        target = row.get("query", {}).get("target_video_id")
        if predicted and target and predicted != target:
            labels.append("wrong_candidate")
        else:
            labels.append("unknown")

    return labels


def print_type_breakdown(rows: list[dict[str, Any]]) -> None:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[query_type(row)].append(row)

    print("type_breakdown:")
    for bucket in sorted(buckets):
        items = buckets[bucket]
        success = sum(1 for row in items if row.get("success") is True)
        print(f"  {bucket}: count={len(items)} success_rate={success / max(1, len(items)):.3f}")


def print_error_breakdown(rows: list[dict[str, Any]]) -> None:
    failed_rows = [row for row in rows if row.get("success") is not True]
    counter: Counter[str] = Counter()
    for row in failed_rows:
        counter.update(error_labels(row))

    print("error_breakdown:")
    if not counter:
        print("  none")
        return

    for label, count in counter.most_common():
        print(f"  {label}: {count}")


def print_failed_cases(rows: list[dict[str, Any]]) -> None:
    failed_rows = [row for row in rows if row.get("success") is not True]
    print("failed_cases:")
    if not failed_rows:
        print("  none")
        return

    for row in failed_rows:
        labels = ",".join(error_labels(row))
        print(
            "  "
            f"query={row.get('query', {}).get('query_id')} "
            f"type={query_type(row)} "
            f"target={row.get('query', {}).get('target_video_id')} "
            f"final={row.get('final_candidate_id')} "
            f"errors={labels}"
        )


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    rows = load_rows(path)

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
    print_type_breakdown(rows)
    print_error_breakdown(rows)
    print_failed_cases(rows)


if __name__ == "__main__":
    main()
