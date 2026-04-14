from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved run traces")
    parser.add_argument("--input", required=True, help="Path or glob for run jsonl files")
    parser.add_argument(
        "--recall-k",
        default="1,3,5",
        help="Comma-separated K values for Recall@K reporting (default: 1,3,5)",
    )
    parser.add_argument("--report-output", help="Optional text report output path")
    parser.add_argument("--json-output", help="Optional JSON summary output path")
    parser.add_argument("--markdown-output", help="Optional Markdown summary output path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_input_paths(value: str) -> list[Path]:
    direct = Path(value)
    if direct.exists():
        return [direct]

    matches = sorted(Path().glob(value))
    if matches:
        return [path for path in matches if path.is_file()]
    raise FileNotFoundError(f"no input matched: {value}")


def parse_recall_ks(value: str) -> list[int]:
    ks: list[int] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        k = int(item)
        if k <= 0:
            raise ValueError(f"Recall@K must use positive integers, got: {k}")
        ks.append(k)

    if not ks:
        raise ValueError("At least one Recall@K value is required")
    return sorted(set(ks))


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


def compute_recall_at_k(
    rows: list[dict[str, Any]],
    ks: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    first_round_hits = {k: 0 for k in ks}
    any_round_hits = {k: 0 for k in ks}
    total = len(rows)

    for row in rows:
        target = row.get("query", {}).get("target_video_id")
        rounds = row.get("rounds", [])
        if not target or not rounds:
            continue

        first_round = rounds[0]
        first_retrieved = first_round.get("retrieved_candidates", [])
        for k in ks:
            if target in first_retrieved[:k]:
                first_round_hits[k] += 1
            if any(target in round_row.get("retrieved_candidates", [])[:k] for round_row in rounds):
                any_round_hits[k] += 1

    first_round_recall = {
        k: first_round_hits[k] / max(1, total)
        for k in ks
    }
    any_round_recall = {
        k: any_round_hits[k] / max(1, total)
        for k in ks
    }
    return first_round_recall, any_round_recall


def print_recall_block(title: str, values: dict[int, float]) -> list[str]:
    lines = [f"{title}:"]
    print(title + ":")
    for k in sorted(values):
        line = f"  R@{k}={values[k]:.3f}"
        lines.append(line)
        print(line)
    return lines


def summarize_rows(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    avg_rounds = sum(len(row.get("rounds", [])) for row in rows) / max(1, total)
    avg_tool_calls = sum(len(row.get("tool_history", [])) for row in rows) / max(1, total)
    first_round_recall, any_round_recall = compute_recall_at_k(rows, ks)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[query_type(row)].append(row)

    type_breakdown = {
        bucket: {
            "count": len(items),
            "success_rate": round(
                sum(1 for item in items if item.get("success") is True) / max(1, len(items)),
                3,
            ),
        }
        for bucket, items in sorted(buckets.items())
    }

    failed_rows = [row for row in rows if row.get("success") is not True]
    error_counter: Counter[str] = Counter()
    failed_cases: list[dict[str, Any]] = []
    for row in failed_rows:
        labels = error_labels(row)
        error_counter.update(labels)
        failed_cases.append(
            {
                "query_id": row.get("query", {}).get("query_id"),
                "type": query_type(row),
                "target": row.get("query", {}).get("target_video_id"),
                "final": row.get("final_candidate_id"),
                "errors": labels,
            }
        )

    return {
        "runs": total,
        "success_rate": round(successes / max(1, total), 3),
        "avg_rounds": round(avg_rounds, 2),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "first_round_top1": round(first_round_recall.get(1, 0.0), 3),
        "first_round_recall": {f"R@{k}": round(v, 3) for k, v in first_round_recall.items()},
        "any_round_recall": {f"R@{k}": round(v, 3) for k, v in any_round_recall.items()},
        "type_breakdown": type_breakdown,
        "error_breakdown": dict(error_counter),
        "failed_cases": failed_cases,
    }


def print_type_breakdown(summary: dict[str, Any]) -> None:
    buckets = summary.get("type_breakdown", {})
    print("type_breakdown:")
    if not buckets:
        print("  none")
        return
    for bucket in sorted(buckets):
        item = buckets[bucket]
        print(f"  {bucket}: count={item['count']} success_rate={item['success_rate']:.3f}")


def print_error_breakdown(summary: dict[str, Any]) -> None:
    counter = summary.get("error_breakdown", {})
    print("error_breakdown:")
    if not counter:
        print("  none")
        return

    for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label}: {count}")


def print_failed_cases(summary: dict[str, Any]) -> None:
    failed_rows = summary.get("failed_cases", [])
    print("failed_cases:")
    if not failed_rows:
        print("  none")
        return

    for row in failed_rows:
        labels = ",".join(row.get("errors", []))
        print(
            "  "
            f"query={row.get('query_id')} "
            f"type={row.get('type')} "
            f"target={row.get('target')} "
            f"final={row.get('final')} "
            f"errors={labels}"
        )


def build_text_lines(input_paths: list[Path], summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    lines = [
        f"inputs={len(input_paths)}",
        f"runs={summary['runs']}",
        f"success_rate={summary['success_rate']:.3f}",
        f"avg_rounds={summary['avg_rounds']:.2f}",
        f"avg_tool_calls={summary['avg_tool_calls']:.2f}",
        f"first_round_top1={summary['first_round_top1']:.3f}",
    ]
    first_round_recall = {
        int(key.split("@", 1)[1]): value
        for key, value in summary["first_round_recall"].items()
    }
    any_round_recall = {
        int(key.split("@", 1)[1]): value
        for key, value in summary["any_round_recall"].items()
    }
    recall_lines = []
    recall_lines.extend(print_recall_block("first_round_recall", first_round_recall))
    recall_lines.extend(print_recall_block("any_round_recall", any_round_recall))
    return lines, recall_lines


def render_markdown_summary(input_paths: list[Path], summary: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        "## Headline Metrics",
        "",
        f"- inputs: `{len(input_paths)}`",
        f"- runs: `{summary['runs']}`",
        f"- success_rate: `{summary['success_rate']}`",
        f"- avg_rounds: `{summary['avg_rounds']}`",
        f"- avg_tool_calls: `{summary['avg_tool_calls']}`",
        f"- first_round_top1: `{summary['first_round_top1']}`",
        "",
        "## Recall",
        "",
        f"- first_round_recall: `{summary['first_round_recall']}`",
        f"- any_round_recall: `{summary['any_round_recall']}`",
        "",
        "## Type Breakdown",
        "",
    ]
    if summary["type_breakdown"]:
        for bucket, item in summary["type_breakdown"].items():
            lines.append(
                f"- {bucket}: `count={item['count']}` `success_rate={item['success_rate']}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Error Breakdown", ""])
    if summary["error_breakdown"]:
        for label, count in summary["error_breakdown"].items():
            lines.append(f"- {label}: `{count}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Failed Cases", ""])
    if summary["failed_cases"]:
        for row in summary["failed_cases"]:
            lines.append(
                f"- query=`{row['query_id']}` type=`{row['type']}` "
                f"target=`{row['target']}` final=`{row['final']}` "
                f"errors=`{','.join(row['errors'])}`"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    paths = resolve_input_paths(args.input)
    ks = parse_recall_ks(args.recall_k)
    for path in paths:
        rows.extend(load_rows(path))

    summary = summarize_rows(rows, ks)
    lines = [
        f"inputs={len(paths)}",
        f"runs={summary['runs']}",
        f"success_rate={summary['success_rate']:.3f}",
        f"avg_rounds={summary['avg_rounds']:.2f}",
        f"avg_tool_calls={summary['avg_tool_calls']:.2f}",
        f"first_round_top1={summary['first_round_top1']:.3f}",
    ]
    for line in lines:
        print(line)
    first_round_recall = {
        int(key.split("@", 1)[1]): value
        for key, value in summary["first_round_recall"].items()
    }
    any_round_recall = {
        int(key.split("@", 1)[1]): value
        for key, value in summary["any_round_recall"].items()
    }
    recall_lines = []
    recall_lines.extend(print_recall_block("first_round_recall", first_round_recall))
    recall_lines.extend(print_recall_block("any_round_recall", any_round_recall))
    print_type_breakdown(summary)
    print_error_breakdown(summary)
    print_failed_cases(summary)

    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            for line in [*lines, *recall_lines]:
                handle.write(line + "\n")
    if args.json_output:
        payload = {
            "inputs": [str(path) for path in paths],
            "metrics": summary,
        }
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown_summary(paths, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
