from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.paper_metrics import MSRVTT_REFERENCE_RESULTS, get_reference_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MSR-VTT T2V-style results against paper references"
    )
    parser.add_argument("--summary", required=True, help="Path to alignment summary JSON or eval JSON summary")
    parser.add_argument(
        "--profiles",
        default="adaptive",
        help="Comma-separated profile names when reading alignment_suite summary.json",
    )
    parser.add_argument(
        "--paper-reference",
        default="avigate_paper",
        choices=sorted(MSRVTT_REFERENCE_RESULTS),
        help="Built-in paper/reference result to compare against",
    )
    parser.add_argument(
        "--include-reproduction",
        action="store_true",
        help="Also include the built-in local AVIGATE reproduction row",
    )
    parser.add_argument("--method-label", default="Ours")
    parser.add_argument("--output-md", help="Optional Markdown output path")
    parser.add_argument("--output-json", help="Optional JSON output path")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")
    return payload


def _profile_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one profile must be provided")
    return names


def _extract_metric_block(payload: dict[str, Any], profiles: list[str]) -> list[dict[str, Any]]:
    if "profiles" in payload:
        rows = []
        available = {
            item["profile"]: item
            for item in payload.get("profiles", [])
            if isinstance(item, dict) and "profile" in item
        }
        for profile in profiles:
            if profile not in available:
                existing = ", ".join(sorted(available))
                raise KeyError(f"profile {profile!r} not found; available: {existing}")
            rows.append(available[profile])
        return rows

    metrics = payload.get("metrics", payload)
    return [
        {
            "profile": "default",
            "planner_name": payload.get("planner_name", "unknown"),
            "metrics": metrics,
        }
    ]


def _metric_value(metric_block: dict[str, Any], prefix: str, k: int) -> float | None:
    recall_block = metric_block.get(prefix, {})
    if not isinstance(recall_block, dict):
        return None
    value = recall_block.get(f"R@{k}")
    if value is None:
        return None
    return round(float(value) * 100.0, 1)


def build_comparison_payload(
    summary_payload: dict[str, Any],
    *,
    profiles: list[str],
    method_label: str,
    paper_reference: str,
    include_reproduction: bool,
) -> dict[str, Any]:
    selected_profiles = _extract_metric_block(summary_payload, profiles)
    references = [get_reference_result(paper_reference)]
    if include_reproduction and paper_reference != "avigate_reproduction":
        references.append(get_reference_result("avigate_reproduction"))

    our_rows: list[dict[str, Any]] = []
    for item in selected_profiles:
        metrics = item["metrics"]
        profile = item["profile"]
        planner_name = item.get("planner_name")
        for view_name, metric_prefix in (
            ("Round-1", "first_round_recall"),
            ("Any-round", "any_round_recall"),
        ):
            our_rows.append(
                {
                    "name": f"{method_label} ({profile}, {view_name})",
                    "profile": profile,
                    "planner_name": planner_name,
                    "view": view_name,
                    "t2v": {
                        "R@1": _metric_value(metrics, metric_prefix, 1),
                        "R@5": _metric_value(metrics, metric_prefix, 5),
                        "R@10": _metric_value(metrics, metric_prefix, 10),
                    },
                    "success_rate": round(float(metrics.get("success_rate", 0.0)) * 100.0, 1),
                    "avg_rounds": metrics.get("avg_rounds"),
                    "avg_tool_calls": metrics.get("avg_tool_calls"),
                    "first_round_top1": round(float(metrics.get("first_round_top1", 0.0)) * 100.0, 1),
                    "type_breakdown": metrics.get("type_breakdown", {}),
                }
            )

    return {
        "dataset": "MSR-VTT",
        "paper_reference": references[0],
        "reference_rows": references,
        "ours_rows": our_rows,
        "notes": [
            "Paper rows use standard MSR-VTT T2V retrieval metrics.",
            "Our rows map replay recall to T2V-style columns for side-by-side reporting.",
            "V2T is intentionally left blank for our method because the current benchmark only evaluates query-to-video retrieval.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MSR-VTT T2V Comparison",
        "",
        f"- dataset: `{payload['dataset']}`",
        f"- primary_reference: `{payload['paper_reference']['name']}`",
        "",
        "## T2V Table",
        "",
        "| Method | T2V R@1 | T2V R@5 | T2V R@10 | V2T R@1 | V2T R@5 | V2T R@10 | RSum | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in payload["reference_rows"]:
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['t2v']['R@1']:.1f} | {row['t2v']['R@5']:.1f} | {row['t2v']['R@10']:.1f} | "
            f"{row['v2t']['R@1']:.1f} | {row['v2t']['R@5']:.1f} | {row['v2t']['R@10']:.1f} | "
            f"{row['rsum']:.1f} | {row['source']} |"
        )

    for row in payload["ours_rows"]:
        t2v = row["t2v"]
        lines.append(
            "| "
            f"{row['name']} | "
            f"{_fmt_metric(t2v['R@1'])} | {_fmt_metric(t2v['R@5'])} | {_fmt_metric(t2v['R@10'])} | "
            "- | - | - | - | "
            f"success={row['success_rate']:.1f}%, avg_rounds={row['avg_rounds']}, avg_tools={row['avg_tool_calls']} |"
        )

    lines.extend(["", "## Notes", ""])
    for note in payload["notes"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Type Breakdown", ""])
    for row in payload["ours_rows"]:
        lines.append(f"### {row['name']}")
        breakdown = row.get("type_breakdown", {})
        if breakdown:
            for bucket, item in breakdown.items():
                lines.append(
                    f"- {bucket}: `count={item['count']}` `success_rate={float(item['success_rate']) * 100.0:.1f}%`"
                )
        else:
            lines.append("- none")

    return "\n".join(lines)


def _fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def main() -> None:
    args = parse_args()
    summary_payload = load_json(args.summary)
    comparison = build_comparison_payload(
        summary_payload,
        profiles=_profile_names(args.profiles),
        method_label=args.method_label,
        paper_reference=args.paper_reference,
        include_reproduction=args.include_reproduction,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = render_markdown(comparison)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if args.output_md:
        print(f"output_md={Path(args.output_md).resolve()}")
    if args.output_json:
        print(f"output_json={Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
