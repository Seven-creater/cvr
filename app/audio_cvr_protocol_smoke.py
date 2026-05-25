from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


B_FILES = {
    "main": "b_main_audio_cvr_triplets.jsonl",
    "extended": "b_extended_audio_cvr_triplets.jsonl",
    "diagnostic": "b_diagnostic_audio_cvr_triplets.jsonl",
    "all": "b_all_audio_cvr_triplets.jsonl",
}
MANIFEST_FILES = (
    "audio_necessity_eval_manifest.json",
    "benchmark_quality_summary.json",
)
NEGATIVE_TYPES = (
    "reference_negative",
    "local_same_source",
    "local_fallback_visual",
    "visual_hard",
    "audio_hard",
    "asr_hard",
    "random_distractor",
)
TYPED_HARD_TYPES = ("visual_hard", "audio_hard", "asr_hard")


def summarize_data(
    *,
    run_root: str | Path,
    output_dir: str | Path,
    run_label: str = "Audio-CVR Protocol",
    human_query_sample: int = 30,
    human_negative_sample: int = 40,
) -> dict[str, Any]:
    run_path = Path(run_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_by_tier = {name: _read_jsonl(run_path / filename) for name, filename in B_FILES.items()}
    main_rows = records_by_tier["main"]
    summary = {
        "run_label": run_label,
        "run_root": str(run_path),
        "output_dir": str(output_path),
        "file_status": _file_status(run_path),
        "tier_counts": {tier: len(rows) for tier, rows in records_by_tier.items()},
        "subtype_counts": _subtype_counts(main_rows),
        "hard_negative_coverage": _hard_negative_coverage(main_rows),
        "negative_quality": _negative_quality(main_rows),
        "manual_review_required": _manual_review_summary(main_rows),
    }
    human_cases = _sample_human_review_cases(
        main_rows,
        query_limit=human_query_sample,
        negative_limit=human_negative_sample,
    )
    (output_path / "data_quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_path / "data_quality_summary.md").write_text(_data_quality_markdown(summary), encoding="utf-8")
    _write_jsonl(output_path / "human_review_cases.jsonl", human_cases)
    (output_path / "human_review_summary.md").write_text(_human_review_markdown(human_cases), encoding="utf-8")
    return summary


def summarize_evals(*, output_dir: str | Path, evals: list[str], run_label: str = "Audio-CVR Protocol") -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eval_summaries = [_load_labeled_eval(raw) for raw in evals]
    table_rows = _gallery_result_rows(eval_summaries)
    hard_rows = _hard_negative_rows(eval_summaries)
    audio_rows = _audio_necessity_rows(table_rows)
    error_rows, topk_errors = _error_rows(eval_summaries)
    summary = {
        "run_label": run_label,
        "output_dir": str(output_path),
        "evals": [{"label": item["label"], "path": str(item["path"])} for item in eval_summaries],
        "gallery_protocol_rows": table_rows,
        "audio_necessity_rows": audio_rows,
        "hard_negative_rows": hard_rows,
        "error_rows": error_rows,
        "topk_error_count": len(topk_errors),
    }
    (output_path / "protocol_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_path / "protocol_smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_path / "gallery_protocol_results.md").write_text(_gallery_results_markdown(table_rows), encoding="utf-8")
    (output_path / "audio_necessity_results.md").write_text(_audio_necessity_markdown(audio_rows), encoding="utf-8")
    (output_path / "hard_negative_breakdown.md").write_text(_hard_negative_markdown(hard_rows), encoding="utf-8")
    (output_path / "topk_errors.md").write_text(_error_markdown(error_rows), encoding="utf-8")
    _write_jsonl(output_path / "topk_errors.jsonl", topk_errors)
    (output_path / "advisor_brief.md").write_text(_advisor_brief_markdown(summary), encoding="utf-8")
    return summary


def _file_status(run_path: Path) -> dict[str, bool]:
    paths = {filename: (run_path / filename).exists() for filename in (*B_FILES.values(), *MANIFEST_FILES)}
    paths.update(
        {
            "b_main_eval_gallery_global.jsonl": (run_path / "b_main_eval_gallery_global.jsonl").exists(),
            "b_main_eval_gallery_local_same_source.jsonl": (run_path / "b_main_eval_gallery_local_same_source.jsonl").exists(),
            "b_main_eval_gallery_hardneg.jsonl": (run_path / "b_main_eval_gallery_hardneg.jsonl").exists(),
        }
    )
    return paths


def _subtype_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        subtype = _first_text(row, "b_subtype", "audio_delta_type", "audio_delta_subtype", default="unknown")
        counts[subtype] += 1
    return dict(sorted(counts.items()))


def _hard_negative_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    coverage = {kind: 0 for kind in NEGATIVE_TYPES}
    missing_reasons: dict[str, Counter[str]] = {kind: Counter() for kind in NEGATIVE_TYPES}
    for row in rows:
        negatives = _negative_items(row)
        present = {str(item.get("type") or item.get("negative_type") or "").strip() for item in negatives}
        if _first_text(row, "reference_video", "reference_path"):
            present.add("reference_negative")
        for kind in NEGATIVE_TYPES:
            if kind in present:
                coverage[kind] += 1
        missing = row.get("hard_negative_missing_reasons") or row.get("audio_delta_hard_negative_missing_reasons") or {}
        if isinstance(missing, dict):
            for kind, reason in missing.items():
                missing_reasons.setdefault(str(kind), Counter())[str(reason or "missing")] += 1
    return {
        kind: {
            "query_count": count,
            "coverage_rate": round(count / max(1, total), 4),
            "missing_reasons": dict(missing_reasons.get(kind, Counter())),
        }
        for kind, count in coverage.items()
    }


def _negative_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    uncertain = 0
    false_negative = 0
    manual_review = 0
    type_counts: Counter[str] = Counter()
    temporal_counts: Counter[str] = Counter()
    for row in rows:
        for item in _negative_items(row):
            total += 1
            neg_type = str(item.get("type") or item.get("negative_type") or "unknown")
            type_counts[neg_type] += 1
            temporal = str(item.get("temporal_relation") or "")
            if temporal:
                temporal_counts[temporal] += 1
            if str(item.get("verification_status") or "").lower() == "uncertain":
                uncertain += 1
            if _truthy(item.get("satisfies_edit")) or _truthy(item.get("verification_accept")):
                false_negative += 1
            if _truthy(item.get("manual_review_required")):
                manual_review += 1
    return {
        "negative_count": total,
        "negative_type_counts": dict(sorted(type_counts.items())),
        "local_same_source_relation_counts": dict(sorted(temporal_counts.items())),
        "uncertain_negative_count": uncertain,
        "uncertain_negative_rate": round(uncertain / max(1, total), 4),
        "false_negative_risk_count": false_negative,
        "false_negative_risk_rate": round(false_negative / max(1, total), 4),
        "manual_review_required_negative_count": manual_review,
        "manual_review_required_negative_rate": round(manual_review / max(1, total), 4),
    }


def _manual_review_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    count = sum(1 for row in rows if _truthy(row.get("manual_review_required")))
    return {"query_count": count, "query_rate": round(count / max(1, total), 4)}


def _sample_human_review_cases(rows: list[dict[str, Any]], *, query_limit: int, negative_limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    priority_rows = sorted(
        rows,
        key=lambda row: (
            not _truthy(row.get("manual_review_required")),
            -float(row.get("asr_degeneracy_risk") or 0.0),
            str(row.get("sample_id") or ""),
        ),
    )
    for row in priority_rows[: max(0, query_limit)]:
        cases.append(
            {
                "case_type": "b_main_query",
                "sample_id": row.get("sample_id"),
                "reference_video": row.get("reference_video"),
                "target_video": row.get("target_video"),
                "edit_text": row.get("edit_text"),
                "b_subtype": row.get("b_subtype") or row.get("audio_delta_type"),
                "manual_review_required": bool(_truthy(row.get("manual_review_required"))),
                "review_label": "",
                "review_notes": "",
            }
        )
    negative_cases: list[dict[str, Any]] = []
    for row in rows:
        for item in _negative_items(row):
            neg_type = str(item.get("type") or item.get("negative_type") or "")
            if neg_type in {"local_same_source", "local_fallback_visual", *TYPED_HARD_TYPES}:
                negative_cases.append(
                    {
                        "case_type": "hard_negative",
                        "sample_id": row.get("sample_id"),
                        "negative_type": neg_type,
                        "video": item.get("video"),
                        "edit_text": row.get("edit_text"),
                        "temporal_relation": item.get("temporal_relation"),
                        "verification_status": item.get("verification_status"),
                        "satisfies_edit": item.get("satisfies_edit"),
                        "review_label": "",
                        "review_notes": "",
                    }
                )
    cases.extend(negative_cases[: max(0, negative_limit)])
    return cases


def _gallery_result_rows(eval_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in eval_summaries:
        summary = item["summary"]
        for model_key, model_label, method_names in (
            ("base_e5", "Base E5", ("base_e5_global_local", "base_e5_global")),
            ("audio_delta_adapter", "Adapter", ("audio_delta_adapter_global_local", "audio_delta_adapter_global")),
        ):
            method_row = _pick_method(summary.get("rows", []), method_names)
            target_beats = (summary.get("target_beats_reference") or {}).get(model_key) or {}
            ref_rank = summary.get("base_reference_rank_summary" if model_key == "base_e5" else "reference_rank_summary") or {}
            rows.append(
                {
                    "gallery_protocol": item["label"],
                    "model": model_label,
                    "R@1": method_row.get("R@1"),
                    "R@5": method_row.get("R@5"),
                    "R@10": method_row.get("R@10"),
                    "target_beats_reference": target_beats.get("target_beats_reference_rate"),
                    "reference_rank_median": ref_rank.get("median_rank"),
                    "target_ref_gap_mean": target_beats.get("target_minus_reference_mean"),
                }
            )
    return rows


def _audio_necessity_rows(table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapter_by_label = {row["gallery_protocol"]: row for row in table_rows if row["model"] == "Adapter"}
    vt = adapter_by_label.get("V+T") or adapter_by_label.get("v+t") or {}
    vt_r1 = _maybe_float(vt.get("R@1"))
    vt_tbr = _maybe_float(vt.get("target_beats_reference"))
    rows: list[dict[str, Any]] = []
    for mode in ("T-only-fullAV", "V-only", "A-only", "V+T", "A+T", "V+A", "V+A+T"):
        row = adapter_by_label.get(mode) or {}
        r1 = _maybe_float(row.get("R@1"))
        tbr = _maybe_float(row.get("target_beats_reference"))
        rows.append(
            {
                "mode": mode,
                "R@1": row.get("R@1"),
                "R@5": row.get("R@5"),
                "R@10": row.get("R@10"),
                "target_beats_reference": row.get("target_beats_reference"),
                "target_ref_gap_mean": row.get("target_ref_gap_mean"),
                "delta_R@1_vs_V+T": round(r1 - vt_r1, 4) if r1 is not None and vt_r1 is not None else None,
                "delta_target_beats_ref_vs_V+T": round(tbr - vt_tbr, 4) if tbr is not None and vt_tbr is not None else None,
            }
        )
    return rows


def _hard_negative_rows(eval_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in eval_summaries:
        summary = item["summary"]
        for model_label, key in (("Base E5", "base_hard_negative_recall_by_type"), ("Adapter", "hard_negative_recall_by_type")):
            values = summary.get(key) or {}
            rows.append(
                {
                    "gallery_protocol": item["label"],
                    "model": model_label,
                    "positive beats reference_negative": _rate(values, "reference_negative"),
                    "positive beats local_same_source": _rate(values, "local_same_source"),
                    "positive beats visual_hard": _rate(values, "visual_hard"),
                    "positive beats audio_hard": _rate(values, "audio_hard"),
                    "positive beats asr_hard": _rate(values, "asr_hard"),
                }
            )
    return rows


def _error_rows(eval_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counter: Counter[str] = Counter()
    topk_errors: list[dict[str, Any]] = []
    for item in eval_summaries:
        path = item["path"] / "per_query_scores.jsonl"
        for row in _read_jsonl(path):
            top1 = row.get("adapter_top1") or {}
            if top1.get("is_target"):
                continue
            error_type = _error_type_from_top1(top1)
            counter[error_type] += 1
            topk_errors.append({"gallery_protocol": item["label"], "error_type": error_type, **row})
    total = sum(counter.values())
    rows = [
        {
            "error_type": key,
            "count": count,
            "percentage": round(count / max(1, total), 4),
            "typical_cause": _error_typical_cause(key),
            "next_action": _error_next_action(key),
        }
        for key, count in sorted(counter.items())
    ]
    return rows, topk_errors


def _error_type_from_top1(top1: dict[str, Any]) -> str:
    if top1.get("is_reference") or top1.get("kind") == "reference_negative":
        return "reference wins"
    kind = str(top1.get("kind") or top1.get("negative_type") or "")
    if kind in {"local_same_source", "local_fallback_visual"}:
        return "local_same_source wins"
    if kind == "visual_hard":
        return "visual_hard wins"
    if kind == "audio_hard":
        return "audio_hard wins"
    if kind == "asr_hard":
        return "asr_hard wins"
    if _truthy(top1.get("satisfies_edit")):
        return "false_negative_suspected"
    return "random wins"


def _error_typical_cause(error_type: str) -> str:
    return {
        "reference wins": "reference and target are visually close; edit direction is weak",
        "local_same_source wins": "same-source clip is close to the target context",
        "visual_hard wins": "model may still rely on visual similarity",
        "audio_hard wins": "model may follow audio cue but ignore video context",
        "asr_hard wins": "speech/topic keyword shortcut may be active",
        "false_negative_suspected": "negative may actually satisfy the edit",
        "random wins": "check gallery index, cache, or low-quality target",
    }.get(error_type, "unknown")


def _error_next_action(error_type: str) -> str:
    return {
        "reference wins": "report as directionality failure; consider later reference/delta losses",
        "local_same_source wins": "inspect false-negative risk and local hard negative quality",
        "visual_hard wins": "inspect video-only shortcut and audio evidence",
        "audio_hard wins": "check full-AV required signal",
        "asr_hard wins": "downgrade ASR-risk samples or cap speech ratio",
        "false_negative_suspected": "remove from negative gallery or send to human review",
        "random wins": "verify paths, positive indices, and media health",
    }.get(error_type, "inspect manually")


def _load_labeled_eval(raw: str) -> dict[str, Any]:
    if "=" not in raw:
        raise ValueError("--eval must be formatted as label=/path/to/eval_dir")
    label, path_text = raw.split("=", 1)
    path = Path(path_text)
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing eval summary: {summary_path}")
    return {"label": label, "path": path, "summary": json.loads(summary_path.read_text(encoding="utf-8"))}


def _pick_method(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        found = next((row for row in rows if row.get("method") == name), None)
        if found:
            return found
    return {}


def _rate(values: dict[str, Any], key: str) -> Any:
    item = values.get(key) or {}
    return item.get("positive_beats_negative_rate")


def _data_quality_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary.get('run_label', 'Audio-CVR Protocol')} Data Quality Summary",
        "",
        f"- run_root: `{summary['run_root']}`",
        "",
        "## Required Files",
        "",
        "| File | Exists |",
        "|---|---:|",
    ]
    for name, exists in summary["file_status"].items():
        lines.append(f"| `{name}` | {'yes' if exists else 'no'} |")
    lines.extend(["", "## Tier And Subtype Counts", "", "| Metric | Value |", "|---|---:|"])
    for tier, count in summary["tier_counts"].items():
        lines.append(f"| B-{tier} count | {count} |")
    for subtype, count in summary["subtype_counts"].items():
        lines.append(f"| {subtype} | {count} |")
    lines.extend(["", "## Hard Negative Coverage", "", "| Type | Query Count | Coverage |", "|---|---:|---:|"])
    for kind in NEGATIVE_TYPES:
        item = summary["hard_negative_coverage"].get(kind, {})
        lines.append(f"| {kind} | {item.get('query_count', 0)} | {_fmt(item.get('coverage_rate'))} |")
    nq = summary["negative_quality"]
    mr = summary["manual_review_required"]
    lines.extend(
        [
            "",
            "## Risk Rates",
            "",
            "| Risk | Count | Rate |",
            "|---|---:|---:|",
            f"| uncertain negative | {nq['uncertain_negative_count']} | {_fmt(nq['uncertain_negative_rate'])} |",
            f"| false-negative risk | {nq['false_negative_risk_count']} | {_fmt(nq['false_negative_risk_rate'])} |",
            f"| manual review required query | {mr['query_count']} | {_fmt(mr['query_rate'])} |",
        ]
    )
    return "\n".join(lines) + "\n"


def _human_review_markdown(cases: list[dict[str, Any]]) -> str:
    counts = Counter(str(case.get("case_type")) for case in cases)
    return "\n".join(
        [
            "# Human Review Cases",
            "",
            "Fill `review_label` with `passed`, `failed`, or `uncertain` in `human_review_cases.jsonl`.",
            "",
            "| Case Type | Count |",
            "|---|---:|",
            *[f"| {key} | {value} |" for key, value in sorted(counts.items())],
            "",
        ]
    )


def _gallery_results_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Gallery Protocol Results",
        "",
        "| gallery protocol | model | R@1 | R@5 | R@10 | target_beats_reference | reference_rank_median | target_ref_gap_mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gallery_protocol']} | {row['model']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} | "
            f"{_fmt(row.get('target_beats_reference'))} | {_fmt(row.get('reference_rank_median'))} | {_fmt(row.get('target_ref_gap_mean'))} |"
        )
    return "\n".join(lines) + "\n"


def _audio_necessity_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Audio Necessity Results",
        "",
        "| mode | R@1 | R@5 | R@10 | target_beats_reference | target_ref_gap_mean | Delta R@1 vs V+T | Delta target_beats_ref vs V+T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} | "
            f"{_fmt(row.get('target_beats_reference'))} | {_fmt(row.get('target_ref_gap_mean'))} | "
            f"{_fmt(row.get('delta_R@1_vs_V+T'))} | {_fmt(row.get('delta_target_beats_ref_vs_V+T'))} |"
        )
    return "\n".join(lines) + "\n"


def _hard_negative_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Hard Negative Breakdown",
        "",
        "| gallery protocol | model | positive beats reference_negative | positive beats local_same_source | positive beats visual_hard | positive beats audio_hard | positive beats asr_hard |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gallery_protocol']} | {row['model']} | {_fmt(row.get('positive beats reference_negative'))} | "
            f"{_fmt(row.get('positive beats local_same_source'))} | {_fmt(row.get('positive beats visual_hard'))} | "
            f"{_fmt(row.get('positive beats audio_hard'))} | {_fmt(row.get('positive beats asr_hard'))} |"
        )
    return "\n".join(lines) + "\n"


def _error_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Top-k Error Types",
        "",
        "| error_type | count | percentage | typical cause | next action |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['error_type']} | {row['count']} | {_fmt(row['percentage'])} | {row['typical_cause']} | {row['next_action']} |"
        )
    return "\n".join(lines) + "\n"


def _advisor_brief_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Advisor Brief",
            "",
            f"本次 {summary.get('run_label', 'Audio-CVR protocol')} 的目的不是追求最终模型性能，而是验证正式 protocol 是否成立。",
            "",
            "汇报时重点看三件事：",
            "",
            "1. random gallery 是否虚高；",
            "2. reference/local/typed hard negatives 是否显著提高难度；",
            "3. 加入 audio 后，V+A+T 是否相比 V+T 提升，并改善 target_beats_reference 和 target-reference score gap。",
            "",
            "如果 random 高但 reference/local 低，这不是失败，而是说明 protocol 成功暴露了 Audio-CVR 的真实难点。",
            "",
            f"- aggregated evals: `{len(summary['evals'])}`",
            f"- top-k error rows: `{summary['topk_error_count']}`",
            "",
        ]
    )


def _negative_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("audio_delta_hard_negatives") or row.get("hard_negatives") or []
    return [item for item in value if isinstance(item, dict)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(payload: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Audio-CVR protocol evaluation outputs for pilot or full-scale runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("summarize-data")
    data.add_argument("--run-root", required=True)
    data.add_argument("--output-dir", required=True)
    data.add_argument("--run-label", default="Audio-CVR Protocol")
    data.add_argument("--human-query-sample", type=int, default=30)
    data.add_argument("--human-negative-sample", type=int, default=40)

    evals = subparsers.add_parser("summarize-evals")
    evals.add_argument("--output-dir", required=True)
    evals.add_argument("--run-label", default="Audio-CVR Protocol")
    evals.add_argument("--eval", action="append", default=[], help="Labelled eval directory, formatted as label=/path/to/eval_dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "summarize-data":
        result = summarize_data(
            run_root=args.run_root,
            output_dir=args.output_dir,
            run_label=args.run_label,
            human_query_sample=args.human_query_sample,
            human_negative_sample=args.human_negative_sample,
        )
    elif args.command == "summarize-evals":
        result = summarize_evals(output_dir=args.output_dir, evals=args.eval, run_label=args.run_label)
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
