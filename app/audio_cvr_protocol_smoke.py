from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
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


def mine_local_same_source(
    *,
    run_root: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    max_per_query: int = 5,
    manifest_paths: list[str | Path] | None = None,
    summary_output: str | Path | None = None,
    coverage_output: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_root)
    input_file = Path(input_path)
    output_file = Path(output_path)
    rows = _read_jsonl(input_file)
    manifests = [Path(path) for path in manifest_paths] if manifest_paths else _default_clip_manifest_paths(run_path)
    clip_rows = _load_clip_inventory(manifests)
    clip_rows.extend(_sibling_clip_inventory(rows))
    by_source = _index_clips_by_source(clip_rows)
    mined: list[dict[str, Any]] = []
    missing_reasons: Counter[str] = Counter()
    for row in rows:
        sample_id = _first_text(row, "sample_id", "proposal_id", "candidate_id")
        reference_video = _first_text(row, "reference_video", "reference_path")
        target_video = _first_text(row, "target_video", "target_path")
        source_ids = _candidate_source_ids(row, reference_video, target_video)
        ref_key = _media_key(reference_video)
        tgt_key = _media_key(target_video)
        ref_index = _segment_index(reference_video, _first_text(row, "reference_clip_id", "reference_id"))
        tgt_index = _segment_index(target_video, _first_text(row, "target_clip_id", "target_id"))
        strict_candidates: list[dict[str, Any]] = []
        for source_id in source_ids:
            for clip in by_source.get(source_id, []):
                video = _first_text(clip, "video", "output_path", "video_path", "clip_path", "path")
                if not video:
                    continue
                clip_key = _media_key(video)
                if clip_key in {ref_key, tgt_key}:
                    continue
                relation = _temporal_relation(clip, ref_index=ref_index, tgt_index=tgt_index)
                strict_candidates.append(
                    {
                        "sample_id": sample_id,
                        "query_sample_id": sample_id,
                        "type": "local_same_source",
                        "negative_type": "local_same_source",
                        "video": video,
                        "source_id": source_id,
                        "raw_source_id": source_id,
                        "candidate_clip_id": _first_text(clip, "clip_id", "candidate_clip_id", default=Path(video).stem),
                        "temporal_relation": relation,
                        "same_source": True,
                        "satisfies_edit": "unknown",
                        "verification_status": "candidate_unverified",
                        "manual_review_required": "true",
                        "reason": "same raw source candidate; requires false-negative guard before formal benchmark use",
                        "missing_reason": "",
                    }
                )
        strict_candidates = _dedupe_candidate_rows(strict_candidates)
        strict_candidates.sort(key=_local_candidate_sort_key)
        selected = strict_candidates[: max(0, int(max_per_query))]
        if not selected:
            fallback = _fallback_visual_candidates(row, sample_id=sample_id, max_per_query=max(0, int(max_per_query)))
            selected.extend(fallback)
            missing_reasons["no_strict_local_same_source_candidate"] += 1
        mined.extend(selected)
    _write_jsonl(output_file, mined)
    summary = _local_same_source_summary(rows, mined, missing_reasons)
    summary_path = Path(summary_output) if summary_output else output_file.with_name("local_same_source_candidate_summary.json")
    coverage_path = Path(coverage_output) if coverage_output else output_file.with_name("local_same_source_coverage.md")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage_path.write_text(_local_same_source_coverage_markdown(summary), encoding="utf-8")
    return {
        **summary,
        "run_root": str(run_path),
        "input_path": str(input_file),
        "output_path": str(output_file),
        "manifest_paths": [str(path) for path in manifests],
        "summary_output": str(summary_path),
        "coverage_output": str(coverage_path),
    }


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


def _default_clip_manifest_paths(run_path: Path) -> list[Path]:
    candidates = [
        run_path / "single_source_annotations.jsonl",
        run_path / "extracted_single_source_clips.jsonl",
        run_path / "clip_manifest.jsonl",
        run_path / "clips_manifest.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def _load_clip_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            video = _first_text(row, "output_path", "video", "video_path", "clip_path", "path")
            if not video:
                continue
            rows.append({**row, "video": video})
    return rows


def _sibling_clip_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    parents: set[Path] = set()
    for row in rows:
        for key in ("reference_video", "target_video", "reference_path", "target_path"):
            value = str(row.get(key) or "").strip()
            if not value:
                continue
            path = Path(value)
            if path.parent and path.parent.exists():
                parents.add(path.parent)
    for parent in sorted(parents):
        for video in sorted(parent.glob("*.mp4")):
            source_id = _source_id_from_path(str(video))
            inventory.append(
                {
                    "clip_id": video.stem,
                    "video": str(video),
                    "output_path": str(video),
                    "source_clip_id": source_id,
                    "raw_source_id": source_id,
                }
            )
    return inventory


def _index_clips_by_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        video = _first_text(row, "video", "output_path", "video_path", "clip_path", "path")
        if not video:
            continue
        source_id = _source_id_from_row(row, video)
        if not source_id:
            continue
        key = (source_id, _media_key(video))
        if key in seen:
            continue
        seen.add(key)
        by_source[source_id].append({**row, "video": video, "raw_source_id": source_id})
    return dict(by_source)


def _candidate_source_ids(row: dict[str, Any], reference_video: str, target_video: str) -> list[str]:
    candidates = [
        _source_id_from_row(row, reference_video),
        _source_id_from_path(reference_video),
        _source_id_from_path(target_video),
    ]
    result: list[str] = []
    for value in candidates:
        if value and value not in result:
            result.append(value)
    return result


def _source_id_from_row(row: dict[str, Any], video: str = "") -> str:
    for key in ("raw_source_id", "source_clip_id", "source_disjoint_group_id", "source_id", "group_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return _normalize_source_id(value)
    return _source_id_from_path(video)


def _source_id_from_path(video: str) -> str:
    raw = str(video or "").replace("\\", "/").strip()
    if not raw:
        return ""
    name = Path(raw).stem
    match = re.search(r"(.+?)__single_\d+", name)
    if match:
        return _normalize_source_id(match.group(1))
    parent = Path(raw).parent.name
    if parent:
        return _normalize_source_id(parent)
    return ""


def _normalize_source_id(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("single_source_"):
        text = text[len("single_source_") :]
    return text


def _segment_index(video: str, clip_id: str = "") -> int | None:
    for value in (clip_id, video):
        text = str(value or "").replace("\\", "/")
        match = re.search(r"(?:__single_|single_)(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def _temporal_relation(clip: dict[str, Any], *, ref_index: int | None, tgt_index: int | None) -> str:
    index = _segment_index(_first_text(clip, "video", "output_path", "path"), _first_text(clip, "clip_id", "candidate_clip_id"))
    anchors = [value for value in (ref_index, tgt_index) if value is not None]
    if index is None or not anchors:
        return "same_source_non_adjacent"
    nearest = min(anchors, key=lambda value: abs(index - value))
    if abs(index - nearest) == 1:
        return "adjacent_before" if index < nearest else "adjacent_after"
    return "same_source_non_adjacent"


def _dedupe_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row.get('sample_id')}|{row.get('negative_type')}|{_media_key(str(row.get('video') or ''))}"
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _local_candidate_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    relation = str(row.get("temporal_relation") or "")
    priority = {
        "adjacent_before": 0,
        "adjacent_after": 0,
        "same_source_non_adjacent": 1,
        "same_group": 2,
        "cross_source_same_context": 3,
        "visual_hard_fallback": 4,
    }.get(relation, 5)
    return priority, str(row.get("video") or "")


def _fallback_visual_candidates(row: dict[str, Any], *, sample_id: str, max_per_query: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in _negative_items(row):
        neg_type = str(item.get("type") or item.get("negative_type") or "")
        if neg_type != "visual_hard":
            continue
        video = _first_text(item, "video", "target_video", "path")
        if not video:
            continue
        candidates.append(
            {
                "sample_id": sample_id,
                "query_sample_id": sample_id,
                "type": "local_fallback_visual",
                "negative_type": "local_fallback_visual",
                "video": video,
                "source_id": _first_text(item, "source_id", "raw_source_id"),
                "raw_source_id": _first_text(item, "source_id", "raw_source_id"),
                "candidate_clip_id": _first_text(item, "candidate_clip_id", "clip_id", default=Path(video).stem),
                "temporal_relation": "visual_hard_fallback",
                "same_source": False,
                "satisfies_edit": item.get("satisfies_edit", "false"),
                "verification_status": item.get("verification_status", "auto_verified"),
                "manual_review_required": item.get("manual_review_required", ""),
                "reason": "fallback because no strict local_same_source candidate exists",
                "missing_reason": "no_strict_local_same_source_candidate",
            }
        )
        if len(candidates) >= max_per_query:
            break
    return candidates


def _local_same_source_summary(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], missing_reasons: Counter[str]) -> dict[str, Any]:
    query_count = len(rows)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_sample[str(candidate.get("sample_id") or "")].append(candidate)
    strict_samples = {
        sample_id
        for sample_id, items in by_sample.items()
        if any(str(item.get("negative_type")) == "local_same_source" for item in items)
    }
    fallback_samples = {
        sample_id
        for sample_id, items in by_sample.items()
        if any(str(item.get("negative_type")) == "local_fallback_visual" for item in items)
    }
    relation_counts: Counter[str] = Counter(str(item.get("temporal_relation") or "unknown") for item in candidates)
    verification_counts: Counter[str] = Counter(str(item.get("verification_status") or "unknown") for item in candidates)
    return {
        "query_count": query_count,
        "candidate_count": len(candidates),
        "strict_local_same_source_query_count": len(strict_samples),
        "strict_local_same_source_coverage": round(len(strict_samples) / max(1, query_count), 4),
        "local_fallback_visual_query_count": len(fallback_samples),
        "local_fallback_visual_rate": round(len(fallback_samples) / max(1, query_count), 4),
        "average_candidates_per_query": round(len(candidates) / max(1, query_count), 4),
        "temporal_relation_counts": dict(sorted(relation_counts.items())),
        "verification_status_counts": dict(sorted(verification_counts.items())),
        "missing_reasons": dict(sorted(missing_reasons.items())),
    }


def _local_same_source_coverage_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Local Same-Source Coverage",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "query_count",
        "candidate_count",
        "strict_local_same_source_query_count",
        "strict_local_same_source_coverage",
        "local_fallback_visual_query_count",
        "local_fallback_visual_rate",
        "average_candidates_per_query",
    ):
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(["", "## Temporal Relations", "", "| relation | count |", "|---|---:|"])
    for key, value in (summary.get("temporal_relation_counts") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Verification Status", "", "| status | count |", "|---|---:|"])
    for key, value in (summary.get("verification_status_counts") or {}).items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def _media_key(raw_path: str) -> str:
    return str(raw_path or "").replace("\\", "/").strip().lower()


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

    mine = subparsers.add_parser("mine-local-same-source")
    mine.add_argument("--run-root", required=True)
    mine.add_argument("--input", required=True)
    mine.add_argument("--output", required=True)
    mine.add_argument("--max-per-query", type=int, default=5)
    mine.add_argument("--manifest-path", action="append", default=[])
    mine.add_argument("--summary-output")
    mine.add_argument("--coverage-output")
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
    elif args.command == "mine-local-same-source":
        result = mine_local_same_source(
            run_root=args.run_root,
            input_path=args.input,
            output_path=args.output,
            max_per_query=args.max_per_query,
            manifest_paths=args.manifest_path or None,
            summary_output=args.summary_output,
            coverage_output=args.coverage_output,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
