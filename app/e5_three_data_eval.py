from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DATASET_ORDER = ("cvr_943", "a_line", "b_line")
DATASET_LABELS = {
    "cvr_943": "原CVR 943",
    "a_line": "Line-A",
    "b_line": "Line-B",
    "overall": "Overall",
}
DEFAULT_RECALL_KS = (1, 5, 10)
DEFAULT_MODE_CONFIGS = (
    {
        "name": "vta_audio_on",
        "label": "V + T + A",
        "input_mode": "V + T + A",
        "description": "全模态基线",
        "run_dir": "vta_audio_on",
    },
    {
        "name": "vt_audio_off",
        "label": "V + T",
        "input_mode": "V + T",
        "description": "关闭 query 和 target gallery 的视频音频，判断声音是否必要",
        "run_dir": "vt_audio_off",
    },
    {
        "name": "va_video_only_audio_on",
        "label": "V + A",
        "input_mode": "V + A",
        "description": "去掉 edit_text，只保留 reference video 和 audio",
        "run_dir": "va_video_only_audio_on",
    },
)


def load_three_data_metadata(triplets_jsonl: str | Path) -> dict[str, dict[str, str]]:
    path = Path(triplets_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"triplets jsonl not found: {path}")
    records: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        sample_id = str(payload.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"line {line_number} missing sample_id")
        dataset = str(payload.get("dataset", "")).strip() or "unknown"
        records[sample_id] = {
            "dataset": dataset,
            "modality": str(payload.get("modality", "")).strip(),
            "original_sample_id": str(payload.get("original_sample_id", "")).strip(),
        }
    if not records:
        raise ValueError(f"triplets jsonl is empty: {path}")
    return records


def summarize_traces_by_dataset(
    *,
    triplets_jsonl: str | Path,
    traces_jsonl: str | Path,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    dataset_order: tuple[str, ...] = DATASET_ORDER,
) -> dict[str, Any]:
    metadata = load_three_data_metadata(triplets_jsonl)
    traces_path = Path(traces_jsonl)
    if not traces_path.exists():
        raise FileNotFoundError(f"traces jsonl not found: {traces_path}")
    recall_ks = _normalize_ks(recall_ks)
    hit_counts: dict[str, Counter[int]] = {}
    query_counts: Counter[str] = Counter()
    missing_sample_ids: list[str] = []

    for line_number, line in enumerate(traces_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        trace = json.loads(line)
        sample_id = str(trace.get("sample_id", "")).strip()
        if sample_id not in metadata:
            missing_sample_ids.append(sample_id or f"line_{line_number}")
            continue
        dataset = metadata[sample_id]["dataset"]
        rank = int(trace.get("target_rank", 10**9))
        query_counts[dataset] += 1
        query_counts["overall"] += 1
        hit_counts.setdefault(dataset, Counter())
        hit_counts.setdefault("overall", Counter())
        for k in recall_ks:
            if rank <= k:
                hit_counts[dataset][k] += 1
                hit_counts["overall"][k] += 1

    if missing_sample_ids:
        raise ValueError(f"{len(missing_sample_ids)} trace rows are missing from triplets metadata, e.g. {missing_sample_ids[:3]}")

    group_names = list(dataset_order)
    extras = sorted(name for name in query_counts if name not in set(group_names) | {"overall"})
    group_names.extend(extras)
    group_names.append("overall")
    groups = {
        name: {
            "dataset": name,
            "label": DATASET_LABELS.get(name, name),
            "query_count": int(query_counts.get(name, 0)),
            "recall": _recall_payload(hit_counts.get(name, Counter()), int(query_counts.get(name, 0)), recall_ks),
        }
        for name in group_names
        if name == "overall" or query_counts.get(name, 0) > 0
    }
    return {
        "traces_jsonl": str(traces_path),
        "total_query_count": int(query_counts.get("overall", 0)),
        "recall_ks": list(recall_ks),
        "groups": groups,
    }


def build_three_data_comparison(
    *,
    triplets_jsonl: str | Path,
    run_root: str | Path,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    mode_configs: tuple[dict[str, str], ...] = DEFAULT_MODE_CONFIGS,
) -> dict[str, Any]:
    root = Path(run_root)
    metadata = load_three_data_metadata(triplets_jsonl)
    full_dir_name = f"full{len(metadata)}"
    dataset_counts = Counter(record["dataset"] for record in metadata.values())
    modes: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    for config in mode_configs:
        traces_path = root / config["run_dir"] / full_dir_name / "traces.jsonl"
        grouped = summarize_traces_by_dataset(
            triplets_jsonl=triplets_jsonl,
            traces_jsonl=traces_path,
            recall_ks=recall_ks,
        )
        mode_entry = {
            **config,
            "traces_jsonl": str(traces_path),
            "groups": grouped["groups"],
        }
        modes.append(mode_entry)
        for dataset_name, group in grouped["groups"].items():
            detailed_rows.append(
                {
                    "mode": config["label"],
                    "mode_name": config["name"],
                    "dataset": dataset_name,
                    "dataset_label": group["label"],
                    "query_count": group["query_count"],
                    **group["recall"],
                }
            )

    return {
        "run_root": str(root),
        "triplets_jsonl": str(Path(triplets_jsonl)),
        "total_triplets": len(metadata),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "recall_ks": list(_normalize_ks(recall_ks)),
        "modes": modes,
        "detailed_rows": detailed_rows,
    }


def write_three_data_comparison(
    *,
    triplets_jsonl: str | Path,
    run_root: str | Path,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
) -> dict[str, Any]:
    root = Path(run_root)
    comparison = build_three_data_comparison(triplets_jsonl=triplets_jsonl, run_root=root, recall_ks=recall_ks)
    (root / "comparison_by_dataset.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "per_mode_grouped_summary.json").write_text(
        json.dumps({"modes": {mode["name"]: mode["groups"] for mode in comparison["modes"]}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "comparison_by_dataset.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize mixed three-data e5 CVR results by dataset")
    parser.add_argument("--triplets-jsonl", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--topk", default="1,5,10")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = write_three_data_comparison(
        triplets_jsonl=args.triplets_jsonl,
        run_root=args.run_root,
        recall_ks=tuple(_parse_topk(args.topk)),
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# e5 Three-Data Mixed Evaluation",
        "",
        f"- run_root: `{comparison['run_root']}`",
        f"- triplets_jsonl: `{comparison['triplets_jsonl']}`",
        f"- total_triplets: `{comparison['total_triplets']}`",
        f"- dataset_counts: `{comparison['dataset_counts']}`",
        "",
        "## 主表（R@1）",
        "",
        "| 输入模式 | 测 原CVR 943 | 测 Line-A | 测 Line-B | 这个设置证明了什么 |",
        "|---|---:|---:|---:|---|",
    ]
    for mode in comparison["modes"]:
        groups = mode["groups"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode["label"],
                    _fmt_group(groups, "cvr_943", "R@1"),
                    _fmt_group(groups, "a_line", "R@1"),
                    _fmt_group(groups, "b_line", "R@1"),
                    mode["description"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 详细表",
            "",
            "| 输入模式 | Dataset | Query Count | R@1 | R@5 | R@10 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["detailed_rows"]:
        if row["dataset"] == "overall":
            continue
        lines.append(
            f"| {row['mode']} | {row['dataset_label']} | {row['query_count']} | "
            f"{_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} |"
        )
    lines.extend(["", "## Overall", "", "| 输入模式 | Query Count | R@1 | R@5 | R@10 |", "|---|---:|---:|---:|---:|"])
    for row in comparison["detailed_rows"]:
        if row["dataset"] != "overall":
            continue
        lines.append(
            f"| {row['mode']} | {row['query_count']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} |"
        )
    return "\n".join(lines) + "\n"


def _recall_payload(hit_counts: Counter[int], query_count: int, recall_ks: tuple[int, ...]) -> dict[str, float]:
    return {f"R@{k}": round(float(hit_counts.get(k, 0)) / max(1, query_count), 4) for k in recall_ks}


def _normalize_ks(raw: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(sorted({int(k) for k in raw if int(k) > 0}))
    if not values:
        raise ValueError("topk values must contain at least one positive integer")
    return values


def _parse_topk(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _fmt_group(groups: dict[str, Any], dataset: str, metric: str) -> str:
    group = groups.get(dataset)
    if not group:
        return "-"
    return _fmt(group["recall"].get(metric))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
