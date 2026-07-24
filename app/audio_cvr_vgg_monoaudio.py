from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HUMAN_CATEGORY = "human"
MUSIC_CATEGORY = "music"
YOUTUBE_ID_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")


def _read_metadata(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subset in ("inter_class", "intra_class"):
        metadata_path = root / subset / "metadata.csv"
        if not metadata_path.is_file():
            continue
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                normalized = {str(key): str(value or "").strip() for key, value in row.items()}
                normalized["subset"] = subset
                normalized["media_path"] = str((root / subset / normalized["file_name"]).resolve())
                rows.append(normalized)
    if not rows:
        raise ValueError(f"no VGG-MonoAudio metadata found under {root}")
    return rows


def _opposite_position(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "left":
        return "right"
    if normalized == "right":
        return "left"
    return ""


def _same_time(value_a: str, value_b: str) -> bool:
    try:
        return abs(float(value_a) - float(value_b)) <= 1e-6
    except (TypeError, ValueError):
        return False


def reversible_pairs(rows: Iterable[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_direction: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_direction[(row["target_file"], row["paired_file"])].append(row)

    output: list[tuple[dict[str, str], dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for (source_a, source_b), forward_rows in sorted(by_direction.items()):
        pair_key = tuple(sorted((source_a, source_b)))
        if not all(pair_key) or pair_key in seen:
            continue
        matches: list[tuple[dict[str, str], dict[str, str]]] = []
        for forward in forward_rows:
            expected_position = _opposite_position(forward.get("target_position", ""))
            if not expected_position:
                continue
            for reverse in by_direction.get((source_b, source_a), []):
                if reverse.get("target_position", "").lower() != expected_position:
                    continue
                if not _same_time(forward.get("target_start_sec", ""), reverse.get("paired_start_sec", "")):
                    continue
                if not _same_time(forward.get("paired_start_sec", ""), reverse.get("target_start_sec", "")):
                    continue
                if not Path(forward["media_path"]).is_file() or not Path(reverse["media_path"]).is_file():
                    continue
                matches.append((forward, reverse))
        if not matches:
            continue
        matches.sort(
            key=lambda pair: (
                0 if pair[0].get("subset") == "inter_class" else 1,
                pair[0]["media_path"],
                pair[1]["media_path"],
            )
        )
        output.append(matches[0])
        seen.add(pair_key)
    return output


def _stable_hash(*values: Any) -> str:
    payload = "|".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_row(
    forward: dict[str, str],
    reverse: dict[str, str],
) -> dict[str, Any]:
    source_a = forward["target_file"]
    source_b = forward["paired_file"]
    pair_key = tuple(sorted((source_a, source_b)))
    digest = _stable_hash("vgg_monoaudio", *pair_key)[:20]
    old_label = forward["label"].strip().lower()
    new_label = reverse["label"].strip().lower()
    old_category = forward["target_category"].strip()
    new_category = reverse["target_category"].strip()
    subtype = (
        "music"
        if old_category.lower() == MUSIC_CATEGORY and new_category.lower() == MUSIC_CATEGORY
        else "sound_event"
    )
    edit_text = f"Replace the sound of {old_label} with the sound of {new_label}."
    source_id = f"vgg_monoaudio_pair:{digest}"
    return {
        "sample_id": f"vgg_monoaudio_{digest}",
        "proposal_id": f"vgg_monoaudio_{digest}",
        "pair_group_id": f"vgg_monoaudio_pair_{digest}",
        "raw_source_id": source_id,
        "source_disjoint_group_id": source_id,
        "component_source_ids": [source_a, source_b],
        "dataset": "vgg_monoaudio",
        "b_subtype": subtype,
        "automatic_split_tier": "main",
        "accepted": True,
        "fallback": False,
        "direction": "forward",
        "is_inverse": False,
        "reference_video": forward["media_path"],
        "target_video": reverse["media_path"],
        "edit_text": edit_text,
        "old_audio": old_label,
        "new_audio": new_label,
        "audio_delta_strength": 1.0,
        "video_context_strength": 1.0,
        "vgg_monoaudio_subset": forward["subset"],
        "vgg_monoaudio_target_position": forward["target_position"],
        "vgg_monoaudio_old_category": old_category,
        "vgg_monoaudio_new_category": new_category,
        "synthetic_visual_composite": True,
        "audio_only_proposal": {
            "difference_type": "music" if subtype == "music" else "audio_event",
            "b_subtype": subtype,
            "reference_audio_content": old_label,
            "target_audio_content": new_label,
            "edit_text": edit_text,
            "annotation_source": "VGG-MonoAudio reversible clean-audio metadata",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _excluded_component_ids(paths: Iterable[str | Path]) -> set[str]:
    output: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            for match in YOUTUBE_ID_RE.finditer(value):
                output.add(match.group(1))

    for value in paths:
        path = Path(value)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    visit(json.loads(line))
    return output


def prepare_candidates(
    *,
    root: str | Path,
    output_dir: str | Path,
    exclude_jsonl_paths: Iterable[str | Path] = (),
    max_component_uses: int = 2,
    max_candidates: int = 120,
    random_seed: int = 20260723,
) -> dict[str, Any]:
    source_root = Path(root).resolve()
    output_root = Path(output_dir)
    rows = _read_metadata(source_root)
    pairs = reversible_pairs(rows)
    excluded_components = _excluded_component_ids(exclude_jsonl_paths)
    non_human_pairs = [
        pair
        for pair in pairs
        if pair[0]["target_category"].lower() != HUMAN_CATEGORY
        and pair[1]["target_category"].lower() != HUMAN_CATEGORY
        and pair[0]["target_file"][:11] not in excluded_components
        and pair[0]["paired_file"][:11] not in excluded_components
    ]
    non_human_pairs.sort(
        key=lambda pair: (
            0 if pair[0]["subset"] == "inter_class" else 1,
            _stable_hash(
                random_seed,
                *sorted((pair[0]["target_file"], pair[0]["paired_file"])),
            ),
        )
    )

    component_counts: Counter[str] = Counter()
    selected_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    component_limit = max(1, int(max_component_uses))
    for pair in non_human_pairs:
        components = (pair[0]["target_file"], pair[0]["paired_file"])
        if any(component_counts[value] >= component_limit for value in components):
            continue
        selected_pairs.append(pair)
        component_counts.update(components)
        if len(selected_pairs) >= max(0, int(max_candidates)):
            break

    all_rows = [_candidate_row(*pair) for pair in non_human_pairs]
    selected_rows = [_candidate_row(*pair) for pair in selected_pairs]
    all_path = output_root / "all_reversible_nonhuman_candidates.jsonl"
    selected_path = output_root / "review_candidates.jsonl"
    _write_jsonl(all_path, all_rows)
    _write_jsonl(selected_path, selected_rows)
    summary = {
        "protocol": "audiocvr_vgg_monoaudio_reversible_v1",
        "selection_uses_model_scores": False,
        "synthetic_visual_composite": True,
        "root": str(source_root),
        "metadata_row_count": len(rows),
        "reversible_same_layout_pair_count": len(pairs),
        "excluded_component_source_count": len(excluded_components),
        "reversible_nonhuman_pair_count": len(non_human_pairs),
        "selected_count": len(selected_rows),
        "max_component_uses": component_limit,
        "max_observed_component_uses": max(component_counts.values(), default=0),
        "unique_component_source_count": len(component_counts),
        "selected_subtypes": dict(sorted(Counter(row["b_subtype"] for row in selected_rows).items())),
        "selected_subsets": dict(
            sorted(Counter(row["vgg_monoaudio_subset"] for row in selected_rows).items())
        ),
        "selected_category_pairs": dict(
            sorted(
                Counter(
                    f"{row['vgg_monoaudio_old_category']}->{row['vgg_monoaudio_new_category']}"
                    for row in selected_rows
                ).items()
            )
        ),
        "random_seed": int(random_seed),
        "outputs": {
            "all_candidates": str(all_path),
            "review_candidates": str(selected_path),
            "summary": str(output_root / "summary.json"),
        },
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare reversible, same-layout Audio-CVR candidates from VGG-MonoAudio."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--max-component-uses", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=20260723)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_candidates(
        root=args.root,
        output_dir=args.output_dir,
        exclude_jsonl_paths=args.exclude_jsonl,
        max_component_uses=args.max_component_uses,
        max_candidates=args.max_candidates,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
