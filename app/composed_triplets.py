from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any


DEFAULT_DATASET_ROOT = "/data02/usr/wangqihao/Demo/test/data"
DEFAULT_OUTPUT_ROOT = "/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs"
DEFAULT_EXPECTED_COUNT = 943


@dataclass(frozen=True)
class RefTargetEditTriplet:
    sample_id: str
    reference_video: str
    target_video: str
    edit_text: str
    reference_caption: str
    source: str
    difference_type: str
    accepted: bool | None
    final_omni_accept: bool | None
    final_omni_quality_score: float | None
    reference_clip_id: str
    target_clip_id: str


@dataclass(frozen=True)
class InvalidSample:
    sample_id: str
    sample_dir: str
    reason: str


def build_triplets(
    dataset_root: str | Path,
    *,
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
) -> tuple[list[RefTargetEditTriplet], list[InvalidSample], dict[str, Any]]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    triplets: list[RefTargetEditTriplet] = []
    invalids: list[InvalidSample] = []
    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for sample_dir in sample_dirs:
        try:
            triplets.append(_read_triplet(sample_dir))
        except Exception as exc:
            invalids.append(InvalidSample(sample_id=sample_dir.name, sample_dir=str(sample_dir), reason=str(exc)))

    summary = _build_summary(
        dataset_root=root,
        triplets=triplets,
        invalids=invalids,
        expected_count=expected_count,
        discovered_count=len(sample_dirs),
    )
    return triplets, invalids, summary


def write_triplet_outputs(
    *,
    output_dir: str | Path,
    triplets: list[RefTargetEditTriplet],
    invalids: list[InvalidSample],
    summary: dict[str, Any],
    materialize_videos: bool = True,
    link_mode: str = "symlink",
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_triplets = (
        _materialize_triplet_media(triplets, root=root, link_mode=link_mode)
        if materialize_videos
        else list(triplets)
    )
    summary = {
        **summary,
        "materialized_triplets_root": str(root / "triplets_media") if materialize_videos else "",
        "materialized_videos": bool(materialize_videos),
        "link_mode": link_mode if materialize_videos else "",
    }
    _write_text_if_changed(
        root / "triplets.jsonl",
        "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in output_triplets),
    )
    _write_triplet_csv(root / "triplets.csv", output_triplets)
    _write_text_if_changed(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if invalids:
        _write_text_if_changed(
            root / "invalid_samples.jsonl",
            "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in invalids),
        )
    else:
        invalid_path = root / "invalid_samples.jsonl"
        if invalid_path.exists():
            invalid_path.unlink()


def build_and_write_triplets(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or _default_output_dir())
    triplets, invalids, summary = build_triplets(args.dataset_root, expected_count=args.expected_count)
    summary = dict(summary)
    summary["output_dir"] = str(output_dir)
    write_triplet_outputs(
        output_dir=output_dir,
        triplets=triplets,
        invalids=invalids,
        summary=summary,
        materialize_videos=not args.no_materialize_videos,
        link_mode=args.link_mode,
    )

    errors: list[str] = []
    if invalids:
        errors.append(f"found {len(invalids)} invalid samples; see {output_dir / 'invalid_samples.jsonl'}")
    if args.expected_count is not None and len(triplets) != args.expected_count:
        errors.append(f"expected {args.expected_count} valid triplets, got {len(triplets)}")
    if errors:
        raise SystemExit("; ".join(errors))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a ref-target-edit triplet manifest from composed video samples")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--no-materialize-videos", action="store_true")
    return parser


def main() -> None:
    build_and_write_triplets(build_parser().parse_args())


def _read_triplet(sample_dir: Path) -> RefTargetEditTriplet:
    reference_video = _required_file(sample_dir / "reference.mp4")
    target_video = _required_file(sample_dir / "target.mp4")
    edit_text_path = _required_file(sample_dir / "edit_text.txt")
    info_path = _required_file(sample_dir / "info.json")

    edit_text = edit_text_path.read_text(encoding="utf-8").strip()
    if not edit_text:
        raise ValueError("edit_text.txt is empty")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    reference_caption = str(info.get("reference_caption", "")).strip()
    if not reference_caption:
        annotation_path = sample_dir / "reference_annotation.json"
        if annotation_path.exists():
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            reference_caption = str(annotation.get("summary", "")).strip()
    if not reference_caption:
        raise ValueError("missing reference_caption")

    return RefTargetEditTriplet(
        sample_id=sample_dir.name,
        reference_video=str(reference_video),
        target_video=str(target_video),
        edit_text=edit_text,
        reference_caption=reference_caption,
        source=str(info.get("source", "")),
        difference_type=str(info.get("difference_type", "")),
        accepted=_optional_bool(info.get("accepted")),
        final_omni_accept=_optional_bool(info.get("final_omni_accept")),
        final_omni_quality_score=_optional_float(info.get("final_omni_quality_score")),
        reference_clip_id=str(info.get("reference_clip_id", "")),
        target_clip_id=str(info.get("target_clip_id", "")),
    )


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"required path is not a file: {path.name}")
    return path


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_summary(
    *,
    dataset_root: Path,
    triplets: list[RefTargetEditTriplet],
    invalids: list[InvalidSample],
    expected_count: int | None,
    discovered_count: int,
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    dataset_counts = {"daily_omni": 0, "worldsense": 0, "unknown": 0}
    difference_type_counts: dict[str, int] = {}
    accepted_true = 0
    accepted_false = 0
    final_omni_accept_true = 0
    for item in triplets:
        source_key = item.source or "unknown"
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        dataset_key = _dataset_key(item)
        dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + 1
        difference_key = item.difference_type or "unknown"
        difference_type_counts[difference_key] = difference_type_counts.get(difference_key, 0) + 1
        if item.accepted is True:
            accepted_true += 1
        elif item.accepted is False:
            accepted_false += 1
        if item.final_omni_accept is True:
            final_omni_accept_true += 1

    return {
        "dataset_root": str(dataset_root),
        "expected_count": expected_count,
        "discovered_sample_dirs": discovered_count,
        "valid_triplets": len(triplets),
        "invalid_samples": len(invalids),
        "source_counts": dict(sorted(source_counts.items())),
        "dataset_counts": dataset_counts,
        "difference_type_counts": dict(sorted(difference_type_counts.items())),
        "accepted_true": accepted_true,
        "accepted_false": accepted_false,
        "cap_exceeded_or_not_accepted": accepted_false,
        "final_omni_accept_true": final_omni_accept_true,
    }


def _dataset_key(item: RefTargetEditTriplet) -> str:
    text = f"{item.sample_id} {item.source}".lower()
    if "worldsense" in text:
        return "worldsense"
    if "daily_omni" in text:
        return "daily_omni"
    return "unknown"


def _write_triplet_csv(path: Path, triplets: list[RefTargetEditTriplet]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RefTargetEditTriplet.__dataclass_fields__.keys()))
        writer.writeheader()
        for item in triplets:
            writer.writerow(asdict(item))
    if path.exists() and path.read_text(encoding="utf-8") == temp_path.read_text(encoding="utf-8"):
        temp_path.unlink()
    else:
        temp_path.replace(path)


def _materialize_triplet_media(
    triplets: list[RefTargetEditTriplet],
    *,
    root: Path,
    link_mode: str,
) -> list[RefTargetEditTriplet]:
    materialized: list[RefTargetEditTriplet] = []
    media_root = root / "triplets_media"
    for item in triplets:
        sample_root = media_root / item.sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        reference_dst = sample_root / "reference.mp4"
        target_dst = sample_root / "target.mp4"
        _materialize_file(Path(item.reference_video), reference_dst, mode=link_mode)
        _materialize_file(Path(item.target_video), target_dst, mode=link_mode)
        _write_text_if_changed(sample_root / "edit_text.txt", item.edit_text + "\n")
        materialized.append(
            replace(
                item,
                reference_video=str(reference_dst),
                target_video=str(target_dst),
            )
        )
    return materialized


def _materialize_file(src: Path, dst: Path, *, mode: str) -> None:
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if _materialized_matches(src, dst):
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _materialized_matches(src: Path, dst: Path) -> bool:
    try:
        if dst.is_symlink():
            return dst.resolve() == src
        if os.path.samefile(src, dst):
            return True
        src_stat = src.stat()
        dst_stat = dst.stat()
        return dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime_ns == src_stat.st_mtime_ns
    except OSError:
        return False


def _write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _default_output_dir() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{DEFAULT_OUTPUT_ROOT}/composed_triplets_full_{stamp}"


if __name__ == "__main__":
    main()
