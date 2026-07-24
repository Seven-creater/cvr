from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MUSIC_CATEGORIES = {
    "accordion",
    "acoustic guitar",
    "banjo",
    "flute",
    "mandolin",
    "shofar",
    "ukulele",
    "violin, fiddle",
}

EDIT_TEXT_BY_CATEGORY = {
    "accordion": "Add accordion music.",
    "acoustic guitar": "Add acoustic guitar music.",
    "baby cry, infant cry": "Add the sound of a baby crying.",
    "banjo": "Add banjo music.",
    "bark": "Add the sound of a dog barking.",
    "bus": "Add the sound of a bus engine.",
    "cat": "Add the sound of a cat meowing.",
    "chainsaw": "Add the sound of a running chainsaw.",
    "church bell": "Add the sound of a church bell ringing.",
    "clock": "Add the sound of a ticking clock.",
    "fixed-wing aircraft, airplane": "Add the sound of an airplane.",
    "flute": "Add flute music.",
    "frying (food)": "Add the sound of food frying.",
    "goat": "Add the sound of a goat bleating.",
    "helicopter": "Add the sound of a helicopter.",
    "horse": "Add the sound of a horse neighing.",
    "mandolin": "Add mandolin music.",
    "motorcycle": "Add the sound of a motorcycle engine.",
    "race car, auto racing": "Add the sound of race cars.",
    "rodents, rats, mice": "Add the sounds of rodents.",
    "shofar": "Add the sound of a shofar.",
    "toilet flush": "Add the sound of a toilet flushing.",
    "train horn": "Add the sound of a train horn.",
    "truck": "Add the sound of a truck engine.",
    "ukulele": "Add ukulele music.",
    "violin, fiddle": "Add violin or fiddle music.",
}

YOUTUBE_SOURCE_RE = re.compile(
    r"(?:avatar|vggsound|ave)[_:/-](?P<video_id>[A-Za-z0-9_-]{11})(?:[_:/-]|$)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AveAnnotation:
    category: str
    video_id: str
    quality: str
    event_start: float
    event_end: float
    source_path: Path


@dataclass(frozen=True)
class BoundaryCandidate:
    annotation: AveAnnotation
    reference_start: float
    target_start: float
    reference_event_overlap: float
    target_event_overlap: float
    overlap_delta: float
    tier: str


TIER_THRESHOLDS = (
    ("high", 1.0, 3.0, 2.5),
    ("medium", 2.0, 3.0, 2.0),
    ("broad", 3.0, 2.0, 1.0),
)


def read_annotations(ave_root: str | Path) -> list[AveAnnotation]:
    root = Path(ave_root)
    annotation_path = root / "Annotations.txt"
    video_root = root / "extracted" / "videos"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"AVE annotations not found: {annotation_path}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"AVE videos not found: {video_root}")

    rows: list[AveAnnotation] = []
    for line_index, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines()):
        if line_index == 0 or not line.strip():
            continue
        parts = line.split("&")
        if len(parts) != 5:
            continue
        category, video_id, quality, start_text, end_text = parts
        source_path = video_root / f"{video_id}.mp4"
        if not source_path.is_file():
            continue
        if "speech" in category.lower():
            continue
        try:
            event_start = float(start_text)
            event_end = float(end_text)
        except ValueError:
            continue
        if event_end <= event_start:
            continue
        rows.append(
            AveAnnotation(
                category=category.strip(),
                video_id=video_id.strip(),
                quality=quality.strip(),
                event_start=event_start,
                event_end=event_end,
                source_path=source_path.resolve(),
            )
        )
    return rows


def boundary_candidate(
    annotation: AveAnnotation,
    *,
    clip_seconds: float = 6.0,
    video_seconds: float = 10.0,
    start_step: float = 1.0,
) -> BoundaryCandidate | None:
    if clip_seconds <= 0 or clip_seconds > video_seconds:
        raise ValueError("clip_seconds must be in (0, video_seconds]")
    starts: list[float] = []
    current = 0.0
    while current <= video_seconds - clip_seconds + 1e-9:
        starts.append(round(current, 6))
        current += start_step
    overlaps = [
        (
            max(
                0.0,
                min(start + clip_seconds, annotation.event_end)
                - max(start, annotation.event_start),
            ),
            start,
        )
        for start in starts
    ]
    reference_overlap, reference_start = min(overlaps, key=lambda value: (value[0], value[1]))
    target_overlap, target_start = max(overlaps, key=lambda value: (value[0], value[1]))
    delta = target_overlap - reference_overlap
    if reference_start == target_start or delta <= 0:
        return None

    tier = ""
    for name, max_reference, min_target, min_delta in TIER_THRESHOLDS:
        if (
            reference_overlap <= max_reference
            and target_overlap >= min_target
            and delta >= min_delta
        ):
            tier = name
            break
    if not tier:
        return None
    return BoundaryCandidate(
        annotation=annotation,
        reference_start=reference_start,
        target_start=target_start,
        reference_event_overlap=reference_overlap,
        target_event_overlap=target_overlap,
        overlap_delta=delta,
        tier=tier,
    )


def extract_youtube_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    output: set[str] = set()
    for row in rows:
        for value in row.values():
            if not isinstance(value, (str, Path)):
                continue
            text = str(value)
            for match in YOUTUBE_SOURCE_RE.finditer(text):
                output.add(match.group("video_id"))
            name = Path(text).stem
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", name):
                output.add(name)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json_atomic(path: Path, payload: Any) -> None:
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


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def _stable_key(candidate: BoundaryCandidate, seed: int) -> str:
    value = (
        f"{seed}|{candidate.annotation.video_id}|{candidate.reference_start}|"
        f"{candidate.target_start}|{candidate.annotation.category}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_sort_key(candidate: BoundaryCandidate, seed: int) -> tuple[Any, ...]:
    tier_rank = {"high": 0, "medium": 1, "broad": 2}
    return (
        tier_rank[candidate.tier],
        -candidate.overlap_delta,
        candidate.reference_event_overlap,
        -candidate.target_event_overlap,
        _stable_key(candidate, seed),
    )


def _clip_path(output_root: Path, video_id: str, role: str) -> Path:
    return output_root / "clips" / f"ave_{video_id}" / f"ave_{video_id}__{role}.mp4"


def _encode_clip(
    *,
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    clip_seconds: float,
    ffmpeg_bin: str,
) -> tuple[bool, str]:
    if output_path.is_file() and output_path.stat().st_size > 0:
        return True, "reused"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.mp4"
    )
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{clip_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(temp_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
            return False, completed.stderr.strip()[-1000:]
        os.replace(temp_path, output_path)
        return True, "encoded"
    finally:
        temp_path.unlink(missing_ok=True)


def _edit_text(category: str) -> str:
    normalized = category.strip().lower()
    return EDIT_TEXT_BY_CATEGORY.get(
        normalized,
        f"Add the sound event of {normalized}.",
    )


def _candidate_row(candidate: BoundaryCandidate, output_root: Path) -> dict[str, Any]:
    annotation = candidate.annotation
    subtype = "music" if annotation.category.lower() in MUSIC_CATEGORIES else "sound_event"
    edit_text = _edit_text(annotation.category)
    sample_digest = hashlib.sha256(
        (
            f"ave|{annotation.video_id}|{candidate.reference_start}|"
            f"{candidate.target_start}|{edit_text}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    source_id = f"ave:{annotation.video_id}"
    reference_path = _clip_path(output_root, annotation.video_id, "reference").resolve()
    target_path = _clip_path(output_root, annotation.video_id, "target").resolve()
    return {
        "sample_id": f"ave_boundary_{sample_digest}",
        "proposal_id": f"ave_boundary_{sample_digest}",
        "pair_group_id": f"ave_pair_{annotation.video_id}",
        "raw_source_id": source_id,
        "source_disjoint_group_id": source_id,
        "dataset": "ave",
        "b_subtype": subtype,
        "automatic_split_tier": "main",
        "accepted": True,
        "fallback": False,
        "direction": "forward",
        "is_inverse": False,
        "reference_video": str(reference_path),
        "target_video": str(target_path),
        "edit_text": edit_text,
        "old_audio": f"background audio without clearly audible {annotation.category.lower()}",
        "new_audio": annotation.category.lower(),
        "audio_delta_strength": min(1.0, 0.65 + candidate.overlap_delta / 10.0),
        "video_context_strength": 0.5,
        "ave_boundary_tier": candidate.tier,
        "ave_category": annotation.category,
        "ave_event_start": annotation.event_start,
        "ave_event_end": annotation.event_end,
        "reference_start_seconds": candidate.reference_start,
        "target_start_seconds": candidate.target_start,
        "reference_event_overlap_seconds": candidate.reference_event_overlap,
        "target_event_overlap_seconds": candidate.target_event_overlap,
        "event_overlap_delta_seconds": candidate.overlap_delta,
        "source_video": str(annotation.source_path),
        "audio_only_proposal": {
            "difference_type": "audio_event",
            "b_subtype": subtype,
            "reference_audio_content": (
                f"background audio without clearly audible {annotation.category.lower()}"
            ),
            "target_audio_content": annotation.category.lower(),
            "edit_text": edit_text,
            "annotation_source": "AVE temporal event boundary",
        },
    }


def prepare_ave_boundary_pool(
    *,
    ave_root: str | Path,
    output_dir: str | Path,
    exclude_jsonl_paths: Iterable[str | Path] = (),
    clip_seconds: float = 6.0,
    max_candidates: int = 1000,
    workers: int = 24,
    random_seed: int = 20260723,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    excluded_rows = [
        row
        for value in exclude_jsonl_paths
        for row in _read_jsonl(Path(value))
    ]
    excluded_video_ids = extract_youtube_ids(excluded_rows)
    annotations = read_annotations(ave_root)
    candidates = [
        candidate
        for annotation in annotations
        if annotation.video_id not in excluded_video_ids
        if (candidate := boundary_candidate(annotation, clip_seconds=clip_seconds)) is not None
    ]
    candidates.sort(key=lambda item: _candidate_sort_key(item, int(random_seed)))
    unique_candidates: list[BoundaryCandidate] = []
    seen_video_ids: set[str] = set()
    for candidate in candidates:
        video_id = candidate.annotation.video_id
        if video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)
        unique_candidates.append(candidate)
    selected = unique_candidates[: max(0, int(max_candidates))]

    _write_jsonl_atomic(
        output_root / "selected_boundary_candidates.jsonl",
        [
            {
                **asdict(candidate),
                "annotation": {
                    **asdict(candidate.annotation),
                    "source_path": str(candidate.annotation.source_path),
                },
            }
            for candidate in selected
        ],
    )

    failures: list[dict[str, Any]] = []
    successful_ids: set[str] = set()

    def encode(candidate: BoundaryCandidate) -> tuple[str, bool, list[str]]:
        errors: list[str] = []
        for role, start in (
            ("reference", candidate.reference_start),
            ("target", candidate.target_start),
        ):
            ok, detail = _encode_clip(
                source_path=candidate.annotation.source_path,
                output_path=_clip_path(output_root, candidate.annotation.video_id, role),
                start_seconds=start,
                clip_seconds=clip_seconds,
                ffmpeg_bin=ffmpeg_bin,
            )
            if not ok:
                errors.append(f"{role}:{detail}")
        return candidate.annotation.video_id, not errors, errors

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(encode, candidate): candidate for candidate in selected}
        for index, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            try:
                video_id, ok, errors = future.result()
            except Exception as exc:
                video_id, ok, errors = candidate.annotation.video_id, False, [
                    f"{type(exc).__name__}:{exc}"
                ]
            if ok:
                successful_ids.add(video_id)
            else:
                failures.append({"video_id": video_id, "errors": errors})
            if index % 50 == 0 or index == len(futures):
                print(
                    f"[ave-boundary] clips {index}/{len(futures)} "
                    f"successful={len(successful_ids)} failed={len(failures)}",
                    flush=True,
                )

    rows = [
        _candidate_row(candidate, output_root)
        for candidate in selected
        if candidate.annotation.video_id in successful_ids
    ]
    candidate_path = output_root / "ave_boundary_candidates.jsonl"
    _write_jsonl_atomic(candidate_path, rows)
    _write_jsonl_atomic(output_root / "clip_failures.jsonl", failures)
    summary = {
        "protocol": "audiocvr_ave_boundary_pool_v1",
        "selection_uses_model_scores": False,
        "ave_root": str(Path(ave_root).resolve()),
        "annotation_count_non_speech_with_media": len(annotations),
        "excluded_youtube_id_count": len(excluded_video_ids),
        "boundary_eligible_count": len(candidates),
        "boundary_eligible_unique_source_count": len(unique_candidates),
        "duplicate_source_candidates_dropped": len(candidates) - len(unique_candidates),
        "selected_count": len(selected),
        "successful_candidate_count": len(rows),
        "clip_failure_count": len(failures),
        "tier_counts": dict(sorted(Counter(candidate.tier for candidate in selected).items())),
        "subtype_counts": dict(sorted(Counter(row["b_subtype"] for row in rows).items())),
        "category_counts": dict(sorted(Counter(row["ave_category"] for row in rows).items())),
        "clip_seconds": float(clip_seconds),
        "max_candidates": int(max_candidates),
        "random_seed": int(random_seed),
        "outputs": {
            "candidates": str(candidate_path),
            "selected_boundaries": str(output_root / "selected_boundary_candidates.jsonl"),
            "clip_failures": str(output_root / "clip_failures.jsonl"),
        },
    }
    _write_json_atomic(output_root / "ave_boundary_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare boundary-directed Audio-CVR candidates from AVE.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--ave-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--exclude-jsonl", action="append", default=[])
    prepare.add_argument("--clip-seconds", type=float, default=6.0)
    prepare.add_argument("--max-candidates", type=int, default=1000)
    prepare.add_argument("--workers", type=int, default=24)
    prepare.add_argument("--random-seed", type=int, default=20260723)
    prepare.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_ave_boundary_pool(
            ave_root=args.ave_root,
            output_dir=args.output_dir,
            exclude_jsonl_paths=args.exclude_jsonl,
            clip_seconds=args.clip_seconds,
            max_candidates=args.max_candidates,
            workers=args.workers,
            random_seed=args.random_seed,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
