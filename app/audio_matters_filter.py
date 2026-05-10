from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

import numpy as np

from app.e5_cvr_eval import DEFAULT_EXPECTED_COUNT, DEFAULT_RUNS_ROOT, E5CVRTriplet, find_latest_triplets, load_triplets_jsonl


DEFAULT_MIN_AUDIO_ANCHOR_SCORE = 0.85
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 12.0
DEFAULT_MIN_RMS = 1e-4
VISUAL_DIFFERENCE_TYPES = {"object_presence", "object_count", "attribute", "scene", "action"}
AUDIO_TEXT_RE = re.compile(
    r"\b(audio|sound|sounds|sounding|speech|speak|speaking|spoken|voice|voiceover|"
    r"narration|music|song|noise|noisy|listen|hear|hearing|audible|applause|"
    r"footstep|footsteps|wind|hum|machine|machinery)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AudioFeature:
    vector: np.ndarray
    rms: float
    sample_count: int


@dataclass(frozen=True)
class AudioMattersDecision:
    triplet: E5CVRTriplet
    accepted: bool
    audio_anchor_score: float
    reasons: list[str]
    metadata: dict[str, Any]


def build_audio_matters_filter(args: argparse.Namespace) -> dict[str, Any]:
    triplets_path = Path(args.triplets_jsonl) if args.triplets_jsonl else find_latest_triplets(args.runs_root)
    expected_count = None if args.expected_count <= 0 else args.expected_count
    triplets = load_triplets_jsonl(triplets_path, expected_count=expected_count)
    output_dir = Path(args.output_dir or _default_output_dir(args.runs_root))
    summary = filter_audio_matters_triplets(
        triplets=triplets,
        triplets_jsonl=triplets_path,
        output_dir=output_dir,
        min_audio_anchor_score=args.min_audio_anchor_score,
        min_rms=args.min_rms,
        ffmpeg=args.ffmpeg,
        sample_rate=args.sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def filter_audio_matters_triplets(
    *,
    triplets: list[E5CVRTriplet],
    triplets_jsonl: str | Path,
    output_dir: str | Path,
    min_audio_anchor_score: float = DEFAULT_MIN_AUDIO_ANCHOR_SCORE,
    min_rms: float = DEFAULT_MIN_RMS,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
    audio_feature_loader: Callable[[str], AudioFeature] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not triplets:
        raise ValueError("triplets must not be empty")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    loader = audio_feature_loader or (
        lambda path: load_audio_feature(
            path,
            ffmpeg=ffmpeg,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
        )
    )

    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    decisions: list[AudioMattersDecision] = []
    for index, triplet in enumerate(triplets, start=1):
        _emit(progress, f"[audio-matters] triplet {index}/{len(triplets)} start sample_id={triplet.sample_id}")
        decision = evaluate_audio_matters_triplet(
            triplet,
            audio_feature_loader=loader,
            min_audio_anchor_score=min_audio_anchor_score,
            min_rms=min_rms,
        )
        decisions.append(decision)
        record = _decision_record(decision)
        if decision.accepted:
            accepted_records.append(record)
            _emit(
                progress,
                f"[audio-matters] triplet {index}/{len(triplets)} accepted "
                f"score={decision.audio_anchor_score:.4f} sample_id={triplet.sample_id}",
            )
        else:
            rejected_records.append(record)
            _emit(
                progress,
                f"[audio-matters] triplet {index}/{len(triplets)} rejected "
                f"score={decision.audio_anchor_score:.4f} reasons={','.join(decision.reasons)} "
                f"sample_id={triplet.sample_id}",
            )

    _write_jsonl(root / "audio_matters_triplets.jsonl", accepted_records)
    _write_jsonl(root / "comparison_ready_triplets.jsonl", accepted_records)
    _write_jsonl(root / "rejected_triplets.jsonl", rejected_records)
    summary = _summary(
        triplets_jsonl=triplets_jsonl,
        output_dir=root,
        decisions=decisions,
        min_audio_anchor_score=min_audio_anchor_score,
        min_rms=min_rms,
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
    )
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(progress, f"[audio-matters] wrote accepted: {root / 'audio_matters_triplets.jsonl'}")
    _emit(progress, f"[audio-matters] wrote rejected: {root / 'rejected_triplets.jsonl'}")
    _emit(progress, f"[audio-matters] wrote summary: {root / 'summary.json'}")
    return summary


def evaluate_audio_matters_triplet(
    triplet: E5CVRTriplet,
    *,
    audio_feature_loader: Callable[[str], AudioFeature],
    min_audio_anchor_score: float = DEFAULT_MIN_AUDIO_ANCHOR_SCORE,
    min_rms: float = DEFAULT_MIN_RMS,
) -> AudioMattersDecision:
    reasons: list[str] = []
    difference_type = triplet.difference_type.strip()
    if difference_type not in VISUAL_DIFFERENCE_TYPES:
        reasons.append(f"non_visual_difference_type:{difference_type or 'missing'}")
    if AUDIO_TEXT_RE.search(triplet.edit_text):
        reasons.append("edit_text_mentions_audio")

    reference_feature: AudioFeature | None = None
    target_feature: AudioFeature | None = None
    score = 0.0
    try:
        reference_feature = audio_feature_loader(triplet.reference_video)
        target_feature = audio_feature_loader(triplet.target_video)
        if reference_feature.rms < min_rms or target_feature.rms < min_rms:
            reasons.append("audio_too_quiet_or_missing")
        else:
            score = audio_anchor_score(reference_feature, target_feature)
            if score < min_audio_anchor_score:
                reasons.append("audio_anchor_score_below_threshold")
    except Exception as exc:
        reasons.append(f"audio_feature_error:{type(exc).__name__}")

    metadata = {
        "audio_anchor_required": True,
        "audio_anchor_type": "similar_audio",
        "audio_anchor_score": round(score, 6),
        "audio_anchor_threshold": min_audio_anchor_score,
        "reference_audio_rms": round(float(reference_feature.rms), 8) if reference_feature is not None else 0.0,
        "target_audio_rms": round(float(target_feature.rms), 8) if target_feature is not None else 0.0,
        "reference_audio_sample_count": reference_feature.sample_count if reference_feature is not None else 0,
        "target_audio_sample_count": target_feature.sample_count if target_feature is not None else 0,
        "visual_delta_type": difference_type,
        "edit_primary_modality": "visual",
        "audio_primary_modality": False,
        "audio_should_be_preserved": True,
    }
    return AudioMattersDecision(
        triplet=triplet,
        accepted=not reasons,
        audio_anchor_score=score,
        reasons=reasons,
        metadata=metadata,
    )


def load_audio_feature(
    path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
) -> AudioFeature:
    audio = _load_audio_waveform(path, ffmpeg=ffmpeg, sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return AudioFeature(vector=_audio_feature_vector(audio), rms=rms, sample_count=int(audio.size))


def audio_anchor_score(reference: AudioFeature, target: AudioFeature) -> float:
    left = np.asarray(reference.vector, dtype=np.float32)
    right = np.asarray(target.vector, dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(1.0, score))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter existing CVR triplets into an audio-anchor visual-edit subset")
    parser.add_argument("--triplets-jsonl")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--min-audio-anchor-score", type=float, default=DEFAULT_MIN_AUDIO_ANCHOR_SCORE)
    parser.add_argument("--min-rms", type=float, default=DEFAULT_MIN_RMS)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max-audio-seconds", type=float, default=DEFAULT_MAX_AUDIO_SECONDS)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main() -> None:
    build_audio_matters_filter(build_parser().parse_args())


def _load_audio_waveform(
    path: str | Path,
    *,
    ffmpeg: str,
    sample_rate: int,
    max_audio_seconds: float,
) -> np.ndarray:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(max_audio_seconds),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    if not result.stdout:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32)


def _audio_feature_vector(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.size == 0:
        return np.zeros((64,), dtype=np.float32)
    waveform = waveform - float(np.mean(waveform))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak
    spectrum = np.abs(np.fft.rfft(waveform))
    if spectrum.size == 0:
        return np.zeros((64,), dtype=np.float32)
    spectrum = np.log1p(spectrum)
    bins = np.array_split(spectrum, 64)
    vector = np.asarray([float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in bins], dtype=np.float32)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return vector
    return vector / vector_norm


def _decision_record(decision: AudioMattersDecision) -> dict[str, Any]:
    triplet = decision.triplet
    return {
        "sample_id": triplet.sample_id,
        "reference_video": triplet.reference_video,
        "target_video": triplet.target_video,
        "edit_text": triplet.edit_text,
        "reference_caption": triplet.reference_caption,
        "source": triplet.source,
        "difference_type": triplet.difference_type,
        **decision.metadata,
        "audio_matters_accepted": decision.accepted,
        "audio_matters_reject_reasons": list(decision.reasons),
    }


def _summary(
    *,
    triplets_jsonl: str | Path,
    output_dir: Path,
    decisions: list[AudioMattersDecision],
    min_audio_anchor_score: float,
    min_rms: float,
    sample_rate: int,
    max_audio_seconds: float,
) -> dict[str, Any]:
    accepted = [decision for decision in decisions if decision.accepted]
    reject_reasons: dict[str, int] = {}
    difference_type_counts: dict[str, int] = {}
    for decision in decisions:
        difference_type_counts[decision.triplet.difference_type] = difference_type_counts.get(decision.triplet.difference_type, 0) + 1
        for reason in decision.reasons:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    scores = [decision.audio_anchor_score for decision in decisions]
    accepted_scores = [decision.audio_anchor_score for decision in accepted]
    return {
        "mode": "audio-matters-filter",
        "triplets_jsonl": str(triplets_jsonl),
        "output_dir": str(output_dir),
        "input_count": len(decisions),
        "accepted_count": len(accepted),
        "rejected_count": len(decisions) - len(accepted),
        "min_audio_anchor_score": min_audio_anchor_score,
        "min_rms": min_rms,
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "visual_difference_types": sorted(VISUAL_DIFFERENCE_TYPES),
        "reject_reason_counts": reject_reasons,
        "difference_type_counts": difference_type_counts,
        "audio_anchor_score": _score_stats(scores),
        "accepted_audio_anchor_score": _score_stats(accepted_scores),
        "outputs": {
            "audio_matters_triplets": str(output_dir / "audio_matters_triplets.jsonl"),
            "comparison_ready_triplets": str(output_dir / "comparison_ready_triplets.jsonl"),
            "rejected_triplets": str(output_dir / "rejected_triplets.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }


def _score_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
        "mean": round(float(sum(values) / len(values)), 6),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            sanitized = {key: value for key, value in record.items() if key != "target_caption"}
            handle.write(json.dumps(sanitized, ensure_ascii=False) + "\n")


def _default_output_dir(runs_root: str | Path) -> str:
    return str(Path(runs_root) / f"audio_matters_filter_{time.strftime('%Y%m%d_%H%M%S')}")


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


if __name__ == "__main__":
    main()
