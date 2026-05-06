from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

VIDEO_MASK_SEMANTICS_VERSION = 3
VIDEO_MASK_POLARITY = "white_generate_black_preserve"
MASK_GENERATION_STRATEGY = "adaptive_repair_v1"
MASK_QUALITY_EXCELLENT = "excellent"
MASK_QUALITY_USABLE = "usable_for_vace"
MASK_QUALITY_DIAGNOSTIC = "diagnostic_only"
MASK_QUALITY_FAILED = "failed"


def _normalized_phrase(value: str) -> str:
    return " ".join(part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in str(value)).split() if part)


def _git_head_short() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_video(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _find_file(path: Path, patterns: tuple[str, ...]) -> Path:
    if path.is_file():
        return path
    for pattern in patterns:
        matches = sorted(path.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"cannot find any of {patterns} under {path}")


def _looks_like_huggingface_pytorch_checkpoint(path: Path) -> bool:
    return (
        path.name == "pytorch_model.bin"
        and (
            (path.parent / "config.json").exists()
            or (path.parent / "model.safetensors").exists()
            or (path.parent / "tokenizer.json").exists()
        )
    )


def _find_grounding_dino_checkpoint(path: Path) -> Path:
    if path.is_file():
        if path.suffix == ".safetensors":
            raise ValueError(
                f"GroundingDINO load_model uses torch.load and cannot read safetensors checkpoints: {path}"
            )
        if _looks_like_huggingface_pytorch_checkpoint(path):
            raise ValueError(
                "GroundingDINO load_model expects a torch checkpoint dict with a 'model' key, "
                f"but this looks like a HuggingFace pytorch_model.bin checkpoint: {path}"
            )
        return path
    try:
        return _find_file(path, ("*.pth", "*.pt"))
    except FileNotFoundError as exc:
        bin_matches = [
            candidate
            for candidate in sorted(path.rglob("*.bin")) if path.exists()
            if not _looks_like_huggingface_pytorch_checkpoint(candidate)
        ]
        if bin_matches:
            return bin_matches[0]
        hf_bins = [
            candidate
            for candidate in sorted(path.rglob("*.bin")) if path.exists()
            if _looks_like_huggingface_pytorch_checkpoint(candidate)
        ]
        if hf_bins:
            raise FileNotFoundError(
                "GroundingDINO checkpoint directory only contains HuggingFace-format "
                "pytorch_model.bin files, but GroundingDINO load_model expects a torch "
                f"checkpoint dict with a 'model' key. First HF bin: {hf_bins[0]}"
            ) from exc
        safetensors = sorted(path.rglob("*.safetensors")) if path.exists() else []
        if safetensors:
            raise FileNotFoundError(
                "GroundingDINO checkpoint directory only contains safetensors files, "
                "but GroundingDINO load_model expects a torch checkpoint (.pth/.pt/.bin). "
                f"First safetensors file: {safetensors[0]}"
            ) from exc
        raise


def _find_dir(path: Path, markers: tuple[str, ...]) -> Path:
    if path.is_dir() and all((path / marker).exists() for marker in markers):
        return path
    for candidate in sorted(path.rglob("*")):
        if candidate.is_dir() and all((candidate / marker).exists() for marker in markers):
            return candidate
    raise FileNotFoundError(f"cannot find model dir with markers {markers} under {path}")


def _ffprobe_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    value = result.stdout.strip()
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / max(float(den), 1.0)
    return float(value or 24.0)


def _extract_frames(video_path: Path, frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(frame_dir / "%05d.jpg"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _encode_mask_video(mask_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(mask_dir / "%05d.png"),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _sample_keyframe_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    raw = [0, frame_count // 4, frame_count // 2, (frame_count * 3) // 4, frame_count - 1]
    return sorted({max(0, min(frame_count - 1, idx)) for idx in raw})


def _sample_detection_frame_indices(frame_count: int, *, max_samples: int = 13) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_samples:
        return list(range(frame_count))
    anchors = set(_sample_keyframe_indices(frame_count))
    sample_count = max(2, max_samples)
    for pos in range(sample_count):
        idx = round(pos * (frame_count - 1) / (sample_count - 1))
        anchors.add(max(0, min(frame_count - 1, idx)))
    return sorted(anchors)


def _select_anchor_detections(
    detections: list[dict[str, Any]],
    *,
    max_anchors: int = 3,
) -> list[dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for detection in detections:
        frame_idx = int(detection.get("frame_idx", 0) or 0)
        previous = by_frame.get(frame_idx)
        if previous is None or float(detection.get("score", 0.0) or 0.0) > float(previous.get("score", 0.0) or 0.0):
            by_frame[frame_idx] = detection
    ranked = sorted(by_frame.values(), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    selected = ranked[: max(1, max_anchors)]
    return sorted(selected, key=lambda item: int(item.get("frame_idx", 0) or 0))


def _visible_spans_from_masks(masks_by_frame: dict[int, Any], frame_count: int) -> list[dict[str, Any]]:
    import numpy as np

    spans: list[dict[str, Any]] = []
    active_start: int | None = None
    coverages: list[float] = []
    for idx in range(frame_count):
        mask = masks_by_frame.get(idx)
        coverage = float(np.asarray(mask).astype(bool).mean()) if mask is not None else 0.0
        if coverage > 0.0:
            if active_start is None:
                active_start = idx
                coverages = []
            coverages.append(coverage)
        elif active_start is not None:
            spans.append(
                {
                    "start_frame": active_start,
                    "end_frame": idx - 1,
                    "frame_count": idx - active_start,
                    "coverage_avg": sum(coverages) / len(coverages) if coverages else 0.0,
                }
            )
            active_start = None
            coverages = []
    if active_start is not None:
        spans.append(
            {
                "start_frame": active_start,
                "end_frame": frame_count - 1,
                "frame_count": frame_count - active_start,
                "coverage_avg": sum(coverages) / len(coverages) if coverages else 0.0,
            }
        )
    return spans


def _mask_quality_tier(gate_errors: list[str], metrics: dict[str, Any]) -> str:
    nonempty = float(metrics.get("mask_nonempty_frame_ratio", 0.0) or 0.0)
    stability = float(metrics.get("mask_temporal_stability", 0.0) or 0.0)
    avg_coverage = float(metrics.get("mask_coverage_ratio_avg", 0.0) or 0.0)
    if nonempty <= 0.0 or avg_coverage <= 0.0:
        return MASK_QUALITY_FAILED
    if not gate_errors:
        if nonempty >= 0.98 and stability >= 0.90:
            return MASK_QUALITY_EXCELLENT
        return MASK_QUALITY_USABLE
    return MASK_QUALITY_DIAGNOSTIC


def _append_adaptive_provenance(
    metrics: dict[str, Any],
    *,
    gate_errors: list[str] | None = None,
    detection_attempts: list[dict[str, Any]] | None = None,
    anchor_frame_indices: list[int] | None = None,
    prompt_type: str = "box",
    repair_rounds: int = 0,
) -> dict[str, Any]:
    errors = list(gate_errors or [])
    quality_tier = _mask_quality_tier(errors, metrics)
    metrics["mask_generation_strategy"] = MASK_GENERATION_STRATEGY
    metrics["detector_cascade"] = metrics.get("detector_cascade", [])
    metrics["detection_attempts"] = detection_attempts or metrics.get("detection_attempts", [])
    metrics["anchor_frame_indices"] = anchor_frame_indices or metrics.get("anchor_frame_indices", [])
    metrics["prompt_type"] = prompt_type
    metrics["reinit_count"] = max(0, len(metrics["anchor_frame_indices"]) - 1)
    metrics["repair_rounds"] = repair_rounds
    metrics["visible_spans"] = metrics.get("visible_spans", [])
    metrics["sparse_full_length"] = bool(float(metrics.get("mask_nonempty_frame_ratio", 0.0) or 0.0) < 0.98)
    metrics["mask_quality_tier"] = quality_tier
    metrics["usable_for_vace"] = quality_tier in {MASK_QUALITY_EXCELLENT, MASK_QUALITY_USABLE}
    metrics["failure_reasons"] = errors
    return metrics


def _write_all_black_mask_video(reference_video: Path, output_mask_video: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="black_sparse_mask_") as tmp:
        tmp_dir = Path(tmp)
        frame_dir = tmp_dir / "frames"
        mask_dir = tmp_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        _extract_frames(reference_video, frame_dir)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"no frames extracted from {reference_video}")
        shape = None
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"cannot read extracted frame: {frame_path}")
            shape = frame.shape[:2]
            cv2.imwrite(str(mask_dir / f"{idx:05d}.png"), np.zeros(shape, dtype="uint8"))
        fps = _ffprobe_fps(reference_video)
        _encode_mask_video(mask_dir, output_mask_video, fps)
        metrics = {
            "frame_count": len(frame_paths),
            "fps": fps,
            "reference_frame_count": len(frame_paths),
            "reference_fps": fps,
            "mask_coverage_ratio_min": 0.0,
            "mask_coverage_ratio_avg": 0.0,
            "mask_coverage_ratio_max": 0.0,
            "mask_temporal_stability": 1.0,
            "mask_nonempty_frame_ratio": 0.0,
            "visible_span_ratio": 0.0,
            "visible_spans": [],
            "sampled_frame_indices": _sample_detection_frame_indices(len(frame_paths)),
            "detected_keyframe_index": None,
            "detected_keyframe_box_coverage": 0.0,
            "box_count": 0,
            "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
            "mask_polarity": VIDEO_MASK_POLARITY,
            "generator_commit": _git_head_short(),
            "device": "",
        }
        return _append_adaptive_provenance(
            metrics,
            gate_errors=["no detector produced a usable mask; emitted all-black diagnostic mask"],
            detection_attempts=[],
            anchor_frame_indices=[],
            prompt_type="none",
        )


def _mask_gate_errors(mask_gate: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    avg_coverage = float(metrics.get("mask_coverage_ratio_avg", 0.0) or 0.0)
    min_coverage = float(mask_gate.get("min_coverage_ratio", 0.0) or 0.0)
    max_coverage = float(mask_gate.get("max_coverage_ratio", 1.0) or 1.0)
    if avg_coverage < min_coverage:
        errors.append(f"avg_coverage {avg_coverage:.4f} < min {min_coverage:.4f}")
    if avg_coverage > max_coverage:
        errors.append(f"avg_coverage {avg_coverage:.4f} > max {max_coverage:.4f}")
    min_stability = float(mask_gate.get("min_temporal_stability", 0.0) or 0.0)
    stability = float(metrics.get("mask_temporal_stability", 0.0) or 0.0)
    if stability < min_stability:
        errors.append(f"temporal_stability {stability:.4f} < min {min_stability:.4f}")
    min_nonempty = float(mask_gate.get("min_nonempty_frame_ratio", 0.0) or 0.0)
    nonempty = float(metrics.get("mask_nonempty_frame_ratio", 0.0) or 0.0)
    if nonempty < min_nonempty:
        errors.append(f"nonempty_frame_ratio {nonempty:.4f} < min {min_nonempty:.4f}")
    if bool(mask_gate.get("mask_not_empty_all_frames")) and nonempty < 1.0:
        errors.append("mask is empty on at least one frame")
    min_box_coverage = float(mask_gate.get("min_detected_keyframe_box_coverage", 0.0) or 0.0)
    box_coverage = float(metrics.get("detected_keyframe_box_coverage", 0.0) or 0.0)
    if box_coverage < min_box_coverage:
        errors.append(f"detected_keyframe_box_coverage {box_coverage:.4f} < min {min_box_coverage:.4f}")
    max_subject_overlap = mask_gate.get("max_subject_overlap_ratio")
    if max_subject_overlap is not None:
        subject_overlap = float(metrics.get("subject_overlap_ratio", 0.0) or 0.0)
        max_subject_overlap_float = float(max_subject_overlap or 0.0)
        if subject_overlap > max_subject_overlap_float:
            errors.append(
                f"subject_overlap_ratio {subject_overlap:.4f} > max {max_subject_overlap_float:.4f}"
            )
    min_background_editable = mask_gate.get("min_background_editable_ratio")
    if min_background_editable is not None:
        background_editable = float(metrics.get("background_editable_ratio", 0.0) or 0.0)
        min_background_editable_float = float(min_background_editable or 0.0)
        if background_editable < min_background_editable_float:
            errors.append(
                f"background_editable_ratio {background_editable:.4f} < min {min_background_editable_float:.4f}"
            )
    min_foreground_coverage = mask_gate.get("min_foreground_subject_coverage_ratio")
    if min_foreground_coverage is not None:
        foreground_coverage = float(metrics.get("foreground_subject_coverage_ratio_avg", 0.0) or 0.0)
        min_foreground_coverage_float = float(min_foreground_coverage or 0.0)
        if foreground_coverage < min_foreground_coverage_float:
            errors.append(
                f"foreground_subject_coverage_ratio {foreground_coverage:.4f} < min {min_foreground_coverage_float:.4f}"
            )
    max_foreground_coverage = mask_gate.get("max_foreground_subject_coverage_ratio")
    if max_foreground_coverage is not None:
        foreground_coverage = float(metrics.get("foreground_subject_coverage_ratio_avg", 0.0) or 0.0)
        max_foreground_coverage_float = float(max_foreground_coverage or 0.0)
        if foreground_coverage > max_foreground_coverage_float:
            errors.append(
                f"foreground_subject_coverage_ratio {foreground_coverage:.4f} > max {max_foreground_coverage_float:.4f}"
            )
    min_foreground_stability = mask_gate.get("min_foreground_subject_temporal_stability")
    if min_foreground_stability is not None:
        foreground_stability = float(metrics.get("foreground_subject_temporal_stability", 0.0) or 0.0)
        min_foreground_stability_float = float(min_foreground_stability or 0.0)
        if foreground_stability < min_foreground_stability_float:
            errors.append(
                f"foreground_subject_temporal_stability {foreground_stability:.4f} < min {min_foreground_stability_float:.4f}"
            )
    min_foreground_nonempty = mask_gate.get("min_foreground_subject_nonempty_frame_ratio")
    if min_foreground_nonempty is not None:
        foreground_nonempty = float(metrics.get("foreground_subject_nonempty_frame_ratio", 0.0) or 0.0)
        min_foreground_nonempty_float = float(min_foreground_nonempty or 0.0)
        if foreground_nonempty < min_foreground_nonempty_float:
            errors.append(
                f"foreground_subject_nonempty_frame_ratio {foreground_nonempty:.4f} < min {min_foreground_nonempty_float:.4f}"
            )
    max_protected_overlap = mask_gate.get("max_protected_overlap_ratio")
    if max_protected_overlap is not None:
        protected_overlap = metrics.get("protected_overlap_ratio_max")
        if protected_overlap is None:
            if bool(mask_gate.get("require_protected_overlap_metrics")):
                errors.append("protected overlap metrics are missing")
        else:
            protected_overlap_float = float(protected_overlap or 0.0)
            max_protected_overlap_float = float(max_protected_overlap or 0.0)
            if protected_overlap_float > max_protected_overlap_float:
                errors.append(
                    f"protected_overlap_ratio {protected_overlap_float:.4f} > max {max_protected_overlap_float:.4f}"
                )
    min_protected_detections = int(mask_gate.get("min_protected_detections", 0) or 0)
    if min_protected_detections > 0:
        protected_details = metrics.get("protected_overlap", [])
        if not isinstance(protected_details, list):
            protected_details = []
        detected_count = sum(
            1
            for item in protected_details
            if isinstance(item, dict) and str(item.get("status", "")).strip() == "detected"
        )
        if detected_count < min_protected_detections:
            errors.append(
                f"protected_detection_count {detected_count} < min {min_protected_detections}"
            )
    return errors


def _box_mask_overlap_ratio(mask: Any, boxes_xyxy: Any) -> float:
    import numpy as np

    if mask is None:
        return 0.0
    mask_bool = np.asarray(mask).astype(bool)
    mask_area = float(mask_bool.sum())
    if mask_area <= 0.0:
        return 0.0
    protected = np.zeros(mask_bool.shape, dtype=bool)
    h, w = mask_bool.shape[:2]
    for raw_box in np.asarray(boxes_xyxy):
        x0, y0, x1, y1 = [int(round(float(value))) for value in raw_box[:4]]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            protected[y0:y1, x0:x1] = True
    return float((mask_bool & protected).sum()) / mask_area


def _mask_overlap_ratio(mask: Any, protected_mask: Any) -> float:
    import numpy as np

    if mask is None or protected_mask is None:
        return 0.0
    mask_bool = np.asarray(mask).astype(bool)
    protected_bool = np.asarray(protected_mask).astype(bool)
    if mask_bool.shape != protected_bool.shape:
        return 0.0
    protected_area = float(protected_bool.sum())
    if protected_area <= 0.0:
        return 0.0
    return float((mask_bool & protected_bool).sum()) / protected_area


def _mask_sequence_metrics(masks_by_frame: dict[int, Any], frame_count: int) -> dict[str, float]:
    import numpy as np

    template_shape = (1, 1)
    for value in masks_by_frame.values():
        candidate = np.asarray(value)
        if candidate.size:
            template_shape = candidate.shape[:2]
            break
    coverages: list[float] = []
    ious: list[float] = []
    nonempty_count = 0
    last_mask: np.ndarray | None = None
    for idx in range(frame_count):
        mask = masks_by_frame.get(idx)
        if mask is None:
            mask = np.zeros(template_shape, dtype="uint8")
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape[:2] != template_shape:
            mask_bool = np.zeros(template_shape, dtype=bool)
        coverage = float(mask_bool.mean())
        coverages.append(coverage)
        if coverage > 0.0:
            nonempty_count += 1
        if last_mask is not None:
            intersection = float(np.logical_and(last_mask, mask_bool).sum())
            union = float(np.logical_or(last_mask, mask_bool).sum())
            ious.append(intersection / union if union > 0 else 0.0)
        last_mask = mask_bool.copy()
    return {
        "coverage_min": min(coverages) if coverages else 0.0,
        "coverage_avg": sum(coverages) / len(coverages) if coverages else 0.0,
        "coverage_max": max(coverages) if coverages else 0.0,
        "temporal_stability": sum(ious) / len(ious) if ious else 1.0,
        "nonempty_frame_ratio": nonempty_count / len(coverages) if coverages else 0.0,
    }


def _run_grounded_sam2(
    *,
    reference_video: Path,
    output_mask_video: Path,
    mask_query: str,
    mask_mode: str,
    grounded_sam2_code: Path,
    grounding_dino_config: Path,
    grounding_dino_checkpoint: Path,
    sam2_config: str,
    sam2_checkpoint: Path,
    box_threshold: float,
    text_threshold: float,
    mask_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(grounded_sam2_code))

    import cv2
    import numpy as np
    import torch
    from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="grounded_sam2_mask_") as tmp:
        tmp_dir = Path(tmp)
        frame_dir = tmp_dir / "frames"
        mask_dir = tmp_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        _extract_frames(reference_video, frame_dir)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"no frames extracted from {reference_video}")

        grounding_model = load_model(
            str(grounding_dino_config),
            str(grounding_dino_checkpoint),
            device=device,
        )
        sampled_frame_indices = _sample_detection_frame_indices(len(frame_paths))
        detections: list[dict[str, Any]] = []
        detection_attempts: list[dict[str, Any]] = []
        for frame_idx in sampled_frame_indices:
            frame_path = frame_paths[frame_idx]
            image_source, image = load_image(str(frame_path))
            boxes, logits, phrases = predict(
                model=grounding_model,
                image=image,
                caption=mask_query,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )
            if boxes is None or len(boxes) == 0:
                detection_attempts.append(
                    {"frame_idx": frame_idx, "query": mask_query, "status": "not_detected", "detector": "groundingdino"}
                )
                continue
            h, w = image_source.shape[:2]
            boxes_np = boxes.detach().cpu().numpy()
            boxes_xyxy = np.zeros_like(boxes_np)
            boxes_xyxy[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * w
            boxes_xyxy[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * h
            boxes_xyxy[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * w
            boxes_xyxy[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * h
            clipped = boxes_xyxy.copy()
            clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, w)
            clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, h)
            box_areas = np.maximum(0, clipped[:, 2] - clipped[:, 0]) * np.maximum(0, clipped[:, 3] - clipped[:, 1])
            coverage = float(box_areas[:3].sum() / max(float(w * h), 1.0))
            logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.array(logits)
            confidence = float(logits_np.max()) if logits_np.size else 0.0
            score = confidence - abs(min(coverage, 1.0) - 0.12)
            detection = {
                "frame_idx": frame_idx,
                "boxes_xyxy": boxes_xyxy,
                "phrases": [str(item) for item in phrases],
                "score": score,
                "box_coverage": coverage,
                "confidence": confidence,
                "box_count": int(len(boxes_xyxy)),
            }
            detections.append(detection)
            detection_attempts.append(
                {
                    "frame_idx": frame_idx,
                    "query": mask_query,
                    "status": "detected",
                    "detector": "groundingdino",
                    "box_count": int(len(boxes_xyxy)),
                    "box_coverage": coverage,
                    "confidence": confidence,
                    "score": score,
                    "phrases": [str(item) for item in phrases],
                }
            )
        if not detections:
            raise RuntimeError(f"GroundingDINO found no box for query in sampled frames: {mask_query}")
        anchor_detections = _select_anchor_detections(detections, max_anchors=3)
        best_detection = max(anchor_detections, key=lambda item: float(item.get("score", 0.0) or 0.0))
        boxes_xyxy = best_detection["boxes_xyxy"]
        keyframe_idx = int(best_detection["frame_idx"])

        predictor = build_sam2_video_predictor(sam2_config, str(sam2_checkpoint), device=device)
        inference_state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(inference_state)
        object_id = 1
        for anchor in anchor_detections:
            for box in anchor["boxes_xyxy"][:3]:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(anchor["frame_idx"]),
                    obj_id=object_id,
                    box=box,
                )
                object_id += 1

        raw_frame_masks: dict[int, np.ndarray] = {}
        frame_masks: dict[int, np.ndarray] = {}
        for frame_idx, _object_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks = (mask_logits > 0.0).detach().cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            union = np.any(masks, axis=0).astype("uint8")
            raw_frame_masks[int(frame_idx)] = union.copy()
            if mask_mode == "edit_background_inverse_subject":
                union = 1 - union
            frame_masks[int(frame_idx)] = union

        protected_overlap_details: list[dict[str, Any]] = []
        protected_overlap_max: float | None = None
        protected_queries = [
            str(item).strip()
            for item in (mask_gate or {}).get("protected_overlap_queries", [])
            if str(item).strip()
        ]
        if protected_queries:
            keyframe_mask = frame_masks.get(keyframe_idx)
            frame_path = frame_paths[keyframe_idx]
            image_source, image = load_image(str(frame_path))
            h, w = image_source.shape[:2]
            overlaps: list[float] = []
            for protected_query in protected_queries:
                try:
                    protected_boxes, _protected_logits, protected_phrases = predict(
                        model=grounding_model,
                        image=image,
                        caption=protected_query,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=device,
                    )
                    if protected_boxes is None or len(protected_boxes) == 0:
                        protected_overlap_details.append(
                            {"query": protected_query, "status": "not_detected", "overlap_ratio": 0.0}
                        )
                        continue
                    boxes_np = protected_boxes.detach().cpu().numpy()
                    protected_xyxy = np.zeros_like(boxes_np)
                    protected_xyxy[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * w
                    protected_xyxy[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * h
                    protected_xyxy[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * w
                    protected_xyxy[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * h
                    overlap = _box_mask_overlap_ratio(keyframe_mask, protected_xyxy)
                    overlaps.append(overlap)
                    protected_overlap_details.append(
                        {
                            "query": protected_query,
                            "status": "detected",
                            "overlap_ratio": overlap,
                            "detected_phrases": [str(item) for item in protected_phrases],
                        }
                    )
                except Exception as exc:
                    protected_overlap_details.append(
                        {"query": protected_query, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    )
            protected_overlap_max = max(overlaps) if overlaps else 0.0

        coverages: list[float] = []
        ious: list[float] = []
        nonempty_count = 0
        last_mask: np.ndarray | None = None
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"cannot read extracted frame: {frame_path}")
            mask = frame_masks.get(idx)
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype="uint8")
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            coverages.append(float(mask.mean()))
            if float(mask.mean()) > 0.0:
                nonempty_count += 1
            if last_mask is not None:
                intersection = float(np.logical_and(last_mask, mask).sum())
                union = float(np.logical_or(last_mask, mask).sum())
                ious.append(intersection / union if union > 0 else 0.0)
            last_mask = mask.copy()
            mask_rgb = (mask * 255).astype("uint8")
            cv2.imwrite(str(mask_dir / f"{idx:05d}.png"), mask_rgb)

        fps = _ffprobe_fps(reference_video)
        _encode_mask_video(mask_dir, output_mask_video, fps)
        background_editable_ratio = sum(coverages) / len(coverages) if coverages else 0.0
        subject_overlap_ratio = None
        foreground_metrics: dict[str, float] = {}
        if mask_mode == "edit_background_inverse_subject":
            foreground_metrics = _mask_sequence_metrics(raw_frame_masks, len(frame_paths))
            subject_overlap_ratio = _mask_overlap_ratio(frame_masks.get(keyframe_idx), raw_frame_masks.get(keyframe_idx))
        metrics = {
            "frame_count": len(frame_paths),
            "fps": fps,
            "reference_frame_count": len(frame_paths),
            "reference_fps": fps,
            "mask_coverage_ratio_min": min(coverages) if coverages else 0.0,
            "mask_coverage_ratio_avg": background_editable_ratio,
            "mask_coverage_ratio_max": max(coverages) if coverages else 0.0,
            "mask_temporal_stability": sum(ious) / len(ious) if ious else 1.0,
            "mask_nonempty_frame_ratio": nonempty_count / len(coverages) if coverages else 0.0,
            "visible_span_ratio": nonempty_count / len(coverages) if coverages else 0.0,
            "visible_spans": _visible_spans_from_masks(frame_masks, len(frame_paths)),
            "reinit_count": max(0, len(anchor_detections) - 1),
            "sampled_frame_indices": sampled_frame_indices,
            "detected_phrases": best_detection["phrases"],
            "detected_keyframe_index": keyframe_idx,
            "detected_keyframe_box_coverage": best_detection["box_coverage"],
            "box_count": int(len(boxes_xyxy)),
            "device": device,
            "grounder": "groundingdino",
            "detector_cascade": ["groundingdino"],
            "detection_attempts": detection_attempts,
            "anchor_frame_indices": [int(item["frame_idx"]) for item in anchor_detections],
            "prompt_type": "box",
            "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
            "mask_polarity": VIDEO_MASK_POLARITY,
            "generator_commit": _git_head_short(),
        }
        if subject_overlap_ratio is not None:
            metrics["subject_overlap_ratio"] = subject_overlap_ratio
            metrics["background_editable_ratio"] = background_editable_ratio
            metrics["foreground_subject_coverage_ratio_min"] = foreground_metrics.get("coverage_min", 0.0)
            metrics["foreground_subject_coverage_ratio_avg"] = foreground_metrics.get("coverage_avg", 0.0)
            metrics["foreground_subject_coverage_ratio_max"] = foreground_metrics.get("coverage_max", 0.0)
            metrics["foreground_subject_temporal_stability"] = foreground_metrics.get("temporal_stability", 0.0)
            metrics["foreground_subject_nonempty_frame_ratio"] = foreground_metrics.get("nonempty_frame_ratio", 0.0)
        if protected_overlap_details:
            metrics["protected_overlap"] = protected_overlap_details
            metrics["protected_overlap_ratio_max"] = protected_overlap_max if protected_overlap_max is not None else 0.0
        return _append_adaptive_provenance(
            metrics,
            detection_attempts=detection_attempts,
            anchor_frame_indices=[int(item["frame_idx"]) for item in anchor_detections],
            prompt_type="box",
        )


def _florence_boxes(
    *,
    image_path: Path,
    mask_query: str,
    florence_model_dir: Path,
    device: str,
) -> tuple[Any, list[str]]:
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(florence_model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(florence_model_dir),
        trust_remote_code=True,
    ).eval().to(device)
    return _florence_boxes_with_model(
        image_path=image_path,
        mask_query=mask_query,
        processor=processor,
        model=model,
        device=device,
    )


def _florence_boxes_with_model(
    *,
    image_path: Path,
    mask_query: str,
    processor: Any,
    model: Any,
    device: str,
) -> tuple[Any, list[str]]:
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    task = "<OPEN_VOCABULARY_DETECTION>"
    prompt = task + mask_query
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
    result = parsed.get(task, {})
    boxes = result.get("bboxes", []) or []
    labels = result.get("bboxes_labels", result.get("labels", [])) or []
    if not boxes:
        raise RuntimeError(f"Florence-2 found no box for query: {mask_query}")
    return np.array(boxes, dtype="float32"), [str(item) for item in labels]


def _run_florence_sam2(
    *,
    reference_video: Path,
    output_mask_video: Path,
    mask_query: str,
    mask_mode: str,
    florence_model_dir: Path,
    grounded_sam2_code: Path,
    sam2_config: str,
    sam2_checkpoint: Path,
    mask_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(grounded_sam2_code))

    import cv2
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(str(florence_model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(florence_model_dir),
        trust_remote_code=True,
    ).eval().to(device)
    with tempfile.TemporaryDirectory(prefix="florence_sam2_mask_") as tmp:
        tmp_dir = Path(tmp)
        frame_dir = tmp_dir / "frames"
        mask_dir = tmp_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        _extract_frames(reference_video, frame_dir)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"no frames extracted from {reference_video}")

        sampled_frame_indices = _sample_detection_frame_indices(len(frame_paths))
        detections: list[dict[str, Any]] = []
        detection_attempts: list[dict[str, Any]] = []
        for frame_idx in sampled_frame_indices:
            try:
                boxes_xyxy, labels = _florence_boxes_with_model(
                    image_path=frame_paths[frame_idx],
                    mask_query=mask_query,
                    processor=processor,
                    model=model,
                    device=device,
                )
            except RuntimeError:
                detection_attempts.append(
                    {"frame_idx": frame_idx, "query": mask_query, "status": "not_detected", "detector": "florence2"}
                )
                continue
            frame = cv2.imread(str(frame_paths[frame_idx]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            clipped = boxes_xyxy.copy()
            clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, w)
            clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, h)
            box_areas = np.maximum(0, clipped[:, 2] - clipped[:, 0]) * np.maximum(0, clipped[:, 3] - clipped[:, 1])
            coverage = float(box_areas[:3].sum() / max(float(w * h), 1.0))
            score = -abs(min(coverage, 1.0) - 0.12) + min(len(boxes_xyxy), 3) * 0.05
            detection = {
                "frame_idx": frame_idx,
                "boxes_xyxy": boxes_xyxy,
                "labels": labels,
                "score": score,
                "box_coverage": coverage,
                "box_count": int(len(boxes_xyxy)),
            }
            detections.append(detection)
            detection_attempts.append(
                {
                    "frame_idx": frame_idx,
                    "query": mask_query,
                    "status": "detected",
                    "detector": "florence2",
                    "box_count": int(len(boxes_xyxy)),
                    "box_coverage": coverage,
                    "score": score,
                    "labels": labels,
                }
            )
        if not detections:
            raise RuntimeError(f"Florence-2 found no box for query in sampled frames: {mask_query}")
        anchor_detections = _select_anchor_detections(detections, max_anchors=3)
        best_detection = max(anchor_detections, key=lambda item: float(item.get("score", 0.0) or 0.0))
        boxes_xyxy = best_detection["boxes_xyxy"]
        labels = best_detection["labels"]
        keyframe_idx = int(best_detection["frame_idx"])
        predictor = build_sam2_video_predictor(sam2_config, str(sam2_checkpoint), device=device)
        inference_state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(inference_state)
        object_id = 1
        for anchor in anchor_detections:
            for box in anchor["boxes_xyxy"][:3]:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(anchor["frame_idx"]),
                    obj_id=object_id,
                    box=box,
                )
                object_id += 1

        raw_frame_masks: dict[int, np.ndarray] = {}
        frame_masks: dict[int, np.ndarray] = {}
        for frame_idx, _object_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks = (mask_logits > 0.0).detach().cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            union = np.any(masks, axis=0).astype("uint8")
            raw_frame_masks[int(frame_idx)] = union.copy()
            if mask_mode == "edit_background_inverse_subject":
                union = 1 - union
            frame_masks[int(frame_idx)] = union

        protected_overlap_details: list[dict[str, Any]] = []
        protected_overlap_max: float | None = None
        protected_queries = [
            str(item).strip()
            for item in (mask_gate or {}).get("protected_overlap_queries", [])
            if str(item).strip()
        ]
        if protected_queries:
            keyframe_mask = frame_masks.get(keyframe_idx)
            overlaps: list[float] = []
            for protected_query in protected_queries:
                try:
                    protected_boxes, protected_labels = _florence_boxes_with_model(
                        image_path=frame_paths[keyframe_idx],
                        mask_query=protected_query,
                        processor=processor,
                        model=model,
                        device=device,
                    )
                    overlap = _box_mask_overlap_ratio(keyframe_mask, protected_boxes)
                    overlaps.append(overlap)
                    protected_overlap_details.append(
                        {
                            "query": protected_query,
                            "status": "detected",
                            "overlap_ratio": overlap,
                            "detected_phrases": protected_labels,
                        }
                    )
                except Exception as exc:
                    protected_overlap_details.append(
                        {"query": protected_query, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    )
            protected_overlap_max = max(overlaps) if overlaps else 0.0

        coverages: list[float] = []
        ious: list[float] = []
        nonempty_count = 0
        last_mask: np.ndarray | None = None
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"cannot read extracted frame: {frame_path}")
            mask = frame_masks.get(idx)
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype="uint8")
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            coverages.append(float(mask.mean()))
            if float(mask.mean()) > 0.0:
                nonempty_count += 1
            if last_mask is not None:
                intersection = float(np.logical_and(last_mask, mask).sum())
                union = float(np.logical_or(last_mask, mask).sum())
                ious.append(intersection / union if union > 0 else 0.0)
            last_mask = mask.copy()
            cv2.imwrite(str(mask_dir / f"{idx:05d}.png"), (mask * 255).astype("uint8"))

        fps = _ffprobe_fps(reference_video)
        _encode_mask_video(mask_dir, output_mask_video, fps)
        background_editable_ratio = sum(coverages) / len(coverages) if coverages else 0.0
        subject_overlap_ratio = None
        foreground_metrics: dict[str, float] = {}
        if mask_mode == "edit_background_inverse_subject":
            foreground_metrics = _mask_sequence_metrics(raw_frame_masks, len(frame_paths))
            subject_overlap_ratio = _mask_overlap_ratio(frame_masks.get(keyframe_idx), raw_frame_masks.get(keyframe_idx))
        metrics = {
            "frame_count": len(frame_paths),
            "fps": fps,
            "reference_frame_count": len(frame_paths),
            "reference_fps": fps,
            "mask_coverage_ratio_min": min(coverages) if coverages else 0.0,
            "mask_coverage_ratio_avg": background_editable_ratio,
            "mask_coverage_ratio_max": max(coverages) if coverages else 0.0,
            "mask_temporal_stability": sum(ious) / len(ious) if ious else 1.0,
            "mask_nonempty_frame_ratio": nonempty_count / len(coverages) if coverages else 0.0,
            "visible_span_ratio": nonempty_count / len(coverages) if coverages else 0.0,
            "visible_spans": _visible_spans_from_masks(frame_masks, len(frame_paths)),
            "reinit_count": max(0, len(anchor_detections) - 1),
            "sampled_frame_indices": sampled_frame_indices,
            "detected_phrases": labels,
            "detected_keyframe_index": keyframe_idx,
            "detected_keyframe_box_coverage": best_detection["box_coverage"],
            "box_count": int(len(boxes_xyxy)),
            "device": device,
            "grounder": "florence2",
            "detector_cascade": ["florence2"],
            "detection_attempts": detection_attempts,
            "anchor_frame_indices": [int(item["frame_idx"]) for item in anchor_detections],
            "prompt_type": "box",
            "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
            "mask_polarity": VIDEO_MASK_POLARITY,
            "generator_commit": _git_head_short(),
        }
        if subject_overlap_ratio is not None:
            metrics["subject_overlap_ratio"] = subject_overlap_ratio
            metrics["background_editable_ratio"] = background_editable_ratio
            metrics["foreground_subject_coverage_ratio_min"] = foreground_metrics.get("coverage_min", 0.0)
            metrics["foreground_subject_coverage_ratio_avg"] = foreground_metrics.get("coverage_avg", 0.0)
            metrics["foreground_subject_coverage_ratio_max"] = foreground_metrics.get("coverage_max", 0.0)
            metrics["foreground_subject_temporal_stability"] = foreground_metrics.get("temporal_stability", 0.0)
            metrics["foreground_subject_nonempty_frame_ratio"] = foreground_metrics.get("nonempty_frame_ratio", 0.0)
        if protected_overlap_details:
            metrics["protected_overlap"] = protected_overlap_details
            metrics["protected_overlap_ratio_max"] = protected_overlap_max if protected_overlap_max is not None else 0.0
        return _append_adaptive_provenance(
            metrics,
            detection_attempts=detection_attempts,
            anchor_frame_indices=[int(item["frame_idx"]) for item in anchor_detections],
            prompt_type="box",
        )


def _mask_query_candidates(plan: dict[str, Any]) -> list[str]:
    raw_candidates = plan.get("mask_query_candidates", [])
    values: list[str] = []
    if isinstance(raw_candidates, list):
        values.extend(str(item).strip() for item in raw_candidates if str(item).strip())
    primary = str(plan.get("mask_query", "")).strip()
    if primary:
        values.insert(0, primary)
    seen: set[str] = set()
    candidates: list[str] = []
    for value in values:
        key = _normalized_phrase(value)
        if key and key not in seen:
            seen.add(key)
            candidates.append(value)
    return candidates or [primary or "object"]


def _candidate_mask_video_path(mask_video: Path, attempt_index: int) -> Path:
    return mask_video.with_name(f"{mask_video.stem}__candidate_{attempt_index:02d}{mask_video.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval")
    parser.add_argument("--mask-plan-path", required=True)
    parser.add_argument("--mask-manifest-path", required=True)
    parser.add_argument("--output-manifest-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--grounded-sam2-code", required=True)
    parser.add_argument("--grounder", choices=("auto", "groundingdino", "florence2"), default="auto")
    parser.add_argument("--florence-model", default="")
    parser.add_argument("--grounding-dino-config", required=True)
    parser.add_argument("--grounding-dino-checkpoint", required=True)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-masks", type=int)
    args = parser.parse_args()

    root = Path(args.root)
    mask_plans = _load_jsonl(Path(args.mask_plan_path))
    original_manifest = _load_jsonl(Path(args.mask_manifest_path))
    manifest_by_plan = {str(row.get("plan_id", "")): dict(row) for row in original_manifest}
    grounded_sam2_code = Path(args.grounded_sam2_code)
    grounding_config = Path(args.grounding_dino_config)
    grounding_checkpoint = None
    grounding_checkpoint_error = ""
    if args.grounder in {"auto", "groundingdino"}:
        try:
            grounding_checkpoint = _find_grounding_dino_checkpoint(Path(args.grounding_dino_checkpoint))
        except Exception as exc:
            if args.grounder == "groundingdino":
                raise
            grounding_checkpoint_error = f"{type(exc).__name__}: {exc}"
    florence_model_dir = None
    if args.grounder in {"auto", "florence2"} and args.florence_model:
        florence_model_dir = _find_dir(Path(args.florence_model), ("config.json",))
    effective_grounder = args.grounder
    if args.grounder == "auto":
        if grounding_checkpoint is not None:
            effective_grounder = "groundingdino"
        elif florence_model_dir is not None:
            effective_grounder = "florence2"
        else:
            raise RuntimeError(
                "auto grounder could not find a usable GroundingDINO checkpoint or Florence-2 model. "
                f"GroundingDINO error: {grounding_checkpoint_error}"
            )
    sam2_checkpoint = _find_file(Path(args.sam2_checkpoint), ("*.pt", "*.pth"))

    output_records: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for plan in mask_plans[: args.max_masks if args.max_masks and args.max_masks > 0 else None]:
        plan_id = str(plan.get("plan_id", "")).strip()
        row = manifest_by_plan.get(plan_id, {"plan_id": plan_id})
        attempts: list[dict[str, Any]] = []
        candidate_videos_to_cleanup: list[Path] = []
        try:
            reference_video = _resolve_video(root, str(plan.get("reference_video", "")))
            mask_video = Path(str(plan.get("mask_video") or row.get("mask_video", "")))
            if not mask_video.is_absolute():
                mask_video = Path(args.mask_manifest_path).resolve().parent / mask_video
            mask_gate = plan.get("mask_gate") if isinstance(plan.get("mask_gate"), dict) else {}
            metrics: dict[str, Any] | None = None
            selected_query = ""
            selected_candidate_video: Path | None = None
            selected_gate_errors: list[str] = []
            selected_status = "failed"
            for attempt_index, candidate_query in enumerate(_mask_query_candidates(plan), start=1):
                candidate_video = _candidate_mask_video_path(mask_video, attempt_index)
                try:
                    if candidate_video.exists():
                        candidate_video.unlink()
                    if effective_grounder == "florence2":
                        if florence_model_dir is None:
                            raise RuntimeError("--florence-model is required when --grounder florence2")
                        candidate_metrics = _run_florence_sam2(
                            reference_video=reference_video,
                            output_mask_video=candidate_video,
                            mask_query=candidate_query,
                            mask_mode=str(plan.get("mask_mode", "")).strip(),
                            florence_model_dir=florence_model_dir,
                            grounded_sam2_code=grounded_sam2_code,
                            sam2_config=str(args.sam2_config),
                            sam2_checkpoint=sam2_checkpoint,
                            mask_gate=mask_gate,
                        )
                    else:
                        if grounding_checkpoint is None:
                            raise RuntimeError("GroundingDINO checkpoint is required")
                        candidate_metrics = _run_grounded_sam2(
                            reference_video=reference_video,
                            output_mask_video=candidate_video,
                            mask_query=candidate_query,
                            mask_mode=str(plan.get("mask_mode", "")).strip(),
                            grounded_sam2_code=grounded_sam2_code,
                            grounding_dino_config=grounding_config,
                            grounding_dino_checkpoint=grounding_checkpoint,
                            sam2_config=str(args.sam2_config),
                            sam2_checkpoint=sam2_checkpoint,
                            box_threshold=args.box_threshold,
                            text_threshold=args.text_threshold,
                            mask_gate=mask_gate,
                        )
                    gate_errors = _mask_gate_errors(mask_gate, candidate_metrics)
                    candidate_metrics = _append_adaptive_provenance(
                        candidate_metrics,
                        gate_errors=gate_errors,
                        detection_attempts=candidate_metrics.get("detection_attempts", []),
                        anchor_frame_indices=candidate_metrics.get("anchor_frame_indices", []),
                        prompt_type=str(candidate_metrics.get("prompt_type", "box") or "box"),
                    )
                    attempts.append(
                        {
                            "mask_query": candidate_query,
                            "status": "generated" if not gate_errors else "failed_gate",
                            "gate_errors": gate_errors,
                            "mask_metrics": candidate_metrics,
                            "candidate_mask_video": str(candidate_video),
                            "mask_quality_tier": candidate_metrics.get("mask_quality_tier"),
                            "usable_for_vace": bool(candidate_metrics.get("usable_for_vace")),
                        }
                    )
                    candidate_videos_to_cleanup.append(candidate_video)
                    if metrics is None:
                        metrics = candidate_metrics
                        selected_query = candidate_query
                        selected_candidate_video = candidate_video
                        selected_gate_errors = gate_errors
                        selected_status = "generated" if not gate_errors else "diagnostic_generated"
                    else:
                        previous_tier = str(metrics.get("mask_quality_tier", MASK_QUALITY_FAILED))
                        current_tier = str(candidate_metrics.get("mask_quality_tier", MASK_QUALITY_FAILED))
                        rank = {
                            MASK_QUALITY_EXCELLENT: 4,
                            MASK_QUALITY_USABLE: 3,
                            MASK_QUALITY_DIAGNOSTIC: 2,
                            MASK_QUALITY_FAILED: 1,
                        }
                        if rank.get(current_tier, 0) > rank.get(previous_tier, 0):
                            metrics = candidate_metrics
                            selected_query = candidate_query
                            selected_candidate_video = candidate_video
                            selected_gate_errors = gate_errors
                            selected_status = "generated" if not gate_errors else "diagnostic_generated"
                    if not gate_errors:
                        break
                except Exception as attempt_exc:
                    if candidate_video.exists():
                        candidate_video.unlink()
                    attempts.append(
                        {
                            "mask_query": candidate_query,
                            "status": "error",
                            "error": f"{type(attempt_exc).__name__}: {attempt_exc}",
                        }
                    )
            if metrics is None:
                mask_video.parent.mkdir(parents=True, exist_ok=True)
                if mask_video.exists():
                    mask_video.unlink()
                metrics = _write_all_black_mask_video(reference_video, mask_video)
                selected_query = str(plan.get("mask_query", "")).strip()
                selected_gate_errors = list(metrics.get("failure_reasons", []))
                selected_status = "diagnostic_generated"
            else:
                if selected_candidate_video is None:
                    raise RuntimeError("selected mask candidate is missing")
                mask_video.parent.mkdir(parents=True, exist_ok=True)
                if mask_video.exists():
                    mask_video.unlink()
                shutil.move(str(selected_candidate_video), str(mask_video))
                for candidate_video in candidate_videos_to_cleanup:
                    if candidate_video != selected_candidate_video and candidate_video.exists():
                        candidate_video.unlink()
            if not attempts and selected_status == "diagnostic_generated":
                attempts.append(
                    {
                        "mask_query": selected_query,
                        "status": "diagnostic_all_black",
                        "gate_errors": selected_gate_errors,
                        "mask_metrics": metrics,
                        "mask_quality_tier": metrics.get("mask_quality_tier"),
                        "usable_for_vace": False,
                    }
                )
            target_instance_description = str(plan.get("target_instance_description", "")).strip()
            mask_target_instance_alignment = {}
            if target_instance_description:
                mask_target_instance_alignment = {
                    "passed": False,
                    "method": "not_checked",
                    "reason": "target_instance_description requires external Omni/contact-sheet alignment review",
                    "target_instance_description": target_instance_description,
                }
            default_usable_for_vace = plan.get("usable_for_vace_default", row.get("usable_for_vace_default", True)) is not False
            maskability_issue = str(plan.get("maskability_issue") or row.get("maskability_issue") or "").strip()
            final_usable_for_vace = bool(metrics.get("usable_for_vace")) and not selected_gate_errors and default_usable_for_vace
            if not default_usable_for_vace:
                selected_status = "diagnostic_generated"
            row.update(
                {
                    "mask_video": str(mask_video),
                    "status": selected_status,
                    "mask_metrics": metrics,
                    "mask_gate_result": {"passed": not selected_gate_errors, "errors": selected_gate_errors},
                    "mask_quality_tier": metrics.get("mask_quality_tier", MASK_QUALITY_FAILED),
                    "usable_for_vace": final_usable_for_vace,
                    "failure_reasons": selected_gate_errors,
                    "maskability_issue": maskability_issue,
                    "usable_for_vace_default": default_usable_for_vace,
                    "sparse_full_length": bool(metrics.get("sparse_full_length", False)),
                    "visible_spans": metrics.get("visible_spans", []),
                    "mask_generation_strategy": MASK_GENERATION_STRATEGY,
                    "detector_cascade": metrics.get("detector_cascade", []),
                    "detection_attempts": metrics.get("detection_attempts", []),
                    "anchor_frame_indices": metrics.get("anchor_frame_indices", []),
                    "prompt_type": metrics.get("prompt_type", ""),
                    "reinit_count": metrics.get("reinit_count", 0),
                    "repair_rounds": metrics.get("repair_rounds", 0),
                    "mask_query": selected_query,
                    "mask_query_candidates": _mask_query_candidates(plan),
                    "mask_attempts": attempts,
                    "mask_mode": str(plan.get("mask_mode", "")).strip(),
                    "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
                    "mask_polarity": VIDEO_MASK_POLARITY,
                    "sampled_frame_indices": metrics.get("sampled_frame_indices", []),
                    "detected_keyframe_index": metrics.get("detected_keyframe_index"),
                    "reference_frame_count": metrics.get("reference_frame_count"),
                    "reference_fps": metrics.get("reference_fps"),
                    "generator_commit": metrics.get("generator_commit", ""),
                }
            )
            if mask_target_instance_alignment:
                row["mask_target_instance_alignment"] = mask_target_instance_alignment
        except Exception as exc:  # pragma: no cover - integration path
            try:
                reference_video = _resolve_video(root, str(plan.get("reference_video", "")))
                mask_video = Path(str(plan.get("mask_video") or row.get("mask_video", "")))
                if not mask_video.is_absolute():
                    mask_video = Path(args.mask_manifest_path).resolve().parent / mask_video
                if reference_video.exists():
                    diagnostic_metrics = _write_all_black_mask_video(reference_video, mask_video)
                    row.update(
                        {
                            "mask_video": str(mask_video),
                            "status": "diagnostic_generated",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "failure_reasons": [f"{type(exc).__name__}: {exc}"],
                            "mask_metrics": diagnostic_metrics,
                            "mask_gate_result": {"passed": False, "errors": diagnostic_metrics.get("failure_reasons", [])},
                            "mask_quality_tier": MASK_QUALITY_FAILED,
                            "usable_for_vace": False,
                            "sparse_full_length": True,
                            "visible_spans": [],
                            "mask_generation_strategy": MASK_GENERATION_STRATEGY,
                            "detector_cascade": diagnostic_metrics.get("detector_cascade", []),
                            "detection_attempts": diagnostic_metrics.get("detection_attempts", []),
                            "anchor_frame_indices": [],
                            "prompt_type": "none",
                            "reinit_count": 0,
                            "repair_rounds": 0,
                            "mask_query": str(plan.get("mask_query", "")).strip(),
                            "mask_query_candidates": _mask_query_candidates(plan),
                            "mask_attempts": attempts,
                            "mask_mode": str(plan.get("mask_mode", "")).strip(),
                            "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
                            "mask_polarity": VIDEO_MASK_POLARITY,
                            "generator_commit": _git_head_short(),
                        }
                    )
                else:
                    raise FileNotFoundError(reference_video)
            except Exception:
                row.update(
                    {
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "mask_query": str(plan.get("mask_query", "")).strip(),
                        "mask_query_candidates": _mask_query_candidates(plan),
                        "mask_attempts": attempts,
                        "mask_mode": str(plan.get("mask_mode", "")).strip(),
                        "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
                        "mask_polarity": VIDEO_MASK_POLARITY,
                        "mask_generation_strategy": MASK_GENERATION_STRATEGY,
                        "mask_quality_tier": MASK_QUALITY_FAILED,
                        "usable_for_vace": False,
                        "generator_commit": _git_head_short(),
                    }
                )
        output_records.append(row)
        report_rows.append(
            {
                "plan_id": plan_id,
                "mask_query": row.get("mask_query", plan.get("mask_query")),
                "status": row.get("status"),
                "mask_video": row.get("mask_video"),
                "failure_reason": row.get("failure_reason", ""),
                "failure_reasons": row.get("failure_reasons", []),
                "mask_quality_tier": row.get("mask_quality_tier", ""),
                "usable_for_vace": row.get("usable_for_vace", False),
                "visible_spans": row.get("visible_spans", []),
                "reinit_count": row.get("reinit_count", 0),
                "mask_metrics": row.get("mask_metrics", {}),
            }
        )

    _write_jsonl(Path(args.output_manifest_path), output_records)
    if args.report_path:
        report = ["# Grounded-SAM-2 Mask Report", ""]
        for row in report_rows:
            reasons = row.get("failure_reasons")
            if isinstance(reasons, list) and reasons:
                reason_text = "; ".join(str(item) for item in reasons)
            else:
                reason_text = str(row.get("failure_reason", ""))
            report.append(
                f"- `{row['plan_id']}` query=`{row.get('mask_query')}` status=`{row.get('status')}` "
                f"tier=`{row.get('mask_quality_tier')}` usable_for_vace=`{row.get('usable_for_vace')}` "
                f"visible_spans=`{len(row.get('visible_spans') or [])}` reinit_count=`{row.get('reinit_count')}` "
                f"mask=`{row.get('mask_video')}` reason=`{reason_text}`"
            )
        Path(args.report_path).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"mask_count": len(output_records), "output_manifest_path": args.output_manifest_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
