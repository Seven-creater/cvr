from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


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
        first_frame_path = frame_paths[0]
        image_source, image = load_image(str(first_frame_path))
        boxes, logits, phrases = predict(
            model=grounding_model,
            image=image,
            caption=mask_query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        if boxes is None or len(boxes) == 0:
            raise RuntimeError(f"GroundingDINO found no box for query: {mask_query}")

        h, w = image_source.shape[:2]
        boxes_np = boxes.detach().cpu().numpy()
        # GroundingDINO boxes are normalized cxcywh.
        boxes_xyxy = np.zeros_like(boxes_np)
        boxes_xyxy[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * h

        predictor = build_sam2_video_predictor(sam2_config, str(sam2_checkpoint), device=device)
        inference_state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(inference_state)
        for object_id, box in enumerate(boxes_xyxy[:3], start=1):
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                box=box,
            )

        frame_masks: dict[int, np.ndarray] = {}
        for frame_idx, _object_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks = (mask_logits > 0.0).detach().cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            union = np.any(masks, axis=0).astype("uint8")
            if mask_mode == "edit_background_inverse_subject":
                union = 1 - union
            frame_masks[int(frame_idx)] = union

        coverages: list[float] = []
        ious: list[float] = []
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
            if last_mask is not None:
                intersection = float(np.logical_and(last_mask, mask).sum())
                union = float(np.logical_or(last_mask, mask).sum())
                ious.append(intersection / union if union > 0 else 0.0)
            last_mask = mask.copy()
            mask_rgb = (mask * 255).astype("uint8")
            cv2.imwrite(str(mask_dir / f"{idx:05d}.png"), mask_rgb)

        fps = _ffprobe_fps(reference_video)
        _encode_mask_video(mask_dir, output_mask_video, fps)
        return {
            "frame_count": len(frame_paths),
            "fps": fps,
            "mask_coverage_ratio_min": min(coverages) if coverages else 0.0,
            "mask_coverage_ratio_avg": sum(coverages) / len(coverages) if coverages else 0.0,
            "mask_coverage_ratio_max": max(coverages) if coverages else 0.0,
            "mask_temporal_stability": sum(ious) / len(ious) if ious else 1.0,
            "detected_phrases": [str(item) for item in phrases],
            "box_count": int(len(boxes_xyxy)),
            "device": device,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval")
    parser.add_argument("--mask-plan-path", required=True)
    parser.add_argument("--mask-manifest-path", required=True)
    parser.add_argument("--output-manifest-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--grounded-sam2-code", required=True)
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
    grounding_checkpoint = _find_file(Path(args.grounding_dino_checkpoint), ("*.pth", "*.pt"))
    sam2_checkpoint = _find_file(Path(args.sam2_checkpoint), ("*.pt", "*.pth"))

    output_records: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for plan in mask_plans[: args.max_masks if args.max_masks and args.max_masks > 0 else None]:
        plan_id = str(plan.get("plan_id", "")).strip()
        row = manifest_by_plan.get(plan_id, {"plan_id": plan_id})
        try:
            reference_video = _resolve_video(root, str(plan.get("reference_video", "")))
            mask_video = Path(str(plan.get("mask_video") or row.get("mask_video", "")))
            if not mask_video.is_absolute():
                mask_video = Path(args.mask_manifest_path).resolve().parent / mask_video
            metrics = _run_grounded_sam2(
                reference_video=reference_video,
                output_mask_video=mask_video,
                mask_query=str(plan.get("mask_query", "")).strip(),
                mask_mode=str(plan.get("mask_mode", "")).strip(),
                grounded_sam2_code=grounded_sam2_code,
                grounding_dino_config=grounding_config,
                grounding_dino_checkpoint=grounding_checkpoint,
                sam2_config=str(args.sam2_config),
                sam2_checkpoint=sam2_checkpoint,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            row.update(
                {
                    "mask_video": str(mask_video),
                    "status": "generated",
                    "mask_metrics": metrics,
                }
            )
        except Exception as exc:  # pragma: no cover - integration path
            row.update({"status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
        output_records.append(row)
        report_rows.append(
            {
                "plan_id": plan_id,
                "mask_query": plan.get("mask_query"),
                "status": row.get("status"),
                "mask_video": row.get("mask_video"),
                "failure_reason": row.get("failure_reason", ""),
                "mask_metrics": row.get("mask_metrics", {}),
            }
        )

    _write_jsonl(Path(args.output_manifest_path), output_records)
    if args.report_path:
        report = ["# Grounded-SAM-2 Mask Report", ""]
        for row in report_rows:
            report.append(
                f"- `{row['plan_id']}` query=`{row.get('mask_query')}` status=`{row.get('status')}` "
                f"mask=`{row.get('mask_video')}` reason=`{row.get('failure_reason', '')}`"
            )
        Path(args.report_path).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"mask_count": len(output_records), "output_manifest_path": args.output_manifest_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
