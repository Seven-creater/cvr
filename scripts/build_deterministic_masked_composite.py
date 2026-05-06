#!/usr/bin/env python3
"""Build a fixed-background masked composite video without generative VACE.

The input mask follows the VACE convention used by this pipeline:
white = editable/generated background, black = retained foreground.  For
deterministic background replacement we use that mask directly as the selector:
black pixels keep the reference frame, white pixels take the fixed src_ref plate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(cmd)}\nsee {log_path}")


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next((row for row in streams if row.get("codec_type") == "video"), {})
    audio_stream = next((row for row in streams if row.get("codec_type") == "audio"), {})
    raw_frames = video_stream.get("nb_read_frames") or video_stream.get("nb_frames") or 0
    try:
        frame_count = int(raw_frames)
    except (TypeError, ValueError):
        frame_count = 0
    return {
        "path": str(path),
        "duration_seconds": float((payload.get("format") or {}).get("duration") or video_stream.get("duration") or 0.0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "frame_count": frame_count,
        "has_video": bool(video_stream),
        "has_audio": bool(audio_stream),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--src-ref-image", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--target-output", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--frame-num", required=True, type=int)
    parser.add_argument("--mask-feather", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root: Path = args.out_root
    log_dir = out_root / "logs"
    review_dir = out_root / "review_inputs"
    metadata_dir = out_root / "metadata"
    for path in (log_dir, review_dir, metadata_dir, args.raw_output.parent, args.target_output.parent):
        path.mkdir(parents=True, exist_ok=True)

    for label, path in (
        ("reference", args.reference),
        ("mask", args.mask),
        ("src_ref_image", args.src_ref_image),
    ):
        if not path.exists() or path.stat().st_size <= 0:
            raise SystemExit(f"missing required {label}: {path}")

    reference_probe = _probe(args.reference)
    width = reference_probe["width"]
    height = reference_probe["height"]
    if not width or not height:
        raise SystemExit(f"reference has no usable video stream: {args.reference}")

    src_ref_plate = review_dir / "src_ref_plate.png"
    alpha_contact = review_dir / "alpha_contact.jpg"
    composite_contact = review_dir / "composite_target_contact.jpg"
    background_video = metadata_dir / "deterministic_background_plate.mp4"
    alpha_mask = metadata_dir / "deterministic_alpha_mask.mp4"

    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.src_ref_image),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=rgb24",
            "-frames:v",
            "1",
            str(src_ref_plate),
        ],
        log_path=log_dir / "deterministic_src_ref_plate.log",
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            f"{args.fps:g}",
            "-i",
            str(src_ref_plate),
            "-frames:v",
            str(args.frame_num),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(background_video),
        ],
        log_path=log_dir / "deterministic_background_video.log",
    )

    mask_filter = f"fps={args.fps:g},scale={width}:{height},format=gray"
    if args.mask_feather > 0:
        mask_filter += f",boxblur=luma_radius={args.mask_feather:g}:luma_power=1"
    mask_filter += f",trim=start_frame=0:end_frame={args.frame_num},setpts=N/{args.fps:g}/TB"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.mask),
            "-filter_complex",
            f"[0:v]{mask_filter}[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(alpha_mask),
        ],
        log_path=log_dir / "deterministic_alpha_mask.log",
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(alpha_mask),
            "-vf",
            "fps=1,scale=240:-1,tile=5x1",
            "-frames:v",
            "1",
            str(alpha_contact),
        ],
        log_path=log_dir / "contact_alpha.log",
    )

    # maskedmerge keeps input 0 where mask is black and input 1 where mask is white.
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.reference),
            "-i",
            str(background_video),
            "-i",
            str(alpha_mask),
            "-filter_complex",
            f"[0:v]fps={args.fps:g},scale={width}:{height},format=yuv420p[ref];"
            f"[1:v]fps={args.fps:g},scale={width}:{height},format=yuv420p[bg];"
            f"[2:v]format=gray[m];"
            "[ref][bg][m]maskedmerge[out]",
            "-map",
            "[out]",
            "-frames:v",
            str(args.frame_num),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(args.raw_output),
        ],
        log_path=log_dir / "deterministic_composite_raw.log",
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.raw_output),
            "-i",
            str(args.reference),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(args.target_output),
        ],
        log_path=log_dir / "deterministic_remux_audio.log",
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.target_output),
            "-vf",
            "fps=1,scale=240:-1,tile=5x1",
            "-frames:v",
            "1",
            str(composite_contact),
        ],
        log_path=log_dir / "contact_composite_target.log",
    )

    metrics = {
        "route": "deterministic_foreground_background_composite",
        "requires_vace": False,
        "mask_polarity": "white_generate_black_preserve",
        "alpha_semantics": "white mask pixels take the fixed background plate; black mask pixels keep the reference foreground",
        "mask_feather_radius": args.mask_feather,
        "src_ref_plate": str(src_ref_plate),
        "background_plate_video": str(background_video),
        "alpha_mask": str(alpha_mask),
        "raw_output": _probe(args.raw_output),
        "target_output": _probe(args.target_output),
    }
    (metadata_dir / "deterministic_composite_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
