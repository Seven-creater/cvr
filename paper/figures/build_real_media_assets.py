#!/usr/bin/env python3
"""Build publication figure assets from a frozen Audio-CVR example."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


FRAME_QUANTILES = (0.2, 0.5, 0.8)
NAVY = (29, 52, 77)
BLUE = (45, 114, 178)
CORAL = (213, 94, 73)
LIGHT = (239, 243, 247)
MID = (192, 203, 214)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_video(path: Path) -> tuple[list[Image.Image], np.ndarray]:
    container = av.open(str(path))
    video_frames = [
        frame.to_image().convert("RGB") for frame in container.decode(video=0)
    ]
    container.close()
    if not video_frames:
        raise RuntimeError(f"No video frames decoded from {path}")

    audio_samples: list[np.ndarray] = []
    container = av.open(str(path))
    if container.streams.audio:
        for frame in container.decode(audio=0):
            block = frame.to_ndarray().astype(np.float32)
            if block.ndim == 2:
                block = block.mean(axis=0)
            audio_samples.append(block.reshape(-1))
    container.close()
    audio = (
        np.concatenate(audio_samples)
        if audio_samples
        else np.zeros(4096, dtype=np.float32)
    )
    peak = float(np.max(np.abs(audio))) or 1.0
    return video_frames, audio / peak


def _pick_frames(frames: list[Image.Image]) -> list[Image.Image]:
    last = len(frames) - 1
    return [frames[round(last * value)] for value in FRAME_QUANTILES]


def _fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _filmstrip(frames: list[Image.Image], output: Path) -> None:
    tile_size = (320, 180)
    gap = 10
    canvas = Image.new(
        "RGB",
        (tile_size[0] * len(frames) + gap * (len(frames) - 1), tile_size[1]),
        "white",
    )
    for index, frame in enumerate(frames):
        canvas.paste(_fit_cover(frame, tile_size), (index * (tile_size[0] + gap), 0))
    canvas.save(output, quality=94, subsampling=0)


def _waveform(
    audio: np.ndarray,
    output: Path,
    color: tuple[int, int, int],
    width: int = 980,
    height: int = 150,
) -> None:
    bins = min(width, max(1, audio.size))
    boundaries = np.linspace(0, audio.size, bins + 1, dtype=int)
    envelope = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        segment = audio[boundaries[index] : boundaries[index + 1]]
        envelope[index] = np.max(np.abs(segment)) if segment.size else 0.0
    if float(envelope.max()) > 0:
        envelope /= float(envelope.max())

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    center = height // 2
    draw.line((0, center, width, center), fill=(*MID, 200), width=2)
    points_top = [(x, center - round(value * (height * 0.42))) for x, value in enumerate(envelope)]
    points_bottom = [
        (x, center + round(value * (height * 0.42)))
        for x, value in reversed(list(enumerate(envelope)))
    ]
    draw.polygon(points_top + points_bottom, fill=(*color, 205))
    canvas.save(output)


def _audio_pair(
    reference_waveform: Image.Image,
    target_waveform: Image.Image,
    output: Path,
) -> None:
    width = max(reference_waveform.width, target_waveform.width)
    canvas = Image.new("RGBA", (width, 330), (255, 255, 255, 0))
    canvas.alpha_composite(reference_waveform, (0, 5))
    canvas.alpha_composite(target_waveform, (0, 175))
    canvas.save(output)


def _muted_pair(
    reference: Image.Image,
    target: Image.Image,
    output: Path,
) -> None:
    tile = (420, 236)
    gap = 30
    canvas = Image.new("RGB", (tile[0] * 2 + gap, tile[1]), LIGHT)
    for index, image in enumerate((reference, target)):
        muted = _fit_cover(image, tile).convert("L").convert("RGB")
        overlay = Image.new("RGB", tile, (235, 239, 243))
        muted = Image.blend(muted, overlay, 0.18)
        canvas.paste(muted, (index * (tile[0] + gap), 0))
    canvas.save(output, quality=94, subsampling=0)


def _gallery(
    selected_reference: Image.Image,
    selected_target: Image.Image,
    distractors: list[Image.Image],
    output: Path,
) -> None:
    tile = (300, 169)
    gap = 18
    images = [selected_target, selected_reference, *distractors[:4]]
    canvas = Image.new("RGB", (tile[0] * 3 + gap * 2, tile[1] * 2 + gap), "white")
    for index, image in enumerate(images):
        row, column = divmod(index, 3)
        canvas.paste(
            _fit_cover(image, tile),
            (column * (tile[0] + gap), row * (tile[1] + gap)),
        )
    canvas.save(output, quality=94, subsampling=0)


def build_assets(candidate_root: Path, selected_index: int, output: Path) -> None:
    manifest_path = candidate_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["candidates"]
    selected = records[selected_index]
    selected_dir = candidate_root / f"{selected_index:02d}_{selected['sample_id'][-8:]}"
    output.mkdir(parents=True, exist_ok=True)

    ref_frames, ref_audio = _decode_video(selected_dir / "reference.mp4")
    tgt_frames, tgt_audio = _decode_video(selected_dir / "target.mp4")
    ref_selected = _pick_frames(ref_frames)
    tgt_selected = _pick_frames(tgt_frames)

    _filmstrip(ref_selected, output / "reference_filmstrip.jpg")
    _filmstrip(tgt_selected, output / "target_filmstrip.jpg")
    _waveform(ref_audio, output / "reference_waveform.png", BLUE)
    _waveform(tgt_audio, output / "target_waveform.png", CORAL)
    _audio_pair(
        Image.open(output / "reference_waveform.png").convert("RGBA"),
        Image.open(output / "target_waveform.png").convert("RGBA"),
        output / "audio_only_pair.png",
    )
    _muted_pair(ref_selected[1], tgt_selected[1], output / "muted_video_pair.jpg")

    distractors: list[Image.Image] = []
    for record in records:
        if record["index"] == selected_index:
            continue
        directory = candidate_root / f"{record['index']:02d}_{record['sample_id'][-8:]}"
        distractors.append(Image.open(directory / "target_frame_2.jpg").convert("RGB"))
    _gallery(
        ref_selected[1],
        tgt_selected[1],
        distractors,
        output / "real_gallery.jpg",
    )

    provenance = {
        "frozen_test_sha256": (
            "70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e"
        ),
        "selection_rule": manifest["selection_rule"],
        "selection_note": (
            "The selected example was chosen for visual legibility from a fixed, "
            "score-independent candidate set."
        ),
        "selected_record": {
            key: selected[key]
            for key in ("sample_id", "dataset", "b_subtype", "edit_text", "raw_source_id")
        },
        "frame_quantiles": FRAME_QUANTILES,
        "reference_video_sha256": _sha256(selected_dir / "reference.mp4"),
        "target_video_sha256": _sha256(selected_dir / "target.mp4"),
        "assets": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("paper/figures/assets/test1000_candidates"),
    )
    parser.add_argument("--selected-index", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/assets/real_example"),
    )
    args = parser.parse_args()
    build_assets(args.candidate_root, args.selected_index, args.output)


if __name__ == "__main__":
    main()
