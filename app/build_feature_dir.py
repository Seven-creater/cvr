from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from app.retriever import load_split_ids

VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov")
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


@dataclass(frozen=True, slots=True)
class BuildSummary:
    output_dir: str
    video_count: int
    text_count: int
    wrote_text_embeddings: bool
    wrote_video_visual_embeddings: bool
    wrote_video_audio_embeddings: bool

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "video_count": self.video_count,
            "text_count": self.text_count,
            "wrote_text_embeddings": self.wrote_text_embeddings,
            "wrote_video_visual_embeddings": self.wrote_video_visual_embeddings,
            "wrote_video_audio_embeddings": self.wrote_video_audio_embeddings,
        }


def load_msrvtt_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_paths(root: Path, stem: str, suffixes: Iterable[str], default_suffix: str) -> list[Path]:
    return [root / f"{stem}{suffix}" for suffix in tuple(suffixes) + (default_suffix,)]


def resolve_media_path(
    media_id: str,
    *,
    root: str | Path | None,
    suffixes: Iterable[str],
    default_suffix: str,
) -> str:
    if not root:
        return ""
    root_path = Path(root)
    for candidate in _candidate_paths(root_path, media_id, suffixes, default_suffix):
        if candidate.exists():
            return str(candidate)
    return str(root_path / f"{media_id}{default_suffix}")


def _sorted_sentences(payload: dict) -> list[dict]:
    rows = payload.get("sentences") or payload.get("annotations") or []
    if not isinstance(rows, list):
        raise ValueError("MSRVTT json must contain a list under 'sentences' or 'annotations'")
    return rows


def _video_entries(payload: dict) -> list[dict]:
    videos = payload.get("videos") or []
    if videos and not isinstance(videos, list):
        raise ValueError("MSRVTT json field 'videos' must be a list")
    return videos


def _caption_text(row: dict) -> str:
    for key in ("caption", "sentence", "text"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"sentence row missing caption text: {row}")


def build_rows(
    *,
    msrvtt_json_path: str | Path,
    split_csv_path: str | Path | None = None,
    video_root: str | Path | None = None,
    audio_root: str | Path | None = None,
    default_video_ext: str = ".mp4",
    default_audio_ext: str = ".wav",
) -> tuple[list[dict], list[dict]]:
    payload = load_msrvtt_json(msrvtt_json_path)
    split_ids = load_split_ids(split_csv_path)

    videos = _video_entries(payload)
    sentences = _sorted_sentences(payload)

    video_ids_from_sentences = []
    seen_video_ids: set[str] = set()
    for row in sentences:
        video_id = str(row["video_id"])
        if split_ids is not None and video_id not in split_ids:
            continue
        if video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            video_ids_from_sentences.append(video_id)

    video_lookup = {str(row["video_id"]): row for row in videos if "video_id" in row}

    video_rows: list[dict] = []
    for video_id in video_ids_from_sentences:
        video_meta = video_lookup.get(video_id, {})
        actual_video_id = str(video_meta.get("video_id", video_id))
        video_rows.append(
            {
                "video_id": actual_video_id,
                "video_path": resolve_media_path(
                    actual_video_id,
                    root=video_root,
                    suffixes=VIDEO_SUFFIXES,
                    default_suffix=default_video_ext,
                ),
                "audio_path": resolve_media_path(
                    actual_video_id,
                    root=audio_root,
                    suffixes=AUDIO_SUFFIXES,
                    default_suffix=default_audio_ext,
                )
                if audio_root
                else None,
            }
        )

    caption_counter: dict[str, int] = {}
    text_rows: list[dict] = []
    for row in sentences:
        video_id = str(row["video_id"])
        if split_ids is not None and video_id not in split_ids:
            continue
        caption_counter[video_id] = caption_counter.get(video_id, 0) + 1
        text_id = row.get("text_id") or row.get("sen_id") or f"{video_id}::caption::{caption_counter[video_id]}"
        text_rows.append(
            {
                "text_id": str(text_id),
                "video_id": video_id,
                "text": _caption_text(row),
            }
        )

    return text_rows, video_rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_embedding(path: str | Path) -> np.ndarray:
    return np.asarray(np.load(Path(path)), dtype=np.float32)


def _write_embedding(path: str | Path, matrix: np.ndarray) -> None:
    np.save(Path(path), np.asarray(matrix, dtype=np.float32))


def build_feature_dir(
    *,
    msrvtt_json_path: str | Path,
    output_dir: str | Path,
    split_csv_path: str | Path | None = None,
    video_root: str | Path | None = None,
    audio_root: str | Path | None = None,
    text_embeddings_in: str | Path | None = None,
    video_visual_embeddings_in: str | Path | None = None,
    video_audio_embeddings_in: str | Path | None = None,
    default_video_ext: str = ".mp4",
    default_audio_ext: str = ".wav",
) -> BuildSummary:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    text_rows, video_rows = build_rows(
        msrvtt_json_path=msrvtt_json_path,
        split_csv_path=split_csv_path,
        video_root=video_root,
        audio_root=audio_root,
        default_video_ext=default_video_ext,
        default_audio_ext=default_audio_ext,
    )
    write_jsonl(output_root / "text_rows.jsonl", text_rows)
    write_jsonl(output_root / "video_rows.jsonl", video_rows)

    wrote_text_embeddings = False
    wrote_video_visual_embeddings = False
    wrote_video_audio_embeddings = False

    if text_embeddings_in:
        matrix = _load_embedding(text_embeddings_in)
        if matrix.shape[0] != len(text_rows):
            raise ValueError("text_embeddings row count does not match text_rows.jsonl")
        _write_embedding(output_root / "text_embeddings.npy", matrix)
        wrote_text_embeddings = True

    if video_visual_embeddings_in:
        matrix = _load_embedding(video_visual_embeddings_in)
        if matrix.shape[0] != len(video_rows):
            raise ValueError("video_visual_embeddings row count does not match video_rows.jsonl")
        _write_embedding(output_root / "video_visual_embeddings.npy", matrix)
        wrote_video_visual_embeddings = True

    if video_audio_embeddings_in:
        matrix = _load_embedding(video_audio_embeddings_in)
        if matrix.shape[0] != len(video_rows):
            raise ValueError("video_audio_embeddings row count does not match video_rows.jsonl")
        _write_embedding(output_root / "video_audio_embeddings.npy", matrix)
        wrote_video_audio_embeddings = True

    summary = BuildSummary(
        output_dir=str(output_root),
        video_count=len(video_rows),
        text_count=len(text_rows),
        wrote_text_embeddings=wrote_text_embeddings,
        wrote_video_visual_embeddings=wrote_video_visual_embeddings,
        wrote_video_audio_embeddings=wrote_video_audio_embeddings,
    )
    with (output_root / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a feature-dir for the minimal MSRVTT retriever")
    parser.add_argument("--msrvtt-json", required=True, help="Path to MSRVTT_data.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature-dir files")
    parser.add_argument("--split-csv", help="Optional split csv such as MSRVTT_JSFUSION_test.csv")
    parser.add_argument("--video-root", help="Optional root directory that contains <video_id>.mp4")
    parser.add_argument("--audio-root", help="Optional root directory that contains <video_id>.wav")
    parser.add_argument("--text-embeddings-in", help="Optional .npy file to copy as text_embeddings.npy")
    parser.add_argument(
        "--video-visual-embeddings-in",
        help="Optional .npy file to copy as video_visual_embeddings.npy",
    )
    parser.add_argument(
        "--video-audio-embeddings-in",
        help="Optional .npy file to copy as video_audio_embeddings.npy",
    )
    parser.add_argument("--default-video-ext", default=".mp4", help="Default suffix when video file is not found")
    parser.add_argument("--default-audio-ext", default=".wav", help="Default suffix when audio file is not found")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_feature_dir(
        msrvtt_json_path=args.msrvtt_json,
        output_dir=args.output_dir,
        split_csv_path=args.split_csv,
        video_root=args.video_root,
        audio_root=args.audio_root,
        text_embeddings_in=args.text_embeddings_in,
        video_visual_embeddings_in=args.video_visual_embeddings_in,
        video_audio_embeddings_in=args.video_audio_embeddings_in,
        default_video_ext=args.default_video_ext,
        default_audio_ext=args.default_audio_ext,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
