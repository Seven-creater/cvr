from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CVRTriplet:
    sample_id: str
    reference_video: str
    target_video: str
    edit_text: str
    reference_caption: str = ""
    source: str = ""
    difference_type: str = ""


@dataclass(frozen=True)
class CVRQueryViews:
    sample_id: str
    reference_video: str
    edit_text: str
    reference_caption: str
    avigate_text_query: str
    e5_text_query: str
    e5_video_text_query: dict[str, str]


def load_cvr_triplets_jsonl(
    path: str | Path,
    *,
    sample_size: int | None = None,
    start_index: int = 0,
) -> list[CVRTriplet]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive")

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"triplets jsonl not found: {root}")

    triplets: list[CVRTriplet] = []
    for line_number, line in enumerate(root.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        triplets.append(_triplet_from_dict(payload, line_number=line_number))

    selected = triplets[start_index:] if sample_size is None else triplets[start_index : start_index + sample_size]
    if sample_size is not None and len(selected) < sample_size:
        raise ValueError(f"requested {sample_size} triplets from {root}, found {len(selected)}")
    return selected


def build_cvr_query_views(triplet: CVRTriplet) -> CVRQueryViews:
    avigate_query = compose_avigate_query(triplet.reference_caption, triplet.edit_text)
    e5_text_query = compose_e5_text_query(triplet.reference_caption, triplet.edit_text)
    return CVRQueryViews(
        sample_id=triplet.sample_id,
        reference_video=triplet.reference_video,
        edit_text=triplet.edit_text,
        reference_caption=triplet.reference_caption,
        avigate_text_query=avigate_query,
        e5_text_query=e5_text_query,
        e5_video_text_query={
            "text": e5_text_query,
            "video": triplet.reference_video,
        },
    )


def compose_avigate_query(reference_caption: str, edit_text: str) -> str:
    caption = str(reference_caption or "").strip()
    if caption and caption[-1] not in ".!?":
        caption = f"{caption}."
    edit = str(edit_text or "").strip().rstrip(".")
    if not edit:
        raise ValueError("edit_text is required")
    if caption:
        return f"{caption} Edit: {edit}."
    return f"Edit: {edit}."


def compose_e5_text_query(reference_caption: str, edit_text: str) -> str:
    edit = str(edit_text or "").strip().rstrip(".")
    if not edit:
        raise ValueError("edit_text is required")
    caption = str(reference_caption or "").strip()
    if caption:
        return f"Find a target video matching this edit of the reference video: {edit}. Reference summary: {caption}."
    return f"Find a target video matching this edit of the reference video: {edit}."


def _triplet_from_dict(payload: dict[str, Any], *, line_number: int) -> CVRTriplet:
    sample_id = str(payload.get("sample_id") or payload.get("video_id") or "").strip()
    reference_video = str(payload.get("reference_video") or "").strip()
    target_video = str(payload.get("target_video") or "").strip()
    edit_text = str(payload.get("edit_text") or "").strip()
    if not sample_id:
        raise ValueError(f"triplet line {line_number} missing sample_id")
    if not reference_video:
        raise ValueError(f"triplet line {line_number} missing reference_video")
    if not target_video:
        raise ValueError(f"triplet line {line_number} missing target_video")
    if not edit_text:
        raise ValueError(f"triplet line {line_number} missing edit_text")
    return CVRTriplet(
        sample_id=sample_id,
        reference_video=reference_video,
        target_video=target_video,
        edit_text=edit_text,
        reference_caption=str(payload.get("reference_caption") or "").strip(),
        source=str(payload.get("source") or "").strip(),
        difference_type=str(payload.get("difference_type") or "").strip(),
    )
