from __future__ import annotations

from dataclasses import dataclass


def parse_topk_values(raw: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not ks or ks[0] <= 0:
        raise ValueError("topk values must be positive integers")
    return ks


@dataclass(frozen=True, slots=True)
class TextRow:
    text_id: str
    video_id: str
    text: str

    @classmethod
    def from_dict(cls, row: dict) -> "TextRow":
        return cls(text_id=str(row["text_id"]), video_id=str(row["video_id"]), text=str(row["text"]))


@dataclass(frozen=True, slots=True)
class VideoRow:
    video_id: str
    video_path: str
    audio_path: str | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "VideoRow":
        return cls(
            video_id=str(row["video_id"]),
            video_path=str(row.get("video_path", "")),
            audio_path=str(row["audio_path"]) if row.get("audio_path") else None,
        )


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    item_id: str
    score: float
    video_id: str | None = None
    text_id: str | None = None
    text: str | None = None
    video_path: str | None = None

    def to_dict(self) -> dict:
        payload = {"rank": self.rank, "item_id": self.item_id, "score": round(self.score, 6)}
        if self.video_id is not None:
            payload["video_id"] = self.video_id
        if self.text_id is not None:
            payload["text_id"] = self.text_id
        if self.text is not None:
            payload["text"] = self.text
        if self.video_path is not None:
            payload["video_path"] = self.video_path
        return payload
