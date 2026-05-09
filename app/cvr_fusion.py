from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval_types import RetrievalHit


@dataclass(frozen=True)
class FusedHit:
    rank: int
    item_id: str
    video_id: str
    fused_score: float
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank": self.rank,
            "item_id": self.item_id,
            "video_id": self.video_id,
            "fused_score": round(float(self.fused_score), 6),
            "source_ranks": dict(self.source_ranks),
            "source_scores": {key: round(float(value), 6) for key, value in self.source_scores.items()},
        }
        if self.video_path is not None:
            payload["video_path"] = self.video_path
        return payload

    def to_retrieval_hit(self) -> RetrievalHit:
        return RetrievalHit(
            rank=self.rank,
            item_id=self.item_id,
            score=float(self.fused_score),
            video_id=self.video_id,
            video_path=self.video_path,
        )


def fuse_video_hits(
    *,
    avigate_hits: list[RetrievalHit],
    e5_hits: list[RetrievalHit],
    topk: int,
    rrf_k: int = 60,
) -> list[FusedHit]:
    if topk <= 0:
        raise ValueError("topk must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    accumulator: dict[str, dict[str, Any]] = {}
    _accumulate_source(accumulator, "avigate", avigate_hits, rrf_k=rrf_k)
    _accumulate_source(accumulator, "e5", e5_hits, rrf_k=rrf_k)

    rows = sorted(
        accumulator.values(),
        key=lambda item: (
            -float(item["fused_score"]),
            min(item["source_ranks"].values()) if item["source_ranks"] else 10**9,
            str(item["video_id"]),
        ),
    )
    fused: list[FusedHit] = []
    for rank, item in enumerate(rows[:topk], start=1):
        fused.append(
            FusedHit(
                rank=rank,
                item_id=str(item["video_id"]),
                video_id=str(item["video_id"]),
                fused_score=float(item["fused_score"]),
                source_ranks=dict(item["source_ranks"]),
                source_scores=dict(item["source_scores"]),
                video_path=item.get("video_path"),
            )
        )
    return fused


def fused_hits_to_retrieval_hits(hits: list[FusedHit]) -> list[RetrievalHit]:
    return [hit.to_retrieval_hit() for hit in hits]


def _accumulate_source(
    accumulator: dict[str, dict[str, Any]],
    source_name: str,
    hits: list[RetrievalHit],
    *,
    rrf_k: int,
) -> None:
    for hit in hits:
        video_id = str(hit.video_id or hit.item_id).strip()
        if not video_id:
            continue
        row = accumulator.setdefault(
            video_id,
            {
                "video_id": video_id,
                "fused_score": 0.0,
                "source_ranks": {},
                "source_scores": {},
                "video_path": hit.video_path,
            },
        )
        row["fused_score"] += 1.0 / (rrf_k + int(hit.rank))
        row["source_ranks"][source_name] = int(hit.rank)
        row["source_scores"][source_name] = float(hit.score)
        if not row.get("video_path") and hit.video_path:
            row["video_path"] = hit.video_path
