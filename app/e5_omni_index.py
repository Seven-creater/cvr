from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.e5_omni_runtime import E5OmniRuntime, E5OmniRuntimeConfig, runtime_config_fingerprint
from app.retrieval_types import RetrievalHit, VideoRow


INDEX_VERSION = 1


@dataclass(frozen=True)
class E5TargetRecord:
    video_id: str
    video_path: str
    audio_path: str | None = None


@dataclass(frozen=True)
class E5TargetIndex:
    records: list[E5TargetRecord]
    embeddings: np.ndarray
    metadata: dict[str, Any]

    @property
    def video_ids(self) -> list[str]:
        return [record.video_id for record in self.records]

    def to_summary(self) -> dict[str, Any]:
        return {
            "video_count": len(self.records),
            "embedding_shape": list(self.embeddings.shape),
            **self.metadata,
        }


def records_from_video_rows(video_rows: list[VideoRow]) -> list[E5TargetRecord]:
    return [
        E5TargetRecord(video_id=row.video_id, video_path=row.video_path, audio_path=row.audio_path)
        for row in video_rows
    ]


def build_or_load_e5_target_index(
    *,
    runtime: E5OmniRuntime,
    records: list[E5TargetRecord],
    index_dir: str | Path,
    force_rebuild: bool = False,
) -> E5TargetIndex:
    if not records:
        raise ValueError("records must not be empty")
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = build_e5_index_key(config=runtime.config, records=records)
    index_json = root / "target_index.json"
    embeddings_path = root / "target_embeddings.npy"
    summary_path = root / "e5_index_summary.json"

    if not force_rebuild:
        loaded = _try_load_index(index_json=index_json, embeddings_path=embeddings_path, expected_key=key)
        if loaded is not None:
            return loaded

    embeddings = runtime.encode_video_documents([record.video_path for record in records])
    if embeddings.shape[0] != len(records):
        raise ValueError(f"encoded {embeddings.shape[0]} videos for {len(records)} records")
    metadata = {
        "version": INDEX_VERSION,
        "cache_key": key,
        "model_path": runtime.config.model_path,
        "runtime_config": asdict(runtime.config),
    }
    index = E5TargetIndex(records=list(records), embeddings=embeddings.astype(np.float32), metadata=metadata)
    np.save(str(embeddings_path), index.embeddings)
    index_json.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "records": [asdict(record) for record in records],
                "embeddings_path": str(embeddings_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(index.to_summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def retrieve_e5_videos(
    *,
    query_embedding: np.ndarray,
    index: E5TargetIndex,
    topk: int,
) -> list[RetrievalHit]:
    if topk <= 0:
        raise ValueError("topk must be positive")
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    matrix = np.asarray(index.embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("index embeddings must be 2D")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(f"query dim {query.shape[0]} does not match index dim {matrix.shape[1]}")
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")[: max(1, int(topk))]
    hits: list[RetrievalHit] = []
    for rank, row_index in enumerate(order, start=1):
        record = index.records[int(row_index)]
        hits.append(
            RetrievalHit(
                rank=rank,
                item_id=record.video_id,
                score=float(scores[row_index]),
                video_id=record.video_id,
                video_path=record.video_path,
            )
        )
    return hits


def build_e5_index_key(*, config: E5OmniRuntimeConfig, records: list[E5TargetRecord]) -> str:
    payload = {
        "version": INDEX_VERSION,
        "runtime": runtime_config_fingerprint(config),
        "records": [_record_fingerprint(record) for record in records],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _try_load_index(
    *,
    index_json: Path,
    embeddings_path: Path,
    expected_key: str,
) -> E5TargetIndex | None:
    if not index_json.exists() or not embeddings_path.exists():
        return None
    try:
        payload = json.loads(index_json.read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        if metadata.get("cache_key") != expected_key:
            return None
        records = [E5TargetRecord(**row) for row in payload["records"]]
        embeddings = np.load(str(embeddings_path)).astype(np.float32)
    except Exception:
        return None
    if embeddings.shape[0] != len(records):
        return None
    return E5TargetIndex(records=records, embeddings=embeddings, metadata=metadata)


def _record_fingerprint(record: E5TargetRecord) -> dict[str, Any]:
    path = Path(record.video_path)
    stat = path.stat()
    return {
        "video_id": record.video_id,
        "video_path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }
