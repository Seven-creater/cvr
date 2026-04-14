from __future__ import annotations

import csv
import hashlib
import json
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return set()

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = "video_id" in header or "id" in header
    ids: set[str] = set()
    if has_header:
        key_index = header.index("video_id") if "video_id" in header else header.index("id")
        for row in rows[1:]:
            if row and len(row) > key_index and row[key_index].strip():
                ids.add(row[key_index].strip())
        return ids

    for row in rows:
        if row and row[0].strip():
            ids.add(row[0].strip())
    return ids


def parse_topk_values(raw: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not ks or ks[0] <= 0:
        raise ValueError("topk values must be positive integers")
    return ks


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def normalize_weights(alpha_visual: float, alpha_audio: float, audio_available: bool) -> tuple[float, float]:
    if not audio_available:
        return 1.0, 0.0
    alpha_visual = max(0.0, float(alpha_visual))
    alpha_audio = max(0.0, float(alpha_audio))
    total = alpha_visual + alpha_audio
    if total == 0.0:
        return 1.0, 0.0
    return alpha_visual / total, alpha_audio / total


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


class TextEncoder(Protocol):
    def encode(self, text: str) -> np.ndarray:
        ...


class HashingTextEncoder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        return normalize_vector(vector)


class SubprocessTextEncoder:
    def __init__(self, command: str) -> None:
        self.command = shlex.split(command)

    def encode(self, text: str) -> np.ndarray:
        result = subprocess.run(
            self.command,
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        if isinstance(payload, dict):
            payload = payload["embedding"]
        return normalize_vector(np.asarray(payload, dtype=np.float32))


@dataclass
class FeatureRetriever:
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    text_embeddings: np.ndarray
    video_visual_embeddings: np.ndarray
    video_audio_embeddings: np.ndarray | None
    text_encoder: TextEncoder | None = None

    def __post_init__(self) -> None:
        if len(self.text_rows) != self.text_embeddings.shape[0]:
            raise ValueError("text_rows and text_embeddings length mismatch")
        if len(self.video_rows) != self.video_visual_embeddings.shape[0]:
            raise ValueError("video_rows and video_visual_embeddings length mismatch")
        if self.video_audio_embeddings is not None and len(self.video_rows) != self.video_audio_embeddings.shape[0]:
            raise ValueError("video_rows and video_audio_embeddings length mismatch")

        self.text_embeddings = l2_normalize(self.text_embeddings)
        self.video_visual_embeddings = l2_normalize(self.video_visual_embeddings)
        if self.video_audio_embeddings is not None:
            self.video_audio_embeddings = l2_normalize(self.video_audio_embeddings)

        self._text_index = {row.text_id: index for index, row in enumerate(self.text_rows)}
        self._video_index = {row.video_id: index for index, row in enumerate(self.video_rows)}
        self._text_lookup = {row.text_id: row for row in self.text_rows}
        self._video_lookup = {row.video_id: row for row in self.video_rows}
        self._video_to_text_ids: dict[str, list[str]] = defaultdict(list)
        for row in self.text_rows:
            self._video_to_text_ids[row.video_id].append(row.text_id)

    @property
    def audio_available(self) -> bool:
        return self.video_audio_embeddings is not None

    @property
    def embedding_dim(self) -> int:
        return int(self.text_embeddings.shape[1])

    @classmethod
    def from_feature_dir(
        cls,
        feature_dir: str | Path,
        split_csv_path: str | Path | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> "FeatureRetriever":
        feature_dir = Path(feature_dir)
        text_rows = [TextRow.from_dict(row) for row in load_jsonl(feature_dir / "text_rows.jsonl")]
        video_rows = [VideoRow.from_dict(row) for row in load_jsonl(feature_dir / "video_rows.jsonl")]
        text_embeddings = np.load(feature_dir / "text_embeddings.npy")
        video_visual_embeddings = np.load(feature_dir / "video_visual_embeddings.npy")
        audio_path = feature_dir / "video_audio_embeddings.npy"
        video_audio_embeddings = np.load(audio_path) if audio_path.exists() else None

        split_ids = load_split_ids(split_csv_path)
        if split_ids is not None:
            text_keep = [index for index, row in enumerate(text_rows) if row.video_id in split_ids]
            video_keep = [index for index, row in enumerate(video_rows) if row.video_id in split_ids]
            text_rows = [text_rows[index] for index in text_keep]
            video_rows = [video_rows[index] for index in video_keep]
            text_embeddings = text_embeddings[text_keep]
            video_visual_embeddings = video_visual_embeddings[video_keep]
            if video_audio_embeddings is not None:
                video_audio_embeddings = video_audio_embeddings[video_keep]

        return cls(
            text_rows=text_rows,
            video_rows=video_rows,
            text_embeddings=text_embeddings,
            video_visual_embeddings=video_visual_embeddings,
            video_audio_embeddings=video_audio_embeddings,
            text_encoder=text_encoder,
        )

    def build_query_embedding(
        self,
        *,
        query_text: str | None = None,
        text_id: str | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        if query_embedding is not None:
            return normalize_vector(query_embedding)
        if text_id is not None:
            return self.text_embeddings[self._text_index[text_id]]
        if query_text is None:
            raise ValueError("query_text, text_id, or query_embedding must be provided")
        if self.text_encoder is None:
            raise ValueError("text_encoder is required for raw query text retrieval")
        vector = self.text_encoder.encode(query_text)
        if vector.shape[-1] != self.embedding_dim:
            raise ValueError("encoded query embedding dimension mismatch")
        return normalize_vector(vector)

    def retrieve_t2v(
        self,
        *,
        query_text: str | None = None,
        text_id: str | None = None,
        query_embedding: np.ndarray | None = None,
        alpha_visual: float = 0.8,
        alpha_audio: float = 0.2,
        topk: int = 10,
    ) -> list[RetrievalHit]:
        topk = max(1, int(topk))
        alpha_visual, alpha_audio = normalize_weights(alpha_visual, alpha_audio, self.audio_available)
        query_vector = self.build_query_embedding(query_text=query_text, text_id=text_id, query_embedding=query_embedding)
        visual_scores = self.video_visual_embeddings @ query_vector
        if self.video_audio_embeddings is not None:
            audio_scores = self.video_audio_embeddings @ query_vector
            scores = (alpha_visual * visual_scores) + (alpha_audio * audio_scores)
        else:
            scores = visual_scores
        order = np.argsort(-scores, kind="stable")[:topk]
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(order, start=1):
            row = self.video_rows[int(index)]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    item_id=row.video_id,
                    score=float(scores[index]),
                    video_id=row.video_id,
                    video_path=row.video_path,
                )
            )
        return hits

    def retrieve_v2t(
        self,
        video_id: str,
        *,
        alpha_visual: float = 0.8,
        alpha_audio: float = 0.2,
        topk: int = 10,
    ) -> list[RetrievalHit]:
        topk = max(1, int(topk))
        video_index = self._video_index[video_id]
        alpha_visual, alpha_audio = normalize_weights(alpha_visual, alpha_audio, self.audio_available)
        visual_query = self.video_visual_embeddings[video_index]
        visual_scores = self.text_embeddings @ visual_query
        if self.video_audio_embeddings is not None:
            audio_query = self.video_audio_embeddings[video_index]
            audio_scores = self.text_embeddings @ audio_query
            scores = (alpha_visual * visual_scores) + (alpha_audio * audio_scores)
        else:
            scores = visual_scores
        order = np.argsort(-scores, kind="stable")[:topk]
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(order, start=1):
            row = self.text_rows[int(index)]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    item_id=row.text_id,
                    score=float(scores[index]),
                    video_id=row.video_id,
                    text_id=row.text_id,
                    text=row.text,
                )
            )
        return hits

    def evaluate_t2v(self, ks: list[int], alpha_visual: float = 0.8, alpha_audio: float = 0.2) -> dict[str, float]:
        scores = {f"R@{k}": 0.0 for k in ks}
        max_k = max(ks)
        for row in self.text_rows:
            hits = self.retrieve_t2v(text_id=row.text_id, alpha_visual=alpha_visual, alpha_audio=alpha_audio, topk=max_k)
            top_ids = [hit.video_id for hit in hits if hit.video_id is not None]
            for k in ks:
                scores[f"R@{k}"] += 1.0 if row.video_id in top_ids[:k] else 0.0
        total = len(self.text_rows)
        return {key: round(value / max(1, total), 4) for key, value in scores.items()}

    def evaluate_v2t(self, ks: list[int], alpha_visual: float = 0.8, alpha_audio: float = 0.2) -> dict[str, float]:
        scores = {f"R@{k}": 0.0 for k in ks}
        max_k = max(ks)
        for row in self.video_rows:
            hits = self.retrieve_v2t(row.video_id, alpha_visual=alpha_visual, alpha_audio=alpha_audio, topk=max_k)
            target_ids = set(self._video_to_text_ids[row.video_id])
            top_ids = [hit.item_id for hit in hits]
            for k in ks:
                scores[f"R@{k}"] += 1.0 if any(item in target_ids for item in top_ids[:k]) else 0.0
        total = len(self.video_rows)
        return {key: round(value / max(1, total), 4) for key, value in scores.items()}

    def evaluate_bidirectional(self, ks: list[int], alpha_visual: float = 0.8, alpha_audio: float = 0.2) -> dict:
        return {
            "dataset": "MSRVTT",
            "video_count": len(self.video_rows),
            "text_count": len(self.text_rows),
            "audio_available": self.audio_available,
            "t2v": self.evaluate_t2v(ks, alpha_visual=alpha_visual, alpha_audio=alpha_audio),
            "v2t": self.evaluate_v2t(ks, alpha_visual=alpha_visual, alpha_audio=alpha_audio),
        }

    def get_video_row(self, video_id: str) -> VideoRow:
        return self._video_lookup[video_id]

    def get_text_row(self, text_id: str) -> TextRow:
        return self._text_lookup[text_id]

    def target_text_ids(self, video_id: str) -> list[str]:
        return list(self._video_to_text_ids[video_id])


def build_text_encoder(kind: str, *, dim: int | None = None, command: str | None = None) -> TextEncoder | None:
    if kind == "none":
        return None
    if kind == "hashing":
        if dim is None:
            raise ValueError("hashing encoder requires dim")
        return HashingTextEncoder(dim=dim)
    if kind == "subprocess":
        if not command:
            raise ValueError("subprocess encoder requires command")
        return SubprocessTextEncoder(command)
    raise ValueError(f"unsupported text encoder kind: {kind}")


def load_feature_retriever(
    *,
    feature_dir: str | Path,
    split_csv_path: str | Path | None = None,
    text_encoder: TextEncoder | None = None,
) -> FeatureRetriever:
    return FeatureRetriever.from_feature_dir(
        feature_dir=feature_dir,
        split_csv_path=split_csv_path,
        text_encoder=text_encoder,
    )


def retrieve_videos_from_text(
    *,
    feature_dir: str | Path,
    query_text: str,
    split_csv_path: str | Path | None = None,
    alpha_visual: float = 0.8,
    alpha_audio: float = 0.2,
    topk: int = 10,
    text_encoder: TextEncoder | None = None,
) -> list[RetrievalHit]:
    retriever = load_feature_retriever(
        feature_dir=feature_dir,
        split_csv_path=split_csv_path,
        text_encoder=text_encoder,
    )
    return retriever.retrieve_t2v(
        query_text=query_text,
        alpha_visual=alpha_visual,
        alpha_audio=alpha_audio,
        topk=topk,
    )


def retrieve_texts_from_video(
    *,
    feature_dir: str | Path,
    video_id: str,
    split_csv_path: str | Path | None = None,
    alpha_visual: float = 0.8,
    alpha_audio: float = 0.2,
    topk: int = 10,
) -> list[RetrievalHit]:
    retriever = load_feature_retriever(feature_dir=feature_dir, split_csv_path=split_csv_path)
    return retriever.retrieve_v2t(
        video_id,
        alpha_visual=alpha_visual,
        alpha_audio=alpha_audio,
        topk=topk,
    )


def evaluate_retriever(
    *,
    feature_dir: str | Path,
    split_csv_path: str | Path | None = None,
    ks: list[int] | None = None,
    alpha_visual: float = 0.8,
    alpha_audio: float = 0.2,
) -> dict:
    retriever = load_feature_retriever(feature_dir=feature_dir, split_csv_path=split_csv_path)
    return retriever.evaluate_bidirectional(ks or [1, 5, 10], alpha_visual=alpha_visual, alpha_audio=alpha_audio)
