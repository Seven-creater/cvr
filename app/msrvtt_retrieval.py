from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def tokenize(text: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [token for token in normalized.split() if token]


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    captions: list[str]
    merged_text: str


@dataclass(frozen=True, slots=True)
class TextRecord:
    text_id: str
    video_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    item_id: str
    score: float
    video_id: str | None = None
    text: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "rank": self.rank,
            "item_id": self.item_id,
            "score": round(self.score, 6),
        }
        if self.video_id is not None:
            payload["video_id"] = self.video_id
        if self.text is not None:
            payload["text"] = self.text
        return payload


class TfIdfIndex:
    def __init__(self, doc_ids: list[str], documents: list[str]) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        self._doc_id_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        doc_tokens = [tokenize(doc) for doc in self.documents]
        document_frequency: Counter[str] = Counter()
        doc_weights: list[dict[str, float]] = []
        total_docs = len(doc_tokens)

        for tokens in doc_tokens:
            counts = Counter(tokens)
            doc_weights.append({token: float(count) for token, count in counts.items()})
            document_frequency.update(counts.keys())

        self._idf = {
            token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }

        for doc_index, weights in enumerate(doc_weights):
            norm = math.sqrt(sum((count * self._idf[token]) ** 2 for token, count in weights.items()))
            if norm == 0.0:
                continue
            for token, count in weights.items():
                normalized_weight = (count * self._idf[token]) / norm
                self._postings[token].append((doc_index, normalized_weight))

    def score(self, query_text: str) -> list[float]:
        counts = Counter(tokenize(query_text))
        if not counts:
            return [0.0] * len(self.doc_ids)

        weighted = {token: count * self._idf.get(token, 0.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0.0:
            return [0.0] * len(self.doc_ids)

        scores = [0.0] * len(self.doc_ids)
        for token, value in weighted.items():
            query_weight = value / norm
            for doc_index, doc_weight in self._postings.get(token, ()):
                scores[doc_index] += query_weight * doc_weight
        return scores

    def rank(self, query_text: str, topk: int | None = None) -> list[tuple[str, float]]:
        scores = self.score(query_text)
        ordered = sorted(
            enumerate(scores),
            key=lambda item: (-item[1], self.doc_ids[item[0]]),
        )
        if topk is not None:
            ordered = ordered[:topk]
        return [(self.doc_ids[index], score) for index, score in ordered]

    def index_of(self, doc_id: str) -> int:
        return self._doc_id_to_index[doc_id]


@dataclass(slots=True)
class MsrvttDataset:
    videos: list[VideoRecord]
    texts: list[TextRecord]
    video_lookup: dict[str, VideoRecord]
    text_lookup: dict[str, TextRecord]
    video_to_text_ids: dict[str, list[str]]
    video_index: TfIdfIndex
    text_index: TfIdfIndex


def load_msrvtt_dataset(
    msrvtt_json_path: str | Path,
    split_csv_path: str | Path | None = None,
    max_videos: int | None = None,
    max_texts_per_video: int | None = None,
) -> MsrvttDataset:
    raw = load_json(msrvtt_json_path)
    split_ids = load_split_ids(split_csv_path)
    captions_by_video: dict[str, list[str]] = defaultdict(list)

    for item in raw.get("sentences", []):
        video_id = item.get("video_id")
        caption = (item.get("caption") or item.get("text") or "").strip()
        if not video_id or not caption:
            continue
        if split_ids is not None and video_id not in split_ids:
            continue
        if max_texts_per_video is not None and len(captions_by_video[video_id]) >= max_texts_per_video:
            continue
        captions_by_video[video_id].append(caption)

    video_ids = sorted(captions_by_video)
    if max_videos is not None:
        video_ids = video_ids[:max_videos]

    videos: list[VideoRecord] = []
    texts: list[TextRecord] = []
    text_lookup: dict[str, TextRecord] = {}
    video_to_text_ids: dict[str, list[str]] = {}

    for video_id in video_ids:
        captions = captions_by_video[video_id]
        merged_text = " ".join(captions)
        videos.append(VideoRecord(video_id=video_id, captions=captions, merged_text=merged_text))

        text_ids: list[str] = []
        for caption_index, caption in enumerate(captions):
            text_id = f"{video_id}::caption::{caption_index}"
            row = TextRecord(text_id=text_id, video_id=video_id, text=caption)
            texts.append(row)
            text_lookup[text_id] = row
            text_ids.append(text_id)
        video_to_text_ids[video_id] = text_ids

    video_lookup = {video.video_id: video for video in videos}
    video_index = TfIdfIndex(
        doc_ids=[video.video_id for video in videos],
        documents=[video.merged_text for video in videos],
    )
    text_index = TfIdfIndex(
        doc_ids=[text.text_id for text in texts],
        documents=[text.text for text in texts],
    )
    return MsrvttDataset(
        videos=videos,
        texts=texts,
        video_lookup=video_lookup,
        text_lookup=text_lookup,
        video_to_text_ids=video_to_text_ids,
        video_index=video_index,
        text_index=text_index,
    )


def retrieve_videos_from_text(text: str, dataset: MsrvttDataset, topk: int = 10) -> list[RetrievalHit]:
    ranking = dataset.video_index.rank(text, topk=topk)
    hits: list[RetrievalHit] = []
    for rank, (video_id, score) in enumerate(ranking, start=1):
        hits.append(
            RetrievalHit(
                rank=rank,
                item_id=video_id,
                score=score,
                video_id=video_id,
                text=dataset.video_lookup[video_id].captions[0] if dataset.video_lookup[video_id].captions else "",
            )
        )
    return hits


def retrieve_texts_from_video(video_id: str, dataset: MsrvttDataset, topk: int = 10) -> list[RetrievalHit]:
    if video_id not in dataset.video_lookup:
        raise KeyError(f"unknown video_id: {video_id}")
    query_text = dataset.video_lookup[video_id].merged_text
    ranking = dataset.text_index.rank(query_text, topk=topk)
    hits: list[RetrievalHit] = []
    for rank, (text_id, score) in enumerate(ranking, start=1):
        text_row = dataset.text_lookup[text_id]
        hits.append(
            RetrievalHit(
                rank=rank,
                item_id=text_id,
                score=score,
                video_id=text_row.video_id,
                text=text_row.text,
            )
        )
    return hits


def _hit_recall(top_ids: Iterable[str], target_ids: set[str], k: int) -> float:
    head = list(top_ids)[:k]
    return 1.0 if any(item in target_ids for item in head) else 0.0


def parse_topk_values(raw: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not ks or ks[0] <= 0:
        raise ValueError("topk values must be positive integers")
    return ks


def evaluate_text_to_video(dataset: MsrvttDataset, ks: list[int]) -> dict:
    max_k = max(ks)
    scores = {f"R@{k}": 0.0 for k in ks}
    total = len(dataset.texts)

    for text_row in dataset.texts:
        ranking = retrieve_videos_from_text(text_row.text, dataset, topk=max_k)
        top_ids = [hit.video_id for hit in ranking if hit.video_id is not None]
        targets = {text_row.video_id}
        for k in ks:
            scores[f"R@{k}"] += _hit_recall(top_ids, targets, k)

    return {key: round(value / max(1, total), 4) for key, value in scores.items()}


def evaluate_video_to_text(dataset: MsrvttDataset, ks: list[int]) -> dict:
    max_k = max(ks)
    scores = {f"R@{k}": 0.0 for k in ks}
    total = len(dataset.videos)

    for video in dataset.videos:
        ranking = retrieve_texts_from_video(video.video_id, dataset, topk=max_k)
        top_ids = [hit.item_id for hit in ranking]
        targets = set(dataset.video_to_text_ids[video.video_id])
        for k in ks:
            scores[f"R@{k}"] += _hit_recall(top_ids, targets, k)

    return {key: round(value / max(1, total), 4) for key, value in scores.items()}


def evaluate_bidirectional(dataset: MsrvttDataset, ks: list[int]) -> dict:
    return {
        "dataset": "MSRVTT",
        "video_count": len(dataset.videos),
        "text_count": len(dataset.texts),
        "t2v": evaluate_text_to_video(dataset, ks),
        "v2t": evaluate_video_to_text(dataset, ks),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MSRVTT retrieval baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("text2video", "video2text", "evaluate"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--msrvtt-json", required=True, help="Path to MSRVTT_data.json")
        sub.add_argument("--split-csv", help="Optional split csv such as MSRVTT_JSFUSION_test.csv")
        sub.add_argument("--max-videos", type=int, help="Optional cap for quick tests")
        sub.add_argument("--max-texts-per-video", type=int, help="Optional caption cap per video")

    text2video = subparsers.choices["text2video"]
    text2video.add_argument("--text", required=True)
    text2video.add_argument("--topk", type=int, default=10)

    video2text = subparsers.choices["video2text"]
    video2text.add_argument("--video-id", required=True)
    video2text.add_argument("--topk", type=int, default=10)

    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--topk", default="1,5,10", help="Comma-separated recall cutoffs")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset = load_msrvtt_dataset(
        msrvtt_json_path=args.msrvtt_json,
        split_csv_path=args.split_csv,
        max_videos=args.max_videos,
        max_texts_per_video=args.max_texts_per_video,
    )

    if args.command == "text2video":
        payload = [hit.to_dict() for hit in retrieve_videos_from_text(args.text, dataset, topk=args.topk)]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "video2text":
        payload = [hit.to_dict() for hit in retrieve_texts_from_video(args.video_id, dataset, topk=args.topk)]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    metrics = evaluate_bidirectional(dataset, ks=parse_topk_values(args.topk))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
