from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import glob
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


DEFAULT_RUNS_ROOT = "/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs"
DEFAULT_E5_MODEL = "/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B"
DEFAULT_EXPECTED_COUNT = 943
DEFAULT_SMOKE_SIZE = 20
DEFAULT_TOPK = "1,5,10"
DEFAULT_VIDEO_MAX_PIXELS = 64 * 28 * 28
QUERY_TEMPLATE = "Edit the reference video so that: {edit_text}"


@dataclass(frozen=True)
class E5CVRTriplet:
    sample_id: str
    reference_video: str
    target_video: str
    edit_text: str
    reference_caption: str = ""
    source: str = ""
    difference_type: str = ""


@dataclass(frozen=True)
class E5RuntimeInfo:
    model_path: str
    device: str
    torch_dtype: str
    requested_attention: str
    used_attention: str
    batch_size: int
    video_max_pixels: int
    video_fps: int


@dataclass(frozen=True)
class TargetRecord:
    sample_id: str
    target_video: str


@dataclass(frozen=True)
class TargetIndex:
    records: list[TargetRecord]
    embeddings: np.ndarray
    metadata: dict[str, Any]


class E5SentenceTransformerEncoder:
    def __init__(self, model: Any, *, batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size

    def encode_document(self, inputs: list[Any]) -> np.ndarray:
        return _encode_with_sentence_transformers(self.model, inputs, batch_size=self.batch_size)


def load_triplets_jsonl(path: str | Path, *, expected_count: int | None = None) -> list[E5CVRTriplet]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"triplets.jsonl not found: {root}")
    triplets: list[E5CVRTriplet] = []
    for line_number, line in enumerate(root.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        triplets.append(_triplet_from_payload(payload, line_number=line_number))
    if expected_count is not None and len(triplets) != expected_count:
        raise ValueError(f"expected {expected_count} triplets, got {len(triplets)}")
    return triplets


def find_latest_triplets(runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    pattern = str(Path(runs_root) / "composed_triplets_full_*" / "triplets.jsonl")
    candidates = [Path(path) for path in glob.glob(pattern)]
    if not candidates:
        raise FileNotFoundError(f"no composed triplets found under {runs_root}; run scripts/build_composed_triplets.sh first")
    return max(candidates, key=lambda path: path.parent.stat().st_mtime_ns)


def load_e5_encoder(
    *,
    model_path: str,
    device: str,
    torch_dtype: str,
    attn_implementation: str,
    batch_size: int,
    video_max_pixels: int,
    video_fps: int,
) -> tuple[E5SentenceTransformerEncoder, E5RuntimeInfo]:
    model_root = Path(model_path)
    if not model_root.exists():
        raise FileNotFoundError(f"e5 model path not found: {model_root}")
    if not (model_root / "config.json").exists():
        raise FileNotFoundError(f"e5 model config.json not found: {model_root / 'config.json'}")

    torch = _import_torch()
    SentenceTransformer = _import_sentence_transformer()
    model_kwargs = _build_model_kwargs(torch, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    try:
        model = SentenceTransformer(str(model_root), device=device, trust_remote_code=True, model_kwargs=model_kwargs)
        used_attention = attn_implementation or "default"
    except Exception as first_error:
        if not attn_implementation:
            raise
        fallback_kwargs = _build_model_kwargs(torch, torch_dtype=torch_dtype, attn_implementation="")
        try:
            model = SentenceTransformer(str(model_root), device=device, trust_remote_code=True, model_kwargs=fallback_kwargs)
        except Exception as second_error:
            raise RuntimeError(f"failed to load e5 with flash attention and fallback: {second_error}") from first_error
        used_attention = "default"

    _configure_video_processing(model, max_pixels=video_max_pixels, fps=video_fps)
    info = E5RuntimeInfo(
        model_path=str(model_root),
        device=device,
        torch_dtype=torch_dtype,
        requested_attention=attn_implementation or "default",
        used_attention=used_attention,
        batch_size=batch_size,
        video_max_pixels=video_max_pixels,
        video_fps=video_fps,
    )
    return E5SentenceTransformerEncoder(model, batch_size=batch_size), info


def build_or_load_target_index(
    *,
    triplets: list[E5CVRTriplet],
    encoder: Any,
    index_dir: str | Path,
    runtime_info: E5RuntimeInfo | dict[str, Any],
    force_rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TargetIndex:
    if not triplets:
        raise ValueError("triplets must not be empty")
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = [TargetRecord(sample_id=item.sample_id, target_video=item.target_video) for item in triplets]
    runtime_payload = asdict(runtime_info) if isinstance(runtime_info, E5RuntimeInfo) else dict(runtime_info)
    cache_key = _target_index_cache_key(records=records, runtime_info=runtime_payload)
    embeddings_path = root / "target_embeddings.npy"
    index_path = root / "target_index.json"

    if not force_rebuild:
        loaded = _try_load_target_index(index_path=index_path, embeddings_path=embeddings_path, cache_key=cache_key)
        if loaded is not None:
            _emit(progress, f"[e5-cvr] loaded target index: {embeddings_path}")
            return loaded

    _emit(progress, f"[e5-cvr] encoding {len(records)} target videos")
    embeddings = _normalize_rows(encoder.encode_document([record.target_video for record in records]))
    if embeddings.shape[0] != len(records):
        raise ValueError(f"target embedding row count mismatch: {embeddings.shape[0]} vs {len(records)}")
    metadata = {
        "cache_key": cache_key,
        "gallery_count": len(records),
        "embedding_shape": list(embeddings.shape),
        "runtime": runtime_payload,
    }
    index = TargetIndex(records=records, embeddings=embeddings.astype(np.float32), metadata=metadata)
    np.save(str(embeddings_path), index.embeddings)
    index_path.write_text(
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
    return index


def run_eval_slice(
    *,
    triplets: list[E5CVRTriplet],
    target_index: TargetIndex,
    encoder: Any,
    output_dir: str | Path,
    sample_size: int,
    recall_ks: tuple[int, ...],
    topk_trace: int,
    runtime_info: E5RuntimeInfo | dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(triplets):
        raise ValueError(f"sample_size {sample_size} exceeds triplet count {len(triplets)}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    recall_ks = _normalize_ks(recall_ks)
    max_trace = max(max(recall_ks), topk_trace)
    selected = triplets[:sample_size]
    target_position = {record.sample_id: index for index, record in enumerate(target_index.records)}
    missing = [item.sample_id for item in selected if item.sample_id not in target_position]
    if missing:
        raise ValueError(f"{len(missing)} query targets are missing from target gallery, e.g. {missing[:3]}")

    hit_counts = {k: 0 for k in recall_ks}
    trace_lines: list[str] = []
    for query_index, triplet in enumerate(selected, start=1):
        _emit(progress, f"[e5-cvr] query {query_index}/{len(selected)} sample_id={triplet.sample_id}")
        query_embedding = _normalize_rows(encoder.encode_document([_query_payload(triplet)]))[0]
        scores = target_index.embeddings @ query_embedding
        order = np.argsort(-scores, kind="stable")
        target_index_value = target_position[triplet.sample_id]
        target_rank = int(np.where(order == target_index_value)[0][0]) + 1
        for k in recall_ks:
            if target_rank <= k:
                hit_counts[k] += 1
        trace_lines.append(
            json.dumps(
                {
                    "sample_id": triplet.sample_id,
                    "reference_video": triplet.reference_video,
                    "target_video": triplet.target_video,
                    "edit_text": triplet.edit_text,
                    "target_rank": target_rank,
                    "target_score": round(float(scores[target_index_value]), 6),
                    "query_index": query_index,
                    "topk_hits": _topk_hits(order=order, scores=scores, target_index=target_index, topk=max_trace),
                },
                ensure_ascii=False,
            )
        )

    runtime_payload = asdict(runtime_info) if isinstance(runtime_info, E5RuntimeInfo) else dict(runtime_info)
    summary = {
        "mode": "e5-cvr-only",
        "query_count": len(selected),
        "gallery_count": len(target_index.records),
        "recall": {f"R@{k}": round(hit_counts[k] / max(1, len(selected)), 4) for k in recall_ks},
        "topk_trace": topk_trace,
        "query_template": QUERY_TEMPLATE,
        "runtime": runtime_payload,
        "target_index": target_index.metadata,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "traces.jsonl").write_text("\n".join(trace_lines) + ("\n" if trace_lines else ""), encoding="utf-8")
    return summary


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root or _default_run_root())
    run_root.mkdir(parents=True, exist_ok=True)
    triplets_path = Path(args.triplets_jsonl) if args.triplets_jsonl else find_latest_triplets(args.runs_root)
    triplets = load_triplets_jsonl(triplets_path, expected_count=args.expected_count)
    encoder, runtime_info = load_e5_encoder(
        model_path=args.e5_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        batch_size=args.batch_size,
        video_max_pixels=args.video_max_pixels,
        video_fps=args.video_fps,
    )
    index = build_or_load_target_index(
        triplets=triplets,
        encoder=encoder,
        index_dir=run_root / "target_index",
        runtime_info=runtime_info,
        force_rebuild=args.force_rebuild_index,
        progress=lambda message: print(message, flush=True),
    )
    recall_ks = tuple(_normalize_ks(parse_topk(args.topk)))
    smoke_summary = run_eval_slice(
        triplets=triplets,
        target_index=index,
        encoder=encoder,
        output_dir=run_root / "smoke20",
        sample_size=min(args.smoke_size, len(triplets)),
        recall_ks=recall_ks,
        topk_trace=args.topk_trace,
        runtime_info=runtime_info,
        progress=lambda message: print(message, flush=True),
    )
    full_summary = run_eval_slice(
        triplets=triplets,
        target_index=index,
        encoder=encoder,
        output_dir=run_root / f"full{len(triplets)}",
        sample_size=len(triplets),
        recall_ks=recall_ks,
        topk_trace=args.topk_trace,
        runtime_info=runtime_info,
        progress=lambda message: print(message, flush=True),
    )
    comparison = {
        "run_root": str(run_root),
        "triplets_jsonl": str(triplets_path),
        "rows": [
            {"split": "smoke20", **smoke_summary["recall"]},
            {"split": f"full{len(triplets)}", **full_summary["recall"]},
        ],
    }
    (run_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run e5-omni only composed video retrieval evaluation")
    parser.add_argument("--triplets-jsonl")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-root")
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--e5-model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--video-max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--smoke-size", type=int, default=DEFAULT_SMOKE_SIZE)
    parser.add_argument("--topk", default=DEFAULT_TOPK)
    parser.add_argument("--topk-trace", type=int, default=10)
    parser.add_argument("--force-rebuild-index", action="store_true")
    return parser


def main() -> None:
    run_workflow(build_parser().parse_args())


def parse_topk(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in str(raw).split(",") if part.strip()})
    return _normalize_ks(tuple(values))


def _triplet_from_payload(payload: dict[str, Any], *, line_number: int) -> E5CVRTriplet:
    sample_id = str(payload.get("sample_id", "")).strip()
    reference_video = str(payload.get("reference_video", "")).strip()
    target_video = str(payload.get("target_video", "")).strip()
    edit_text = str(payload.get("edit_text", "")).strip()
    if not sample_id:
        raise ValueError(f"line {line_number} missing sample_id")
    if not reference_video:
        raise ValueError(f"line {line_number} missing reference_video")
    if not target_video:
        raise ValueError(f"line {line_number} missing target_video")
    if not edit_text:
        raise ValueError(f"line {line_number} missing edit_text")
    return E5CVRTriplet(
        sample_id=sample_id,
        reference_video=reference_video,
        target_video=target_video,
        edit_text=edit_text,
        reference_caption=str(payload.get("reference_caption", "")).strip(),
        source=str(payload.get("source", "")).strip(),
        difference_type=str(payload.get("difference_type", "")).strip(),
    )


def _query_payload(triplet: E5CVRTriplet) -> dict[str, str]:
    return {
        "video": triplet.reference_video,
        "text": QUERY_TEMPLATE.format(edit_text=triplet.edit_text.strip().rstrip(".")),
    }


def _encode_with_sentence_transformers(model: Any, inputs: list[Any], *, batch_size: int) -> np.ndarray:
    kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    try:
        return _as_2d_float32(model.encode_document(inputs, **kwargs))
    except TypeError:
        kwargs.pop("convert_to_numpy", None)
        kwargs.pop("show_progress_bar", None)
        return _as_2d_float32(model.encode_document(inputs, **kwargs))


def _configure_video_processing(model: Any, *, max_pixels: int, fps: int) -> None:
    processing = {"video": {"max_pixels": max_pixels, "do_sample_frames": True, "fps": fps}}
    try:
        target = model[0]
    except Exception:
        target = model
    existing = getattr(target, "processing_kwargs", None)
    if isinstance(existing, dict):
        existing.update(processing)


def _build_model_kwargs(torch_module: Any, *, torch_dtype: str, attn_implementation: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if torch_dtype and torch_dtype != "auto" and hasattr(torch_module, torch_dtype):
        kwargs["torch_dtype"] = getattr(torch_module, torch_dtype)
    if attn_implementation and attn_implementation != "default":
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _try_load_target_index(*, index_path: Path, embeddings_path: Path, cache_key: str) -> TargetIndex | None:
    if not index_path.exists() or not embeddings_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        if metadata.get("cache_key") != cache_key:
            return None
        records = [TargetRecord(**record) for record in payload["records"]]
        embeddings = np.load(str(embeddings_path)).astype(np.float32)
    except Exception:
        return None
    if embeddings.shape[0] != len(records):
        return None
    return TargetIndex(records=records, embeddings=_normalize_rows(embeddings), metadata=metadata)


def _target_index_cache_key(*, records: list[TargetRecord], runtime_info: dict[str, Any]) -> str:
    payload = {
        "runtime": runtime_info,
        "records": [_record_fingerprint(record) for record in records],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _record_fingerprint(record: TargetRecord) -> dict[str, Any]:
    path = Path(record.target_video)
    stat = path.stat()
    return {
        "sample_id": record.sample_id,
        "target_video": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _topk_hits(*, order: np.ndarray, scores: np.ndarray, target_index: TargetIndex, topk: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rank, row_index in enumerate(order[:topk], start=1):
        record = target_index.records[int(row_index)]
        hits.append(
            {
                "rank": rank,
                "sample_id": record.sample_id,
                "target_video": record.target_video,
                "score": round(float(scores[int(row_index)]), 6),
            }
        )
    return hits


def _normalize_rows(value: Any) -> np.ndarray:
    array = _as_2d_float32(value)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


def _as_2d_float32(value: Any) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected 2D embedding array, got shape {array.shape}")
    return array


def _normalize_ks(raw: tuple[int, ...]) -> list[int]:
    values = sorted({int(k) for k in raw if int(k) > 0})
    if not values:
        raise ValueError("topk values must contain at least one positive integer")
    return values


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# e5-omni CVR Only Comparison",
        "",
        f"- run_root: `{comparison['run_root']}`",
        f"- triplets_jsonl: `{comparison['triplets_jsonl']}`",
        "",
        "| Split | R@1 | R@5 | R@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(f"| {row['split']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _default_run_root() -> str:
    return f"{DEFAULT_RUNS_ROOT}/e5_cvr_eval_{time.strftime('%Y%m%d_%H%M%S')}"


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to run e5-omni CVR evaluation") from exc
    return torch


def _import_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError('sentence-transformers is required; install "sentence_transformers[image,audio,video]"') from exc
    return SentenceTransformer


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


if __name__ == "__main__":
    main()
