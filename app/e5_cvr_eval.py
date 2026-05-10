from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import numpy as np


DEFAULT_RUNS_ROOT = "/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs"
DEFAULT_E5_MODEL = "/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B"
DEFAULT_REFERENCE_AUDIO_CACHE_DIR = f"{DEFAULT_RUNS_ROOT}/e5_reference_audio_media_cache"
DEFAULT_EXPECTED_COUNT = 943
DEFAULT_SMOKE_SIZE = 20
DEFAULT_TOPK = "1,5,10"
DEFAULT_VIDEO_MAX_PIXELS = 64 * 28 * 28
QUERY_TEMPLATE = "Edit the reference video so that: {edit_text}"
QUERY_MODE_COMPOSED = "composed"
QUERY_MODE_VIDEO_ONLY = "video-only"
QUERY_MODES = (QUERY_MODE_COMPOSED, QUERY_MODE_VIDEO_ONLY)
REFERENCE_AUDIO_MODE_ORIGINAL = "original"
REFERENCE_AUDIO_MODE_MUTED = "muted"
REFERENCE_AUDIO_MODE_SILENT = "silent"
REFERENCE_AUDIO_MODES = (REFERENCE_AUDIO_MODE_ORIGINAL, REFERENCE_AUDIO_MODE_MUTED, REFERENCE_AUDIO_MODE_SILENT)
VIDEO_AUDIO_MODE_ON = "on"
VIDEO_AUDIO_MODE_OFF = "off"
VIDEO_AUDIO_MODES = (VIDEO_AUDIO_MODE_ON, VIDEO_AUDIO_MODE_OFF)


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
    video_audio_mode: str
    load_audio_from_video: bool
    processor_video_kwargs_sanitizer: bool


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


def prepare_reference_audio_triplets(
    *,
    triplets: list[E5CVRTriplet],
    reference_audio_mode: str,
    cache_dir: str | Path,
    output_dir: str | Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    command_runner: Callable[[list[str]], None] | None = None,
    stream_probe: Callable[[Path], list[dict[str, Any]]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[E5CVRTriplet], dict[str, Any]]:
    mode = _normalize_reference_audio_mode(reference_audio_mode)
    if mode == REFERENCE_AUDIO_MODE_ORIGINAL:
        return list(triplets), _reference_audio_summary(
            mode=mode,
            cache_dir="",
            total=len(triplets),
            generated=0,
            reused=0,
        )
    if not triplets:
        raise ValueError("triplets must not be empty")

    cache_root = Path(cache_dir)
    output_root = Path(output_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    triplets_path = output_root / f"reference_{mode}_triplets.jsonl"
    manifest_path = output_root / f"reference_{mode}_media_manifest.jsonl"
    summary_path = output_root / f"reference_{mode}_media_summary.json"
    prepared: list[E5CVRTriplet] = []
    generated = 0
    reused = 0
    probe = stream_probe or (lambda path: _probe_streams(path, ffprobe=ffprobe))
    runner = command_runner or _run_command

    with triplets_path.open("w", encoding="utf-8") as triplets_file, manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index, triplet in enumerate(triplets, start=1):
            source = Path(triplet.reference_video)
            if not source.exists():
                raise FileNotFoundError(f"reference video not found for {triplet.sample_id}: {source}")
            prepared_path = _reference_audio_path(source=source, sample_id=triplet.sample_id, cache_dir=cache_root, mode=mode)
            if _reference_audio_video_is_valid(prepared_path, mode=mode, stream_probe=probe):
                action = "reuse"
                reused += 1
                _emit(progress, f"[e5-cvr] reference-audio {mode} reuse {index}/{len(triplets)} sample_id={triplet.sample_id} role=reference path={prepared_path}")
            else:
                _emit(progress, f"[e5-cvr] reference-audio {mode} start {index}/{len(triplets)} sample_id={triplet.sample_id} role=reference src={source}")
                _rewrite_reference_audio(source=source, output=prepared_path, mode=mode, ffmpeg=ffmpeg, command_runner=runner)
                if not _reference_audio_video_is_valid(prepared_path, mode=mode, stream_probe=probe):
                    raise RuntimeError(f"{mode} reference video failed validation: {prepared_path}")
                action = "generated"
                generated += 1
                _emit(progress, f"[e5-cvr] reference-audio {mode} done {index}/{len(triplets)} sample_id={triplet.sample_id} role=reference path={prepared_path}")
            prepared_triplet = replace(triplet, reference_video=str(prepared_path))
            prepared.append(prepared_triplet)
            triplets_file.write(json.dumps(asdict(prepared_triplet), ensure_ascii=False) + "\n")
            triplets_file.flush()
            manifest_file.write(
                json.dumps(
                    {
                        "sample_id": triplet.sample_id,
                        "action": action,
                        "role": "reference",
                        "original_reference_video": triplet.reference_video,
                        "prepared_reference_video": str(prepared_path),
                        "reference_audio_mode": mode,
                        "target_video": triplet.target_video,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest_file.flush()

    summary = _reference_audio_summary(
        mode=mode,
        cache_dir=str(cache_root),
        total=len(triplets),
        generated=generated,
        reused=reused,
    )
    summary["triplets_path"] = str(triplets_path)
    summary["manifest_path"] = str(manifest_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(progress, f"[e5-cvr] wrote reference {mode} triplets: {triplets_path}")
    _emit(progress, f"[e5-cvr] wrote reference {mode} media manifest: {manifest_path}")
    _emit(progress, f"[e5-cvr] wrote reference {mode} media summary: {summary_path}")
    return prepared, summary


def load_e5_encoder(
    *,
    model_path: str,
    device: str,
    torch_dtype: str,
    attn_implementation: str,
    batch_size: int,
    video_max_pixels: int,
    video_fps: int,
    video_audio_mode: str = VIDEO_AUDIO_MODE_ON,
) -> tuple[E5SentenceTransformerEncoder, E5RuntimeInfo]:
    video_audio_mode = _normalize_video_audio_mode(video_audio_mode)
    load_audio_from_video = _video_audio_loads(video_audio_mode)
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

    processor_video_kwargs_sanitizer = _configure_video_processing(
        model,
        max_pixels=video_max_pixels,
        fps=video_fps,
        load_audio_from_video=load_audio_from_video,
    )
    if load_audio_from_video and not processor_video_kwargs_sanitizer:
        raise RuntimeError(
            "audio-in-video e5 evaluation needs a processor __call__ sanitizer for load_audio_from_video; "
            "could not find a patchable processor/tokenizer on the SentenceTransformer module"
        )
    info = E5RuntimeInfo(
        model_path=str(model_root),
        device=device,
        torch_dtype=torch_dtype,
        requested_attention=attn_implementation or "default",
        used_attention=used_attention,
        batch_size=batch_size,
        video_max_pixels=video_max_pixels,
        video_fps=video_fps,
        video_audio_mode=video_audio_mode,
        load_audio_from_video=load_audio_from_video,
        processor_video_kwargs_sanitizer=processor_video_kwargs_sanitizer,
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
        if index_path.exists() or embeddings_path.exists():
            raise ValueError(
                "target index exists but cache key does not match this runtime; "
                "do not reuse audio-off/no-sound e5 results for audio-enabled CVR. "
                f"Use a fresh target index dir or pass --force-rebuild-index: {root}"
            )

    target_batch_size = _positive_int(runtime_payload.get("batch_size"), default=1)
    _emit(progress, f"[e5-cvr] encoding {len(records)} target videos batch_size={target_batch_size}")
    embeddings = _encode_records_with_progress(
        encoder=encoder,
        records=records,
        batch_size=target_batch_size,
        progress=progress,
    )
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
    _emit(progress, f"[e5-cvr] wrote target embeddings: {embeddings_path}")
    _emit(progress, f"[e5-cvr] wrote target index: {index_path}")
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
    query_mode: str = QUERY_MODE_COMPOSED,
    reference_audio_mode: str = REFERENCE_AUDIO_MODE_ORIGINAL,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    query_mode = _normalize_query_mode(query_mode)
    reference_audio_mode = _normalize_reference_audio_mode(reference_audio_mode)
    runtime_payload = asdict(runtime_info) if isinstance(runtime_info, E5RuntimeInfo) else dict(runtime_info)
    video_audio_mode = str(runtime_payload.get("video_audio_mode", VIDEO_AUDIO_MODE_OFF))
    load_audio_from_video = bool(runtime_payload.get("load_audio_from_video", False))
    processor_video_kwargs_sanitizer = bool(runtime_payload.get("processor_video_kwargs_sanitizer", False))
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
    traces_path = output_root / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as traces_file:
        for query_index, triplet in enumerate(selected, start=1):
            _emit(
                progress,
                f"[e5-cvr] query {query_index}/{len(selected)} start sample_id={triplet.sample_id} "
                f"mode={query_mode} video_audio_mode={video_audio_mode}",
            )
            query_embedding = _normalize_rows(encoder.encode_document([_query_payload(triplet, query_mode=query_mode)]))[0]
            scores = target_index.embeddings @ query_embedding
            order = np.argsort(-scores, kind="stable")
            target_index_value = target_position[triplet.sample_id]
            target_rank = int(np.where(order == target_index_value)[0][0]) + 1
            for k in recall_ks:
                if target_rank <= k:
                    hit_counts[k] += 1
            traces_file.write(
                json.dumps(
                    {
                        "sample_id": triplet.sample_id,
                        "reference_video": triplet.reference_video,
                        "target_video": triplet.target_video,
                        "edit_text": triplet.edit_text,
                        "target_rank": target_rank,
                        "target_score": round(float(scores[target_index_value]), 6),
                        "query_index": query_index,
                        "query_mode": query_mode,
                        "query_used_text": _query_uses_text(query_mode),
                        "video_audio_mode": video_audio_mode,
                        "load_audio_from_video": load_audio_from_video,
                        "processor_video_kwargs_sanitizer": processor_video_kwargs_sanitizer,
                        "reference_audio_mode": reference_audio_mode,
                        "target_audio_mode": "original",
                        "reference_audio_transform": _reference_audio_transform(reference_audio_mode),
                        "audio_removed_scope": "reference_only" if _reference_audio_removed(reference_audio_mode) else "none",
                        "audio_removed": _reference_audio_removed(reference_audio_mode),
                        "topk_hits": _topk_hits(order=order, scores=scores, target_index=target_index, topk=max_trace),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            traces_file.flush()
            _emit(progress, f"[e5-cvr] query {query_index}/{len(selected)} done rank={target_rank}")

    summary = {
        "mode": "e5-cvr-only",
        "query_mode": query_mode,
        "query_input": _query_input_label(query_mode),
        "uses_edit_text_for_embedding": _query_uses_text(query_mode),
        "video_audio_mode": video_audio_mode,
        "load_audio_from_video": load_audio_from_video,
        "processor_video_kwargs_sanitizer": processor_video_kwargs_sanitizer,
        "reference_audio_mode": reference_audio_mode,
        "target_audio_mode": "original",
        "reference_audio_transform": _reference_audio_transform(reference_audio_mode),
        "audio_removed_scope": "reference_only" if _reference_audio_removed(reference_audio_mode) else "none",
        "audio_removed": _reference_audio_removed(reference_audio_mode),
        "query_count": len(selected),
        "gallery_count": len(target_index.records),
        "recall": {f"R@{k}": round(hit_counts[k] / max(1, len(selected)), 4) for k in recall_ks},
        "topk_trace": topk_trace,
        "query_template": QUERY_TEMPLATE if _query_uses_text(query_mode) else "",
        "runtime": runtime_payload,
        "target_index": target_index.metadata,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(progress, f"[e5-cvr] wrote traces: {traces_path}")
    _emit(progress, f"[e5-cvr] wrote summary: {output_root / 'summary.json'}")
    return summary


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    query_mode = _normalize_query_mode(args.query_mode)
    reference_audio_mode = _normalize_reference_audio_mode(args.reference_audio_mode)
    video_audio_mode = _normalize_video_audio_mode(args.video_audio_mode)
    run_root = Path(args.run_root or _default_run_root(query_mode=query_mode, reference_audio_mode=reference_audio_mode, video_audio_mode=video_audio_mode))
    run_root.mkdir(parents=True, exist_ok=True)
    triplets_path = Path(args.triplets_jsonl) if args.triplets_jsonl else find_latest_triplets(args.runs_root)
    triplets = load_triplets_jsonl(triplets_path, expected_count=args.expected_count)
    gallery_triplets_path = Path(args.gallery_triplets_jsonl) if args.gallery_triplets_jsonl else triplets_path
    gallery_expected_count = args.gallery_expected_count if args.gallery_triplets_jsonl else args.expected_count
    gallery_triplets = (
        load_triplets_jsonl(gallery_triplets_path, expected_count=gallery_expected_count)
        if gallery_triplets_path != triplets_path
        else list(triplets)
    )
    triplets, reference_audio_summary = prepare_reference_audio_triplets(
        triplets=triplets,
        reference_audio_mode=reference_audio_mode,
        cache_dir=args.reference_audio_cache_dir,
        output_dir=run_root,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        progress=lambda message: print(message, flush=True),
    )
    encoder, runtime_info = load_e5_encoder(
        model_path=args.e5_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        batch_size=args.batch_size,
        video_max_pixels=args.video_max_pixels,
        video_fps=args.video_fps,
        video_audio_mode=video_audio_mode,
    )
    index = build_or_load_target_index(
        triplets=gallery_triplets,
        encoder=encoder,
        index_dir=Path(args.target_index_dir) if args.target_index_dir else run_root / "target_index",
        runtime_info=runtime_info,
        force_rebuild=args.force_rebuild_index,
        progress=lambda message: print(message, flush=True),
    )
    target_index_dir = Path(args.target_index_dir) if args.target_index_dir else run_root / "target_index"
    target_index_reference = {
        "target_index_dir": str(target_index_dir),
        "gallery_triplets_jsonl": str(gallery_triplets_path),
        "gallery_count": len(index.records),
        "target_index": index.metadata,
    }
    (run_root / "target_index_reference.json").write_text(
        json.dumps(target_index_reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[e5-cvr] wrote target index reference: {run_root / 'target_index_reference.json'}", flush=True)
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
        query_mode=query_mode,
        reference_audio_mode=reference_audio_mode,
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
        query_mode=query_mode,
        reference_audio_mode=reference_audio_mode,
        progress=lambda message: print(message, flush=True),
    )
    comparison = {
        "run_root": str(run_root),
        "triplets_jsonl": str(triplets_path),
        "gallery_triplets_jsonl": str(gallery_triplets_path),
        "query_count": len(triplets),
        "gallery_count": len(index.records),
        "query_mode": query_mode,
        "query_input": _query_input_label(query_mode),
        "uses_edit_text_for_embedding": _query_uses_text(query_mode),
        "video_audio_mode": video_audio_mode,
        "load_audio_from_video": runtime_info.load_audio_from_video,
        "processor_video_kwargs_sanitizer": runtime_info.processor_video_kwargs_sanitizer,
        "reference_audio_mode": reference_audio_mode,
        "target_audio_mode": "original",
        "audio_removed_scope": reference_audio_summary["audio_removed_scope"],
        "audio_removed": reference_audio_summary["audio_removed"],
        "reference_audio": reference_audio_summary,
        "rows": [
            {"split": "smoke20", **smoke_summary["recall"]},
            {"split": f"full{len(triplets)}", **full_summary["recall"]},
        ],
    }
    (run_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    print(f"[e5-cvr] wrote comparison json: {run_root / 'comparison.json'}", flush=True)
    print(f"[e5-cvr] wrote comparison md: {run_root / 'comparison.md'}", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run e5-omni only composed video retrieval evaluation")
    parser.add_argument("--triplets-jsonl")
    parser.add_argument("--gallery-triplets-jsonl")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-root")
    parser.add_argument("--target-index-dir")
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--gallery-expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--e5-model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--video-max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--video-audio-mode", choices=VIDEO_AUDIO_MODES, default=VIDEO_AUDIO_MODE_ON)
    parser.add_argument("--smoke-size", type=int, default=DEFAULT_SMOKE_SIZE)
    parser.add_argument("--query-mode", choices=QUERY_MODES, default=QUERY_MODE_COMPOSED)
    parser.add_argument("--reference-audio-mode", choices=REFERENCE_AUDIO_MODES, default=REFERENCE_AUDIO_MODE_ORIGINAL)
    parser.add_argument("--reference-audio-cache-dir", default=DEFAULT_REFERENCE_AUDIO_CACHE_DIR)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
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


def _reference_audio_summary(*, mode: str, cache_dir: str, total: int, generated: int, reused: int) -> dict[str, Any]:
    return {
        "reference_audio_mode": mode,
        "target_audio_mode": "original",
        "audio_removed_scope": "reference_only" if _reference_audio_removed(mode) else "none",
        "audio_removed": _reference_audio_removed(mode),
        "reference_audio_transform": _reference_audio_transform(mode),
        "media_cache_dir": cache_dir,
        "total": total,
        "generated_count": generated,
        "reused_count": reused,
    }


def _reference_audio_removed(mode: str) -> bool:
    return _normalize_reference_audio_mode(mode) != REFERENCE_AUDIO_MODE_ORIGINAL


def _reference_audio_transform(mode: str) -> str:
    mode = _normalize_reference_audio_mode(mode)
    if mode == REFERENCE_AUDIO_MODE_MUTED:
        return "strip"
    if mode == REFERENCE_AUDIO_MODE_SILENT:
        return "silent"
    return "none"


def _reference_audio_path(*, source: Path, sample_id: str, cache_dir: Path, mode: str) -> Path:
    mode = _normalize_reference_audio_mode(mode)
    fingerprint = _file_fingerprint_hash(source)
    return cache_dir / mode / f"{_safe_filename_part(sample_id)}_{fingerprint}.mp4"


def _file_fingerprint_hash(path: Path) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _safe_filename_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return (safe or "sample")[:80]


def _rewrite_reference_audio(
    *,
    source: Path,
    output: Path,
    mode: str,
    ffmpeg: str,
    command_runner: Callable[[list[str]], None],
) -> None:
    mode = _normalize_reference_audio_mode(mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == REFERENCE_AUDIO_MODE_MUTED:
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            str(output),
        ]
    elif mode == REFERENCE_AUDIO_MODE_SILENT:
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=16000",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
    else:
        raise ValueError(f"cannot rewrite reference audio for mode {mode!r}")
    command_runner(command)


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _reference_audio_video_is_valid(path: Path, *, mode: str, stream_probe: Callable[[Path], list[dict[str, Any]]]) -> bool:
    if not path.exists():
        return False
    mode = _normalize_reference_audio_mode(mode)
    try:
        streams = stream_probe(path)
    except Exception:
        return False
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    if mode == REFERENCE_AUDIO_MODE_MUTED:
        return has_video and not has_audio
    if mode == REFERENCE_AUDIO_MODE_SILENT:
        return has_video and has_audio
    return has_video


def _probe_streams(path: Path, *, ffprobe: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    return streams if isinstance(streams, list) else []


def _query_payload(triplet: E5CVRTriplet, *, query_mode: str) -> str | dict[str, str]:
    query_mode = _normalize_query_mode(query_mode)
    if query_mode == QUERY_MODE_VIDEO_ONLY:
        return triplet.reference_video
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


def _configure_video_processing(model: Any, *, max_pixels: int, fps: int, load_audio_from_video: bool) -> bool:
    processing = {
        "video": {
            "max_pixels": max_pixels,
            "do_sample_frames": True,
            "fps": fps,
            "load_audio_from_video": load_audio_from_video,
        }
    }
    try:
        target = model[0]
    except Exception:
        target = model
    existing = getattr(target, "processing_kwargs", None)
    if isinstance(existing, dict):
        for key, value in processing.items():
            if isinstance(existing.get(key), dict):
                existing[key].update(value)
            else:
                existing[key] = value
    else:
        try:
            setattr(target, "processing_kwargs", processing)
        except Exception:
            pass
    return _patch_processor_video_kwargs_sanitizer(target) if load_audio_from_video else False


def _patch_processor_video_kwargs_sanitizer(target: Any) -> bool:
    patched = False
    seen: set[int] = set()
    for attr_name in ("processor", "tokenizer"):
        processor = getattr(target, attr_name, None)
        if processor is None or id(processor) in seen:
            continue
        seen.add(id(processor))
        patched = _patch_processor_instance_video_kwargs_sanitizer(processor) or patched
    return patched


def _patch_processor_instance_video_kwargs_sanitizer(processor: Any) -> bool:
    processor_cls = processor.__class__
    marker = "_cvr_load_audio_from_video_sanitizer"
    if getattr(processor_cls, marker, False):
        return True
    original_call = getattr(processor_cls, "__call__", None)
    if original_call is None:
        return False

    def sanitized_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        videos_kwargs = kwargs.get("videos_kwargs")
        if isinstance(videos_kwargs, dict) and "load_audio_from_video" in videos_kwargs:
            videos_kwargs = dict(videos_kwargs)
            videos_kwargs.pop("load_audio_from_video", None)
            kwargs["videos_kwargs"] = videos_kwargs
        return original_call(self, *args, **kwargs)

    sanitized_call.__name__ = getattr(original_call, "__name__", "__call__")
    sanitized_call.__doc__ = getattr(original_call, "__doc__", None)
    setattr(processor_cls, "__call__", sanitized_call)
    setattr(processor_cls, marker, True)
    return True


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


def _encode_records_with_progress(
    *,
    encoder: Any,
    records: list[TargetRecord],
    batch_size: int,
    progress: Callable[[str], None] | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = len(records)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = records[start:stop]
        sample_ids = ",".join(record.sample_id for record in batch[:3])
        suffix = "..." if len(batch) > 3 else ""
        _emit(progress, f"[e5-cvr] target {start + 1}-{stop}/{total} start sample_id={sample_ids}{suffix}")
        encoded = _normalize_rows(encoder.encode_document([record.target_video for record in batch]))
        if encoded.shape[0] != len(batch):
            raise ValueError(f"target batch row count mismatch: {encoded.shape[0]} vs {len(batch)}")
        chunks.append(encoded)
        _emit(progress, f"[e5-cvr] target {start + 1}-{stop}/{total} done")
    if not chunks:
        raise ValueError("no target embeddings were encoded")
    return _normalize_rows(np.vstack(chunks))


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


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_query_mode(value: str) -> str:
    if value not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of {', '.join(QUERY_MODES)}, got {value!r}")
    return value


def _normalize_reference_audio_mode(value: str) -> str:
    if value not in REFERENCE_AUDIO_MODES:
        raise ValueError(f"reference_audio_mode must be one of {', '.join(REFERENCE_AUDIO_MODES)}, got {value!r}")
    return value


def _normalize_video_audio_mode(value: str) -> str:
    if value not in VIDEO_AUDIO_MODES:
        raise ValueError(f"video_audio_mode must be one of {', '.join(VIDEO_AUDIO_MODES)}, got {value!r}")
    return value


def _video_audio_loads(value: str) -> bool:
    return _normalize_video_audio_mode(value) == VIDEO_AUDIO_MODE_ON


def _query_uses_text(query_mode: str) -> bool:
    return _normalize_query_mode(query_mode) == QUERY_MODE_COMPOSED


def _query_input_label(query_mode: str) -> str:
    return "reference_video + edit_text" if _query_uses_text(query_mode) else "reference_video"


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# e5-omni CVR Only Comparison",
        "",
        f"- run_root: `{comparison['run_root']}`",
        f"- triplets_jsonl: `{comparison['triplets_jsonl']}`",
        f"- query_mode: `{comparison['query_mode']}`",
        f"- query_input: `{comparison['query_input']}`",
        f"- uses_edit_text_for_embedding: `{str(comparison['uses_edit_text_for_embedding']).lower()}`",
        f"- video_audio_mode: `{comparison['video_audio_mode']}`",
        f"- load_audio_from_video: `{str(comparison['load_audio_from_video']).lower()}`",
        f"- processor_video_kwargs_sanitizer: `{str(comparison['processor_video_kwargs_sanitizer']).lower()}`",
        f"- reference_audio_mode: `{comparison['reference_audio_mode']}`",
        f"- target_audio_mode: `{comparison['target_audio_mode']}`",
        f"- reference_audio_transform: `{comparison['reference_audio']['reference_audio_transform']}`",
        f"- audio_removed_scope: `{comparison['audio_removed_scope']}`",
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


def _default_run_root(*, query_mode: str, reference_audio_mode: str, video_audio_mode: str) -> str:
    query_mode = _normalize_query_mode(query_mode)
    reference_audio_mode = _normalize_reference_audio_mode(reference_audio_mode)
    video_audio_mode = _normalize_video_audio_mode(video_audio_mode)
    audio_prefix = "e5_audio_on" if video_audio_mode == VIDEO_AUDIO_MODE_ON else "e5_audio_off"
    if reference_audio_mode == REFERENCE_AUDIO_MODE_ORIGINAL:
        suffix = "video_only_eval" if query_mode == QUERY_MODE_VIDEO_ONLY else "composed_eval"
    else:
        query_part = "video_only" if query_mode == QUERY_MODE_VIDEO_ONLY else "composed"
        suffix = f"{query_part}_ref_{reference_audio_mode}_eval"
    prefix = f"{audio_prefix}_{suffix}"
    return f"{DEFAULT_RUNS_ROOT}/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


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
