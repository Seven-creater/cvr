from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app.e5_audio_delta_train import prepare_omnicvr_records


MODEL_LABEL = "Audio-as-Text VLM2Vec reproduction"
UPSTREAM_REPOSITORY = "https://github.com/TIGER-AI-Lab/VLM2Vec"
UPSTREAM_BRANCH = "v1"
UPSTREAM_CHECKPOINT = "TIGER-Lab/VLM2Vec-Qwen2VL-7B"
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
MODES = ("V_T", "V_A_T")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _l2(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    denominator = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denominator, 1e-12)


def _resolve_media(raw: str, roots: Sequence[Path]) -> Path:
    candidate = Path(raw).expanduser()
    attempts = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    raise FileNotFoundError(f"media not found: {raw}; roots={[str(root) for root in roots]}")


def _media_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    payload = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return {
        "media_key": _sha256_text(payload),
        "path": str(path.resolve()),
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _subtype(row: dict[str, Any]) -> str:
    return str(row.get("b_subtype") or row.get("audio_delta_type") or "unknown")


def _audio_cvr_records(
    input_path: Path,
    output_dir: Path,
    roots: Sequence[Path],
    *,
    split_name: str,
    expected_count: int | None,
) -> dict[str, Any]:
    source_rows = _load_jsonl(input_path)
    if expected_count is not None and len(source_rows) != expected_count:
        raise ValueError(f"{input_path}: expected {expected_count} rows, found {len(source_rows)}")
    sample_ids = [str(row.get("sample_id") or "").strip() for row in source_rows]
    if any(not value for value in sample_ids):
        raise ValueError(f"{input_path}: sample_id values must be non-empty")
    sample_counts = Counter(sample_ids)

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        reference = _resolve_media(str(row["reference_video"]), roots)
        target = _resolve_media(str(row["target_video"]), roots)
        direction = str(row.get("direction") or "forward")
        record_id = (
            sample_ids[index]
            if sample_counts[sample_ids[index]] == 1
            else f"{sample_ids[index]}::{direction}"
        )
        normalized.append(
            {
                "sample_id": sample_ids[index],
                "record_id": record_id,
                "reference_video": str(reference),
                "target_video": str(target),
                "edit_text": str(row["edit_text"]).strip(),
                "subtype": _subtype(row),
                "dataset": str(row.get("dataset") or "unknown"),
                "raw_source_id": str(
                    row.get("source_disjoint_group_id")
                    or row.get("raw_source_id")
                    or row.get("group_id")
                    or sample_ids[index]
                ),
                "direction": direction,
                "is_inverse": bool(row.get("is_inverse", False)),
            }
        )
    record_ids = [row["record_id"] for row in normalized]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{input_path}: (sample_id, direction) records must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / f"{split_name}_triplets.jsonl", normalized)

    if split_name != "test":
        return {
            "split": split_name,
            "count": len(normalized),
            "unique_media_count": len(
                {row[key] for row in normalized for key in ("reference_video", "target_video")}
            ),
        }

    gallery: list[dict[str, Any]] = []
    for kind, key in (("target", "target_video"), ("reference", "reference_video")):
        for row in normalized:
            gallery.append(
                {
                    "gallery_id": f"audiocvr::{kind}::{row['sample_id']}",
                    "media_path": row[key],
                    "kind": kind,
                    "sample_id": row["sample_id"],
                }
            )
    gallery_lookup = {row["gallery_id"]: index for index, row in enumerate(gallery)}
    records = []
    for row in normalized:
        sample_id = row["sample_id"]
        records.append(
            {
                **row,
                "positive_index": gallery_lookup[f"audiocvr::target::{sample_id}"],
                "reference_index": gallery_lookup[f"audiocvr::reference::{sample_id}"],
                "candidate_indices": list(range(len(gallery))),
            }
        )
    _atomic_jsonl(output_dir / "test_records.jsonl", records)
    _atomic_jsonl(output_dir / "test_gallery.jsonl", gallery)
    return {
        "split": split_name,
        "count": len(records),
        "gallery_count": len(gallery),
        "unique_media_count": len({row["media_path"] for row in gallery}),
        "subtypes": dict(Counter(row["subtype"] for row in records)),
        "datasets": dict(Counter(row["dataset"] for row in records)),
    }


def prepare_records(
    *,
    audio_test: Path,
    audio_train: Path,
    audio_val: Path,
    omnicvr_annotations: Path,
    omnicvr_videos: Path,
    output_dir: Path,
    media_roots: Sequence[Path],
    audio_test_sha256: str,
    omnicvr_query_count: int,
    omnicvr_gallery_size: int,
) -> dict[str, Any]:
    actual_sha = _sha256_file(audio_test)
    if audio_test_sha256 and actual_sha != audio_test_sha256:
        raise ValueError(f"Audio-CVR test SHA256={actual_sha}, expected={audio_test_sha256}")
    roots = [Path.cwd(), *media_roots]
    audio_dir = output_dir / "audiocvr"
    omni_dir = output_dir / "omnicvr"
    audio_summary = {
        "test": _audio_cvr_records(audio_test, audio_dir, roots, split_name="test", expected_count=1000),
        "train": _audio_cvr_records(audio_train, audio_dir, roots, split_name="train", expected_count=None),
        "val": _audio_cvr_records(audio_val, audio_dir, roots, split_name="val", expected_count=None),
    }
    omni_summary = prepare_omnicvr_records(
        annotation_path=omnicvr_annotations,
        videos_dir=omnicvr_videos,
        output_dir=omni_dir / "prepared_e5",
        start_index=0,
        query_count=omnicvr_query_count,
        expected_gallery_size=omnicvr_gallery_size,
        require_existing_media=True,
    )
    omni_records = _load_jsonl(omni_dir / "prepared_e5" / "eval.jsonl")
    omni_gallery = _load_jsonl(omni_dir / "prepared_e5" / "eval_gallery.jsonl")
    indices = json.loads(
        (omni_dir / "prepared_e5" / "eval_gallery_positive_indices.json").read_text(encoding="utf-8")
    )
    omni_gallery_lookup = {row["gallery_id"]: index for index, row in enumerate(omni_gallery)}
    normalized_omni_records: list[dict[str, Any]] = []
    for index, row in enumerate(omni_records):
        candidate_indices = [omni_gallery_lookup[value] for value in row["candidate_gallery_ids"]]
        normalized_omni_records.append(
            {
                "sample_id": row["sample_id"],
                "reference_video": row["reference_video"],
                "target_video": row["target_video"],
                "edit_text": row["edit_text"],
                "subtype": "audio_center",
                "dataset": "omnicvr",
                "raw_source_id": row["source_id"],
                "positive_index": int(indices["positive_gallery_index"][index]),
                "reference_index": int(indices["reference_gallery_index"][index]),
                "candidate_indices": candidate_indices,
            }
        )
    normalized_omni_gallery = [
        {
            "gallery_id": row["gallery_id"],
            "media_path": row["video"],
            "kind": "candidate",
            "sample_id": row["gallery_id"],
        }
        for row in omni_gallery
    ]
    _atomic_jsonl(omni_dir / "test_records.jsonl", normalized_omni_records)
    _atomic_jsonl(omni_dir / "test_gallery.jsonl", normalized_omni_gallery)

    all_media: dict[str, dict[str, Any]] = {}
    for dataset in ("audiocvr", "omnicvr"):
        for row in _load_jsonl(output_dir / dataset / "test_gallery.jsonl"):
            identity = _media_identity(Path(row["media_path"]))
            all_media.setdefault(identity["media_key"], identity)
    for split in ("train", "val"):
        for row in _load_jsonl(audio_dir / f"{split}_triplets.jsonl"):
            for key in ("reference_video", "target_video"):
                identity = _media_identity(Path(row[key]))
                all_media.setdefault(identity["media_key"], identity)
    media_rows = sorted(all_media.values(), key=lambda row: row["media_key"])
    _atomic_jsonl(output_dir / "media_inventory.jsonl", media_rows)

    source_sets = {}
    for split in ("train", "val", "test"):
        rows = _load_jsonl(audio_dir / f"{split}_triplets.jsonl")
        source_sets[split] = {row["raw_source_id"] for row in rows}
    source_overlap = {
        "train_val": len(source_sets["train"] & source_sets["val"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "val_test": len(source_sets["val"] & source_sets["test"]),
    }
    if any(source_overlap.values()):
        raise ValueError(f"Audio-CVR source leakage: {source_overlap}")
    summary = {
        "model_label": MODEL_LABEL,
        "audio_test_sha256": actual_sha,
        "audio": audio_summary,
        "omnicvr": {
            "query_count": len(normalized_omni_records),
            "gallery_union_count": len(normalized_omni_gallery),
            "official_prepare_summary": omni_summary,
        },
        "media_inventory_count": len(media_rows),
        "audio_source_overlap": source_overlap,
    }
    _atomic_json(output_dir / "prepare_summary.json", summary)
    return summary


def _audio_waveform(path: Path, sample_rate: int = 16000) -> np.ndarray:
    import av

    arrays: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("video has no audio stream")
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            converted = resampler.resample(frame)
            for item in converted if isinstance(converted, list) else [converted]:
                if item is None:
                    continue
                arrays.append(np.asarray(item.to_ndarray(), dtype=np.float32).reshape(-1))
        flushed = resampler.resample(None)
        for item in flushed if isinstance(flushed, list) else [flushed]:
            if item is not None:
                arrays.append(np.asarray(item.to_ndarray(), dtype=np.float32).reshape(-1))
    if not arrays:
        raise ValueError("decoded audio is empty")
    waveform = np.concatenate(arrays)
    if not np.isfinite(waveform).all():
        raise ValueError("decoded audio contains NaN or Inf")
    return waveform


class Qwen2AudioCaptioner:
    def __init__(self, model_path: Path, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device)
        self.model.eval()

    def caption(self, path: Path) -> str:
        waveform = _audio_waveform(path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "audio.wav"},
                    {
                        "type": "text",
                        "text": (
                            "Describe only what is audible. Mention speech content at a high level, "
                            "sound events, music or instruments, foreground/background relations, and "
                            "temporal changes. Do not infer visual content. Be concrete and concise."
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=prompt,
            audios=[waveform],
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=72,
                do_sample=False,
                use_cache=True,
            )
        generated = generated[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise ValueError("Qwen2-Audio returned an empty caption")
        return text


def caption_audio(
    *,
    inventory_path: Path,
    cache_dir: Path,
    model_path: Path,
    shard_index: int,
    shard_count: int,
    device: str,
    retries: int,
) -> dict[str, Any]:
    rows = _load_jsonl(inventory_path)
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    pending = [row for row in selected if not (cache_dir / f"{row['media_key']}.json").is_file()]
    captioner = Qwen2AudioCaptioner(model_path, device) if pending else None
    encoded = failed = 0
    for row in pending:
        output_path = cache_dir / f"{row['media_key']}.json"
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                caption = captioner.caption(Path(row["path"])) if captioner else ""
                _atomic_json(
                    output_path,
                    {
                        **row,
                        "caption": caption,
                        "model_path": str(model_path),
                        "model_label": "Qwen2-Audio audio captioner",
                        "attempt": attempt,
                    },
                )
                encoded += 1
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(min(8, attempt * 2))
        if last_error is not None:
            failed += 1
            _atomic_json(
                cache_dir / "failures" / f"{row['media_key']}.json",
                {**row, "error_type": type(last_error).__name__, "error": str(last_error)},
            )
    summary = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(selected),
        "reused_count": len(selected) - len(pending),
        "encoded_count": encoded,
        "failed_count": failed,
    }
    _atomic_json(cache_dir / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}.json", summary)
    return summary


def audit_captions(inventory_path: Path, cache_dir: Path, output_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(inventory_path)
    complete = [row for row in rows if (cache_dir / f"{row['media_key']}.json").is_file()]
    failures = [row for row in rows if (cache_dir / "failures" / f"{row['media_key']}.json").is_file()]
    missing = [row["media_key"] for row in rows if row not in complete]
    summary = {
        "inventory_count": len(rows),
        "complete_count": len(complete),
        "failure_marker_count": len(failures),
        "missing_count": len(missing),
        "missing_keys": missing[:100],
        "complete": not missing,
    }
    _atomic_json(output_path, summary)
    return summary


def _caption_lookup(media_inventory: Path, caption_cache: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _load_jsonl(media_inventory):
        path = caption_cache / f"{row['media_key']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"caption missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[row["path"]] = str(payload["caption"]).strip()
    return result


def _embedding_item(
    *,
    dataset: str,
    split: str,
    mode: str,
    role: str,
    item_id: str,
    media_path: str,
    edit_text: str,
    caption: str,
) -> dict[str, Any]:
    if role == "query":
        if mode == "V_T":
            text = f"{IMAGE_TOKEN}\nRepresent this retrieval query. Modification: {edit_text}"
        else:
            text = (
                f"{IMAGE_TOKEN}\nRepresent this retrieval query. "
                f"Reference audio: {caption}\nModification: {edit_text}"
            )
    else:
        if mode == "V_T":
            text = f"{IMAGE_TOKEN}\nRepresent this candidate video for retrieval."
        else:
            text = f"{IMAGE_TOKEN}\nRepresent this candidate video. Candidate audio: {caption}"
    identity = {
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "role": role,
        "item_id": item_id,
        "media_path": media_path,
        "text": text,
        "preprocessing_version": "four_frame_contact_sheet_v1",
    }
    return {"embedding_key": _sha256_text(json.dumps(identity, sort_keys=True)), **identity}


def prepare_embedding_inventory(
    *,
    records_dir: Path,
    caption_cache: Path,
    output_path: Path,
) -> dict[str, Any]:
    captions = _caption_lookup(records_dir / "media_inventory.jsonl", caption_cache)
    items: dict[str, dict[str, Any]] = {}
    for dataset in ("audiocvr", "omnicvr"):
        records = _load_jsonl(records_dir / dataset / "test_records.jsonl")
        gallery = _load_jsonl(records_dir / dataset / "test_gallery.jsonl")
        for mode in MODES:
            for row in records:
                item = _embedding_item(
                    dataset=dataset,
                    split="test",
                    mode=mode,
                    role="query",
                    item_id=row["sample_id"],
                    media_path=row["reference_video"],
                    edit_text=row["edit_text"],
                    caption=captions[row["reference_video"]],
                )
                items[item["embedding_key"]] = item
            for row in gallery:
                item = _embedding_item(
                    dataset=dataset,
                    split="test",
                    mode=mode,
                    role="document",
                    item_id=row["gallery_id"],
                    media_path=row["media_path"],
                    edit_text="",
                    caption=captions[row["media_path"]],
                )
                items[item["embedding_key"]] = item
    for split in ("train", "val"):
        for row in _load_jsonl(records_dir / "audiocvr" / f"{split}_triplets.jsonl"):
            for mode in MODES:
                for role, media_key in (("query", "reference_video"), ("target", "target_video"), ("reference", "reference_video")):
                    item = _embedding_item(
                        dataset="audiocvr",
                        split=split,
                        mode=mode,
                        role="query" if role == "query" else "document",
                        item_id=f"{row['record_id']}::{role}",
                        media_path=row[media_key],
                        edit_text=row["edit_text"] if role == "query" else "",
                        caption=captions[row[media_key]],
                    )
                    items[item["embedding_key"]] = item
    rows = sorted(items.values(), key=lambda row: row["embedding_key"])
    _atomic_jsonl(output_path, rows)
    summary = {
        "item_count": len(rows),
        "by_dataset": dict(Counter(row["dataset"] for row in rows)),
        "by_mode": dict(Counter(row["mode"] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
    }
    _atomic_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _video_contact_sheet(path: Path, size: int = 336) -> Any:
    import av
    from PIL import Image

    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        if duration > 0:
            targets = np.linspace(0.1 * duration, 0.9 * duration, 4)
            for target in targets:
                container.seek(int(target * int(av.time_base)), any_frame=False, backward=True)
                selected = None
                for frame in container.decode(video=0):
                    selected = frame.to_image().convert("RGB")
                    break
                if selected is not None:
                    frames.append(selected)
        if not frames:
            for frame in container.decode(video=0):
                frames.append(frame.to_image().convert("RGB"))
                if len(frames) >= 4:
                    break
    if not frames:
        raise ValueError("video has no decodable frame")
    while len(frames) < 4:
        frames.append(frames[-1].copy())
    frames = frames[:4]
    canvas = Image.new("RGB", (size * 2, size * 2))
    for index, frame in enumerate(frames):
        frame.thumbnail((size, size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (size, size))
        tile.paste(frame, ((size - frame.width) // 2, (size - frame.height) // 2))
        canvas.paste(tile, ((index % 2) * size, (index // 2) * size))
    return canvas


class VLM2VecEncoder:
    def __init__(self, base_model: Path, adapter_model: Path, device: str) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(base_model), local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            str(base_model),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_model), is_trainable=False).merge_and_unload()
        self.model.to(device).eval()

    def encode(self, items: Sequence[dict[str, Any]]) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for item in items:
            image = _video_contact_sheet(Path(item["media_path"]))
            inputs = self.processor(
                images=[image],
                text=[item["text"]],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                output = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            hidden = output.hidden_states[-1]
            end = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[
                self.torch.arange(hidden.shape[0], device=hidden.device),
                end,
            ]
            pooled = self.torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
            results.append(pooled[0].cpu().numpy().astype(np.float32))
        return results


def encode_vlm2vec(
    *,
    inventory_path: Path,
    cache_dir: Path,
    base_model: Path,
    adapter_model: Path,
    shard_index: int,
    shard_count: int,
    device: str,
    batch_size: int,
    retries: int,
) -> dict[str, Any]:
    rows = _load_jsonl(inventory_path)
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    pending = [row for row in selected if not (cache_dir / f"{row['embedding_key']}.npy").is_file()]
    encoder = VLM2VecEncoder(base_model, adapter_model, device) if pending else None
    encoded = failed = 0
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        vectors: list[np.ndarray] | None = None
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                vectors = encoder.encode(batch) if encoder else []
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(min(8, attempt * 2))
        if vectors is None and len(batch) > 1:
            for item in batch:
                try:
                    vector = encoder.encode([item])[0] if encoder else np.empty(0)
                    _atomic_npy(cache_dir / f"{item['embedding_key']}.npy", vector)
                    encoded += 1
                except Exception as error:
                    failed += 1
                    _atomic_json(
                        cache_dir / "failures" / f"{item['embedding_key']}.json",
                        {**item, "error_type": type(error).__name__, "error": str(error)},
                    )
            continue
        if vectors is None:
            failed += 1
            item = batch[0]
            _atomic_json(
                cache_dir / "failures" / f"{item['embedding_key']}.json",
                {**item, "error_type": type(last_error).__name__, "error": str(last_error)},
            )
            continue
        for item, vector in zip(batch, vectors):
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"invalid embedding for {item['embedding_key']}")
            _atomic_npy(cache_dir / f"{item['embedding_key']}.npy", vector)
            encoded += 1
    summary = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(selected),
        "reused_count": len(selected) - len(pending),
        "encoded_count": encoded,
        "failed_count": failed,
    }
    _atomic_json(cache_dir / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}.json", summary)
    return summary


def audit_embeddings(inventory_path: Path, cache_dir: Path, output_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(inventory_path)
    complete = 0
    invalid: list[str] = []
    missing: list[str] = []
    dimensions: Counter[int] = Counter()
    for row in rows:
        path = cache_dir / f"{row['embedding_key']}.npy"
        if not path.is_file():
            missing.append(row["embedding_key"])
            continue
        value = np.load(path, allow_pickle=False)
        if value.ndim != 1 or not np.isfinite(value).all():
            invalid.append(row["embedding_key"])
            continue
        complete += 1
        dimensions[int(value.shape[0])] += 1
    summary = {
        "inventory_count": len(rows),
        "complete_count": complete,
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "dimensions": {str(key): value for key, value in dimensions.items()},
        "missing_keys": missing[:100],
        "invalid_keys": invalid[:100],
        "complete": complete == len(rows) and not invalid,
    }
    _atomic_json(output_path, summary)
    return summary


def _inventory_lookup(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    return {
        (row["dataset"], row["split"], row["mode"], row["role"], row["item_id"]): row
        for row in _load_jsonl(path)
    }


def _vector(
    lookup: dict[tuple[str, str, str, str, str], dict[str, Any]],
    cache_dir: Path,
    key: tuple[str, str, str, str, str],
) -> np.ndarray:
    row = lookup[key]
    return np.load(cache_dir / f"{row['embedding_key']}.npy", allow_pickle=False).astype(np.float32)


def _dataset_embeddings(
    records_dir: Path,
    inventory_path: Path,
    cache_dir: Path,
    dataset: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    records = _load_jsonl(records_dir / dataset / "test_records.jsonl")
    gallery = _load_jsonl(records_dir / dataset / "test_gallery.jsonl")
    lookup = _inventory_lookup(inventory_path)
    query = np.stack(
        [
            _vector(lookup, cache_dir, (dataset, "test", mode, "query", row["sample_id"]))
            for row in records
        ]
    )
    document = np.stack(
        [
            _vector(lookup, cache_dir, (dataset, "test", mode, "document", row["gallery_id"]))
            for row in gallery
        ]
    )
    return records, gallery, _l2(query), _l2(document)


def _metric_summary(
    scores: np.ndarray,
    records: Sequence[dict[str, Any]],
    *,
    mask_reference: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    work = np.asarray(scores, dtype=np.float32).copy()
    positive = np.asarray([row["positive_index"] for row in records], dtype=np.int64)
    reference = np.asarray([row["reference_index"] for row in records], dtype=np.int64)
    allowed = np.zeros(work.shape, dtype=bool)
    for index, row in enumerate(records):
        allowed[index, np.asarray(row["candidate_indices"], dtype=np.int64)] = True
    work[~allowed] = -np.inf
    rows = np.arange(len(records))
    positive_scores = work[rows, positive].copy()
    reference_scores = work[rows, reference].copy()
    reference_work = work.copy()
    reference_ranks = 1 + np.sum(reference_work > reference_scores[:, None], axis=1)
    if mask_reference:
        work[rows, reference] = -np.inf
    ranks = 1 + np.sum(work > positive_scores[:, None], axis=1)
    top1 = np.argmax(work, axis=1)
    values = {
        "rank": ranks,
        "reference_rank": reference_ranks,
        "correct_at_1": ranks <= 1,
        "correct_at_5": ranks <= 5,
        "correct_at_10": ranks <= 10,
        "reciprocal_rank": 1.0 / ranks,
        "target_beats_reference": positive_scores > reference_scores,
        "gap": positive_scores - reference_scores,
        "top1_is_reference": top1 == reference,
    }
    summary = {
        "query_count": len(records),
        "candidate_count_min": min(len(row["candidate_indices"]) for row in records) - int(mask_reference),
        "candidate_count_max": max(len(row["candidate_indices"]) for row in records) - int(mask_reference),
        "reference_in_gallery": not mask_reference,
        "R@1": float(np.mean(values["correct_at_1"])),
        "R@5": float(np.mean(values["correct_at_5"])),
        "R@10": float(np.mean(values["correct_at_10"])),
        "MRR": float(np.mean(values["reciprocal_rank"])),
        "target_rank_mean": float(np.mean(ranks)),
        "target_rank_median": float(np.median(ranks)),
        "target_beats_reference": float(np.mean(values["target_beats_reference"])),
        "target_reference_gap_mean": float(np.mean(values["gap"])),
        "reference_rank_mean": float(np.mean(reference_ranks)),
        "reference_rank_median": float(np.median(reference_ranks)),
        "top1_own_reference_rate": float(np.mean(values["top1_is_reference"])),
    }
    return summary, values


def _paired_test(a: np.ndarray, b: np.ndarray, *, iterations: int, seed: int) -> dict[str, Any]:
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(iterations, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(iterations, len(difference)))
    randomized = (difference[None, :] * signs).mean(axis=1)
    observed = float(difference.mean())
    return {
        "mean_difference": observed,
        "bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "paired_randomization_p_two_sided": float(
            (np.sum(np.abs(randomized) >= abs(observed)) + 1) / (iterations + 1)
        ),
        "iterations": iterations,
    }


def _binomial_tail(n: int, k: int) -> float:
    return sum(math.comb(n, value) for value in range(k, n + 1)) / (2**n)


def _mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p_value = min(1.0, 2.0 * _binomial_tail(discordant, max(a_only, b_only))) if discordant else 1.0
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_two_sided": p_value}


def evaluate_zero_shot(
    *,
    records_dir: Path,
    inventory_path: Path,
    cache_dir: Path,
    output_dir: Path,
    iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    all_values: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for dataset in ("audiocvr", "omnicvr"):
        dataset_results: dict[str, Any] = {}
        for mode in MODES:
            records, _, query, document = _dataset_embeddings(
                records_dir, inventory_path, cache_dir, dataset, mode
            )
            scores = query @ document.T
            with_ref, with_values = _metric_summary(scores, records, mask_reference=False)
            without_ref, without_values = _metric_summary(scores, records, mask_reference=True)
            with_ref["reference_induced_R@1_drop"] = without_ref["R@1"] - with_ref["R@1"]
            dataset_results[mode] = {
                "with_reference": with_ref,
                "without_reference": without_ref,
            }
            all_values[(dataset, mode, "with")] = with_values
            all_values[(dataset, mode, "without")] = without_values
        vat = all_values[(dataset, "V_A_T", "with")]
        vt = all_values[(dataset, "V_T", "with")]
        statistics = {
            "audio_gain_R@1": _paired_test(
                vat["correct_at_1"], vt["correct_at_1"], iterations=iterations, seed=20260724
            ),
            "audio_gain_target_beats_reference": _paired_test(
                vat["target_beats_reference"],
                vt["target_beats_reference"],
                iterations=iterations,
                seed=20260725,
            ),
            "audio_gain_gap": _paired_test(vat["gap"], vt["gap"], iterations=iterations, seed=20260726),
            "audio_gain_R@1_mcnemar": _mcnemar(vat["correct_at_1"], vt["correct_at_1"]),
        }
        dataset_results["statistics"] = statistics
        all_results[dataset] = dataset_results
    _atomic_json(output_dir / "zero_shot_results.json", all_results)
    return all_results


class LowRankResidualAdapter:
    def __init__(self, dimension: int, rank: int, device: str) -> None:
        import torch

        self.torch = torch
        self.query_a = torch.nn.Linear(dimension, rank, bias=False, device=device)
        self.query_b = torch.nn.Linear(rank, dimension, bias=False, device=device)
        self.document_a = torch.nn.Linear(dimension, rank, bias=False, device=device)
        self.document_b = torch.nn.Linear(rank, dimension, bias=False, device=device)
        torch.nn.init.normal_(self.query_a.weight, std=0.02)
        torch.nn.init.zeros_(self.query_b.weight)
        torch.nn.init.normal_(self.document_a.weight, std=0.02)
        torch.nn.init.zeros_(self.document_b.weight)

    def parameters(self) -> list[Any]:
        return [
            *self.query_a.parameters(),
            *self.query_b.parameters(),
            *self.document_a.parameters(),
            *self.document_b.parameters(),
        ]

    def query(self, value: Any) -> Any:
        return self.torch.nn.functional.normalize(value + self.query_b(self.query_a(value)), p=2, dim=-1)

    def document(self, value: Any) -> Any:
        return self.torch.nn.functional.normalize(
            value + self.document_b(self.document_a(value)), p=2, dim=-1
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "query_a": self.query_a.state_dict(),
            "query_b": self.query_b.state_dict(),
            "document_a": self.document_a.state_dict(),
            "document_b": self.document_b.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.query_a.load_state_dict(state["query_a"])
        self.query_b.load_state_dict(state["query_b"])
        self.document_a.load_state_dict(state["document_a"])
        self.document_b.load_state_dict(state["document_b"])


def _triplet_arrays(
    records_dir: Path,
    inventory_path: Path,
    cache_dir: Path,
    split: str,
    mode: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    records = _load_jsonl(records_dir / "audiocvr" / f"{split}_triplets.jsonl")
    lookup = _inventory_lookup(inventory_path)
    query = []
    target = []
    reference = []
    for row in records:
        sample_id = row["record_id"]
        query.append(
            _vector(lookup, cache_dir, ("audiocvr", split, mode, "query", f"{sample_id}::query"))
        )
        target.append(
            _vector(lookup, cache_dir, ("audiocvr", split, mode, "document", f"{sample_id}::target"))
        )
        reference.append(
            _vector(
                lookup,
                cache_dir,
                ("audiocvr", split, mode, "document", f"{sample_id}::reference"),
            )
        )
    return records, _l2(np.stack(query)), _l2(np.stack(target)), _l2(np.stack(reference))


def train_adapter(
    *,
    records_dir: Path,
    inventory_path: Path,
    cache_dir: Path,
    output_dir: Path,
    seed: int,
    rank: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records, query_np, target_np, reference_np = _triplet_arrays(
        records_dir, inventory_path, cache_dir, "train", "V_A_T"
    )
    dimension = int(query_np.shape[1])
    adapter = LowRankResidualAdapter(dimension, rank, device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=0.01)
    query = torch.tensor(query_np, device=device)
    target = torch.tensor(target_np, device=device)
    reference = torch.tensor(reference_np, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(records), (batch_size,), generator=generator).to(device)
        q = adapter.query(query[indices])
        t = adapter.document(target[indices])
        r = adapter.document(reference[indices])
        logits = q @ t.T / 0.05
        labels = torch.arange(len(indices), device=device)
        contrastive = torch.nn.functional.cross_entropy(logits, labels)
        directional = torch.nn.functional.softplus(0.1 + torch.sum(q * r, dim=1) - torch.sum(q * t, dim=1)).mean()
        loss = contrastive + directional
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "contrastive": float(contrastive.detach().cpu()),
                "directional": float(directional.detach().cpu()),
            }
        )
    if not all(math.isfinite(row["loss"]) for row in losses):
        raise ValueError("adapter loss contains NaN or Inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "dimension": dimension,
            "rank": rank,
            "seed": seed,
            "steps": steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        output_dir / "adapter.pt",
    )
    _atomic_jsonl(output_dir / "loss_curve.jsonl", losses)

    val_records, val_query, val_target, val_reference = _triplet_arrays(
        records_dir, inventory_path, cache_dir, "val", "V_A_T"
    )
    with torch.inference_mode():
        vq = adapter.query(torch.tensor(val_query, device=device))
        vt = adapter.document(torch.tensor(val_target, device=device))
        vr = adapter.document(torch.tensor(val_reference, device=device))
        target_score = torch.sum(vq * vt, dim=1)
        reference_score = torch.sum(vq * vr, dim=1)
    summary = {
        "model_label": "task-adapted Audio-as-Text VLM2Vec reproduction",
        "seed": seed,
        "train_count": len(records),
        "val_count": len(val_records),
        "rank": rank,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "trainable_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "val_target_beats_reference": float(torch.mean((target_score > reference_score).float()).cpu()),
        "val_target_reference_gap": float(torch.mean(target_score - reference_score).cpu()),
        "test_used_for_selection": False,
    }
    _atomic_json(output_dir / "train_summary.json", summary)
    return summary


def evaluate_adapters(
    *,
    records_dir: Path,
    inventory_path: Path,
    cache_dir: Path,
    adapter_dirs: Sequence[Path],
    output_dir: Path,
    device: str,
    iterations: int,
) -> dict[str, Any]:
    import torch

    per_seed: dict[str, Any] = {}
    values_by_seed: dict[tuple[int, str, str, str], dict[str, np.ndarray]] = {}
    for adapter_dir in adapter_dirs:
        checkpoint = torch.load(adapter_dir / "adapter.pt", map_location=device, weights_only=False)
        seed = int(checkpoint["seed"])
        adapter = LowRankResidualAdapter(int(checkpoint["dimension"]), int(checkpoint["rank"]), device)
        adapter.load_state_dict(checkpoint["state_dict"])
        seed_results: dict[str, Any] = {}
        with torch.inference_mode():
            for dataset in ("audiocvr", "omnicvr"):
                dataset_results: dict[str, Any] = {}
                for mode in MODES:
                    records, _, query, document = _dataset_embeddings(
                        records_dir, inventory_path, cache_dir, dataset, mode
                    )
                    adapted_query = adapter.query(torch.tensor(query, device=device)).cpu().numpy()
                    adapted_document = adapter.document(torch.tensor(document, device=device)).cpu().numpy()
                    scores = adapted_query @ adapted_document.T
                    with_ref, with_values = _metric_summary(scores, records, mask_reference=False)
                    without_ref, without_values = _metric_summary(scores, records, mask_reference=True)
                    with_ref["reference_induced_R@1_drop"] = without_ref["R@1"] - with_ref["R@1"]
                    dataset_results[mode] = {
                        "with_reference": with_ref,
                        "without_reference": without_ref,
                    }
                    values_by_seed[(seed, dataset, mode, "with")] = with_values
                    values_by_seed[(seed, dataset, mode, "without")] = without_values
                seed_results[dataset] = dataset_results
        per_seed[str(seed)] = seed_results

    seeds = sorted(int(value) for value in per_seed)
    aggregate: dict[str, Any] = {}
    for dataset in ("audiocvr", "omnicvr"):
        aggregate[dataset] = {}
        for mode in MODES:
            aggregate[dataset][mode] = {}
            for reference_state in ("with_reference", "without_reference"):
                metric_rows = [
                    per_seed[str(seed)][dataset][mode][reference_state]
                    for seed in seeds
                ]
                aggregate[dataset][mode][reference_state] = {
                    key: {
                        "mean": float(np.mean([row[key] for row in metric_rows])),
                        "std": float(np.std([row[key] for row in metric_rows])),
                    }
                    for key in ("R@1", "R@5", "R@10", "MRR", "target_beats_reference", "target_reference_gap_mean")
                }
        seed_statistics = {}
        for seed in seeds:
            vat = values_by_seed[(seed, dataset, "V_A_T", "with")]
            vt = values_by_seed[(seed, dataset, "V_T", "with")]
            seed_statistics[str(seed)] = {
                "audio_gain_R@1": _paired_test(
                    vat["correct_at_1"], vt["correct_at_1"], iterations=iterations, seed=seed
                ),
                "audio_gain_gap": _paired_test(
                    vat["gap"], vt["gap"], iterations=iterations, seed=seed + 1000
                ),
                "mcnemar": _mcnemar(vat["correct_at_1"], vt["correct_at_1"]),
            }
        aggregate[dataset]["statistics"] = seed_statistics
    result = {"seeds": seeds, "per_seed": per_seed, "mean_std": aggregate}
    _atomic_json(output_dir / "adapter_results.json", result)
    return result


def summarize(
    *,
    zero_shot_path: Path,
    adapter_path: Path,
    prepare_summary_path: Path,
    output_dir: Path,
) -> None:
    zero = json.loads(zero_shot_path.read_text(encoding="utf-8"))
    adapted = json.loads(adapter_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepare_summary_path.read_text(encoding="utf-8"))
    lines = [
        "# Audio-as-Text VLM2Vec Reference Diagnostic",
        "",
        "This is an independent reproduction built from the public VLM2Vec-Qwen2VL-7B adapter and",
        "Qwen2-Audio captions. It is not the unreleased official AudioVLM2Vec checkpoint.",
        "",
        f"- Audio-CVR Test1000 SHA256: `{prepared['audio_test_sha256']}`",
        f"- Audio-CVR queries: {prepared['audio']['test']['count']}",
        f"- OmniCVR queries: {prepared['omnicvr']['query_count']}",
        "",
    ]
    for dataset in ("audiocvr", "omnicvr"):
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Model | Mode | Source | R@1 | R@5 | R@10 | Target>Source | Gap |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in MODES:
            for state, label in (("with_reference", "with"), ("without_reference", "masked")):
                row = zero[dataset][mode][state]
                lines.append(
                    f"| zero-shot | {mode} | {label} | {row['R@1']:.4f} | {row['R@5']:.4f} | "
                    f"{row['R@10']:.4f} | {row['target_beats_reference']:.4f} | "
                    f"{row['target_reference_gap_mean']:.4f} |"
                )
        for mode in MODES:
            for state, label in (("with_reference", "with"), ("without_reference", "masked")):
                row = adapted["mean_std"][dataset][mode][state]
                lines.append(
                    f"| adapter mean±std | {mode} | {label} | "
                    f"{row['R@1']['mean']:.4f}±{row['R@1']['std']:.4f} | "
                    f"{row['R@5']['mean']:.4f}±{row['R@5']['std']:.4f} | "
                    f"{row['R@10']['mean']:.4f}±{row['R@10']['std']:.4f} | "
                    f"{row['target_beats_reference']['mean']:.4f}±{row['target_beats_reference']['std']:.4f} | "
                    f"{row['target_reference_gap_mean']['mean']:.4f}±{row['target_reference_gap_mean']['std']:.4f} |"
                )
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(
        output_dir / "model_provenance.json",
        {
            "model_label": MODEL_LABEL,
            "official_audiovlm2vec_checkpoint_public": False,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_branch": UPSTREAM_BRANCH,
            "upstream_checkpoint": UPSTREAM_CHECKPOINT,
            "audio_captioner": "Qwen2-Audio",
            "visual_temporal_representation": "four-frame contact sheet",
            "test_selection_uses_metrics": False,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=MODEL_LABEL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--audio-test", type=Path, required=True)
    prepare.add_argument("--audio-train", type=Path, required=True)
    prepare.add_argument("--audio-val", type=Path, required=True)
    prepare.add_argument("--omnicvr-annotations", type=Path, required=True)
    prepare.add_argument("--omnicvr-videos", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--media-root", type=Path, action="append", default=[])
    prepare.add_argument("--audio-test-sha256", default="")
    prepare.add_argument("--omnicvr-query-count", type=int, default=1000)
    prepare.add_argument("--omnicvr-gallery-size", type=int, default=2000)

    caption = subparsers.add_parser("caption-audio")
    caption.add_argument("--inventory", type=Path, required=True)
    caption.add_argument("--cache-dir", type=Path, required=True)
    caption.add_argument("--model", type=Path, required=True)
    caption.add_argument("--shard-index", type=int, required=True)
    caption.add_argument("--shard-count", type=int, required=True)
    caption.add_argument("--device", default="cuda")
    caption.add_argument("--retries", type=int, default=4)

    caption_audit = subparsers.add_parser("audit-captions")
    caption_audit.add_argument("--inventory", type=Path, required=True)
    caption_audit.add_argument("--cache-dir", type=Path, required=True)
    caption_audit.add_argument("--output", type=Path, required=True)

    inventory = subparsers.add_parser("prepare-embedding-inventory")
    inventory.add_argument("--records-dir", type=Path, required=True)
    inventory.add_argument("--caption-cache", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)

    encode = subparsers.add_parser("encode")
    encode.add_argument("--inventory", type=Path, required=True)
    encode.add_argument("--cache-dir", type=Path, required=True)
    encode.add_argument("--base-model", type=Path, required=True)
    encode.add_argument("--adapter-model", type=Path, required=True)
    encode.add_argument("--shard-index", type=int, required=True)
    encode.add_argument("--shard-count", type=int, required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--batch-size", type=int, default=2)
    encode.add_argument("--retries", type=int, default=4)

    embedding_audit = subparsers.add_parser("audit-embeddings")
    embedding_audit.add_argument("--inventory", type=Path, required=True)
    embedding_audit.add_argument("--cache-dir", type=Path, required=True)
    embedding_audit.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-zero-shot")
    evaluate.add_argument("--records-dir", type=Path, required=True)
    evaluate.add_argument("--inventory", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--iterations", type=int, default=20000)

    train = subparsers.add_parser("train-adapter")
    train.add_argument("--records-dir", type=Path, required=True)
    train.add_argument("--inventory", type=Path, required=True)
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--rank", type=int, default=32)
    train.add_argument("--steps", type=int, default=400)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--device", default="cuda")

    adapter_eval = subparsers.add_parser("evaluate-adapters")
    adapter_eval.add_argument("--records-dir", type=Path, required=True)
    adapter_eval.add_argument("--inventory", type=Path, required=True)
    adapter_eval.add_argument("--cache-dir", type=Path, required=True)
    adapter_eval.add_argument("--adapter-dir", type=Path, action="append", required=True)
    adapter_eval.add_argument("--output-dir", type=Path, required=True)
    adapter_eval.add_argument("--device", default="cuda")
    adapter_eval.add_argument("--iterations", type=int, default=20000)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--zero-shot", type=Path, required=True)
    summary.add_argument("--adapter-results", type=Path, required=True)
    summary.add_argument("--prepare-summary", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_records(
            audio_test=args.audio_test,
            audio_train=args.audio_train,
            audio_val=args.audio_val,
            omnicvr_annotations=args.omnicvr_annotations,
            omnicvr_videos=args.omnicvr_videos,
            output_dir=args.output_dir,
            media_roots=args.media_root,
            audio_test_sha256=args.audio_test_sha256,
            omnicvr_query_count=args.omnicvr_query_count,
            omnicvr_gallery_size=args.omnicvr_gallery_size,
        )
    elif args.command == "caption-audio":
        result = caption_audio(
            inventory_path=args.inventory,
            cache_dir=args.cache_dir,
            model_path=args.model,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            retries=args.retries,
        )
    elif args.command == "audit-captions":
        result = audit_captions(args.inventory, args.cache_dir, args.output)
    elif args.command == "prepare-embedding-inventory":
        result = prepare_embedding_inventory(
            records_dir=args.records_dir,
            caption_cache=args.caption_cache,
            output_path=args.output,
        )
    elif args.command == "encode":
        result = encode_vlm2vec(
            inventory_path=args.inventory,
            cache_dir=args.cache_dir,
            base_model=args.base_model,
            adapter_model=args.adapter_model,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            batch_size=args.batch_size,
            retries=args.retries,
        )
    elif args.command == "audit-embeddings":
        result = audit_embeddings(args.inventory, args.cache_dir, args.output)
    elif args.command == "evaluate-zero-shot":
        result = evaluate_zero_shot(
            records_dir=args.records_dir,
            inventory_path=args.inventory,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            iterations=args.iterations,
        )
    elif args.command == "train-adapter":
        result = train_adapter(
            records_dir=args.records_dir,
            inventory_path=args.inventory,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            rank=args.rank,
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "evaluate-adapters":
        result = evaluate_adapters(
            records_dir=args.records_dir,
            inventory_path=args.inventory,
            cache_dir=args.cache_dir,
            adapter_dirs=args.adapter_dir,
            output_dir=args.output_dir,
            device=args.device,
            iterations=args.iterations,
        )
    elif args.command == "summarize":
        summarize(
            zero_shot_path=args.zero_shot,
            adapter_path=args.adapter_results,
            prepare_summary_path=args.prepare_summary,
            output_dir=args.output_dir,
        )
        result = {"output_dir": str(args.output_dir)}
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
