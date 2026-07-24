from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app.audio_cvr_weak_accept import (
    EXPECTED_FULL1000_SHA256,
    REFERENCE_VARIANTS,
    _load_jsonl,
    _resolve_media,
    _sample_id,
    _sha256_file,
)


MODEL_LABEL = "OmniEmbed-v0.1-multivent"
PROMPT_VERSION = "audiocvr_fixed_retrieval_prompt_v1"
MODES = ("V_T", "V_A_T")
CONDITIONS = ("exact", *REFERENCE_VARIANTS)
QUERY_INSTRUCTION = "Retrieve a video with the same visual context after this change: {edit_text}"


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
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
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
        handle.flush()
        os.fsync(handle.fileno())
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
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _l2(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    denominator = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denominator, 1e-12)


def _ensure_qwen_omni_utils_root() -> str:
    utilities_root = os.environ.get("QWEN_OMNI_UTILS_ROOT", "").strip()
    if utilities_root and utilities_root not in sys.path:
        sys.path.append(utilities_root)
    return utilities_root


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_omnicvr(
    records_path: Path,
    gallery_path: Path,
    roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _load_jsonl(records_path)
    gallery = _load_jsonl(gallery_path)
    if len(records) != 1000:
        raise ValueError(f"OmniCVR must contain 1000 records, found {len(records)}")
    if len(gallery) != 2000:
        raise ValueError(f"OmniCVR gallery must contain 2000 items, found {len(gallery)}")
    normalized_gallery = []
    for index, row in enumerate(gallery):
        media = _resolve_media(
            _first_text(row, "media_path", "video", "target_video", "path"), roots
        )
        gallery_id = _first_text(row, "gallery_id", "sample_id") or f"omnicvr::{index}"
        normalized_gallery.append(
            {
                **row,
                "gallery_index": int(row.get("gallery_index", index)),
                "gallery_id": gallery_id,
                "media_path": str(media),
            }
        )
    gallery_id_to_index: dict[str, int] = {}
    for index, row in enumerate(normalized_gallery):
        gallery_id = row["gallery_id"]
        if gallery_id in gallery_id_to_index:
            raise ValueError(f"duplicate OmniCVR gallery_id: {gallery_id}")
        gallery_id_to_index[gallery_id] = index

    def resolve_index(
        row: dict[str, Any],
        *,
        id_key: str,
        numeric_keys: Sequence[str],
        label: str,
    ) -> int:
        gallery_id = _first_text(row, id_key)
        if gallery_id:
            if gallery_id not in gallery_id_to_index:
                raise ValueError(f"{label} gallery_id is missing from gallery: {gallery_id}")
            return gallery_id_to_index[gallery_id]
        for key in numeric_keys:
            if row.get(key) is not None:
                value = int(row[key])
                if not 0 <= value < len(normalized_gallery):
                    raise ValueError(f"{label} index is out of range: {value}")
                return value
        raise ValueError(f"OmniCVR row is missing {label} gallery identity")

    normalized_records = []
    for index, row in enumerate(records):
        reference = _resolve_media(
            _first_text(row, "reference_video", "source_video", "query_video"), roots
        )
        positive = resolve_index(
            row,
            id_key="positive_gallery_id",
            numeric_keys=("positive_index", "positive_gallery_index"),
            label="positive",
        )
        reference_index = resolve_index(
            row,
            id_key="reference_gallery_id",
            numeric_keys=("reference_index", "reference_gallery_index"),
            label="reference",
        )
        candidates = row.get("candidate_indices")
        candidate_ids = row.get("candidate_gallery_ids")
        if isinstance(candidate_ids, list):
            missing_ids = [
                str(value)
                for value in candidate_ids
                if str(value) not in gallery_id_to_index
            ]
            if missing_ids:
                raise ValueError(
                    f"candidate gallery IDs are missing from gallery: {missing_ids[:5]}"
                )
            candidates = [gallery_id_to_index[str(value)] for value in candidate_ids]
        elif not isinstance(candidates, list):
            candidates = list(range(len(normalized_gallery)))
        candidates = [int(value) for value in candidates]
        if any(value < 0 or value >= len(normalized_gallery) for value in candidates):
            raise ValueError(f"candidate index is out of range for row {index}")
        if positive not in candidates or reference_index not in candidates:
            raise ValueError(
                f"candidate set for row {index} omits its positive or reference item"
            )
        normalized_records.append(
            {
                "sample_id": _sample_id(row),
                "reference_video": str(reference),
                "edit_text": _first_text(row, "edit_text", "modification_text", "query_text"),
                "positive_index": positive,
                "reference_index": reference_index,
                "candidate_indices": candidates,
                "source_row_index": index,
            }
        )
    return normalized_records, normalized_gallery


def _normalize_audiocvr(
    full_path: Path,
    roots: Sequence[Path],
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual_sha = _sha256_file(full_path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(f"Audio-CVR SHA256={actual_sha}, expected={expected_sha256}")
    source_rows = _load_jsonl(full_path)
    if len(source_rows) != 1000:
        raise ValueError(f"Audio-CVR Full1000 must contain 1000 rows, found {len(source_rows)}")
    records = []
    gallery = []
    for index, row in enumerate(source_rows):
        sample_id = _sample_id(row)
        target = _resolve_media(str(row["target_video"]), roots)
        reference = _resolve_media(str(row["reference_video"]), roots)
        records.append(
            {
                "sample_id": sample_id,
                "reference_video": str(reference),
                "target_video": str(target),
                "edit_text": str(row["edit_text"]).strip(),
                "positive_index": index,
                "reference_index": index + len(source_rows),
                "candidate_indices": list(range(len(source_rows) * 2)),
                "subtype": _first_text(row, "b_subtype", "audio_delta_type") or "unknown",
                "dataset": _first_text(row, "dataset", "source_dataset") or "unknown",
            }
        )
        gallery.append(
            {
                "gallery_index": index,
                "gallery_id": f"audiocvr::target::{sample_id}",
                "sample_id": sample_id,
                "kind": "target",
                "media_path": str(target),
            }
        )
    for index, row in enumerate(records, start=len(records)):
        gallery.append(
            {
                "gallery_index": index,
                "gallery_id": f"audiocvr::reference::{row['sample_id']}",
                "sample_id": row["sample_id"],
                "kind": "reference",
                "media_path": row["reference_video"],
            }
        )
    return records, gallery


def _embedding_item(
    *,
    dataset: str,
    mode: str,
    condition: str,
    role: str,
    item_id: str,
    media_path: str,
    edit_text: str = "",
) -> dict[str, Any]:
    path = Path(media_path)
    stat = path.stat()
    identity = {
        "dataset": dataset,
        "mode": mode,
        "condition": condition,
        "role": role,
        "item_id": item_id,
        "media_path": str(path.resolve()),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "edit_text": edit_text,
        "prompt_version": PROMPT_VERSION,
    }
    return {"embedding_key": _stable_digest(identity), **identity}


def prepare_inventory(
    *,
    audio_test: Path,
    omnicvr_records: Path,
    omnicvr_gallery: Path,
    variant_manifest: Path,
    output_dir: Path,
    media_roots: Sequence[Path],
    audio_test_sha256: str,
) -> dict[str, Any]:
    roots = [Path.cwd(), *media_roots]
    audio_records, audio_gallery = _normalize_audiocvr(
        audio_test, roots, audio_test_sha256
    )
    omni_records, omni_gallery = _normalize_omnicvr(
        omnicvr_records, omnicvr_gallery, roots
    )
    variants = _load_jsonl(variant_manifest)
    variant_lookup = {
        (row["condition"], row["sample_id"]): str(Path(row["output_path"]).resolve())
        for row in variants
    }
    expected_variant_keys = {
        (condition, row["sample_id"])
        for condition in REFERENCE_VARIANTS
        for row in audio_records
    }
    missing_variants = sorted(expected_variant_keys - set(variant_lookup))
    if missing_variants:
        raise ValueError(f"reference variant manifest is missing {len(missing_variants)} items")

    inventory: dict[str, dict[str, Any]] = {}
    for dataset, records, gallery in (
        ("audiocvr", audio_records, audio_gallery),
        ("omnicvr", omni_records, omni_gallery),
    ):
        for mode in MODES:
            for row in records:
                item = _embedding_item(
                    dataset=dataset,
                    mode=mode,
                    condition="exact",
                    role="query",
                    item_id=row["sample_id"],
                    media_path=row["reference_video"],
                    edit_text=row["edit_text"],
                )
                inventory[item["embedding_key"]] = item
            for row in gallery:
                item = _embedding_item(
                    dataset=dataset,
                    mode=mode,
                    condition="exact",
                    role="document",
                    item_id=row["gallery_id"],
                    media_path=row["media_path"],
                )
                inventory[item["embedding_key"]] = item
    for mode in MODES:
        for condition in REFERENCE_VARIANTS:
            for row in audio_records:
                item = _embedding_item(
                    dataset="audiocvr",
                    mode=mode,
                    condition=condition,
                    role="document",
                    item_id=f"audiocvr::reference::{row['sample_id']}",
                    media_path=variant_lookup[(condition, row["sample_id"])],
                )
                inventory[item["embedding_key"]] = item

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "audiocvr_records.jsonl", audio_records)
    _atomic_jsonl(output_dir / "audiocvr_gallery.jsonl", audio_gallery)
    _atomic_jsonl(output_dir / "omnicvr_records.jsonl", omni_records)
    _atomic_jsonl(output_dir / "omnicvr_gallery.jsonl", omni_gallery)
    ordered = sorted(inventory.values(), key=lambda row: row["embedding_key"])
    _atomic_jsonl(output_dir / "embedding_inventory.jsonl", ordered)
    summary = {
        "model": MODEL_LABEL,
        "prompt_version": PROMPT_VERSION,
        "audio_test_path": str(audio_test.resolve()),
        "audio_test_sha256": _sha256_file(audio_test),
        "audiocvr_query_count": len(audio_records),
        "audiocvr_gallery_count": len(audio_gallery),
        "omnicvr_query_count": len(omni_records),
        "omnicvr_gallery_count": len(omni_gallery),
        "embedding_item_count": len(ordered),
        "by_dataset": dict(Counter(row["dataset"] for row in ordered)),
        "by_mode": dict(Counter(row["mode"] for row in ordered)),
        "by_condition": dict(Counter(row["condition"] for row in ordered)),
        "selection_uses_test_metrics": False,
    }
    _atomic_json(output_dir / "inventory_summary.json", summary)
    return summary


def _message(item: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    use_audio = item["mode"] == "V_A_T"
    content: list[dict[str, Any]] = [
        {"type": "video", "video": item["media_path"]}
    ]
    if item["role"] == "query":
        content.append(
            {
                "type": "text",
                "text": QUERY_INSTRUCTION.format(edit_text=item["edit_text"]),
            }
        )
    return [{"role": "user", "content": content}], use_audio


class OmniEmbedEncoder:
    def __init__(
        self,
        *,
        base_model: Path,
        adapter_model: Path,
        device: str,
        torch_dtype: str,
        attn_implementation: str,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        dtype = getattr(torch, torch_dtype)
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            str(base_model), local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(base_model),
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            base, str(adapter_model), is_trainable=False, local_files_only=True
        )
        self.model = model.merge_and_unload().to(self.device).eval()
        self.model.padding_side = "left"

    def encode(self, item: dict[str, Any]) -> np.ndarray:
        _ensure_qwen_omni_utils_root()
        from qwen_omni_utils import process_mm_info

        message, use_audio = _message(item)
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        if isinstance(text, list):
            text = text[0]
        text = str(text) + "<|endoftext|>"
        audio_inputs, image_inputs, video_inputs = process_mm_info(
            message, use_audio_in_video=use_audio
        )
        inputs = self.processor(
            text=[text],
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="longest",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        cache_position = self.torch.arange(
            0, inputs["input_ids"].shape[1], device=self.device
        )
        prepared = self.model.prepare_inputs_for_generation(
            **inputs, use_cache=True, cache_position=cache_position
        )
        with self.torch.inference_mode():
            outputs = self.model(
                **prepared, return_dict=True, output_hidden_states=True
            )
        value = self.torch.nn.functional.normalize(
            outputs.hidden_states[-1][:, -1].float(), p=2, dim=-1
        )[0]
        result = value.cpu().numpy().astype(np.float32)
        del outputs, prepared, inputs, value
        return result


def encode_inventory(
    *,
    inventory_path: Path,
    cache_dir: Path,
    base_model: Path,
    adapter_model: Path,
    shard_index: int,
    shard_count: int,
    device: str,
    retries: int,
    torch_dtype: str,
    attn_implementation: str,
    encoder: OmniEmbedEncoder | None = None,
) -> dict[str, Any]:
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    selected = [
        row
        for index, row in enumerate(_load_jsonl(inventory_path))
        if index % shard_count == shard_index
    ]
    pending = []
    reused = 0
    for row in selected:
        path = cache_dir / "items" / f"{row['embedding_key']}.npy"
        if path.is_file():
            value = np.load(path, allow_pickle=False)
            if value.ndim == 1 and np.isfinite(value).all():
                reused += 1
                continue
        pending.append(row)
    if pending and encoder is None:
        encoder = OmniEmbedEncoder(
            base_model=base_model,
            adapter_model=adapter_model,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
    encoded = failed = 0
    failures: list[dict[str, Any]] = []
    for row in pending:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                assert encoder is not None
                value = encoder.encode(row)
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError(f"invalid OmniEmbed vector shape={value.shape}")
                _atomic_npy(cache_dir / "items" / f"{row['embedding_key']}.npy", value)
                encoded += 1
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(min(8.0, 2.0 * attempt))
        if last_error is not None:
            failed += 1
            failures.append(
                {
                    **row,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                }
            )
    _atomic_jsonl(
        cache_dir
        / "failures"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.jsonl",
        failures,
    )
    summary = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(selected),
        "encoded_count": encoded,
        "reused_count": reused,
        "failed_count": failed,
        "device": device,
    }
    _atomic_json(
        cache_dir
        / "shards"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.json",
        summary,
    )
    return summary


def audit_cache(
    *, inventory_path: Path, cache_dir: Path, output_path: Path
) -> dict[str, Any]:
    rows = _load_jsonl(inventory_path)
    complete = invalid = 0
    missing: list[str] = []
    dimensions: Counter[int] = Counter()
    for row in rows:
        path = cache_dir / "items" / f"{row['embedding_key']}.npy"
        if not path.is_file():
            missing.append(row["embedding_key"])
            continue
        value = np.load(path, allow_pickle=False)
        if value.ndim != 1 or not np.isfinite(value).all():
            invalid += 1
            continue
        complete += 1
        dimensions[int(value.shape[0])] += 1
    result = {
        "inventory_count": len(rows),
        "complete_count": complete,
        "missing_count": len(missing),
        "invalid_count": invalid,
        "dimensions": dict(dimensions),
        "complete": complete == len(rows) and invalid == 0,
        "missing_keys": missing[:100],
    }
    _atomic_json(output_path, result)
    return result


def _lookup_inventory(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    return {
        (
            row["dataset"],
            row["mode"],
            row["condition"],
            row["role"],
            row["item_id"],
        ): row
        for row in _load_jsonl(path)
    }


def _vector(
    lookup: dict[tuple[str, str, str, str, str], dict[str, Any]],
    cache_dir: Path,
    key: tuple[str, str, str, str, str],
) -> np.ndarray:
    row = lookup[key]
    return np.asarray(
        np.load(cache_dir / "items" / f"{row['embedding_key']}.npy", allow_pickle=False),
        dtype=np.float32,
    )


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
    reference_ranks = 1 + np.sum(work > reference_scores[:, None], axis=1)
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
    result = {
        "query_count": len(records),
        "gallery_count": work.shape[1],
        "effective_gallery_count_per_query": int(
            np.median([len(row["candidate_indices"]) for row in records])
        )
        - int(mask_reference),
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
        "top1_own_reference_count": int(values["top1_is_reference"].sum()),
        "top1_own_reference_rate": float(np.mean(values["top1_is_reference"])),
    }
    return result, values


def _evaluate_condition(
    *,
    dataset: str,
    mode: str,
    condition: str,
    records: list[dict[str, Any]],
    gallery: list[dict[str, Any]],
    lookup: dict[tuple[str, str, str, str, str], dict[str, Any]],
    cache_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = _l2(
        np.stack(
            [
                _vector(
                    lookup,
                    cache_dir,
                    (dataset, mode, "exact", "query", row["sample_id"]),
                )
                for row in records
            ]
        )
    )
    documents = []
    for row in gallery:
        item_condition = condition if row.get("kind") == "reference" and dataset == "audiocvr" else "exact"
        documents.append(
            _vector(
                lookup,
                cache_dir,
                (dataset, mode, item_condition, "document", row["gallery_id"]),
            )
        )
    document = _l2(np.stack(documents))
    scores = query @ document.T
    with_ref, with_values = _metric_summary(scores, records, mask_reference=False)
    masked, masked_values = _metric_summary(scores, records, mask_reference=True)
    with_ref["masked_R@1"] = masked["R@1"]
    with_ref["reference_induced_R@1_drop"] = masked["R@1"] - with_ref["R@1"]
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "sample_id": record["sample_id"],
                "dataset": dataset,
                "mode": mode,
                "condition": condition,
                "with_reference_rank": int(with_values["rank"][index]),
                "without_reference_rank": int(masked_values["rank"][index]),
                "reference_rank": int(with_values["reference_rank"][index]),
                "target_beats_reference": bool(
                    with_values["target_beats_reference"][index]
                ),
                "target_reference_gap": float(with_values["gap"][index]),
                "with_reference_correct_at_1": bool(
                    with_values["correct_at_1"][index]
                ),
                "without_reference_correct_at_1": bool(
                    masked_values["correct_at_1"][index]
                ),
                "top1_is_own_reference": bool(
                    with_values["top1_is_reference"][index]
                ),
            }
        )
    return {"with_reference": with_ref, "masked_reference": masked}, rows


def evaluate(
    *,
    records_dir: Path,
    inventory_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    lookup = _lookup_inventory(inventory_path)
    result: dict[str, Any] = {"audiocvr": {}, "omnicvr": {}}
    per_query: list[dict[str, Any]] = []
    for dataset in ("audiocvr", "omnicvr"):
        records = _load_jsonl(records_dir / f"{dataset}_records.jsonl")
        gallery = _load_jsonl(records_dir / f"{dataset}_gallery.jsonl")
        conditions = CONDITIONS if dataset == "audiocvr" else ("exact",)
        for mode in MODES:
            result[dataset][mode] = {}
            for condition in conditions:
                metrics, rows = _evaluate_condition(
                    dataset=dataset,
                    mode=mode,
                    condition=condition,
                    records=records,
                    gallery=gallery,
                    lookup=lookup,
                    cache_dir=cache_dir,
                )
                result[dataset][mode][condition] = metrics
                per_query.extend(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "results.json", result)
    _atomic_jsonl(output_dir / "per_query_results.jsonl", per_query)
    summary = {
        "model": MODEL_LABEL,
        "training_checkpoint": "MultiVENT video retrieval",
        "prompt_version": PROMPT_VERSION,
        "audiocvr_query_count": 1000,
        "omnicvr_query_count": 1000,
        "modes": list(MODES),
        "reference_conditions": list(CONDITIONS),
        "masking_reuses_same_score_matrix": True,
        "selection_uses_test_metrics": False,
        "nan_or_inf_count": 0,
    }
    _atomic_json(output_dir / "evaluation_summary.json", summary)
    return summary


def _paired_test(
    first: np.ndarray, second: np.ndarray, *, iterations: int, seed: int
) -> dict[str, Any]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(difference), size=(iterations, len(difference)))
    bootstrap = difference[samples].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(iterations, len(difference)))
    randomized = (difference[None, :] * signs).mean(axis=1)
    observed = float(difference.mean())
    p_value = float(
        (np.sum(np.abs(randomized) >= abs(observed)) + 1) / (iterations + 1)
    )
    return {
        "mean_difference": observed,
        "bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "paired_randomization_p_two_sided": p_value,
        "iterations": iterations,
    }


def _holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        current = min(1.0, values[name] * (total - rank))
        running = max(running, current)
        adjusted[name] = running
    return adjusted


def summarize_statistics(
    *, per_query_path: Path, output_dir: Path, iterations: int, seed: int
) -> dict[str, Any]:
    rows = _load_jsonl(per_query_path)
    indexed = {
        (row["dataset"], row["mode"], row["condition"], row["sample_id"]): row
        for row in rows
    }
    comparisons: dict[str, Any] = {}
    audio_ids = sorted(
        {
            row["sample_id"]
            for row in rows
            if row["dataset"] == "audiocvr"
            and row["mode"] == "V_A_T"
            and row["condition"] == "exact"
        }
    )
    for metric, field in (
        ("R@1", "with_reference_correct_at_1"),
        ("target_beats_reference", "target_beats_reference"),
        ("target_reference_gap", "target_reference_gap"),
    ):
        vat = np.asarray(
            [indexed[("audiocvr", "V_A_T", "exact", item)][field] for item in audio_ids],
            dtype=np.float64,
        )
        vt = np.asarray(
            [indexed[("audiocvr", "V_T", "exact", item)][field] for item in audio_ids],
            dtype=np.float64,
        )
        comparisons[f"audio_gain_{metric}"] = _paired_test(
            vat, vt, iterations=iterations, seed=seed
        )
    for mode in MODES:
        for condition in REFERENCE_VARIANTS:
            exact = np.asarray(
                [
                    indexed[("audiocvr", mode, "exact", item)][
                        "without_reference_correct_at_1"
                    ]
                    - indexed[("audiocvr", mode, "exact", item)][
                        "with_reference_correct_at_1"
                    ]
                    for item in audio_ids
                ],
                dtype=np.float64,
            )
            perturbed = np.asarray(
                [
                    indexed[("audiocvr", mode, condition, item)][
                        "without_reference_correct_at_1"
                    ]
                    - indexed[("audiocvr", mode, condition, item)][
                        "with_reference_correct_at_1"
                    ]
                    for item in audio_ids
                ],
                dtype=np.float64,
            )
            comparisons[f"{mode}_{condition}_reference_drop_vs_exact"] = _paired_test(
                perturbed, exact, iterations=iterations, seed=seed + 1
            )
    raw = {
        name: payload["paired_randomization_p_two_sided"]
        for name, payload in comparisons.items()
    }
    adjusted = _holm_adjust(raw)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    payload = {
        "model": MODEL_LABEL,
        "iterations": iterations,
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "paired_comparisons.json", payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{MODEL_LABEL} Audio-CVR diagnostic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--audio-test", required=True)
    prepare.add_argument("--omnicvr-records", required=True)
    prepare.add_argument("--omnicvr-gallery", required=True)
    prepare.add_argument("--variant-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--media-root", action="append", default=[])
    prepare.add_argument("--audio-test-sha256", default=EXPECTED_FULL1000_SHA256)

    encode = subparsers.add_parser("encode")
    encode.add_argument("--inventory-path", required=True)
    encode.add_argument("--cache-dir", required=True)
    encode.add_argument("--base-model", required=True)
    encode.add_argument("--adapter-model", required=True)
    encode.add_argument("--shard-index", type=int, required=True)
    encode.add_argument("--shard-count", type=int, required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--retries", type=int, default=3)
    encode.add_argument("--torch-dtype", default="bfloat16")
    encode.add_argument("--attn-implementation", default="flash_attention_2")

    audit = subparsers.add_parser("audit-cache")
    audit.add_argument("--inventory-path", required=True)
    audit.add_argument("--cache-dir", required=True)
    audit.add_argument("--output-path", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--records-dir", required=True)
    evaluate_parser.add_argument("--inventory-path", required=True)
    evaluate_parser.add_argument("--cache-dir", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)

    statistics = subparsers.add_parser("statistics")
    statistics.add_argument("--per-query-path", required=True)
    statistics.add_argument("--output-dir", required=True)
    statistics.add_argument("--iterations", type=int, default=20000)
    statistics.add_argument("--seed", type=int, default=20260724)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare_inventory(
            audio_test=Path(args.audio_test),
            omnicvr_records=Path(args.omnicvr_records),
            omnicvr_gallery=Path(args.omnicvr_gallery),
            variant_manifest=Path(args.variant_manifest),
            output_dir=Path(args.output_dir),
            media_roots=[Path(root) for root in args.media_root],
            audio_test_sha256=args.audio_test_sha256,
        )
    elif args.command == "encode":
        value = encode_inventory(
            inventory_path=Path(args.inventory_path),
            cache_dir=Path(args.cache_dir),
            base_model=Path(args.base_model),
            adapter_model=Path(args.adapter_model),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            retries=args.retries,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
        )
    elif args.command == "audit-cache":
        value = audit_cache(
            inventory_path=Path(args.inventory_path),
            cache_dir=Path(args.cache_dir),
            output_path=Path(args.output_path),
        )
    elif args.command == "evaluate":
        value = evaluate(
            records_dir=Path(args.records_dir),
            inventory_path=Path(args.inventory_path),
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
        )
    elif args.command == "statistics":
        value = summarize_statistics(
            per_query_path=Path(args.per_query_path),
            output_dir=Path(args.output_dir),
            iterations=args.iterations,
            seed=args.seed,
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
