from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Callable

import numpy as np

from app.e5_cvr_eval import (
    DEFAULT_E5_MODEL,
    DEFAULT_VIDEO_MAX_PIXELS,
    VIDEO_AUDIO_MODE_ON,
    _normalize_rows,
    load_e5_encoder,
)


DEFAULT_RUNS_ROOT = "/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs"
DEFAULT_DATA_ROOT = "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval"
QUERY_TEMPLATE = "Edit the reference video so that: {edit_text}"
DEFAULT_NEGATIVE_TYPES = ("reference_negative", "visual_hard", "audio_hard", "asr_hard")
DEFAULT_LOSS_OPTIONS = {
    "training_profile": "v1",
    "contrastive_objective": "ce",
    "dcl_debias_prob": 0.1,
    "dcl_negative_floor": 1e-6,
    "disable_delta_loss": False,
    "disable_hard_negatives": False,
    "disable_reference_negative": False,
    "disable_edit_type_loss": False,
    "disable_local_segments": False,
    "disable_global_local_mix": False,
    "local_mix_weight": 0.5,
    "curriculum_stage": 4,
    "enable_hardness_weighting": False,
    "hardness_temperature": 0.07,
    "hardness_weight_min": 0.25,
    "hardness_weight_max": 4.0,
    "enable_multi_positive": False,
    "enable_coral_align": False,
    "enable_memory_bank": False,
    "enable_false_negative_filtering": False,
    "enable_modality_temperature": False,
    "modality_temperature_init": 0.05,
    "modality_temperature_min": 0.005,
    "modality_temperature_max": 0.2,
    "enable_quantile_negative_curriculum": False,
    "negative_keep_ratio_start": 1.0,
    "negative_keep_ratio_end": 0.5,
    "negative_curriculum_warmup_ratio": 0.1,
    "easy_negative_weight": 0.1,
    "enable_batch_whitening": False,
    "temperature": 0.05,
    "lambda_delta": 0.5,
    "lambda_hn": 0.5,
    "lambda_ref": 0.3,
    "lambda_edit_type": 0.3,
    "lambda_visual": 0.05,
    "lambda_hw_hn": 0.0,
    "lambda_multi_positive": 0.0,
    "lambda_coral_align": 0.0,
    "lambda_memory_bank": 0.0,
    "lambda_batch_whitening": 0.0,
    "false_negative_sim_threshold": 0.92,
    "false_negative_soft_weight": 0.15,
}
NEGATIVE_CURRICULUM_STAGE = {
    "reference_negative": 1,
    "visual_hard": 2,
    "audio_hard": 3,
    "asr_hard": 4,
}


@dataclass(frozen=True)
class AudioDeltaRecord:
    sample_id: str
    reference_video: str
    target_video: str
    edit_text: str
    edit_type: str
    audio_delta_type: str
    old_audio: str
    new_audio: str
    direction: str
    split_tier: str
    raw_source_id: str
    pair_group_id: str
    inverse_pair_group_id: str
    shortcut_label: str
    audio_delta_strength: float
    video_context_strength: float
    asr_degeneracy_risk: float
    visual_shortcut_risk: float
    full_av_required: bool
    hard_negatives: tuple[dict[str, str], ...]
    source_payload: dict[str, Any]


@dataclass(frozen=True)
class EvalGalleryItem:
    gallery_id: str
    video: str
    raw_source_id: str
    kind: str
    source_payload: dict[str, Any]


class DeterministicEncoder:
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode_document(self, inputs: list[Any]) -> np.ndarray:
        rows = [_hash_embedding(item, dim=self.dim) for item in inputs]
        return _normalize_rows(np.asarray(rows, dtype=np.float32))


def prepare_records(
    *,
    run_root: str | Path,
    output_dir: str | Path | None = None,
    train_paths: list[str | Path] | None = None,
    eval_paths: list[str | Path] | None = None,
    max_train_records: int = 8,
    max_eval_records: int = 4,
    eval_gallery_size: int = 0,
    distractor_pool_path: str | Path | None = None,
    distractor_seed: int = 13,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    dataset_root = Path(run_root)
    output_root = Path(output_dir) if output_dir else dataset_root / "e5_audio_delta_records"
    output_root.mkdir(parents=True, exist_ok=True)
    train_paths = train_paths or _default_train_paths(dataset_root)
    eval_paths = eval_paths or _default_eval_paths(dataset_root)
    train_records = _dedupe_records(_load_records_from_paths(train_paths))
    eval_records = _dedupe_records(_load_records_from_paths(eval_paths))
    if not train_records:
        raise ValueError(f"no training records found from: {[str(path) for path in train_paths]}")
    if not eval_records:
        eval_records = train_records[:]
    if max_train_records > 0:
        train_records = train_records[:max_train_records]
    if max_eval_records > 0:
        eval_records = eval_records[:max_eval_records]
    train_path = output_root / "train.jsonl"
    eval_path = output_root / "eval.jsonl"
    _write_jsonl(train_path, [asdict(record) for record in train_records])
    _write_jsonl(eval_path, [asdict(record) for record in eval_records])
    gallery_summary: dict[str, Any] | None = None
    gallery_path = output_root / "eval_gallery.jsonl"
    if eval_gallery_size and eval_gallery_size > 0:
        gallery_items, positive_indices, gallery_summary = _build_eval_gallery(
            dataset_root=dataset_root,
            train_records=train_records,
            eval_records=eval_records,
            total_gallery_size=eval_gallery_size,
            distractor_pool_path=distractor_pool_path,
            seed=distractor_seed,
        )
        _write_jsonl(gallery_path, [asdict(item) for item in gallery_items])
        (output_root / "eval_gallery_positive_indices.json").write_text(
            json.dumps({"positive_gallery_index": positive_indices}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "dataset_run_root": str(dataset_root),
        "output_dir": str(output_root),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "train_paths": [str(path) for path in train_paths],
        "eval_paths": [str(path) for path in eval_paths],
        "outputs": {
            "train": str(train_path),
            "eval": str(eval_path),
            "eval_gallery": str(gallery_path) if gallery_summary else None,
            "eval_gallery_positive_indices": str(output_root / "eval_gallery_positive_indices.json") if gallery_summary else None,
        },
        "eval_gallery": gallery_summary,
        "eval_protocol": "pilot_only_random_distractor_gallery" if gallery_summary else "default_aligned_eval_targets",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(progress, f"[e5-audio-delta] prepared train={len(train_records)} eval={len(eval_records)} records_dir={output_root}")
    return summary


def cache_embeddings(
    *,
    records_dir: str | Path,
    output_dir: str | Path,
    encoder: Any | None = None,
    mock_encoder: bool = False,
    e5_model: str = DEFAULT_E5_MODEL,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    batch_size: int = 1,
    video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    video_fps: int = 1,
    local_segments: int = 0,
    local_segment_mode: str = "prompt",
    local_segment_cache_dir: str | Path | None = None,
    segment_overlap: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    records_root = Path(records_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        if mock_encoder:
            encoder = DeterministicEncoder()
            runtime_info: dict[str, Any] = {"model_path": "mock-deterministic", "dim": encoder.dim, "video_audio_mode": VIDEO_AUDIO_MODE_ON}
        else:
            encoder, info = load_e5_encoder(
                model_path=e5_model,
                device=device,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                batch_size=batch_size,
                video_max_pixels=video_max_pixels,
                video_fps=video_fps,
                video_audio_mode=VIDEO_AUDIO_MODE_ON,
            )
            runtime_info = asdict(info)
    else:
        runtime_info = {"model_path": "injected-encoder", "video_audio_mode": VIDEO_AUDIO_MODE_ON}
    train_records = load_audio_delta_records(records_root / "train.jsonl")
    eval_records = load_audio_delta_records(records_root / "eval.jsonl")
    eval_gallery = load_eval_gallery_items(records_root / "eval_gallery.jsonl") if (records_root / "eval_gallery.jsonl").exists() else []
    train_summary = _cache_split_embeddings(
        records=train_records,
        split="train",
        encoder=encoder,
        output_root=output_root,
        runtime_info=runtime_info,
        local_segments=local_segments,
        local_segment_mode=local_segment_mode,
        local_segment_cache_dir=local_segment_cache_dir,
        segment_overlap=segment_overlap,
        progress=progress,
    )
    eval_summary = _cache_split_embeddings(
        records=eval_records,
        split="eval",
        eval_gallery=eval_gallery,
        encoder=encoder,
        output_root=output_root,
        runtime_info=runtime_info,
        local_segments=local_segments,
        local_segment_mode=local_segment_mode,
        local_segment_cache_dir=local_segment_cache_dir,
        segment_overlap=segment_overlap,
        progress=progress,
    )
    summary = {
        "records_dir": str(records_root),
        "output_dir": str(output_root),
        "runtime": runtime_info,
        "local_segments": local_segments,
        "local_segment_mode": local_segment_mode,
        "local_segment_cache_dir": str(local_segment_cache_dir) if local_segment_cache_dir else None,
        "segment_overlap": segment_overlap,
        "train": train_summary,
        "eval": eval_summary,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def train_adapter(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    steps: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 13,
    device: str = "cuda",
    training_profile: str = "v1",
    contrastive_objective: str | None = None,
    dcl_debias_prob: float = 0.1,
    dcl_negative_floor: float = 1e-6,
    disable_delta_loss: bool = False,
    disable_hard_negatives: bool = False,
    disable_reference_negative: bool = False,
    disable_edit_type_loss: bool = False,
    disable_local_segments: bool = False,
    disable_global_local_mix: bool = False,
    local_mix_weight: float = 0.5,
    curriculum_stage: int = 4,
    enable_hardness_weighting: bool | None = None,
    hardness_temperature: float = 0.07,
    hardness_weight_min: float = 0.25,
    hardness_weight_max: float = 4.0,
    enable_multi_positive: bool | None = None,
    enable_coral_align: bool | None = None,
    enable_memory_bank: bool | None = None,
    enable_false_negative_filtering: bool | None = None,
    enable_modality_temperature: bool | None = None,
    modality_temperature_init: float = 0.05,
    modality_temperature_min: float = 0.005,
    modality_temperature_max: float = 0.2,
    enable_quantile_negative_curriculum: bool | None = None,
    negative_keep_ratio_start: float = 1.0,
    negative_keep_ratio_end: float = 0.5,
    negative_curriculum_warmup_ratio: float = 0.1,
    easy_negative_weight: float = 0.1,
    enable_batch_whitening: bool | None = None,
    lambda_hw_hn: float | None = None,
    lambda_multi_positive: float | None = None,
    lambda_coral_align: float | None = None,
    lambda_memory_bank: float | None = None,
    lambda_batch_whitening: float | None = None,
    memory_bank_size: int = 4096,
    warmup_ratio: float = 0.05,
    min_learning_rate_ratio: float = 0.1,
    temperature_start: float = 0.07,
    temperature_end: float = 0.03,
    false_negative_sim_threshold: float = 0.92,
    false_negative_soft_weight: float = 0.15,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    torch = _import_torch()
    cache_root = Path(cache_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data = _load_embedding_npz(cache_root / "train_embeddings.npz")
    records = load_audio_delta_records(cache_root / "train_records.jsonl")
    if not records:
        raise ValueError("train records are empty")
    dim = int(data["query"].shape[1])
    device_obj = _torch_device(torch, device)
    torch.manual_seed(seed)
    model = _AudioDeltaAdapter(torch, dim, modality_temperature_init=modality_temperature_init).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device_obj) for key, value in data.items()}
    count = int(tensors["query"].shape[0])
    rng = random.Random(seed)
    profile_options = _training_profile_options(
        training_profile=training_profile,
        enable_hardness_weighting=enable_hardness_weighting,
        enable_multi_positive=enable_multi_positive,
        enable_coral_align=enable_coral_align,
        enable_memory_bank=enable_memory_bank,
        enable_false_negative_filtering=enable_false_negative_filtering,
        enable_modality_temperature=enable_modality_temperature,
        enable_quantile_negative_curriculum=enable_quantile_negative_curriculum,
        enable_batch_whitening=enable_batch_whitening,
        lambda_hw_hn=lambda_hw_hn,
        lambda_multi_positive=lambda_multi_positive,
        lambda_coral_align=lambda_coral_align,
        lambda_memory_bank=lambda_memory_bank,
        lambda_batch_whitening=lambda_batch_whitening,
    )
    loss_options = _loss_options(
        training_profile=training_profile,
        dcl_debias_prob=dcl_debias_prob,
        dcl_negative_floor=dcl_negative_floor,
        disable_delta_loss=disable_delta_loss,
        disable_hard_negatives=disable_hard_negatives,
        disable_reference_negative=disable_reference_negative,
        disable_edit_type_loss=disable_edit_type_loss,
        disable_local_segments=disable_local_segments,
        disable_global_local_mix=disable_global_local_mix,
        local_mix_weight=local_mix_weight,
        curriculum_stage=curriculum_stage,
        hardness_temperature=hardness_temperature,
        hardness_weight_min=hardness_weight_min,
        hardness_weight_max=hardness_weight_max,
        false_negative_sim_threshold=false_negative_sim_threshold,
        false_negative_soft_weight=false_negative_soft_weight,
        modality_temperature_init=modality_temperature_init,
        modality_temperature_min=modality_temperature_min,
        modality_temperature_max=modality_temperature_max,
        negative_keep_ratio_start=negative_keep_ratio_start,
        negative_keep_ratio_end=negative_keep_ratio_end,
        negative_curriculum_warmup_ratio=negative_curriculum_warmup_ratio,
        easy_negative_weight=easy_negative_weight,
        **profile_options,
    )
    if contrastive_objective is not None:
        loss_options["contrastive_objective"] = str(contrastive_objective)
    if str(training_profile or "v1") in {"v2_research", "e5_omni_recipe"}:
        loss_options["disable_local_segments"] = True
        loss_options["disable_global_local_mix"] = True
    max_steps = max(1, steps)
    warmup_steps = int(max_steps * max(0.0, warmup_ratio))
    memory_bank: list[Any] = []
    losses_path = output_root / "loss_curve.jsonl"
    with losses_path.open("w", encoding="utf-8") as losses_file:
        for step in range(1, max_steps + 1):
            indices = [rng.randrange(count) for _ in range(min(max(1, batch_size), count))]
            batch = {key: value[indices] if value.shape[0] == count else value for key, value in tensors.items()}
            batch_records = [records[index] for index in indices]
            lr = _scheduled_learning_rate(
                base_lr=learning_rate,
                step=step,
                total_steps=max_steps,
                warmup_steps=warmup_steps,
                min_ratio=min_learning_rate_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            temperature = _scheduled_temperature(
                step=step,
                total_steps=max_steps,
                start=temperature_start if training_profile == "v2_research" else 0.05,
                end=temperature_end if training_profile == "v2_research" else 0.05,
            )
            step_loss_options = dict(loss_options)
            step_loss_options["temperature"] = temperature
            step_loss_options["current_step"] = step
            step_loss_options["total_steps"] = max_steps
            if loss_options["enable_memory_bank"] and step > warmup_steps and memory_bank:
                step_loss_options["memory_bank"] = torch.cat(memory_bank, dim=0)
            else:
                step_loss_options["memory_bank"] = None
            optimizer.zero_grad(set_to_none=True)
            losses = _adapter_losses(torch, model, batch, batch_records, step_loss_options)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite adapter loss at step {step}: {float(loss.detach().cpu())}")
            loss.backward()
            optimizer.step()
            if loss_options["enable_memory_bank"] and memory_bank_size > 0:
                with torch.no_grad():
                    projected = model.doc(batch["target"]).detach()
                    memory_bank.append(projected)
                    bank_tensor = torch.cat(memory_bank, dim=0)
                    if bank_tensor.shape[0] > memory_bank_size:
                        bank_tensor = bank_tensor[-memory_bank_size:]
                    memory_bank = [bank_tensor]
            row = {
                "step": step,
                "lr": round(float(lr), 10),
                "temperature": round(float(temperature), 6),
                "memory_bank_size": int(torch.cat(memory_bank, dim=0).shape[0]) if memory_bank else 0,
                **{name: round(float(value.detach().cpu()), 6) for name, value in losses.items()},
            }
            losses_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            losses_file.flush()
            _emit(progress, f"[e5-audio-delta] train step={step}/{steps} loss={row['total']:.6f}")
    adapter_path = output_root / "adapter.pt"
    torch.save(model.state_dict(), adapter_path)
    config = {
        "dim": dim,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device_obj),
        "training_profile": training_profile,
        "loss_options": loss_options,
        "schedule": {
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "min_learning_rate_ratio": min_learning_rate_ratio,
            "temperature_start": temperature_start,
            "temperature_end": temperature_end,
            "memory_bank_size": memory_bank_size,
        },
        "losses_path": str(losses_path),
        "adapter_path": str(adapter_path),
    }
    (output_root / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"cache_dir": str(cache_root), "output_dir": str(output_root), **config}
    (output_root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def eval_adapter(
    *,
    cache_dir: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    topk: tuple[int, ...] = (1, 5, 10),
    device: str = "cuda",
    disable_local_segments: bool = False,
    disable_global_local_mix: bool = False,
    local_mix_weight: float = 0.5,
) -> dict[str, Any]:
    torch = _import_torch()
    cache_root = Path(cache_dir)
    adapter_root = Path(adapter_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data = _load_embedding_npz(cache_root / "eval_embeddings.npz")
    records = load_audio_delta_records(cache_root / "eval_records.jsonl")
    if not records:
        raise ValueError("eval records are empty")
    positive_gallery_index = np.asarray(data["positive_gallery_index"], dtype=np.int64) if "positive_gallery_index" in data else np.arange(len(records), dtype=np.int64)
    topk = _normalize_topk(topk)
    gallery = data["gallery"] if "gallery" in data else data["target"]
    base_scores = _score_matrix_np(data["query"], gallery)
    base = _recall_from_scores(base_scores, topk=topk, positive_index=positive_gallery_index)
    has_local = _has_local_segments(data) and not disable_local_segments
    gallery_segments = data["gallery_segments"] if "gallery_segments" in data else data["target_segments"]
    base_local_scores = _local_score_matrix_np(data["query"], gallery_segments) if has_local else None
    base_mix_scores = _mix_scores(base_scores, base_local_scores, local_mix_weight, disable_global_local_mix)
    dim = int(data["query"].shape[1])
    device_obj = _torch_device(torch, device)
    model = _AudioDeltaAdapter(torch, dim).to(device_obj)
    state = torch.load(adapter_root / "adapter.pt", map_location=device_obj)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        query = torch.as_tensor(data["query"], dtype=torch.float32, device=device_obj)
        target = torch.as_tensor(gallery, dtype=torch.float32, device=device_obj)
        reference = torch.as_tensor(data["reference"], dtype=torch.float32, device=device_obj)
        negative = torch.as_tensor(data["negative"], dtype=torch.float32, device=device_obj)
        adapted_query = model.query(query)
        adapted_target = model.doc(target)
        adapted_reference = model.doc(reference)
        adapted_negative = model.doc(negative)
        adapted_scores = (adapted_query @ adapted_target.T).detach().cpu().numpy()
        adapted_reference_scores = torch.sum(adapted_query * adapted_reference, dim=-1).detach().cpu().numpy()
        adapted_negative_scores = torch.einsum("bd,bnd->bn", adapted_query, adapted_negative).detach().cpu().numpy()
        adapted_local_scores = None
        if has_local:
            target_segments = model.doc(torch.as_tensor(gallery_segments, dtype=torch.float32, device=device_obj))
            adapted_local_scores = _local_score_matrix_torch(torch, adapted_query, target_segments).detach().cpu().numpy()
        adapted_mix_scores = _mix_scores(adapted_scores, adapted_local_scores, local_mix_weight, disable_global_local_mix)
    adapted = _recall_from_scores(adapted_scores, topk=topk, positive_index=positive_gallery_index)
    rows = [
        {"method": "base_e5_global", **base},
        {"method": "audio_delta_adapter_global", **adapted},
    ]
    if has_local and base_local_scores is not None and adapted_local_scores is not None:
        rows.extend(
            [
                {"method": "base_e5_local", **_recall_from_scores(base_local_scores, topk=topk, positive_index=positive_gallery_index)},
                {"method": "base_e5_global_local", **_recall_from_scores(base_mix_scores, topk=topk, positive_index=positive_gallery_index)},
                {"method": "audio_delta_adapter_local", **_recall_from_scores(adapted_local_scores, topk=topk, positive_index=positive_gallery_index)},
                {"method": "audio_delta_adapter_global_local", **_recall_from_scores(adapted_mix_scores, topk=topk, positive_index=positive_gallery_index)},
            ]
        )
    comparison = {
        "cache_dir": str(cache_root),
        "adapter_dir": str(adapter_root),
        "output_dir": str(output_root),
        "eval_count": len(records),
        "gallery_count": int(gallery.shape[0]),
        "topk": list(topk),
        "has_local_segments": has_local,
        "local_mix_weight": local_mix_weight,
        "rows": rows,
        "by_split_tier": _grouped_recall_summary(base_mix_scores, adapted_mix_scores, records, "split_tier", topk, positive_index=positive_gallery_index),
        "by_audio_delta_type": _grouped_recall_summary(base_mix_scores, adapted_mix_scores, records, "audio_delta_type", topk, positive_index=positive_gallery_index),
        "by_shortcut_label": _grouped_recall_summary(base_mix_scores, adapted_mix_scores, records, "shortcut_label", topk, positive_index=positive_gallery_index),
        "reference_rank_summary": _reference_rank_summary(adapted_mix_scores, adapted_reference_scores),
        "delta_score_distribution": _delta_score_distribution(adapted_mix_scores, adapted_reference_scores, positive_index=positive_gallery_index),
        "hard_negative_recall_by_type": _hard_negative_recall_by_type(adapted_scores, adapted_negative_scores, records, positive_index=positive_gallery_index),
        "diagnostics_path": str(output_root / "diagnostics.json"),
    }
    (output_root / "summary.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics = {
        "reference_rank_summary": comparison["reference_rank_summary"],
        "delta_score_distribution": comparison["delta_score_distribution"],
        "hard_negative_recall_by_type": comparison["hard_negative_recall_by_type"],
        "by_shortcut_label": comparison["by_shortcut_label"],
        "by_split_tier": comparison["by_split_tier"],
    }
    (output_root / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def train_lora_plan(*, output_dir: str | Path) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "dry_run_only",
        "reason": "LoRA is V1 optional; V0 adapter smoke must pass first.",
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "default_lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
        "recommended_launcher": "accelerate launch --num_processes 4 --gpu_ids 0,1,2,3",
    }
    (output_root / "lora_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def build_splits(
    *,
    run_root: str | Path,
    output_dir: str | Path | None = None,
    input_paths: list[str | Path] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 13,
) -> dict[str, Any]:
    dataset_root = Path(run_root)
    output_root = Path(output_dir) if output_dir else dataset_root / "b_splits"
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in input_paths] if input_paths else _default_train_paths(dataset_root)
    records = _dedupe_records(_load_records_from_paths(paths))
    if not records:
        raise ValueError(f"no records found for split building from: {[str(path) for path in paths]}")
    diagnostic = [record for record in records if _is_diagnostic_record(record)]
    eligible = [record for record in records if record not in diagnostic]
    groups: dict[str, list[AudioDeltaRecord]] = defaultdict(list)
    for record in eligible:
        groups[_split_group_id(record)].append(record)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    train_cut = int(len(group_ids) * train_ratio)
    val_cut = train_cut + int(len(group_ids) * val_ratio)
    split_group_ids = {
        "train": set(group_ids[:train_cut]),
        "val": set(group_ids[train_cut:val_cut]),
        "test": set(group_ids[val_cut:]),
    }
    split_records = {
        name: [record for group_id in sorted(group_ids_for_split) for record in groups[group_id]]
        for name, group_ids_for_split in split_group_ids.items()
    }
    test_main = _one_direction_per_pair([record for record in split_records["test"] if record.split_tier == "main" and record.direction != "inverse"])
    test_inverse = [record for record in split_records["test"] if record.direction == "inverse"]
    outputs = {
        "train": output_root / "train.jsonl",
        "val": output_root / "val.jsonl",
        "test_main": output_root / "test_main.jsonl",
        "test_inverse_diagnostic": output_root / "test_inverse_diagnostic.jsonl",
        "diagnostic": output_root / "diagnostic.jsonl",
    }
    _write_jsonl(outputs["train"], [asdict(record) for record in split_records["train"]])
    _write_jsonl(outputs["val"], [asdict(record) for record in split_records["val"]])
    _write_jsonl(outputs["test_main"], [asdict(record) for record in test_main])
    _write_jsonl(outputs["test_inverse_diagnostic"], [asdict(record) for record in test_inverse])
    _write_jsonl(outputs["diagnostic"], [asdict(record) for record in diagnostic])
    summary = {
        "dataset_run_root": str(dataset_root),
        "output_dir": str(output_root),
        "input_paths": [str(path) for path in paths],
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "source_disjoint_group_count": len(group_ids),
        "counts": {
            "all": len(records),
            "train": len(split_records["train"]),
            "val": len(split_records["val"]),
            "test_pool": len(split_records["test"]),
            "test_main": len(test_main),
            "test_inverse_diagnostic": len(test_inverse),
            "diagnostic": len(diagnostic),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "leakage_checks": _split_leakage_checks(split_records, test_main, test_inverse, diagnostic),
    }
    (output_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_ablations(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    steps: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    seed: int = 13,
    training_profile: str = "v2_research",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    cache_root = Path(cache_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if training_profile in {"v2_research", "e5_omni_recipe"}:
        configs = [
            ("full_v2", {"training_profile": "v2_research"}),
            ("without_modality_temperature", {"training_profile": "v2_research", "enable_modality_temperature": False}),
            ("without_quantile_negative_curriculum", {"training_profile": "v2_research", "enable_quantile_negative_curriculum": False}),
            ("without_false_negative_debiasing", {"training_profile": "v2_research", "enable_false_negative_filtering": False}),
            ("without_hardness_weighting", {"training_profile": "v2_research", "enable_hardness_weighting": False, "lambda_hw_hn": 0.0}),
            ("without_multi_positive", {"training_profile": "v2_research", "enable_multi_positive": False, "lambda_multi_positive": 0.0}),
            ("without_coral_align", {"training_profile": "v2_research", "enable_coral_align": False, "lambda_coral_align": 0.0}),
            ("without_batch_whitening", {"training_profile": "v2_research", "enable_batch_whitening": False, "lambda_batch_whitening": 0.0}),
            ("without_memory_bank", {"training_profile": "v2_research", "enable_memory_bank": False, "lambda_memory_bank": 0.0}),
            ("without_false_negative_filtering", {"training_profile": "v2_research", "enable_false_negative_filtering": False}),
            ("without_local_segments", {"training_profile": "v2_research", "disable_local_segments": True}),
            ("without_delta", {"training_profile": "v2_research", "disable_delta_loss": True}),
            ("without_reference_negative", {"training_profile": "v2_research", "disable_reference_negative": True}),
            ("without_hard_negatives", {"training_profile": "v2_research", "disable_hard_negatives": True}),
            (
                "v1_loss_only",
                {
                    "training_profile": "v1",
                    "enable_hardness_weighting": False,
                    "enable_multi_positive": False,
                    "enable_coral_align": False,
                    "enable_memory_bank": False,
                    "enable_false_negative_filtering": False,
                    "enable_modality_temperature": False,
                    "enable_quantile_negative_curriculum": False,
                    "enable_batch_whitening": False,
                },
            ),
        ]
    else:
        configs = [
            ("full", {"training_profile": "v1"}),
            ("without_delta", {"disable_delta_loss": True}),
            ("without_hard_negatives", {"disable_hard_negatives": True}),
            ("without_reference_negative", {"disable_reference_negative": True}),
            ("without_edit_type", {"disable_edit_type_loss": True}),
            ("without_local_segments", {"disable_local_segments": True}),
            ("global_only", {"disable_local_segments": True, "disable_global_local_mix": True}),
        ]
    rows: list[dict[str, Any]] = []
    for name, overrides in configs:
        _emit(progress, f"[e5-audio-delta] ablation {name} start")
        adapter_dir = output_root / name / "adapter"
        eval_dir = output_root / name / "eval"
        train_summary = train_adapter(
            cache_dir=cache_root,
            output_dir=adapter_dir,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            progress=progress,
            **overrides,
        )
        eval_summary = eval_adapter(
            cache_dir=cache_root,
            adapter_dir=adapter_dir,
            output_dir=eval_dir,
            device=device,
            disable_local_segments=bool(overrides.get("disable_local_segments", False)),
            disable_global_local_mix=bool(overrides.get("disable_global_local_mix", False)),
        )
        adapted_row = next((row for row in eval_summary["rows"] if row["method"] == "audio_delta_adapter_global_local"), None)
        if adapted_row is None:
            adapted_row = next((row for row in eval_summary["rows"] if row["method"] == "audio_delta_adapter_global"), {})
        loss_tail = _last_jsonl_row(adapter_dir / "loss_curve.jsonl")
        diagnostics = eval_summary.get("diagnostics_path")
        rows.append(
            {
                "ablation": name,
                "adapter_dir": str(adapter_dir),
                "eval_dir": str(eval_dir),
                "steps": train_summary["steps"],
                **adapted_row,
                "reference_negative_average_rank": (eval_summary.get("reference_rank_summary") or {}).get("mean_rank"),
                "delta_score_pos_mean": (eval_summary.get("delta_score_distribution") or {}).get("mean"),
                "delta_score_neg_mean": 0.0,
                "effective_negative_count": loss_tail.get("effective_negative_count"),
                "tau_text": loss_tail.get("tau_text"),
                "tau_audio": loss_tail.get("tau_audio"),
                "tau_video": loss_tail.get("tau_video"),
                "diagnostics_path": diagnostics,
            }
        )
        _emit(progress, f"[e5-audio-delta] ablation {name} done")
    summary = {"cache_dir": str(cache_root), "output_dir": str(output_root), "rows": rows}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "comparison.md").write_text(_ablation_markdown(summary), encoding="utf-8")
    return summary


def load_audio_delta_records(path: str | Path) -> list[AudioDeltaRecord]:
    root = Path(path)
    if not root.exists():
        return []
    return [_record_from_payload(json.loads(line), line_number=index) for index, line in enumerate(root.read_text(encoding="utf-8-sig").splitlines(), start=1) if line.strip()]


def load_eval_gallery_items(path: str | Path) -> list[EvalGalleryItem]:
    root = Path(path)
    if not root.exists():
        return []
    items: list[EvalGalleryItem] = []
    for index, line in enumerate(root.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{root} line {index}: expected a JSON object")
        gallery_id = _first_text(payload, "gallery_id", default=f"gallery_{index:06d}")
        video = _first_text(payload, "video", "target_video", "output_path", "path")
        if not video:
            raise ValueError(f"{root} line {index}: missing gallery video path")
        items.append(
            EvalGalleryItem(
                gallery_id=gallery_id,
                video=video,
                raw_source_id=_first_text(payload, "raw_source_id", "source_clip_id", "group_id", default=gallery_id),
                kind=_first_text(payload, "kind", default="distractor"),
                source_payload=dict(payload),
            )
        )
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small AudioDelta adapter on e5-omni embeddings")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-run-root", "--run-root", dest="dataset_run_root", required=True)
    prepare.add_argument("--output-dir")
    prepare.add_argument("--train-path", action="append", default=[])
    prepare.add_argument("--eval-path", action="append", default=[])
    prepare.add_argument("--max-train-records", type=int, default=8)
    prepare.add_argument("--max-eval-records", type=int, default=4)
    prepare.add_argument(
        "--eval-gallery-size",
        type=int,
        default=0,
        help="Pilot-only: expand eval gallery with random distractors for small-data recipe checks; not for final full-dataset benchmark.",
    )
    prepare.add_argument(
        "--distractor-pool-path",
        help="Pilot-only distractor pool JSONL. Defaults to the current run's annotation/segment manifests.",
    )
    prepare.add_argument("--distractor-seed", type=int, default=13)

    cache = subparsers.add_parser("cache-embeddings")
    cache.add_argument("--records-dir", required=True)
    cache.add_argument("--output-dir", required=True)
    cache.add_argument("--mock-encoder", action="store_true")
    cache.add_argument("--e5-model", default=DEFAULT_E5_MODEL)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--torch-dtype", default="bfloat16")
    cache.add_argument("--attn-implementation", default="flash_attention_2")
    cache.add_argument("--batch-size", type=int, default=1)
    cache.add_argument("--video-max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    cache.add_argument("--video-fps", type=int, default=1)
    cache.add_argument("--local-segments", type=int, default=0, help="Encode this many temporal local views per video; 0 disables local cache.")
    cache.add_argument("--local-segment-mode", choices=("prompt", "ffmpeg"), default="prompt")
    cache.add_argument("--local-segment-cache-dir")
    cache.add_argument("--segment-overlap", type=float, default=0.0)

    train = subparsers.add_parser("train-adapter")
    train.add_argument("--cache-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--steps", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="cuda")
    train.add_argument("--training-profile", choices=("v1", "v2_research", "e5_omni_recipe"), default="v1")
    train.add_argument("--contrastive-objective", choices=("ce", "masked_dcl"))
    train.add_argument("--dcl-debias-prob", type=float, default=0.1)
    train.add_argument("--dcl-negative-floor", type=float, default=1e-6)
    train.add_argument("--disable-delta-loss", action="store_true")
    train.add_argument("--disable-hard-negatives", action="store_true")
    train.add_argument("--disable-reference-negative", action="store_true")
    train.add_argument("--disable-edit-type-loss", action="store_true")
    train.add_argument("--disable-local-segments", action="store_true")
    train.add_argument("--disable-global-local-mix", action="store_true")
    train.add_argument("--local-mix-weight", type=float, default=0.5)
    train.add_argument("--curriculum-stage", type=int, default=4)
    train.add_argument("--enable-hardness-weighting", action="store_true", default=None)
    train.add_argument("--disable-hardness-weighting", action="store_false", dest="enable_hardness_weighting")
    train.add_argument("--hardness-temperature", type=float, default=0.07)
    train.add_argument("--hardness-weight-min", type=float, default=0.25)
    train.add_argument("--hardness-weight-max", type=float, default=4.0)
    train.add_argument("--enable-multi-positive", action="store_true", default=None)
    train.add_argument("--disable-multi-positive", action="store_false", dest="enable_multi_positive")
    train.add_argument("--enable-coral-align", action="store_true", default=None)
    train.add_argument("--disable-coral-align", action="store_false", dest="enable_coral_align")
    train.add_argument("--enable-memory-bank", action="store_true", default=None)
    train.add_argument("--disable-memory-bank", action="store_false", dest="enable_memory_bank")
    train.add_argument("--enable-false-negative-filtering", action="store_true", default=None)
    train.add_argument("--disable-false-negative-filtering", action="store_false", dest="enable_false_negative_filtering")
    train.add_argument("--enable-modality-temperature", action="store_true", default=None)
    train.add_argument("--disable-modality-temperature", action="store_false", dest="enable_modality_temperature")
    train.add_argument("--modality-temperature-init", type=float, default=0.05)
    train.add_argument("--modality-temperature-min", type=float, default=0.005)
    train.add_argument("--modality-temperature-max", type=float, default=0.2)
    train.add_argument("--enable-quantile-negative-curriculum", action="store_true", default=None)
    train.add_argument("--disable-quantile-negative-curriculum", action="store_false", dest="enable_quantile_negative_curriculum")
    train.add_argument("--negative-keep-ratio-start", type=float, default=1.0)
    train.add_argument("--negative-keep-ratio-end", type=float, default=0.5)
    train.add_argument("--negative-curriculum-warmup-ratio", type=float, default=0.1)
    train.add_argument("--easy-negative-weight", type=float, default=0.1)
    train.add_argument("--enable-batch-whitening", action="store_true", default=None)
    train.add_argument("--disable-batch-whitening", action="store_false", dest="enable_batch_whitening")
    train.add_argument("--lambda-hw-hn", type=float)
    train.add_argument("--lambda-multi-positive", type=float)
    train.add_argument("--lambda-coral-align", type=float)
    train.add_argument("--lambda-memory-bank", type=float)
    train.add_argument("--lambda-batch-whitening", type=float)
    train.add_argument("--memory-bank-size", type=int, default=4096)
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    train.add_argument("--temperature-start", type=float, default=0.07)
    train.add_argument("--temperature-end", type=float, default=0.03)
    train.add_argument("--false-negative-sim-threshold", type=float, default=0.92)
    train.add_argument("--false-negative-soft-weight", type=float, default=0.15)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--cache-dir", required=True)
    evaluate.add_argument("--adapter-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--topk", default="1,5,10")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--disable-local-segments", action="store_true")
    evaluate.add_argument("--disable-global-local-mix", action="store_true")
    evaluate.add_argument("--local-mix-weight", type=float, default=0.5)

    split = subparsers.add_parser("build-splits")
    split.add_argument("--dataset-run-root", "--run-root", dest="dataset_run_root", required=True)
    split.add_argument("--output-dir")
    split.add_argument("--input-path", action="append", default=[])
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=13)

    ablate = subparsers.add_parser("run-ablations")
    ablate.add_argument("--cache-dir", required=True)
    ablate.add_argument("--output-dir", required=True)
    ablate.add_argument("--steps", type=int, default=10)
    ablate.add_argument("--batch-size", type=int, default=4)
    ablate.add_argument("--learning-rate", type=float, default=1e-3)
    ablate.add_argument("--device", default="cuda")
    ablate.add_argument("--seed", type=int, default=13)
    ablate.add_argument("--training-profile", choices=("v1", "v2_research", "e5_omni_recipe"), default="v2_research")

    lora = subparsers.add_parser("train-lora")
    lora.add_argument("--output-dir", required=True)
    lora.add_argument("--dry-run", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    progress = lambda message: print(message, flush=True)
    if args.command == "prepare":
        result = prepare_records(
            run_root=args.dataset_run_root,
            output_dir=args.output_dir,
            train_paths=args.train_path or None,
            eval_paths=args.eval_path or None,
            max_train_records=args.max_train_records,
            max_eval_records=args.max_eval_records,
            eval_gallery_size=args.eval_gallery_size,
            distractor_pool_path=args.distractor_pool_path,
            distractor_seed=args.distractor_seed,
            progress=progress,
        )
    elif args.command == "cache-embeddings":
        result = cache_embeddings(
            records_dir=args.records_dir,
            output_dir=args.output_dir,
            mock_encoder=args.mock_encoder,
            e5_model=args.e5_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            batch_size=args.batch_size,
            video_max_pixels=args.video_max_pixels,
            video_fps=args.video_fps,
            local_segments=args.local_segments,
            local_segment_mode=args.local_segment_mode,
            local_segment_cache_dir=args.local_segment_cache_dir,
            segment_overlap=args.segment_overlap,
            progress=progress,
        )
    elif args.command == "train-adapter":
        result = train_adapter(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            training_profile=args.training_profile,
            contrastive_objective=args.contrastive_objective,
            dcl_debias_prob=args.dcl_debias_prob,
            dcl_negative_floor=args.dcl_negative_floor,
            disable_delta_loss=args.disable_delta_loss,
            disable_hard_negatives=args.disable_hard_negatives,
            disable_reference_negative=args.disable_reference_negative,
            disable_edit_type_loss=args.disable_edit_type_loss,
            disable_local_segments=args.disable_local_segments,
            disable_global_local_mix=args.disable_global_local_mix,
            local_mix_weight=args.local_mix_weight,
            curriculum_stage=args.curriculum_stage,
            enable_hardness_weighting=args.enable_hardness_weighting,
            hardness_temperature=args.hardness_temperature,
            hardness_weight_min=args.hardness_weight_min,
            hardness_weight_max=args.hardness_weight_max,
            enable_multi_positive=args.enable_multi_positive,
            enable_coral_align=args.enable_coral_align,
            enable_memory_bank=args.enable_memory_bank,
            enable_false_negative_filtering=args.enable_false_negative_filtering,
            enable_modality_temperature=args.enable_modality_temperature,
            modality_temperature_init=args.modality_temperature_init,
            modality_temperature_min=args.modality_temperature_min,
            modality_temperature_max=args.modality_temperature_max,
            enable_quantile_negative_curriculum=args.enable_quantile_negative_curriculum,
            negative_keep_ratio_start=args.negative_keep_ratio_start,
            negative_keep_ratio_end=args.negative_keep_ratio_end,
            negative_curriculum_warmup_ratio=args.negative_curriculum_warmup_ratio,
            easy_negative_weight=args.easy_negative_weight,
            enable_batch_whitening=args.enable_batch_whitening,
            lambda_hw_hn=args.lambda_hw_hn,
            lambda_multi_positive=args.lambda_multi_positive,
            lambda_coral_align=args.lambda_coral_align,
            lambda_memory_bank=args.lambda_memory_bank,
            lambda_batch_whitening=args.lambda_batch_whitening,
            memory_bank_size=args.memory_bank_size,
            warmup_ratio=args.warmup_ratio,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            false_negative_sim_threshold=args.false_negative_sim_threshold,
            false_negative_soft_weight=args.false_negative_soft_weight,
            progress=progress,
        )
    elif args.command == "eval":
        result = eval_adapter(
            cache_dir=args.cache_dir,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            topk=tuple(int(part.strip()) for part in str(args.topk).split(",") if part.strip()),
            device=args.device,
            disable_local_segments=args.disable_local_segments,
            disable_global_local_mix=args.disable_global_local_mix,
            local_mix_weight=args.local_mix_weight,
        )
    elif args.command == "build-splits":
        result = build_splits(
            run_root=args.dataset_run_root,
            output_dir=args.output_dir,
            input_paths=args.input_path or None,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif args.command == "run-ablations":
        result = run_ablations(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            training_profile=args.training_profile,
            progress=progress,
        )
    elif args.command == "train-lora":
        result = train_lora_plan(output_dir=args.output_dir)
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _cache_split_embeddings(
    *,
    records: list[AudioDeltaRecord],
    split: str,
    eval_gallery: list[EvalGalleryItem] | None = None,
    encoder: Any,
    output_root: Path,
    runtime_info: dict[str, Any],
    local_segments: int,
    local_segment_mode: str,
    local_segment_cache_dir: str | Path | None,
    segment_overlap: float,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"{split} records are empty")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("query", "target", "reference", "edit", "old_audio", "new_audio")}
    local_segments = max(0, int(local_segments))
    local_segment_mode = _normalize_local_segment_mode(local_segment_mode)
    local_cache_root = Path(local_segment_cache_dir) if local_segment_cache_dir else output_root / "local_media_cache"
    target_segment_rows: list[np.ndarray] = []
    reference_segment_rows: list[np.ndarray] = []
    negative_rows: list[list[np.ndarray]] = []
    negative_segment_rows: list[list[np.ndarray]] = []
    negative_mask: list[list[float]] = []
    negative_effective_mask: list[list[float]] = []
    negative_types: list[list[str]] = []
    positive_group_ids: list[str] = []
    gallery_items = list(eval_gallery or [])
    manifest_path = output_root / f"{split}_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index, record in enumerate(records, start=1):
            _emit(progress, f"[e5-audio-delta] cache {split} {index}/{len(records)} sample_id={record.sample_id}")
            arrays["query"].append(_encode_one(encoder, _query_payload(record)))
            arrays["target"].append(_encode_one(encoder, _video_payload(record.target_video)))
            arrays["reference"].append(_encode_one(encoder, _video_payload(record.reference_video)))
            arrays["edit"].append(_encode_one(encoder, record.edit_text))
            arrays["old_audio"].append(_encode_one(encoder, record.old_audio or record.edit_text))
            arrays["new_audio"].append(_encode_one(encoder, record.new_audio or record.edit_text))
            if local_segments > 0:
                target_segment_rows.append(
                    _encode_many(
                        encoder,
                        _local_video_payloads(
                            record.target_video,
                            role="target",
                            count=local_segments,
                            mode=local_segment_mode,
                            cache_root=local_cache_root,
                            sample_id=record.sample_id,
                            segment_overlap=segment_overlap,
                        ),
                    )
                )
                reference_segment_rows.append(
                    _encode_many(
                        encoder,
                        _local_video_payloads(
                            record.reference_video,
                            role="reference",
                            count=local_segments,
                            mode=local_segment_mode,
                            cache_root=local_cache_root,
                            sample_id=record.sample_id,
                            segment_overlap=segment_overlap,
                        ),
                    )
                )
            neg_vectors: list[np.ndarray] = []
            neg_segment_vectors: list[np.ndarray] = []
            neg_mask_row: list[float] = []
            neg_effective_row: list[float] = []
            neg_type_row: list[str] = []
            for negative in _ordered_negatives(record):
                video = str(negative.get("video", "")).strip()
                if not video:
                    continue
                neg_vectors.append(_encode_one(encoder, _video_payload(video)))
                if local_segments > 0:
                    neg_segment_vectors.append(
                        _encode_many(
                            encoder,
                            _local_video_payloads(
                                video,
                                role=str(negative.get("type", "negative")),
                                count=local_segments,
                                mode=local_segment_mode,
                                cache_root=local_cache_root,
                                sample_id=record.sample_id,
                                segment_overlap=segment_overlap,
                            ),
                        )
                    )
                neg_mask_row.append(1.0)
                neg_effective_row.append(_static_negative_effective_weight(record, negative))
                neg_type_row.append(str(negative.get("type", "")).strip() or "unknown")
            while len(neg_vectors) < len(DEFAULT_NEGATIVE_TYPES):
                neg_vectors.append(np.zeros_like(arrays["target"][-1]))
                if local_segments > 0:
                    neg_segment_vectors.append(np.zeros((local_segments, arrays["target"][-1].shape[0]), dtype=np.float32))
                neg_mask_row.append(0.0)
                neg_effective_row.append(0.0)
                neg_type_row.append("")
            negative_rows.append(neg_vectors[: len(DEFAULT_NEGATIVE_TYPES)])
            if local_segments > 0:
                negative_segment_rows.append(neg_segment_vectors[: len(DEFAULT_NEGATIVE_TYPES)])
            negative_mask.append(neg_mask_row[: len(DEFAULT_NEGATIVE_TYPES)])
            negative_effective_mask.append(neg_effective_row[: len(DEFAULT_NEGATIVE_TYPES)])
            negative_types.append(neg_type_row[: len(DEFAULT_NEGATIVE_TYPES)])
            positive_group_id = _positive_group_id(record)
            positive_group_ids.append(positive_group_id)
            manifest_file.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "positive_group_id": positive_group_id,
                        "negative_types": neg_type_row,
                        "negative_effective_mask": neg_effective_row,
                        "local_segment_mode": local_segment_mode,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest_file.flush()
    stacked = {key: np.vstack(value).astype(np.float32) for key, value in arrays.items()}
    stacked["negative"] = np.asarray(negative_rows, dtype=np.float32)
    if local_segments > 0:
        stacked["target_segments"] = np.asarray(target_segment_rows, dtype=np.float32)
        stacked["reference_segments"] = np.asarray(reference_segment_rows, dtype=np.float32)
        stacked["negative_segments"] = np.asarray(negative_segment_rows, dtype=np.float32)
    stacked["negative_mask"] = np.asarray(negative_mask, dtype=np.float32)
    stacked["negative_effective_mask"] = np.asarray(negative_effective_mask, dtype=np.float32)
    stacked["positive_group_index"] = np.asarray(_positive_group_indices(positive_group_ids), dtype=np.int64)
    if split == "eval" and gallery_items:
        gallery_vectors: list[np.ndarray] = []
        gallery_segment_rows: list[np.ndarray] = []
        positive_gallery_index: list[int] = []
        gallery_records_path = output_root / "eval_gallery.jsonl"
        _write_jsonl(gallery_records_path, [asdict(item) for item in gallery_items])
        gallery_lookup = {item.gallery_id: index for index, item in enumerate(gallery_items)}
        for item_index, item in enumerate(gallery_items, start=1):
            _emit(progress, f"[e5-audio-delta] cache eval gallery {item_index}/{len(gallery_items)} gallery_id={item.gallery_id}")
            gallery_vectors.append(_encode_one(encoder, _video_payload(item.video)))
            if local_segments > 0:
                gallery_segment_rows.append(
                    _encode_many(
                        encoder,
                        _local_video_payloads(
                            item.video,
                            role=item.kind or "gallery",
                            count=local_segments,
                            mode=local_segment_mode,
                            cache_root=local_cache_root,
                            sample_id=item.gallery_id,
                            segment_overlap=segment_overlap,
                        ),
                    )
                )
        for record in records:
            gallery_id = f"positive::{record.sample_id}"
            positive_gallery_index.append(int(gallery_lookup[gallery_id]))
        stacked["gallery"] = np.vstack(gallery_vectors).astype(np.float32)
        stacked["positive_gallery_index"] = np.asarray(positive_gallery_index, dtype=np.int64)
        if local_segments > 0:
            stacked["gallery_segments"] = np.asarray(gallery_segment_rows, dtype=np.float32)
    npz_path = output_root / f"{split}_embeddings.npz"
    np.savez(str(npz_path), **stacked)
    records_path = output_root / f"{split}_records.jsonl"
    _write_jsonl(records_path, [asdict(record) for record in records])
    metadata = {
        "split": split,
        "record_count": len(records),
        "embedding_shape": list(stacked["query"].shape),
        "negative_shape": list(stacked["negative"].shape),
        "gallery_shape": list(stacked["gallery"].shape) if "gallery" in stacked else None,
        "local_segments": local_segments,
        "local_segment_mode": local_segment_mode,
        "local_segment_cache_dir": str(local_cache_root) if local_segments > 0 and local_segment_mode == "ffmpeg" else None,
        "target_segments_shape": list(stacked["target_segments"].shape) if "target_segments" in stacked else None,
        "gallery_segments_shape": list(stacked["gallery_segments"].shape) if "gallery_segments" in stacked else None,
        "runtime": runtime_info,
        "embeddings_path": str(npz_path),
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
    }
    (output_root / f"{split}_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _adapter_losses(torch: Any, model: Any, batch: dict[str, Any], records: list[AudioDeltaRecord], options: dict[str, Any]) -> dict[str, Any]:
    query = model.query(batch["query"])
    target = model.doc(batch["target"])
    reference = model.doc(batch["reference"])
    edit = model.edit(batch["edit"])
    old_audio = model.edit(batch["old_audio"])
    new_audio = model.edit(batch["new_audio"])
    negative = model.doc(batch["negative"])
    neg_mask = batch["negative_mask"]
    has_local = _batch_has_local(batch) and not options["disable_local_segments"]
    target_segments = model.doc(batch["target_segments"]) if has_local else None
    reference_segments = model.doc(batch["reference_segments"]) if has_local else None
    negative_segments = model.doc(batch["negative_segments"]) if has_local else None
    use_modality_temperature = bool(options["enable_modality_temperature"])
    temperature = max(1e-6, float(options["temperature"]))
    tau_query = _modality_tau(torch, model, ("text", "audio", "video"), options, fallback=temperature, device=query.device)
    tau_target = _modality_tau(torch, model, ("audio", "video"), options, fallback=temperature, device=query.device)
    tau_text = _modality_tau(torch, model, ("text",), options, fallback=temperature, device=query.device)
    tau_audio = _modality_tau(torch, model, ("audio",), options, fallback=temperature, device=query.device)
    tau_video = _modality_tau(torch, model, ("video",), options, fallback=temperature, device=query.device)
    tau_cvr = _pair_tau(tau_query, tau_target)
    tau_audio_text = _pair_tau(tau_audio, tau_text)
    global_logits = query @ target.T
    local_logits = _local_score_matrix_torch(torch, query, target_segments) if target_segments is not None else None
    retrieval_scores = _mix_torch_scores(global_logits, local_logits, float(options["local_mix_weight"]), bool(options["disable_global_local_mix"]))
    retrieval_logits = retrieval_scores / tau_cvr
    labels = torch.arange(retrieval_logits.shape[0], device=retrieval_logits.device)
    loss_ce = torch.nn.functional.cross_entropy(retrieval_logits, labels)
    loss_cvr = loss_ce
    loss_multi_positive = _multi_positive_loss(torch, retrieval_logits, batch.get("positive_group_index")) if options["enable_multi_positive"] else torch.zeros((), device=retrieval_logits.device)
    pos_global = torch.sum(query * target, dim=-1)
    ref_global = torch.sum(query * reference, dim=-1)
    pos_local = _paired_local_scores_torch(torch, query, target_segments) if target_segments is not None else None
    ref_local = _paired_local_scores_torch(torch, query, reference_segments) if reference_segments is not None else None
    pos = _mix_torch_scores(pos_global, pos_local, float(options["local_mix_weight"]), bool(options["disable_global_local_mix"]))
    ref = _mix_torch_scores(ref_global, ref_local, float(options["local_mix_weight"]), bool(options["disable_global_local_mix"]))
    pos_metric = _scale_for_modality_temperature(pos, tau_cvr, use_modality_temperature)
    ref_metric = _scale_for_modality_temperature(ref, tau_cvr, use_modality_temperature)
    loss_ref = torch.relu(0.2 - pos_metric + ref_metric).mean()
    if options["disable_reference_negative"]:
        loss_ref = torch.zeros((), device=retrieval_logits.device)
    neg_scores_global = torch.einsum("bd,bnd->bn", query, negative)
    neg_scores_local = _negative_local_scores_torch(torch, query, negative_segments) if negative_segments is not None else None
    neg_scores = _mix_torch_scores(neg_scores_global, neg_scores_local, float(options["local_mix_weight"]), bool(options["disable_global_local_mix"]))
    neg_metric = _scale_for_modality_temperature(neg_scores, tau_cvr, use_modality_temperature)
    curriculum_mask = _curriculum_mask(torch, records, neg_mask, int(options["curriculum_stage"]))
    type_weight = curriculum_mask
    if "negative_effective_mask" in batch:
        type_weight = type_weight * batch["negative_effective_mask"]
    false_negative_weight = torch.ones_like(type_weight)
    if options["enable_false_negative_filtering"]:
        false_negative_weight = _false_negative_weights(
            torch,
            records,
            neg_scores,
            threshold=float(options["false_negative_sim_threshold"]),
            soft_weight=float(options["false_negative_soft_weight"]),
        )
    base_negative_weight = type_weight * false_negative_weight
    quantile_weight = _quantile_negative_curriculum_weights(
        torch,
        neg_metric,
        base_negative_weight,
        enabled=bool(options["enable_quantile_negative_curriculum"]),
        step=int(options.get("current_step", 1)),
        total_steps=int(options.get("total_steps", 1)),
        warmup_ratio=float(options["negative_curriculum_warmup_ratio"]),
        keep_ratio_start=float(options["negative_keep_ratio_start"]),
        keep_ratio_end=float(options["negative_keep_ratio_end"]),
        easy_weight=float(options["easy_negative_weight"]),
    )
    effective_mask = base_negative_weight * quantile_weight
    loss_masked_dcl, dcl_effective_negative_count = _masked_dcl_loss(
        torch,
        retrieval_logits,
        neg_metric,
        effective_mask,
        batch.get("positive_group_index"),
        debias_prob=float(options["dcl_debias_prob"]),
        negative_floor=float(options["dcl_negative_floor"]),
    )
    if str(options.get("contrastive_objective", "ce")) == "masked_dcl":
        loss_cvr = loss_masked_dcl
    margins = torch.relu(0.2 - pos_metric[:, None] + neg_metric)
    hn = margins * effective_mask
    loss_hn = hn.sum() / effective_mask.sum().clamp_min(1.0)
    if options["disable_hard_negatives"]:
        loss_hn = torch.zeros((), device=retrieval_logits.device)
    loss_hw_hn = torch.zeros((), device=retrieval_logits.device)
    hardness = torch.ones_like(effective_mask)
    if options["enable_hardness_weighting"] and not options["disable_hard_negatives"]:
        hardness = _hardness_weights(
            torch,
            records,
            neg_metric,
            effective_mask,
            temperature=float(options["hardness_temperature"]),
            weight_min=float(options["hardness_weight_min"]),
            weight_max=float(options["hardness_weight_max"]),
        )
        loss_hw_hn = (margins * effective_mask * hardness).sum() / effective_mask.sum().clamp_min(1.0)
    final_negative_weight = effective_mask * hardness
    delta_losses: list[Any] = []
    edit_type_losses: list[Any] = []
    for index, record in enumerate(records):
        edit_type = _normalize_edit_type(record.edit_type, record.edit_text)
        target_edit = _mix_torch_scores(
            torch.sum(target[index] * edit[index], dim=-1),
            _single_local_score_torch(torch, target_segments[index], edit[index]) if target_segments is not None else None,
            float(options["local_mix_weight"]),
            bool(options["disable_global_local_mix"]),
        )
        target_edit = _scale_for_modality_temperature(target_edit, tau_audio_text, use_modality_temperature)
        reference_edit = _mix_torch_scores(
            torch.sum(reference[index] * edit[index], dim=-1),
            _single_local_score_torch(torch, reference_segments[index], edit[index]) if reference_segments is not None else None,
            float(options["local_mix_weight"]),
            bool(options["disable_global_local_mix"]),
        )
        reference_edit = _scale_for_modality_temperature(reference_edit, tau_audio_text, use_modality_temperature)
        target_old = _mix_torch_scores(
            torch.sum(target[index] * old_audio[index], dim=-1),
            _single_local_score_torch(torch, target_segments[index], old_audio[index]) if target_segments is not None else None,
            float(options["local_mix_weight"]),
            bool(options["disable_global_local_mix"]),
        )
        target_old = _scale_for_modality_temperature(target_old, tau_audio_text, use_modality_temperature)
        target_new = _mix_torch_scores(
            torch.sum(target[index] * new_audio[index], dim=-1),
            _single_local_score_torch(torch, target_segments[index], new_audio[index]) if target_segments is not None else None,
            float(options["local_mix_weight"]),
            bool(options["disable_global_local_mix"]),
        )
        target_new = _scale_for_modality_temperature(target_new, tau_audio_text, use_modality_temperature)
        reference_old = _mix_torch_scores(
            torch.sum(reference[index] * old_audio[index], dim=-1),
            _single_local_score_torch(torch, reference_segments[index], old_audio[index]) if reference_segments is not None else None,
            float(options["local_mix_weight"]),
            bool(options["disable_global_local_mix"]),
        )
        reference_old = _scale_for_modality_temperature(reference_old, tau_audio_text, use_modality_temperature)
        reference_new = _mix_torch_scores(
            torch.sum(reference[index] * new_audio[index], dim=-1),
            _single_local_score_torch(torch, reference_segments[index], new_audio[index]) if reference_segments is not None else None,
            float(options["local_mix_weight"]),
            bool(options["disable_global_local_mix"]),
        )
        reference_new = _scale_for_modality_temperature(reference_new, tau_audio_text, use_modality_temperature)
        if edit_type in {"remove", "decrease"}:
            delta = reference_edit - target_edit
            delta_losses.append(torch.relu(0.2 - delta))
        elif edit_type == "replace":
            edit_type_losses.append(torch.relu(0.2 - target_new + target_old))
            edit_type_losses.append(torch.relu(0.2 - reference_old + reference_new))
        else:
            delta = target_edit - reference_edit
            delta_losses.append(torch.relu(0.2 - delta))
    loss_delta = torch.stack(delta_losses).mean() if delta_losses else torch.zeros((), device=retrieval_logits.device)
    loss_edit_type = torch.stack(edit_type_losses).mean() if edit_type_losses else torch.zeros((), device=retrieval_logits.device)
    if options["disable_delta_loss"]:
        loss_delta = torch.zeros((), device=retrieval_logits.device)
    if options["disable_edit_type_loss"]:
        loss_edit_type = torch.zeros((), device=retrieval_logits.device)
    visual_sim = torch.sum(target * reference, dim=-1)
    loss_visual = torch.relu(0.05 - visual_sim).mean()
    zero = torch.zeros((), device=retrieval_logits.device)
    doc_for_align = torch.cat([target, reference], dim=0)
    edit_for_align = torch.cat([edit, old_audio, new_audio], dim=0)
    delta_vec = target - reference
    loss_coral_query_target = _coral_loss(torch, query, target) if options["enable_coral_align"] else zero
    loss_coral_doc_edit = _coral_loss(torch, doc_for_align, edit_for_align) if options["enable_coral_align"] and str(options.get("training_profile")) not in {"v2_research", "e5_omni_recipe"} else zero
    loss_coral_delta_edit = _coral_loss(torch, delta_vec, edit) if options["enable_coral_align"] else zero
    if str(options.get("training_profile")) in {"v2_research", "e5_omni_recipe"}:
        loss_coral_delta_edit = zero
    loss_coral_align = loss_coral_query_target + loss_coral_doc_edit + loss_coral_delta_edit
    loss_batch_whitening = (
        _batch_whitening_loss(torch, torch.cat([query, target], dim=0))
        if options["enable_batch_whitening"]
        else zero
    )
    loss_memory_bank = _memory_bank_loss(torch, pos, query, options.get("memory_bank"), temperature=float(tau_cvr.detach().cpu())) if options["enable_memory_bank"] else zero
    total = (
        loss_cvr
        + float(options["lambda_delta"]) * loss_delta
        + float(options["lambda_hn"]) * loss_hn
        + float(options["lambda_ref"]) * loss_ref
        + float(options["lambda_edit_type"]) * loss_edit_type
        + float(options["lambda_visual"]) * loss_visual
        + float(options["lambda_hw_hn"]) * loss_hw_hn
        + float(options["lambda_multi_positive"]) * loss_multi_positive
        + float(options["lambda_coral_align"]) * loss_coral_align
        + float(options["lambda_memory_bank"]) * loss_memory_bank
        + float(options["lambda_batch_whitening"]) * loss_batch_whitening
    )
    hard_score_mean, easy_score_mean = _hard_easy_score_means(torch, neg_metric, base_negative_weight, quantile_weight)
    suspected_false_negative_count = ((false_negative_weight < 1.0) & (type_weight > 0)).sum()
    kept_negative_count = ((quantile_weight >= 1.0) & (base_negative_weight > 0)).sum()
    masked_easy_negative_count = ((quantile_weight < 1.0) & (base_negative_weight > 0)).sum()
    return {
        "total": total,
        "loss_cvr": loss_cvr,
        "loss_ce": loss_ce,
        "loss_masked_dcl": loss_masked_dcl,
        "loss_delta": loss_delta,
        "loss_hn": loss_hn,
        "loss_ref": loss_ref,
        "loss_edit_type": loss_edit_type,
        "loss_visual": loss_visual,
        "loss_hw_hn": loss_hw_hn,
        "loss_multi_positive": loss_multi_positive,
        "loss_coral_align": loss_coral_align,
        "loss_coral_query_target": loss_coral_query_target,
        "loss_coral_doc_edit": loss_coral_doc_edit,
        "loss_coral_delta_edit": loss_coral_delta_edit,
        "loss_batch_whitening": loss_batch_whitening,
        "loss_memory_bank": loss_memory_bank,
        "effective_negative_count": dcl_effective_negative_count.detach() if str(options.get("contrastive_objective", "ce")) == "masked_dcl" else effective_mask.sum().detach(),
        "kept_negative_count": kept_negative_count.detach(),
        "masked_easy_negative_count": masked_easy_negative_count.detach(),
        "suspected_false_negative_count": suspected_false_negative_count.detach(),
        "avg_negative_weight": final_negative_weight.mean().detach() if "final_negative_weight" in locals() else zero,
        "avg_hard_negative_score": hard_score_mean.detach(),
        "avg_easy_negative_score": easy_score_mean.detach(),
        "tau_text": tau_text.detach(),
        "tau_audio": tau_audio.detach(),
        "tau_video": tau_video.detach(),
        "tau_query": tau_query.detach(),
        "tau_target": tau_target.detach(),
        "tau_audio_text": tau_audio_text.detach(),
        "effective_temperature_cvr": tau_cvr.detach(),
        "effective_temperature_delta": tau_audio_text.detach(),
        "cov_doc_trace": _covariance_trace(torch, doc_for_align).detach(),
        "cov_query_trace": _covariance_trace(torch, query).detach(),
        "cov_target_trace": _covariance_trace(torch, target).detach(),
        "cov_query_target_gap": torch.abs(_covariance_trace(torch, query) - _covariance_trace(torch, target)).detach(),
        "cov_edit_trace": _covariance_trace(torch, edit_for_align).detach(),
        "cov_delta_trace": _covariance_trace(torch, delta_vec).detach(),
        "whitening_mean_norm": torch.cat([query, target], dim=0).mean(dim=0).norm().detach(),
        "whitening_cov_gap": _batch_whitening_loss(torch, torch.cat([query, target], dim=0)).detach(),
        "whitening_enabled": torch.as_tensor(1.0 if options["enable_batch_whitening"] else 0.0, device=retrieval_logits.device),
    }


def _loss_options(**overrides: Any) -> dict[str, Any]:
    options = dict(DEFAULT_LOSS_OPTIONS)
    options.update(overrides)
    options["local_mix_weight"] = min(1.0, max(0.0, float(options["local_mix_weight"])))
    options["curriculum_stage"] = max(1, min(4, int(options["curriculum_stage"])))
    options["training_profile"] = str(options.get("training_profile") or "v1")
    options["contrastive_objective"] = str(options.get("contrastive_objective") or "ce")
    for key in (
        "lambda_delta",
        "lambda_hn",
        "lambda_ref",
        "lambda_edit_type",
        "lambda_visual",
        "lambda_hw_hn",
        "lambda_multi_positive",
        "lambda_coral_align",
        "lambda_memory_bank",
        "lambda_batch_whitening",
        "dcl_debias_prob",
        "dcl_negative_floor",
        "hardness_temperature",
        "hardness_weight_min",
        "hardness_weight_max",
        "false_negative_sim_threshold",
        "false_negative_soft_weight",
        "temperature",
        "modality_temperature_init",
        "modality_temperature_min",
        "modality_temperature_max",
        "negative_keep_ratio_start",
        "negative_keep_ratio_end",
        "negative_curriculum_warmup_ratio",
        "easy_negative_weight",
    ):
        options[key] = float(options[key])
    return options


def _training_profile_options(
    *,
    training_profile: str,
    enable_hardness_weighting: bool | None,
    enable_multi_positive: bool | None,
    enable_coral_align: bool | None,
    enable_memory_bank: bool | None,
    enable_false_negative_filtering: bool | None,
    enable_modality_temperature: bool | None,
    enable_quantile_negative_curriculum: bool | None,
    enable_batch_whitening: bool | None,
    lambda_hw_hn: float | None,
    lambda_multi_positive: float | None,
    lambda_coral_align: float | None,
    lambda_memory_bank: float | None,
    lambda_batch_whitening: float | None,
) -> dict[str, Any]:
    profile = str(training_profile or "v1")
    if profile not in {"v1", "v2_research", "e5_omni_recipe"}:
        raise ValueError(f"unknown training profile: {training_profile}")
    enabled = profile in {"v2_research", "e5_omni_recipe"}
    result = {
        "contrastive_objective": "masked_dcl" if enabled else "ce",
        "enable_hardness_weighting": False if enable_hardness_weighting is None else enable_hardness_weighting,
        "enable_multi_positive": False if enable_multi_positive is None else enable_multi_positive,
        "enable_coral_align": enabled if enable_coral_align is None else enable_coral_align,
        "enable_memory_bank": False if enable_memory_bank is None else enable_memory_bank,
        "enable_false_negative_filtering": enabled if enable_false_negative_filtering is None else enable_false_negative_filtering,
        "enable_modality_temperature": enabled if enable_modality_temperature is None else enable_modality_temperature,
        "enable_quantile_negative_curriculum": enabled if enable_quantile_negative_curriculum is None else enable_quantile_negative_curriculum,
        "enable_batch_whitening": enabled if enable_batch_whitening is None else enable_batch_whitening,
        "lambda_delta": 0.0 if enabled else DEFAULT_LOSS_OPTIONS["lambda_delta"],
        "lambda_hn": 0.0 if enabled else DEFAULT_LOSS_OPTIONS["lambda_hn"],
        "lambda_ref": 0.0 if enabled else DEFAULT_LOSS_OPTIONS["lambda_ref"],
        "lambda_edit_type": 0.0 if enabled else DEFAULT_LOSS_OPTIONS["lambda_edit_type"],
        "lambda_visual": 0.0 if enabled else DEFAULT_LOSS_OPTIONS["lambda_visual"],
        "lambda_hw_hn": 0.0,
        "lambda_multi_positive": 0.0,
        "lambda_coral_align": 0.05 if enabled else 0.0,
        "lambda_memory_bank": 0.0,
        "lambda_batch_whitening": 0.01 if enabled else 0.0,
    }
    if lambda_hw_hn is not None:
        result["lambda_hw_hn"] = lambda_hw_hn
    if lambda_multi_positive is not None:
        result["lambda_multi_positive"] = lambda_multi_positive
    if lambda_coral_align is not None:
        result["lambda_coral_align"] = lambda_coral_align
    if lambda_memory_bank is not None:
        result["lambda_memory_bank"] = lambda_memory_bank
    if lambda_batch_whitening is not None:
        result["lambda_batch_whitening"] = lambda_batch_whitening
    return result


def _scheduled_learning_rate(*, base_lr: float, step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if total_steps <= 1:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def _scheduled_temperature(*, step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = (step - 1) / max(1, total_steps - 1)
    return start + (end - start) * min(1.0, max(0.0, progress))


def _multi_positive_loss(torch: Any, logits: Any, positive_group_index: Any | None) -> Any:
    if positive_group_index is None or logits.shape[0] <= 1:
        labels = torch.arange(logits.shape[0], device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels)
    groups = positive_group_index.to(device=logits.device)
    same_group = groups[:, None].eq(groups[None, :])
    same_group.fill_diagonal_(True)
    masked = logits.masked_fill(~same_group, -1e9)
    numerator = torch.logsumexp(masked, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def _masked_dcl_loss(
    torch: Any,
    retrieval_logits: Any,
    explicit_negative_logits: Any,
    explicit_negative_weight: Any,
    positive_group_index: Any | None,
    *,
    debias_prob: float,
    negative_floor: float,
) -> tuple[Any, Any]:
    if retrieval_logits.shape[0] == 0:
        zero = torch.zeros((), device=retrieval_logits.device)
        return zero, zero
    debias_prob = min(0.95, max(0.0, float(debias_prob)))
    negative_floor = max(1e-12, float(negative_floor))
    batch_size = int(retrieval_logits.shape[0])
    losses: list[Any] = []
    effective_counts: list[Any] = []
    for row_index in range(batch_size):
        pos_logit = retrieval_logits[row_index, row_index]
        row_logits: list[Any] = []
        row_weights: list[Any] = []
        for col_index in range(batch_size):
            if col_index == row_index:
                continue
            weight = torch.ones((), dtype=retrieval_logits.dtype, device=retrieval_logits.device)
            if positive_group_index is not None:
                same_group = positive_group_index[row_index] == positive_group_index[col_index]
                weight = torch.where(same_group, torch.zeros_like(weight), weight)
            row_logits.append(retrieval_logits[row_index, col_index])
            row_weights.append(weight)
        for neg_index in range(explicit_negative_logits.shape[1]):
            row_logits.append(explicit_negative_logits[row_index, neg_index])
            row_weights.append(explicit_negative_weight[row_index, neg_index])
        if not row_logits:
            losses.append(torch.zeros((), dtype=retrieval_logits.dtype, device=retrieval_logits.device))
            effective_counts.append(torch.zeros((), dtype=retrieval_logits.dtype, device=retrieval_logits.device))
            continue
        neg_logits = torch.stack(row_logits)
        neg_weights = torch.stack(row_weights).to(dtype=retrieval_logits.dtype, device=retrieval_logits.device)
        max_logit = torch.maximum(pos_logit, neg_logits.max())
        pos_exp = torch.exp(pos_logit - max_logit)
        neg_sum = (torch.exp(neg_logits - max_logit) * neg_weights).sum()
        effective_count = neg_weights.sum()
        if debias_prob > 0.0:
            neg_sum = (neg_sum - debias_prob * effective_count * pos_exp) / max(1e-6, 1.0 - debias_prob)
        neg_sum = torch.clamp(neg_sum, min=negative_floor)
        losses.append(-(torch.log(pos_exp) - torch.log(pos_exp + neg_sum)))
        effective_counts.append(effective_count)
    return torch.stack(losses).mean(), torch.stack(effective_counts).mean()


def _modality_tau(torch: Any, model: Any, modalities: tuple[str, ...], options: dict[str, Any], *, fallback: float, device: Any) -> Any:
    if not options.get("enable_modality_temperature") or not hasattr(model, "modality_temperature"):
        return torch.as_tensor(float(fallback), dtype=torch.float32, device=device)
    return model.modality_temperature(
        modalities,
        tau_min=float(options["modality_temperature_min"]),
        tau_max=float(options["modality_temperature_max"]),
    )


def _pair_tau(left: Any, right: Any) -> Any:
    return 0.5 * (left + right)


def _scale_for_modality_temperature(score: Any, tau: Any, enabled: bool) -> Any:
    return score / tau if enabled else score


def _quantile_negative_curriculum_weights(
    torch: Any,
    neg_scores: Any,
    active_weight: Any,
    *,
    enabled: bool,
    step: int,
    total_steps: int,
    warmup_ratio: float,
    keep_ratio_start: float,
    keep_ratio_end: float,
    easy_weight: float,
) -> Any:
    if not enabled or neg_scores.numel() == 0:
        return torch.ones_like(active_weight)
    warmup_steps = int(max(0, total_steps) * max(0.0, float(warmup_ratio)))
    if step <= warmup_steps:
        ratio = float(keep_ratio_start)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        ratio = float(keep_ratio_start) + (float(keep_ratio_end) - float(keep_ratio_start)) * min(1.0, max(0.0, progress))
    ratio = min(1.0, max(0.0, ratio))
    easy_weight = min(1.0, max(0.0, float(easy_weight)))
    rows: list[Any] = []
    for row_index in range(neg_scores.shape[0]):
        row_active = active_weight[row_index] > 0
        active_count = int(row_active.sum().detach().cpu())
        if active_count <= 0:
            rows.append(torch.zeros_like(active_weight[row_index]))
            continue
        keep_count = max(1, int(np.ceil(active_count * ratio)))
        active_scores = neg_scores[row_index].detach().masked_fill(~row_active, -1e9)
        keep_indices = torch.topk(active_scores, k=min(keep_count, active_scores.shape[0]), dim=0).indices
        row = torch.full_like(active_weight[row_index], fill_value=easy_weight)
        row = torch.where(row_active, row, torch.zeros_like(row))
        row[keep_indices] = 1.0
        rows.append(row)
    return torch.stack(rows, dim=0)


def _hard_easy_score_means(torch: Any, neg_scores: Any, active_weight: Any, quantile_weight: Any) -> tuple[Any, Any]:
    active = active_weight > 0
    hard = active & (quantile_weight >= 1.0)
    easy = active & (quantile_weight < 1.0)
    zero = torch.zeros((), device=neg_scores.device, dtype=neg_scores.dtype)
    hard_mean = neg_scores[hard].mean() if bool(hard.any().detach().cpu()) else zero
    easy_mean = neg_scores[easy].mean() if bool(easy.any().detach().cpu()) else zero
    return hard_mean, easy_mean


def _batch_covariance(torch: Any, value: Any) -> Any:
    if value.shape[0] <= 1:
        return torch.zeros((value.shape[-1], value.shape[-1]), device=value.device, dtype=value.dtype)
    centered = value - value.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(1, value.shape[0] - 1)


def _coral_loss(torch: Any, left: Any, right: Any) -> Any:
    if left.shape[0] <= 1 or right.shape[0] <= 1:
        return torch.zeros((), device=left.device)
    left_cov = _batch_covariance(torch, left)
    right_cov = _batch_covariance(torch, right)
    dim = max(1, int(left.shape[-1]))
    return ((left_cov - right_cov) ** 2).sum() / (4.0 * dim * dim)


def _batch_whitening_loss(torch: Any, value: Any) -> Any:
    if value.shape[0] <= 1:
        return torch.zeros((), device=value.device)
    mean_loss = torch.sum(value.mean(dim=0) ** 2)
    cov = _batch_covariance(torch, value)
    identity = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    dim = max(1, int(value.shape[-1]))
    return mean_loss + ((cov - identity) ** 2).sum() / (dim * dim)


def _covariance_trace(torch: Any, value: Any) -> Any:
    return torch.trace(_batch_covariance(torch, value))


def _memory_bank_loss(torch: Any, pos_scores: Any, query: Any, memory_bank: Any | None, *, temperature: float) -> Any:
    if memory_bank is None or memory_bank.numel() == 0:
        return torch.zeros((), device=query.device)
    bank = memory_bank.to(device=query.device, dtype=query.dtype)
    pos_logits = (pos_scores / max(1e-6, temperature)).unsqueeze(1)
    bank_logits = query @ bank.T / max(1e-6, temperature)
    all_logits = torch.cat([pos_logits, bank_logits], dim=1)
    return (torch.logsumexp(all_logits, dim=1) - pos_logits.squeeze(1)).mean()


def _false_negative_weights(torch: Any, records: list[AudioDeltaRecord], neg_scores: Any, *, threshold: float, soft_weight: float) -> Any:
    rows: list[list[float]] = []
    detached = neg_scores.detach().cpu()
    for row_index, record in enumerate(records):
        negatives = _ordered_negatives(record)
        row: list[float] = []
        for neg_index in range(len(DEFAULT_NEGATIVE_TYPES)):
            if neg_index >= len(negatives):
                row.append(0.0)
                continue
            negative = negatives[neg_index]
            neg_type = str(negative.get("type", ""))
            neg_group = str(negative.get("pair_group_id") or negative.get("inverse_pair_group_id") or "")
            record_group = str(record.inverse_pair_group_id or record.pair_group_id or "")
            if neg_group and neg_group == record_group:
                row.append(0.0)
            elif neg_type != "reference_negative" and float(detached[row_index, neg_index]) >= threshold:
                row.append(float(soft_weight))
            else:
                row.append(1.0)
        rows.append(row)
    return torch.as_tensor(rows, dtype=neg_scores.dtype, device=neg_scores.device)


def _hardness_weights(
    torch: Any,
    records: list[AudioDeltaRecord],
    neg_scores: Any,
    active_mask: Any,
    *,
    temperature: float,
    weight_min: float,
    weight_max: float,
) -> Any:
    rows: list[Any] = []
    temp = max(1e-6, float(temperature))
    for row_index, record in enumerate(records):
        negatives = _ordered_negatives(record)
        type_mask = []
        for neg_index in range(len(DEFAULT_NEGATIVE_TYPES)):
            neg_type = str(negatives[neg_index].get("type", "")) if neg_index < len(negatives) else ""
            type_mask.append(1.0 if neg_type in {"visual_hard", "audio_hard", "asr_hard"} else 0.0)
        type_tensor = torch.as_tensor(type_mask, dtype=active_mask.dtype, device=active_mask.device)
        row_mask = active_mask[row_index] * type_tensor
        if float(row_mask.sum().detach().cpu()) <= 0:
            rows.append(torch.ones_like(active_mask[row_index]))
            continue
        masked = (neg_scores[row_index].detach() / temp).masked_fill(row_mask <= 0, -1e9)
        weights = torch.nn.functional.softmax(masked, dim=0) * row_mask.sum().clamp_min(1.0)
        weights = torch.clamp(weights, min=weight_min, max=weight_max)
        rows.append(torch.where(row_mask > 0, weights, torch.ones_like(weights)))
    return torch.stack(rows, dim=0)


def _batch_has_local(batch: dict[str, Any]) -> bool:
    return "target_segments" in batch and "reference_segments" in batch and "negative_segments" in batch


def _has_local_segments(data: dict[str, np.ndarray]) -> bool:
    return "target_segments" in data and data["target_segments"].ndim == 3 and data["target_segments"].shape[1] > 0


def _score_matrix_np(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _normalize_np(query) @ _normalize_np(target).T


def _local_score_matrix_np(query: np.ndarray, target_segments: np.ndarray) -> np.ndarray:
    query_norm = _normalize_np(query)
    flat_segments = target_segments.reshape(-1, target_segments.shape[-1])
    segment_norm = _normalize_np(flat_segments).reshape(target_segments.shape)
    return np.einsum("bd,nsd->bns", query_norm, segment_norm).max(axis=-1)


def _mix_scores(global_scores: np.ndarray, local_scores: np.ndarray | None, local_mix_weight: float, disable_mix: bool) -> np.ndarray:
    if local_scores is None or disable_mix:
        return global_scores
    weight = min(1.0, max(0.0, float(local_mix_weight)))
    return ((1.0 - weight) * global_scores) + (weight * local_scores)


def _local_score_matrix_torch(torch: Any, query: Any, target_segments: Any) -> Any:
    return torch.einsum("bd,nsd->bns", query, target_segments).max(dim=-1).values


def _paired_local_scores_torch(torch: Any, query: Any, target_segments: Any) -> Any:
    return torch.einsum("bd,bsd->bs", query, target_segments).max(dim=-1).values


def _negative_local_scores_torch(torch: Any, query: Any, negative_segments: Any) -> Any:
    return torch.einsum("bd,bnsd->bns", query, negative_segments).max(dim=-1).values


def _single_local_score_torch(torch: Any, segments: Any, vector: Any) -> Any:
    return torch.einsum("sd,d->s", segments, vector).max()


def _mix_torch_scores(global_scores: Any, local_scores: Any | None, local_mix_weight: float, disable_mix: bool) -> Any:
    if local_scores is None or disable_mix:
        return global_scores
    weight = min(1.0, max(0.0, float(local_mix_weight)))
    return ((1.0 - weight) * global_scores) + (weight * local_scores)


def _curriculum_mask(torch: Any, records: list[AudioDeltaRecord], neg_mask: Any, curriculum_stage: int) -> Any:
    keep_rows: list[list[float]] = []
    for record in records:
        negatives = _ordered_negatives(record)
        row: list[float] = []
        for index in range(len(DEFAULT_NEGATIVE_TYPES)):
            if index >= len(negatives):
                row.append(0.0)
                continue
            neg_type = str(negatives[index].get("type", ""))
            required_stage = NEGATIVE_CURRICULUM_STAGE.get(neg_type, 4)
            row.append(1.0 if required_stage <= curriculum_stage else 0.0)
        keep_rows.append(row)
    return neg_mask * torch.as_tensor(keep_rows, dtype=neg_mask.dtype, device=neg_mask.device)


def _grouped_recall_summary(
    base_scores: np.ndarray,
    adapted_scores: np.ndarray,
    records: list[AudioDeltaRecord],
    field_name: str,
    topk: tuple[int, ...],
    positive_index: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(getattr(record, field_name, "unknown") or "unknown")].append(index)
    result: dict[str, Any] = {}
    for group, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        result[group] = {
            "count": int(len(indices)),
            "base_e5": _recall_from_scores(base_scores[idx], topk=topk, positive_index=positive_index[idx]),
            "audio_delta_adapter": _recall_from_scores(adapted_scores[idx], topk=topk, positive_index=positive_index[idx]),
        }
    return result


def _reference_rank_summary(scores: np.ndarray, reference_scores: np.ndarray) -> dict[str, Any]:
    ranks: list[int] = []
    for index in range(scores.shape[0]):
        rank = int((scores[index] > reference_scores[index]).sum() + 1)
        ranks.append(rank)
    return _rank_list_summary(ranks)


def _delta_score_distribution(scores: np.ndarray, reference_scores: np.ndarray, *, positive_index: np.ndarray) -> dict[str, Any]:
    deltas = np.asarray([scores[index, int(positive_index[index])] - reference_scores[index] for index in range(scores.shape[0])], dtype=np.float32)
    if deltas.size == 0:
        return {"count": 0}
    return {
        "count": int(deltas.size),
        "mean": round(float(deltas.mean()), 6),
        "median": round(float(np.median(deltas)), 6),
        "min": round(float(deltas.min()), 6),
        "max": round(float(deltas.max()), 6),
        "positive_delta_rate": round(float((deltas > 0).mean()), 4),
    }


def _hard_negative_recall_by_type(scores: np.ndarray, negative_scores: np.ndarray, records: list[AudioDeltaRecord], *, positive_index: np.ndarray) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row_index, record in enumerate(records):
        positive = float(scores[row_index, int(positive_index[row_index])])
        negatives = _ordered_negatives(record)
        for neg_index, negative in enumerate(negatives[: negative_scores.shape[1]]):
            neg_type = str(negative.get("type", "unknown") or "unknown")
            buckets[neg_type].append(positive > float(negative_scores[row_index, neg_index]))
    return {
        neg_type: {"count": len(values), "positive_beats_negative_rate": round(sum(values) / max(1, len(values)), 4)}
        for neg_type, values in sorted(buckets.items())
    }


def _rank_list_summary(ranks: list[int]) -> dict[str, Any]:
    if not ranks:
        return {"count": 0}
    arr = np.asarray(ranks, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean_rank": round(float(arr.mean()), 4),
        "median_rank": round(float(np.median(arr)), 4),
        "max_rank": int(arr.max()),
        "rank_le_1_rate": round(float((arr <= 1).mean()), 4),
        "rank_le_5_rate": round(float((arr <= 5).mean()), 4),
    }


def _AudioDeltaAdapter(torch: Any, dim: int, *, modality_temperature_init: float = 0.05) -> Any:
    class ModalityAwareTemperature(torch.nn.Module):
        def __init__(self, init_tau: float = 0.05) -> None:
            super().__init__()
            init = float(np.log(max(1e-6, init_tau)))
            self.log_tau_text = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
            self.log_tau_audio = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
            self.log_tau_video = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))

        def forward(self, modalities: tuple[str, ...], *, tau_min: float, tau_max: float) -> Any:
            params = {
                "text": self.log_tau_text,
                "audio": self.log_tau_audio,
                "video": self.log_tau_video,
            }
            values = [torch.exp(params[name]).clamp(min=tau_min, max=tau_max) for name in modalities if name in params]
            if not values:
                values = [torch.exp(self.log_tau_text).clamp(min=tau_min, max=tau_max)]
            return torch.stack(values).mean()

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_proj = torch.nn.Linear(dim, dim, bias=False)
            self.doc_proj = torch.nn.Linear(dim, dim, bias=False)
            self.edit_proj = torch.nn.Linear(dim, dim, bias=False)
            self.modality_temperature = ModalityAwareTemperature(modality_temperature_init)
            torch.nn.init.eye_(self.query_proj.weight)
            torch.nn.init.eye_(self.doc_proj.weight)
            torch.nn.init.eye_(self.edit_proj.weight)

        def query(self, value: Any) -> Any:
            return torch.nn.functional.normalize(self.query_proj(value), dim=-1)

        def doc(self, value: Any) -> Any:
            return torch.nn.functional.normalize(self.doc_proj(value), dim=-1)

        def edit(self, value: Any) -> Any:
            return torch.nn.functional.normalize(self.edit_proj(value), dim=-1)

    return Adapter()


def _record_from_payload(payload: dict[str, Any], *, line_number: int) -> AudioDeltaRecord:
    sample_id = _first_text(payload, "sample_id", "proposal_id", "candidate_id") or f"record_{line_number:06d}"
    reference_video = _first_text(payload, "reference_video", "reference_path")
    target_video = _first_text(payload, "target_video", "target_path")
    edit_text = _first_text(payload, "edit_text", "refined_edit_text", "audio_only_edit_text")
    if not reference_video:
        raise ValueError(f"line {line_number} missing reference_video")
    if not target_video:
        raise ValueError(f"line {line_number} missing target_video")
    if not edit_text:
        raise ValueError(f"line {line_number} missing edit_text")
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    audio_delta_analysis = payload.get("audio_delta_analysis") if isinstance(payload.get("audio_delta_analysis"), dict) else {}
    hard_negatives = payload.get("audio_delta_hard_negatives") or payload.get("hard_negatives") or []
    return AudioDeltaRecord(
        sample_id=sample_id,
        reference_video=reference_video,
        target_video=target_video,
        edit_text=edit_text,
        edit_type=_normalize_edit_type(_first_text(payload, "edit_type", "inverse_generation_rule"), edit_text),
        audio_delta_type=_first_text(payload, "audio_delta_type", "b_subtype", default="speech_topic"),
        old_audio=_first_text(payload, "old_audio", "reference_audio_content", default=str(audio_delta_analysis.get("reference_audio_content", ""))),
        new_audio=_first_text(payload, "new_audio", "target_audio_content", default=str(audio_delta_analysis.get("target_audio_content", ""))),
        direction=_first_text(payload, "direction", default="inverse" if payload.get("is_inverse") else "forward"),
        split_tier=_first_text(payload, "split_tier", default="extended"),
        raw_source_id=_first_text(payload, "raw_source_id", "source_disjoint_group_id", "source_clip_id", "group_id", default=sample_id),
        pair_group_id=_first_text(payload, "pair_group_id", default=sample_id),
        inverse_pair_group_id=_first_text(payload, "inverse_pair_group_id", default=_first_text(payload, "pair_group_id", default=sample_id)),
        shortcut_label=_first_text(payload, "shortcut_label", default="unknown"),
        audio_delta_strength=_float_value(payload.get("audio_delta_strength", quality.get("audio_delta_strength", audio_delta_analysis.get("audio_delta_strength", 0.0)))),
        video_context_strength=_float_value(payload.get("video_context_strength", quality.get("video_context_strength", 0.0))),
        asr_degeneracy_risk=_float_value(payload.get("asr_degeneracy_risk", quality.get("asr_degeneracy_risk", 0.0))),
        visual_shortcut_risk=_float_value(payload.get("visual_shortcut_risk", quality.get("visual_shortcut_risk", 0.0))),
        full_av_required=bool(payload.get("full_av_required", quality.get("full_av_required", False))),
        hard_negatives=tuple(_normalize_negative_items(hard_negatives)),
        source_payload=dict(payload),
    )


def _normalize_negative_items(items: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if not isinstance(items, list):
        return result
    for index, item in enumerate(items):
        if isinstance(item, dict):
            video = str(item.get("video") or item.get("target_video") or item.get("path") or "").strip()
            neg_type = str(item.get("type") or DEFAULT_NEGATIVE_TYPES[min(index, len(DEFAULT_NEGATIVE_TYPES) - 1)]).strip()
            pair_group_id = str(item.get("pair_group_id") or "").strip()
            inverse_pair_group_id = str(item.get("inverse_pair_group_id") or "").strip()
        else:
            video = str(item).strip()
            neg_type = DEFAULT_NEGATIVE_TYPES[min(index, len(DEFAULT_NEGATIVE_TYPES) - 1)]
            pair_group_id = ""
            inverse_pair_group_id = ""
        if video:
            normalized = {"type": neg_type, "video": video}
            if pair_group_id:
                normalized["pair_group_id"] = pair_group_id
            if inverse_pair_group_id:
                normalized["inverse_pair_group_id"] = inverse_pair_group_id
            result.append(normalized)
    return result


def _default_train_paths(root: Path) -> list[Path]:
    candidates = [
        root / "b_splits" / "train.jsonl",
        root / "b_train_bidirectional_triplets.jsonl",
        root / "b_main_audio_cvr_triplets.jsonl",
        root / "b_extended_audio_cvr_triplets.jsonl",
        root / "b_all_audio_cvr_triplets.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def _default_eval_paths(root: Path) -> list[Path]:
    candidates = [
        root / "b_splits" / "val.jsonl",
        root / "b_splits" / "test_main.jsonl",
        root / "b_main_audio_cvr_triplets.jsonl",
        root / "b_all_audio_cvr_triplets.jsonl",
        root / "b_extended_audio_cvr_triplets.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def _default_distractor_pool_paths(root: Path) -> list[Path]:
    candidates = [
        root / "single_source_annotations.jsonl",
        root / "extracted_single_source_clips.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def _build_eval_gallery(
    *,
    dataset_root: Path,
    train_records: list[AudioDeltaRecord],
    eval_records: list[AudioDeltaRecord],
    total_gallery_size: int,
    distractor_pool_path: str | Path | None,
    seed: int,
) -> tuple[list[EvalGalleryItem], list[int], dict[str, Any]]:
    desired_total = max(len(eval_records), int(total_gallery_size))
    positives: list[EvalGalleryItem] = [
        EvalGalleryItem(
            gallery_id=f"positive::{record.sample_id}",
            video=record.target_video,
            raw_source_id=str(record.raw_source_id or record.sample_id),
            kind="positive",
            source_payload={"sample_id": record.sample_id, "pair_group_id": record.pair_group_id},
        )
        for record in eval_records
    ]
    forbidden_video_keys = {
        _media_key(record.reference_video)
        for record in (*train_records, *eval_records)
    }
    forbidden_video_keys.update(_media_key(record.target_video) for record in (*train_records, *eval_records))
    for record in (*train_records, *eval_records):
        for negative in record.hard_negatives:
            video = str(negative.get("video") or "").strip()
            if video:
                forbidden_video_keys.add(_media_key(video))
    forbidden_source_ids = {str(record.raw_source_id or "").strip() for record in eval_records if str(record.raw_source_id or "").strip()}
    pool_paths = [Path(distractor_pool_path)] if distractor_pool_path else _default_distractor_pool_paths(dataset_root)
    distractor_candidates = _load_distractor_pool(pool_paths)
    distractors: list[EvalGalleryItem] = []
    seen_gallery_keys = {_media_key(item.video) for item in positives}
    rng = random.Random(seed)
    rng.shuffle(distractor_candidates)
    for payload in distractor_candidates:
        video = _first_text(payload, "output_path", "video", "video_path", "clip_path", "path", "target_video", "reference_video")
        if not video:
            continue
        video_key = _media_key(video)
        source_id = _first_text(payload, "raw_source_id", "source_clip_id", "group_id", default="")
        if video_key in forbidden_video_keys or video_key in seen_gallery_keys:
            continue
        if source_id and source_id in forbidden_source_ids:
            continue
        gallery_id = _first_text(payload, "clip_id", "gallery_id", default=f"distractor_{len(distractors):06d}")
        distractors.append(
            EvalGalleryItem(
                gallery_id=f"distractor::{gallery_id}",
                video=video,
                raw_source_id=source_id or gallery_id,
                kind="distractor",
                source_payload=dict(payload),
            )
        )
        seen_gallery_keys.add(video_key)
        if len(positives) + len(distractors) >= desired_total:
            break
    gallery_items = positives + distractors
    rng.shuffle(gallery_items)
    gallery_lookup = {item.gallery_id: index for index, item in enumerate(gallery_items)}
    positive_indices = [int(gallery_lookup[f"positive::{record.sample_id}"]) for record in eval_records]
    summary = {
        "gallery_count": len(gallery_items),
        "positive_count": len(positives),
        "distractor_count": len(distractors),
        "requested_gallery_size": int(total_gallery_size),
        "pool_paths": [str(path) for path in pool_paths],
        "forbidden_source_count": len(forbidden_source_ids),
    }
    return gallery_items, positive_indices, summary


def _load_distractor_pool(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_records_from_paths(paths: list[str | Path]) -> list[AudioDeltaRecord]:
    records: list[AudioDeltaRecord] = []
    for path in paths:
        records.extend(load_audio_delta_records(path))
    return records


def _dedupe_records(records: list[AudioDeltaRecord]) -> list[AudioDeltaRecord]:
    seen: set[str] = set()
    result: list[AudioDeltaRecord] = []
    for record in records:
        key = f"{record.sample_id}|{record.direction}|{record.reference_video}|{record.target_video}|{record.edit_text}"
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _media_key(raw_path: str) -> str:
    return _resolve_media_path(raw_path).replace("\\", "/").lower()


def _is_diagnostic_record(record: AudioDeltaRecord) -> bool:
    tier = str(record.split_tier or "").lower()
    label = str(record.shortcut_label or "").lower()
    return tier == "diagnostic" or "asr" in label or "shortcut" in label and record.visual_shortcut_risk > 0.5


def _split_group_id(record: AudioDeltaRecord) -> str:
    raw_source = str(record.raw_source_id or "").strip()
    if raw_source:
        return raw_source
    return str(record.inverse_pair_group_id or record.pair_group_id or record.sample_id)


def _one_direction_per_pair(records: list[AudioDeltaRecord]) -> list[AudioDeltaRecord]:
    seen: set[str] = set()
    result: list[AudioDeltaRecord] = []
    for record in sorted(records, key=lambda item: (item.inverse_pair_group_id, item.sample_id)):
        group = str(record.inverse_pair_group_id or record.pair_group_id or record.sample_id)
        if group in seen:
            continue
        seen.add(group)
        result.append(record)
    return result


def _split_leakage_checks(
    split_records: dict[str, list[AudioDeltaRecord]],
    test_main: list[AudioDeltaRecord],
    test_inverse: list[AudioDeltaRecord],
    diagnostic: list[AudioDeltaRecord],
) -> dict[str, Any]:
    source_to_split: dict[str, str] = {}
    source_leaks: list[str] = []
    pair_to_split: dict[str, str] = {}
    pair_leaks: list[str] = []
    for split_name, records in split_records.items():
        for record in records:
            source = _split_group_id(record)
            if source in source_to_split and source_to_split[source] != split_name:
                source_leaks.append(source)
            source_to_split[source] = split_name
            pair = str(record.inverse_pair_group_id or record.pair_group_id or record.sample_id)
            if pair in pair_to_split and pair_to_split[pair] != split_name:
                pair_leaks.append(pair)
            pair_to_split[pair] = split_name
    test_main_pairs = [str(record.inverse_pair_group_id or record.pair_group_id or record.sample_id) for record in test_main]
    return {
        "raw_source_cross_split_leaks": sorted(set(source_leaks)),
        "pair_group_cross_split_leaks": sorted(set(pair_leaks)),
        "test_main_unique_pair_groups": len(test_main_pairs) == len(set(test_main_pairs)),
        "test_inverse_count": len(test_inverse),
        "diagnostic_count": len(diagnostic),
    }


def _query_payload(record: AudioDeltaRecord) -> dict[str, str]:
    return {"video": _resolve_media_path(record.reference_video), "text": QUERY_TEMPLATE.format(edit_text=record.edit_text.strip().rstrip("."))}


def _video_payload(video_path: str) -> dict[str, str]:
    return {"video": _resolve_media_path(video_path)}


def _resolve_media_path(raw_path: str) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return raw
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    default_root = Path(DEFAULT_DATA_ROOT)
    for candidate in (default_root / path, Path.cwd() / path):
        if candidate.exists():
            return str(candidate)
    if raw.startswith(("clips/", "raw/", "raw_datasets/")):
        return str(default_root / path)
    return raw


def _encode_one(encoder: Any, payload: Any) -> np.ndarray:
    return _normalize_rows(encoder.encode_document([payload]))[0]


def _encode_many(encoder: Any, payloads: list[Any]) -> np.ndarray:
    return _normalize_rows(encoder.encode_document(payloads))


def _local_video_payloads(
    video_path: str,
    *,
    role: str,
    count: int,
    mode: str = "prompt",
    cache_root: str | Path | None = None,
    sample_id: str = "",
    segment_overlap: float = 0.0,
) -> list[dict[str, str]]:
    resolved = _resolve_media_path(video_path)
    total = max(1, int(count))
    mode = _normalize_local_segment_mode(mode)
    if mode == "ffmpeg":
        segment_paths = _ffmpeg_local_segment_paths(
            resolved,
            role=role,
            count=total,
            cache_root=Path(cache_root) if cache_root else Path.cwd() / "local_media_cache",
            sample_id=sample_id,
            segment_overlap=segment_overlap,
        )
        if segment_paths:
            return [{"video": str(path)} for path in segment_paths]
    payloads: list[dict[str, str]] = []
    for index in range(total):
        payloads.append(
            {
                "video": resolved,
                "text": (
                    f"Focus on local temporal segment {index + 1} of {total} for the {role} video. "
                    "Represent short-lived speech, music, sound events, and visual context in this segment."
                ),
            }
        )
    return payloads


def _normalize_local_segment_mode(mode: str) -> str:
    value = str(mode or "prompt").strip().lower()
    if value not in {"prompt", "ffmpeg"}:
        raise ValueError(f"unknown local segment mode: {mode}")
    return value


def _ffmpeg_local_segment_paths(
    video_path: str,
    *,
    role: str,
    count: int,
    cache_root: Path,
    sample_id: str,
    segment_overlap: float,
) -> list[Path]:
    source = Path(video_path)
    if not source.exists():
        return []
    duration = _media_duration_seconds(source)
    if duration <= 0:
        return []
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_sample = _safe_filename(sample_id or source.stem)
    segment_count = max(1, int(count))
    base_length = duration / segment_count
    overlap = max(0.0, min(0.8, float(segment_overlap)))
    segment_length = min(duration, base_length * (1.0 + overlap))
    paths: list[Path] = []
    for index in range(segment_count):
        start = min(max(0.0, index * base_length - (segment_length - base_length) / 2.0), max(0.0, duration - segment_length))
        out = cache_root / f"{safe_sample}__{_safe_filename(role)}__seg{index + 1:02d}.mp4"
        if not out.exists() or out.stat().st_size == 0:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{segment_length:.3f}",
                "-c",
                "copy",
                str(out),
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (OSError, subprocess.CalledProcessError):
                return []
        paths.append(out)
    return paths


def _media_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError):
        return 0.0
    return max(0.0, _float_value(result.stdout.strip()))


def _safe_filename(value: str) -> str:
    text = str(value or "item")
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)[:96] or "item"


def _positive_group_id(record: AudioDeltaRecord) -> str:
    return str(record.inverse_pair_group_id or record.pair_group_id or record.sample_id)


def _positive_group_indices(group_ids: list[str]) -> list[int]:
    mapping: dict[str, int] = {}
    result: list[int] = []
    for group_id in group_ids:
        if group_id not in mapping:
            mapping[group_id] = len(mapping)
        result.append(mapping[group_id])
    return result


def _static_negative_effective_weight(record: AudioDeltaRecord, negative: dict[str, str]) -> float:
    record_group = str(record.inverse_pair_group_id or record.pair_group_id or "")
    neg_group = str(negative.get("inverse_pair_group_id") or negative.get("pair_group_id") or "")
    if neg_group and record_group and neg_group == record_group:
        return 0.0
    return 1.0


def _ordered_negatives(record: AudioDeltaRecord) -> list[dict[str, str]]:
    by_type = {str(item.get("type", "")): item for item in record.hard_negatives}
    ordered = [by_type[key] for key in DEFAULT_NEGATIVE_TYPES if key in by_type]
    ordered.extend(item for item in record.hard_negatives if item not in ordered)
    return ordered[: len(DEFAULT_NEGATIVE_TYPES)]


def _normalize_edit_type(raw: str, edit_text: str) -> str:
    value = str(raw or "").strip().lower().replace("_", "-")
    text = str(edit_text or "").strip().lower()
    if "replace" in value or text.startswith("replace ") or " with " in text:
        return "replace"
    if "remove" in value or text.startswith("remove "):
        return "remove"
    if "decrease" in value or "lower " in text or "reduce " in text:
        return "decrease"
    if "increase" in value or "louder" in text or "raise " in text:
        return "increase"
    if "add" in value or text.startswith("add "):
        return "add"
    return value or "replace"


def _recall_from_scores(scores: np.ndarray, *, topk: tuple[int, ...], positive_index: np.ndarray | None = None) -> dict[str, float]:
    total = scores.shape[0]
    if positive_index is None:
        positive_index = np.arange(total, dtype=np.int64)
    positive_index = np.asarray(positive_index, dtype=np.int64)
    if positive_index.shape[0] != total:
        raise ValueError("positive_index size must match query count")
    order = np.argsort(-scores, axis=1, kind="stable")
    result: dict[str, float] = {}
    for k in topk:
        hits = 0
        for index in range(total):
            if int(positive_index[index]) in order[index, : min(k, scores.shape[1])]:
                hits += 1
        result[f"R@{k}"] = round(hits / max(1, total), 4)
    return result


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# AudioDelta-E5 Smoke Evaluation",
        "",
        f"- eval_count: `{comparison['eval_count']}`",
        f"- gallery_count: `{comparison.get('gallery_count', comparison['eval_count'])}`",
        "",
        "| Method | R@1 | R@5 | R@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(f"| {row['method']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} |")
    return "\n".join(lines) + "\n"


def _ablation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AudioDelta-E5 Ablation",
        "",
        "| Ablation | R@1 | R@5 | R@10 | Ref Avg Rank | Delta Pos Mean | Delta Neg Mean | Effective Neg | Tau T | Tau A | Tau V |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['ablation']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} | "
            f"{_fmt(row.get('reference_negative_average_rank'))} | {_fmt(row.get('delta_score_pos_mean'))} | {_fmt(row.get('delta_score_neg_mean'))} | "
            f"{_fmt(row.get('effective_negative_count'))} | {_fmt(row.get('tau_text'))} | {_fmt(row.get('tau_audio'))} | {_fmt(row.get('tau_video'))} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _normalize_topk(raw: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(sorted({int(k) for k in raw if int(k) > 0}))
    return values or (1, 5, 10)


def _normalize_np(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (value / norms).astype(np.float32)


def _load_embedding_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"embedding cache not found: {path}")
    loaded = np.load(str(path))
    return {key: loaded[key].astype(np.float32) for key in loaded.files}


def _hash_embedding(item: Any, *, dim: int) -> np.ndarray:
    text = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, dict) else str(item)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    counter = 0
    while len(values) < dim:
        block = hashlib.sha256(digest + counter.to_bytes(4, "little")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in block)
        counter += 1
    return np.asarray(values[:dim], dtype=np.float32)


def _first_text(payload: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}
    try:
        return json.loads(rows[-1])
    except json.JSONDecodeError:
        return {}


def _torch_device(torch: Any, raw: str) -> Any:
    if str(raw).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for AudioDelta-E5 adapter training") from exc
    return torch


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


if __name__ == "__main__":
    main()
