from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
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
    summary = {
        "dataset_run_root": str(dataset_root),
        "output_dir": str(output_root),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "train_paths": [str(path) for path in train_paths],
        "eval_paths": [str(path) for path in eval_paths],
        "outputs": {"train": str(train_path), "eval": str(eval_path)},
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
    train_summary = _cache_split_embeddings(
        records=train_records,
        split="train",
        encoder=encoder,
        output_root=output_root,
        runtime_info=runtime_info,
        progress=progress,
    )
    eval_summary = _cache_split_embeddings(
        records=eval_records,
        split="eval",
        encoder=encoder,
        output_root=output_root,
        runtime_info=runtime_info,
        progress=progress,
    )
    summary = {
        "records_dir": str(records_root),
        "output_dir": str(output_root),
        "runtime": runtime_info,
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
    model = _AudioDeltaAdapter(torch, dim).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device_obj) for key, value in data.items()}
    count = int(tensors["query"].shape[0])
    rng = random.Random(seed)
    losses_path = output_root / "loss_curve.jsonl"
    with losses_path.open("w", encoding="utf-8") as losses_file:
        for step in range(1, max(1, steps) + 1):
            indices = [rng.randrange(count) for _ in range(min(max(1, batch_size), count))]
            batch = {key: value[indices] if value.shape[0] == count else value for key, value in tensors.items()}
            batch_records = [records[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            losses = _adapter_losses(torch, model, batch, batch_records)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite adapter loss at step {step}: {float(loss.detach().cpu())}")
            loss.backward()
            optimizer.step()
            row = {"step": step, **{name: round(float(value.detach().cpu()), 6) for name, value in losses.items()}}
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
    topk = _normalize_topk(topk)
    base_scores = _normalize_np(data["query"]) @ _normalize_np(data["target"]).T
    base = _recall_from_scores(base_scores, topk=topk)
    dim = int(data["query"].shape[1])
    device_obj = _torch_device(torch, device)
    model = _AudioDeltaAdapter(torch, dim).to(device_obj)
    state = torch.load(adapter_root / "adapter.pt", map_location=device_obj)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        query = torch.as_tensor(data["query"], dtype=torch.float32, device=device_obj)
        target = torch.as_tensor(data["target"], dtype=torch.float32, device=device_obj)
        adapted_query = model.query(query)
        adapted_target = model.doc(target)
        adapted_scores = (adapted_query @ adapted_target.T).detach().cpu().numpy()
    adapted = _recall_from_scores(adapted_scores, topk=topk)
    comparison = {
        "cache_dir": str(cache_root),
        "adapter_dir": str(adapter_root),
        "output_dir": str(output_root),
        "eval_count": len(records),
        "topk": list(topk),
        "rows": [
            {"method": "base_e5", **base},
            {"method": "audio_delta_adapter", **adapted},
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def load_audio_delta_records(path: str | Path) -> list[AudioDeltaRecord]:
    root = Path(path)
    if not root.exists():
        return []
    return [_record_from_payload(json.loads(line), line_number=index) for index, line in enumerate(root.read_text(encoding="utf-8-sig").splitlines(), start=1) if line.strip()]


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

    train = subparsers.add_parser("train-adapter")
    train.add_argument("--cache-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--steps", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="cuda")

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--cache-dir", required=True)
    evaluate.add_argument("--adapter-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--topk", default="1,5,10")
    evaluate.add_argument("--device", default="cuda")

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
            progress=progress,
        )
    elif args.command == "eval":
        result = eval_adapter(
            cache_dir=args.cache_dir,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            topk=tuple(int(part.strip()) for part in str(args.topk).split(",") if part.strip()),
            device=args.device,
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
    encoder: Any,
    output_root: Path,
    runtime_info: dict[str, Any],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"{split} records are empty")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("query", "target", "reference", "edit", "old_audio", "new_audio")}
    negative_rows: list[list[np.ndarray]] = []
    negative_mask: list[list[float]] = []
    negative_types: list[list[str]] = []
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
            neg_vectors: list[np.ndarray] = []
            neg_mask_row: list[float] = []
            neg_type_row: list[str] = []
            for negative in _ordered_negatives(record):
                video = str(negative.get("video", "")).strip()
                if not video:
                    continue
                neg_vectors.append(_encode_one(encoder, _video_payload(video)))
                neg_mask_row.append(1.0)
                neg_type_row.append(str(negative.get("type", "")).strip() or "unknown")
            while len(neg_vectors) < len(DEFAULT_NEGATIVE_TYPES):
                neg_vectors.append(np.zeros_like(arrays["target"][-1]))
                neg_mask_row.append(0.0)
                neg_type_row.append("")
            negative_rows.append(neg_vectors[: len(DEFAULT_NEGATIVE_TYPES)])
            negative_mask.append(neg_mask_row[: len(DEFAULT_NEGATIVE_TYPES)])
            negative_types.append(neg_type_row[: len(DEFAULT_NEGATIVE_TYPES)])
            manifest_file.write(json.dumps({"sample_id": record.sample_id, "negative_types": neg_type_row}, ensure_ascii=False) + "\n")
            manifest_file.flush()
    stacked = {key: np.vstack(value).astype(np.float32) for key, value in arrays.items()}
    stacked["negative"] = np.asarray(negative_rows, dtype=np.float32)
    stacked["negative_mask"] = np.asarray(negative_mask, dtype=np.float32)
    npz_path = output_root / f"{split}_embeddings.npz"
    np.savez(str(npz_path), **stacked)
    records_path = output_root / f"{split}_records.jsonl"
    _write_jsonl(records_path, [asdict(record) for record in records])
    metadata = {
        "split": split,
        "record_count": len(records),
        "embedding_shape": list(stacked["query"].shape),
        "negative_shape": list(stacked["negative"].shape),
        "runtime": runtime_info,
        "embeddings_path": str(npz_path),
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
    }
    (output_root / f"{split}_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _adapter_losses(torch: Any, model: Any, batch: dict[str, Any], records: list[AudioDeltaRecord]) -> dict[str, Any]:
    query = model.query(batch["query"])
    target = model.doc(batch["target"])
    reference = model.doc(batch["reference"])
    edit = model.edit(batch["edit"])
    old_audio = model.edit(batch["old_audio"])
    new_audio = model.edit(batch["new_audio"])
    negative = model.doc(batch["negative"])
    neg_mask = batch["negative_mask"]
    logits = query @ target.T / 0.05
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_cvr = torch.nn.functional.cross_entropy(logits, labels)
    pos = torch.sum(query * target, dim=-1)
    ref = torch.sum(query * reference, dim=-1)
    loss_ref = torch.relu(0.2 - pos + ref).mean()
    neg_scores = torch.einsum("bd,bnd->bn", query, negative)
    hn = torch.relu(0.2 - pos[:, None] + neg_scores) * neg_mask
    loss_hn = hn.sum() / neg_mask.sum().clamp_min(1.0)
    delta_losses: list[Any] = []
    edit_type_losses: list[Any] = []
    for index, record in enumerate(records):
        edit_type = _normalize_edit_type(record.edit_type, record.edit_text)
        if edit_type in {"remove", "decrease"}:
            delta = torch.sum(reference[index] * edit[index]) - torch.sum(target[index] * edit[index])
            delta_losses.append(torch.relu(0.2 - delta))
        elif edit_type == "replace":
            edit_type_losses.append(torch.relu(0.2 - torch.sum(target[index] * new_audio[index]) + torch.sum(target[index] * old_audio[index])))
            edit_type_losses.append(torch.relu(0.2 - torch.sum(reference[index] * old_audio[index]) + torch.sum(reference[index] * new_audio[index])))
        else:
            delta = torch.sum(target[index] * edit[index]) - torch.sum(reference[index] * edit[index])
            delta_losses.append(torch.relu(0.2 - delta))
    loss_delta = torch.stack(delta_losses).mean() if delta_losses else torch.zeros((), device=logits.device)
    loss_edit_type = torch.stack(edit_type_losses).mean() if edit_type_losses else torch.zeros((), device=logits.device)
    visual_sim = torch.sum(target * reference, dim=-1)
    loss_visual = torch.relu(0.05 - visual_sim).mean()
    total = loss_cvr + 0.5 * loss_delta + 0.5 * loss_hn + 0.3 * loss_ref + 0.3 * loss_edit_type + 0.05 * loss_visual
    return {
        "total": total,
        "loss_cvr": loss_cvr,
        "loss_delta": loss_delta,
        "loss_hn": loss_hn,
        "loss_ref": loss_ref,
        "loss_edit_type": loss_edit_type,
        "loss_visual": loss_visual,
    }


def _AudioDeltaAdapter(torch: Any, dim: int) -> Any:
    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_proj = torch.nn.Linear(dim, dim, bias=False)
            self.doc_proj = torch.nn.Linear(dim, dim, bias=False)
            self.edit_proj = torch.nn.Linear(dim, dim, bias=False)
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
        else:
            video = str(item).strip()
            neg_type = DEFAULT_NEGATIVE_TYPES[min(index, len(DEFAULT_NEGATIVE_TYPES) - 1)]
        if video:
            result.append({"type": neg_type, "video": video})
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


def _recall_from_scores(scores: np.ndarray, *, topk: tuple[int, ...]) -> dict[str, float]:
    total = scores.shape[0]
    order = np.argsort(-scores, axis=1, kind="stable")
    result: dict[str, float] = {}
    for k in topk:
        hits = 0
        for index in range(total):
            if index in order[index, : min(k, scores.shape[1])]:
                hits += 1
        result[f"R@{k}"] = round(hits / max(1, total), 4)
    return result


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# AudioDelta-E5 Smoke Evaluation",
        "",
        f"- eval_count: `{comparison['eval_count']}`",
        "",
        "| Method | R@1 | R@5 | R@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(f"| {row['method']} | {_fmt(row.get('R@1'))} | {_fmt(row.get('R@5'))} | {_fmt(row.get('R@10'))} |")
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
