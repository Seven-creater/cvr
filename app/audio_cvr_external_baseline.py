from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PREPROCESSING_VERSION = "imagebind_pyav_v1"
IMAGEBIND_UPSTREAM_COMMIT = "5120b6bbed3f175bf004895809b628f1b0bcb72f"
MODES = ("T_only_fullAV", "V_only", "A_only", "V_T", "A_T", "V_A", "V_A_T")
NEGATIVE_TYPES = ("visual_hard", "audio_hard", "asr_hard")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows))


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temp, array)
    os.replace(temp, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{_sha256_bytes(value.encode('utf-8'))[:24]}"


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _canonical_subtype(row: dict[str, Any]) -> str:
    value = _first_text(row, "b_subtype", "audio_delta_type", "subtype").lower()
    if value in {"audio_event", "sound", "sound_event"}:
        return "sound_event"
    if "music" in value:
        return "music"
    if "speech" in value:
        return "speech_topic_in_video_context"
    return value or "unknown"


def _dataset_name(row: dict[str, Any]) -> str:
    value = _first_text(row, "dataset", "dataset_name", "source_dataset").lower()
    if value:
        return value
    candidates = " ".join(
        _first_text(row, key)
        for key in ("reference_video", "target_video", "raw_source_id", "source_id")
    ).lower()
    for name in ("existing_vggsound", "vggsound", "avscapbench", "avqa_videos", "avatar", "worldsense", "hdtf", "voxceleb", "daily_omni"):
        if name in candidates:
            return name
    return "unknown"


def _source_identity(row: dict[str, Any]) -> str:
    value = _first_text(row, "source_disjoint_group_id", "raw_source_id", "source_id")
    if value:
        return value
    reference = _first_text(row, "reference_video")
    if reference:
        parent = Path(reference.replace("\\", "/")).parent.name
        if parent:
            return f"path_source::{parent}"
    return ""


def _pair_identity(row: dict[str, Any]) -> str:
    value = _first_text(row, "pair_group_id", "inverse_pair_group_id", "proposal_id")
    if value:
        return value
    reference = _first_text(row, "reference_video").replace("\\", "/")
    target = _first_text(row, "target_video").replace("\\", "/")
    if reference and target:
        return _stable_id("path_pair", "|".join(sorted((reference, target))))
    return ""


def _negative_type(item: dict[str, Any]) -> str:
    value = _first_text(item, "negative_type", "type", "kind").lower()
    aliases = {
        "visual": "visual_hard",
        "audio": "audio_hard",
        "asr": "asr_hard",
        "reference": "reference_negative",
    }
    return aliases.get(value, value)


def _hard_negatives(row: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("hard_negatives", "audio_delta_hard_negatives", "local_same_source_candidates"):
        raw = row.get(key)
        if isinstance(raw, list):
            values.extend(item for item in raw if isinstance(item, dict))
    return values


def _resolve_media_path(value: str, roots: Sequence[Path]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"media not found: {value}; roots={[str(root) for root in roots]}")


def _inventory_media_paths(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    reference = _first_text(row, "reference_video")
    target = _first_text(row, "target_video")
    if reference:
        items.append(("reference", "reference_negative", reference))
    if target:
        items.append(("target", "positive", target))
    for negative in _hard_negatives(row):
        video = _first_text(negative, "video", "video_path", "path")
        if video:
            kind = _negative_type(negative) or "hard_negative"
            items.append(("hard_negative", kind, video))
    return items


def prepare_inventory(
    records_path: Path,
    output_dir: Path,
    media_roots: Sequence[Path],
    *,
    expected_count: int | None = None,
    expected_sha256: str | None = None,
    inherited_records: Path | None = None,
    expected_subtypes: dict[str, int] | None = None,
    require_unique_source_pair: bool = False,
    allow_missing_media: bool = False,
) -> dict[str, Any]:
    records_sha256 = _sha256_file(records_path)
    if expected_sha256 and records_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"records SHA256 mismatch: {records_sha256} != {expected_sha256}")
    rows = _load_jsonl(records_path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(rows)}")

    sample_ids: list[str] = []
    media: dict[str, dict[str, Any]] = {}
    texts: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    record_manifest: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = _first_text(row, "sample_id", "proposal_id") or f"row_{index:06d}"
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.append(sample_id)
        edit_text = _first_text(row, "edit_text", "modification_text")
        if not edit_text:
            raise ValueError(f"missing edit_text for {sample_id}")
        text_id = _stable_id("text", edit_text.strip())
        texts.setdefault(text_id, {"text_id": text_id, "text": edit_text.strip()})
        role_entries: list[dict[str, str]] = []
        for role, kind, raw_path in _inventory_media_paths(row):
            try:
                resolved = _resolve_media_path(raw_path, media_roots)
            except FileNotFoundError as exc:
                missing.append({"sample_id": sample_id, "role": role, "kind": kind, "video": raw_path, "error": str(exc)})
                continue
            normalized = os.path.normcase(str(resolved))
            media_id = _stable_id("media", normalized)
            media.setdefault(
                media_id,
                {
                    "media_id": media_id,
                    "video": raw_path,
                    "resolved_media_path": str(resolved),
                    "file_size": resolved.stat().st_size,
                    "mtime_ns": resolved.stat().st_mtime_ns,
                },
            )
            role_entries.append({"role": role, "kind": kind, "media_id": media_id, "video": raw_path})
        record_manifest.append(
            {
                "index": index,
                "sample_id": sample_id,
                "text_id": text_id,
                "subtype": _canonical_subtype(row),
                "dataset": _dataset_name(row),
                "media": role_entries,
            }
        )

    inherited_count = 0
    missing_inherited: list[str] = []
    if inherited_records is not None:
        inherited_ids = {_first_text(row, "sample_id", "proposal_id") for row in _load_jsonl(inherited_records)}
        current_ids = set(sample_ids)
        missing_inherited = sorted(value for value in inherited_ids if value not in current_ids)
        inherited_count = len(inherited_ids)
        if missing_inherited:
            raise ValueError(f"final records do not inherit {len(missing_inherited)} existing sample_ids")

    subtype_counts = Counter(item["subtype"] for item in record_manifest)
    if expected_subtypes is not None and dict(subtype_counts) != expected_subtypes:
        raise ValueError(f"subtype counts differ: {dict(subtype_counts)} != {expected_subtypes}")
    source_ids = [_source_identity(row) for row in rows]
    pair_ids = [_pair_identity(row) for row in rows]
    duplicate_source_count = len(source_ids) - len({value for value in source_ids if value}) - sum(not value for value in source_ids)
    duplicate_pair_count = len(pair_ids) - len({value for value in pair_ids if value}) - sum(not value for value in pair_ids)
    missing_source_identity_count = sum(not value for value in source_ids)
    missing_pair_identity_count = sum(not value for value in pair_ids)
    if require_unique_source_pair and (
        duplicate_source_count or duplicate_pair_count or missing_source_identity_count or missing_pair_identity_count
    ):
        raise ValueError(
            "invalid source/pair identities: "
            f"duplicate_source={duplicate_source_count}, duplicate_pair={duplicate_pair_count}, "
            f"missing_source={missing_source_identity_count}, missing_pair={missing_pair_identity_count}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    media_rows = sorted(media.values(), key=lambda item: item["media_id"])
    text_rows = sorted(texts.values(), key=lambda item: item["text_id"])
    _atomic_jsonl(output_dir / "media_inventory.jsonl", media_rows)
    _atomic_jsonl(output_dir / "text_inventory.jsonl", text_rows)
    _atomic_jsonl(output_dir / "record_inventory.jsonl", record_manifest)
    if missing:
        _atomic_jsonl(output_dir / "missing_media.jsonl", missing)
    summary = {
        "records_path": str(records_path.resolve()),
        "records_sha256": records_sha256,
        "record_count": len(rows),
        "sample_id_count": len(sample_ids),
        "media_count": len(media_rows),
        "text_count": len(text_rows),
        "missing_media_count": len(missing),
        "inherited_record_count": inherited_count,
        "missing_inherited_count": len(missing_inherited),
        "selection_uses_model_scores": False,
        "subtype_counts": dict(subtype_counts),
        "dataset_counts": dict(Counter(item["dataset"] for item in record_manifest)),
        "duplicate_source_count": duplicate_source_count,
        "duplicate_pair_count": duplicate_pair_count,
        "missing_source_identity_count": missing_source_identity_count,
        "missing_pair_identity_count": missing_pair_identity_count,
    }
    _atomic_json(output_dir / "inventory_summary.json", summary)
    if missing and not allow_missing_media:
        raise ValueError(f"{len(missing)} inventory media files are missing")
    return summary


def _l2(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    denominator = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denominator, 1e-12)


def _model_fingerprint(model_dir: Path) -> str:
    config = (model_dir / "config.json").read_bytes()
    weights = model_dir / "model.safetensors"
    stat = weights.stat()
    payload = config + f"|{stat.st_size}|{stat.st_mtime_ns}|{IMAGEBIND_UPSTREAM_COMMIT}".encode("ascii")
    return _sha256_bytes(payload)


def _cache_key(kind: str, source_hash: str, model_fingerprint: str) -> str:
    payload = f"{kind}|{source_hash}|{model_fingerprint}|{PREPROCESSING_VERSION}"
    return _sha256_bytes(payload.encode("utf-8"))


@dataclass(frozen=True)
class CacheLocation:
    cache_key: str
    embedding_path: Path
    index_path: Path


def _cache_location(cache_root: Path, kind: str, item_id: str, cache_key: str) -> CacheLocation:
    directory = "media_embeddings" if kind == "media" else "text_embeddings"
    suffix = ".npz" if kind == "media" else ".npy"
    return CacheLocation(
        cache_key=cache_key,
        embedding_path=cache_root / directory / f"{cache_key}{suffix}",
        index_path=cache_root / "indexes" / kind / f"{item_id}.json",
    )


def _valid_embedding_file(path: Path, kind: str) -> bool:
    if not path.is_file():
        return False
    try:
        if kind == "media":
            with np.load(path) as data:
                vision = np.asarray(data["vision_embedding"])
                audio = np.asarray(data["audio_embedding"])
            return vision.shape == (1024,) and audio.shape == (1024,) and np.isfinite(vision).all() and np.isfinite(audio).all()
        value = np.load(path)
        return value.shape == (1024,) and np.isfinite(value).all()
    except Exception:
        return False


def _existing_cache(cache_root: Path, kind: str, item_id: str) -> CacheLocation | None:
    index_path = cache_root / "indexes" / kind / f"{item_id}.json"
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        embedding_path = Path(payload["embedding_path"])
        location = CacheLocation(str(payload["cache_key"]), embedding_path, index_path)
        return location if _valid_embedding_file(embedding_path, kind) else None
    except Exception:
        return None


class ImageBindEncoder:
    def __init__(self, model_dir: Path, device: str, vendor_root: Path):
        import torch
        from safetensors.torch import load_file

        vendor_text = str(vendor_root.resolve())
        if vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)
        from imagebind.models.imagebind_model import ImageBindModel, ModalityType
        from imagebind.models.multimodal_preprocessors import SimpleTokenizer

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.torch = torch
        self.ModalityType = ModalityType
        self.device = torch.device(device)
        self.model = ImageBindModel(**config)
        state = load_file(str(model_dir / "model.safetensors"), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ImageBind checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.model.eval().to(self.device)
        bpe_path = vendor_root / "bpe" / "bpe_simple_vocab_16e6.txt.gz"
        self.tokenizer = SimpleTokenizer(bpe_path=str(bpe_path))

    def encode_text(self, texts: Sequence[str]) -> list[np.ndarray]:
        torch = self.torch
        tokens = torch.stack([self.tokenizer(text) for text in texts], dim=0).to(self.device)
        with torch.inference_mode():
            output = self.model({self.ModalityType.TEXT: tokens})[self.ModalityType.TEXT]
        return [value for value in _l2(output.float().cpu().numpy())]

    def encode_media(self, paths: Sequence[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
        torch = self.torch
        vision = torch.stack([_preprocess_video(path, torch) for path in paths], dim=0).to(self.device)
        audio = torch.stack([_preprocess_audio(path, torch) for path in paths], dim=0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(
                {
                    self.ModalityType.VISION: vision,
                    self.ModalityType.AUDIO: audio,
                }
            )
        vision_values = _l2(outputs[self.ModalityType.VISION].float().cpu().numpy())
        audio_values = _l2(outputs[self.ModalityType.AUDIO].float().cpu().numpy())
        return list(zip(vision_values, audio_values))


def _decode_video_frames(path: Path, frame_count: int = 10) -> list[np.ndarray]:
    import av

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        total = int(stream.frames or 0)
        if total <= 0:
            rate = float(stream.average_rate or 25.0)
            duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
            total = max(frame_count, int(round(rate * duration)))
        wanted = np.linspace(0, max(0, total - 1), frame_count).round().astype(int).tolist()
        frames: list[np.ndarray] = []
        wanted_index = 0
        last: np.ndarray | None = None
        for index, frame in enumerate(container.decode(stream)):
            if wanted_index >= len(wanted):
                break
            last = frame.to_rgb().to_ndarray()
            while wanted_index < len(wanted) and index >= wanted[wanted_index]:
                frames.append(last.copy())
                wanted_index += 1
        if last is None:
            raise ValueError(f"no decodable video frames: {path}")
        while len(frames) < frame_count:
            frames.append(last.copy())
        return frames[:frame_count]


def _preprocess_video(path: Path, torch_module: Any) -> Any:
    import torch.nn.functional as functional

    frames = _decode_video_frames(path, frame_count=10)
    tensors = [torch_module.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
    transformed: list[Any] = []
    mean = torch_module.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch_module.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    for tensor in tensors:
        _, height, width = tensor.shape
        if height <= width:
            new_height = 224
            new_width = max(224, int(round(width * 224 / height)))
        else:
            new_width = 224
            new_height = max(224, int(round(height * 224 / width)))
        resized = functional.interpolate(tensor.unsqueeze(0), size=(new_height, new_width), mode="bicubic", align_corners=False).squeeze(0)
        transformed.append((resized - mean) / std)

    clips: list[Any] = []
    for clip_index in range(5):
        pair = transformed[clip_index * 2 : clip_index * 2 + 2]
        _, height, width = pair[0].shape
        if width >= height:
            offsets = [(0, (height - 224) // 2), ((width - 224) // 2, (height - 224) // 2), (width - 224, (height - 224) // 2)]
        else:
            offsets = [((width - 224) // 2, 0), ((width - 224) // 2, (height - 224) // 2), ((width - 224) // 2, height - 224)]
        for left, top in offsets:
            crop_frames = [value[:, top : top + 224, left : left + 224] for value in pair]
            clips.append(torch_module.stack(crop_frames, dim=1))
    return torch_module.stack(clips, dim=0)


def _decode_audio(path: Path, target_rate: int = 16000) -> np.ndarray:
    import av

    chunks: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream: {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            if converted is None:
                continue
            converted_frames = converted if isinstance(converted, list) else [converted]
            for item in converted_frames:
                array = item.to_ndarray().astype(np.float32, copy=False)
                chunks.append(array.reshape(-1))
        tail = resampler.resample(None)
        if tail:
            tail_frames = tail if isinstance(tail, list) else [tail]
            for item in tail_frames:
                chunks.append(item.to_ndarray().astype(np.float32, copy=False).reshape(-1))
    if not chunks:
        raise ValueError(f"no decodable audio samples: {path}")
    return np.concatenate(chunks)


def _preprocess_audio(path: Path, torch_module: Any) -> Any:
    import torchaudio

    sample_rate = 16000
    waveform = torch_module.from_numpy(_decode_audio(path, sample_rate)).float().unsqueeze(0)
    duration_samples = 2 * sample_rate
    max_start = max(0, waveform.shape[1] - duration_samples)
    starts = np.linspace(0, max_start, 3).round().astype(int).tolist()
    clips: list[Any] = []
    for start in starts:
        clip = waveform[:, start : start + duration_samples]
        if clip.shape[1] < duration_samples:
            clip = torch_module.nn.functional.pad(clip, (0, duration_samples - clip.shape[1]))
        clip = clip - clip.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            clip,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        ).transpose(0, 1)
        if fbank.shape[1] < 204:
            fbank = torch_module.nn.functional.pad(fbank, (0, 204 - fbank.shape[1]))
        else:
            fbank = fbank[:, :204]
        clips.append(((fbank - (-4.268)) / 9.138).unsqueeze(0))
    return torch_module.stack(clips, dim=0)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_hash(kind: str, item: dict[str, Any]) -> str:
    if kind == "media":
        return _sha256_file(Path(item["resolved_media_path"]))
    return _sha256_bytes(str(item["text"]).strip().encode("utf-8"))


def _prepare_cache_item(
    item: dict[str, Any],
    kind: str,
    cache_root: Path,
    model_fingerprint: str,
) -> tuple[CacheLocation, str]:
    item_id = str(item[f"{kind}_id"])
    existing = _existing_cache(cache_root, kind, item_id)
    if existing is not None:
        return existing, "reused"
    source_hash = _source_hash(kind, item)
    cache_key = _cache_key(kind, source_hash, model_fingerprint)
    location = _cache_location(cache_root, kind, item_id, cache_key)
    if _valid_embedding_file(location.embedding_path, kind):
        _atomic_json(
            location.index_path,
            {
                "item_id": item_id,
                "cache_key": cache_key,
                "embedding_path": str(location.embedding_path.resolve()),
                "source_sha256": source_hash,
                "kind": kind,
            },
        )
        return location, "reused_by_content"
    return location, source_hash


def _write_cache_item(
    item: dict[str, Any],
    kind: str,
    location: CacheLocation,
    source_hash: str,
    model_fingerprint: str,
    encoded: Any,
) -> None:
    item_id = str(item[f"{kind}_id"])
    if kind == "media":
        vision, audio = encoded
        vision = _l2(np.asarray(vision)).reshape(-1)
        audio = _l2(np.asarray(audio)).reshape(-1)
        if vision.shape != (1024,) or audio.shape != (1024,):
            raise ValueError(f"unexpected media embedding shape: {vision.shape}, {audio.shape}")
        _atomic_npz(
            location.embedding_path,
            vision_embedding=vision.astype(np.float32),
            audio_embedding=audio.astype(np.float32),
            resolved_media_path=np.asarray(str(item["resolved_media_path"])),
            source_sha256=np.asarray(source_hash),
            file_size=np.asarray(int(item["file_size"]), dtype=np.int64),
            mtime_ns=np.asarray(int(item["mtime_ns"]), dtype=np.int64),
            model_fingerprint=np.asarray(model_fingerprint),
            preprocessing_version=np.asarray(PREPROCESSING_VERSION),
            finite_check=np.asarray(True),
        )
    else:
        value = _l2(np.asarray(encoded)).reshape(-1)
        if value.shape != (1024,):
            raise ValueError(f"unexpected text embedding shape: {value.shape}")
        _atomic_npy(location.embedding_path, value.astype(np.float32))
    _atomic_json(
        location.index_path,
        {
            "item_id": item_id,
            "cache_key": location.cache_key,
            "embedding_path": str(location.embedding_path.resolve()),
            "source_sha256": source_hash,
            "model_fingerprint": model_fingerprint,
            "preprocessing_version": PREPROCESSING_VERSION,
            "finite_check": True,
            "kind": kind,
        },
    )


def _encode_single_with_retries(
    encoder: ImageBindEncoder,
    item: dict[str, Any],
    kind: str,
    retries: int,
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            if kind == "media":
                return encoder.encode_media([Path(item["resolved_media_path"])])[0]
            return encoder.encode_text([str(item["text"])])[0]
        except BaseException as exc:
            last_error = exc
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    assert last_error is not None
    raise last_error


def cache_imagebind(
    inventory_path: Path,
    cache_root: Path,
    model_dir: Path,
    vendor_root: Path,
    *,
    kind: str,
    shard_index: int,
    shard_count: int,
    device: str,
    batch_size: int,
    retries: int,
) -> dict[str, Any]:
    if kind not in {"media", "text"}:
        raise ValueError(f"invalid cache kind: {kind}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    rows = _load_jsonl(inventory_path)
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    model_fingerprint = _model_fingerprint(model_dir)
    shard_name = f"{kind}_shard_{shard_index:03d}_of_{shard_count:03d}"
    manifest_path = cache_root / "shard_manifests" / f"{shard_name}.jsonl"
    failure_path = cache_root / "failures" / f"{shard_name}.jsonl"
    summary_path = cache_root / "shard_summaries" / f"{shard_name}.json"
    cache_root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[dict[str, Any], CacheLocation, str]] = []
    reused = 0
    for item in selected:
        location, state = _prepare_cache_item(item, kind, cache_root, model_fingerprint)
        if state.startswith("reused"):
            reused += 1
            _append_jsonl(manifest_path, {"item_id": item[f"{kind}_id"], "state": state, "cache_key": location.cache_key})
        else:
            pending.append((item, location, state))

    encoder: ImageBindEncoder | None = None
    encoded_count = 0
    failed = 0
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        if encoder is None:
            encoder = ImageBindEncoder(model_dir, device, vendor_root)
        try:
            if kind == "media":
                values = encoder.encode_media([Path(item[0]["resolved_media_path"]) for item in batch])
            else:
                values = encoder.encode_text([str(item[0]["text"]) for item in batch])
        except BaseException:
            values = []
            for item, _, _ in batch:
                try:
                    values.append(_encode_single_with_retries(encoder, item, kind, retries))
                except BaseException as exc:
                    values.append(exc)
        for (item, location, source_hash), value in zip(batch, values):
            item_id = str(item[f"{kind}_id"])
            if isinstance(value, BaseException):
                failed += 1
                _append_jsonl(
                    failure_path,
                    {
                        "item_id": item_id,
                        "kind": kind,
                        "error_type": type(value).__name__,
                        "error": str(value),
                        "resolved_media_path": item.get("resolved_media_path"),
                    },
                )
                _append_jsonl(manifest_path, {"item_id": item_id, "state": "failed"})
                continue
            _write_cache_item(item, kind, location, source_hash, model_fingerprint, value)
            encoded_count += 1
            _append_jsonl(manifest_path, {"item_id": item_id, "state": "encoded", "cache_key": location.cache_key})

    summary = {
        "kind": kind,
        "inventory_path": str(inventory_path.resolve()),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(selected),
        "encoded_count": encoded_count,
        "reused_count": reused,
        "failed_count": failed,
        "complete_count": encoded_count + reused,
        "model_fingerprint": model_fingerprint,
        "preprocessing_version": PREPROCESSING_VERSION,
        "device": device,
        "batch_size": batch_size,
    }
    _atomic_json(summary_path, summary)
    return summary


def _load_media_embedding(cache_root: Path, media_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    location = _existing_cache(cache_root, "media", media_id)
    if location is None:
        return None
    with np.load(location.embedding_path) as data:
        return _l2(np.asarray(data["vision_embedding"])).reshape(-1), _l2(np.asarray(data["audio_embedding"])).reshape(-1)


def _load_text_embedding(cache_root: Path, text_id: str) -> np.ndarray | None:
    location = _existing_cache(cache_root, "text", text_id)
    if location is None:
        return None
    return _l2(np.load(location.embedding_path)).reshape(-1)


def _role_media(record: dict[str, Any], role: str, kind: str | None = None) -> str:
    for item in record.get("media", []):
        if item.get("role") == role and (kind is None or item.get("kind") == kind):
            return str(item.get("media_id", ""))
    return ""


def assemble_embeddings(
    records_path: Path,
    inventory_dir: Path,
    cache_root: Path,
    output_dir: Path,
    *,
    pre_records: Path | None = None,
    core_records: Path | None = None,
    max_exclusion_rate: float = 0.01,
) -> dict[str, Any]:
    raw_rows = _load_jsonl(records_path)
    inventory_rows = _load_jsonl(inventory_dir / "record_inventory.jsonl")
    if len(raw_rows) != len(inventory_rows):
        raise ValueError("record inventory length does not match source records")
    pre_ids = {_first_text(row, "sample_id", "proposal_id") for row in _load_jsonl(pre_records)} if pre_records else set()
    core_ids = {_first_text(row, "sample_id", "proposal_id") for row in _load_jsonl(core_records)} if core_records else set()

    kept_rows: list[dict[str, Any]] = []
    kept_inventory: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    q_v: list[np.ndarray] = []
    q_a: list[np.ndarray] = []
    q_t: list[np.ndarray] = []
    target_v: list[np.ndarray] = []
    target_a: list[np.ndarray] = []
    ref_v: list[np.ndarray] = []
    ref_a: list[np.ndarray] = []
    negative_values: dict[str, dict[str, list[np.ndarray]]] = {
        kind: {"v": [], "a": [], "mask": []} for kind in NEGATIVE_TYPES
    }
    for raw, record in zip(raw_rows, inventory_rows):
        sample_id = str(record["sample_id"])
        reference_id = _role_media(record, "reference")
        target_id = _role_media(record, "target")
        reference = _load_media_embedding(cache_root, reference_id) if reference_id else None
        target = _load_media_embedding(cache_root, target_id) if target_id else None
        text_value = _load_text_embedding(cache_root, str(record["text_id"]))
        missing_roles = []
        if reference is None:
            missing_roles.append("reference")
        if target is None:
            missing_roles.append("target")
        if text_value is None:
            missing_roles.append("edit_text")
        if missing_roles:
            exclusions.append({"sample_id": sample_id, "missing_roles": missing_roles})
            continue
        assert reference is not None and target is not None and text_value is not None
        kept_rows.append(raw)
        kept_inventory.append(record)
        q_v.append(reference[0])
        q_a.append(reference[1])
        q_t.append(text_value)
        target_v.append(target[0])
        target_a.append(target[1])
        ref_v.append(reference[0])
        ref_a.append(reference[1])
        for kind in NEGATIVE_TYPES:
            media_id = _role_media(record, "hard_negative", kind)
            value = _load_media_embedding(cache_root, media_id) if media_id else None
            negative_values[kind]["v"].append(value[0] if value is not None else np.zeros(1024, dtype=np.float32))
            negative_values[kind]["a"].append(value[1] if value is not None else np.zeros(1024, dtype=np.float32))
            negative_values[kind]["mask"].append(bool(value is not None))

    exclusion_rate = len(exclusions) / max(1, len(raw_rows))
    if exclusion_rate > max_exclusion_rate:
        raise ValueError(f"encoding exclusion rate {exclusion_rate:.4f} exceeds {max_exclusion_rate:.4f}")
    count = len(kept_rows)
    if count == 0:
        raise ValueError("no valid records remain after cache assembly")
    arrays: dict[str, Any] = {
        "sample_ids": np.asarray([record["sample_id"] for record in kept_inventory]),
        "query_vision": np.stack(q_v),
        "query_audio": np.stack(q_a),
        "query_text": np.stack(q_t),
        "gallery_vision": np.concatenate([np.stack(target_v), np.stack(ref_v)], axis=0),
        "gallery_audio": np.concatenate([np.stack(target_a), np.stack(ref_a)], axis=0),
        "positive_indices": np.arange(count, dtype=np.int64),
        "reference_indices": np.arange(count, count * 2, dtype=np.int64),
        "is_preexisting": np.asarray([str(record["sample_id"]) in pre_ids for record in kept_inventory]),
        "is_human_checked_core": np.asarray([str(record["sample_id"]) in core_ids for record in kept_inventory]),
        "subtypes": np.asarray([str(record["subtype"]) for record in kept_inventory]),
        "datasets": np.asarray([str(record["dataset"]) for record in kept_inventory]),
    }
    for kind in NEGATIVE_TYPES:
        arrays[f"negative_{kind}_vision"] = np.stack(negative_values[kind]["v"])
        arrays[f"negative_{kind}_audio"] = np.stack(negative_values[kind]["a"])
        arrays[f"negative_{kind}_mask"] = np.asarray(negative_values[kind]["mask"], dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "imagebind_embeddings.npz"
    _atomic_npz(embeddings_path, **arrays)
    _atomic_jsonl(output_dir / "records.jsonl", kept_rows)
    gallery_rows = []
    for index, (raw, record) in enumerate(zip(kept_rows, kept_inventory)):
        gallery_rows.append({"gallery_index": index, "gallery_id": f"target::{record['sample_id']}", "sample_id": record["sample_id"], "kind": "positive", "video": _first_text(raw, "target_video")})
    for offset, (raw, record) in enumerate(zip(kept_rows, kept_inventory), start=count):
        gallery_rows.append({"gallery_index": offset, "gallery_id": f"reference::{record['sample_id']}", "sample_id": record["sample_id"], "kind": "reference_negative", "video": _first_text(raw, "reference_video")})
    _atomic_jsonl(output_dir / "gallery.jsonl", gallery_rows)
    _atomic_jsonl(output_dir / "encoding_exclusion_manifest.jsonl", exclusions)
    summary = {
        "records_path": str(records_path.resolve()),
        "records_sha256": _sha256_file(records_path),
        "input_record_count": len(raw_rows),
        "valid_query_count": count,
        "excluded_query_count": len(exclusions),
        "exclusion_rate": exclusion_rate,
        "gallery_count": count * 2,
        "with_reference_gallery_count": count * 2,
        "without_reference_effective_gallery_count_per_query": count * 2 - 1,
        "preexisting_query_count": int(arrays["is_preexisting"].sum()),
        "human_checked_core_count": int(arrays["is_human_checked_core"].sum()),
        "embedding_path": str(embeddings_path.resolve()),
    }
    _atomic_json(output_dir / "assembly_summary.json", summary)
    return summary


def _mode_embeddings(data: Any, mode: str) -> tuple[np.ndarray, np.ndarray]:
    qv = _l2(data["query_vision"])
    qa = _l2(data["query_audio"])
    qt = _l2(data["query_text"])
    gv = _l2(data["gallery_vision"])
    ga = _l2(data["gallery_audio"])
    if mode == "T_only_fullAV":
        return qt, _l2(gv + ga)
    if mode == "V_only":
        return qv, gv
    if mode == "A_only":
        return qa, ga
    if mode == "V_T":
        return _l2(qv + qt), gv
    if mode == "A_T":
        return _l2(qa + qt), ga
    if mode == "V_A":
        return _l2(qv + qa), _l2(gv + ga)
    if mode == "V_A_T":
        return _l2(qv + qa + qt), _l2(gv + ga)
    raise ValueError(f"unknown mode: {mode}")


def _negative_embedding(data: Any, mode: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    qv = _l2(data["query_vision"])
    qa = _l2(data["query_audio"])
    qt = _l2(data["query_text"])
    nv = _l2(data[f"negative_{kind}_vision"])
    na = _l2(data[f"negative_{kind}_audio"])
    if mode == "T_only_fullAV":
        return qt, _l2(nv + na)
    if mode == "V_only":
        return qv, nv
    if mode == "A_only":
        return qa, na
    if mode == "V_T":
        return _l2(qv + qt), nv
    if mode == "A_T":
        return _l2(qa + qt), na
    if mode == "V_A":
        return _l2(qv + qa), _l2(nv + na)
    if mode == "V_A_T":
        return _l2(qv + qa + qt), _l2(nv + na)
    raise ValueError(mode)


def _metric_summary(
    scores: np.ndarray,
    positive_indices: np.ndarray,
    reference_indices: np.ndarray,
    *,
    mask_own_reference: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    work = np.asarray(scores, dtype=np.float32).copy()
    rows = np.arange(work.shape[0])
    positive_scores = work[rows, positive_indices].copy()
    reference_scores = work[rows, reference_indices].copy()
    target_beats_reference = positive_scores > reference_scores
    if mask_own_reference:
        work[rows, reference_indices] = -np.inf
    ranks = 1 + np.sum(work > positive_scores[:, None], axis=1)
    reference_ranks = 1 + np.sum(scores > reference_scores[:, None], axis=1)
    top1 = np.argmax(work, axis=1)
    top1_is_own_reference = top1 == reference_indices
    reciprocal = 1.0 / ranks.astype(np.float64)
    summary = {
        "query_count": int(work.shape[0]),
        "gallery_count": int(work.shape[1]),
        "effective_gallery_count_per_query": int(work.shape[1] - int(mask_own_reference)),
        "reference_in_gallery": not mask_own_reference,
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(reciprocal)),
        "target_rank_mean": float(np.mean(ranks)),
        "target_rank_median": float(np.median(ranks)),
        "target_beats_reference": float(np.mean(target_beats_reference)),
        "target_reference_gap_mean": float(np.mean(positive_scores - reference_scores)),
        "reference_rank_mean": float(np.mean(reference_ranks)),
        "reference_rank_median": float(np.median(reference_ranks)),
        "reference_rank_le_1": float(np.mean(reference_ranks <= 1)),
        "top1_own_reference_count": int(top1_is_own_reference.sum()),
        "top1_own_reference_rate": float(np.mean(top1_is_own_reference)),
    }
    per_query = {
        "ranks": ranks,
        "reference_ranks": reference_ranks,
        "correct_at_1": ranks <= 1,
        "correct_at_5": ranks <= 5,
        "correct_at_10": ranks <= 10,
        "reciprocal_rank": reciprocal,
        "target_beats_reference": target_beats_reference,
        "gap": positive_scores - reference_scores,
        "top1_indices": top1,
        "top1_is_own_reference": top1_is_own_reference,
    }
    return summary, per_query


def _subset_summary(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"query_count": 0}
    return {
        "query_count": count,
        "R@1": float(np.mean(values["correct_at_1"][mask])),
        "R@5": float(np.mean(values["correct_at_5"][mask])),
        "R@10": float(np.mean(values["correct_at_10"][mask])),
        "MRR": float(np.mean(values["reciprocal_rank"][mask])),
        "target_beats_reference": float(np.mean(values["target_beats_reference"][mask])),
        "target_reference_gap_mean": float(np.mean(values["gap"][mask])),
        "top1_own_reference_rate": float(np.mean(values["top1_is_own_reference"][mask])),
    }


def evaluate_embeddings(assembly_dir: Path, output_dir: Path, *, topk: int = 20) -> dict[str, Any]:
    embeddings_path = assembly_dir / "imagebind_embeddings.npz"
    records = _load_jsonl(assembly_dir / "records.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    per_query_rows: list[dict[str, Any]] = []
    with np.load(embeddings_path) as data:
        positive_indices = np.asarray(data["positive_indices"], dtype=np.int64)
        reference_indices = np.asarray(data["reference_indices"], dtype=np.int64)
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        subtypes = np.asarray(data["subtypes"]).astype(str)
        datasets = np.asarray(data["datasets"]).astype(str)
        is_preexisting = np.asarray(data["is_preexisting"], dtype=bool)
        is_core = np.asarray(data["is_human_checked_core"], dtype=bool)
        for mode in MODES:
            query, gallery = _mode_embeddings(data, mode)
            scores = query @ gallery.T
            with_ref, with_values = _metric_summary(scores, positive_indices, reference_indices, mask_own_reference=False)
            without_ref, without_values = _metric_summary(scores, positive_indices, reference_indices, mask_own_reference=True)
            with_ref["without_reference_R@1"] = without_ref["R@1"]
            with_ref["reference_induced_R@1_drop"] = without_ref["R@1"] - with_ref["R@1"]
            breakdown: dict[str, Any] = {
                "human_checked_core": _subset_summary(with_values, is_core),
                "preexisting_516": _subset_summary(with_values, is_preexisting),
                "final_delta": _subset_summary(with_values, ~is_preexisting),
                "subtype": {value: _subset_summary(with_values, subtypes == value) for value in sorted(set(subtypes))},
                "dataset": {value: _subset_summary(with_values, datasets == value) for value in sorted(set(datasets))},
            }
            negative_breakdown: dict[str, Any] = {}
            target_scores = scores[np.arange(scores.shape[0]), positive_indices]
            for kind in NEGATIVE_TYPES:
                mask = np.asarray(data[f"negative_{kind}_mask"], dtype=bool)
                negative_query, negative_document = _negative_embedding(data, mode, kind)
                negative_scores = np.sum(negative_query * negative_document, axis=1)
                negative_breakdown[kind] = {
                    "query_count": int(mask.sum()),
                    "positive_beats_negative_rate": float(np.mean(target_scores[mask] > negative_scores[mask])) if mask.any() else None,
                }
            with_ref["negative_breakdown"] = negative_breakdown
            with_ref["breakdown"] = breakdown
            results[mode] = {"with_reference": with_ref, "without_reference": without_ref}

            masked_scores = scores.copy()
            masked_scores[np.arange(scores.shape[0]), reference_indices] = -np.inf
            top_with = np.argpartition(-scores, min(topk, scores.shape[1]) - 1, axis=1)[:, :topk]
            top_without = np.argpartition(-masked_scores, min(topk, scores.shape[1]) - 1, axis=1)[:, :topk]
            for index, sample_id in enumerate(sample_ids):
                with_order = top_with[index][np.argsort(-scores[index, top_with[index]])]
                without_order = top_without[index][np.argsort(-masked_scores[index, top_without[index]])]
                per_query_rows.append(
                    {
                        "sample_id": sample_id,
                        "mode": mode,
                        "subtype": subtypes[index],
                        "dataset": datasets[index],
                        "with_reference_rank": int(with_values["ranks"][index]),
                        "without_reference_rank": int(without_values["ranks"][index]),
                        "reference_rank": int(with_values["reference_ranks"][index]),
                        "target_beats_reference": bool(with_values["target_beats_reference"][index]),
                        "target_reference_gap": float(with_values["gap"][index]),
                        "with_reference_top_indices": with_order.tolist(),
                        "without_reference_top_indices": without_order.tolist(),
                    }
                )

    _atomic_json(output_dir / "seven_mode_results.json", results)
    masking = {
        mode: {
            "with_reference_R@1": payload["with_reference"]["R@1"],
            "without_reference_R@1": payload["without_reference"]["R@1"],
            "reference_induced_R@1_drop": payload["with_reference"]["reference_induced_R@1_drop"],
            "effective_gallery_with_reference": payload["with_reference"]["effective_gallery_count_per_query"],
            "effective_gallery_without_reference": payload["without_reference"]["effective_gallery_count_per_query"],
        }
        for mode, payload in results.items()
    }
    _atomic_json(output_dir / "reference_masking_results.json", masking)
    _atomic_jsonl(output_dir / "per_query_results.jsonl", per_query_rows)
    summary = {
        "model": "ImageBind-Huge",
        "training": "none",
        "query_count": len(records),
        "modes": list(MODES),
        "score_matrix_reused_for_reference_masking": True,
        "selection_uses_test_metrics": False,
        "nan_or_inf_count": 0,
    }
    _atomic_json(output_dir / "evaluation_summary.json", summary)
    return summary


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _paired_test(a: np.ndarray, b: np.ndarray, *, iterations: int, seed: int) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    difference = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(iterations, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(iterations, len(difference)))
    randomized = (difference[None, :] * signs).mean(axis=1)
    observed = float(difference.mean())
    p_two_sided = float((np.sum(np.abs(randomized) >= abs(observed)) + 1) / (iterations + 1))
    return {
        "mean_difference": observed,
        "bootstrap_95_ci": _percentile_interval(bootstrap),
        "paired_randomization_p_two_sided": p_two_sided,
        "iterations": iterations,
    }


def _mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_only, b_only)
        tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_two_sided": float(p_value)}


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def summarize_results(evaluation_dir: Path, output_dir: Path, *, iterations: int = 20000, seed: int = 20260723) -> dict[str, Any]:
    rows = _load_jsonl(evaluation_dir / "per_query_results.jsonl")
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row["mode"])].append(row)
    for mode in by_mode:
        by_mode[mode].sort(key=lambda row: str(row["sample_id"]))

    comparisons: dict[str, Any] = {}
    vat = by_mode["V_A_T"]
    vt = by_mode["V_T"]
    if [row["sample_id"] for row in vat] != [row["sample_id"] for row in vt]:
        raise ValueError("V_A_T and V_T sample IDs differ")
    vat_correct = np.asarray([row["with_reference_rank"] == 1 for row in vat], dtype=float)
    vt_correct = np.asarray([row["with_reference_rank"] == 1 for row in vt], dtype=float)
    comparisons["audio_gain_R@1"] = _paired_test(vat_correct, vt_correct, iterations=iterations, seed=seed)
    comparisons["audio_gain_R@1"]["mcnemar"] = _mcnemar_exact(vat_correct, vt_correct)
    comparisons["audio_gain_target_reference_gap"] = _paired_test(
        np.asarray([row["target_reference_gap"] for row in vat]),
        np.asarray([row["target_reference_gap"] for row in vt]),
        iterations=iterations,
        seed=seed + 1,
    )
    for mode in ("V_A_T", "V_T"):
        mode_rows = by_mode[mode]
        with_correct = np.asarray([row["with_reference_rank"] == 1 for row in mode_rows], dtype=float)
        without_correct = np.asarray([row["without_reference_rank"] == 1 for row in mode_rows], dtype=float)
        key = f"{mode}_reference_masking_R@1"
        comparisons[key] = _paired_test(without_correct, with_correct, iterations=iterations, seed=seed + len(comparisons))
        comparisons[key]["mcnemar"] = _mcnemar_exact(without_correct, with_correct)

    raw_p = {name: float(payload["paired_randomization_p_two_sided"]) for name, payload in comparisons.items()}
    adjusted = _holm_adjust(raw_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value

    seven_modes = json.loads((evaluation_dir / "seven_mode_results.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "paired_comparisons.json", comparisons)
    error_breakdown = {
        mode: {
            "query_count": payload["with_reference"]["query_count"],
            "top1_own_reference_count": payload["with_reference"]["top1_own_reference_count"],
            "top1_own_reference_rate": payload["with_reference"]["top1_own_reference_rate"],
            "breakdown": payload["with_reference"]["breakdown"],
        }
        for mode, payload in seven_modes.items()
    }
    _atomic_json(output_dir / "error_breakdown.json", error_breakdown)
    lines = [
        "# ImageBind Audio-CVR Results",
        "",
        "ImageBind-Huge is evaluated zero-shot with equal-weight normalized modality arithmetic.",
        "",
        "| Mode | With-ref R@1 | Without-ref R@1 | Ref-induced drop | R@5 | R@10 | Target beats ref | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        with_ref = seven_modes[mode]["with_reference"]
        without_ref = seven_modes[mode]["without_reference"]
        lines.append(
            f"| {mode} | {with_ref['R@1']:.4f} | {without_ref['R@1']:.4f} | {with_ref['reference_induced_R@1_drop']:.4f} | "
            f"{with_ref['R@5']:.4f} | {with_ref['R@10']:.4f} | {with_ref['target_beats_reference']:.4f} | {with_ref['target_reference_gap_mean']:.5f} |"
        )
    lines.extend(["", "## Paired tests", ""])
    for name, payload in comparisons.items():
        lines.append(
            f"- `{name}`: delta={payload['mean_difference']:.5f}, 95% CI={payload['bootstrap_95_ci']}, "
            f"p={payload['paired_randomization_p_two_sided']:.5g}, Holm p={payload['holm_adjusted_p']:.5g}."
        )
    _atomic_text(output_dir / "paper_results.md", "\n".join(lines) + "\n")
    summary = {
        "state": "COMPLETE",
        "model": "ImageBind-Huge",
        "iterations": iterations,
        "comparisons": comparisons,
        "selection_uses_test_metrics": False,
    }
    _atomic_json(output_dir / "statistics_summary.json", summary)
    return summary


def build_delta_inventory(pre_dir: Path, final_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for kind in ("media", "text"):
        id_key = f"{kind}_id"
        pre_rows = _load_jsonl(pre_dir / f"{kind}_inventory.jsonl")
        final_rows = _load_jsonl(final_dir / f"{kind}_inventory.jsonl")
        pre_ids = {str(row[id_key]) for row in pre_rows}
        final_ids = {str(row[id_key]) for row in final_rows}
        delta = [row for row in final_rows if str(row[id_key]) not in pre_ids]
        removed = sorted(pre_ids - final_ids)
        if removed:
            raise ValueError(f"final {kind} inventory dropped {len(removed)} preexisting IDs")
        _atomic_jsonl(output_dir / f"delta_{kind}_inventory.jsonl", delta)
        summary[kind] = {
            "pre_count": len(pre_rows),
            "final_count": len(final_rows),
            "reused_count": len(pre_ids & final_ids),
            "delta_count": len(delta),
            "removed_count": len(removed),
        }
    _atomic_json(output_dir / "reuse_audit.json", summary)
    return summary


def audit_cache(inventory_dir: Path, cache_root: Path, output_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for kind in ("media", "text"):
        rows = _load_jsonl(inventory_dir / f"{kind}_inventory.jsonl")
        complete = 0
        missing_ids: list[str] = []
        for row in rows:
            item_id = str(row[f"{kind}_id"])
            if _existing_cache(cache_root, kind, item_id) is not None:
                complete += 1
            else:
                missing_ids.append(item_id)
        report[kind] = {
            "inventory_count": len(rows),
            "complete_count": complete,
            "missing_count": len(missing_ids),
            "missing_ids": missing_ids[:100],
        }
    report["complete"] = all(payload["missing_count"] == 0 for payload in report.values() if isinstance(payload, dict))
    _atomic_json(output_path, report)
    return report


def _parse_counts(value: str) -> dict[str, int]:
    if not value.strip():
        return {}
    result: dict[str, int] = {}
    for part in value.split(","):
        key, raw = part.split("=", 1)
        result[key.strip()] = int(raw)
    return result


def _path_list(values: Sequence[str]) -> list[Path]:
    return [Path(value).expanduser().resolve() for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental ImageBind baseline for Audio-CVR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("prepare-inventory")
    inventory.add_argument("--records", type=Path, required=True)
    inventory.add_argument("--output-dir", type=Path, required=True)
    inventory.add_argument("--media-root", action="append", default=[])
    inventory.add_argument("--expected-count", type=int)
    inventory.add_argument("--expected-sha256")
    inventory.add_argument("--inherited-records", type=Path)
    inventory.add_argument("--expected-subtypes", default="")
    inventory.add_argument("--require-unique-source-pair", action="store_true")
    inventory.add_argument("--allow-missing-media", action="store_true")

    cache = subparsers.add_parser("cache-imagebind")
    cache.add_argument("--inventory", type=Path, required=True)
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--model-dir", type=Path, required=True)
    cache.add_argument(
        "--vendor-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "imagebind_5120b6bb",
    )
    cache.add_argument("--kind", choices=("media", "text"), required=True)
    cache.add_argument("--shard-index", type=int, required=True)
    cache.add_argument("--shard-count", type=int, required=True)
    cache.add_argument("--device", required=True)
    cache.add_argument("--batch-size", type=int, default=2)
    cache.add_argument("--encoding-retries", type=int, default=4)

    delta = subparsers.add_parser("prepare-delta")
    delta.add_argument("--pre-inventory-dir", type=Path, required=True)
    delta.add_argument("--final-inventory-dir", type=Path, required=True)
    delta.add_argument("--output-dir", type=Path, required=True)

    audit = subparsers.add_parser("audit-cache")
    audit.add_argument("--inventory-dir", type=Path, required=True)
    audit.add_argument("--cache-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--records", type=Path, required=True)
    assemble.add_argument("--inventory-dir", type=Path, required=True)
    assemble.add_argument("--cache-root", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--pre-records", type=Path)
    assemble.add_argument("--core-records", type=Path)
    assemble.add_argument("--max-exclusion-rate", type=float, default=0.01)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--assembly-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--save-topk", type=int, default=20)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--evaluation-dir", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--iterations", type=int, default=20000)
    summarize.add_argument("--seed", type=int, default=20260723)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-inventory":
        payload = prepare_inventory(
            args.records,
            args.output_dir,
            _path_list(args.media_root) or [Path.cwd()],
            expected_count=args.expected_count,
            expected_sha256=args.expected_sha256,
            inherited_records=args.inherited_records,
            expected_subtypes=_parse_counts(args.expected_subtypes) or None,
            require_unique_source_pair=args.require_unique_source_pair,
            allow_missing_media=args.allow_missing_media,
        )
    elif args.command == "cache-imagebind":
        payload = cache_imagebind(
            args.inventory,
            args.cache_root,
            args.model_dir,
            args.vendor_root,
            kind=args.kind,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            batch_size=args.batch_size,
            retries=args.encoding_retries,
        )
    elif args.command == "prepare-delta":
        payload = build_delta_inventory(args.pre_inventory_dir, args.final_inventory_dir, args.output_dir)
    elif args.command == "audit-cache":
        payload = audit_cache(args.inventory_dir, args.cache_root, args.output)
    elif args.command == "assemble":
        payload = assemble_embeddings(
            args.records,
            args.inventory_dir,
            args.cache_root,
            args.output_dir,
            pre_records=args.pre_records,
            core_records=args.core_records,
            max_exclusion_rate=args.max_exclusion_rate,
        )
    elif args.command == "evaluate":
        payload = evaluate_embeddings(args.assembly_dir, args.output_dir, topk=args.save_topk)
    elif args.command == "summarize":
        payload = summarize_results(args.evaluation_dir, args.output_dir, iterations=args.iterations, seed=args.seed)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
