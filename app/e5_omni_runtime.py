from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_E5_OMNI_MODEL_PATH = (
    "/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B"
)


@dataclass(frozen=True)
class E5OmniRuntimeConfig:
    model_path: str = DEFAULT_E5_OMNI_MODEL_PATH
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    normalize_embeddings: bool = True
    batch_size: int = 1
    video_max_pixels: int = 128 * 28 * 28
    video_fps: int = 1


@dataclass(frozen=True)
class E5EmbeddingMetadata:
    model_path: str
    device: str
    torch_dtype: str
    normalize_embeddings: bool
    embedding_dim: int
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class E5OmniRuntime:
    def __init__(self, *, config: E5OmniRuntimeConfig, model: Any) -> None:
        self.config = config
        self.model = model

    def encode_text_query(self, text: str) -> np.ndarray:
        if not str(text).strip():
            raise ValueError("text query is required")
        return self._encode_with("encode_query", [str(text).strip()])[0]

    def encode_video_document(self, video_path: str | Path) -> np.ndarray:
        return self.encode_video_documents([video_path])[0]

    def encode_video_documents(self, video_paths: list[str | Path]) -> np.ndarray:
        if not video_paths:
            raise ValueError("video_paths must not be empty")
        inputs = [str(_check_video_path(path)) for path in video_paths]
        return self._encode_with("encode_document", inputs)

    def encode_video_text_query(self, *, video_path: str | Path, text: str) -> np.ndarray:
        if not str(text).strip():
            raise ValueError("text is required for a composed video-text query")
        video = str(_check_video_path(video_path))
        payload = {"video": video, "text": str(text).strip()}
        return self._encode_with("encode_document", [payload])[0]

    def metadata_for(self, *, source: str, embedding: np.ndarray) -> E5EmbeddingMetadata:
        source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
        return E5EmbeddingMetadata(
            model_path=self.config.model_path,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
            normalize_embeddings=self.config.normalize_embeddings,
            embedding_dim=int(np.asarray(embedding).shape[-1]),
            source_hash=source_hash,
        )

    def _encode_with(self, method_name: str, inputs: list[Any]) -> np.ndarray:
        method = getattr(self.model, method_name)
        kwargs = {
            "batch_size": self.config.batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self.config.normalize_embeddings:
            kwargs["normalize_embeddings"] = True
        try:
            encoded = method(inputs, **kwargs)
        except TypeError:
            kwargs.pop("normalize_embeddings", None)
            encoded = method(inputs, **kwargs)
        array = _as_2d_float32(encoded)
        if self.config.normalize_embeddings:
            array = _normalize_rows(array)
        return array


def load_e5_omni_runtime(config: E5OmniRuntimeConfig) -> E5OmniRuntime:
    model_root = Path(config.model_path)
    if not model_root.exists():
        raise FileNotFoundError(f"e5-omni model path not found: {model_root}")
    if not (model_root / "config.json").exists():
        raise FileNotFoundError(f"e5-omni config.json not found under: {model_root}")

    SentenceTransformer = _sentence_transformer_cls()
    model_kwargs = _build_model_kwargs(config)
    init_kwargs: dict[str, Any] = {
        "device": config.device,
        "trust_remote_code": config.trust_remote_code,
    }
    if model_kwargs:
        init_kwargs["model_kwargs"] = model_kwargs
    try:
        model = SentenceTransformer(str(model_root), **init_kwargs)
    except TypeError:
        init_kwargs.pop("trust_remote_code", None)
        model = SentenceTransformer(str(model_root), **init_kwargs)
    _configure_video_processing(model, config)
    return E5OmniRuntime(config=config, model=model)


def runtime_config_fingerprint(config: E5OmniRuntimeConfig) -> str:
    payload = asdict(config)
    model_root = Path(config.model_path)
    payload["model_path"] = _path_fingerprint(model_root)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sentence_transformer_cls() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - server dependency
        raise RuntimeError("sentence-transformers is required to load e5-omni") from exc
    return SentenceTransformer


def _build_model_kwargs(config: E5OmniRuntimeConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.torch_dtype and config.torch_dtype != "auto":
        torch = _torch()
        if hasattr(torch, config.torch_dtype):
            kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)
    if config.attn_implementation and config.attn_implementation != "none":
        kwargs["attn_implementation"] = config.attn_implementation
    return kwargs


def _configure_video_processing(model: Any, config: E5OmniRuntimeConfig) -> None:
    processing_kwargs = {
        "video": {
            "max_pixels": int(config.video_max_pixels),
            "do_sample_frames": True,
            "fps": int(config.video_fps),
        }
    }
    try:
        first_module = model[0]
    except Exception:
        first_module = None
    for candidate in (first_module, model):
        existing = getattr(candidate, "processing_kwargs", None)
        if isinstance(existing, dict):
            existing.update(processing_kwargs)
            return


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - server dependency
        raise RuntimeError("PyTorch is required to load e5-omni") from exc
    return torch


def _check_video_path(path: str | Path) -> Path:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"video file not found: {video_path}")
    return video_path


def _as_2d_float32(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected 2D embeddings, got shape {array.shape}")
    return array


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


def _path_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }
