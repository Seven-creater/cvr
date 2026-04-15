from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from app.retrieval_types import RetrievalHit, TextRow, VideoRow


VIDEO_SUFFIXES = (".mp4", ".webm")
DEFAULT_AUDIO_SUFFIX = ".wav"


@dataclass(frozen=True, slots=True)
class AvigateRuntimeConfig:
    model_dir: str
    checkpoint_path: str
    data_json_path: str
    test_csv_path: str
    video_root: str
    audio_root: str
    clip_weight_path: str
    device: str = "cuda"
    batch_size_val: int = 100
    max_words: int = 32
    max_frames: int = 12
    sim_header: str = "seqTransf"
    cross_num_hidden_layers: int = 4
    audio_query_layers: int = 4
    temperature: float = 1.0
    feature_framerate: int = 1
    eval_frame_order: int = 0
    slice_framepos: int = 2
    beta: float = 0.2
    margin_bd: float = 0.1


@dataclass(slots=True)
class AvigateRuntime:
    config: AvigateRuntimeConfig
    model: Any
    tokenizer: Any
    device: str
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    text_input_mask: Any
    text_sequence_output: Any
    video_mask: Any
    video_visual_output: Any
    video_audio_output: Any
    _video_index: dict[str, int] = field(default_factory=dict)
    _text_index: dict[str, int] = field(default_factory=dict)
    _video_to_text_ids: dict[str, list[str]] = field(default_factory=dict)
    _sim_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self._video_index:
            self._video_index = {row.video_id: index for index, row in enumerate(self.video_rows)}
        if not self._text_index:
            self._text_index = {row.text_id: index for index, row in enumerate(self.text_rows)}
        if not self._video_to_text_ids:
            mapping: dict[str, list[str]] = defaultdict(list)
            for row in self.text_rows:
                mapping[row.video_id].append(row.text_id)
            self._video_to_text_ids = dict(mapping)

    @property
    def audio_available(self) -> bool:
        return bool(self.config.audio_root)

    def target_text_ids(self, video_id: str) -> list[str]:
        return list(self._video_to_text_ids[video_id])

    def similarity_matrix(self, *, batch_size: int | None = None) -> np.ndarray:
        if self._sim_matrix is None:
            self._sim_matrix = _compute_similarity_matrix(self, batch_size=batch_size)
        return self._sim_matrix

    def score_text_query(self, query_text: str, *, batch_size: int | None = None) -> np.ndarray:
        torch = _torch()
        query_ids, query_mask, query_segment = _build_text_inputs(self.tokenizer, query_text, self.config.max_words)
        with torch.no_grad():
            query_sequence = self.model.get_sequence_output(
                torch.from_numpy(query_ids).to(self.device),
                torch.from_numpy(query_segment).to(self.device),
                torch.from_numpy(query_mask).to(self.device),
            )

        scores: list[np.ndarray] = []
        video_total = len(self.video_rows)
        chunk = batch_size or self.config.batch_size_val
        for start in range(0, video_total, chunk):
            end = min(start + chunk, video_total)
            with torch.no_grad():
                logits, *_rest = self.model.get_similarity_logits(
                    query_sequence,
                    self.video_visual_output[start:end].to(self.device),
                    self.video_audio_output[start:end].to(self.device),
                    torch.from_numpy(query_mask).to(self.device),
                    self.video_mask[start:end].to(self.device),
                    loose_type=self.model.loose_type,
                )
            scores.append(logits.squeeze(0).detach().cpu().numpy())
        return np.concatenate(scores, axis=0)

    def score_video_query(self, video_id: str, *, batch_size: int | None = None) -> np.ndarray:
        if self._sim_matrix is not None:
            return self._sim_matrix[:, self._video_index[video_id]]

        torch = _torch()
        index = self._video_index[video_id]
        visual_output = self.video_visual_output[index : index + 1].to(self.device)
        audio_output = self.video_audio_output[index : index + 1].to(self.device)
        video_mask = self.video_mask[index : index + 1].to(self.device)

        scores: list[np.ndarray] = []
        text_total = len(self.text_rows)
        chunk = batch_size or self.config.batch_size_val
        for start in range(0, text_total, chunk):
            end = min(start + chunk, text_total)
            with torch.no_grad():
                logits, *_rest = self.model.get_similarity_logits(
                    self.text_sequence_output[start:end].to(self.device),
                    visual_output,
                    audio_output,
                    self.text_input_mask[start:end].to(self.device),
                    video_mask,
                    loose_type=self.model.loose_type,
                )
            scores.append(logits[:, 0].detach().cpu().numpy())
        return np.concatenate(scores, axis=0)


def load_avigate_runtime(config: AvigateRuntimeConfig) -> AvigateRuntime:
    for raw_path in (
        config.checkpoint_path,
        config.data_json_path,
        config.test_csv_path,
        config.video_root,
        config.audio_root,
        config.clip_weight_path,
    ):
        if not Path(raw_path).exists():
            raise FileNotFoundError(f"required AVIGATE path does not exist: {raw_path}")

    torch = _torch()
    CLIP4Clip, SimpleTokenizer, RawVideoExtractor = _import_avigate_vendor()
    text_rows, video_rows = _load_msrvtt_split_rows(config)
    tokenizer = SimpleTokenizer()
    task_config = _build_task_config(config)

    state_dict = torch.load(config.checkpoint_path, map_location="cpu")
    model = CLIP4Clip.from_pretrained("cross-base", state_dict=state_dict, task_config=task_config)
    model = model.to(config.device)
    model.eval()

    text_input_ids, text_input_mask, text_segment_ids = _encode_corpus_text_inputs(
        tokenizer=tokenizer,
        text_rows=text_rows,
        max_words=config.max_words,
    )
    text_sequence_output = _encode_corpus_text_outputs(
        model=model,
        device=config.device,
        text_input_ids=text_input_ids,
        text_input_mask=text_input_mask,
        text_segment_ids=text_segment_ids,
        batch_size=config.batch_size_val,
    )
    video_mask, video_visual_output, video_audio_output = _encode_corpus_video_outputs(
        model=model,
        device=config.device,
        video_rows=video_rows,
        config=config,
        raw_video_extractor=RawVideoExtractor(framerate=config.feature_framerate, size=224),
        batch_size=config.batch_size_val,
    )

    return AvigateRuntime(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        text_rows=text_rows,
        video_rows=video_rows,
        text_input_mask=torch.from_numpy(text_input_mask),
        text_sequence_output=text_sequence_output,
        video_mask=video_mask,
        video_visual_output=video_visual_output,
        video_audio_output=video_audio_output,
    )


def retrieve_videos_from_text_official(
    query_text: str,
    runtime: Any,
    topk: int = 10,
) -> list[RetrievalHit]:
    scores = np.asarray(runtime.score_text_query(query_text), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[: max(1, int(topk))]
    hits: list[RetrievalHit] = []
    for rank, index in enumerate(order, start=1):
        row = runtime.video_rows[int(index)]
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


def retrieve_texts_from_video_official(
    video_id: str,
    runtime: Any,
    topk: int = 10,
) -> list[RetrievalHit]:
    scores = np.asarray(runtime.score_video_query(video_id), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[: max(1, int(topk))]
    hits: list[RetrievalHit] = []
    for rank, index in enumerate(order, start=1):
        row = runtime.text_rows[int(index)]
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


def evaluate_avigate_official(runtime: Any, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    matrix = np.asarray(runtime.similarity_matrix(), dtype=np.float32)
    ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    t2v = {f"R@{k}": 0.0 for k in ks}
    v2t = {f"R@{k}": 0.0 for k in ks}

    video_index = {row.video_id: index for index, row in enumerate(runtime.video_rows)}
    video_to_text_ids = {row.video_id: set(runtime.target_text_ids(row.video_id)) for row in runtime.video_rows}

    for text_index, row in enumerate(runtime.text_rows):
        ranked_videos = np.argsort(-matrix[text_index], kind="stable")
        target_index = video_index[row.video_id]
        for k in ks:
            t2v[f"R@{k}"] += 1.0 if target_index in ranked_videos[:k] else 0.0

    for video_index_value, row in enumerate(runtime.video_rows):
        ranked_texts = np.argsort(-matrix[:, video_index_value], kind="stable")
        top_text_ids = [runtime.text_rows[int(index)].text_id for index in ranked_texts]
        target_ids = video_to_text_ids[row.video_id]
        for k in ks:
            v2t[f"R@{k}"] += 1.0 if any(text_id in target_ids for text_id in top_text_ids[:k]) else 0.0

    text_total = max(1, len(runtime.text_rows))
    video_total = max(1, len(runtime.video_rows))
    return {
        "dataset": "MSRVTT",
        "video_count": len(runtime.video_rows),
        "text_count": len(runtime.text_rows),
        "audio_available": bool(runtime.audio_available),
        "t2v": {key: round(value / text_total, 4) for key, value in t2v.items()},
        "v2t": {key: round(value / video_total, 4) for key, value in v2t.items()},
    }


def _compute_similarity_matrix(runtime: AvigateRuntime, *, batch_size: int | None = None) -> np.ndarray:
    torch = _torch()
    chunks: list[np.ndarray] = []
    text_total = len(runtime.text_rows)
    video_total = len(runtime.video_rows)
    chunk = batch_size or runtime.config.batch_size_val
    for text_start in range(0, text_total, chunk):
        text_end = min(text_start + chunk, text_total)
        row_chunks: list[np.ndarray] = []
        for video_start in range(0, video_total, chunk):
            video_end = min(video_start + chunk, video_total)
            with torch.no_grad():
                logits, *_rest = runtime.model.get_similarity_logits(
                    runtime.text_sequence_output[text_start:text_end].to(runtime.device),
                    runtime.video_visual_output[video_start:video_end].to(runtime.device),
                    runtime.video_audio_output[video_start:video_end].to(runtime.device),
                    runtime.text_input_mask[text_start:text_end].to(runtime.device),
                    runtime.video_mask[video_start:video_end].to(runtime.device),
                    loose_type=runtime.model.loose_type,
                )
            row_chunks.append(logits.detach().cpu().numpy())
        chunks.append(np.concatenate(row_chunks, axis=1))
    return np.concatenate(chunks, axis=0)


def _load_msrvtt_split_rows(config: AvigateRuntimeConfig) -> tuple[list[TextRow], list[VideoRow]]:
    text_rows: list[TextRow] = []
    video_rows: list[VideoRow] = []
    seen_video_ids: set[str] = set()
    text_counter: dict[str, int] = defaultdict(int)
    with Path(config.test_csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = str(row["video_id"]).strip()
            sentence = str(row.get("sentence") or row.get("caption") or row.get("text") or "").strip()
            if not sentence:
                raise ValueError(f"missing sentence text for video_id={video_id}")
            text_counter[video_id] += 1
            text_rows.append(
                TextRow(
                    text_id=f"{video_id}::caption::{text_counter[video_id]}",
                    video_id=video_id,
                    text=sentence,
                )
            )
            if video_id not in seen_video_ids:
                seen_video_ids.add(video_id)
                video_rows.append(
                    VideoRow(
                        video_id=video_id,
                        video_path=_resolve_video_path(video_id, config.video_root),
                        audio_path=str(Path(config.audio_root) / f"{video_id}{DEFAULT_AUDIO_SUFFIX}"),
                    )
                )
    return text_rows, video_rows


def _resolve_video_path(video_id: str, video_root: str) -> str:
    root = Path(video_root)
    for suffix in VIDEO_SUFFIXES:
        candidate = root / f"{video_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(root / f"{video_id}{VIDEO_SUFFIXES[0]}")


def _build_task_config(config: AvigateRuntimeConfig) -> SimpleNamespace:
    return SimpleNamespace(
        local_rank=0,
        pretrained_clip_name=config.clip_weight_path,
        cross_num_hidden_layers=config.cross_num_hidden_layers,
        audio_query_layers=config.audio_query_layers,
        temperature=config.temperature,
        sim_header=config.sim_header,
        linear_patch="2d",
        max_words=config.max_words,
        max_frames=config.max_frames,
        loose_type=True,
        beta=config.beta,
        margin_BD=config.margin_bd,
    )


def _encode_corpus_text_inputs(
    *,
    tokenizer: Any,
    text_rows: list[TextRow],
    max_words: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_batches: list[np.ndarray] = []
    mask_batches: list[np.ndarray] = []
    segment_batches: list[np.ndarray] = []
    for row in text_rows:
        input_ids, input_mask, segment_ids = _build_text_inputs(tokenizer, row.text, max_words)
        ids_batches.append(input_ids)
        mask_batches.append(input_mask)
        segment_batches.append(segment_ids)
    return (
        np.concatenate(ids_batches, axis=0),
        np.concatenate(mask_batches, axis=0),
        np.concatenate(segment_batches, axis=0),
    )


def _build_text_inputs(tokenizer: Any, text: str, max_words: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    special = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>"}
    words = tokenizer.tokenize(text)
    words = [special["CLS_TOKEN"]] + words
    total_length_with_cls = max_words - 1
    if len(words) > total_length_with_cls:
        words = words[:total_length_with_cls]
    words = words + [special["SEP_TOKEN"]]
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return (
        np.asarray([input_ids], dtype=np.int64),
        np.asarray([input_mask], dtype=np.int64),
        np.asarray([segment_ids], dtype=np.int64),
    )


def _encode_corpus_text_outputs(
    *,
    model: Any,
    device: str,
    text_input_ids: np.ndarray,
    text_input_mask: np.ndarray,
    text_segment_ids: np.ndarray,
    batch_size: int,
) -> Any:
    torch = _torch()
    batches: list[Any] = []
    for start in range(0, text_input_ids.shape[0], batch_size):
        end = min(start + batch_size, text_input_ids.shape[0])
        with torch.no_grad():
            output = model.get_sequence_output(
                torch.from_numpy(text_input_ids[start:end]).to(device),
                torch.from_numpy(text_segment_ids[start:end]).to(device),
                torch.from_numpy(text_input_mask[start:end]).to(device),
            )
        batches.append(output.detach().cpu())
    return torch.cat(batches, dim=0)


def _encode_corpus_video_outputs(
    *,
    model: Any,
    device: str,
    video_rows: list[VideoRow],
    config: AvigateRuntimeConfig,
    raw_video_extractor: Any,
    batch_size: int,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    video_masks: list[Any] = []
    visual_outputs: list[Any] = []
    audio_outputs: list[Any] = []
    for start in range(0, len(video_rows), batch_size):
        end = min(start + batch_size, len(video_rows))
        batch_rows = video_rows[start:end]
        video_batch, video_mask = _load_video_batch(
            [row.video_id for row in batch_rows],
            video_root=config.video_root,
            raw_video_extractor=raw_video_extractor,
            max_frames=config.max_frames,
            frame_order=config.eval_frame_order,
            slice_framepos=config.slice_framepos,
        )
        audio_batch = _load_audio_batch(
            [row.video_id for row in batch_rows],
            audio_root=config.audio_root,
        )
        with torch.no_grad():
            visual_output = model.get_visual_output(torch.from_numpy(video_batch).to(device), torch.from_numpy(video_mask).to(device))
            audio_output = model.get_audio_output(torch.from_numpy(audio_batch).to(device))
        video_masks.append(torch.from_numpy(video_mask))
        visual_outputs.append(visual_output.detach().cpu())
        audio_outputs.append(audio_output.detach().cpu())
    return torch.cat(video_masks, dim=0), torch.cat(visual_outputs, dim=0), torch.cat(audio_outputs, dim=0)


def _load_video_batch(
    video_ids: list[str],
    *,
    video_root: str,
    raw_video_extractor: Any,
    max_frames: int,
    frame_order: int,
    slice_framepos: int,
) -> tuple[np.ndarray, np.ndarray]:
    video_mask = np.zeros((len(video_ids), 1, max_frames), dtype=np.int64)
    video = np.zeros(
        (len(video_ids), 1, max_frames, 1, 3, raw_video_extractor.size, raw_video_extractor.size),
        dtype=np.float32,
    )
    max_video_length = [0] * len(video_ids)

    for index, video_id in enumerate(video_ids):
        video_path = _resolve_video_path(video_id, video_root)
        raw_video_data = raw_video_extractor.get_video_data(video_path)["video"]
        if len(raw_video_data.shape) > 3:
            raw_video_slice = raw_video_extractor.process_raw_data(raw_video_data)
            if max_frames < raw_video_slice.shape[0]:
                if slice_framepos == 0:
                    video_slice = raw_video_slice[:max_frames, ...]
                elif slice_framepos == 1:
                    video_slice = raw_video_slice[-max_frames:, ...]
                else:
                    sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_index, ...]
            else:
                video_slice = raw_video_slice

            video_slice = raw_video_extractor.process_frame_order(video_slice, frame_order=frame_order)
            slice_len = int(video_slice.shape[0])
            max_video_length[index] = max(max_video_length[index], slice_len)
            if slice_len >= 1:
                video[index, 0, :slice_len, ...] = video_slice.cpu().numpy()

    for index, length in enumerate(max_video_length):
        video_mask[index, 0, :length] = 1

    return video, video_mask


def _load_audio_batch(
    video_ids: list[str],
    *,
    audio_root: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    torch = _torch()
    torchaudio = _torchaudio()

    target_length = 1024
    norm_mean = -5.118
    norm_std = 3.2527153
    fbanks = torch.zeros((len(video_ids), 1, target_length, 128), dtype=torch.float32)
    for index, video_id in enumerate(video_ids):
        audio_path = Path(audio_root) / f"{video_id}{DEFAULT_AUDIO_SUFFIX}"
        if not audio_path.exists():
            continue
        waveform, sr = torchaudio.load(str(audio_path))
        if sample_rate != sr:
            resample = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resample(waveform)
        waveform -= waveform.mean()
        frame_shift = waveform.shape[1] * 1000 / (sample_rate * target_length)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=frame_shift,
        )
        frame_count = fbank.shape[0]
        padding = target_length - frame_count
        if padding > 0:
            zero_pad = torch.nn.ZeroPad2d((0, 0, 0, padding))
            fbank = zero_pad(fbank)
        elif padding < 0:
            fbank = fbank[:target_length, :]
        fbank = (fbank - norm_mean) / (norm_std * 2)
        fbanks[index, 0] = fbank
    return fbanks.numpy()


def _import_avigate_vendor() -> tuple[Any, Any, Any]:
    try:
        from app.avigate_vendor.modeling import CLIP4Clip
        from app.avigate_vendor.rawvideo_util import RawVideoExtractor
        from app.avigate_vendor.tokenization_clip import SimpleTokenizer
    except Exception as exc:  # pragma: no cover - exercised in server integration
        raise RuntimeError(
            "failed to import vendored AVIGATE dependencies; check timm/torchaudio/opencv/ftfy/regex availability"
        ) from exc
    return CLIP4Clip, SimpleTokenizer, RawVideoExtractor


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required to run AVIGATE official retrieval") from exc
    return torch


def _torchaudio():
    try:
        import torchaudio
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torchaudio is required to run AVIGATE official retrieval") from exc
    return torchaudio
