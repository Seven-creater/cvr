from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import uuid
from unittest import mock

import numpy as np

from app.avigate_official import (
    AvigateRuntimeConfig,
    _load_audio_fbank,
    evaluate_avigate_official,
    load_avigate_runtime,
    retrieve_texts_from_video_official,
    retrieve_videos_from_text_official,
)
from app.retrieval_types import TextRow, VideoRow


@dataclass
class FakeRuntime:
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    text_score_map: dict[tuple[str, str], np.ndarray]
    video_score_map: dict[tuple[str, str], np.ndarray]
    matrix: np.ndarray
    audio_available: bool = True
    text_calls: list[tuple[str, str]] = None
    video_calls: list[tuple[str, str]] = None

    def __post_init__(self) -> None:
        self._video_to_text: dict[str, list[str]] = {}
        for row in self.text_rows:
            self._video_to_text.setdefault(row.video_id, []).append(row.text_id)
        self.text_calls = []
        self.video_calls = []

    def score_text_query(self, query_text: str, *, audio_mode: str = "on") -> np.ndarray:
        self.text_calls.append((query_text, audio_mode))
        return self.text_score_map[(query_text, audio_mode)]

    def score_video_query(self, video_id: str, *, audio_mode: str = "on") -> np.ndarray:
        self.video_calls.append((video_id, audio_mode))
        return self.video_score_map[(video_id, audio_mode)]

    def target_text_ids(self, video_id: str) -> list[str]:
        return list(self._video_to_text[video_id])

    def similarity_matrix(self) -> np.ndarray:
        return self.matrix


class AvigateOfficialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_rows = [
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            VideoRow(video_id="video2", video_path="/tmp/video2.mp4"),
        ]
        self.text_rows = [
            TextRow(text_id="t1", video_id="video1", text="query one"),
            TextRow(text_id="t2", video_id="video1", text="query one alt"),
            TextRow(text_id="t3", video_id="video2", text="query two"),
            TextRow(text_id="t4", video_id="video2", text="query two alt"),
        ]
        self.runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={
                ("query one", "on"): np.asarray([0.9, 0.1], dtype=np.float32),
                ("query one", "off"): np.asarray([0.1, 0.9], dtype=np.float32),
            },
            video_score_map={
                ("video2", "on"): np.asarray([0.2, 0.1, 0.9, 0.8], dtype=np.float32),
                ("video2", "off"): np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
            },
            matrix=np.asarray(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.1, 0.9],
                    [0.2, 0.8],
                ],
                dtype=np.float32,
            ),
        )

    def test_retrieve_videos_from_text_official_sorts_scores(self) -> None:
        hits = retrieve_videos_from_text_official("query one", self.runtime, topk=2)
        self.assertEqual(["video1", "video2"], [hit.video_id for hit in hits])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_retrieve_texts_from_video_official_returns_text_hits(self) -> None:
        hits = retrieve_texts_from_video_official("video2", self.runtime, topk=3)
        self.assertEqual(["t3", "t4", "t1"], [hit.text_id for hit in hits])
        self.assertEqual("query two", hits[0].text)

    def test_retrieve_official_passes_audio_mode(self) -> None:
        video_hits = retrieve_videos_from_text_official("query one", self.runtime, topk=2, audio_mode="off")
        text_hits = retrieve_texts_from_video_official("video2", self.runtime, topk=2, audio_mode="off")

        self.assertEqual([("query one", "off")], self.runtime.text_calls)
        self.assertEqual([("video2", "off")], self.runtime.video_calls)
        self.assertEqual(["video2", "video1"], [hit.video_id for hit in video_hits])
        self.assertEqual(["t1", "t2"], [hit.text_id for hit in text_hits])

    def test_evaluate_avigate_official_matches_baseline_shape(self) -> None:
        metrics = evaluate_avigate_official(self.runtime, ks=(1, 5, 10))
        self.assertEqual("MSRVTT", metrics["dataset"])
        self.assertEqual(2, metrics["video_count"])
        self.assertEqual(4, metrics["text_count"])
        self.assertEqual({"R@1": 1.0, "R@5": 1.0, "R@10": 1.0}, metrics["t2v"])
        self.assertEqual({"R@1": 1.0, "R@5": 1.0, "R@10": 1.0}, metrics["v2t"])

    def test_load_avigate_runtime_rejects_missing_paths_before_heavy_imports(self) -> None:
        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        root = temp_parent / f"tmp-avigate-official-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            config = AvigateRuntimeConfig(
                model_dir=str(root / "model"),
                checkpoint_path=str(root / "missing.bin"),
                data_json_path=str(root / "missing.json"),
                test_csv_path=str(root / "missing.csv"),
                video_root=str(root / "missing_videos"),
                audio_root=str(root / "missing_audio"),
                clip_weight_path=str(root / "missing_clip.pt"),
            )
            with self.assertRaises(FileNotFoundError):
                load_avigate_runtime(config)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_load_audio_fbank_falls_back_to_librosa(self) -> None:
        fake_torch = mock.Mock()
        fake_tensor = mock.Mock()
        fake_tensor.float.return_value = fake_tensor
        fake_torch.from_numpy.return_value = fake_tensor

        fake_librosa = mock.Mock()
        fake_librosa.load.return_value = (np.asarray([0.1, -0.2, 0.3], dtype=np.float32), 16000)
        fake_librosa.feature.melspectrogram.return_value = np.ones((128, 4), dtype=np.float32)
        fake_librosa.power_to_db.return_value = np.ones((128, 4), dtype=np.float32)

        fbank, returned_librosa = _load_audio_fbank(
            audio_path=Path("/tmp/fake.wav"),
            sample_rate=16000,
            target_length=1024,
            torch_module=fake_torch,
            torchaudio_module=None,
            librosa_module=fake_librosa,
        )

        self.assertIs(fake_tensor, fbank)
        self.assertIs(fake_librosa, returned_librosa)
        fake_librosa.load.assert_called_once()
        fake_librosa.feature.melspectrogram.assert_called_once()
        fake_librosa.power_to_db.assert_called_once()

    def test_load_avigate_runtime_reuses_cached_corpus_outputs(self) -> None:
        import torch
        original_torch_load = torch.load

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir:
            root = Path(temp_dir)
            (root / "model").mkdir()
            (root / "videos").mkdir()
            (root / "audio").mkdir()
            checkpoint = root / "checkpoint.bin"
            data_json = root / "data.json"
            split_csv = root / "split.csv"
            clip_weight = root / "clip.pt"
            checkpoint.write_bytes(b"x")
            data_json.write_text("{}", encoding="utf-8")
            split_csv.write_text("video_id,sentence\nvideo1,caption one\n", encoding="utf-8")
            clip_weight.write_bytes(b"x")

            fake_model = mock.Mock()
            fake_model.to.return_value = fake_model
            fake_model.eval.return_value = None
            fake_model.loose_type = True
            fake_tokenizer_cls = mock.Mock()
            fake_tokenizer = fake_tokenizer_cls.return_value
            fake_clip_cls = mock.Mock()
            fake_clip_cls.from_pretrained.return_value = fake_model
            fake_extractor_cls = mock.Mock()
            rows = (
                [TextRow(text_id="t1", video_id="video1", text="caption one")],
                [VideoRow(video_id="video1", video_path=str(root / "videos" / "video1.mp4"), audio_path=str(root / "audio" / "video1.wav"))],
            )
            checkpoint_loader = lambda path, *args, **kwargs: {} if str(path) == str(checkpoint) else original_torch_load(path, *args, **kwargs)

            with (
                mock.patch("app.avigate_official._import_avigate_vendor", return_value=(fake_clip_cls, fake_tokenizer_cls, fake_extractor_cls)),
                mock.patch("app.avigate_official._load_msrvtt_split_rows", return_value=rows),
                mock.patch("app.avigate_official._encode_corpus_text_inputs", return_value=(
                    np.asarray([[1, 2]], dtype=np.int64),
                    np.asarray([[1, 1]], dtype=np.int64),
                    np.asarray([[0, 0]], dtype=np.int64),
                )),
                mock.patch("app.avigate_official._encode_corpus_text_outputs", return_value=torch.ones((1, 2))),
                mock.patch("app.avigate_official._encode_corpus_video_outputs", return_value=(
                    torch.ones((1, 1, 2)),
                    torch.ones((1, 1, 2)),
                    torch.ones((1, 1, 2)),
                )),
                mock.patch("app.avigate_official._RUNTIME_CACHE", {}),
                mock.patch("app.avigate_official._torch", return_value=torch),
                mock.patch("torch.load", side_effect=checkpoint_loader),
            ):
                config = AvigateRuntimeConfig(
                    model_dir=str(root / "model"),
                    checkpoint_path=str(checkpoint),
                    data_json_path=str(data_json),
                    test_csv_path=str(split_csv),
                    video_root=str(root / "videos"),
                    audio_root=str(root / "audio"),
                    clip_weight_path=str(clip_weight),
                    cache_dir=str(root / "cache"),
                    device="cpu",
                )
                runtime1 = load_avigate_runtime(config)
                self.assertTrue(Path(runtime1.cache_path).exists())

            with (
                mock.patch("app.avigate_official._import_avigate_vendor", return_value=(fake_clip_cls, fake_tokenizer_cls, fake_extractor_cls)),
                mock.patch("app.avigate_official._load_msrvtt_split_rows", return_value=rows),
                mock.patch("app.avigate_official._encode_corpus_text_inputs") as encode_inputs,
                mock.patch("app.avigate_official._encode_corpus_text_outputs") as encode_text,
                mock.patch("app.avigate_official._encode_corpus_video_outputs") as encode_video,
                mock.patch("app.avigate_official._RUNTIME_CACHE", {}),
                mock.patch("app.avigate_official._torch", return_value=torch),
                mock.patch("torch.load", side_effect=checkpoint_loader),
            ):
                runtime2 = load_avigate_runtime(config)
                self.assertEqual(runtime1.cache_path, runtime2.cache_path)
                encode_inputs.assert_not_called()
                encode_text.assert_not_called()
                encode_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
