from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
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
    text_score_map: dict[str, np.ndarray]
    video_score_map: dict[str, np.ndarray]
    matrix: np.ndarray
    audio_available: bool = True

    def __post_init__(self) -> None:
        self._video_to_text: dict[str, list[str]] = {}
        for row in self.text_rows:
            self._video_to_text.setdefault(row.video_id, []).append(row.text_id)

    def score_text_query(self, query_text: str) -> np.ndarray:
        return self.text_score_map[query_text]

    def score_video_query(self, video_id: str) -> np.ndarray:
        return self.video_score_map[video_id]

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
                "query one": np.asarray([0.9, 0.1], dtype=np.float32),
            },
            video_score_map={
                "video2": np.asarray([0.2, 0.1, 0.9, 0.8], dtype=np.float32),
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

    def test_evaluate_avigate_official_matches_baseline_shape(self) -> None:
        metrics = evaluate_avigate_official(self.runtime, ks=(1, 5, 10))
        self.assertEqual("MSRVTT", metrics["dataset"])
        self.assertEqual(2, metrics["video_count"])
        self.assertEqual(4, metrics["text_count"])
        self.assertEqual({"R@1": 1.0, "R@5": 1.0, "R@10": 1.0}, metrics["t2v"])
        self.assertEqual({"R@1": 1.0, "R@5": 1.0, "R@10": 1.0}, metrics["v2t"])

    def test_load_avigate_runtime_rejects_missing_paths_before_heavy_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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


if __name__ == "__main__":
    unittest.main()
