from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.avigate_agent import run_cvr_agent_case
from app.cvr_fusion import fuse_video_hits, fused_hits_to_retrieval_hits
from app.cvr_pipeline import run_cvr_fusion_eval, run_e5_only_eval
from app.cvr_query_builder import CVRTriplet, build_cvr_query_views, load_cvr_triplets_jsonl
from app.e5_omni_index import (
    E5TargetIndex,
    E5TargetRecord,
    build_or_load_e5_target_index,
    retrieve_e5_videos,
)
from app.e5_omni_runtime import E5OmniRuntime, E5OmniRuntimeConfig
from app.omni_checker import MockOmniChecker
from app.retrieval_types import RetrievalHit, TextRow, VideoRow


class FakeE5Model:
    def encode_document(self, inputs, **kwargs):
        _ = kwargs
        rows = []
        for item in inputs:
            if isinstance(item, dict):
                text = item.get("text", "")
                rows.append([0.0, 1.0] if "make it blue" in text else [1.0, 0.0])
            elif "video2" in str(item):
                rows.append([0.0, 1.0])
            else:
                rows.append([1.0, 0.0])
        return np.asarray(rows, dtype=np.float32)

    def encode_query(self, inputs, **kwargs):
        _ = kwargs
        return np.asarray([[1.0, 0.0] for _item in inputs], dtype=np.float32)


@dataclass
class FakeAvigateRuntime:
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    score_map: dict[str, np.ndarray]
    text_calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._video_index = {row.video_id: index for index, row in enumerate(self.video_rows)}

    def score_text_query(self, query_text: str, *, audio_mode: str = "on") -> np.ndarray:
        _ = audio_mode
        self.text_calls.append(query_text)
        return self.score_map[query_text]


class CVRIntegrationTests(unittest.TestCase):
    def test_query_builder_uses_reference_caption_without_target_caption(self) -> None:
        triplet = CVRTriplet(
            sample_id="video1",
            reference_video="/tmp/ref.mp4",
            target_video="/tmp/target.mp4",
            edit_text="make it blue",
            reference_caption="a red car drives",
        )

        views = build_cvr_query_views(triplet)

        self.assertEqual("a red car drives. Edit: make it blue.", views.avigate_text_query)
        self.assertIn("make it blue", views.e5_text_query)
        self.assertEqual("/tmp/ref.mp4", views.e5_video_text_query["video"])

    def test_load_triplets_jsonl_rejects_missing_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triplets.jsonl"
            path.write_text('{"sample_id": "x", "reference_video": "ref.mp4"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing target_video"):
                load_cvr_triplets_jsonl(path)

    def test_e5_runtime_encodes_video_text_and_video_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "video1.mp4").write_bytes(b"x")
            (root / "video2.mp4").write_bytes(b"x")
            runtime = E5OmniRuntime(
                config=E5OmniRuntimeConfig(model_path=str(root), device="cpu", batch_size=2),
                model=FakeE5Model(),
            )

            docs = runtime.encode_video_documents([root / "video1.mp4", root / "video2.mp4"])
            query = runtime.encode_video_text_query(video_path=root / "video1.mp4", text="make it blue")

            self.assertEqual((2, 2), docs.shape)
            self.assertEqual((2,), query.shape)
            self.assertAlmostEqual(1.0, float(np.linalg.norm(query)), places=5)

    def test_e5_target_index_caches_and_retrieves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "video1.mp4").write_bytes(b"x")
            (root / "video2.mp4").write_bytes(b"x")
            runtime = E5OmniRuntime(
                config=E5OmniRuntimeConfig(model_path=str(root), device="cpu"),
                model=FakeE5Model(),
            )
            records = [
                E5TargetRecord(video_id="video1", video_path=str(root / "video1.mp4")),
                E5TargetRecord(video_id="video2", video_path=str(root / "video2.mp4")),
            ]

            index = build_or_load_e5_target_index(runtime=runtime, records=records, index_dir=root / "index")
            loaded = build_or_load_e5_target_index(runtime=runtime, records=records, index_dir=root / "index")
            hits = retrieve_e5_videos(query_embedding=np.asarray([0.0, 1.0], dtype=np.float32), index=loaded, topk=2)

            self.assertEqual(["video1", "video2"], index.video_ids)
            self.assertEqual(["video2", "video1"], [hit.video_id for hit in hits])
            self.assertEqual((2, 2), loaded.embeddings.shape)

    def test_fusion_dedupes_and_keeps_source_evidence(self) -> None:
        avigate_hits = [
            RetrievalHit(rank=1, item_id="video1", score=0.9, video_id="video1"),
            RetrievalHit(rank=2, item_id="video2", score=0.7, video_id="video2"),
        ]
        e5_hits = [
            RetrievalHit(rank=1, item_id="video2", score=0.8, video_id="video2"),
            RetrievalHit(rank=2, item_id="video3", score=0.6, video_id="video3"),
        ]

        fused = fuse_video_hits(avigate_hits=avigate_hits, e5_hits=e5_hits, topk=3, rrf_k=10)

        self.assertEqual("video2", fused[0].video_id)
        self.assertEqual({"avigate": 2, "e5": 1}, fused[0].source_ranks)
        self.assertEqual(["video2", "video1", "video3"], [hit.video_id for hit in fused])

    def test_e5_only_eval_writes_recall_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"x")
            triplet = CVRTriplet(
                sample_id="video2",
                reference_video=str(root / "ref.mp4"),
                target_video=str(root / "video2.mp4"),
                edit_text="make it blue",
                reference_caption="a car drives",
            )
            index = E5TargetIndex(
                records=[
                    E5TargetRecord(video_id="video1", video_path=str(root / "video1.mp4")),
                    E5TargetRecord(video_id="video2", video_path=str(root / "video2.mp4")),
                ],
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                metadata={},
            )
            runtime = E5OmniRuntime(
                config=E5OmniRuntimeConfig(model_path=str(root), device="cpu"),
                model=FakeE5Model(),
            )

            summary = run_e5_only_eval(
                e5_runtime=runtime,
                e5_index=index,
                triplets=[triplet],
                recall_ks=(1, 2),
                topk=2,
                output_dir=root / "e5_only",
            )

            self.assertEqual({"R@1": 1.0, "R@2": 1.0}, summary["recall"])
            self.assertTrue((root / "e5_only" / "traces.jsonl").exists())

    def test_cvr_agent_trace_records_avigate_e5_fused_and_final(self) -> None:
        runtime = FakeAvigateRuntime(
            text_rows=[],
            video_rows=[
                VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
                VideoRow(video_id="video2", video_path="/tmp/video2.mp4"),
            ],
            score_map={},
        )
        checker = MockOmniChecker(
            cvr_t2v_rerank_results={
                "a car. Edit: make it blue.": {
                    "ordered_video_ids": ["video2", "video1"],
                    "top_choice_video_id": "video2",
                    "confidence": 0.9,
                    "reason": "video2 reflects the edit",
                }
            }
        )
        avigate_hits = [RetrievalHit(rank=1, item_id="video1", score=0.9, video_id="video1")]
        e5_hits = [RetrievalHit(rank=1, item_id="video2", score=0.8, video_id="video2")]
        fused_hits = [
            RetrievalHit(rank=1, item_id="video1", score=0.02, video_id="video1"),
            RetrievalHit(rank=2, item_id="video2", score=0.02, video_id="video2"),
        ]

        trace = run_cvr_agent_case(
            sample_id="video2",
            query_text="a car. Edit: make it blue.",
            reference_video_path="/tmp/ref.mp4",
            edit_text="make it blue",
            reference_caption="a car",
            runtime=runtime,
            checker=checker,
            target_video_id="video2",
            avigate_hits=avigate_hits,
            e5_hits=e5_hits,
            fused_hits=fused_hits,
            fused_evidence=[{"video_id": "video2", "source_ranks": {"e5": 1}, "source_scores": {"e5": 0.8}}],
            rerank_window=2,
        )

        self.assertEqual("cvr-agent", trace["mode"])
        self.assertEqual("video2", trace["final_result"]["video_id"])
        self.assertIn("avigate_hits", trace)
        self.assertIn("e5_hits", trace)
        self.assertIn("fused_evidence", trace)
        self.assertEqual(5, trace["omni_calls"])

    def test_fusion_eval_reports_all_three_recall_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ref.mp4").write_bytes(b"x")
            triplet = CVRTriplet(
                sample_id="video2",
                reference_video=str(root / "ref.mp4"),
                target_video=str(root / "video2.mp4"),
                edit_text="make it blue",
                reference_caption="a car",
            )
            avigate_runtime = FakeAvigateRuntime(
                text_rows=[],
                video_rows=[
                    VideoRow(video_id="video1", video_path=str(root / "video1.mp4")),
                    VideoRow(video_id="video2", video_path=str(root / "video2.mp4")),
                ],
                score_map={"a car. Edit: make it blue.": np.asarray([0.9, 0.1], dtype=np.float32)},
            )
            e5_runtime = E5OmniRuntime(
                config=E5OmniRuntimeConfig(model_path=str(root), device="cpu"),
                model=FakeE5Model(),
            )
            e5_index = E5TargetIndex(
                records=[
                    E5TargetRecord(video_id="video1", video_path=str(root / "video1.mp4")),
                    E5TargetRecord(video_id="video2", video_path=str(root / "video2.mp4")),
                ],
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                metadata={},
            )

            summary = run_cvr_fusion_eval(
                avigate_runtime=avigate_runtime,
                e5_runtime=e5_runtime,
                e5_index=e5_index,
                triplets=[triplet],
                recall_ks=(1, 2),
                avigate_topk=2,
                e5_topk=2,
                fused_topk=2,
                output_dir=root / "fusion",
            )

            self.assertEqual({"R@1": 0.0, "R@2": 1.0}, summary["avigate_recall"])
            self.assertEqual({"R@1": 1.0, "R@2": 1.0}, summary["e5_recall"])
            self.assertEqual({"R@1": 0.0, "R@2": 1.0}, summary["fused_recall"])


if __name__ == "__main__":
    unittest.main()
