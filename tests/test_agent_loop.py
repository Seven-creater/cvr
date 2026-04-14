from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.agent_loop import RetrievalParams, run_agent_case, summarize_agent_traces
from app.omni_checker import MockOmniChecker
from app.retriever import FeatureRetriever


class MockEncoder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = {key: np.asarray(value, dtype=np.float32) for key, value in mapping.items()}

    def encode(self, text: str) -> np.ndarray:
        return self.mapping[text]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        write_jsonl(
            self.root / "video_rows.jsonl",
            [
                {"video_id": "video1", "video_path": "/tmp/video1.mp4"},
                {"video_id": "video2", "video_path": "/tmp/video2.mp4"},
            ],
        )
        write_jsonl(
            self.root / "text_rows.jsonl",
            [
                {"text_id": "t1", "video_id": "video1", "text": "guitar stage music"},
                {"text_id": "t2", "video_id": "video2", "text": "dog park running"},
            ],
        )
        np.save(self.root / "text_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.save(self.root / "video_visual_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.save(self.root / "video_audio_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_t2v_agent_can_retry_then_improve(self) -> None:
        retriever = FeatureRetriever.from_feature_dir(
            self.root,
            text_encoder=MockEncoder({"bad query": [0.0, 1.0], "good query": [1.0, 0.0]}),
        )
        checker = MockOmniChecker(
            t2v_results={
                "bad query::video2": {
                    "is_match": False,
                    "confidence": 0.2,
                    "visual_match": 0.2,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["guitar"],
                    "reason": "wrong",
                    "rewrite_suggestion": "good query",
                },
                "bad query::video1": {
                    "is_match": False,
                    "confidence": 0.2,
                    "visual_match": 0.2,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["guitar"],
                    "reason": "needs rewrite",
                    "rewrite_suggestion": "good query",
                },
                "good query::video1": {
                    "is_match": True,
                    "confidence": 0.9,
                    "visual_match": 0.9,
                    "audio_match": 0.6,
                    "main_events": ["guitar"],
                    "missing_elements": [],
                    "reason": "good",
                    "rewrite_suggestion": "good query",
                },
            }
        )
        trace = run_agent_case(
            mode="t2v",
            retriever=retriever,
            checker=checker,
            query_text="bad query",
            target_video_id="video1",
            initial_params=RetrievalParams(),
            max_rounds=3,
        )
        self.assertGreaterEqual(len(trace["rounds"]), 2)
        self.assertEqual(trace["target_rank_after"], 1)
        self.assertNotIn("target_video_id", trace["rounds"][0]["controller_steps"][0]["state_text"])

    def test_v2t_agent_can_loop(self) -> None:
        retriever = FeatureRetriever.from_feature_dir(self.root)
        checker = MockOmniChecker(
            v2t_results={
                "video1::t2": {
                    "is_match": False,
                    "confidence": 0.1,
                    "visual_match": 0.1,
                    "audio_match": 0.1,
                    "main_events": [],
                    "missing_elements": ["guitar"],
                    "reason": "wrong text",
                    "rewrite_suggestion": "",
                },
                "video1::t1": {
                    "is_match": True,
                    "confidence": 0.9,
                    "visual_match": 0.9,
                    "audio_match": 0.5,
                    "main_events": ["guitar"],
                    "missing_elements": [],
                    "reason": "correct text",
                    "rewrite_suggestion": "",
                },
            }
        )
        trace = run_agent_case(
            mode="v2t",
            retriever=retriever,
            checker=checker,
            query_video_id="video1",
            target_text_ids=["t1"],
            initial_params=RetrievalParams(alpha_visual=0.2, alpha_audio=0.8, topk=10),
            max_rounds=3,
        )
        self.assertLessEqual(len(trace["rounds"]), 3)
        self.assertEqual(trace["success"], True)

    def test_summary_reports_rates(self) -> None:
        summary = summarize_agent_traces(
            [
                {
                    "rounds": [{}, {}],
                    "checker_call_count": 3,
                    "final_submission": {"rank": 2},
                    "target_rank_before": 5,
                    "target_rank_after": 1,
                },
                {
                    "rounds": [{}],
                    "checker_call_count": 1,
                    "final_submission": {"rank": 1},
                    "target_rank_before": 1,
                    "target_rank_after": 1,
                },
            ],
            [1, 5, 10],
        )
        self.assertEqual(summary["retry_rate"], 0.5)
        self.assertEqual(summary["submit_top1_rate"], 0.5)
        self.assertEqual(summary["submit_top2_rate"], 0.5)
