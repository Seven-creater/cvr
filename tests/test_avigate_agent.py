from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.avigate_agent import (
    run_official_agent_partial_eval,
    run_t2v_official_agent_case,
    run_v2t_official_agent_case,
)
from app.omni_checker import MockOmniChecker
from app.retrieval_types import TextRow, VideoRow


@dataclass
class FakeRuntime:
    text_rows: list[TextRow]
    video_rows: list[VideoRow]
    text_score_map: dict[tuple[str, str], np.ndarray]
    video_score_map: dict[tuple[str, str], np.ndarray]
    text_calls: list[tuple[str, str]] = field(default_factory=list)
    video_calls: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._video_index = {row.video_id: index for index, row in enumerate(self.video_rows)}
        self._text_index = {row.text_id: index for index, row in enumerate(self.text_rows)}

    def score_text_query(self, query_text: str, *, audio_mode: str = "on") -> np.ndarray:
        self.text_calls.append((query_text, audio_mode))
        return self.text_score_map[(query_text, audio_mode)]

    def score_video_query(self, video_id: str, *, audio_mode: str = "on") -> np.ndarray:
        self.video_calls.append((video_id, audio_mode))
        return self.video_score_map[(video_id, audio_mode)]

    def target_text_ids(self, video_id: str) -> list[str]:
        return [row.text_id for row in self.text_rows if row.video_id == video_id]


class AvigateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_rows = [
            VideoRow(video_id="video1", video_path="/tmp/video1.mp4"),
            VideoRow(video_id="video2", video_path="/tmp/video2.mp4"),
            VideoRow(video_id="video3", video_path="/tmp/video3.mp4"),
        ]
        self.text_rows = [
            TextRow(text_id="t1", video_id="video1", text="a person is cooking"),
            TextRow(text_id="t2", video_id="video2", text="a dog is running"),
            TextRow(text_id="t3", video_id="video3", text="a person fixes a computer"),
        ]

    def test_t2v_agent_uses_query_understanding_and_reranks_prefix(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={
                ("cook query", "on"): np.asarray([0.1, 0.9, 0.2], dtype=np.float32),
                ("better cook query", "off"): np.asarray([0.8, 0.9, 0.1], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_understanding_results={
                "cook query": {
                    "retrieval_text": "better cook query",
                    "summary": "person cooks food",
                    "main_events": ["cooking"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "irrelevant",
                    "reason": "visual signal is enough",
                }
            },
            video_description_results={
                "video2": {
                    "summary": "a dog runs outdoors",
                    "main_events": ["running"],
                    "objects": ["dog"],
                    "scene": "park",
                    "audio_cues": [],
                    "audio_relevance": "unknown",
                },
                "video1": {
                    "summary": "a person cooks in a kitchen",
                    "main_events": ["stirs food"],
                    "objects": ["pot"],
                    "scene": "kitchen",
                    "audio_cues": ["sizzle"],
                    "audio_relevance": "helpful",
                },
            },
            t2v_rerank_results={
                "better cook query": {
                    "ordered_video_ids": ["video1", "video2"],
                    "top_choice_video_id": "video1",
                    "confidence": 0.91,
                    "reason": "video1 matches cooking better",
                }
            },
        )

        trace = run_t2v_official_agent_case(
            query_text="cook query",
            runtime=runtime,
            checker=checker,
            target_video_id="video1",
            topk=3,
            rerank_window=2,
        )

        self.assertEqual([("better cook query", "off")], runtime.text_calls)
        self.assertEqual("video1", trace["target_video_id"])
        self.assertEqual("better cook query", trace["retrieval_hints"]["query_text_override"])
        self.assertEqual("off", trace["retrieval_hints"]["audio_mode"])
        self.assertEqual(["video2", "video1", "video3"], [hit["video_id"] for hit in trace["initial_hits"]])
        self.assertEqual(["video1", "video2", "video3"], [hit["video_id"] for hit in trace["reranked_hits"]])
        self.assertEqual("video1", trace["final_result"]["video_id"])
        self.assertEqual(2, trace["final_result"]["original_rank"])
        self.assertEqual(4, trace["omni_calls"])
        self.assertFalse(trace["fallback_used"])

    def test_v2t_agent_describes_video_then_reranks(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={},
            video_score_map={
                ("video3", "off"): np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            },
        )
        checker = MockOmniChecker(
            video_description_results={
                "video3": {
                    "summary": "a person repairs a computer",
                    "main_events": ["opens computer case"],
                    "objects": ["desktop computer"],
                    "scene": "desk",
                    "audio_cues": [],
                    "audio_relevance": "irrelevant",
                }
            },
            v2t_rerank_results={
                "video3": {
                    "ordered_text_ids": ["t3", "t1", "t2"],
                    "top_choice_text_id": "t3",
                    "confidence": 0.88,
                    "reason": "t3 best matches the repair scene",
                }
            },
        )

        trace = run_v2t_official_agent_case(
            query_video_id="video3",
            runtime=runtime,
            checker=checker,
            topk=3,
        )

        self.assertEqual([("video3", "off")], runtime.video_calls)
        self.assertEqual("off", trace["retrieval_hints"]["audio_mode"])
        self.assertEqual(["t1", "t2", "t3"], [hit["text_id"] for hit in trace["initial_hits"]])
        self.assertEqual(["t3", "t1", "t2"], [hit["text_id"] for hit in trace["reranked_hits"]])
        self.assertEqual("t3", trace["final_result"]["text_id"])
        self.assertEqual(3, trace["final_result"]["original_rank"])
        self.assertEqual(2, trace["omni_calls"])
        self.assertFalse(trace["fallback_used"])

    def test_t2v_agent_falls_back_to_initial_hits_when_rerank_is_invalid(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={
                ("cook query", "on"): np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_understanding_results={
                "cook query": {
                    "retrieval_text": "cook query",
                    "summary": "person cooks",
                    "main_events": ["cooking"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                    "reason": "same query works",
                }
            },
            video_description_results={
                "video1": {
                    "summary": "person cooks",
                    "main_events": ["cooking"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                },
                "video2": {
                    "summary": "dog runs",
                    "main_events": ["running"],
                    "objects": ["dog"],
                    "scene": "park",
                    "audio_cues": [],
                    "audio_relevance": "unknown",
                },
            },
            t2v_rerank_results={
                "cook query": {
                    "ordered_video_ids": ["video999"],
                    "top_choice_video_id": "video999",
                    "confidence": 0.2,
                    "reason": "invalid id should trigger repair fallback",
                }
            },
        )

        trace = run_t2v_official_agent_case(
            query_text="cook query",
            runtime=runtime,
            checker=checker,
            topk=3,
            rerank_window=2,
        )

        self.assertTrue(trace["fallback_used"])
        self.assertEqual("t2v_rerank", trace["fallback_stage"])
        self.assertEqual(
            [hit["video_id"] for hit in trace["initial_hits"]],
            [hit["video_id"] for hit in trace["reranked_hits"]],
        )

    def test_t2v_agent_reuses_cached_video_descriptions_across_cases(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={
                ("repeat query", "on"): np.asarray([0.9, 0.8, 0.1], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_understanding_results={
                "repeat query": {
                    "retrieval_text": "repeat query",
                    "summary": "person cooks",
                    "main_events": ["cooking"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                    "reason": "same query works",
                }
            },
            video_description_results={
                "video1": {
                    "summary": "person cooks",
                    "main_events": ["cooking"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                }
            },
            t2v_rerank_results={
                "repeat query": {
                    "ordered_video_ids": ["video1"],
                    "top_choice_video_id": "video1",
                    "confidence": 0.9,
                    "reason": "video1 is the only described candidate",
                }
            },
        )

        run_t2v_official_agent_case(
            query_text="repeat query",
            runtime=runtime,
            checker=checker,
            topk=1,
            omni_concurrency=1,
            rerank_window=1,
        )
        run_t2v_official_agent_case(
            query_text="repeat query",
            runtime=runtime,
            checker=checker,
            topk=1,
            omni_concurrency=1,
            rerank_window=1,
        )

        self.assertEqual(["video1"], checker.video_description_calls)

    def test_partial_eval_uses_reranked_hits_for_final_recall(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=[self.video_rows[2]],
            text_score_map={},
            video_score_map={
                ("video3", "on"): np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            },
        )
        checker = MockOmniChecker(
            video_description_results={
                "video3": {
                    "summary": "a person repairs a computer",
                    "main_events": ["opens computer case"],
                    "objects": ["desktop computer"],
                    "scene": "desk",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                }
            },
            v2t_rerank_results={
                "video3": {
                    "ordered_text_ids": ["t3", "t1", "t2"],
                    "top_choice_text_id": "t3",
                    "confidence": 0.88,
                    "reason": "t3 best matches the repair scene",
                }
            },
        )

        result = run_official_agent_partial_eval(
            mode="v2t",
            runtime=runtime,
            checker=checker,
            sample_size=1,
            topk=3,
            recall_ks=(1, 3),
        )

        summary = result["summary"]
        self.assertEqual({"R@1": 0.0, "R@3": 1.0}, summary["round1_recall"])
        self.assertEqual({"R@1": 1.0, "R@3": 1.0}, summary["final_recall"])
        self.assertEqual(1.0, summary["final_top1_accuracy"])
        self.assertEqual(2.0, summary["avg_omni_calls"])
        self.assertEqual(0.0, summary["audio_off_rate"])
        self.assertEqual(0.0, summary["fallback_rate"])

    def test_partial_eval_writes_summary_and_tracks_t2v_rewrite_rate(self) -> None:
        runtime = FakeRuntime(
            text_rows=[TextRow(text_id="t1", video_id="video1", text="cook query")],
            video_rows=self.video_rows,
            text_score_map={
                ("better cook query", "off"): np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_understanding_results={
                "cook query": {
                    "retrieval_text": "better cook query",
                    "summary": "person cooks",
                    "main_events": ["cooking"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "irrelevant",
                    "reason": "visual signal is enough",
                }
            },
            video_description_results={
                "video2": {
                    "summary": "a dog runs",
                    "main_events": ["running"],
                    "objects": ["dog"],
                    "scene": "park",
                    "audio_cues": [],
                    "audio_relevance": "unknown",
                },
                "video1": {
                    "summary": "a person cooks",
                    "main_events": ["stirs food"],
                    "objects": ["pan"],
                    "scene": "kitchen",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                },
            },
            t2v_rerank_results={
                "better cook query": {
                    "ordered_video_ids": ["video1", "video2"],
                    "top_choice_video_id": "video1",
                    "confidence": 0.91,
                    "reason": "video1 matches cooking",
                }
            },
        )
        progress_messages: list[str] = []
        runs_dir = Path(__file__).resolve().parents[1] / "runs"
        with tempfile.TemporaryDirectory(dir=runs_dir) as temp_dir:
            result = run_official_agent_partial_eval(
                mode="t2v",
                runtime=runtime,
                checker=checker,
                sample_size=1,
                topk=3,
                rerank_window=2,
                recall_ks=(1, 2),
                output_dir=temp_dir,
                progress=progress_messages.append,
            )

            summary = result["summary"]
            self.assertEqual({"R@1": 0.0, "R@2": 1.0}, summary["round1_recall"])
            self.assertEqual({"R@1": 1.0, "R@2": 1.0}, summary["final_recall"])
            self.assertEqual(1.0, summary["query_rewrite_rate"])
            self.assertEqual(1.0, summary["audio_off_rate"])
            self.assertEqual(4.0, summary["avg_omni_calls"])

            summary_path = Path(result["summary_path"])
            traces_path = Path(result["traces_path"])
            self.assertTrue(summary_path.exists())
            self.assertTrue(traces_path.exists())
            self.assertEqual(summary, json.loads(summary_path.read_text(encoding="utf-8")))
            self.assertEqual(1, len(traces_path.read_text(encoding="utf-8").strip().splitlines()))
            self.assertTrue(any("start 1/1" in message for message in progress_messages))
            self.assertTrue(any("done 1/1" in message for message in progress_messages))

    def test_partial_eval_start_index_selects_later_rows(self) -> None:
        runtime = FakeRuntime(
            text_rows=self.text_rows,
            video_rows=self.video_rows,
            text_score_map={
                ("a dog is running", "on"): np.asarray([0.2, 0.9, 0.1], dtype=np.float32),
            },
            video_score_map={},
        )
        checker = MockOmniChecker(
            t2v_understanding_results={
                "a dog is running": {
                    "retrieval_text": "a dog is running",
                    "summary": "dog runs",
                    "main_events": ["running"],
                    "objects": ["dog"],
                    "scene": "park",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                    "reason": "query is already specific",
                }
            },
            video_description_results={
                "video2": {
                    "summary": "a dog runs outdoors",
                    "main_events": ["running"],
                    "objects": ["dog"],
                    "scene": "park",
                    "audio_cues": [],
                    "audio_relevance": "helpful",
                }
            },
            t2v_rerank_results={
                "a dog is running": {
                    "ordered_video_ids": ["video2"],
                    "top_choice_video_id": "video2",
                    "confidence": 0.95,
                    "reason": "video2 is the only described candidate",
                }
            },
        )

        result = run_official_agent_partial_eval(
            mode="t2v",
            runtime=runtime,
            checker=checker,
            start_index=1,
            sample_size=1,
            topk=1,
            rerank_window=1,
            recall_ks=(1,),
        )

        self.assertEqual([("a dog is running", "on")], runtime.text_calls)
        self.assertEqual({"R@1": 1.0}, result["summary"]["final_recall"])


if __name__ == "__main__":
    unittest.main()
