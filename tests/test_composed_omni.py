from __future__ import annotations

import base64
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from app.composed_omni import OpenAIComposedDataClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ComposedOmniClientTests(unittest.TestCase):
    def test_annotate_clip_materializes_local_video_path_as_data_url(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "two cats on a sofa",
                                "subjects": ["cats"],
                                "object_counts": {"cat": 2},
                                "actions": ["resting"],
                                "scene": "living room",
                                "attributes": ["orange"],
                                "on_screen_text": [],
                                "speech": "soft narration",
                                "audio_events": [],
                                "modalities": ["visual"],
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            request_holder["timeout"] = timeout
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-composed-omni-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8092/v1",
                api_key="EMPTY",
                model="captioner-model",
                timeout_seconds=33.0,
            )

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.annotate_clip(clip_path=str(clip_path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual("two cats on a sofa", normalized["summary"])
        self.assertEqual(["soft narration"], normalized["speech"])
        request = request_holder["request"]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual({"type": "json_object"}, request_body["response_format"])
        user_content = request_body["messages"][1]["content"]
        encoded_url = user_content[0]["video_url"]["url"]
        self.assertTrue(encoded_url.startswith("data:video/mp4;base64,"))
        encoded_bytes = base64.b64decode(encoded_url.split(",", 1)[1])
        self.assertEqual(b"fake-mp4-bytes", encoded_bytes)
        self.assertEqual(33.0, request_holder["timeout"])

    def test_propose_pair_uses_json_object_response_format(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "edit_text": "change one cat into two cats",
                                "modalities": ["visual", "audio"],
                                "reference_caption": "one cat on a sofa",
                                "target_caption": "two cats on a sofa",
                                "difference": {
                                    "type": "object_count",
                                    "from": "one cat",
                                    "to": "two cats",
                                    "description": "the cat count increases",
                                },
                                "proposal_reason": "same room and same subject with one clear change",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            request_holder["timeout"] = timeout
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8092/v1",
            api_key="EMPTY",
            model="instruct-model",
            timeout_seconds=22.0,
        )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            normalized, _raw_payload = client.propose_pair(
                reference_annotation={
                    "clip_id": "ref",
                    "summary": "one cat on a sofa",
                    "subjects": ["cat"],
                    "object_counts": {"cat": 1},
                    "actions": ["resting"],
                    "scene": "living room",
                    "attributes": ["orange"],
                    "on_screen_text": [],
                    "speech": [],
                    "audio_events": [],
                    "modalities": ["visual"],
                },
                target_annotation={
                    "clip_id": "target",
                    "summary": "two cats on a sofa",
                    "subjects": ["cats"],
                    "object_counts": {"cat": 2},
                    "actions": ["resting"],
                    "scene": "living room",
                    "attributes": ["orange"],
                    "on_screen_text": [],
                    "speech": [],
                    "audio_events": ["cat meow"],
                    "modalities": ["visual", "audio"],
                },
                hard_negative_candidates=[
                    {
                        "clip_id": "neg1",
                        "summary": "one dog on a sofa",
                    }
                ],
            )

        self.assertEqual("change one cat into two cats", normalized["edit_text"])
        request = request_holder["request"]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual({"type": "json_object"}, request_body["response_format"])
        self.assertEqual("instruct-model", request_body["model"])
        system_prompt = request_body["messages"][0]["content"]
        self.assertIn("compare 1-3 possible differences internally", system_prompt)
        self.assertIn("Reject vague edit_text", system_prompt)
        self.assertIn("include both from/to text", system_prompt)
        self.assertEqual(22.0, request_holder["timeout"])

    def test_refine_b_line_edit_text_requires_specific_audio_instruction(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "refined_edit_text": "change the speech from discussing the bakery opening to discussing the mayor's remarks",
                                "edit_text_specificity_score": 0.91,
                                "reject_if_unspecific": False,
                                "edit_text_reject_reason": "",
                                "speech_or_audio_evidence": [
                                    "reference speech mentions a bakery opening",
                                    "target speech mentions the mayor's remarks",
                                ],
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            request_holder["timeout"] = timeout
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-composed-omni-refine-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            ref_path = tmp_dir / "ref.mp4"
            tgt_path = tmp_dir / "tgt.mp4"
            ref_path.write_bytes(b"fake-ref")
            tgt_path.write_bytes(b"fake-tgt")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8092/v1",
                api_key="EMPTY",
                model="instruct-model",
                timeout_seconds=44.0,
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.refine_b_line_edit_text(
                    reference_clip_path=str(ref_path),
                    target_clip_path=str(tgt_path),
                    model_fields={"edit_text": "speech content has been altered", "difference": {"type": "speech"}},
                    final_verification={"audio_primary": True, "visual_locked": True},
                    reference_annotation={"speech": ["bakery opening"], "summary": "same speaker"},
                    target_annotation={"speech": ["mayor remarks"], "summary": "same speaker"},
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(
            "change the speech from discussing the bakery opening to discussing the mayor's remarks",
            normalized["refined_edit_text"],
        )
        self.assertAlmostEqual(0.91, normalized["edit_text_specificity_score"])
        request = request_holder["request"]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual({"type": "json_object"}, request_body["response_format"])
        system_prompt = request_body["messages"][0]["content"]
        self.assertIn("Reject vague wording such as unintelligible speech", system_prompt)
        self.assertIn("not transcribed", system_prompt)
        self.assertIn("target audio", system_prompt)
        self.assertEqual(44.0, request_holder["timeout"])

    def test_refine_b_line_speech_content_listens_for_specific_topics(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reference_speech_content": "budget planning",
                                "target_speech_content": "health services",
                                "speech_transcription_confidence": 0.92,
                                "speech_language": "English",
                                "refined_edit_text": "change the speech from discussing budget planning to discussing health services",
                                "reject_if_still_unclear": False,
                                "speech_rewrite_reject_reason": "",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            request_holder["timeout"] = timeout
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-composed-omni-speech-rewrite-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            ref_path = tmp_dir / "ref.mp4"
            tgt_path = tmp_dir / "tgt.mp4"
            ref_path.write_bytes(b"fake-ref")
            tgt_path.write_bytes(b"fake-tgt")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8092/v1",
                api_key="EMPTY",
                model="instruct-model",
                timeout_seconds=55.0,
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.refine_b_line_speech_content(
                    reference_clip_path=str(ref_path),
                    target_clip_path=str(tgt_path),
                    model_fields={
                        "edit_text": "change the speech from discussing A to discussing B",
                        "difference": {"type": "speech"},
                    },
                    final_verification={"audio_primary": True, "visual_locked": True},
                    edit_text_refinement={
                        "refined_edit_text": "change the speech from discussing A to discussing B",
                        "edit_text_specificity_score": 0.95,
                    },
                    reference_annotation={"speech": ["budget planning"], "summary": "same speaker"},
                    target_annotation={"speech": ["health services"], "summary": "same speaker"},
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual("budget planning", normalized["reference_speech_content"])
        self.assertEqual("health services", normalized["target_speech_content"])
        self.assertEqual(
            "change the speech from discussing budget planning to discussing health services",
            normalized["refined_edit_text"],
        )
        self.assertFalse(normalized["reject_if_still_unclear"])
        request = request_holder["request"]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual({"type": "json_object"}, request_body["response_format"])
        system_prompt = request_body["messages"][0]["content"]
        user_content = request_body["messages"][1]["content"]
        user_text = "\n".join(item.get("text", "") for item in user_content if item.get("type") == "text")
        self.assertIn("listen to the reference and target clips", system_prompt)
        self.assertIn("Paraphrase is allowed", system_prompt)
        self.assertIn("Do not output placeholders", system_prompt)
        self.assertIn("listening to the audio only", user_text)
        self.assertEqual(55.0, request_holder["timeout"])

    def test_request_json_repairs_malformed_json_response(self) -> None:
        requests: list[object] = []
        repaired_payload = {
            "edit_text": "change one cat into two cats",
            "modalities": ["visual"],
            "reference_caption": "one cat on a sofa",
            "target_caption": "two cats on a sofa",
            "difference": {
                "type": "object_count",
                "from": "one cat",
                "to": "two cats",
                "description": "the cat count increases",
            },
            "proposal_reason": "same room and same subject with one clear change",
        }
        responses = [
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"edit_text": "change one cat into two cats" "modalities": ["visual"]}'
                            }
                        }
                    ]
                }
            ),
            _FakeHTTPResponse({"choices": [{"message": {"content": json.dumps(repaired_payload)}}]}),
        ]

        def fake_urlopen(request, timeout):
            requests.append(request)
            return responses.pop(0)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8092/v1",
            api_key="EMPTY",
            model="instruct-model",
            timeout_seconds=22.0,
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            normalized, _raw_payload = client.propose_pair(
                reference_annotation={"clip_id": "ref", "summary": "one cat"},
                target_annotation={"clip_id": "target", "summary": "two cats"},
                hard_negative_candidates=[],
            )

        self.assertEqual("change one cat into two cats", normalized["edit_text"])
        self.assertEqual(2, len(requests))
        repair_request_body = json.loads(requests[1].data.decode("utf-8"))
        self.assertIn("Repair the malformed JSON-like", repair_request_body["messages"][0]["content"])
        self.assertNotIn("video_url", json.dumps(repair_request_body))

    def test_propose_single_source_pair_materializes_both_videos_and_requires_evidence(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "edit_text": "add a picture-in-picture demonstration overlay",
                                "modalities": ["visual"],
                                "reference_caption": "a woman speaks without an overlay",
                                "target_caption": "a woman speaks with a picture-in-picture demo overlay",
                                "difference": {
                                    "type": "object_presence",
                                    "from": "no picture-in-picture demonstration overlay",
                                    "to": "picture-in-picture demonstration overlay",
                                    "description": "a picture-in-picture demonstration overlay appears",
                                },
                                "dominant_delta": {
                                    "type": "object_presence",
                                    "from": "no overlay",
                                    "to": "picture-in-picture overlay",
                                    "reason": "the overlay is the clearest visual difference",
                                },
                                "reference_state": {
                                    "main_speaker": "woman presenter",
                                    "inset_subjects": [],
                                    "product_overlay": "",
                                    "composition": "speaker-only talking-head shot",
                                    "internal_transitions": [],
                                },
                                "target_state": {
                                    "main_speaker": "woman presenter",
                                    "inset_subjects": ["brow treatment demonstrator"],
                                    "product_overlay": "",
                                    "composition": "talking-head shot with picture-in-picture overlay",
                                    "internal_transitions": [],
                                },
                                "delta_temporal_extent": {
                                    "reference": "no overlay throughout",
                                    "target": "overlay appears for most of the target clip",
                                    "target_coverage": 0.86,
                                    "evidence": "target frames show the overlay",
                                },
                                "subject_roles": {
                                    "main_speaker": "woman presenter",
                                    "inset_subjects": ["brow treatment demonstrator"],
                                    "product_overlay": "",
                                },
                                "is_segment_wide_delta": True,
                                "discarded_deltas": ["minor blouse/shirt wording"],
                                "evidence": ["target has a picture-in-picture overlay while reference does not"],
                                "confidence": 0.86,
                                "accept": True,
                                "reject_reason": "",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.propose_single_source_pair(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    reference_annotation={"clip_id": "ref", "summary": "a woman speaks"},
                    target_annotation={"clip_id": "target", "summary": "a woman speaks with overlay"},
                )

        self.assertTrue(normalized["accept"])
        self.assertEqual("object_presence", normalized["difference"]["type"])
        self.assertEqual(["target has a picture-in-picture overlay while reference does not"], normalized["evidence"])
        self.assertTrue(normalized["is_segment_wide_delta"])
        self.assertEqual(0.86, normalized["delta_temporal_extent"]["target_coverage"])
        self.assertEqual(["brow treatment demonstrator"], normalized["subject_roles"]["inset_subjects"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        content = request_body["messages"][1]["content"]
        self.assertEqual("Reference clip:", content[0]["text"])
        self.assertTrue(content[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual("Target clip:", content[2]["text"])
        self.assertTrue(content[3]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertIn("picture-in-picture", request_body["messages"][0]["content"])
        self.assertIn("main speaker", request_body["messages"][0]["content"])
        self.assertIn("inset", request_body["messages"][0]["content"])

    def test_propose_single_source_pair_includes_v4_audio_line_guidance(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "edit_text": "change the shot from a studio anchor to aerial flood footage",
                                "modalities": ["visual"],
                                "reference_caption": "a news anchor speaks in a studio",
                                "target_caption": "aerial flood footage is shown",
                                "difference": {"type": "scene", "from": "studio anchor", "to": "aerial flood footage", "description": "large visual shot change"},
                                "dominant_delta": {"type": "scene", "from": "studio", "to": "flood aerial", "reason": "large visual change"},
                                "reference_state": {"main_speaker": "anchor", "inset_subjects": [], "product_overlay": "", "composition": "studio anchor", "internal_transitions": []},
                                "target_state": {"main_speaker": "", "inset_subjects": [], "product_overlay": "", "composition": "aerial footage", "internal_transitions": []},
                                "delta_temporal_extent": {"reference": "studio", "target": "aerial", "target_coverage": 0.9, "evidence": "target shows flood aerial"},
                                "subject_roles": {"main_speaker": "anchor", "inset_subjects": [], "product_overlay": ""},
                                "is_segment_wide_delta": True,
                                "discarded_deltas": [],
                                "evidence": ["target is flood aerial footage"],
                                "confidence": 0.9,
                                "accept": True,
                                "reject_reason": "",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client.propose_single_source_pair(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    reference_annotation={"clip_id": "ref", "summary": "a news anchor speaks"},
                    target_annotation={"clip_id": "target", "summary": "flood aerial footage"},
                    candidate={
                        "audio_dataset_line": "visual_audio_anchor",
                        "quality": {"audio_line_quality_profile": "v4_strict"},
                        "instruction": "v4_strict: accept only large visual changes like a studio anchor shot to flood aerial footage",
                    },
                    audio_dataset_line="visual_audio_anchor",
                )

        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        system_prompt = request_body["messages"][0]["content"]
        user_text = request_body["messages"][1]["content"][-1]["text"]
        self.assertIn("news/program audio context", system_prompt)
        self.assertIn("studio anchor shot to flood aerial footage", user_text)
        self.assertIn("v4_strict", user_text)

    def test_audio_line_single_source_prompts_stay_compact(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "edit_text": "change the speech from discussing budget plans to discussing health services",
                                "modalities": ["audio"],
                                "reference_caption": "a speaker talks at a desk",
                                "target_caption": "the same speaker talks at a desk",
                                "difference": {"type": "speech", "from": "budget plans", "to": "health services", "description": "speech topic changes"},
                                "dominant_delta": {"type": "speech", "from": "budget", "to": "health", "reason": "spoken content differs"},
                                "reference_state": {},
                                "target_state": {},
                                "delta_temporal_extent": {"reference": "whole clip", "target": "whole clip", "target_coverage": 0.9, "evidence": "speech evidence"},
                                "subject_roles": {},
                                "is_segment_wide_delta": True,
                                "discarded_deltas": [],
                                "evidence": ["the spoken topic changes"],
                                "confidence": 0.9,
                                "accept": True,
                                "reject_reason": "",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        long_annotation = {
            "clip_id": "clip",
            "summary": "same livestream speaker at a desk " * 80,
            "scene": "indoor livestream desk " * 60,
            "subjects": ["speaker at desk"] * 20,
            "actions": ["speaking to camera"] * 20,
            "speech": ["the speaker talks about budget planning and transportation funding " * 20 for _ in range(8)],
            "speakers_and_transcript": ["speaker: a long transcript about policy funding " * 25 for _ in range(8)],
            "audio_events": ["speech", "room ambience", "chair noise"] * 10,
            "modalities": ["visual", "audio"],
        }
        candidate = {
            "audio_dataset_line": "speech_audio_content",
            "quality": {"audio_line_quality_profile": "v5_audio_primary"},
            "instruction": "B-line speech content candidate " * 80,
        }

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client.propose_single_source_pair(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    reference_annotation=long_annotation,
                    target_annotation=long_annotation,
                    candidate=candidate,
                    audio_dataset_line="speech_audio_content",
                )

        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        system_prompt = request_body["messages"][0]["content"]
        user_text = request_body["messages"][1]["content"][-1]["text"]
        self.assertLess(len(system_prompt), 2300)
        self.assertLess(len(user_text), 3200)
        self.assertIn("speech_audio_content", system_prompt)
        self.assertIn("B-line", user_text)

    def test_verify_single_source_pair_final_materializes_videos_and_scores_quality(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "accept": True,
                                "confidence": 0.82,
                                "quality_score": 0.74,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "observable_delta": True,
                                "single_primary_delta": True,
                                "text_or_ocr_driven": False,
                                "segment_wide": True,
                                "edit_text_accurate": True,
                                "main_reject_reason": "",
                                "evidence": ["target has a stable product overlay; reference does not"],
                                "recommended_edit_text": "add a static product image overlay on the left",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.verify_single_source_pair_final(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    model_fields={
                        "edit_text": "add a static product image overlay on the left",
                        "dominant_delta": {"type": "object_presence", "from": "no overlay", "to": "product overlay"},
                    },
                    reference_annotation={"clip_id": "ref", "summary": "speaker only"},
                    target_annotation={"clip_id": "target", "summary": "speaker with product overlay"},
                    local_gate_report={"passed": True, "hard_reject": [], "review_required": []},
                )

        self.assertTrue(normalized["accept"])
        self.assertEqual(0.74, normalized["quality_score"])
        self.assertEqual("add a static product image overlay on the left", normalized["recommended_edit_text"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        content = request_body["messages"][1]["content"]
        self.assertEqual("Reference clip for final verification:", content[0]["text"])
        self.assertTrue(content[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual("Target clip for final verification:", content[2]["text"])
        self.assertTrue(content[3]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertIn("quality_score", request_body["messages"][0]["content"])
        self.assertIn("0.7 is borderline", request_body["messages"][0]["content"])

    def test_verify_single_source_pair_final_b_line_requires_audio_primary_fields(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "accept": True,
                                "confidence": 0.88,
                                "quality_score": 0.82,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "observable_delta": True,
                                "single_primary_delta": True,
                                "text_or_ocr_driven": False,
                                "segment_wide": True,
                                "edit_text_accurate": True,
                                "main_reject_reason": "",
                                "evidence": ["the same podium shot has different spoken content"],
                                "recommended_edit_text": "change the speech from discussing budget to discussing health",
                                "audio_primary": True,
                                "visual_locked": True,
                                "visual_too_different_for_B": False,
                                "edit_text_audio_only": True,
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.verify_single_source_pair_final(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    model_fields={
                        "edit_text": "change the speech from discussing budget to discussing health",
                        "dominant_delta": {"type": "speech", "from": "budget", "to": "health"},
                    },
                    reference_annotation={"clip_id": "ref", "summary": "speaker discusses budget"},
                    target_annotation={"clip_id": "target", "summary": "speaker discusses health"},
                    local_gate_report={"passed": True, "hard_reject": [], "review_required": []},
                    audio_dataset_line="speech_audio_content",
                )

        self.assertTrue(normalized["audio_primary"])
        self.assertTrue(normalized["visual_locked"])
        self.assertFalse(normalized["visual_too_different_for_B"])
        self.assertTrue(normalized["edit_text_audio_only"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        system_prompt = request_body["messages"][0]["content"]
        user_text = request_body["messages"][1]["content"][-1]["text"]
        self.assertIn("audio_primary", system_prompt)
        self.assertIn("visual_locked", system_prompt)
        self.assertIn("edit_text_audio_only", user_text)

    def test_audio_line_final_verification_repairs_missing_auxiliary_schema_fields(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "accept": True,
                                "confidence": 0.81,
                                "quality_score": 0.78,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "observable_delta": True,
                                "audio_primary": True,
                                "visual_locked": True,
                                "visual_too_different_for_B": False,
                                "edit_text_audio_only": True,
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.verify_single_source_pair_final(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    model_fields={
                        "edit_text": "change the speech from discussing budget to discussing health",
                        "modalities": ["audio"],
                        "difference": {
                            "type": "speech",
                            "from": "budget",
                            "to": "health",
                            "description": "spoken topic changes",
                        },
                        "dominant_delta": {
                            "type": "speech",
                            "from": "budget",
                            "to": "health",
                            "reason": "the same podium shot has a different spoken topic",
                        },
                        "evidence": ["reference speech discusses budget; target speech discusses health"],
                        "is_segment_wide_delta": True,
                        "confidence": 0.84,
                        "accept": True,
                    },
                    reference_annotation={"clip_id": "ref", "summary": "speaker discusses budget"},
                    target_annotation={"clip_id": "target", "summary": "speaker discusses health"},
                    local_gate_report={"passed": True, "hard_reject": [], "review_required": [], "all_issues": []},
                    audio_dataset_line="speech_audio_content",
                )

        self.assertTrue(normalized["accept"])
        self.assertTrue(normalized["single_primary_delta"])
        self.assertTrue(normalized["segment_wide"])
        self.assertTrue(normalized["edit_text_accurate"])
        self.assertEqual("", normalized["main_reject_reason"])
        self.assertEqual(
            "change the speech from discussing budget to discussing health",
            normalized["recommended_edit_text"],
        )
        self.assertIn("reference speech discusses budget", normalized["evidence"][0])
        self.assertTrue(normalized["audio_primary"])
        self.assertTrue(normalized["visual_locked"])
        self.assertFalse(normalized["visual_too_different_for_B"])
        self.assertTrue(normalized["edit_text_audio_only"])
        self.assertIn("single_primary_delta", normalized["schema_repaired_fields"])
        self.assertIn("evidence", normalized["schema_repaired_fields"])

    def test_verify_single_source_pair_final_caps_rejected_quality_score(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "accept": False,
                                "confidence": 0.95,
                                "quality_score": 0.9,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "observable_delta": True,
                                "single_primary_delta": True,
                                "text_or_ocr_driven": False,
                                "segment_wide": True,
                                "edit_text_accurate": True,
                                "main_reject_reason": "the clips are effectively the same apart from product text",
                                "evidence": ["only the product image label changes"],
                                "recommended_edit_text": "change the product image label",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.verify_single_source_pair_final(
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                    model_fields={"edit_text": "change the product image label"},
                    reference_annotation={"clip_id": "ref", "summary": "speaker with product image"},
                    target_annotation={"clip_id": "target", "summary": "speaker with similar product image"},
                    local_gate_report={"passed": True, "hard_reject": [], "review_required": []},
                )

        self.assertFalse(normalized["accept"])
        self.assertEqual(0.69, normalized["quality_score"])

    def test_plan_video_edit_materializes_reference_video_and_normalizes_plan(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "should_generate": True,
                                "source_prompt": "A close-up video of a person holding a mobile phone at a desk.",
                                "target_prompt": "A close-up video of the same person at the same desk holding a tablet instead of the mobile phone.",
                                "edit_token": "tablet",
                                "preserve_tokens": ["person", "desk", "camera motion", "lighting", "timing"],
                                "negative_prompt": "Do not change the person, desk, camera, lighting, timing, or visible text.",
                                "edit_region": "hand-held object",
                                "model_route": "vace_controlled",
                                "reason": "The phone is visible and can be locally replaced.",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            request_holder["timeout"] = timeout
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-video-edit-plan-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
                timeout_seconds=44.0,
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.plan_video_edit(
                    reference_clip_path=str(clip_path),
                    reference_annotation={"summary": "a person holds a mobile phone at a desk"},
                    candidate={
                        "edit_text": "replace the mobile phone with a tablet",
                        "difference": {"type": "object_presence", "from": "mobile phone", "to": "tablet"},
                    },
                    route_hint="vace_controlled",
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertTrue(normalized["should_generate"])
        self.assertEqual("tablet", normalized["edit_token"])
        self.assertEqual("vace_controlled", normalized["model_route"])
        self.assertIn("camera motion", normalized["preserve_tokens"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        self.assertEqual({"type": "json_object"}, request_body["response_format"])
        user_content = request_body["messages"][1]["content"]
        self.assertEqual("video_url", user_content[0]["type"])
        self.assertTrue(user_content[0]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertIn("Candidate edit JSON", user_content[1]["text"])
        self.assertEqual(44.0, request_holder["timeout"])

    def test_plan_video_edit_repairs_empty_target_prompt_without_discarding_plan(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "should_generate": True,
                                "source_prompt": "A close-up video of a hand writing on white paper.",
                                "target_prompt": "",
                                "edit_token": "blue star sticker",
                                "preserve_tokens": ["hand", "paper", "camera motion", "lighting"],
                                "negative_prompt": "Do not change the hand, paper, writing, camera, timing, or lighting.",
                                "edit_region": "top-right paper surface",
                                "model_route": "vace_controlled",
                                "reason": "The paper surface is visible and localized.",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-video-edit-plan-repair-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.plan_video_edit(
                    reference_clip_path=str(clip_path),
                    reference_annotation={"summary": "a hand writes on white paper"},
                    candidate={
                        "edit_text": "add a blue star sticker to the top-right corner of the paper",
                        "difference": {"type": "object_presence", "from": "no blue star sticker", "to": "blue star sticker"},
                    },
                    route_hint="vace_controlled",
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertIn("add a blue star sticker", normalized["target_prompt"])
        self.assertIn("Preserve all other visible content", normalized["target_prompt"])
        self.assertEqual(["target_prompt"], normalized["repaired_fields"])
        self.assertEqual("blue star sticker", normalized["edit_token"])

    def test_plan_video_edit_repairs_missing_edit_region_without_discarding_plan(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "should_generate": True,
                                "source_prompt": "A product showcase of a rotating robot platform.",
                                "target_prompt": "A product showcase of a rotating robot platform with a robot action figure added in the background.",
                                "edit_token": "robot action figure",
                                "preserve_tokens": ["platform", "camera motion", "lighting", "timing"],
                                "negative_prompt": "Do not change the platform, camera, lighting, timing, or scene.",
                                "edit_region": "",
                                "model_route": "vace_controlled",
                                "reason": "The edit is localized behind the platform.",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-video-edit-plan-region-repair-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.plan_video_edit(
                    reference_clip_path=str(clip_path),
                    reference_annotation={"summary": "a robot platform rotates in a dark studio"},
                    candidate={
                        "edit_text": "add a robot action figure to the background",
                        "difference": {
                            "type": "object_presence",
                            "from": "no robot action figure",
                            "to": "robot action figure",
                        },
                    },
                    route_hint="vace_controlled",
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual("background", normalized["edit_region"])
        self.assertEqual(["edit_region"], normalized["repaired_fields"])
        self.assertEqual("robot action figure", normalized["edit_token"])

    def test_plan_video_edit_repairs_missing_preserve_tokens_without_discarding_plan(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "should_generate": True,
                                "source_prompt": "A product showcase of a rotating platform in a dark studio.",
                                "target_prompt": "A product showcase of a rotating platform in a dark studio with a robot action figure in the background.",
                                "edit_token": "robot action figure",
                                "edit_region": "background",
                                "model_route": "vace_controlled",
                                "reason": "The background can be edited locally while preserving the platform.",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-video-edit-plan-preserve-repair-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.plan_video_edit(
                    reference_clip_path=str(clip_path),
                    reference_annotation={
                        "summary": "a rotating platform in a dark studio",
                        "subjects": ["rotating platform"],
                        "object_counts": {"platform": 1},
                        "actions": ["rotating"],
                        "scene": "dark studio",
                    },
                    candidate={
                        "edit_text": "add a robot action figure to the background",
                        "difference": {
                            "type": "object_presence",
                            "from": "no robot action figure",
                            "to": "robot action figure",
                        },
                    },
                    route_hint="vace_controlled",
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertIn("rotating platform", normalized["preserve_tokens"])
        self.assertIn("camera motion", normalized["preserve_tokens"])
        self.assertNotIn("robot action figure", normalized["preserve_tokens"])
        self.assertIn("Do not change rotating platform", normalized["negative_prompt"])
        self.assertEqual(["negative_prompt", "preserve_tokens"], normalized["repaired_fields"])

    def test_detective_annotation_runs_observer_then_final_pass(self) -> None:
        requests = []
        observer_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "visual_observations": ["one person holds a guitar"],
                                "audio_observations": ["guitar music"],
                                "text_observations": [],
                                "timeline": ["person sits", "person plays"],
                                "uncertainties": [],
                                "follow_up_questions": ["is there speech?"],
                            }
                        )
                    }
                }
            ]
        }
        final_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "a person plays guitar on a small stage",
                                "subjects": ["person", "guitar"],
                                "object_counts": {"person": 1, "guitar": 1},
                                "actions": ["playing guitar"],
                                "scene": "small stage",
                                "attributes": ["indoor"],
                                "on_screen_text": [],
                                "speech": [],
                                "audio_events": ["guitar music"],
                                "modalities": ["visual", "audio"],
                                "storyline": ["person sits with guitar", "person plays music"],
                                "visible_text": [],
                                "speakers_and_transcript": [],
                                "detective_notes": ["audio confirms guitar performance"],
                            }
                        )
                    }
                }
            ]
        }
        responses = [_FakeHTTPResponse(observer_payload), _FakeHTTPResponse(final_payload)]

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return responses.pop(0)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-detective-omni-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, raw_payload = client.annotate_clip_detective(clip_path=str(clip_path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(2, len(requests))
        self.assertIn("observer", requests[0]["messages"][0]["content"].lower())
        self.assertIn("detective agent", requests[1]["messages"][0]["content"].lower())
        self.assertEqual("a person plays guitar on a small stage", normalized["summary"])
        self.assertEqual(["person sits with guitar", "person plays music"], normalized["storyline"])
        self.assertEqual("person sits with guitar", normalized["events"][0]["visual"])
        self.assertEqual(["guitar music"], normalized["audio_events"])
        self.assertEqual(["observer", "detective_final"], [item["stage"] for item in raw_payload["detective_trajectory"]])

    def test_detective_annotation_infers_audio_events_from_event_audio_and_notes(self) -> None:
        observer_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "visual_observations": ["a robot stands on a platform"],
                                "audio_observations": ["low electronic hum"],
                                "text_observations": [],
                                "timeline": ["robot rotates slowly"],
                                "uncertainties": [],
                                "follow_up_questions": [],
                            }
                        )
                    }
                }
            ]
        }
        final_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "a robot stands on a platform with a low electronic hum in the background",
                                "subjects": ["robot"],
                                "object_counts": {"robot": 1},
                                "actions": ["standing"],
                                "scene": "dark studio platform",
                                "attributes": ["metallic"],
                                "on_screen_text": [],
                                "speech": [],
                                "audio_events": [],
                                "modalities": ["visual"],
                                "storyline": ["the robot rotates in place"],
                                "events": [
                                    {
                                        "start": 0,
                                        "end": 4,
                                        "visual": "the robot rotates in place",
                                        "audio": "low electronic hum and occasional beeps",
                                        "objects": ["robot"],
                                        "actions": ["rotating"],
                                    }
                                ],
                                "visible_text": [],
                                "speakers_and_transcript": [],
                                "detective_notes": ["background electronic hum remains constant"],
                            }
                        )
                    }
                }
            ]
        }
        responses = [_FakeHTTPResponse(observer_payload), _FakeHTTPResponse(final_payload)]

        def fake_urlopen(_request, timeout=None):
            return responses.pop(0)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-detective-omni-audio-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            clip_path = tmp_dir / "clip.mp4"
            clip_path.write_bytes(b"fake-mp4-bytes")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
            )
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.annotate_clip_detective(clip_path=str(clip_path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertIn("audio", normalized["modalities"])
        self.assertTrue(any("electronic hum" in item for item in normalized["audio_events"]))

    def test_propose_pair_normalizes_subject_difference_alias(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "edit_text": "change the woman into a man",
                                "modalities": ["visual"],
                                "reference_caption": "a woman speaks at a podium",
                                "target_caption": "a man speaks to the camera",
                                "difference": {
                                    "type": "subject",
                                    "from": "woman",
                                    "to": "man",
                                    "description": "the main subject changes",
                                },
                                "proposal_reason": "the main visible person changes",
                            }
                        )
                    }
                }
            ]
        }

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8092/v1",
            api_key="EMPTY",
            model="instruct-model",
        )

        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(response_payload)):
            normalized, _raw_payload = client.propose_pair(
                reference_annotation={"clip_id": "ref", "summary": "a woman speaks at a podium"},
                target_annotation={"clip_id": "target", "summary": "a man speaks to the camera"},
                hard_negative_candidates=[],
            )

        self.assertEqual("object_presence", normalized["difference"]["type"])

    def test_judge_pair_normalizes_scores_and_booleans(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "single_main_difference": True,
                                "same_context_score": 1.2,
                                "edit_match_score": 0.86,
                                "target_uniqueness_score": 0.78,
                                "audio_required": "yes",
                                "hard_negative_quality": "good",
                                "accept": "true",
                                "reject_reason": "",
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            normalized, _raw_payload = client.judge_pair(
                proposal={"edit_text": "change one cat into two cats"},
                reference_annotation={"clip_id": "ref", "summary": "one cat"},
                target_annotation={"clip_id": "target", "summary": "two cats"},
                hard_negative_candidates=[],
            )

        self.assertTrue(normalized["accept"])
        self.assertTrue(normalized["audio_required"])
        self.assertEqual(1.0, normalized["same_context_score"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        self.assertEqual({"type": "json_object"}, request_body["response_format"])

    def test_verify_pair_difference_normalizes_nested_schema(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "caption_delta": {
                                    "caption_equivalent": "false",
                                    "has_concrete_difference": "yes",
                                    "difference_matches_edit": True,
                                    "concrete_differences": ["one cat becomes two cats"],
                                    "reason": "the count changes",
                                },
                                "edit_projection": {
                                    "projected_target_caption": "two orange cats rest on a sofa",
                                    "target_matches_projection": True,
                                    "score": 1.2,
                                    "missing_requirements": [],
                                    "reason": "projection matches the target",
                                },
                                "edit_necessity": {
                                    "edit_needed": True,
                                    "reference_satisfies_edit": False,
                                    "target_satisfies_edit": True,
                                    "score": 0.88,
                                    "reason": "the reference has one cat",
                                },
                                "edit_text_quality_check": {
                                    "not_caption_like": "true",
                                    "matches_modality": "yes",
                                    "single_primary_difference": True,
                                    "reference_does_not_satisfy": True,
                                    "target_satisfies": True,
                                    "score": 1.2,
                                    "failure_reason": "",
                                },
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            normalized, _raw_payload = client.verify_pair_difference(
                proposal={"edit_text": "change one cat into two cats"},
                reference_annotation={"clip_id": "ref", "summary": "one cat on a sofa"},
                target_annotation={"clip_id": "target", "summary": "two cats on a sofa"},
            )

        self.assertFalse(normalized["caption_delta"]["caption_equivalent"])
        self.assertTrue(normalized["caption_delta"]["has_concrete_difference"])
        self.assertEqual(1.0, normalized["edit_projection"]["score"])
        self.assertEqual(0.88, normalized["edit_necessity"]["score"])
        self.assertTrue(normalized["edit_text_quality_check"]["matches_modality"])
        self.assertEqual(1.0, normalized["edit_text_quality_check"]["score"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        self.assertIn("edit-required difference", request_body["messages"][1]["content"][0]["text"])
        self.assertIn("edit_text_quality_check", request_body["messages"][1]["content"][0]["text"])

    def test_audio_line_single_source_pair_repairs_missing_auxiliary_schema_fields(self) -> None:
        requests: list[object] = []
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "edit_text": "change the speech from discussing budgets to discussing healthcare",
                                "difference": {
                                    "type": "speech",
                                    "from": "discussing budgets",
                                    "to": "discussing healthcare",
                                    "description": "the spoken topic changes while the scene remains the same",
                                },
                                "modalities": ["audio"],
                                "confidence": 0.84,
                                "accept": True,
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            requests.append(request)
            return _FakeHTTPResponse(response_payload)

        temp_parent = Path.cwd() / "runs"
        temp_parent.mkdir(exist_ok=True)
        tmp_dir = temp_parent / f"tmp-composed-omni-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            ref_path = tmp_dir / "ref.mp4"
            tgt_path = tmp_dir / "tgt.mp4"
            ref_path.write_bytes(b"fake-reference")
            tgt_path.write_bytes(b"fake-target")
            client = OpenAIComposedDataClient(
                base_url="http://127.0.0.1:8093/v1",
                api_key="EMPTY",
                model="qwen3-omni",
            )

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.propose_single_source_pair(
                    reference_clip_path=str(ref_path),
                    target_clip_path=str(tgt_path),
                    reference_annotation={
                        "clip_id": "ref",
                        "summary": "a presenter speaks at a podium about budgets",
                        "scene": "podium speech",
                        "subjects": ["presenter"],
                        "actions": ["speaking"],
                        "speech": ["budget policy comments"],
                        "audio_events": [],
                        "modalities": ["visual", "audio"],
                    },
                    target_annotation={
                        "clip_id": "tgt",
                        "summary": "the same presenter speaks at the podium about healthcare",
                        "scene": "podium speech",
                        "subjects": ["presenter"],
                        "actions": ["speaking"],
                        "speech": ["healthcare policy comments"],
                        "audio_events": [],
                        "modalities": ["visual", "audio"],
                    },
                    whole_annotation=None,
                    candidate={
                        "audio_dataset_line": "speech_audio_content",
                        "heuristic_difference": {
                            "type": "speech",
                            "from": "budget policy comments",
                            "to": "healthcare policy comments",
                            "description": "spoken topic changes",
                        },
                    },
                    audio_dataset_line="speech_audio_content",
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(1, len(requests))
        self.assertEqual("speech", normalized["difference"]["type"])
        self.assertEqual("speech", normalized["dominant_delta"]["type"])
        self.assertEqual("a presenter speaks at a podium about budgets", normalized["reference_caption"])
        self.assertEqual("the same presenter speaks at the podium about healthcare", normalized["target_caption"])
        self.assertGreaterEqual(normalized["delta_temporal_extent"]["target_coverage"], 0.55)
        self.assertTrue(normalized["is_segment_wide_delta"])
        self.assertIn("dominant_delta", normalized["schema_repaired_fields"])
        self.assertIn("reference_state", normalized["schema_repaired_fields"])
        self.assertIn("target_state", normalized["schema_repaired_fields"])

    def test_verify_pair_difference_materializes_reference_and_target_videos(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "caption_delta": {
                                    "caption_equivalent": False,
                                    "has_concrete_difference": True,
                                    "difference_matches_edit": True,
                                },
                                "edit_projection": {
                                    "projected_target_caption": "a dollhouse appears in the background",
                                    "target_matches_projection": True,
                                    "score": 0.9,
                                },
                                "edit_necessity": {
                                    "edit_needed": True,
                                    "reference_satisfies_edit": False,
                                    "target_satisfies_edit": True,
                                    "score": 0.9,
                                },
                                "edit_text_quality_check": {
                                    "not_caption_like": True,
                                    "matches_modality": True,
                                    "single_primary_difference": True,
                                    "reference_does_not_satisfy": True,
                                    "target_satisfies": True,
                                    "score": 0.9,
                                },
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client.verify_pair_difference(
                    proposal={"edit_text": "add a dollhouse to the background"},
                    reference_annotation={"clip_id": "ref", "summary": "a woman speaks"},
                    target_annotation={"clip_id": "target", "summary": "a woman speaks near a dollhouse"},
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                )

        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        content = request_body["messages"][1]["content"]
        self.assertEqual("Reference video for final verification:", content[0]["text"])
        self.assertTrue(content[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual("Target video for final verification:", content[2]["text"])
        self.assertTrue(content[3]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        verification_prompt = content[4]["text"]
        self.assertIn("actual videos as the primary evidence", verification_prompt)
        self.assertIn("reference video already contains or satisfies", verification_prompt)

    def test_verify_audio_anchor_visual_pair_materializes_videos_and_rejects_weak_deltas(self) -> None:
        request_holder: dict[str, object] = {}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "accept": True,
                                "reject_reason": "",
                                "recommended_edit_text": "change the gray wall close-up into a classroom blackboard view",
                                "visual_delta_type": "scene",
                                "visual_delta_strength": 0.82,
                                "near_duplicate_risk": 0.2,
                                "reference_satisfies_edit": False,
                                "target_satisfies_edit": True,
                                "caption_equivalent": False,
                                "order_only_scene_reorder": False,
                                "weak_synonym_or_wording_delta": False,
                                "evidence": ["The target shows a blackboard classroom while the reference is a gray wall close-up."],
                            }
                        )
                    }
                }
            ]
        }

        def fake_urlopen(request, timeout):
            request_holder["request"] = request
            return _FakeHTTPResponse(response_payload)

        client = OpenAIComposedDataClient(
            base_url="http://127.0.0.1:8093/v1",
            api_key="EMPTY",
            model="qwen3-omni",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.mp4"
            target_path = Path(temp_dir) / "target.mp4"
            reference_path.write_bytes(b"reference-video")
            target_path.write_bytes(b"target-video")
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                normalized, _raw_payload = client.verify_audio_anchor_visual_pair(
                    proposal={
                        "edit_text": "change the gray wall close-up into a classroom blackboard view",
                        "audio_anchor_score": 0.94,
                    },
                    reference_annotation={"clip_id": "ref", "summary": "a gray wall close-up with speech"},
                    target_annotation={"clip_id": "target", "summary": "a classroom blackboard with the same speech"},
                    reference_clip_path=str(reference_path),
                    target_clip_path=str(target_path),
                )

        self.assertTrue(normalized["accept"])
        self.assertEqual("scene", normalized["visual_delta_type"])
        self.assertEqual(0.82, normalized["visual_delta_strength"])
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        system_prompt = request_body["messages"][0]["content"]
        self.assertIn("visual_edit_audio_anchor", system_prompt)
        self.assertIn("bright core", system_prompt)
        content = request_body["messages"][1]["content"]
        self.assertEqual("Reference video for audio-anchor visual verification:", content[0]["text"])
        self.assertTrue(content[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual("Target video for audio-anchor visual verification:", content[2]["text"])
        self.assertTrue(content[3]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertIn("Audio similarity means the clips may share context", content[4]["text"])


if __name__ == "__main__":
    unittest.main()
