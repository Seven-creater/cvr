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
        request_body = json.loads(request_holder["request"].data.decode("utf-8"))
        content = request_body["messages"][1]["content"]
        self.assertEqual("Reference clip:", content[0]["text"])
        self.assertTrue(content[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual("Target clip:", content[2]["text"])
        self.assertTrue(content[3]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertIn("picture-in-picture", request_body["messages"][0]["content"])

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


if __name__ == "__main__":
    unittest.main()
