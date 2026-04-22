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
        self.assertEqual(22.0, request_holder["timeout"])

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


if __name__ == "__main__":
    unittest.main()
