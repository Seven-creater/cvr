from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.omni_checker import _extract_json, _materialize_video_url


ALLOWED_MODALITIES = {"visual", "audio"}
ALLOWED_DIFFERENCE_TYPES = {
    "object_count",
    "object_presence",
    "attribute",
    "action",
    "scene",
    "audio_event",
    "speech",
}

REQUIRED_CLIP_ANNOTATION_FIELDS = (
    "summary",
    "subjects",
    "object_counts",
    "actions",
    "scene",
    "attributes",
    "on_screen_text",
    "speech",
    "audio_events",
    "modalities",
)

REQUIRED_PAIR_PROPOSAL_FIELDS = (
    "edit_text",
    "modalities",
    "reference_caption",
    "target_caption",
    "difference",
    "proposal_reason",
)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_modalities(value: Any) -> list[str]:
    normalized = []
    for item in _string_list(value):
        lowered = item.lower()
        if lowered in ALLOWED_MODALITIES and lowered not in normalized:
            normalized.append(lowered)
    return normalized


def _normalize_object_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        normalized[key] = count
    return normalized


def _validate_difference(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("difference must be an object")
    difference_type = str(value.get("type", "")).strip()
    if difference_type not in ALLOWED_DIFFERENCE_TYPES:
        raise ValueError(f"unsupported difference.type={difference_type!r}")
    result = {
        "type": difference_type,
    }
    for field_name in ("from", "to", "description"):
        field_value = str(value.get(field_name, "")).strip()
        if field_value:
            result[field_name] = field_value
    if not any(field_name in result for field_name in ("from", "to", "description")):
        raise ValueError("difference must include from/to/description")
    return result


def _missing_fields(payload: dict[str, Any], required_fields: tuple[str, ...]) -> list[str]:
    return [field_name for field_name in required_fields if field_name not in payload]


def _clip_annotation_system_prompt() -> str:
    return (
        "You annotate short video clips for composed video retrieval dataset construction. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"summary": string, "subjects": [string], "object_counts": {string: integer}, '
        '"actions": [string], "scene": string, "attributes": [string], "on_screen_text": [string], '
        '"speech": [string], "audio_events": [string], "modalities": ["visual"|"audio", ...]}. '
        "All keys are mandatory. "
        "Use concrete and distinguishing details. Keep phrases short. "
        "Include 'audio' in modalities only when the clip contains useful audible content such as speech, music, or sound events."
    )


def _pair_proposal_system_prompt() -> str:
    difference_types = ", ".join(sorted(ALLOWED_DIFFERENCE_TYPES))
    return (
        "You draft candidate pairs for a composed video retrieval dataset. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"edit_text": string, "modalities": ["visual"|"audio", ...], '
        '"reference_caption": string, "target_caption": string, '
        '"difference": {"type": string, "from": string, "to": string, "description": string}, '
        '"proposal_reason": string}. '
        f"Allowed difference.type values: {difference_types}. "
        "Keep edit_text short and only describe the change from reference to target. "
        "Prefer a single key difference instead of multiple simultaneous changes."
    )


def _build_clip_annotation_user_content(clip_path: str) -> list[dict[str, Any]]:
    prompt = (
        "Task: describe this clip for composed retrieval dataset construction.\n"
        "Focus on the main subject, object counts, actions, scene, attributes, visible text, speech, and audio events.\n"
        "Use details that help distinguish this clip from similar clips in the same series."
    )
    return [
        {"type": "video_url", "video_url": {"url": clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_pair_proposal_user_content(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    hard_negative_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt = (
        "Task: draft a composed retrieval pair proposal.\n"
        f"Reference annotation JSON:\n{json.dumps(reference_annotation, ensure_ascii=False)}\n"
        f"Target annotation JSON:\n{json.dumps(target_annotation, ensure_ascii=False)}\n"
        f"Related negative candidate annotations JSON:\n{json.dumps(hard_negative_candidates, ensure_ascii=False)}\n"
        "Write a short edit_text that changes the reference into the target. "
        "Use one primary difference type only. "
        "If audio is important, include it in modalities. "
        "Keep captions factual and concise."
    )
    return [{"type": "text", "text": prompt}]


class OpenAIComposedDataClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def annotate_clip(self, *, clip_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_clip_annotation_user_content(clip_path),
            system_prompt=_clip_annotation_system_prompt(),
            max_tokens=1024,
        )
        return _normalize_clip_annotation_payload(raw_payload), raw_payload

    def propose_pair(
        self,
        *,
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        hard_negative_candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_pair_proposal_user_content(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                hard_negative_candidates=hard_negative_candidates,
            ),
            system_prompt=_pair_proposal_system_prompt(),
            max_tokens=1200,
        )
        return _normalize_pair_proposal_payload(raw_payload), raw_payload

    def _request_json(
        self,
        *,
        user_content: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        request_content: list[dict[str, Any]] = []
        for item in user_content:
            if item.get("type") == "video_url":
                video_url = dict(item["video_url"])
                video_url["url"] = _materialize_video_url(str(video_url["url"]))
                request_content.append({"type": "video_url", "video_url": video_url})
                continue
            request_content.append(item)

        payload = {
            "model": self.model,
            "modalities": ["text"],
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_content},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"composed data request failed: {detail}") from exc
        content = raw_response["choices"][0]["message"]["content"]
        payload = _extract_json(content)
        if not isinstance(payload, dict):
            raise ValueError("model response must decode to a JSON object")
        return payload


def _normalize_clip_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_CLIP_ANNOTATION_FIELDS)
    if missing_fields:
        raise ValueError(f"clip annotation missing fields: {missing_fields}")

    modalities = _normalize_modalities(payload.get("modalities"))
    if not modalities:
        audio_signals = _string_list(payload.get("speech")) or _string_list(payload.get("audio_events"))
        modalities = ["visual", "audio"] if audio_signals else ["visual"]

    normalized = {
        "summary": str(payload.get("summary", "")).strip(),
        "subjects": _string_list(payload.get("subjects")),
        "object_counts": _normalize_object_counts(payload.get("object_counts")),
        "actions": _string_list(payload.get("actions")),
        "scene": str(payload.get("scene", "")).strip(),
        "attributes": _string_list(payload.get("attributes")),
        "on_screen_text": _string_list(payload.get("on_screen_text")),
        "speech": _string_list(payload.get("speech")),
        "audio_events": _string_list(payload.get("audio_events")),
        "modalities": modalities,
    }
    if not normalized["summary"]:
        raise ValueError("clip annotation summary is required")
    return normalized


def _normalize_pair_proposal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_PAIR_PROPOSAL_FIELDS)
    if missing_fields:
        raise ValueError(f"pair proposal missing fields: {missing_fields}")

    modalities = _normalize_modalities(payload.get("modalities"))
    if not modalities:
        modalities = ["visual"]

    normalized = {
        "edit_text": str(payload.get("edit_text", "")).strip(),
        "modalities": modalities,
        "reference_caption": str(payload.get("reference_caption", "")).strip(),
        "target_caption": str(payload.get("target_caption", "")).strip(),
        "difference": _validate_difference(payload.get("difference")),
        "proposal_reason": str(payload.get("proposal_reason", "")).strip(),
    }
    for field_name in ("edit_text", "reference_caption", "target_caption", "proposal_reason"):
        if not normalized[field_name]:
            raise ValueError(f"pair proposal {field_name} is required")
    return normalized
