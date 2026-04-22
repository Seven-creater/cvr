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
    "visible_text",
}
DIFFERENCE_TYPE_ALIASES = {
    "subject": "object_presence",
    "person": "object_presence",
    "people": "object_presence",
    "object": "object_presence",
    "entity": "object_presence",
    "count": "object_count",
    "number": "object_count",
    "object_number": "object_count",
    "activity": "action",
    "movement": "action",
    "sound": "audio_event",
    "audio": "audio_event",
    "music": "audio_event",
    "voice": "speech",
    "spoken": "speech",
    "ocr": "visible_text",
    "screen_text": "visible_text",
    "text": "visible_text",
    "visible_text_change": "visible_text",
    "background": "scene",
    "location": "scene",
    "color": "attribute",
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

REQUIRED_PAIR_JUDGE_FIELDS = (
    "reference_satisfies_edit",
    "target_satisfies_edit",
    "single_main_difference",
    "same_context_score",
    "edit_match_score",
    "target_uniqueness_score",
    "audio_required",
    "hard_negative_quality",
    "accept",
    "reject_reason",
)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _detail_list(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            parts = [str(part).strip() for part in item.values() if str(part).strip()]
            text = " | ".join(parts)
        else:
            text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


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
    difference_type = str(value.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    difference_type = DIFFERENCE_TYPE_ALIASES.get(difference_type, difference_type)
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


def _detective_observation_system_prompt() -> str:
    return (
        "You are the observer in an Omni-Captioner style detective loop. "
        "Inspect the video and return exactly one JSON object and nothing else. "
        'Required schema: {"visual_observations": [string], "audio_observations": [string], '
        '"text_observations": [string], "timeline": [string], "uncertainties": [string], '
        '"follow_up_questions": [string]}. '
        "Capture concrete details that distinguish this clip from similar clips. "
        "Do not infer unsupported identities or events."
    )


def _detective_toolbox_system_prompt() -> str:
    return (
        "You are an independent observer inside an Omni-Captioner style Tool Box. "
        "Use the supplied tool observations plus the video to answer concrete questions about the clip. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"visual_observations": [string], "audio_observations": [string], '
        '"text_observations": [string], "timeline": [string], "uncertainties": [string], '
        '"follow_up_questions": [string]}. '
        "Prefer timestamped facts when possible. Separate visual, audio, visible text, and speech evidence. "
        "Do not hide uncertainty."
    )


def _detective_final_system_prompt() -> str:
    return (
        "You are the detective agent for composed video retrieval data construction. "
        "Use the video and prior observations to produce a low-hallucination, fine-grained annotation. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"summary": string, "subjects": [string], "object_counts": {string: integer}, '
        '"actions": [string], "scene": string, "attributes": [string], "on_screen_text": [string], '
        '"speech": [string], "audio_events": [string], "modalities": ["visual"|"audio", ...], '
        '"storyline": [string], "visible_text": [string], "speakers_and_transcript": [string], '
        '"detective_notes": [string]}. '
        "Keep the summary concise, but preserve discriminative subject, action, audio, OCR, and timeline details. "
        "Use 'audio' in modalities only when audible information helps distinguish the clip."
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
        "If the main subject/person/object appears or disappears, use object_presence rather than subject. "
        "Prefer fine-grained action, audio_event, object_count, or object_presence changes over broad scene changes when both apply. "
        "Use scene only when the location or background is the primary edit. "
        "Keep edit_text short and only describe the change from reference to target. "
        "Prefer a single key difference instead of multiple simultaneous changes."
    )


def _pair_judge_system_prompt() -> str:
    return (
        "You are a strict judge for composed video retrieval dataset construction. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"reference_satisfies_edit": boolean, "target_satisfies_edit": boolean, '
        '"single_main_difference": boolean, "same_context_score": number, "edit_match_score": number, '
        '"target_uniqueness_score": number, "audio_required": boolean, '
        '"hard_negative_quality": "good"|"weak"|"bad", "accept": boolean, "reject_reason": string}. '
        "Accept only when the reference does not satisfy the edit, the target does satisfy it, "
        "there is one main difference, the context is similar, and negatives are close but wrong. "
        "Use scores from 0.0 to 1.0."
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


def _build_detective_observation_user_content(clip_path: str) -> list[dict[str, Any]]:
    prompt = (
        "Observation pass: inspect the clip like an independent observer.\n"
        "List visible subjects, object counts, actions, scene, visible text, speech, music, sound events, and timeline beats.\n"
        "Also list any uncertainties that a later detective pass should be careful about."
    )
    return [
        {"type": "video_url", "video_url": {"url": clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_detective_toolbox_user_content(
    *,
    clip_path: str,
    tool_observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt = (
        "Tool-box observation pass: inspect the clip using the structured tool observations below.\n"
        f"Tool observations JSON:\n{json.dumps(tool_observations, ensure_ascii=False)}\n"
        "Return concrete evidence for visual events, audio events, visible text, speech/transcript, timeline beats, "
        "and remaining uncertainties."
    )
    return [
        {"type": "video_url", "video_url": {"url": clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_detective_final_user_content(
    *,
    clip_path: str,
    observations: dict[str, Any],
    tool_observations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tool_text = ""
    if tool_observations:
        tool_text = f"Tool observations JSON:\n{json.dumps(tool_observations, ensure_ascii=False)}\n"
    prompt = (
        "Final detective pass: synthesize a structured clip annotation for composed retrieval.\n"
        f"{tool_text}"
        f"Prior observer JSON:\n{json.dumps(observations, ensure_ascii=False)}\n"
        "Prefer facts supported by the video or observer notes. "
        "Make the annotation useful for later finding pairs that differ by one clear visual/audio/text change."
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
    heuristic_pair: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    heuristic_text = ""
    if heuristic_pair:
        heuristic_text = f"Heuristic pair hint JSON:\n{json.dumps(heuristic_pair, ensure_ascii=False)}\n"
    prompt = (
        "Task: draft a composed retrieval pair proposal.\n"
        f"Reference annotation JSON:\n{json.dumps(reference_annotation, ensure_ascii=False)}\n"
        f"Target annotation JSON:\n{json.dumps(target_annotation, ensure_ascii=False)}\n"
        f"Related negative candidate annotations JSON:\n{json.dumps(hard_negative_candidates, ensure_ascii=False)}\n"
        f"{heuristic_text}"
        "Write a short edit_text that changes the reference into the target. "
        "Use one primary difference type only. "
        "The chosen difference.type, edit_text, modalities, and difference.from/to must all describe the same main change. "
        "Prefer action/audio/object differences over broad scene differences if they are visible or audible. "
        "If the clips come from the same source context and the main localized change is in speech, audio, or visible text, "
        "prefer speech/audio_event/visible_text over attribute or scene. "
        "Only include audio in modalities when the edit actually requires listening. "
        "Do not mention secondary audio, speech, or visible-text details in edit_text unless they are the chosen primary difference. "
        "Keep captions factual and concise."
    )
    return [{"type": "text", "text": prompt}]


def _build_pair_judge_user_content(
    *,
    proposal: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    hard_negative_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt = (
        "Task: judge whether this candidate is a high-quality composed retrieval sample.\n"
        f"Pair proposal JSON:\n{json.dumps(proposal, ensure_ascii=False)}\n"
        f"Reference annotation JSON:\n{json.dumps(reference_annotation, ensure_ascii=False)}\n"
        f"Target annotation JSON:\n{json.dumps(target_annotation, ensure_ascii=False)}\n"
        f"Hard negative annotations JSON:\n{json.dumps(hard_negative_candidates, ensure_ascii=False)}\n"
        "The edit_text must describe the change only. The reference should not satisfy the edit; "
        "the target should satisfy it. Reject broad scene-only changes unless the context remains clearly related. "
        "If the pair proposal JSON says the clips share the same source context, treat localized speech/audio/visible-text changes "
        "as valid composed edits when the rest of the scene stays aligned. "
        "If you reject the pair, reject_reason must be a non-empty sentence naming the main failed gate or threshold."
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

    def annotate_clip_detective(
        self,
        *,
        clip_path: str,
        tool_observations: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if tool_observations:
            observations = self._request_json(
                user_content=_build_detective_toolbox_user_content(
                    clip_path=clip_path,
                    tool_observations=tool_observations,
                ),
                system_prompt=_detective_toolbox_system_prompt(),
                max_tokens=1400,
            )
        else:
            observations = self._request_json(
                user_content=_build_detective_observation_user_content(clip_path),
                system_prompt=_detective_observation_system_prompt(),
                max_tokens=1200,
            )
        final_payload = self._request_json(
            user_content=_build_detective_final_user_content(
                clip_path=clip_path,
                observations=observations,
                tool_observations=tool_observations,
            ),
            system_prompt=_detective_final_system_prompt(),
            max_tokens=1800,
        )
        trajectory = list(tool_observations or []) + [
            {"stage": "observer", "payload": observations},
            {"stage": "detective_final", "payload": final_payload},
        ]
        normalized = _normalize_detective_clip_annotation_payload(final_payload)
        normalized["detective_trajectory"] = trajectory
        return normalized, {
            "observer": observations,
            "detective_final": final_payload,
            "detective_trajectory": trajectory,
        }

    def propose_pair(
        self,
        *,
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        hard_negative_candidates: list[dict[str, Any]],
        heuristic_pair: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_pair_proposal_user_content(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                hard_negative_candidates=hard_negative_candidates,
                heuristic_pair=heuristic_pair,
            ),
            system_prompt=_pair_proposal_system_prompt(),
            max_tokens=1200,
        )
        return _normalize_pair_proposal_payload(raw_payload), raw_payload

    def judge_pair(
        self,
        *,
        proposal: dict[str, Any],
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        hard_negative_candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_pair_judge_user_content(
                proposal=proposal,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                hard_negative_candidates=hard_negative_candidates,
            ),
            system_prompt=_pair_judge_system_prompt(),
            max_tokens=900,
        )
        return _normalize_pair_judge_payload(raw_payload), raw_payload

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


def _normalize_detective_clip_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_clip_annotation_payload(payload)
    visible_text = _detail_list(payload.get("visible_text"))
    transcript = _detail_list(payload.get("speakers_and_transcript"))
    if visible_text and not normalized["on_screen_text"]:
        normalized["on_screen_text"] = visible_text
    if transcript and not normalized["speech"]:
        normalized["speech"] = transcript
    normalized.update(
        {
            "storyline": _detail_list(payload.get("storyline")),
            "visible_text": visible_text,
            "speakers_and_transcript": transcript,
            "detective_notes": _detail_list(payload.get("detective_notes")),
            "uncertainties": _detail_list(payload.get("uncertainties")),
        }
    )
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


def _normalize_pair_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_PAIR_JUDGE_FIELDS)
    if missing_fields:
        raise ValueError(f"pair judge missing fields: {missing_fields}")

    hard_negative_quality = str(payload.get("hard_negative_quality", "")).strip().lower()
    if hard_negative_quality not in {"good", "weak", "bad"}:
        hard_negative_quality = "weak"

    return {
        "reference_satisfies_edit": _bool_value(payload.get("reference_satisfies_edit")),
        "target_satisfies_edit": _bool_value(payload.get("target_satisfies_edit")),
        "single_main_difference": _bool_value(payload.get("single_main_difference")),
        "same_context_score": _score_value(payload.get("same_context_score")),
        "edit_match_score": _score_value(payload.get("edit_match_score")),
        "target_uniqueness_score": _score_value(payload.get("target_uniqueness_score")),
        "audio_required": _bool_value(payload.get("audio_required")),
        "hard_negative_quality": hard_negative_quality,
        "accept": _bool_value(payload.get("accept")),
        "reject_reason": str(payload.get("reject_reason", "")).strip(),
    }


def _score_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "pass", "accept", "accepted"}
