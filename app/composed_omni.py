from __future__ import annotations

import json
import re
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

REQUIRED_PAIR_VERIFICATION_FIELDS = (
    "caption_delta",
    "edit_projection",
    "edit_necessity",
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NON_SPEECH_AUDIO_TOKENS = {
    "ambient",
    "ambience",
    "applause",
    "bark",
    "barking",
    "beep",
    "bell",
    "bird",
    "birds",
    "buzz",
    "buzzing",
    "chain",
    "chainsaw",
    "cheer",
    "cheering",
    "chirp",
    "chirping",
    "clap",
    "clapping",
    "crash",
    "crowd",
    "drum",
    "electronic",
    "engine",
    "footstep",
    "gunshot",
    "hiss",
    "hum",
    "instrument",
    "laugh",
    "laughter",
    "machine",
    "mechanical",
    "melody",
    "music",
    "noise",
    "orchestra",
    "orchestral",
    "piano",
    "rain",
    "ring",
    "ringing",
    "river",
    "roar",
    "rumble",
    "rustle",
    "rustling",
    "score",
    "siren",
    "song",
    "splash",
    "static",
    "stream",
    "thunder",
    "traffic",
    "water",
    "waves",
    "whir",
    "whirring",
    "whoosh",
    "wind",
}
GENERIC_SPEECH_AUDIO_TOKENS = {
    "dialog",
    "dialogue",
    "narrate",
    "narrates",
    "narrating",
    "narration",
    "narrator",
    "speaker",
    "speak",
    "speaking",
    "speech",
    "talk",
    "talking",
    "voice",
    "voiceover",
}


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


def _tokenize_text(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(str(value).lower()))


def _is_speech_like_audio_value(value: str) -> bool:
    tokens = _tokenize_text(value)
    if not tokens:
        return False
    if tokens & NON_SPEECH_AUDIO_TOKENS:
        return False
    return bool(tokens & GENERIC_SPEECH_AUDIO_TOKENS)


def _is_non_speech_audio_value(value: str) -> bool:
    tokens = _tokenize_text(value)
    return bool(tokens & NON_SPEECH_AUDIO_TOKENS) and not _is_speech_like_audio_value(value)


def _collect_non_speech_audio_terms(payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add(value: Any) -> None:
        for item in _detail_list(value):
            if _is_non_speech_audio_value(item) and item not in terms:
                terms.append(item)

    add(payload.get("audio_events"))
    add(payload.get("audio_observations"))
    add(payload.get("detective_notes"))
    add(payload.get("summary"))
    events = payload.get("events", [])
    if isinstance(events, list):
        for item in events:
            if isinstance(item, dict):
                add(item.get("audio"))
                add(item.get("audio_events"))
    return terms


def _merge_audio_events(existing: list[str], inferred: list[str]) -> list[str]:
    merged = list(existing)
    for item in inferred:
        if item not in merged:
            merged.append(item)
    return merged


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
        "Use speech only for spoken words, transcript paraphrases, or speaker delivery. "
        "Use audio_events only for non-speech audio such as music, applause, wind, footsteps, machinery, animal sounds, hums, or ambient noise. "
        "If a clip has both speech and non-speech audio, put language in speech and non-language sounds in audio_events. "
        "Never use audio_events for generic labels like speech, narration, talking, or voiceover. "
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
        "In audio_observations, call out non-speech audio explicitly: music, ambience, hums, applause, animal sounds, mechanical noise, water, wind, or similar sounds. "
        "Do not collapse non-speech audio into vague phrases like audio or background sound. "
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
        "When audio is present, name non-speech sounds explicitly instead of vague placeholders. "
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
        '"storyline": [string], "events": [{"start": number, "end": number, "visual": string, '
        '"audio": string, "objects": [string], "actions": [string]}], '
        '"visible_text": [string], "speakers_and_transcript": [string], '
        '"detective_notes": [string]}. '
        "Keep the summary concise, but preserve discriminative subject, action, audio, OCR, and timeline details. "
        "Use speech only for spoken-language content. Use audio_events only for non-speech sounds such as music, applause, environmental ambience, hums, machinery, footsteps, animal sounds, water, or wind. "
        "If non-speech audio exists, name it explicitly in audio_events and also mention it in events[].audio or detective_notes. "
        "Never fill audio_events with speech, narration, talking, or voiceover. "
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
        "Use action only when the action itself changes and both clips contain action/storyline/event evidence for that change. "
        "Do not label object appearance, color/attribute, visible text, speech topic, or scene/background changes as action. "
        "Use speech only for concrete spoken-language content, speaker, tone, or dialogue changes with transcript-backed speech evidence. "
        "Use audio_event only for non-speech sounds such as music, applause, machinery, footsteps, wind, animals, or environmental sounds. "
        "Never label a pure speech topic or narration change as audio_event. "
        "Use scene only when the location or background is the primary edit. "
        "Keep edit_text short and only describe the change from reference to target. "
        "edit_text must be an imperative edit, not a caption. "
        "Do not copy a full reference_caption or target_caption into edit_text. "
        "Do not mention visual subjects in audio_event edit_text. "
        "Do not mention speech/topic/content in audio_event edit_text. "
        "Do not mention audio, speech, visible text, or OCR unless that modality is the chosen difference.type. "
        "Use exactly one main difference. "
        "Good object_presence: add a dollhouse to the background. Bad object_presence: change no dollhouse into 1 dollhouse. "
        "Good object_count: reduce the number of pillows from four to three. Bad object_count: the number of pillows decreases from 4 to 3. "
        "Good action: change the gesture from making a small circle to waving. Bad action: the man is speaking and then waving. "
        "Good audio_event: add a whoosh sound. Good audio_event: replace electronic hum with scratching sounds. "
        "Bad audio_event: add a woman with blonde hair to the audio. Bad audio_event: add speech or no background noise to the audio. "
        "Good speech: change the speech from discussing cold email to discussing affiliate marketing. Bad speech: change the man to talk about another topic. "
        "Good visible_text: change on-screen text from cold email to mass emails. Bad visible_text: change the whole speech and on-screen text. "
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
        "For speech edits, require transcript-backed speech evidence on both sides and audio_required=true; generic 'talking' or 'speaking to camera' is not enough. "
        "For audio_event edits, accept only non-speech sound/music/environment changes; reject if the audio difference is only speech or narration content. "
        "Use scores from 0.0 to 1.0."
    )


def _pair_verification_system_prompt() -> str:
    return (
        "You verify whether a composed retrieval pair truly needs the edit text. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"caption_delta": {"caption_equivalent": boolean, '
        '"has_concrete_difference": boolean, "difference_matches_edit": boolean, '
        '"concrete_differences": [string], "reason": string}, '
        '"edit_projection": {"projected_target_caption": string, '
        '"target_matches_projection": boolean, "score": number, '
        '"missing_requirements": [string], "reason": string}, '
        '"edit_necessity": {"edit_needed": boolean, "reference_satisfies_edit": boolean, '
        '"target_satisfies_edit": boolean, "score": number, "reason": string}, '
        '"edit_text_quality_check": {"not_caption_like": boolean, "matches_modality": boolean, '
        '"single_primary_difference": boolean, "reference_does_not_satisfy": boolean, '
        '"target_satisfies": boolean, "score": number, "failure_reason": string}}. '
        "Reject pairs where the reference and target captions are semantically equivalent, "
        "where no concrete visual/audio/text difference is present, or where the edit is not necessary. "
        "For speech pairs, state what the reference speech says and what the target speech says; if either side lacks transcript-backed speech content, mark difference_matches_edit=false. "
        "For audio_event pairs, reject speech-only/narration-only changes as audio_event; audio_event must be non-language sound evidence. "
        "In edit_text_quality_check, reject caption-like edit_text, modality leakage, multiple primary differences, and cases where reference already satisfies the edit. "
        "The projected target caption should describe what the reference would become after applying the edit."
    )


def _build_clip_annotation_user_content(clip_path: str) -> list[dict[str, Any]]:
    prompt = (
        "Task: describe this clip for composed retrieval dataset construction.\n"
        "Focus on the main subject, object counts, actions, scene, attributes, visible text, spoken language, and non-speech audio events.\n"
        "If there is music, ambience, hum, applause, footsteps, animal sound, water, wind, or machine noise, list it explicitly in audio_events.\n"
        "Keep speech and non-speech audio separate.\n"
        "Use details that help distinguish this clip from similar clips in the same series."
    )
    return [
        {"type": "video_url", "video_url": {"url": clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_detective_observation_user_content(clip_path: str) -> list[dict[str, Any]]:
    prompt = (
        "Observation pass: inspect the clip like an independent observer.\n"
        "List visible subjects, object counts, actions, scene, visible text, speech, non-speech audio events, and timeline beats.\n"
        "Name non-speech audio explicitly, for example background music, applause, electronic hum, wind, machinery, footsteps, or animal sounds.\n"
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
        "Return concrete evidence for visual events, non-speech audio events, visible text, speech/transcript, timeline beats, "
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
        "When there is non-speech audio, write it explicitly into audio_events and mention it in the event timeline or detective_notes. "
        "Keep speech/transcript separate from non-speech audio. "
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
        "edit_text must be an imperative edit, not a caption, and must not copy a full reference or target caption. "
        "Prefer action/audio/object differences over broad scene differences if they are visible or audible. "
        "For action proposals, edit_text must be a verb/action change such as starts/stops/changes from doing X to doing Y, "
        "and both reference and target storyline/events/actions must support that action change. "
        "If the evidence is mainly object presence, color, visible text, speech content, or audio, choose that type instead of action. "
        "Separate speech from audio_event: speech is language content or speaker delivery; audio_event is non-speech music, environment, or event sound. "
        "For speech, edit_text must name the specific spoken content change and must be grounded in transcript-backed evidence, not just 'talks about a different topic'. "
        "For audio_event, edit_text must name a non-speech sound change and must not be only narration/speech. "
        "For audio_event, never mention visual subjects such as person, woman, man, room, background, toy, or dollhouse in edit_text. "
        "Use type-specific edit_text style: object_presence add/remove X; object_count change the number of X from A to B; action change the action from X to Y; audio_event add/remove/replace sound X; speech change speech from X to Y; visible_text change on-screen text from X to Y. "
        "Reject your own proposal if edit_text sounds like a caption, contains multiple changes, or leaks another modality. "
        "If the clips come from the same source context and the main localized change is in speech, audio, or visible text, "
        "prefer speech/audio_event/visible_text over attribute or scene. "
        "Use event/timeline evidence to choose a difference that is concrete, localized, and needed for retrieval. "
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
        "For action edits, require concrete action evidence on both sides; reject if action is inferred only from a vague summary "
        "or if the actual difference is object/color/text/speech/audio rather than a verb/action change. "
        "For speech edits, require transcript-backed lexical speech evidence on both sides, audio_required=true, and a target speech change that cannot be judged from visuals alone. "
        "For audio_event edits, reject if audio_events only say speech/narration/talking/voiceover; require music or non-language sound evidence. "
        "If the pair proposal JSON says the clips share the same source context, treat localized speech/audio/visible-text changes "
        "as valid composed edits when the rest of the scene stays aligned. "
        "If you reject the pair, reject_reason must be a non-empty sentence naming the main failed gate or threshold."
    )
    return [{"type": "text", "text": prompt}]


def _build_pair_verification_user_content(
    *,
    proposal: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    reference_clip_path: str | None = None,
    target_clip_path: str | None = None,
) -> list[dict[str, Any]]:
    prompt = (
        "Task: verify whether this pair has a real edit-required difference.\n"
        "If reference and target videos are attached, use the actual videos as the primary evidence, "
        "then use the annotations as supporting evidence. Do not accept a pair only because caption wording differs.\n"
        f"Pair proposal JSON:\n{json.dumps(proposal, ensure_ascii=False)}\n"
        f"Reference annotation JSON:\n{json.dumps(reference_annotation, ensure_ascii=False)}\n"
        f"Target annotation JSON:\n{json.dumps(target_annotation, ensure_ascii=False)}\n"
        "Step 1 caption_delta: decide whether the two captions/annotations are effectively the same. "
        "If they are the same clip content with only wording differences, set caption_equivalent=true.\n"
        "Step 2 edit_projection: apply edit_text to the reference caption and write the expected target caption. "
        "Then judge whether the actual target annotation matches that projection.\n"
        "Step 3 edit_necessity: decide whether the edit is needed. "
        "The reference must not satisfy the edit, and the target must satisfy it. "
        "Use concrete visual, audio, speech, visible-text, object-count, action, and event/timeline evidence. "
        "Reject if the actual reference video already contains or satisfies the requested edit, even when the reference annotation omits it. "
        "For object_presence/background edits, check whether the object is already visible in the reference and whether it is materially visible in the target, "
        "not just present in a boundary frame or described by loose caption wording. "
        "For action edit_text, set difference_matches_edit=false unless the reference and target have different concrete actions "
        "supported by action/storyline/event evidence. "
        "For speech edit_text, explicitly check: what transcript-backed speech does the reference contain, what transcript-backed speech does the target contain, "
        "whether the edit requires listening, and whether visuals alone would fail to distinguish the target. "
        "If speech evidence is generic or missing, set target_matches_projection=false or difference_matches_edit=false. "
        "For audio_event edit_text, set difference_matches_edit=false if the change is only spoken topic/narration rather than a non-speech sound. "
        "The pair proposal may include deterministic edit_text_quality and observable_difference gates; use them as local evidence. "
        "Fill edit_text_quality_check for edit-text surface problems only: not_caption_like=false if edit_text copies a caption; "
        "matches_modality=false if audio_event edit_text mentions visual subjects or speech; single_primary_difference=false if it mixes modalities. "
        "Do not use edit_text_quality_check to duplicate edit_necessity; reference/target satisfaction belongs in edit_necessity."
    )
    content: list[dict[str, Any]] = []
    if reference_clip_path:
        content.extend(
            [
                {"type": "text", "text": "Reference video for final verification:"},
                {"type": "video_url", "video_url": {"url": reference_clip_path}},
            ]
        )
    if target_clip_path:
        content.extend(
            [
                {"type": "text", "text": "Target video for final verification:"},
                {"type": "video_url", "video_url": {"url": target_clip_path}},
            ]
        )
    content.append({"type": "text", "text": prompt})
    return content


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

    def verify_pair_difference(
        self,
        *,
        proposal: dict[str, Any],
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        reference_clip_path: str | None = None,
        target_clip_path: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_pair_verification_user_content(
                proposal=proposal,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                reference_clip_path=reference_clip_path,
                target_clip_path=target_clip_path,
            ),
            system_prompt=_pair_verification_system_prompt(),
            max_tokens=1300,
        )
        return _normalize_pair_verification_payload(raw_payload), raw_payload

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
    normalized["audio_events"] = _merge_audio_events(normalized["audio_events"], _collect_non_speech_audio_terms(payload))
    if normalized["audio_events"] and "audio" not in normalized["modalities"]:
        normalized["modalities"] = list(normalized["modalities"]) + ["audio"]
    if not normalized["summary"]:
        raise ValueError("clip annotation summary is required")
    return normalized


def _normalize_detective_clip_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_clip_annotation_payload(payload)
    visible_text = _detail_list(payload.get("visible_text"))
    transcript = _detail_list(payload.get("speakers_and_transcript"))
    storyline = _detail_list(payload.get("storyline"))
    if visible_text and not normalized["on_screen_text"]:
        normalized["on_screen_text"] = visible_text
    if transcript and not normalized["speech"]:
        normalized["speech"] = transcript
    normalized.update(
        {
            "storyline": storyline,
            "events": _event_list(payload.get("events"), fallback_storyline=storyline),
            "visible_text": visible_text,
            "speakers_and_transcript": transcript,
            "detective_notes": _detail_list(payload.get("detective_notes")),
            "uncertainties": _detail_list(payload.get("uncertainties")),
        }
    )
    normalized["audio_events"] = _merge_audio_events(normalized["audio_events"], _collect_non_speech_audio_terms(normalized))
    if normalized["audio_events"] and "audio" not in normalized["modalities"]:
        normalized["modalities"] = list(normalized["modalities"]) + ["audio"]
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


def _normalize_pair_verification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_PAIR_VERIFICATION_FIELDS)
    if missing_fields:
        raise ValueError(f"pair verification missing fields: {missing_fields}")

    caption_delta = payload.get("caption_delta")
    edit_projection = payload.get("edit_projection")
    edit_necessity = payload.get("edit_necessity")
    edit_text_quality_check = payload.get("edit_text_quality_check", {})
    if not isinstance(caption_delta, dict):
        raise ValueError("pair verification caption_delta must be an object")
    if not isinstance(edit_projection, dict):
        raise ValueError("pair verification edit_projection must be an object")
    if not isinstance(edit_necessity, dict):
        raise ValueError("pair verification edit_necessity must be an object")
    if not isinstance(edit_text_quality_check, dict):
        edit_text_quality_check = {}

    return {
        "caption_delta": {
            "caption_equivalent": _bool_value(caption_delta.get("caption_equivalent")),
            "has_concrete_difference": _bool_value(caption_delta.get("has_concrete_difference")),
            "difference_matches_edit": _bool_value(caption_delta.get("difference_matches_edit")),
            "concrete_differences": _string_list(caption_delta.get("concrete_differences")),
            "reason": str(caption_delta.get("reason", "")).strip(),
        },
        "edit_projection": {
            "projected_target_caption": str(edit_projection.get("projected_target_caption", "")).strip(),
            "target_matches_projection": _bool_value(edit_projection.get("target_matches_projection")),
            "score": _score_value(edit_projection.get("score")),
            "missing_requirements": _string_list(edit_projection.get("missing_requirements")),
            "reason": str(edit_projection.get("reason", "")).strip(),
        },
        "edit_necessity": {
            "edit_needed": _bool_value(edit_necessity.get("edit_needed")),
            "reference_satisfies_edit": _bool_value(edit_necessity.get("reference_satisfies_edit")),
            "target_satisfies_edit": _bool_value(edit_necessity.get("target_satisfies_edit")),
            "score": _score_value(edit_necessity.get("score")),
            "reason": str(edit_necessity.get("reason", "")).strip(),
        },
        "edit_text_quality_check": {
            "not_caption_like": _bool_value(edit_text_quality_check.get("not_caption_like", True)),
            "matches_modality": _bool_value(edit_text_quality_check.get("matches_modality", True)),
            "single_primary_difference": _bool_value(edit_text_quality_check.get("single_primary_difference", True)),
            "reference_does_not_satisfy": _bool_value(edit_text_quality_check.get("reference_does_not_satisfy", True)),
            "target_satisfies": _bool_value(edit_text_quality_check.get("target_satisfies", True)),
            "score": _score_value(edit_text_quality_check.get("score", 1.0)),
            "failure_reason": str(edit_text_quality_check.get("failure_reason", "")).strip(),
        },
    }


def _score_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _event_list(value: Any, *, fallback_storyline: list[str] | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                event = {
                    "start": _score_or_zero(item.get("start")),
                    "end": _score_or_zero(item.get("end")),
                    "visual": str(item.get("visual", "")).strip(),
                    "audio": str(item.get("audio", "")).strip(),
                    "objects": _string_list(item.get("objects")),
                    "actions": _string_list(item.get("actions")),
                }
                if event["visual"] or event["audio"] or event["objects"] or event["actions"]:
                    events.append(event)
            else:
                text = str(item).strip()
                if text:
                    events.append({"start": 0.0, "end": 0.0, "visual": text, "audio": "", "objects": [], "actions": []})
    if not events and fallback_storyline:
        for item in fallback_storyline:
            text = str(item).strip()
            if text:
                events.append({"start": 0.0, "end": 0.0, "visual": text, "audio": "", "objects": [], "actions": []})
    return events


def _score_or_zero(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "pass", "accept", "accepted"}
