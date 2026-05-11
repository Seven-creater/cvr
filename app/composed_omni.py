from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
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
REQUIRED_AUDIO_ANCHOR_VISUAL_VERIFICATION_FIELDS = (
    "accept",
    "reject_reason",
    "recommended_edit_text",
    "visual_delta_type",
    "visual_delta_strength",
    "near_duplicate_risk",
    "reference_satisfies_edit",
    "target_satisfies_edit",
    "caption_equivalent",
    "order_only_scene_reorder",
    "weak_synonym_or_wording_delta",
    "evidence",
)

REQUIRED_SINGLE_SOURCE_PAIR_FIELDS = (
    "edit_text",
    "modalities",
    "reference_caption",
    "target_caption",
    "difference",
    "dominant_delta",
    "reference_state",
    "target_state",
    "delta_temporal_extent",
    "subject_roles",
    "is_segment_wide_delta",
    "discarded_deltas",
    "evidence",
    "confidence",
    "accept",
    "reject_reason",
)

REQUIRED_SINGLE_SOURCE_FINAL_VERIFICATION_FIELDS = (
    "accept",
    "confidence",
    "quality_score",
    "reference_satisfies_edit",
    "target_satisfies_edit",
    "observable_delta",
    "single_primary_delta",
    "text_or_ocr_driven",
    "segment_wide",
    "edit_text_accurate",
    "main_reject_reason",
    "evidence",
    "recommended_edit_text",
)

REQUIRED_B_LINE_EDIT_TEXT_REFINEMENT_FIELDS = (
    "refined_edit_text",
    "edit_text_specificity_score",
    "reject_if_unspecific",
    "edit_text_reject_reason",
    "speech_or_audio_evidence",
)

REQUIRED_B_LINE_SPEECH_REWRITE_FIELDS = (
    "reference_speech_content",
    "target_speech_content",
    "speech_transcription_confidence",
    "speech_language",
    "refined_edit_text",
    "reject_if_still_unclear",
    "speech_rewrite_reject_reason",
)

REQUIRED_VIDEO_EDIT_PLAN_FIELDS = (
    "should_generate",
    "source_prompt",
    "target_prompt",
    "edit_token",
    "preserve_tokens",
    "negative_prompt",
    "edit_region",
    "model_route",
    "reason",
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


def _materialize_image_url(raw_url: str) -> str:
    if raw_url.startswith(("http://", "https://", "data:")):
        return raw_url
    image_path = Path(raw_url)
    if raw_url.startswith("file://"):
        image_path = Path(urllib.request.url2pathname(raw_url.removeprefix("file://")))
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    content = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{content}"


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


def _detective_observation_system_prompt(*, audio_focused: bool = False) -> str:
    prompt = (
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
    if audio_focused:
        prompt += (
            " This is an audio-dataset pass: listen carefully and separate speech/transcript, crowd reaction, applause, music, ambience, and other sound events. "
            "If a person is speaking, summarize what they are saying with specific topic words or short paraphrases; do not write only 'speech' or 'talking'. "
            "If an audible claim is uncertain, mark it in uncertainties instead of inventing vague hum/click labels."
        )
    return prompt


def _detective_toolbox_system_prompt(*, audio_focused: bool = False) -> str:
    prompt = (
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
    if audio_focused:
        prompt += (
            " For this audio-focused pass, verify whether the sound is speech, crowd/applause/music/ambient/environment, or uncertain. "
            "For speech, capture speaker role and content/topic changes that could distinguish adjacent 6-second clips. "
            "Avoid weak labels like electronic hum, click, or tone unless the video audio clearly supports them."
        )
    return prompt


def _detective_final_system_prompt(*, audio_focused: bool = False) -> str:
    prompt = (
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
    if audio_focused:
        prompt += (
            " This is an audio-focused refresh for an audio retrieval dataset. "
            "Be conservative: include speech/transcript only when you can hear language content, and include audio_events only for clear sounds such as crowd cheering, applause, music, rain, wind, machinery, or other verifiable events. "
            "For speech, write a short transcript/paraphrase or topic-specific summary in speech and speakers_and_transcript, so adjacent clips from the same speaker can be paired by changed spoken content. "
            "Do not use vague hum/click/tone guesses as distinguishing evidence unless they are unmistakable."
        )
    return prompt


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
        "First compare 1-3 possible differences internally, then choose the single most concrete and verifiable difference; mention the discarded alternatives briefly in proposal_reason, not as extra JSON fields. "
        "If the main subject/person/object appears or disappears, use object_presence rather than subject. "
        "For this dataset pass, prefer clean visual subject swaps, object swaps, scene swaps, and non-speech audio changes over speech/topic or OCR/text edits. "
        "Treat speech and visible_text as diagnostic-only unless no stronger clean visual or non-speech audio difference exists. "
        "Use action only when the action itself changes and both clips contain action/storyline/event evidence for that change. "
        "Do not label object appearance, color/attribute, visible text, speech topic, or scene/background changes as action. "
        "Use speech only for concrete spoken-language content, speaker, tone, or dialogue changes with transcript-backed speech evidence. "
        "Use audio_event only for non-speech sounds such as music, applause, machinery, footsteps, wind, animals, or environmental sounds. "
        "Never label a pure speech topic or narration change as audio_event. "
        "Use scene only when the location or background is the primary edit and the two clips still share strong context; reject loose stock-like scene shifts. "
        "If the heuristic hint or evidence says these are cross-video template-compatible talking-head clips, and the dominant change is the person on screen, use difference.type=attribute and write edit_text as 'change the speaker from <visual signature> to <visual signature>'. "
        "For cross-video object swaps, prefer 'replace the held object from <A> to <B>' or 'change the featured object from <A> to <B>'. "
        "For scene swaps with stable clip templates, prefer 'change the setting from <A> to <B>'. "
        "Keep edit_text short and only describe the change from reference to target. "
        "edit_text must be an imperative edit, not a caption. "
        "Reject vague edit_text such as 'make the school', 'make it nice', or 'make it a touristy country'. "
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
        "For visible_text, include both from/to text in edit_text; never output only the target text. "
        "Prefer a single key difference instead of multiple simultaneous changes."
    )


def _normalize_audio_dataset_line(value: str | None) -> str:
    line = str(value or "standard").strip().lower().replace("-", "_")
    if line in {"", "none"}:
        return "standard"
    if line not in {"standard", "visual_audio_anchor", "speech_audio_content"}:
        return "standard"
    return line


def _single_source_pair_system_prompt(audio_dataset_line: str | None = None) -> str:
    difference_types = ", ".join(sorted(ALLOWED_DIFFERENCE_TYPES))
    line = _normalize_audio_dataset_line(audio_dataset_line)
    if line == "visual_audio_anchor":
        return (
            "You compare two 6s clips from the same source video for A-line visual_audio_anchor. "
            "Use attached videos as primary evidence; annotations and candidate JSON are hints only. Return exactly one JSON object. "
            'Schema: {"edit_text": string, "modalities": [string], "reference_caption": string, "target_caption": string, '
            '"difference": {"type": string, "from": string, "to": string, "description": string}, '
            '"dominant_delta": {"type": string, "from": string, "to": string, "reason": string}, '
            '"reference_state": object, "target_state": object, "delta_temporal_extent": object, "subject_roles": object, '
            '"is_segment_wide_delta": boolean, "discarded_deltas": [string], "evidence": [string], '
            '"confidence": number, "accept": boolean, "reject_reason": string}. '
            "Audio is preserved news/program audio context only; edit_text must be visual-only and must not mention audio, sound, speech, music, transcript, narration, or voice. "
            "Good A style: same news/program audio context while the picture changes from a studio anchor shot to flood aerial footage. "
            "Accept one large stable visual delta: scene, action, object presence/count, or clear subject/composition change. "
            "Reject near-duplicate visuals, tiny hand/object motion, brightness/framing changes, visible-text-only changes, speech/audio-only changes, or multiple competing visual changes. "
            "If rejecting, still return the best tentative visual edit and clear reject_reason."
        )
    if line == "speech_audio_content":
        return (
            "You compare two 6s clips from the same source video for B-line speech_audio_content. "
            "Use attached videos as primary evidence; annotations and candidate JSON are hints only. Return exactly one JSON object. "
            'Schema: {"edit_text": string, "modalities": [string], "reference_caption": string, "target_caption": string, '
            '"difference": {"type": string, "from": string, "to": string, "description": string}, '
            '"dominant_delta": {"type": string, "from": string, "to": string, "reason": string}, '
            '"reference_state": object, "target_state": object, "delta_temporal_extent": object, "subject_roles": object, '
            '"is_segment_wide_delta": boolean, "discarded_deltas": [string], "evidence": [string], '
            '"confidence": number, "accept": boolean, "reject_reason": string}. '
            "Keep visual context practically locked: same speaker/scene/program/match/context is enough; minor framing/pose/action changes are okay. "
            "Use type=speech for concrete spoken topic/phrase/lyric changes, or audio_event for concrete non-speech sounds. "
            "edit_text must be audio-only, e.g. 'change the speech from discussing A to discussing B' or 'add crowd cheering to the audio'. "
            "Reject unintelligible/not-transcribed/unspecified/generic target-audio text, visual edit_text, or visually dominated pairs."
        )
    base_prompt = (
        "You compare two short clips cut from the same original video for composed video retrieval. "
        "Follow an Omni-Captioner/Omni-Detective style: inspect both videos, use the segment annotations as evidence, "
        "then choose the single clearest change from reference to target. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"edit_text": string, "modalities": ["visual"|"audio", ...], '
        '"reference_caption": string, "target_caption": string, '
        '"difference": {"type": string, "from": string, "to": string, "description": string}, '
        '"dominant_delta": {"type": string, "from": string, "to": string, "reason": string}, '
        '"reference_state": {"main_speaker": string, "inset_subjects": [string], "product_overlay": string, '
        '"composition": string, "internal_transitions": [string]}, '
        '"target_state": {"main_speaker": string, "inset_subjects": [string], "product_overlay": string, '
        '"composition": string, "internal_transitions": [string]}, '
        '"delta_temporal_extent": {"reference": string, "target": string, "target_coverage": number, "evidence": string}, '
        '"subject_roles": {"main_speaker": string, "inset_subjects": [string], "product_overlay": string}, '
        '"is_segment_wide_delta": boolean, '
        '"discarded_deltas": [string], "evidence": [string], "confidence": number, '
        '"accept": boolean, "reject_reason": string}. '
        f"Allowed difference.type values: {difference_types}. "
        "Prefer concrete visual changes that are obvious to a human reviewer: object presence/count, a product shown or held, "
        "picture-in-picture overlay appearing or disappearing, action/gesture changes, scene/composition/background changes, then attributes. "
        "For beauty/tutorial/talking-head videos, do not choose tiny wording differences like blouse vs shirt or long brown hair vs long hair. "
        "If a product, applicator, hand-held object, product close-up, or picture-in-picture demo is visible in one clip and not the other, "
        "choose object_presence or scene/composition instead of clothing or hair attributes. "
        "Speech and visible text are auxiliary evidence only for this pass; do not use them as the primary edit. "
        "Distinguish the main speaker from inset-video subjects and product-overlay imagery; never describe an inset man/woman "
        "as replacing the main speaker. "
        "Only accept if the dominant delta is stable for most of the target clip; if it appears only briefly or only at the end, "
        "set is_segment_wide_delta=false and explain the transient timing. "
        "Write edit_text as a short imperative edit that changes the reference into the target, for example: "
        "'add a picture-in-picture demonstration overlay', "
        "'change the shot from face-only speaking to holding a mascara wand', or "
        "'add a static product image overlay on the left'. "
        "Do not write 'product close-up' unless the speaker mostly disappears and the product dominates the frame. "
        "Do not write 'full-screen product presentation' unless there is no speaker or overlay layout. "
        "Do not write 'man speaking' as the primary subject unless it is explicitly an inset video and it persists for most of the clip. "
        "Reject the pair if the only difference is a near-duplicate attribute wording change, unclear clothing/hair wording, "
        "a transient final-moment overlay, an internally changing segment, or multiple equally strong unrelated changes. "
        "When rejecting, still fill the best tentative edit_text and evidence, set accept=false, and explain reject_reason."
    )
    if line == "visual_audio_anchor":
        return (
            base_prompt
            + " This pass is the visual_audio_anchor A line. The two clips should keep similar or continuous audio context; "
            "audio is only an anchor and must not be the edit. Accept only one clear visual delta. "
            "The edit_text must be visual-only and must not mention audio, sound, speech, music, transcript, narration, or voice. "
            "Good A-line style: the same news/program audio context continues while the picture changes from a studio anchor shot to flood aerial footage. "
            "Reject weak A-line cases: same scene with tiny hand/object motion, brightness shifts, camera distance changes, visible text changes, or wording-only attributes. "
            "Reject if the target only changes speech/audio/text, if the visual change is near-duplicate, or if the reference already satisfies the visual edit."
        )
    if line == "speech_audio_content":
        return (
            base_prompt
            + " B line: audio-sensitive retrieval. Keep same speaker/scene/program/match/context; minor framing/pose/action differences are okay. "
            "Use speech for concrete spoken topic/phrase changes; audio_event only for clear non-speech sounds. "
            "edit_text must be audio-only and specific. Reject vague/untranscribed/target-audio wording, visual edit_text, or visually dominated pairs."
        )
    return base_prompt


def _single_source_final_verification_system_prompt(audio_dataset_line: str | None = None) -> str:
    line = _normalize_audio_dataset_line(audio_dataset_line)
    if line == "visual_audio_anchor":
        return (
            "You are the final verifier for A-line visual_audio_anchor. Use attached videos as primary evidence; local_gate_report is diagnostic only. "
            "Return exactly one JSON object with schema: "
            '{"accept": boolean, "confidence": number, "quality_score": number, '
            '"reference_satisfies_edit": boolean, "target_satisfies_edit": boolean, "observable_delta": boolean, '
            '"single_primary_delta": boolean, "text_or_ocr_driven": boolean, "segment_wide": boolean, '
            '"edit_text_accurate": boolean, "main_reject_reason": string, "evidence": [string], '
            '"recommended_edit_text": string, "large_visual_delta": boolean, "audio_context_preserved": boolean}. '
            "Accept only if edit_text is visual-only, reference does not satisfy it, target clearly satisfies it, large_visual_delta=true, "
            "audio_context_preserved=true, and the target difference is stable for most of the clip. "
            "Reject near-duplicates, tiny lighting/framing/gesture changes, OCR/text edits, speech/audio edits, or inaccurate edit_text. "
            "If accept=false, set quality_score below 0.7 and explain main_reject_reason."
        )
    if line == "speech_audio_content":
        return (
            "You are the final verifier for B-line speech_audio_content. Use attached videos as primary evidence; local_gate_report is diagnostic only. "
            "Return exactly one JSON object with schema: "
            '{"accept": boolean, "confidence": number, "quality_score": number, '
            '"reference_satisfies_edit": boolean, "target_satisfies_edit": boolean, "observable_delta": boolean, '
            '"single_primary_delta": boolean, "text_or_ocr_driven": boolean, "segment_wide": boolean, '
            '"edit_text_accurate": boolean, "main_reject_reason": string, "evidence": [string], '
            '"recommended_edit_text": string, "audio_primary": boolean, "visual_locked": boolean, '
            '"visual_too_different_for_B": boolean, "edit_text_audio_only": boolean}. '
            "Accept when audio is primary, edit_text is audio-only, reference lacks it, target has it, and same speaker/scene/program/match/context remains clear. "
            "segment_wide=false is allowed if target audio evidence is clear. Minor framing/pose/action changes are okay. "
            "Reject vague audio, visual edit_text, or visually dominated pairs. "
            "If accept=false, set quality_score below 0.7 and explain main_reject_reason."
        )
    base_prompt = (
        "You are the final strict verifier for a single-source composed video retrieval pair. "
        "The candidate has already passed an initial pair-comparison step and local gates; your job is to decide whether it should enter human review as an accepted sample. "
        "Use the attached reference and target videos as primary evidence. Use captions, dominant_delta, and local_gate_report only as supporting evidence. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"accept": boolean, "confidence": number, "quality_score": number, '
        '"reference_satisfies_edit": boolean, "target_satisfies_edit": boolean, '
        '"observable_delta": boolean, "single_primary_delta": boolean, '
        '"text_or_ocr_driven": boolean, "segment_wide": boolean, '
        '"edit_text_accurate": boolean, "main_reject_reason": string, '
        '"evidence": [string], "recommended_edit_text": string}. '
        "Set quality_score from 0.0 to 1.0 for dataset usefulness: 1.0 is a clean, obvious, single-delta pair; "
        "0.7 is borderline but acceptable for human review; below 0.7 should normally be rejected. "
        "If accept=false, quality_score must be below 0.7. "
        "Accept only if the reference does not satisfy the edit, the target clearly satisfies it, "
        "there is an obvious real difference, the edit_text names the main difference accurately, "
        "and the difference is stable for most of the target clip. "
        "Reject if the candidate is driven by subtitles, visible text, title cards, lower-thirds, product-label/OCR wording, or boundary-frame text. "
        "Reject if reference and target are effectively the same, if the target does not visibly/audibly satisfy the edit, "
        "if the edit_text exaggerates the composition, or if there is a stronger unmentioned difference. "
        "Specifically reject inaccurate phrases such as 'product close-up', 'full-screen product presentation', or 'speaker replacement' "
        "unless the actual target video proves that phrase. "
        "Distinguish the main speaker from picture-in-picture/inset subjects; do not let an inset person count as the main speaker. "
        "Inspect the beginning, middle, and end of both clips before deciding. "
        "If any of text_or_ocr_driven=true, observable_delta=false, target_satisfies_edit=false, "
        "or edit_text_accurate=false, set accept=false and explain main_reject_reason."
    )
    if line == "visual_audio_anchor":
        return (
            base_prompt
            + " For this visual_audio_anchor A line, accept only if the edit is visual and the audio/speech is not the primary change. "
            'Also include "large_visual_delta": boolean and "audio_context_preserved": boolean in the JSON object. '
            "The edit_text must not mention audio, sound, speech, music, transcript, narration, or voice. "
            "The useful target is a large visual shot/scene/subject/action change under similar audio context, not a near-duplicate. "
            "Set accept=true only if large_visual_delta=true and audio_context_preserved=true. "
            "Reject near-duplicate visual changes, order-only changes, tiny lighting/framing shifts, and any pair where reference already satisfies the visual edit."
        )
    if line == "speech_audio_content":
        return (
            base_prompt
            + " For speech_audio_content B, observable_delta may be speech or non-speech audio. "
            'Also include "audio_primary": boolean, "visual_locked": boolean, "visual_too_different_for_B": boolean, '
            'and "edit_text_audio_only": boolean in the JSON object. '
            "Accept speech only with concrete spoken evidence; audio_event only for concrete non-speech sound/music/environment changes. "
            "visual_locked=true if same speaker/scene/program/context is clear despite minor framing/pose/action changes. "
            "Do not reject only because audio is not present for the full 6 seconds; report segment_wide=false but it can pass. "
            "Set accept=true only if audio_primary=true, visual_locked=true, visual_too_different_for_B=false, and edit_text_audio_only=true. "
            "Reject generic audio, visual edit_text, or cases where visuals alone identify the target."
        )
    return base_prompt


def _b_line_edit_text_refinement_system_prompt() -> str:
    return (
        "You refine B-line speech_audio_content edit_text for an audio-sensitive CVR dataset. "
        "Use the attached videos as primary evidence and return exactly one JSON object. "
        'Schema: {"refined_edit_text": string, "edit_text_specificity_score": number, '
        '"reject_if_unspecific": boolean, "edit_text_reject_reason": string, "speech_or_audio_evidence": [string]}. '
        "The edit_text must be audio-only and specific enough for retrieval. "
        "Acceptable speech forms: 'change the speech from discussing {specific topic A} to discussing {specific topic B}', "
        "'change the voice from saying \"{short phrase A}\" to saying \"{short phrase B}\"', or "
        "'change the singing from {specific vocal content/style A} to {specific vocal content/style B}'. "
        "Acceptable audio-event forms: 'replace {specific sound A} with {specific sound B}', "
        "'add {specific sound/event} to the audio', or 'remove {specific sound/event} from the audio'. "
        "Reject vague wording such as unintelligible speech, not transcribed, unclear content, unknown sound, target audio, reference audio, or generic audio differs. "
        "Reject if the text describes people, objects, scenes, camera, frame, subtitles, clothing, fishing, boats, rivers, or other visual content. "
        "If you cannot name the concrete speech topic/phrase or concrete sound event, set reject_if_unspecific=true."
    )


def _b_line_speech_rewrite_system_prompt() -> str:
    return (
        "You are the speech-listening repair step for B-line speech_audio_content CVR data. "
        "Ignore visual differences except for sanity; listen to the reference and target clips and identify concrete spoken or sung content. "
        "Return exactly one JSON object with schema: "
        '{"reference_speech_content": string, "target_speech_content": string, '
        '"speech_transcription_confidence": number, "speech_language": string, '
        '"refined_edit_text": string, "reject_if_still_unclear": boolean, "speech_rewrite_reject_reason": string}. '
        "Paraphrase is allowed when exact transcription is hard, but it must name a real topic, phrase, lyric, or semantic content on each side. "
        "Do not output placeholders such as A/B, topic A/topic B, unknown, unintelligible, not transcribed, speech content changed, target audio, or reference audio. "
        "Use forms like 'change the speech from discussing budget planning to discussing health services', "
        "'change the voice from saying \"the crazy one\" to saying \"our generation\"', or "
        "'change the singing from \"morning light\" to \"our generation\"'. "
        "If you cannot hear specific content for both clips, set reject_if_still_unclear=true."
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
        "Reject if the pair is a near-duplicate without a visible/audible/textual delta, or if it needs multiple broad changes to map reference to target. "
        "For speech edits, require transcript-backed speech evidence on both sides and audio_required=true; generic 'talking' or 'speaking to camera' is not enough. "
        "For audio_event edits, accept only non-speech sound/music/environment changes; reject if the audio difference is only speech or narration content. "
        "For visible_text edits, require concrete OCR/on-screen text evidence on both sides and a target that is unique beyond template similarity. "
        "For this dataset pass, reject speech and visible_text as final accepted types even if they are real; they may still be useful as diagnostics. "
        "Cross-video template-compatible subject/object/scene swaps are allowed when the clip template remains stable and the main difference is single and well evidenced. "
        "Use scores from 0.0 to 1.0."
    )


def _audio_anchor_visual_verification_system_prompt() -> str:
    return (
        "You are the strict final verifier for the audio-anchor visual-edit line of a composed video retrieval dataset. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"accept": boolean, "reject_reason": string, "recommended_edit_text": string, '
        '"visual_delta_type": string, "visual_delta_strength": number, "near_duplicate_risk": number, '
        '"reference_satisfies_edit": boolean, "target_satisfies_edit": boolean, '
        '"caption_equivalent": boolean, "order_only_scene_reorder": boolean, '
        '"weak_synonym_or_wording_delta": boolean, "evidence": [string]}. '
        "This line is visual_edit_audio_anchor: audio is preserved context only. The edit_text must describe one clear visual change. "
        "Accept only if the actual target video visibly satisfies the edit, the actual reference video does not, "
        "and the difference is a useful retrieval target beyond nearly identical frames or caption wording. "
        "Reject synonym or weak attribute edits such as bright core to luminous core, microphone clipped to shirt to receding hairline, "
        "tiny lighting changes, closer/farther camera framing, or reordered shots of the same scenes. "
        "Reject if the clips are effectively the same, if the reference already satisfies the edit, "
        "if the target only changes speech/audio/text, or if multiple stronger visual changes compete with the edit. "
        "Inspect the beginning, middle, and end of both clips before deciding. "
        "Set visual_delta_strength from 0.0 to 1.0; values below 0.70 are not useful. "
        "Set near_duplicate_risk from 0.0 to 1.0; values above 0.85 indicate the pair is too similar unless the visible edit is obvious."
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
        "Reject broad stock-pair shifts where the edit would require changing the whole clip instead of one primary difference. "
        "For speech pairs, state what the reference speech says and what the target speech says; if either side lacks transcript-backed speech content, mark difference_matches_edit=false. "
        "For audio_event pairs, reject speech-only/narration-only changes as audio_event; audio_event must be non-language sound evidence. "
        "For visible_text pairs, quote the observed reference and target on-screen text and reject if either side lacks OCR evidence. "
        "Ignore intro cards, outro cards, title cards, lower-thirds, and boundary-frame text as final accepted deltas for this dataset pass. "
        "Cross-video template-compatible subject/object/scene swaps are valid if the template stays aligned and the target uniquely satisfies the edit. "
        "In edit_text_quality_check, reject caption-like edit_text, modality leakage, multiple primary differences, and cases where reference already satisfies the edit. "
        "The projected target caption should describe what the reference would become after applying the edit."
    )


def _video_edit_planner_system_prompt() -> str:
    return (
        "You are the strongest Omni video-edit prompt planner for synthetic composed video retrieval. "
        "Watch the short reference clip and use the candidate edit as a hint, not as something to blindly copy. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"should_generate": boolean, "edit_text": string, "difference": object, '
        '"source_prompt": string, "target_prompt": string, '
        '"edit_token": string, "preserve_tokens": [string], "negative_prompt": string, '
        '"edit_region": string, "mask_query": string, "preserve_regions": [string], '
        '"model_route": string, "reason": string}. '
        "First understand the reference video: main subjects, stable scene, visible text, actions, editable attributes, and bad edits. "
        "If the candidate edit is unsuitable but the reference clip has a safer single visual edit, revise edit_text and difference to that safer edit instead of rejecting. "
        "Prefer VACE-friendly edits on existing content in this order: clothing/outfit changes, background replacement, video style changes, object replacement inside an existing maskable region, object removal/inpainting, large subject color/material changes, lighting/weather/time-of-day changes. "
        "Do not revise to naked small-object insertion, tiny accessories, exact count edits, or precise text edits. "
        "Only revise to attribute/color/material edits, existing-object replacements, or simple action edits that are visibly supported by the reference. "
        "The source_prompt must faithfully describe the reference video. "
        "The target_prompt must be a positive description of the desired edited video, not a bare command. "
        "For object replacement, describe the target object in the original source-object location and explicitly state that the source object is no longer visible. "
        "For object removal, describe the source-object area as clean/naturally filled and explicitly state that the object is no longer visible. "
        "The target_prompt must preserve the same subject, camera, lighting, timing, and layout, while applying exactly one visual edit. "
        "The edit_token is the one object, attribute, or action concept the editor should change. "
        "mask_query is the visual target that Grounded-SAM-2 should segment, for example robot body, shirt, cup, glasses, or the foreground subject for background edits. "
        "For background edits, do not use mask_query=background; use the foreground subject so the mask can be inverted to edit the background. "
        "preserve_tokens are concepts that must stay unchanged, but must not include the object being replaced or removed. "
        "preserve_regions are visual regions that must stay unchanged outside the mask. "
        "negative_prompt must explicitly forbid changing people, scene, camera, visible text, timing, and unrelated objects, but must not forbid changing the source object being replaced or removed. "
        "edit_region should be localized when possible, such as top-right paper surface, wall area, desk surface, hand-held object, or background. "
        "For VACE, route clothing/background/style/object-replacement/removal/color/material edits to vace_controlled. "
        "Do not route no-object -> object insertion such as stickers, plants, badges, nose rings, posters, text, or labels to VACE unless a deterministic mask/overlay editor is explicitly available. "
        "Reject by setting should_generate=false when the edit is audio-only, speech/topic, visible text/OCR, broad scene replacement, impossible for this reference, or likely to change the whole clip. "
        "Do not write malformed phrases like 'Add only no chair' or 'Add only tablet' for replacements; write the clean target-video description."
    )


def _stable_clip_selector_system_prompt() -> str:
    return (
        "You select short stable clips for synthetic video editing. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"start_sec": number, "end_sec": number, "stability_score": number, '
        '"camera_motion": "static"|"slow_pan"|"unstable"|"unknown", "main_subjects": [string], '
        '"visible_text_risk": boolean, "recommended_for_vace": boolean, "reason": string}. '
        "Choose a 5-8 second window with one clear subject, stable camera, minimal scene cuts, minimal visible text, "
        "and an edit-friendly visual structure. Prefer windows suitable for masked VACE editing."
    )


def _build_stable_clip_selector_user_content(
    *,
    source_video_path: str,
    media_info: dict[str, Any],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[dict[str, Any]]:
    prompt = (
        "Task: choose one stable short window from this source video for later VACE editing.\n"
        f"Media info JSON:\n{json.dumps(media_info, ensure_ascii=False)}\n"
        f"Window length must be between {min_clip_seconds:.1f} and {max_clip_seconds:.1f} seconds.\n"
        "Prefer a window with a single main subject, no/low visible text risk, no rapid cuts, and a maskable object or subject.\n"
        "If no good window exists, set recommended_for_vace=false and still return the least bad 5-8 second window."
    )
    return [
        {"type": "video_url", "video_url": {"url": source_video_path}},
        {"type": "text", "text": prompt},
    ]


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


def _build_detective_observation_user_content(clip_path: str, *, audio_focused: bool = False) -> list[dict[str, Any]]:
    prompt = (
        "Observation pass: inspect the clip like an independent observer.\n"
        "List visible subjects, object counts, actions, scene, visible text, speech, non-speech audio events, and timeline beats.\n"
        "Name non-speech audio explicitly, for example background music, applause, electronic hum, wind, machinery, footsteps, or animal sounds.\n"
        "Also list any uncertainties that a later detective pass should be careful about."
    )
    if audio_focused:
        prompt += (
            "\nAudio-focused requirement: explicitly state whether there is speech/transcript, crowd cheering, applause, music, ambience, or no reliable distinctive audio. "
            "If speech is present, capture what is being said as a short transcript/paraphrase or topic summary. "
            "Do not guess low hum/click/tone as a dataset signal if it is not clearly audible."
        )
    return [
        {"type": "video_url", "video_url": {"url": clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_detective_toolbox_user_content(
    *,
    clip_path: str,
    tool_observations: list[dict[str, Any]],
    audio_focused: bool = False,
) -> list[dict[str, Any]]:
    prompt = (
        "Tool-box observation pass: inspect the clip using the structured tool observations below.\n"
        f"Tool observations JSON:\n{json.dumps(tool_observations, ensure_ascii=False)}\n"
        "Return concrete evidence for visual events, non-speech audio events, visible text, speech/transcript, timeline beats, "
        "and remaining uncertainties."
    )
    if audio_focused:
        prompt += "\nAudio-focused requirement: prioritize transcript/paraphrase/topic, crowd, applause, music, and ambient evidence; flag uncertain audio instead of inventing it."
    return [
        {"type": "video_url", "video_url": {"url": clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_detective_final_user_content(
    *,
    clip_path: str,
    observations: dict[str, Any],
    tool_observations: list[dict[str, Any]] | None = None,
    audio_focused: bool = False,
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
    if audio_focused:
        prompt += (
            "\nAudio-focused requirement: make speech, speakers_and_transcript, and audio_events useful for pairing. "
            "For one-person speech or livestream clips, write the actual spoken topic/content for this 6-second segment, not a generic label. "
            "Use audio_events for concrete crowd/applause/music/environment sounds; put uncertain or vague sounds in detective_notes/uncertainties, not as a confident audio_event."
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
        "Prefer clean subject/object/scene or non-speech audio differences over speech or OCR differences. "
        "For action proposals, edit_text must be a verb/action change such as starts/stops/changes from doing X to doing Y, "
        "and both reference and target storyline/events/actions must support that action change. "
        "If the evidence is mainly object presence, color, visible text, speech content, or audio, choose that type instead of action. "
        "Separate speech from audio_event: speech is language content or speaker delivery; audio_event is non-speech music, environment, or event sound. "
        "For speech, edit_text must name the specific spoken content change and must be grounded in transcript-backed evidence, not just 'talks about a different topic'. "
        "For audio_event, edit_text must name a non-speech sound change and must not be only narration/speech. "
        "For audio_event, never mention visual subjects such as person, woman, man, room, background, toy, or dollhouse in edit_text. "
        "If the heuristic hint says same_template_cluster and the primary difference is attribute with speaker signatures, write edit_text as 'change the speaker from <signature> to <signature>'. "
        "If the heuristic hint says same_template_cluster and the primary difference is object_presence, prefer 'replace the held object from <A> to <B>' or 'change the featured object from <A> to <B>'. "
        "If the heuristic hint says same_template_cluster and the primary difference is scene, prefer 'change the setting from <A> to <B>'. "
        "If the heuristic hint says audio_anchor_required, treat audio as preserved context only: choose a visual difference and do not mention audio, sound, speech, or music in edit_text. "
        "Do not turn reordered shots of the same scenes, people, or objects into object_presence/object_count edits; reject or choose a truly localized evidence-backed difference instead. "
        "For people/personnel/staff/crowd edits, verify that the reference actually lacks that group and the target clearly contains it; do not infer large counts from a busy room caption. "
        "Use type-specific edit_text style: object_presence add/remove X; object_count change the number of X from A to B; action change the action from X to Y; audio_event add/remove/replace sound X; speech change speech from X to Y; visible_text change on-screen text from X to Y. "
        "Reject your own proposal if edit_text sounds like a caption, contains multiple changes, or leaks another modality. "
        "Do not choose speech or visible_text when a clean subject, object, scene, or non-speech audio delta is available. "
        "Use event/timeline evidence to choose a difference that is concrete, localized, and needed for retrieval. "
        "Only include audio in modalities when the edit actually requires listening. "
        "Do not mention secondary audio, speech, or visible-text details in edit_text unless they are the chosen primary difference. "
        "Keep captions factual and concise."
    )
    return [{"type": "text", "text": prompt}]


def _build_single_source_pair_user_content(
    *,
    reference_clip_path: str,
    target_clip_path: str,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    whole_annotation: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    line = _normalize_audio_dataset_line(candidate.get("audio_dataset_line") if isinstance(candidate, dict) else None)
    ann_char_limit = 360 if line == "speech_audio_content" else 520 if line == "visual_audio_anchor" else 4200
    candidate_char_limit = 260 if line == "speech_audio_content" else 650 if line == "visual_audio_anchor" else 3000
    context_text = ""
    if whole_annotation:
        whole_limit = 0 if line == "speech_audio_content" else 140 if line == "visual_audio_anchor" else 2200
        if whole_limit:
            context_text = f"Whole source video context JSON:\n{_prompt_json(whole_annotation, max_chars=whole_limit)}\n"
    candidate_text = ""
    if candidate:
        candidate_text = f"Chronological pair candidate JSON:\n{_prompt_json(candidate, max_chars=candidate_char_limit)}\n"
    line_text = ""
    if line == "visual_audio_anchor":
        line_text = (
            "A-line rule: write a visual-only edit. Audio is preserved context. "
            "If the visual change is not large and stable, reject instead of proposing an attribute edit.\n"
        )
    elif line == "speech_audio_content":
        line_text = (
            "B-line: audio primary; visuals same context. Use speech/audio_event only; reject vague audio or visual edits.\n"
        )
    prompt = (
        "Compare two same-source 6s clips and write one evidence-backed edit. Videos are primary evidence; annotations are hints.\n"
        f"{line_text}"
        f"{context_text}"
        f"Ref annotation:\n{_prompt_json(reference_annotation, max_chars=ann_char_limit)}\n"
        f"Tgt annotation:\n{_prompt_json(target_annotation, max_chars=ann_char_limit)}\n"
        f"{candidate_text}"
        "Return JSON only. The edit must be concrete and human-verifiable."
    )
    return [
        {"type": "text", "text": "Reference clip:"},
        {"type": "video_url", "video_url": {"url": reference_clip_path}},
        {"type": "text", "text": "Target clip:"},
        {"type": "video_url", "video_url": {"url": target_clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_single_source_final_verification_user_content(
    *,
    reference_clip_path: str,
    target_clip_path: str,
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    local_gate_report: dict[str, Any],
    whole_annotation: dict[str, Any] | None = None,
    audio_dataset_line: str | None = None,
) -> list[dict[str, Any]]:
    line = _normalize_audio_dataset_line(audio_dataset_line)
    ann_char_limit = 520 if line == "speech_audio_content" else 440 if line == "visual_audio_anchor" else 3600
    context_text = ""
    if whole_annotation:
        whole_limit = 120 if line in {"visual_audio_anchor", "speech_audio_content"} else 1800
        context_text = f"Whole source video context JSON:\n{_prompt_json(whole_annotation, max_chars=whole_limit)}\n"
    line_text = ""
    if line == "visual_audio_anchor":
        line_text = (
            "A-line final rule: accept only large visual edits; audio must not be the edit. "
            "Return large_visual_delta and audio_context_preserved.\n"
        )
    elif line == "speech_audio_content":
        line_text = (
            "B-line final rule: accept if the audible speech/audio change is primary and visuals share the same person, same scene, same program, "
            "or same broadcast/view context. Minor framing, pose, camera, or action changes are acceptable. "
            "Do not reject only because the audio evidence covers part of the 6s clip; report segment_wide=false but accept when target clearly contains the requested audio. "
            "Reject only when visual changes are dominant enough that listening is unnecessary. Return audio_primary, visual_locked, visual_too_different_for_B, and edit_text_audio_only.\n"
        )
    prompt = (
        "Task: final-check whether this single-source pair should be accepted.\n"
        "Use the actual videos first. Check the reference and target at approximately 0.2s, 2.5s, and 4.8s, "
        "plus any other moment needed to verify the edit.\n"
        f"{line_text}"
        f"{context_text}"
        f"Pair proposal JSON:\n{_prompt_json(model_fields, max_chars=900 if line in {'visual_audio_anchor', 'speech_audio_content'} else 1800)}\n"
        f"Local gate report JSON:\n{_prompt_json(local_gate_report, max_chars=450 if line in {'visual_audio_anchor', 'speech_audio_content'} else 1000)}\n"
        f"Reference segment annotation JSON:\n{_prompt_json(reference_annotation, max_chars=ann_char_limit)}\n"
        f"Target segment annotation JSON:\n{_prompt_json(target_annotation, max_chars=ann_char_limit)}\n"
        "Answer the required schema exactly. The accept field must be false if the target does not visibly/audibly satisfy edit_text, "
        "if the reference already satisfies edit_text, if the pair is text/OCR-driven, or if the edit_text describes the wrong subject or composition."
    )
    return [
        {"type": "text", "text": "Reference clip for final verification:"},
        {"type": "video_url", "video_url": {"url": reference_clip_path}},
        {"type": "text", "text": "Target clip for final verification:"},
        {"type": "video_url", "video_url": {"url": target_clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_b_line_edit_text_refinement_user_content(
    *,
    reference_clip_path: str,
    target_clip_path: str,
    model_fields: dict[str, Any],
    final_verification: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    prompt = (
        "Task: refine the B-line edit_text only. Do not change the pair decision except by rejecting unspecific audio text.\n"
        f"Pair proposal JSON:\n{_prompt_json(model_fields, max_chars=900)}\n"
        f"Final verification JSON:\n{_prompt_json(final_verification, max_chars=600)}\n"
        f"Reference audio-focused annotation JSON:\n{_prompt_json(reference_annotation, max_chars=520)}\n"
        f"Target audio-focused annotation JSON:\n{_prompt_json(target_annotation, max_chars=520)}\n"
        "Return the required JSON schema. The refined_edit_text must not mention visual content. "
        "If the audio evidence is only 'speech present', 'not transcribed', 'unintelligible', or a generic sound, reject it."
    )
    return [
        {"type": "text", "text": "Reference clip for edit-text refinement:"},
        {"type": "video_url", "video_url": {"url": reference_clip_path}},
        {"type": "text", "text": "Target clip for edit-text refinement:"},
        {"type": "video_url", "video_url": {"url": target_clip_path}},
        {"type": "text", "text": prompt},
    ]


def _build_b_line_speech_rewrite_user_content(
    *,
    reference_clip_path: str,
    target_clip_path: str,
    model_fields: dict[str, Any],
    final_verification: dict[str, Any],
    edit_text_refinement: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    prompt = (
        "Task: repair a B-line speech edit_text by listening to the audio only.\n"
        f"Current pair proposal JSON:\n{_prompt_json(model_fields, max_chars=650)}\n"
        f"Final verification JSON:\n{_prompt_json(final_verification, max_chars=420)}\n"
        f"Prior edit-text refinement JSON:\n{_prompt_json(edit_text_refinement, max_chars=360)}\n"
        f"Reference audio annotation JSON:\n{_prompt_json(reference_annotation, max_chars=420)}\n"
        f"Target audio annotation JSON:\n{_prompt_json(target_annotation, max_chars=420)}\n"
        "Return only the required JSON. The refined edit_text must be specific and audio-only. "
        "Reject if both clips are only generic speaking/talking, if the content is unclear, or if the two sides have the same spoken content."
    )
    return [
        {"type": "text", "text": "Reference clip for speech rewrite:"},
        {"type": "video_url", "video_url": {"url": reference_clip_path}},
        {"type": "text", "text": "Target clip for speech rewrite:"},
        {"type": "video_url", "video_url": {"url": target_clip_path}},
        {"type": "text", "text": prompt},
    ]


def _prompt_json(payload: dict[str, Any], *, max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 18)].rstrip() + "... [truncated]"


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
        "If the pair proposal JSON says audio_anchor_required, reject edits that make audio, sound, speech, or music the primary difference; audio should only anchor the context for a visual edit. "
        "Reject visible_text and speech proposals for final accepted output in this dataset pass; they can still be diagnostic. "
        "If the pair proposal JSON says the clips are same_template_cluster, allow cross-video subject/object/scene swaps when the template stays stable and the main delta is single. "
        "If you reject the pair, reject_reason must be a non-empty sentence naming the main failed gate or threshold."
    )
    return [{"type": "text", "text": prompt}]


def _build_audio_anchor_visual_verification_user_content(
    *,
    proposal: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    reference_clip_path: str,
    target_clip_path: str,
) -> list[dict[str, Any]]:
    prompt = (
        "Task: final-check this audio-anchor visual-edit candidate.\n"
        "Use the attached reference and target videos as primary evidence. Use annotations and audio_anchor_score only as supporting context.\n"
        f"Pair proposal JSON:\n{json.dumps(proposal, ensure_ascii=False)}\n"
        f"Reference annotation JSON:\n{json.dumps(reference_annotation, ensure_ascii=False)}\n"
        f"Target annotation JSON:\n{json.dumps(target_annotation, ensure_ascii=False)}\n"
        "Rules:\n"
        "- Audio similarity means the clips may share context; do not use audio as the edited attribute.\n"
        "- edit_text must be visual-only and must not mention audio, speech, sound, music, or transcript.\n"
        "- Reject if the proposed edit is only wording, brightness, synonym phrasing, shot order, or a near-duplicate visual state.\n"
        "- Reject if the reference already visually satisfies edit_text or the target does not clearly satisfy it.\n"
        "- If rejecting but a cleaner visual edit exists, put it in recommended_edit_text; otherwise leave it empty.\n"
        "- Provide concrete evidence from the videos, not only from captions."
    )
    return [
        {"type": "text", "text": "Reference video for audio-anchor visual verification:"},
        {"type": "video_url", "video_url": {"url": _materialize_video_url(reference_clip_path)}},
        {"type": "text", "text": "Target video for audio-anchor visual verification:"},
        {"type": "video_url", "video_url": {"url": _materialize_video_url(target_clip_path)}},
        {"type": "text", "text": prompt},
    ]


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
        "For people/personnel/staff/crowd object edits, reject if both videos already show that group or if the only difference is the order of shared shots. "
        "For action edit_text, set difference_matches_edit=false unless the reference and target have different concrete actions "
        "supported by action/storyline/event evidence. "
        "For speech edit_text, explicitly check: what transcript-backed speech does the reference contain, what transcript-backed speech does the target contain, "
        "whether the edit requires listening, and whether visuals alone would fail to distinguish the target. "
        "If speech evidence is generic or missing, set target_matches_projection=false or difference_matches_edit=false. "
        "For audio_event edit_text, set difference_matches_edit=false if the change is only spoken topic/narration rather than a non-speech sound. "
        "For visible_text edit_text, reject title-card, lower-third, intro/outro, or boundary-frame-only evidence. "
        "For same_template_cluster subject/object/scene edits, accept cross-video pairs when the clips share a stable template and the target uniquely matches the proposed swap. "
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


def _build_video_edit_planner_user_content(
    *,
    reference_clip_path: str,
    reference_annotation: dict[str, Any],
    candidate: dict[str, Any],
    route_hint: str,
) -> list[dict[str, Any]]:
    prompt = (
        "Task: build a controlled video-edit prompt plan for this reference clip.\n"
        "The goal is to create a synthetic target video for composed video retrieval: reference video + edit_text -> target video.\n"
        "The video editor must be used with purpose. First understand what is actually in the reference video, then decide whether the candidate edit is suitable.\n"
        f"Route hint: {route_hint}\n"
        f"Reference annotation JSON:\n{json.dumps(reference_annotation, ensure_ascii=False)}\n"
        f"Candidate edit JSON:\n{json.dumps(candidate, ensure_ascii=False)}\n"
        "Rules:\n"
        "- Keep the clip short-context identity: same scene, subject identity, camera motion, timing, lighting, and layout.\n"
        "- Prefer VACE edits in this order: change clothing/outfit, replace background, style transfer, replace an existing object, remove/inpaint an object, change large color/material, change lighting/weather/time.\n"
        "- Do not propose naked small-object insertion such as a sticker, plant, badge, nose ring, poster, label, logo, or text unless a deterministic mask/overlay editor is explicitly available.\n"
        "- If the candidate edit is not suitable but another local visual edit is suitable for this exact reference, output the safer revised edit_text and difference.\n"
        "- The safest VACE edit is usually a large, visible attribute change on the main subject, not adding a new object to the background.\n"
        "- If the edit can be achieved by fixing a selected reference image or background plate into the masked region, prefer that deterministic route first and reserve VACE for seam repair, harmonization, or hidden-content synthesis.\n"
        "- For full background replacement in a stable talking-head shot, prefer a fixed foreground/background composite route before a full generative background edit.\n"
        "- Do not plan visible-text edits unless OCR-backed text editing is explicitly available.\n"
        "- Do not plan audio_event or speech edits for a video editor.\n"
        "- Do not use a universal edit. The edit must fit objects/actions visible in this reference video.\n"
        "- Set should_generate=false only when no safe single local visual edit is available for this video.\n"
        "Return JSON only."
    )
    return [
        {"type": "video_url", "video_url": {"url": reference_clip_path}},
        {"type": "text", "text": prompt},
    ]


def _src_ref_image_audit_system_prompt() -> str:
    return (
        "You audit generated source reference images for VACE video editing. "
        "Select only candidates that should be passed as src_ref_images to VACE. "
        "The selected images must match the edit role, target object or scene, camera perspective, scale, and material; "
        "they must avoid visible text, watermarks, logos, extra people, wrong objects, or product-flatlay views that will confuse the video editor. "
        "For replacement objects, select at most 1-2 strong object references. "
        "For clothing, select at most 1 strong candidate and prefer wearable upper-body references suitable for a person, not flat product catalog images. "
        "Reject empty garments, hangers, ghost mannequins, flat lays, and product-only catalog shots. "
        "For black jacket edits, only select a cropped human torso wearing an open black long-sleeved jacket over a black shirt/T-shirt, with shoulders, arms, sleeves, and open jacket structure visible. "
        "For backgrounds, prefer 16:9 empty scene plates with no people or text. "
        "Return JSON only with fields: selected_indices, audit, rejected, reason. "
        "selected_indices are 1-based indices from the candidate list."
    )


def _build_src_ref_image_audit_user_content(
    *,
    src_ref_plan: dict[str, Any],
    candidate_image_paths: list[str],
    max_selected: int,
) -> list[dict[str, Any]]:
    prompt = (
        "Audit these generated candidate images for a VACE src_ref_images input.\n"
        f"Max selected images: {max_selected}\n"
        f"Source reference plan JSON:\n{json.dumps(src_ref_plan, ensure_ascii=False)}\n"
        "For each candidate, decide whether it is useful as a source reference image. "
        "Reject images with wrong target, visible text/watermark/logo, poor composition, mismatched perspective, or confusing extra objects.\n"
        "Return JSON schema:\n"
        "{"
        "\"selected_indices\":[1],"
        "\"audit\":[{\"index\":1,\"score\":0.0,\"verdict\":\"select|reject\",\"reason\":\"...\"}],"
        "\"rejected\":[{\"index\":2,\"reason\":\"...\"}],"
        "\"reason\":\"short final selection reason\""
        "}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, image_path in enumerate(candidate_image_paths, start=1):
        content.append({"type": "text", "text": f"Candidate {index}: {image_path}"})
        content.append({"type": "image_url", "image_url": {"url": image_path}})
    return content


def _normalize_src_ref_image_audit_payload(
    payload: dict[str, Any],
    *,
    candidate_count: int,
    max_selected: int,
) -> dict[str, Any]:
    selected_indices: list[int] = []
    raw_indices = payload.get("selected_indices")
    if isinstance(raw_indices, list):
        for raw_index in raw_indices:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= candidate_count and index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= max(1, max_selected):
                break

    audit_rows = payload.get("audit")
    if not isinstance(audit_rows, list):
        audit_rows = []
    normalized_audit = []
    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= candidate_count:
            continue
        normalized_audit.append(
            {
                "index": index,
                "score": row.get("score"),
                "verdict": str(row.get("verdict", "")).strip(),
                "reason": str(row.get("reason", "")).strip(),
            }
        )

    rejected_rows = payload.get("rejected")
    if not isinstance(rejected_rows, list):
        rejected_rows = []
    rejected = []
    for row in rejected_rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= index <= candidate_count:
            rejected.append({"index": index, "reason": str(row.get("reason", "")).strip()})

    return {
        "selected_indices": selected_indices,
        "audit": normalized_audit,
        "rejected": rejected,
        "reason": str(payload.get("reason", "")).strip(),
    }


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
        audio_focused: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if tool_observations:
            observations = self._request_json(
                user_content=_build_detective_toolbox_user_content(
                    clip_path=clip_path,
                    tool_observations=tool_observations,
                    audio_focused=audio_focused,
                ),
                system_prompt=_detective_toolbox_system_prompt(audio_focused=audio_focused),
                max_tokens=1400,
            )
        else:
            observations = self._request_json(
                user_content=_build_detective_observation_user_content(clip_path, audio_focused=audio_focused),
                system_prompt=_detective_observation_system_prompt(audio_focused=audio_focused),
                max_tokens=1200,
            )
        final_payload = self._request_json(
            user_content=_build_detective_final_user_content(
                clip_path=clip_path,
                observations=observations,
                tool_observations=tool_observations,
                audio_focused=audio_focused,
            ),
            system_prompt=_detective_final_system_prompt(audio_focused=audio_focused),
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

    def propose_single_source_pair(
        self,
        *,
        reference_clip_path: str,
        target_clip_path: str,
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        whole_annotation: dict[str, Any] | None = None,
        candidate: dict[str, Any] | None = None,
        audio_dataset_line: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        line = _normalize_audio_dataset_line(audio_dataset_line)
        raw_payload = self._request_json(
            user_content=_build_single_source_pair_user_content(
                reference_clip_path=reference_clip_path,
                target_clip_path=target_clip_path,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                whole_annotation=whole_annotation,
                candidate=candidate,
            ),
            system_prompt=_single_source_pair_system_prompt(line),
            max_tokens=1200 if line == "speech_audio_content" else 1100 if line == "visual_audio_anchor" else 1500,
        )
        if line in {"visual_audio_anchor", "speech_audio_content"}:
            raw_payload = _repair_audio_line_single_source_pair_payload(
                raw_payload,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                candidate=candidate,
                audio_dataset_line=line,
            )
        return _normalize_single_source_pair_payload(raw_payload), raw_payload

    def verify_single_source_pair_final(
        self,
        *,
        reference_clip_path: str,
        target_clip_path: str,
        model_fields: dict[str, Any],
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        local_gate_report: dict[str, Any],
        whole_annotation: dict[str, Any] | None = None,
        audio_dataset_line: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_single_source_final_verification_user_content(
                reference_clip_path=reference_clip_path,
                target_clip_path=target_clip_path,
                model_fields=model_fields,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                local_gate_report=local_gate_report,
                whole_annotation=whole_annotation,
                audio_dataset_line=audio_dataset_line,
            ),
            system_prompt=_single_source_final_verification_system_prompt(audio_dataset_line),
            max_tokens=900,
        )
        line = _normalize_audio_dataset_line(audio_dataset_line)
        if line in {"visual_audio_anchor", "speech_audio_content"}:
            raw_payload = _repair_audio_line_single_source_final_verification_payload(
                raw_payload,
                model_fields=model_fields,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                local_gate_report=local_gate_report,
                audio_dataset_line=line,
            )
        return _normalize_single_source_final_verification_payload(raw_payload), raw_payload

    def refine_b_line_edit_text(
        self,
        *,
        reference_clip_path: str,
        target_clip_path: str,
        model_fields: dict[str, Any],
        final_verification: dict[str, Any],
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_b_line_edit_text_refinement_user_content(
                reference_clip_path=reference_clip_path,
                target_clip_path=target_clip_path,
                model_fields=model_fields,
                final_verification=final_verification,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            ),
            system_prompt=_b_line_edit_text_refinement_system_prompt(),
            max_tokens=700,
        )
        return _normalize_b_line_edit_text_refinement_payload(raw_payload), raw_payload

    def refine_b_line_speech_content(
        self,
        *,
        reference_clip_path: str,
        target_clip_path: str,
        model_fields: dict[str, Any],
        final_verification: dict[str, Any],
        edit_text_refinement: dict[str, Any],
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_b_line_speech_rewrite_user_content(
                reference_clip_path=reference_clip_path,
                target_clip_path=target_clip_path,
                model_fields=model_fields,
                final_verification=final_verification,
                edit_text_refinement=edit_text_refinement,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            ),
            system_prompt=_b_line_speech_rewrite_system_prompt(),
            max_tokens=700,
        )
        return _normalize_b_line_speech_rewrite_payload(raw_payload), raw_payload

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

    def verify_audio_anchor_visual_pair(
        self,
        *,
        proposal: dict[str, Any],
        reference_annotation: dict[str, Any],
        target_annotation: dict[str, Any],
        reference_clip_path: str,
        target_clip_path: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_audio_anchor_visual_verification_user_content(
                proposal=proposal,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                reference_clip_path=reference_clip_path,
                target_clip_path=target_clip_path,
            ),
            system_prompt=_audio_anchor_visual_verification_system_prompt(),
            max_tokens=900,
        )
        return _normalize_audio_anchor_visual_verification_payload(raw_payload), raw_payload

    def plan_video_edit(
        self,
        *,
        reference_clip_path: str,
        reference_annotation: dict[str, Any],
        candidate: dict[str, Any],
        route_hint: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_video_edit_planner_user_content(
                reference_clip_path=reference_clip_path,
                reference_annotation=reference_annotation,
                candidate=candidate,
                route_hint=route_hint,
            ),
            system_prompt=_video_edit_planner_system_prompt(),
            max_tokens=1300,
        )
        repaired_payload = _repair_video_edit_plan_payload(
            raw_payload,
            candidate=candidate,
            reference_annotation=reference_annotation,
            route_hint=route_hint,
        )
        return _normalize_video_edit_plan_payload(repaired_payload), raw_payload

    def audit_src_ref_images(
        self,
        *,
        src_ref_plan: dict[str, Any],
        candidate_image_paths: list[str],
        max_selected: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_src_ref_image_audit_user_content(
                src_ref_plan=src_ref_plan,
                candidate_image_paths=candidate_image_paths,
                max_selected=max_selected,
            ),
            system_prompt=_src_ref_image_audit_system_prompt(),
            max_tokens=1000,
        )
        return _normalize_src_ref_image_audit_payload(
            raw_payload,
            candidate_count=len(candidate_image_paths),
            max_selected=max_selected,
        ), raw_payload

    def select_stable_clip_window(
        self,
        *,
        source_video_path: str,
        media_info: dict[str, Any],
        min_clip_seconds: float,
        max_clip_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_payload = self._request_json(
            user_content=_build_stable_clip_selector_user_content(
                source_video_path=source_video_path,
                media_info=media_info,
                min_clip_seconds=min_clip_seconds,
                max_clip_seconds=max_clip_seconds,
            ),
            system_prompt=_stable_clip_selector_system_prompt(),
            max_tokens=700,
        )
        return _normalize_stable_clip_selection_payload(
            raw_payload,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
            duration_seconds=float(media_info.get("duration_seconds") or 0.0),
        ), raw_payload

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
            if item.get("type") == "image_url":
                image_url = dict(item["image_url"])
                image_url["url"] = _materialize_image_url(str(image_url["url"]))
                request_content.append({"type": "image_url", "image_url": image_url})
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
            raise RuntimeError(f"composed data request failed: HTTP {exc.code}: {detail}") from exc
        content = raw_response["choices"][0]["message"]["content"]
        try:
            payload = _extract_json(content)
        except Exception as parse_exc:
            payload = self._repair_malformed_json_response(
                malformed_content=str(content),
                parse_error=parse_exc,
                max_tokens=max_tokens,
            )
        if not isinstance(payload, dict):
            raise ValueError("model response must decode to a JSON object")
        return payload

    def _repair_malformed_json_response(
        self,
        *,
        malformed_content: str,
        parse_error: BaseException,
        max_tokens: int,
    ) -> dict[str, Any]:
        repair_prompt = (
            "Repair the malformed JSON-like model response into exactly one valid JSON object. "
            "Preserve the original keys and meanings as much as possible. Do not add markdown or commentary. "
            "If a value is unclear, use an empty string, empty list, false, 0, or null rather than inventing evidence."
        )
        repair_user_text = (
            f"Parse error: {type(parse_error).__name__}: {parse_error}\n"
            "Malformed response:\n"
            f"{malformed_content[:8000]}"
        )
        repair_payload = {
            "model": self.model,
            "modalities": ["text"],
            "max_tokens": min(max(700, int(max_tokens or 900)), 1400),
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": repair_prompt},
                {"role": "user", "content": [{"type": "text", "text": repair_user_text}]},
            ],
        }
        repair_request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(repair_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(repair_request, timeout=self.timeout_seconds) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"composed data JSON repair failed: HTTP {exc.code}: {detail}") from parse_error
        repair_content = raw_response["choices"][0]["message"]["content"]
        try:
            repaired = _extract_json(str(repair_content))
        except Exception as repair_exc:
            raise ValueError(
                "model response JSON repair failed after original parse error "
                f"{type(parse_error).__name__}: {parse_error}; repair error "
                f"{type(repair_exc).__name__}: {repair_exc}"
            ) from parse_error
        if not isinstance(repaired, dict):
            raise ValueError("model JSON repair did not return a JSON object")
        return repaired


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


def _normalize_single_source_pair_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_SINGLE_SOURCE_PAIR_FIELDS)
    if missing_fields:
        raise ValueError(f"single-source pair proposal missing fields: {missing_fields}")

    modalities = _normalize_modalities(payload.get("modalities")) or ["visual"]
    dominant_delta = payload.get("dominant_delta")
    if not isinstance(dominant_delta, dict):
        raise ValueError("single-source pair dominant_delta must be an object")
    reference_state = _normalize_single_source_state(payload.get("reference_state"))
    target_state = _normalize_single_source_state(payload.get("target_state"))
    delta_temporal_extent = _normalize_delta_temporal_extent(payload.get("delta_temporal_extent"))
    subject_roles = _normalize_single_source_subject_roles(payload.get("subject_roles"))
    normalized = {
        "edit_text": str(payload.get("edit_text", "")).strip(),
        "modalities": modalities,
        "reference_caption": str(payload.get("reference_caption", "")).strip(),
        "target_caption": str(payload.get("target_caption", "")).strip(),
        "difference": _validate_difference(payload.get("difference")),
        "dominant_delta": {
            "type": str(dominant_delta.get("type", "")).strip(),
            "from": str(dominant_delta.get("from", "")).strip(),
            "to": str(dominant_delta.get("to", "")).strip(),
            "reason": str(dominant_delta.get("reason", "")).strip(),
        },
        "reference_state": reference_state,
        "target_state": target_state,
        "delta_temporal_extent": delta_temporal_extent,
        "subject_roles": subject_roles,
        "is_segment_wide_delta": _bool_value(payload.get("is_segment_wide_delta")),
        "discarded_deltas": _detail_list(payload.get("discarded_deltas")),
        "evidence": _detail_list(payload.get("evidence")),
        "confidence": _score_value(payload.get("confidence")),
        "accept": _bool_value(payload.get("accept")),
        "reject_reason": str(payload.get("reject_reason", "")).strip(),
        "schema_repaired_fields": _detail_list(payload.get("schema_repaired_fields")),
    }
    for field_name in ("edit_text", "reference_caption", "target_caption"):
        if not normalized[field_name]:
            raise ValueError(f"single-source pair {field_name} is required")
    if not normalized["dominant_delta"]["type"] or not normalized["dominant_delta"]["reason"]:
        raise ValueError("single-source pair dominant_delta type and reason are required")
    if not normalized["evidence"]:
        raise ValueError("single-source pair evidence is required")
    if not normalized["delta_temporal_extent"]["evidence"]:
        raise ValueError("single-source pair delta_temporal_extent evidence is required")
    return normalized


def _repair_audio_line_single_source_pair_payload(
    payload: dict[str, Any],
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    candidate: dict[str, Any] | None,
    audio_dataset_line: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    repaired = dict(payload)
    repaired_fields = _detail_list(repaired.get("schema_repaired_fields"))
    line = _normalize_audio_dataset_line(audio_dataset_line)
    candidate_payload = candidate if isinstance(candidate, dict) else {}
    heuristic_difference = candidate_payload.get("heuristic_difference")
    if not isinstance(heuristic_difference, dict):
        heuristic_difference = candidate_payload.get("difference")
    if not isinstance(heuristic_difference, dict):
        heuristic_difference = {}

    difference = _coerce_single_source_difference(repaired.get("difference"))
    dominant_delta = repaired.get("dominant_delta") if isinstance(repaired.get("dominant_delta"), dict) else {}
    if difference is None:
        difference = _coerce_single_source_difference(dominant_delta)
    if difference is None:
        difference = _coerce_single_source_difference(heuristic_difference)
    if difference is not None:
        existing_difference = repaired.get("difference") if isinstance(repaired.get("difference"), dict) else {}
        merged_difference = {
            "type": difference["type"],
            "from": str(existing_difference.get("from", difference.get("from", ""))).strip(),
            "to": str(existing_difference.get("to", difference.get("to", ""))).strip(),
            "description": str(existing_difference.get("description", difference.get("description", ""))).strip(),
        }
        if not merged_difference["description"]:
            merged_difference["description"] = _single_source_difference_description(merged_difference)
        repaired["difference"] = merged_difference
        if "difference" in _missing_fields(payload, ("difference",)):
            repaired_fields.append("difference")

    diff = repaired.get("difference") if isinstance(repaired.get("difference"), dict) else {}
    diff_type = str(diff.get("type", "")).strip()
    edit_text = str(repaired.get("edit_text", "")).strip()
    if not edit_text:
        edit_text = str(repaired.get("recommended_edit_text", "")).strip()
    if not edit_text and diff_type:
        edit_text = _single_source_edit_text_from_difference(diff, line)
    if edit_text and not str(repaired.get("edit_text", "")).strip():
        repaired["edit_text"] = edit_text
        repaired_fields.append("edit_text")

    modalities = _normalize_modalities(repaired.get("modalities"))
    if not modalities:
        modalities = ["audio"] if diff_type in {"speech", "audio_event"} or line == "speech_audio_content" else ["visual"]
        repaired["modalities"] = modalities
        repaired_fields.append("modalities")

    if not str(repaired.get("reference_caption", "")).strip():
        repaired["reference_caption"] = _single_source_caption_from_annotation(reference_annotation, fallback="reference clip")
        repaired_fields.append("reference_caption")
    if not str(repaired.get("target_caption", "")).strip():
        repaired["target_caption"] = _single_source_caption_from_annotation(target_annotation, fallback="target clip")
        repaired_fields.append("target_caption")

    repaired["dominant_delta"] = _repair_single_source_dominant_delta(
        repaired.get("dominant_delta"),
        difference=diff,
        evidence=repaired.get("evidence"),
    )
    if not isinstance(payload.get("dominant_delta"), dict):
        repaired_fields.append("dominant_delta")

    repaired["reference_state"] = _repair_single_source_state_payload(
        repaired.get("reference_state"),
        annotation=reference_annotation,
    )
    if not isinstance(payload.get("reference_state"), dict):
        repaired_fields.append("reference_state")
    repaired["target_state"] = _repair_single_source_state_payload(
        repaired.get("target_state"),
        annotation=target_annotation,
    )
    if not isinstance(payload.get("target_state"), dict):
        repaired_fields.append("target_state")

    if "accept" not in repaired and edit_text and diff_type:
        repaired["accept"] = True
        repaired_fields.append("accept")

    confidence = _score_value(repaired.get("confidence"))
    if confidence <= 0.0 and _bool_value(repaired.get("accept")):
        repaired["confidence"] = 0.72
        repaired_fields.append("confidence")
        confidence = 0.72

    repaired["delta_temporal_extent"] = _repair_single_source_delta_extent(
        repaired.get("delta_temporal_extent"),
        difference=diff,
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        confidence=confidence,
        accepted=_bool_value(repaired.get("accept")),
    )
    if not isinstance(payload.get("delta_temporal_extent"), dict):
        repaired_fields.append("delta_temporal_extent")

    repaired["subject_roles"] = _repair_single_source_subject_roles_payload(
        repaired.get("subject_roles"),
        target_state=repaired["target_state"],
    )
    if not isinstance(payload.get("subject_roles"), dict):
        repaired_fields.append("subject_roles")

    if "is_segment_wide_delta" not in repaired:
        repaired["is_segment_wide_delta"] = bool(_bool_value(repaired.get("accept")) and confidence >= 0.55)
        repaired_fields.append("is_segment_wide_delta")
    if "discarded_deltas" not in repaired or not isinstance(repaired.get("discarded_deltas"), list):
        repaired["discarded_deltas"] = _detail_list(repaired.get("discarded_deltas"))
        repaired_fields.append("discarded_deltas")

    evidence = _detail_list(repaired.get("evidence"))
    if not evidence:
        evidence = _single_source_repair_evidence(
            difference=diff,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            dominant_delta=repaired["dominant_delta"],
        )
        if evidence:
            repaired["evidence"] = evidence
            repaired_fields.append("evidence")

    if not str(repaired.get("reject_reason", "")).strip():
        repaired["reject_reason"] = "" if _bool_value(repaired.get("accept")) else "model did not accept the pair"
        if not _bool_value(repaired.get("accept")):
            repaired_fields.append("reject_reason")

    if repaired_fields:
        repaired["schema_repaired_fields"] = sorted(set(repaired_fields))
    return repaired


def _coerce_single_source_difference(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    difference_type = str(value.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    difference_type = DIFFERENCE_TYPE_ALIASES.get(difference_type, difference_type)
    if difference_type not in ALLOWED_DIFFERENCE_TYPES:
        return None
    result = {
        "type": difference_type,
        "from": str(value.get("from", "")).strip(),
        "to": str(value.get("to", "")).strip(),
        "description": str(value.get("description", value.get("reason", ""))).strip(),
    }
    if not (result["from"] or result["to"] or result["description"]):
        return None
    if not result["description"]:
        result["description"] = _single_source_difference_description(result)
    return result


def _single_source_difference_description(difference: dict[str, Any]) -> str:
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if from_value and to_value:
        return f"changes from {from_value} to {to_value}"
    if to_value:
        return f"target changes to {to_value}"
    if from_value:
        return f"reference starts from {from_value}"
    return "single-source pair has one proposed difference"


def _single_source_edit_text_from_difference(difference: dict[str, Any], audio_dataset_line: str) -> str:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if difference_type == "speech":
        if from_value and to_value:
            return f"change the speech from {from_value} to {to_value}"
        if to_value:
            return f"change the speech to {to_value}"
    if difference_type == "audio_event":
        if from_value and to_value:
            return f"replace {from_value} in the audio with {to_value}"
        if to_value:
            return f"add {to_value} to the audio"
    if audio_dataset_line == "speech_audio_content":
        return ""
    if difference_type == "object_presence" and to_value:
        return f"add {to_value}" if not from_value or from_value.lower() in {"none", "absent", "missing", "no"} else f"replace {from_value} with {to_value}"
    if difference_type == "object_count" and from_value and to_value:
        return f"change the count from {from_value} to {to_value}"
    if difference_type == "scene" and from_value and to_value:
        return f"change the scene from {from_value} to {to_value}"
    if difference_type == "action" and from_value and to_value:
        return f"change the action from {from_value} to {to_value}"
    if difference_type == "attribute" and from_value and to_value:
        return f"change {from_value} to {to_value}"
    return ""


def _single_source_caption_from_annotation(annotation: dict[str, Any], *, fallback: str) -> str:
    summary = str(annotation.get("summary", "")).strip()
    if summary:
        return summary
    scene = str(annotation.get("scene", "")).strip()
    subjects = _detail_list(annotation.get("subjects"))
    actions = _detail_list(annotation.get("actions"))
    parts = subjects[:2] + actions[:2]
    if scene:
        parts.append(scene)
    return ", ".join(parts) if parts else fallback


def _repair_single_source_dominant_delta(value: Any, *, difference: dict[str, Any], evidence: Any) -> dict[str, Any]:
    dominant = dict(value) if isinstance(value, dict) else {}
    reason = str(dominant.get("reason", "")).strip()
    if not reason:
        evidence_items = _detail_list(evidence)
        reason = evidence_items[0] if evidence_items else str(difference.get("description", "")).strip()
    return {
        "type": str(dominant.get("type", difference.get("type", ""))).strip(),
        "from": str(dominant.get("from", difference.get("from", ""))).strip(),
        "to": str(dominant.get("to", difference.get("to", ""))).strip(),
        "reason": reason or "dominant delta inferred from the proposed difference",
    }


def _repair_single_source_state_payload(value: Any, *, annotation: dict[str, Any]) -> dict[str, Any]:
    state = dict(value) if isinstance(value, dict) else {}
    transcript = _detail_list(annotation.get("speakers_and_transcript")) or _detail_list(annotation.get("speech"))
    subjects = _detail_list(annotation.get("subjects"))
    actions = _detail_list(annotation.get("actions"))
    scene = str(annotation.get("scene", "")).strip()
    composition = str(state.get("composition", "")).strip() or ", ".join([item for item in [scene, *subjects[:2], *actions[:2]] if item])
    return {
        "main_speaker": str(state.get("main_speaker", "")).strip() or (subjects[0] if subjects else ""),
        "inset_subjects": _detail_list(state.get("inset_subjects")),
        "product_overlay": str(state.get("product_overlay", "")).strip(),
        "composition": composition or str(annotation.get("summary", "")).strip(),
        "internal_transitions": _detail_list(state.get("internal_transitions")) or transcript[:2],
    }


def _repair_single_source_delta_extent(
    value: Any,
    *,
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    confidence: float,
    accepted: bool,
) -> dict[str, Any]:
    extent = dict(value) if isinstance(value, dict) else {}
    coverage = _score_value(extent.get("target_coverage"))
    if coverage <= 0.0 and accepted:
        coverage = max(0.6, min(0.9, confidence or 0.72))
    evidence = str(extent.get("evidence", "")).strip()
    if not evidence:
        evidence = str(difference.get("description", "")).strip() or _single_source_difference_description(difference)
    return {
        "reference": str(extent.get("reference", "")).strip()
        or _single_source_caption_from_annotation(reference_annotation, fallback="reference clip"),
        "target": str(extent.get("target", "")).strip()
        or _single_source_caption_from_annotation(target_annotation, fallback="target clip"),
        "target_coverage": coverage,
        "evidence": evidence,
    }


def _repair_single_source_subject_roles_payload(value: Any, *, target_state: dict[str, Any]) -> dict[str, Any]:
    roles = dict(value) if isinstance(value, dict) else {}
    return {
        "main_speaker": str(roles.get("main_speaker", target_state.get("main_speaker", ""))).strip(),
        "inset_subjects": _detail_list(roles.get("inset_subjects", target_state.get("inset_subjects", []))),
        "product_overlay": str(roles.get("product_overlay", target_state.get("product_overlay", ""))).strip(),
    }


def _single_source_repair_evidence(
    *,
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    dominant_delta: dict[str, Any],
) -> list[str]:
    evidence: list[str] = []
    reason = str(dominant_delta.get("reason", "")).strip()
    if reason:
        evidence.append(reason)
    description = str(difference.get("description", "")).strip()
    if description and description not in evidence:
        evidence.append(description)
    ref_caption = _single_source_caption_from_annotation(reference_annotation, fallback="")
    tgt_caption = _single_source_caption_from_annotation(target_annotation, fallback="")
    if ref_caption and tgt_caption:
        evidence.append(f"reference: {ref_caption}; target: {tgt_caption}")
    return evidence[:3]


def _normalize_single_source_final_verification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_SINGLE_SOURCE_FINAL_VERIFICATION_FIELDS)
    if missing_fields:
        raise ValueError(f"single-source final verification missing fields: {missing_fields}")

    accept = _bool_value(payload.get("accept"))
    quality_score = _score_value(payload.get("quality_score"))
    if not accept:
        quality_score = min(quality_score, 0.69)
    normalized = {
        "accept": accept,
        "confidence": _score_value(payload.get("confidence")),
        "quality_score": quality_score,
        "reference_satisfies_edit": _bool_value(payload.get("reference_satisfies_edit")),
        "target_satisfies_edit": _bool_value(payload.get("target_satisfies_edit")),
        "observable_delta": _bool_value(payload.get("observable_delta")),
        "single_primary_delta": _bool_value(payload.get("single_primary_delta")),
        "text_or_ocr_driven": _bool_value(payload.get("text_or_ocr_driven")),
        "segment_wide": _bool_value(payload.get("segment_wide")),
        "edit_text_accurate": _bool_value(payload.get("edit_text_accurate")),
        "main_reject_reason": str(payload.get("main_reject_reason", "")).strip(),
        "evidence": _detail_list(payload.get("evidence")),
        "recommended_edit_text": str(payload.get("recommended_edit_text", "")).strip(),
        "schema_repaired_fields": _detail_list(payload.get("schema_repaired_fields")),
    }
    optional_bool_fields = (
        "audio_primary",
        "visual_locked",
        "visual_too_different_for_B",
        "edit_text_audio_only",
        "large_visual_delta",
        "audio_context_preserved",
    )
    for field_name in optional_bool_fields:
        if field_name in payload:
            normalized[field_name] = _bool_value(payload.get(field_name))
    if "visual_locked" not in normalized and "visual_context_sufficient" in payload:
        normalized["visual_locked"] = _bool_value(payload.get("visual_context_sufficient"))
    if normalized["accept"] and not normalized["evidence"]:
        raise ValueError("single-source final verification evidence is required for accept=true")
    if not normalized["accept"] and not normalized["main_reject_reason"]:
        raise ValueError("single-source final verification reject reason is required for accept=false")
    return normalized


def _normalize_b_line_edit_text_refinement_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_B_LINE_EDIT_TEXT_REFINEMENT_FIELDS)
    if missing_fields:
        raise ValueError(f"B-line edit-text refinement missing fields: {missing_fields}")
    refined = str(payload.get("refined_edit_text", "")).strip()
    reject = _bool_value(payload.get("reject_if_unspecific"))
    score = _score_value(payload.get("edit_text_specificity_score"))
    reason = str(payload.get("edit_text_reject_reason", "")).strip()
    evidence = _detail_list(payload.get("speech_or_audio_evidence"))
    if not reject and (not refined or score < 0.70):
        reject = True
        reason = reason or "refined edit_text is not specific enough"
    if reject and not reason:
        reason = "B-line edit_text is not specific enough"
    return {
        "refined_edit_text": refined,
        "edit_text_specificity_score": score,
        "reject_if_unspecific": reject,
        "edit_text_reject_reason": reason,
        "speech_or_audio_evidence": evidence,
    }


def _normalize_b_line_speech_rewrite_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_B_LINE_SPEECH_REWRITE_FIELDS)
    if missing_fields:
        raise ValueError(f"B-line speech rewrite missing fields: {missing_fields}")
    reference_content = str(payload.get("reference_speech_content", "")).strip()
    target_content = str(payload.get("target_speech_content", "")).strip()
    refined = str(payload.get("refined_edit_text", "")).strip()
    reject = _bool_value(payload.get("reject_if_still_unclear"))
    confidence = _score_value(payload.get("speech_transcription_confidence"))
    reason = str(payload.get("speech_rewrite_reject_reason", "")).strip()
    if not reject and (not reference_content or not target_content or not refined or confidence < 0.70):
        reject = True
        reason = reason or "speech rewrite is not specific enough"
    if reject and not reason:
        reason = "speech content is still unclear"
    return {
        "reference_speech_content": reference_content,
        "target_speech_content": target_content,
        "speech_transcription_confidence": confidence,
        "speech_language": str(payload.get("speech_language", "")).strip(),
        "refined_edit_text": refined,
        "reject_if_still_unclear": reject,
        "speech_rewrite_reject_reason": reason,
    }


def _repair_audio_line_single_source_final_verification_payload(
    payload: dict[str, Any],
    *,
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    local_gate_report: dict[str, Any],
    audio_dataset_line: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    repaired = dict(payload)
    repaired_fields = _detail_list(repaired.get("schema_repaired_fields"))
    line = _normalize_audio_dataset_line(audio_dataset_line)
    model_difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(model_difference.get("type", "")).strip()
    model_confidence = _score_value(model_fields.get("confidence"))

    if "accept" not in repaired:
        repaired["accept"] = _final_verification_accept_from_partial_payload(
            repaired,
            model_fields=model_fields,
            audio_dataset_line=line,
        )
        repaired_fields.append("accept")
    accept = _bool_value(repaired.get("accept"))

    if "confidence" not in repaired:
        repaired["confidence"] = max(model_confidence, 0.72) if accept else min(model_confidence, 0.45)
        repaired_fields.append("confidence")
    confidence = _score_value(repaired.get("confidence"))

    if "quality_score" not in repaired:
        repaired["quality_score"] = max(confidence, 0.72) if accept else 0.0
        repaired_fields.append("quality_score")
    quality_score = _score_value(repaired.get("quality_score"))
    if not accept and quality_score >= 0.7:
        repaired["quality_score"] = 0.69
        repaired_fields.append("quality_score")

    bool_defaults = _final_verification_bool_defaults(
        accept=accept,
        model_fields=model_fields,
        local_gate_report=local_gate_report,
        audio_dataset_line=line,
    )
    for field_name in (
        "reference_satisfies_edit",
        "target_satisfies_edit",
        "observable_delta",
        "single_primary_delta",
        "text_or_ocr_driven",
        "segment_wide",
        "edit_text_accurate",
    ):
        if field_name not in repaired:
            repaired[field_name] = bool_defaults[field_name]
            repaired_fields.append(field_name)

    if "recommended_edit_text" not in repaired:
        repaired["recommended_edit_text"] = str(model_fields.get("edit_text", "")).strip() if accept else ""
        repaired_fields.append("recommended_edit_text")
    if "evidence" not in repaired or not _detail_list(repaired.get("evidence")):
        evidence = _final_verification_repair_evidence(
            payload=repaired,
            model_fields=model_fields,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
        if evidence:
            repaired["evidence"] = evidence
            repaired_fields.append("evidence")
        elif "evidence" not in repaired:
            repaired["evidence"] = []
            repaired_fields.append("evidence")

    if "main_reject_reason" not in repaired or (not accept and not str(repaired.get("main_reject_reason", "")).strip()):
        repaired["main_reject_reason"] = "" if accept else _final_verification_reject_reason(
            payload=repaired,
            model_fields=model_fields,
            local_gate_report=local_gate_report,
        )
        repaired_fields.append("main_reject_reason")

    if line == "speech_audio_content":
        optional_defaults = {
            "audio_primary": accept and difference_type in {"speech", "audio_event"},
            "visual_locked": accept and not _local_gate_mentions_visual_too_different(local_gate_report),
            "visual_too_different_for_B": False if accept else _local_gate_mentions_visual_too_different(local_gate_report),
            "edit_text_audio_only": accept and difference_type in {"speech", "audio_event"},
        }
        for field_name, default_value in optional_defaults.items():
            if field_name not in repaired:
                repaired[field_name] = default_value
                repaired_fields.append(field_name)
    elif line == "visual_audio_anchor":
        optional_defaults = {
            "large_visual_delta": accept and difference_type in {"scene", "action", "object_presence", "object_count", "attribute"},
            "audio_context_preserved": accept,
        }
        for field_name, default_value in optional_defaults.items():
            if field_name not in repaired:
                repaired[field_name] = default_value
                repaired_fields.append(field_name)

    if repaired_fields:
        repaired["schema_repaired_fields"] = sorted(set(repaired_fields))
    return repaired


def _final_verification_accept_from_partial_payload(
    payload: dict[str, Any],
    *,
    model_fields: dict[str, Any],
    audio_dataset_line: str,
) -> bool:
    if "quality_score" in payload and _score_value(payload.get("quality_score")) < 0.7:
        return False
    if "confidence" in payload and _score_value(payload.get("confidence")) < 0.45:
        return False
    if "target_satisfies_edit" in payload and not _bool_value(payload.get("target_satisfies_edit")):
        return False
    if "reference_satisfies_edit" in payload and _bool_value(payload.get("reference_satisfies_edit")):
        return False
    if "observable_delta" in payload and not _bool_value(payload.get("observable_delta")):
        return False
    if "edit_text_accurate" in payload and not _bool_value(payload.get("edit_text_accurate")):
        return False
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    line = _normalize_audio_dataset_line(audio_dataset_line)
    if line == "speech_audio_content" and difference_type not in {"speech", "audio_event"}:
        return False
    if line == "visual_audio_anchor" and difference_type not in {"scene", "action", "object_presence", "object_count", "attribute"}:
        return False
    return bool(model_fields.get("accept"))


def _final_verification_bool_defaults(
    *,
    accept: bool,
    model_fields: dict[str, Any],
    local_gate_report: dict[str, Any],
    audio_dataset_line: str,
) -> dict[str, bool]:
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    local_issues = _detail_list(local_gate_report.get("all_issues")) + _detail_list(local_gate_report.get("hard_reject"))
    text_or_ocr = any("ocr" in str(issue).lower() or "text" in str(issue).lower() for issue in local_issues)
    segment_wide = bool(model_fields.get("is_segment_wide_delta")) if "is_segment_wide_delta" in model_fields else accept
    if audio_dataset_line == "speech_audio_content" and difference_type in {"speech", "audio_event"}:
        text_or_ocr = False
    return {
        "reference_satisfies_edit": False if accept else _bool_value(model_fields.get("reference_satisfies_edit")),
        "target_satisfies_edit": accept,
        "observable_delta": accept,
        "single_primary_delta": accept,
        "text_or_ocr_driven": text_or_ocr,
        "segment_wide": bool(segment_wide),
        "edit_text_accurate": accept,
    }


def _final_verification_repair_evidence(
    *,
    payload: dict[str, Any],
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> list[str]:
    evidence: list[str] = []
    evidence.extend(_detail_list(payload.get("evidence")))
    evidence.extend(_detail_list(model_fields.get("evidence")))
    dominant_delta = model_fields.get("dominant_delta") if isinstance(model_fields.get("dominant_delta"), dict) else {}
    reason = str(dominant_delta.get("reason", "")).strip()
    if reason:
        evidence.append(reason)
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    description = str(difference.get("description", "")).strip()
    if description:
        evidence.append(description)
    if not evidence:
        ref_caption = _single_source_caption_from_annotation(reference_annotation, fallback="")
        tgt_caption = _single_source_caption_from_annotation(target_annotation, fallback="")
        if ref_caption and tgt_caption:
            evidence.append(f"reference: {ref_caption}; target: {tgt_caption}")
    seen: set[str] = set()
    compact: list[str] = []
    for item in evidence:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        compact.append(text)
        if len(compact) >= 4:
            break
    return compact


def _final_verification_reject_reason(
    *,
    payload: dict[str, Any],
    model_fields: dict[str, Any],
    local_gate_report: dict[str, Any],
) -> str:
    for field_name in ("main_reject_reason", "reject_reason", "reason"):
        reason = str(payload.get(field_name, "")).strip()
        if reason:
            return reason
    issues = _detail_list(local_gate_report.get("all_issues")) or _detail_list(local_gate_report.get("hard_reject"))
    if issues:
        return "; ".join(str(issue) for issue in issues[:4])
    reason = str(model_fields.get("reject_reason", "")).strip()
    if reason:
        return reason
    return "final verifier did not accept the pair"


def _local_gate_mentions_visual_too_different(local_gate_report: dict[str, Any]) -> bool:
    issues = _detail_list(local_gate_report.get("all_issues")) + _detail_list(local_gate_report.get("hard_reject"))
    return any("visual_too_different_for_b" in str(issue).lower() for issue in issues)


def _normalize_single_source_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("single-source pair state fields must be objects")
    return {
        "main_speaker": str(value.get("main_speaker", "")).strip(),
        "inset_subjects": _detail_list(value.get("inset_subjects")),
        "product_overlay": str(value.get("product_overlay", "")).strip(),
        "composition": str(value.get("composition", "")).strip(),
        "internal_transitions": _detail_list(value.get("internal_transitions")),
    }


def _normalize_delta_temporal_extent(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("single-source pair delta_temporal_extent must be an object")
    return {
        "reference": str(value.get("reference", "")).strip(),
        "target": str(value.get("target", "")).strip(),
        "target_coverage": _score_value(value.get("target_coverage")),
        "evidence": str(value.get("evidence", "")).strip(),
    }


def _normalize_single_source_subject_roles(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("single-source pair subject_roles must be an object")
    return {
        "main_speaker": str(value.get("main_speaker", "")).strip(),
        "inset_subjects": _detail_list(value.get("inset_subjects")),
        "product_overlay": str(value.get("product_overlay", "")).strip(),
    }


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


def _normalize_audio_anchor_visual_verification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_AUDIO_ANCHOR_VISUAL_VERIFICATION_FIELDS)
    if missing_fields:
        raise ValueError(f"audio-anchor visual verification missing fields: {missing_fields}")

    visual_delta_type = str(payload.get("visual_delta_type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    visual_delta_type = DIFFERENCE_TYPE_ALIASES.get(visual_delta_type, visual_delta_type)
    if visual_delta_type not in ALLOWED_DIFFERENCE_TYPES:
        visual_delta_type = ""
    accept = _bool_value(payload.get("accept"))
    normalized = {
        "accept": accept,
        "reject_reason": str(payload.get("reject_reason", "")).strip(),
        "recommended_edit_text": str(payload.get("recommended_edit_text", "")).strip(),
        "visual_delta_type": visual_delta_type,
        "visual_delta_strength": _score_value(payload.get("visual_delta_strength")),
        "near_duplicate_risk": _score_value(payload.get("near_duplicate_risk")),
        "reference_satisfies_edit": _bool_value(payload.get("reference_satisfies_edit")),
        "target_satisfies_edit": _bool_value(payload.get("target_satisfies_edit")),
        "caption_equivalent": _bool_value(payload.get("caption_equivalent")),
        "order_only_scene_reorder": _bool_value(payload.get("order_only_scene_reorder")),
        "weak_synonym_or_wording_delta": _bool_value(payload.get("weak_synonym_or_wording_delta")),
        "evidence": _detail_list(payload.get("evidence")),
    }
    if normalized["accept"] and not normalized["evidence"]:
        raise ValueError("audio-anchor visual verification evidence is required for accept=true")
    if not normalized["accept"] and not normalized["reject_reason"]:
        raise ValueError("audio-anchor visual verification reject_reason is required for accept=false")
    return normalized


def _repair_video_edit_plan_payload(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    reference_annotation: dict[str, Any] | None = None,
    route_hint: str,
) -> dict[str, Any]:
    repaired = dict(payload)
    repaired_fields: list[str] = _string_list(repaired.get("repaired_fields"))
    source_prompt = str(repaired.get("source_prompt", "")).strip()
    edit_text = str(candidate.get("edit_text", "")).strip()
    difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
    edit_token = str(repaired.get("edit_token", "")).strip()
    if not edit_token:
        for field_name in ("to", "description", "from"):
            value = str(difference.get(field_name, "")).strip()
            if value and value.lower() not in {"none", "missing", "absent"} and not value.lower().startswith("no "):
                edit_token = value
                repaired["edit_token"] = edit_token
                repaired_fields.append("edit_token")
                break
    preserve_tokens = _string_list(repaired.get("preserve_tokens"))
    if not preserve_tokens:
        preserve_tokens = _infer_video_edit_preserve_tokens(
            reference_annotation=reference_annotation or {},
            payload=repaired,
            difference=difference,
            edit_token=edit_token,
        )
        if preserve_tokens:
            repaired["preserve_tokens"] = preserve_tokens
            repaired_fields.append("preserve_tokens")
    if source_prompt and not str(repaired.get("target_prompt", "")).strip():
        edit_instruction = edit_text or (f"add or change {edit_token}" if edit_token else "apply the requested edit")
        repaired["target_prompt"] = (
            f"{source_prompt.rstrip('.')} Apply exactly one localized edit: {edit_instruction}. "
            "Preserve all other visible content, camera motion, lighting, timing, and layout."
        )
        repaired_fields.append("target_prompt")
    if not str(repaired.get("negative_prompt", "")).strip() and preserve_tokens:
        protected = ", ".join(preserve_tokens[:6])
        repaired["negative_prompt"] = (
            f"Do not change {protected}. Do not change people, scene, camera, lighting, visible text, "
            "timing, or unrelated objects."
        )
        repaired_fields.append("negative_prompt")
    if route_hint and not str(repaired.get("model_route", "")).strip():
        repaired["model_route"] = route_hint
        repaired_fields.append("model_route")
    if not str(repaired.get("edit_region", "")).strip():
        edit_region = _infer_video_edit_region(
            payload=repaired,
            candidate=candidate,
            difference=difference,
            edit_token=edit_token,
        )
        if edit_region:
            repaired["edit_region"] = edit_region
            repaired_fields.append("edit_region")
    if not str(repaired.get("reason", "")).strip():
        repaired["reason"] = "Planner response was repaired conservatively from the candidate edit."
        repaired_fields.append("reason")
    if repaired_fields:
        repaired["repaired_fields"] = sorted(set(repaired_fields))
    return repaired


def _infer_video_edit_preserve_tokens(
    *,
    reference_annotation: dict[str, Any],
    payload: dict[str, Any],
    difference: dict[str, Any],
    edit_token: str,
) -> list[str]:
    values: list[str] = []
    if isinstance(reference_annotation, dict):
        values.extend(_string_list(reference_annotation.get("subjects")))
        values.extend(_normalize_object_counts(reference_annotation.get("object_counts")).keys())
        values.extend(_string_list(reference_annotation.get("actions")))
        scene = str(reference_annotation.get("scene", "")).strip()
        if scene:
            values.append(scene)
        values.extend(_string_list(reference_annotation.get("on_screen_text")))

    source_prompt = str(payload.get("source_prompt", "")).strip()
    if source_prompt:
        for phrase in (
            "camera motion",
            "camera angle",
            "lighting",
            "timing",
            "layout",
            "background",
            "foreground",
        ):
            if phrase in source_prompt.lower():
                values.append(phrase)

    values.extend(["camera motion", "lighting", "timing", "layout"])
    edit_key = _phrase_key(edit_token)
    source_key = _phrase_key(str(difference.get("from", "")).strip())
    target_key = _phrase_key(str(difference.get("to", "")).strip())
    preserve_tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = _phrase_key(item)
        if not item or not key or key == edit_key or key == source_key or key == target_key or key in seen:
            continue
        seen.add(key)
        preserve_tokens.append(item)
        if len(preserve_tokens) >= 8:
            break
    if not preserve_tokens:
        preserve_tokens = ["original subject", "scene", "camera motion", "lighting", "timing", "layout"]
    return preserve_tokens


def _phrase_key(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def _infer_video_edit_region(
    *,
    payload: dict[str, Any],
    candidate: dict[str, Any],
    difference: dict[str, Any],
    edit_token: str,
) -> str:
    text_parts = [
        candidate.get("edit_text", ""),
        payload.get("target_prompt", ""),
        payload.get("source_prompt", ""),
        difference.get("description", ""),
        difference.get("to", ""),
        difference.get("from", ""),
    ]
    text = " ".join(str(part) for part in text_parts if part).lower()
    region_patterns = (
        (r"\btop[- ]right\b|\bupper[- ]right\b", "top-right region"),
        (r"\btop[- ]left\b|\bupper[- ]left\b", "top-left region"),
        (r"\bbottom[- ]right\b|\blower[- ]right\b", "bottom-right region"),
        (r"\bbottom[- ]left\b|\blower[- ]left\b", "bottom-left region"),
        (r"\bbackground\b|\bbackdrop\b|\bin the back\b", "background"),
        (r"\bforeground\b|\bfront\b", "foreground"),
        (r"\bwall\b|\bposter\b|\bpainting\b|\bframed picture\b|\bwall art\b", "wall area"),
        (r"\bpaper\b|\bpage\b|\bnotebook\b|\bworksheet\b", "paper surface"),
        (r"\bdesk\b|\btable\b|\bcounter\b|\bsurface\b", "desk/table surface"),
        (r"\bfloor\b|\bground\b", "floor area"),
        (r"\bhand[- ]held\b|\bin (?:the )?hand\b|\bholding\b|\bheld\b", "hand-held object"),
        (r"\bcenter\b|\bmiddle\b", "center region"),
        (r"\bleft side\b|\bon the left\b", "left side"),
        (r"\bright side\b|\bon the right\b", "right side"),
    )
    for pattern, region in region_patterns:
        if re.search(pattern, text):
            return region
    if edit_token:
        return f"localized region around {edit_token}"
    return ""


def _normalize_video_edit_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = _missing_fields(payload, REQUIRED_VIDEO_EDIT_PLAN_FIELDS)
    if missing_fields:
        raise ValueError(f"video edit plan missing fields: {missing_fields}")

    preserve_tokens = _string_list(payload.get("preserve_tokens"))
    preserve_regions = _string_list(payload.get("preserve_regions"))
    normalized = {
        "should_generate": _bool_value(payload.get("should_generate")),
        "edit_text": str(payload.get("edit_text", "")).strip(),
        "difference": payload.get("difference") if isinstance(payload.get("difference"), dict) else {},
        "source_prompt": str(payload.get("source_prompt", "")).strip(),
        "target_prompt": str(payload.get("target_prompt", "")).strip(),
        "edit_token": str(payload.get("edit_token", "")).strip(),
        "preserve_tokens": preserve_tokens,
        "negative_prompt": str(payload.get("negative_prompt", "")).strip(),
        "edit_region": str(payload.get("edit_region", "")).strip(),
        "mask_query": str(payload.get("mask_query", "")).strip(),
        "preserve_regions": preserve_regions,
        "model_route": str(payload.get("model_route", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
        "repaired_fields": _string_list(payload.get("repaired_fields")),
    }
    for field_name in ("source_prompt", "target_prompt", "edit_token", "negative_prompt", "edit_region", "reason"):
        if not normalized[field_name]:
            raise ValueError(f"video edit plan {field_name} is required")
    if not normalized["preserve_tokens"]:
        raise ValueError("video edit plan preserve_tokens is required")
    return normalized


def _normalize_stable_clip_selection_payload(
    payload: dict[str, Any],
    *,
    min_clip_seconds: float,
    max_clip_seconds: float,
    duration_seconds: float,
) -> dict[str, Any]:
    start = _score_or_zero(payload.get("start_sec"))
    try:
        end = float(payload.get("end_sec"))
    except (TypeError, ValueError):
        end = start + min_clip_seconds
    start = max(0.0, min(start, max(0.0, duration_seconds)))
    end = max(start, min(end, duration_seconds if duration_seconds > 0 else end))
    if end - start < min_clip_seconds:
        end = min(duration_seconds if duration_seconds > 0 else start + min_clip_seconds, start + min_clip_seconds)
    if end - start > max_clip_seconds:
        end = start + max_clip_seconds
    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "stability_score": _score_value(payload.get("stability_score")),
        "camera_motion": str(payload.get("camera_motion", "unknown")).strip() or "unknown",
        "main_subjects": _string_list(payload.get("main_subjects")),
        "visible_text_risk": _bool_value(payload.get("visible_text_risk")),
        "recommended_for_vace": _bool_value(payload.get("recommended_for_vace", True)),
        "reason": str(payload.get("reason", "")).strip(),
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
