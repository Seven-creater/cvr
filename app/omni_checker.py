from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request

from app.retrieval_types import TextRow, VideoRow

VALID_AUDIO_RELEVANCE = {"required", "helpful", "irrelevant", "unknown"}

REQUIRED_T2V_QUERY_FIELDS = (
    "retrieval_text",
    "summary",
    "main_events",
    "objects",
    "scene",
    "audio_cues",
    "audio_relevance",
    "reason",
)

REQUIRED_VIDEO_DESCRIPTION_FIELDS = (
    "summary",
    "main_events",
    "objects",
    "scene",
    "audio_cues",
    "audio_relevance",
)

REQUIRED_T2V_RERANK_FIELDS = (
    "ordered_video_ids",
    "top_choice_video_id",
    "confidence",
    "reason",
)

REQUIRED_V2T_RERANK_FIELDS = (
    "ordered_text_ids",
    "top_choice_text_id",
    "confidence",
    "reason",
)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_audio_relevance(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in VALID_AUDIO_RELEVANCE:
        return normalized
    return "unknown"


def _extract_json(text: str) -> dict:
    text = str(text).strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _missing_fields(payload: dict, required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if field not in payload]


def _file_path_from_url(raw_url: str) -> Path | None:
    if raw_url.startswith("file://"):
        parsed = urllib.parse.urlparse(raw_url)
        return Path(urllib.request.url2pathname(parsed.path))
    if raw_url.startswith(("http://", "https://", "data:")):
        return None
    return Path(raw_url)


def _materialize_video_url(raw_url: str) -> str:
    file_path = _file_path_from_url(raw_url)
    if file_path is None:
        return raw_url
    if not file_path.exists():
        raise FileNotFoundError(f"video file not found: {file_path}")
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "video/mp4"
    content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{content}"


@dataclass(frozen=True, slots=True)
class RetrievalHints:
    query_text_override: str | None
    audio_mode: str = "on"
    fallback_used: bool = False

    @classmethod
    def from_query_understanding(
        cls,
        query_text: str,
        understanding: "T2VQueryUnderstanding",
    ) -> "RetrievalHints":
        raw_override = understanding.retrieval_text.strip()
        query_text_override = raw_override if raw_override and raw_override != query_text else None
        audio_mode = "off" if not understanding.fallback_used and understanding.audio_relevance == "irrelevant" else "on"
        return cls(
            query_text_override=query_text_override,
            audio_mode=audio_mode,
            fallback_used=understanding.fallback_used,
        )

    @classmethod
    def from_video_description(cls, description: "VideoDescription") -> "RetrievalHints":
        audio_mode = "off" if not description.fallback_used and description.audio_relevance == "irrelevant" else "on"
        return cls(query_text_override=None, audio_mode=audio_mode, fallback_used=description.fallback_used)

    def to_dict(self) -> dict:
        payload = {
            "audio_mode": self.audio_mode,
            "fallback_used": self.fallback_used,
        }
        if self.query_text_override is not None:
            payload["query_text_override"] = self.query_text_override
        return payload


@dataclass(frozen=True, slots=True)
class T2VQueryUnderstanding:
    retrieval_text: str
    summary: str
    main_events: list[str]
    objects: list[str]
    scene: str
    audio_cues: list[str]
    audio_relevance: str
    reason: str
    fallback_used: bool = False

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        original_query_text: str,
    ) -> "T2VQueryUnderstanding":
        retrieval_text = str(payload.get("retrieval_text") or original_query_text).strip() or original_query_text
        return cls(
            retrieval_text=retrieval_text,
            summary=str(payload.get("summary", "")).strip(),
            main_events=_string_list(payload.get("main_events")),
            objects=_string_list(payload.get("objects")),
            scene=str(payload.get("scene", "")).strip(),
            audio_cues=_string_list(payload.get("audio_cues")),
            audio_relevance=_normalize_audio_relevance(payload.get("audio_relevance")),
            reason=str(payload.get("reason", "")).strip(),
            fallback_used=bool(payload.get("fallback_used", False)),
        )

    def to_dict(self) -> dict:
        return {
            "retrieval_text": self.retrieval_text,
            "summary": self.summary,
            "main_events": list(self.main_events),
            "objects": list(self.objects),
            "scene": self.scene,
            "audio_cues": list(self.audio_cues),
            "audio_relevance": self.audio_relevance,
            "reason": self.reason,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True, slots=True)
class VideoDescription:
    summary: str
    main_events: list[str]
    objects: list[str]
    scene: str
    audio_cues: list[str]
    audio_relevance: str
    fallback_used: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "VideoDescription":
        return cls(
            summary=str(payload.get("summary", "")).strip(),
            main_events=_string_list(payload.get("main_events")),
            objects=_string_list(payload.get("objects")),
            scene=str(payload.get("scene", "")).strip(),
            audio_cues=_string_list(payload.get("audio_cues")),
            audio_relevance=_normalize_audio_relevance(payload.get("audio_relevance")),
            fallback_used=bool(payload.get("fallback_used", False)),
        )

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "main_events": list(self.main_events),
            "objects": list(self.objects),
            "scene": self.scene,
            "audio_cues": list(self.audio_cues),
            "audio_relevance": self.audio_relevance,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True, slots=True)
class T2VRerankResult:
    ordered_video_ids: list[str]
    top_choice_video_id: str
    confidence: float
    reason: str
    fallback_used: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "T2VRerankResult":
        ordered_video_ids = _string_list(payload.get("ordered_video_ids"))
        top_choice = str(payload.get("top_choice_video_id", "")).strip()
        if not top_choice and ordered_video_ids:
            top_choice = ordered_video_ids[0]
        return cls(
            ordered_video_ids=ordered_video_ids,
            top_choice_video_id=top_choice,
            confidence=_as_float(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")).strip(),
            fallback_used=bool(payload.get("fallback_used", False)),
        )

    def to_dict(self) -> dict:
        return {
            "ordered_video_ids": list(self.ordered_video_ids),
            "top_choice_video_id": self.top_choice_video_id,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True, slots=True)
class V2TRerankResult:
    ordered_text_ids: list[str]
    top_choice_text_id: str
    confidence: float
    reason: str
    fallback_used: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "V2TRerankResult":
        ordered_text_ids = _string_list(payload.get("ordered_text_ids"))
        top_choice = str(payload.get("top_choice_text_id", "")).strip()
        if not top_choice and ordered_text_ids:
            top_choice = ordered_text_ids[0]
        return cls(
            ordered_text_ids=ordered_text_ids,
            top_choice_text_id=top_choice,
            confidence=_as_float(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")).strip(),
            fallback_used=bool(payload.get("fallback_used", False)),
        )

    def to_dict(self) -> dict:
        return {
            "ordered_text_ids": list(self.ordered_text_ids),
            "top_choice_text_id": self.top_choice_text_id,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "fallback_used": self.fallback_used,
        }


class OmniChecker(Protocol):
    def understand_t2v_query(self, query_text: str) -> T2VQueryUnderstanding:
        ...

    def describe_video(self, video: VideoRow) -> VideoDescription:
        ...

    def rerank_t2v(self, query_understanding: T2VQueryUnderstanding, candidates: list[dict]) -> T2VRerankResult:
        ...

    def rerank_v2t(
        self,
        query_video: VideoRow,
        video_description: VideoDescription,
        candidates: list[dict],
    ) -> V2TRerankResult:
        ...


def build_t2v_query_understanding_user_content(query_text: str) -> list[dict]:
    prompt = (
        "Task: understand a text-to-video retrieval query.\n"
        f"Original query: {query_text}\n"
        "Return a retrieval-focused rewrite and structured cues. "
        "Do not mention dataset labels."
    )
    return [{"type": "text", "text": prompt}]


def build_video_description_user_content(video: VideoRow) -> list[dict]:
    if not video.video_path:
        raise ValueError("video.video_path is required")
    prompt = (
        "Task: describe this video for retrieval reranking.\n"
        "Summarize the visible actions, important objects, scene, and useful audio cues. "
        "Do not mention dataset labels."
    )
    return [
        {"type": "video_url", "video_url": {"url": video.video_path}},
        {"type": "text", "text": prompt},
    ]


def build_t2v_rerank_user_content(query_understanding: T2VQueryUnderstanding, candidates: list[dict]) -> list[dict]:
    prompt = (
        "Task: rerank candidate videos for a text-to-video query.\n"
        f"Query understanding JSON:\n{json.dumps(query_understanding.to_dict(), ensure_ascii=False)}\n"
        f"Candidate videos JSON:\n{json.dumps(candidates, ensure_ascii=False)}\n"
        "Rank only the provided candidate video_ids. Keep the full candidate set."
    )
    return [{"type": "text", "text": prompt}]


def build_v2t_rerank_user_content(video_description: VideoDescription, candidates: list[dict]) -> list[dict]:
    prompt = (
        "Task: rerank candidate texts for a video-to-text query.\n"
        f"Video description JSON:\n{json.dumps(video_description.to_dict(), ensure_ascii=False)}\n"
        f"Candidate texts JSON:\n{json.dumps(candidates, ensure_ascii=False)}\n"
        "Rank only the provided candidate text_ids. Keep the full candidate set."
    )
    return [{"type": "text", "text": prompt}]


def _t2v_query_system_prompt() -> str:
    return (
        "You are a retrieval query understanding assistant. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"retrieval_text": string, "summary": string, "main_events": [string], '
        '"objects": [string], "scene": string, "audio_cues": [string], '
        '"audio_relevance": "required|helpful|irrelevant|unknown", "reason": string}. '
        "All keys are mandatory."
    )


def _video_description_system_prompt() -> str:
    return (
        "You are a video description assistant for retrieval reranking. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"summary": string, "main_events": [string], "objects": [string], '
        '"scene": string, "audio_cues": [string], '
        '"audio_relevance": "required|helpful|irrelevant|unknown"}. '
        "All keys are mandatory."
    )


def _t2v_rerank_system_prompt() -> str:
    return (
        "You are a text-to-video reranker. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"ordered_video_ids": [string], "top_choice_video_id": string, '
        '"confidence": float, "reason": string}. '
        "Use only candidate video_ids that appear in the input, include every candidate exactly once, and do not invent ids."
    )


def _v2t_rerank_system_prompt() -> str:
    return (
        "You are a video-to-text reranker. "
        "Return exactly one JSON object and nothing else. "
        'Required schema: {"ordered_text_ids": [string], "top_choice_text_id": string, '
        '"confidence": float, "reason": string}. '
        "Use only candidate text_ids that appear in the input, include every candidate exactly once, and do not invent ids."
    )


def _fallback_query_understanding(query_text: str, *, reason: str) -> T2VQueryUnderstanding:
    return T2VQueryUnderstanding(
        retrieval_text=query_text,
        summary=query_text,
        main_events=[],
        objects=[],
        scene="",
        audio_cues=[],
        audio_relevance="unknown",
        reason=reason,
        fallback_used=True,
    )


def _fallback_video_description() -> VideoDescription:
    return VideoDescription(
        summary="",
        main_events=[],
        objects=[],
        scene="",
        audio_cues=[],
        audio_relevance="unknown",
        fallback_used=True,
    )


def _fallback_t2v_rerank(candidate_ids: list[str], *, reason: str) -> T2VRerankResult:
    return T2VRerankResult(
        ordered_video_ids=list(candidate_ids),
        top_choice_video_id=candidate_ids[0] if candidate_ids else "",
        confidence=0.0,
        reason=reason,
        fallback_used=True,
    )


def _fallback_v2t_rerank(candidate_ids: list[str], *, reason: str) -> V2TRerankResult:
    return V2TRerankResult(
        ordered_text_ids=list(candidate_ids),
        top_choice_text_id=candidate_ids[0] if candidate_ids else "",
        confidence=0.0,
        reason=reason,
        fallback_used=True,
    )


class OpenAIOmniChecker:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _request_payload(self, user_content: list[dict], system_prompt: str, *, max_tokens: int = 512) -> dict:
        request_content: list[dict] = []
        for item in user_content:
            if item.get("type") == "video_url":
                video_url = dict(item["video_url"])
                video_url["url"] = _materialize_video_url(str(video_url["url"]))
                request_content.append({"type": "video_url", "video_url": video_url})
            else:
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
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on live service
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"omni checker request failed: {detail}") from exc
        content = raw["choices"][0]["message"]["content"]
        return _extract_json(content)

    def understand_t2v_query(self, query_text: str) -> T2VQueryUnderstanding:
        try:
            payload = self._request_payload(
                build_t2v_query_understanding_user_content(query_text),
                _t2v_query_system_prompt(),
            )
        except Exception as exc:
            return _fallback_query_understanding(query_text, reason=f"query_understanding_failed:{type(exc).__name__}")
        if _missing_fields(payload, REQUIRED_T2V_QUERY_FIELDS):
            return _fallback_query_understanding(query_text, reason="query_understanding_incomplete")
        return T2VQueryUnderstanding.from_dict(payload, original_query_text=query_text)

    def describe_video(self, video: VideoRow) -> VideoDescription:
        try:
            payload = self._request_payload(
                build_video_description_user_content(video),
                _video_description_system_prompt(),
            )
        except Exception:
            return _fallback_video_description()
        if _missing_fields(payload, REQUIRED_VIDEO_DESCRIPTION_FIELDS):
            return _fallback_video_description()
        return VideoDescription.from_dict(payload)

    def rerank_t2v(self, query_understanding: T2VQueryUnderstanding, candidates: list[dict]) -> T2VRerankResult:
        candidate_ids = [str(item.get("video_id", "")).strip() for item in candidates if str(item.get("video_id", "")).strip()]
        try:
            payload = self._request_payload(
                build_t2v_rerank_user_content(query_understanding, candidates),
                _t2v_rerank_system_prompt(),
            )
        except Exception as exc:
            return _fallback_t2v_rerank(candidate_ids, reason=f"t2v_rerank_failed:{type(exc).__name__}")
        if _missing_fields(payload, REQUIRED_T2V_RERANK_FIELDS):
            return _fallback_t2v_rerank(candidate_ids, reason="t2v_rerank_incomplete")
        return T2VRerankResult.from_dict(payload)

    def rerank_v2t(
        self,
        query_video: VideoRow,
        video_description: VideoDescription,
        candidates: list[dict],
    ) -> V2TRerankResult:
        _ = query_video
        candidate_ids = [str(item.get("text_id", "")).strip() for item in candidates if str(item.get("text_id", "")).strip()]
        try:
            payload = self._request_payload(
                build_v2t_rerank_user_content(video_description, candidates),
                _v2t_rerank_system_prompt(),
            )
        except Exception as exc:
            return _fallback_v2t_rerank(candidate_ids, reason=f"v2t_rerank_failed:{type(exc).__name__}")
        if _missing_fields(payload, REQUIRED_V2T_RERANK_FIELDS):
            return _fallback_v2t_rerank(candidate_ids, reason="v2t_rerank_incomplete")
        return V2TRerankResult.from_dict(payload)


class MockOmniChecker:
    def __init__(
        self,
        *,
        t2v_understanding_results: dict[str, T2VQueryUnderstanding | dict] | None = None,
        video_description_results: dict[str, VideoDescription | dict] | None = None,
        t2v_rerank_results: dict[str, T2VRerankResult | dict] | None = None,
        v2t_rerank_results: dict[str, V2TRerankResult | dict] | None = None,
    ) -> None:
        self.t2v_understanding_results = dict(t2v_understanding_results or {})
        self.video_description_results = dict(video_description_results or {})
        self.t2v_rerank_results = dict(t2v_rerank_results or {})
        self.v2t_rerank_results = dict(v2t_rerank_results or {})

    def understand_t2v_query(self, query_text: str) -> T2VQueryUnderstanding:
        resolved = self.t2v_understanding_results.get(query_text)
        if isinstance(resolved, T2VQueryUnderstanding):
            return resolved
        if isinstance(resolved, dict):
            return T2VQueryUnderstanding.from_dict(resolved, original_query_text=query_text)
        return T2VQueryUnderstanding(
            retrieval_text=query_text,
            summary=query_text,
            main_events=[],
            objects=[],
            scene="",
            audio_cues=[],
            audio_relevance="unknown",
            reason="mock default",
        )

    def describe_video(self, video: VideoRow) -> VideoDescription:
        resolved = self.video_description_results.get(video.video_id)
        if isinstance(resolved, VideoDescription):
            return resolved
        if isinstance(resolved, dict):
            return VideoDescription.from_dict(resolved)
        return VideoDescription(
            summary=f"summary for {video.video_id}",
            main_events=[],
            objects=[],
            scene="",
            audio_cues=[],
            audio_relevance="unknown",
        )

    def rerank_t2v(self, query_understanding: T2VQueryUnderstanding, candidates: list[dict]) -> T2VRerankResult:
        resolved = self.t2v_rerank_results.get(query_understanding.retrieval_text)
        if isinstance(resolved, T2VRerankResult):
            return resolved
        if isinstance(resolved, dict):
            return T2VRerankResult.from_dict(resolved)
        candidate_ids = [str(item.get("video_id", "")).strip() for item in candidates if str(item.get("video_id", "")).strip()]
        return T2VRerankResult(
            ordered_video_ids=candidate_ids,
            top_choice_video_id=candidate_ids[0] if candidate_ids else "",
            confidence=0.0,
            reason="mock default",
        )

    def rerank_v2t(
        self,
        query_video: VideoRow,
        video_description: VideoDescription,
        candidates: list[dict],
    ) -> V2TRerankResult:
        _ = video_description
        resolved = self.v2t_rerank_results.get(query_video.video_id)
        if isinstance(resolved, V2TRerankResult):
            return resolved
        if isinstance(resolved, dict):
            return V2TRerankResult.from_dict(resolved)
        candidate_ids = [str(item.get("text_id", "")).strip() for item in candidates if str(item.get("text_id", "")).strip()]
        return V2TRerankResult(
            ordered_text_ids=candidate_ids,
            top_choice_text_id=candidate_ids[0] if candidate_ids else "",
            confidence=0.0,
            reason="mock default",
        )
