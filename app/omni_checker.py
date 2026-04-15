from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from app.retrieval_types import TextRow, VideoRow

REQUIRED_CHECKER_FIELDS = (
    "is_match",
    "confidence",
    "visual_match",
    "audio_match",
    "main_events",
    "missing_elements",
    "reason",
    "rewrite_suggestion",
)


@dataclass(frozen=True, slots=True)
class CheckerResult:
    is_match: bool
    confidence: float
    visual_match: float
    audio_match: float
    main_events: list[str]
    missing_elements: list[str]
    reason: str
    rewrite_suggestion: str

    @classmethod
    def from_dict(cls, payload: dict) -> "CheckerResult":
        def _as_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "match"}
            return bool(value)

        def _as_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return cls(
            is_match=_as_bool(payload.get("is_match", False)),
            confidence=_as_float(payload.get("confidence", 0.0)),
            visual_match=_as_float(payload.get("visual_match", 0.0)),
            audio_match=_as_float(payload.get("audio_match", 0.0)),
            main_events=[str(item) for item in payload.get("main_events", [])],
            missing_elements=[str(item) for item in payload.get("missing_elements", [])],
            reason=str(payload.get("reason", "")),
            rewrite_suggestion=str(payload.get("rewrite_suggestion", "")),
        )

    def to_dict(self) -> dict:
        return {
            "is_match": self.is_match,
            "confidence": round(self.confidence, 4),
            "visual_match": round(self.visual_match, 4),
            "audio_match": round(self.audio_match, 4),
            "main_events": list(self.main_events),
            "missing_elements": list(self.missing_elements),
            "reason": self.reason,
            "rewrite_suggestion": self.rewrite_suggestion,
        }


class OmniChecker(Protocol):
    def inspect_t2v(self, query_text: str, candidate_video: VideoRow, *, rank: int, score: float) -> CheckerResult:
        ...

    def inspect_v2t(self, query_video: VideoRow, candidate_text: TextRow, *, rank: int, score: float) -> CheckerResult:
        ...


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _fallback_payload(raw_text: str) -> dict:
    return {
        "is_match": False,
        "confidence": 0.0,
        "visual_match": 0.0,
        "audio_match": 0.0,
        "main_events": [],
        "missing_elements": ["unstructured_response"],
        "reason": raw_text.strip(),
        "rewrite_suggestion": "",
    }


def _complete_payload(payload: dict) -> dict:
    completed = {
        "is_match": payload.get("is_match", False),
        "confidence": payload.get("confidence", 0.0),
        "visual_match": payload.get("visual_match", 0.0),
        "audio_match": payload.get("audio_match", 0.0),
        "main_events": payload.get("main_events", []),
        "missing_elements": payload.get("missing_elements", []),
        "reason": payload.get("reason", ""),
        "rewrite_suggestion": payload.get("rewrite_suggestion", ""),
    }
    missing_fields = [field for field in REQUIRED_CHECKER_FIELDS if field not in payload]
    if missing_fields:
        missing_items = [str(item) for item in completed["missing_elements"]]
        missing_items.extend([f"missing_field:{field}" for field in missing_fields])
        completed["missing_elements"] = missing_items
        if not completed["reason"]:
            completed["reason"] = "incomplete_json_response"
    return completed


def _checker_system_prompt(kind: str) -> str:
    subject = "video retrieval" if kind == "t2v" else "text retrieval"
    return (
        f"You are a {subject} checker. "
        "Return exactly one JSON object and nothing else. "
        "All 8 keys are mandatory. "
        'Required schema: {"is_match": bool, "confidence": float, "visual_match": float, '
        '"audio_match": float, "main_events": [string], "missing_elements": [string], '
        '"reason": string, "rewrite_suggestion": string}. '
        "The float fields must be numeric values between 0.0 and 1.0. "
        'A reply like {"is_match": true} is invalid because keys are missing. '
        'Example valid reply: {"is_match": true, "confidence": 0.82, "visual_match": 0.91, '
        '"audio_match": 0.35, "main_events": ["person pours milk", "person cuts vanilla bean"], '
        '"missing_elements": [], "reason": "The visible cooking actions match the query.", '
        '"rewrite_suggestion": ""}.'
    )


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


def build_t2v_user_content(query_text: str, candidate_video: VideoRow, *, rank: int, score: float) -> list[dict]:
    if not candidate_video.video_path:
        raise ValueError("candidate_video.video_path is required")
    prompt = (
        f"Task: text-to-video retrieval check\n"
        f"Query text: {query_text}\n"
        f"Candidate rank: {rank}\n"
        f"Retriever score: {score:.6f}\n"
        "Please inspect whether the candidate video truly matches the query. "
        "Do not use any dataset labels. "
        "Your answer is invalid unless all 8 JSON fields are present."
    )
    return [
        {"type": "video_url", "video_url": {"url": candidate_video.video_path}},
        {"type": "text", "text": prompt},
    ]


def build_v2t_user_content(query_video: VideoRow, candidate_text: TextRow, *, rank: int, score: float) -> list[dict]:
    if not query_video.video_path:
        raise ValueError("query_video.video_path is required")
    prompt = (
        f"Task: video-to-text retrieval check\n"
        f"Candidate text: {candidate_text.text}\n"
        f"Candidate rank: {rank}\n"
        f"Retriever score: {score:.6f}\n"
        "Please inspect whether the text truly describes the query video. "
        "Do not use any dataset labels. "
        "Your answer is invalid unless all 8 JSON fields are present."
    )
    return [
        {"type": "video_url", "video_url": {"url": query_video.video_path}},
        {"type": "text", "text": prompt},
    ]


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

    def _request(self, user_content: list[dict], system_prompt: str) -> CheckerResult:
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
            "max_tokens": 256,
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
        except urllib.error.HTTPError as exc:  # pragma: no cover
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"omni checker request failed: {detail}") from exc
        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = _complete_payload(_extract_json(content))
        except ValueError:
            parsed = _fallback_payload(str(content))
        return CheckerResult.from_dict(parsed)

    def inspect_t2v(self, query_text: str, candidate_video: VideoRow, *, rank: int, score: float) -> CheckerResult:
        return self._request(
            build_t2v_user_content(query_text, candidate_video, rank=rank, score=score),
            _checker_system_prompt("t2v"),
        )

    def inspect_v2t(self, query_video: VideoRow, candidate_text: TextRow, *, rank: int, score: float) -> CheckerResult:
        return self._request(
            build_v2t_user_content(query_video, candidate_text, rank=rank, score=score),
            _checker_system_prompt("v2t"),
        )


class MockOmniChecker:
    def __init__(
        self,
        *,
        t2v_results: dict[str, CheckerResult | dict] | None = None,
        v2t_results: dict[str, CheckerResult | dict] | None = None,
    ) -> None:
        self.t2v_results = {
            key: value if isinstance(value, CheckerResult) else CheckerResult.from_dict(value)
            for key, value in (t2v_results or {}).items()
        }
        self.v2t_results = {
            key: value if isinstance(value, CheckerResult) else CheckerResult.from_dict(value)
            for key, value in (v2t_results or {}).items()
        }

    def _resolve(self, store: dict, primary_key: str, secondary_key: str) -> CheckerResult | None:
        return store.get(secondary_key) or store.get(primary_key)

    def inspect_t2v(self, query_text: str, candidate_video: VideoRow, *, rank: int, score: float) -> CheckerResult:
        resolved = self._resolve(self.t2v_results, candidate_video.video_id, f"{query_text}::{candidate_video.video_id}")
        return resolved or CheckerResult(
            is_match=False,
            confidence=0.1,
            visual_match=0.1,
            audio_match=0.1,
            main_events=[],
            missing_elements=["unknown"],
            reason="mock fallback",
            rewrite_suggestion=query_text,
        )

    def inspect_v2t(self, query_video: VideoRow, candidate_text: TextRow, *, rank: int, score: float) -> CheckerResult:
        resolved = self._resolve(self.v2t_results, candidate_text.text_id, f"{query_video.video_id}::{candidate_text.text_id}")
        return resolved or CheckerResult(
            is_match=False,
            confidence=0.1,
            visual_match=0.1,
            audio_match=0.1,
            main_events=[],
            missing_elements=["unknown"],
            reason="mock fallback",
            rewrite_suggestion="",
        )
