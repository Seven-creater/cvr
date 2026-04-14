from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _first(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RetrievalParams:
    video_weight: float = 0.7
    audio_weight: float = 0.3
    object_focus: str = "none"
    temporal_focus: str = "global"
    topk: int = 5

    def __post_init__(self) -> None:
        total = self.video_weight + self.audio_weight
        if total <= 0:
            raise ValueError("video_weight + audio_weight must be positive")
        self.video_weight = max(0.0, self.video_weight / total)
        self.audio_weight = max(0.0, self.audio_weight / total)
        if self.topk <= 0:
            raise ValueError("topk must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryCase:
    query_id: str
    source_video_id: str
    edit_instruction: str
    target_video_id: str | None = None
    preserve_tags: list[str] = field(default_factory=list)
    required_audio_tags: list[str] = field(default_factory=list)
    required_objects: list[str] = field(default_factory=list)
    required_temporal: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryCase":
        return cls(
            query_id=_first(payload, "query_id", "id"),
            source_video_id=_first(payload, "source_video_id", "source_id", "source_video", "source"),
            edit_instruction=_first(payload, "edit_instruction", "instruction", "text", "query"),
            target_video_id=_first(payload, "target_video_id", "target_id", "target_video", "target"),
            preserve_tags=list(_ensure_list(_first(payload, "preserve_tags", "preserve", "preserved_tags", default=[]))),
            required_audio_tags=list(_ensure_list(_first(payload, "required_audio_tags", "audio_tags", "audio_requirements", default=[]))),
            required_objects=list(_ensure_list(_first(payload, "required_objects", "objects", "object_requirements", default=[]))),
            required_temporal=_first(payload, "required_temporal", "temporal_focus", "temporal"),
            notes=_first(payload, "notes", "meta", default=""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def without_target(self) -> "QueryCase":
        payload = self.to_dict()
        payload["target_video_id"] = None
        return QueryCase.from_dict(payload)


@dataclass(slots=True)
class TextQueryCase:
    query_id: str
    text: str
    target_video_id: str
    required_audio_tags: list[str] = field(default_factory=list)
    required_objects: list[str] = field(default_factory=list)
    required_temporal: str | None = None
    scene_tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateMetadata:
    video_id: str
    title: str
    summary: str
    caption: str
    asr: str
    audio_tags: list[str] = field(default_factory=list)
    visual_objects: list[str] = field(default_factory=list)
    scene_tags: list[str] = field(default_factory=list)
    temporal_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateMetadata":
        return cls(
            video_id=_first(payload, "video_id", "candidate_id", "id"),
            title=_first(payload, "title", "name", default=""),
            summary=_first(payload, "summary", "description", "text", default=""),
            caption=_first(payload, "caption", "video_caption", default=""),
            asr=_first(payload, "asr", "transcript", "speech_text", default=""),
            audio_tags=list(_ensure_list(_first(payload, "audio_tags", "audio_events", default=[]))),
            visual_objects=list(_ensure_list(_first(payload, "visual_objects", "objects", default=[]))),
            scene_tags=list(_ensure_list(_first(payload, "scene_tags", "scene_labels", "tags", default=[]))),
            temporal_tags=list(_ensure_list(_first(payload, "temporal_tags", "temporal", "time_tags", default=[]))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalCandidate:
    candidate_id: str
    score: float
    video_score: float
    audio_score: float
    summary: str
    audio_tags: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InspectionRecord:
    candidate_id: str
    title: str
    summary: str
    caption: str
    asr: str
    audio_tags: list[str]
    visual_objects: list[str]
    temporal_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareResult:
    candidate_id: str
    satisfied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCallTrace:
    tool_name: str
    arguments: dict[str, Any]
    result_preview: dict[str, Any]
    round_index: int
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoundRecord:
    round_index: int
    retrieval_params: RetrievalParams | None = None
    retrieved_candidates: list[str] = field(default_factory=list)
    inspected_candidates: list[str] = field(default_factory=list)
    comparisons: dict[str, CompareResult] = field(default_factory=dict)
    decision: str = "pending"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["comparisons"] = {
            candidate_id: result.to_dict()
            for candidate_id, result in self.comparisons.items()
        }
        if self.retrieval_params is not None:
            payload["retrieval_params"] = self.retrieval_params.to_dict()
        return payload


@dataclass(slots=True)
class RunTrace:
    query: QueryCase
    planner_name: str
    planner_metadata: dict[str, Any] = field(default_factory=dict)
    rounds: list[RoundRecord] = field(default_factory=list)
    tool_history: list[ToolCallTrace] = field(default_factory=list)
    final_candidate_id: str | None = None
    final_explanation: str = ""
    success: bool | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "planner_name": self.planner_name,
            "planner_metadata": dict(self.planner_metadata),
            "rounds": [item.to_dict() for item in self.rounds],
            "tool_history": [item.to_dict() for item in self.tool_history],
            "final_candidate_id": self.final_candidate_id,
            "final_explanation": self.final_explanation,
            "success": self.success,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class BanditActionRecord:
    action_id: str
    action_type: str
    description: str
    retrieval_params: dict[str, Any] | None = None
    candidate_id: str | None = None
    candidate_strategy: str | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BanditSample:
    query_id: str
    source_video_id: str
    round_index: int
    planner_name: str
    observation_text: str
    state: dict[str, Any]
    available_actions: list[dict[str, Any]]
    action: BanditActionRecord
    reward: float
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    final_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "source_video_id": self.source_video_id,
            "round_index": self.round_index,
            "planner_name": self.planner_name,
            "observation_text": self.observation_text,
            "state": dict(self.state),
            "available_actions": [dict(item) for item in self.available_actions],
            "action": self.action.to_dict(),
            "reward": self.reward,
            "reward_breakdown": dict(self.reward_breakdown),
            "done": self.done,
            "final_success": self.final_success,
        }
