from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.composed_omni import ALLOWED_DIFFERENCE_TYPES, OpenAIComposedDataClient


DEFAULT_DATA_ROOT = "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval"
LAYOUT_DIRS = ("raw", "clips", "metadata", "captions", "pairs", "splits", "reports", "caches")
DEFAULT_RAW_INDEX_NAME = "raw_assets.jsonl"
DEFAULT_CLIP_MANIFEST_NAME = "clips.jsonl"
DEFAULT_CLIP_ANNOTATIONS_NAME = "clip_annotations.jsonl"
DEFAULT_PAIR_PROPOSALS_NAME = "pilot_candidates.jsonl"
DEFAULT_CLIP_GROUPS_NAME = "clip_groups.jsonl"
DEFAULT_DETECTIVE_CLIP_PLAN_NAME = "clip_plan_detective.jsonl"
DEFAULT_EVENT_CLIP_MANIFEST_NAME = "extracted_event_clips.jsonl"
DEFAULT_ACCEPTED_PAIRS_NAME = "accepted_pairs.jsonl"
DEFAULT_SYNTHETIC_JUDGED_PAIRS_NAME = "judged_synthetic_pair_proposals.jsonl"
DEFAULT_SYNTHETIC_ACCEPTED_PAIRS_NAME = "accepted_synthetic_pairs.jsonl"
DEFAULT_SYNTHETIC_PILOT_REVIEW_NAME = "synthetic_pilot_review.md"
DEFAULT_MINED_PAIR_CANDIDATES_NAME = "mined_pair_candidates.jsonl"
DEFAULT_CANDIDATE_MINING_REPORT_NAME = "candidate_mining_report.md"
DEFAULT_MAX_MINED_PAIR_CANDIDATES = 240
DEFAULT_ZERO_ACCEPTED_STOP_AFTER = 0
MIN_SINGLE_SOURCE_FINAL_OMNI_QUALITY_SCORE = 0.70
DEFAULT_SELECTED_SINGLE_SOURCE_NAME = "selected_source_video.json"
DEFAULT_SINGLE_SOURCE_CANDIDATES_NAME = "selected_source_candidates.jsonl"
DEFAULT_SINGLE_SOURCE_CLIP_PLAN_NAME = "single_source_clip_plan.jsonl"
DEFAULT_SINGLE_SOURCE_CLIP_GROUPS_NAME = "single_source_clip_groups.jsonl"
DEFAULT_SINGLE_SOURCE_WHOLE_MANIFEST_NAME = "selected_source_manifest.jsonl"
DEFAULT_SINGLE_SOURCE_PAIR_CANDIDATES_NAME = "single_source_pair_candidates.jsonl"
DEFAULT_SINGLE_SOURCE_PAIR_REPORT_NAME = "single_source_pair_report.md"
DEFAULT_VIDEO_EDIT_PLAN_NAME = "video_edit_plan.jsonl"
DEFAULT_VIDEO_EDIT_PLANNER_CACHE_NAME = "video_edit_planner_cache.jsonl"
DEFAULT_VIDEO_MASK_PLAN_NAME = "video_mask_plan.jsonl"
DEFAULT_VIDEO_MASK_MANIFEST_NAME = "video_mask_manifest.jsonl"
DEFAULT_OMNI_STABLE_CLIP_SELECTION_CACHE_NAME = "omni_stable_clip_selection_cache.jsonl"
DEFAULT_REFERENCE_UNDERSTANDING_CACHE_NAME = "reference_understanding_cache.jsonl"
DEFAULT_SRC_REF_IMAGE_PLAN_NAME = "src_ref_image_plan.jsonl"
DEFAULT_SRC_REF_IMAGE_SELECTION_NAME = "src_ref_image_selection.jsonl"
DEFAULT_AUDIO_EDIT_PLAN_NAME = "audio_edit_plan.jsonl"
DEFAULT_LICENSE_NOTE = "internal research pilot only"
VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
ALLOWED_MODALITIES = {"visual", "audio"}
ALLOWED_SOURCE_TYPES = {"natural", "synthetic_edit"}
MAX_PAIR_CANDIDATES = 40
MAX_PAIR_LOCAL_COMPARISONS = 240
MAX_TEMPLATE_CLUSTER_COMPARISONS = 480
MIN_PAIR_CONTEXT_SCORE = 0.03
MAX_PAIR_CHANGED_TYPES = 5
MIN_PAIR_EDIT_MATCH_SCORE = 0.15
PAIR_PRIORITY = (
    "attribute",
    "object_presence",
    "object_count",
    "action",
    "scene",
    "audio_event",
    "speech",
    "visible_text",
)
HIGH_CONTEXT_PAIR_PRIORITY = PAIR_PRIORITY
DIVERSE_PAIR_BUCKET_TARGETS = {
    "attribute": 4,
    "object_presence": 4,
    "object_count": 2,
    "action": 4,
    "scene": 3,
    "audio_event": 3,
    "speech": 1,
    "visible_text": 1,
}
MIN_ACCEPT_SAME_CONTEXT_SCORE = 0.55
MIN_ACCEPT_EDIT_MATCH_SCORE = 0.75
MIN_ACCEPT_TARGET_UNIQUENESS_SCORE = 0.70
MIN_ACCEPT_EDIT_NECESSITY_SCORE = 0.70
MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE = 0.75
MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE = 0.65
MIN_ACCEPT_ACTION_EVIDENCE_SCORE = 0.65
MIN_ACCEPT_SPEECH_EVIDENCE_SCORE = 0.75
MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE = 0.70
MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE = 0.70
MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE = 0.75
MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE = 0.85
MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE = 0.95
MIN_VIDEO_MASK_COVERAGE_RATIO = 0.02
MAX_VIDEO_MASK_COVERAGE_RATIO = 0.65
MIN_VIDEO_MASK_TEMPORAL_STABILITY = 0.75
MIN_VIDEO_MASK_NONEMPTY_FRAME_RATIO = 0.90
MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE = 0.995
VISUAL_DIFFERENCE_TYPES = {"object_count", "object_presence", "attribute", "action", "scene", "visible_text"}
VACE_BG_REPLACE_COMPOSITE_ROUTE = "vace_bg_replace_composite_first_frame_mv2v"
DETERMINISTIC_BG_COMPOSITE_ROUTE = "deterministic_foreground_background_composite"
DETERMINISTIC_MASKED_REFERENCE_PASTE_ROUTE = "deterministic_masked_reference_paste"
GUIDED_COMPOSITE_REFINE_VACE_ROUTE = "guided_composite_refine_vace"
VACE_FULL_GENERATIVE_ROUTE = "vace_full_generative"
SYNTHETIC_VISUAL_ROUTES = {
    "vace_controlled",
    "ltx2_retake",
    "tokenflow_style",
    VACE_BG_REPLACE_COMPOSITE_ROUTE,
    DETERMINISTIC_BG_COMPOSITE_ROUTE,
    DETERMINISTIC_MASKED_REFERENCE_PASTE_ROUTE,
    GUIDED_COMPOSITE_REFINE_VACE_ROUTE,
    VACE_FULL_GENERATIVE_ROUTE,
}
SYNTHETIC_AUDIO_ROUTES = {"deterministic_overlay", "foleycrafter_temporal", "frieren_benchmark", "audio_deterministic"}
VACE_ATTRIBUTE_MARKERS = {
    "attribute",
    "color",
    "colour",
    "bright",
    "yellow",
    "red",
    "blue",
    "green",
    "silver",
    "gold",
    "black",
    "white",
    "body",
    "shell",
    "surface",
    "material",
    "metal",
    "metallic",
    "matte",
    "plastic",
    "texture",
    "style",
    "visor",
    "light",
    "clothing",
    "shirt",
    "jacket",
    "dress",
    "vehicle",
    "car",
    "robot",
    "background",
    "backdrop",
    "room",
    "street",
    "office",
    "kitchen",
    "laboratory",
    "lab",
    "cyberpunk",
    "anime",
    "cinematic",
    "neon",
    "weather",
    "rain",
    "night",
    "day",
}
VACE_TINY_OR_INSERTION_MARKERS = {
    "sticker",
    "poster",
    "plant",
    "potted",
    "badge",
    "button",
    "logo",
    "label",
    "sign",
    "text",
    "caption",
    "nose ring",
    "earring",
    "ear ring",
    "necklace",
    "bracelet",
    "watch",
    "flower",
    "cube",
    "eraser",
}
VACE_BACKGROUND_STYLE_MARKERS = {
    "background",
    "backdrop",
    "room",
    "street",
    "office",
    "kitchen",
    "laboratory",
    "lab",
    "cyberpunk",
    "anime",
    "oil painting",
    "cinematic",
    "neon",
    "night",
    "day",
    "rain",
    "sunset",
    "studio",
}
VACE_EXPLORATION_OBJECT_REPLACEMENTS = {
    "cup": "bottle",
    "mug": "bottle",
    "glass": "bottle",
    "phone": "tablet",
    "smartphone": "tablet",
    "mobile phone": "tablet",
    "laptop": "tablet",
    "computer": "tablet",
    "book": "notebook",
    "bag": "backpack",
    "tote bag": "backpack",
    "box": "suitcase",
    "chair": "stool",
    "bottle": "thermos",
    "toy": "wooden toy",
}
VACE_EXPLORATION_REMOVABLE_OBJECTS = {
    "cup",
    "mug",
    "glass",
    "phone",
    "smartphone",
    "mobile phone",
    "bag",
    "tote bag",
    "backpack",
    "glasses",
    "sunglasses",
    "hat",
    "chair",
    "box",
    "bottle",
}
VACE_SCREEN_TEXT_OBJECTS = {
    "computer",
    "desktop",
    "laptop",
    "monitor",
    "screen",
    "tablet",
    "television",
    "tv",
}
VACE_SEATED_SUPPORT_OBJECTS = {"bench", "chair", "seat", "sofa", "stool"}
VACE_GENERIC_MULTI_INSTANCE_MASK_OBJECTS = {
    "bag",
    "bench",
    "bottle",
    "box",
    "chair",
    "cup",
    "desk",
    "glass",
    "man",
    "person",
    "phone",
    "seat",
    "sofa",
    "stool",
    "table",
    "woman",
}
VACE_TEXT_OR_LOGO_EDIT_MARKERS = {
    "caption",
    "country",
    "flag",
    "letter",
    "letters",
    "logo",
    "made in",
    "map",
    "ocr",
    "subtitle",
    "text",
    "tourism",
    "touristy",
    "watermark",
    "word",
}
VACE_BROAD_SCENE_EDIT_MARKERS = {
    "make it like",
    "turn their",
    "turn the scene",
    "turn this",
    "turn it into",
    "loose stock pair",
}
VACE_CLOTHING_OBJECT_MARKERS = {
    "clothing",
    "outfit",
    "shirt",
    "jacket",
    "coat",
    "dress",
    "blouse",
    "robe",
    "hoodie",
    "sweater",
    "vest",
    "pants",
    "skirt",
}
VACE_BLACK_JACKET_REQUIRED_PHRASE = "open black long-sleeved jacket"
VACE_BLACK_JACKET_PROMPT = (
    "A man in a blue fedora wearing an open black long-sleeved jacket over the same black T-shirt "
    "plays a ukulele and sings into a microphone against the same brick wall."
)
VACE_BLACK_JACKET_SRC_REF_TARGET = "open black long-sleeved jacket over a black T-shirt"
VACE_BLACK_JACKET_FORBIDDEN_PROMPT_MARKERS = {
    "patterned shirt",
    "dark shirt",
    "navy shirt",
    "polo",
    "black clothing",
    "change only",
}
VACE_CLOTHING_SRC_REF_ARTIFACT_MARKERS = {
    "empty jacket",
    "flat lay",
    "hanger",
    "ghost mannequin",
    "mannequin",
    "product catalog",
    "catalog",
}
VACE_BACKGROUND_SRC_REF_WIDTH = 1664
VACE_BACKGROUND_SRC_REF_HEIGHT = 928
VACE_STRUCTURAL_CLOTHING_TARGET_MARKERS = {
    "open jacket",
    "open black jacket",
    "open black long sleeved jacket",
    "long sleeve jacket",
    "long sleeved jacket",
    "long sleeves",
    "outerwear",
    "layered jacket",
    "jacket over",
    "coat over",
    "blazer",
}
VACE_OUTERWEAR_MARKERS = {"jacket", "coat", "blazer", "outerwear"}
VACE_NON_OUTERWEAR_CLOTHING_MARKERS = {"shirt", "t shirt", "tee", "short sleeve", "short sleeved", "outfit", "clothing", "blouse", "robe"}
VIDEO_MASK_SEMANTICS_VERSION = 3
VIDEO_MASK_POLARITY = "white_generate_black_preserve"
VACE_GENERIC_PERSON_MASK_QUERIES = {"man", "woman", "person", "people", "subject", "main subject"}
VACE_TINY_FULLFRAME_OBJECTS = {"cup", "mug"}
VACE_WORN_OBJECT_MARKERS = {"backpack", "bag", "handbag", "purse", "satchel"}
VACE_MULTI_SHOT_MASK_MARKERS = {
    "various locations",
    "multiple locations",
    "different locations",
    "various scenes",
    "multiple scenes",
    "scene changes",
    "seen in various",
    "montage",
}
VACE_CLOTHING_FORBIDDEN_RESULT_MARKERS = {"vest", "polo", "undershirt", "tank top", "sleeveless"}
VACE_DARK_COLOR_MARKERS = {"black", "dark", "navy", "deep navy", "deep navy blue", "charcoal"}
VACE_BACKGROUND_MAX_SUBJECT_OVERLAP_RATIO = 0.20
VACE_BACKGROUND_MIN_FOREGROUND_SUBJECT_COVERAGE_RATIO = 0.04
VACE_BACKGROUND_MAX_FOREGROUND_SUBJECT_COVERAGE_RATIO = 0.70
VACE_BACKGROUND_ORIGINAL_SCENE_MARKERS = {
    "brick wall",
    "curtain",
    "desk",
    "door",
    "indoor room",
    "kitchen",
    "living room",
    "office",
    "same room",
    "stage",
    "studio",
    "sunlit room",
    "window",
}
VACE_BACKGROUND_OVERLAY_FAILURE_MARKERS = {
    "blue filter",
    "blue haze",
    "blue overlay",
    "blue tint",
    "blue tinted",
    "blue wash",
    "color cast",
    "overlay",
    "semi transparent",
    "transparent overlay",
}
VACE_BACKGROUND_TARGET_SYNONYMS = {
    "laboratory": {"laboratory", "lab"},
    "lab": {"laboratory", "lab"},
    "futuristic": {"futuristic", "sci fi", "science fiction", "high tech", "hi tech"},
}
VACE_BACKGROUND_REPLACE_PRESERVE_DENY_MARKERS = {
    "background",
    "door",
    "lighting",
    "layout",
    "original background",
    "room",
    "source background",
    "sunlit room",
    "wall",
    "window",
}
VACE_BACKGROUND_REPLACE_REGION_DENY_MARKERS = {
    "background",
    "door",
    "room",
    "wall",
    "window",
}
VACE_BACKGROUND_REPLACE_LOCK_DENY_PATTERNS = {
    "door",
    "do not change lighting",
    "do not change layout",
    "original background",
    "original room",
    "preserve lighting",
    "preserve lighting exactly",
    "preserve layout",
    "preserve layout exactly",
    "preserve source background",
    "room",
    "source background",
    "sunlit room",
    "wall",
    "window",
}
VACE_BACKGROUND_REPLACE_NEGATIVE_PROMPT = (
    "do not change the subject identity, face, hair, glasses, pose, body position, mouth motion, "
    "or camera framing. no extra people, no text, no flicker, no unstable background, no distorted face, "
    "no duplicated body parts, no deformed hands, no blurry face."
)
VACE_SEMANTIC_PRESERVE_OBJECT_MARKERS = {
    "beard",
    "face",
    "glasses",
    "guitar",
    "hat",
    "instrument",
    "man",
    "microphone",
    "person",
    "ukulele",
    "woman",
}
INTRACLIP_CHANGE_MARKERS = (
    "change from",
    "changes from",
    "changed from",
    "changes to",
    "changed to",
    "replace",
    "replaced by",
    "replaces",
    "transition from",
    "transitions from",
    "turns into",
    "becomes",
    "followed by",
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
GENERIC_SPEECH_TOKENS = {
    "camera",
    "conversation",
    "dialog",
    "dialogue",
    "discuss",
    "discusses",
    "discussing",
    "female",
    "interview",
    "male",
    "man",
    "monologue",
    "narrate",
    "narrates",
    "narrating",
    "narration",
    "narrator",
    "person",
    "says",
    "speak",
    "speaker",
    "speaking",
    "speech",
    "talk",
    "talking",
    "voice",
    "voiceover",
    "woman",
}
GENERIC_SPEECH_PHRASES = {
    "speaks to camera",
    "speaking to camera",
    "talks to camera",
    "talking to camera",
    "speaks directly to the camera",
    "speaking directly to the camera",
    "speech",
    "narration",
    "talking",
    "voiceover",
}
GENERIC_EDIT_TEXT_PHRASES = {
    "change the mood",
    "make it better",
    "make it cinematic",
    "make it nice",
    "make it more cinematic",
    "make the video better",
    "make the scene better",
    "make the scene more interesting",
    "change the topic",
    "change the vibe",
}
EDIT_ACTION_VERBS = {
    "add",
    "adds",
    "appear",
    "appears",
    "begin",
    "begins",
    "change",
    "changes",
    "convert",
    "converts",
    "delete",
    "deletes",
    "disappear",
    "disappears",
    "increase",
    "increases",
    "insert",
    "inserts",
    "introduce",
    "introduced",
    "introduces",
    "launch",
    "launched",
    "make",
    "remove",
    "removes",
    "replace",
    "replaced",
    "replaces",
    "start",
    "starts",
    "swap",
    "swaps",
    "turn",
    "turns",
    "wave",
    "waving",
}
EDIT_TEXT_AUDIO_TOKENS = {
    "audio",
    "hum",
    "music",
    "noise",
    "scratch",
    "scratching",
    "sound",
    "speech",
    "voice",
    "whoosh",
}
EDIT_TEXT_VISUAL_TOKENS = {
    "background",
    "color",
    "colour",
    "object",
    "scene",
    "shirt",
    "text",
    "video",
    "visible",
}
VISUAL_DESCRIPTION_TOKENS = {
    "background",
    "beard",
    "blue",
    "camera",
    "clothes",
    "forest",
    "glasses",
    "hat",
    "jacket",
    "looking",
    "scene",
    "shirt",
    "standing",
    "wearing",
}
GENERIC_HUMAN_GROUP_TOKENS = {
    "audience",
    "controllers",
    "crew",
    "crowd",
    "employees",
    "group",
    "operators",
    "people",
    "personnel",
    "persons",
    "staff",
    "team",
    "workers",
}
OBJECT_ALIAS_GROUPS = (
    (
        "dollhouse",
        "toy house",
        "toy home",
        "play house",
        "playhouse",
    ),
    (
        "framed picture",
        "framed pictures",
        "picture",
        "pictures",
        "painting",
        "paintings",
        "poster",
        "posters",
        "wall art",
        "artwork",
        "frame",
        "frames",
    ),
    (
        "personnel",
        "people",
        "staff",
        "crowd",
        "workers",
        "persons",
        "team",
        "crew",
        "operators",
        "controllers",
        "employees",
    ),
)
BACKGROUND_DECOR_OBJECTS = {"framed picture"}
OBJECT_LABEL_STOPWORDS = {
    "a",
    "an",
    "the",
    "present",
    "visible",
    "appears",
    "appear",
    "shown",
    "showing",
}
MIN_COMPETING_DIFFERENCE_STRENGTH = 0.72
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
    "machine",
    "mechanical",
    "laugh",
    "laughter",
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
SPEECH_ONLY_AUDIO_PATTERNS = (
    "only speech",
    "speech only",
    "contains only speech",
    "contains speech only",
    "only narration",
    "narration only",
    "only talking",
    "talking only",
    "only voiceover",
    "voiceover only",
)
SPEECH_CONTENT_EDIT_PATTERNS = (
    "speech",
    "spoken content",
    "transcript",
    "narration",
    "narrator",
    "voiceover",
    "says",
    "say ",
    "topic",
    "talks about",
    "talk about",
    "discussing",
    "discussion",
)
NON_SPEECH_AUDIO_ABSENCE_PATTERNS = (
    "no background music",
    "no background noise",
    "no ambient noise",
    "no ambient sound",
    "no ambient sounds",
    "no distinctive audio",
    "no non speech audio",
    "without background music",
    "without background noise",
    "without ambient noise",
)
EDIT_TEXT_START_VERBS = {
    "add",
    "change",
    "include",
    "increase",
    "introduce",
    "make",
    "reduce",
    "remove",
    "replace",
    "start",
    "starts",
    "stop",
    "stops",
    "switch",
    "turn",
}
EDIT_TEXT_CAPTION_MAX_TOKENS = 24
EDIT_TEXT_VISUAL_LEAK_TOKENS = {
    "background",
    "blonde",
    "camera",
    "desk",
    "dollhouse",
    "hair",
    "man",
    "nose",
    "person",
    "room",
    "shirt",
    "speaking",
    "toy",
    "woman",
}
EDIT_TEXT_AUDIO_TOKENS = NON_SPEECH_AUDIO_TOKENS | {"audio", "sound", "sounds", "effect", "effects"}
EDIT_TEXT_VISIBLE_TEXT_TOKENS = {"caption", "ocr", "on", "screen", "text", "subtitle", "subtitles"}
EDIT_TEXT_SPEECH_TOKENS = GENERIC_SPEECH_TOKENS | {"transcript", "spoken", "says", "say", "topic", "topics"}
NATURAL_PAIR_GATE_LABELS = {
    "bad_imperative_edit_text": "bad_imperative_edit_text: edit_text is vague, malformed, or not an edit command",
    "too_similar_without_observable_delta": "too_similar_without_observable_delta: near-duplicate visual pair has no frame-backed delta",
    "too_broad_or_loose_pair": "too_broad_or_loose_pair: broad scene change without enough shared context",
    "visible_text_disabled": "visible_text_disabled: visible-text edits are diagnostic only for this dataset pass",
    "ocr_template_risk": "ocr_template_risk: visible-text edit lacks reliable OCR evidence or target uniqueness",
    "audio_event_too_similar": "audio_event_too_similar: audio_event from/to values are too similar to be a useful edit",
    "audio_secondary_due_to_visual_delta": "audio_secondary_due_to_visual_delta: stronger visual differences make audio a secondary delta",
    "visible_text_fragment_edit": "visible_text_fragment_edit: target visible text is only a fragment of the source text",
}
VISIBLE_TEXT_FRAGMENT_MIN_SOURCE_TOKENS = 2
VISIBLE_TEXT_FRAGMENT_MAX_TARGET_TOKEN_RATIO = 0.75
FINAL_ACCEPT_BUCKET_TARGETS = {
    "attribute": 6,
    "object_presence": 4,
    "action": 3,
    "scene": 3,
    "audio_event": 4,
    "speech": 0,
    "visible_text": 0,
}
EXPLORATION_SMALL_ACCEPT_BUCKET_TARGETS = {
    "attribute": 1,
    "object": 1,
    "action": 1,
    "scene": 1,
    "audio_event": 1,
}
FINAL_DISABLED_DIFFERENCE_TYPES = {"speech", "visible_text"}
DOMINANT_VISUAL_DIFFERENCE_TYPES = ("attribute", "object_presence", "object_count", "scene", "action")
AUDIO_PRIMARY_MIN_SAME_CONTEXT_SCORE = 0.86
AUDIO_PRIMARY_MIN_TEMPLATE_COMPATIBILITY_SCORE = 0.82
AUDIO_PRIMARY_MIN_VISUAL_NEAR_DUPLICATE_SCORE = 0.98
MIN_TEMPLATE_SEMANTIC_CONTEXT_SCORE = 0.35
MIN_TEMPLATE_COMPATIBILITY_SCORE = 0.72
MIN_TEMPLATE_CLEAN_STABILITY_SCORE = 0.75
MIN_TEMPLATE_SINGLE_DELTA_BUNDLE_SCORE = 0.75
MIN_TEMPLATE_TARGET_UNIQUENESS_SCORE = 0.75
MIN_TEMPLATE_DIFFERENCE_STRENGTH_SCORE = 0.75
DEFAULT_ACCEPTANCE_PROFILE = "final"
EXPLORATION_ACCEPTANCE_PROFILE = "exploration"
AUDIO_MATTERS_ACCEPTANCE_PROFILE = "audio_matters"
B_AUDIO_REVIEW_ACCEPTANCE_PROFILE = "b_audio_review"
B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE = "b_audio_context_cvr"
B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE = "b_audio_blind_review"
STANDARD_AUDIO_DATASET_LINE = "standard"
VISUAL_AUDIO_ANCHOR_LINE = "visual_audio_anchor"
SPEECH_AUDIO_CONTENT_LINE = "speech_audio_content"
AUDIO_LINE_QUALITY_PROFILE_V4_STRICT = "v4_strict"
V4_A_STRONG_VISUAL_TYPES = {"scene", "action", "object_presence"}
V4_VAGUE_AUDIO_TERMS = (
    "buzz",
    "buzzing",
    "click",
    "clicking",
    "electronic tone",
    "electronic hum",
    "hum",
    "humming",
    "low frequency",
    "low-frequency",
    "tone",
)
V4_CONCRETE_AUDIO_TERMS = (
    "applause",
    "cheer",
    "cheering",
    "chant",
    "crowd",
    "music",
    "song",
    "whistle",
    "siren",
    "bell",
    "rain",
    "water",
    "wind",
    "engine",
    "machinery",
    "footstep",
    "footsteps",
)
V4_B_MIN_VISUAL_CONTEXT_SIMILARITY = 0.30
V4_B_MAX_VISUAL_DELTA_STRENGTH = 0.55
B_LINE_AUDIO_EDIT_TERMS = (
    "audio",
    "sound",
    "speech",
    "spoken",
    "says",
    "say",
    "talk",
    "talking",
    "discuss",
    "discussing",
    "commentary",
    "commentator",
    "narration",
    "voice",
    "words",
    "transcript",
    "cheer",
    "cheering",
    "applause",
    "music",
    "song",
    "ambient",
    "ambience",
    "crowd",
)
B_LINE_VISUAL_EDIT_TERMS = (
    "shot",
    "scene",
    "camera",
    "view",
    "visual",
    "frame",
    "background",
    "foreground",
    "object",
    "person",
    "people",
    "man ",
    "men ",
    "woman ",
    "women ",
    "shirt",
    "boat",
    "river",
    "fishing",
    "podium",
    "microphone",
    "color",
    "colour",
    "close up",
    "close-up",
    "gesture",
    "gestures",
    "smile",
    "walking",
    "walks",
    "walk off",
    "off screen",
    "off-screen",
    "subscribe",
    "button",
    "bell icon",
    "wide shot",
    "full orchestra",
    "text",
    "subtitle",
    "logo",
)
A_LINE_FINAL_RESCUABLE_LOCAL_ISSUE_PREFIXES = (
    "low_pair_video_confidence:",
    "visual_too_similar_for_A:",
)
B_LINE_FINAL_RESCUABLE_LOCAL_ISSUE_PREFIXES = (
    "low_pair_video_confidence:",
    "speech_audio_content edit must include audio modality",
    "speech_audio_content speech edit lacks transcript-backed evidence",
    "speech_audio_content audio_event edit lacks non-speech audio evidence",
    "edit_text_not_audio_only:",
    "visual_too_different_for_B:",
    "audio_not_primary:",
    "vague_audio_event:",
    "video_context_too_weak_for_B:",
    "asr_degeneracy_risk_too_high:",
)
AUDIO_DATASET_LINE_NAMES = {
    STANDARD_AUDIO_DATASET_LINE,
    VISUAL_AUDIO_ANCHOR_LINE,
    SPEECH_AUDIO_CONTENT_LINE,
}
ACCEPTANCE_PROFILE_NAMES = {
    DEFAULT_ACCEPTANCE_PROFILE,
    EXPLORATION_ACCEPTANCE_PROFILE,
    AUDIO_MATTERS_ACCEPTANCE_PROFILE,
    B_AUDIO_REVIEW_ACCEPTANCE_PROFILE,
    B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE,
    B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE,
}
ACCEPTANCE_PROFILE_CONFIGS = {
    DEFAULT_ACCEPTANCE_PROFILE: {
        "template_semantic_context_score": MIN_TEMPLATE_SEMANTIC_CONTEXT_SCORE,
        "template_compatibility_score": MIN_TEMPLATE_COMPATIBILITY_SCORE,
        "template_clean_stability_score": MIN_TEMPLATE_CLEAN_STABILITY_SCORE,
        "template_single_delta_bundle_score": MIN_TEMPLATE_SINGLE_DELTA_BUNDLE_SCORE,
        "template_target_uniqueness_score": MIN_TEMPLATE_TARGET_UNIQUENESS_SCORE,
        "template_difference_strength_score": MIN_TEMPLATE_DIFFERENCE_STRENGTH_SCORE,
        "same_context_score": MIN_ACCEPT_SAME_CONTEXT_SCORE,
        "edit_match_score": MIN_ACCEPT_EDIT_MATCH_SCORE,
        "target_uniqueness_score": MIN_ACCEPT_TARGET_UNIQUENESS_SCORE,
        "difference_strength_score": MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE,
        "edit_necessity_score": MIN_ACCEPT_EDIT_NECESSITY_SCORE,
        "edit_target_alignment_score": MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE,
        "action_evidence_score": MIN_ACCEPT_ACTION_EVIDENCE_SCORE,
        "non_speech_audio_event_score": MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE,
        "edit_text_quality_score": MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE,
    },
    EXPLORATION_ACCEPTANCE_PROFILE: {
        "template_semantic_context_score": 0.25,
        "template_compatibility_score": 0.55,
        "template_clean_stability_score": 0.55,
        "template_single_delta_bundle_score": 0.50,
        "template_target_uniqueness_score": 0.45,
        "template_difference_strength_score": 0.50,
        "same_context_score": 0.45,
        "edit_match_score": 0.45,
        "target_uniqueness_score": 0.45,
        "difference_strength_score": 0.50,
        "edit_necessity_score": 0.55,
        "edit_target_alignment_score": 0.55,
        "action_evidence_score": 0.45,
        "non_speech_audio_event_score": 0.45,
        "edit_text_quality_score": 0.55,
    },
    AUDIO_MATTERS_ACCEPTANCE_PROFILE: {
        "template_semantic_context_score": 0.25,
        "template_compatibility_score": 0.55,
        "template_clean_stability_score": 0.55,
        "template_single_delta_bundle_score": 0.40,
        "template_target_uniqueness_score": 0.40,
        "template_difference_strength_score": 0.50,
        "same_context_score": 0.45,
        "edit_match_score": 0.45,
        "target_uniqueness_score": 0.40,
        "difference_strength_score": 0.50,
        "edit_necessity_score": 0.50,
        "edit_target_alignment_score": 0.50,
        "action_evidence_score": 0.40,
        "non_speech_audio_event_score": 0.45,
        "edit_text_quality_score": 0.55,
        "audio_anchor_score": 0.86,
        "visual_delta_strength": 0.70,
        "near_duplicate_risk": 0.85,
    },
    B_AUDIO_REVIEW_ACCEPTANCE_PROFILE: {
        "template_semantic_context_score": 0.25,
        "template_compatibility_score": 0.55,
        "template_clean_stability_score": 0.55,
        "template_single_delta_bundle_score": 0.50,
        "template_target_uniqueness_score": 0.40,
        "template_difference_strength_score": 0.45,
        "same_context_score": 0.40,
        "edit_match_score": 0.45,
        "target_uniqueness_score": 0.40,
        "difference_strength_score": 0.45,
        "edit_necessity_score": 0.50,
        "edit_target_alignment_score": 0.50,
        "action_evidence_score": 0.40,
        "non_speech_audio_event_score": 0.40,
        "edit_text_quality_score": 0.50,
    },
    B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE: {
        "template_semantic_context_score": 0.25,
        "template_compatibility_score": 0.55,
        "template_clean_stability_score": 0.55,
        "template_single_delta_bundle_score": 0.50,
        "template_target_uniqueness_score": 0.40,
        "template_difference_strength_score": 0.45,
        "same_context_score": 0.40,
        "edit_match_score": 0.45,
        "target_uniqueness_score": 0.40,
        "difference_strength_score": 0.45,
        "edit_necessity_score": 0.50,
        "edit_target_alignment_score": 0.50,
        "action_evidence_score": 0.40,
        "non_speech_audio_event_score": 0.40,
        "edit_text_quality_score": 0.50,
        "video_context_strength": 0.45,
        "asr_degeneracy_risk": 0.55,
    },
    B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE: {
        "template_semantic_context_score": 0.25,
        "template_compatibility_score": 0.55,
        "template_clean_stability_score": 0.55,
        "template_single_delta_bundle_score": 0.50,
        "template_target_uniqueness_score": 0.40,
        "template_difference_strength_score": 0.45,
        "same_context_score": 0.40,
        "edit_match_score": 0.45,
        "target_uniqueness_score": 0.40,
        "difference_strength_score": 0.45,
        "edit_necessity_score": 0.50,
        "edit_target_alignment_score": 0.50,
        "action_evidence_score": 0.40,
        "non_speech_audio_event_score": 0.40,
        "edit_text_quality_score": 0.50,
        "video_context_strength": 0.45,
        "asr_degeneracy_risk": 0.55,
        "visual_delta_strength": 0.55,
    },
}
SAME_TEMPLATE_CLUSTER_RELATION = "same_template_cluster"
DEFAULT_DIAGNOSTIC_BUNDLE_NAME = "diagnostic_bundle"
DIAGNOSTIC_BUCKET_KEYS = ("ocr", "near_duplicate", "over_broad", "audio_weak")
GENERIC_HUMAN_OBJECT_LABELS = {
    "adult",
    "audience",
    "crowd",
    "girl",
    "guy",
    "host",
    "human",
    "kid",
    "man",
    "men",
    "people",
    "person",
    "presenter",
    "speaker",
    "staff",
    "woman",
    "women",
}


def _normalize_acceptance_profile(value: str | None) -> str:
    profile = str(value or DEFAULT_ACCEPTANCE_PROFILE).strip().lower().replace("-", "_")
    if profile == "b_context_cvr":
        profile = B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE
    if profile in {"b_audio_blind", "b_blind_review", "blind_audio_review"}:
        profile = B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE
    if profile not in ACCEPTANCE_PROFILE_NAMES:
        allowed = ", ".join(sorted(ACCEPTANCE_PROFILE_NAMES))
        raise ValueError(f"unsupported acceptance_profile={value!r}; expected one of: {allowed}")
    return profile


def _normalize_audio_dataset_line(value: str | None) -> str:
    line = str(value or STANDARD_AUDIO_DATASET_LINE).strip().lower().replace("-", "_")
    if line in {"", "none"}:
        line = STANDARD_AUDIO_DATASET_LINE
    if line not in AUDIO_DATASET_LINE_NAMES:
        allowed = ", ".join(sorted(AUDIO_DATASET_LINE_NAMES))
        raise ValueError(f"unsupported audio_dataset_line={value!r}; expected one of: {allowed}")
    return line


def _acceptance_profile_config(acceptance_profile: str | None) -> dict[str, float]:
    return ACCEPTANCE_PROFILE_CONFIGS[_normalize_acceptance_profile(acceptance_profile)]


def _profile_threshold(acceptance_profile: str | None, key: str) -> float:
    return _score_float(_acceptance_profile_config(acceptance_profile).get(key))


def _is_exploration_profile(acceptance_profile: str | None) -> bool:
    return _normalize_acceptance_profile(acceptance_profile) == EXPLORATION_ACCEPTANCE_PROFILE


def _is_audio_matters_profile(acceptance_profile: str | None) -> bool:
    return _normalize_acceptance_profile(acceptance_profile) == AUDIO_MATTERS_ACCEPTANCE_PROFILE


def _is_b_audio_review_profile(acceptance_profile: str | None) -> bool:
    return _normalize_acceptance_profile(acceptance_profile) in {
        B_AUDIO_REVIEW_ACCEPTANCE_PROFILE,
        B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE,
        B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE,
    }


def _is_b_audio_context_cvr_profile(acceptance_profile: str | None) -> bool:
    return _normalize_acceptance_profile(acceptance_profile) == B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE


def _is_b_audio_blind_review_profile(acceptance_profile: str | None) -> bool:
    return _normalize_acceptance_profile(acceptance_profile) == B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE


def _uses_soft_local_gate_profile(acceptance_profile: str | None) -> bool:
    profile = _normalize_acceptance_profile(acceptance_profile)
    return profile in {
        EXPLORATION_ACCEPTANCE_PROFILE,
        AUDIO_MATTERS_ACCEPTANCE_PROFILE,
        B_AUDIO_REVIEW_ACCEPTANCE_PROFILE,
        B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE,
        B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE,
    }
SUBJECT_SIGNATURE_MARKER_TOKENS = {
    "bald",
    "beard",
    "bearded",
    "blonde",
    "blond",
    "brown",
    "brunette",
    "curly",
    "earring",
    "earrings",
    "glasses",
    "gray",
    "grey",
    "hair",
    "hat",
    "headscarf",
    "hoodie",
    "jacket",
    "mustache",
    "moustache",
    "necklace",
    "ponytail",
    "receding",
    "red",
    "robe",
    "shirt",
    "suit",
    "sweater",
    "tie",
    "vest",
    "wearing",
    "white",
}
SCENE_SIGNATURE_MARKER_TOKENS = {
    "airport",
    "auditorium",
    "beach",
    "bedroom",
    "classroom",
    "conference",
    "corridor",
    "desk",
    "forest",
    "hallway",
    "kitchen",
    "lab",
    "laboratory",
    "lecture",
    "living",
    "office",
    "outdoor",
    "park",
    "podium",
    "stage",
    "store",
    "street",
    "studio",
    "workshop",
}
TITLE_CARD_HINT_TOKENS = {
    "credits",
    "headline",
    "intro",
    "logo",
    "lower third",
    "outro",
    "subtitle",
    "subtitles",
    "title",
    "watermark",
}


@dataclass(frozen=True)
class RawAsset:
    asset_id: str
    dataset: str
    path: str
    relative_path: str
    file_name: str
    extension: str
    size_bytes: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "dataset": self.dataset,
            "path": self.path,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class ClipManifestRecord:
    clip_id: str
    source_asset_id: str | None
    source_path: str
    output_path: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    role: str | None
    notes: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "clip_id": self.clip_id,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.source_asset_id:
            payload["source_asset_id"] = self.source_asset_id
        if self.role:
            payload["role"] = self.role
        if self.notes:
            payload["notes"] = self.notes
        return payload


def ensure_layout(root: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    paths = {"root": root_path}
    for name in LAYOUT_DIRS:
        path = root_path / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def discover_raw_sources(root: str | Path) -> list[tuple[str, Path]]:
    raw_datasets_root = Path(root) / "raw_datasets"
    if not raw_datasets_root.exists():
        return []
    sources: list[tuple[str, Path]] = []
    for candidate in sorted(raw_datasets_root.iterdir()):
        if candidate.is_dir():
            sources.append((candidate.name, candidate))
    return sources


def index_raw_sources(
    *,
    root: str | Path,
    sources: list[tuple[str, str | Path]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    output = Path(output_path) if output_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME

    assets: list[RawAsset] = []
    per_dataset_counts: dict[str, int] = {}
    for dataset_name, raw_source in sources:
        source_path = Path(raw_source)
        if not source_path.exists():
            raise FileNotFoundError(f"raw source does not exist: {source_path}")
        if not source_path.is_dir():
            raise NotADirectoryError(f"raw source must be a directory: {source_path}")

        count = 0
        for path in sorted(source_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            stat = path.stat()
            relative_path = path.relative_to(source_path).as_posix()
            assets.append(
                RawAsset(
                    asset_id=_build_asset_id(dataset_name, relative_path),
                    dataset=dataset_name,
                    path=str(path),
                    relative_path=relative_path,
                    file_name=path.name,
                    extension=path.suffix.lower(),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            count += 1
        per_dataset_counts[dataset_name] = count

    _write_jsonl(output, [asset.to_dict() for asset in assets])
    report_path = layout["reports"] / "raw_assets_summary.md"
    report_path.write_text(_build_raw_summary_report(output, per_dataset_counts), encoding="utf-8")
    return {
        "output_path": str(output),
        "report_path": str(report_path),
        "asset_count": len(assets),
        "dataset_counts": per_dataset_counts,
    }


def extract_clips(
    *,
    root: str | Path,
    plan_path: str | Path,
    raw_index_path: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    plan = list(_load_jsonl(Path(plan_path)))
    if not plan:
        raise ValueError("clip plan is empty")

    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    output_manifest = Path(output_manifest_path) if output_manifest_path else layout["metadata"] / DEFAULT_CLIP_MANIFEST_NAME

    commands: list[list[str]] = []
    records: list[ClipManifestRecord] = []
    seen_clip_ids: set[str] = set()
    for line_number, item in enumerate(plan, start=1):
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError(f"clip plan line {line_number}: clip_id is required")
        if clip_id in seen_clip_ids:
            raise ValueError(f"clip plan line {line_number}: duplicate clip_id={clip_id}")
        seen_clip_ids.add(clip_id)

        source_asset_id = str(item.get("source_asset_id", "")).strip() or None
        source_path = str(item.get("source_path", "")).strip()
        if source_asset_id:
            if source_asset_id not in raw_index:
                raise ValueError(f"clip plan line {line_number}: unknown source_asset_id={source_asset_id}")
            source_path = raw_index[source_asset_id]["path"]
        if not source_path:
            raise ValueError(f"clip plan line {line_number}: source_asset_id or source_path is required")

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"clip plan line {line_number}: source video not found: {source}")

        start_seconds = _as_non_negative_float(item.get("start_seconds"), f"clip plan line {line_number}: start_seconds")
        end_seconds = _as_non_negative_float(item.get("end_seconds"), f"clip plan line {line_number}: end_seconds")
        if end_seconds <= start_seconds:
            raise ValueError(f"clip plan line {line_number}: end_seconds must be greater than start_seconds")

        output_value = str(item.get("output_path", "")).strip() or f"clips/{clip_id}.mp4"
        output_path = _resolve_under_root(layout["root"], output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        role = str(item.get("role", "")).strip() or None
        notes = str(item.get("notes", "")).strip() or None
        command = build_ffmpeg_extract_command(
            source_path=source,
            output_path=output_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            overwrite=overwrite,
        )
        commands.append(command)
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_asset_id=source_asset_id,
                source_path=str(source),
                output_path=_display_path(layout["root"], output_path),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=end_seconds - start_seconds,
                role=role,
                notes=notes,
            )
        )

        if not dry_run:
            subprocess.run(command, check=True)

    if not dry_run:
        _write_jsonl(output_manifest, [record.to_dict() for record in records])

    return {
        "plan_path": str(plan_path),
        "dry_run": dry_run,
        "clip_count": len(records),
        "output_manifest_path": str(output_manifest),
        "commands": [" ".join(command) for command in commands],
    }


def plan_detective_event_clips(
    *,
    root: str | Path,
    source_clips_path: str | Path,
    clip_plan_output_path: str | Path | None = None,
    clip_groups_output_path: str | Path | None = None,
    max_source_videos: int = 100,
    segment_seconds: float = 8.0,
    min_clip_seconds: float = 3.0,
    max_clip_seconds: float = 15.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    source_clips = list(_load_jsonl(Path(source_clips_path)))
    if not source_clips:
        raise ValueError("source clip manifest is empty")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")
    if max_clip_seconds < min_clip_seconds:
        raise ValueError("max_clip_seconds must be >= min_clip_seconds")
    if segment_seconds < min_clip_seconds or segment_seconds > max_clip_seconds:
        raise ValueError("segment_seconds must stay within min/max clip seconds")

    clip_plan_output = Path(clip_plan_output_path) if clip_plan_output_path else layout["metadata"] / DEFAULT_DETECTIVE_CLIP_PLAN_NAME
    clip_groups_output = Path(clip_groups_output_path) if clip_groups_output_path else layout["metadata"] / DEFAULT_CLIP_GROUPS_NAME

    plan_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    used_source_keys: set[str] = set()
    used_clip_ids: set[str] = set()
    single_segment_records: list[dict[str, Any]] = []
    skipped_count = 0
    probed_count = 0

    for item in source_clips:
        if len(used_source_keys) >= max_source_videos:
            break
        source_path = _source_clip_video_path(layout["root"], item)
        if not source_path.exists():
            skipped_count += 1
            continue
        source_key = str(source_path.resolve())
        if source_key in used_source_keys:
            continue
        used_source_keys.add(source_key)
        media = probe_media(source_path)
        probed_count += 1
        duration = _source_clip_duration_seconds(item, media)
        if duration < min_clip_seconds:
            skipped_count += 1
            continue

        source_clip_id = str(item.get("clip_id", "")).strip() or _stable_hash(source_key)
        dataset = str(item.get("dataset", "unknown")).strip() or "unknown"
        source_group_id = f"group_{dataset}_{_stable_hash(source_key)}"
        segments = _event_segments(
            duration_seconds=duration,
            segment_seconds=segment_seconds,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
        )
        candidate_clip_ids: list[str] = []
        for segment_index, (start_seconds, end_seconds) in enumerate(segments, start=1):
            clip_id = f"{_safe_id(source_clip_id)}__seg_{segment_index:03d}"
            if clip_id in used_clip_ids:
                continue
            used_clip_ids.add(clip_id)
            output_path = f"clips/detective/{dataset}/{clip_id}.mp4"
            record = {
                "clip_id": clip_id,
                "source_path": str(source_path),
                "output_path": output_path,
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "role": "event_clip",
                "notes": "planned by Omni-Detective event segmentation",
                "dataset": dataset,
                "source_clip_id": source_clip_id,
                "group_id": source_group_id,
                "source_row_ids": list(item.get("source_row_ids", [])),
                "text_fields": item.get("text_fields", {}),
                "media_probe": media,
            }
            source_asset_id = str(item.get("source_asset_id", "")).strip()
            if source_asset_id:
                record["source_asset_id"] = source_asset_id
            plan_records.append(record)
            candidate_clip_ids.append(clip_id)

        if len(candidate_clip_ids) >= 2:
            group_records.append(
                {
                    "group_id": source_group_id,
                    "dataset": dataset,
                    "group_reason": "same_source_video",
                    "source_clip_ids": [source_clip_id],
                    "candidate_clip_ids": candidate_clip_ids,
                    "group_tags": _group_tags_from_clip(item),
                    "source_path": _display_source_path(layout["root"], str(source_path)),
                    "media_probe": media,
                }
            )
        elif candidate_clip_ids:
            single_segment_records.append(
                {
                    "dataset": dataset,
                    "clip_id": candidate_clip_ids[0],
                    "source_clip_id": source_clip_id,
                    "tokens": sorted(_group_tokens_from_clip(item)),
                }
            )

    group_records.extend(_semantic_singleton_groups(single_segment_records))
    _write_jsonl(clip_plan_output, plan_records)
    _write_jsonl(clip_groups_output, group_records)
    return {
        "source_clips_path": str(source_clips_path),
        "clip_plan_output_path": str(clip_plan_output),
        "clip_groups_output_path": str(clip_groups_output),
        "source_video_count": len(used_source_keys),
        "probed_count": probed_count,
        "skipped_count": skipped_count,
        "planned_clip_count": len(plan_records),
        "group_count": len(group_records),
        "segment_seconds": segment_seconds,
        "min_clip_seconds": min_clip_seconds,
        "max_clip_seconds": max_clip_seconds,
    }


def select_single_source_video(
    *,
    root: str | Path,
    source_clips_path: str | Path,
    output_path: str | Path | None = None,
    candidates_output_path: str | Path | None = None,
    selection_annotations_path: str | Path | None = None,
    dataset: str = "daily_omni",
    min_duration_seconds: float = 28.0,
    max_duration_seconds: float = 32.0,
    top_k: int = 8,
    max_source_videos_scan: int = 2000,
    max_eligible_candidates: int | None = None,
    selection_mode: str = "local_score",
    random_seed: int | None = None,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    selection_mode = str(selection_mode).strip() or "local_score"
    if selection_mode not in {"local_score", "random", "first"}:
        raise ValueError("selection_mode must be local_score, random, or first")
    layout = ensure_layout(root)
    source_rows = list(_load_jsonl(Path(source_clips_path)))
    if not source_rows:
        raise ValueError("source clip manifest is empty")
    if min_duration_seconds <= 0:
        raise ValueError("min_duration_seconds must be positive")
    if max_duration_seconds < min_duration_seconds:
        raise ValueError("max_duration_seconds must be >= min_duration_seconds")

    output = Path(output_path) if output_path else layout["metadata"] / DEFAULT_SELECTED_SINGLE_SOURCE_NAME
    candidates_output = (
        Path(candidates_output_path)
        if candidates_output_path
        else layout["metadata"] / DEFAULT_SINGLE_SOURCE_CANDIDATES_NAME
    )
    selection_annotations_output = Path(selection_annotations_path) if selection_annotations_path else None

    candidates: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    seen_source_paths: set[str] = set()
    scanned = 0
    probed = 0
    for item in source_rows:
        if scanned >= max(0, max_source_videos_scan):
            break
        scanned += 1
        source_path = _source_clip_video_path(layout["root"], item)
        source_key = str(source_path)
        if not _is_single_source_raw_video_candidate(
            root=layout["root"],
            item=item,
            source_path=source_path,
            dataset=dataset,
        ):
            skipped_reasons["not_daily_omni_raw_video"] += 1
            continue
        if source_key in seen_source_paths:
            skipped_reasons["duplicate_source_path"] += 1
            continue
        seen_source_paths.add(source_key)
        if not source_path.exists():
            skipped_reasons["missing_source_video"] += 1
            continue
        media = probe_media(source_path)
        probed += 1
        duration = _source_clip_duration_seconds(item, media)
        if "error" in media:
            skipped_reasons["probe_error"] += 1
            continue
        if not media.get("has_video"):
            skipped_reasons["missing_video_stream"] += 1
            continue
        if not media.get("has_audio"):
            skipped_reasons["missing_audio_stream"] += 1
            continue
        if duration < min_duration_seconds or duration > max_duration_seconds:
            skipped_reasons["outside_duration_window"] += 1
            continue

        source_clip_id = str(item.get("clip_id", "")).strip() or _stable_hash(str(source_path))
        candidate = {
            "source_clip_id": source_clip_id,
            "dataset": str(item.get("dataset", dataset)).strip() or dataset,
            "source_path": str(source_path),
            "source_path_display": _display_source_path(layout["root"], str(source_path)),
            "duration_seconds": round(duration, 3),
            "media_probe": media,
            "source_row_ids": list(item.get("source_row_ids", [])),
            "text_fields": item.get("text_fields", {}),
            "local_selection_score": _single_source_local_selection_score(item=item, media=media),
            "selection_notes": [
                "daily_omni raw video",
                "28-32s duration",
                "audio and video streams present",
            ],
        }
        source_asset_id = str(item.get("source_asset_id", "")).strip()
        if source_asset_id:
            candidate["source_asset_id"] = source_asset_id
        candidates.append(candidate)
        if max_eligible_candidates is not None and max_eligible_candidates > 0 and len(candidates) >= max_eligible_candidates:
            break

    ordered_candidates = list(candidates)
    if selection_mode == "random":
        random.Random(random_seed).shuffle(ordered_candidates)
    elif selection_mode == "local_score":
        ordered_candidates.sort(
            key=lambda record: (
                -_score_float(record.get("local_selection_score")),
                abs(float(record.get("duration_seconds") or 0.0) - 30.0),
                str(record.get("source_clip_id", "")),
            )
        )
    top_candidates = ordered_candidates[: max(1, top_k)]
    selection_annotations: list[dict[str, Any]] = []
    selection_method = f"{selection_mode}_local_probe"
    selected = top_candidates[0] if top_candidates else None
    if top_candidates and base_url and model:
        client = OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        selection_method = "omni_detective_top_k"
        best_score = -1.0
        for rank, candidate in enumerate(top_candidates, start=1):
            source_path = Path(str(candidate["source_path"]))
            try:
                normalized, raw_model_output = client.annotate_clip_detective(
                    clip_path=str(source_path),
                    tool_observations=_build_toolbox_observations(source_path),
                )
                annotation = _single_source_selection_annotation(candidate, normalized, raw_model_output)
                selection_score = _single_source_omni_selection_score(annotation)
                annotation["selection_rank_before_omni"] = rank
                annotation["omni_selection_score"] = selection_score
                selection_annotations.append(annotation)
                candidate["omni_selection_score"] = selection_score
                candidate["omni_summary"] = annotation.get("summary", "")
                candidate["omni_selection_reasons"] = _single_source_selection_reasons(annotation)
                if selection_score > best_score:
                    best_score = selection_score
                    selected = candidate
            except Exception as exc:
                candidate["omni_selection_error"] = f"{type(exc).__name__}: {exc}"
        top_candidates.sort(
            key=lambda record: (
                -_score_float(record.get("omni_selection_score", record.get("local_selection_score"))),
                -_score_float(record.get("local_selection_score")),
                str(record.get("source_clip_id", "")),
            )
        )
        selected = top_candidates[0]

    if not selected:
        raise ValueError(
            "no eligible single-source daily_omni raw video found; require 28-32s, audio, video, and local file presence"
        )

    selected_record = dict(selected)
    selected_record["selection_method"] = selection_method
    selected_record["selection_top_k"] = len(top_candidates)
    selected_record["selection_mode"] = selection_mode
    selected_record["random_seed"] = random_seed
    selected_record["selection_constraints"] = {
        "dataset": dataset,
        "min_duration_seconds": min_duration_seconds,
        "max_duration_seconds": max_duration_seconds,
        "requires_audio": True,
        "requires_video": True,
        "requires_raw_daily_omni_video": True,
    }
    _write_jsonl(candidates_output, top_candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected_record, ensure_ascii=False, indent=2), encoding="utf-8")
    if selection_annotations_output is not None:
        _write_jsonl(selection_annotations_output, selection_annotations)
    return {
        "source_clips_path": str(source_clips_path),
        "output_path": str(output),
        "candidates_output_path": str(candidates_output),
        "selection_annotations_path": str(selection_annotations_output or ""),
        "scanned_count": scanned,
        "probed_count": probed,
        "eligible_count": len(candidates),
        "top_k_count": len(top_candidates),
        "max_eligible_candidates": max_eligible_candidates,
        "selection_mode": selection_mode,
        "random_seed": random_seed,
        "selected_source_clip_id": selected_record["source_clip_id"],
        "selected_source_path": selected_record["source_path"],
        "selection_method": selection_method,
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_single_source_clips(
    *,
    root: str | Path,
    selected_source_path: str | Path,
    clip_plan_output_path: str | Path | None = None,
    clip_groups_output_path: str | Path | None = None,
    whole_manifest_output_path: str | Path | None = None,
    segment_seconds: float = 5.0,
    min_clip_seconds: float = 3.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    selected = json.loads(Path(selected_source_path).read_text(encoding="utf-8"))
    source_path = Path(str(selected.get("source_path", "")).strip())
    if not source_path.exists():
        raise FileNotFoundError(f"selected source video not found: {source_path}")
    media = selected.get("media_probe") if isinstance(selected.get("media_probe"), dict) else probe_media(source_path)
    duration = float(selected.get("duration_seconds") or media.get("duration_seconds") or 0.0)
    if duration <= 0:
        raise ValueError("selected source video duration is unavailable")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")

    clip_plan_output = Path(clip_plan_output_path) if clip_plan_output_path else layout["metadata"] / DEFAULT_SINGLE_SOURCE_CLIP_PLAN_NAME
    clip_groups_output = Path(clip_groups_output_path) if clip_groups_output_path else layout["metadata"] / DEFAULT_SINGLE_SOURCE_CLIP_GROUPS_NAME
    whole_manifest_output = Path(whole_manifest_output_path) if whole_manifest_output_path else layout["metadata"] / DEFAULT_SINGLE_SOURCE_WHOLE_MANIFEST_NAME

    source_clip_id = str(selected.get("source_clip_id", "")).strip() or _stable_hash(str(source_path))
    dataset = str(selected.get("dataset", "daily_omni")).strip() or "daily_omni"
    safe_source_id = _safe_id(source_clip_id)
    window_start = _optional_float(selected.get("source_window_start_seconds")) or 0.0
    window_duration = _optional_float(selected.get("source_window_duration_seconds"))
    if window_duration is None or window_duration <= 0:
        window_duration = max(0.0, duration - window_start)
    window_end = window_start + window_duration
    if window_start < 0:
        raise ValueError("source_window_start_seconds must be non-negative")
    if window_duration <= 0:
        raise ValueError("source_window_duration_seconds must be positive")
    if window_end > duration + 0.05:
        raise ValueError(
            "selected source window exceeds media duration: "
            f"start={window_start:.3f} duration={window_duration:.3f} media_duration={duration:.3f}"
        )
    segments = _fixed_single_source_segments(
        duration_seconds=window_duration,
        segment_seconds=segment_seconds,
        min_clip_seconds=min_clip_seconds,
    )
    if len(segments) < 4:
        raise ValueError(f"single-source pilot needs at least 4 valid segments; planned={len(segments)}")

    plan_records: list[dict[str, Any]] = []
    candidate_clip_ids: list[str] = []
    for segment_index, (start_seconds, end_seconds) in enumerate(segments, start=1):
        absolute_start_seconds = round(window_start + start_seconds, 3)
        absolute_end_seconds = round(window_start + end_seconds, 3)
        clip_id = f"{safe_source_id}__single_{segment_index:03d}"
        candidate_clip_ids.append(clip_id)
        plan_records.append(
            {
                "clip_id": clip_id,
                "source_path": str(source_path),
                "output_path": f"clips/single_source/{safe_source_id}/{clip_id}.mp4",
                "start_seconds": absolute_start_seconds,
                "end_seconds": absolute_end_seconds,
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "role": "single_source_segment",
                "notes": f"fixed {segment_seconds:g}s single-source Omni pair segment",
                "dataset": dataset,
                "source_clip_id": source_clip_id,
                "source_window_start_seconds": round(window_start, 3),
                "source_window_duration_seconds": round(window_duration, 3),
                "relative_start_seconds": round(start_seconds, 3),
                "relative_end_seconds": round(end_seconds, 3),
                "group_id": f"single_source_{safe_source_id}",
                "source_row_ids": list(selected.get("source_row_ids", [])),
                "text_fields": selected.get("text_fields", {}),
                "media_probe": media,
            }
        )

    group_record = {
        "group_id": f"single_source_{safe_source_id}",
        "dataset": dataset,
        "group_reason": "single_source_video",
        "source_clip_ids": [source_clip_id],
        "candidate_clip_ids": candidate_clip_ids,
        "group_tags": ["single_source", dataset, "fixed_segments"],
        "source_path": _display_source_path(layout["root"], str(source_path)),
        "media_probe": media,
        "segment_seconds": segment_seconds,
        "source_window_start_seconds": round(window_start, 3),
        "source_window_duration_seconds": round(window_duration, 3),
    }
    whole_clip_id = f"{safe_source_id}__whole_window"
    whole_record = {
        "clip_id": whole_clip_id,
        "source_path": str(source_path),
        "output_path": f"clips/single_source/{safe_source_id}/{whole_clip_id}.mp4",
        "start_seconds": round(window_start, 3),
        "end_seconds": round(window_end, 3),
        "duration_seconds": round(window_duration, 3),
        "role": "single_source_whole_video",
        "notes": "30s single-source window for global Omni description",
        "dataset": dataset,
        "source_clip_id": source_clip_id,
        "source_window_start_seconds": round(window_start, 3),
        "source_window_duration_seconds": round(window_duration, 3),
        "group_id": f"single_source_{safe_source_id}",
        "source_row_ids": list(selected.get("source_row_ids", [])),
        "text_fields": selected.get("text_fields", {}),
        "media_probe": media,
    }
    _write_jsonl(clip_plan_output, plan_records)
    _write_jsonl(clip_groups_output, [group_record])
    _write_jsonl(whole_manifest_output, [whole_record])
    return {
        "selected_source_path": str(selected_source_path),
        "clip_plan_output_path": str(clip_plan_output),
        "clip_groups_output_path": str(clip_groups_output),
        "whole_manifest_output_path": str(whole_manifest_output),
        "source_clip_id": source_clip_id,
        "source_path": str(source_path),
        "source_window_start_seconds": round(window_start, 3),
        "source_window_duration_seconds": round(window_duration, 3),
        "segment_count": len(plan_records),
        "pair_count": len(plan_records) * (len(plan_records) - 1) // 2,
        "segment_seconds": segment_seconds,
        "min_clip_seconds": min_clip_seconds,
    }


def mine_single_source_pairs(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    clip_groups_path: str | Path,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> dict[str, Any]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    groups_path = Path(clip_groups_path)
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_SINGLE_SOURCE_PAIR_CANDIDATES_NAME
    report = Path(report_path) if report_path else layout["reports"] / DEFAULT_SINGLE_SOURCE_PAIR_REPORT_NAME
    annotations = list(_load_jsonl(annotations_path))
    groups = list(_load_jsonl(groups_path))
    if not annotations:
        raise ValueError("clip annotations are empty")
    if not groups:
        raise ValueError("clip groups are empty")

    annotations_by_id = {str(item.get("clip_id", "")).strip(): item for item in annotations if str(item.get("clip_id", "")).strip()}
    mined_records: list[dict[str, Any]] = []
    fallback_candidate_count = 0
    usable_group_count = 0
    skipped_group_count = 0
    expected_pair_count = 0
    segment_count = 0
    report_group = groups[0]
    for selected_group in groups:
        candidate_clip_ids = [str(value).strip() for value in selected_group.get("candidate_clip_ids", []) if str(value).strip()]
        ordered_annotations = [
            annotations_by_id[clip_id]
            for clip_id in candidate_clip_ids
            if clip_id in annotations_by_id and not bool(annotations_by_id[clip_id].get("fallback_used"))
        ]
        ordered_annotations.sort(key=lambda item: (_clip_start_seconds(item), str(item.get("clip_id", ""))))
        if len(ordered_annotations) < 4:
            skipped_group_count += 1
            continue
        usable_group_count += 1
        segment_count += len(ordered_annotations)
        expected_pair_count += len(ordered_annotations) * (len(ordered_annotations) - 1) // 2
        group_metadata = {
            "group_id": str(selected_group.get("group_id", "single_source_video")),
            "group_reason": "single_source_video",
        }
        for left_index, reference_annotation in enumerate(ordered_annotations):
            for target_annotation in ordered_annotations[left_index + 1 :]:
                candidate, fallback_used = _single_source_pair_candidate(
                    root=layout["root"],
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    annotations=ordered_annotations,
                    group_metadata=group_metadata,
                    acceptance_profile=acceptance_profile,
                )
                fallback_candidate_count += 1 if fallback_used else 0
                record = _mined_pair_candidate_record(candidate, group_metadata=group_metadata)
                record["candidate_index"] = len(mined_records) + 1
                record["single_source_pair"] = True
                record["chronological_pair"] = True
                record["candidate_stage"] = "enumeration_only"
                record["requires_pair_video_comparison"] = True
                record["reference_start_seconds"] = _clip_start_seconds(reference_annotation)
                record["target_start_seconds"] = _clip_start_seconds(target_annotation)
                if fallback_used:
                    record["risk_flags"] = _dedupe_strings(list(record.get("risk_flags", [])) + ["fallback_single_source_difference"])
                if record.get("difference", {}).get("type") in FINAL_DISABLED_DIFFERENCE_TYPES:
                    record["risk_flags"] = _dedupe_strings(list(record.get("risk_flags", [])) + ["diagnostic_only_final_disabled_type"])
                mined_records.append(record)
    if usable_group_count <= 0:
        raise ValueError("single-source pair mining needs at least one group with 4 usable annotations")

    _write_jsonl(output, mined_records)
    report.write_text(
        _build_single_source_pair_report(
            output_path=output,
            group=report_group,
            annotations=annotations,
            candidates=mined_records,
            fallback_candidate_count=fallback_candidate_count,
            acceptance_profile=acceptance_profile,
        ),
        encoding="utf-8",
    )
    return {
        "clip_annotations_path": str(annotations_path),
        "clip_groups_path": str(groups_path),
        "output_path": str(output),
        "report_path": str(report),
        "segment_count": segment_count,
        "candidate_count": len(mined_records),
        "expected_pair_count": expected_pair_count,
        "fallback_candidate_count": fallback_candidate_count,
        "group_count": len(groups),
        "usable_group_count": usable_group_count,
        "skipped_group_count": skipped_group_count,
        "difference_type_counts": dict(Counter(str(item.get("difference", {}).get("type", "")) for item in mined_records)),
        "acceptance_profile": acceptance_profile,
    }


def propose_single_source_pairs(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    pair_candidates_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    accepted_output_path: str | Path | None = None,
    whole_annotation_path: str | Path | None = None,
    timeout_seconds: float = 180.0,
    max_accepted_pairs: int = 5,
    max_proposals: int | None = None,
    zero_accepted_stop_after: int = DEFAULT_ZERO_ACCEPTED_STOP_AFTER,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
    audio_dataset_line: str = STANDARD_AUDIO_DATASET_LINE,
    accepted_progress_path: str | Path | None = None,
    rejected_progress_path: str | Path | None = None,
    omni_retries: int = 0,
    fail_on_transient_omni_errors: bool = False,
) -> dict[str, Any]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    audio_dataset_line = _normalize_audio_dataset_line(audio_dataset_line)
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    candidates_path = Path(pair_candidates_path)
    output = Path(output_path) if output_path else layout["pairs"] / "ranked_single_source_pairs.jsonl"
    accepted_output = Path(accepted_output_path) if accepted_output_path else layout["pairs"] / DEFAULT_ACCEPTED_PAIRS_NAME
    accepted_progress_output = Path(accepted_progress_path) if accepted_progress_path else None
    rejected_progress_output = Path(rejected_progress_path) if rejected_progress_path else None
    annotations = list(_load_jsonl(annotations_path))
    candidates = list(_load_jsonl(candidates_path))
    if not annotations:
        raise ValueError("single-source annotations are empty")
    if not candidates:
        raise ValueError("single-source pair candidates are empty")

    whole_annotations = list(_load_jsonl(Path(whole_annotation_path))) if whole_annotation_path else []
    whole_annotation = whole_annotations[0] if whole_annotations else {}
    annotations_by_id = {
        str(item.get("clip_id", "")).strip(): item
        for item in annotations
        if str(item.get("clip_id", "")).strip()
    }
    existing_records = _load_records_by_key(output, "proposal_id")
    if not output.exists():
        _write_jsonl(output, [])
    if not accepted_output.exists():
        _write_jsonl(accepted_output, [])

    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            int(item.get("candidate_index") or 0),
            str(item.get("candidate_id", "")),
        ),
    )
    output_records: list[dict[str, Any]] = []
    accepted_total_count = 0
    rejected_count = 0
    fallback_count = 0
    early_stop_reason = ""
    omni_retries = max(0, int(omni_retries or 0))

    def persist_progress() -> None:
        current_accepted = _select_single_source_quality_passed_records(output_records)
        _write_jsonl(output, output_records)
        _write_jsonl(accepted_output, current_accepted)

    for candidate in ordered_candidates:
        if max_proposals is not None and len(output_records) >= max_proposals:
            break
        reference_clip_id = str(candidate.get("reference_clip_id", "")).strip()
        target_clip_id = str(candidate.get("target_clip_id", "")).strip()
        reference_annotation = annotations_by_id.get(reference_clip_id)
        target_annotation = annotations_by_id.get(target_clip_id)
        if reference_annotation is None or target_annotation is None:
            continue
        reference_video = str(candidate.get("reference_video") or reference_annotation.get("output_path", "")).strip()
        target_video = str(candidate.get("target_video") or target_annotation.get("output_path", "")).strip()
        proposal_id = str(candidate.get("proposal_id") or candidate.get("candidate_id") or _build_proposal_id(reference_video, target_video))
        print(
            "[propose-single-source-pairs] start "
            f"proposal_index={len(output_records) + 1} proposal_id={proposal_id}",
            file=sys.stderr,
            flush=True,
        )
        if proposal_id in existing_records:
            record = dict(existing_records[proposal_id])
            record = _recheck_existing_single_source_pair_record(record, acceptance_profile=acceptance_profile)
        else:
            reference_path = _resolve_under_root(layout["root"], reference_video)
            target_path = _resolve_under_root(layout["root"], target_video)
            if (
                audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE
                and _is_b_audio_blind_review_profile(acceptance_profile)
                and proposal_id not in existing_records
            ):
                fallback_used = False
                raw_model_output: dict[str, Any] = {}
                raw_final_omni_output: dict[str, Any] = {}
                audio_only_proposal: dict[str, Any] = {}
                raw_audio_only_proposal: dict[str, Any] = {}
                audio_only_verification: dict[str, Any] = {}
                raw_audio_only_verification: dict[str, Any] = {}
                full_av_consistency: dict[str, Any] = {}
                raw_full_av_consistency: dict[str, Any] = {}
                extraction_error = ""
                reference_audio_path = target_audio_path = Path()
                try:
                    cache_dir = output.parent / "audio_only_cache"
                    reference_audio_path = _extract_audio_only_cache(
                        video_path=reference_path,
                        cache_dir=cache_dir,
                        clip_id=reference_clip_id,
                    )
                    target_audio_path = _extract_audio_only_cache(
                        video_path=target_path,
                        cache_dir=cache_dir,
                        clip_id=target_clip_id,
                    )
                except Exception as exc:
                    extraction_error = f"missing_audio_track: {type(exc).__name__}: {exc}"

                quality_seed = {
                    "difference": {"type": "speech"},
                    "edit_text": "",
                    "modalities": ["audio"],
                    "confidence": 0.0,
                    "accept": False,
                }
                quality = _single_source_pair_quality(
                    candidate=candidate,
                    model_fields=quality_seed,
                    acceptance_profile=acceptance_profile,
                )
                context_type = _b_line_video_context_type(reference_annotation, target_annotation)
                video_context_strength = _b_line_video_context_strength(reference_annotation, target_annotation, quality)
                asr_degeneracy_risk = _b_line_asr_degeneracy_risk(reference_annotation, target_annotation, quality)
                quality.update(
                    {
                        "b_audio_blind_review": True,
                        "video_context_type": context_type,
                        "video_context_strength": video_context_strength,
                        "asr_degeneracy_risk": asr_degeneracy_risk,
                    }
                )
                blind_local_issues: list[str] = []
                if extraction_error:
                    blind_local_issues.append(extraction_error)
                if video_context_strength < _profile_threshold(B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE, "video_context_strength"):
                    blind_local_issues.append("blind_review_video_context_too_weak")
                if asr_degeneracy_risk > _profile_threshold(B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE, "asr_degeneracy_risk"):
                    blind_local_issues.append("blind_review_asr_degeneracy_risk_too_high")
                if context_type in {"asr_only", "generic_talking_head"}:
                    blind_local_issues.append(f"blind_review_asr_degeneracy_risk_too_high: context_type={context_type}")
                local_gate_report = {
                    "passed": not blind_local_issues,
                    "hard_reject": blind_local_issues,
                    "review_required": [],
                    "difference_type": "audio_only_pending",
                    "confidence": 0.0,
                    "acceptance_profile": acceptance_profile,
                    "audio_dataset_line": audio_dataset_line,
                    "reference_video_exists": reference_path.exists(),
                    "target_video_exists": target_path.exists(),
                    "visual_context_type": context_type,
                    "video_context_strength": video_context_strength,
                    "asr_degeneracy_risk": asr_degeneracy_risk,
                }

                model_fields = _b_audio_blind_review_model_fields({})
                if not blind_local_issues:
                    try:
                        audio_only_proposal, raw_audio_only_proposal = _call_omni_with_retries(
                            label=f"audio_only_proposal:{proposal_id}",
                            retries=omni_retries,
                            fail_on_transient=fail_on_transient_omni_errors,
                            func=lambda: client.propose_b_line_audio_only_pair(
                                reference_audio_path=str(reference_audio_path),
                                target_audio_path=str(target_audio_path),
                                metadata={
                                    "proposal_id": proposal_id,
                                    "reference_clip_id": reference_clip_id,
                                    "target_clip_id": target_clip_id,
                                },
                            ),
                        )
                        model_fields = _b_audio_blind_review_model_fields(audio_only_proposal)
                        audio_only_verification, raw_audio_only_verification = _call_omni_with_retries(
                            label=f"audio_only_verify:{proposal_id}",
                            retries=omni_retries,
                            fail_on_transient=fail_on_transient_omni_errors,
                            func=lambda: client.verify_b_line_audio_only_edit(
                                reference_audio_path=str(reference_audio_path),
                                target_audio_path=str(target_audio_path),
                                edit_text=str(model_fields.get("edit_text", "")).strip(),
                                audio_only_proposal=audio_only_proposal,
                            ),
                        )
                        full_av_consistency, raw_full_av_consistency = _call_omni_with_retries(
                            label=f"full_av_consistency:{proposal_id}",
                            retries=omni_retries,
                            fail_on_transient=fail_on_transient_omni_errors,
                            func=lambda: client.verify_b_line_full_av_consistency(
                                reference_clip_path=str(reference_path),
                                target_clip_path=str(target_path),
                                edit_text=str(model_fields.get("edit_text", "")).strip(),
                                audio_only_evidence={
                                    "proposal": audio_only_proposal,
                                    "verification": audio_only_verification,
                                },
                                local_gate_report=local_gate_report,
                            ),
                        )
                    except Exception as exc:
                        if fail_on_transient_omni_errors and _is_transient_omni_exception(exc):
                            print(
                                "[propose-single-source-pairs] transient blind review error; shard will fail for retry "
                                f"proposal_id={proposal_id} error={type(exc).__name__}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            raise
                        blind_local_issues.append(f"blind_review_error: {type(exc).__name__}: {exc}")
                        local_gate_report["passed"] = False
                        local_gate_report["hard_reject"] = blind_local_issues
                        raw_audio_only_proposal = raw_audio_only_proposal or {"error": f"{type(exc).__name__}: {exc}"}

                difference = dict(model_fields.get("difference") or {})
                difference_type = str(difference.get("type", "")).strip()
                confidence = _score_float(model_fields.get("confidence"))
                quality.update(
                    _single_source_pair_quality(
                        candidate=candidate,
                        model_fields=model_fields,
                        acceptance_profile=acceptance_profile,
                    )
                )
                quality.update(
                    {
                        "b_audio_blind_review": True,
                        "video_context_type": context_type,
                        "video_context_strength": video_context_strength,
                        "asr_degeneracy_risk": asr_degeneracy_risk,
                    }
                )
                blocking_issues = _dedupe_strings(
                    blind_local_issues
                    + _b_audio_blind_review_issues(
                        audio_only_proposal=audio_only_proposal,
                        audio_only_verification=audio_only_verification,
                        full_av_consistency=full_av_consistency,
                        quality=quality,
                    )
                )
                accepted = not blocking_issues
                final_omni_accept = accepted
                reject_reason = "" if accepted else "; ".join(blocking_issues)
                final_omni_verification = {
                    "accept": accepted,
                    "confidence": min(
                        _score_float(audio_only_proposal.get("confidence")),
                        _score_float(audio_only_verification.get("confidence")),
                        _score_float(full_av_consistency.get("confidence")),
                    )
                    if accepted
                    else 0.0,
                    "quality_score": min(
                        _score_float(audio_only_proposal.get("confidence")),
                        _score_float(audio_only_verification.get("confidence")),
                        _score_float(full_av_consistency.get("confidence")),
                    )
                    if accepted
                    else 0.0,
                    "reference_satisfies_edit": _boolish(audio_only_verification.get("reference_satisfies_edit")),
                    "target_satisfies_edit": _boolish(audio_only_verification.get("target_satisfies_edit")),
                    "observable_delta": _boolish(audio_only_verification.get("audio_difference_specific")),
                    "single_primary_delta": True,
                    "text_or_ocr_driven": False,
                    "segment_wide": True,
                    "edit_text_accurate": _boolish(audio_only_verification.get("accept")),
                    "main_reject_reason": reject_reason,
                    "evidence": _dedupe_strings(
                        _normalize_list(audio_only_proposal.get("evidence", []))
                        + _normalize_list(audio_only_verification.get("evidence", []))
                        + _normalize_list(full_av_consistency.get("evidence", []))
                    ),
                    "recommended_edit_text": "",
                    "audio_primary": True,
                    "visual_locked": _boolish(full_av_consistency.get("visual_context_preserved")),
                    "visual_too_different_for_B": _boolish(full_av_consistency.get("visual_shortcut_risk")),
                    "edit_text_audio_only": _boolish(audio_only_verification.get("edit_text_audio_only")),
                    "visual_context_preserved": _boolish(full_av_consistency.get("visual_context_preserved")),
                    "video_context_strength": video_context_strength,
                    "asr_degeneracy_risk": asr_degeneracy_risk,
                    "not_asr_only": asr_degeneracy_risk <= _profile_threshold(B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE, "asr_degeneracy_risk"),
                }
                hard_negative_paths = _single_source_hard_negative_paths(
                    root=layout["root"],
                    candidate=candidate,
                    annotations=annotations,
                    reference_clip_id=reference_clip_id,
                    target_clip_id=target_clip_id,
                )
                judge = {
                    "reference_satisfies_edit": _boolish(audio_only_verification.get("reference_satisfies_edit")),
                    "target_satisfies_edit": accepted,
                    "single_main_difference": accepted,
                    "same_context_score": quality["same_context_score"],
                    "edit_match_score": quality["edit_match_score"],
                    "target_uniqueness_score": quality["target_uniqueness_score"],
                    "audio_required": True,
                    "hard_negative_quality": "weak",
                    "accept": accepted,
                    "reject_reason": reject_reason,
                }
                verification = _single_source_pair_verification(model_fields, accepted=accepted, reject_reason=reject_reason)
                source = {
                    "platform": str(target_annotation.get("dataset") or reference_annotation.get("dataset") or "daily_omni"),
                    "url": target_path.resolve().as_uri(),
                    "license_note": DEFAULT_LICENSE_NOTE,
                }
                b_subtype = str(audio_only_proposal.get("b_subtype", "")).strip()
                if b_subtype == "speech_topic":
                    b_subtype = "speech_topic_in_video_context"
                elif b_subtype not in {"music", "sound_event"}:
                    b_subtype = _b_line_subtype_from_evidence(
                        difference_type=difference_type,
                        edit_text=str(model_fields.get("edit_text", "")).strip(),
                        reference_annotation=reference_annotation,
                        target_annotation=target_annotation,
                    )
                record = {
                    "proposal_id": proposal_id,
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "candidate_stage": "audio_only_blind_review",
                    "group_id": str(candidate.get("group_id", "")),
                    "group_reason": str(candidate.get("group_reason", "single_source_video")),
                    "reference_clip_id": reference_clip_id,
                    "target_clip_id": target_clip_id,
                    "reference_video": reference_video,
                    "target_video": target_video,
                    "edit_text": str(model_fields.get("edit_text", "")).strip(),
                    "modalities": ["audio"],
                    "reference_caption": str(audio_only_proposal.get("reference_audio_content", "")).strip(),
                    "target_caption": str(audio_only_proposal.get("target_audio_content", "")).strip(),
                    "difference": difference,
                    "audio_dataset_line": audio_dataset_line,
                    "audio_line_quality_profile": str(quality.get("audio_line_quality_profile", "")).strip(),
                    "audio_only_reference_content": str(audio_only_proposal.get("reference_audio_content", "")).strip(),
                    "audio_only_target_content": str(audio_only_proposal.get("target_audio_content", "")).strip(),
                    "audio_only_edit_text": str(audio_only_proposal.get("edit_text", "")).strip(),
                    "audio_only_accept": _boolish(audio_only_verification.get("accept")),
                    "audio_only_proposal": audio_only_proposal,
                    "audio_only_verification": audio_only_verification,
                    "raw_audio_only_proposal": raw_audio_only_proposal,
                    "raw_audio_only_verification": raw_audio_only_verification,
                    "full_av_consistency": full_av_consistency,
                    "raw_full_av_consistency": raw_full_av_consistency,
                    "visual_shortcut_risk": _boolish(full_av_consistency.get("visual_shortcut_risk")),
                    "full_av_consistency_accept": _boolish(full_av_consistency.get("accept")),
                    "dominant_delta": dict(model_fields.get("dominant_delta", {})),
                    "reference_state": dict(model_fields.get("reference_state", {})) if isinstance(model_fields.get("reference_state"), dict) else {},
                    "target_state": dict(model_fields.get("target_state", {})) if isinstance(model_fields.get("target_state"), dict) else {},
                    "delta_temporal_extent": dict(model_fields.get("delta_temporal_extent", {})) if isinstance(model_fields.get("delta_temporal_extent"), dict) else {},
                    "subject_roles": dict(model_fields.get("subject_roles", {})) if isinstance(model_fields.get("subject_roles"), dict) else {},
                    "is_segment_wide_delta": bool(model_fields.get("is_segment_wide_delta")),
                    "discarded_deltas": list(model_fields.get("discarded_deltas", [])),
                    "pair_video_evidence": list(model_fields.get("evidence", [])),
                    "confidence": confidence,
                    "model_accepted": _boolish(audio_only_proposal.get("accept")),
                    "local_gate_passed": not blind_local_issues,
                    "final_omni_accept": final_omni_accept,
                    "final_accept_source": "audio_only_blind_review",
                    "local_gate_report": local_gate_report,
                    "final_omni_verification": final_omni_verification,
                    "single_source_delta_family": _single_source_delta_family_from_fields(model_fields),
                    "single_source_pair_acceptance_issues": blocking_issues,
                    "single_source_pair_review_required": [],
                    "b_line_edit_text_repaired": False,
                    "b_line_original_edit_text": "",
                    "raw_proposed_edit_text": str(audio_only_proposal.get("edit_text", "")).strip(),
                    "edit_text_refinement": {},
                    "raw_edit_text_refinement": {},
                    "refined_edit_text": "",
                    "edit_text_specificity_score": 0.0,
                    "edit_text_reject_reason": "",
                    "speech_or_audio_evidence": _normalize_list(audio_only_proposal.get("evidence", [])),
                    "speech_rewrite": {},
                    "raw_speech_rewrite": {},
                    "speech_rewrite_refined_edit_text": "",
                    "speech_rewrite_confidence": 0.0,
                    "speech_rewrite_reject_reason": "",
                    "speech_rewrite_used": False,
                    "b_subtype": b_subtype,
                    "video_context_type": context_type,
                    "video_context_strength": video_context_strength,
                    "asr_degeneracy_risk": asr_degeneracy_risk,
                    "speech_role": str(reference_annotation.get("speech_role") or target_annotation.get("speech_role") or "").strip(),
                    "audio_evidence": _dedupe_strings(
                        [
                            str(audio_only_proposal.get("reference_audio_content", "")).strip(),
                            str(audio_only_proposal.get("target_audio_content", "")).strip(),
                            *_normalize_list(audio_only_proposal.get("evidence", [])),
                            *_normalize_list(audio_only_verification.get("evidence", [])),
                        ]
                    ),
                    "visual_context_evidence": _dedupe_strings(
                        [
                            str(reference_annotation.get("scene", "")).strip(),
                            str(target_annotation.get("scene", "")).strip(),
                            str(reference_annotation.get("summary", "")).strip(),
                            str(target_annotation.get("summary", "")).strip(),
                        ]
                    ),
                    "recommended_edit_text": "",
                    "hard_negatives": hard_negative_paths,
                    "quality": quality,
                    "heuristic_quality": dict(candidate.get("quality", {})) if isinstance(candidate.get("quality"), dict) else {},
                    "source_context": {
                        "relation": "same_source_video",
                        "single_source_pair": True,
                        "template_route": "audio_only_blind_review",
                        "score": quality["same_context_score"],
                    },
                    "source": source,
                    "proposal_reason": "audio-only blind review",
                    "evidence": {
                        **_evidence_from_annotations(reference_annotation, target_annotation),
                        "audio_only_proposal": _normalize_list(audio_only_proposal.get("evidence", [])),
                        "audio_only_verification": _normalize_list(audio_only_verification.get("evidence", [])),
                        "full_av_consistency": _normalize_list(full_av_consistency.get("evidence", [])),
                    },
                    "judge": judge,
                    "verification": verification,
                    "edit_text_quality": _edit_text_quality_payload(
                        edit_text=str(model_fields.get("edit_text", "")),
                        difference=difference,
                        modalities=["audio"],
                        reference_caption=str(audio_only_proposal.get("reference_audio_content", "")),
                        target_caption=str(audio_only_proposal.get("target_audio_content", "")),
                    ),
                    "observable_difference": {
                        "passed": accepted,
                        "frame_backed": False,
                        "failure_reason": "" if accepted else reject_reason,
                        "reference_evidence": _normalize_list(audio_only_proposal.get("evidence", [])),
                        "target_evidence": _normalize_list(audio_only_verification.get("evidence", [])),
                        "supporting_fields": ["audio_only_blind_review"],
                    },
                    "dominant_delta_decision": dict(model_fields.get("dominant_delta", {})),
                    "accepted": accepted,
                    "fallback_used": fallback_used,
                    "raw_model_output": raw_audio_only_proposal,
                    "raw_final_omni_output": raw_full_av_consistency,
                    "single_source_pair": True,
                }
                if bool(record.get("accepted")):
                    accepted_total_count += 1
                else:
                    rejected_count += 1
                output_records.append(record)
                _apply_single_source_delta_uniqueness(
                    output_records,
                    max_accepted_pairs=max_accepted_pairs,
                    acceptance_profile=acceptance_profile,
                )
                if bool(record.get("accepted")) and accepted_progress_output is not None:
                    _append_jsonl_record(accepted_progress_output, record)
                if not bool(record.get("accepted")) and rejected_progress_output is not None:
                    _append_jsonl_record(rejected_progress_output, record)
                persist_progress()
                current_accepted = _select_single_source_quality_passed_records(output_records)
                print(
                    "[propose-single-source-pairs] wrote "
                    f"proposal_count={len(output_records)} accepted_current={len(current_accepted)} "
                    f"proposal_id={record.get('proposal_id', '')} "
                    f"accepted={bool(record.get('accepted'))} "
                    f"final_omni_accept={bool(record.get('final_omni_accept'))} "
                    f"final_omni_quality_score={_score_float((record.get('final_omni_verification') or {}).get('quality_score')) if isinstance(record.get('final_omni_verification'), dict) else 0.0:.2f} "
                    f"difference_type={record.get('difference', {}).get('type', '') if isinstance(record.get('difference'), dict) else ''} "
                    f"delta_family={record.get('single_source_delta_family', '')} "
                    f"fallback={bool(record.get('fallback_used'))} "
                    f"issues={';'.join(str(issue) for issue in record.get('single_source_pair_acceptance_issues', []))} "
                    f"edit_text={str(record.get('edit_text', '')).replace(chr(10), ' ')[:180]}",
                    file=sys.stderr,
                    flush=True,
                )
                if (
                    zero_accepted_stop_after
                    and zero_accepted_stop_after > 0
                    and len(output_records) >= zero_accepted_stop_after
                    and not current_accepted
                ):
                    early_stop_reason = (
                        f"zero accepted after {len(output_records)} single-source pair comparisons; "
                        "inspect selected source, segment captions, or pair-level Omni output"
                    )
                    break
                continue
            raw_model_output: dict[str, Any]
            fallback_used = False
            try:
                model_fields, raw_model_output = _call_omni_with_retries(
                    label=f"proposal:{proposal_id}",
                    retries=omni_retries,
                    fail_on_transient=fail_on_transient_omni_errors,
                    func=lambda: client.propose_single_source_pair(
                        reference_clip_path=str(reference_path),
                        target_clip_path=str(target_path),
                        reference_annotation=_single_source_line_annotation_prompt_view(reference_annotation, audio_dataset_line),
                        target_annotation=_single_source_line_annotation_prompt_view(target_annotation, audio_dataset_line),
                        whole_annotation=_single_source_whole_prompt_view(whole_annotation) if whole_annotation else None,
                        candidate=_single_source_candidate_prompt_view(candidate),
                        audio_dataset_line=audio_dataset_line,
                    ),
                )
            except Exception as exc:
                if fail_on_transient_omni_errors and _is_transient_omni_exception(exc):
                    print(
                        "[propose-single-source-pairs] transient omni error; shard will fail for retry "
                        f"proposal_id={proposal_id} error={type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                print(
                    "[propose-single-source-pairs] fallback after omni proposal error "
                    f"proposal_id={proposal_id} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                model_fields = _single_source_rejected_model_fields(
                    candidate=candidate,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                fallback_used = True

            model_fields = _repair_single_source_audio_line_model_fields(
                model_fields=model_fields,
                audio_dataset_line=audio_dataset_line,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            difference = dict(model_fields.get("difference") or {})
            difference_type = str(difference.get("type", "")).strip()
            confidence = _score_float(model_fields.get("confidence"))
            hard_negative_paths = _single_source_hard_negative_paths(
                root=layout["root"],
                candidate=candidate,
                annotations=annotations,
                reference_clip_id=reference_clip_id,
                target_clip_id=target_clip_id,
            )
            quality = _single_source_pair_quality(
                candidate=candidate,
                model_fields=model_fields,
                acceptance_profile=acceptance_profile,
            )
            if audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE and _is_b_audio_context_cvr_profile(acceptance_profile):
                context_type = _b_line_video_context_type(reference_annotation, target_annotation)
                video_context_strength = _b_line_video_context_strength(reference_annotation, target_annotation, quality)
                asr_degeneracy_risk = _b_line_asr_degeneracy_risk(reference_annotation, target_annotation, quality)
                quality.update(
                    {
                        "b_context_cvr": True,
                        "video_context_type": context_type,
                        "video_context_strength": video_context_strength,
                        "asr_degeneracy_risk": asr_degeneracy_risk,
                    }
                )
            edit_text_quality = _edit_text_quality_payload(
                edit_text=str(model_fields.get("edit_text", "")),
                difference=difference,
                modalities=list(model_fields.get("modalities", [])),
                reference_caption=str(model_fields.get("reference_caption", "")),
                target_caption=str(model_fields.get("target_caption", "")),
            )
            acceptance_issues = _single_source_pair_acceptance_issues(
                model_fields=model_fields,
                edit_text_quality=edit_text_quality,
                acceptance_profile=acceptance_profile,
                audio_dataset_line=audio_dataset_line,
                candidate_quality=quality,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            local_gate_report = _single_source_local_gate_report(
                acceptance_issues=acceptance_issues,
                fallback_used=fallback_used,
                difference_type=difference_type,
                confidence=confidence,
                acceptance_profile=acceptance_profile,
                audio_dataset_line=audio_dataset_line,
                reference_video_exists=reference_path.exists(),
                target_video_exists=target_path.exists(),
            )
            local_hard_rejects = list(local_gate_report.get("hard_reject", []))
            model_accepted = (
                bool(model_fields.get("accept"))
                and bool(local_gate_report.get("passed"))
                and not fallback_used
            )
            should_run_final_omni = bool(model_accepted)
            if not should_run_final_omni:
                should_run_final_omni = _a_line_can_run_final_rescue(
                    audio_dataset_line=audio_dataset_line,
                    model_fields=model_fields,
                    fallback_used=fallback_used,
                    reference_video_exists=reference_path.exists(),
                    target_video_exists=target_path.exists(),
                ) or _b_line_can_run_final_rescue(
                    audio_dataset_line=audio_dataset_line,
                    model_fields=model_fields,
                    fallback_used=fallback_used,
                    reference_video_exists=reference_path.exists(),
                    target_video_exists=target_path.exists(),
                )
            raw_final_omni_output: dict[str, Any] = {}
            if should_run_final_omni:
                try:
                    final_omni_verification, raw_final_omni_output = _call_omni_with_retries(
                        label=f"final_verify:{proposal_id}",
                        retries=omni_retries,
                        fail_on_transient=fail_on_transient_omni_errors,
                        func=lambda: client.verify_single_source_pair_final(
                            reference_clip_path=str(reference_path),
                            target_clip_path=str(target_path),
                            model_fields=model_fields,
                            reference_annotation=_single_source_line_annotation_prompt_view(reference_annotation, audio_dataset_line),
                            target_annotation=_single_source_line_annotation_prompt_view(target_annotation, audio_dataset_line),
                            local_gate_report=local_gate_report,
                            whole_annotation=_single_source_whole_prompt_view(whole_annotation) if whole_annotation else None,
                            audio_dataset_line=audio_dataset_line,
                        ),
                    )
                except Exception as exc:
                    if fail_on_transient_omni_errors and _is_transient_omni_exception(exc):
                        print(
                            "[propose-single-source-pairs] transient final verify error; shard will fail for retry "
                            f"proposal_id={proposal_id} error={type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise
                    final_omni_verification = _single_source_skipped_final_verification(
                        f"final_omni_verification_error: {type(exc).__name__}: {exc}"
                    )
                    raw_final_omni_output = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                skip_reason = "initial_pair_model_rejected"
                if local_hard_rejects:
                    skip_reason = "; ".join(local_hard_rejects)
                elif not bool(model_fields.get("accept")):
                    skip_reason = str(model_fields.get("reject_reason", "")).strip() or skip_reason
                final_omni_verification = _single_source_skipped_final_verification(skip_reason)
            final_issues = _single_source_final_verification_issues(
                final_omni_verification,
                acceptance_profile=acceptance_profile,
                audio_dataset_line=audio_dataset_line,
                model_fields=model_fields,
            )
            final_review_required = _single_source_final_verification_review_required(
                final_omni_verification,
                acceptance_profile=acceptance_profile,
                audio_dataset_line=audio_dataset_line,
            )
            local_review_required = list(local_gate_report.get("review_required", []))
            blocking_local_hard_rejects = list(local_hard_rejects)
            if audio_dataset_line == VISUAL_AUDIO_ANCHOR_LINE and not final_issues:
                blocking_local_hard_rejects = _a_line_unrescued_local_hard_rejects(
                    blocking_local_hard_rejects,
                    final_omni_verification,
                )
            elif audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE and not final_issues:
                blocking_local_hard_rejects = _b_line_unrescued_local_hard_rejects(
                    blocking_local_hard_rejects,
                    final_omni_verification,
                )
                if not blocking_local_hard_rejects and "audio" not in list(model_fields.get("modalities", [])):
                    model_fields["modalities"] = list(model_fields.get("modalities", [])) + ["audio"]
            blocking_issues = _dedupe_strings(
                blocking_local_hard_rejects + (local_review_required if final_issues else []) + final_issues
            )
            final_omni_accept = bool(should_run_final_omni and not final_issues and _boolish(final_omni_verification.get("accept")))
            accepted = bool(final_omni_accept and not blocking_issues)
            edit_text_refinement: dict[str, Any] = {}
            raw_edit_text_refinement: dict[str, Any] = {}
            speech_rewrite: dict[str, Any] = {}
            raw_speech_rewrite: dict[str, Any] = {}
            speech_rewrite_used = False
            if accepted and audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE and _is_b_audio_review_profile(acceptance_profile):
                try:
                    edit_text_refinement, raw_edit_text_refinement = _call_omni_with_retries(
                        label=f"edit_refine:{proposal_id}",
                        retries=omni_retries,
                        fail_on_transient=fail_on_transient_omni_errors,
                        func=lambda: client.refine_b_line_edit_text(
                            reference_clip_path=str(reference_path),
                            target_clip_path=str(target_path),
                            model_fields=model_fields,
                            final_verification=final_omni_verification,
                            reference_annotation=_single_source_line_annotation_prompt_view(reference_annotation, audio_dataset_line),
                            target_annotation=_single_source_line_annotation_prompt_view(target_annotation, audio_dataset_line),
                        ),
                    )
                except Exception as exc:
                    if fail_on_transient_omni_errors and _is_transient_omni_exception(exc):
                        print(
                            "[propose-single-source-pairs] transient edit refinement error; shard will fail for retry "
                            f"proposal_id={proposal_id} error={type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise
                    edit_text_refinement = {
                        "refined_edit_text": "",
                        "edit_text_specificity_score": 0.0,
                        "reject_if_unspecific": True,
                        "edit_text_reject_reason": f"edit_text_refinement_error: {type(exc).__name__}: {exc}",
                        "speech_or_audio_evidence": [],
                    }
                    raw_edit_text_refinement = {"error": f"{type(exc).__name__}: {exc}"}
                refinement_issues = _b_line_edit_text_refinement_issues(edit_text_refinement)
                should_run_speech_rewrite = _b_line_should_run_speech_rewrite(
                    model_fields=model_fields,
                    final_verification=final_omni_verification,
                    edit_text_refinement=edit_text_refinement,
                    refinement_issues=refinement_issues,
                )
                if should_run_speech_rewrite:
                    try:
                        speech_rewrite, raw_speech_rewrite = _call_omni_with_retries(
                            label=f"speech_rewrite:{proposal_id}",
                            retries=omni_retries,
                            fail_on_transient=fail_on_transient_omni_errors,
                            func=lambda: client.refine_b_line_speech_content(
                                reference_clip_path=str(reference_path),
                                target_clip_path=str(target_path),
                                model_fields=model_fields,
                                final_verification=final_omni_verification,
                                edit_text_refinement=edit_text_refinement,
                                reference_annotation=_single_source_line_annotation_prompt_view(reference_annotation, audio_dataset_line),
                                target_annotation=_single_source_line_annotation_prompt_view(target_annotation, audio_dataset_line),
                            ),
                        )
                    except Exception as exc:
                        if fail_on_transient_omni_errors and _is_transient_omni_exception(exc):
                            print(
                                "[propose-single-source-pairs] transient speech rewrite error; shard will fail for retry "
                                f"proposal_id={proposal_id} error={type(exc).__name__}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            raise
                        speech_rewrite = {
                            "reference_speech_content": "",
                            "target_speech_content": "",
                            "speech_transcription_confidence": 0.0,
                            "speech_language": "",
                            "refined_edit_text": "",
                            "reject_if_still_unclear": True,
                            "speech_rewrite_reject_reason": f"speech_rewrite_error: {type(exc).__name__}: {exc}",
                        }
                        raw_speech_rewrite = {"error": f"{type(exc).__name__}: {exc}"}
                    speech_rewrite_issues = _b_line_speech_rewrite_issues(speech_rewrite)
                    if speech_rewrite_issues:
                        accepted = False
                        final_omni_accept = False
                        blocking_issues = _dedupe_strings(blocking_issues + refinement_issues + speech_rewrite_issues)
                    else:
                        refined_edit = str(speech_rewrite.get("refined_edit_text", "")).strip()
                        if refined_edit:
                            if not model_fields.get("b_line_original_edit_text"):
                                model_fields["b_line_original_edit_text"] = str(model_fields.get("edit_text", "")).strip()
                            model_fields["edit_text"] = refined_edit
                            speech_rewrite_used = True
                elif refinement_issues:
                    accepted = False
                    final_omni_accept = False
                    blocking_issues = _dedupe_strings(blocking_issues + refinement_issues)
                else:
                    refined_edit = str(edit_text_refinement.get("refined_edit_text", "")).strip()
                    if refined_edit:
                        if not model_fields.get("b_line_original_edit_text"):
                            model_fields["b_line_original_edit_text"] = str(model_fields.get("edit_text", "")).strip()
                        model_fields["edit_text"] = refined_edit
            reject_reason = str(model_fields.get("reject_reason", "")).strip()
            if not accepted:
                reject_reason = "; ".join([item for item in [reject_reason, *blocking_issues] if item]).strip()
                reject_reason = reject_reason or "single-source pair model rejected"
            judge = {
                "reference_satisfies_edit": False,
                "target_satisfies_edit": accepted,
                "single_main_difference": accepted,
                "same_context_score": quality["same_context_score"],
                "edit_match_score": quality["edit_match_score"],
                "target_uniqueness_score": quality["target_uniqueness_score"],
                "audio_required": "audio" in list(model_fields.get("modalities", [])),
                "hard_negative_quality": "weak",
                "accept": accepted,
                "reject_reason": reject_reason,
            }
            verification = _single_source_pair_verification(model_fields, accepted=accepted, reject_reason=reject_reason)
            source = {
                "platform": str(target_annotation.get("dataset") or reference_annotation.get("dataset") or "daily_omni"),
                "url": _resolve_under_root(layout["root"], target_video).resolve().as_uri(),
                "license_note": DEFAULT_LICENSE_NOTE,
            }
            source_context = {
                "relation": "same_source_video",
                "single_source_pair": True,
                "template_route": "single_source_pair_video_comparison",
                "score": quality["same_context_score"],
            }
            b_subtype = ""
            video_context_type = str(quality.get("video_context_type", "")).strip()
            video_context_strength = _score_float(quality.get("video_context_strength"))
            asr_degeneracy_risk = _score_float(quality.get("asr_degeneracy_risk"))
            if audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE:
                b_subtype = _b_line_subtype_from_evidence(
                    difference_type=difference_type,
                    edit_text=str(model_fields.get("edit_text", "")).strip(),
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                )
                if "video_context_strength" in final_omni_verification:
                    video_context_strength = _score_float(final_omni_verification.get("video_context_strength"))
                if "asr_degeneracy_risk" in final_omni_verification:
                    asr_degeneracy_risk = _score_float(final_omni_verification.get("asr_degeneracy_risk"))
            record = {
                "proposal_id": proposal_id,
                "candidate_id": str(candidate.get("candidate_id", "")),
                "candidate_stage": "pair_video_comparison",
                "group_id": str(candidate.get("group_id", "")),
                "group_reason": str(candidate.get("group_reason", "single_source_video")),
                "reference_clip_id": reference_clip_id,
                "target_clip_id": target_clip_id,
                "reference_video": reference_video,
                "target_video": target_video,
                "edit_text": str(model_fields.get("edit_text", "")).strip(),
                "modalities": list(model_fields.get("modalities", [])),
                "reference_caption": str(model_fields.get("reference_caption", "")).strip(),
                "target_caption": str(model_fields.get("target_caption", "")).strip(),
                "difference": difference,
                "audio_dataset_line": audio_dataset_line,
                "audio_line_quality_profile": str(quality.get("audio_line_quality_profile", "")).strip(),
                "audio_matters_line": "visual_edit_audio_anchor" if audio_dataset_line == VISUAL_AUDIO_ANCHOR_LINE else "",
                "dominant_delta": dict(model_fields.get("dominant_delta", {})),
                "reference_state": dict(model_fields.get("reference_state", {})) if isinstance(model_fields.get("reference_state"), dict) else {},
                "target_state": dict(model_fields.get("target_state", {})) if isinstance(model_fields.get("target_state"), dict) else {},
                "delta_temporal_extent": dict(model_fields.get("delta_temporal_extent", {})) if isinstance(model_fields.get("delta_temporal_extent"), dict) else {},
                "subject_roles": dict(model_fields.get("subject_roles", {})) if isinstance(model_fields.get("subject_roles"), dict) else {},
                "is_segment_wide_delta": bool(model_fields.get("is_segment_wide_delta")),
                "discarded_deltas": list(model_fields.get("discarded_deltas", [])),
                "pair_video_evidence": list(model_fields.get("evidence", [])),
                "confidence": confidence,
                "model_accepted": bool(model_fields.get("accept")) and not fallback_used,
                "local_gate_passed": bool(local_gate_report.get("passed")),
                "final_omni_accept": final_omni_accept,
                "final_accept_source": "local_gate_and_final_omni",
                "local_gate_report": local_gate_report,
                "final_omni_verification": final_omni_verification,
                "single_source_delta_family": _single_source_delta_family_from_fields(model_fields),
                "single_source_pair_acceptance_issues": blocking_issues,
                "single_source_pair_review_required": _dedupe_strings(local_review_required + final_review_required),
                "b_line_edit_text_repaired": bool(model_fields.get("b_line_edit_text_repaired")),
                "b_line_original_edit_text": str(model_fields.get("b_line_original_edit_text", "")).strip(),
                "raw_proposed_edit_text": str(model_fields.get("b_line_original_edit_text") or raw_model_output.get("edit_text") or "").strip()
                if isinstance(raw_model_output, dict)
                else str(model_fields.get("b_line_original_edit_text", "")).strip(),
                "edit_text_refinement": edit_text_refinement,
                "raw_edit_text_refinement": raw_edit_text_refinement,
                "refined_edit_text": str(edit_text_refinement.get("refined_edit_text", "")).strip(),
                "edit_text_specificity_score": _score_float(edit_text_refinement.get("edit_text_specificity_score")),
                "edit_text_reject_reason": str(edit_text_refinement.get("edit_text_reject_reason", "")).strip(),
                "speech_or_audio_evidence": _normalize_list(edit_text_refinement.get("speech_or_audio_evidence", [])),
                "speech_rewrite": speech_rewrite,
                "raw_speech_rewrite": raw_speech_rewrite,
                "speech_rewrite_refined_edit_text": str(speech_rewrite.get("refined_edit_text", "")).strip(),
                "speech_rewrite_confidence": _score_float(speech_rewrite.get("speech_transcription_confidence")),
                "speech_rewrite_reject_reason": str(speech_rewrite.get("speech_rewrite_reject_reason", "")).strip(),
                "speech_rewrite_used": bool(speech_rewrite_used),
                "b_subtype": b_subtype,
                "video_context_type": video_context_type,
                "video_context_strength": video_context_strength,
                "asr_degeneracy_risk": asr_degeneracy_risk,
                "speech_role": str(reference_annotation.get("speech_role") or target_annotation.get("speech_role") or "").strip(),
                "audio_evidence": _dedupe_strings(
                    _normalize_list(reference_annotation.get("audio_events", []))
                    + _normalize_list(target_annotation.get("audio_events", []))
                    + _normalize_list(edit_text_refinement.get("speech_or_audio_evidence", []))
                    + _normalize_list(speech_rewrite.get("reference_speech_content", []))
                    + _normalize_list(speech_rewrite.get("target_speech_content", []))
                ),
                "visual_context_evidence": _dedupe_strings(
                    [
                        str(reference_annotation.get("scene", "")).strip(),
                        str(target_annotation.get("scene", "")).strip(),
                        str(reference_annotation.get("summary", "")).strip(),
                        str(target_annotation.get("summary", "")).strip(),
                    ]
                ),
                "recommended_edit_text": str(final_omni_verification.get("recommended_edit_text", "")).strip()
                or _single_source_recommended_edit_text(model_fields),
                "hard_negatives": hard_negative_paths,
                "quality": quality,
                "heuristic_quality": dict(candidate.get("quality", {})) if isinstance(candidate.get("quality"), dict) else {},
                "source_context": source_context,
                "source": source,
                "proposal_reason": str(model_fields.get("dominant_delta", {}).get("reason", "")).strip(),
                "evidence": {
                    **_evidence_from_annotations(reference_annotation, target_annotation),
                    "pair_video_comparison": list(model_fields.get("evidence", [])),
                    "discarded_deltas": list(model_fields.get("discarded_deltas", [])),
                },
                "judge": judge,
                "verification": verification,
                "edit_text_quality": edit_text_quality,
                "observable_difference": {
                    "passed": accepted,
                    "frame_backed": bool(model_fields.get("evidence")),
                    "failure_reason": "" if accepted else reject_reason,
                    "reference_evidence": [],
                    "target_evidence": list(model_fields.get("evidence", [])),
                    "supporting_fields": ["pair_video_comparison"],
                },
                "dominant_delta_decision": dict(model_fields.get("dominant_delta", {})),
                "accepted": accepted,
                "fallback_used": fallback_used,
                "raw_model_output": raw_model_output,
                "raw_final_omni_output": raw_final_omni_output,
                "single_source_pair": True,
            }
        if bool(record.get("fallback_used")):
            fallback_count += 1
        if bool(record.get("accepted")):
            accepted_total_count += 1
        else:
            rejected_count += 1
        output_records.append(record)
        _apply_single_source_delta_uniqueness(
            output_records,
            max_accepted_pairs=max_accepted_pairs,
            acceptance_profile=acceptance_profile,
        )
        if bool(record.get("accepted")) and accepted_progress_output is not None:
            _append_jsonl_record(accepted_progress_output, record)
        if not bool(record.get("accepted")) and rejected_progress_output is not None:
            _append_jsonl_record(rejected_progress_output, record)
        persist_progress()
        current_accepted = _select_single_source_quality_passed_records(output_records)
        print(
            "[propose-single-source-pairs] wrote "
            f"proposal_count={len(output_records)} accepted_current={len(current_accepted)} "
            f"proposal_id={record.get('proposal_id', '')} "
            f"accepted={bool(record.get('accepted'))} "
            f"final_omni_accept={bool(record.get('final_omni_accept'))} "
            f"final_omni_quality_score={_score_float((record.get('final_omni_verification') or {}).get('quality_score')) if isinstance(record.get('final_omni_verification'), dict) else 0.0:.2f} "
            f"difference_type={record.get('difference', {}).get('type', '') if isinstance(record.get('difference'), dict) else ''} "
            f"delta_family={record.get('single_source_delta_family', '')} "
            f"fallback={bool(record.get('fallback_used'))} "
            f"issues={';'.join(str(issue) for issue in record.get('single_source_pair_acceptance_issues', []))} "
            f"edit_text={str(record.get('edit_text', '')).replace(chr(10), ' ')[:180]}",
            file=sys.stderr,
            flush=True,
        )
        if (
            zero_accepted_stop_after
            and zero_accepted_stop_after > 0
            and len(output_records) >= zero_accepted_stop_after
            and not current_accepted
        ):
            early_stop_reason = (
                f"zero accepted after {len(output_records)} single-source pair comparisons; "
                "inspect selected source, segment captions, or pair-level Omni output"
            )
            print(f"[propose-single-source-pairs] EARLY_STOP: {early_stop_reason}", file=sys.stderr, flush=True)
            break

    accepted_records = _select_single_source_quality_passed_records(output_records)
    _apply_single_source_delta_uniqueness(
        output_records,
        max_accepted_pairs=max_accepted_pairs,
        acceptance_profile=acceptance_profile,
    )
    accepted_records = _select_single_source_quality_passed_records(output_records)
    _write_jsonl(output, output_records)
    _write_jsonl(accepted_output, accepted_records)
    return {
        "clip_annotations_path": str(annotations_path),
        "pair_candidates_path": str(candidates_path),
        "whole_annotation_path": str(whole_annotation_path or ""),
        "output_path": str(output),
        "accepted_output_path": str(accepted_output),
        "candidate_count": len(candidates),
        "proposal_count": len(output_records),
        "accepted_count": len(accepted_records),
        "accepted_total_count": accepted_total_count,
        "rejected_count": rejected_count,
        "fallback_count": fallback_count,
        "early_stop_reason": early_stop_reason,
        "acceptance_profile": acceptance_profile,
        "audio_dataset_line": audio_dataset_line,
        "omni_retries": omni_retries,
        "fail_on_transient_omni_errors": bool(fail_on_transient_omni_errors),
    }


def _call_omni_with_retries(*, label: str, retries: int, fail_on_transient: bool, func: Any) -> Any:
    attempts = max(0, int(retries or 0)) + 1
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            transient = _is_transient_omni_exception(exc)
            if not transient or attempt >= attempts:
                raise
            wait_seconds = min(30.0, 2.0 * attempt)
            print(
                "[omni-retry] transient error "
                f"label={label} attempt={attempt}/{attempts} wait={wait_seconds:.1f}s "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"omni call {label} did not run")


def _is_transient_omni_exception(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "connection refused",
        "connection reset",
        "connection aborted",
        "remote end closed",
        "remotedisconnected",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "http error 502",
        "http error 503",
        "http error 504",
        "max retries exceeded",
        "no route to host",
        "jsondecodeerror",
        "response did not contain a json object",
        "model response must decode to a json object",
        "expecting ',' delimiter",
        "expecting property name enclosed in double quotes",
        "unterminated string",
    )
    return any(marker in text for marker in transient_markers)


def annotate_clips(
    *,
    root: str | Path,
    clips_manifest_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    concurrency: int = 1,
) -> dict[str, Any]:
    return _annotate_clips_impl(
        root=root,
        clips_manifest_path=clips_manifest_path,
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
        detective=False,
        concurrency=concurrency,
    )


def detective_annotate_clips(
    *,
    root: str | Path,
    clips_manifest_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    concurrency: int = 1,
    audio_focused: bool = False,
) -> dict[str, Any]:
    return _annotate_clips_impl(
        root=root,
        clips_manifest_path=clips_manifest_path,
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
        detective=True,
        concurrency=concurrency,
        audio_focused=audio_focused,
    )


def _annotate_clips_impl(
    *,
    root: str | Path,
    clips_manifest_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None,
    overwrite: bool,
    timeout_seconds: float,
    detective: bool,
    concurrency: int,
    audio_focused: bool = False,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    manifest_path = Path(clips_manifest_path)
    clips = list(_load_jsonl(manifest_path))
    if not clips:
        raise ValueError("clip manifest is empty")

    output = Path(output_path) if output_path else layout["captions"] / DEFAULT_CLIP_ANNOTATIONS_NAME
    # Long Omni annotation runs must be restartable.  Even when callers pass
    # overwrite=True for a fresh run, keep already written records as a resume
    # cache; delete the output file explicitly to force a full re-annotation.
    existing_records = _load_records_by_key(output, "clip_id")
    if not output.exists():
        _write_jsonl(output, [])
    concurrency = max(1, int(concurrency or 1))

    def annotate_one(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        local_client = OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        clip_id = str(item.get("clip_id", "")).strip()
        clip_path = _resolve_under_root(layout["root"], str(item.get("output_path", "")).strip())
        if not clip_path.exists():
            raise FileNotFoundError(f"clip output does not exist: {clip_path}")

        fallback_reason = ""
        detective_fallback_reason = ""
        detective_fallback_used = False
        detective_to_single_pass = False
        raw_model_output: dict[str, Any] = {}
        if detective:
            tool_observations = _build_toolbox_observations(clip_path)
            try:
                normalized, raw_model_output = local_client.annotate_clip_detective(
                    clip_path=str(clip_path),
                    tool_observations=tool_observations,
                    audio_focused=audio_focused,
                )
                fallback_used = False
            except Exception as detective_exc:
                detective_fallback_used = True
                detective_fallback_reason = "detective_to_single_pass"
                try:
                    normalized, single_pass_output = local_client.annotate_clip(clip_path=str(clip_path))
                    raw_model_output = {
                        "detective_error": f"{type(detective_exc).__name__}: {detective_exc}",
                        "single_pass_fallback": single_pass_output,
                    }
                    normalized["storyline"] = []
                    normalized["visible_text"] = []
                    normalized["speakers_and_transcript"] = []
                    normalized["detective_notes"] = ["detective annotation failed; used single-pass annotation"]
                    normalized["detective_trajectory"] = [
                        *tool_observations,
                        {"stage": "detective_error", "error": raw_model_output["detective_error"]},
                        {"stage": "single_pass_fallback", "payload": single_pass_output},
                    ]
                    normalized["uncertainties"] = ["detective annotation failed; used single-pass annotation"]
                    fallback_used = False
                    detective_to_single_pass = True
                except Exception as single_pass_exc:
                    normalized = _fallback_clip_annotation()
                    raw_model_output = {
                        "detective_error": f"{type(detective_exc).__name__}: {detective_exc}",
                        "single_pass_error": f"{type(single_pass_exc).__name__}: {single_pass_exc}",
                    }
                    fallback_used = True
                    fallback_reason = "annotation_fallback"
                    detective_fallback_reason = "detective_and_single_pass_failed"
        else:
            try:
                normalized, raw_model_output = local_client.annotate_clip(clip_path=str(clip_path))
                fallback_used = False
            except Exception as exc:
                normalized = _fallback_clip_annotation()
                raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                fallback_used = True
                fallback_reason = "annotation_fallback"

        record = {
            "clip_id": clip_id,
            "output_path": _display_path(layout["root"], clip_path),
            "summary": normalized["summary"],
            "subjects": list(normalized["subjects"]),
            "object_counts": dict(normalized["object_counts"]),
            "actions": list(normalized["actions"]),
            "scene": normalized["scene"],
            "attributes": list(normalized["attributes"]),
            "on_screen_text": list(normalized["on_screen_text"]),
            "speech": list(normalized["speech"]),
            "audio_events": list(normalized["audio_events"]),
            "modalities": list(normalized["modalities"]),
            "source_asset_id": str(item.get("source_asset_id", "")).strip() or None,
            "fallback_used": fallback_used,
            "raw_model_output": raw_model_output,
        }
        if detective:
            record.update(
                {
                    "storyline": list(normalized.get("storyline", [])),
                    "visible_text": list(normalized.get("visible_text", [])),
                    "speakers_and_transcript": list(normalized.get("speakers_and_transcript", [])),
                    "detective_notes": list(normalized.get("detective_notes", [])),
                    "detective_trajectory": list(normalized.get("detective_trajectory", [])),
                    "uncertainties": list(normalized.get("uncertainties", [])),
                    "detective_fallback_used": detective_fallback_used,
                    "audio_focused_annotation": bool(audio_focused),
                }
            )
            if detective_fallback_reason:
                record["detective_fallback_reason"] = detective_fallback_reason
        record.update(_clip_manifest_metadata(item=item, root=layout["root"]))
        if fallback_reason:
            record["fallback_reason"] = fallback_reason
        return record, detective_to_single_pass

    records_by_clip_id: dict[str, dict[str, Any]] = {}
    pending_items: list[dict[str, Any]] = []
    annotated_count = 0
    reused_count = 0
    detective_to_single_pass_count = 0
    for item in clips:
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError("clip manifest contains an entry without clip_id")

        if clip_id in existing_records:
            records_by_clip_id[clip_id] = existing_records[clip_id]
            reused_count += 1
        else:
            pending_items.append(item)

    if concurrency <= 1:
        for item in pending_items:
            record, detective_to_single_pass = annotate_one(item)
            records_by_clip_id[str(record["clip_id"])] = record
            annotated_count += 1
            if detective_to_single_pass:
                detective_to_single_pass_count += 1
            _append_jsonl_record(output, record)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(annotate_one, item) for item in pending_items]
            for future in as_completed(futures):
                record, detective_to_single_pass = future.result()
                records_by_clip_id[str(record["clip_id"])] = record
                annotated_count += 1
                if detective_to_single_pass:
                    detective_to_single_pass_count += 1
                _append_jsonl_record(output, record)

    output_records: list[dict[str, Any]] = []
    for item in clips:
        clip_id = str(item.get("clip_id", "")).strip()
        record = records_by_clip_id[clip_id]
        output_records.append(record)

    fallback_count = 0
    for record in output_records:
        if bool(record.get("fallback_used")):
            fallback_count += 1

    _write_jsonl(output, output_records)
    return {
        "clips_manifest_path": str(manifest_path),
        "output_path": str(output),
        "clip_count": len(output_records),
        "annotated_count": annotated_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
        "annotation_mode": "detective" if detective else "single_pass",
        "audio_focused_annotation": bool(audio_focused),
        "detective_to_single_pass_count": detective_to_single_pass_count if detective else 0,
        "concurrency": concurrency,
    }


def _clip_manifest_metadata(*, item: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    dataset = str(item.get("dataset", "")).strip()
    if dataset:
        metadata["dataset"] = dataset

    source_row_ids = [str(value).strip() for value in item.get("source_row_ids", []) if str(value).strip()]
    if source_row_ids:
        metadata["source_row_ids"] = source_row_ids

    text_fields = item.get("text_fields")
    if isinstance(text_fields, dict) and text_fields:
        metadata["text_fields"] = text_fields

    source_path = str(item.get("source_path", "")).strip()
    if source_path:
        metadata["source_path"] = _display_source_path(root, source_path)

    clip_timing: dict[str, Any] = {}
    for field_name in ("start_seconds", "end_seconds", "duration_seconds"):
        if field_name in item:
            try:
                clip_timing[field_name] = round(float(item[field_name]), 3)
            except (TypeError, ValueError):
                continue
    role = str(item.get("role", "")).strip()
    notes = str(item.get("notes", "")).strip()
    if role:
        clip_timing["role"] = role
    if notes:
        clip_timing["notes"] = notes
    if clip_timing:
        metadata["source_clip"] = clip_timing
    return metadata


def _display_source_path(root: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)
    return raw_path


def propose_pairs(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    annotations = list(_load_jsonl(annotations_path))
    if not annotations:
        raise ValueError("clip annotations are empty")

    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_PAIR_PROPOSALS_NAME
    existing_records = {} if overwrite else _load_records_by_key(output, "proposal_id")
    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    candidates = _build_pair_candidates(root=layout["root"], annotations=annotations)
    output_records: list[dict[str, Any]] = []
    proposed_count = 0
    reused_count = 0
    fallback_count = 0
    for candidate in candidates:
        proposal_id = candidate["proposal_id"]
        if proposal_id in existing_records:
            record = existing_records[proposal_id]
            reused_count += 1
        else:
            reference_annotation = candidate["reference_annotation"]
            target_annotation = candidate["target_annotation"]
            raw_model_output: dict[str, Any] = {}
            try:
                model_fields, raw_model_output = client.propose_pair(
                    reference_annotation=_annotation_prompt_view(reference_annotation),
                    target_annotation=_annotation_prompt_view(target_annotation),
                    hard_negative_candidates=[
                        _annotation_prompt_view(annotation) for annotation in candidate["hard_negative_annotations"]
                    ],
                )
                fallback_used = False
            except Exception as exc:
                model_fields = _fallback_pair_model_fields(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    primary_difference=candidate["primary_difference"],
                )
                raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                fallback_used = True

            candidate, model_fields, direction_corrected = _maybe_reorient_candidate_for_model_fields(
                root=layout["root"],
                candidate=candidate,
                model_fields=model_fields,
                annotations=annotations,
            )
            if direction_corrected:
                proposal_id = candidate["proposal_id"]
                reference_annotation = candidate["reference_annotation"]
                target_annotation = candidate["target_annotation"]
            source = _build_source_metadata(
                root=layout["root"],
                target_annotation=target_annotation,
                raw_index=raw_index,
            )
            record = {
                "proposal_id": proposal_id,
                "reference_video": reference_annotation["output_path"],
                "target_video": target_annotation["output_path"],
                "edit_text": model_fields["edit_text"],
                "modalities": list(model_fields["modalities"]),
                "reference_caption": model_fields["reference_caption"],
                "target_caption": model_fields["target_caption"],
                "difference": model_fields["difference"],
                "hard_negatives": list(candidate["hard_negative_paths"]),
                "quality": {
                    "same_context_score": candidate["quality"]["same_context_score"],
                    "edit_match_score": candidate["quality"]["edit_match_score"],
                    "target_uniqueness_score": candidate["quality"]["target_uniqueness_score"],
                },
                "source_context": dict(candidate["source_context"]),
                "source": source,
                "proposal_reason": model_fields["proposal_reason"],
                "direction_corrected": direction_corrected,
                "fallback_used": fallback_used,
                "raw_model_output": raw_model_output,
            }
            proposed_count += 1

        if bool(record.get("fallback_used")):
            fallback_count += 1
        output_records.append(record)

    _write_jsonl(output, output_records)
    return {
        "clip_annotations_path": str(annotations_path),
        "output_path": str(output),
        "candidate_count": len(candidates),
        "proposal_count": len(output_records),
        "proposed_count": proposed_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
    }


def mine_pair_candidates(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    clip_groups_path: str | Path,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    max_candidates: int = DEFAULT_MAX_MINED_PAIR_CANDIDATES,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> dict[str, Any]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    groups_path = Path(clip_groups_path)
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_MINED_PAIR_CANDIDATES_NAME
    report = Path(report_path) if report_path else layout["reports"] / DEFAULT_CANDIDATE_MINING_REPORT_NAME
    annotations = list(_load_jsonl(annotations_path))
    groups = list(_load_jsonl(groups_path))
    if not annotations:
        raise ValueError("clip annotations are empty")
    if not groups:
        raise ValueError("clip groups are empty")

    annotations_by_id: dict[str, dict[str, Any]] = {}
    duplicate_annotation_count = 0
    for item in annotations:
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            continue
        if clip_id in annotations_by_id:
            duplicate_annotation_count += 1
        annotations_by_id[clip_id] = item

    candidate_by_id: dict[str, dict[str, Any]] = {}
    skipped_groups: Counter[str] = Counter()
    group_candidate_counts: Counter[str] = Counter()
    usable_annotations = [
        item
        for item in annotations_by_id.values()
        if not bool(item.get("fallback_used")) and _annotation_has_signal(item)
    ]

    for group in groups:
        group_metadata = {
            "group_id": str(group.get("group_id", "")).strip(),
            "group_reason": str(group.get("group_reason", "")).strip(),
        }
        candidate_clip_ids = [str(value).strip() for value in group.get("candidate_clip_ids", []) if str(value).strip()]
        group_annotations = [
            annotations_by_id[clip_id]
            for clip_id in candidate_clip_ids
            if clip_id in annotations_by_id
            and not bool(annotations_by_id[clip_id].get("fallback_used"))
            and _annotation_has_signal(annotations_by_id[clip_id])
        ]
        if len(group_annotations) < 4:
            skipped_groups["too_few_usable_annotations"] += 1
            continue
        group_candidates = _build_pair_candidates(root=layout["root"], annotations=group_annotations)
        group_candidate_counts[group_metadata["group_id"] or "unknown"] += len(group_candidates)
        for candidate in group_candidates:
            mined = _mined_pair_candidate_record(candidate, group_metadata=group_metadata)
            existing = candidate_by_id.get(mined["candidate_id"])
            if existing is None or _score_float(mined["scores"].get("local_candidate_score")) > _score_float(
                existing.get("scores", {}).get("local_candidate_score")
            ):
                candidate_by_id[mined["candidate_id"]] = mined

    template_cluster_candidates = _build_template_cluster_pair_candidates(
        root=layout["root"],
        annotations=usable_annotations,
        acceptance_profile=acceptance_profile,
    )
    if template_cluster_candidates:
        template_metadata = {
            "group_id": "template_cluster_cross_video",
            "group_reason": "cross-video template-cluster mining for clean subject/object/scene replacements",
        }
        group_candidate_counts[template_metadata["group_id"]] += len(template_cluster_candidates)
        for candidate in template_cluster_candidates:
            mined = _mined_pair_candidate_record(candidate, group_metadata=template_metadata)
            existing = candidate_by_id.get(mined["candidate_id"])
            if existing is None or _score_float(mined["scores"].get("local_candidate_score")) > _score_float(
                existing.get("scores", {}).get("local_candidate_score")
            ):
                candidate_by_id[mined["candidate_id"]] = mined

    if len(candidate_by_id) < max_candidates and len(usable_annotations) >= 4:
        global_metadata = {
            "group_id": "global_same_context_backfill",
            "group_reason": "backfill from all same-source or same-dataset annotations after grouped mining",
        }
        for candidate in _build_pair_candidates(root=layout["root"], annotations=usable_annotations):
            mined = _mined_pair_candidate_record(candidate, group_metadata=global_metadata)
            candidate_by_id.setdefault(mined["candidate_id"], mined)

    mined_records = sorted(
        candidate_by_id.values(),
        key=lambda item: (
            -_score_float(item.get("scores", {}).get("local_candidate_score")),
            str(item.get("candidate_id", "")),
        ),
    )[: max(0, max_candidates)]
    _write_jsonl(output, mined_records)
    report.write_text(
        _build_candidate_mining_report(
            output_path=output,
            annotations_count=len(annotations),
            unique_annotation_count=len(annotations_by_id),
            duplicate_annotation_count=duplicate_annotation_count,
            group_count=len(groups),
            skipped_groups=skipped_groups,
            group_candidate_counts=group_candidate_counts,
            candidates=mined_records,
            acceptance_profile=acceptance_profile,
        ),
        encoding="utf-8",
    )
    return {
        "clip_annotations_path": str(annotations_path),
        "clip_groups_path": str(groups_path),
        "output_path": str(output),
        "report_path": str(report),
        "annotation_count": len(annotations),
        "unique_annotation_count": len(annotations_by_id),
        "duplicate_annotation_count": duplicate_annotation_count,
        "group_count": len(groups),
        "candidate_count": len(mined_records),
        "risk_flag_counts": dict(_candidate_risk_flag_counts(mined_records)),
        "difference_type_counts": dict(Counter(str(item.get("difference", {}).get("type", "")) for item in mined_records)),
        "source_relation_counts": dict(
            Counter(str(item.get("source_context", {}).get("relation", "")) for item in mined_records)
        ),
        "skipped_groups": dict(skipped_groups),
        "acceptance_profile": acceptance_profile,
    }


def _mined_pair_candidate_record(
    candidate: dict[str, Any],
    *,
    group_metadata: dict[str, str],
) -> dict[str, Any]:
    reference_annotation = candidate["reference_annotation"]
    target_annotation = candidate["target_annotation"]
    difference = dict(candidate["primary_difference"])
    quality = dict(candidate["quality"])
    source_context = dict(candidate.get("source_context", {}))
    risk_flags = _candidate_risk_flags(candidate)
    scores = {
        "same_context_score": _score_float(quality.get("same_context_score")),
        "difference_strength_score": _score_float(quality.get("difference_strength_score")),
        "single_delta_score": round(max(0.0, 1.0 - max(0, len(candidate.get("changed_difference_types", [])) - 1) * 0.2), 3),
        "target_uniqueness_score": _score_float(quality.get("target_uniqueness_score")),
        "edit_match_score": _score_float(quality.get("edit_match_score")),
        "semantic_context_score": _score_float(quality.get("semantic_context_score")),
        "template_compatibility_score": _score_float(
            quality.get("template_compatibility_score", source_context.get("template_compatibility_score"))
        ),
        "clean_stability_score": _score_float(
            quality.get("clean_stability_score", source_context.get("clean_stability_score"))
        ),
        "single_delta_bundle_score": _score_float(quality.get("single_delta_bundle_score")),
        "local_candidate_score": _score_float(candidate.get("composite_score")),
    }
    return {
        "candidate_id": candidate["proposal_id"],
        "proposal_id": candidate["proposal_id"],
        "reference_clip_id": str(reference_annotation.get("clip_id", "")).strip(),
        "target_clip_id": str(target_annotation.get("clip_id", "")).strip(),
        "reference_video": str(reference_annotation.get("output_path", "")).strip(),
        "target_video": str(target_annotation.get("output_path", "")).strip(),
        "difference": difference,
        "changed_difference_types": list(candidate.get("changed_difference_types", [])),
        "modalities": _infer_pair_modalities(reference_annotation, target_annotation, difference["type"]),
        "source_context": source_context,
        "scores": scores,
        "quality": quality,
        "dominant_delta_decision": dict(candidate.get("dominant_delta_decision", quality.get("dominant_delta_decision", {}))),
        "acceptance_profile": str(quality.get("acceptance_profile", DEFAULT_ACCEPTANCE_PROFILE)),
        "evidence": dict(candidate.get("difference_evidence", {})),
        "risk_flags": risk_flags,
        "group_id": group_metadata.get("group_id", ""),
        "group_reason": group_metadata.get("group_reason", ""),
        "subject_signature_bundle": {
            "reference": list(source_context.get("subject_signature_bundle", {}).get("reference", []))
            if isinstance(source_context.get("subject_signature_bundle"), dict)
            else [],
            "target": list(source_context.get("subject_signature_bundle", {}).get("target", []))
            if isinstance(source_context.get("subject_signature_bundle"), dict)
            else [],
        },
        "object_signature_bundle": {
            "reference": list(source_context.get("object_signature_bundle", {}).get("reference", []))
            if isinstance(source_context.get("object_signature_bundle"), dict)
            else [],
            "target": list(source_context.get("object_signature_bundle", {}).get("target", []))
            if isinstance(source_context.get("object_signature_bundle"), dict)
            else [],
        },
        "scene_signature_bundle": {
            "reference": list(source_context.get("scene_signature_bundle", {}).get("reference", []))
            if isinstance(source_context.get("scene_signature_bundle"), dict)
            else [],
            "target": list(source_context.get("scene_signature_bundle", {}).get("target", []))
            if isinstance(source_context.get("scene_signature_bundle"), dict)
            else [],
        },
        "hard_negative_clip_ids": [
            str(annotation.get("clip_id", "")).strip()
            for annotation in candidate.get("hard_negative_annotations", [])
            if str(annotation.get("clip_id", "")).strip()
        ],
        "hard_negative_paths": list(candidate.get("hard_negative_paths", [])),
    }


def _candidate_risk_flags(candidate: dict[str, Any]) -> list[str]:
    difference = candidate.get("primary_difference", {})
    difference_type = str(difference.get("type", "")).strip()
    quality = candidate.get("quality", {}) if isinstance(candidate.get("quality"), dict) else {}
    source_context = candidate.get("source_context", {}) if isinstance(candidate.get("source_context"), dict) else {}
    risk_flags: list[str] = []
    changed_types = list(candidate.get("changed_difference_types", []))
    if len(changed_types) > 1:
        risk_flags.append("multi_delta")
    if _score_float(quality.get("difference_strength_score")) < MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE:
        risk_flags.append("weak_difference_strength")
    if _score_float(quality.get("clean_stability_score")) and _score_float(quality.get("clean_stability_score")) < MIN_TEMPLATE_CLEAN_STABILITY_SCORE:
        risk_flags.append("unclean_or_unstable_clip")
    if _score_float(quality.get("title_card_or_boundary_text")) >= 1.0:
        risk_flags.append("boundary_text_only")
    if difference_type == "scene":
        if _score_float(quality.get("same_context_score")) < 0.75 or str(source_context.get("relation", "")) in {
            "same_dataset",
            "unknown",
            "cross_dataset",
        }:
            risk_flags.append("too_broad_scene")
    if difference_type == "visible_text":
        risk_flags.append("ocr_title_card_only")
        if _visible_text_fragment_edit(difference):
            risk_flags.append("visible_text_fragment_edit")
        if _score_float(quality.get("target_uniqueness_score")) < MIN_ACCEPT_TARGET_UNIQUENESS_SCORE:
            risk_flags.append("ocr_template_risk")
    if difference_type == "speech":
        risk_flags.append("speech_content_disabled")
        if _score_float(quality.get("speech_transcript_backed")) < 1.0:
            risk_flags.append("speech_unbacked")
    if difference_type == "audio_event" and _difference_values_are_too_similar(
        str(difference.get("from", "")),
        str(difference.get("to", "")),
    ):
        risk_flags.append("audio_too_similar")
    if difference_type == "audio_event" and _score_float(quality.get("audio_primary_allowed", 1.0)) < 1.0:
        risk_flags.append("audio_secondary_due_to_visual_delta")
    if difference_type in VISUAL_DIFFERENCE_TYPES:
        observable = _observable_difference_gate(
            reference_annotation=candidate["reference_annotation"],
            target_annotation=candidate["target_annotation"],
            difference=difference,
            visual_near_duplicate_score=None,
        )
        if not bool(observable.get("passed")):
            risk_flags.append("too_similar_without_observable_delta")
    if str(source_context.get("relation", "")) == SAME_TEMPLATE_CLUSTER_RELATION:
        if _score_float(quality.get("template_compatibility_score", source_context.get("template_compatibility_score"))) < MIN_TEMPLATE_COMPATIBILITY_SCORE:
            risk_flags.append("cross_video_over_broad")
        if difference_type == "attribute" and _score_float(quality.get("subject_signature_bundle_count")) < 1.0:
            risk_flags.append("subject_bundle_underspecified")
        subject_bundle = source_context.get("subject_signature_bundle", {})
        if (
            difference_type == "attribute"
            and isinstance(subject_bundle, dict)
            and subject_bundle.get("reference") == subject_bundle.get("target")
        ):
            risk_flags.append("near_duplicate_same_subject")
    return _dedupe_strings(risk_flags)


def _candidate_risk_flag_counts(candidates: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        flags = candidate.get("risk_flags", [])
        if isinstance(flags, list) and flags:
            counts.update(str(item) for item in flags)
        else:
            counts["none"] += 1
    return counts


def _annotation_subject_signature_bundle(annotation: dict[str, Any]) -> list[str]:
    bundle: list[str] = []
    for subject in _normalize_list(annotation.get("subjects", [])):
        normalized = _normalized_phrase(subject)
        if normalized in {"man", "woman"} and normalized not in bundle:
            bundle.append(normalized)
    for attribute in _normalize_list(annotation.get("attributes", [])):
        normalized = _normalized_phrase(attribute)
        if normalized and _tokenize_text(normalized) & SUBJECT_SIGNATURE_MARKER_TOKENS and normalized not in bundle:
            bundle.append(normalized)
    summary = _normalized_phrase(str(annotation.get("summary", "")))
    for phrase in (
        "black jacket",
        "brown shirt",
        "maroon hoodie",
        "red hair",
        "curly hair",
        "receding hairline",
        "wearing glasses",
        "glasses",
        "beard",
        "earrings",
        "necklace",
    ):
        if phrase in summary and phrase not in bundle:
            bundle.append(phrase)
    return bundle[:4]


def _annotation_object_signature_bundle(annotation: dict[str, Any]) -> list[str]:
    counts = _normalize_object_counts(annotation.get("object_counts", {}))
    bundle: list[str] = []
    for label in sorted(counts):
        normalized = _normalized_phrase(label)
        if not normalized or normalized in GENERIC_HUMAN_OBJECT_LABELS:
            continue
        bundle.append(normalized)
    return bundle[:3]


def _coarse_scene_label(annotation: dict[str, Any]) -> str:
    scene_text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("scene", "")).strip(),
                str(annotation.get("summary", "")).strip(),
            ]
        )
    )
    for token in sorted(SCENE_SIGNATURE_MARKER_TOKENS, key=len, reverse=True):
        if token in scene_text:
            return token
    return ""


def _annotation_scene_signature_bundle(annotation: dict[str, Any]) -> list[str]:
    bundle: list[str] = []
    coarse_scene = _coarse_scene_label(annotation)
    if coarse_scene:
        bundle.append(coarse_scene)
    normalized_scene = _normalized_phrase(str(annotation.get("scene", "")))
    if normalized_scene and normalized_scene not in bundle:
        bundle.append(normalized_scene)
    return bundle[:2]


def _title_card_or_boundary_text(annotation: dict[str, Any]) -> bool:
    visible_text = _normalize_list(annotation.get("visible_text") or annotation.get("on_screen_text", []))
    summary = _normalized_phrase(str(annotation.get("summary", "")))
    if any(token in summary for token in TITLE_CARD_HINT_TOKENS):
        return True
    if not visible_text:
        return False
    has_subjects = bool(_normalize_list(annotation.get("subjects", [])) or _normalize_object_counts(annotation.get("object_counts", {})))
    has_actions = bool(_action_terms_from_annotation(annotation))
    return not has_subjects or not has_actions


def _human_subject_count(annotation: dict[str, Any]) -> int:
    counts = _normalize_object_counts(annotation.get("object_counts", {}))
    total = 0
    for label, count in counts.items():
        if _normalized_phrase(label) in GENERIC_HUMAN_OBJECT_LABELS:
            total += count
    if total > 0:
        return total
    return sum(1 for subject in _normalize_list(annotation.get("subjects", [])) if _normalized_phrase(subject) in GENERIC_HUMAN_OBJECT_LABELS)


def _is_talking_head_template(annotation: dict[str, Any]) -> bool:
    if _human_subject_count(annotation) != 1:
        return False
    actions = _action_terms_from_annotation(annotation)
    merged_text = _normalized_phrase(" ".join(actions + [str(annotation.get("summary", ""))]))
    scene = _normalized_phrase(str(annotation.get("scene", "")))
    if any(token in merged_text for token in ("speak", "talk", "present", "lecture", "interview", "address")):
        return True
    return any(token in scene for token in ("desk", "stage", "studio", "podium", "lecture"))


def _template_family(annotation: dict[str, Any]) -> str:
    if _is_talking_head_template(annotation):
        return "talking_head"
    if _human_subject_count(annotation) <= 1 and _annotation_object_signature_bundle(annotation):
        return "single_subject_scene"
    if _human_subject_count(annotation) == 0 and _annotation_object_signature_bundle(annotation):
        return "showcase"
    return "general"


def _clean_stability_score(annotation: dict[str, Any]) -> float:
    score = 0.55
    if _annotation_has_signal(annotation):
        score += 0.10
    if _is_talking_head_template(annotation):
        score += 0.15
    if _annotation_subject_signature_bundle(annotation):
        score += 0.10
    if _annotation_scene_signature_bundle(annotation):
        score += 0.10
    if _title_card_or_boundary_text(annotation):
        score -= 0.35
    summary = _normalized_phrase(str(annotation.get("summary", "")))
    if any(token in summary for token in ("montage", "compilation", "multiple scenes", "quick cuts", "rapid cuts")):
        score -= 0.20
    if len(_normalize_list(annotation.get("uncertainties", []))) >= 2:
        score -= 0.10
    return max(0.0, min(1.0, round(score, 3)))


def _template_compatibility_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_family = _template_family(left)
    right_family = _template_family(right)
    score = 0.0
    if left_family == right_family:
        score += 0.45
    elif {left_family, right_family} <= {"talking_head", "single_subject_scene"}:
        score += 0.28

    left_scene = _coarse_scene_label(left)
    right_scene = _coarse_scene_label(right)
    if left_scene and right_scene:
        if left_scene == right_scene:
            score += 0.20
        else:
            score += _scene_similarity(left_scene, right_scene) * 0.10

    left_humans = _human_subject_count(left)
    right_humans = _human_subject_count(right)
    if left_humans == right_humans and left_humans <= 2:
        score += 0.15

    action_similarity = _jaccard(_tokenize_values(_action_terms_from_annotation(left)), _tokenize_values(_action_terms_from_annotation(right)))
    score += action_similarity * 0.10
    if not _title_card_or_boundary_text(left) and not _title_card_or_boundary_text(right):
        score += 0.10
    return max(0.0, min(1.0, round(score, 3)))


def _single_delta_bundle_score(
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    *,
    difference_type: str,
) -> float:
    subject_changed = _annotation_subject_signature_bundle(reference_annotation) != _annotation_subject_signature_bundle(target_annotation)
    object_changed = _annotation_object_signature_bundle(reference_annotation) != _annotation_object_signature_bundle(target_annotation)
    scene_changed = _annotation_scene_signature_bundle(reference_annotation) != _annotation_scene_signature_bundle(target_annotation)
    action_changed = bool(
        _first_unique(_action_terms_from_annotation(reference_annotation), _action_terms_from_annotation(target_annotation))
        or _first_unique(_action_terms_from_annotation(target_annotation), _action_terms_from_annotation(reference_annotation))
    )
    audio_changed = bool(
        _first_unique(_non_speech_audio_terms(reference_annotation), _non_speech_audio_terms(target_annotation))
        or _first_unique(_non_speech_audio_terms(target_annotation), _non_speech_audio_terms(reference_annotation))
    )
    text_changed = bool(
        _first_unique(_visible_text_values(reference_annotation), _visible_text_values(target_annotation))
        or _first_unique(_visible_text_values(target_annotation), _visible_text_values(reference_annotation))
    )
    changed_bundle_count = sum(
        1 for changed in (subject_changed, object_changed, scene_changed, action_changed, audio_changed, text_changed) if changed
    )
    score = 1.0 if changed_bundle_count <= 1 else 0.85 if changed_bundle_count == 2 else 0.70 if changed_bundle_count == 3 else 0.55
    expected_change = {
        "attribute": subject_changed,
        "object_presence": object_changed,
        "scene": scene_changed,
        "action": action_changed,
        "audio_event": audio_changed,
        "visible_text": text_changed,
    }.get(difference_type, True)
    if not expected_change:
        score -= 0.35
    return max(0.0, min(1.0, round(score, 3)))


def _same_template_cluster_source_context(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    compatibility = _template_compatibility_score(left, right)
    clean_stability = min(_clean_stability_score(left), _clean_stability_score(right))
    dataset = str(left.get("dataset", "")).strip() or str(right.get("dataset", "")).strip()
    return {
        "relation": SAME_TEMPLATE_CLUSTER_RELATION,
        "score": round(0.28 + compatibility * 0.52, 3),
        "dataset": dataset,
        "template_family": _template_family(left),
        "template_compatibility_score": compatibility,
        "clean_stability_score": clean_stability,
    }


def _template_final_threshold_warnings(quality: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field_name, threshold_key in (
        ("semantic_context_score", "template_semantic_context_score"),
        ("template_compatibility_score", "template_compatibility_score"),
        ("clean_stability_score", "template_clean_stability_score"),
        ("single_delta_bundle_score", "template_single_delta_bundle_score"),
        ("target_uniqueness_score", "template_target_uniqueness_score"),
        ("difference_strength_score", "template_difference_strength_score"),
    ):
        value = _score_float(quality.get(field_name))
        threshold = _profile_threshold(DEFAULT_ACCEPTANCE_PROFILE, threshold_key)
        if value < threshold:
            warnings.append(f"{field_name}_below_final_threshold")
    return warnings


def _template_cluster_key(annotation: dict[str, Any]) -> str:
    dataset = str(annotation.get("dataset", "")).strip() or "unknown"
    return f"{dataset}:{_template_family(annotation)}"


def _build_template_cluster_pair_candidates(
    *,
    root: Path,
    annotations: list[dict[str, Any]],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> list[dict[str, Any]]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    clusters: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        if bool(annotation.get("fallback_used")) or not _annotation_has_signal(annotation):
            continue
        clusters.setdefault(_template_cluster_key(annotation), []).append(annotation)

    candidates: list[dict[str, Any]] = []
    comparison_count = 0
    for cluster_annotations in clusters.values():
        cluster_annotations.sort(
            key=lambda item: (-_clean_stability_score(item), str(item.get("clip_id", "")))
        )
        for left_index, left in enumerate(cluster_annotations):
            for right in cluster_annotations[left_index + 1 :]:
                if comparison_count >= MAX_TEMPLATE_CLUSTER_COMPARISONS:
                    break
                if str(left.get("source_path", "")).strip() == str(right.get("source_path", "")).strip():
                    continue
                comparison_count += 1
                candidate = _build_cross_video_template_candidate(
                    root=root,
                    reference_annotation=left,
                    target_annotation=right,
                    annotations=annotations,
                    acceptance_profile=acceptance_profile,
                )
                if candidate is not None:
                    candidates.append(candidate)
            if comparison_count >= MAX_TEMPLATE_CLUSTER_COMPARISONS:
                break
        if comparison_count >= MAX_TEMPLATE_CLUSTER_COMPARISONS:
            break
    candidates.sort(key=lambda item: (-item["composite_score"], item["proposal_id"]))
    return candidates


def _build_cross_video_template_candidate(
    *,
    root: Path,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> dict[str, Any] | None:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    if reference_annotation["clip_id"] == target_annotation["clip_id"]:
        return None
    if str(reference_annotation.get("dataset", "")).strip() != str(target_annotation.get("dataset", "")).strip():
        return None

    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    template_compatibility_score = _template_compatibility_score(reference_annotation, target_annotation)
    clean_stability_score = min(_clean_stability_score(reference_annotation), _clean_stability_score(target_annotation))
    if (
        semantic_context_score < _profile_threshold(acceptance_profile, "template_semantic_context_score")
        or template_compatibility_score < _profile_threshold(acceptance_profile, "template_compatibility_score")
        or clean_stability_score < _profile_threshold(acceptance_profile, "template_clean_stability_score")
    ):
        return None

    source_context = _same_template_cluster_source_context(reference_annotation, target_annotation)
    subject_reference = _annotation_subject_signature_bundle(reference_annotation)
    subject_target = _annotation_subject_signature_bundle(target_annotation)
    object_reference = _annotation_object_signature_bundle(reference_annotation)
    object_target = _annotation_object_signature_bundle(target_annotation)
    scene_reference = _annotation_scene_signature_bundle(reference_annotation)
    scene_target = _annotation_scene_signature_bundle(target_annotation)
    audio_reference = _non_speech_audio_terms(reference_annotation)
    audio_target = _non_speech_audio_terms(target_annotation)
    reference_actions = _action_terms_from_annotation(reference_annotation)
    target_actions = _action_terms_from_annotation(target_annotation)

    candidate_specs: list[tuple[str, dict[str, Any]]] = []
    if _is_talking_head_template(reference_annotation) and _is_talking_head_template(target_annotation):
        if subject_reference and subject_target and subject_reference != subject_target:
            candidate_specs.append(
                (
                    "cross_video_template_subject",
                    {
                        "type": "attribute",
                        "from": f"speaker with {', '.join(subject_reference[:4])}",
                        "to": f"speaker with {', '.join(subject_target[:4])}",
                        "description": "the speaker's visual signature changes while the presentation template stays similar",
                    },
                )
            )
    if object_reference and object_target and object_reference != object_target:
        candidate_specs.append(
            (
                "cross_video_template_object",
                {
                    "type": "object_presence",
                    "from": _first_item(object_reference) or "featured object",
                    "to": _first_item(object_target) or "featured object",
                    "description": "the featured object changes while the shot template stays similar",
                },
            )
        )
    if scene_reference and scene_target and scene_reference != scene_target:
        candidate_specs.append(
            (
                "cross_video_template_scene",
                {
                    "type": "scene",
                    "from": _first_item(scene_reference) or str(reference_annotation.get("scene", "")).strip(),
                    "to": _first_item(scene_target) or str(target_annotation.get("scene", "")).strip(),
                    "description": "the setting changes while the clip template remains aligned",
                },
            )
        )
    removed_action = _first_unique(reference_actions, target_actions)
    added_action = _first_unique(target_actions, reference_actions)
    if removed_action and added_action:
        candidate_specs.append(
            (
                "cross_video_template_action",
                {
                    "type": "action",
                    "from": removed_action,
                    "to": added_action,
                    "description": "the action changes between otherwise template-compatible clips",
                },
            )
        )
    removed_audio = _first_unique(audio_reference, audio_target)
    added_audio = _first_unique(audio_target, audio_reference)
    if removed_audio and added_audio:
        candidate_specs.append(
            (
                "cross_video_audio_nonspeech",
                {
                    "type": "audio_event",
                    "from": removed_audio,
                    "to": added_audio,
                    "description": "the non-speech audio event changes while the visual template stays similar",
                },
            )
        )

    best_candidate: dict[str, Any] | None = None
    best_tuple: tuple[float, float, float, str] | None = None
    for route_name, difference in candidate_specs:
        difference_type = str(difference.get("type", "")).strip()
        single_delta_bundle_score = _single_delta_bundle_score(
            reference_annotation,
            target_annotation,
            difference_type=difference_type,
        )
        if single_delta_bundle_score < _profile_threshold(acceptance_profile, "template_single_delta_bundle_score"):
            continue
        hard_negative_annotations = _select_hard_negative_annotations(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            annotations=annotations,
            primary_difference=difference,
        )
        if len(hard_negative_annotations) < 2:
            continue
        target_uniqueness_score = _target_uniqueness_score(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            annotations=annotations,
            primary_difference=difference,
        )
        if target_uniqueness_score < _profile_threshold(acceptance_profile, "template_target_uniqueness_score"):
            continue
        difference_strength_score = _difference_strength_score(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=difference,
            changed_types=[difference_type],
        )
        if difference_type == "attribute":
            difference_strength_score = max(difference_strength_score, 0.80)
        if difference_strength_score < _profile_threshold(acceptance_profile, "template_difference_strength_score"):
            continue

        same_context_score = max(semantic_context_score, round(template_compatibility_score * 0.95, 3))
        edit_match_score = max(
            _edit_match_score(
                same_context_score=same_context_score,
                primary_difference_type=difference_type,
                changed_types=[difference_type],
            ),
            round(template_compatibility_score * 0.45 + single_delta_bundle_score * 0.35 + clean_stability_score * 0.20, 3),
        )
        quality = {
            "same_context_score": round(same_context_score, 3),
            "semantic_context_score": round(semantic_context_score, 3),
            "edit_match_score": round(edit_match_score, 3),
            "target_uniqueness_score": round(target_uniqueness_score, 3),
            "difference_strength_score": round(difference_strength_score, 3),
            "difference_type": difference_type,
            "template_compatibility_score": round(template_compatibility_score, 3),
            "clean_stability_score": round(clean_stability_score, 3),
            "single_delta_bundle_score": round(single_delta_bundle_score, 3),
            "talking_head_template": 1.0 if _is_talking_head_template(reference_annotation) and _is_talking_head_template(target_annotation) else 0.0,
            "title_card_or_boundary_text": 1.0
            if _title_card_or_boundary_text(reference_annotation) or _title_card_or_boundary_text(target_annotation)
            else 0.0,
            "subject_signature_bundle_count": float(min(len(subject_reference), len(subject_target))),
            "acceptance_profile": acceptance_profile,
        }
        if _is_exploration_profile(acceptance_profile):
            quality["exploration_warnings"] = _template_final_threshold_warnings(quality)
        if difference_type == "audio_event":
            quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(reference_annotation, target_annotation)
            quality["has_audio_modality"] = 1.0
            audio_decision = _dominant_delta_decision(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=difference,
                quality=quality,
                source_context=source_context,
            )
            if not audio_decision["audio_primary_allowed"]:
                quality["exploration_warnings"] = _dedupe_strings(
                    _normalize_list(quality.get("exploration_warnings", []))
                    + list(audio_decision.get("failure_flags", []))
                )
            if not audio_decision["audio_primary_allowed"] and audio_decision.get("dominant_type") in DOMINANT_VISUAL_DIFFERENCE_TYPES:
                retargeted_difference_type = str(audio_decision.get("dominant_type", "")).strip()
                retargeted_difference = _dominant_visual_difference_from_annotations(
                    reference_annotation,
                    target_annotation,
                    difference_type=retargeted_difference_type,
                )
                if retargeted_difference is None or retargeted_difference.get("type") != retargeted_difference_type:
                    continue
                changed_types = list(retargeted_difference.pop("changed_types"))
                difference = retargeted_difference
                difference_type = retargeted_difference_type
                route_name = f"{route_name}_retargeted_{difference_type}"
                quality["difference_type"] = difference_type
                quality["retargeted_from_audio_secondary"] = 1.0
                quality["retargeted_from_difference_type"] = "audio_event"
                quality["edit_match_score"] = round(
                    _edit_match_score(
                        same_context_score=same_context_score,
                        primary_difference_type=difference_type,
                        changed_types=changed_types,
                    ),
                    3,
                )
                quality["difference_strength_score"] = round(
                    _difference_strength_score(
                        reference_annotation=reference_annotation,
                        target_annotation=target_annotation,
                        primary_difference=difference,
                        changed_types=changed_types,
                    ),
                    3,
                )
                quality["exploration_warnings"] = _dedupe_strings(
                    _normalize_list(quality.get("exploration_warnings", []))
                    + ["retargeted_from_audio_secondary"]
                )
                difference_strength_score = _score_float(quality.get("difference_strength_score"))
            else:
                changed_types = [difference_type]
        else:
            changed_types = [difference_type]
        source_context_for_candidate = {
            **source_context,
            "template_route": route_name,
            "subject_signature_bundle": {"reference": subject_reference, "target": subject_target},
            "object_signature_bundle": {"reference": object_reference, "target": object_target},
            "scene_signature_bundle": {"reference": scene_reference, "target": scene_target},
        }
        dominant_delta_decision = _dominant_delta_decision(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=difference,
            quality=quality,
            source_context=source_context_for_candidate,
        )
        quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
        quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
        quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
        quality["dominant_delta_decision"] = dominant_delta_decision
        composite_score = _candidate_composite_score(quality, source_context_for_candidate)
        reference_path = _display_path(root, _resolve_under_root(root, reference_annotation["output_path"]))
        target_path = _display_path(root, _resolve_under_root(root, target_annotation["output_path"]))
        hard_negative_paths = [
            _display_path(root, _resolve_under_root(root, annotation["output_path"]))
            for annotation in hard_negative_annotations[:3]
        ]
        candidate = {
            "proposal_id": _build_proposal_id(reference_path, target_path),
            "reference_annotation": _sanitize_annotation_for_output(reference_annotation, root),
            "target_annotation": _sanitize_annotation_for_output(target_annotation, root),
            "primary_difference": difference,
            "changed_difference_types": changed_types,
            "quality": quality,
            "composite_score": composite_score,
            "source_context": source_context_for_candidate,
            "dominant_delta_decision": dominant_delta_decision,
            "difference_evidence": _difference_evidence_from_annotations(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                primary_difference=difference,
            ),
            "hard_negative_annotations": [_sanitize_annotation_for_output(annotation, root) for annotation in hard_negative_annotations[:3]],
            "hard_negative_paths": hard_negative_paths,
        }
        candidate_tuple = (composite_score, difference_strength_score, target_uniqueness_score, route_name)
        if best_tuple is None or candidate_tuple > best_tuple:
            best_tuple = candidate_tuple
            best_candidate = candidate
    return best_candidate


def _build_candidate_mining_report(
    *,
    output_path: Path,
    annotations_count: int,
    unique_annotation_count: int,
    duplicate_annotation_count: int,
    group_count: int,
    skipped_groups: Counter[str],
    group_candidate_counts: Counter[str],
    candidates: list[dict[str, Any]],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> str:
    difference_counts = Counter(str(item.get("difference", {}).get("type", "")) for item in candidates)
    relation_counts = Counter(str(item.get("source_context", {}).get("relation", "")) for item in candidates)
    route_counts = Counter(str(item.get("source_context", {}).get("template_route", "")) for item in candidates if str(item.get("source_context", {}).get("template_route", "")).strip())
    risk_counts = _candidate_risk_flag_counts(candidates)
    lines = [
        "# Candidate Mining Report",
        "",
        f"- Output: `{output_path}`",
        f"- Annotation rows: `{annotations_count}`",
        f"- Unique annotations: `{unique_annotation_count}`",
        f"- Duplicate annotation rows: `{duplicate_annotation_count}`",
        f"- Clip groups: `{group_count}`",
        f"- Mined candidates: `{len(candidates)}`",
        f"- Acceptance profile: `{_normalize_acceptance_profile(acceptance_profile)}`",
        "",
        "## Difference Type Counts",
    ]
    for key, value in sorted(difference_counts.items()):
        lines.append(f"- `{key or 'unknown'}`: `{value}`")
    if not difference_counts:
        lines.append("- none")
    lines.extend(["", "## Source Relation Counts"])
    for key, value in sorted(relation_counts.items()):
        lines.append(f"- `{key or 'unknown'}`: `{value}`")
    if not relation_counts:
        lines.append("- none")
    lines.extend(["", "## Template Route Counts"])
    for key, value in sorted(route_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    if not route_counts:
        lines.append("- none")
    lines.extend(["", "## Risk Flag Counts"])
    for key, value in sorted(risk_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    if not risk_counts:
        lines.append("- none")
    lines.extend(["", "## Skipped Groups"])
    for key, value in sorted(skipped_groups.items()):
        lines.append(f"- `{key}`: `{value}`")
    if not skipped_groups:
        lines.append("- none")
    lines.extend(["", "## Top Groups"])
    for key, value in group_candidate_counts.most_common(10):
        lines.append(f"- `{key}`: `{value}`")
    if not group_candidate_counts:
        lines.append("- none")
    lines.extend(["", "## Top Candidates"])
    for candidate in candidates[:20]:
        difference = candidate.get("difference", {})
        scores = candidate.get("scores", {})
        flags = candidate.get("risk_flags", [])
        lines.append(
            "- "
            f"`{candidate.get('candidate_id')}` "
            f"`{difference.get('type', 'unknown')}` "
            f"`{difference.get('from', '')}` -> `{difference.get('to', '')}` "
            f"score=`{scores.get('local_candidate_score', 0.0)}` "
            f"route=`{candidate.get('source_context', {}).get('template_route', candidate.get('source_context', {}).get('relation', ''))}` "
            f"risks=`{','.join(flags) if flags else 'none'}`"
        )
    if not candidates:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _candidate_from_mined_record(
    mined: dict[str, Any],
    *,
    annotations_by_id: dict[str, dict[str, Any]],
    annotations: list[dict[str, Any]],
    root: Path,
) -> dict[str, Any] | None:
    reference_clip_id = str(mined.get("reference_clip_id", "")).strip()
    target_clip_id = str(mined.get("target_clip_id", "")).strip()
    if not reference_clip_id or not target_clip_id:
        return None
    reference_annotation = annotations_by_id.get(reference_clip_id)
    target_annotation = annotations_by_id.get(target_clip_id)
    if reference_annotation is None or target_annotation is None:
        return None
    difference = mined.get("difference", {})
    if not isinstance(difference, dict) or not str(difference.get("type", "")).strip():
        detected = _detect_primary_difference(reference_annotation, target_annotation)
        if detected is None:
            return None
        changed_types = list(detected.pop("changed_types"))
        difference = detected
    else:
        difference = dict(difference)
        changed_types = [
            str(item).strip()
            for item in mined.get("changed_difference_types", [])
            if str(item).strip()
        ] or [str(difference.get("type", "")).strip()]

    hard_negative_annotations = [
        annotations_by_id[clip_id]
        for clip_id in [str(item).strip() for item in mined.get("hard_negative_clip_ids", []) if str(item).strip()]
        if clip_id in annotations_by_id and clip_id not in {reference_clip_id, target_clip_id}
    ]
    if len(hard_negative_annotations) < 2:
        hard_negative_annotations = _select_hard_negative_annotations(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            annotations=annotations,
            primary_difference=difference,
        )
    if len(hard_negative_annotations) < 2:
        return None

    quality = dict(mined.get("quality", {})) if isinstance(mined.get("quality"), dict) else {}
    if "acceptance_profile" not in quality and str(mined.get("acceptance_profile", "")).strip():
        quality["acceptance_profile"] = _normalize_acceptance_profile(str(mined.get("acceptance_profile")))
    scores = mined.get("scores", {}) if isinstance(mined.get("scores"), dict) else {}
    for source_key, target_key in (
        ("same_context_score", "same_context_score"),
        ("semantic_context_score", "semantic_context_score"),
        ("edit_match_score", "edit_match_score"),
        ("target_uniqueness_score", "target_uniqueness_score"),
        ("difference_strength_score", "difference_strength_score"),
        ("template_compatibility_score", "template_compatibility_score"),
        ("clean_stability_score", "clean_stability_score"),
        ("single_delta_bundle_score", "single_delta_bundle_score"),
        ("audio_anchor_score", "audio_anchor_score"),
        ("audio_anchor_context_score", "audio_anchor_context_score"),
        ("audio_anchor_min_rms", "audio_anchor_min_rms"),
    ):
        if target_key not in quality and source_key in scores:
            quality[target_key] = _score_float(scores.get(source_key))
    quality.setdefault("difference_type", str(difference.get("type", "")).strip())
    source_context = mined.get("source_context", {}) if isinstance(mined.get("source_context"), dict) else {}
    if not source_context:
        source_context = _source_context(reference_annotation, target_annotation)
    dominant_delta_decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=difference,
        quality=quality,
        source_context=source_context,
    )
    quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
    quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
    quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
    quality["dominant_delta_decision"] = dominant_delta_decision

    reference_path = _display_path(root, _resolve_under_root(root, reference_annotation["output_path"]))
    target_path = _display_path(root, _resolve_under_root(root, target_annotation["output_path"]))
    hard_negative_paths = [
        _display_path(root, _resolve_under_root(root, annotation["output_path"])) for annotation in hard_negative_annotations[:3]
    ]
    candidate = {
        "proposal_id": str(mined.get("proposal_id") or mined.get("candidate_id") or _build_proposal_id(reference_path, target_path)),
        "reference_annotation": _sanitize_annotation_for_output(reference_annotation, root),
        "target_annotation": _sanitize_annotation_for_output(target_annotation, root),
        "primary_difference": difference,
        "changed_difference_types": changed_types,
        "quality": quality,
        "composite_score": _score_float(scores.get("local_candidate_score"))
        or _candidate_composite_score(quality, source_context),
        "source_context": dict(source_context),
        "dominant_delta_decision": dominant_delta_decision,
        "difference_evidence": dict(mined.get("evidence", {}))
        if isinstance(mined.get("evidence"), dict)
        else _difference_evidence_from_annotations(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=difference,
        ),
        "hard_negative_annotations": [
            _sanitize_annotation_for_output(annotation, root) for annotation in hard_negative_annotations[:3]
        ],
        "hard_negative_paths": hard_negative_paths,
        "mined_candidate": {
            "candidate_id": mined.get("candidate_id", ""),
            "candidate_kind": mined.get("candidate_kind", ""),
            "risk_flags": list(mined.get("risk_flags", [])) if isinstance(mined.get("risk_flags"), list) else [],
            "scores": dict(scores),
        },
    }
    return _retarget_audio_secondary_candidate_to_dominant_visual(candidate)


def propose_group_pairs(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    clip_groups_path: str | Path,
    mined_candidates_path: str | Path | None = None,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    accepted_output_path: str | Path | None = None,
    accepted_progress_path: str | Path | None = None,
    rejected_progress_path: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    max_accepted_pairs: int = 10,
    max_proposals: int | None = None,
    zero_accepted_stop_after: int = DEFAULT_ZERO_ACCEPTED_STOP_AFTER,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
    strict_audio_matters_visual_anchor: bool = True,
) -> dict[str, Any]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    layout = ensure_layout(root)
    annotations_path = Path(clip_annotations_path)
    groups_path = Path(clip_groups_path)
    mined_path = Path(mined_candidates_path) if mined_candidates_path else None
    print(
        "[propose-group-pairs] load inputs "
        f"annotations_path={annotations_path} groups_path={groups_path} mined_candidates_path={mined_path or ''}",
        file=sys.stderr,
        flush=True,
    )
    annotations = list(_load_jsonl(annotations_path))
    groups = list(_load_jsonl(groups_path))
    if not annotations:
        raise ValueError("clip annotations are empty")
    if not groups:
        raise ValueError("clip groups are empty")
    print(
        f"[propose-group-pairs] loaded annotations={len(annotations)} groups={len(groups)}",
        file=sys.stderr,
        flush=True,
    )

    output = Path(output_path) if output_path else layout["pairs"] / "judged_pair_proposals.jsonl"
    accepted_output = Path(accepted_output_path) if accepted_output_path else layout["pairs"] / DEFAULT_ACCEPTED_PAIRS_NAME
    accepted_progress_output = Path(accepted_progress_path) if accepted_progress_path else None
    rejected_progress_output = Path(rejected_progress_path) if rejected_progress_path else None
    # Long pair proposal runs are model-call heavy. Preserve already written
    # proposal rows as a resume cache even when callers pass --overwrite.
    existing_records = _load_records_by_key(output, "proposal_id")
    if not output.exists():
        _write_jsonl(output, [])
    if not accepted_output.exists():
        _write_jsonl(accepted_output, [])
    if accepted_progress_output is not None:
        accepted_progress_output.parent.mkdir(parents=True, exist_ok=True)
        accepted_progress_output.write_text("", encoding="utf-8")
    if rejected_progress_output is not None:
        rejected_progress_output.parent.mkdir(parents=True, exist_ok=True)
        rejected_progress_output.write_text("", encoding="utf-8")
    print("[propose-group-pairs] load raw asset index", file=sys.stderr, flush=True)
    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    print(f"[propose-group-pairs] raw asset index loaded rows={len(raw_index)}", file=sys.stderr, flush=True)
    annotations_by_id: dict[str, dict[str, Any]] = {}
    duplicate_annotation_count = 0
    for item in annotations:
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            continue
        if clip_id in annotations_by_id:
            duplicate_annotation_count += 1
        annotations_by_id[clip_id] = item
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    output_records: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    candidate_count = 0
    proposed_count = 0
    reused_count = 0
    fallback_count = 0
    rejected_count = 0
    accepted_total_count = 0
    pre_propose_rejected_count = 0
    pre_propose_reject_counts: Counter[str] = Counter()
    early_stop_reason = ""
    seen_proposal_ids: set[str] = set()

    def persist_progress() -> None:
        current_accepted = _select_final_accepted_records(
            output_records,
            max_accepted_pairs=max_accepted_pairs,
            acceptance_profile=acceptance_profile,
        )
        _write_jsonl(output, output_records)
        _write_jsonl(accepted_output, current_accepted)

    def filter_pre_propose_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal pre_propose_rejected_count
        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            reasons = _candidate_pre_propose_reject_reasons(
                candidate,
                acceptance_profile=acceptance_profile,
            )
            if reasons:
                pre_propose_rejected_count += 1
                pre_propose_reject_counts.update(reasons)
                continue
            filtered.append(candidate)
        return filtered

    if mined_path is not None:
        mined_records = list(_load_jsonl(mined_path))
        print(
            f"[propose-group-pairs] loaded mined candidates rows={len(mined_records)}",
            file=sys.stderr,
            flush=True,
        )
        mined_candidates = filter_pre_propose_candidates([
            candidate
            for candidate in (
                _candidate_from_mined_record(
                    record,
                    annotations_by_id=annotations_by_id,
                    annotations=list(annotations_by_id.values()),
                    root=layout["root"],
                )
                for record in mined_records
            )
            if candidate is not None
        ])
        if pre_propose_rejected_count:
            print(
                "[propose-group-pairs] pre-propose filtered "
                f"count={pre_propose_rejected_count} reasons={dict(pre_propose_reject_counts)}",
                file=sys.stderr,
                flush=True,
            )
        candidate_batches = [
            (
                1,
                1,
                {
                    "group_id": "mined_pair_candidates",
                    "group_reason": f"local candidate mining from {mined_path}",
                },
                mined_candidates,
                list(annotations_by_id.values()),
            )
        ]
    else:
        candidate_batches = []
        for group_index, group in enumerate(groups, start=1):
            group_metadata = {
                "group_id": str(group.get("group_id", "")).strip(),
                "group_reason": str(group.get("group_reason", "")).strip(),
            }
            candidate_clip_ids = [str(value).strip() for value in group.get("candidate_clip_ids", []) if str(value).strip()]
            group_annotations = [
                annotations_by_id[clip_id]
                for clip_id in candidate_clip_ids
                if clip_id in annotations_by_id and not bool(annotations_by_id[clip_id].get("fallback_used"))
            ]
            print(
                "[propose-group-pairs] group "
                f"{group_index}/{len(groups)} group_id={group_metadata['group_id']} "
                f"candidate_clip_ids={len(candidate_clip_ids)} usable_annotations={len(group_annotations)}",
                file=sys.stderr,
                flush=True,
            )
            if len(group_annotations) < 4:
                continue
            candidates = filter_pre_propose_candidates(_build_pair_candidates(root=layout["root"], annotations=group_annotations))
            candidate_batches.append((group_index, len(groups), group_metadata, candidates, group_annotations))

    for group_index, group_total, group_metadata, candidates, group_annotations in candidate_batches:
        if early_stop_reason:
            break
        print(
            "[propose-group-pairs] group "
            f"{group_index}/{group_total} built_candidates={len(candidates)} total_candidate_count={candidate_count + len(candidates)}",
            file=sys.stderr,
            flush=True,
        )
        candidate_count += len(candidates)
        for candidate in candidates:
            if max_proposals is not None and len(output_records) >= max_proposals:
                break
            proposal_id = candidate["proposal_id"]
            if proposal_id in seen_proposal_ids:
                continue
            seen_proposal_ids.add(proposal_id)
            reference_annotation = candidate["reference_annotation"]
            target_annotation = candidate["target_annotation"]
            print(
                "[propose-group-pairs] start "
                f"proposal_index={len(output_records) + 1} group_id={group_metadata['group_id']} "
                f"proposal_id={proposal_id}",
                file=sys.stderr,
                flush=True,
            )
            if proposal_id in existing_records:
                record = existing_records[proposal_id]
                reused_count += 1
            else:
                raw_model_output: dict[str, Any] = {}
                judge_raw_output: dict[str, Any] = {}
                verification_raw_output: dict[str, Any] = {}
                audio_anchor_visual_raw_output: dict[str, Any] = {}
                audio_anchor_visual_verification: dict[str, Any] = {}
                verification_skipped_before_video = False
                try:
                    model_fields, raw_model_output = client.propose_pair(
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        target_annotation=_annotation_prompt_view(target_annotation),
                        hard_negative_candidates=[
                            _annotation_prompt_view(annotation) for annotation in candidate["hard_negative_annotations"]
                        ],
                        heuristic_pair={
                            "primary_difference": dict(candidate["primary_difference"]),
                            "changed_difference_types": list(candidate["changed_difference_types"]),
                            "heuristic_quality": dict(candidate["quality"]),
                            "source_context": dict(candidate["source_context"]),
                            "mined_candidate": dict(candidate.get("mined_candidate", {})),
                        },
                    )
                    proposal_fallback_used = False
                except Exception as exc:
                    model_fields = _fallback_pair_model_fields(
                        reference_annotation=reference_annotation,
                        target_annotation=target_annotation,
                        primary_difference=candidate["primary_difference"],
                    )
                    raw_model_output = {"error": f"{type(exc).__name__}: {exc}"}
                    proposal_fallback_used = True

                model_fields = _repair_pair_model_fields(
                    model_fields=model_fields,
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    source_context=candidate["source_context"],
                )
                direction_corrected = False
                oriented_candidate, model_fields, direction_corrected = _maybe_reorient_candidate_for_model_fields(
                    root=layout["root"],
                    candidate=candidate,
                    model_fields=model_fields,
                    annotations=group_annotations,
                )
                if direction_corrected:
                    seen_proposal_ids.discard(proposal_id)
                    proposal_id = oriented_candidate["proposal_id"]
                    if proposal_id in seen_proposal_ids:
                        continue
                    seen_proposal_ids.add(proposal_id)
                    candidate = oriented_candidate
                    reference_annotation = candidate["reference_annotation"]
                    target_annotation = candidate["target_annotation"]
                source = _build_source_metadata(
                    root=layout["root"],
                    target_annotation=target_annotation,
                    raw_index=raw_index,
                )
                proposal_quality = _quality_for_model_fields(
                    base_quality={**candidate["quality"], "source_context": dict(candidate["source_context"])},
                    model_fields=model_fields,
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                )
                proposal_quality["acceptance_profile"] = acceptance_profile
                edit_text_quality = _edit_text_quality_payload(
                    edit_text=model_fields["edit_text"],
                    difference=model_fields["difference"],
                    modalities=model_fields["modalities"],
                    reference_caption=model_fields["reference_caption"],
                    target_caption=model_fields["target_caption"],
                )
                observable_difference = _observable_difference_gate(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    difference=model_fields["difference"],
                    visual_near_duplicate_score=proposal_quality.get("visual_near_duplicate_score"),
                )
                _apply_structured_gate_quality(
                    proposal_quality,
                    edit_text_quality=edit_text_quality,
                    observable_difference=observable_difference,
                )
                proposal_difference_evidence = _difference_evidence_from_annotations(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    primary_difference=model_fields["difference"],
                )
                proposal_view = {
                    "proposal_id": proposal_id,
                    "edit_text": model_fields["edit_text"],
                    "modalities": list(model_fields["modalities"]),
                    "reference_caption": model_fields["reference_caption"],
                    "target_caption": model_fields["target_caption"],
                    "difference": model_fields["difference"],
                    "quality": dict(proposal_quality),
                    "heuristic_primary_difference": dict(candidate["primary_difference"]),
                    "changed_difference_types": list(candidate["changed_difference_types"]),
                    "source_context": dict(candidate["source_context"]),
                    "difference_evidence": dict(proposal_difference_evidence),
                    "edit_text_quality": dict(edit_text_quality),
                    "observable_difference": dict(observable_difference),
                    "acceptance_profile": acceptance_profile,
                    "acceptance_thresholds": {
                        "same_context_score": _profile_threshold(acceptance_profile, "same_context_score"),
                        "edit_match_score": _profile_threshold(acceptance_profile, "edit_match_score"),
                        "target_uniqueness_score": _profile_threshold(acceptance_profile, "target_uniqueness_score"),
                        "difference_strength_score": _profile_threshold(acceptance_profile, "difference_strength_score"),
                        "action_evidence_score_for_action_edits": _profile_threshold(acceptance_profile, "action_evidence_score"),
                        "speech_evidence_score_for_speech_edits": MIN_ACCEPT_SPEECH_EVIDENCE_SCORE,
                        "speech_specificity_score_for_speech_edits": MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE,
                        "non_speech_audio_event_score_for_audio_event_edits": _profile_threshold(acceptance_profile, "non_speech_audio_event_score"),
                        "max_visual_near_duplicate_score_for_visual_edits": MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE,
                        "audio_anchor_score_for_audio_matters": _profile_threshold(acceptance_profile, "audio_anchor_score"),
                        "visual_delta_strength_for_audio_matters": _profile_threshold(acceptance_profile, "visual_delta_strength"),
                        "near_duplicate_risk_for_audio_matters": _profile_threshold(acceptance_profile, "near_duplicate_risk"),
                    },
                }
                try:
                    judge, judge_raw_output = client.judge_pair(
                        proposal=proposal_view,
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        target_annotation=_annotation_prompt_view(target_annotation),
                        hard_negative_candidates=[
                            _annotation_prompt_view(annotation) for annotation in candidate["hard_negative_annotations"]
                        ],
                    )
                    judge_fallback_used = False
                except Exception as exc:
                    judge = _fallback_pair_judge(candidate["quality"], reason=f"{type(exc).__name__}: {exc}")
                    judge_raw_output = {"error": f"{type(exc).__name__}: {exc}"}
                    judge_fallback_used = True

                judge = _finalize_pair_judge(judge)
                pre_verification_quality = _effective_pair_quality(judge, None, proposal_quality)
                if not _should_skip_pair_video_verification(judge, pre_verification_quality, acceptance_profile=acceptance_profile):
                    try:
                        (
                            verification,
                            verification_raw_output,
                            verification_context_retry_used,
                        ) = _verify_pair_difference_with_context_retry(
                            client,
                            proposal=proposal_view,
                            reference_annotation=_annotation_prompt_view(reference_annotation),
                            target_annotation=_annotation_prompt_view(target_annotation),
                            reference_clip_path=str(
                                _resolve_under_root(layout["root"], reference_annotation["output_path"])
                            ),
                            target_clip_path=str(_resolve_under_root(layout["root"], target_annotation["output_path"])),
                        )
                        verification_fallback_used = False
                    except Exception as exc:
                        verification = _fallback_pair_verification(reason=f"{type(exc).__name__}: {exc}")
                        verification_raw_output = {"error": f"{type(exc).__name__}: {exc}"}
                        verification_context_retry_used = False
                        verification_fallback_used = True
                else:
                    skip_reason = _compose_reject_reason(judge, None, pre_verification_quality)
                    verification = _fallback_pair_verification(
                        reason=f"pre-verification reject; video verification skipped: {skip_reason}"
                    )
                    verification_raw_output = {
                        "skipped": True,
                        "stage": "pre_verification_gate",
                        "reason": skip_reason,
                    }
                    verification_context_retry_used = False
                    verification_fallback_used = False
                    verification_skipped_before_video = True
                verification = _finalize_pair_verification(verification)
                fallback_used = proposal_fallback_used or judge_fallback_used or verification_fallback_used
                effective_quality = _effective_pair_quality(judge, verification, proposal_quality)
                if _is_audio_matters_profile(acceptance_profile) and strict_audio_matters_visual_anchor:
                    if _boolish(judge.get("accept")) and _boolish(judge.get("single_main_difference")):
                        try:
                            audio_anchor_visual_verification, audio_anchor_visual_raw_output = client.verify_audio_anchor_visual_pair(
                                proposal=proposal_view,
                                reference_annotation=_annotation_prompt_view(reference_annotation),
                                target_annotation=_annotation_prompt_view(target_annotation),
                                reference_clip_path=str(
                                    _resolve_under_root(layout["root"], reference_annotation["output_path"])
                                ),
                                target_clip_path=str(_resolve_under_root(layout["root"], target_annotation["output_path"])),
                            )
                        except Exception as exc:
                            audio_anchor_visual_verification = _fallback_audio_anchor_visual_verification(
                                reason=f"{type(exc).__name__}: {exc}"
                            )
                            audio_anchor_visual_raw_output = {"error": f"{type(exc).__name__}: {exc}"}
                            fallback_used = True
                    else:
                        audio_anchor_visual_verification = _fallback_audio_anchor_visual_verification(
                            reason="skipped because the pair judge did not accept a single main difference"
                        )
                        audio_anchor_visual_raw_output = {
                            "skipped": True,
                            "reason": "pair judge did not accept a single main difference",
                        }
                    _apply_audio_anchor_visual_quality(effective_quality, audio_anchor_visual_verification)
                accepted = _judge_accepts(judge, verification, effective_quality, acceptance_profile=acceptance_profile)
                if accepted:
                    judge["reject_reason"] = ""
                else:
                    judge["reject_reason"] = _compose_reject_reason(judge, verification, effective_quality)
                speech_quality = _speech_quality_payload(effective_quality)
                audio_event_quality = _audio_event_quality_payload(effective_quality)
                record = {
                    "proposal_id": proposal_id,
                    "group_id": group_metadata["group_id"],
                    "group_reason": group_metadata["group_reason"],
                    "reference_clip_id": reference_annotation.get("clip_id", ""),
                    "target_clip_id": target_annotation.get("clip_id", ""),
                    "reference_video": reference_annotation["output_path"],
                    "target_video": target_annotation["output_path"],
                    "edit_text": model_fields["edit_text"],
                    "modalities": list(model_fields["modalities"]),
                    "reference_caption": model_fields["reference_caption"],
                    "target_caption": model_fields["target_caption"],
                    "difference": model_fields["difference"],
                    "audio_matters_line": "visual_edit_audio_anchor"
                    if _is_audio_matters_profile(acceptance_profile)
                    else "",
                    "hard_negatives": list(candidate["hard_negative_paths"]),
                    "judge_quality": {
                        "same_context_score": judge["same_context_score"],
                        "edit_match_score": judge["edit_match_score"],
                        "target_uniqueness_score": judge["target_uniqueness_score"],
                    },
                    "quality": effective_quality,
                    "heuristic_quality": dict(proposal_quality),
                    "source_context": dict(candidate["source_context"]),
                    "source": source,
                    "proposal_reason": model_fields["proposal_reason"],
                    "direction_corrected": direction_corrected,
                    "evidence": _evidence_from_annotations(
                        reference_annotation,
                        target_annotation,
                        difference_evidence=proposal_difference_evidence,
                    ),
                    "judge": judge,
                    "verification": verification,
                    "audio_anchor_visual_verification": audio_anchor_visual_verification,
                    "speech_quality": speech_quality,
                    "audio_event_quality": audio_event_quality,
                    "edit_text_quality": edit_text_quality,
                    "observable_difference": observable_difference,
                    "dominant_delta_decision": dict(effective_quality.get("dominant_delta_decision", {})),
                    "transcript_backed": speech_quality.get("transcript_backed"),
                    "accepted": accepted,
                    "fallback_used": fallback_used,
                    "raw_model_output": raw_model_output,
                    "raw_judge_output": judge_raw_output,
                    "raw_verification_output": verification_raw_output,
                    "raw_audio_anchor_visual_output": audio_anchor_visual_raw_output,
                    "verification_annotation_only_retry_used": verification_context_retry_used,
                    "verification_skipped_before_video": verification_skipped_before_video,
                }
                proposed_count += 1

            if "verification" not in record:
                record = dict(record)
                record["verification"] = _fallback_pair_verification(reason="existing record has no verification")
                record["accepted"] = False
                record["fallback_used"] = True
                judge = dict(record.get("judge", {}))
                judge["reject_reason"] = _compose_reject_reason(judge, record["verification"], record.get("quality"))
                record["judge"] = judge
            record = _prepare_record_for_acceptance(
                record,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                acceptance_profile=acceptance_profile,
            )
            judge = dict(record.get("judge", {}))
            verification = record.get("verification", {})
            quality = record.get("quality", {})
            record["accepted"] = _judge_accepts(judge, verification, quality, acceptance_profile=acceptance_profile)
            if bool(record.get("accepted")):
                judge["reject_reason"] = ""
                record["judge"] = judge
            else:
                judge["accept"] = False
                judge["reject_reason"] = _compose_reject_reason(judge, verification, quality)
                record["judge"] = judge
            acceptance_issues = _pair_record_acceptance_issues(
                root=layout["root"],
                record=record,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                acceptance_profile=acceptance_profile,
            )
            if acceptance_issues:
                record = _reject_record_with_acceptance_issues(record, acceptance_issues)
                quality = dict(record.get("quality", {}))
                if any("single clip" in issue for issue in acceptance_issues):
                    quality["intraclip_change_conflict"] = 1.0
                record["quality"] = quality
            if bool(record.get("fallback_used")):
                fallback_count += 1
            if bool(record.get("accepted")):
                accepted_total_count += 1
            else:
                rejected_count += 1
            output_records.append(record)
            persist_progress()
            current_accepted_records = _select_final_accepted_records(
                output_records,
                max_accepted_pairs=max_accepted_pairs,
                acceptance_profile=acceptance_profile,
            )
            edit_preview = str(record.get("edit_text", "")).replace("\n", " ").strip()
            if len(edit_preview) > 160:
                edit_preview = edit_preview[:157].rstrip() + "..."
            difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
            quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
            if bool(record.get("accepted")) and accepted_progress_output is not None:
                _append_jsonl_record(
                    accepted_progress_output,
                    {
                        "event": "accepted_sample",
                        "proposal_index": len(output_records),
                        "accepted_current": len(current_accepted_records),
                        "proposal_id": record.get("proposal_id", ""),
                        "reference_video": record.get("reference_video", ""),
                        "target_video": record.get("target_video", ""),
                        "edit_text": record.get("edit_text", ""),
                        "difference_type": difference.get("type", ""),
                        "same_context_score": quality.get("same_context_score"),
                        "edit_match_score": quality.get("edit_match_score"),
                        "target_uniqueness_score": quality.get("target_uniqueness_score"),
                        "difference_strength_score": quality.get("difference_strength_score"),
                        "audio_anchor_score": quality.get("audio_anchor_score"),
                        "omni_visual_accept": quality.get("omni_visual_accept"),
                        "visual_delta_strength": quality.get("visual_delta_strength"),
                        "near_duplicate_risk": quality.get("near_duplicate_risk"),
                    },
                )
            if bool(record.get("accepted")):
                print(
                    "[propose-group-pairs] ACCEPTED_SAMPLE "
                    f"proposal_index={len(output_records)} accepted_current={len(current_accepted_records)} "
                    f"proposal_id={record.get('proposal_id', '')} "
                    f"difference_type={difference.get('type', '')} "
                    f"audio_anchor_score={quality.get('audio_anchor_score', '')} "
                    f"reference_video={record.get('reference_video', '')} "
                    f"target_video={record.get('target_video', '')} "
                    f"edit_text={edit_preview}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                judge_payload = record.get("judge", {}) if isinstance(record.get("judge"), dict) else {}
                reject_preview = str(judge_payload.get("reject_reason", "")).replace("\n", " ").strip()
                if len(reject_preview) > 160:
                    reject_preview = reject_preview[:157].rstrip() + "..."
                print(
                    "[propose-group-pairs] REJECTED_PROPOSAL "
                    f"proposal_index={len(output_records)} proposal_id={record.get('proposal_id', '')} "
                    f"difference_type={difference.get('type', '')} reject_reason={reject_preview}",
                    file=sys.stderr,
                    flush=True,
                )
                if rejected_progress_output is not None:
                    _append_jsonl_record(
                        rejected_progress_output,
                        {
                            "event": "rejected_sample",
                            "proposal_index": len(output_records),
                            "proposal_id": record.get("proposal_id", ""),
                            "reference_video": record.get("reference_video", ""),
                            "target_video": record.get("target_video", ""),
                            "edit_text": record.get("edit_text", ""),
                            "difference_type": difference.get("type", ""),
                            "audio_anchor_score": quality.get("audio_anchor_score"),
                            "omni_visual_accept": quality.get("omni_visual_accept"),
                            "visual_delta_strength": quality.get("visual_delta_strength"),
                            "near_duplicate_risk": quality.get("near_duplicate_risk"),
                            "reject_reason": reject_preview,
                        },
                    )
            print(
                "[propose-group-pairs] wrote "
                f"proposal_count={len(output_records)} accepted_current="
                f"{len(current_accepted_records)} "
                f"accepted={bool(record.get('accepted'))} fallback={bool(record.get('fallback_used'))} "
                f"skipped_video={bool(record.get('verification_skipped_before_video'))}",
                file=sys.stderr,
                flush=True,
            )
            if (
                zero_accepted_stop_after
                and zero_accepted_stop_after > 0
                and len(output_records) >= zero_accepted_stop_after
                and not current_accepted_records
            ):
                early_stop_reason = (
                    f"zero accepted after {len(output_records)} judged proposals; "
                    "stop early because candidate mining or gate logic needs inspection"
                )
                print(f"[propose-group-pairs] EARLY_STOP: {early_stop_reason}", file=sys.stderr, flush=True)
                break
        if max_proposals is not None and len(output_records) >= max_proposals:
            break

    accepted_records = _select_final_accepted_records(
        output_records,
        max_accepted_pairs=max_accepted_pairs,
        acceptance_profile=acceptance_profile,
    )
    _write_jsonl(output, output_records)
    _write_jsonl(accepted_output, accepted_records)
    verification_counts = _pair_verification_counts(output_records)
    return {
        "clip_annotations_path": str(annotations_path),
        "clip_groups_path": str(groups_path),
        "output_path": str(output),
        "accepted_output_path": str(accepted_output),
        "accepted_progress_path": str(accepted_progress_output or ""),
        "rejected_progress_path": str(rejected_progress_output or ""),
        "mined_candidates_path": str(mined_path) if mined_path else "",
        "group_count": len(groups),
        "annotation_count": len(annotations),
        "unique_annotation_count": len(annotations_by_id),
        "duplicate_annotation_count": duplicate_annotation_count,
        "candidate_count": candidate_count,
        "proposal_count": len(output_records),
        "accepted_count": len(accepted_records),
        "accepted_total_count": accepted_total_count,
        "rejected_count": rejected_count,
        "proposed_count": proposed_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
        "pre_propose_rejected_count": pre_propose_rejected_count,
        "pre_propose_reject_counts": dict(pre_propose_reject_counts),
        "early_stop_reason": early_stop_reason,
        "zero_accepted_stop_after": zero_accepted_stop_after,
        "verification_counts": verification_counts,
        "thresholds": {
            "same_context_score": _profile_threshold(acceptance_profile, "same_context_score"),
            "edit_match_score": _profile_threshold(acceptance_profile, "edit_match_score"),
            "target_uniqueness_score": _profile_threshold(acceptance_profile, "target_uniqueness_score"),
            "edit_necessity_score": _profile_threshold(acceptance_profile, "edit_necessity_score"),
            "edit_target_alignment_score": _profile_threshold(acceptance_profile, "edit_target_alignment_score"),
            "difference_strength_score": _profile_threshold(acceptance_profile, "difference_strength_score"),
            "action_evidence_score_for_action_edits": _profile_threshold(acceptance_profile, "action_evidence_score"),
            "speech_evidence_score_for_speech_edits": MIN_ACCEPT_SPEECH_EVIDENCE_SCORE,
            "speech_specificity_score_for_speech_edits": MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE,
            "non_speech_audio_event_score_for_audio_event_edits": _profile_threshold(acceptance_profile, "non_speech_audio_event_score"),
            "audio_anchor_score_for_audio_matters": _profile_threshold(acceptance_profile, "audio_anchor_score"),
            "visual_delta_strength_for_audio_matters": _profile_threshold(acceptance_profile, "visual_delta_strength"),
            "near_duplicate_risk_for_audio_matters": _profile_threshold(acceptance_profile, "near_duplicate_risk"),
        },
        "acceptance_profile": acceptance_profile,
    }


def plan_video_edits(
    *,
    root: str | Path,
    pair_candidates_path: str | Path,
    clip_annotations_path: str | Path,
    output_path: str | Path | None = None,
    max_plans: int = 10,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
    planning_mode: str = "production",
    planner_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    planning_mode = str(planning_mode).strip() or "production"
    if planning_mode not in {"production", "exploration"}:
        raise ValueError("planning_mode must be 'production' or 'exploration'")
    layout = ensure_layout(root)
    candidates = list(_load_jsonl(Path(pair_candidates_path)))
    original_candidate_count = len(candidates)
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not candidates:
        raise ValueError("pair candidates file is empty")
    if not annotations:
        raise ValueError("clip annotations are empty")

    annotation_lookup = _annotation_lookup(root=layout["root"], annotations=annotations)
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_VIDEO_EDIT_PLAN_NAME
    cache_output = Path(planner_cache_path) if planner_cache_path else output.with_name(DEFAULT_VIDEO_EDIT_PLANNER_CACHE_NAME)
    planner_cache = _load_video_edit_planner_cache(cache_output)
    planner_cache_dirty = False
    planner_client = (
        OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        if base_url and model
        else None
    )

    plans: list[dict[str, Any]] = []
    skipped_by_type: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    cache_hits = 0
    cache_misses = 0
    if planning_mode == "exploration":
        exploration_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            reference_video = str(candidate.get("reference_video", "")).strip()
            if not reference_video:
                skipped_by_type["unknown"] += 1
                skipped_reasons["missing_reference_video"] += 1
                continue
            reference_annotation = _annotation_for_video_edit_plan(
                root=layout["root"],
                lookup=annotation_lookup,
                record=candidate,
                video_field="reference_video",
                caption_field="reference_caption",
            )
            if not _annotation_is_usable_for_reference_understanding(reference_annotation):
                skipped_reasons["reference_annotation_unusable"] += 1
                continue
            generated = _video_edit_exploration_candidates(candidate, reference_annotation)
            if generated:
                exploration_candidates.extend(generated)
                skipped_reasons["exploration_ideation_from_reference"] += len(generated)
            else:
                skipped_reasons["exploration_no_suitable_reference_edit"] += 1
        candidates = exploration_candidates

    seen_sources: set[str] = set()
    seen_plan_keys: set[tuple[str, str, str, str, str]] = set()
    for candidate in candidates:
        if len(plans) >= max_plans:
            break
        difference = dict(candidate.get("difference") or {})
        difference_type = str(difference.get("type", "")).strip()
        route = _video_edit_model_route(difference_type)
        safe_visual_ideation_used = False

        reference_video = str(candidate.get("reference_video", "")).strip()
        if not reference_video:
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["missing_reference_video"] += 1
            continue
        if planning_mode != "exploration" and reference_video in seen_sources:
            skipped_reasons["duplicate_reference_video"] += 1
            continue

        reference_annotation = _annotation_for_video_edit_plan(
            root=layout["root"],
            lookup=annotation_lookup,
            record=candidate,
            video_field="reference_video",
            caption_field="reference_caption",
        )
        if not _annotation_is_usable_for_reference_understanding(reference_annotation):
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["reference_annotation_unusable"] += 1
            continue
        if route not in {None, "vace_controlled"} and planner_client is not None:
            ideation_candidate = _safe_visual_ideation_candidate(candidate, reference_annotation)
            if ideation_candidate is not None:
                candidate = ideation_candidate
                difference = dict(candidate.get("difference") or {})
                difference_type = str(difference.get("type", "")).strip()
                route = _video_edit_model_route(difference_type)
                safe_visual_ideation_used = True
                skipped_reasons["safe_visual_ideation_from_non_vace_candidate"] += 1
        if route is None:
            ideation_candidate = (
                _safe_visual_ideation_candidate(candidate, reference_annotation)
                if planner_client is not None
                else None
            )
            if ideation_candidate is None:
                skipped_by_type[difference_type or "unknown"] += 1
                skipped_reasons["unsupported_difference_type"] += 1
                continue
            candidate = ideation_candidate
            difference = dict(candidate.get("difference") or {})
            difference_type = str(difference.get("type", "")).strip()
            route = _video_edit_model_route(difference_type)
            safe_visual_ideation_used = True
            skipped_reasons["safe_visual_ideation_from_unsupported_type"] += 1
            if route is None:
                skipped_by_type[difference_type or "unknown"] += 1
                skipped_reasons["unsupported_difference_type"] += 1
                continue
        risk = _video_edit_risk_assessment(reference_annotation, difference_type=difference_type)
        if planning_mode == "exploration":
            risk = _relax_visual_exploration_risk(risk, candidate)
        if safe_visual_ideation_used:
            risk = _relax_safe_visual_ideation_risk(risk, candidate)
        if not risk["allow_generation"]:
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons[f"high_risk_reference_{risk['risk_level']}"] += 1
            for reason in risk["risk_reasons"]:
                skipped_reasons[f"risk_{reason}"] += 1
            continue
        edit_text = str(candidate.get("edit_text", "")).strip() or _build_fallback_edit_text(difference)
        edit_token = str(candidate.get("suggested_edit_token", "")).strip() or _video_edit_token(difference, edit_text)
        edit_region = str(candidate.get("suggested_edit_region", "")).strip() or _video_edit_region(edit_text, difference, reference_annotation, route)
        if difference_type in {"object_presence", "object_count"} and (not edit_token or not edit_region):
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["missing_object_edit_token_or_region"] += 1
            continue

        source_prompt = _video_edit_source_prompt(reference_annotation, candidate)
        target_prompt = _video_edit_target_prompt(
            source_prompt=source_prompt,
            edit_text=edit_text,
            difference=difference,
            edit_token=edit_token,
        )
        preserve_tokens = _video_edit_preserve_tokens(reference_annotation, difference, edit_token, edit_text=edit_text)
        negative_prompt = _video_edit_negative_prompt(preserve_tokens, risk=risk)
        planner_metadata: dict[str, Any] = {
            "stage": "heuristic_prompt_planner",
            "input": "annotation_and_candidate_edit",
            "output": "source_prompt_target_prompt_edit_token_preserve_constraints",
            "fallback_used": True,
        }
        raw_planner_output: dict[str, Any] = {}
        planned_mask_query = str(candidate.get("suggested_mask_query", "")).strip()
        planned_preserve_regions: list[str] = [
            str(item).strip()
            for item in candidate.get("suggested_preserve_regions", [])
            if str(item).strip()
        ] if isinstance(candidate.get("suggested_preserve_regions"), list) else []
        planner_input = {
            "edit_text": edit_text,
            "difference": difference,
            "reference_video": reference_video,
            "reference_caption": str(candidate.get("reference_caption", "")).strip(),
            "model_route_hint": route,
            "planning_mode": planning_mode,
            "exploration_family": str(candidate.get("exploration_family", "")).strip(),
        }
        planner_cache_key = _video_edit_planner_cache_key(
            model=model,
            planning_mode=planning_mode,
            route=route,
            reference_video=reference_video,
            reference_annotation=reference_annotation,
            candidate=planner_input,
        )
        cached_planner_record = planner_cache.get(planner_cache_key)
        if cached_planner_record:
            cache_hits += 1
        elif planner_client is not None:
            cache_misses += 1
        if planner_client is not None or cached_planner_record is not None:
            try:
                if cached_planner_record is not None:
                    planned = dict(cached_planner_record.get("planned", {}))
                    raw_planner_output = dict(cached_planner_record.get("raw_planner_output", {}))
                else:
                    planned, raw_planner_output = planner_client.plan_video_edit(
                        reference_clip_path=str(_resolve_under_root(layout["root"], reference_video)),
                        reference_annotation=_annotation_prompt_view(reference_annotation),
                        candidate=planner_input,
                        route_hint=route,
                    )
                    planner_cache[planner_cache_key] = {
                        "cache_key": planner_cache_key,
                        "model": model,
                        "planning_mode": planning_mode,
                        "route": route,
                        "reference_video": reference_video,
                        "candidate": planner_input,
                        "planned": planned,
                        "raw_planner_output": raw_planner_output,
                    }
                    planner_cache_dirty = True
                if not bool(planned.get("should_generate")):
                    skipped_by_type[difference_type or "unknown"] += 1
                    skipped_reasons["model_planner_rejected"] += 1
                    continue
                planned_edit_text = str(planned.get("edit_text", "")).strip()
                if planned_edit_text:
                    edit_text = planned_edit_text
                planned_difference = planned.get("difference")
                if isinstance(planned_difference, dict) and str(planned_difference.get("type", "")).strip():
                    planned_difference = _normalize_model_planned_visual_difference(
                        dict(planned_difference),
                        edit_text=edit_text,
                    )
                    planned_difference_type = str(planned_difference.get("type", "")).strip()
                    planned_difference_route = _video_edit_model_route(planned_difference_type)
                    if planned_difference_route is None:
                        skipped_by_type[planned_difference_type or "unknown"] += 1
                        skipped_reasons["model_planner_revised_to_unsupported_difference_type"] += 1
                        continue
                    difference = dict(planned_difference)
                    difference_type = planned_difference_type
                    route = planned_difference_route
                    risk = _video_edit_risk_assessment(reference_annotation, difference_type=difference_type)
                    if planning_mode == "exploration":
                        risk = _relax_visual_exploration_risk(risk, candidate)
                    if safe_visual_ideation_used:
                        risk = _relax_safe_visual_ideation_risk(risk, candidate)
                    if not risk["allow_generation"]:
                        skipped_by_type[difference_type or "unknown"] += 1
                        skipped_reasons[f"model_planner_revised_to_high_risk_{risk['risk_level']}"] += 1
                        for reason in risk["risk_reasons"]:
                            skipped_reasons[f"risk_{reason}"] += 1
                        continue
                source_prompt = str(planned["source_prompt"]).strip()
                target_prompt = str(planned["target_prompt"]).strip()
                edit_token = str(planned["edit_token"]).strip()
                preserve_tokens = [str(item).strip() for item in planned["preserve_tokens"] if str(item).strip()]
                negative_prompt = str(planned["negative_prompt"]).strip()
                negative_prompt = _merge_video_edit_locks(negative_prompt, risk)
                edit_region = str(planned["edit_region"]).strip()
                planned_mask_query = str(planned.get("mask_query", "")).strip()
                planned_preserve_regions = [
                    str(item).strip()
                    for item in planned.get("preserve_regions", [])
                    if str(item).strip()
                ]
                planned_route = str(planned.get("model_route", "")).strip()
                if planned_route in SYNTHETIC_VISUAL_ROUTES and _planned_route_matches_difference(
                    planned_route,
                    difference_type,
                ):
                    route = planned_route
                planner_metadata = {
                    "stage": "strongest_omni_prompt_planner",
                    "input": "short_clip_reference_video",
                    "output": "source_prompt_target_prompt_edit_token_preserve_constraints",
                    "fallback_used": False,
                    "cache_hit": cached_planner_record is not None,
                    "model": model,
                    "reason": str(planned.get("reason", "")).strip(),
                    "repaired_fields": list(planned.get("repaired_fields", [])),
                }
            except Exception as exc:
                raw_planner_output = {"error": f"{type(exc).__name__}: {exc}"}
                skipped_reasons["model_planner_fallback"] += 1
                planner_metadata = {
                    "stage": "heuristic_prompt_planner",
                    "input": "annotation_and_candidate_edit",
                    "output": "source_prompt_target_prompt_edit_token_preserve_constraints",
                    "fallback_used": True,
                    "model": model,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
        suitability = _video_edit_route_suitability(
            route=route,
            difference=difference,
            edit_text=edit_text,
            edit_token=edit_token,
            edit_region=edit_region,
            reference_annotation=reference_annotation,
        )
        if not suitability["allow_generation"]:
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons[str(suitability["reason"])] += 1
            continue
        default_mask_query = _video_mask_query(
            difference=difference,
            edit_text=edit_text,
            edit_token=edit_token,
            edit_region=edit_region,
            route=route,
            suitability=suitability,
            reference_annotation=reference_annotation,
        )
        mask_query = default_mask_query
        if planned_mask_query:
            if not (
                str(difference.get("type", "")).strip() == "scene"
                and _normalized_phrase(planned_mask_query) == "background"
            ):
                mask_query = planned_mask_query
        target_instance_description = str(
            candidate.get("target_instance_description")
            or raw_planner_output.get("target_instance_description", "")
        ).strip()
        if target_instance_description and (
            _is_existing_object_replacement(difference, edit_text)
            or _is_object_removal(difference, edit_text)
        ):
            mask_query = target_instance_description
        mask_query = _video_mask_query_for_plan(
            {
                "difference": difference,
                "edit_text": edit_text,
                "edit_token": edit_token,
                "edit_region": edit_region,
                "exploration_family": str(candidate.get("exploration_family", "")).strip(),
                "reference_understanding": _video_edit_reference_understanding(reference_annotation),
            },
            mask_query,
        )
        target_prompt, preserve_tokens, negative_prompt, prompt_repairs = _repair_video_edit_prompt_contract(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_text=edit_text,
            difference=difference,
            edit_token=edit_token,
            preserve_tokens=preserve_tokens,
            negative_prompt=negative_prompt,
            mask_query=mask_query,
            risk=risk,
        )
        background_replace_plan = _is_background_replace_edit(
            difference,
            edit_text,
            edit_region=edit_region,
            mask_query=mask_query,
            target_prompt=target_prompt,
        )
        if background_replace_plan:
            repaired_risk = _background_replace_risk_locks(risk)
            if repaired_risk != risk:
                risk = repaired_risk
                prompt_repairs.append("visual_edit_risk_locks_rewritten_for_background_replace")
            repaired_preserve_regions = _filter_background_replace_preserve_regions(planned_preserve_regions)
            if repaired_preserve_regions != planned_preserve_regions:
                planned_preserve_regions = repaired_preserve_regions
                prompt_repairs.append("preserve_regions_rewritten_for_background_replace")
            suitability = dict(suitability)
            suitability.update(
                {
                    "production_allowed": True,
                    "plain_masked_vace_production": False,
                    "recommended_route": DETERMINISTIC_BG_COMPOSITE_ROUTE,
                    "fallback_route": VACE_BG_REPLACE_COMPOSITE_ROUTE,
                    "refine_route": GUIDED_COMPOSITE_REFINE_VACE_ROUTE,
                    "reason": "background_replace_prefers_fixed_deterministic_composite",
                    "priority": "production_candidate",
                    "requires_vace": False,
                    "route_decision_source": "omni_planner_plus_local_policy",
                }
            )
        if prompt_repairs:
            planner_metadata = dict(planner_metadata)
            repaired_fields = list(planner_metadata.get("repaired_fields", []))
            repaired_fields.extend(prompt_repairs)
            planner_metadata["repaired_fields"] = sorted(set(repaired_fields))
            planner_metadata["post_lint_repaired"] = True
        plan_lint = _video_edit_plan_lint(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_text=edit_text,
            difference=difference,
            edit_token=edit_token,
            preserve_tokens=preserve_tokens,
            negative_prompt=negative_prompt,
            reference_annotation=reference_annotation,
            mask_query=mask_query,
            preserve_regions=planned_preserve_regions,
            risk=risk,
            target_instance_description=target_instance_description,
        )
        if not plan_lint["passed"]:
            skipped_by_type[difference_type or "unknown"] += 1
            for reason in plan_lint["errors"]:
                skipped_reasons[f"plan_lint_{reason}"] += 1
            continue
        plan_key = (
            reference_video,
            str(difference.get("type", "")).strip(),
            _normalized_phrase(str(difference.get("from", "")).strip()),
            _normalized_phrase(str(difference.get("to", "")).strip()),
            _normalized_phrase(edit_text),
        )
        if plan_key in seen_plan_keys:
            skipped_reasons["duplicate_exploration_plan"] += 1
            continue
        control_plan = _video_edit_control_plan(route)
        preserve_regions = _video_preserve_regions(
            preserve_tokens=preserve_tokens,
            edit_region=edit_region,
            reference_annotation=reference_annotation,
        )
        if background_replace_plan:
            preserve_regions = _filter_background_replace_preserve_regions(preserve_regions)
        if planned_preserve_regions:
            preserve_regions = planned_preserve_regions
        background_replace_policy = _background_replace_route_policy() if background_replace_plan else {}
        mask_plan_name = "grounded_sam2_video_mask" if route == "vace_controlled" else (
            "none" if route == "audio_deterministic" else "local_roi"
        )
        src_ref_requirements = _src_ref_requirement_for_video_plan(
            {
                "difference": difference,
                "edit_text": edit_text,
                "edit_token": edit_token,
                "edit_region": edit_region,
                "model_route": route,
                "exploration_family": str(candidate.get("exploration_family", "")).strip(),
            }
        )
        plan = {
            "plan_id": str(candidate.get("proposal_id", "")).strip()
            or f"video_edit_plan_{_stable_hash(reference_video + edit_text)}",
            "reference_video": reference_video,
            "source_candidate_edit_text": str(candidate.get("source_candidate_edit_text", edit_text)).strip(),
            "source_candidate_difference": candidate.get("source_candidate_difference", difference),
            "edit_text": edit_text,
            "planner": planner_metadata,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "edit_token": edit_token,
            "preserve_tokens": preserve_tokens,
            "negative_prompt": negative_prompt,
            "edit_region": edit_region,
            "mask_query": mask_query,
            "target_instance_description": target_instance_description,
            "preserve_regions": preserve_regions,
            "mask_plan": mask_plan_name,
            "control_plan": control_plan,
            "model_route": route,
            "route": "vace14b_masked_v2v" if route == "vace_controlled" else route,
            "vace_inputs": {
                "src_video": reference_video,
                "src_mask": "to_be_generated" if route == "vace_controlled" else "",
                "src_ref_images": [],
            },
            "src_ref_requirements": src_ref_requirements,
            "difference": difference,
            "raw_planner_output": raw_planner_output,
            "reference_understanding": _video_edit_reference_understanding(reference_annotation),
            "route_suitability": suitability,
            "plan_lint": plan_lint,
            "visual_edit_risk": risk,
            "background_replace_policy": background_replace_policy,
            "route_execution_order": (
                [
                    DETERMINISTIC_BG_COMPOSITE_ROUTE,
                    GUIDED_COMPOSITE_REFINE_VACE_ROUTE,
                    VACE_BG_REPLACE_COMPOSITE_ROUTE,
                ]
                if background_replace_plan
                else [route]
            ),
            "planning_mode": planning_mode,
            "exploration_family": str(candidate.get("exploration_family", "")).strip(),
            "exploration_goal": str(candidate.get("exploration_goal", "")).strip(),
            "generation_defaults": _video_edit_generation_defaults(route),
            "validation_requirements": {
                "visual_near_duplicate_min": MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE,
                "preserve_reference_audio": route != "audio_deterministic",
                "single_edit_token": True,
                "requires_mask": route == "vace_controlled",
                "outside_mask_visual_near_duplicate_min": MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE,
                "mask_gate": _video_mask_gate_defaults(
                    mask_mode=_video_mask_mode({"difference": difference, "edit_region": edit_region, "mask_query": mask_query}),
                    mask_query=mask_query,
                    plan={
                        "difference": difference,
                        "edit_text": edit_text,
                        "edit_token": edit_token,
                        "exploration_family": str(candidate.get("exploration_family", "")).strip(),
                    },
                )
                if route == "vace_controlled"
                else {},
            },
        }
        plans.append(plan)
        seen_plan_keys.add(plan_key)
        if route != "audio_deterministic" and planning_mode != "exploration":
            seen_sources.add(reference_video)

    if planner_cache_dirty:
        _write_jsonl(cache_output, list(planner_cache.values()))
    _write_jsonl(output, plans)
    return {
        "candidate_count": original_candidate_count,
        "expanded_candidate_count": len(candidates),
        "plan_count": len(plans),
        "planning_mode": planning_mode,
        "output_path": str(output),
        "planner_cache_path": str(cache_output),
        "planner_cache_hits": cache_hits,
        "planner_cache_misses": cache_misses,
        "skipped_by_type": dict(skipped_by_type),
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_audio_edits(
    *,
    root: str | Path,
    pair_candidates_path: str | Path,
    clip_annotations_path: str | Path,
    output_path: str | Path | None = None,
    max_plans: int = 10,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    candidates = list(_load_jsonl(Path(pair_candidates_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not candidates:
        raise ValueError("pair candidates file is empty")
    if not annotations:
        raise ValueError("clip annotations are empty")

    annotation_lookup = _annotation_lookup(root=layout["root"], annotations=annotations)
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_AUDIO_EDIT_PLAN_NAME
    plans: list[dict[str, Any]] = []
    skipped_by_type: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    for candidate in candidates:
        if len(plans) >= max_plans:
            break
        difference = dict(candidate.get("difference") or {})
        difference_type = str(difference.get("type", "")).strip()
        edit_text = str(candidate.get("edit_text", "")).strip() or _build_fallback_edit_text(difference)
        reference_video = str(candidate.get("reference_video", "")).strip()
        if not reference_video:
            skipped_by_type[difference_type] += 1
            skipped_reasons["missing_reference_video"] += 1
            continue
        reference_annotation = _annotation_for_video_edit_plan(
            root=layout["root"],
            lookup=annotation_lookup,
            record=candidate,
            video_field="reference_video",
            caption_field="reference_caption",
        )
        source_candidate_edit_text = str(candidate.get("source_candidate_edit_text", edit_text)).strip()
        source_candidate_difference = candidate.get("source_candidate_difference", difference)
        speech_issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
        if speech_issues or difference_type == "speech":
            skipped_by_type[difference_type or "unknown"] += 1
            skipped_reasons["speech_content_or_speech_only_audio"] += 1
            continue
        if difference_type != "audio_event":
            ideation_candidate = _safe_audio_ideation_candidate(candidate, reference_annotation)
            if ideation_candidate is None:
                skipped_by_type[difference_type or "unknown"] += 1
                skipped_reasons["not_audio_event"] += 1
                continue
            candidate = ideation_candidate
            difference = dict(candidate.get("difference") or {})
            difference_type = str(difference.get("type", "")).strip()
            edit_text = str(candidate.get("edit_text", "")).strip()
            source_candidate_edit_text = str(candidate.get("source_candidate_edit_text", source_candidate_edit_text)).strip()
            source_candidate_difference = candidate.get("source_candidate_difference", source_candidate_difference)
            skipped_reasons["safe_audio_ideation_from_non_audio_candidate"] += 1
        speech_issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
        if speech_issues:
            skipped_by_type[difference_type] += 1
            skipped_reasons["speech_content_or_speech_only_audio"] += 1
            continue
        expected_event = _audio_expected_event(difference, edit_text)
        if not expected_event:
            skipped_by_type[difference_type] += 1
            skipped_reasons["missing_expected_audio_event"] += 1
            continue
        plan_id = str(candidate.get("proposal_id", "")).strip() or f"audio_edit_plan_{_stable_hash(reference_video + edit_text)}"
        route = _audio_edit_route(expected_event, reference_annotation)
        suitability = _audio_edit_route_suitability(
            expected_event=expected_event,
            difference=difference,
            edit_text=edit_text,
            reference_annotation=reference_annotation,
        )
        if not suitability["allow_generation"]:
            skipped_by_type[difference_type] += 1
            skipped_reasons[str(suitability["reason"])] += 1
            continue
        target_video = str(candidate.get("target_video", "")).strip() or f"clips/synthetic_audio/{plan_id}.mp4"
        audio_plan = {
            "route": route,
            "audio_prompt": _audio_edit_prompt(expected_event, reference_annotation, edit_text),
            "negative_audio_prompt": "speech, narration, talking, voiceover, crowd chatter, unrelated music",
            "timing_strategy": _audio_timing_strategy(expected_event, reference_annotation),
            "preserve_video": True,
            "mixing": "overlay",
            "expected_event": expected_event,
        }
        plans.append(
            {
                "plan_id": plan_id,
                "reference_video": reference_video,
                "target_video": target_video,
                "source_candidate_edit_text": source_candidate_edit_text,
                "source_candidate_difference": source_candidate_difference,
                "edit_text": edit_text,
                "difference": difference,
                "planner": {
                    "stage": "strongest_omni_audio_prompt_planner",
                    "input": "short_clip_reference_video_and_audio_understanding",
                    "output": "non_speech_audio_event_plan",
                },
                "audio_reference_understanding": _audio_edit_reference_understanding(reference_annotation),
                "route_suitability": suitability,
                "audio_edit_plan": audio_plan,
                "generation_defaults": {
                    "preserve_video_stream": True,
                    "generate_video": False,
                    "visual_near_duplicate_min": MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE,
                    "duration_drift_max": 0.10,
                },
            }
        )

    _write_jsonl(output, plans)
    return {
        "candidate_count": len(candidates),
        "plan_count": len(plans),
        "output_path": str(output),
        "skipped_by_type": dict(skipped_by_type),
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_stable_omni_clips(
    *,
    root: str | Path,
    raw_index_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    max_source_videos: int = 50,
    min_clip_seconds: float = 5.0,
    max_clip_seconds: float = 8.0,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    raw_index_file = Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME
    raw_index = _load_raw_asset_index(raw_index_file)
    if not raw_index:
        raise ValueError("raw asset index is empty")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")
    if max_clip_seconds < min_clip_seconds:
        raise ValueError("max_clip_seconds must be >= min_clip_seconds")

    output = Path(output_path) if output_path else layout["metadata"] / "omni_stable_clip_plan.jsonl"
    cache_output = Path(cache_path) if cache_path else layout["caches"] / DEFAULT_OMNI_STABLE_CLIP_SELECTION_CACHE_NAME
    cache = _load_records_by_key(cache_output, "cache_key")
    _write_jsonl(output, [])
    client = (
        OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model or "",
            timeout_seconds=timeout_seconds,
        )
        if base_url and model
        else None
    )

    plan_records: list[dict[str, Any]] = []
    cache_records = dict(cache)
    cache_hits = 0
    cache_misses = 0
    skipped_reasons: Counter[str] = Counter()
    assets = sorted(raw_index.values(), key=lambda item: (str(item.get("dataset", "")), str(item.get("relative_path", ""))))
    for asset in assets[: max(0, max_source_videos)]:
        source_path = Path(str(asset.get("path", "")).strip())
        if not source_path.exists():
            skipped_reasons["missing_source_video"] += 1
            continue
        media = probe_media(source_path)
        duration = float(media.get("duration_seconds") or 0.0)
        if duration < min_clip_seconds:
            skipped_reasons["too_short"] += 1
            continue

        cache_key = _stable_json_hash(
            {
                "asset_id": asset.get("asset_id"),
                "path": str(source_path),
                "mtime_ns": asset.get("mtime_ns"),
                "min_clip_seconds": min_clip_seconds,
                "max_clip_seconds": max_clip_seconds,
                "model": model or "",
            }
        )
        cached_record = cache.get(cache_key)
        if cached_record is not None:
            cache_hits += 1
            selection = dict(cached_record.get("selection") or {})
        else:
            cache_misses += 1
            selection = _heuristic_stable_clip_selection(
                media=media,
                min_clip_seconds=min_clip_seconds,
                max_clip_seconds=max_clip_seconds,
            )
            if client is not None:
                try:
                    model_selection, raw_payload = client.select_stable_clip_window(
                        source_video_path=str(source_path),
                        media_info=media,
                        min_clip_seconds=min_clip_seconds,
                        max_clip_seconds=max_clip_seconds,
                    )
                    selection = _coerce_stable_clip_selection(
                        model_selection,
                        fallback=selection,
                        media=media,
                        min_clip_seconds=min_clip_seconds,
                        max_clip_seconds=max_clip_seconds,
                    )
                    selection["planner"] = {
                        "stage": "strongest_omni_stable_clip_selector",
                        "fallback_used": False,
                        "model": model,
                        "raw_payload": raw_payload,
                    }
                except Exception as exc:
                    selection["planner"] = {
                        "stage": "heuristic_stable_clip_selector",
                        "fallback_used": True,
                        "model": model,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    }
            else:
                selection["planner"] = {
                    "stage": "heuristic_stable_clip_selector",
                    "fallback_used": True,
                    "reason": "no Omni endpoint supplied",
                }
            cache_record = {
                "cache_key": cache_key,
                "asset_id": asset.get("asset_id"),
                "source_video": str(source_path),
                "selection": selection,
            }
            cache_records[cache_key] = cache_record
            _append_jsonl_record(cache_output, cache_record)

        if not bool(selection.get("recommended_for_vace", True)):
            skipped_reasons["not_recommended_for_vace"] += 1
            continue
        start_seconds = float(selection.get("start_sec", 0.0) or 0.0)
        end_seconds = float(selection.get("end_sec", 0.0) or 0.0)
        if end_seconds - start_seconds < min_clip_seconds or end_seconds - start_seconds > max_clip_seconds + 1e-6:
            skipped_reasons["invalid_selected_window"] += 1
            continue

        dataset = str(asset.get("dataset", "raw")).strip() or "raw"
        clip_id = f"{dataset}_{Path(str(asset.get('relative_path', source_path.name))).stem}__omni_{_stable_hash(cache_key)[:8]}"
        clip_record = {
            "clip_id": clip_id,
            "source_asset_id": str(asset.get("asset_id", "")).strip(),
            "source_path": str(source_path),
            "output_path": f"clips/omni_stable/{clip_id}.mp4",
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "duration_seconds": round(end_seconds - start_seconds, 3),
            "role": "reference",
            "notes": "Omni-selected stable short clip for VACE planning",
            "stable_clip_selection": selection,
        }
        plan_records.append(clip_record)
        _append_jsonl_record(output, clip_record)

    _write_jsonl(output, plan_records)
    _write_jsonl(cache_output, list(cache_records.values()))
    return {
        "raw_index_path": str(raw_index_file),
        "clip_plan_count": len(plan_records),
        "output_path": str(output),
        "cache_path": str(cache_output),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "skipped_reasons": dict(skipped_reasons),
    }


def cache_reference_understandings(
    *,
    root: str | Path,
    clip_annotations_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not annotations:
        raise ValueError("clip annotations are empty")
    output = Path(output_path) if output_path else layout["caches"] / DEFAULT_REFERENCE_UNDERSTANDING_CACHE_NAME
    records: list[dict[str, Any]] = []
    skipped_unusable_count = 0
    for annotation in annotations:
        if not _annotation_is_usable_for_reference_understanding(annotation):
            skipped_unusable_count += 1
            continue
        reference_video = str(annotation.get("output_path") or annotation.get("source_path") or "").strip()
        clip_id = str(annotation.get("clip_id", "")).strip() or _stable_hash(reference_video)
        visual_understanding = _video_edit_reference_understanding(annotation)
        audio_understanding = _audio_edit_reference_understanding(annotation)
        stable_targets = _stable_edit_targets_from_understanding(visual_understanding, annotation)
        records.append(
            {
                "clip_id": clip_id,
                "reference_video": reference_video,
                "summary": str(annotation.get("summary", "")).strip(),
                "subjects": _dedupe_strings(_normalize_list(annotation.get("subjects", []))),
                "actions": _dedupe_strings(_normalize_list(annotation.get("actions", []))),
                "scene": str(annotation.get("scene", "")).strip(),
                "camera_motion": str(annotation.get("camera_motion", "")).strip() or "unknown",
                "visible_text": _dedupe_strings(
                    _normalize_list(annotation.get("visible_text", []))
                    + _normalize_list(annotation.get("on_screen_text", []))
                ),
                "stable_edit_targets": stable_targets,
                "bad_edits": visual_understanding.get("bad_edits", []),
                "reference_understanding": visual_understanding,
                "audio_reference_understanding": audio_understanding,
            }
        )
    _write_jsonl(output, records)
    return {
        "clip_annotations_path": str(clip_annotations_path),
        "understanding_count": len(records),
        "skipped_unusable_annotation_count": skipped_unusable_count,
        "output_path": str(output),
    }


def plan_video_masks(
    *,
    root: str | Path,
    video_edit_plan_path: str | Path,
    output_path: str | Path | None = None,
    mask_manifest_path: str | Path | None = None,
    max_masks: int | None = None,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    edit_plans = list(_load_jsonl(Path(video_edit_plan_path)))
    if not edit_plans:
        raise ValueError("video edit plan file is empty")

    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_VIDEO_MASK_PLAN_NAME
    manifest_output = Path(mask_manifest_path) if mask_manifest_path else output.with_name(DEFAULT_VIDEO_MASK_MANIFEST_NAME)
    mask_dir = output.parent / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_plans: list[dict[str, Any]] = []
    mask_manifest: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    for edit_plan in edit_plans:
        if max_masks is not None and max_masks > 0 and len(mask_plans) >= max_masks:
            break
        if str(edit_plan.get("model_route", "")).strip() != "vace_controlled":
            skipped_reasons["non_vace_route"] += 1
            continue
        plan_id = str(edit_plan.get("plan_id", "")).strip()
        if not plan_id:
            skipped_reasons["missing_plan_id"] += 1
            continue
        mask_query = str(edit_plan.get("mask_query", "")).strip() or _video_mask_query(
            difference=dict(edit_plan.get("difference") or {}),
            edit_text=str(edit_plan.get("edit_text", "")).strip(),
            edit_token=str(edit_plan.get("edit_token", "")).strip(),
            edit_region=str(edit_plan.get("edit_region", "")).strip(),
            route=str(edit_plan.get("model_route", "")).strip(),
            suitability=edit_plan.get("route_suitability") if isinstance(edit_plan.get("route_suitability"), dict) else {},
            reference_annotation=edit_plan.get("reference_understanding")
            if isinstance(edit_plan.get("reference_understanding"), dict)
            else {},
        )
        mask_query = _video_mask_query_for_plan(edit_plan, mask_query)
        if not mask_query:
            skipped_reasons["missing_mask_query"] += 1
            continue
        reference_video = str(edit_plan.get("reference_video", "")).strip()
        reference_path = _resolve_under_root(layout["root"], reference_video)
        if not reference_video or not reference_path.exists():
            skipped_reasons["missing_reference_video"] += 1
            continue
        mask_mode = _video_mask_mode({**edit_plan, "mask_query": mask_query})
        maskability_issue = _video_maskability_issue(edit_plan, mask_query=mask_query, mask_mode=mask_mode)
        if maskability_issue:
            skipped_reasons[maskability_issue] += 1
        mask_query_candidates = _video_mask_query_candidates_for_plan(edit_plan, mask_query)
        safe_id = _safe_id(plan_id)
        mask_video = mask_dir / f"{safe_id}_mask.mp4"
        mask_record = {
            "plan_id": plan_id,
            "reference_video": reference_video,
            "reference_video_absolute": str(reference_path),
            "mask_video": str(mask_video),
            "mask_query": mask_query,
            "mask_query_candidates": mask_query_candidates,
            "mask_mode": mask_mode,
            "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
            "mask_polarity": VIDEO_MASK_POLARITY,
            "edit_region": str(edit_plan.get("edit_region", "")).strip(),
            "preserve_regions": _video_preserve_regions(
                preserve_tokens=_normalize_list(edit_plan.get("preserve_tokens", [])),
                edit_region=str(edit_plan.get("edit_region", "")).strip(),
                reference_annotation={},
            ),
            "toolchain": {
                "grounder": "GroundingDINO_or_Florence-2",
                "segmenter": "SAM2.1_video_predictor",
                "wrapper": "Grounded-SAM-2",
            },
            "mask_gate": _video_mask_gate_defaults(mask_mode=mask_mode, mask_query=mask_query, plan=edit_plan),
            "mask_generation_strategy": "adaptive_repair_v1",
            "generate_diagnostic_mask": bool(maskability_issue),
            "maskability_issue": maskability_issue,
            "usable_for_vace_default": not bool(maskability_issue),
            "status": "planned",
        }
        mask_plans.append(mask_record)
        mask_manifest.append(
            {
                "plan_id": plan_id,
                "reference_video": reference_video,
                "mask_video": str(mask_video),
                "mask_query": mask_query,
                "mask_query_candidates": mask_query_candidates,
                "mask_mode": mask_mode,
                "mask_semantics_version": VIDEO_MASK_SEMANTICS_VERSION,
                "mask_polarity": VIDEO_MASK_POLARITY,
                "mask_generation_strategy": "adaptive_repair_v1",
                "generate_diagnostic_mask": bool(maskability_issue),
                "maskability_issue": maskability_issue,
                "usable_for_vace_default": not bool(maskability_issue),
                "status": "planned",
            }
        )

    _write_jsonl(output, mask_plans)
    _write_jsonl(manifest_output, mask_manifest)
    return {
        "video_edit_plan_path": str(video_edit_plan_path),
        "mask_plan_count": len(mask_plans),
        "output_path": str(output),
        "mask_manifest_path": str(manifest_output),
        "skipped_reasons": dict(skipped_reasons),
    }


def plan_src_ref_images(
    *,
    root: str | Path,
    video_edit_plan_path: str | Path,
    output_path: str | Path | None = None,
    image_root: str | Path | None = None,
    num_candidates: int = 4,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    edit_plans = list(_load_jsonl(Path(video_edit_plan_path)))
    if not edit_plans:
        raise ValueError("video edit plan file is empty")
    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_SRC_REF_IMAGE_PLAN_NAME
    image_base = Path(image_root) if image_root else output.parent / "src_ref_images"
    plans: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    for edit_plan in edit_plans:
        requirement = _src_ref_requirement_for_video_plan(edit_plan)
        if not requirement.get("required") and not requirement.get("recommended"):
            skipped_reason = (
                "structural_clothing_tryon_required"
                if str(requirement.get("reason", "")).strip() == "structural_clothing_tryon_required"
                else "src_ref_not_needed"
            )
            skipped_reasons[skipped_reason] += 1
            continue
        plan_id = str(edit_plan.get("plan_id", "")).strip()
        if not plan_id:
            skipped_reasons["missing_plan_id"] += 1
            continue
        target = str(requirement.get("target", "")).strip()
        if not target:
            skipped_reasons["missing_src_ref_target"] += 1
            continue
        safe_id = _safe_id(plan_id)
        candidate_dir = image_base / safe_id
        plans.append(
            {
                "plan_id": plan_id,
                "reference_video": str(edit_plan.get("reference_video", "")).strip(),
                "edit_text": str(edit_plan.get("edit_text", "")).strip(),
                "difference": edit_plan.get("difference", {}),
                "target_object": target,
                "src_ref_role": str(requirement.get("role", "")).strip(),
                "required": bool(requirement.get("required")),
                "recommended": bool(requirement.get("recommended")),
                "image_prompts": _src_ref_image_prompts(requirement=requirement, edit_plan=edit_plan),
                "negative_prompt": _src_ref_image_negative_prompt(requirement),
                "num_candidates": max(1, int(num_candidates)),
                "candidate_dir": str(candidate_dir),
                "image_width": VACE_BACKGROUND_SRC_REF_WIDTH
                if str(requirement.get("role", "")).strip() == "background_reference"
                else 0,
                "image_height": VACE_BACKGROUND_SRC_REF_HEIGHT
                if str(requirement.get("role", "")).strip() == "background_reference"
                else 0,
                "planner": {
                    "stage": "src_ref_image_requirement_planner",
                    "input": "video_edit_plan_and_omni_reference_understanding",
                    "output": "image_generation_prompts_for_vace_src_ref_images",
                },
            }
        )
    _write_jsonl(output, plans)
    return {
        "video_edit_plan_path": str(video_edit_plan_path),
        "plan_count": len(plans),
        "output_path": str(output),
        "image_root": str(image_base),
        "skipped_reasons": dict(skipped_reasons),
    }


def select_src_ref_images(
    *,
    root: str | Path,
    src_ref_image_plan_path: str | Path,
    output_path: str | Path | None = None,
    max_selected: int = 2,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    model: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    ensure_layout(root)
    image_plans = list(_load_jsonl(Path(src_ref_image_plan_path)))
    if not image_plans:
        raise ValueError("src_ref image plan file is empty")
    output = Path(output_path) if output_path else Path(src_ref_image_plan_path).with_name(DEFAULT_SRC_REF_IMAGE_SELECTION_NAME)
    audit_client = (
        OpenAIComposedDataClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        if base_url and model
        else None
    )
    records: list[dict[str, Any]] = []
    selected_count = 0
    missing_count = 0
    audit_rejected_count = 0
    audit_failed_count = 0
    for plan in image_plans:
        candidate_dir = Path(str(plan.get("candidate_dir", "")).strip())
        candidates = _find_src_ref_image_candidates(candidate_dir)
        audited = sorted(
            (_audit_src_ref_image_candidate(path, plan) for path in candidates),
            key=lambda item: (-float(item.get("score", 0.0)), str(item.get("path", ""))),
        )
        eligible_audited = [item for item in audited if bool(item.get("eligible", True))]
        role = str(plan.get("src_ref_role", "")).strip()
        selection_limit = max(1, int(max_selected))
        if role == "replacement_object":
            selection_limit = min(selection_limit, 2)
        if role == "clothing_reference":
            selection_limit = 1
        selected_audits = eligible_audited[:selection_limit]
        selection_method = "deterministic_src_ref_quality_audit"
        selection_reason = "selected highest-scoring candidate image(s) by deterministic VACE src_ref quality audit"
        omni_audit: dict[str, Any] | None = None
        raw_omni_audit: dict[str, Any] | None = None
        if audit_client and eligible_audited:
            try:
                candidate_image_paths = [str(item["path"]) for item in eligible_audited]
                omni_audit, raw_omni_audit = audit_client.audit_src_ref_images(
                    src_ref_plan=plan,
                    candidate_image_paths=candidate_image_paths,
                    max_selected=selection_limit,
                )
                selected_indices = [
                    int(index)
                    for index in omni_audit.get("selected_indices", [])
                    if isinstance(index, int) or str(index).isdigit()
                ]
                selected_audits = [
                    eligible_audited[index - 1]
                    for index in selected_indices
                    if 1 <= index <= len(eligible_audited)
                ]
                selection_method = "omni_src_ref_image_audit"
                selection_reason = str(omni_audit.get("reason", "")).strip() or (
                    "selected candidate image(s) by Omni src_ref visual audit"
                    if selected_audits
                    else "Omni audit rejected all generated candidate images"
                )
            except Exception as exc:
                selected_audits = []
                selection_method = "omni_src_ref_image_audit_failed"
                selection_reason = f"Omni src_ref audit failed: {exc}"
                audit_failed_count += 1
        selected = [str(item["path"]) for item in selected_audits]
        selected_set = set(selected)
        rejected = [
            {
                "path": str(item.get("path", "")),
                "reason": (
                    "not selected by Omni src_ref audit"
                    if audit_client and audited
                    else "lower deterministic src_ref audit score"
                ),
                "audit": item,
            }
            for item in audited
            if str(item.get("path", "")) not in selected_set
        ]
        if selected:
            status = "selected"
            selected_count += 1
        elif audited and not eligible_audited:
            status = "rejected_by_deterministic_audit"
            selection_reason = "all generated candidate images failed deterministic VACE src_ref quality gates"
            audit_rejected_count += 1
        elif audited and audit_client:
            status = "rejected_by_omni_audit" if selection_method == "omni_src_ref_image_audit" else "omni_audit_failed"
            audit_rejected_count += int(selection_method == "omni_src_ref_image_audit")
        else:
            status = "missing_candidates"
            selection_reason = "no generated candidate images found"
            missing_count += 1
        record = {
            "plan_id": str(plan.get("plan_id", "")).strip(),
            "selected_src_ref_images": selected,
            "rejected": rejected,
            "status": status,
            "required": bool(plan.get("required")),
            "src_ref_role": str(plan.get("src_ref_role", "")).strip(),
            "selection_reason": selection_reason,
            "selection_method": selection_method,
            "candidate_dir": str(candidate_dir),
            "candidate_audit": audited,
        }
        if omni_audit is not None:
            record["omni_audit"] = omni_audit
        if raw_omni_audit is not None:
            record["raw_omni_audit"] = raw_omni_audit
        records.append(record)
    _write_jsonl(output, records)
    return {
        "src_ref_image_plan_path": str(src_ref_image_plan_path),
        "selection_count": len(records),
        "selected_plan_count": selected_count,
        "missing_candidate_count": missing_count,
        "audit_rejected_count": audit_rejected_count,
        "audit_failed_count": audit_failed_count,
        "output_path": str(output),
    }


def _manual_review_bundle_issues(metadata: dict[str, Any]) -> list[str]:
    generation = metadata.get("generation", {}) if isinstance(metadata.get("generation"), dict) else {}
    route = _synthetic_generation_route(generation)
    if route not in SYNTHETIC_VISUAL_ROUTES:
        return []
    background_route = _background_replace_actual_route(generation)
    deterministic_background = background_route == DETERMINISTIC_BG_COMPOSITE_ROUTE
    issues: list[str] = []
    if not metadata.get("copied_src_video_for_vace"):
        issues.append("incomplete_review_bundle: missing src_video_for_vace")
    if (route == "vace_controlled" or deterministic_background) and not metadata.get("copied_src_mask"):
        issues.append("incomplete_review_bundle: missing src_mask")
    if not metadata.get("copied_raw_generated_video"):
        issues.append("incomplete_review_bundle: missing raw_generated_video")
    src_ref_requirements = (
        generation.get("src_ref_requirements", {}) if isinstance(generation.get("src_ref_requirements"), dict) else {}
    )
    if src_ref_requirements.get("required") and not metadata.get("copied_src_ref_images"):
        issues.append("incomplete_review_bundle: missing required src_ref_images")
    copied_review_inputs = str(metadata.get("copied_review_inputs", "")).strip()
    if not copied_review_inputs:
        issues.append("incomplete_review_bundle: missing review_inputs_dir")
    else:
        review_inputs_path = Path(copied_review_inputs)
        required_review_inputs = [
            "vace_prompt.txt",
            "preflight_report.json",
            "duration_metrics.json",
            "vace_command.json",
            "reference_contact.jpg",
            "src_video_contact.jpg",
            "raw_target_contact.jpg",
            "target_contact.jpg",
        ]
        if route == "vace_controlled" or deterministic_background:
            required_review_inputs.append("mask_contact.jpg")
        if deterministic_background:
            required_review_inputs.extend(
                [
                    "src_ref_plate.png",
                    "alpha_contact.jpg",
                    "composite_target_contact.jpg",
                    "deterministic_composite_metrics.json",
                    "deterministic_composite_command.json",
                    "post_vace_or_composite_verdict.json",
                ]
            )
        for filename in required_review_inputs:
            if not (review_inputs_path / filename).exists():
                issues.append(f"incomplete_review_bundle: missing review_inputs/{filename}")
    if not metadata.get("duration_metrics"):
        issues.append("incomplete_review_bundle: missing duration_metrics")
    if route == "vace_controlled" and not metadata.get("mask_metrics"):
        issues.append("incomplete_review_bundle: missing mask_metrics")
    if not metadata.get("post_vace_verdict"):
        issues.append("incomplete_review_bundle: missing post_vace_verdict")
    if deterministic_background and not metadata.get("deterministic_composite_metrics"):
        issues.append("incomplete_review_bundle: missing deterministic_composite_metrics")
    return issues


def _audio_anchor_review_metadata(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    heuristic_quality = record.get("heuristic_quality", {}) if isinstance(record.get("heuristic_quality"), dict) else {}
    source_context = record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {}
    containers = [quality, heuristic_quality, source_context]

    def first_present(key: str) -> Any:
        for container in containers:
            value = container.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    raw_score = first_present("audio_anchor_score")
    raw_required = first_present("audio_anchor_required")
    raw_type = first_present("audio_anchor_type")
    if raw_score is None and raw_required is None and raw_type is None:
        return {}

    audio_anchor: dict[str, Any] = {
        "audio_anchor_score": _score_float(raw_score) if raw_score is not None else None,
        "audio_anchor_required": (_score_float(raw_required) >= 1.0) if raw_required is not None else bool(raw_score is not None),
        "audio_anchor_type": str(raw_type or "similar_or_same_natural_audio").strip(),
        "edit_primary_modality": str(first_present("edit_primary_modality") or "").strip(),
    }
    for key in ("audio_anchor_context_score", "audio_anchor_min_rms"):
        value = first_present(key)
        if value is not None:
            audio_anchor[key] = _score_float(value)
    warnings = first_present("audio_matters_warnings")
    if warnings is None:
        warnings = first_present("exploration_warnings")
    audio_anchor["audio_matters_warnings"] = _normalize_list(warnings)
    return audio_anchor


def build_manual_review_bundle(
    *,
    root: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    clip_annotations_path: str | Path | None = None,
    limit: int | None = None,
    copy_videos: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    pairs = list(_load_jsonl(Path(pairs_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path))) if clip_annotations_path else []
    annotation_lookup = _annotation_lookup(root=root_path, annotations=annotations) if annotations else {}
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    missing_videos: list[str] = []
    incomplete_review_bundle_count = 0
    selected_pairs = pairs[: limit if limit and limit > 0 else None]
    for index, record in enumerate(selected_pairs, start=1):
        sample_id = str(record.get("sample_id") or record.get("proposal_id") or f"sample_{index:04d}").strip()
        safe_sample_id = _safe_id(sample_id)
        item_dir = output_root / f"{index:04d}_{safe_sample_id}"
        item_dir.mkdir(parents=True, exist_ok=True)

        reference_video_raw = str(record.get("reference_video", "")).strip()
        target_video_raw = str(record.get("target_video", "")).strip()
        reference_path = _resolve_under_root(root_path, reference_video_raw) if reference_video_raw else Path()
        target_path = _resolve_under_root(root_path, target_video_raw) if target_video_raw else Path()
        reference_annotation = _review_annotation_for_record(
            root=root_path,
            lookup=annotation_lookup,
            record=record,
            video_field="reference_video",
            clip_id_field="reference_clip_id",
        )
        target_annotation = _review_annotation_for_record(
            root=root_path,
            lookup=annotation_lookup,
            record=record,
            video_field="target_video",
            clip_id_field="target_clip_id",
        )
        reference_caption = (
            str(record.get("reference_caption", "")).strip()
            or str(reference_annotation.get("summary", "")).strip()
            or str(reference_annotation.get("caption", "")).strip()
        )
        target_caption = (
            str(record.get("target_caption", "")).strip()
            or str(target_annotation.get("summary", "")).strip()
            or str(target_annotation.get("caption", "")).strip()
        )
        reference_copy = item_dir / "reference.mp4"
        target_copy = item_dir / "target.mp4"
        if copy_videos:
            if reference_path.exists():
                shutil.copy2(reference_path, reference_copy)
            else:
                missing_videos.append(str(reference_path or reference_video_raw))
            if target_path.exists():
                shutil.copy2(target_path, target_copy)
            else:
                missing_videos.append(str(target_path or target_video_raw))
        generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
        src_ref_images = _normalize_list(generation.get("src_ref_images", []))
        copied_src_ref_images: list[str] = []
        if copy_videos and src_ref_images:
            src_ref_dir = item_dir / "src_ref_images"
            src_ref_dir.mkdir(parents=True, exist_ok=True)
            for image_index, image_raw in enumerate(src_ref_images, start=1):
                image_path = _resolve_under_root(root_path, image_raw)
                if image_path.exists():
                    image_copy = src_ref_dir / f"{image_index:03d}_{_safe_id(image_path.name)}"
                    shutil.copy2(image_path, image_copy)
                    copied_src_ref_images.append(str(image_copy))
                else:
                    missing_videos.append(str(image_path or image_raw))
        src_mask = str(generation.get("src_mask", "")).strip()
        mask_copy_path = ""
        if copy_videos and src_mask:
            mask_path = _resolve_under_root(root_path, src_mask)
            if mask_path.exists():
                mask_copy = item_dir / "mask.mp4"
                shutil.copy2(mask_path, mask_copy)
                mask_copy_path = str(mask_copy)
            else:
                missing_videos.append(str(mask_path or src_mask))
        src_video_for_vace = str(generation.get("src_video_for_vace", "")).strip()
        src_video_for_vace_copy_path = ""
        if copy_videos and src_video_for_vace:
            src_video_for_vace_path = _resolve_under_root(root_path, src_video_for_vace)
            if src_video_for_vace_path.exists():
                src_video_for_vace_copy = item_dir / "src_video_for_vace.mp4"
                shutil.copy2(src_video_for_vace_path, src_video_for_vace_copy)
                src_video_for_vace_copy_path = str(src_video_for_vace_copy)
            else:
                missing_videos.append(str(src_video_for_vace_path or src_video_for_vace))
        postprocess = generation.get("postprocess", {}) if isinstance(generation.get("postprocess"), dict) else {}
        raw_generated_video = str(postprocess.get("raw_generated_video", "")).strip()
        raw_generated_video_copy_path = ""
        if copy_videos and raw_generated_video:
            raw_generated_path = _resolve_under_root(root_path, raw_generated_video)
            if raw_generated_path.exists():
                raw_generated_copy = item_dir / "raw_target.mp4"
                shutil.copy2(raw_generated_path, raw_generated_copy)
                raw_generated_video_copy_path = str(raw_generated_copy)
            else:
                missing_videos.append(str(raw_generated_path or raw_generated_video))
        review_inputs_dir = str(generation.get("review_inputs_dir", "")).strip()
        copied_review_inputs = ""
        if copy_videos and review_inputs_dir:
            review_inputs_path = _resolve_under_root(root_path, review_inputs_dir)
            if review_inputs_path.exists() and review_inputs_path.is_dir():
                copied_review_inputs_path = item_dir / "review_inputs"
                if copied_review_inputs_path.exists():
                    shutil.rmtree(copied_review_inputs_path)
                shutil.copytree(review_inputs_path, copied_review_inputs_path)
                copied_review_inputs = str(copied_review_inputs_path)
            else:
                missing_videos.append(str(review_inputs_path or review_inputs_dir))

        quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
        heuristic_quality = record.get("heuristic_quality", {}) if isinstance(record.get("heuristic_quality"), dict) else {}
        source_context = record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {}
        judge = record.get("judge", {}) if isinstance(record.get("judge"), dict) else {}
        audio_anchor = _audio_anchor_review_metadata(record)

        metadata = {
            "index": index,
            "sample_id": sample_id,
            "proposal_id": record.get("proposal_id"),
            "difference": record.get("difference", {}),
            "edit_text": str(record.get("edit_text", "")).strip(),
            "reference_video": reference_video_raw,
            "target_video": target_video_raw,
            "reference_video_absolute": str(reference_path) if reference_video_raw else "",
            "target_video_absolute": str(target_path) if target_video_raw else "",
            "reference_caption": reference_caption,
            "target_caption": target_caption,
            "reference_omni_description": _review_annotation_description(reference_annotation, fallback_caption=reference_caption),
            "target_omni_description": _review_annotation_description(target_annotation, fallback_caption=target_caption),
            "dominant_delta_decision": record.get("dominant_delta_decision", {}),
            "secondary_deltas": _normalize_list(
                (record.get("dominant_delta_decision", {}) if isinstance(record.get("dominant_delta_decision"), dict) else {}).get("secondary_delta_types", [])
            ),
            "verification": record.get("verification", {}),
            "observable_difference": record.get("observable_difference", {}),
            "competing_difference": record.get("competing_difference", {}),
            "audio_matters_line": str(record.get("audio_matters_line", "")).strip(),
            "audio_anchor_visual_verification": record.get("audio_anchor_visual_verification", {}),
            "quality": quality,
            "heuristic_quality": heuristic_quality,
            "source_context": source_context,
            "judge": judge,
            "audio_anchor": audio_anchor,
            "audio_anchor_score": audio_anchor.get("audio_anchor_score"),
            "audio_anchor_type": audio_anchor.get("audio_anchor_type", ""),
            "audio_matters_warnings": audio_anchor.get("audio_matters_warnings", []),
            "generation": generation,
            "src_ref_images": src_ref_images,
            "copied_src_ref_images": copied_src_ref_images,
            "src_mask": src_mask,
            "copied_src_mask": mask_copy_path,
            "src_video_for_vace": src_video_for_vace,
            "copied_src_video_for_vace": src_video_for_vace_copy_path,
            "raw_generated_video": raw_generated_video,
            "copied_raw_generated_video": raw_generated_video_copy_path,
            "copied_review_inputs": copied_review_inputs,
            "mask_metrics": generation.get("mask_metrics", {}),
            "duration_metrics": generation.get("duration_metrics", {}),
            "vace_command": generation.get("vace_command", {}),
            "deterministic_composite_metrics": generation.get("deterministic_composite_metrics", {}),
            "post_vace_or_composite_verdict": generation.get("post_vace_or_composite_verdict", generation.get("post_vace_verdict", {})),
            "post_vace_verdict": generation.get("post_vace_verdict", {}),
        }
        review_bundle_issues = _manual_review_bundle_issues(metadata)
        metadata["review_bundle_issues"] = review_bundle_issues
        metadata["incomplete_review_bundle"] = bool(review_bundle_issues)
        if review_bundle_issues:
            incomplete_review_bundle_count += 1
        final_verification = record.get("final_omni_verification", {}) if isinstance(record.get("final_omni_verification"), dict) else {}
        local_gate_report = record.get("local_gate_report", {}) if isinstance(record.get("local_gate_report"), dict) else {}
        review_metadata = {
            "line": str(record.get("audio_dataset_line", "")).strip(),
            "audio_line_quality_profile": str(record.get("audio_line_quality_profile") or quality.get("audio_line_quality_profile", "")).strip(),
            "edit_text": str(record.get("edit_text", "")).strip(),
            "difference": record.get("difference", {}),
            "reference_video": reference_video_raw,
            "target_video": target_video_raw,
            "visual_verdict": {
                "visual_delta_strength": quality.get("visual_delta_strength"),
                "visual_context_similarity": quality.get("visual_context_similarity"),
                "reference_satisfies_edit": final_verification.get("reference_satisfies_edit"),
                "target_satisfies_edit": final_verification.get("target_satisfies_edit"),
                "observable_delta": final_verification.get("observable_delta"),
                "single_primary_delta": final_verification.get("single_primary_delta"),
                "evidence": final_verification.get("evidence", []),
            },
            "audio_verdict": {
                "audio_anchor_score": audio_anchor.get("audio_anchor_score"),
                "audio_anchor_type": audio_anchor.get("audio_anchor_type", ""),
                "speech_evidence_score": quality.get("speech_evidence_score"),
                "speech_specificity_score": quality.get("speech_specificity_score"),
                "non_speech_audio_event_score": quality.get("non_speech_audio_event_score"),
                "audio_content_delta_strength": quality.get("audio_content_delta_strength"),
            },
            "omni_accept": bool(record.get("accepted")),
            "omni_reject_reason": str(judge.get("reject_reason", "")).strip(),
            "local_gate_report": local_gate_report,
            "pair_video_evidence": record.get("pair_video_evidence", []),
            "reference_omni_description": metadata["reference_omni_description"],
            "target_omni_description": metadata["target_omni_description"],
        }
        (item_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "review_metadata.json").write_text(
            json.dumps(review_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "semantic_evaluation_result.json").write_text(
            json.dumps(metadata.get("post_vace_or_composite_verdict", metadata.get("post_vace_verdict", {})), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "post_vace_or_composite_verdict.json").write_text(
            json.dumps(metadata.get("post_vace_or_composite_verdict", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "mask_metrics.json").write_text(
            json.dumps(metadata.get("mask_metrics", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "duration_metrics.json").write_text(
            json.dumps(metadata.get("duration_metrics", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        review_md = _manual_review_item_markdown(
            metadata=metadata,
            reference_filename="reference.mp4" if copy_videos and reference_copy.exists() else "",
            target_filename="target.mp4" if copy_videos and target_copy.exists() else "",
        )
        (item_dir / "review.md").write_text(review_md, encoding="utf-8")
        (item_dir / "description.md").write_text(
            _manual_review_description_markdown(metadata),
            encoding="utf-8",
        )
        items.append(
            {
                "index": index,
                "sample_id": sample_id,
                "difference_type": (record.get("difference") or {}).get("type") if isinstance(record.get("difference"), dict) else "",
                "edit_text": str(record.get("edit_text", "")).strip(),
                "item_dir": str(item_dir),
                "review_md": str(item_dir / "review.md"),
                "review_metadata": str(item_dir / "review_metadata.json"),
                "reference_video": str(reference_copy if copy_videos and reference_copy.exists() else reference_path),
                "target_video": str(target_copy if copy_videos and target_copy.exists() else target_path),
                "incomplete_review_bundle": bool(review_bundle_issues),
                "review_bundle_issues": review_bundle_issues,
                "audio_anchor_score": audio_anchor.get("audio_anchor_score"),
                "audio_matters_line": str(record.get("audio_matters_line", "")).strip(),
                "omni_visual_accept": quality.get("omni_visual_accept"),
                "visual_delta_strength": quality.get("visual_delta_strength"),
                "near_duplicate_risk": quality.get("near_duplicate_risk"),
            }
        )

    _write_jsonl(output_root / "review_items.jsonl", items)
    index_md = _manual_review_index_markdown(items=items, source_pairs_path=str(pairs_path), missing_videos=missing_videos)
    (output_root / "index.md").write_text(index_md, encoding="utf-8")
    return {
        "pair_count": len(pairs),
        "bundle_count": len(items),
        "incomplete_review_bundle_count": incomplete_review_bundle_count,
        "output_dir": str(output_root),
        "index_path": str(output_root / "index.md"),
        "items_path": str(output_root / "review_items.jsonl"),
        "missing_video_count": len(missing_videos),
        "missing_videos": missing_videos,
    }


def _diagnostic_bucket_key(record: dict[str, Any]) -> str:
    difference_type = str(record.get("difference", {}).get("type", "")).strip()
    reject_reason = str(record.get("judge", {}).get("reject_reason", "")).lower()
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    if difference_type == "visible_text" or _score_float(quality.get("visible_text_disabled")) >= 1.0:
        return "ocr"
    if (
        difference_type == "audio_event"
        or _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0
        or _score_float(quality.get("audio_event_too_similar")) >= 1.0
    ):
        return "audio_weak"
    if (
        _score_float(quality.get("too_similar_without_observable_delta")) >= 1.0
        or "caption_equivalent" in reject_reason
        or "near-duplicate" in reject_reason
        or "near duplicate" in reject_reason
    ):
        return "near_duplicate"
    if (
        _score_float(quality.get("too_broad_or_loose_pair")) >= 1.0
        or "broad scene" in reject_reason
        or "competing stronger difference" in reject_reason
        or "multiple broad changes" in reject_reason
    ):
        return "over_broad"
    return ""


def build_diagnostic_review_bundle(
    *,
    root: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    clip_annotations_path: str | Path | None = None,
    limit_per_bucket: int = 5,
    copy_videos: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records = list(_load_jsonl(Path(pairs_path)))
    selected: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    for record in records:
        if bool(record.get("accepted")):
            continue
        bucket = _diagnostic_bucket_key(record)
        if not bucket or bucket_counts[bucket] >= max(0, limit_per_bucket):
            continue
        bucket_counts[bucket] += 1
        selected.append(record)

    selected_path = output_root / "_diagnostic_selected_pairs.jsonl"
    _write_jsonl(selected_path, selected)
    bundle_summary = build_manual_review_bundle(
        root=root,
        pairs_path=selected_path,
        output_dir=output_root,
        clip_annotations_path=clip_annotations_path,
        copy_videos=copy_videos,
    )
    summary = {
        "pairs_path": str(pairs_path),
        "selected_count": len(selected),
        "bucket_counts": dict(bucket_counts),
        "output_dir": str(output_root),
        "bundle_summary": bundle_summary,
    }
    (output_root / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_single_source_review_bundle(
    *,
    root: str | Path,
    selected_source_path: str | Path,
    segments_manifest_path: str | Path,
    clip_annotations_path: str | Path,
    ranked_pairs_path: str | Path,
    accepted_pairs_path: str | Path,
    output_dir: str | Path,
    copy_videos: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    selected_path = Path(selected_source_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    segments = list(_load_jsonl(Path(segments_manifest_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    ranked_pairs = list(_load_jsonl(Path(ranked_pairs_path)))
    accepted_pairs = list(_load_jsonl(Path(accepted_pairs_path)))
    annotation_lookup = _annotation_lookup(root=root_path, annotations=annotations)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(selected_path, output_root / "selected_source_video.json")
    source_path = Path(str(selected.get("source_path", "")).strip())
    missing_videos: list[str] = []
    if copy_videos:
        if source_path.exists():
            shutil.copy2(source_path, output_root / "source_30s.mp4")
        else:
            missing_videos.append(str(source_path))

    segments_dir = output_root / "segments"
    if copy_videos:
        segments_dir.mkdir(parents=True, exist_ok=True)
    segment_items: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        clip_id = str(segment.get("clip_id", "")).strip()
        segment_path = _resolve_under_root(root_path, str(segment.get("output_path", "")).strip())
        annotation = annotation_lookup.get(clip_id) or {}
        copied_name = ""
        if copy_videos:
            copied_name = f"{index:03d}_{_safe_id(clip_id)}.mp4"
            if segment_path.exists():
                shutil.copy2(segment_path, segments_dir / copied_name)
            else:
                missing_videos.append(str(segment_path))
        segment_items.append(
            {
                "index": index,
                "clip_id": clip_id,
                "start_seconds": segment.get("start_seconds"),
                "end_seconds": segment.get("end_seconds"),
                "video": copied_name,
                "summary": str(annotation.get("summary", "")).strip(),
                "description": _review_annotation_description(annotation, fallback_caption=str(annotation.get("summary", ""))),
            }
        )

    ranked_copy = output_root / "ranked_single_source_pairs.jsonl"
    accepted_copy = output_root / "accepted_pairs.jsonl"
    _write_jsonl(ranked_copy, ranked_pairs)
    _write_jsonl(accepted_copy, accepted_pairs)
    (output_root / "segment_descriptions.md").write_text(
        _single_source_segment_descriptions_markdown(segment_items),
        encoding="utf-8",
    )
    (output_root / "all_pair_ranking.md").write_text(
        _single_source_pair_ranking_markdown(ranked_pairs),
        encoding="utf-8",
    )

    top_pair_bundle = build_manual_review_bundle(
        root=root_path,
        pairs_path=accepted_copy,
        output_dir=output_root / "top_pairs",
        clip_annotations_path=clip_annotations_path,
        copy_videos=copy_videos,
    )
    pair_review_summary = _build_single_source_pair_review_items(
        root=root_path,
        output_dir=output_root / "pair_review",
        ranked_pairs=ranked_pairs,
        annotation_lookup=annotation_lookup,
        copy_videos=copy_videos,
    )
    summary = {
        "selected_source_path": str(selected_path),
        "segments_manifest_path": str(segments_manifest_path),
        "clip_annotations_path": str(clip_annotations_path),
        "ranked_pairs_path": str(ranked_pairs_path),
        "accepted_pairs_path": str(accepted_pairs_path),
        "output_dir": str(output_root),
        "segment_count": len(segments),
        "ranked_pair_count": len(ranked_pairs),
        "accepted_pair_count": len(accepted_pairs),
        "top_pair_bundle": top_pair_bundle,
        "pair_review": pair_review_summary,
        "missing_video_count": len(missing_videos),
        "missing_videos": missing_videos,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "index.md").write_text(
        _single_source_review_index_markdown(summary=summary, selected=selected, segment_items=segment_items),
        encoding="utf-8",
    )
    return summary


def _single_source_segment_descriptions_markdown(segment_items: list[dict[str, Any]]) -> str:
    lines = ["# Segment Descriptions", ""]
    for item in segment_items:
        lines.extend(
            [
                f"## {item['index']:03d} `{item['clip_id']}`",
                "",
                f"- Time: `{item.get('start_seconds')}` -> `{item.get('end_seconds')}`",
                f"- Video: `{item.get('video', '')}`",
                "",
                f"```json\n{json.dumps(item.get('description', {}), ensure_ascii=False, indent=2)}\n```"
                if isinstance(item.get("description"), dict)
                else (str(item.get("description", "")).strip() or str(item.get("summary", "")).strip() or "No annotation."),
                "",
            ]
        )
    return "\n".join(lines)


def _single_source_pair_ranking_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Ranked Single Source Pairs",
        "",
        "| # | accepted | type | edit_text | reference | target | reject_reason |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, record in enumerate(records, start=1):
        difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
        judge = record.get("judge", {}) if isinstance(record.get("judge"), dict) else {}
        lines.append(
            "| "
            f"{index} | "
            f"{'yes' if bool(record.get('accepted')) else 'no'} | "
            f"{_markdown_table_cell(str(difference.get('type', '')))} | "
            f"{_markdown_table_cell(str(record.get('edit_text', '')))} | "
            f"{_markdown_table_cell(str(record.get('reference_clip_id', '')))} | "
            f"{_markdown_table_cell(str(record.get('target_clip_id', '')))} | "
            f"{_markdown_table_cell(str(judge.get('reject_reason', '')))} |"
        )
    if not records:
        lines.append("| 0 | no | none | none | none | none | none |")
    lines.append("")
    return "\n".join(lines)


def _build_single_source_pair_review_items(
    *,
    root: Path,
    output_dir: Path,
    ranked_pairs: list[dict[str, Any]],
    annotation_lookup: dict[str, dict[str, Any]],
    copy_videos: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_counts: Counter[str] = Counter()
    missing_videos: list[str] = []
    for index, record in enumerate(ranked_pairs, start=1):
        bucket = "accepted" if bool(record.get("accepted")) else "diagnostic"
        bucket_counts[bucket] += 1
        difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
        item_dir = output_dir / bucket / f"{index:03d}_{_safe_id(str(difference.get('type', 'pair')))}"
        item_dir.mkdir(parents=True, exist_ok=True)
        reference_video = str(record.get("reference_video", "")).strip()
        target_video = str(record.get("target_video", "")).strip()
        resolved_videos: dict[str, Path] = {}
        if copy_videos:
            for filename, raw_path in (("reference.mp4", reference_video), ("target.mp4", target_video)):
                resolved = _resolve_under_root(root, raw_path)
                if resolved.exists():
                    shutil.copy2(resolved, item_dir / filename)
                    resolved_videos[filename] = resolved
                else:
                    missing_videos.append(str(resolved))
        (item_dir / "edit_text.txt").write_text(str(record.get("edit_text", "")).strip() + "\n", encoding="utf-8")
        (item_dir / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        local_gate_report = record.get("local_gate_report") if isinstance(record.get("local_gate_report"), dict) else {}
        final_omni_verification = (
            record.get("final_omni_verification") if isinstance(record.get("final_omni_verification"), dict) else {}
        )
        (item_dir / "local_gate_report.json").write_text(
            json.dumps(local_gate_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "final_omni_verification.json").write_text(
            json.dumps(final_omni_verification, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (item_dir / "final_reason.md").write_text(
            _single_source_final_reason_markdown(record),
            encoding="utf-8",
        )
        reference_annotation = annotation_lookup.get(str(record.get("reference_clip_id", "")).strip(), {})
        target_annotation = annotation_lookup.get(str(record.get("target_clip_id", "")).strip(), {})
        contact_sheet_path = _write_single_source_pair_contact_sheet(
            output_path=item_dir / "contact_sheet.jpg",
            reference_video=resolved_videos.get("reference.mp4") or _resolve_under_root(root, reference_video),
            target_video=resolved_videos.get("target.mp4") or _resolve_under_root(root, target_video),
            label=str(record.get("proposal_id", "")),
        )
        (item_dir / "description.md").write_text(
            _single_source_pair_description_markdown(
                record=record,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                contact_sheet_path=contact_sheet_path.name if contact_sheet_path else "",
            ),
            encoding="utf-8",
        )
    summary = {
        "output_dir": str(output_dir),
        "pair_count": len(ranked_pairs),
        "bucket_counts": dict(bucket_counts),
        "missing_video_count": len(missing_videos),
        "missing_videos": missing_videos,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _single_source_final_reason_markdown(record: dict[str, Any]) -> str:
    local_gate_report = record.get("local_gate_report") if isinstance(record.get("local_gate_report"), dict) else {}
    final_omni = record.get("final_omni_verification") if isinstance(record.get("final_omni_verification"), dict) else {}
    return "\n".join(
        [
            "# Final Pair Decision",
            "",
            f"- accepted: `{bool(record.get('accepted'))}`",
            f"- final_accept_source: `{record.get('final_accept_source', '')}`",
            f"- final_omni_accept: `{bool(record.get('final_omni_accept'))}`",
            f"- final_omni_quality_score: `{final_omni.get('quality_score', '')}`",
            f"- final_omni_confidence: `{final_omni.get('confidence', '')}`",
            f"- main_reject_reason: {str(final_omni.get('main_reject_reason', '')).strip()}",
            "",
            "## Local Gate Report",
            "",
            f"```json\n{json.dumps(local_gate_report, ensure_ascii=False, indent=2)}\n```",
            "",
            "## Final Omni Verification",
            "",
            f"```json\n{json.dumps(final_omni, ensure_ascii=False, indent=2)}\n```",
            "",
        ]
    )


def _write_single_source_pair_contact_sheet(
    *,
    output_path: Path,
    reference_video: Path,
    target_video: Path,
    label: str,
) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    thumb_w, thumb_h = 170, 96
    sheet = Image.new("RGB", (thumb_w * 6 + 36, thumb_h + 44), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), label[:150], fill=(0, 0, 0))
    x = 4
    for side_label, video_path in (("REF", reference_video), ("TGT", target_video)):
        draw.text((x, 22), side_label, fill=(0, 0, 0))
        for time_seconds in (0.2, 2.5, 4.8):
            frame = _read_video_frame_image(video_path, time_seconds=time_seconds, size=(thumb_w, thumb_h))
            if frame is None:
                frame = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                placeholder = ImageDraw.Draw(frame)
                placeholder.text((8, 38), "frame unavailable", fill=(80, 80, 80))
            sheet.paste(frame, (x, 40))
            draw.text((x, thumb_h + 40), f"{time_seconds:.1f}s", fill=(0, 0, 0))
            x += thumb_w + 4
        x += 8
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return output_path


def _read_video_frame_image(video_path: Path, *, time_seconds: float, size: tuple[int, int]) -> Any:
    reader = None
    try:
        import imageio.v2 as imageio
        from PIL import Image

        reader = imageio.get_reader(str(video_path), "ffmpeg")
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 25.0)
        duration = float(metadata.get("duration") or 5.0)
        frame_index = max(0, int(min(time_seconds, max(0.0, duration - 0.05)) * fps))
        try:
            frame = reader.get_data(frame_index)
        except Exception:
            frame = reader.get_data(0)
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        return canvas
    except Exception:
        return None
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def _single_source_pair_description_markdown(
    *,
    record: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    contact_sheet_path: str = "",
) -> str:
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    judge = record.get("judge", {}) if isinstance(record.get("judge"), dict) else {}
    issues = _normalize_list(record.get("single_source_pair_acceptance_issues", []))
    lines = [
        f"# Single Source Pair `{record.get('proposal_id', '')}`",
        "",
        f"- accepted: `{bool(record.get('accepted'))}`",
        f"- final_omni_accept: `{bool(record.get('final_omni_accept'))}`",
        f"- edit_text: {str(record.get('edit_text', '')).strip()}",
        f"- recommended_edit_text: {str(record.get('recommended_edit_text', record.get('edit_text', ''))).strip()}",
        f"- difference: `{json.dumps(difference, ensure_ascii=False)}`",
        f"- confidence: `{record.get('confidence', '')}`",
        f"- final_omni_quality_score: `{(record.get('final_omni_verification') or {}).get('quality_score', '') if isinstance(record.get('final_omni_verification'), dict) else ''}`",
        f"- issue_tags: `{', '.join(issues) if issues else 'none'}`",
        f"- reject_reason: {str(judge.get('reject_reason', '')).strip()}",
        f"- delta_family: `{record.get('single_source_delta_family', '')}`",
        "",
        "## Contact Sheet",
        "",
        f"![reference/target frames]({contact_sheet_path})" if contact_sheet_path else "- Contact sheet unavailable.",
        "",
        "## Dominant Delta",
        "",
        f"```json\n{json.dumps(record.get('dominant_delta', record.get('dominant_delta_decision', {})), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Temporal Support",
        "",
        f"```json\n{json.dumps(record.get('delta_temporal_extent', {}), ensure_ascii=False, indent=2)}\n```",
        f"- is_segment_wide_delta: `{bool(record.get('is_segment_wide_delta'))}`",
        "",
        "## Subject Roles",
        "",
        f"```json\n{json.dumps(record.get('subject_roles', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Local Gate Report",
        "",
        f"```json\n{json.dumps(record.get('local_gate_report', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Final Omni Verification",
        "",
        f"```json\n{json.dumps(record.get('final_omni_verification', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Reference State",
        "",
        f"```json\n{json.dumps(record.get('reference_state', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Target State",
        "",
        f"```json\n{json.dumps(record.get('target_state', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Pair Video Evidence",
        "",
    ]
    evidence = record.get("pair_video_evidence", [])
    if isinstance(evidence, list) and evidence:
        lines.extend([f"- {item}" for item in evidence])
    else:
        lines.append("- No pair-level evidence recorded.")
    lines.extend(
        [
            "",
            "## Discarded Deltas",
            "",
        ]
    )
    discarded = record.get("discarded_deltas", [])
    if isinstance(discarded, list) and discarded:
        lines.extend([f"- {item}" for item in discarded])
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Reference Segment Description",
            "",
            f"```json\n{json.dumps(_review_annotation_description(reference_annotation), ensure_ascii=False, indent=2)}\n```",
            "",
            "## Target Segment Description",
            "",
            f"```json\n{json.dumps(_review_annotation_description(target_annotation), ensure_ascii=False, indent=2)}\n```",
            "",
            "## Review Focus",
            "",
            "- edit_text 是否描述了视频里最明显、可验证的主差异？",
            "- 如果有产品、画中画、手持物或特写，是否优先于衣服/头发措辞差异？",
            "- 是否只有一个主变化，而不是多个无关变化拼在一起？",
            "",
        ]
    )
    return "\n".join(lines)


def _single_source_review_index_markdown(
    *,
    summary: dict[str, Any],
    selected: dict[str, Any],
    segment_items: list[dict[str, Any]],
) -> str:
    lines = [
        "# Single Source Omni Pair Review",
        "",
        f"- Source clip id: `{selected.get('source_clip_id', '')}`",
        f"- Source path: `{selected.get('source_path', '')}`",
        f"- Segments: `{summary.get('segment_count', 0)}`",
        f"- Ranked pairs: `{summary.get('ranked_pair_count', 0)}`",
        f"- Accepted pairs: `{summary.get('accepted_pair_count', 0)}`",
        f"- Missing videos: `{summary.get('missing_video_count', 0)}`",
        "",
        "## Files",
        "",
        "- `source_30s.mp4`",
        "- `segments/`",
        "- `segment_descriptions.md`",
        "- `all_pair_ranking.md`",
        "- `top_pairs/`",
        "",
        "## Segment Order",
    ]
    for item in segment_items:
        lines.append(
            f"- `{item['clip_id']}` `{item.get('start_seconds')}` -> `{item.get('end_seconds')}`: "
            f"{str(item.get('summary', '')).strip()}"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def validate_known_pairs(
    *,
    root: str | Path,
    known_pairs_path: str | Path,
    clip_annotations_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_path: str | Path | None = None,
    accepted_output_path: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    max_accepted_pairs: int = 10,
) -> dict[str, Any]:
    layout = ensure_layout(root)
    known_pairs = list(_load_jsonl(Path(known_pairs_path)))
    annotations = list(_load_jsonl(Path(clip_annotations_path)))
    if not known_pairs:
        raise ValueError("known pairs file is empty")
    if not annotations:
        raise ValueError("clip annotations are empty")

    output = Path(output_path) if output_path else layout["pairs"] / DEFAULT_SYNTHETIC_JUDGED_PAIRS_NAME
    accepted_output = Path(accepted_output_path) if accepted_output_path else layout["pairs"] / DEFAULT_SYNTHETIC_ACCEPTED_PAIRS_NAME
    existing_records = {} if overwrite else _load_records_by_key(output, "proposal_id")
    raw_index = _load_raw_asset_index(Path(raw_index_path) if raw_index_path else layout["metadata"] / DEFAULT_RAW_INDEX_NAME)
    annotation_lookup = _annotation_lookup(root=layout["root"], annotations=annotations)
    client = OpenAIComposedDataClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    output_records: list[dict[str, Any]] = []
    proposed_count = 0
    reused_count = 0
    fallback_count = 0
    rejected_count = 0
    accepted_total_count = 0
    seen_proposal_ids: set[str] = set()

    for line_number, pair in enumerate(known_pairs, start=1):
        reference_annotation = _annotation_for_known_pair(
            root=layout["root"],
            lookup=annotation_lookup,
            pair=pair,
            clip_id_field="reference_clip_id",
            video_field="reference_video",
            line_number=line_number,
        )
        target_annotation = _annotation_for_known_pair(
            root=layout["root"],
            lookup=annotation_lookup,
            pair=pair,
            clip_id_field="target_clip_id",
            video_field="target_video",
            line_number=line_number,
        )

        reference_video = _known_pair_video_path(layout["root"], pair, reference_annotation, "reference_video")
        target_video = _known_pair_video_path(layout["root"], pair, target_annotation, "target_video")
        proposal_id = str(pair.get("proposal_id", "")).strip() or _build_proposal_id(reference_video, target_video)
        if proposal_id in seen_proposal_ids:
            continue
        seen_proposal_ids.add(proposal_id)

        if proposal_id in existing_records:
            record = existing_records[proposal_id]
            reused_count += 1
        else:
            raw_judge_output: dict[str, Any] = {}
            raw_verification_output: dict[str, Any] = {}
            model_fields = _known_pair_model_fields(
                pair=pair,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            model_fields = _repair_pair_model_fields(
                model_fields=model_fields,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            source = _build_source_metadata(
                root=layout["root"],
                target_annotation=target_annotation,
                raw_index=raw_index,
            )
            source["source_type"] = str(pair.get("source_type", "synthetic_edit")).strip() or "synthetic_edit"
            source_context = _known_pair_source_context(pair)
            hard_negative_annotations = _known_pair_hard_negative_annotations(
                root=layout["root"],
                lookup=annotation_lookup,
                annotations=annotations,
                pair=pair,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=model_fields["difference"],
            )
            hard_negative_paths = _known_pair_hard_negative_paths(
                root=layout["root"],
                pair=pair,
                hard_negative_annotations=hard_negative_annotations,
            )
            base_quality = _known_pair_base_quality(
                root=layout["root"],
                pair=pair,
                annotations=annotations,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=model_fields["difference"],
                source_context=source_context,
            )
            proposal_quality = _quality_for_model_fields(
                base_quality=base_quality,
                model_fields=model_fields,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
            )
            edit_text_quality = _edit_text_quality_payload(
                edit_text=model_fields["edit_text"],
                difference=model_fields["difference"],
                modalities=model_fields["modalities"],
                reference_caption=model_fields["reference_caption"],
                target_caption=model_fields["target_caption"],
            )
            observable_difference = _observable_difference_gate(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                difference=model_fields["difference"],
                visual_near_duplicate_score=proposal_quality.get("visual_near_duplicate_score"),
            )
            _apply_structured_gate_quality(
                proposal_quality,
                edit_text_quality=edit_text_quality,
                observable_difference=observable_difference,
            )
            proposal_difference_evidence = _difference_evidence_from_annotations(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                primary_difference=model_fields["difference"],
            )
            proposal_view = {
                "proposal_id": proposal_id,
                "edit_text": model_fields["edit_text"],
                "modalities": list(model_fields["modalities"]),
                "reference_caption": model_fields["reference_caption"],
                "target_caption": model_fields["target_caption"],
                "difference": model_fields["difference"],
                "quality": dict(proposal_quality),
                "source_context": dict(source_context),
                "generation": dict(pair.get("generation", {})),
                "difference_evidence": dict(proposal_difference_evidence),
                "edit_text_quality": dict(edit_text_quality),
                "observable_difference": dict(observable_difference),
                "acceptance_thresholds": {
                    "same_context_score": MIN_ACCEPT_SAME_CONTEXT_SCORE,
                    "edit_match_score": MIN_ACCEPT_EDIT_MATCH_SCORE,
                    "target_uniqueness_score": MIN_ACCEPT_TARGET_UNIQUENESS_SCORE,
                    "difference_strength_score": MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE,
                    "max_visual_near_duplicate_score_for_visual_edits": MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE,
                    "edit_text_quality_score": MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE,
                },
            }
            try:
                judge, raw_judge_output = client.judge_pair(
                    proposal=proposal_view,
                    reference_annotation=_annotation_prompt_view(reference_annotation),
                    target_annotation=_annotation_prompt_view(target_annotation),
                    hard_negative_candidates=[
                        _annotation_prompt_view(annotation) for annotation in hard_negative_annotations
                    ],
                )
                judge_fallback_used = False
            except Exception as exc:
                judge = _fallback_pair_judge(proposal_quality, reason=f"{type(exc).__name__}: {exc}")
                raw_judge_output = {"error": f"{type(exc).__name__}: {exc}"}
                judge_fallback_used = True

            try:
                (
                    verification,
                    raw_verification_output,
                    verification_context_retry_used,
                ) = _verify_pair_difference_with_context_retry(
                    client,
                    proposal=proposal_view,
                    reference_annotation=_annotation_prompt_view(reference_annotation),
                    target_annotation=_annotation_prompt_view(target_annotation),
                    reference_clip_path=str(_resolve_under_root(layout["root"], reference_video)),
                    target_clip_path=str(_resolve_under_root(layout["root"], target_video)),
                )
                verification_fallback_used = False
            except Exception as exc:
                verification = _fallback_pair_verification(reason=f"{type(exc).__name__}: {exc}")
                raw_verification_output = {"error": f"{type(exc).__name__}: {exc}"}
                verification_context_retry_used = False
                verification_fallback_used = True

            judge = _finalize_pair_judge(judge)
            verification = _finalize_pair_verification(verification)
            fallback_used = judge_fallback_used or verification_fallback_used
            effective_quality = _effective_pair_quality(judge, verification, proposal_quality)
            accepted = _judge_accepts(judge, verification, effective_quality)
            if not accepted:
                judge["reject_reason"] = _compose_reject_reason(judge, verification, effective_quality)
            speech_quality = _speech_quality_payload(effective_quality)
            audio_event_quality = _audio_event_quality_payload(effective_quality)
            record = {
                "proposal_id": proposal_id,
                "source_type": str(pair.get("source_type", "synthetic_edit")).strip() or "synthetic_edit",
                "generation": dict(pair.get("generation", {})),
                "group_id": str(pair.get("group_id", "synthetic_edit")).strip() or "synthetic_edit",
                "group_reason": str(pair.get("group_reason", "known_synthetic_pair")).strip() or "known_synthetic_pair",
                "reference_clip_id": reference_annotation.get("clip_id", ""),
                "target_clip_id": target_annotation.get("clip_id", ""),
                "reference_video": reference_video,
                "target_video": target_video,
                "edit_text": model_fields["edit_text"],
                "modalities": list(model_fields["modalities"]),
                "reference_caption": model_fields["reference_caption"],
                "target_caption": model_fields["target_caption"],
                "difference": model_fields["difference"],
                "hard_negatives": hard_negative_paths,
                "judge_quality": {
                    "same_context_score": judge["same_context_score"],
                    "edit_match_score": judge["edit_match_score"],
                    "target_uniqueness_score": judge["target_uniqueness_score"],
                },
                "quality": effective_quality,
                "heuristic_quality": dict(proposal_quality),
                "source_context": dict(source_context),
                "source": source,
                "proposal_reason": str(pair.get("proposal_reason", "known pair validation")).strip(),
                "evidence": _evidence_from_annotations(
                    reference_annotation,
                    target_annotation,
                    difference_evidence=proposal_difference_evidence,
                ),
                "judge": judge,
                "verification": verification,
                "speech_quality": speech_quality,
                "audio_event_quality": audio_event_quality,
                "edit_text_quality": edit_text_quality,
                "observable_difference": observable_difference,
                "transcript_backed": speech_quality.get("transcript_backed"),
                "accepted": accepted,
                "fallback_used": fallback_used,
                "raw_model_output": {"known_pair": True},
                "raw_judge_output": raw_judge_output,
                "raw_verification_output": raw_verification_output,
                "verification_annotation_only_retry_used": verification_context_retry_used,
            }
            proposed_count += 1

        record = _apply_post_vace_semantic_verdict(
            record,
            target_annotation=target_annotation,
        )
        record = _prepare_record_for_acceptance(
            record,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
        judge = dict(record.get("judge", {}))
        verification = record.get("verification", {})
        quality = record.get("quality", {})
        record["accepted"] = _judge_accepts(judge, verification, quality)
        if not bool(record.get("accepted")):
            judge["accept"] = False
            judge["reject_reason"] = _compose_reject_reason(judge, verification, quality)
            record["judge"] = judge
        acceptance_issues = _pair_record_acceptance_issues(
            root=layout["root"],
            record=record,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
        acceptance_issues.extend(_known_pair_generation_issues(record))
        if acceptance_issues:
            record = _reject_record_with_acceptance_issues(record, acceptance_issues)
        if bool(record.get("fallback_used")):
            fallback_count += 1
        if bool(record.get("accepted")):
            accepted_total_count += 1
        else:
            rejected_count += 1
        output_records.append(record)

    accepted_records = _select_final_accepted_records(output_records, max_accepted_pairs=max_accepted_pairs)
    _write_jsonl(output, output_records)
    _write_jsonl(accepted_output, accepted_records)
    verification_counts = _pair_verification_counts(output_records)
    return {
        "known_pairs_path": str(known_pairs_path),
        "clip_annotations_path": str(clip_annotations_path),
        "output_path": str(output),
        "accepted_output_path": str(accepted_output),
        "pair_count": len(known_pairs),
        "proposal_count": len(output_records),
        "accepted_count": len(accepted_records),
        "accepted_total_count": accepted_total_count,
        "rejected_count": rejected_count,
        "proposed_count": proposed_count,
        "reused_count": reused_count,
        "fallback_count": fallback_count,
        "verification_counts": verification_counts,
    }


def validate_pilot_dataset(
    *,
    root: str | Path,
    pilot_jsonl_path: str | Path,
    gallery_output_path: str | Path,
    report_output_path: str | Path,
) -> dict[str, Any]:
    root_path = Path(root)
    pilot_records = list(_load_jsonl(Path(pilot_jsonl_path)))
    if not pilot_records:
        raise ValueError("pilot dataset is empty")

    errors: list[str] = []
    seen_sample_ids: set[str] = set()
    seen_proposal_ids: set[str] = set()
    seen_pair_keys: set[tuple[str, str]] = set()
    difference_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()
    source_type_difference_counter: Counter[str] = Counter()
    speech_count = 0
    high_quality_speech_count = 0
    transcript_backed_speech_count = 0
    non_speech_audio_event_count = 0
    same_context_scores: list[float] = []
    difference_strength_scores: list[float] = []
    source_context_counter: Counter[str] = Counter()
    gallery_accumulator: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(pilot_records, start=1):
        errors.extend(_validate_pilot_record(root_path, record, index))

        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            if sample_id in seen_sample_ids:
                errors.append(f"pilot line {index}: duplicate sample_id={sample_id}")
            seen_sample_ids.add(sample_id)

        proposal_id = str(record.get("proposal_id", "")).strip()
        if not proposal_id:
            errors.append(f"pilot line {index}: proposal_id is required")
        elif proposal_id in seen_proposal_ids:
            errors.append(f"pilot line {index}: duplicate proposal_id={proposal_id}")
        else:
            seen_proposal_ids.add(proposal_id)

        reference_video = str(record.get("reference_video", "")).strip()
        target_video = str(record.get("target_video", "")).strip()
        if reference_video and target_video:
            record_source_type = str(record.get("source_type", "natural")).strip() or "natural"
            expected_proposal_id = _build_proposal_id(reference_video, target_video)
            if record_source_type != "synthetic_edit" and proposal_id and proposal_id != expected_proposal_id:
                errors.append(
                    f"pilot line {index}: proposal_id={proposal_id} does not match expected {expected_proposal_id}"
                )
            pair_key = (reference_video, target_video)
            if pair_key in seen_pair_keys:
                errors.append(f"pilot line {index}: duplicate reference-target pair={pair_key}")
            seen_pair_keys.add(pair_key)

        modalities = [str(item).strip() for item in record.get("modalities", []) if str(item).strip()]
        modality_counter.update(modalities)

        source_type = str(record.get("source_type", "natural")).strip() or "natural"
        source_type_counter[source_type] += 1

        difference = record.get("difference", {})
        difference_type = str(difference.get("type", "")).strip()
        if difference_type:
            difference_counter[difference_type] += 1
            source_type_difference_counter[f"{source_type}:{difference_type}"] += 1

        quality = record.get("quality", {})
        if isinstance(quality, dict):
            if difference_type == "speech":
                speech_count += 1
                if _score_float(quality.get("speech_transcript_backed")) >= 1.0:
                    transcript_backed_speech_count += 1
                if (
                    _score_float(quality.get("speech_evidence_score")) >= MIN_ACCEPT_SPEECH_EVIDENCE_SCORE
                    and _score_float(quality.get("speech_specificity_score")) >= MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE
                ):
                    high_quality_speech_count += 1
            if (
                difference_type == "audio_event"
                and _score_float(quality.get("non_speech_audio_event_score")) >= MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE
            ):
                non_speech_audio_event_count += 1
            try:
                same_context_scores.append(float(quality.get("same_context_score", 0.0)))
            except (TypeError, ValueError):
                pass
            if "difference_strength_score" in quality:
                try:
                    difference_strength_scores.append(float(quality.get("difference_strength_score", 0.0)))
                except (TypeError, ValueError):
                    pass

        source_context = record.get("source_context", {})
        if isinstance(source_context, dict):
            source_context_counter[str(source_context.get("relation", "unknown"))] += 1

        target_video = str(record.get("target_video", "")).strip()
        if target_video:
            _merge_gallery_entry(
                accumulator=gallery_accumulator,
                video_path=target_video,
                sample_id=sample_id,
                role="target",
            )
        for negative in record.get("hard_negatives", []):
            negative_path = str(negative).strip()
            if negative_path:
                _merge_gallery_entry(
                    accumulator=gallery_accumulator,
                    video_path=negative_path,
                    sample_id=sample_id,
                    role="hard_negative",
                )

    if errors:
        raise ValueError("\n".join(errors[:20]))

    gallery_records = [
        {
            "gallery_id": _build_gallery_id(video_path),
            "video_path": video_path,
            "sample_ids": sorted(entry["sample_ids"]),
            "roles": sorted(entry["roles"]),
        }
        for video_path, entry in sorted(gallery_accumulator.items())
    ]
    _write_jsonl(Path(gallery_output_path), gallery_records)

    verification_counts = _load_pair_verification_counts(Path(pilot_jsonl_path))
    summary = {
        "sample_count": len(pilot_records),
        "gallery_count": len(gallery_records),
        "modality_counts": dict(sorted(modality_counter.items())),
        "difference_type_counts": dict(sorted(difference_counter.items())),
        "source_type_counts": dict(sorted(source_type_counter.items())),
        "source_type_difference_counts": dict(sorted(source_type_difference_counter.items())),
        "source_context_counts": dict(sorted(source_context_counter.items())),
        "quality_summary": _quality_summary(same_context_scores),
        "difference_strength_summary": _score_summary(difference_strength_scores, "difference_strength"),
        "verification_counts": verification_counts,
        "speech_audio_quality_counts": {
            "speech_count": speech_count,
            "high_quality_speech_count": high_quality_speech_count,
            "transcript_backed_speech_count": transcript_backed_speech_count,
            "non_speech_audio_event_count": non_speech_audio_event_count,
            "speech_rejected_as_too_generic_count": verification_counts.get("speech_rejected_as_too_generic_count", 0),
            "audio_event_rejected_as_speech_only_count": verification_counts.get(
                "audio_event_rejected_as_speech_only_count",
                0,
            ),
        },
        "automated_acceptance": {
            "sample_count_between_5_and_10": 5 <= len(pilot_records) <= 10,
            "audio_samples_at_least_2": modality_counter.get("audio", 0) >= 2,
            "non_speech_audio_samples_at_least_1": non_speech_audio_event_count >= 1,
            "speech_samples_all_have_evidence": speech_count == high_quality_speech_count,
            "speech_samples_all_transcript_backed": speech_count == transcript_backed_speech_count,
            "object_change_samples_at_least_2": difference_counter.get("object_count", 0)
            + difference_counter.get("object_presence", 0)
            >= 2,
            "action_samples_at_least_1": difference_counter.get("action", 0) >= 1,
        },
    }
    Path(report_output_path).write_text(_build_pilot_report(summary), encoding="utf-8")
    summary["gallery_output_path"] = str(gallery_output_path)
    summary["report_output_path"] = str(report_output_path)
    return summary


def _quality_summary(same_context_scores: list[float]) -> dict[str, float]:
    if not same_context_scores:
        return {"same_context_min": 0.0, "same_context_avg": 0.0, "same_context_max": 0.0}
    return {
        "same_context_min": round(min(same_context_scores), 3),
        "same_context_avg": round(sum(same_context_scores) / len(same_context_scores), 3),
        "same_context_max": round(max(same_context_scores), 3),
    }


def _score_summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_min": 0.0, f"{prefix}_avg": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_min": round(min(values), 3),
        f"{prefix}_avg": round(sum(values) / len(values), 3),
        f"{prefix}_max": round(max(values), 3),
    }


def _load_pair_verification_counts(pilot_jsonl_path: Path) -> dict[str, int]:
    candidate_names = ["judged_pair_proposals.jsonl"]
    if "synthetic" in pilot_jsonl_path.name:
        candidate_names.insert(0, DEFAULT_SYNTHETIC_JUDGED_PAIRS_NAME)
    for candidate_name in candidate_names:
        judged_path = pilot_jsonl_path.with_name(candidate_name)
        if judged_path.exists():
            return _pair_verification_counts(list(_load_jsonl(judged_path)))
    return _empty_pair_verification_counts()


def _empty_pair_verification_counts() -> dict[str, int]:
    return {
        "verification_passed_count": 0,
        "verification_passed_rejected_count": 0,
        "verification_override_accept_count": 0,
        "caption_equivalent_reject_count": 0,
        "missing_delta_reject_count": 0,
        "difference_mismatch_reject_count": 0,
        "edit_projection_reject_count": 0,
        "edit_not_needed_reject_count": 0,
        "speech_rejected_as_too_generic_count": 0,
        "speech_rejected_for_missing_transcript_count": 0,
        "audio_event_rejected_as_speech_only_count": 0,
        "good_edit_text_count": 0,
        "bad_edit_text_rejected_count": 0,
        "caption_like_edit_rejected_count": 0,
        "modality_leakage_rejected_count": 0,
        "near_duplicate_without_delta_rejected_count": 0,
        "visual_presence_contradiction_reject_count": 0,
        "visible_text_without_ocr_reject_count": 0,
        "audio_event_without_independent_audio_evidence_reject_count": 0,
        "competing_difference_reject_count": 0,
        "duplicate_target_reject_count": 0,
        "synthetic_context_override_count": 0,
        "synthetic_visual_count": 0,
        "synthetic_audio_count": 0,
        "deterministic_audio_count": 0,
        "foleycrafter_audio_count": 0,
        "frieren_audio_count": 0,
        "speech_content_reject_count": 0,
        "audio_stream_missing_reject_count": 0,
        "visual_changed_in_audio_sample_reject_count": 0,
        "audio_event_not_detected_reject_count": 0,
        "audio_remux_count": 0,
        "missing_target_audio_reject_count": 0,
        "accepted_after_verification_count": 0,
    }


def _pair_verification_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = _empty_pair_verification_counts()
    accepted_target_counts = Counter(
        str(record.get("target_video", "")).strip()
        for record in records
        if bool(record.get("accepted")) and str(record.get("target_video", "")).strip()
    )
    counts["duplicate_target_reject_count"] = sum(max(0, count - 1) for count in accepted_target_counts.values())
    for record in records:
        verification = record.get("verification")
        if not isinstance(verification, dict):
            continue
        quality = record.get("quality", {})
        if not isinstance(quality, dict):
            quality = {}
        if bool(record.get("accepted")) and not _structured_edit_text_failures(quality):
            counts["good_edit_text_count"] += 1
        if _score_float(quality.get("synthetic_context_override")) >= 1.0:
            counts["synthetic_context_override_count"] += 1
        generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
        route = _synthetic_generation_route(generation)
        if str(record.get("source_type", "")).strip() == "synthetic_edit" and bool(record.get("accepted")):
            if route in SYNTHETIC_AUDIO_ROUTES:
                counts["synthetic_audio_count"] += 1
                if route in {"deterministic_overlay", "audio_deterministic"}:
                    counts["deterministic_audio_count"] += 1
                elif route == "foleycrafter_temporal":
                    counts["foleycrafter_audio_count"] += 1
                elif route == "frieren_benchmark":
                    counts["frieren_audio_count"] += 1
            else:
                counts["synthetic_visual_count"] += 1
        postprocess = generation.get("postprocess", {}) if isinstance(generation.get("postprocess"), dict) else {}
        if postprocess.get("audio_copied_from_reference"):
            counts["audio_remux_count"] += 1
        reject_reason_text = str(record.get("judge", {}).get("reject_reason", "")).lower() if isinstance(record.get("judge"), dict) else ""
        if "missing audio copied from the reference" in reject_reason_text:
            counts["missing_target_audio_reject_count"] += 1
        if "missing audio" in reject_reason_text:
            counts["audio_stream_missing_reject_count"] += 1
        if "speech content edits are disabled" in reject_reason_text or "speech difference type is disabled" in reject_reason_text:
            counts["speech_content_reject_count"] += 1
        if "audio synthetic target changed visual stream" in reject_reason_text:
            counts["visual_changed_in_audio_sample_reject_count"] += 1
        if "audio_event target sound was not detected" in reject_reason_text:
            counts["audio_event_not_detected_reject_count"] += 1
        verification_passed = _verification_accepts(verification)
        if verification_passed:
            counts["verification_passed_count"] += 1
        if bool(record.get("accepted")) and verification_passed:
            counts["accepted_after_verification_count"] += 1
            judge = record.get("judge", {})
            if isinstance(judge, dict) and not _boolish(judge.get("accept")):
                counts["verification_override_accept_count"] += 1
            continue
        if verification_passed and not bool(record.get("accepted")):
            counts["verification_passed_rejected_count"] += 1
        if not bool(record.get("accepted")):
            difference_type = str(record.get("difference", {}).get("type", "")).strip()
            if difference_type == "speech" and (
                _score_float(quality.get("speech_evidence_score")) < MIN_ACCEPT_SPEECH_EVIDENCE_SCORE
                or _score_float(quality.get("speech_specificity_score")) < MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE
            ):
                counts["speech_rejected_as_too_generic_count"] += 1
            if difference_type == "speech" and _score_float(quality.get("speech_transcript_backed")) < 1.0:
                counts["speech_rejected_for_missing_transcript_count"] += 1
            if (
                difference_type == "audio_event"
                and _score_float(quality.get("non_speech_audio_event_score")) < MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE
            ):
                counts["audio_event_rejected_as_speech_only_count"] += 1
            edit_text_failures = _structured_edit_text_failures(quality)
            if edit_text_failures:
                counts["bad_edit_text_rejected_count"] += 1
            if any("caption-like" in failure for failure in edit_text_failures):
                counts["caption_like_edit_rejected_count"] += 1
            if any("leaks another modality" in failure for failure in edit_text_failures):
                counts["modality_leakage_rejected_count"] += 1
            if _observable_difference_rejects(quality):
                counts["near_duplicate_without_delta_rejected_count"] += 1
            observable = record.get("observable_difference", {})
            if isinstance(observable, dict):
                observable_reason = str(observable.get("failure_reason", "")).strip().lower()
                if "already appears to contain equivalent object" in observable_reason:
                    counts["visual_presence_contradiction_reject_count"] += 1
                if "visible_text lacks" in observable_reason:
                    counts["visible_text_without_ocr_reject_count"] += 1
            if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
                counts["audio_event_without_independent_audio_evidence_reject_count"] += 1
            if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
                counts["competing_difference_reject_count"] += 1
        caption_delta = verification.get("caption_delta", {})
        edit_projection = verification.get("edit_projection", {})
        edit_necessity = verification.get("edit_necessity", {})
        if _boolish(caption_delta.get("caption_equivalent")):
            counts["caption_equivalent_reject_count"] += 1
        if not _boolish(caption_delta.get("has_concrete_difference")):
            counts["missing_delta_reject_count"] += 1
        if not _boolish(caption_delta.get("difference_matches_edit")):
            counts["difference_mismatch_reject_count"] += 1
        if (
            not _boolish(edit_projection.get("target_matches_projection"))
            or _score_float(edit_projection.get("score")) < MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE
        ):
            counts["edit_projection_reject_count"] += 1
        if (
            not _boolish(edit_necessity.get("edit_needed"))
            or _boolish(edit_necessity.get("reference_satisfies_edit"))
            or not _boolish(edit_necessity.get("target_satisfies_edit"))
            or _score_float(edit_necessity.get("score")) < MIN_ACCEPT_EDIT_NECESSITY_SCORE
        ):
            counts["edit_not_needed_reject_count"] += 1
    return counts


def probe_media(source_path: str | Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout or "{}")
    except Exception as exc:
        return {
            "duration_seconds": 0.0,
            "has_audio": False,
            "has_video": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    duration = _media_duration(payload, video_stream)
    return {
        "duration_seconds": round(duration, 3),
        "has_audio": bool(audio_stream),
        "has_video": bool(video_stream),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": round(_parse_fraction(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "")), 3),
    }


def _build_toolbox_observations(clip_path: Path) -> list[dict[str, Any]]:
    media = probe_media(clip_path)
    duration = float(media.get("duration_seconds") or 0.0)
    frame_times = _sample_frame_times(duration)
    audio_note = (
        "audio track present; inspect speech, music, acoustic events, and audio-visual synchronization"
        if media.get("has_audio")
        else "no audio stream detected by ffprobe"
    )
    return [
        {
            "tool": "media_probe",
            "observation": media,
        },
        {
            "tool": "frame_sampler",
            "observation": {
                "sample_times": frame_times,
                "instruction": "use these timestamps as key visual moments for subjects, actions, scene, and visible text",
            },
        },
        {
            "tool": "audio_observer",
            "observation": {
                "note": audio_note,
                "max_audio_window_seconds": 30.0,
            },
        },
        {
            "tool": "ocr_asr_observer",
            "observation": {
                "instruction": "extract visible text and spoken content when present; leave uncertainty if unreadable or inaudible",
            },
        },
    ]


def _sample_frame_times(duration_seconds: float) -> list[float]:
    if duration_seconds <= 0:
        return []
    count = 3 if duration_seconds <= 6 else 6
    if count == 1:
        return [round(duration_seconds / 2, 3)]
    step = duration_seconds / (count + 1)
    return [round(step * index, 3) for index in range(1, count + 1)]


def _media_duration(payload: dict[str, Any], video_stream: dict[str, Any]) -> float:
    for raw_value in (
        payload.get("format", {}).get("duration") if isinstance(payload.get("format"), dict) else None,
        video_stream.get("duration"),
    ):
        try:
            duration = float(raw_value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    return 0.0


def _parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_clip_video_path(root: Path, item: dict[str, Any]) -> Path:
    source_path = str(item.get("source_path", "")).strip()
    if source_path:
        path = Path(source_path)
        return path if path.is_absolute() else root / path
    output_path = str(item.get("output_path", "")).strip()
    if output_path:
        return _resolve_under_root(root, output_path)
    return root / "__missing_source_clip__"


def _source_clip_duration_seconds(item: dict[str, Any], media: dict[str, Any]) -> float:
    for value in (
        media.get("duration_seconds"),
        item.get("duration_seconds"),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    start = _optional_float(item.get("start_seconds"))
    end = _optional_float(item.get("end_seconds"))
    if start is not None and end is not None and end > start:
        return end - start
    return 0.0


def _is_single_source_raw_video_candidate(
    *,
    root: Path,
    item: dict[str, Any],
    source_path: Path,
    dataset: str,
) -> bool:
    if str(item.get("dataset", "")).strip() != dataset:
        return False
    path_text = _display_source_path(root, str(source_path)).replace("\\", "/").lower()
    if "/clips/" in f"/{path_text}":
        return False
    return f"raw/{dataset}/video/" in path_text or f"raw_datasets/{dataset}/" in path_text


def _single_source_local_selection_score(*, item: dict[str, Any], media: dict[str, Any]) -> float:
    duration = float(media.get("duration_seconds") or item.get("duration_seconds") or 0.0)
    score = 1.0 - min(1.0, abs(duration - 30.0) / 4.0) * 0.35
    if media.get("has_audio"):
        score += 0.20
    if media.get("has_video"):
        score += 0.20
    width = int(media.get("width") or 0)
    height = int(media.get("height") or 0)
    if width >= 480 and height >= 270:
        score += 0.10
    text_blob = _normalized_phrase(json.dumps(item.get("text_fields", {}), ensure_ascii=False))
    if any(token in text_blob for token in ("subtitle", "caption", "title card", "text only")):
        score -= 0.15
    if len(_tokenize_text(text_blob)) >= 6:
        score += 0.05
    return round(max(0.0, min(1.0, score)), 3)


def _single_source_selection_annotation(
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    raw_model_output: dict[str, Any],
) -> dict[str, Any]:
    return {
        "clip_id": f"{_safe_id(str(candidate.get('source_clip_id', 'source')))}__selection",
        "output_path": str(candidate.get("source_path", "")),
        "source_path": str(candidate.get("source_path", "")),
        "dataset": str(candidate.get("dataset", "")),
        "summary": str(normalized.get("summary", "")).strip(),
        "subjects": list(normalized.get("subjects", [])),
        "object_counts": dict(normalized.get("object_counts", {})),
        "actions": list(normalized.get("actions", [])),
        "scene": str(normalized.get("scene", "")).strip(),
        "attributes": list(normalized.get("attributes", [])),
        "on_screen_text": list(normalized.get("on_screen_text", [])),
        "visible_text": list(normalized.get("visible_text", [])),
        "speech": list(normalized.get("speech", [])),
        "audio_events": list(normalized.get("audio_events", [])),
        "modalities": list(normalized.get("modalities", [])),
        "storyline": list(normalized.get("storyline", [])),
        "speakers_and_transcript": list(normalized.get("speakers_and_transcript", [])),
        "detective_notes": list(normalized.get("detective_notes", [])),
        "uncertainties": list(normalized.get("uncertainties", [])),
        "fallback_used": False,
        "raw_model_output": raw_model_output,
    }


def _single_source_omni_selection_score(annotation: dict[str, Any]) -> float:
    score = _clean_stability_score(annotation) * 0.45
    score += min(0.20, len(_normalize_list(annotation.get("storyline", []))) * 0.05)
    score += min(0.12, len(_action_terms_from_annotation(annotation)) * 0.03)
    score += min(0.12, len(_normalize_object_counts(annotation.get("object_counts", {}))) * 0.04)
    score += min(0.08, len(_annotation_subject_signature_bundle(annotation)) * 0.03)
    if _title_card_or_boundary_text(annotation):
        score -= 0.25
    if len(_normalize_list(annotation.get("visible_text", []))) >= 3:
        score -= 0.10
    if len(_speech_texts_from_annotation(annotation)) >= 3:
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 3)


def _single_source_selection_reasons(annotation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _clean_stability_score(annotation) >= 0.75:
        reasons.append("clean/stable according to Omni annotation")
    if len(_normalize_list(annotation.get("storyline", []))) >= 2:
        reasons.append("multiple timeline moments available inside 30s")
    if _action_terms_from_annotation(annotation):
        reasons.append("action evidence available")
    if _annotation_object_signature_bundle(annotation):
        reasons.append("object evidence available")
    if not _title_card_or_boundary_text(annotation):
        reasons.append("not title-card/text-only dominant")
    return reasons[:5]


def _fixed_single_source_segments(
    *,
    duration_seconds: float,
    segment_seconds: float,
    min_clip_seconds: float,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + segment_seconds, duration_seconds)
        if end - start >= min_clip_seconds:
            segments.append((round(start, 3), round(end, 3)))
        start += segment_seconds
    return segments


def _clip_start_seconds(annotation: dict[str, Any]) -> float:
    source_clip = annotation.get("source_clip", {}) if isinstance(annotation.get("source_clip"), dict) else {}
    return _optional_float(source_clip.get("start_seconds")) or _optional_float(annotation.get("start_seconds")) or 0.0


def _single_source_pair_candidate(
    *,
    root: Path,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    group_metadata: dict[str, str],
    acceptance_profile: str,
) -> tuple[dict[str, Any], bool]:
    scored = _score_ordered_pair(
        root=root,
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        compute_visual_near_duplicate=False,
    )
    if scored is not None:
        source_context = dict(scored.get("source_context", {}))
        source_context["relation"] = "same_source_video"
        source_context["single_source_pair"] = True
        source_context["template_route"] = "single_source_chronological"
        scored["source_context"] = source_context
        quality = dict(scored.get("quality", {}))
        quality["acceptance_profile"] = acceptance_profile
        scored["quality"] = quality
        scored["composite_score"] = _candidate_composite_score(quality, source_context)
        return scored, False

    detected = _detect_primary_difference(reference_annotation, target_annotation)
    if detected is None:
        detected = {
            "type": "scene",
            "from": _short_difference_value(str(reference_annotation.get("summary", "")) or "earlier segment"),
            "to": _short_difference_value(str(target_annotation.get("summary", "")) or "later segment"),
            "description": "the later segment differs from the earlier segment but local fields did not isolate a clean single delta",
            "changed_types": ["scene"],
        }
    changed_types = list(detected.pop("changed_types"))
    difference = detected
    hard_negative_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("clip_id") not in {reference_annotation.get("clip_id"), target_annotation.get("clip_id")}
    ][:3]
    if len(hard_negative_annotations) < 2:
        hard_negative_annotations = annotations[:2]
    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    source_context = {
        **_source_context(reference_annotation, target_annotation),
        "relation": "same_source_video",
        "single_source_pair": True,
        "template_route": "single_source_chronological",
        "group_id": group_metadata.get("group_id", ""),
    }
    same_context_score = _pair_context_score(
        semantic_context_score=semantic_context_score,
        source_context=source_context,
    )
    difference_strength_score = _difference_strength_score(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference=difference,
        changed_types=changed_types,
    )
    quality = {
        "same_context_score": round(same_context_score, 3),
        "semantic_context_score": round(semantic_context_score, 3),
        "edit_match_score": round(
            max(
                MIN_PAIR_EDIT_MATCH_SCORE,
                _edit_match_score(
                    same_context_score=same_context_score,
                    primary_difference_type=str(difference.get("type", "")).strip(),
                    changed_types=changed_types,
                ),
            ),
            3,
        ),
        "target_uniqueness_score": round(
            _target_uniqueness_score(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                annotations=annotations,
                primary_difference=difference,
            ),
            3,
        ),
        "difference_strength_score": round(difference_strength_score, 3),
        "difference_type": str(difference.get("type", "")).strip(),
        "acceptance_profile": acceptance_profile,
        "single_source_fallback_candidate": 1.0,
    }
    if difference["type"] == "action":
        quality["action_evidence_score"] = _action_evidence_score(reference_annotation, target_annotation)
    if difference["type"] == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
        quality["has_audio_modality"] = 1.0
    if difference["type"] == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(reference_annotation, target_annotation)
        quality["has_audio_modality"] = 1.0
    dominant_delta_decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=difference,
        quality=quality,
        source_context=source_context,
    )
    quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
    quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
    quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
    quality["dominant_delta_decision"] = dominant_delta_decision
    reference_path = _display_path(root, _resolve_under_root(root, reference_annotation["output_path"]))
    target_path = _display_path(root, _resolve_under_root(root, target_annotation["output_path"]))
    hard_negative_paths = [
        _display_path(root, _resolve_under_root(root, annotation["output_path"])) for annotation in hard_negative_annotations[:3]
    ]
    return (
        {
            "proposal_id": _build_proposal_id(reference_path, target_path),
            "reference_annotation": _sanitize_annotation_for_output(reference_annotation, root),
            "target_annotation": _sanitize_annotation_for_output(target_annotation, root),
            "primary_difference": difference,
            "changed_difference_types": changed_types,
            "quality": quality,
            "composite_score": _candidate_composite_score(quality, source_context),
            "source_context": source_context,
            "dominant_delta_decision": dominant_delta_decision,
            "difference_evidence": _difference_evidence_from_annotations(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                primary_difference=difference,
            ),
            "hard_negative_annotations": [
                _sanitize_annotation_for_output(annotation, root) for annotation in hard_negative_annotations[:3]
            ],
            "hard_negative_paths": hard_negative_paths,
        },
        True,
    )


def _single_source_candidate_prompt_view(candidate: dict[str, Any]) -> dict[str, Any]:
    difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
    audio_dataset_line = _normalize_audio_dataset_line(candidate.get("audio_dataset_line"))
    quality = dict(candidate.get("quality", {})) if isinstance(candidate.get("quality"), dict) else {}
    profile = str(quality.get("audio_line_quality_profile") or candidate.get("audio_line_quality_profile") or "").strip()
    instruction = "Do not copy the heuristic difference if the videos show a stronger product, overlay, object, action, or composition change."
    if audio_dataset_line == VISUAL_AUDIO_ANCHOR_LINE:
        instruction = (
            "A-line visual_audio_anchor: audio is preserved context only. Choose a clear visual delta and do not mention audio, sound, speech, music, or transcript in edit_text."
        )
        if profile == AUDIO_LINE_QUALITY_PROFILE_V4_STRICT:
            instruction += (
                " v4_strict: accept only large visual shot/scene/subject/action changes, like changing a news anchor shot to flood aerial footage. "
                "Reject near-duplicate visuals, lighting changes, tiny hand/object changes, camera distance changes, visible-text-only edits, and wording-only attribute edits."
            )
    elif audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE:
        instruction = (
            "B-line speech_audio_content: choose speech content or concrete non-speech audio_event as the primary delta; reject if visuals are the main change."
        )
        if profile == AUDIO_LINE_QUALITY_PROFILE_V4_STRICT:
            instruction += (
                " v4_strict: the visual context must stay similar, like two cricket broadcast clips where crowd cheering or speech content changes. "
                "Reject pairs with large visual scene/subject changes and reject vague hum/click/tone guesses unless there is explicit evidence."
            )
        elif profile == "v5_audio_primary":
            instruction += (
                " v5_audio_primary: prioritize speech content differences in same-source clips, such as the same speaker or livestream moving from one topic to another. "
                "Use speech when spoken words, transcript, paraphrase, or topic changes are the main difference. Use audio_event only for concrete non-speech sound changes. "
                "Keep edit_text audio-only; visual differences are warnings unless they make listening unnecessary."
            )
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "candidate_index": candidate.get("candidate_index"),
        "audio_dataset_line": audio_dataset_line,
        "reference_clip_id": str(candidate.get("reference_clip_id", "")),
        "target_clip_id": str(candidate.get("target_clip_id", "")),
        "reference_start_seconds": candidate.get("reference_start_seconds"),
        "target_start_seconds": candidate.get("target_start_seconds"),
        "enumeration_hint_only": True,
        "heuristic_difference": {
            "type": str(difference.get("type", "")),
            "from": str(difference.get("from", "")),
            "to": str(difference.get("to", "")),
            "description": str(difference.get("description", "")),
        },
        "risk_flags": list(candidate.get("risk_flags", [])) if isinstance(candidate.get("risk_flags"), list) else [],
        "quality": quality,
        "instruction": instruction,
    }


def _single_source_whole_prompt_view(annotation: dict[str, Any]) -> dict[str, Any]:
    if not annotation:
        return {}
    return {
        "clip_id": str(annotation.get("clip_id", "whole_source")),
        "summary": _truncate_text(annotation.get("summary", ""), 700),
        "subjects": _prompt_list(annotation.get("subjects", []), limit=8, text_limit=80),
        "object_counts": dict(annotation.get("object_counts", {})) if isinstance(annotation.get("object_counts"), dict) else {},
        "actions": _prompt_list(annotation.get("actions", []), limit=8, text_limit=80),
        "scene": _truncate_text(annotation.get("scene", ""), 300),
        "attributes": _prompt_list(annotation.get("attributes", []), limit=8, text_limit=120),
        "storyline": _prompt_list(annotation.get("storyline", []), limit=8, text_limit=220),
        "events": _prompt_list(annotation.get("events", []), limit=8, text_limit=220),
        "visible_text": _prompt_list(annotation.get("visible_text", []), limit=8, text_limit=120),
        "speakers_and_transcript": _prompt_list(annotation.get("speakers_and_transcript", []), limit=6, text_limit=220),
        "audio_events": _prompt_list(annotation.get("audio_events", []), limit=8, text_limit=120),
    }


def _single_source_rejected_model_fields(*, candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
    if not difference or str(difference.get("type", "")).strip() not in ALLOWED_DIFFERENCE_TYPES:
        difference = {
            "type": "scene",
            "from": "reference segment",
            "to": "target segment",
            "description": "single-source pair comparison failed before a reliable dominant delta could be written",
        }
    return {
        "edit_text": "",
        "modalities": ["visual"],
        "reference_caption": "single-source reference segment",
        "target_caption": "single-source target segment",
        "difference": dict(difference),
        "dominant_delta": {
            "type": str(difference.get("type", "scene")),
            "from": str(difference.get("from", "")),
            "to": str(difference.get("to", "")),
            "reason": reason,
        },
        "reference_state": {"main_speaker": "", "inset_subjects": [], "product_overlay": "", "composition": "", "internal_transitions": []},
        "target_state": {"main_speaker": "", "inset_subjects": [], "product_overlay": "", "composition": "", "internal_transitions": []},
        "delta_temporal_extent": {"reference": "", "target": "", "target_coverage": 0.0, "evidence": reason},
        "subject_roles": {"main_speaker": "", "inset_subjects": [], "product_overlay": ""},
        "is_segment_wide_delta": False,
        "discarded_deltas": [],
        "evidence": [reason],
        "confidence": 0.0,
        "accept": False,
        "reject_reason": reason,
    }


def _recheck_existing_single_source_pair_record(
    record: dict[str, Any],
    *,
    acceptance_profile: str,
) -> dict[str, Any]:
    if not bool(record.get("single_source_pair")):
        return record
    difference = dict(record.get("difference", {})) if isinstance(record.get("difference"), dict) else {}
    record_evidence = record.get("evidence", {}) if isinstance(record.get("evidence"), dict) else {}
    pair_video_evidence = record.get("pair_video_evidence")
    if not isinstance(pair_video_evidence, list):
        pair_video_evidence = record_evidence.get("pair_video_comparison", [])
    if not isinstance(pair_video_evidence, list):
        pair_video_evidence = []
    model_fields = {
        "edit_text": str(record.get("edit_text", "")).strip(),
        "modalities": list(record.get("modalities", [])) if isinstance(record.get("modalities"), list) else ["visual"],
        "reference_caption": str(record.get("reference_caption", "")).strip(),
        "target_caption": str(record.get("target_caption", "")).strip(),
        "difference": difference,
        "dominant_delta": dict(record.get("dominant_delta", {})) if isinstance(record.get("dominant_delta"), dict) else {},
        "reference_state": dict(record.get("reference_state", {})) if isinstance(record.get("reference_state"), dict) else {},
        "target_state": dict(record.get("target_state", {})) if isinstance(record.get("target_state"), dict) else {},
        "delta_temporal_extent": dict(record.get("delta_temporal_extent", {})) if isinstance(record.get("delta_temporal_extent"), dict) else {},
        "subject_roles": dict(record.get("subject_roles", {})) if isinstance(record.get("subject_roles"), dict) else {},
        "is_segment_wide_delta": bool(record.get("is_segment_wide_delta")),
        "discarded_deltas": list(record.get("discarded_deltas", [])) if isinstance(record.get("discarded_deltas"), list) else [],
        "evidence": list(pair_video_evidence),
        "confidence": _score_float(record.get("confidence")),
        "accept": bool(record.get("model_accepted", record.get("accepted"))),
        "reject_reason": str(record.get("judge", {}).get("reject_reason", "")).strip() if isinstance(record.get("judge"), dict) else "",
    }
    edit_text_quality = _edit_text_quality_payload(
        edit_text=str(model_fields.get("edit_text", "")),
        difference=difference,
        modalities=list(model_fields.get("modalities", [])),
        reference_caption=str(model_fields.get("reference_caption", "")),
        target_caption=str(model_fields.get("target_caption", "")),
    )
    issues = _single_source_pair_acceptance_issues(
        model_fields=model_fields,
        edit_text_quality=edit_text_quality,
        acceptance_profile=acceptance_profile,
        audio_dataset_line=str(record.get("audio_dataset_line") or STANDARD_AUDIO_DATASET_LINE),
        candidate_quality=record.get("quality", {}) if isinstance(record.get("quality"), dict) else {},
        reference_annotation=record.get("reference_annotation", {}) if isinstance(record.get("reference_annotation"), dict) else None,
        target_annotation=record.get("target_annotation", {}) if isinstance(record.get("target_annotation"), dict) else None,
    )
    local_gate_report = _single_source_local_gate_report(
        acceptance_issues=issues,
        fallback_used=bool(record.get("fallback_used")),
        difference_type=str(difference.get("type", "")).strip(),
        confidence=_score_float(record.get("confidence")),
        acceptance_profile=acceptance_profile,
        audio_dataset_line=str(record.get("audio_dataset_line") or STANDARD_AUDIO_DATASET_LINE),
        reference_video_exists=True,
        target_video_exists=True,
    )
    final_omni_verification = (
        record.get("final_omni_verification")
        if isinstance(record.get("final_omni_verification"), dict)
        else _single_source_skipped_final_verification("final_omni_verification_missing")
    )
    final_issues = _single_source_final_verification_issues(
        final_omni_verification,
        acceptance_profile=acceptance_profile,
        audio_dataset_line=str(record.get("audio_dataset_line") or STANDARD_AUDIO_DATASET_LINE),
        model_fields=model_fields,
    )
    final_review_required = _single_source_final_verification_review_required(
        final_omni_verification,
        acceptance_profile=acceptance_profile,
        audio_dataset_line=str(record.get("audio_dataset_line") or STANDARD_AUDIO_DATASET_LINE),
    )
    local_review_required = list(local_gate_report.get("review_required", []))
    blocking_local_hard_rejects = list(local_gate_report.get("hard_reject", []))
    record_audio_line = _normalize_audio_dataset_line(str(record.get("audio_dataset_line") or STANDARD_AUDIO_DATASET_LINE))
    if record_audio_line == VISUAL_AUDIO_ANCHOR_LINE and not final_issues:
        blocking_local_hard_rejects = _a_line_unrescued_local_hard_rejects(
            blocking_local_hard_rejects,
            final_omni_verification,
        )
    elif record_audio_line == SPEECH_AUDIO_CONTENT_LINE and not final_issues:
        blocking_local_hard_rejects = _b_line_unrescued_local_hard_rejects(
            blocking_local_hard_rejects,
            final_omni_verification,
        )
    blocking_issues = _dedupe_strings(
        blocking_local_hard_rejects + (local_review_required if final_issues else []) + final_issues
    )
    record["local_gate_report"] = local_gate_report
    record["final_omni_verification"] = final_omni_verification
    record["final_omni_accept"] = bool(
        _boolish(final_omni_verification.get("accept")) and not final_issues and not blocking_local_hard_rejects
    )
    record["final_accept_source"] = "local_gate_and_final_omni"
    record["single_source_pair_acceptance_issues"] = blocking_issues
    record["single_source_pair_review_required"] = _dedupe_strings(local_review_required + final_review_required)
    record["recommended_edit_text"] = record.get("recommended_edit_text") or _single_source_recommended_edit_text(model_fields)
    record["single_source_delta_family"] = record.get("single_source_delta_family") or _single_source_delta_family_from_fields(model_fields)
    record["model_accepted"] = bool(model_fields.get("accept")) and not bool(record.get("fallback_used"))
    record["local_gate_passed"] = bool(local_gate_report.get("passed"))
    _set_single_source_record_acceptance(
        record,
        accepted=bool(record["model_accepted"] and record["final_omni_accept"] and not blocking_issues),
        extra_issues=blocking_issues,
    )
    return record


def _single_source_hard_negative_paths(
    *,
    root: Path,
    candidate: dict[str, Any],
    annotations: list[dict[str, Any]],
    reference_clip_id: str,
    target_clip_id: str,
) -> list[str]:
    paths = [
        str(item).strip()
        for item in candidate.get("hard_negative_paths", [])
        if str(item).strip()
    ] if isinstance(candidate.get("hard_negative_paths"), list) else []
    valid_paths: list[str] = []
    for path in paths:
        resolved = _resolve_under_root(root, path)
        if resolved.exists() and path not in valid_paths:
            valid_paths.append(path)
    if len(valid_paths) >= 2:
        return valid_paths[:3]
    for annotation in annotations:
        clip_id = str(annotation.get("clip_id", "")).strip()
        if clip_id in {reference_clip_id, target_clip_id}:
            continue
        output_path = str(annotation.get("output_path", "")).strip()
        if output_path and _resolve_under_root(root, output_path).exists() and output_path not in valid_paths:
            valid_paths.append(output_path)
        if len(valid_paths) >= 3:
            break
    return valid_paths


def _single_source_pair_quality(
    *,
    candidate: dict[str, Any],
    model_fields: dict[str, Any],
    acceptance_profile: str,
) -> dict[str, Any]:
    scores = candidate.get("scores", {}) if isinstance(candidate.get("scores"), dict) else {}
    heuristic_quality = candidate.get("quality", {}) if isinstance(candidate.get("quality"), dict) else {}
    confidence = _score_float(model_fields.get("confidence"))
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    extent = model_fields.get("delta_temporal_extent") if isinstance(model_fields.get("delta_temporal_extent"), dict) else {}
    quality = {
        "same_context_score": max(0.65, _score_float(scores.get("same_context_score", heuristic_quality.get("same_context_score")))),
        "semantic_context_score": _score_float(scores.get("semantic_context_score", heuristic_quality.get("semantic_context_score"))),
        "edit_match_score": confidence,
        "target_uniqueness_score": max(confidence, _score_float(scores.get("target_uniqueness_score", heuristic_quality.get("target_uniqueness_score")))),
        "difference_strength_score": max(confidence, _score_float(scores.get("difference_strength_score", heuristic_quality.get("difference_strength_score")))),
        "difference_type": difference_type,
        "pair_video_comparison_confidence": confidence,
        "delta_target_coverage": _score_float(extent.get("target_coverage")),
        "is_segment_wide_delta": 1.0 if bool(model_fields.get("is_segment_wide_delta")) else 0.0,
        "acceptance_profile": acceptance_profile,
    }
    for key in (
        "audio_anchor_score",
        "audio_anchor_required",
        "audio_anchor_type",
        "audio_anchor_context_score",
        "audio_anchor_min_rms",
        "edit_primary_modality",
        "speech_evidence_score",
        "speech_specificity_score",
        "speech_transcript_backed",
        "non_speech_audio_event_score",
        "has_audio_modality",
        "audio_dataset_line",
        "audio_line_quality_profile",
        "visual_delta_strength",
        "visual_context_similarity",
        "audio_content_delta_strength",
        "b_subtype",
        "video_context_type",
        "video_context_strength",
        "asr_degeneracy_risk",
    ):
        if key in heuristic_quality:
            quality[key] = heuristic_quality[key]
        elif key in scores:
            quality[key] = scores[key]
    return quality


def _single_source_pair_acceptance_issues(
    *,
    model_fields: dict[str, Any],
    edit_text_quality: dict[str, Any],
    acceptance_profile: str,
    audio_dataset_line: str = STANDARD_AUDIO_DATASET_LINE,
    candidate_quality: dict[str, Any] | None = None,
    reference_annotation: dict[str, Any] | None = None,
    target_annotation: dict[str, Any] | None = None,
) -> list[str]:
    audio_dataset_line = _normalize_audio_dataset_line(audio_dataset_line)
    reasons = _single_source_model_reject_issues(
        model_fields,
        edit_text_quality,
        audio_dataset_line=audio_dataset_line,
    )
    confidence = _score_float(model_fields.get("confidence"))
    threshold = _profile_threshold(acceptance_profile, "edit_match_score")
    if confidence < threshold:
        reasons.append(f"low_pair_video_confidence: {confidence:.2f} < {threshold:.2f}")

    extent = model_fields.get("delta_temporal_extent") if isinstance(model_fields.get("delta_temporal_extent"), dict) else {}
    if not extent:
        reasons.append("missing_delta_temporal_extent")
    target_coverage = _score_float(extent.get("target_coverage")) if extent else 0.0
    extent_text = _normalized_phrase(
        " ".join(
            str(extent.get(key, ""))
            for key in ("reference", "target", "evidence")
        )
        if extent
        else ""
    )
    if extent and target_coverage <= 0.0:
        reasons.append("missing_delta_target_coverage")
    elif 0.0 < target_coverage < 0.55:
        reasons.append("transient_delta_not_segment_wide")
    if any(marker in extent_text for marker in ("brief", "briefly", "last moment", "end of clip", "only at the end", "final moment")):
        reasons.append("transient_delta_not_segment_wide")
    if not bool(model_fields.get("is_segment_wide_delta")):
        reasons.append("transient_delta_not_segment_wide")

    reference_state = model_fields.get("reference_state") if isinstance(model_fields.get("reference_state"), dict) else {}
    target_state = model_fields.get("target_state") if isinstance(model_fields.get("target_state"), dict) else {}
    internal_transition_text = _normalized_phrase(
        " ".join(
            _normalize_list(reference_state.get("internal_transitions", []))
            + _normalize_list(target_state.get("internal_transitions", []))
        )
    )
    edit_text = str(model_fields.get("edit_text", "")).strip()
    normalized_edit = _normalized_phrase(edit_text)
    if internal_transition_text and (
        not bool(model_fields.get("is_segment_wide_delta"))
        or any(marker in internal_transition_text for marker in ("appears", "disappears", "then", "followed by", "changes from"))
    ):
        reasons.append("segment_internal_transition")

    target_role_text = _normalized_phrase(
        " ".join(
            [
                str(target_state.get("main_speaker", "")),
                str(target_state.get("product_overlay", "")),
                str(target_state.get("composition", "")),
            ]
            + _normalize_list(target_state.get("inset_subjects", []))
        )
    )
    roles = model_fields.get("subject_roles") if isinstance(model_fields.get("subject_roles"), dict) else {}
    main_speaker = _normalized_phrase(str(roles.get("main_speaker", target_state.get("main_speaker", ""))))
    inset_subjects = _normalized_phrase(" ".join(_normalize_list(roles.get("inset_subjects", target_state.get("inset_subjects", [])))))
    product_overlay = _normalized_phrase(str(roles.get("product_overlay", target_state.get("product_overlay", ""))))

    if "close up" in normalized_edit or "closeup" in normalized_edit:
        if main_speaker or "speaker" in target_role_text or "woman" in target_role_text:
            reasons.append("composition_label_mismatch: product close-up claimed while speaker remains visible")
    if "full screen" in normalized_edit or "fullscreen" in normalized_edit:
        if product_overlay or "overlay" in target_role_text or main_speaker:
            reasons.append("composition_label_mismatch: full-screen claimed while speaker or overlay remains visible")
    if "change the shot from the speaker" in normalized_edit and (main_speaker or "speaker" in target_role_text):
        reasons.append("composition_label_mismatch: target still contains the speaker")
    if "man speaking" in normalized_edit and "inset" not in normalized_edit and "picture in picture" not in normalized_edit:
        if inset_subjects:
            reasons.append("subject_role_mismatch: inset man described as primary subject")
    if "woman receiving" in normalized_edit and "inset" not in normalized_edit and "picture in picture" not in normalized_edit:
        if inset_subjects:
            reasons.append("subject_role_mismatch: inset woman described as primary subject")
    text_driven_issue = _single_source_text_driven_product_change_issue(
        model_fields=model_fields,
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    )
    if text_driven_issue:
        reasons.append(text_driven_issue)
    reasons.extend(
        _single_source_audio_line_acceptance_issues(
            model_fields=model_fields,
            audio_dataset_line=audio_dataset_line,
            candidate_quality=candidate_quality,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
    )

    return _dedupe_strings(reasons)


def _single_source_local_gate_report(
    *,
    acceptance_issues: list[str],
    fallback_used: bool,
    difference_type: str,
    confidence: float,
    acceptance_profile: str,
    reference_video_exists: bool,
    target_video_exists: bool,
    audio_dataset_line: str = STANDARD_AUDIO_DATASET_LINE,
) -> dict[str, Any]:
    audio_dataset_line = _normalize_audio_dataset_line(audio_dataset_line)
    hard_rejects: list[str] = []
    review_required: list[str] = []
    threshold = _profile_threshold(acceptance_profile, "edit_match_score")

    if fallback_used:
        hard_rejects.append("fallback_pair_proposal")
    if not reference_video_exists:
        hard_rejects.append("reference_video_missing")
    if not target_video_exists:
        hard_rejects.append("target_video_missing")
    disabled_difference_types = set(FINAL_DISABLED_DIFFERENCE_TYPES)
    if audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE:
        disabled_difference_types.discard("speech")
    if difference_type in disabled_difference_types:
        hard_rejects.append(f"{difference_type} is diagnostic-only for single-source accepted pairs")
    if audio_dataset_line == VISUAL_AUDIO_ANCHOR_LINE and difference_type not in DOMINANT_VISUAL_DIFFERENCE_TYPES:
        hard_rejects.append(f"{difference_type} is not allowed for visual_audio_anchor")
    if audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE and difference_type not in {"speech", "audio_event"}:
        hard_rejects.append(f"{difference_type} is not allowed for speech_audio_content")
    if confidence < threshold:
        hard_rejects.append(f"low_pair_video_confidence: {confidence:.2f} < {threshold:.2f}")

    for issue in acceptance_issues:
        normalized = str(issue).strip()
        if not normalized:
            continue
        if normalized.startswith(
            (
                "transient_delta_not_segment_wide",
                "segment_internal_transition",
                "composition_label_mismatch",
                "subject_role_mismatch",
            )
        ):
            review_required.append(normalized)
        else:
            hard_rejects.append(normalized)

    hard_rejects = _dedupe_strings(hard_rejects)
    review_required = _dedupe_strings(review_required)
    return {
        "passed": not hard_rejects,
        "hard_reject": hard_rejects,
        "review_required": review_required,
        "all_issues": _dedupe_strings(hard_rejects + review_required),
        "confidence": round(confidence, 3),
        "confidence_threshold": threshold,
        "frame_check_points_seconds": [0.2, 2.5, 4.8],
    }


def _single_source_skipped_final_verification(reason: str) -> dict[str, Any]:
    return {
        "accept": False,
        "confidence": 0.0,
        "quality_score": 0.0,
        "reference_satisfies_edit": False,
        "target_satisfies_edit": False,
        "observable_delta": False,
        "single_primary_delta": False,
        "text_or_ocr_driven": False,
        "segment_wide": False,
        "edit_text_accurate": False,
        "main_reject_reason": reason,
        "evidence": [],
        "recommended_edit_text": "",
        "audio_primary": False,
        "visual_locked": False,
        "visual_too_different_for_B": False,
        "edit_text_audio_only": False,
        "visual_context_preserved": False,
        "video_context_strength": 0.0,
        "asr_degeneracy_risk": 1.0,
        "not_asr_only": False,
        "large_visual_delta": False,
        "audio_context_preserved": False,
        "skipped": True,
    }


def _a_line_can_run_final_rescue(
    *,
    audio_dataset_line: str,
    model_fields: dict[str, Any],
    fallback_used: bool,
    reference_video_exists: bool,
    target_video_exists: bool,
) -> bool:
    if _normalize_audio_dataset_line(audio_dataset_line) != VISUAL_AUDIO_ANCHOR_LINE:
        return False
    if fallback_used or not reference_video_exists or not target_video_exists:
        return False
    if not bool(model_fields.get("accept")):
        return False
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    return str(difference.get("type", "")).strip() in DOMINANT_VISUAL_DIFFERENCE_TYPES


def _b_line_can_run_final_rescue(
    *,
    audio_dataset_line: str,
    model_fields: dict[str, Any],
    fallback_used: bool,
    reference_video_exists: bool,
    target_video_exists: bool,
) -> bool:
    if _normalize_audio_dataset_line(audio_dataset_line) != SPEECH_AUDIO_CONTENT_LINE:
        return False
    if fallback_used or not reference_video_exists or not target_video_exists:
        return False
    if not bool(model_fields.get("accept")):
        return False
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    return str(difference.get("type", "")).strip() in {"speech", "audio_event"}


def _extract_audio_only_cache(*, video_path: Path, cache_dir: Path, clip_id: str) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"video file not found for audio extraction: {video_path}")
    stat = video_path.stat()
    cache_key = json.dumps(
        {
            "path": str(video_path.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        },
        sort_keys=True,
    )
    safe_clip_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip_id).strip("_") or "clip"
    output_path = cache_dir / f"{safe_clip_id}_{_stable_hash(cache_key)}.wav"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.wav")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(tmp_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            raise RuntimeError("ffmpeg produced an empty wav")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return output_path


def _b_audio_blind_review_model_fields(audio_only_proposal: dict[str, Any]) -> dict[str, Any]:
    difference_type = str(audio_only_proposal.get("difference_type", "")).strip()
    if difference_type == "speech_topic":
        difference_type = "speech"
    if difference_type not in {"speech", "audio_event"}:
        difference_type = "audio_event" if str(audio_only_proposal.get("b_subtype", "")) in {"music", "sound_event"} else "speech"
    reference_audio = str(audio_only_proposal.get("reference_audio_content", "")).strip()
    target_audio = str(audio_only_proposal.get("target_audio_content", "")).strip()
    edit_text = str(audio_only_proposal.get("edit_text", "")).strip()
    return {
        "edit_text": edit_text,
        "modalities": ["audio"],
        "reference_caption": reference_audio,
        "target_caption": target_audio,
        "difference": {
            "type": difference_type,
            "from": reference_audio,
            "to": target_audio,
            "description": "; ".join(_normalize_list(audio_only_proposal.get("evidence", []))) or edit_text,
        },
        "dominant_delta": {
            "type": difference_type,
            "from": reference_audio,
            "to": target_audio,
            "reason": "audio-only blind proposal",
        },
        "reference_state": {"composition": "", "main_speaker": ""},
        "target_state": {"composition": "", "main_speaker": ""},
        "delta_temporal_extent": {
            "reference": reference_audio,
            "target": target_audio,
            "target_coverage": _score_float(audio_only_proposal.get("confidence")),
            "evidence": "; ".join(_normalize_list(audio_only_proposal.get("evidence", []))),
        },
        "subject_roles": {"main_speaker": "", "inset_subjects": [], "product_overlay": ""},
        "is_segment_wide_delta": True,
        "discarded_deltas": [],
        "evidence": _normalize_list(audio_only_proposal.get("evidence", [])),
        "confidence": _score_float(audio_only_proposal.get("confidence")),
        "accept": _boolish(audio_only_proposal.get("accept")),
        "reject_reason": str(audio_only_proposal.get("reject_reason", "")).strip(),
    }


def _b_audio_blind_review_issues(
    *,
    audio_only_proposal: dict[str, Any],
    audio_only_verification: dict[str, Any],
    full_av_consistency: dict[str, Any],
    quality: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    difference_type = str(audio_only_proposal.get("difference_type", "")).strip()
    if difference_type == "speech_topic":
        difference_type = "speech"
    if difference_type not in {"speech", "audio_event"}:
        issues.append(f"audio_only_invalid_difference_type: {difference_type or 'missing'}")
    edit_text = str(audio_only_proposal.get("edit_text", "")).strip()
    issues.extend(_b_line_edit_text_audio_only_issues(edit_text, difference_type))
    if not _boolish(audio_only_proposal.get("accept")):
        reason = str(audio_only_proposal.get("reject_reason", "")).strip()
        issues.append("audio_only_proposal_reject" + (f": {reason}" if reason else ""))
    if _score_float(audio_only_proposal.get("confidence")) < 0.70:
        issues.append(f"audio_only_proposal_confidence_below_threshold: {_score_float(audio_only_proposal.get('confidence')):.2f} < 0.70")
    if not _boolish(audio_only_proposal.get("audio_difference_specific")):
        issues.append("audio_only_difference_not_specific")
    if not _boolish(audio_only_proposal.get("edit_text_audio_only")):
        issues.append("audio_only_edit_text_not_audio_only")
    if _b_line_audio_phrase_is_hollow(str(audio_only_proposal.get("reference_audio_content", ""))) or _b_line_audio_phrase_is_hollow(
        str(audio_only_proposal.get("target_audio_content", ""))
    ):
        issues.append("audio_only_hollow_content")

    if not _boolish(audio_only_verification.get("accept")):
        reason = str(audio_only_verification.get("reject_reason", "")).strip()
        issues.append("audio_only_verification_reject" + (f": {reason}" if reason else ""))
    if _boolish(audio_only_verification.get("reference_satisfies_edit")):
        issues.append("audio_only_reference_satisfies_edit")
    if not _boolish(audio_only_verification.get("target_satisfies_edit")):
        issues.append("audio_only_target_missing_edit")
    if not _boolish(audio_only_verification.get("audio_difference_specific")):
        issues.append("audio_only_verification_not_specific")
    if not _boolish(audio_only_verification.get("edit_text_audio_only")):
        issues.append("audio_only_verification_edit_text_not_audio_only")
    if _score_float(audio_only_verification.get("confidence")) < 0.70:
        issues.append(f"audio_only_verification_confidence_below_threshold: {_score_float(audio_only_verification.get('confidence')):.2f} < 0.70")

    if not _boolish(full_av_consistency.get("accept")):
        reason = str(full_av_consistency.get("reject_reason", "")).strip()
        issues.append("full_av_consistency_reject" + (f": {reason}" if reason else ""))
    if not _boolish(full_av_consistency.get("visual_context_preserved")):
        issues.append("full_av_visual_context_not_preserved")
    if _boolish(full_av_consistency.get("visual_shortcut_risk")):
        issues.append("visual_shortcut_risk")
    if not _boolish(full_av_consistency.get("audio_edit_still_valid")):
        issues.append("full_av_audio_edit_not_valid")
    if _score_float(full_av_consistency.get("confidence")) < 0.60:
        issues.append(f"full_av_consistency_confidence_below_threshold: {_score_float(full_av_consistency.get('confidence')):.2f} < 0.60")

    if _score_float(quality.get("video_context_strength")) < _profile_threshold(B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE, "video_context_strength"):
        issues.append("blind_review_video_context_too_weak")
    if _score_float(quality.get("asr_degeneracy_risk")) > _profile_threshold(B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE, "asr_degeneracy_risk"):
        issues.append("blind_review_asr_degeneracy_risk_too_high")
    if _score_float(quality.get("visual_delta_strength")) > _profile_threshold(B_AUDIO_BLIND_REVIEW_ACCEPTANCE_PROFILE, "visual_delta_strength"):
        issues.append("visual_shortcut_risk: local visual_delta_strength too high")
    return _dedupe_strings(issues)


def _a_line_unrescued_local_hard_rejects(
    local_hard_rejects: list[str],
    final_verification: dict[str, Any],
) -> list[str]:
    if not (
        _boolish(final_verification.get("accept"))
        and _boolish(final_verification.get("large_visual_delta"))
        and _boolish(final_verification.get("audio_context_preserved"))
    ):
        return local_hard_rejects

    unrescued: list[str] = []
    for issue in local_hard_rejects:
        normalized = str(issue).strip()
        if not normalized:
            continue
        if any(normalized.startswith(prefix) for prefix in A_LINE_FINAL_RESCUABLE_LOCAL_ISSUE_PREFIXES):
            continue
        unrescued.append(normalized)
    return _dedupe_strings(unrescued)


def _b_line_unrescued_local_hard_rejects(
    local_hard_rejects: list[str],
    final_verification: dict[str, Any],
) -> list[str]:
    if not (
        _boolish(final_verification.get("accept"))
        and _boolish(final_verification.get("audio_primary"))
        and _boolish(final_verification.get("visual_locked"))
        and not _boolish(final_verification.get("visual_too_different_for_B"))
        and _boolish(final_verification.get("edit_text_audio_only"))
    ):
        return local_hard_rejects

    unrescued: list[str] = []
    for issue in local_hard_rejects:
        normalized = str(issue).strip()
        if not normalized:
            continue
        if normalized.startswith("edit_text_not_audio_only: identical audio endpoints") or normalized.startswith(
            "edit_text_not_audio_only: hollow speech target"
        ):
            unrescued.append(normalized)
            continue
        if any(normalized.startswith(prefix) for prefix in B_LINE_FINAL_RESCUABLE_LOCAL_ISSUE_PREFIXES):
            continue
        unrescued.append(normalized)
    return _dedupe_strings(unrescued)


def _single_source_final_verification_issues(
    final_verification: dict[str, Any],
    *,
    acceptance_profile: str,
    audio_dataset_line: str = STANDARD_AUDIO_DATASET_LINE,
    model_fields: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(final_verification, dict) or not final_verification:
        return ["final_omni_verification_missing"]
    audio_dataset_line = _normalize_audio_dataset_line(audio_dataset_line)
    b_audio_review = audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE and _is_b_audio_review_profile(acceptance_profile)
    b_audio_context_cvr = audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE and _is_b_audio_context_cvr_profile(acceptance_profile)
    difference = model_fields.get("difference") if isinstance(model_fields, dict) and isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    threshold = _profile_threshold(acceptance_profile, "edit_match_score")
    issues: list[str] = []
    confidence = _score_float(final_verification.get("confidence"))
    quality_score = _score_float(final_verification.get("quality_score"))
    quality_threshold = _single_source_final_omni_quality_threshold(
        acceptance_profile=acceptance_profile,
        audio_dataset_line=audio_dataset_line,
    )
    reason = str(final_verification.get("main_reject_reason", "")).strip()
    if not _boolish(final_verification.get("accept")):
        issues.append("final_omni_reject" + (f": {reason}" if reason else ""))
    if confidence < threshold:
        issues.append(f"final_omni_low_confidence: {confidence:.2f} < {threshold:.2f}")
    if quality_score < quality_threshold:
        issues.append(
            "final_omni_quality_score_below_threshold: "
            f"{quality_score:.2f} < {quality_threshold:.2f}"
        )
    if _boolish(final_verification.get("reference_satisfies_edit")):
        issues.append("final_omni_reference_satisfies_edit")
    if not _boolish(final_verification.get("target_satisfies_edit")):
        issues.append("final_omni_target_missing_edit")
    if not _boolish(final_verification.get("observable_delta")):
        issues.append("final_omni_missing_observable_delta")
    if not _boolish(final_verification.get("single_primary_delta")):
        issues.append("final_omni_not_single_primary_delta")
    if _boolish(final_verification.get("text_or_ocr_driven")):
        issues.append("final_omni_text_or_ocr_driven")
    if not _boolish(final_verification.get("segment_wide")) and not b_audio_review:
        issues.append("final_omni_delta_not_segment_wide")
    if not _boolish(final_verification.get("edit_text_accurate")):
        issues.append("final_omni_edit_text_inaccurate")
    if audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE:
        if (
            not _boolish(final_verification.get("audio_primary"))
            and (not b_audio_review or _boolish(final_verification.get("visual_too_different_for_B")) or difference_type not in {"speech", "audio_event"})
        ):
            issues.append("final_omni_audio_not_primary")
        if (
            not _boolish(final_verification.get("visual_locked"))
            and (not b_audio_review or _boolish(final_verification.get("visual_too_different_for_B")))
        ):
            issues.append("final_omni_visual_not_locked")
        if _boolish(final_verification.get("visual_too_different_for_B")):
            issues.append("final_omni_visual_too_different_for_B")
        if not _boolish(final_verification.get("edit_text_audio_only")):
            issues.append("final_omni_edit_text_not_audio_only")
        if b_audio_context_cvr:
            video_context_strength = _score_float(final_verification.get("video_context_strength"))
            asr_degeneracy_risk = _score_float(final_verification.get("asr_degeneracy_risk"))
            not_asr_only = True if "not_asr_only" not in final_verification else _boolish(final_verification.get("not_asr_only"))
            if "visual_context_preserved" in final_verification and not _boolish(final_verification.get("visual_context_preserved")):
                issues.append("final_omni_visual_context_not_preserved")
            if video_context_strength < _profile_threshold(B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE, "video_context_strength"):
                issues.append(
                    "final_omni_video_context_too_weak: "
                    f"{video_context_strength:.2f} < 0.45"
                )
            if asr_degeneracy_risk > _profile_threshold(B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE, "asr_degeneracy_risk"):
                issues.append(
                    "final_omni_asr_degeneracy_risk_too_high: "
                    f"{asr_degeneracy_risk:.2f} > 0.55"
                )
            if not not_asr_only:
                issues.append("final_omni_asr_only")
    elif audio_dataset_line == VISUAL_AUDIO_ANCHOR_LINE:
        if "large_visual_delta" in final_verification and not _boolish(final_verification.get("large_visual_delta")):
            issues.append("final_omni_visual_delta_too_small_for_A")
        if "audio_context_preserved" in final_verification and not _boolish(final_verification.get("audio_context_preserved")):
            issues.append("final_omni_audio_context_not_preserved_for_A")
    return _dedupe_strings(issues)


def _single_source_final_omni_quality_threshold(
    *,
    acceptance_profile: str,
    audio_dataset_line: str,
) -> float:
    if _normalize_audio_dataset_line(audio_dataset_line) == SPEECH_AUDIO_CONTENT_LINE and _is_b_audio_review_profile(acceptance_profile):
        return 0.60
    return MIN_SINGLE_SOURCE_FINAL_OMNI_QUALITY_SCORE


def _single_source_final_verification_review_required(
    final_verification: dict[str, Any],
    *,
    acceptance_profile: str,
    audio_dataset_line: str = STANDARD_AUDIO_DATASET_LINE,
) -> list[str]:
    if not isinstance(final_verification, dict) or not final_verification:
        return []
    if not (
        _normalize_audio_dataset_line(audio_dataset_line) == SPEECH_AUDIO_CONTENT_LINE
        and _is_b_audio_review_profile(acceptance_profile)
    ):
        return []
    review: list[str] = []
    if not _boolish(final_verification.get("segment_wide")):
        review.append("final_omni_delta_not_segment_wide")
    if not _boolish(final_verification.get("audio_primary")):
        review.append("final_omni_audio_not_primary")
    if not _boolish(final_verification.get("visual_locked")):
        review.append("final_omni_visual_not_locked")
    if _is_b_audio_context_cvr_profile(acceptance_profile):
        if "asr_degeneracy_risk" in final_verification and _score_float(final_verification.get("asr_degeneracy_risk")) > 0.40:
            review.append("asr_degeneracy_risk_review")
        if "video_context_strength" in final_verification and _score_float(final_verification.get("video_context_strength")) < 0.65:
            review.append("video_context_strength_review")
    return _dedupe_strings(review)


def _b_line_edit_text_refinement_issues(refinement: dict[str, Any]) -> list[str]:
    if not isinstance(refinement, dict) or not refinement:
        return ["edit_text_refinement_missing"]
    refined_edit = str(refinement.get("refined_edit_text", "")).strip()
    difference_type = "audio_event" if any(term in _normalized_phrase(refined_edit) for term in ("audio", "sound", "music", "cheer", "applause", "ambient")) else "speech"
    issues = _b_line_edit_text_audio_only_issues(refined_edit, difference_type)
    score = _score_float(refinement.get("edit_text_specificity_score"))
    if score < 0.70:
        issues.append(f"edit_text_specificity_score_below_threshold: {score:.2f} < 0.70")
    if _boolish(refinement.get("reject_if_unspecific")):
        reason = str(refinement.get("edit_text_reject_reason", "")).strip()
        issues.append("edit_text_refinement_reject" + (f": {reason}" if reason else ""))
    return _dedupe_strings(issues)


def _b_line_should_run_speech_rewrite(
    *,
    model_fields: dict[str, Any],
    final_verification: dict[str, Any],
    edit_text_refinement: dict[str, Any],
    refinement_issues: list[str],
) -> bool:
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    if str(difference.get("type", "")).strip() != "speech":
        return False
    if _boolish(final_verification.get("visual_too_different_for_B")):
        return False
    if not _boolish(final_verification.get("target_satisfies_edit")):
        return False
    refined_edit = str(edit_text_refinement.get("refined_edit_text") or model_fields.get("edit_text") or "").strip()
    specificity_issues = _b_line_edit_text_specificity_issues(refined_edit, "speech")
    if refinement_issues or specificity_issues:
        return True
    return _score_float(model_fields.get("confidence")) < MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE


def _b_line_speech_rewrite_issues(rewrite: dict[str, Any]) -> list[str]:
    if not isinstance(rewrite, dict) or not rewrite:
        return ["speech_rewrite_missing"]
    issues: list[str] = []
    refined_edit = str(rewrite.get("refined_edit_text", "")).strip()
    reference_content = str(rewrite.get("reference_speech_content", "")).strip()
    target_content = str(rewrite.get("target_speech_content", "")).strip()
    issues.extend(_b_line_edit_text_audio_only_issues(refined_edit, "speech"))
    confidence = _score_float(rewrite.get("speech_transcription_confidence"))
    if confidence < 0.70:
        issues.append(f"speech_rewrite_confidence_below_threshold: {confidence:.2f} < 0.70")
    if _boolish(rewrite.get("reject_if_still_unclear")):
        reason = str(rewrite.get("speech_rewrite_reject_reason", "")).strip()
        issues.append("speech_rewrite_reject" + (f": {reason}" if reason else ""))
    if _b_line_audio_phrase_is_hollow(reference_content) or _b_line_audio_phrase_is_hollow(target_content):
        issues.append("speech_rewrite_hollow_content")
    if reference_content and target_content and _normalized_phrase(reference_content) == _normalized_phrase(target_content):
        issues.append("speech_rewrite_identical_content")
    return _dedupe_strings(issues)


def _b_line_video_context_text(annotation: dict[str, Any] | None) -> str:
    if not isinstance(annotation, dict):
        return ""
    return _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                str(annotation.get("scene", "")),
                " ".join(_normalize_list(annotation.get("subjects", []))),
                " ".join(_normalize_list(annotation.get("actions", []))),
                " ".join(_normalize_list(annotation.get("events", []))),
                " ".join(_normalize_list(annotation.get("storyline", []))),
                " ".join(_normalize_list(annotation.get("detective_notes", []))),
                str(annotation.get("video_context_type", "")),
                str(annotation.get("speech_role", "")),
            ]
        )
    )


def _b_line_video_context_type(reference_annotation: dict[str, Any] | None, target_annotation: dict[str, Any] | None) -> str:
    text = _b_line_video_context_text(reference_annotation) + " " + _b_line_video_context_text(target_annotation)
    if any(term in text for term in ("news", "report", "anchor", "broadcast", "journalist")):
        return "news/reporting"
    if any(term in text for term in ("sport", "match", "game", "cricket", "football", "basketball", "player", "commentary")):
        return "sports_commentary"
    if any(term in text for term in ("tutorial", "instruction", "cook", "recipe", "repair", "demo", "demonstration", "how to")):
        return "tutorial_instruction"
    if any(term in text for term in ("interview", "podium", "press", "stage", "panel")):
        return "interview_context"
    if any(term in text for term in ("livestream", "live stream", "streamer", "vlog", "studio", "desk")):
        return "livestream_context"
    if any(term in text for term in ("singing", "song", "music", "guitar", "piano", "performance", "concert")):
        return "performance_or_singing"
    if any(term in text for term in ("meeting", "conference call", "webinar", "zoom", "slide deck")):
        return "asr_only"
    if any(term in text for term in ("talking head", "speaking to camera", "speaker")):
        return "generic_talking_head"
    return "unknown"


def _b_line_video_context_strength(
    reference_annotation: dict[str, Any] | None,
    target_annotation: dict[str, Any] | None,
    candidate_quality: dict[str, Any] | None = None,
) -> float:
    candidate_quality = candidate_quality if isinstance(candidate_quality, dict) else {}
    provided = max(
        _score_float(candidate_quality.get("video_context_strength")),
        _score_float((reference_annotation or {}).get("video_context_strength") if isinstance(reference_annotation, dict) else 0.0),
        _score_float((target_annotation or {}).get("video_context_strength") if isinstance(target_annotation, dict) else 0.0),
    )
    context_type = _b_line_video_context_type(reference_annotation, target_annotation)
    visual_similarity = _score_float(candidate_quality.get("visual_context_similarity"))
    text = _b_line_video_context_text(reference_annotation) + " " + _b_line_video_context_text(target_annotation)
    evidence_bonus = 0.0
    if context_type in {
        "news/reporting",
        "sports_commentary",
        "tutorial_instruction",
        "interview_context",
        "livestream_context",
        "performance_or_singing",
    }:
        evidence_bonus += 0.35
    if len(_tokenize_text(text)) >= 8:
        evidence_bonus += 0.20
    if any(term in text for term in ("field", "kitchen", "stadium", "podium", "studio", "stage", "outdoor", "classroom")):
        evidence_bonus += 0.15
    if visual_similarity > 0:
        evidence_bonus += min(0.25, visual_similarity * 0.25)
    return round(min(1.0, max(provided, evidence_bonus)), 3)


def _b_line_asr_degeneracy_risk(
    reference_annotation: dict[str, Any] | None,
    target_annotation: dict[str, Any] | None,
    candidate_quality: dict[str, Any] | None = None,
) -> float:
    candidate_quality = candidate_quality if isinstance(candidate_quality, dict) else {}
    provided = max(
        _score_float(candidate_quality.get("asr_degeneracy_risk")),
        _score_float((reference_annotation or {}).get("asr_degeneracy_risk") if isinstance(reference_annotation, dict) else 0.0),
        _score_float((target_annotation or {}).get("asr_degeneracy_risk") if isinstance(target_annotation, dict) else 0.0),
    )
    context_type = _b_line_video_context_type(reference_annotation, target_annotation)
    text = _b_line_video_context_text(reference_annotation) + " " + _b_line_video_context_text(target_annotation)
    risk = provided
    if context_type in {"asr_only", "generic_talking_head", "unknown"}:
        risk = max(risk, 0.62 if context_type != "unknown" else 0.56)
    if any(term in text for term in ("black screen", "static image", "podcast", "audio only", "meeting", "webinar", "zoom")):
        risk = max(risk, 0.78)
    if any(term in text for term in ("news", "sport", "match", "tutorial", "cook", "repair", "interview", "performance", "livestream")):
        risk = min(risk or 0.45, 0.45)
    return round(min(1.0, max(0.0, risk)), 3)


def _b_line_subtype_from_evidence(
    *,
    difference_type: str,
    edit_text: str,
    reference_annotation: dict[str, Any] | None,
    target_annotation: dict[str, Any] | None,
) -> str:
    text = _normalized_phrase(
        " ".join(
            [
                difference_type,
                edit_text,
                " ".join(_normalize_list((reference_annotation or {}).get("audio_events", [])) if isinstance(reference_annotation, dict) else []),
                " ".join(_normalize_list((target_annotation or {}).get("audio_events", [])) if isinstance(target_annotation, dict) else []),
                " ".join(_normalize_list((reference_annotation or {}).get("speech", [])) if isinstance(reference_annotation, dict) else []),
                " ".join(_normalize_list((target_annotation or {}).get("speech", [])) if isinstance(target_annotation, dict) else []),
            ]
        )
    )
    if any(term in text for term in ("music", "song", "singing", "guitar", "piano", "lyric", "melody")):
        return "music"
    if difference_type == "speech":
        return "speech_topic_in_video_context"
    return "sound_event"


def _single_source_model_reject_issues(
    model_fields: dict[str, Any],
    edit_text_quality: dict[str, Any],
    *,
    audio_dataset_line: str = STANDARD_AUDIO_DATASET_LINE,
) -> list[str]:
    audio_dataset_line = _normalize_audio_dataset_line(audio_dataset_line)
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    edit_text = str(model_fields.get("edit_text", "")).strip()
    reasons: list[str] = []
    disabled_difference_types = set(FINAL_DISABLED_DIFFERENCE_TYPES)
    if audio_dataset_line == SPEECH_AUDIO_CONTENT_LINE:
        disabled_difference_types.discard("speech")
    if difference_type in disabled_difference_types:
        reasons.append(f"{difference_type} is diagnostic-only for single-source accepted pairs")
    if difference_type == "attribute":
        normalized_edit = _normalized_phrase(edit_text)
        if _difference_values_are_too_similar(from_value, to_value):
            reasons.append("weak_attribute_wording: from/to values are too similar")
        if any(
            phrase in normalized_edit
            for phrase in (
                "blouse to shirt",
                "shirt to blouse",
                "dark blue blouse to dark blue shirt",
                "long brown hair to long hair",
            )
        ):
            reasons.append("weak_attribute_wording: clothing or hair wording change is not a meaningful single-source edit")
    if _score_float(edit_text_quality.get("score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "edit_text_quality_score"):
        reasons.append("bad_edit_text_quality")
    return _dedupe_strings(reasons)


def _b_line_edit_text_audio_only_issues(edit_text: str, difference_type: str) -> list[str]:
    normalized = _normalized_phrase(edit_text)
    if not normalized:
        return ["edit_text_not_audio_only: empty edit_text"]
    issues: list[str] = []
    issues.extend(_b_line_edit_text_specificity_issues(edit_text, difference_type))
    if not any(term in normalized for term in B_LINE_AUDIO_EDIT_TERMS):
        issues.append("edit_text_not_audio_only: missing speech/audio wording")
    visual_terms = [term.strip() for term in B_LINE_VISUAL_EDIT_TERMS if term in normalized]
    if visual_terms:
        issues.append(f"edit_text_not_audio_only: visual wording {visual_terms[0]}")
    if difference_type == "speech" and not any(
        term in normalized
        for term in (
            "speech",
            "spoken",
            "says",
            "say",
            "talk",
            "talking",
            "discuss",
            "discussing",
            "commentary",
            "commentator",
            "narration",
            "voice",
            "words",
            "transcript",
        )
    ):
        issues.append("edit_text_not_audio_only: speech edit lacks speech-content wording")
    if difference_type == "audio_event" and not any(
        term in normalized
        for term in (
            "audio",
            "sound",
            "music",
            "song",
            "cheer",
            "cheering",
            "applause",
            "ambient",
            "ambience",
            "crowd",
        )
    ):
        issues.append("edit_text_not_audio_only: audio_event edit lacks concrete sound wording")
    return _dedupe_strings(issues)


def _b_line_edit_text_specificity_issues(edit_text: str, difference_type: str) -> list[str]:
    normalized = _normalized_phrase(edit_text)
    issues: list[str] = []
    generic_phrases = (
        "speech content has been altered",
        "audio content differs",
        "audio content has been altered",
        "the audio content differs",
        "target audio",
        "reference audio",
        "add target audio to the audio",
        "replace reference audio",
    )
    placeholder_patterns = (
        r"\bfrom discussing a to discussing b\b",
        r"\bfrom topic a to topic b\b",
        r"\bfrom content a to content b\b",
        r"\bfrom phrase a to phrase b\b",
        r"\bfrom saying a to saying b\b",
        r"\bspecific topic a\b",
        r"\bspecific topic b\b",
        r"\bspecific sound a\b",
        r"\bspecific sound b\b",
    )
    hollow_markers = (
        "unintelligible",
        "inaudible",
        "not transcribed",
        "not clearly transcribed",
        "not clear enough",
        "not clearly discernible",
        "not clearly intelligible",
        "content is not clear",
        "content not clear",
        "content is unspecified",
        "specific content is not",
        "speech is present but",
        "unknown",
        "unspecified",
    )
    if any(phrase in normalized for phrase in generic_phrases):
        issues.append("edit_text_not_audio_only: generic audio placeholder")
    if any(re.search(pattern, normalized) for pattern in placeholder_patterns):
        issues.append("edit_text_not_audio_only: placeholder audio wording")
    if any(marker in normalized for marker in hollow_markers):
        issues.append("edit_text_not_audio_only: hollow audio wording")
    clauses = [clause.strip() for clause in re.split(r"[;\n]+", edit_text) if clause.strip()]
    for clause in clauses[1:]:
        clause_norm = _normalized_phrase(clause)
        has_audio_word = any(term in clause_norm for term in B_LINE_AUDIO_EDIT_TERMS)
        has_visual_word = any(term.strip() in clause_norm for term in B_LINE_VISUAL_EDIT_TERMS)
        if has_visual_word and not has_audio_word:
            issues.append("edit_text_not_audio_only: visual clause in audio edit")
            break
    if difference_type == "speech" and normalized in {"speech", "speaking", "audio", "sound", "voice"}:
        issues.append("edit_text_not_audio_only: hollow speech edit")
    return _dedupe_strings(issues)


def _single_source_audio_line_acceptance_issues(
    *,
    model_fields: dict[str, Any],
    audio_dataset_line: str,
    candidate_quality: dict[str, Any] | None,
    reference_annotation: dict[str, Any] | None,
    target_annotation: dict[str, Any] | None,
) -> list[str]:
    line = _normalize_audio_dataset_line(audio_dataset_line)
    if line == STANDARD_AUDIO_DATASET_LINE:
        return []
    candidate_quality = candidate_quality if isinstance(candidate_quality, dict) else {}
    profile = str(candidate_quality.get("audio_line_quality_profile", "")).strip()
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    modalities = {str(item).strip().lower() for item in _normalize_list(model_fields.get("modalities", []))}
    edit_text = _normalized_phrase(str(model_fields.get("edit_text", "")))
    issues: list[str] = []
    audio_words = ("audio", "sound", "speech", "music", "transcript", "narration", "voice")
    if line == VISUAL_AUDIO_ANCHOR_LINE:
        if difference_type not in DOMINANT_VISUAL_DIFFERENCE_TYPES:
            issues.append(f"visual_audio_anchor requires visual difference type, got {difference_type or 'missing'}")
        if "audio" in modalities:
            issues.append("visual_audio_anchor edit must not require audio modality")
        if any(word in edit_text for word in audio_words):
            issues.append("visual_audio_anchor edit_text mentions audio/speech terms")
        if profile == AUDIO_LINE_QUALITY_PROFILE_V4_STRICT:
            visual_delta_strength = _score_float(candidate_quality.get("visual_delta_strength"))
            if difference_type not in V4_A_STRONG_VISUAL_TYPES:
                issues.append(f"visual_too_similar_for_A: {difference_type or 'missing'} is not a large visual delta type")
            if visual_delta_strength < 0.45:
                issues.append(f"visual_too_similar_for_A: visual_delta_strength {visual_delta_strength:.2f} < 0.45")
    elif line == SPEECH_AUDIO_CONTENT_LINE:
        if difference_type not in {"speech", "audio_event"}:
            issues.append(f"speech_audio_content requires speech or audio_event difference type, got {difference_type or 'missing'}")
        if "audio" not in modalities:
            issues.append("speech_audio_content edit must include audio modality")
        issues.extend(_b_line_edit_text_audio_only_issues(edit_text, difference_type))
        endpoint_issue = _b_line_difference_endpoint_issue(difference, difference_type)
        if endpoint_issue:
            issues.append(endpoint_issue)
        if difference_type == "speech":
            if not reference_annotation or not target_annotation or not _speech_is_transcript_backed(reference_annotation, target_annotation):
                issues.append("speech_audio_content speech edit lacks transcript-backed evidence")
        if difference_type == "audio_event":
            score = (
                _non_speech_audio_event_score(reference_annotation, target_annotation)
                if reference_annotation and target_annotation
                else 0.0
            )
            if score < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "non_speech_audio_event_score"):
                issues.append("speech_audio_content audio_event edit lacks non-speech audio evidence")
        if profile == AUDIO_LINE_QUALITY_PROFILE_V4_STRICT:
            visual_delta_strength = _score_float(candidate_quality.get("visual_delta_strength"))
            visual_context_similarity = _score_float(candidate_quality.get("visual_context_similarity"))
            if visual_delta_strength > V4_B_MAX_VISUAL_DELTA_STRENGTH:
                issues.append(
                    "visual_too_different_for_B: "
                    f"visual_delta_strength {visual_delta_strength:.2f} > {V4_B_MAX_VISUAL_DELTA_STRENGTH:.2f}"
                )
            if visual_context_similarity < V4_B_MIN_VISUAL_CONTEXT_SIMILARITY:
                issues.append(
                    "visual_too_different_for_B: "
                    f"visual_context_similarity {visual_context_similarity:.2f} < {V4_B_MIN_VISUAL_CONTEXT_SIMILARITY:.2f}"
                )
            if difference_type == "audio_event":
                evidence_text = _normalized_phrase(
                    " ".join(
                        [
                            str(difference.get("from", "")),
                            str(difference.get("to", "")),
                            str(difference.get("description", "")),
                            " ".join(_normalize_list(reference_annotation.get("audio_events", []))) if reference_annotation else "",
                            " ".join(_normalize_list(target_annotation.get("audio_events", []))) if target_annotation else "",
                            " ".join(model_fields.get("evidence", [])) if isinstance(model_fields.get("evidence"), list) else "",
                        ]
                    )
                )
                has_concrete_audio = any(term in evidence_text for term in V4_CONCRETE_AUDIO_TERMS)
                has_vague_audio = any(term in evidence_text for term in V4_VAGUE_AUDIO_TERMS)
                if not has_concrete_audio:
                    issues.append("audio_not_primary: missing concrete audio event evidence")
                if has_vague_audio and not has_concrete_audio:
                    issues.append("vague_audio_event: vague hum/click/tone without explicit evidence")
        if _is_b_audio_context_cvr_profile(str(candidate_quality.get("acceptance_profile", ""))):
            video_context_strength = _score_float(candidate_quality.get("video_context_strength"))
            asr_degeneracy_risk = _score_float(candidate_quality.get("asr_degeneracy_risk"))
            context_type = str(candidate_quality.get("video_context_type", "")).strip()
            if video_context_strength < _profile_threshold(B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE, "video_context_strength"):
                issues.append(
                    "video_context_too_weak_for_B: "
                    f"video_context_strength {video_context_strength:.2f} < 0.45"
                )
            if asr_degeneracy_risk > _profile_threshold(B_AUDIO_CONTEXT_CVR_ACCEPTANCE_PROFILE, "asr_degeneracy_risk"):
                issues.append(
                    "asr_degeneracy_risk_too_high: "
                    f"asr_degeneracy_risk {asr_degeneracy_risk:.2f} > 0.55"
                )
            if context_type in {"asr_only", "generic_talking_head"}:
                issues.append(f"asr_degeneracy_risk_too_high: context_type={context_type}")
            datasets = {
                str((reference_annotation or {}).get("dataset", "")).strip().lower() if isinstance(reference_annotation, dict) else "",
                str((target_annotation or {}).get("dataset", "")).strip().lower() if isinstance(target_annotation, dict) else "",
            }
            if any("ami" in dataset for dataset in datasets if dataset):
                issues.append("diagnostic_asr_auxiliary_source: AMI-AV is not accepted into main B-line")
    return _dedupe_strings(issues)


def _b_line_difference_endpoint_issue(difference: dict[str, Any], difference_type: str) -> str:
    if difference_type not in {"speech", "audio_event"}:
        return ""
    from_value = _clean_b_line_audio_phrase(str(difference.get("from", "")).strip(), difference_type=difference_type)
    to_value = _clean_b_line_audio_phrase(str(difference.get("to", "")).strip(), difference_type=difference_type)
    if from_value and to_value and _normalized_phrase(from_value) == _normalized_phrase(to_value):
        return "edit_text_not_audio_only: identical audio endpoints"
    hollow_markers = (
        "not transcribed",
        "not described",
        "unclear",
        "unknown",
        "speaking",
        "speech",
        "audio",
        "sound",
    )
    if difference_type == "speech" and to_value and _normalized_phrase(to_value) in hollow_markers:
        return "edit_text_not_audio_only: hollow speech target"
    return ""


def _single_source_model_reject_reason(
    model_fields: dict[str, Any],
    edit_text_quality: dict[str, Any],
) -> str:
    return "; ".join(_single_source_model_reject_issues(model_fields, edit_text_quality))


def _single_source_recommended_edit_text(model_fields: dict[str, Any]) -> str:
    edit_text = str(model_fields.get("edit_text", "")).strip()
    normalized = _normalized_phrase(edit_text)
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if "product close up" in normalized or "product closeup" in normalized or "close up" in normalized:
        if "static" in _normalized_phrase(to_value) or "product" in _normalized_phrase(to_value):
            return "add a static product image overlay on the left"
        return "change the composition to show a product image overlay beside the speaker"
    if "full screen" in normalized or "fullscreen" in normalized:
        return "change the picture-in-picture demonstration to a static product image overlay"
    if "man speaking" in normalized and "inset" not in normalized and "picture in picture" not in normalized:
        return edit_text.replace("man speaking", "inset video showing a man speaking")
    if from_value and to_value and not edit_text:
        return f"change {from_value} to {to_value}"
    return edit_text


def _repair_single_source_audio_line_model_fields(
    *,
    model_fields: dict[str, Any],
    audio_dataset_line: str,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    if _normalize_audio_dataset_line(audio_dataset_line) != SPEECH_AUDIO_CONTENT_LINE:
        return model_fields
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    if difference_type not in {"speech", "audio_event"}:
        return model_fields

    repaired = dict(model_fields)
    repaired["modalities"] = _ensure_audio_modality(repaired.get("modalities"))
    candidate_edit = _b_line_audio_only_edit_text_from_difference(
        difference=difference,
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    )
    if not candidate_edit:
        return repaired
    current_issues = _b_line_edit_text_audio_only_issues(str(repaired.get("edit_text", "")), difference_type)
    candidate_issues = _b_line_edit_text_audio_only_issues(candidate_edit, difference_type)
    current_quality = _edit_text_quality_payload(
        edit_text=str(repaired.get("edit_text", "")),
        difference=difference,
        modalities=repaired.get("modalities", []),
        reference_caption=str(repaired.get("reference_caption", "")),
        target_caption=str(repaired.get("target_caption", "")),
    )
    candidate_quality = _edit_text_quality_payload(
        edit_text=candidate_edit,
        difference=difference,
        modalities=repaired.get("modalities", []),
        reference_caption=str(reference_annotation.get("summary", "")),
        target_caption=str(target_annotation.get("summary", "")),
    )
    if _b_line_difference_endpoint_issue(difference, difference_type):
        return repaired
    should_replace = bool(current_issues and not candidate_issues) or (
        _score_float(candidate_quality.get("score")) > _score_float(current_quality.get("score"))
        and not candidate_issues
    )
    if should_replace:
        repaired["edit_text"] = candidate_edit
        repaired["b_line_edit_text_repaired"] = True
        repaired["b_line_original_edit_text"] = str(model_fields.get("edit_text", "")).strip()
    return repaired


def _ensure_audio_modality(value: Any) -> list[str]:
    modalities = [str(item).strip() for item in _normalize_list(value) if str(item).strip()]
    if "audio" not in {item.lower() for item in modalities}:
        modalities.append("audio")
    return modalities


def _b_line_audio_only_edit_text_from_difference(
    *,
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> str:
    difference_type = str(difference.get("type", "")).strip()
    from_value = _clean_b_line_audio_phrase(str(difference.get("from", "")).strip(), difference_type=difference_type)
    to_value = _clean_b_line_audio_phrase(str(difference.get("to", "")).strip(), difference_type=difference_type)
    if difference_type == "speech":
        if not from_value:
            from_value = _clean_b_line_audio_phrase(_first_speech_phrase(reference_annotation), difference_type=difference_type)
        if not to_value:
            to_value = _clean_b_line_audio_phrase(_first_speech_phrase(target_annotation), difference_type=difference_type)
        if _b_line_audio_phrase_is_hollow(from_value) or _b_line_audio_phrase_is_hollow(to_value):
            return ""
        if from_value and to_value:
            return f"change the speech from {from_value} to {to_value}"
        if to_value:
            return f"change the speech to {to_value}"
    if difference_type == "audio_event":
        if _b_line_audio_phrase_is_hollow(from_value) or _b_line_audio_phrase_is_hollow(to_value):
            return ""
        if from_value and to_value:
            return f"replace {from_value} in the audio with {to_value}"
        if to_value:
            return f"add {to_value} to the audio"
        if from_value:
            return f"remove {from_value} from the audio"
    return ""


def _b_line_audio_phrase_is_hollow(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    hollow_exact = {
        "a",
        "b",
        "speech",
        "speaking",
        "discussing speaking",
        "discussing a",
        "discussing b",
        "topic a",
        "topic b",
        "content a",
        "content b",
        "audio",
        "sound",
        "voice",
        "unknown",
        "unspecified",
        "target audio",
        "reference audio",
    }
    if normalized in hollow_exact:
        return True
    return bool(
        any(
            marker in normalized
            for marker in (
                "unintelligible",
                "inaudible",
                "not transcribed",
                "not clearly",
                "not clear",
                "not discernible",
                "not intelligible",
                "content is not",
                "content not",
                "specific content is not",
            )
        )
    )


def _clean_b_line_audio_phrase(value: str, *, difference_type: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\b(?:a|an|the)\s+(?:man|woman|person|speaker|presenter|commentator|narrator)\s+(?:is\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:man|woman|person|speaker|presenter|commentator|narrator)\s+(?:is\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:talking|speaking)\s+about\b", "discussing", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:talks|speaks)\s+about\b", "discussing", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:saying|says|said)\b", "saying", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:in|on)\s+(?:the\s+)?(?:shot|scene|camera|view|frame|background|foreground)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:shot|scene|camera|view|frame|background|foreground|visual|subtitle|logo|text)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-")
    if difference_type == "speech" and text and not any(term in _normalized_phrase(text) for term in ("discuss", "discussing", "say", "saying", "speech", "spoken", "commentary", "narration", "voice")):
        text = f"discussing {text}"
    return text


def _first_speech_phrase(annotation: dict[str, Any]) -> str:
    for key in ("speakers_and_transcript", "speech"):
        values = _normalize_list(annotation.get(key, []))
        for value in values:
            text = str(value).strip()
            if text:
                return text
    return ""


def _single_source_text_driven_product_change_issue(
    *,
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any] | None,
    target_annotation: dict[str, Any] | None,
) -> str:
    if not reference_annotation or not target_annotation:
        return ""
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    if difference_type not in {"object_presence", "object_count", "attribute", "scene"}:
        return ""
    reference_text = _visible_text_values(reference_annotation)
    target_text = _visible_text_values(target_annotation)
    if not reference_text or not target_text or not _strong_visible_text_delta(reference_annotation, target_annotation):
        return ""

    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    edit_text = str(model_fields.get("edit_text", "")).strip()
    description = str(difference.get("description", "")).strip()
    dominant_delta = model_fields.get("dominant_delta") if isinstance(model_fields.get("dominant_delta"), dict) else {}
    combined = _normalized_phrase(
        " ".join(
            [
                edit_text,
                description,
                from_value,
                to_value,
                str(dominant_delta.get("from", "")),
                str(dominant_delta.get("to", "")),
                str(dominant_delta.get("reason", "")),
            ]
        )
    )
    product_overlay_terms = (
        "product image",
        "product overlay",
        "static image",
        "brand",
        "label",
        "skincare",
        "professional",
        "pen",
        "roller",
    )
    if not any(term in combined for term in product_overlay_terms):
        return ""

    reference_overlap = _visible_text_token_overlap_ratio(from_value, reference_text)
    target_overlap = _visible_text_token_overlap_ratio(to_value, target_text)
    if reference_overlap >= 0.50 and target_overlap >= 0.50:
        return "text_driven_product_overlay_change: product from/to values are primarily OCR or packaging text"
    if "visible text" in combined or "text on" in combined or "label" in combined:
        return "text_driven_product_overlay_change: product change is described through on-screen text"
    return ""


def _visible_text_token_overlap_ratio(value: str, visible_text_values: list[str]) -> float:
    value_tokens = {
        token
        for token in _tokenize_text(_normalized_phrase(value))
        if len(token) >= 3 and token not in {"the", "and", "with", "from", "into", "image", "static", "product"}
    }
    if not value_tokens:
        return 0.0
    text_tokens = {
        token
        for token in _tokenize_text(_normalized_phrase(" ".join(visible_text_values)))
        if len(token) >= 3
    }
    if not text_tokens:
        return 0.0
    return len(value_tokens & text_tokens) / max(1, len(value_tokens))


def _single_source_delta_family_from_fields(model_fields: dict[str, Any]) -> str:
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    text = _normalized_phrase(
        " ".join(
            [
                str(model_fields.get("edit_text", "")),
                str(difference.get("type", "")),
                str(difference.get("from", "")),
                str(difference.get("to", "")),
                str(difference.get("description", "")),
                str(model_fields.get("dominant_delta", {}).get("from", "")) if isinstance(model_fields.get("dominant_delta"), dict) else "",
                str(model_fields.get("dominant_delta", {}).get("to", "")) if isinstance(model_fields.get("dominant_delta"), dict) else "",
            ]
        )
    )
    if any(marker in text for marker in ("picture in picture", "pip", "inset")):
        if text.startswith("add ") or "no overlay" in text or "no picture in picture" in text:
            return "add_pip_demo"
        if "static product" in text or "product image" in text or ("product" in text and "overlay" in text):
            return "pip_demo_to_product_overlay"
        if "brow" in text or "eyebrow" in text or "treatment" in text:
            return "pip_subject_change"
        if "man" in text:
            return "add_pip_inset_man"
        return "pip_overlay_change"
    if "product" in text or "brow lift" in text or "revlon" in text:
        if "no product" in text or text.startswith("add "):
            return "add_product_overlay"
        if "close up" in text or "closeup" in text:
            return "product_closeup_claim"
        return "product_overlay_change"
    difference_type = str(difference.get("type", "pair")).strip() or "pair"
    return f"{difference_type}:{_stable_hash(text)[:8]}"


def _apply_single_source_delta_uniqueness(
    records: list[dict[str, Any]],
    *,
    max_accepted_pairs: int,
    acceptance_profile: str,
) -> None:
    if not records or not any(bool(record.get("single_source_pair")) for record in records):
        return
    # `max_accepted_pairs` is kept for API compatibility, but single-source
    # production now treats all final-Omni-passed pairs as dataset candidates.
    # We still remove duplicate delta families; we no longer demote clean pairs
    # just because a per-source cap was reached.
    uniqueness_issue_prefixes = ("duplicate_delta_family", "single_source_accept_cap_exceeded")
    eligible: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("single_source_pair")):
            continue
        issues = [
            str(issue).strip()
            for issue in record.get("single_source_pair_acceptance_issues", [])
            if str(issue).strip() and not str(issue).strip().startswith(uniqueness_issue_prefixes)
        ]
        record["single_source_pair_acceptance_issues"] = issues
        base_accepted = bool(record.get("model_accepted", record.get("accepted")))
        if not base_accepted or issues:
            _set_single_source_record_acceptance(record, accepted=False, extra_issues=issues)
            continue
        family = str(record.get("single_source_delta_family", "")).strip()
        if not family:
            family = _single_source_delta_family_from_record(record)
            record["single_source_delta_family"] = family
        eligible.append(record)

    selected_families: set[str] = set()
    for record in sorted(eligible, key=_accepted_record_sort_key):
        family = str(record.get("single_source_delta_family", "")).strip()
        if family and family in selected_families:
            _set_single_source_record_acceptance(record, accepted=False, extra_issues=[f"duplicate_delta_family:{family}"])
            continue
        if family:
            selected_families.add(family)
        _set_single_source_record_acceptance(record, accepted=True, extra_issues=[])


def _single_source_delta_family_from_record(record: dict[str, Any]) -> str:
    model_fields = {
        "edit_text": record.get("edit_text", ""),
        "difference": record.get("difference", {}),
        "dominant_delta": record.get("dominant_delta", {}),
    }
    return _single_source_delta_family_from_fields(model_fields)


def _set_single_source_record_acceptance(
    record: dict[str, Any],
    *,
    accepted: bool,
    extra_issues: list[str],
) -> None:
    issues = _dedupe_strings(
        [
            str(issue).strip()
            for issue in record.get("single_source_pair_acceptance_issues", [])
            if str(issue).strip()
        ]
        + [str(issue).strip() for issue in extra_issues if str(issue).strip()]
    )
    record["single_source_pair_acceptance_issues"] = issues
    record["accepted"] = bool(accepted and not issues)
    reason = "; ".join(issues)
    if not record["accepted"] and not reason:
        reason = "single-source pair rejected"
    judge = record.get("judge") if isinstance(record.get("judge"), dict) else {}
    judge["accept"] = record["accepted"]
    judge["target_satisfies_edit"] = record["accepted"]
    judge["single_main_difference"] = record["accepted"]
    judge["reject_reason"] = "" if record["accepted"] else _append_reason(judge.get("reject_reason", ""), reason)
    record["judge"] = judge

    observable = record.get("observable_difference") if isinstance(record.get("observable_difference"), dict) else {}
    observable["passed"] = record["accepted"]
    observable["failure_reason"] = "" if record["accepted"] else _append_reason(observable.get("failure_reason", ""), reason)
    record["observable_difference"] = observable

    verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
    if not record["accepted"]:
        verification["passed"] = False
        verification["failures"] = _dedupe_strings(_normalize_list(verification.get("failures", [])) + issues)
        edit_text_quality_check = verification.get("edit_text_quality_check") if isinstance(verification.get("edit_text_quality_check"), dict) else {}
        edit_text_quality_check["single_primary_difference"] = False
        edit_text_quality_check["target_satisfies"] = False
        edit_text_quality_check["failure_reason"] = _append_reason(edit_text_quality_check.get("failure_reason", ""), reason)
        verification["edit_text_quality_check"] = edit_text_quality_check
    record["verification"] = verification


def _single_source_pair_verification(
    model_fields: dict[str, Any],
    *,
    accepted: bool,
    reject_reason: str,
) -> dict[str, Any]:
    evidence = list(model_fields.get("evidence", [])) if isinstance(model_fields.get("evidence"), list) else []
    confidence = _score_float(model_fields.get("confidence"))
    difference = model_fields.get("difference") if isinstance(model_fields.get("difference"), dict) else {}
    reason = "; ".join(evidence) if evidence else reject_reason
    return _finalize_pair_verification(
        {
            "caption_delta": {
                "caption_equivalent": not accepted,
                "has_concrete_difference": accepted,
                "difference_matches_edit": accepted,
                "concrete_differences": [str(difference.get("description", "")).strip()] if str(difference.get("description", "")).strip() else [],
                "reason": reason,
            },
            "edit_projection": {
                "projected_target_caption": str(model_fields.get("target_caption", "")).strip(),
                "target_matches_projection": accepted,
                "score": confidence if accepted else 0.0,
                "missing_requirements": [] if accepted else [reject_reason or "single-source pair rejected"],
                "reason": reason,
            },
            "edit_necessity": {
                "edit_needed": accepted,
                "reference_satisfies_edit": False,
                "target_satisfies_edit": accepted,
                "score": confidence if accepted else 0.0,
                "reason": reason,
            },
            "edit_text_quality_check": {
                "not_caption_like": bool(str(model_fields.get("edit_text", "")).strip()),
                "matches_modality": accepted,
                "single_primary_difference": accepted,
                "reference_does_not_satisfy": True,
                "target_satisfies": accepted,
                "score": confidence if accepted else 0.0,
                "failure_reason": "" if accepted else reject_reason,
            },
        }
    )


def _short_difference_value(value: str, *, max_tokens: int = 8) -> str:
    tokens = [token for token in re.split(r"\s+", value.strip()) if token]
    return " ".join(tokens[:max_tokens]) or "segment"


def _build_single_source_pair_report(
    *,
    output_path: Path,
    group: dict[str, Any],
    annotations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    fallback_candidate_count: int,
    acceptance_profile: str,
) -> str:
    difference_counts = Counter(str(item.get("difference", {}).get("type", "")) for item in candidates)
    risk_counts = _candidate_risk_flag_counts(candidates)
    expected_pair_count = len(annotations) * (len(annotations) - 1) // 2
    lines = [
        "# Single Source Pair Mining Report",
        "",
        f"- Output: `{output_path}`",
        f"- Group: `{group.get('group_id', '')}`",
        f"- Segments: `{len(annotations)}`",
        f"- Expected pairs n*(n-1)/2: `{expected_pair_count}`",
        f"- Mined pairs: `{len(candidates)}`",
        f"- Fallback heuristic pairs: `{fallback_candidate_count}`",
        f"- Acceptance profile: `{acceptance_profile}`",
        "",
        "## Difference Type Counts",
    ]
    for key, value in sorted(difference_counts.items()):
        lines.append(f"- `{key or 'unknown'}`: `{value}`")
    if not difference_counts:
        lines.append("- none")
    lines.extend(["", "## Risk Flag Counts"])
    for key, value in sorted(risk_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    if not risk_counts:
        lines.append("- none")
    lines.extend(["", "## Segment Order"])
    for annotation in annotations:
        lines.append(
            f"- `{annotation.get('clip_id', '')}` "
            f"{_clip_start_seconds(annotation):.3f}s: {str(annotation.get('summary', '')).strip()}"
        )
    lines.extend(["", "## Candidate Pairs"])
    for candidate in candidates:
        difference = candidate.get("difference", {})
        lines.append(
            "- "
            f"`{candidate.get('candidate_id', '')}` "
            f"{candidate.get('reference_start_seconds', 0.0):.3f}s -> "
            f"{candidate.get('target_start_seconds', 0.0):.3f}s "
            f"`{difference.get('type', 'unknown')}` "
            f"`{difference.get('from', '')}` -> `{difference.get('to', '')}`"
        )
    if not candidates:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_segments(
    *,
    duration_seconds: float,
    segment_seconds: float,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[tuple[float, float]]:
    if duration_seconds < min_clip_seconds:
        return []
    if duration_seconds <= max_clip_seconds:
        return [(0.0, duration_seconds)]
    segment_length = min(max(segment_seconds, min_clip_seconds), max_clip_seconds)
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + segment_length, duration_seconds)
        if end - start >= min_clip_seconds:
            segments.append((start, end))
        elif segments:
            previous_start, _previous_end = segments[-1]
            segments[-1] = (previous_start, duration_seconds)
        start += segment_length
    return segments


def _group_tags_from_clip(item: dict[str, Any]) -> list[str]:
    tokens = _group_tokens_from_clip(item)
    return sorted(tokens)[:8]


def _group_tokens_from_clip(item: dict[str, Any]) -> set[str]:
    tokens = set()
    tokens.update(_text_field_tokens(item.get("text_fields", {})))
    tokens.update(_tokenize_text(str(item.get("dataset", ""))))
    tokens.update(_tokenize_text(str(item.get("clip_id", ""))))
    return tokens


def _semantic_singleton_groups(items: list[dict[str, Any]], *, group_size: int = 8) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_dataset.setdefault(str(item.get("dataset", "unknown")), []).append(item)

    groups: list[dict[str, Any]] = []
    for dataset, dataset_items in sorted(by_dataset.items()):
        dataset_items.sort(key=lambda item: (item.get("tokens", []), item["clip_id"]))
        for group_index, start in enumerate(range(0, len(dataset_items), group_size), start=1):
            chunk = dataset_items[start : start + group_size]
            if len(chunk) < 2:
                continue
            clip_ids = [str(item["clip_id"]) for item in chunk]
            token_counter: Counter[str] = Counter()
            for item in chunk:
                token_counter.update(item.get("tokens", []))
            group_tags = [token for token, _count in token_counter.most_common(8)]
            groups.append(
                {
                    "group_id": f"group_{dataset}_semantic_{group_index:03d}",
                    "dataset": dataset,
                    "group_reason": "semantic_cluster",
                    "source_clip_ids": [str(item.get("source_clip_id", "")) for item in chunk],
                    "candidate_clip_ids": clip_ids,
                    "group_tags": group_tags,
                }
            )
    return groups


def build_ffmpeg_extract_command(
    *,
    source_path: str | Path,
    output_path: str | Path,
    start_seconds: float,
    end_seconds: float,
    overwrite: bool,
) -> list[str]:
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-ss",
        _format_seconds(start_seconds),
        "-to",
        _format_seconds(end_seconds),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _build_asset_id(dataset_name: str, relative_path: str) -> str:
    stem = Path(relative_path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "video"
    slug = slug[:32]
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}__{slug}__{digest}"


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _stable_json_hash(value: Any) -> str:
    return _stable_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _load_video_edit_planner_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        key = str(payload.get("cache_key", "")).strip()
        if key:
            cache[key] = payload
    return cache


def _video_edit_planner_cache_key(
    *,
    model: str | None,
    planning_mode: str,
    route: str,
    reference_video: str,
    reference_annotation: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    payload = {
        "model": model or "",
        "planning_mode": planning_mode,
        "route": route,
        "reference_video": reference_video,
        "reference_annotation": _annotation_prompt_view(reference_annotation),
        "candidate": candidate,
    }
    return _stable_json_hash(payload)


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")[:80] or "clip"


def _build_gallery_id(video_path: str) -> str:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    return f"gallery__{digest}"


def _build_proposal_id(reference_path: str, target_path: str) -> str:
    digest = hashlib.sha1(f"{reference_path}::{target_path}".encode("utf-8")).hexdigest()[:16]
    return f"proposal__{digest}"


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _merge_gallery_entry(*, accumulator: dict[str, dict[str, Any]], video_path: str, sample_id: str, role: str) -> None:
    entry = accumulator.setdefault(video_path, {"sample_ids": set(), "roles": set()})
    if sample_id:
        entry["sample_ids"].add(sample_id)
    entry["roles"].add(role)


def _build_raw_summary_report(output_path: Path, dataset_counts: dict[str, int]) -> str:
    lines = [
        "# Raw Asset Index Summary",
        "",
        f"- Index: `{output_path}`",
        f"- Total assets: `{sum(dataset_counts.values())}`",
        "",
        "| Dataset | Video Count |",
        "|---|---:|",
    ]
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"| `{dataset}` | `{count}` |")
    return "\n".join(lines) + "\n"


def _build_pilot_report(summary: dict[str, Any]) -> str:
    acceptance = summary["automated_acceptance"]
    lines = [
        "# Pilot Review Summary",
        "",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Gallery count: `{summary['gallery_count']}`",
        "",
        "## Modality Counts",
    ]
    for key, value in summary["modality_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary["modality_counts"]:
        lines.append("- none")

    lines.extend(["", "## Difference Type Counts"])
    for key, value in summary["difference_type_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary["difference_type_counts"]:
        lines.append("- none")

    lines.extend(["", "## Source Type Counts"])
    for key, value in summary.get("source_type_counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary.get("source_type_counts"):
        lines.append("- none")

    lines.extend(["", "## Source Type Difference Counts"])
    for key, value in summary.get("source_type_difference_counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary.get("source_type_difference_counts"):
        lines.append("- none")

    lines.extend(["", "## Source Context Counts"])
    for key, value in summary["source_context_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not summary["source_context_counts"]:
        lines.append("- none")

    quality = summary["quality_summary"]
    lines.extend(
        [
            "",
            "## Quality Summary",
            f"- `same_context_min`: `{quality['same_context_min']}`",
            f"- `same_context_avg`: `{quality['same_context_avg']}`",
            f"- `same_context_max`: `{quality['same_context_max']}`",
        ]
    )
    strength = summary.get("difference_strength_summary", {})
    if strength:
        lines.extend(
            [
                "",
                "## Difference Strength Summary",
                f"- `difference_strength_min`: `{strength.get('difference_strength_min', 0.0)}`",
                f"- `difference_strength_avg`: `{strength.get('difference_strength_avg', 0.0)}`",
                f"- `difference_strength_max`: `{strength.get('difference_strength_max', 0.0)}`",
            ]
        )

    speech_audio_counts = summary.get("speech_audio_quality_counts", {})
    if speech_audio_counts:
        lines.extend(["", "## Speech / Audio Quality Counts"])
        for key in (
            "speech_count",
            "high_quality_speech_count",
            "transcript_backed_speech_count",
            "non_speech_audio_event_count",
            "speech_rejected_as_too_generic_count",
            "audio_event_rejected_as_speech_only_count",
        ):
            lines.append(f"- `{key}`: `{speech_audio_counts.get(key, 0)}`")

    lines.extend(["", "## Automated Acceptance Checks"])
    for key, value in acceptance.items():
        lines.append(f"- `{key}`: `{'PASS' if value else 'FAIL'}`")
    verification_counts = summary.get("verification_counts", {})
    if verification_counts:
        lines.extend(["", "## Synthetic Route Counts"])
        for key in (
            "synthetic_visual_count",
            "synthetic_audio_count",
            "deterministic_audio_count",
            "foleycrafter_audio_count",
            "frieren_audio_count",
            "audio_remux_count",
            "speech_content_reject_count",
            "audio_stream_missing_reject_count",
            "visual_changed_in_audio_sample_reject_count",
            "audio_event_not_detected_reject_count",
        ):
            lines.append(f"- `{key}`: `{verification_counts.get(key, 0)}`")
        lines.extend(["", "## Edit Text / Difference Gate Counts"])
        for key in (
            "good_edit_text_count",
            "bad_edit_text_rejected_count",
            "caption_like_edit_rejected_count",
            "modality_leakage_rejected_count",
            "near_duplicate_without_delta_rejected_count",
            "visual_presence_contradiction_reject_count",
            "visible_text_without_ocr_reject_count",
            "audio_event_without_independent_audio_evidence_reject_count",
            "competing_difference_reject_count",
            "duplicate_target_reject_count",
        ):
            lines.append(f"- `{key}`: `{verification_counts.get(key, 0)}`")
        lines.extend(["", "## Verification Reject Counts"])
        for key in (
            "verification_passed_count",
            "verification_passed_rejected_count",
            "verification_override_accept_count",
            "caption_equivalent_reject_count",
            "missing_delta_reject_count",
            "difference_mismatch_reject_count",
            "edit_projection_reject_count",
            "edit_not_needed_reject_count",
            "speech_rejected_as_too_generic_count",
            "speech_rejected_for_missing_transcript_count",
            "audio_event_rejected_as_speech_only_count",
            "accepted_after_verification_count",
        ):
            lines.append(f"- `{key}`: `{verification_counts.get(key, 0)}`")
    lines.append("")
    lines.append("Manual review is still required for semantic correctness and target uniqueness.")
    return "\n".join(lines) + "\n"


def _validate_pilot_record(root: Path, record: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    sample_id = str(record.get("sample_id", "")).strip()
    if not sample_id:
        errors.append(f"pilot line {line_number}: sample_id is required")

    reference_video = str(record.get("reference_video", "")).strip()
    target_video = str(record.get("target_video", "")).strip()
    edit_text = str(record.get("edit_text", "")).strip()
    reference_caption = str(record.get("reference_caption", "")).strip()
    target_caption = str(record.get("target_caption", "")).strip()
    source_type = str(record.get("source_type", "natural")).strip() or "natural"
    if source_type not in ALLOWED_SOURCE_TYPES:
        errors.append(f"pilot line {line_number}: unsupported source_type={source_type!r}")
    if source_type == "synthetic_edit":
        errors.extend(f"pilot line {line_number}: {issue}" for issue in _known_pair_generation_issues(record))

    for field_name, value in (
        ("reference_video", reference_video),
        ("target_video", target_video),
        ("edit_text", edit_text),
        ("reference_caption", reference_caption),
        ("target_caption", target_caption),
    ):
        if not value:
            errors.append(f"pilot line {line_number}: {field_name} is required")

    if reference_video and target_video and reference_video == target_video:
        errors.append(f"pilot line {line_number}: reference_video and target_video must differ")

    for field_name, raw_value in (("reference_video", reference_video), ("target_video", target_video)):
        if raw_value:
            resolved = _resolve_under_root(root, raw_value)
            if not resolved.exists():
                errors.append(f"pilot line {line_number}: {field_name} does not exist: {raw_value}")

    modalities = record.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        errors.append(f"pilot line {line_number}: modalities must be a non-empty list")
    else:
        invalid_modalities = sorted({str(item).strip() for item in modalities} - ALLOWED_MODALITIES)
        if invalid_modalities:
            errors.append(f"pilot line {line_number}: invalid modalities={invalid_modalities}")

    difference = record.get("difference")
    if not isinstance(difference, dict):
        errors.append(f"pilot line {line_number}: difference must be an object")
    else:
        difference_type = str(difference.get("type", "")).strip()
        if difference_type not in ALLOWED_DIFFERENCE_TYPES:
            errors.append(f"pilot line {line_number}: unsupported difference.type={difference_type!r}")
        if difference_type in FINAL_DISABLED_DIFFERENCE_TYPES:
            errors.append(
                f"pilot line {line_number}: {difference_type} difference type is disabled for final Omni-CVR samples"
            )
        if not any(str(difference.get(key, "")).strip() for key in ("from", "to", "description")):
            errors.append(f"pilot line {line_number}: difference must include from/to/description")

    hard_negatives = record.get("hard_negatives")
    if not isinstance(hard_negatives, list) or not hard_negatives:
        errors.append(f"pilot line {line_number}: hard_negatives must be a non-empty list")
    else:
        normalized_negatives = [str(item).strip() for item in hard_negatives if str(item).strip()]
        if len(normalized_negatives) != len(hard_negatives):
            errors.append(f"pilot line {line_number}: hard_negatives must only contain non-empty strings")
        if reference_video and reference_video in normalized_negatives:
            errors.append(f"pilot line {line_number}: reference_video cannot appear in hard_negatives")
        if target_video and target_video in normalized_negatives:
            errors.append(f"pilot line {line_number}: target_video cannot appear in hard_negatives")
        for negative_path in normalized_negatives:
            resolved = _resolve_under_root(root, negative_path)
            if not resolved.exists():
                errors.append(f"pilot line {line_number}: hard_negative does not exist: {negative_path}")

    quality = record.get("quality")
    if not isinstance(quality, dict):
        errors.append(f"pilot line {line_number}: quality must be an object")
    else:
        for field_name in ("same_context_score", "edit_match_score", "target_uniqueness_score"):
            if field_name not in quality:
                errors.append(f"pilot line {line_number}: quality.{field_name} is required")
                continue
            try:
                float(quality[field_name])
            except (TypeError, ValueError):
                errors.append(f"pilot line {line_number}: quality.{field_name} must be numeric")

    source = record.get("source")
    if not isinstance(source, dict):
        errors.append(f"pilot line {line_number}: source must be an object")
    else:
        for field_name in ("platform", "url", "license_note"):
            if not str(source.get(field_name, "")).strip():
                errors.append(f"pilot line {line_number}: source.{field_name} is required")

    return errors


def _build_pair_candidates(*, root: Path, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [annotation for annotation in annotations if _annotation_has_signal(annotation)]
    eligible.sort(key=lambda annotation: (-_annotation_pairing_signal_score(annotation), str(annotation.get("clip_id", ""))))
    candidates: list[dict[str, Any]] = []
    comparison_count = 0
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            if comparison_count >= MAX_PAIR_LOCAL_COMPARISONS:
                break
            comparison_count += 1
            forward = _score_ordered_pair(
                root=root,
                reference_annotation=left,
                target_annotation=right,
                annotations=eligible,
                compute_visual_near_duplicate=False,
            )
            backward = _score_ordered_pair(
                root=root,
                reference_annotation=right,
                target_annotation=left,
                annotations=eligible,
                compute_visual_near_duplicate=False,
            )
            chosen = _select_better_pair(forward, backward)
            if chosen is not None:
                candidates.append(chosen)
        if comparison_count >= MAX_PAIR_LOCAL_COMPARISONS:
            break
    candidates.sort(key=lambda item: (-item["composite_score"], item["proposal_id"]))
    return _select_diverse_pair_candidates(candidates, max_candidates=MAX_PAIR_CANDIDATES)


def _annotation_pairing_signal_score(annotation: dict[str, Any]) -> float:
    score = 0.0
    if _non_speech_audio_terms(annotation):
        score += 5.0
    if _speech_texts_from_annotation(annotation):
        score += 1.0
    if _visible_text_values(annotation):
        score += 1.0
    score += min(3.0, len(_normalize_object_counts(annotation.get("object_counts", {}))) * 0.5)
    score += min(2.0, len(_action_terms_from_annotation(annotation)) * 0.4)
    score += min(2.0, len(_normalize_list(annotation.get("attributes", []))) * 0.25)
    score += min(1.0, len(_normalize_list(annotation.get("storyline", []))) * 0.2)
    score += _clean_stability_score(annotation) * 2.0
    score += min(1.5, len(_annotation_subject_signature_bundle(annotation)) * 0.5)
    return score


def _select_diverse_pair_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for difference_type, target_count in DIVERSE_PAIR_BUCKET_TARGETS.items():
        bucket_count = 0
        for candidate in candidates:
            if len(selected) >= max_candidates or bucket_count >= target_count:
                break
            if candidate["proposal_id"] in selected_ids:
                continue
            if candidate["primary_difference"]["type"] != difference_type:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["proposal_id"])
            bucket_count += 1

    for difference_type, target_count in DIVERSE_PAIR_BUCKET_TARGETS.items():
        bucket_count = sum(
            1 for candidate in selected if candidate["primary_difference"]["type"] == difference_type
        )
        for candidate in candidates:
            if len(selected) >= max_candidates or bucket_count >= target_count:
                break
            if candidate["proposal_id"] in selected_ids:
                continue
            if difference_type not in candidate.get("changed_difference_types", []):
                continue
            retargeted = _retarget_pair_candidate(candidate, difference_type)
            if retargeted is None:
                continue
            selected.append(retargeted)
            selected_ids.add(retargeted["proposal_id"])
            bucket_count += 1

    for candidate in candidates:
        if len(selected) >= max_candidates:
            break
        if candidate["proposal_id"] in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["proposal_id"])

    return selected


def _retarget_pair_candidate(candidate: dict[str, Any], difference_type: str) -> dict[str, Any] | None:
    if candidate["primary_difference"]["type"] == difference_type:
        return candidate

    reference_annotation = candidate["reference_annotation"]
    target_annotation = candidate["target_annotation"]
    primary_difference = _dominant_visual_difference_from_annotations(
        reference_annotation,
        target_annotation,
        difference_type=difference_type,
    )
    if primary_difference is None or primary_difference["type"] != difference_type:
        return None

    changed_types = primary_difference.pop("changed_types")
    same_context_score = _score_float(candidate["quality"].get("same_context_score"))
    edit_match_score = _edit_match_score(
        same_context_score=same_context_score,
        primary_difference_type=difference_type,
        changed_types=changed_types,
    )
    if edit_match_score < MIN_PAIR_EDIT_MATCH_SCORE:
        return None

    retargeted = dict(candidate)
    source_context = dict(candidate.get("source_context", {}))
    retargeted["primary_difference"] = primary_difference
    retargeted["changed_difference_types"] = list(changed_types)
    quality = dict(candidate["quality"])
    quality["edit_match_score"] = round(edit_match_score, 3)
    quality["difference_strength_score"] = round(
        _difference_strength_score(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=primary_difference,
            changed_types=changed_types,
        ),
        3,
    )
    quality["difference_type"] = primary_difference["type"]
    if primary_difference["type"] == "action":
        quality["action_evidence_score"] = _action_evidence_score(reference_annotation, target_annotation)
    if primary_difference["type"] == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
        quality["has_audio_modality"] = 1.0
    if primary_difference["type"] == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(
            reference_annotation,
            target_annotation,
        )
        quality["has_audio_modality"] = 1.0
    dominant_delta_decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=primary_difference,
        quality=quality,
        source_context=source_context,
    )
    quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
    quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
    quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
    quality["dominant_delta_decision"] = dominant_delta_decision
    retargeted["quality"] = quality
    retargeted["source_context"] = source_context
    retargeted["dominant_delta_decision"] = dominant_delta_decision
    retargeted["composite_score"] = _candidate_composite_score(quality, source_context)
    retargeted["difference_evidence"] = _difference_evidence_from_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference=primary_difference,
    )
    return retargeted


def _dominant_visual_difference_from_annotations(
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    *,
    difference_type: str,
) -> dict[str, Any] | None:
    if difference_type == "attribute" and _is_talking_head_template(reference_annotation) and _is_talking_head_template(target_annotation):
        reference_signature = _annotation_subject_signature_bundle(reference_annotation)
        target_signature = _annotation_subject_signature_bundle(target_annotation)
        if reference_signature and target_signature and reference_signature != target_signature:
            return {
                "type": "attribute",
                "from": f"speaker with {', '.join(reference_signature[:4])}",
                "to": f"speaker with {', '.join(target_signature[:4])}",
                "description": "the speaker's visual signature changes while the presentation template stays similar",
                "changed_types": ["attribute"],
            }

    priority_order = (difference_type,) + tuple(item for item in PAIR_PRIORITY if item != difference_type)
    return _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=priority_order,
    )


def _retarget_audio_secondary_candidate_to_dominant_visual(candidate: dict[str, Any]) -> dict[str, Any] | None:
    difference = candidate.get("primary_difference", {})
    if not isinstance(difference, dict) or str(difference.get("type", "")).strip() != "audio_event":
        return candidate

    quality = dict(candidate.get("quality", {})) if isinstance(candidate.get("quality"), dict) else {}
    source_context = dict(candidate.get("source_context", {})) if isinstance(candidate.get("source_context"), dict) else {}
    decision = candidate.get("dominant_delta_decision") if isinstance(candidate.get("dominant_delta_decision"), dict) else {}
    if not decision:
        decision = _dominant_delta_decision(
            reference_annotation=candidate["reference_annotation"],
            target_annotation=candidate["target_annotation"],
            difference=difference,
            quality=quality,
            source_context=source_context,
        )
    if bool(decision.get("audio_primary_allowed")):
        return candidate

    dominant_type = str(decision.get("dominant_type", "")).strip()
    if dominant_type not in DOMINANT_VISUAL_DIFFERENCE_TYPES:
        return None

    retargeted = _retarget_pair_candidate(candidate, dominant_type)
    if retargeted is None:
        return None

    retargeted_quality = dict(retargeted.get("quality", {}))
    retargeted_quality["retargeted_from_audio_secondary"] = 1.0
    retargeted_quality["retargeted_from_difference_type"] = "audio_event"
    retargeted_quality["exploration_warnings"] = _dedupe_strings(
        _normalize_list(retargeted_quality.get("exploration_warnings", []))
        + ["retargeted_from_audio_secondary"]
        + list(decision.get("failure_flags", []))
    )
    retargeted_source_context = dict(retargeted.get("source_context", {}))
    retargeted_source_context["retargeted_from_difference_type"] = "audio_event"
    retargeted_source_context["retarget_reason"] = str(decision.get("reason", "")).strip()
    retargeted["quality"] = retargeted_quality
    retargeted["source_context"] = retargeted_source_context
    retargeted["composite_score"] = _candidate_composite_score(retargeted_quality, retargeted_source_context)

    mined_candidate = dict(retargeted.get("mined_candidate", {})) if isinstance(retargeted.get("mined_candidate"), dict) else {}
    if mined_candidate:
        mined_candidate["retargeted_from_difference_type"] = "audio_event"
        retargeted["mined_candidate"] = mined_candidate
    return retargeted


def _candidate_pre_propose_reject_reasons(
    candidate: dict[str, Any],
    *,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> list[str]:
    if not _is_exploration_profile(acceptance_profile):
        return []

    difference = candidate.get("primary_difference", {})
    difference_type = str(difference.get("type", "")).strip() if isinstance(difference, dict) else ""
    reference_annotation = candidate.get("reference_annotation", {})
    target_annotation = candidate.get("target_annotation", {})
    reasons: list[str] = []
    if difference_type in FINAL_DISABLED_DIFFERENCE_TYPES:
        reasons.append(f"disabled_primary_{difference_type}")
    if difference_type not in {"visible_text", "speech"}:
        if _strong_visible_text_delta(reference_annotation, target_annotation):
            reasons.append("competing_disabled_visible_text")
        if _strong_speech_delta(reference_annotation, target_annotation):
            reasons.append("competing_disabled_speech")
    return _dedupe_strings(reasons)


def _maybe_reorient_candidate_for_model_fields(
    *,
    root: Path,
    candidate: dict[str, Any],
    model_fields: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    difference = model_fields.get("difference", {})
    if not isinstance(difference, dict):
        return candidate, model_fields, False
    if not _model_difference_prefers_reverse_direction(
        difference=difference,
        reference_annotation=candidate["reference_annotation"],
        target_annotation=candidate["target_annotation"],
    ):
        return candidate, model_fields, False

    swapped = _score_ordered_pair(
        root=root,
        reference_annotation=candidate["target_annotation"],
        target_annotation=candidate["reference_annotation"],
        annotations=annotations,
    )
    if swapped is None:
        return candidate, model_fields, False
    difference_type = str(difference.get("type", "")).strip()
    if swapped["primary_difference"]["type"] != difference_type and difference_type in swapped.get("changed_difference_types", []):
        retargeted = _retarget_pair_candidate(swapped, difference_type)
        if retargeted is not None:
            swapped = retargeted
    if swapped["primary_difference"]["type"] != difference_type:
        return candidate, model_fields, False

    oriented_fields = dict(model_fields)
    oriented_fields["reference_caption"] = str(swapped["reference_annotation"].get("summary", "")).strip() or str(
        model_fields.get("target_caption", "")
    ).strip()
    oriented_fields["target_caption"] = str(swapped["target_annotation"].get("summary", "")).strip() or str(
        model_fields.get("reference_caption", "")
    ).strip()
    reason = str(oriented_fields.get("proposal_reason", "")).strip()
    correction_reason = "direction corrected because difference.from/to matched target-to-reference evidence"
    oriented_fields["proposal_reason"] = f"{reason} {correction_reason}".strip()
    return swapped, oriented_fields, True


def _model_difference_prefers_reverse_direction(
    *,
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> bool:
    forward_score = _difference_direction_alignment_score(difference, reference_annotation, target_annotation)
    reverse_score = _difference_direction_alignment_score(difference, target_annotation, reference_annotation)
    return reverse_score >= 0.72 and reverse_score >= forward_score + 0.20


def _difference_direction_alignment_score(
    difference: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> float:
    difference_type = str(difference.get("type", "")).strip()
    if not difference_type:
        return 0.0
    priority_order = (difference_type,) + tuple(item for item in PAIR_PRIORITY if item != difference_type)
    detected = _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=priority_order,
    )
    if not detected or detected.get("type") != difference_type:
        return 0.0
    from_score = _difference_value_similarity(
        str(difference.get("from", "")),
        str(detected.get("from", "")),
    )
    to_score = _difference_value_similarity(
        str(difference.get("to", "")),
        str(detected.get("to", "")),
    )
    return round((from_score + to_score) / 2.0, 3)


def _difference_value_similarity(left: str, right: str) -> float:
    left_norm = _normalized_phrase(left)
    right_norm = _normalized_phrase(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.95
    left_absent = left_norm.startswith("no ") or left_norm in {"none", "no distinctive audio event"}
    right_absent = right_norm.startswith("no ") or right_norm in {"none", "no distinctive audio event"}
    if left_absent != right_absent:
        return 0.0
    if left_absent and right_absent:
        return 1.0
    left_tokens = _tokenize_text(_strip_presence_prefix(left_norm))
    right_tokens = _tokenize_text(_strip_presence_prefix(right_norm))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def _difference_values_are_too_similar(left: str, right: str, *, threshold: float = 0.85) -> bool:
    left_absent = _absence_like_phrase(left) or _is_audio_absence_edit_phrase(left)
    right_absent = _absence_like_phrase(right) or _is_audio_absence_edit_phrase(right)
    if left_absent != right_absent:
        return False
    left_norm = _normalized_phrase(_strip_presence_prefix(left))
    right_norm = _normalized_phrase(_strip_presence_prefix(right))
    if not left_norm or not right_norm:
        return False
    return _difference_value_similarity(left_norm, right_norm) >= threshold


def _visible_text_fragment_edit(difference: dict[str, Any]) -> bool:
    from_norm = _normalized_phrase(str(difference.get("from", "")))
    to_norm = _normalized_phrase(str(difference.get("to", "")))
    if not from_norm or not to_norm or from_norm == to_norm:
        return bool(from_norm and to_norm and from_norm == to_norm)
    from_tokens = _tokenize_text(from_norm)
    to_tokens = _tokenize_text(to_norm)
    if (
        not from_tokens
        or not to_tokens
        or len(from_tokens) < VISIBLE_TEXT_FRAGMENT_MIN_SOURCE_TOKENS
        or len(to_tokens) >= len(from_tokens)
    ):
        return False
    target_is_source_subspan = to_norm in from_norm
    target_tokens_are_subset = to_tokens <= from_tokens
    target_ratio = len(to_tokens) / max(1, len(from_tokens))
    return bool(
        (target_is_source_subspan or target_tokens_are_subset)
        and target_ratio <= VISIBLE_TEXT_FRAGMENT_MAX_TARGET_TOKEN_RATIO
    )


def _retarget_audio_secondary_model_fields(
    *,
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    source_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    difference = model_fields.get("difference", {})
    if not isinstance(difference, dict) or str(difference.get("type", "")).strip() != "audio_event":
        return model_fields

    source_context = source_context if isinstance(source_context, dict) else {}
    quality_seed = {
        "same_context_score": _score_float(source_context.get("score")) or _same_context_score(reference_annotation, target_annotation),
        "template_compatibility_score": _score_float(source_context.get("template_compatibility_score")),
    }
    decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=difference,
        quality=quality_seed,
        source_context=source_context,
    )
    if decision["audio_primary_allowed"]:
        return model_fields

    dominant_type = str(decision.get("dominant_type", "")).strip()
    if dominant_type not in DOMINANT_VISUAL_DIFFERENCE_TYPES:
        return model_fields

    detected = _dominant_visual_difference_from_annotations(
        reference_annotation,
        target_annotation,
        difference_type=dominant_type,
    )
    if detected is None or detected.get("type") != dominant_type:
        return model_fields

    changed_types = list(detected.pop("changed_types"))
    repaired = dict(model_fields)
    repaired["difference"] = detected
    repaired["modalities"] = _infer_pair_modalities(reference_annotation, target_annotation, dominant_type)
    repaired["edit_text"] = _build_fallback_edit_text(detected)
    reason = str(repaired.get("proposal_reason", "")).strip()
    repaired["proposal_reason"] = (
        f"{reason} retargeted from secondary audio_event to dominant {dominant_type} "
        "because stronger visual deltas define the pair"
    ).strip()
    repaired["retargeted_from_difference_type"] = "audio_event"
    repaired["retargeted_changed_difference_types"] = changed_types
    return repaired


def _repair_pair_model_fields(
    *,
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    source_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repaired = dict(model_fields)
    if str(repaired.get("difference", {}).get("type", "")).strip() == "audio_event":
        repaired = _normalize_audio_event_model_fields(repaired)
        repaired = _retarget_audio_secondary_model_fields(
            model_fields=repaired,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            source_context=source_context,
        )
    current_quality = _edit_text_quality_payload(
        edit_text=str(repaired.get("edit_text", "")),
        difference=repaired.get("difference", {}),
        modalities=repaired.get("modalities", []),
        reference_caption=str(repaired.get("reference_caption", "")),
        target_caption=str(repaired.get("target_caption", "")),
    )
    if _edit_text_quality_passes(current_quality):
        return repaired

    template_edit = _build_fallback_edit_text(repaired.get("difference", {}))
    template_quality = _edit_text_quality_payload(
        edit_text=template_edit,
        difference=repaired.get("difference", {}),
        modalities=repaired.get("modalities", []),
        reference_caption=str(reference_annotation.get("summary", "")),
        target_caption=str(target_annotation.get("summary", "")),
    )
    if _edit_text_quality_passes(template_quality):
        repaired["edit_text"] = template_edit
        reason = str(repaired.get("proposal_reason", "")).strip()
        repaired["proposal_reason"] = f"{reason} edit_text normalized from evidence template".strip()
    return repaired


def _edit_text_quality_payload(
    *,
    edit_text: str,
    difference: dict[str, Any],
    modalities: list[str] | tuple[str, ...] | Any,
    reference_caption: str,
    target_caption: str,
) -> dict[str, Any]:
    text = str(edit_text).strip()
    tokens = _tokenize_text(text)
    difference_type = str(difference.get("type", "")).strip()
    modality_set = {str(item).strip() for item in modalities if str(item).strip()} if isinstance(modalities, (list, tuple, set)) else set()
    bad_patterns: list[str] = []

    if not text:
        bad_patterns.append("edit_text is empty")
    if any(phrase in _normalized_phrase(text) for phrase in GENERIC_EDIT_TEXT_PHRASES):
        bad_patterns.append("edit_text is too broad")

    first_token = _normalized_phrase(text).split()[0] if _normalized_phrase(text).split() else ""
    is_imperative_edit = first_token in EDIT_TEXT_START_VERBS or first_token in EDIT_ACTION_VERBS
    if not is_imperative_edit:
        bad_patterns.append("edit_text is not an imperative edit")

    matches_difference_type = _edit_text_matches_difference_type(
        edit_text=text,
        difference=difference,
        modalities=modality_set,
    )
    if not matches_difference_type:
        bad_patterns.append(f"edit_text does not match difference type {difference_type or 'unknown'}")

    single_change = _edit_text_single_change(text, difference_type)
    if not single_change:
        bad_patterns.append("edit_text appears to contain multiple unrelated changes")

    not_caption_like = _edit_text_not_caption_like(
        edit_text=text,
        reference_caption=reference_caption,
        target_caption=target_caption,
    )
    if not not_caption_like:
        bad_patterns.append("edit_text reads like a caption instead of an edit instruction")

    no_modality_leakage = _edit_text_no_modality_leakage(text, modalities, difference_type)
    if not no_modality_leakage:
        bad_patterns.append("edit_text mentions a modality outside the declared difference")

    malformed_presence = _edit_text_has_malformed_presence(text)
    if malformed_presence:
        bad_patterns.append("edit_text uses malformed or vague edit wording")

    score = 1.0
    for failed, penalty in (
        (not bool(text), 0.50),
        (not is_imperative_edit, 0.30),
        (not matches_difference_type, 0.35),
        (not single_change, 0.25),
        (not not_caption_like, 0.35),
        (not no_modality_leakage, 0.35),
        (malformed_presence, 0.35),
        (any(phrase in _normalized_phrase(text) for phrase in GENERIC_EDIT_TEXT_PHRASES), 0.30),
    ):
        if failed:
            score -= penalty
    score = round(max(0.0, min(1.0, score)), 3)
    return {
        "score": score,
        "is_imperative_edit": is_imperative_edit,
        "matches_difference_type": matches_difference_type,
        "single_change": single_change,
        "not_caption_like": not_caption_like,
        "no_modality_leakage": no_modality_leakage,
        "bad_patterns": bad_patterns,
    }


def _edit_text_quality_passes(payload: dict[str, Any]) -> bool:
    return bool(
        _score_float(payload.get("score")) >= MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE
        and bool(payload.get("is_imperative_edit"))
        and bool(payload.get("matches_difference_type"))
        and bool(payload.get("single_change"))
        and bool(payload.get("not_caption_like"))
        and bool(payload.get("no_modality_leakage"))
        and not payload.get("bad_patterns")
    )


def _edit_text_quality_has_bad_imperative(payload: dict[str, Any]) -> bool:
    bad_patterns = " ".join(str(item) for item in payload.get("bad_patterns", []))
    return bool(
        not payload.get("is_imperative_edit")
        or "too broad" in bad_patterns
        or "malformed" in bad_patterns
        or "empty" in bad_patterns
    )


def _edit_text_matches_difference_type(
    *,
    edit_text: str,
    difference: dict[str, Any],
    modalities: set[str],
) -> bool:
    tokens = _tokenize_text(edit_text)
    difference_type = str(difference.get("type", "")).strip()
    from_tokens = _tokenize_text(str(difference.get("from", "")))
    to_tokens = _tokenize_text(str(difference.get("to", "")))
    delta_tokens = from_tokens | to_tokens
    if not difference_type:
        return bool(tokens & EDIT_ACTION_VERBS)
    if difference_type in {"object_count", "object_presence"}:
        leaked_modality_tokens = {"audio", "sound", "sounds", "speech", "transcript", "spoken", "voiceover", "narration"}
        return bool(tokens & delta_tokens) and not bool(tokens & leaked_modality_tokens)
    if difference_type == "action":
        return bool({"action", "gesture", "doing"} & tokens or tokens & delta_tokens or tokens & EDIT_ACTION_VERBS)
    if difference_type == "audio_event":
        return bool("audio" in modalities and tokens & EDIT_TEXT_AUDIO_TOKENS) and not (
            _is_speech_only_or_absence_audio_phrase(edit_text) or bool(tokens & EDIT_TEXT_VISUAL_LEAK_TOKENS)
        )
    if difference_type == "speech":
        return bool("audio" in modalities and tokens & EDIT_TEXT_SPEECH_TOKENS)
    if difference_type == "visible_text":
        return bool(tokens & EDIT_TEXT_VISIBLE_TEXT_TOKENS)
    if difference_type in {"attribute", "scene"}:
        return bool(tokens & EDIT_ACTION_VERBS or tokens & delta_tokens)
    return bool(tokens & EDIT_ACTION_VERBS)


def _edit_text_single_change(edit_text: str, difference_type: str) -> bool:
    normalized = _normalized_phrase(edit_text)
    if not normalized:
        return False
    if len(normalized.split()) > 32:
        return False
    multi_markers = ("and also", "as well as", " plus ")
    if any(marker in normalized for marker in multi_markers):
        return False
    tokens = _tokenize_text(edit_text)
    modality_hits = 0
    if tokens & EDIT_TEXT_AUDIO_TOKENS:
        modality_hits += 1
    if tokens & EDIT_TEXT_SPEECH_TOKENS:
        modality_hits += 1
    if tokens & EDIT_TEXT_VISIBLE_TEXT_TOKENS:
        modality_hits += 1
    return not (difference_type not in {"integrated", "speech"} and modality_hits > 1)


def _edit_text_not_caption_like(*, edit_text: str, reference_caption: str, target_caption: str) -> bool:
    text = edit_text.strip()
    if not text:
        return False
    text_tokens = _tokenize_text(text)
    if len(text_tokens) > EDIT_TEXT_CAPTION_MAX_TOKENS:
        return False
    normalized_text = _normalized_phrase(text)
    if len(text_tokens) <= 8:
        return True
    for caption in (reference_caption, target_caption):
        normalized_caption = _normalized_phrase(caption)
        caption_tokens = _tokenize_text(caption)
        if not caption_tokens:
            continue
        if normalized_text and normalized_text in normalized_caption:
            return False
        if _jaccard(text_tokens, caption_tokens) >= 0.72:
            return False
    return True


def _edit_text_no_modality_leakage(
    edit_text: str,
    modalities: list[str] | tuple[str, ...] | Any,
    difference_type: str,
) -> bool:
    modality_set = {str(item).strip() for item in modalities if str(item).strip()} if isinstance(modalities, (list, tuple, set)) else set()
    tokens = _tokenize_text(edit_text)
    if difference_type == "audio_event":
        if "audio" not in modality_set:
            return False
        if tokens & EDIT_TEXT_VISUAL_LEAK_TOKENS:
            return False
        if _is_speech_only_or_absence_audio_phrase(edit_text):
            return False
        return True
    if difference_type == "speech":
        return "audio" in modality_set and not bool(tokens & (NON_SPEECH_AUDIO_TOKENS - {"voice"}))
    if difference_type == "visible_text":
        return not bool((tokens & EDIT_TEXT_AUDIO_TOKENS) or (tokens & GENERIC_SPEECH_TOKENS))
    if difference_type in VISUAL_DIFFERENCE_TYPES and tokens & {"audio", "sound", "sounds", "speech", "transcript", "spoken"}:
        return False
    return True


def _edit_text_has_malformed_presence(edit_text: str) -> bool:
    normalized = _normalized_phrase(edit_text)
    if normalized.startswith("change no ") and " into " in normalized and re.search(r"\b\d+\b", normalized):
        return True
    tokens = normalized.split()
    return bool(tokens[:2] == ["make", "the"] and len(tokens) <= 3)


def _observable_difference_gate(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
    visual_near_duplicate_score: Any,
) -> dict[str, Any]:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    near_duplicate_score = _score_float(visual_near_duplicate_score)
    if near_duplicate_score >= MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE:
        near_duplicate_risk = "high"
    elif near_duplicate_score >= 0.97:
        near_duplicate_risk = "medium"
    else:
        near_duplicate_risk = "low"

    if difference_type not in VISUAL_DIFFERENCE_TYPES:
        return {
            "passed": True,
            "type": difference_type,
            "frame_backed": False,
            "reference_missing": [],
            "target_present": [],
            "reference_value": from_value,
            "target_value": to_value,
            "supporting_fields": ["non_visual_difference_type"],
            "near_duplicate_risk": near_duplicate_risk,
            "visual_near_duplicate_score": near_duplicate_score,
            "failure_reason": "",
        }

    supporting_fields: list[str] = []
    reference_missing: list[str] = []
    target_present: list[str] = []
    reference_evidence: list[str] = []
    target_evidence: list[str] = []
    reference_counts = _normalize_object_counts(reference_annotation.get("object_counts", {}))
    target_counts = _normalize_object_counts(target_annotation.get("object_counts", {}))
    reference_actions = _normalize_list(reference_annotation.get("actions", []))
    target_actions = _normalize_list(target_annotation.get("actions", []))
    reference_text = _annotation_observable_text(reference_annotation)
    target_text = _annotation_observable_text(target_annotation)
    conflict_reasons: list[str] = []

    if difference_type in {"object_count", "object_presence"}:
        label = _strip_presence_prefix(to_value) or _strip_presence_prefix(from_value)
        canonical_label = _canonical_object_label(label)
        reference_mentions_label = _annotation_mentions_presence_label(reference_annotation, label)
        target_mentions_label = _annotation_mentions_presence_label(target_annotation, label)
        reference_label_count = _object_count_for_label(reference_counts, label)
        target_label_count = _object_count_for_label(target_counts, label)
        if label and reference_label_count != target_label_count:
            supporting_fields.append("object_counts")
            reference_evidence.append(f"object_counts:{reference_label_count}")
            target_evidence.append(f"object_counts:{target_label_count}")
        if label and not reference_mentions_label and target_mentions_label:
            reference_missing.append(label)
            target_present.append(label)
            supporting_fields.append("summary")
        if label and _presence_value_claims_absent(from_value) and reference_mentions_label:
            conflict_reasons.append(f"reference already appears to contain equivalent object: {label}")
        if label and _presence_value_claims_absent(to_value) and target_mentions_label:
            conflict_reasons.append(f"target still appears to contain {label}")
        if (
            canonical_label in BACKGROUND_DECOR_OBJECTS
            and target_mentions_label
            and not _annotation_has_label_frame_evidence(target_annotation, label)
        ):
            conflict_reasons.append(f"background decor object lacks frame-level evidence: {label}")
    elif difference_type == "action":
        if _first_unique(reference_actions, target_actions) or _first_unique(target_actions, reference_actions):
            supporting_fields.append("actions")
            reference_evidence.append(_first_unique(reference_actions, target_actions))
            target_evidence.append(_first_unique(target_actions, reference_actions))
        if from_value and _text_mentions_phrase(reference_text, from_value):
            supporting_fields.append("storyline")
        if to_value and _text_mentions_phrase(target_text, to_value):
            supporting_fields.append("events")
    elif difference_type == "visible_text":
        reference_visible_text = _visible_text_values(reference_annotation)
        target_visible_text = _visible_text_values(target_annotation)
        if reference_visible_text != target_visible_text:
            supporting_fields.append("visible_text")
        if from_value and not _text_collection_mentions_phrase(reference_visible_text, from_value):
            conflict_reasons.append(f"visible_text lacks reference OCR/frame evidence for {from_value}")
        if to_value and not _text_collection_mentions_phrase(target_visible_text, to_value):
            conflict_reasons.append(f"visible_text lacks target OCR/frame evidence for {to_value}")
        reference_evidence.extend(reference_visible_text)
        target_evidence.extend(target_visible_text)
    elif difference_type == "attribute":
        if _normalize_list(reference_annotation.get("attributes", [])) != _normalize_list(target_annotation.get("attributes", [])):
            supporting_fields.append("attributes")
        if to_value and _text_mentions_phrase(target_text, to_value):
            supporting_fields.append("summary")
    elif difference_type == "scene":
        if str(reference_annotation.get("scene", "")).strip() != str(target_annotation.get("scene", "")).strip():
            supporting_fields.append("scene")

    supporting_fields = _dedupe_strings(supporting_fields)
    competing_reasons = _competing_difference_reasons(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference_type=difference_type,
    )
    conflict_reasons.extend(competing_reasons)
    hard_fields = {"object_counts", "actions", "events", "visible_text", "attributes", "scene"}
    frame_backed = bool(set(supporting_fields) & hard_fields)
    passed = bool(supporting_fields)
    if conflict_reasons:
        passed = False
    if near_duplicate_risk == "high" and not bool(set(supporting_fields) & hard_fields):
        passed = False
    if passed:
        failure_reason = ""
    elif conflict_reasons:
        failure_reason = "; ".join(_dedupe_strings(conflict_reasons))
    else:
        failure_reason = "no observable annotation delta supports this visual edit"
    return {
        "passed": passed,
        "type": difference_type,
        "frame_backed": frame_backed,
        "reference_missing": _dedupe_strings(reference_missing),
        "target_present": _dedupe_strings(target_present),
        "reference_evidence": _dedupe_strings(reference_evidence),
        "target_evidence": _dedupe_strings(target_evidence),
        "reference_value": from_value,
        "target_value": to_value,
        "supporting_fields": supporting_fields,
        "near_duplicate_risk": near_duplicate_risk,
        "visual_near_duplicate_score": near_duplicate_score,
        "failure_reason": failure_reason,
    }


def _annotation_observable_text(annotation: dict[str, Any]) -> str:
    texts: list[str] = []
    for field in ("summary", "scene"):
        value = str(annotation.get(field, "")).strip()
        if value:
            texts.append(value)
    texts.extend(_normalize_list(annotation.get("storyline", [])))
    texts.extend(_normalize_list(annotation.get("visible_text", [])))
    for event in annotation.get("events", []):
        if isinstance(event, dict):
            texts.extend(_normalize_list([event.get("visual", "")]))
            texts.extend(_normalize_list(event.get("actions", [])))
            texts.extend(_normalize_list(event.get("objects", [])))
        else:
            texts.extend(_normalize_list([event]))
    return " ".join(texts)


def _visible_text_values(annotation: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("visible_text", "on_screen_text", "ocr_text"):
        values.extend(_normalize_list(annotation.get(field, [])))
    for event in annotation.get("events", []):
        if not isinstance(event, dict):
            continue
        values.extend(_normalize_list(event.get("visible_text", [])))
        values.extend(_normalize_list(event.get("on_screen_text", [])))
        values.extend(_normalize_list(event.get("ocr_text", [])))
    return _dedupe_strings(values)


def _text_collection_mentions_phrase(values: list[str], phrase: str) -> bool:
    if not phrase:
        return True
    return any(_text_mentions_phrase(value, phrase) for value in values)


def _competing_difference_reasons(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference_type: str,
) -> list[str]:
    reasons: list[str] = []
    if primary_difference_type != "action" and _strong_action_delta(reference_annotation, target_annotation):
        reasons.append("single_main_difference failed: competing stronger action difference")
    if primary_difference_type not in {"visible_text", "speech"} and _strong_visible_text_delta(reference_annotation, target_annotation):
        reasons.append("single_main_difference failed: competing stronger visible_text difference")
    if primary_difference_type not in {"speech", "visible_text"} and _strong_speech_delta(reference_annotation, target_annotation):
        reasons.append("single_main_difference failed: competing stronger speech difference")
    return reasons


def _strong_action_delta(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    reference_actions = _action_terms_from_annotation(reference_annotation)
    target_actions = _action_terms_from_annotation(target_annotation)
    if not reference_actions or not target_actions:
        return False
    if not _first_unique(reference_actions, target_actions) or not _first_unique(target_actions, reference_actions):
        return False
    return _list_delta_strength(reference_actions, target_actions) >= MIN_COMPETING_DIFFERENCE_STRENGTH


def _strong_visible_text_delta(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    reference_text = _visible_text_values(reference_annotation)
    target_text = _visible_text_values(target_annotation)
    if not reference_text or not target_text:
        return False
    if not _first_unique(reference_text, target_text) or not _first_unique(target_text, reference_text):
        return False
    return _list_delta_strength(reference_text, target_text) >= MIN_COMPETING_DIFFERENCE_STRENGTH


def _strong_speech_delta(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    reference_speech = _speech_texts_from_annotation(reference_annotation)
    target_speech = _speech_texts_from_annotation(target_annotation)
    if not reference_speech or not target_speech:
        return False
    if not _first_unique(reference_speech, target_speech) or not _first_unique(target_speech, reference_speech):
        return False
    return _list_delta_strength(reference_speech, target_speech) >= MIN_COMPETING_DIFFERENCE_STRENGTH


def _annotation_mentions_value(annotation: dict[str, Any], value: str) -> bool:
    if not value:
        return False
    texts = [
        _annotation_observable_text(annotation),
        " ".join(_normalize_object_counts(annotation.get("object_counts", {})).keys()),
        " ".join(_normalize_list(annotation.get("subjects", []))),
    ]
    return any(_text_mentions_phrase(text, value) for text in texts)


def _annotation_mentions_presence_label(annotation: dict[str, Any], label: str) -> bool:
    if not label:
        return False
    if _annotation_mentions_value(annotation, label):
        return True
    for alias in _object_label_aliases(label):
        if alias != _normalized_object_label(label) and _annotation_mentions_value(annotation, alias):
            return True
    label_tokens = _tokenize_text(label)
    if not label_tokens or not (label_tokens & GENERIC_HUMAN_GROUP_TOKENS):
        return False
    texts = [
        _annotation_observable_text(annotation),
        " ".join(_normalize_object_counts(annotation.get("object_counts", {})).keys()),
        " ".join(_normalize_list(annotation.get("subjects", []))),
    ]
    annotation_tokens = _tokenize_text(" ".join(texts))
    if not (annotation_tokens & GENERIC_HUMAN_GROUP_TOKENS):
        return False
    context_tokens = {
        token
        for token in label_tokens
        if token not in GENERIC_HUMAN_GROUP_TOKENS and not token.isdigit()
    }
    return not context_tokens or bool(context_tokens & annotation_tokens)


def _normalized_object_label(value: str) -> str:
    label = _strip_presence_prefix(value)
    normalized_tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(label.lower()):
        if token.isdigit() or token in OBJECT_LABEL_STOPWORDS:
            continue
        normalized_tokens.append(_singular_object_token(token))
    return " ".join(normalized_tokens)


def _singular_object_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _canonical_object_label(value: str) -> str:
    normalized = _normalized_object_label(value)
    if not normalized:
        return ""
    for alias_group in OBJECT_ALIAS_GROUPS:
        normalized_group = [_normalized_object_label(alias) for alias in alias_group]
        if normalized in normalized_group:
            return normalized_group[0]
    return normalized


def _object_label_aliases(label: str) -> list[str]:
    normalized = _normalized_object_label(label)
    canonical = _canonical_object_label(label)
    aliases = [normalized, canonical]
    for alias_group in OBJECT_ALIAS_GROUPS:
        normalized_group = [_normalized_object_label(alias) for alias in alias_group]
        if canonical in normalized_group or normalized in normalized_group:
            aliases.extend(normalized_group)
    return _dedupe_strings([alias for alias in aliases if alias])


def _annotation_has_label_frame_evidence(annotation: dict[str, Any], label: str) -> bool:
    aliases = _object_label_aliases(label)
    if not aliases:
        return False
    for container_name in ("events", "storyline"):
        container = annotation.get(container_name, [])
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict):
                values = [item.get("visual", ""), item.get("description", "")]
                values.extend(_normalize_list(item.get("objects", [])))
                values.extend(_normalize_list(item.get("actions", [])))
            else:
                values = [item]
            text = " ".join(str(value) for value in values)
            if any(_text_mentions_phrase(text, alias) for alias in aliases):
                return True
    return False


def _presence_value_claims_absent(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if normalized.startswith("no "):
        return True
    count = _first_integer(normalized)
    return count == 0


def _object_count_for_label(counts: dict[str, int], label: str) -> int:
    label_tokens = _tokenize_text(label)
    canonical_label = _canonical_object_label(label)
    for key, count in counts.items():
        if canonical_label and _canonical_object_label(key) == canonical_label:
            return count
        key_tokens = _tokenize_text(key)
        if label_tokens and key_tokens and (label_tokens <= key_tokens or key_tokens <= label_tokens):
            return count
    return 0


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)
    return result


def _apply_structured_gate_quality(
    quality: dict[str, Any],
    *,
    edit_text_quality: dict[str, Any],
    observable_difference: dict[str, Any],
) -> None:
    quality["edit_text_quality_score"] = _score_float(edit_text_quality.get("score"))
    quality["edit_text_is_imperative"] = 1.0 if edit_text_quality.get("is_imperative_edit") else 0.0
    quality["edit_text_matches_difference_type"] = 1.0 if edit_text_quality.get("matches_difference_type") else 0.0
    quality["edit_text_single_change"] = 1.0 if edit_text_quality.get("single_change") else 0.0
    quality["edit_text_not_caption_like"] = 1.0 if edit_text_quality.get("not_caption_like") else 0.0
    quality["edit_text_no_modality_leakage"] = 1.0 if edit_text_quality.get("no_modality_leakage") else 0.0
    quality["observable_difference_passed"] = 1.0 if observable_difference.get("passed") else 0.0
    quality["observable_difference_frame_backed"] = 1.0 if observable_difference.get("frame_backed") else 0.0
    quality["near_duplicate_without_delta"] = 1.0 if observable_difference.get("near_duplicate_risk") == "high" and not observable_difference.get("passed") else 0.0
    quality["bad_imperative_edit_text"] = 1.0 if _edit_text_quality_has_bad_imperative(edit_text_quality) else 0.0


def _competing_difference_gate(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
) -> dict[str, Any]:
    reasons = _competing_difference_reasons(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference_type=str(difference.get("type", "")).strip(),
    )
    return {
        "passed": not reasons,
        "failure_reason": "; ".join(_dedupe_strings(reasons)),
    }


def _audio_event_independent_evidence_gate(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
) -> dict[str, Any]:
    if str(difference.get("type", "")).strip() != "audio_event":
        return {
            "passed": True,
            "reference_evidence": [],
            "target_evidence": [],
            "supporting_fields": [],
            "failure_reason": "",
        }
    reference_terms = _non_speech_audio_terms(reference_annotation)
    target_terms = _non_speech_audio_terms(target_annotation)
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    from_absent = _is_audio_absence_edit_phrase(from_value)
    to_absent = _is_audio_absence_edit_phrase(to_value)
    reference_supported = (
        (from_absent and not _audio_terms_match(reference_terms, to_value))
        or _audio_terms_match(reference_terms, from_value)
    )
    target_supported = (
        (to_absent and not _audio_terms_match(target_terms, from_value))
        or _audio_terms_match(target_terms, to_value)
    )
    terms_differ = bool(_first_unique(reference_terms, target_terms) or _first_unique(target_terms, reference_terms))
    passed = bool(reference_terms or target_terms) and reference_supported and target_supported and terms_differ
    failure_reason = ""
    if not passed:
        failure_reason = "audio_event lacks independent non-speech audio evidence"
    return {
        "passed": passed,
        "reference_evidence": reference_terms,
        "target_evidence": target_terms,
        "supporting_fields": ["audio_events"] if reference_terms or target_terms else [],
        "failure_reason": failure_reason,
    }


def _natural_pair_quality_gate(
    *,
    record: dict[str, Any],
    edit_text_quality: dict[str, Any],
    observable_difference: dict[str, Any],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> dict[str, Any]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    if str(record.get("source_type", "natural")).strip() == "synthetic_edit":
        return {"passed": True, "failure_codes": [], "failure_reason": ""}

    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    edit_text = str(record.get("edit_text", "")).strip()
    normalized_edit = _normalized_phrase(edit_text)
    failure_codes: list[str] = []

    if _edit_text_quality_has_bad_imperative(edit_text_quality):
        failure_codes.append("bad_imperative_edit_text")

    if (
        difference_type in VISUAL_DIFFERENCE_TYPES
        and _score_float(quality.get("visual_near_duplicate_score")) >= MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE
        and not _boolish(observable_difference.get("frame_backed"))
    ):
        failure_codes.append("too_similar_without_observable_delta")

    if difference_type == "scene":
        same_context = _score_float(quality.get("same_context_score"))
        source_context = record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {}
        relation = str(source_context.get("relation", "")).strip()
        loose_edit = (
            normalized_edit.startswith("make it ")
            or normalized_edit.startswith("make it like")
            or normalized_edit.startswith("turn it into")
        )
        if relation == "cross_dataset" or loose_edit or (
            not _is_exploration_profile(acceptance_profile)
            and same_context < 0.75
        ):
            failure_codes.append("too_broad_or_loose_pair")

    if difference_type == "visible_text":
        failure_codes.append("visible_text_disabled")
        has_from_to = bool(str(difference.get("from", "")).strip() and str(difference.get("to", "")).strip())
        if (
            not has_from_to
            or not _boolish(observable_difference.get("passed"))
            or not _boolish(observable_difference.get("frame_backed"))
            or _score_float(quality.get("target_uniqueness_score")) < MIN_ACCEPT_TARGET_UNIQUENESS_SCORE
        ):
            failure_codes.append("ocr_template_risk")
        if has_from_to and _visible_text_fragment_edit(difference):
            failure_codes.append("visible_text_fragment_edit")

    if difference_type == "audio_event" and _difference_values_are_too_similar(
        str(difference.get("from", "")),
        str(difference.get("to", "")),
    ):
        failure_codes.append("audio_event_too_similar")
    if difference_type == "audio_event" and _score_float(quality.get("audio_primary_allowed", 1.0)) < 1.0:
        failure_codes.append("audio_secondary_due_to_visual_delta")

    failure_codes = _dedupe_strings(failure_codes)
    reasons = [NATURAL_PAIR_GATE_LABELS[code] for code in failure_codes if code in NATURAL_PAIR_GATE_LABELS]
    return {
        "passed": not failure_codes,
        "failure_codes": failure_codes,
        "failure_reason": "; ".join(reasons),
    }


def _is_audio_absence_edit_phrase(value: str) -> bool:
    return _is_non_speech_absence_audio_phrase(value) or _absence_like_phrase(value)


def _audio_terms_match(terms: list[str], phrase: str) -> bool:
    if not phrase:
        return True
    phrase_tokens = _tokenize_text(phrase)
    if not phrase_tokens:
        return False
    for term in terms:
        term_tokens = _tokenize_text(term)
        if not term_tokens:
            continue
        if _text_mentions_phrase(term, phrase) or _text_mentions_phrase(phrase, term):
            return True
        if _jaccard(phrase_tokens, term_tokens) >= 0.5:
            return True
    return False


def _ensure_structured_gate_fields(
    record: dict[str, Any],
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> dict[str, Any]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    record = dict(record)
    quality = dict(record.get("quality", {}))
    edit_text_quality = dict(record.get("edit_text_quality") or {})
    if not edit_text_quality:
        edit_text_quality = _edit_text_quality_payload(
            edit_text=str(record.get("edit_text", "")),
            difference=record.get("difference", {}),
            modalities=record.get("modalities", []),
            reference_caption=str(record.get("reference_caption", "")),
            target_caption=str(record.get("target_caption", "")),
        )
    observable_difference = dict(record.get("observable_difference") or {})
    if not observable_difference:
        observable_difference = _observable_difference_gate(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=record.get("difference", {}),
            visual_near_duplicate_score=quality.get("visual_near_duplicate_score"),
        )
    _apply_structured_gate_quality(
        quality,
        edit_text_quality=edit_text_quality,
        observable_difference=observable_difference,
    )
    natural_pair_gate = dict(record.get("natural_pair_gate") or {})
    if not natural_pair_gate:
        natural_pair_gate = _natural_pair_quality_gate(
            record={**record, "quality": quality},
            edit_text_quality=edit_text_quality,
            observable_difference=observable_difference,
            acceptance_profile=acceptance_profile,
        )
    for code in NATURAL_PAIR_GATE_LABELS:
        quality[code] = 0.0
    for code in natural_pair_gate.get("failure_codes", []):
        if code in NATURAL_PAIR_GATE_LABELS:
            quality[code] = 1.0
    competing_difference = dict(record.get("competing_difference") or {})
    if not competing_difference:
        competing_difference = _competing_difference_gate(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=record.get("difference", {}),
        )
    audio_event_evidence = dict(record.get("audio_event_evidence") or {})
    if not audio_event_evidence:
        audio_event_evidence = _audio_event_independent_evidence_gate(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=record.get("difference", {}),
        )
    quality["competing_difference_passed"] = 1.0 if competing_difference.get("passed") else 0.0
    quality["audio_event_independent_evidence_passed"] = 1.0 if audio_event_evidence.get("passed") else 0.0
    verification = dict(record.get("verification", {}))
    if verification:
        existing_check = dict(verification.get("edit_text_quality_check", {}))
        edit_necessity = dict(verification.get("edit_necessity", {}))
        judge = dict(record.get("judge", {}))
        reference_does_not_satisfy = not _boolish(
            edit_necessity.get(
                "reference_satisfies_edit",
                judge.get("reference_satisfies_edit", False),
            )
        )
        target_satisfies = _boolish(
            edit_necessity.get(
                "target_satisfies_edit",
                judge.get("target_satisfies_edit", False),
            )
        )
        local_check = {
            "not_caption_like": bool(edit_text_quality.get("not_caption_like")),
            "matches_modality": bool(edit_text_quality.get("no_modality_leakage")),
            "single_primary_difference": bool(edit_text_quality.get("single_change")),
            "reference_does_not_satisfy": reference_does_not_satisfy,
            "target_satisfies": target_satisfies,
            "score": _score_float(edit_text_quality.get("score")),
            "failure_reason": "; ".join(edit_text_quality.get("bad_patterns", [])),
        }
        local_reason = str(local_check.get("failure_reason", "")).strip()
        model_reason = str(existing_check.get("failure_reason", "")).strip()
        failure_reason = local_reason
        if local_reason and model_reason:
            failure_reason = f"{local_reason}; model verifier note: {model_reason}"
        verification["edit_text_quality_check"] = {
            "not_caption_like": local_check["not_caption_like"],
            "matches_modality": local_check["matches_modality"],
            "single_primary_difference": local_check["single_primary_difference"],
            "reference_does_not_satisfy": local_check["reference_does_not_satisfy"],
            "target_satisfies": local_check["target_satisfies"],
            "score": local_check["score"],
            "failure_reason": failure_reason,
        }
        _sync_observable_difference_failure(
            verification,
            observable_difference=observable_difference,
        )
        _sync_synthetic_audio_verification_from_evidence(
            record,
            verification=verification,
            audio_event_evidence=audio_event_evidence,
        )
        if _uses_soft_local_gate_profile(acceptance_profile):
            soft_gate_warnings = _dedupe_strings(
                _normalize_list(quality.get("exploration_warnings", []))
                + _normalize_list(natural_pair_gate.get("failure_codes", []))
                + ([] if bool(competing_difference.get("passed", True)) else ["competing_difference"])
                + ([] if bool(audio_event_evidence.get("passed", True)) else ["audio_event_weak_evidence"])
            )
            quality["exploration_warnings"] = soft_gate_warnings
            if _is_audio_matters_profile(acceptance_profile):
                quality["audio_matters_warnings"] = soft_gate_warnings
        else:
            _sync_local_gate_failure(
                verification,
                passed=bool(competing_difference.get("passed", True)),
                reason=str(competing_difference.get("failure_reason", "")).strip(),
            )
            _sync_local_gate_failure(
                verification,
                passed=bool(audio_event_evidence.get("passed", True)),
                reason=str(audio_event_evidence.get("failure_reason", "")).strip(),
            )
            _sync_local_gate_failure(
                verification,
                passed=bool(natural_pair_gate.get("passed", True)),
                reason=str(natural_pair_gate.get("failure_reason", "")).strip(),
            )
        verification["passed"] = _verification_accepts(verification)
        verification["failures"] = _verification_failures(verification)
        record["verification"] = verification
    record["quality"] = quality
    record["quality"]["acceptance_profile"] = acceptance_profile
    record["edit_text_quality"] = edit_text_quality
    record["observable_difference"] = observable_difference
    record["natural_pair_gate"] = natural_pair_gate
    record["competing_difference"] = competing_difference
    record["audio_event_evidence"] = audio_event_evidence
    return record


def _sync_synthetic_audio_verification_from_evidence(
    record: dict[str, Any],
    *,
    verification: dict[str, Any],
    audio_event_evidence: dict[str, Any],
) -> None:
    if str(record.get("source_type", "")).strip() != "synthetic_edit":
        return
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    if not _is_audio_synthetic_route(_synthetic_generation_route(generation)):
        return
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    if str(difference.get("type", "")).strip() != "audio_event":
        return
    if not _boolish(audio_event_evidence.get("passed")):
        return

    expected_event = _synthetic_audio_expected_event(record)
    reason = (
        "synthetic audio plan and independent audio evidence confirm "
        f"target contains the requested non-speech audio event: {expected_event}"
    )
    caption_delta = verification.setdefault("caption_delta", {})
    caption_delta["caption_equivalent"] = False
    caption_delta["has_concrete_difference"] = True
    caption_delta["difference_matches_edit"] = True
    differences = _normalize_list(caption_delta.get("concrete_differences", []))
    if expected_event and not any(_text_mentions_phrase(item, expected_event) for item in differences):
        differences.append(f"target contains {expected_event}; reference does not")
    caption_delta["concrete_differences"] = differences
    caption_delta["reason"] = _append_reason(caption_delta.get("reason"), reason)

    edit_projection = verification.setdefault("edit_projection", {})
    edit_projection["target_matches_projection"] = True
    edit_projection["score"] = max(_score_float(edit_projection.get("score")), 0.9)
    edit_projection["reason"] = _append_reason(edit_projection.get("reason"), reason)

    edit_necessity = verification.setdefault("edit_necessity", {})
    edit_necessity["edit_needed"] = True
    edit_necessity["reference_satisfies_edit"] = False
    edit_necessity["target_satisfies_edit"] = True
    edit_necessity["score"] = max(_score_float(edit_necessity.get("score")), 0.9)
    edit_necessity["reason"] = _append_reason(edit_necessity.get("reason"), reason)


def _sync_observable_difference_failure(
    verification: dict[str, Any],
    *,
    observable_difference: dict[str, Any],
) -> None:
    if _boolish(observable_difference.get("passed", True)):
        return
    reason = str(observable_difference.get("failure_reason", "")).strip()
    if not reason:
        reason = "observable_difference gate found no concrete visual delta evidence"
    reason = f"observable_difference gate failed: {reason}"
    _sync_local_gate_failure(verification, passed=False, reason=reason)


def _sync_local_gate_failure(
    verification: dict[str, Any],
    *,
    passed: bool,
    reason: str,
) -> None:
    if passed:
        return
    reason = reason.strip() or "local quality gate failed"
    existing_reason = str(verification.get("observable_difference_failure", "")).strip()
    if existing_reason:
        verification["observable_difference_failure"] = _append_reason(existing_reason, reason)
    else:
        verification["observable_difference_failure"] = reason

    caption_delta = verification.setdefault("caption_delta", {})
    caption_delta["has_concrete_difference"] = False
    caption_delta["difference_matches_edit"] = False
    caption_delta["reason"] = _append_reason(caption_delta.get("reason"), reason)

    edit_projection = verification.setdefault("edit_projection", {})
    edit_projection["target_matches_projection"] = False
    edit_projection["score"] = min(_score_float(edit_projection.get("score")), 0.0)
    edit_projection["reason"] = _append_reason(edit_projection.get("reason"), reason)

    edit_necessity = verification.setdefault("edit_necessity", {})
    edit_necessity["edit_needed"] = False
    if "reference already appears to contain" in reason:
        edit_necessity["reference_satisfies_edit"] = True
    if "target still appears to contain" in reason:
        edit_necessity["target_satisfies_edit"] = False
    edit_necessity["score"] = min(_score_float(edit_necessity.get("score")), 0.0)
    edit_necessity["reason"] = _append_reason(edit_necessity.get("reason"), reason)


def _prepare_record_for_acceptance(
    record: dict[str, Any],
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> dict[str, Any]:
    record = _ensure_structured_gate_fields(
        record,
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        acceptance_profile=acceptance_profile,
    )
    judge = dict(record.get("judge", {}))
    verification = record.get("verification", {})
    heuristic_quality = record.get("heuristic_quality")
    if not isinstance(heuristic_quality, dict) or not heuristic_quality:
        heuristic_quality = record.get("quality", {})
    local_gate_quality = dict(record.get("quality", {}))
    record["quality"] = _effective_pair_quality(judge, verification, heuristic_quality)
    _carry_local_gate_quality(record["quality"], local_gate_quality)
    record["quality"]["acceptance_profile"] = _normalize_acceptance_profile(acceptance_profile)
    dominant_delta_decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=record.get("difference", {}) if isinstance(record.get("difference"), dict) else {},
        quality=record["quality"],
        source_context=record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {},
    )
    record["quality"]["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
    record["quality"]["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
    record["quality"]["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
    record["quality"]["dominant_delta_decision"] = dominant_delta_decision
    record["dominant_delta_decision"] = dominant_delta_decision
    if _is_exploration_profile(acceptance_profile):
        record["quality"]["exploration_verification_passed"] = 1.0 if _verification_accepts(verification) else 0.0
    return record


def _carry_local_gate_quality(target_quality: dict[str, Any], source_quality: dict[str, Any]) -> None:
    for key in (
        "edit_text_quality_score",
        "edit_text_is_imperative",
        "edit_text_matches_difference_type",
        "edit_text_single_change",
        "edit_text_not_caption_like",
        "edit_text_no_modality_leakage",
        "observable_difference_passed",
        "observable_difference_frame_backed",
        "near_duplicate_without_delta",
        "bad_imperative_edit_text",
        "too_similar_without_observable_delta",
        "too_broad_or_loose_pair",
        "visible_text_disabled",
        "ocr_template_risk",
        "audio_event_too_similar",
        "visible_text_fragment_edit",
        "template_compatibility_score",
        "clean_stability_score",
        "single_delta_bundle_score",
        "title_card_or_boundary_text",
        "talking_head_template",
        "competing_difference_passed",
        "audio_event_independent_evidence_passed",
        "synthetic_context_override",
        "acceptance_profile",
        "exploration_warnings",
        "audio_matters_warnings",
        "exploration_verification_passed",
        "dominant_delta_type",
        "audio_primary_allowed",
        "audio_anchor_required",
        "audio_anchor_score",
        "audio_anchor_context_score",
        "audio_anchor_min_rms",
        "audio_matters_line",
        "omni_visual_accept",
        "omni_reject_reason",
        "visual_delta_type",
        "visual_delta_strength",
        "near_duplicate_risk",
        "reference_satisfies_edit",
        "target_satisfies_edit",
        "caption_equivalent",
        "order_only_scene_reorder",
        "weak_synonym_or_wording_delta",
        "visual_competing_delta_score",
        "edit_primary_modality",
        "dominant_delta_decision",
    ):
        if key in source_quality:
            target_quality[key] = source_quality[key]


def _quality_for_model_fields(
    *,
    base_quality: dict[str, Any],
    model_fields: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    if str(model_fields.get("difference", {}).get("type", "")).strip() == "audio_event":
        model_fields = _normalize_audio_event_model_fields(model_fields)
    quality = dict(base_quality)
    difference = model_fields.get("difference", {})
    difference_type = str(difference.get("type", "")).strip()
    quality["difference_type"] = difference_type
    quality["has_audio_modality"] = 1.0 if "audio" in set(model_fields.get("modalities", [])) else 0.0
    if difference_type == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
    if difference_type == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(
            reference_annotation,
            target_annotation,
        )
    if _has_intraclip_difference_conflict(
        difference=difference,
        reference_caption=str(model_fields.get("reference_caption", "")),
        target_caption=str(model_fields.get("target_caption", "")),
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    ):
        quality["intraclip_change_conflict"] = 1.0
    source_context = base_quality.get("source_context", {}) if isinstance(base_quality.get("source_context"), dict) else {}
    dominant_delta_decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=difference,
        quality=quality,
        source_context=source_context,
    )
    quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
    quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
    quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
    quality["dominant_delta_decision"] = dominant_delta_decision
    return quality


def _score_ordered_pair(
    *,
    root: Path,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    compute_visual_near_duplicate: bool = False,
) -> dict[str, Any] | None:
    if reference_annotation["clip_id"] == target_annotation["clip_id"]:
        return None

    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    source_context = _source_context(reference_annotation, target_annotation)
    if source_context["relation"] == "cross_dataset":
        return None
    same_context_score = _pair_context_score(
        semantic_context_score=semantic_context_score,
        source_context=source_context,
    )
    priority_order = _difference_priority_order(same_context_score=same_context_score)
    primary_difference = _detect_primary_difference(
        reference_annotation,
        target_annotation,
        priority_order=priority_order,
    )
    if primary_difference is None:
        return None
    changed_types = primary_difference.pop("changed_types")
    if same_context_score < MIN_PAIR_CONTEXT_SCORE:
        return None
    if len(changed_types) > MAX_PAIR_CHANGED_TYPES:
        return None

    edit_match_score = _edit_match_score(
        same_context_score=same_context_score,
        primary_difference_type=primary_difference["type"],
        changed_types=changed_types,
    )
    if edit_match_score < MIN_PAIR_EDIT_MATCH_SCORE:
        return None

    hard_negative_annotations = _select_hard_negative_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=primary_difference,
    )
    if len(hard_negative_annotations) < 2:
        return None

    target_uniqueness_score = _target_uniqueness_score(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=primary_difference,
    )
    visual_near_duplicate_score = None
    if compute_visual_near_duplicate:
        visual_near_duplicate_score = _visual_near_duplicate_score(
            _resolve_under_root(root, reference_annotation["output_path"]),
            _resolve_under_root(root, target_annotation["output_path"]),
        )
    hard_negative_paths = [
        _display_path(root, _resolve_under_root(root, annotation["output_path"])) for annotation in hard_negative_annotations[:3]
    ]
    if len(hard_negative_paths) < 2:
        return None

    reference_path = _display_path(root, _resolve_under_root(root, reference_annotation["output_path"]))
    target_path = _display_path(root, _resolve_under_root(root, target_annotation["output_path"]))
    if reference_path in hard_negative_paths:
        return None
    if target_path in hard_negative_paths:
        return None

    quality = {
        "same_context_score": round(same_context_score, 3),
        "semantic_context_score": round(semantic_context_score, 3),
        "edit_match_score": round(edit_match_score, 3),
        "target_uniqueness_score": round(target_uniqueness_score, 3),
        "difference_strength_score": round(
            _difference_strength_score(
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                primary_difference=primary_difference,
                changed_types=changed_types,
            ),
            3,
        ),
        "difference_type": primary_difference["type"],
    }
    if str(source_context.get("relation", "")) == SAME_TEMPLATE_CLUSTER_RELATION:
        quality["template_compatibility_score"] = _score_float(source_context.get("template_compatibility_score"))
        quality["clean_stability_score"] = _score_float(source_context.get("clean_stability_score"))
        quality["single_delta_bundle_score"] = _single_delta_bundle_score(
            reference_annotation,
            target_annotation,
            difference_type=str(primary_difference.get("type", "")).strip(),
        )
        quality["talking_head_template"] = 1.0 if _is_talking_head_template(reference_annotation) and _is_talking_head_template(target_annotation) else 0.0
        quality["title_card_or_boundary_text"] = 1.0 if _title_card_or_boundary_text(reference_annotation) or _title_card_or_boundary_text(target_annotation) else 0.0
        quality["subject_signature_bundle_count"] = float(
            min(
                len(_annotation_subject_signature_bundle(reference_annotation)),
                len(_annotation_subject_signature_bundle(target_annotation)),
            )
        )
    if primary_difference["type"] == "action":
        quality["action_evidence_score"] = _action_evidence_score(reference_annotation, target_annotation)
    if primary_difference["type"] == "speech":
        quality["speech_evidence_score"] = _speech_evidence_score(reference_annotation, target_annotation)
        quality["speech_specificity_score"] = _speech_specificity_score(reference_annotation, target_annotation)
        quality["speech_transcript_backed"] = 1.0 if _speech_is_transcript_backed(reference_annotation, target_annotation) else 0.0
        quality["has_audio_modality"] = 1.0
    if primary_difference["type"] == "audio_event":
        quality["non_speech_audio_event_score"] = _non_speech_audio_event_score(
            reference_annotation,
            target_annotation,
        )
        quality["has_audio_modality"] = 1.0
    if visual_near_duplicate_score is not None:
        quality["visual_near_duplicate_score"] = round(visual_near_duplicate_score, 3)
    dominant_delta_decision = _dominant_delta_decision(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        difference=primary_difference,
        quality=quality,
        source_context=source_context,
    )
    quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
    quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
    quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
    quality["dominant_delta_decision"] = dominant_delta_decision
    composite_score = _candidate_composite_score(quality, source_context)
    return {
        "proposal_id": _build_proposal_id(reference_path, target_path),
        "reference_annotation": _sanitize_annotation_for_output(reference_annotation, root),
        "target_annotation": _sanitize_annotation_for_output(target_annotation, root),
        "primary_difference": primary_difference,
        "changed_difference_types": list(changed_types),
        "quality": quality,
        "composite_score": composite_score,
        "source_context": source_context,
        "dominant_delta_decision": dominant_delta_decision,
        "difference_evidence": _difference_evidence_from_annotations(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            primary_difference=primary_difference,
        ),
        "hard_negative_annotations": [_sanitize_annotation_for_output(annotation, root) for annotation in hard_negative_annotations[:3]],
        "hard_negative_paths": hard_negative_paths,
    }


def _candidate_composite_score(quality: dict[str, Any], source_context: dict[str, Any]) -> float:
    composite_score = round(
        _score_float(quality.get("same_context_score")) * 0.45
        + _score_float(quality.get("edit_match_score")) * 0.35
        + _score_float(quality.get("target_uniqueness_score")) * 0.15
        + _score_float(quality.get("difference_strength_score")) * 0.05,
        4,
    )
    composite_score += _score_float(source_context.get("score")) * 0.08
    composite_score += _score_float(quality.get("template_compatibility_score", source_context.get("template_compatibility_score"))) * 0.06
    composite_score += _score_float(quality.get("single_delta_bundle_score")) * 0.05
    composite_score += _score_float(quality.get("clean_stability_score", source_context.get("clean_stability_score"))) * 0.04
    return round(composite_score, 4)


def _visual_near_duplicate_score(left_path: Path, right_path: Path) -> float | None:
    if not left_path.exists() or not right_path.exists():
        return None
    left_frames = _sample_video_rgb_frames(left_path)
    right_frames = _sample_video_rgb_frames(right_path)
    if not left_frames or not right_frames:
        return None

    best_scores: list[float] = []
    for left_frame in left_frames:
        left_hash = _average_frame_hash(left_frame)
        frame_scores: list[float] = []
        for right_frame in right_frames:
            pixel_score = 1.0 - _frame_mae(left_frame, right_frame)
            hash_score = 1.0 - _hash_hamming(left_hash, _average_frame_hash(right_frame))
            frame_scores.append(max(0.0, min(1.0, min(pixel_score, hash_score))))
        best_scores.append(max(frame_scores))
    return sum(best_scores) / len(best_scores)


def _sample_video_rgb_frames(path: Path, *, size: int = 32, max_frames: int = 6) -> list[bytes]:
    if shutil.which("ffmpeg") is None:
        return []
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps=1,scale={size}:{size},format=rgb24",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    frame_size = size * size * 3
    data = completed.stdout
    return [data[index : index + frame_size] for index in range(0, len(data), frame_size) if len(data[index : index + frame_size]) == frame_size]


def _frame_mae(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(abs(a - b) for a, b in zip(left, right)) / (255.0 * len(left))


def _average_frame_hash(frame: bytes) -> tuple[bool, ...]:
    if not frame:
        return tuple()
    luminance = [
        (int(frame[index]) * 299 + int(frame[index + 1]) * 587 + int(frame[index + 2]) * 114) // 1000
        for index in range(0, len(frame) - 2, 3)
    ]
    if not luminance:
        return tuple()
    mean_value = sum(luminance) / len(luminance)
    return tuple(value >= mean_value for value in luminance)


def _hash_hamming(left: tuple[bool, ...], right: tuple[bool, ...]) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(1 for left_bit, right_bit in zip(left, right) if left_bit != right_bit) / len(left)


def _sanitize_annotation_for_output(annotation: dict[str, Any], root: Path) -> dict[str, Any]:
    sanitized = dict(annotation)
    sanitized["output_path"] = _display_path(root, _resolve_under_root(root, annotation["output_path"]))
    return sanitized


def _select_better_pair(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    left_tuple = (
        left["composite_score"],
        left["quality"]["edit_match_score"],
        left["quality"]["same_context_score"],
        left["proposal_id"],
    )
    right_tuple = (
        right["composite_score"],
        right["quality"]["edit_match_score"],
        right["quality"]["same_context_score"],
        right["proposal_id"],
    )
    return left if left_tuple >= right_tuple else right


def _annotation_has_signal(annotation: dict[str, Any]) -> bool:
    return bool(
        str(annotation.get("summary", "")).strip()
        or annotation.get("subjects")
        or annotation.get("actions")
        or annotation.get("audio_events")
        or _timeline_audio_terms(annotation)
        or annotation.get("speech")
    )


def _source_context(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {str(value).strip() for value in left.get("source_row_ids", []) if str(value).strip()}
    right_rows = {str(value).strip() for value in right.get("source_row_ids", []) if str(value).strip()}
    shared_rows = sorted(left_rows & right_rows)
    if shared_rows:
        return {
            "relation": "shared_source_row",
            **_source_temporal_context(left, right, default_score=0.9),
            "shared_source_row_ids": shared_rows,
        }

    left_source_path = str(left.get("source_path", "")).strip()
    right_source_path = str(right.get("source_path", "")).strip()
    if left_source_path and left_source_path == right_source_path:
        return {
            "relation": "same_source_video",
            **_source_temporal_context(left, right, default_score=0.65),
        }

    left_dataset = str(left.get("dataset", "")).strip()
    right_dataset = str(right.get("dataset", "")).strip()
    if left_dataset and right_dataset:
        if left_dataset == right_dataset:
            text_score = _source_text_similarity(left, right)
            template_compatibility = _template_compatibility_score(left, right)
            clean_stability = min(_clean_stability_score(left), _clean_stability_score(right))
            if template_compatibility >= MIN_TEMPLATE_COMPATIBILITY_SCORE and clean_stability >= MIN_TEMPLATE_CLEAN_STABILITY_SCORE:
                return {
                    **_same_template_cluster_source_context(left, right),
                    "text_similarity": round(text_score, 3),
                }
            return {
                "relation": "same_dataset",
                "score": round(0.25 + text_score * 0.35, 3),
                "dataset": left_dataset,
                "text_similarity": round(text_score, 3),
            }
        return {"relation": "cross_dataset", "score": 0.0, "datasets": [left_dataset, right_dataset]}

    text_score = _source_text_similarity(left, right)
    return {"relation": "unknown", "score": round(text_score * 0.2, 3), "text_similarity": round(text_score, 3)}


def _pair_context_score(*, semantic_context_score: float, source_context: dict[str, Any]) -> float:
    source_score = _score_float(source_context.get("score"))
    relation = str(source_context.get("relation", "")).strip()
    if relation in {"shared_source_row", "same_source_video", SAME_TEMPLATE_CLUSTER_RELATION, "synthetic_from_reference"}:
        return max(semantic_context_score, source_score)
    return semantic_context_score


def _source_temporal_context(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    default_score: float,
) -> dict[str, Any]:
    left_bounds = _clip_time_bounds(left)
    right_bounds = _clip_time_bounds(right)
    if left_bounds is None or right_bounds is None:
        return {"score": round(default_score, 3), "temporal_relation": "unknown"}

    left_start, left_end = left_bounds
    right_start, right_end = right_bounds
    gap_seconds = max(0.0, max(left_start, right_start) - min(left_end, right_end))
    if gap_seconds <= 0.5:
        score = 0.9
        temporal_relation = "adjacent_or_overlapping"
    elif gap_seconds <= 8.0:
        score = 0.78
        temporal_relation = "nearby"
    elif gap_seconds <= 16.0:
        score = 0.65
        temporal_relation = "loose"
    else:
        score = 0.45
        temporal_relation = "distant"
    return {
        "score": round(score, 3),
        "temporal_relation": temporal_relation,
        "temporal_gap_seconds": round(gap_seconds, 3),
    }


def _clip_time_bounds(annotation: dict[str, Any]) -> tuple[float, float] | None:
    source_clip = annotation.get("source_clip")
    if not isinstance(source_clip, dict):
        return None
    try:
        start_seconds = float(source_clip["start_seconds"])
        end_seconds = float(source_clip["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if end_seconds <= start_seconds:
        return None
    return start_seconds, end_seconds


def _source_text_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _jaccard(_text_field_tokens(left.get("text_fields", {})), _text_field_tokens(right.get("text_fields", {})))


def _text_field_tokens(text_fields: Any) -> set[str]:
    if not isinstance(text_fields, dict):
        return set()
    tokens: set[str] = set()
    for value in text_fields.values():
        if isinstance(value, list):
            for item in value:
                tokens.update(_tokenize_text(str(item)))
        else:
            tokens.update(_tokenize_text(str(value)))
    return tokens


def _same_context_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    subject_score = _jaccard(_tokenize_values(left.get("subjects", [])), _tokenize_values(right.get("subjects", [])))
    scene_score = _scene_similarity(str(left.get("scene", "")), str(right.get("scene", "")))
    summary_score = _jaccard(_tokenize_text(str(left.get("summary", ""))), _tokenize_text(str(right.get("summary", ""))))
    text_score = _jaccard(
        _tokenize_values(left.get("on_screen_text", [])),
        _tokenize_values(right.get("on_screen_text", [])),
    )
    attribute_score = _jaccard(_tokenize_values(left.get("attributes", [])), _tokenize_values(right.get("attributes", [])))
    score = (
        subject_score * 0.35
        + scene_score * 0.30
        + summary_score * 0.20
        + text_score * 0.10
        + attribute_score * 0.05
    )
    return max(0.0, min(1.0, score))


def _difference_priority_order(*, same_context_score: float) -> tuple[str, ...]:
    if same_context_score >= 0.70:
        return HIGH_CONTEXT_PAIR_PRIORITY
    return PAIR_PRIORITY


def _scene_similarity(left: str, right: str) -> float:
    left_value = left.strip().lower()
    right_value = right.strip().lower()
    if not left_value or not right_value:
        return 0.0
    if left_value == right_value:
        return 1.0
    return _jaccard(_tokenize_text(left_value), _tokenize_text(right_value))


def _detect_primary_difference(
    reference: dict[str, Any],
    target: dict[str, Any],
    *,
    priority_order: tuple[str, ...] = PAIR_PRIORITY,
) -> dict[str, Any] | None:
    differences: dict[str, dict[str, Any]] = {}

    reference_counts = _normalize_object_counts(reference.get("object_counts", {}))
    target_counts = _normalize_object_counts(target.get("object_counts", {}))
    shared_count_labels = sorted(set(reference_counts) & set(target_counts))
    for label in shared_count_labels:
        if reference_counts[label] != target_counts[label]:
            differences["object_count"] = {
                "type": "object_count",
                "from": f"{reference_counts[label]} {label}",
                "to": f"{target_counts[label]} {label}",
                "description": f"the count of {label} changes from {reference_counts[label]} to {target_counts[label]}",
            }
            break

    reference_only = sorted(set(reference_counts) - set(target_counts))
    target_only = sorted(set(target_counts) - set(reference_counts))
    if "object_presence" not in differences:
        if target_only:
            label = target_only[0]
            differences["object_presence"] = {
                "type": "object_presence",
                "from": f"no {label}",
                "to": f"{target_counts[label]} {label}",
                "description": f"{label} appears in the target clip",
            }
        elif reference_only:
            label = reference_only[0]
            differences["object_presence"] = {
                "type": "object_presence",
                "from": f"{reference_counts[label]} {label}",
                "to": f"no {label}",
                "description": f"{label} disappears in the target clip",
            }

    reference_actions = _action_terms_from_annotation(reference)
    target_actions = _action_terms_from_annotation(target)
    added_action = _first_unique(target_actions, reference_actions)
    removed_action = _first_unique(reference_actions, target_actions)
    if added_action or removed_action:
        differences["action"] = {
            "type": "action",
            "from": removed_action or _first_item(reference_actions) or "current action",
            "to": added_action or _first_item(target_actions) or "new action",
            "description": "the main action changes between the clips and is supported by action/timeline evidence",
        }

    reference_audio = _non_speech_audio_terms(reference)
    target_audio = _non_speech_audio_terms(target)
    added_audio = _first_unique(target_audio, reference_audio)
    removed_audio = _first_unique(reference_audio, target_audio)
    if added_audio or removed_audio:
        differences["audio_event"] = {
            "type": "audio_event",
            "from": removed_audio or _first_item(reference_audio) or "no distinctive audio event",
            "to": added_audio or _first_item(target_audio) or "no distinctive audio event",
            "description": "the audible event changes between the clips",
        }

    reference_attributes = _normalize_list(reference.get("attributes", []))
    target_attributes = _normalize_list(target.get("attributes", []))
    added_attribute = _first_unique(target_attributes, reference_attributes)
    removed_attribute = _first_unique(reference_attributes, target_attributes)
    if added_attribute or removed_attribute:
        differences["attribute"] = {
            "type": "attribute",
            "from": removed_attribute or _first_item(reference_attributes) or "current attribute",
            "to": added_attribute or _first_item(target_attributes) or "new attribute",
            "description": "an attribute of the scene or subject changes",
        }

    reference_scene = str(reference.get("scene", "")).strip()
    target_scene = str(target.get("scene", "")).strip()
    if reference_scene and target_scene and reference_scene.lower() != target_scene.lower():
        differences["scene"] = {
            "type": "scene",
            "from": reference_scene,
            "to": target_scene,
            "description": "the scene changes between the clips",
        }

    reference_speech = _speech_texts_from_annotation(reference)
    target_speech = _speech_texts_from_annotation(target)
    added_speech = _first_unique(target_speech, reference_speech)
    removed_speech = _first_unique(reference_speech, target_speech)
    if (added_speech or removed_speech) and _speech_evidence_score(reference, target) >= MIN_ACCEPT_SPEECH_EVIDENCE_SCORE:
        differences["speech"] = {
            "type": "speech",
            "from": removed_speech or _first_item(reference_speech) or "no speech",
            "to": added_speech or _first_item(target_speech) or "new speech",
            "description": "the spoken content changes between the clips",
        }

    reference_text = _normalize_list(reference.get("visible_text") or reference.get("on_screen_text", []))
    target_text = _normalize_list(target.get("visible_text") or target.get("on_screen_text", []))
    added_text = _first_unique(target_text, reference_text)
    removed_text = _first_unique(reference_text, target_text)
    if added_text or removed_text:
        differences["visible_text"] = {
            "type": "visible_text",
            "from": removed_text or _first_item(reference_text) or "no visible text",
            "to": added_text or _first_item(target_text) or "new visible text",
            "description": "the visible on-screen text changes between the clips",
        }

    changed_types = [difference_type for difference_type in priority_order if difference_type in differences]
    if not changed_types:
        return None
    primary = dict(differences[changed_types[0]])
    primary["changed_types"] = changed_types
    return primary


def _visual_delta_types(reference: dict[str, Any], target: dict[str, Any]) -> list[str]:
    types: list[str] = []
    reference_subject = _annotation_subject_signature_bundle(reference)
    target_subject = _annotation_subject_signature_bundle(target)
    if reference_subject and target_subject and reference_subject != target_subject:
        types.append("attribute")

    reference_counts = _normalize_object_counts(reference.get("object_counts", {}))
    target_counts = _normalize_object_counts(target.get("object_counts", {}))
    if any(reference_counts[label] != target_counts[label] for label in sorted(set(reference_counts) & set(target_counts))):
        types.append("object_count")
    reference_objects = _annotation_object_signature_bundle(reference)
    target_objects = _annotation_object_signature_bundle(target)
    if reference_objects and target_objects and reference_objects != target_objects:
        types.append("object_presence")

    reference_scene = _annotation_scene_signature_bundle(reference)
    target_scene = _annotation_scene_signature_bundle(target)
    if reference_scene and target_scene and reference_scene != target_scene:
        types.append("scene")

    reference_actions = _action_terms_from_annotation(reference)
    target_actions = _action_terms_from_annotation(target)
    if _first_unique(reference_actions, target_actions) and _first_unique(target_actions, reference_actions):
        types.append("action")

    return [difference_type for difference_type in DOMINANT_VISUAL_DIFFERENCE_TYPES if difference_type in set(types)]


def _audio_event_changed(reference: dict[str, Any], target: dict[str, Any]) -> bool:
    reference_audio = _non_speech_audio_terms(reference)
    target_audio = _non_speech_audio_terms(target)
    return bool(_first_unique(reference_audio, target_audio) or _first_unique(target_audio, reference_audio))


def _audio_primary_allowed(
    *,
    quality: dict[str, Any],
    source_context: dict[str, Any],
    visual_delta_types: list[str],
    audio_changed: bool,
) -> bool:
    if not audio_changed or visual_delta_types:
        return False
    same_context_score = _score_float(quality.get("same_context_score"))
    template_compatibility_score = _score_float(
        quality.get("template_compatibility_score", source_context.get("template_compatibility_score"))
    )
    visual_near_duplicate_score = _score_float(quality.get("visual_near_duplicate_score"))
    relation = str(source_context.get("relation", "")).strip()
    if visual_near_duplicate_score >= AUDIO_PRIMARY_MIN_VISUAL_NEAR_DUPLICATE_SCORE:
        return True
    if relation == "same_source_video" and same_context_score >= 0.78:
        return True
    return bool(
        same_context_score >= AUDIO_PRIMARY_MIN_SAME_CONTEXT_SCORE
        and template_compatibility_score >= AUDIO_PRIMARY_MIN_TEMPLATE_COMPATIBILITY_SCORE
    )


def _dominant_delta_decision(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
    quality: dict[str, Any],
    source_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_context = source_context if isinstance(source_context, dict) else {}
    proposed_type = str(difference.get("type", "")).strip()
    visual_delta_types = _visual_delta_types(reference_annotation, target_annotation)
    audio_changed = _audio_event_changed(reference_annotation, target_annotation)
    audio_allowed = _audio_primary_allowed(
        quality=quality,
        source_context=source_context,
        visual_delta_types=visual_delta_types,
        audio_changed=audio_changed,
    )
    dominant_type = proposed_type
    reason = "proposed difference is the dominant observable delta"
    if proposed_type == "audio_event" and not audio_allowed:
        dominant_type = visual_delta_types[0] if visual_delta_types else "diagnostic_audio_event"
        if visual_delta_types:
            reason = "audio_event is secondary because stronger visual deltas exist"
        else:
            reason = "audio_event lacks near-identical visual context for audio-primary acceptance"
    elif proposed_type not in visual_delta_types and visual_delta_types and proposed_type not in {"speech", "visible_text"}:
        dominant_type = visual_delta_types[0]
        reason = "a stronger visual delta is more suitable as the pair theme"

    secondary = [item for item in [*visual_delta_types, "audio_event" if audio_changed else ""] if item and item != dominant_type]
    visual_competing_delta_score = 0.0
    if visual_delta_types:
        visual_competing_delta_score = min(1.0, 0.45 + 0.15 * len(visual_delta_types))

    flags: list[str] = []
    if proposed_type == "audio_event" and not audio_allowed:
        flags.append("audio_secondary_due_to_visual_delta" if visual_delta_types else "audio_primary_context_too_weak")

    return {
        "proposed_type": proposed_type,
        "dominant_type": dominant_type,
        "audio_primary_allowed": audio_allowed,
        "visual_competing_delta_score": round(visual_competing_delta_score, 3),
        "visual_delta_types": visual_delta_types,
        "secondary_delta_types": _dedupe_strings(secondary),
        "failure_flags": flags,
        "reason": reason,
    }


def _edit_match_score(
    *,
    same_context_score: float,
    primary_difference_type: str,
    changed_types: list[str],
) -> float:
    if primary_difference_type not in PAIR_PRIORITY:
        return 0.0
    base_score = 0.5 + same_context_score * 0.35
    if primary_difference_type in {"object_count", "object_presence", "action", "audio_event", "speech", "visible_text"}:
        base_score += 0.1
    if primary_difference_type in {"audio_event", "speech", "visible_text"} and same_context_score >= 0.70:
        base_score += 0.08
    penalty = max(0, len(changed_types) - 1) * 0.10
    return max(0.0, min(1.0, base_score - penalty))


def _difference_strength_score(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference: dict[str, Any],
    changed_types: list[str],
) -> float:
    difference_type = str(primary_difference.get("type", "")).strip()
    from_value = str(primary_difference.get("from", "")).strip().lower()
    to_value = str(primary_difference.get("to", "")).strip().lower()
    if not difference_type or not from_value or not to_value or from_value == to_value:
        return 0.0

    if difference_type == "object_count":
        score = _object_count_delta_score(from_value, to_value)
    elif difference_type == "object_presence":
        score = 0.82 if from_value.startswith("no ") or to_value.startswith("no ") else 0.70
    elif difference_type == "action":
        score = _action_evidence_score(reference_annotation, target_annotation)
    elif difference_type == "audio_event":
        score = _non_speech_audio_event_score(reference_annotation, target_annotation)
    elif difference_type == "speech":
        score = _speech_evidence_score(reference_annotation, target_annotation)
    elif difference_type == "visible_text":
        score = _list_delta_strength(
            reference_annotation.get("visible_text") or reference_annotation.get("on_screen_text", []),
            target_annotation.get("visible_text") or target_annotation.get("on_screen_text", []),
        )
    elif difference_type == "attribute":
        score = _list_delta_strength(reference_annotation.get("attributes", []), target_annotation.get("attributes", []))
    elif difference_type == "scene":
        score = 0.65 + _scene_similarity(
            str(reference_annotation.get("scene", "")),
            str(target_annotation.get("scene", "")),
        ) * 0.10
    else:
        score = 0.0

    evidence = _difference_evidence_from_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        primary_difference=primary_difference,
    )
    if evidence["supporting_evidence"] and difference_type != "action":
        score = max(score, 0.70)
    if len(changed_types) == 1:
        score += 0.08
    else:
        score -= min(0.15, (len(changed_types) - 1) * 0.04)
    return round(max(0.0, min(1.0, score)), 3)


def _object_count_delta_score(from_value: str, to_value: str) -> float:
    from_count = _first_integer(from_value)
    to_count = _first_integer(to_value)
    if from_count is None or to_count is None or from_count == to_count:
        return 0.65
    delta = abs(to_count - from_count)
    return min(1.0, 0.74 + min(delta, 4) * 0.05)


def _first_integer(value: str) -> int | None:
    match = re.search(r"\d+", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _list_delta_strength(left: Any, right: Any) -> float:
    left_values = _normalize_list(left)
    right_values = _normalize_list(right)
    if left_values == right_values:
        return 0.0
    token_overlap = _jaccard(_tokenize_values(left_values), _tokenize_values(right_values))
    return max(0.62, min(0.92, 0.92 - token_overlap * 0.25))


def _action_terms_from_annotation(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add_values(value: Any) -> None:
        for item in _normalize_list(value):
            if item not in terms:
                terms.append(item)

    add_values(annotation.get("actions", []))
    for container_name in ("events", "storyline"):
        container = annotation.get(container_name, [])
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict):
                add_values(item.get("actions", []))
                action_value = item.get("action")
                if action_value:
                    add_values([action_value])
    return terms


def _has_timeline_action_evidence(annotation: dict[str, Any]) -> bool:
    return bool(_timeline_evidence(annotation))


def _action_evidence_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_terms = _action_terms_from_annotation(reference_annotation)
    target_terms = _action_terms_from_annotation(target_annotation)
    if not reference_terms or not target_terms:
        return 0.0
    if not _first_unique(reference_terms, target_terms) and not _first_unique(target_terms, reference_terms):
        return 0.0

    score = _list_delta_strength(reference_terms, target_terms)
    reference_has_timeline = _has_timeline_action_evidence(reference_annotation)
    target_has_timeline = _has_timeline_action_evidence(target_annotation)
    if reference_has_timeline and target_has_timeline:
        return round(max(score, 0.74), 3)
    if reference_annotation.get("actions") and target_annotation.get("actions"):
        return round(min(score, 0.62), 3)
    return round(min(score, 0.55), 3)


def _speech_texts_from_annotation(annotation: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    def add_text(value: Any) -> None:
        for item in _normalize_list(value):
            if item not in texts:
                texts.append(item)

    add_text(annotation.get("speech", []))
    transcript = annotation.get("speakers_and_transcript", [])
    if isinstance(transcript, list):
        for item in transcript:
            if isinstance(item, dict):
                add_text(
                    [
                        item.get("content")
                        or item.get("transcript")
                        or item.get("text")
                        or item.get("utterance")
                        or ""
                    ]
                )
            else:
                add_text([item])
    return texts


def _speech_specificity_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_score = _speech_specificity_score_for_texts(_speech_texts_from_annotation(reference_annotation))
    target_score = _speech_specificity_score_for_texts(_speech_texts_from_annotation(target_annotation))
    if reference_score == 0.0 or target_score == 0.0:
        return 0.0
    return round(min(reference_score, target_score), 3)


def _speech_specificity_score_for_texts(texts: list[str]) -> float:
    if not texts:
        return 0.0
    best_score = 0.0
    for text in texts:
        normalized = text.strip().lower()
        if not normalized:
            continue
        if normalized in GENERIC_SPEECH_PHRASES:
            best_score = max(best_score, 0.2)
            continue
        tokens = _tokenize_text(normalized)
        content_tokens = {
            token
            for token in tokens
            if token not in GENERIC_SPEECH_TOKENS and token not in VISUAL_DESCRIPTION_TOKENS
        }
        generic_overlap = len(tokens & GENERIC_SPEECH_TOKENS)
        score = 0.0
        if len(content_tokens) >= 6:
            score = 0.9
        elif len(content_tokens) >= 4:
            score = 0.78
        elif len(content_tokens) >= 3 and len(normalized) >= 35:
            score = 0.72
        elif len(content_tokens) >= 2 and generic_overlap:
            score = 0.55
        else:
            score = 0.3
        if any(phrase in normalized for phrase in GENERIC_SPEECH_PHRASES) and len(content_tokens) < 4:
            score = min(score, 0.55)
        best_score = max(best_score, score)
    return round(best_score, 3)


def _speech_evidence_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_texts = _speech_texts_from_annotation(reference_annotation)
    target_texts = _speech_texts_from_annotation(target_annotation)
    if not reference_texts or not target_texts:
        return 0.0
    if not _first_unique(reference_texts, target_texts) and not _first_unique(target_texts, reference_texts):
        return 0.0

    specificity = _speech_specificity_score(reference_annotation, target_annotation)
    if specificity < MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE:
        return round(min(specificity, 0.69), 3)

    has_reference_transcript = _has_transcript_evidence(reference_annotation)
    has_target_transcript = _has_transcript_evidence(target_annotation)
    if not (has_reference_transcript and has_target_transcript):
        return round(min(specificity, 0.69), 3)

    score = _list_delta_strength(reference_texts, target_texts)
    score = max(score, 0.88)
    return round(min(score, specificity), 3)


def _has_transcript_evidence(annotation: dict[str, Any]) -> bool:
    transcript = annotation.get("speakers_and_transcript", [])
    if not isinstance(transcript, list):
        return False
    for item in transcript:
        if isinstance(item, dict):
            if str(item.get("content") or item.get("transcript") or item.get("text") or "").strip():
                return True
        elif str(item).strip():
            return True
    return False


def _speech_is_transcript_backed(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> bool:
    return _has_transcript_evidence(reference_annotation) and _has_transcript_evidence(target_annotation)


def _non_speech_audio_terms(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in (
        _normalize_list(annotation.get("audio_events", []))
        + _timeline_audio_terms(annotation)
        + _annotation_audio_text_terms(annotation)
    ):
        if not _is_speech_like_audio_event(item) and item not in terms:
            terms.append(item)
    return terms


def _annotation_audio_text_terms(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add(value: Any) -> None:
        for item in _normalize_list(value):
            if _is_non_speech_audio_phrase(item) and item not in terms:
                terms.append(item)

    add(annotation.get("summary", ""))
    add(annotation.get("detective_notes", []))
    add(annotation.get("audio_observations", []))
    return terms


def _timeline_audio_terms(annotation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    def add_if_relevant(value: Any) -> None:
        for item in _normalize_list(value):
            if _is_non_speech_audio_phrase(item) and item not in terms:
                terms.append(item)

    container = annotation.get("events", [])
    if not isinstance(container, list):
        return terms
    for item in container:
        if isinstance(item, dict):
            add_if_relevant([item.get("audio", "")])
            add_if_relevant(item.get("audio_events", []))
    return terms


def _is_non_speech_audio_phrase(value: str) -> bool:
    tokens = _tokenize_text(value)
    if _is_speech_only_or_absence_audio_phrase(value):
        return False
    return bool(tokens & NON_SPEECH_AUDIO_TOKENS) and not _is_speech_like_audio_event(value)


def _is_speech_like_audio_event(value: str) -> bool:
    tokens = _tokenize_text(value)
    if not tokens:
        return False
    if tokens & NON_SPEECH_AUDIO_TOKENS:
        return False
    return bool(tokens & GENERIC_SPEECH_TOKENS)


def _is_speech_only_or_absence_audio_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in SPEECH_ONLY_AUDIO_PATTERNS + NON_SPEECH_AUDIO_ABSENCE_PATTERNS)


def _is_non_speech_absence_audio_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in NON_SPEECH_AUDIO_ABSENCE_PATTERNS)


def _is_speech_only_audio_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in SPEECH_ONLY_AUDIO_PATTERNS)


def _speech_content_edit_issues(*, edit_text: str, difference: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    difference_type = str(difference.get("type", "")).strip()
    if difference_type == "speech":
        issues.append("speech difference type is disabled for final Omni-CVR samples")
    if difference_type == "visible_text":
        issues.append("visible_text difference type is disabled for final Omni-CVR samples")

    text_parts = [
        edit_text,
        str(difference.get("from", "")),
        str(difference.get("to", "")),
        str(difference.get("description", "")),
    ]
    normalized = _normalized_phrase(" ".join(text_parts))
    if normalized and any(pattern in normalized for pattern in SPEECH_CONTENT_EDIT_PATTERNS):
        issues.append("speech content edits are disabled for final Omni-CVR samples")

    if difference_type == "audio_event":
        from_value = str(difference.get("from", "")).strip()
        to_value = str(difference.get("to", "")).strip()
        if _is_speech_only_audio_phrase(from_value) or _is_speech_only_audio_phrase(to_value):
            issues.append("audio_event must not use speech-only or narration-only text as the main difference")

    deduped: list[str] = []
    for issue in issues:
        if issue not in deduped:
            deduped.append(issue)
    return deduped


def _split_profiled_speech_content_issues(
    *,
    edit_text: str,
    difference: dict[str, Any],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> tuple[list[str], list[str]]:
    issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
    if not _is_exploration_profile(acceptance_profile):
        return issues, []
    difference_type = str(difference.get("type", "")).strip()
    if difference_type != "audio_event":
        return issues, []

    hard_issues: list[str] = []
    warning_issues: list[str] = []
    for issue in issues:
        if "speech difference type is disabled" in issue or "visible_text difference type is disabled" in issue:
            hard_issues.append(issue)
        else:
            warning_issues.append(issue)
    return hard_issues, warning_issues


def _is_exploration_audio_speech_content_reject(judge: dict[str, Any], quality: dict[str, Any]) -> bool:
    if str(quality.get("difference_type", "")).strip() != "audio_event":
        return False
    reject_reason = str(judge.get("reject_reason", "")).lower()
    return bool(
        "speech content edits are disabled" in reject_reason
        or "speech-only or narration-only" in reject_reason
        or "speech only" in reject_reason
        or "narration only" in reject_reason
    )


def _normalize_audio_event_model_fields(model_fields: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(model_fields)
    difference = dict(normalized.get("difference", {}))
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if _is_non_speech_absence_audio_phrase(from_value) and to_value:
        normalized["edit_text"] = f"add {to_value} to the audio"
    elif _is_non_speech_absence_audio_phrase(to_value) and from_value:
        normalized["edit_text"] = f"remove {from_value} from the audio"
    normalized["difference"] = difference
    return normalized


def _non_speech_audio_event_score(reference_annotation: dict[str, Any], target_annotation: dict[str, Any]) -> float:
    reference_terms = _non_speech_audio_terms(reference_annotation)
    target_terms = _non_speech_audio_terms(target_annotation)
    if not reference_terms and not target_terms:
        return 0.0
    if not _first_unique(reference_terms, target_terms) and not _first_unique(target_terms, reference_terms):
        return 0.0
    return round(max(_list_delta_strength(reference_terms, target_terms), 0.70), 3)


def _difference_evidence_from_annotations(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference: dict[str, Any],
) -> dict[str, Any]:
    difference_type = str(primary_difference.get("type", "")).strip()
    evidence: list[str] = []
    if difference_type in {"object_count", "object_presence"}:
        evidence.append(
            "object_counts: "
            f"{reference_annotation.get('object_counts', {})} -> {target_annotation.get('object_counts', {})}"
        )
    if difference_type == "action":
        evidence.append(_change_text(_action_terms_from_annotation(reference_annotation), _action_terms_from_annotation(target_annotation)))
        evidence.append(
            "action_evidence_score: "
            f"{_action_evidence_score(reference_annotation, target_annotation):.3f}"
        )
    if difference_type == "audio_event":
        evidence.append(_change_text(_non_speech_audio_terms(reference_annotation), _non_speech_audio_terms(target_annotation)))
        evidence.append(
            "non_speech_audio_event_score: "
            f"{_non_speech_audio_event_score(reference_annotation, target_annotation):.3f}"
        )
    if difference_type == "speech":
        evidence.append(_change_text(_speech_texts_from_annotation(reference_annotation), _speech_texts_from_annotation(target_annotation)))
        evidence.append(
            "speech_evidence_score: "
            f"{_speech_evidence_score(reference_annotation, target_annotation):.3f}"
        )
        evidence.append(
            "speech_specificity_score: "
            f"{_speech_specificity_score(reference_annotation, target_annotation):.3f}"
        )
    if difference_type == "visible_text":
        evidence.append(
            _change_text(
                reference_annotation.get("visible_text") or reference_annotation.get("on_screen_text", []),
                target_annotation.get("visible_text") or target_annotation.get("on_screen_text", []),
            )
        )
    if difference_type == "attribute":
        evidence.append(_change_text(reference_annotation.get("attributes", []), target_annotation.get("attributes", [])))
    if difference_type == "scene":
        evidence.append(f"scene: {reference_annotation.get('scene', '')} -> {target_annotation.get('scene', '')}")

    reference_events = _timeline_evidence(reference_annotation)
    target_events = _timeline_evidence(target_annotation)
    if reference_events or target_events:
        evidence.append(f"events: {' | '.join(reference_events[:2]) or 'none'} -> {' | '.join(target_events[:2]) or 'none'}")

    return {
        "difference_type": difference_type,
        "from": str(primary_difference.get("from", "")).strip(),
        "to": str(primary_difference.get("to", "")).strip(),
        "supporting_evidence": [item for item in evidence if item.strip() and not item.strip().endswith("-> none")],
        "reference_events": reference_events,
        "target_events": target_events,
    }


def _normalize_events_for_evidence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    events: list[str] = []
    for item in value:
        if isinstance(item, dict):
            pieces = [
                str(item.get("visual", "")).strip(),
                str(item.get("audio", "")).strip(),
                "; ".join(_normalize_list(item.get("objects", []))),
                "; ".join(_normalize_list(item.get("actions", []))),
            ]
            text = " / ".join(piece for piece in pieces if piece)
        else:
            text = str(item).strip()
        if text:
            events.append(text)
    return events


def _timeline_evidence(annotation: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for field_name in ("events", "storyline"):
        for item in _normalize_events_for_evidence(annotation.get(field_name, [])):
            if item not in evidence:
                evidence.append(item)
    return evidence


def _select_hard_negative_annotations(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    primary_difference: dict[str, Any],
) -> list[dict[str, Any]]:
    scored_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for other in annotations:
        if other["clip_id"] in {reference_annotation["clip_id"], target_annotation["clip_id"]}:
            continue

        context_score = max(
            _same_context_score(reference_annotation, other),
            _same_context_score(target_annotation, other),
        )
        score = context_score
        other_difference = _detect_primary_difference(reference_annotation, other)
        if other_difference is not None and other_difference["type"] == primary_difference["type"]:
            score -= 0.2
        if other["output_path"] == target_annotation["output_path"]:
            continue
        scored_candidates.append((score, other["clip_id"], other))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored_candidates[:3]]


def _target_uniqueness_score(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
    primary_difference: dict[str, Any],
) -> float:
    competitor_scores = []
    priority_order = (primary_difference["type"],) + tuple(
        item for item in PAIR_PRIORITY if item != primary_difference["type"]
    )
    for other in annotations:
        if other["clip_id"] in {reference_annotation["clip_id"], target_annotation["clip_id"]}:
            continue
        context_score = _same_context_score(target_annotation, other)
        other_difference = _detect_primary_difference(
            reference_annotation,
            other,
            priority_order=priority_order,
        )
        competitor_scores.append(
            _target_competitor_score(
                context_score=context_score,
                primary_difference=primary_difference,
                competitor_difference=other_difference,
            )
        )
    if not competitor_scores:
        return 1.0
    highest_competitor = max(competitor_scores)
    return max(0.0, min(1.0, 1.0 - highest_competitor * 0.75))


def _target_competitor_score(
    *,
    context_score: float,
    primary_difference: dict[str, Any],
    competitor_difference: dict[str, Any] | None,
) -> float:
    if competitor_difference is None:
        return context_score * 0.35
    if competitor_difference["type"] != primary_difference["type"]:
        return context_score * 0.35
    if _difference_targets_overlap(primary_difference, competitor_difference):
        return context_score
    return context_score * 0.35


def _difference_targets_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_target = str(left.get("to", "")).strip().lower()
    right_target = str(right.get("to", "")).strip().lower()
    if left_target and right_target and left_target == right_target:
        return True
    if left_target and right_target:
        left_tokens = _tokenize_text(left_target)
        right_tokens = _tokenize_text(right_target)
        return _jaccard(left_tokens, right_tokens) >= 0.67
    left_tokens = _tokenize_text(str(left.get("description", "")))
    right_tokens = _tokenize_text(str(right.get("description", "")))
    return _jaccard(left_tokens, right_tokens) >= 0.50


def _annotation_prompt_view(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": annotation["clip_id"],
        "output_path": annotation["output_path"],
        "summary": _truncate_text(annotation.get("summary", ""), 700),
        "subjects": _prompt_list(annotation.get("subjects", []), limit=8, text_limit=80),
        "object_counts": dict(annotation.get("object_counts", {})),
        "actions": _prompt_list(annotation.get("actions", []), limit=8, text_limit=80),
        "scene": _truncate_text(annotation.get("scene", ""), 300),
        "attributes": _prompt_list(annotation.get("attributes", []), limit=8, text_limit=120),
        "on_screen_text": _prompt_list(annotation.get("on_screen_text", []), limit=8, text_limit=120),
        "speech": _prompt_list(annotation.get("speech", []), limit=6, text_limit=180),
        "audio_events": _prompt_list(annotation.get("audio_events", []), limit=8, text_limit=120),
        "modalities": _prompt_list(annotation.get("modalities", []), limit=4, text_limit=40),
        "storyline": _prompt_list(annotation.get("storyline", []), limit=6, text_limit=220),
        "events": _prompt_list(annotation.get("events", []), limit=8, text_limit=220),
        "visible_text": _prompt_list(annotation.get("visible_text", []), limit=8, text_limit=120),
        "speakers_and_transcript": _prompt_list(annotation.get("speakers_and_transcript", []), limit=6, text_limit=220),
        "uncertainties": _prompt_list(annotation.get("uncertainties", []), limit=6, text_limit=160),
    }


def _single_source_line_annotation_prompt_view(annotation: dict[str, Any], audio_dataset_line: str) -> dict[str, Any]:
    line = _normalize_audio_dataset_line(audio_dataset_line)
    base = {
        "clip_id": annotation.get("clip_id", ""),
        "output_path": annotation.get("output_path", ""),
        "summary": _truncate_text(annotation.get("summary", ""), 160),
        "scene": _truncate_text(annotation.get("scene", ""), 90),
        "subjects": _prompt_list(annotation.get("subjects", []), limit=4, text_limit=45),
        "actions": _prompt_list(annotation.get("actions", []), limit=4, text_limit=45),
        "modalities": _prompt_list(annotation.get("modalities", []), limit=4, text_limit=32),
    }
    if line == SPEECH_AUDIO_CONTENT_LINE:
        base.update(
            {
                "visual_context_only": {
                    "attributes": _prompt_list(annotation.get("attributes", []), limit=2, text_limit=45),
                    "visible_text": _prompt_list(annotation.get("visible_text", []), limit=1, text_limit=40),
                },
                "speech": _prompt_list(annotation.get("speech", []), limit=2, text_limit=110),
                "speakers_and_transcript": _prompt_list(
                    annotation.get("speakers_and_transcript", []), limit=2, text_limit=120
                ),
                "audio_events": _prompt_list(annotation.get("audio_events", []), limit=4, text_limit=55),
                "audio_refresh_annotation": bool(annotation.get("audio_refresh_annotation")),
                "video_context_type": _truncate_text(annotation.get("video_context_type", ""), 55),
                "video_context_strength": _score_float(annotation.get("video_context_strength")),
                "speech_role": _truncate_text(annotation.get("speech_role", ""), 55),
                "speech_topic_or_step": _truncate_text(annotation.get("speech_topic_or_step", ""), 90),
                "music_description": _truncate_text(annotation.get("music_description", ""), 70),
                "asr_degeneracy_risk": _score_float(annotation.get("asr_degeneracy_risk")),
            }
        )
        return base
    if line == VISUAL_AUDIO_ANCHOR_LINE:
        base.update(
            {
                "attributes": _prompt_list(annotation.get("attributes", []), limit=3, text_limit=50),
                "object_counts": dict(list(annotation.get("object_counts", {}).items())[:6])
                if isinstance(annotation.get("object_counts"), dict)
                else {},
                "visible_text": _prompt_list(annotation.get("visible_text", []), limit=1, text_limit=40),
                "audio_context_hint": {
                    "speech": _prompt_list(annotation.get("speech", []), limit=1, text_limit=55),
                    "audio_events": _prompt_list(annotation.get("audio_events", []), limit=2, text_limit=45),
                },
            }
        )
        return base
    return _annotation_prompt_view(annotation)


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _prompt_list(value: Any, *, limit: int, text_limit: int) -> list[Any]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    compact: list[Any] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            compact.append(
                {
                    str(key): _truncate_text(raw_value, text_limit)
                    for key, raw_value in item.items()
                    if key in {"time", "timestamp", "description", "visual", "audio", "text", "action", "event", "objects"}
                }
            )
        else:
            text = _truncate_text(item, text_limit)
            if text:
                compact.append(text)
    return compact


def _fallback_clip_annotation() -> dict[str, Any]:
    return {
        "summary": "",
        "subjects": [],
        "object_counts": {},
        "actions": [],
        "scene": "",
        "attributes": [],
        "on_screen_text": [],
        "speech": [],
        "audio_events": [],
        "modalities": ["visual"],
    }


def _fallback_pair_model_fields(
    *,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "edit_text": _build_fallback_edit_text(primary_difference),
        "modalities": _infer_pair_modalities(reference_annotation, target_annotation, primary_difference["type"]),
        "reference_caption": str(reference_annotation.get("summary", "")).strip(),
        "target_caption": str(target_annotation.get("summary", "")).strip(),
        "difference": dict(primary_difference),
        "proposal_reason": f"heuristic fallback based on {primary_difference['type']}",
    }


def _build_fallback_edit_text(primary_difference: dict[str, Any]) -> str:
    difference_type = primary_difference["type"]
    from_value = str(primary_difference.get("from", "")).strip()
    to_value = str(primary_difference.get("to", "")).strip()
    if difference_type == "object_count":
        from_count, from_label = _count_and_label(from_value)
        to_count, to_label = _count_and_label(to_value)
        label = to_label or from_label or "object"
        if from_count is not None and to_count is not None:
            return f"change the number of {label} from {from_count} to {to_count}"
        return f"change the number of {label} from {from_value} to {to_value}"
    if difference_type == "object_presence":
        if from_value.lower().startswith("no ") and to_value:
            return f"add {_object_phrase_for_edit(to_value)}"
        if to_value.lower().startswith("no ") and from_value:
            return f"remove {_object_phrase_for_edit(from_value)}"
        return f"replace {_object_phrase_for_edit(from_value)} with {_object_phrase_for_edit(to_value)}"
    if difference_type == "action":
        return f"change the action from {from_value} to {to_value}"
    if difference_type == "audio_event":
        if _is_non_speech_absence_audio_phrase(from_value) and to_value:
            return f"add {to_value} to the audio"
        if _is_non_speech_absence_audio_phrase(to_value) and from_value:
            return f"remove {from_value} from the audio"
        return f"replace {from_value} with {to_value} in the audio"
    if difference_type == "attribute":
        if from_value.startswith("speaker with ") and to_value.startswith("speaker with "):
            return f"change the speaker from {from_value.removeprefix('speaker with ').strip()} to {to_value.removeprefix('speaker with ').strip()}"
        return f"change the attribute from {from_value} to {to_value}"
    if difference_type == "scene":
        return f"change the setting from {from_value} to {to_value}"
    if difference_type == "speech":
        return f"change the speech from {_short_edit_phrase(from_value)} to {_short_edit_phrase(to_value)}"
    if difference_type == "visible_text":
        return f"change on-screen text from {_short_edit_phrase(from_value)} to {_short_edit_phrase(to_value)}"
    return str(primary_difference.get("description", "")).strip() or f"change {from_value} to {to_value}"


def _count_and_label(value: str) -> tuple[int | None, str]:
    match = re.match(r"\s*(\d+)\s+(.+?)\s*$", value)
    if not match:
        return None, _strip_presence_prefix(value)
    return int(match.group(1)), _strip_presence_prefix(match.group(2))


def _strip_presence_prefix(value: str) -> str:
    normalized = str(value).strip()
    normalized = re.sub(r"^\s*no\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^\s*\d+\s+", "", normalized)
    return normalized.strip()


def _object_phrase_for_edit(value: str) -> str:
    label = _strip_presence_prefix(value)
    if not label:
        return "the object"
    first_token = label.split()[0].lower()
    if first_token in {"a", "an", "the"}:
        return label
    article = "an" if first_token[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


def _short_edit_phrase(value: str, *, max_words: int = 12) -> str:
    words = str(value).strip().split()
    if len(words) <= max_words:
        return str(value).strip()
    return " ".join(words[:max_words])


def _infer_pair_modalities(
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    primary_difference_type: str,
) -> list[str]:
    modalities: list[str] = []
    for item in list(reference_annotation.get("modalities", [])) + list(target_annotation.get("modalities", [])):
        value = str(item).strip().lower()
        if value in ALLOWED_MODALITIES and value not in modalities:
            modalities.append(value)
    if primary_difference_type in {"audio_event", "speech"} and "audio" not in modalities:
        modalities.append("audio")
    if "visual" not in modalities:
        modalities.insert(0, "visual")
    return modalities


def _fallback_pair_judge(quality: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "reference_satisfies_edit": False,
        "target_satisfies_edit": False,
        "single_main_difference": False,
        "same_context_score": _score_float(quality.get("same_context_score")),
        "edit_match_score": _score_float(quality.get("edit_match_score")),
        "target_uniqueness_score": _score_float(quality.get("target_uniqueness_score")),
        "audio_required": False,
        "hard_negative_quality": "weak",
        "accept": False,
        "reject_reason": f"pair judge fallback: {reason}",
    }


def _is_verification_context_limit_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "context length",
            "context window",
            "input length",
            "max_model_len",
            "maximum context",
            "too many tokens",
            "token limit",
        )
    )


def _verify_pair_difference_with_context_retry(
    client: OpenAIComposedDataClient,
    *,
    proposal: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    reference_clip_path: str,
    target_clip_path: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    try:
        verification, raw_output = client.verify_pair_difference(
            proposal=proposal,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            reference_clip_path=reference_clip_path,
            target_clip_path=target_clip_path,
        )
        return verification, raw_output, False
    except Exception as exc:
        if not _is_verification_context_limit_error(exc):
            raise
        first_error = f"{type(exc).__name__}: {exc}"
        try:
            verification, retry_raw_output = client.verify_pair_difference(
                proposal=proposal,
                reference_annotation=reference_annotation,
                target_annotation=target_annotation,
                reference_clip_path=None,
                target_clip_path=None,
            )
        except Exception as retry_exc:
            raise RuntimeError(
                "annotation-only verification retry failed after video verification "
                f"context error: {first_error}; retry error: {type(retry_exc).__name__}: {retry_exc}"
            ) from retry_exc
        return (
            verification,
            {
                "video_verification_error": first_error,
                "annotation_only_retry_used": True,
                "annotation_only_retry": retry_raw_output,
            },
            True,
        )


def _fallback_pair_verification(*, reason: str) -> dict[str, Any]:
    return {
        "caption_delta": {
            "caption_equivalent": True,
            "has_concrete_difference": False,
            "difference_matches_edit": False,
            "concrete_differences": [],
            "reason": f"pair verification fallback: {reason}",
        },
        "edit_projection": {
            "projected_target_caption": "",
            "target_matches_projection": False,
            "score": 0.0,
            "missing_requirements": ["verification unavailable"],
            "reason": f"pair verification fallback: {reason}",
        },
        "edit_necessity": {
            "edit_needed": False,
            "reference_satisfies_edit": False,
            "target_satisfies_edit": False,
            "score": 0.0,
            "reason": f"pair verification fallback: {reason}",
        },
    }


def _fallback_audio_anchor_visual_verification(*, reason: str) -> dict[str, Any]:
    return {
        "accept": False,
        "reject_reason": f"audio-anchor visual verification fallback: {reason}",
        "recommended_edit_text": "",
        "visual_delta_type": "",
        "visual_delta_strength": 0.0,
        "near_duplicate_risk": 1.0,
        "reference_satisfies_edit": False,
        "target_satisfies_edit": False,
        "caption_equivalent": True,
        "order_only_scene_reorder": False,
        "weak_synonym_or_wording_delta": False,
        "evidence": [],
    }


def _apply_audio_anchor_visual_quality(quality: dict[str, Any], review: dict[str, Any]) -> None:
    quality["audio_matters_line"] = "visual_edit_audio_anchor"
    quality["omni_visual_accept"] = 1.0 if _boolish(review.get("accept")) else 0.0
    quality["omni_reject_reason"] = str(review.get("reject_reason", "")).strip()
    quality["visual_delta_type"] = str(review.get("visual_delta_type", "")).strip()
    quality["visual_delta_strength"] = _score_float(review.get("visual_delta_strength"))
    quality["near_duplicate_risk"] = _score_float(review.get("near_duplicate_risk"))
    quality["reference_satisfies_edit"] = 1.0 if _boolish(review.get("reference_satisfies_edit")) else 0.0
    quality["target_satisfies_edit"] = 1.0 if _boolish(review.get("target_satisfies_edit")) else 0.0
    quality["caption_equivalent"] = 1.0 if _boolish(review.get("caption_equivalent")) else 0.0
    quality["order_only_scene_reorder"] = 1.0 if _boolish(review.get("order_only_scene_reorder")) else 0.0
    quality["weak_synonym_or_wording_delta"] = 1.0 if _boolish(review.get("weak_synonym_or_wording_delta")) else 0.0


def _finalize_pair_verification(verification: dict[str, Any]) -> dict[str, Any]:
    caption_delta = dict(verification.get("caption_delta", {}))
    edit_projection = dict(verification.get("edit_projection", {}))
    edit_necessity = dict(verification.get("edit_necessity", {}))
    edit_text_quality_check = dict(verification.get("edit_text_quality_check", {}))
    normalized = {
        "caption_delta": {
            "caption_equivalent": _boolish(caption_delta.get("caption_equivalent")),
            "has_concrete_difference": _boolish(caption_delta.get("has_concrete_difference")),
            "difference_matches_edit": _boolish(caption_delta.get("difference_matches_edit")),
            "concrete_differences": _normalize_list(caption_delta.get("concrete_differences", [])),
            "reason": str(caption_delta.get("reason", "")).strip(),
        },
        "edit_projection": {
            "projected_target_caption": str(edit_projection.get("projected_target_caption", "")).strip(),
            "target_matches_projection": _boolish(edit_projection.get("target_matches_projection")),
            "score": _score_float(edit_projection.get("score")),
            "missing_requirements": _normalize_list(edit_projection.get("missing_requirements", [])),
            "reason": str(edit_projection.get("reason", "")).strip(),
        },
        "edit_necessity": {
            "edit_needed": _boolish(edit_necessity.get("edit_needed")),
            "reference_satisfies_edit": _boolish(edit_necessity.get("reference_satisfies_edit")),
            "target_satisfies_edit": _boolish(edit_necessity.get("target_satisfies_edit")),
            "score": _score_float(edit_necessity.get("score")),
            "reason": str(edit_necessity.get("reason", "")).strip(),
        },
        "edit_text_quality_check": {
            "not_caption_like": _boolish(edit_text_quality_check.get("not_caption_like", True)),
            "matches_modality": _boolish(edit_text_quality_check.get("matches_modality", True)),
            "single_primary_difference": _boolish(edit_text_quality_check.get("single_primary_difference", True)),
            "reference_does_not_satisfy": _boolish(edit_text_quality_check.get("reference_does_not_satisfy", True)),
            "target_satisfies": _boolish(edit_text_quality_check.get("target_satisfies", True)),
            "score": _score_float(edit_text_quality_check.get("score", 1.0)),
            "failure_reason": str(edit_text_quality_check.get("failure_reason", "")).strip(),
        },
    }
    _apply_verification_semantic_rejections(normalized)
    normalized["passed"] = _verification_accepts(normalized)
    normalized["failures"] = _verification_failures(normalized)
    return normalized


def _apply_verification_semantic_rejections(verification: dict[str, Any]) -> None:
    if not _verification_describes_order_only_difference(verification):
        return
    reason = "same content appears in a different shot/order sequence, not an edit-required target difference"
    caption_delta = verification.setdefault("caption_delta", {})
    caption_delta["caption_equivalent"] = True
    caption_delta["has_concrete_difference"] = False
    caption_delta["difference_matches_edit"] = False
    caption_delta["reason"] = _append_reason(caption_delta.get("reason"), reason)

    edit_projection = verification.setdefault("edit_projection", {})
    edit_projection["target_matches_projection"] = False
    edit_projection["score"] = min(_score_float(edit_projection.get("score")), 0.0)
    edit_projection["reason"] = _append_reason(edit_projection.get("reason"), reason)

    edit_necessity = verification.setdefault("edit_necessity", {})
    edit_necessity["edit_needed"] = False
    edit_necessity["reference_satisfies_edit"] = True
    edit_necessity["target_satisfies_edit"] = True
    edit_necessity["score"] = min(_score_float(edit_necessity.get("score")), 0.0)
    edit_necessity["reason"] = _append_reason(edit_necessity.get("reason"), reason)


def _verification_describes_order_only_difference(verification: dict[str, Any]) -> bool:
    text_parts: list[str] = []
    for section_name in ("caption_delta", "edit_projection", "edit_necessity"):
        section = verification.get(section_name, {})
        if not isinstance(section, dict):
            continue
        text_parts.append(str(section.get("reason", "")))
        text_parts.append(str(section.get("projected_target_caption", "")))
        text_parts.extend(_normalize_list(section.get("concrete_differences", [])))
        text_parts.extend(_normalize_list(section.get("missing_requirements", [])))
    text = _normalized_phrase(" ".join(text_parts))
    if not text:
        return False
    order_markers = (
        "different order",
        "different sequence",
        "order differs",
        "sequence differs",
        "reordered",
        "reverse order",
        "reversed order",
        "shot order",
        "sequence order",
        "temporal order",
        "just the order",
        "only the order",
        "只是顺序",
        "顺序不同",
        "镜头顺序",
    )
    has_order_marker = any(marker in text for marker in order_markers)
    if not has_order_marker:
        return False
    same_content_markers = (
        "same shots",
        "same elements",
        "same scenes",
        "same content",
        "both videos",
        "both clips",
        "both contain",
        "both show",
        "only",
        "just",
        "merely",
        "相同",
        "只是",
    )
    return any(marker in text for marker in same_content_markers)


def _append_reason(existing: Any, reason: str) -> str:
    existing_text = str(existing or "").strip()
    if not existing_text:
        return reason
    if reason in existing_text:
        return existing_text
    return f"{existing_text} {reason}"


def _finalize_pair_judge(judge: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(judge)
    accepted = _judge_accepts(normalized)
    if accepted:
        normalized["reject_reason"] = ""
        return normalized
    normalized["reject_reason"] = _compose_reject_reason(normalized)
    return normalized


def _effective_pair_quality(
    judge: dict[str, Any],
    verification: dict[str, Any] | None,
    heuristic_quality: dict[str, Any] | None,
) -> dict[str, float]:
    heuristic_quality = heuristic_quality or {}
    verification_edit_score = 0.0
    verification_accepted = verification is not None and _verification_accepts(verification)
    if verification_accepted:
        edit_projection = verification.get("edit_projection", {})
        edit_necessity = verification.get("edit_necessity", {})
        verification_edit_score = min(
            _score_float(edit_projection.get("score")),
            _score_float(edit_necessity.get("score")),
        )
    if "difference_strength_score" in heuristic_quality:
        difference_strength_score = _score_float(heuristic_quality.get("difference_strength_score"))
    else:
        difference_strength_score = MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE if verification_accepted else 0.0
    result: dict[str, Any] = {
        "same_context_score": max(
            _score_float(judge.get("same_context_score")),
            _score_float(heuristic_quality.get("same_context_score")),
        ),
        "edit_match_score": max(
            _score_float(judge.get("edit_match_score")),
            verification_edit_score,
        ),
        "target_uniqueness_score": max(
            _score_float(judge.get("target_uniqueness_score")),
            _score_float(heuristic_quality.get("target_uniqueness_score")),
        ),
        "difference_strength_score": difference_strength_score,
    }
    if "visual_near_duplicate_score" in heuristic_quality:
        result["visual_near_duplicate_score"] = _score_float(heuristic_quality.get("visual_near_duplicate_score"))
    if "difference_type" in heuristic_quality:
        result["difference_type"] = str(heuristic_quality.get("difference_type", "")).strip()
    if "action_evidence_score" in heuristic_quality:
        result["action_evidence_score"] = _score_float(heuristic_quality.get("action_evidence_score"))
    if "speech_evidence_score" in heuristic_quality:
        result["speech_evidence_score"] = _score_float(heuristic_quality.get("speech_evidence_score"))
    if "speech_specificity_score" in heuristic_quality:
        result["speech_specificity_score"] = _score_float(heuristic_quality.get("speech_specificity_score"))
    if "speech_transcript_backed" in heuristic_quality:
        result["speech_transcript_backed"] = _score_float(heuristic_quality.get("speech_transcript_backed"))
    if "non_speech_audio_event_score" in heuristic_quality:
        result["non_speech_audio_event_score"] = _score_float(
            heuristic_quality.get("non_speech_audio_event_score")
        )
    if "has_audio_modality" in heuristic_quality:
        result["has_audio_modality"] = _score_float(heuristic_quality.get("has_audio_modality"))
    for key in (
        "edit_text_quality_score",
        "edit_text_is_imperative",
        "edit_text_matches_difference_type",
        "edit_text_single_change",
        "edit_text_not_caption_like",
        "edit_text_no_modality_leakage",
        "observable_difference_passed",
        "observable_difference_frame_backed",
        "near_duplicate_without_delta",
        "synthetic_context_override",
        "audio_primary_allowed",
        "audio_anchor_required",
        "audio_anchor_score",
        "audio_anchor_context_score",
        "audio_anchor_min_rms",
        "visual_competing_delta_score",
        "competing_difference_passed",
        "intraclip_change_conflict",
        "audio_event_independent_evidence_passed",
        "audio_event_too_similar",
    ):
        if key in heuristic_quality:
            result[key] = _score_float(heuristic_quality.get(key))
    for key in ("dominant_delta_type", "dominant_delta_decision", "edit_primary_modality", "acceptance_profile"):
        if key in heuristic_quality:
            result[key] = heuristic_quality[key]
    return result


def _speech_quality_payload(quality: dict[str, Any]) -> dict[str, Any]:
    if str(quality.get("difference_type", "")).strip() != "speech":
        return {}
    return {
        "transcript_backed": _score_float(quality.get("speech_transcript_backed")) >= 1.0,
        "evidence_score": _score_float(quality.get("speech_evidence_score")),
        "specificity_score": _score_float(quality.get("speech_specificity_score")),
        "audio_required": _score_float(quality.get("has_audio_modality")) >= 1.0,
    }


def _audio_event_quality_payload(quality: dict[str, Any]) -> dict[str, Any]:
    if str(quality.get("difference_type", "")).strip() != "audio_event":
        return {}
    return {
        "non_speech_score": _score_float(quality.get("non_speech_audio_event_score")),
        "audio_required": _score_float(quality.get("has_audio_modality")) >= 1.0,
        "audio_primary_allowed": _score_float(quality.get("audio_primary_allowed", 1.0)) >= 1.0,
        "visual_competing_delta_score": _score_float(quality.get("visual_competing_delta_score")),
    }


def _compose_reject_reason(
    judge: dict[str, Any],
    verification: dict[str, Any] | None = None,
    effective_quality: dict[str, Any] | None = None,
) -> str:
    original_reason = str(judge.get("reject_reason", "")).strip()
    failures: list[str] = []
    if judge.get("reference_satisfies_edit"):
        failures.append("reference already satisfies the edit")
    if not judge.get("target_satisfies_edit"):
        failures.append("target does not satisfy the edit")
    if not judge.get("single_main_difference"):
        failures.append("the pair does not contain a single main difference")
    hard_negative_quality = str(judge.get("hard_negative_quality", "")).strip().lower()
    if hard_negative_quality not in {"good", "weak"}:
        failures.append(f"hard_negative_quality is {hard_negative_quality or 'bad'}")

    quality = effective_quality or judge
    acceptance_profile = _normalize_acceptance_profile(str(quality.get("acceptance_profile", DEFAULT_ACCEPTANCE_PROFILE)))
    same_context_score = _score_float(quality.get("same_context_score"))
    same_context_threshold = _profile_threshold(acceptance_profile, "same_context_score")
    if same_context_score < same_context_threshold:
        failures.append(
            f"same_context_score {same_context_score:.3f} is below {same_context_threshold:.2f}"
        )
    edit_match_score = _score_float(quality.get("edit_match_score"))
    edit_match_threshold = _profile_threshold(acceptance_profile, "edit_match_score")
    if edit_match_score < edit_match_threshold:
        failures.append(
            f"edit_match_score {edit_match_score:.3f} is below {edit_match_threshold:.2f}"
        )
    target_uniqueness_score = _score_float(quality.get("target_uniqueness_score"))
    target_uniqueness_threshold = _profile_threshold(acceptance_profile, "target_uniqueness_score")
    if target_uniqueness_score < target_uniqueness_threshold:
        failures.append(
            f"target_uniqueness_score {target_uniqueness_score:.3f} is below {target_uniqueness_threshold:.2f}"
        )
    if "difference_strength_score" in quality:
        difference_strength_score = _score_float(quality.get("difference_strength_score"))
    else:
        difference_strength_score = _profile_threshold(acceptance_profile, "difference_strength_score")
    difference_strength_threshold = _profile_threshold(acceptance_profile, "difference_strength_score")
    if difference_strength_score < difference_strength_threshold:
        failures.append(
            f"difference_strength_score {difference_strength_score:.3f} is below {difference_strength_threshold:.2f}"
        )
    visual_near_duplicate_score = _score_float(quality.get("visual_near_duplicate_score"))
    difference_type = str(quality.get("difference_type", "")).strip()
    if _visual_near_duplicate_rejects(visual_near_duplicate_score, difference_type):
        failures.append(
            f"visual_near_duplicate_score {visual_near_duplicate_score:.3f} is too high for visual difference type {difference_type}"
        )
    if difference_type == "action":
        action_evidence_score = _score_float(quality.get("action_evidence_score"))
        action_threshold = _profile_threshold(acceptance_profile, "action_evidence_score")
        if action_evidence_score < action_threshold:
            failures.append(
                f"action_evidence_score {action_evidence_score:.3f} is below {action_threshold:.2f}"
            )
    if difference_type == "speech":
        if _score_float(quality.get("has_audio_modality")) < 1.0:
            failures.append("speech edit is missing audio modality")
        if not _boolish(judge.get("audio_required")):
            failures.append("speech edit must be marked audio_required")
        if _score_float(quality.get("speech_transcript_backed")) < 1.0:
            failures.append("speech edit is not backed by transcript evidence on both clips")
        speech_evidence_score = _score_float(quality.get("speech_evidence_score"))
        if speech_evidence_score < MIN_ACCEPT_SPEECH_EVIDENCE_SCORE:
            failures.append(
                f"speech_evidence_score {speech_evidence_score:.3f} is below {MIN_ACCEPT_SPEECH_EVIDENCE_SCORE:.2f}"
            )
        speech_specificity_score = _score_float(quality.get("speech_specificity_score"))
        if speech_specificity_score < MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE:
            failures.append(
                f"speech_specificity_score {speech_specificity_score:.3f} is below {MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE:.2f}"
            )
    if difference_type == "audio_event":
        if "audio_primary_allowed" in quality and _score_float(quality.get("audio_primary_allowed")) < 1.0:
            decision = quality.get("dominant_delta_decision", {})
            reason = str(decision.get("reason", "")).strip() if isinstance(decision, dict) else ""
            failures.append(
                "audio_event cannot be the primary edit"
                + (f": {reason}" if reason else "")
            )
        non_speech_audio_event_score = _score_float(quality.get("non_speech_audio_event_score"))
        audio_threshold = _profile_threshold(acceptance_profile, "non_speech_audio_event_score")
        if non_speech_audio_event_score < audio_threshold:
            failures.append(
                f"non_speech_audio_event_score {non_speech_audio_event_score:.3f} is below {audio_threshold:.2f}"
            )
    if _score_float(quality.get("intraclip_change_conflict")) >= 1.0:
        failures.append("the proposed edit appears to describe an intra-clip transition instead of a cross-clip difference")
    failures.extend(_structured_edit_text_failures(quality))
    if not _uses_soft_local_gate_profile(acceptance_profile):
        if _observable_difference_rejects(quality):
            failures.append("observable_difference gate found no concrete visual delta evidence")
        failures.extend(_natural_pair_quality_failures(quality))
        if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
            failures.append("single_main_difference failed: competing stronger difference")
        if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
            failures.append("audio_event lacks independent non-speech audio evidence")
    elif _is_audio_matters_profile(acceptance_profile):
        audio_anchor_score = _score_float(quality.get("audio_anchor_score"))
        audio_anchor_threshold = _profile_threshold(acceptance_profile, "audio_anchor_score")
        if audio_anchor_score < audio_anchor_threshold:
            failures.append(
                f"audio_anchor_score {audio_anchor_score:.3f} is below {audio_anchor_threshold:.2f}"
            )
        if "omni_visual_accept" in quality and _score_float(quality.get("omni_visual_accept")) < 1.0:
            reason = str(quality.get("omni_reject_reason", "")).strip()
            failures.append("audio-anchor visual verifier rejected the pair" + (f": {reason}" if reason else ""))
        if _score_float(quality.get("caption_equivalent")) >= 1.0:
            failures.append("caption_delta says reference and target are equivalent")
        if _score_float(quality.get("order_only_scene_reorder")) >= 1.0:
            failures.append("the proposed scene edit is only a reordered sequence of shared shots")
        if _score_float(quality.get("weak_synonym_or_wording_delta")) >= 1.0:
            failures.append("the proposed edit is a weak synonym or wording-only visual delta")
        if _score_float(quality.get("reference_satisfies_edit")) >= 1.0:
            failures.append("reference already satisfies the audio-anchor visual edit")
        if "target_satisfies_edit" in quality and _score_float(quality.get("target_satisfies_edit")) < 1.0:
            failures.append("target does not satisfy the audio-anchor visual edit")
        if "visual_delta_strength" in quality:
            visual_delta_strength = _score_float(quality.get("visual_delta_strength"))
            visual_delta_threshold = _profile_threshold(acceptance_profile, "visual_delta_strength")
            if visual_delta_strength < visual_delta_threshold:
                failures.append(
                    f"visual_delta_strength {visual_delta_strength:.3f} is below {visual_delta_threshold:.2f}"
                )
        if "near_duplicate_risk" in quality:
            near_duplicate_risk = _score_float(quality.get("near_duplicate_risk"))
            near_duplicate_threshold = _profile_threshold(acceptance_profile, "near_duplicate_risk")
            if near_duplicate_risk > near_duplicate_threshold:
                failures.append(
                    f"near_duplicate_risk {near_duplicate_risk:.3f} is above {near_duplicate_threshold:.2f}"
                )
    if verification is not None:
        failures.extend(_verification_failures(verification))
    if not judge.get("accept"):
        failures.append("the model judge did not accept the pair")

    unique_failures: list[str] = []
    for failure in failures:
        if failure not in unique_failures:
            unique_failures.append(failure)
    if original_reason and unique_failures:
        return f"{original_reason} Final gate check: {'; '.join(unique_failures)}."
    if original_reason:
        return original_reason
    if unique_failures:
        return "; ".join(unique_failures)
    return "the pair was rejected without a structured reason from the judge"


def _audio_matters_visual_difference_type(quality: dict[str, Any]) -> str:
    difference_type = str(quality.get("difference_type", "")).strip()
    if difference_type:
        return difference_type
    dominant_delta_type = str(quality.get("dominant_delta_type", "")).strip()
    return dominant_delta_type


def _audio_matters_base_quality_accepts(quality: dict[str, Any]) -> bool:
    difference_type = _audio_matters_visual_difference_type(quality)
    if difference_type not in DOMINANT_VISUAL_DIFFERENCE_TYPES:
        return False
    if difference_type in FINAL_DISABLED_DIFFERENCE_TYPES:
        return False
    if _score_float(quality.get("audio_anchor_required")) < 1.0:
        return False
    if _score_float(quality.get("audio_anchor_score")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "audio_anchor_score"):
        return False
    if str(quality.get("edit_primary_modality", "visual")).strip() not in {"", "visual"}:
        return False
    if _score_float(quality.get("same_context_score")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "same_context_score"):
        return False
    if _score_float(quality.get("edit_match_score")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "edit_match_score"):
        return False
    if _score_float(quality.get("target_uniqueness_score")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "target_uniqueness_score"):
        return False
    if (
        "difference_strength_score" in quality
        and _score_float(quality.get("difference_strength_score")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "difference_strength_score")
    ):
        return False
    if difference_type == "action" and _score_float(quality.get("action_evidence_score")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "action_evidence_score"):
        return False
    if _visual_near_duplicate_rejects(
        _score_float(quality.get("visual_near_duplicate_score")),
        difference_type,
    ):
        return False
    if _score_float(quality.get("intraclip_change_conflict")) >= 1.0:
        return False
    if _structured_edit_text_failures(quality):
        return False
    return True


def _audio_matters_verification_accepts(verification: dict[str, Any]) -> bool:
    caption_delta = verification.get("caption_delta", {})
    edit_projection = verification.get("edit_projection", {})
    edit_necessity = verification.get("edit_necessity", {})
    edit_text_quality_check = verification.get("edit_text_quality_check", {})
    edit_threshold = _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "edit_target_alignment_score")
    necessity_threshold = _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "edit_necessity_score")
    if not bool(
        not _boolish(caption_delta.get("caption_equivalent"))
        and _boolish(caption_delta.get("has_concrete_difference"))
        and _boolish(caption_delta.get("difference_matches_edit"))
        and _boolish(edit_projection.get("target_matches_projection"))
        and _score_float(edit_projection.get("score")) >= edit_threshold
        and _boolish(edit_necessity.get("edit_needed"))
        and not _boolish(edit_necessity.get("reference_satisfies_edit"))
        and _boolish(edit_necessity.get("target_satisfies_edit"))
        and _score_float(edit_necessity.get("score")) >= necessity_threshold
    ):
        return False
    if not isinstance(edit_text_quality_check, dict) or not edit_text_quality_check:
        return True
    return bool(
        _boolish(edit_text_quality_check.get("not_caption_like"))
        and _boolish(edit_text_quality_check.get("matches_modality"))
        and _boolish(edit_text_quality_check.get("single_primary_difference"))
        and _boolish(edit_text_quality_check.get("reference_does_not_satisfy"))
        and _boolish(edit_text_quality_check.get("target_satisfies"))
        and _score_float(edit_text_quality_check.get("score")) >= _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "edit_text_quality_score")
    )


def _audio_matters_visual_review_accepts(quality: dict[str, Any]) -> bool:
    if "omni_visual_accept" not in quality:
        return True
    if _score_float(quality.get("omni_visual_accept")) < 1.0:
        return False
    if _score_float(quality.get("reference_satisfies_edit")) >= 1.0:
        return False
    if _score_float(quality.get("target_satisfies_edit")) < 1.0:
        return False
    if _score_float(quality.get("caption_equivalent")) >= 1.0:
        return False
    if _score_float(quality.get("order_only_scene_reorder")) >= 1.0:
        return False
    if _score_float(quality.get("weak_synonym_or_wording_delta")) >= 1.0:
        return False
    if _score_float(quality.get("visual_delta_strength")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "visual_delta_strength"):
        return False
    if _score_float(quality.get("near_duplicate_risk")) > _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "near_duplicate_risk"):
        return False
    return True


def _audio_matters_judge_accepts(
    judge: dict[str, Any],
    verification: dict[str, Any] | None,
    quality: dict[str, Any],
) -> bool:
    hard_negative_quality = str(judge.get("hard_negative_quality", "")).strip().lower()
    if not _boolish(judge.get("accept")):
        return False
    if not _boolish(judge.get("single_main_difference")):
        return False
    if _boolish(judge.get("reference_satisfies_edit")) or not _boolish(judge.get("target_satisfies_edit")):
        return False
    if hard_negative_quality not in {"good", "weak"}:
        return False
    if not _audio_matters_base_quality_accepts(quality):
        return False
    if not _audio_matters_visual_review_accepts(quality):
        return False
    if verification is None:
        return True
    return _audio_matters_verification_accepts(verification)


def _audio_matters_should_skip_video_verification(
    judge: dict[str, Any],
    quality: dict[str, Any],
) -> bool:
    hard_negative_quality = str(judge.get("hard_negative_quality", "")).strip().lower()
    if not _boolish(judge.get("accept")):
        return True
    if not _boolish(judge.get("single_main_difference")):
        return True
    if _boolish(judge.get("reference_satisfies_edit")) or not _boolish(judge.get("target_satisfies_edit")):
        return True
    if hard_negative_quality not in {"good", "weak"}:
        return True
    return not _audio_matters_base_quality_accepts(quality)


def _judge_accepts(
    judge: dict[str, Any],
    verification: dict[str, Any] | None = None,
    effective_quality: dict[str, Any] | None = None,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> bool:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    quality = effective_quality or judge
    if _is_audio_matters_profile(acceptance_profile):
        return _audio_matters_judge_accepts(judge, verification, quality)
    if _is_exploration_profile(acceptance_profile):
        return _exploration_judge_accepts(judge, verification, quality)
    judge_accepted = bool(
        not judge.get("reference_satisfies_edit")
        and judge.get("target_satisfies_edit")
        and judge.get("single_main_difference")
        and judge.get("hard_negative_quality") in {"good", "weak"}
        and _score_float(quality.get("same_context_score")) >= MIN_ACCEPT_SAME_CONTEXT_SCORE
        and _score_float(quality.get("edit_match_score")) >= MIN_ACCEPT_EDIT_MATCH_SCORE
        and _score_float(quality.get("target_uniqueness_score")) >= MIN_ACCEPT_TARGET_UNIQUENESS_SCORE
        and (
            "difference_strength_score" not in quality
            or _score_float(quality.get("difference_strength_score")) >= MIN_ACCEPT_DIFFERENCE_STRENGTH_SCORE
        )
        and not _visual_near_duplicate_rejects(
            _score_float(quality.get("visual_near_duplicate_score")),
            str(quality.get("difference_type", "")).strip(),
        )
        and (
            str(quality.get("difference_type", "")).strip() != "action"
            or _score_float(quality.get("action_evidence_score")) >= MIN_ACCEPT_ACTION_EVIDENCE_SCORE
        )
        and (
            str(quality.get("difference_type", "")).strip() != "speech"
            or (
                _score_float(quality.get("has_audio_modality")) >= 1.0
                and _boolish(judge.get("audio_required"))
                and _score_float(quality.get("speech_transcript_backed")) >= 1.0
                and _score_float(quality.get("speech_evidence_score")) >= MIN_ACCEPT_SPEECH_EVIDENCE_SCORE
                and _score_float(quality.get("speech_specificity_score")) >= MIN_ACCEPT_SPEECH_SPECIFICITY_SCORE
            )
        )
        and (
            str(quality.get("difference_type", "")).strip() != "audio_event"
            or _score_float(quality.get("non_speech_audio_event_score")) >= MIN_ACCEPT_NON_SPEECH_AUDIO_EVENT_SCORE
        )
        and _score_float(quality.get("intraclip_change_conflict")) < 1.0
        and not _structured_edit_text_failures(quality)
        and not _observable_difference_rejects(quality)
        and not _natural_pair_quality_failures(quality)
        and _score_float(quality.get("competing_difference_passed", 1.0)) >= 1.0
        and _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) >= 1.0
    )
    if verification is None:
        return bool(judge.get("accept")) and judge_accepted
    return judge_accepted and _verification_accepts(verification)


def _exploration_judge_accepts(
    judge: dict[str, Any],
    verification: dict[str, Any] | None,
    quality: dict[str, Any],
) -> bool:
    difference_type = str(quality.get("difference_type", "")).strip()
    hard_negative_quality = str(judge.get("hard_negative_quality", "")).strip().lower()
    if difference_type in FINAL_DISABLED_DIFFERENCE_TYPES:
        return False
    if not bool(judge.get("accept")) and not _is_exploration_audio_speech_content_reject(judge, quality):
        return False
    if _boolish(judge.get("reference_satisfies_edit")) or not _boolish(judge.get("target_satisfies_edit")):
        return False
    if not _boolish(judge.get("single_main_difference")):
        return False
    if hard_negative_quality not in {"good", "weak"}:
        return False
    if _score_float(quality.get("same_context_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "same_context_score"):
        return False
    if _score_float(quality.get("edit_match_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "edit_match_score"):
        return False
    if _score_float(quality.get("target_uniqueness_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "target_uniqueness_score"):
        return False
    if (
        "difference_strength_score" in quality
        and _score_float(quality.get("difference_strength_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "difference_strength_score")
    ):
        return False
    if difference_type == "action" and _score_float(quality.get("action_evidence_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "action_evidence_score"):
        return False
    if difference_type == "audio_event":
        if "audio_primary_allowed" in quality and _score_float(quality.get("audio_primary_allowed")) < 1.0:
            return False
        if _score_float(quality.get("audio_event_too_similar")) >= 1.0:
            return False
        if _score_float(quality.get("non_speech_audio_event_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "non_speech_audio_event_score"):
            return False
    if _score_float(quality.get("intraclip_change_conflict")) >= 1.0:
        return False
    if "edit_text_is_imperative" in quality and _score_float(quality.get("edit_text_is_imperative")) < 1.0:
        return False
    if "edit_text_quality_score" in quality and _score_float(quality.get("edit_text_quality_score")) < _profile_threshold(EXPLORATION_ACCEPTANCE_PROFILE, "edit_text_quality_score"):
        return False
    return True


def _should_skip_pair_video_verification(
    judge: dict[str, Any],
    quality: dict[str, Any],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> bool:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    if _is_audio_matters_profile(acceptance_profile):
        return _audio_matters_should_skip_video_verification(judge, quality)
    hard_negative_quality = str(judge.get("hard_negative_quality", "")).strip().lower()
    if (
        not bool(judge.get("accept"))
        and not (
            _is_exploration_profile(acceptance_profile)
            and _is_exploration_audio_speech_content_reject(judge, quality)
        )
    ) or (
        _boolish(judge.get("reference_satisfies_edit"))
        or not _boolish(judge.get("target_satisfies_edit"))
        or not _boolish(judge.get("single_main_difference"))
        or hard_negative_quality not in {"good", "weak"}
    ):
        return True
    if _is_exploration_profile(acceptance_profile):
        difference_type = str(quality.get("difference_type", "")).strip()
        return bool(
            difference_type in FINAL_DISABLED_DIFFERENCE_TYPES
            or (difference_type == "audio_event" and "audio_primary_allowed" in quality and _score_float(quality.get("audio_primary_allowed")) < 1.0)
            or _score_float(quality.get("intraclip_change_conflict")) >= 1.0
        )
    if _structured_edit_text_failures(quality):
        return True
    if _observable_difference_rejects(quality):
        return True
    if _natural_pair_quality_failures(quality):
        return True
    if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
        return True
    if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
        return True
    return False


def _visual_near_duplicate_rejects(score: float, difference_type: str) -> bool:
    return bool(
        difference_type in VISUAL_DIFFERENCE_TYPES
        and score >= MAX_ACCEPT_VISUAL_NEAR_DUPLICATE_SCORE
    )


def _verification_accepts(verification: dict[str, Any]) -> bool:
    caption_delta = verification.get("caption_delta", {})
    edit_projection = verification.get("edit_projection", {})
    edit_necessity = verification.get("edit_necessity", {})
    edit_text_quality_check = verification.get("edit_text_quality_check", {})
    return bool(
        not _boolish(caption_delta.get("caption_equivalent"))
        and _boolish(caption_delta.get("has_concrete_difference"))
        and _boolish(caption_delta.get("difference_matches_edit"))
        and _boolish(edit_projection.get("target_matches_projection"))
        and _score_float(edit_projection.get("score")) >= MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE
        and _boolish(edit_necessity.get("edit_needed"))
        and not _boolish(edit_necessity.get("reference_satisfies_edit"))
        and _boolish(edit_necessity.get("target_satisfies_edit"))
        and _score_float(edit_necessity.get("score")) >= MIN_ACCEPT_EDIT_NECESSITY_SCORE
        and _verification_edit_text_quality_accepts(edit_text_quality_check)
    )


def _verification_failures(verification: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    caption_delta = verification.get("caption_delta", {})
    edit_projection = verification.get("edit_projection", {})
    edit_necessity = verification.get("edit_necessity", {})
    edit_text_quality_check = verification.get("edit_text_quality_check", {})
    observable_difference_failure = str(verification.get("observable_difference_failure", "")).strip()
    if observable_difference_failure:
        failures.append(observable_difference_failure)
    if _boolish(caption_delta.get("caption_equivalent")):
        failures.append("caption_delta says reference and target are equivalent")
    if not _boolish(caption_delta.get("has_concrete_difference")):
        failures.append("caption_delta found no concrete difference")
    if not _boolish(caption_delta.get("difference_matches_edit")):
        failures.append("caption_delta difference does not match the edit")
    projection_score = _score_float(edit_projection.get("score"))
    if not _boolish(edit_projection.get("target_matches_projection")):
        failures.append("edit_projection does not match the target")
    if projection_score < MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE:
        failures.append(
            f"edit_projection score {projection_score:.3f} is below {MIN_ACCEPT_EDIT_TARGET_ALIGNMENT_SCORE:.2f}"
        )
    necessity_score = _score_float(edit_necessity.get("score"))
    if not _boolish(edit_necessity.get("edit_needed")):
        failures.append("edit_necessity says the edit is not needed")
    if _boolish(edit_necessity.get("reference_satisfies_edit")):
        failures.append("edit_necessity says the reference already satisfies the edit")
    if not _boolish(edit_necessity.get("target_satisfies_edit")):
        failures.append("edit_necessity says the target does not satisfy the edit")
    if necessity_score < MIN_ACCEPT_EDIT_NECESSITY_SCORE:
        failures.append(
            f"edit_necessity score {necessity_score:.3f} is below {MIN_ACCEPT_EDIT_NECESSITY_SCORE:.2f}"
        )
    if not _verification_edit_text_quality_accepts(edit_text_quality_check):
        reason = str(edit_text_quality_check.get("failure_reason", "")).strip() if isinstance(edit_text_quality_check, dict) else ""
        failures.append(f"edit_text_quality_check failed{': ' + reason if reason else ''}")
    return failures


def _structured_edit_text_failures(quality: dict[str, Any]) -> list[str]:
    if "edit_text_quality_score" not in quality:
        return []
    failures: list[str] = []
    edit_text_quality_score = _score_float(quality.get("edit_text_quality_score"))
    if edit_text_quality_score < MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE:
        failures.append(
            f"edit_text_quality_score {edit_text_quality_score:.3f} is below {MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE:.2f}"
        )
    for key, label in (
        ("edit_text_is_imperative", "edit_text is not an imperative edit"),
        ("edit_text_matches_difference_type", "edit_text does not match the difference type"),
        ("edit_text_single_change", "edit_text does not describe a single primary change"),
        ("edit_text_not_caption_like", "edit_text is caption-like"),
        ("edit_text_no_modality_leakage", "edit_text leaks another modality"),
    ):
        if key in quality and _score_float(quality.get(key)) < 1.0:
            failures.append(label)
    return failures


def _natural_pair_quality_failures(quality: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, label in NATURAL_PAIR_GATE_LABELS.items():
        if _score_float(quality.get(key)) >= 1.0:
            failures.append(label)
    return failures


def _observable_difference_rejects(quality: dict[str, Any]) -> bool:
    difference_type = str(quality.get("difference_type", "")).strip()
    if difference_type not in VISUAL_DIFFERENCE_TYPES:
        return False
    if "observable_difference_passed" in quality and _score_float(quality.get("observable_difference_passed")) < 1.0:
        return True
    if "near_duplicate_without_delta" in quality and _score_float(quality.get("near_duplicate_without_delta")) >= 1.0:
        return True
    return False


def _verification_edit_text_quality_accepts(check: Any) -> bool:
    if not isinstance(check, dict) or not check:
        return True
    return bool(
        _boolish(check.get("not_caption_like"))
        and _boolish(check.get("matches_modality"))
        and _boolish(check.get("single_primary_difference"))
        and _boolish(check.get("reference_does_not_satisfy"))
        and _boolish(check.get("target_satisfies"))
        and _score_float(check.get("score")) >= MIN_ACCEPT_EDIT_TEXT_QUALITY_SCORE
    )


def _score_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "pass", "accept", "accepted"}


def _evidence_from_annotations(
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    *,
    difference_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_audio_terms = _non_speech_audio_terms(reference_annotation) or _normalize_list(reference_annotation.get("audio_events", []))
    target_audio_terms = _non_speech_audio_terms(target_annotation) or _normalize_list(target_annotation.get("audio_events", []))
    reference_actions = _action_terms_from_annotation(reference_annotation)
    target_actions = _action_terms_from_annotation(target_annotation)
    return {
        "reference_summary": str(reference_annotation.get("summary", "")).strip(),
        "target_summary": str(target_annotation.get("summary", "")).strip(),
        "reference_storyline": list(reference_annotation.get("storyline", [])),
        "target_storyline": list(target_annotation.get("storyline", [])),
        "reference_events": list(reference_annotation.get("events", [])),
        "target_events": list(target_annotation.get("events", [])),
        "reference_timeline_evidence": _timeline_evidence(reference_annotation),
        "target_timeline_evidence": _timeline_evidence(target_annotation),
        "reference_actions": reference_actions,
        "target_actions": target_actions,
        "action_change": _change_text(reference_actions, target_actions),
        "audio_change": _change_text(reference_audio_terms, target_audio_terms),
        "visible_text_change": _change_text(
            reference_annotation.get("visible_text") or reference_annotation.get("on_screen_text", []),
            target_annotation.get("visible_text") or target_annotation.get("on_screen_text", []),
        ),
        "difference_evidence": dict(difference_evidence or {}),
    }


def _change_text(left: Any, right: Any) -> str:
    left_values = _normalize_list(left)
    right_values = _normalize_list(right)
    if left_values == right_values:
        return ""
    return f"{'; '.join(left_values) or 'none'} -> { '; '.join(right_values) or 'none'}"


def _normalized_phrase(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def _text_mentions_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalized_phrase(phrase)
    if not normalized_phrase:
        return False
    return normalized_phrase in _normalized_phrase(text)


def _has_intraclip_change_description(text: str, from_value: str, to_value: str) -> bool:
    normalized_text = _normalized_phrase(text)
    if not normalized_text:
        return False
    if not _text_mentions_phrase(text, from_value) or not _text_mentions_phrase(text, to_value):
        return False
    return any(marker in normalized_text for marker in INTRACLIP_CHANGE_MARKERS)


def _annotation_difference_texts(annotation: dict[str, Any], difference_type: str) -> list[str]:
    if difference_type == "speech":
        return _speech_texts_from_annotation(annotation)
    if difference_type == "audio_event":
        values = _non_speech_audio_terms(annotation)
        values.extend(_normalize_list(annotation.get("detective_notes", [])))
        values.extend(_normalize_list(annotation.get("summary", "")))
        return values
    return []


def _has_intraclip_difference_conflict(
    *,
    difference: dict[str, Any],
    reference_caption: str,
    target_caption: str,
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> bool:
    difference_type = str(difference.get("type", "")).strip()
    if difference_type not in {"speech", "audio_event"}:
        return False
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if not from_value or not to_value:
        return False

    texts_to_check = [
        reference_caption,
        target_caption,
        *_annotation_difference_texts(reference_annotation, difference_type),
        *_annotation_difference_texts(target_annotation, difference_type),
    ]
    return any(_has_intraclip_change_description(text, from_value, to_value) for text in texts_to_check)


def _reject_record_with_acceptance_issues(record: dict[str, Any], acceptance_issues: list[str]) -> dict[str, Any]:
    updated = dict(record)
    judge = dict(updated.get("judge", {}))
    judge["accept"] = False
    judge["reject_reason"] = "; ".join(acceptance_issues)
    updated["judge"] = judge
    verification = dict(updated.get("verification", {})) if isinstance(updated.get("verification"), dict) else {}
    failures = [str(item) for item in verification.get("failures", []) if str(item).strip()]
    for issue in acceptance_issues:
        failure = f"acceptance gate failed: {issue}"
        if failure not in failures:
            failures.append(failure)
    verification["failures"] = failures
    verification["passed"] = False
    updated["verification"] = verification
    updated["accepted"] = False
    return updated


def _semantic_verdict_text(record: dict[str, Any], target_annotation: dict[str, Any]) -> str:
    verification = record.get("verification", {}) if isinstance(record.get("verification"), dict) else {}
    chunks: list[str] = [
        str(record.get("target_caption", "")),
        str(target_annotation.get("summary", "")),
        str(target_annotation.get("scene", "")),
        json.dumps(target_annotation.get("attributes", {}), ensure_ascii=False),
    ]
    chunks.extend(_normalize_list(target_annotation.get("subjects", [])))
    chunks.extend(_normalize_list(target_annotation.get("actions", [])))
    chunks.extend(_normalize_object_counts(target_annotation.get("object_counts", {})).keys())
    for section_name in ("caption_delta", "edit_projection", "edit_necessity"):
        section = verification.get(section_name, {}) if isinstance(verification.get(section_name), dict) else {}
        chunks.extend(
            [
                str(section.get("projected_target_caption", "")),
                str(section.get("reason", "")),
                " ".join(_normalize_list(section.get("concrete_differences", []))),
                " ".join(_normalize_list(section.get("missing_requirements", []))),
            ]
        )
    return _normalized_phrase(" ".join(chunks))


def _target_annotation_text(target_annotation: dict[str, Any]) -> str:
    chunks: list[str] = [
        str(target_annotation.get("summary", "")),
        str(target_annotation.get("scene", "")),
        json.dumps(target_annotation.get("attributes", {}), ensure_ascii=False),
    ]
    chunks.extend(_normalize_list(target_annotation.get("subjects", [])))
    chunks.extend(_normalize_list(target_annotation.get("actions", [])))
    chunks.extend(_normalize_object_counts(target_annotation.get("object_counts", {})).keys())
    return _normalized_phrase(" ".join(chunks))


def _semantic_target_markers(value: str, *, drop_tokens: set[str] | None = None) -> list[str]:
    drops = set(drop_tokens or set())
    markers: list[str] = []
    for token in TOKEN_PATTERN.findall(_normalized_phrase(value)):
        if token in STOPWORDS or token in drops or len(token) <= 2:
            continue
        if token not in markers:
            markers.append(token)
    return markers


def _semantic_preserve_markers(preserve_tokens: list[str]) -> list[str]:
    markers: list[str] = []
    for token in preserve_tokens:
        normalized = _normalized_phrase(token)
        for marker in VACE_SEMANTIC_PRESERVE_OBJECT_MARKERS:
            if marker in normalized and marker not in markers:
                markers.append(marker)
    return markers


def _semantic_missing_preserve_markers(text: str, preserve_tokens: list[str]) -> list[str]:
    return [
        marker
        for marker in _semantic_preserve_markers(preserve_tokens)
        if marker not in text
    ]


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _post_vace_semantic_family(record: dict[str, Any], post_vace_verdict: dict[str, Any]) -> str:
    explicit = _normalized_phrase(str(post_vace_verdict.get("semantic_gate_family", "")))
    if explicit:
        return explicit
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    exploration_family = _normalized_phrase(str(generation.get("exploration_family", "")))
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    edit_text = str(record.get("edit_text", "")).strip()
    edit_token = str(generation.get("edit_token") or record.get("edit_token") or "").strip()
    if _is_black_jacket_target(difference, edit_text, edit_token):
        return "black_jacket"
    if exploration_family.startswith("clothing") or _is_clothing_edit(difference, edit_text, edit_token):
        return "clothing"
    if exploration_family == "background_change" or str(difference.get("type", "")).strip() == "scene":
        return "background"
    if _is_existing_object_replacement(difference, edit_text):
        return "object_replacement"
    if _is_object_removal(difference, edit_text):
        return "object_removal"
    return ""


def _clothing_semantic_errors(record: dict[str, Any], target_annotation: dict[str, Any]) -> list[str]:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    edit_text = str(record.get("edit_text", "")).strip()
    edit_token = str(generation.get("edit_token") or record.get("edit_token") or "").strip()
    target_clothing = _video_edit_target_object(difference, edit_text, edit_token) or edit_token
    target_text = _target_annotation_text(target_annotation)
    errors: list[str] = []
    garment_markers = [token for token in _semantic_target_markers(target_clothing) if token in VACE_CLOTHING_OBJECT_MARKERS]
    descriptor_markers = [
        token
        for token in _semantic_target_markers(target_clothing, drop_tokens=VACE_CLOTHING_OBJECT_MARKERS | {"man", "woman", "person", "wearing", "solid", "deep", "bright"})
    ]
    if garment_markers and not all(marker in target_text for marker in garment_markers):
        errors.append("target_annotation_missing_target_clothing")
    if descriptor_markers and not any(marker in target_text for marker in descriptor_markers):
        errors.append("target_annotation_missing_target_clothing_descriptor")
    for marker in sorted(VACE_CLOTHING_FORBIDDEN_RESULT_MARKERS):
        if marker in target_text:
            errors.append(f"target_annotation_forbidden_clothing_result:{marker}")
    for marker in _semantic_missing_preserve_markers(target_text, _normalize_list(generation.get("preserve_tokens", []))):
        errors.append(f"target_annotation_missing_preserved_object:{marker}")
    return errors


def _background_semantic_errors(record: dict[str, Any], target_annotation: dict[str, Any]) -> list[str]:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    target_text = _target_annotation_text(target_annotation)
    target_markers = _semantic_target_markers(
        str(difference.get("to", "")).strip(),
        drop_tokens={"background", "room", "scene", "original"},
    )
    errors: list[str] = []
    missing_target_background = target_markers and not all(
        _background_marker_is_present(target_text, marker) for marker in target_markers
    )
    source_markers = _background_source_markers(record)
    retained_source_markers = [
        marker
        for marker in source_markers
        if _text_mentions_phrase(target_text, marker)
    ]
    overlay_only = _background_overlay_failure(target_text) and (missing_target_background or bool(retained_source_markers))
    if missing_target_background:
        _append_unique(errors, "target_annotation_missing_target_background")
        _append_unique(errors, "target_background_missing")
    if retained_source_markers:
        _append_unique(errors, "original_background_retained")
        for marker in retained_source_markers[:4]:
            _append_unique(errors, f"target_annotation_retains_source_background:{marker}")
        if any(marker in {"sunlit room", "indoor room", "same room", "window", "door"} for marker in retained_source_markers):
            _append_unique(errors, "background_not_replaced_original_room_still_visible")
    if overlay_only:
        _append_unique(errors, "futuristic_lab_only_blue_overlay")
    for marker in _semantic_missing_preserve_markers(target_text, _normalize_list(generation.get("preserve_tokens", []))):
        _append_unique(errors, f"target_annotation_missing_preserved_object:{marker}")
    if not any(error.startswith("target_annotation_missing_preserved_object:") for error in errors) and (
        missing_target_background or retained_source_markers or overlay_only
    ):
        _append_unique(errors, "subject_preserved_but_edit_failed")
    return errors


def _background_marker_is_present(target_text: str, marker: str) -> bool:
    marker_key = _normalized_phrase(marker)
    candidates = VACE_BACKGROUND_TARGET_SYNONYMS.get(marker_key, {marker_key})
    for candidate in candidates:
        if _background_marker_is_negated(target_text, candidate):
            return False
    return any(_text_mentions_phrase(target_text, candidate) for candidate in candidates)


def _background_marker_is_negated(target_text: str, marker: str) -> bool:
    marker_key = _normalized_phrase(marker)
    if not marker_key:
        return False
    if any(
        phrase in target_text
        for phrase in (
            f"no {marker_key}",
            f"not {marker_key}",
            f"without {marker_key}",
            f"missing {marker_key}",
            f"lacks {marker_key}",
            f"lack {marker_key}",
        )
    ):
        return True
    tokens = target_text.split()
    marker_tokens = marker_key.split()
    negators = {"no", "not", "without", "missing", "lacks", "lack"}
    for index, token in enumerate(tokens):
        if token not in negators:
            continue
        window = tokens[index + 1 : index + 6]
        for start in range(0, max(len(window) - len(marker_tokens) + 1, 0)):
            if window[start : start + len(marker_tokens)] == marker_tokens:
                return True
    return False


def _background_source_markers(record: dict[str, Any]) -> list[str]:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    raw_planner_output = (
        generation.get("raw_planner_output", {}) if isinstance(generation.get("raw_planner_output"), dict) else {}
    )
    chunks = [
        str(difference.get("from", "")),
        str(record.get("reference_caption", "")),
        str(record.get("source_prompt", "")),
        str(generation.get("source_prompt", "")),
        str(raw_planner_output.get("source_prompt", "")),
    ]
    chunks.extend(_normalize_list(generation.get("preserve_regions", [])))
    chunks.extend(_normalize_list(raw_planner_output.get("preserve_regions", [])))
    source_text = _normalized_phrase(" ".join(chunks))
    markers: list[str] = []
    for marker in sorted(VACE_BACKGROUND_ORIGINAL_SCENE_MARKERS, key=len, reverse=True):
        if _text_mentions_phrase(source_text, marker):
            _append_unique(markers, _normalized_phrase(marker))
    return markers


def _background_overlay_failure(target_text: str) -> bool:
    return any(_text_mentions_phrase(target_text, marker) for marker in VACE_BACKGROUND_OVERLAY_FAILURE_MARKERS)


def _object_edit_semantic_errors(
    record: dict[str, Any],
    target_annotation: dict[str, Any],
    *,
    removal_only: bool,
) -> list[str]:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    edit_text = str(record.get("edit_text", "")).strip()
    source_object = _video_edit_source_object(difference, edit_text)
    target_object = _video_edit_target_object(difference, edit_text, str(generation.get("edit_token", "")).strip())
    target_text = _target_annotation_text(target_annotation)
    errors: list[str] = []
    if source_object and _text_mentions_phrase(target_text, source_object):
        errors.append("target_annotation_still_contains_source_object")
    if not removal_only and target_object and not _text_mentions_phrase(target_text, target_object):
        errors.append("target_annotation_missing_target_object")
    for marker in _semantic_missing_preserve_markers(target_text, _normalize_list(generation.get("preserve_tokens", []))):
        errors.append(f"target_annotation_missing_preserved_object:{marker}")
    return errors


def _black_jacket_semantic_errors(record: dict[str, Any], target_annotation: dict[str, Any]) -> list[str]:
    text = _semantic_verdict_text(record, target_annotation)
    errors: list[str] = []
    if not ("black" in text and "jacket" in text):
        errors.append("target_annotation_missing_black_jacket")
    if "open" not in text:
        errors.append("target_annotation_missing_open_jacket_structure")
    if not any(marker in text for marker in ("long sleeve", "long sleeved", "long sleeves", "sleeves")):
        errors.append("target_annotation_missing_long_sleeves")
    for marker in ("dark shirt", "navy shirt", "polo", "patterned shirt"):
        if marker in text:
            errors.append(f"target_annotation_forbidden_marker:{marker}")
    for marker in ("ukulele", "microphone"):
        if marker not in text:
            errors.append(f"target_annotation_missing_preserved_object:{marker}")
    if not any(marker in text for marker in ("man", "person", "face")):
        errors.append("target_annotation_missing_preserved_subject")
    if not any(marker in text for marker in ("play", "plays", "playing")):
        errors.append("target_annotation_missing_preserved_ukulele_action")
    return errors


def _apply_post_vace_semantic_verdict(
    record: dict[str, Any],
    *,
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    post_vace_verdict = (
        generation.get("post_vace_verdict", {}) if isinstance(generation.get("post_vace_verdict"), dict) else {}
    )
    if not post_vace_verdict.get("semantic_gate_required"):
        return record
    semantic_family = _post_vace_semantic_family(record, post_vace_verdict)
    if not semantic_family:
        return record

    if semantic_family == "black_jacket":
        errors = _black_jacket_semantic_errors(record, target_annotation)
    elif semantic_family == "clothing":
        errors = _clothing_semantic_errors(record, target_annotation)
    elif semantic_family == "background":
        errors = _background_semantic_errors(record, target_annotation)
    elif semantic_family == "object_replacement":
        errors = _object_edit_semantic_errors(record, target_annotation, removal_only=False)
    elif semantic_family == "object_removal":
        errors = _object_edit_semantic_errors(record, target_annotation, removal_only=True)
    else:
        return record
    updated = dict(record)
    updated_generation = dict(generation)
    updated_verdict = dict(post_vace_verdict)
    updated_verdict.update(
        {
            "semantic_gate_family": semantic_family,
            "semantic_gate_checked_from": "omni_validation_annotation",
            "semantic_gate_passed": not errors,
            "semantic_gate_errors": errors,
        }
    )
    if errors:
        updated_verdict["stage"] = "failed_semantic_gate"
    else:
        updated_verdict["stage"] = "passed_semantic_gate"
    updated_generation["post_vace_verdict"] = updated_verdict
    updated["generation"] = updated_generation
    return updated


def _pair_record_acceptance_issues(
    *,
    root: Path,
    record: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> list[str]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    issues: list[str] = []
    for field_name in ("reference_video", "target_video"):
        raw_path = str(record.get(field_name, "")).strip()
        if raw_path and not _resolve_under_root(root, raw_path).exists():
            issues.append(f"{field_name} does not exist: {raw_path}")

    for negative_path in [str(item).strip() for item in record.get("hard_negatives", []) if str(item).strip()]:
        if not _resolve_under_root(root, negative_path).exists():
            issues.append(f"hard_negative does not exist: {negative_path}")

    if _has_intraclip_difference_conflict(
        difference=record.get("difference", {}),
        reference_caption=str(record.get("reference_caption", "")),
        target_caption=str(record.get("target_caption", "")),
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
    ):
        issues.append("the proposed difference appears inside a single clip instead of between reference and target")
    difference = record.get("difference", {})
    quality = record.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}
        record["quality"] = quality
    speech_issues, speech_warnings = _split_profiled_speech_content_issues(
        edit_text=str(record.get("edit_text", "")),
        difference=difference,
        acceptance_profile=acceptance_profile,
    )
    issues.extend(speech_issues)
    if speech_warnings:
        quality["exploration_warnings"] = _dedupe_strings(
            _normalize_list(quality.get("exploration_warnings", []))
            + [f"diagnostic_audio_speech_content: {issue}" for issue in speech_warnings]
        )
    if str(difference.get("type", "")).strip() == "audio_event":
        from_value = str(difference.get("from", "")).strip()
        to_value = str(difference.get("to", "")).strip()
        if _is_speech_only_audio_phrase(from_value) or _is_speech_only_audio_phrase(to_value):
            issue = "audio_event must not use speech-only or narration-only text as the main difference"
            if _is_exploration_profile(acceptance_profile):
                quality["exploration_warnings"] = _dedupe_strings(
                    _normalize_list(quality.get("exploration_warnings", []))
                    + [f"diagnostic_audio_speech_content: {issue}"]
                )
            else:
                issues.append(issue)
        dominant_delta_decision = _dominant_delta_decision(
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
            difference=difference,
            quality=quality,
            source_context=record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {},
        )
        quality["dominant_delta_type"] = dominant_delta_decision["dominant_type"]
        quality["audio_primary_allowed"] = 1.0 if dominant_delta_decision["audio_primary_allowed"] else 0.0
        quality["visual_competing_delta_score"] = dominant_delta_decision["visual_competing_delta_score"]
        quality["dominant_delta_decision"] = dominant_delta_decision
        record["dominant_delta_decision"] = dominant_delta_decision
        if not dominant_delta_decision["audio_primary_allowed"]:
            reason = str(dominant_delta_decision.get("reason", "")).strip()
            visual_types = ", ".join(dominant_delta_decision.get("visual_delta_types", []))
            suffix = f": {reason}" if reason else ""
            if visual_types:
                suffix += f" ({visual_types})"
            issues.append(f"audio_event cannot be the primary edit for this pair{suffix}")
    if isinstance(quality, dict):
        if _uses_soft_local_gate_profile(acceptance_profile):
            if "edit_text_is_imperative" in quality and _score_float(quality.get("edit_text_is_imperative")) < 1.0:
                issues.append("edit_text is not an imperative edit")
            if _score_float(quality.get("intraclip_change_conflict")) >= 1.0:
                issues.append("the proposed edit appears to describe an intra-clip transition instead of a cross-clip difference")
            if _is_audio_matters_profile(acceptance_profile):
                if _observable_difference_rejects(quality):
                    issues.append("observable_difference gate found no concrete visual delta evidence")
                if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
                    issues.append("single_main_difference failed: competing stronger difference")
                if "omni_visual_accept" in quality and _score_float(quality.get("omni_visual_accept")) < 1.0:
                    reason = str(quality.get("omni_reject_reason", "")).strip()
                    issues.append("audio-anchor visual verifier rejected the pair" + (f": {reason}" if reason else ""))
                if _score_float(quality.get("caption_equivalent")) >= 1.0:
                    issues.append("caption_delta says reference and target are equivalent")
                if _score_float(quality.get("order_only_scene_reorder")) >= 1.0:
                    issues.append("the proposed scene edit is only a reordered sequence of shared shots")
                if _score_float(quality.get("weak_synonym_or_wording_delta")) >= 1.0:
                    issues.append("the proposed edit is a weak synonym or wording-only visual delta")
                if _score_float(quality.get("reference_satisfies_edit")) >= 1.0:
                    issues.append("reference already satisfies the audio-anchor visual edit")
                if "target_satisfies_edit" in quality and _score_float(quality.get("target_satisfies_edit")) < 1.0:
                    issues.append("target does not satisfy the audio-anchor visual edit")
                if "visual_delta_strength" in quality and _score_float(quality.get("visual_delta_strength")) < _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "visual_delta_strength"):
                    issues.append("visual_delta_strength is below the audio-matters threshold")
                if "near_duplicate_risk" in quality and _score_float(quality.get("near_duplicate_risk")) > _profile_threshold(AUDIO_MATTERS_ACCEPTANCE_PROFILE, "near_duplicate_risk"):
                    issues.append("near_duplicate_risk is above the audio-matters threshold")
        else:
            issues.extend(_structured_edit_text_failures(quality))
            if _observable_difference_rejects(quality):
                issues.append("observable_difference gate found no concrete visual delta evidence")
            issues.extend(_natural_pair_quality_failures(quality))
            if _score_float(quality.get("competing_difference_passed", 1.0)) < 1.0:
                issues.append("single_main_difference failed: competing stronger difference")
            if _score_float(quality.get("audio_event_independent_evidence_passed", 1.0)) < 1.0:
                issues.append("audio_event lacks independent non-speech audio evidence")
    issues.extend(
        _synthetic_edit_record_issues(
            root=root,
            record=record,
            reference_annotation=reference_annotation,
            target_annotation=target_annotation,
        )
    )
    return issues


def _synthetic_edit_record_issues(
    *,
    root: Path,
    record: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> list[str]:
    if str(record.get("source_type", "natural")).strip() != "synthetic_edit":
        return []
    issues: list[str] = []
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    difference_type = str(record.get("difference", {}).get("type", "")).strip()
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    route = _synthetic_generation_route(generation)
    is_audio_route = _is_audio_synthetic_route(route)
    background_route = _background_replace_actual_route(generation)
    deterministic_background = background_route == DETERMINISTIC_BG_COMPOSITE_ROUTE
    source_context = record.get("source_context", {}) if isinstance(record.get("source_context"), dict) else {}
    relation = str(source_context.get("relation", "")).strip()
    visual_score = _score_float(quality.get("visual_near_duplicate_score"))
    if (
        relation == "synthetic_from_reference"
        and difference_type in VISUAL_DIFFERENCE_TYPES
        and not is_audio_route
        and visual_score < MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE
    ):
        issues.append(
            f"synthetic target does not preserve reference visual context: visual_near_duplicate_score {visual_score:.3f} is below {MIN_SYNTHETIC_VISUAL_CONTEXT_SCORE:.2f}"
        )
    if is_audio_route and visual_score < MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE:
        issues.append(
            f"audio synthetic target changed visual stream: visual_near_duplicate_score {visual_score:.3f} is below {MIN_SYNTHETIC_AUDIO_VISUAL_CONTEXT_SCORE:.2f}"
        )

    reference_path = _resolve_under_root(root, str(record.get("reference_video", "")).strip())
    target_path = _resolve_under_root(root, str(record.get("target_video", "")).strip())
    reference_media = probe_media(reference_path)
    target_media = probe_media(target_path)
    if "error" not in target_media and not target_media.get("has_video"):
        issues.append("synthetic target is missing a video stream")
    if "error" not in reference_media and "error" not in target_media:
        reference_duration = float(reference_media.get("duration_seconds") or 0.0)
        target_duration = float(target_media.get("duration_seconds") or 0.0)
        if reference_media.get("has_audio") and not target_media.get("has_audio"):
            issues.append("synthetic target is missing audio copied from the reference")
        if reference_duration > 0 and target_duration > 0:
            ratio = abs(reference_duration - target_duration) / reference_duration
            if ratio > 0.10:
                issues.append(
                    f"synthetic target duration drift {ratio:.3f} exceeds 0.10 from the reference"
                )
    postprocess = generation.get("postprocess", {}) if isinstance(generation.get("postprocess"), dict) else {}
    if (
        difference_type in VISUAL_DIFFERENCE_TYPES
        and not is_audio_route
        and "error" not in reference_media
        and reference_media.get("has_audio")
        and not postprocess.get("audio_copied_from_reference")
    ):
        issues.append("visual synthetic edits must record generation.postprocess.audio_copied_from_reference=true")
    if difference_type in VISUAL_DIFFERENCE_TYPES and not is_audio_route:
        plain_background_issue = _plain_background_replacement_vace_issue(record, generation)
        if plain_background_issue:
            issues.append(plain_background_issue)
        src_ref_requirements = (
            generation.get("src_ref_requirements", {}) if isinstance(generation.get("src_ref_requirements"), dict) else {}
        )
        src_ref_images = _normalize_list(generation.get("src_ref_images", []))
        if src_ref_requirements.get("required") and not src_ref_images:
            issues.append("visual synthetic VACE preflight missing required src_ref_images")
        for src_ref_image in src_ref_images:
            if not _resolve_under_root(root, src_ref_image).exists():
                issues.append(f"visual synthetic selected src_ref_image does not exist: {src_ref_image}")
        src_mask = str(generation.get("src_mask", "")).strip()
        if route == "vace_controlled" or deterministic_background:
            if not src_mask:
                issues.append("visual synthetic src_mask is required for masked visual routes")
            elif not _resolve_under_root(root, src_mask).exists():
                issues.append(f"visual synthetic src_mask does not exist: {src_mask}")
        src_video_for_vace = str(generation.get("src_video_for_vace", "")).strip()
        if route == "vace_controlled" or deterministic_background:
            if not src_video_for_vace:
                issues.append("visual synthetic src_video_for_vace is required for masked visual routes")
            elif not _resolve_under_root(root, src_video_for_vace).exists():
                issues.append(f"visual synthetic src_video_for_vace does not exist: {src_video_for_vace}")
        duration_metrics = generation.get("duration_metrics", {}) if isinstance(generation.get("duration_metrics"), dict) else {}
        duration_gate = duration_metrics.get("duration_gate", {}) if isinstance(duration_metrics.get("duration_gate"), dict) else {}
        if not duration_gate:
            issues.append("visual synthetic duration gate is required")
        elif not _boolish(duration_gate.get("passed")):
            gate_errors = _normalize_list(duration_gate.get("errors", []))
            issues.append(
                "visual synthetic duration gate failed"
                + (": " + "; ".join(gate_errors) if gate_errors else "")
            )
        max_drift = _score_float(duration_metrics.get("max_duration_drift_seconds", 0.5))
        for drift_field in ("raw_duration_drift_seconds", "target_duration_drift_seconds"):
            drift = _score_float(duration_metrics.get(drift_field))
            if drift > max_drift:
                issues.append(f"visual synthetic {drift_field} {drift:.3f}s exceeds {max_drift:.3f}s")
        post_vace_verdict = (
            generation.get("post_vace_verdict", {}) if isinstance(generation.get("post_vace_verdict"), dict) else {}
        )
        if post_vace_verdict.get("semantic_gate_required") and not _boolish(
            post_vace_verdict.get("semantic_gate_passed")
        ):
            issues.append("visual synthetic post-VACE semantic gate has not passed")
        if deterministic_background:
            deterministic_metrics = generation.get("deterministic_composite_metrics", {})
            if not isinstance(deterministic_metrics, dict) or not deterministic_metrics:
                issues.append("visual synthetic deterministic background route requires deterministic_composite_metrics")
    if is_audio_route:
        expected_event = _synthetic_audio_expected_event(record)
        if not expected_event:
            issues.append("audio_event target sound was not detected by audio observer: expected_event is missing")
        elif not _audio_terms_mention_event(_non_speech_audio_terms(target_annotation), expected_event):
            issues.append(f"audio_event target sound was not detected by audio observer: {expected_event}")
        elif _audio_terms_mention_event(_non_speech_audio_terms(reference_annotation), expected_event):
            issues.append(f"reference audio already contains requested audio event: {expected_event}")
    return issues


def _accepted_record_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float, str]:
    quality = record.get("quality", {})
    return (
        -_score_float(quality.get("difference_strength_score")),
        -_score_float(quality.get("same_context_score")),
        -_score_float(quality.get("target_uniqueness_score")),
        -_score_float(quality.get("edit_match_score")),
        str(record.get("proposal_id", "")).strip(),
    )


def _accepted_record_signature(record: dict[str, Any]) -> tuple[str, ...]:
    difference = record.get("difference", {})
    from_value = _normalized_phrase(str(difference.get("from", "")).strip())
    to_value = _normalized_phrase(str(difference.get("to", "")).strip())
    if not from_value and not to_value:
        from_value = _normalized_phrase(str(record.get("edit_text", "")).strip())
    if str(record.get("source_type", "natural")).strip() == "synthetic_edit":
        return (
            "synthetic_edit",
            str(record.get("proposal_id", "")).strip(),
            str(record.get("reference_video", "")).strip(),
            str(record.get("target_video", "")).strip(),
            str(difference.get("type", "")).strip(),
            from_value,
            to_value,
        )
    return (
        str(record.get("reference_video", "")).strip(),
        str(record.get("target_video", "")).strip(),
        str(difference.get("type", "")).strip(),
        from_value,
        to_value,
        str(record.get("source_context", {}).get("relation", "")).strip(),
    )


def _select_final_accepted_records(
    records: list[dict[str, Any]],
    *,
    max_accepted_pairs: int,
    acceptance_profile: str = DEFAULT_ACCEPTANCE_PROFILE,
) -> list[dict[str, Any]]:
    acceptance_profile = _normalize_acceptance_profile(acceptance_profile)
    accepted_candidates = sorted(
        [record for record in records if bool(record.get("accepted"))],
        key=_accepted_record_sort_key,
    )
    if not accepted_candidates or max_accepted_pairs <= 0:
        return []

    selected: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    selected_ids: set[str] = set()
    selected_target_videos: set[str] = set()
    selected_reference_videos: set[str] = set()
    bucket_targets = EXPLORATION_SMALL_ACCEPT_BUCKET_TARGETS if _is_exploration_profile(acceptance_profile) else FINAL_ACCEPT_BUCKET_TARGETS

    def bucket_key(record: dict[str, Any]) -> str:
        difference_type = str(record.get("difference", {}).get("type", "")).strip()
        if _is_exploration_profile(acceptance_profile) and difference_type in {"object_presence", "object_count"}:
            return "object"
        return difference_type

    def try_select(record: dict[str, Any]) -> bool:
        signature = _accepted_record_signature(record)
        proposal_id = str(record.get("proposal_id", "")).strip()
        reference_video = str(record.get("reference_video", "")).strip()
        target_video = str(record.get("target_video", "")).strip()
        difference_type = str(record.get("difference", {}).get("type", "")).strip()
        if difference_type in FINAL_DISABLED_DIFFERENCE_TYPES and str(record.get("source_type", "natural")).strip() != "synthetic_edit":
            return False
        if signature in seen_signatures or proposal_id in selected_ids:
            return False
        if _is_exploration_profile(acceptance_profile) and reference_video and reference_video in selected_reference_videos:
            return False
        if _is_exploration_profile(acceptance_profile) and difference_type == "audio_event":
            quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
            if "audio_primary_allowed" in quality and _score_float(quality.get("audio_primary_allowed")) < 1.0:
                return False
        if target_video and target_video in selected_target_videos:
            return False
        selected.append(record)
        seen_signatures.add(signature)
        selected_ids.add(proposal_id)
        if reference_video:
            selected_reference_videos.add(reference_video)
        if target_video:
            selected_target_videos.add(target_video)
        return True

    for target_bucket, target_count in bucket_targets.items():
        bucket_count = 0
        for record in accepted_candidates:
            if len(selected) >= max_accepted_pairs or bucket_count >= target_count:
                break
            if bucket_key(record) != target_bucket:
                continue
            if try_select(record):
                bucket_count += 1

    for record in accepted_candidates:
        if len(selected) >= max_accepted_pairs:
            break
        if _is_exploration_profile(acceptance_profile):
            target_bucket = bucket_key(record)
            if target_bucket not in bucket_targets:
                continue
            if sum(1 for item in selected if bucket_key(item) == target_bucket) >= bucket_targets[target_bucket]:
                continue
        try_select(record)

    return [_accepted_sample_from_record(record, index + 1) for index, record in enumerate(selected)]


def _select_single_source_quality_passed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_candidates = sorted(
        [
            record
            for record in records
            if bool(record.get("single_source_pair")) and bool(record.get("accepted"))
        ],
        key=_accepted_record_sort_key,
    )
    return [_accepted_sample_from_record(record, index + 1) for index, record in enumerate(accepted_candidates)]


def _accepted_sample_from_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    source_type = str(record.get("source_type", "natural")).strip() or "natural"
    if source_type == "synthetic_edit":
        identity = str(record.get("proposal_id") or record.get("target_video") or record.get("edit_text") or index)
        sample_id = f"covr_omni_synth_{_stable_hash(identity)[:8]}"
    else:
        sample_id = f"covr_omni_pilot_{index:04d}"
    return {
        "sample_id": sample_id,
        "proposal_id": record["proposal_id"],
        "reference_clip_id": record.get("reference_clip_id", ""),
        "target_clip_id": record.get("target_clip_id", ""),
        "reference_video": record["reference_video"],
        "target_video": record["target_video"],
        "source_type": source_type,
        "edit_text": record["edit_text"],
        "modalities": list(record["modalities"]),
        "reference_caption": record["reference_caption"],
        "target_caption": record["target_caption"],
        "difference": dict(record["difference"]),
        "audio_dataset_line": str(record.get("audio_dataset_line", "")).strip(),
        "audio_matters_line": str(record.get("audio_matters_line", "")).strip(),
        "hard_negatives": list(record["hard_negatives"]),
        "quality": dict(record["quality"]),
        "source": dict(record["source"]),
        "generation": dict(record.get("generation", {})),
        "source_context": dict(record.get("source_context", {})),
        "direction_corrected": bool(record.get("direction_corrected")),
        "evidence": dict(record.get("evidence", {})),
        "judge": dict(record.get("judge", {})),
        "verification": dict(record.get("verification", {})),
        "audio_anchor_visual_verification": dict(record.get("audio_anchor_visual_verification", {})),
        "edit_text_quality": dict(record.get("edit_text_quality", {})),
        "observable_difference": dict(record.get("observable_difference", {})),
        "dominant_delta_decision": dict(record.get("dominant_delta_decision", {})),
        "competing_difference": dict(record.get("competing_difference", {})),
        "audio_event_evidence": dict(record.get("audio_event_evidence", {})),
        "speech_quality": dict(record.get("speech_quality", {})),
        "audio_event_quality": dict(record.get("audio_event_quality", {})),
        "transcript_backed": record.get("transcript_backed"),
        "group_id": record.get("group_id", ""),
        "group_reason": record.get("group_reason", ""),
    }


def _build_source_metadata(
    *,
    root: Path,
    target_annotation: dict[str, Any],
    raw_index: dict[str, dict[str, Any]],
) -> dict[str, str]:
    asset_id = str(target_annotation.get("source_asset_id", "")).strip()
    raw_asset = raw_index.get(asset_id, {})
    platform = str(raw_asset.get("dataset") or "unknown").strip()
    raw_path = str(raw_asset.get("path", "")).strip()
    if raw_path:
        resolved_path = Path(raw_path)
    else:
        resolved_path = _resolve_under_root(root, target_annotation["output_path"])
    url = resolved_path.resolve().as_uri() if resolved_path.is_absolute() or resolved_path.exists() else ""
    return {
        "platform": platform or "unknown",
        "url": url,
        "license_note": DEFAULT_LICENSE_NOTE,
    }


def _load_raw_asset_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in _load_jsonl(path):
        asset_id = str(item.get("asset_id", "")).strip()
        if asset_id:
            mapping[asset_id] = item
    return mapping


def _load_records_by_key(path: Path, key_name: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in _load_jsonl(path):
        key = str(item.get(key_name, "")).strip()
        if key:
            mapping[key] = item
    return mapping


def _annotation_lookup(*, root: Path, annotations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        clip_id = str(annotation.get("clip_id", "")).strip()
        if clip_id:
            mapping[clip_id] = annotation
        output_path = str(annotation.get("output_path", "")).strip()
        if output_path:
            resolved = _resolve_under_root(root, output_path)
            for key in _path_lookup_keys(root, resolved, output_path):
                mapping.setdefault(key, annotation)
    return mapping


def _path_lookup_keys(root: Path, resolved_path: Path, raw_path: str | Path) -> list[str]:
    keys: list[str] = []

    def add(value: str) -> None:
        normalized = value.replace("\\", "/").strip()
        if normalized and normalized not in keys:
            keys.append(normalized)

    add(str(raw_path))
    add(str(resolved_path))
    try:
        add(str(resolved_path.resolve()))
    except OSError:
        pass
    try:
        add(resolved_path.resolve().relative_to(root.resolve()).as_posix())
    except (OSError, ValueError):
        pass
    return keys


def _annotation_for_video_edit_plan(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    record: dict[str, Any],
    video_field: str,
    caption_field: str,
) -> dict[str, Any]:
    raw_path = str(record.get(video_field, "")).strip()
    if raw_path:
        resolved = _resolve_under_root(root, raw_path)
        for key in _path_lookup_keys(root, resolved, raw_path):
            if key in lookup:
                return lookup[key]
    caption = str(record.get(caption_field, "")).strip()
    return {
        "clip_id": _safe_id(raw_path or caption or "unknown_clip"),
        "output_path": raw_path,
        "summary": caption,
        "subjects": [],
        "object_counts": {},
        "actions": [],
        "scene": "",
        "attributes": [],
        "visible_text": [],
        "speech": [],
        "audio_events": [],
        "modalities": ["visual"],
    }


def _review_annotation_for_record(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    record: dict[str, Any],
    video_field: str,
    clip_id_field: str,
) -> dict[str, Any]:
    clip_id = str(record.get(clip_id_field, "")).strip()
    if clip_id and clip_id in lookup:
        return lookup[clip_id]
    raw_path = str(record.get(video_field, "")).strip()
    if raw_path:
        resolved = _resolve_under_root(root, raw_path)
        for key in _path_lookup_keys(root, resolved, raw_path):
            if key in lookup:
                return lookup[key]
    return {}


def _manual_review_item_markdown(
    *,
    metadata: dict[str, Any],
    reference_filename: str,
    target_filename: str,
) -> str:
    difference = metadata.get("difference") if isinstance(metadata.get("difference"), dict) else {}
    verification = metadata.get("verification") if isinstance(metadata.get("verification"), dict) else {}
    observable = metadata.get("observable_difference") if isinstance(metadata.get("observable_difference"), dict) else {}
    competing = metadata.get("competing_difference") if isinstance(metadata.get("competing_difference"), dict) else {}
    audio_anchor = metadata.get("audio_anchor") if isinstance(metadata.get("audio_anchor"), dict) else {}
    lines = [
        f"# {metadata.get('index')}. {metadata.get('sample_id')} | {difference.get('type', '')}",
        "",
        f"- 修改文本: {metadata.get('edit_text', '')}",
        f"- 参考视频描述: {metadata.get('reference_caption', '')}",
        f"- 目标视频描述: {metadata.get('target_caption', '')}",
        f"- difference: `{json.dumps(difference, ensure_ascii=False)}`",
        f"- dominant_delta_decision: `{json.dumps(metadata.get('dominant_delta_decision', {}), ensure_ascii=False)}`",
        "- 详细双视频描述: `description.md`",
        f"- verification.passed: `{verification.get('passed')}`",
        f"- observable_difference.passed: `{observable.get('passed')}`",
        f"- competing_difference.passed: `{competing.get('passed')}`",
        *(
            [
                f"- audio_anchor_score: `{metadata.get('audio_anchor_score')}`",
                f"- audio_anchor_type: `{metadata.get('audio_anchor_type', '')}`",
                f"- audio_matters_warnings: `{json.dumps(metadata.get('audio_matters_warnings', []), ensure_ascii=False)}`",
                f"- audio_matters_line: `{metadata.get('audio_matters_line', '')}`",
                f"- omni_visual_accept: `{(metadata.get('quality') or {}).get('omni_visual_accept')}`",
                f"- omni_reject_reason: `{(metadata.get('quality') or {}).get('omni_reject_reason', '')}`",
                f"- visual_delta_strength: `{(metadata.get('quality') or {}).get('visual_delta_strength')}`",
                f"- near_duplicate_risk: `{(metadata.get('quality') or {}).get('near_duplicate_risk')}`",
                f"- reference_satisfies_edit: `{(metadata.get('quality') or {}).get('reference_satisfies_edit')}`",
                f"- target_satisfies_edit: `{(metadata.get('quality') or {}).get('target_satisfies_edit')}`",
            ]
            if audio_anchor
            else []
        ),
        f"- src_ref_images: `{json.dumps(metadata.get('src_ref_images', []), ensure_ascii=False)}`",
        f"- src_mask: `{metadata.get('src_mask', '')}`",
        f"- src_video_for_vace: `{metadata.get('src_video_for_vace', '')}`",
        f"- raw_generated_video: `{metadata.get('raw_generated_video', '')}`",
        f"- duration_metrics: `{json.dumps(metadata.get('duration_metrics', {}), ensure_ascii=False)}`",
        f"- mask_metrics: `{json.dumps(metadata.get('mask_metrics', {}), ensure_ascii=False)}`",
        f"- incomplete_review_bundle: `{metadata.get('incomplete_review_bundle')}`",
        f"- review_bundle_issues: `{json.dumps(metadata.get('review_bundle_issues', []), ensure_ascii=False)}`",
        "",
        "## 视频文件",
        "",
    ]
    if reference_filename:
        lines.append(f"- 参考视频本地副本: `{reference_filename}`")
    lines.append(f"- 参考视频原路径: `{metadata.get('reference_video_absolute', '')}`")
    if target_filename:
        lines.append(f"- 目标视频本地副本: `{target_filename}`")
    lines.append(f"- 目标视频原路径: `{metadata.get('target_video_absolute', '')}`")
    copied_refs = _normalize_list(metadata.get("copied_src_ref_images", []))
    if copied_refs:
        lines.extend(["", "## src_ref_images", ""])
        for copied in copied_refs:
            lines.append(f"- `{copied}`")
    if metadata.get("copied_src_mask"):
        lines.extend(["", "## mask", "", f"- `{metadata.get('copied_src_mask')}`"])
    if metadata.get("copied_src_video_for_vace"):
        lines.extend(["", "## src_video_for_vace", "", f"- `{metadata.get('copied_src_video_for_vace')}`"])
    if metadata.get("copied_raw_generated_video"):
        lines.extend(["", "## raw target", "", f"- `{metadata.get('copied_raw_generated_video')}`"])
    if metadata.get("copied_review_inputs"):
        lines.extend(["", "## review_inputs", "", f"- `{metadata.get('copied_review_inputs')}`"])
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "- `semantic_evaluation_result.json`",
            "- `mask_metrics.json`",
            "- `duration_metrics.json`",
        ]
    )
    lines.extend(
        [
            "",
            "## 人工核验问题",
            "",
            "- reference 和 target 是否还是同一视频上下文？",
            "- target 是否只体现 edit_text 的一个主差异？",
            "- edit_text 方向是否正确？",
            "- 是否有额外换场景、换动作、换文字、换主体？",
            "- 如果是视觉 synthetic，target 是否保留 reference audio？",
            "- 如果是音频 synthetic，画面是否完全一致，差异是否只来自音频？",
            "",
        ]
    )
    return "\n".join(lines)


def _review_annotation_description(annotation: dict[str, Any], *, fallback_caption: str = "") -> dict[str, Any]:
    return {
        "summary": str(annotation.get("summary") or annotation.get("caption") or fallback_caption).strip(),
        "subjects": _normalize_list(annotation.get("subjects", []))[:8],
        "subject_signature": _annotation_subject_signature_bundle(annotation)[:8],
        "attributes": _normalize_list(annotation.get("attributes", []))[:8],
        "scene": str(annotation.get("scene", "")).strip(),
        "scene_signature": _annotation_scene_signature_bundle(annotation)[:8],
        "actions": _action_terms_from_annotation(annotation)[:8],
        "object_counts": _normalize_object_counts(annotation.get("object_counts", {})),
        "object_signature": _annotation_object_signature_bundle(annotation)[:8],
        "visible_text": _visible_text_values(annotation)[:8],
        "speech": _speech_texts_from_annotation(annotation)[:5],
        "audio_events": _non_speech_audio_terms(annotation)[:8],
    }


def _manual_review_description_markdown(metadata: dict[str, Any]) -> str:
    reference_description = metadata.get("reference_omni_description", {})
    target_description = metadata.get("target_omni_description", {})
    dominant_delta = metadata.get("dominant_delta_decision", {})
    difference = metadata.get("difference") if isinstance(metadata.get("difference"), dict) else {}
    audio_anchor = metadata.get("audio_anchor") if isinstance(metadata.get("audio_anchor"), dict) else {}
    lines = [
        f"# Pair Description: {metadata.get('sample_id', '')}",
        "",
        f"- edit_text: {metadata.get('edit_text', '')}",
        f"- difference: `{json.dumps(difference, ensure_ascii=False)}`",
        f"- dominant_delta_decision: `{json.dumps(dominant_delta, ensure_ascii=False)}`",
        f"- secondary_deltas: `{json.dumps(metadata.get('secondary_deltas', []), ensure_ascii=False)}`",
        *(
            [f"- audio_anchor: `{json.dumps(audio_anchor, ensure_ascii=False)}`"]
            if audio_anchor
            else []
        ),
        "",
        "## Reference Omni Description",
        "",
        f"```json\n{json.dumps(reference_description, ensure_ascii=False, indent=2)}\n```",
        "",
        "## Target Omni Description",
        "",
        f"```json\n{json.dumps(target_description, ensure_ascii=False, indent=2)}\n```",
        "",
        "## Review Focus",
        "",
        "- 先判断 edit_text 是否描述了最显著主差异。",
        "- 如果主体、场景、物体或动作明显变化，弱音频差异不能作为主主题。",
        "- 只接受一个清楚、稳定、可描述的主变化。",
        "",
    ]
    return "\n".join(lines)


def _manual_review_index_markdown(
    *,
    items: list[dict[str, Any]],
    source_pairs_path: str,
    missing_videos: list[str],
) -> str:
    lines = [
        "# Manual Review Bundle",
        "",
        f"- Source pairs: `{source_pairs_path}`",
        f"- Sample count: `{len(items)}`",
        f"- Missing video count: `{len(missing_videos)}`",
        f"- Incomplete review bundle count: `{sum(1 for item in items if item.get('incomplete_review_bundle'))}`",
        "",
        "## Samples",
        "",
        "| # | sample_id | type | audio | status | edit_text | folder |",
        "|---|-----------|------|-------|--------|-----------|--------|",
    ]
    for item in items:
        status = "incomplete" if item.get("incomplete_review_bundle") else "complete"
        audio_score = item.get("audio_anchor_score")
        audio_text = "" if audio_score is None else f"{_score_float(audio_score):.3f}"
        lines.append(
            f"| {item['index']} | `{item['sample_id']}` | `{item.get('difference_type', '')}` | {audio_text} | "
            f"{status} | {item.get('edit_text', '')} | `{Path(item['item_dir']).name}` |"
        )
    if missing_videos:
        lines.extend(["", "## Missing Videos", ""])
        lines.extend(f"- `{path}`" for path in missing_videos)
    lines.append("")
    return "\n".join(lines)


def _video_edit_model_route(difference_type: str) -> str | None:
    difference_type = str(difference_type).strip()
    if difference_type == "object_presence":
        return "vace_controlled"
    if difference_type == "attribute":
        return "vace_controlled"
    if difference_type == "scene":
        return "vace_controlled"
    if difference_type == "action":
        return "ltx2_retake"
    return None


def _safe_visual_ideation_candidate(candidate: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any] | None:
    anchor = _safe_visual_edit_anchor(annotation)
    if anchor is None:
        return None
    edit_text, difference, reason = anchor
    source_edit_text = str(candidate.get("edit_text", "")).strip()
    proposal_seed = str(candidate.get("proposal_id", "")) or str(candidate.get("reference_video", "")) + edit_text
    revised = dict(candidate)
    revised["proposal_id"] = f"{str(candidate.get('proposal_id', '')).strip() or 'candidate'}__visual_ideation_{_stable_hash(proposal_seed)[:8]}"
    revised["edit_text"] = edit_text
    revised["difference"] = difference
    revised["source_candidate_edit_text"] = source_edit_text
    revised["source_candidate_difference"] = candidate.get("difference", {})
    revised["candidate_source"] = "safe_visual_ideation_from_reference"
    revised["ideation_reason"] = reason
    return revised


def _video_edit_exploration_candidates(candidate: dict[str, Any], annotation: dict[str, Any]) -> list[dict[str, Any]]:
    reference_video = str(candidate.get("reference_video", "")).strip()
    if not reference_video:
        return []

    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))
    object_names = list(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    scene = str(annotation.get("scene", "")).strip()
    summary = str(annotation.get("summary", "")).strip()
    text = _normalized_phrase(" ".join([summary, scene, " ".join(subjects), " ".join(object_names)]))
    source_edit_text = str(candidate.get("edit_text", "")).strip()
    source_difference = candidate.get("difference", {})
    base_proposal = str(candidate.get("proposal_id", "")).strip() or _stable_hash(reference_video)[:8]

    def build(
        *,
        family: str,
        edit_text: str,
        difference: dict[str, Any],
        edit_token: str,
        edit_region: str,
        mask_query: str,
        goal: str,
    ) -> dict[str, Any]:
        seed = "|".join([base_proposal, reference_video, family, edit_text])
        revised = dict(candidate)
        revised["proposal_id"] = f"{base_proposal}__vace_explore_{_safe_id(family)}_{_stable_hash(seed)[:8]}"
        revised["edit_text"] = edit_text
        revised["difference"] = difference
        revised["source_candidate_edit_text"] = source_edit_text
        revised["source_candidate_difference"] = source_difference
        revised["candidate_source"] = "vace_exploration_from_reference"
        revised["exploration_family"] = family
        revised["exploration_goal"] = goal
        revised["suggested_edit_token"] = edit_token
        revised["suggested_edit_region"] = edit_region
        revised["suggested_mask_query"] = mask_query
        revised["suggested_preserve_regions"] = _exploration_preserve_regions(annotation, edit_region)
        return revised

    candidates: list[dict[str, Any]] = []
    if any(marker in text for marker in ("robot", "robotic", "action figure")):
        candidates.append(
            build(
                family="attribute_color",
                edit_text="change the robot body color from black and gold to bright yellow",
                difference={
                    "type": "attribute",
                    "from": "black and gold robot body",
                    "to": "bright yellow robot body",
                    "description": "The existing robot body changes from black and gold to bright yellow.",
                },
                edit_token="bright yellow robot body",
                edit_region="robot body",
                mask_query="robot body",
                goal="test existing-subject color editing",
            )
        )
        candidates.append(
            build(
                family="attribute_material",
                edit_text="change the robot body material from black and gold plastic to metallic silver",
                difference={
                    "type": "attribute",
                    "from": "black and gold plastic robot body",
                    "to": "metallic silver robot body",
                    "description": "The existing robot body material changes to metallic silver.",
                },
                edit_token="metallic silver robot body",
                edit_region="robot body",
                mask_query="robot body",
                goal="test material and surface editing",
            )
        )

    if any(marker in text for marker in ("shirt", "jacket", "coat", "dress", "clothing", "outfit")):
        candidates.append(
            build(
                family="clothing_color",
                edit_text="change the clothing color to deep navy blue",
                difference={
                    "type": "attribute",
                    "from": "original clothing color",
                    "to": "deep navy blue clothing",
                    "description": "The existing clothing changes to deep navy blue.",
                },
                edit_token="deep navy blue clothing",
                edit_region="clothing",
                mask_query="clothing",
                goal="test clothing recoloring",
            )
        )
        candidates.append(
            build(
                family="clothing_type",
                edit_text="change the patterned shirt into a solid black shirt",
                difference={
                    "type": "attribute",
                    "from": "patterned shirt",
                    "to": "solid black shirt",
                    "description": "The existing patterned shirt changes into a solid black shirt.",
                },
                edit_token="solid black shirt",
                edit_region="clothing",
                mask_query="clothing",
                goal="test safe masked clothing type change without structural outerwear",
            )
        )

    if any(marker in text for marker in ("car", "vehicle", "truck", "bus")):
        candidates.append(
            build(
                family="vehicle_color",
                edit_text="change the vehicle body color to bright orange",
                difference={
                    "type": "attribute",
                    "from": "original vehicle body color",
                    "to": "bright orange vehicle body",
                    "description": "The existing vehicle body changes to bright orange.",
                },
                edit_token="bright orange vehicle body",
                edit_region="vehicle body",
                mask_query="vehicle body",
                goal="test large vehicle color editing",
            )
        )

    if any(marker in text for marker in ("room", "office", "kitchen", "street", "studio", "wall", "background")):
        candidates.append(
            build(
                family="background_change",
                edit_text="change the background to a futuristic laboratory",
                difference={
                    "type": "scene",
                    "from": "original background",
                    "to": "futuristic laboratory background",
                    "description": "The background changes to a futuristic laboratory while the main subject remains.",
                },
                edit_token="futuristic laboratory background",
                edit_region="background",
                mask_query=_foreground_mask_query_from_annotation(annotation),
                goal="test masked background replacement",
            )
        )
        candidates.append(
            build(
                family="style_lighting",
                edit_text="change the scene style to cinematic neon lighting",
                difference={
                    "type": "scene",
                    "from": "original scene style",
                    "to": "cinematic neon lighting style",
                    "description": "The scene style changes to cinematic neon lighting.",
                },
                edit_token="cinematic neon lighting style",
                edit_region="background",
                mask_query=_foreground_mask_query_from_annotation(annotation),
                goal="test style and lighting editing",
            )
        )

    replacement_count = 0
    removal_count = 0
    for object_name in object_names + subjects:
        normalized_object = _normalized_phrase(object_name)
        if not normalized_object or normalized_object in {"person", "man", "woman", "people", "hand", "hands"}:
            continue
        replacement = VACE_EXPLORATION_OBJECT_REPLACEMENTS.get(normalized_object)
        if (
            replacement
            and replacement_count < 2
            and not _reference_has_seated_support_conflict(annotation, object_name)
        ):
            replacement_count += 1
            candidates.append(
                build(
                    family="object_replacement",
                    edit_text=f"replace the {object_name} with a {replacement}",
                    difference={
                        "type": "object_presence",
                        "from": object_name,
                        "to": replacement,
                        "description": f"The existing {object_name} is replaced by a {replacement}.",
                    },
                    edit_token=replacement,
                    edit_region=object_name,
                    mask_query=object_name,
                    goal="test masked object replacement",
                )
            )
        if normalized_object in VACE_EXPLORATION_REMOVABLE_OBJECTS and removal_count < 2:
            removal_count += 1
            candidates.append(
                build(
                    family="object_removal",
                    edit_text=f"remove the {object_name} from the scene",
                    difference={
                        "type": "object_presence",
                        "from": object_name,
                        "to": f"no {object_name}",
                        "description": f"The existing {object_name} is removed from the scene.",
                    },
                    edit_token=object_name,
                    edit_region=object_name,
                    mask_query=object_name,
                    goal="test masked object removal and inpainting",
                )
            )

    return candidates


def _exploration_preserve_regions(annotation: dict[str, Any], edit_region: str) -> list[str]:
    values: list[str] = []
    values.extend(_normalize_list(annotation.get("subjects", [])))
    values.extend(list(_normalize_object_counts(annotation.get("object_counts", {})).keys()))
    values.extend(_normalize_list(annotation.get("actions", [])))
    scene = str(annotation.get("scene", "")).strip()
    if scene:
        values.append(scene)
    values.extend(["camera motion", "lighting", "timing", "visible text"])
    edit_key = _normalized_phrase(edit_region)
    preserved: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalized_phrase(value)
        if not key or key == edit_key or key in seen:
            continue
        seen.add(key)
        preserved.append(str(value).strip())
        if len(preserved) >= 8:
            break
    return preserved


def _safe_visual_edit_anchor(annotation: dict[str, Any]) -> tuple[str, dict[str, Any], str] | None:
    values: list[str] = [
        str(annotation.get("summary", "")),
        str(annotation.get("scene", "")),
    ]
    values.extend(_normalize_list(annotation.get("subjects", [])))
    values.extend(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    text = _normalized_phrase(" ".join(values))
    anchors = (
        (
            ("robot", "robotic", "action figure"),
            "change the robot body color from black and gold to bright yellow",
            "black and gold robot body",
            "bright yellow robot body",
            "robot body",
        ),
        (
            ("car", "vehicle", "truck", "bus"),
            "change the vehicle color to bright red",
            "original vehicle color",
            "bright red vehicle body",
            "vehicle body",
        ),
        (
            ("shirt", "jacket", "coat", "dress", "clothing"),
            "change the clothing color to bright blue",
            "original clothing color",
            "bright blue clothing",
            "clothing",
        ),
        (
            ("room", "office", "kitchen", "street"),
            "change the background to a futuristic laboratory",
            "original background",
            "futuristic laboratory background",
            "background",
        ),
    )
    for markers, edit_text, from_value, to_value, region in anchors:
        if any(marker in text for marker in markers):
            difference = {
                "type": "attribute",
                "from": from_value,
                "to": to_value,
                "description": f"The existing {region} changes from {from_value} to {to_value}.",
            }
            return edit_text, difference, f"reference has a stable existing {region} for attribute-based VACE editing"
    return None


def _relax_safe_visual_ideation_risk(risk: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("candidate_source", "")) != "safe_visual_ideation_from_reference":
        return risk
    difference = candidate.get("difference") if isinstance(candidate.get("difference"), dict) else {}
    if str(difference.get("type", "")).strip() != "attribute":
        return risk
    edit_text = str(candidate.get("edit_text", "")).lower()
    stable_surface_markers = {
        "robot",
        "vehicle",
        "clothing",
        "shirt",
        "jacket",
        "body",
        "color",
        "colour",
        "background",
        "laboratory",
        "lab",
        "style",
        "cyberpunk",
    }
    if not any(marker in edit_text for marker in stable_surface_markers):
        return risk
    risk_reasons = [str(reason) for reason in risk.get("risk_reasons", [])]
    hard_reasons = {"visible_text_present", "scene_or_shot_change", "ui_or_text_heavy_scene", "many_subjects"}
    if any(reason in hard_reasons for reason in risk_reasons):
        return risk
    relaxed = dict(risk)
    relaxed["allow_generation"] = True
    relaxed["risk_level"] = "medium" if risk_reasons else str(risk.get("risk_level", "low"))
    relaxed["safe_visual_ideation_relaxed"] = True
    relaxed["relaxed_risk_reasons"] = [
        reason
        for reason in risk_reasons
        if reason in {"multiple_actions", "multi_event_timeline", "speaking_person", "long_storyline"}
    ]
    locks = [
        str(item).strip()
        for item in risk.get("locks", [])
        if str(item).strip()
    ]
    extra_locks = [
        "limit the edit to the named existing subject attribute only",
        "preserve all text, hands, people, actions, motion order, and background content exactly",
    ]
    for lock in extra_locks:
        if lock not in locks:
            locks.append(lock)
    relaxed["locks"] = locks
    return relaxed


def _relax_visual_exploration_risk(risk: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("candidate_source", "")) != "vace_exploration_from_reference":
        return risk
    risk_reasons = [str(reason) for reason in risk.get("risk_reasons", [])]
    relaxed = dict(risk)
    relaxed["allow_generation"] = True
    relaxed["risk_level"] = "exploration_high" if risk_reasons else "exploration_low"
    relaxed["vace_exploration_relaxed"] = True
    relaxed["relaxed_risk_reasons"] = risk_reasons
    locks = [
        str(item).strip()
        for item in risk.get("locks", [])
        if str(item).strip()
    ]
    for lock in (
        "this is an exploration run; generate the requested single masked edit even if the reference is risky",
        "preserve all non-masked regions, visible text, people, camera motion, action timing, and scene layout exactly",
    ):
        if lock not in locks:
            locks.append(lock)
    relaxed["locks"] = locks
    return relaxed


def _normalize_model_planned_visual_difference(difference: dict[str, Any], *, edit_text: str) -> dict[str, Any]:
    return dict(difference)


def _video_edit_route_suitability(
    *,
    route: str,
    difference: dict[str, Any],
    edit_text: str,
    edit_token: str,
    edit_region: str,
    reference_annotation: dict[str, Any],
) -> dict[str, Any]:
    if route != "vace_controlled":
        return {"allow_generation": True, "reason": "route_supported"}

    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    combined_text = _normalized_phrase(" ".join([edit_text, edit_token, from_value, to_value, edit_region]))
    combined_tokens = set(TOKEN_PATTERN.findall(combined_text))
    tiny_markers = {_normalized_phrase(marker) for marker in VACE_TINY_OR_INSERTION_MARKERS}
    if any(marker and marker in combined_text for marker in tiny_markers):
        return {
            "allow_generation": False,
            "reason": "vace_rejects_tiny_or_naked_object_edit",
            "priority": "D",
        }

    if difference_type == "object_presence":
        if _absence_like_phrase(from_value) and not _absence_like_phrase(to_value):
            return {
                "allow_generation": False,
                "reason": "vace_rejects_naked_object_insertion",
                "priority": "D",
            }
        if _absence_like_phrase(to_value) and not _absence_like_phrase(from_value):
            return {
                "allow_generation": True,
                "reason": "object_removal_or_inpainting",
                "priority": "S",
            }
        return {
            "allow_generation": True,
            "reason": "existing_object_replacement",
            "priority": "S",
        }

    if difference_type == "attribute":
        if _absence_like_phrase(from_value) or _absence_like_phrase(to_value):
            return {
                "allow_generation": False,
                "reason": "vace_rejects_absence_based_attribute",
                "priority": "D",
            }
        if not (combined_tokens & VACE_ATTRIBUTE_MARKERS):
            return {
                "allow_generation": False,
                "reason": "vace_attribute_lacks_large_visible_property",
                "priority": "C",
            }
        priority = "S" if any(marker in combined_text for marker in ("clothing", "shirt", "jacket", "dress", "background", "style", "robot body", "vehicle")) else "A"
        return {
            "allow_generation": True,
            "reason": "existing_subject_attribute_edit",
            "priority": priority,
        }

    if difference_type == "scene":
        if _reference_has_multi_shot_mask_risk(reference_annotation):
            return {
                "allow_generation": False,
                "reason": "vace_background_edit_multi_scene_reference",
                "priority": "C",
            }
        if _reference_has_multiple_foreground_subjects(reference_annotation):
            return {
                "allow_generation": False,
                "reason": "vace_background_edit_multi_subject_reference",
                "priority": "C",
            }
        if not any(marker in combined_text for marker in VACE_BACKGROUND_STYLE_MARKERS):
            return {
                "allow_generation": False,
                "reason": "vace_scene_edit_lacks_background_or_style_target",
                "priority": "C",
            }
        return {
            "allow_generation": True,
            "reason": "background_or_style_edit",
            "priority": "S",
        }

    return {
        "allow_generation": False,
        "reason": f"vace_rejects_{difference_type or 'unknown'}_edit",
        "priority": "D",
    }


def _video_mask_query(
    *,
    difference: dict[str, Any],
    edit_text: str,
    edit_token: str,
    edit_region: str,
    route: str,
    suitability: dict[str, Any],
    reference_annotation: dict[str, Any] | None = None,
) -> str:
    if route != "vace_controlled":
        return ""
    difference_type = str(difference.get("type", "")).strip()
    reason = str(suitability.get("reason", "")).strip()
    combined = _normalized_phrase(" ".join([edit_text, edit_token, edit_region, str(difference.get("description", ""))]))
    if difference_type == "scene" or "background" in reason or "background" in combined:
        return _foreground_mask_query_from_annotation(reference_annotation or {})
    if any(marker in combined for marker in ("shirt", "jacket", "coat", "dress", "clothing", "outfit")):
        return "clothing"
    if any(marker in combined for marker in ("robot body", "robotic body", "robot shell")):
        return "robot body"
    if any(marker in combined for marker in ("vehicle body", "car body", "truck body", "bus body")):
        return "vehicle body"
    if difference_type == "object_presence" and _absence_like_phrase(str(difference.get("to", ""))):
        from_value = str(difference.get("from", "")).strip()
        if from_value and not _absence_like_phrase(from_value):
            return from_value[:120]
    if "replacement" in reason or difference_type in {"object_presence", "object_count"}:
        from_value = str(difference.get("from", "")).strip()
        if from_value and not _absence_like_phrase(from_value):
            return from_value[:120]
    if edit_region and not edit_region.startswith("localized region around"):
        return edit_region[:120]
    return (edit_token or str(difference.get("target", "")).strip() or edit_region)[:120]


def _foreground_mask_query_from_annotation(annotation: dict[str, Any]) -> str:
    candidates: list[str] = []
    candidates.extend(_normalize_list(annotation.get("main_subjects", [])))
    candidates.extend(_normalize_list(annotation.get("subjects", [])))
    reference_understanding = annotation.get("reference_understanding")
    if isinstance(reference_understanding, dict):
        candidates.extend(_normalize_list(reference_understanding.get("main_subjects", [])))
    candidates.extend(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    generic = {"background", "scene", "room", "wall", "floor", "table", "desk", "lighting", "camera motion"}
    for candidate in candidates:
        item = str(candidate).strip()
        key = _normalized_phrase(item)
        if not item or key in generic:
            continue
        for token in ("man", "woman", "person", "girl", "boy", "robot", "vehicle", "car", "dog", "cat"):
            if token in key.split():
                return token
    for candidate in candidates:
        item = str(candidate).strip()
        key = _normalized_phrase(item)
        if item and key not in generic:
            return item[:120]
    return "main subject"


def _foreground_mask_query_candidates_from_annotation(annotation: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    if isinstance(annotation, dict):
        candidates.extend(_normalize_list(annotation.get("main_subjects", [])))
        candidates.extend(_normalize_list(annotation.get("subjects", [])))
        reference_understanding = annotation.get("reference_understanding")
        if isinstance(reference_understanding, dict):
            candidates.extend(_normalize_list(reference_understanding.get("main_subjects", [])))
        candidates.extend(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    generic = {"background", "scene", "room", "wall", "floor", "table", "desk", "lighting", "camera motion"}
    ordered: list[str] = []
    primary = _foreground_mask_query_from_annotation(annotation)
    if primary:
        ordered.append(primary)
    for candidate in candidates:
        item = str(candidate).strip()
        key = _normalized_phrase(item)
        if item and key not in generic:
            ordered.append(item[:120])
            for token in ("man", "woman", "person", "girl", "boy", "robot", "vehicle", "car", "dog", "cat"):
                if token in key.split():
                    ordered.append(token)
                    break
    return _dedupe_strings(ordered)[:6] or ["main subject"]


def _mask_query_is_generic_person(mask_query: str) -> bool:
    return _normalized_phrase(mask_query) in VACE_GENERIC_PERSON_MASK_QUERIES


def _video_mask_query_candidates_for_plan(plan: dict[str, Any], primary_query: str) -> list[str]:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    edit_text = str(plan.get("edit_text", "")).strip()
    edit_token = str(plan.get("edit_token", "")).strip()
    exploration_family = _normalized_phrase(str(plan.get("exploration_family", "")))
    reference_annotation = (
        plan.get("reference_understanding") if isinstance(plan.get("reference_understanding"), dict) else {}
    )
    normalized_query = _normalized_phrase(primary_query)
    candidates: list[str] = [primary_query]
    if (
        _is_clothing_edit(difference, edit_text, edit_token)
        or exploration_family.startswith("clothing")
        or any(marker in normalized_query for marker in ("clothing", "shirt", "jacket", "outfit", "blouse", "robe"))
    ):
        for value in (
            str(difference.get("from", "")),
            str(difference.get("to", "")),
            str(plan.get("edit_region", "")),
            primary_query,
            edit_token,
            edit_text,
        ):
            value_key = _normalized_phrase(value)
            for marker in ("shirt", "jacket", "coat", "dress", "hoodie", "sweater", "vest", "pants", "skirt", "blouse", "robe"):
                if marker in value_key.split():
                    candidates.append(marker)
                    candidates.append(f"torso {marker}")
                    if marker == "robe":
                        candidates.append("character robe")
                    break
        candidates.extend(["torso clothing", "clothing"])
    elif str(difference.get("type", "")).strip() == "scene" or "background" in _normalized_phrase(
        str(plan.get("edit_region", ""))
    ):
        candidates.extend(_foreground_mask_query_candidates_from_annotation(reference_annotation))
    else:
        for value in (str(plan.get("edit_region", "")), edit_token, str(difference.get("from", ""))):
            value = value.strip()
            if value:
                candidates.append(value[:120])
    return _dedupe_strings([item for item in candidates if str(item).strip()])[:8]


def _video_mask_query_for_plan(plan: dict[str, Any], mask_query: str) -> str:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    edit_text = str(plan.get("edit_text", "")).strip()
    edit_token = str(plan.get("edit_token", "")).strip()
    exploration_family = _normalized_phrase(str(plan.get("exploration_family", "")))
    reference_annotation = (
        plan.get("reference_understanding") if isinstance(plan.get("reference_understanding"), dict) else {}
    )
    normalized_query = _normalized_phrase(mask_query)
    if (
        _is_clothing_edit(difference, edit_text, edit_token)
        or exploration_family.startswith("clothing")
        or any(marker in normalized_query for marker in ("clothing", "shirt", "jacket", "outfit", "blouse", "robe"))
    ):
        for value in (
            str(difference.get("from", "")),
            str(difference.get("to", "")),
            str(plan.get("edit_region", "")),
            mask_query,
            edit_token,
            edit_text,
        ):
            value_key = _normalized_phrase(value)
            for marker in ("shirt", "jacket", "coat", "dress", "hoodie", "sweater", "vest", "pants", "skirt", "blouse", "robe"):
                if marker in value_key.split():
                    return marker
        return "torso clothing"
    if str(difference.get("type", "")).strip() == "scene" or "background" in _normalized_phrase(
        str(plan.get("edit_region", ""))
    ):
        return _foreground_mask_query_from_annotation(reference_annotation)
    return mask_query


def _reference_has_multi_shot_mask_risk(annotation: dict[str, Any]) -> bool:
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                str(annotation.get("scene", "")),
                str(annotation.get("stable_scene", "")),
                " ".join(_normalize_list(annotation.get("actions", []))),
                " ".join(_normalize_list(annotation.get("visible_text", []))),
            ]
        )
    )
    return any(marker in text for marker in VACE_MULTI_SHOT_MASK_MARKERS)


def _reference_has_worn_object_conflict(annotation: dict[str, Any], source_object: str, edit_region: str) -> bool:
    source_tokens = set(TOKEN_PATTERN.findall(_normalized_phrase(source_object)))
    if not source_tokens & VACE_WORN_OBJECT_MARKERS:
        return False
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                str(annotation.get("scene", "")),
                " ".join(_normalize_list(annotation.get("actions", []))),
                edit_region,
            ]
        )
    )
    return any(marker in text for marker in ("wearing", "wears", "worn", "on back", "back", "shoulder", "holding"))


def _reference_has_multiple_foreground_subjects(annotation: dict[str, Any]) -> bool:
    subject_terms: set[str] = set()
    for value in (
        _normalize_list(annotation.get("main_subjects", []))
        + _normalize_list(annotation.get("subjects", []))
    ):
        key = _normalized_phrase(value)
        if not key:
            continue
        if any(token in key.split() for token in ("man", "woman", "person", "girl", "boy")):
            subject_terms.add(key)
    if len(subject_terms) > 1:
        return True
    counts = _normalize_object_counts(annotation.get("object_counts", {}))
    for name, count in counts.items():
        key = _normalized_phrase(name)
        if count > 1 and any(token in key.split() for token in ("man", "woman", "person", "girl", "boy")):
            return True
    stable_scene = _normalized_phrase(str(annotation.get("stable_scene", "")))
    if " and " in str(annotation.get("stable_scene", "")).lower() and sum(
        1
        for marker in ("room", "studio", "wall", "background", "scene")
        if marker in stable_scene
    ) >= 2:
        return True
    return False


def _is_low_contrast_dark_clothing_edit(plan: dict[str, Any]) -> bool:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    edit_text = str(plan.get("edit_text", "")).strip()
    edit_token = str(plan.get("edit_token", "")).strip()
    if not _is_clothing_edit(difference, edit_text, edit_token):
        return False
    source = _normalized_phrase(_video_edit_source_object(difference, edit_text))
    target = _normalized_phrase(_video_edit_target_object(difference, edit_text, edit_token))
    if not source or not target:
        return False

    def has_dark_marker(text: str) -> bool:
        return any(marker in text for marker in VACE_DARK_COLOR_MARKERS)

    source_has_garment = any(marker in source.split() for marker in VACE_CLOTHING_OBJECT_MARKERS)
    target_has_garment = any(marker in target.split() for marker in VACE_CLOTHING_OBJECT_MARKERS)
    return source_has_garment and target_has_garment and has_dark_marker(source) and has_dark_marker(target)


def _video_maskability_issue(
    plan: dict[str, Any],
    *,
    mask_query: str,
    mask_mode: str,
) -> str:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    edit_text = str(plan.get("edit_text", "")).strip()
    source_object = _video_edit_source_object(difference, edit_text)
    source_key = _normalized_phrase(source_object)
    target_instance_description = str(plan.get("target_instance_description", "")).strip()
    reference_annotation = (
        plan.get("reference_understanding") if isinstance(plan.get("reference_understanding"), dict) else {}
    )
    if _is_low_contrast_dark_clothing_edit(plan):
        return "low_contrast_dark_clothing_color_edit"
    if _reference_has_multi_shot_mask_risk(reference_annotation):
        return "multi_shot_mask_route_unsupported"
    if (
        mask_mode in {"replace_masked_object", "remove_or_inpaint_masked_object"}
        and source_key in VACE_TINY_FULLFRAME_OBJECTS
    ):
        return "small_object_too_tiny_for_fullframe_vace"
    if (
        mask_mode in {"replace_masked_object", "remove_or_inpaint_masked_object"}
        and source_object
        and _normalized_phrase(mask_query) == source_key
        and _reference_has_multiple_visible_instances(reference_annotation, source_object)
        and not target_instance_description
    ):
        return "ambiguous_multi_instance_mask_query"
    if _reference_has_worn_object_conflict(
        reference_annotation,
        source_object,
        str(plan.get("edit_region", "")).strip(),
    ):
        return "subject_contact_or_worn_object_high_risk"
    if (
        mask_mode == "edit_background_inverse_subject"
        and _mask_query_is_generic_person(mask_query)
        and _reference_has_multiple_visible_instances(reference_annotation, mask_query)
    ):
        return "ambiguous_foreground_subject_for_background_mask"
    if mask_mode == "edit_background_inverse_subject" and _reference_has_multiple_foreground_subjects(reference_annotation):
        return "multi_subject_background_mask_route_unsupported"
    return ""


def _video_preserve_regions(
    *,
    preserve_tokens: list[str],
    edit_region: str,
    reference_annotation: dict[str, Any],
) -> list[str]:
    values = [str(item).strip() for item in preserve_tokens if str(item).strip()]
    values.extend(_normalize_list(reference_annotation.get("subjects", [])))
    scene = str(reference_annotation.get("scene", "")).strip()
    if scene:
        values.append(scene)
    values.extend(["camera motion", "lighting", "timing"])
    edit_key = _normalized_phrase(edit_region)
    regions: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalized_phrase(value)
        if not key or key == edit_key or key in seen:
            continue
        seen.add(key)
        regions.append(value)
        if len(regions) >= 8:
            break
    return regions


def _video_mask_mode(plan: dict[str, Any]) -> str:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    edit_region = _normalized_phrase(str(plan.get("edit_region", "")))
    mask_query = _normalized_phrase(str(plan.get("mask_query", "")))
    difference_type = str(difference.get("type", "")).strip()
    combined = _normalized_phrase(
        " ".join(
            [
                str(plan.get("edit_text", "")),
                str(plan.get("edit_token", "")),
                str(plan.get("edit_region", "")),
                str(difference.get("from", "")),
                str(difference.get("to", "")),
                str(plan.get("mask_query", "")),
            ]
        )
    )
    if mask_query != "background" and any(marker in combined for marker in VACE_CLOTHING_OBJECT_MARKERS):
        return "edit_masked_region"
    if difference_type == "scene" or "background" in edit_region or mask_query == "background":
        return "edit_background_inverse_subject"
    if difference_type == "object_presence" and _absence_like_phrase(str(difference.get("to", ""))):
        return "remove_or_inpaint_masked_object"
    if difference_type in {"object_presence", "object_count"}:
        return "replace_masked_object"
    return "edit_masked_region"


def _video_mask_gate_defaults(
    *,
    mask_mode: str = "",
    mask_query: str = "",
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_query = _normalized_phrase(mask_query)
    family = _normalized_phrase(str((plan or {}).get("exploration_family", "")))
    difference = (plan or {}).get("difference") if isinstance((plan or {}).get("difference"), dict) else {}
    edit_text = str((plan or {}).get("edit_text", "")).strip()
    edit_token = str((plan or {}).get("edit_token", "")).strip()
    protected_queries: list[str] = []
    max_protected_overlap = 0.0
    if (
        _is_clothing_edit(difference, edit_text, edit_token)
        or family.startswith("clothing")
        or any(marker in normalized_query for marker in ("clothing", "shirt", "jacket", "outfit"))
    ):
        min_coverage = 0.03
        max_coverage = 0.30
        min_detected_box_coverage = 0.035
        protected_queries = ["face", "hands", "ukulele", "guitar", "instrument", "microphone"]
        max_protected_overlap = 0.18
    elif mask_mode in {"replace_masked_object", "remove_or_inpaint_masked_object"}:
        min_coverage = 0.01
        max_coverage = 0.15
        min_detected_box_coverage = 0.005
    elif mask_mode == "edit_background_inverse_subject":
        min_coverage = 0.20
        max_coverage = 0.90
        min_detected_box_coverage = 0.10
    else:
        min_coverage = MIN_VIDEO_MASK_COVERAGE_RATIO
        max_coverage = MAX_VIDEO_MASK_COVERAGE_RATIO
        min_detected_box_coverage = 0.0
    gate = {
        "min_coverage_ratio": min_coverage,
        "max_coverage_ratio": max_coverage,
        "min_temporal_stability": MIN_VIDEO_MASK_TEMPORAL_STABILITY,
        "min_nonempty_frame_ratio": MIN_VIDEO_MASK_NONEMPTY_FRAME_RATIO,
        "min_detected_keyframe_box_coverage": min_detected_box_coverage,
        "mask_not_empty_all_frames": True,
        "mask_target_matches_query": True,
    }
    if protected_queries:
        gate["protected_overlap_queries"] = protected_queries
        gate["max_protected_overlap_ratio"] = max_protected_overlap
        gate["min_protected_detections"] = 2
        gate["require_protected_overlap_metrics"] = True
    if mask_mode == "edit_background_inverse_subject":
        gate["max_subject_overlap_ratio"] = VACE_BACKGROUND_MAX_SUBJECT_OVERLAP_RATIO
        gate["min_background_editable_ratio"] = min_coverage
        gate["min_foreground_subject_coverage_ratio"] = VACE_BACKGROUND_MIN_FOREGROUND_SUBJECT_COVERAGE_RATIO
        gate["max_foreground_subject_coverage_ratio"] = VACE_BACKGROUND_MAX_FOREGROUND_SUBJECT_COVERAGE_RATIO
        gate["min_foreground_subject_temporal_stability"] = MIN_VIDEO_MASK_TEMPORAL_STABILITY
        gate["min_foreground_subject_nonempty_frame_ratio"] = MIN_VIDEO_MASK_NONEMPTY_FRAME_RATIO
    return gate


def _heuristic_stable_clip_selection(
    *,
    media: dict[str, Any],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> dict[str, Any]:
    duration = float(media.get("duration_seconds") or 0.0)
    clip_seconds = min(max_clip_seconds, duration)
    if clip_seconds < min_clip_seconds:
        clip_seconds = duration
    start = 0.0
    end = min(duration, start + clip_seconds)
    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "stability_score": 0.5,
        "camera_motion": "unknown",
        "main_subjects": [],
        "visible_text_risk": False,
        "recommended_for_vace": True,
        "reason": "heuristic first stable-length window; Omni selection was not available",
    }


def _coerce_stable_clip_selection(
    selection: dict[str, Any],
    *,
    fallback: dict[str, Any],
    media: dict[str, Any],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> dict[str, Any]:
    duration = float(media.get("duration_seconds") or 0.0)
    try:
        start = max(0.0, float(selection.get("start_sec", fallback.get("start_sec", 0.0))))
        end = max(start, float(selection.get("end_sec", fallback.get("end_sec", start))))
    except (TypeError, ValueError):
        start = float(fallback.get("start_sec", 0.0) or 0.0)
        end = float(fallback.get("end_sec", min(duration, start + max_clip_seconds)) or 0.0)
    if end > duration:
        end = duration
    window = end - start
    if window < min_clip_seconds or window > max_clip_seconds:
        fallback_start = float(fallback.get("start_sec", 0.0) or 0.0)
        fallback_end = float(fallback.get("end_sec", min(duration, fallback_start + max_clip_seconds)) or 0.0)
        start, end = fallback_start, fallback_end
    coerced = {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "stability_score": _score_float(selection.get("stability_score", fallback.get("stability_score", 0.5))),
        "camera_motion": str(selection.get("camera_motion", fallback.get("camera_motion", "unknown"))).strip() or "unknown",
        "main_subjects": _dedupe_strings(_normalize_list(selection.get("main_subjects", fallback.get("main_subjects", []))))[:6],
        "visible_text_risk": _boolish(selection.get("visible_text_risk", fallback.get("visible_text_risk", False))),
        "recommended_for_vace": _boolish(selection.get("recommended_for_vace", fallback.get("recommended_for_vace", True))),
        "reason": str(selection.get("reason", fallback.get("reason", ""))).strip(),
    }
    return coerced


def _stable_edit_targets_from_understanding(
    visual_understanding: dict[str, Any],
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for attribute in visual_understanding.get("editable_attributes", []):
        if not isinstance(attribute, dict):
            continue
        target = str(attribute.get("target", "")).strip()
        current = str(attribute.get("current", "")).strip()
        safe_targets = _normalize_list(attribute.get("safe_targets", []))
        if target and safe_targets:
            targets.append(
                {
                    "target": target,
                    "edit_family": "attribute_color",
                    "suggested_edit": f"change {target} from {current or 'its current appearance'} to {safe_targets[0]}",
                    "mask_query": target,
                    "needs_src_ref_images": False,
                    "src_ref_request": "",
                    "safe_targets": safe_targets,
                }
            )
    object_names = list(_normalize_object_counts(annotation.get("object_counts", {})).keys())
    for source_name, replacement in VACE_EXPLORATION_OBJECT_REPLACEMENTS.items():
        if any(_text_mentions_phrase(name, source_name) for name in object_names) and not _reference_has_seated_support_conflict(
            annotation,
            source_name,
        ):
            targets.append(
                {
                    "target": source_name,
                    "edit_family": "object_replacement",
                    "suggested_edit": f"replace the {source_name} with a {replacement}",
                    "mask_query": source_name,
                    "needs_src_ref_images": True,
                    "src_ref_request": f"a realistic {replacement}, isolated, plain background, no hands, no text",
                }
            )
            break
    return targets[:8]


def _annotation_is_usable_for_reference_understanding(annotation: dict[str, Any]) -> bool:
    if bool(annotation.get("fallback_used")):
        return False
    if str(annotation.get("detective_fallback_reason", "")).strip() == "detective_and_single_pass_failed":
        return False
    if str(annotation.get("fallback_reason", "")).strip() == "annotation_fallback":
        return False
    text_fields = [
        str(annotation.get("summary", "")).strip(),
        str(annotation.get("scene", "")).strip(),
    ]
    list_fields = (
        _normalize_list(annotation.get("subjects", []))
        + _normalize_list(annotation.get("actions", []))
        + _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
        + _normalize_list(annotation.get("audio_events", []))
        + _normalize_list(annotation.get("speech", []))
        + _normalize_list(annotation.get("storyline", []))
        + _normalize_list(annotation.get("events", []))
    )
    if any(text_fields) or any(str(item).strip() for item in list_fields):
        return True
    object_counts = annotation.get("object_counts")
    return isinstance(object_counts, dict) and bool(object_counts)


def _src_ref_requirement_for_video_plan(plan: dict[str, Any]) -> dict[str, Any]:
    difference = plan.get("difference") if isinstance(plan.get("difference"), dict) else {}
    difference_type = str(difference.get("type", "")).strip()
    edit_text = _normalized_phrase(str(plan.get("edit_text", "")))
    edit_token = str(plan.get("edit_token", "")).strip()
    family = str(plan.get("exploration_family", "")).strip()
    target = str(difference.get("to", "")).strip() or edit_token
    from_value = str(difference.get("from", "")).strip()

    structural_clothing_reason = _structural_clothing_edit_reason(
        difference,
        str(plan.get("edit_text", "")),
        edit_token,
        str(plan.get("source_prompt", "")),
    )
    if structural_clothing_reason:
        return {
            "required": False,
            "recommended": False,
            "role": "none",
            "target": "",
            "source_object": from_value,
            "reason": structural_clothing_reason,
        }

    if family == "object_replacement" or ("replace" in edit_text and difference_type in {"object_presence", "object_count"}):
        return {
            "required": True,
            "recommended": True,
            "role": "replacement_object",
            "target": target,
            "source_object": from_value,
            "reason": "object replacement needs a visual reference image for the replacement object",
        }
    if family == "background_change" or difference_type == "scene" or "background" in edit_text:
        return {
            "required": True,
            "recommended": True,
            "role": "background_reference",
            "target": target or "target background",
            "source_object": from_value,
            "reason": "background replacement benefits from a clean background reference image",
        }
    if family == "clothing_type" or any(token in edit_text for token in ("shirt", "jacket", "dress", "outfit", "clothing")) and "replace" in edit_text:
        if _is_black_jacket_target(difference, str(plan.get("edit_text", "")), edit_token):
            target = VACE_BLACK_JACKET_SRC_REF_TARGET
        return {
            "required": True,
            "recommended": True,
            "role": "clothing_reference",
            "target": target or edit_token or "target clothing",
            "source_object": from_value,
            "reason": "clothing type replacement needs a reference image for the target clothing",
        }
    if family == "object_removal" or edit_text.startswith("remove "):
        return {
            "required": False,
            "recommended": False,
            "role": "none",
            "target": "",
            "source_object": from_value or edit_token,
            "reason": "object removal uses mask inpainting and does not need src_ref_images",
        }
    return {
        "required": False,
        "recommended": False,
        "role": "none",
        "target": "",
        "source_object": from_value,
        "reason": "attribute/color/material edits can run with video + mask + prompt",
    }


def _src_ref_image_prompts(*, requirement: dict[str, Any], edit_plan: dict[str, Any]) -> list[str]:
    target = str(requirement.get("target", "")).strip() or "target object"
    role = str(requirement.get("role", "")).strip()
    source_object = str(requirement.get("source_object", "")).strip()
    if role == "clothing_reference" and _is_black_jacket_target(
        edit_plan.get("difference", {}) if isinstance(edit_plan.get("difference"), dict) else {},
        str(edit_plan.get("edit_text", "")),
        str(edit_plan.get("edit_token", "")),
    ):
        target = VACE_BLACK_JACKET_SRC_REF_TARGET
    if role == "background_reference":
        return [
            f"a clean 16:9 horizontal wide reference image of {target}, empty scene plate, natural camera perspective, no people, no text, no watermark",
            f"{target}, 16:9 landscape empty background plate matching a talking-head video perspective, cinematic but realistic lighting, no foreground subject, no readable text",
        ]
    if role == "clothing_reference":
        target_with_article = _article_clothing_phrase(target)
        if _normalized_phrase(VACE_BLACK_JACKET_REQUIRED_PHRASE) in _normalized_phrase(target):
            return [
                f"cropped upper-body photo of a standing musician wearing {target_with_article}, arms bent as if holding a small instrument, shoulders and full sleeves visible, no face, no text, no logo",
                f"a realistic human torso wearing {target_with_article}, open jacket front and sleeve structure visible, arms slightly bent, neutral background, no product catalog layout, no watermark",
            ]
        return [
            f"a realistic cropped upper-body photo of a person wearing {target_with_article}, garment silhouette clearly visible, arms slightly bent as if holding a small instrument, front three-quarter view, no face, no text, no logo",
            f"a standing musician torso wearing {target_with_article}, natural human shoulder fit, arms visible, neutral background, no product catalog layout, no watermark",
        ]
    if role == "replacement_object":
        source_hint = f" matching the viewpoint and scale of a {source_object}" if source_object else ""
        if _normalized_phrase(source_object) in {"cup", "mug", "glass"} and "bottle" in _normalized_phrase(target):
            return [
                f"a realistic small tabletop {target}{source_hint}, upright on a plain studio surface, three-quarter view, cup-sized proportion, no hands, no people, no text, no logo",
                f"{target}, small bottle reference for replacing a {source_object} on a table, visible side profile and cap, neutral lighting, plain background, no watermark",
            ]
        return [
            f"a realistic {target}, isolated product reference{source_hint}, three-quarter view, plain white background, no hands, no people, no text, no logo",
            f"{target}, clean object reference image with visible side and top shape, centered, neutral lighting, plain background, perspective suitable to replace a {source_object or 'source object'} in a live-action shot, no watermark",
        ]
    return [
        f"a realistic {target}, isolated product reference, three-quarter view, plain white background, no hands, no people, no text, no logo",
        f"{target}, clean object reference image with visible side and top shape, centered, neutral lighting, transparent or plain background, no watermark",
    ]


def _src_ref_image_negative_prompt(requirement: dict[str, Any]) -> str:
    role = str(requirement.get("role", "")).strip()
    base = "text, watermark, logo, blur, clutter, extra objects, distorted shape"
    if role == "replacement_object":
        return f"hands, people, scene background, {base}"
    if role == "background_reference":
        return f"people, foreground subject, readable signs, {base}"
    if role == "clothing_reference":
        return f"flat lay, hanger, empty jacket, ghost mannequin, product catalog, face, full person identity, readable brand logo, {base}"
    return base


def _find_src_ref_image_candidates(candidate_dir: Path) -> list[Path]:
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(path for path in candidate_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _audit_src_ref_image_candidate(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    role = str(plan.get("src_ref_role", "")).strip()
    score = 0.50
    reasons: list[str] = ["candidate file exists"]
    warnings: list[str] = []
    width = 0
    height = 0
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            width, height = image.size
        score += 0.15
        reasons.append(f"readable image {width}x{height}")
    except Exception:
        warnings.append("image dimensions unavailable for deterministic audit")

    if width > 0 and height > 0:
        aspect = width / max(height, 1)
        if role == "background_reference":
            if 1.45 <= aspect <= 1.95:
                score += 0.20
                reasons.append("background candidate is close to 16:9")
            else:
                score -= 0.35
                warnings.append("background candidate is not close to 16:9")
        elif role == "clothing_reference":
            if 0.55 <= aspect <= 1.35:
                score += 0.15
                reasons.append("clothing candidate has plausible torso/reference aspect")
            else:
                warnings.append("clothing candidate aspect may be hard to fit to a person")
        elif role == "replacement_object":
            if 0.45 <= aspect <= 2.20:
                score += 0.10
                reasons.append("replacement object candidate has usable aspect")

    name_key = _normalized_phrase(path.name)
    if any(token in name_key for token in ("text", "logo", "watermark", "person", "hand", "face")):
        score -= 0.25
        warnings.append("filename suggests a forbidden visual artifact")
    clothing_artifact_reasons = (
        ["clothing_src_ref_product_or_empty_jacket_artifact"]
        if role == "clothing_reference"
        and any(marker in name_key for marker in VACE_CLOTHING_SRC_REF_ARTIFACT_MARKERS)
        else []
    )
    if clothing_artifact_reasons:
        score -= 0.50
        warnings.append("filename suggests an empty/product/mannequin clothing reference")
    background_reject_reasons = (
        ["background_src_ref_not_16x9"]
        if role == "background_reference" and width > 0 and height > 0 and not (1.45 <= width / max(height, 1) <= 1.95)
        else []
    )
    hard_reject_reasons = background_reject_reasons + clothing_artifact_reasons
    return {
        "path": str(path),
        "score": round(max(0.0, min(1.0, score)), 3),
        "width": width,
        "height": height,
        "eligible": not hard_reject_reasons,
        "hard_reject_reasons": hard_reject_reasons,
        "reasons": reasons,
        "warnings": warnings,
    }


def _video_edit_reference_understanding(annotation: dict[str, Any]) -> dict[str, Any]:
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))[:6]
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:6]
    visible_text = _dedupe_strings(
        _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
    )[:6]
    object_names = list(_normalize_object_counts(annotation.get("object_counts", {})).keys())[:6]
    summary = str(annotation.get("summary", "")).strip()
    scene = str(annotation.get("scene", "")).strip()
    stable_clip_selection = (
        annotation.get("stable_clip_selection") if isinstance(annotation.get("stable_clip_selection"), dict) else {}
    )
    editable_attributes: list[dict[str, Any]] = []
    text = _normalized_phrase(" ".join([summary, scene, " ".join(subjects), " ".join(object_names)]))
    if any(marker in text for marker in ("robot", "robotic", "action figure")):
        editable_attributes.append(
            {
                "type": "color",
                "target": "robot body",
                "current": "black and gold",
                "safe_targets": ["bright yellow", "silver", "red"],
            }
        )
    if any(marker in text for marker in ("car", "vehicle", "truck", "bus")):
        editable_attributes.append(
            {
                "type": "color",
                "target": "vehicle body",
                "current": "original vehicle color",
                "safe_targets": ["bright red", "blue", "silver"],
            }
        )
    if any(marker in text for marker in ("shirt", "jacket", "coat", "dress", "clothing")):
        editable_attributes.append(
            {
                "type": "color",
                "target": "clothing",
                "current": "original clothing color",
                "safe_targets": ["bright blue", "red", "green"],
            }
        )
    return {
        "main_subjects": subjects or object_names,
        "stable_scene": scene or summary,
        "camera_motion": str(
            annotation.get("camera_motion")
            or stable_clip_selection.get("camera_motion")
            or "unknown"
        ).strip()
        or "unknown",
        "stability_score": _score_float(
            stable_clip_selection.get("stability_score", annotation.get("stability_score", 0.5))
        ),
        "recommended_for_vace": _boolish(
            stable_clip_selection.get("recommended_for_vace", annotation.get("recommended_for_vace", True))
        ),
        "stable_clip_reason": str(
            stable_clip_selection.get("reason") or annotation.get("stable_clip_reason") or ""
        ).strip(),
        "visible_text": visible_text,
        "actions": actions,
        "editable_attributes": editable_attributes,
        "bad_edits": [
            "add small background object",
            "add text",
            "add tiny accessory",
            "change exact object count",
        ],
    }


def _planned_route_matches_difference(route: str, difference_type: str) -> bool:
    expected_route = _video_edit_model_route(difference_type)
    return bool(expected_route and route == expected_route)


def _video_edit_token(difference: dict[str, Any], edit_text: str) -> str:
    for field_name in ("to", "description", "from"):
        value = str(difference.get(field_name, "")).strip()
        if value and not _absence_like_phrase(value):
            return value[:120]
    tokens = TOKEN_PATTERN.findall(edit_text.lower())
    if not tokens:
        return ""
    return " ".join(tokens[-5:])[:120]


def _absence_like_phrase(value: str) -> bool:
    normalized = _normalized_phrase(value)
    return bool(
        not normalized
        or normalized.startswith("no ")
        or normalized.startswith("none")
        or normalized in {"absent", "missing", "nothing", "no distinctive audio event"}
    )


def _video_edit_region(
    edit_text: str,
    difference: dict[str, Any],
    annotation: dict[str, Any],
    route: str,
) -> str:
    if route == "audio_deterministic":
        return "audio track"
    text = " ".join(
        str(value).strip()
        for value in (
            edit_text,
            difference.get("description", ""),
            difference.get("to", ""),
            annotation.get("summary", ""),
        )
        if str(value).strip()
    ).lower()
    region_patterns = (
        ("top-right", "top-right region"),
        ("top right", "top-right region"),
        ("top-left", "top-left region"),
        ("top left", "top-left region"),
        ("bottom-right", "bottom-right region"),
        ("bottom right", "bottom-right region"),
        ("bottom-left", "bottom-left region"),
        ("bottom left", "bottom-left region"),
        ("background", "background"),
        ("foreground", "foreground"),
        ("wall", "wall area"),
        ("paper", "paper surface"),
        ("desk", "desk surface"),
        ("table", "table surface"),
        ("robot body", "robot body"),
        ("robot", "robot body"),
        ("vehicle", "vehicle body"),
        ("car", "vehicle body"),
        ("clothing", "clothing"),
        ("shirt", "clothing"),
        ("jacket", "clothing"),
        ("visor", "visor"),
        ("floor", "floor area"),
        ("center", "center region"),
        ("left", "left side"),
        ("right", "right side"),
    )
    for marker, region in region_patterns:
        if marker in text:
            return region
    edit_token = _video_edit_token(difference, edit_text)
    if edit_token:
        return f"localized region around {edit_token}"
    return ""


def _video_edit_source_prompt(annotation: dict[str, Any], record: dict[str, Any]) -> str:
    summary = str(annotation.get("summary") or record.get("reference_caption", "")).strip()
    scene = str(annotation.get("scene", "")).strip()
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))[:4]
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:3]
    clauses = [summary or "the reference video"]
    if scene:
        clauses.append(f"scene: {scene}")
    if subjects:
        clauses.append("main subjects: " + ", ".join(subjects))
    if actions:
        clauses.append("actions: " + ", ".join(actions))
    return ". ".join(clauses).strip().rstrip(".") + "."


def _is_existing_object_replacement(difference: dict[str, Any], edit_text: str = "") -> bool:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    if difference_type != "object_presence":
        return False
    if _absence_like_phrase(from_value) or _absence_like_phrase(to_value):
        return False
    return "replace" in _normalized_phrase(edit_text) or bool(from_value and to_value)


def _is_object_removal(difference: dict[str, Any], edit_text: str = "") -> bool:
    difference_type = str(difference.get("type", "")).strip()
    from_value = str(difference.get("from", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    return bool(
        difference_type == "object_presence"
        and not _absence_like_phrase(from_value)
        and (_absence_like_phrase(to_value) or _normalized_phrase(edit_text).startswith("remove "))
    )


def _video_edit_source_object(difference: dict[str, Any], edit_text: str = "") -> str:
    from_value = str(difference.get("from", "")).strip()
    if from_value and not _absence_like_phrase(from_value):
        return from_value
    match = re.search(r"\breplace\s+(?:the\s+)?(.+?)\s+with\b", str(edit_text), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bremove\s+(?:the\s+)?(.+?)(?:\s+from|\s+in|\s*$)", str(edit_text), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _video_edit_target_object(difference: dict[str, Any], edit_text: str = "", edit_token: str = "") -> str:
    to_value = str(difference.get("to", "")).strip()
    if to_value and not _absence_like_phrase(to_value):
        return to_value
    match = re.search(r"\bwith\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|$)", str(edit_text), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return str(edit_token).strip()


def _video_edit_exclusion_keys(difference: dict[str, Any], *, edit_token: str = "", mask_query: str = "") -> set[str]:
    exclusions: set[str] = set()
    for value in (
        _video_edit_source_object(difference),
        _video_edit_target_object(difference, edit_token=edit_token),
        edit_token,
    ):
        key = _normalized_phrase(value)
        if key:
            exclusions.add(key)
    difference_type = str(difference.get("type", "")).strip()
    if difference_type != "scene":
        mask_key = _normalized_phrase(mask_query)
        if mask_key:
            exclusions.add(mask_key)
    if _is_existing_object_replacement(difference) or _is_object_removal(difference):
        from_value = _video_edit_source_object(difference)
        for token in TOKEN_PATTERN.findall(from_value.lower()):
            if token:
                exclusions.add(token)
    return exclusions


def _is_background_replace_edit(
    difference: dict[str, Any],
    edit_text: str = "",
    *,
    edit_region: str = "",
    mask_query: str = "",
    target_prompt: str = "",
) -> bool:
    if str(difference.get("type", "")).strip() not in {"scene", "background"}:
        return False
    source = _normalized_phrase(str(difference.get("from", "")).strip())
    target = _normalized_phrase(str(difference.get("to", "")).strip())
    if not source or not target or source == target:
        return False
    text = _normalized_phrase(
        " ".join(
            [
                str(edit_text),
                str(edit_region),
                str(mask_query),
                str(target_prompt),
                str(difference.get("description", "")),
                source,
                target,
            ]
        )
    )
    background_region = (
        "background" in text
        or _normalized_phrase(edit_region) == "scene"
        or _normalized_phrase(mask_query) == "background"
    )
    replace_verb = any(
        marker in text
        for marker in (
            "background to",
            "change background",
            "change the background",
            "replace background",
            "replace the background",
            "set the background",
            "swap background",
        )
    )
    return background_region and (replace_verb or "background" in source or "background" in target)


def _is_background_replace_lock_denied(value: str) -> bool:
    key = _normalized_phrase(value)
    if "preserve" in key and ("lighting" in key or "layout" in key):
        return True
    if "do not change" in key and ("lighting" in key or "layout" in key):
        return True
    return any(pattern in key for pattern in VACE_BACKGROUND_REPLACE_LOCK_DENY_PATTERNS)


def _background_replace_route_policy() -> dict[str, Any]:
    return {
        "plain_masked_vace_production": False,
        "plain_masked_vace_allowed_for": ["background_restyle", "soft_repaint", "low_structural_delta"],
        "recommended_route": DETERMINISTIC_BG_COMPOSITE_ROUTE,
        "fallback_route": VACE_BG_REPLACE_COMPOSITE_ROUTE,
        "refine_route": GUIDED_COMPOSITE_REFINE_VACE_ROUTE,
        "requires_composite_first_frame": False,
        "requires_fixed_reference_plate": True,
        "requires_vace": False,
        "deterministic_composite_production": True,
        "reason": "full background replacement should fix the target background plate by deterministic compositing; VACE is reserved for optional seam or generative repair",
    }


def _background_replace_target_background(difference: dict[str, Any], edit_text: str = "", edit_token: str = "") -> str:
    target = str(difference.get("to", "") or edit_token or "").strip()
    if not target:
        match = re.search(r"\bbackground\s+(?:to|into|with)\s+(.+?)(?:\.|$)", str(edit_text), flags=re.IGNORECASE)
        if match:
            target = match.group(1).strip()
    target_key = _normalized_phrase(target)
    if "laboratory" in target_key or "lab" in target_key:
        return (
            "a clean blue-white futuristic laboratory interior, with smooth illuminated wall panels "
            "and lab benches in the background"
        )
    target = re.sub(r"\boriginal\b", "", target, flags=re.IGNORECASE)
    target = re.sub(r"\bbackground\b", "", target, flags=re.IGNORECASE).strip(" .,;:")
    if not target:
        target = "the requested new environment"
    article = _indefinite_article_for_phrase(target)
    return f"{article} {target} background"


def _background_replace_foreground_clause(source_prompt: str) -> str:
    prompt = str(source_prompt).strip().rstrip(".")
    if not prompt:
        return "The foreground subject"
    prompt = re.split(r"\.\s*(?:scene:|main subjects:|actions:)", prompt, maxsplit=1, flags=re.IGNORECASE)[0]
    patterns = [
        r"\s+in\s+front\s+of\s+(?:a|an|the)?\s*[^.,;]*(?:background|room|wall|window|door|office|kitchen|studio|stage|street)[^.,;]*",
        r"\s+(?:in|inside|within|against|before)\s+(?:a|an|the)?\s*[^.,;]*(?:background|room|wall|window|door|office|kitchen|studio|stage|street)[^.,;]*",
        r"\s+with\s+(?:a|an|the)?\s*[^.,;]*(?:background|room|wall|window|door)[^.,;]*",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r"\s+", " ", prompt).strip(" .,;:")
    return prompt or "The foreground subject"


def _background_replace_target_prompt(source_prompt: str, difference: dict[str, Any], edit_text: str = "", edit_token: str = "") -> str:
    foreground = _background_replace_foreground_clause(source_prompt)
    target_background = _background_replace_target_background(difference, edit_text, edit_token)
    framing = "stable frontal medium-close-up framing" if "to the camera" in _normalized_phrase(source_prompt) else "stable camera framing"
    return f"{foreground} in {target_background}, {framing}."


def _filter_background_replace_preserve_tokens(preserve_tokens: list[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw_item in preserve_tokens:
        item = str(raw_item).strip()
        key = _normalized_phrase(item)
        if not item or not key or key in seen:
            continue
        if key in VACE_BACKGROUND_REPLACE_PRESERVE_DENY_MARKERS:
            continue
        if any(marker in key for marker in VACE_BACKGROUND_REPLACE_PRESERVE_DENY_MARKERS):
            continue
        filtered.append(item)
        seen.add(key)
    if not any(_normalized_phrase(item) == "camera framing" for item in filtered):
        filtered.append("camera framing")
    if not any("timing" in _normalized_phrase(item) for item in filtered):
        filtered.append("motion timing")
    return filtered[:10]


def _filter_background_replace_preserve_regions(preserve_regions: list[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw_item in preserve_regions:
        item = str(raw_item).strip()
        key = _normalized_phrase(item)
        if not item or not key or key in seen:
            continue
        if key in VACE_BACKGROUND_REPLACE_REGION_DENY_MARKERS:
            continue
        if any(marker in key for marker in VACE_BACKGROUND_REPLACE_REGION_DENY_MARKERS):
            continue
        filtered.append(item)
        seen.add(key)
    return filtered


def _background_replace_risk_locks(risk: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(risk or {})
    locks = [
        str(item).strip()
        for item in updated.get("locks", [])
        if str(item).strip() and not _is_background_replace_lock_denied(str(item))
    ]
    for lock in (
        "preserve foreground identity and face",
        "preserve foreground pose, speaking motion, and timing",
        "preserve camera framing",
    ):
        if lock not in locks:
            locks.append(lock)
    updated["locks"] = locks
    updated["background_replace_locks"] = {
        "foreground": {
            "preserve_identity": True,
            "preserve_face": True,
            "preserve_pose": True,
            "preserve_mouth_motion": True,
            "preserve_timing": True,
            "preserve_camera_framing": True,
        },
        "background": {
            "preserve_source_background": False,
            "preserve_source_layout": False,
            "preserve_source_lighting": False,
            "preserve_windows_doors_walls": False,
        },
    }
    return updated


def _filter_video_edit_preserve_tokens(
    preserve_tokens: list[str],
    *,
    difference: dict[str, Any],
    edit_token: str,
    mask_query: str = "",
) -> list[str]:
    exclusions = _video_edit_exclusion_keys(difference, edit_token=edit_token, mask_query=mask_query)
    filtered: list[str] = []
    seen: set[str] = set()
    for raw_item in preserve_tokens:
        item = str(raw_item).strip()
        key = _normalized_phrase(item)
        if not item or not key or key in seen:
            continue
        if key in exclusions or any(excluded and excluded in key.split() for excluded in exclusions):
            continue
        filtered.append(item)
        seen.add(key)
    return filtered


def _indefinite_article_for_phrase(phrase: str) -> str:
    key = _normalized_phrase(phrase)
    if not key:
        return "a"
    return "an" if key[0] in {"a", "e", "i", "o", "u"} else "a"


def _article_clothing_phrase(phrase: str) -> str:
    value = str(phrase).strip()
    key = _normalized_phrase(value)
    if not value or key.startswith(("a ", "an ", "the ")):
        return value
    if key.endswith("clothing"):
        return value
    return f"{_indefinite_article_for_phrase(value)} {value}"


def _is_clothing_edit(difference: dict[str, Any], edit_text: str = "", edit_token: str = "") -> bool:
    difference_type = str(difference.get("type", "")).strip()
    if difference_type != "attribute":
        return False
    target = _normalized_phrase(_video_edit_target_object(difference, edit_text, edit_token))
    combined = _normalized_phrase(
        " ".join(
            [
                str(edit_text),
                str(edit_token),
                str(difference.get("from", "")),
                str(difference.get("to", "")),
                str(difference.get("description", "")),
            ]
        )
    )
    return any(marker in target.split() for marker in VACE_CLOTHING_OBJECT_MARKERS) or any(
        marker in combined for marker in ("outfit", "clothing", "shirt", "jacket", "coat", "dress", "hoodie", "sweater")
    )


def _is_black_jacket_target(difference: dict[str, Any], edit_text: str = "", edit_token: str = "") -> bool:
    text = _normalized_phrase(
        " ".join(
            [
                _video_edit_target_object(difference, edit_text, edit_token),
                str(edit_text),
                str(edit_token),
                str(difference.get("to", "")),
                str(difference.get("description", "")),
            ]
        )
    )
    return "black jacket" in text


def _structural_clothing_edit_reason(
    difference: dict[str, Any],
    edit_text: str = "",
    edit_token: str = "",
    source_prompt: str = "",
) -> str:
    if not _is_clothing_edit(difference, edit_text, edit_token):
        return ""
    target_key = _normalized_phrase(_video_edit_target_object(difference, edit_text, edit_token))
    source_key = _normalized_phrase(" ".join([str(difference.get("from", "")), source_prompt]))
    combined = _normalized_phrase(
        " ".join(
            [
                target_key,
                str(edit_text),
                str(edit_token),
                str(difference.get("to", "")),
                str(difference.get("description", "")),
            ]
        )
    )
    if _is_black_jacket_target(difference, edit_text, edit_token):
        return "structural_clothing_tryon_required"
    if any(marker in combined for marker in VACE_STRUCTURAL_CLOTHING_TARGET_MARKERS):
        return "structural_clothing_tryon_required"
    target_is_outerwear = any(marker in target_key for marker in VACE_OUTERWEAR_MARKERS)
    source_is_outerwear = any(marker in source_key for marker in VACE_OUTERWEAR_MARKERS)
    source_is_non_outerwear = any(marker in source_key for marker in VACE_NON_OUTERWEAR_CLOTHING_MARKERS)
    if target_is_outerwear and source_is_non_outerwear and not source_is_outerwear:
        return "structural_clothing_tryon_required"
    return ""


def _black_jacket_target_prompt(source_prompt: str) -> str:
    source_key = _normalized_phrase(source_prompt)
    musician_markers = {"blue fedora", "ukulele", "microphone", "brick wall"}
    if len([marker for marker in musician_markers if marker in source_key]) >= 3:
        return VACE_BLACK_JACKET_PROMPT
    prompt = str(source_prompt).strip()
    target_with_article = _article_clothing_phrase(VACE_BLACK_JACKET_SRC_REF_TARGET)
    for source_clothing in _source_clothing_phrases(prompt):
        source_key = _normalized_phrase(source_clothing)
        if source_key and source_key not in {"black jacket", _normalized_phrase(VACE_BLACK_JACKET_SRC_REF_TARGET)}:
            prompt = re.sub(re.escape(source_clothing), target_with_article, prompt, count=1, flags=re.IGNORECASE)
            break
    if _normalized_phrase(VACE_BLACK_JACKET_REQUIRED_PHRASE) not in _normalized_phrase(prompt):
        prompt = f"{prompt.rstrip('.')} wearing {target_with_article}."
    return prompt.strip()


def _source_clothing_phrases(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:(?:a|an|the)\s+)?(?:(?:[a-z][a-z-]*)\s+){0,4}"
        r"(?:clothing|outfit|shirt|jacket|coat|dress|hoodie|sweater|vest)\b",
        flags=re.IGNORECASE,
    )
    phrases: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(str(text)):
        phrase = match.group(0).strip(" .,;:")
        if " and " in phrase.lower():
            phrase = re.split(r"\s+and\s+", phrase, flags=re.IGNORECASE)[-1].strip()
        phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase, flags=re.IGNORECASE).strip()
        key = _normalized_phrase(phrase)
        if not key or key in seen:
            continue
        phrases.append(phrase)
        seen.add(key)
    return phrases


def _clothing_target_prompt(*, source_prompt: str, edit_text: str, difference: dict[str, Any], edit_token: str = "") -> str:
    if _is_black_jacket_target(difference, edit_text, edit_token):
        return _black_jacket_target_prompt(source_prompt)
    target = _video_edit_target_object(difference, edit_text, edit_token) or edit_token or str(difference.get("to", "")).strip()
    target = target or "target clothing"
    target_with_article = _article_clothing_phrase(target)
    prompt = str(source_prompt).strip()
    for source_clothing in _source_clothing_phrases(prompt):
        source_key = _normalized_phrase(source_clothing)
        target_key = _normalized_phrase(target)
        if source_key and source_key != target_key:
            prompt = re.sub(re.escape(source_clothing), target_with_article, prompt, count=1, flags=re.IGNORECASE)
            break
    if _normalized_phrase(target) not in _normalized_phrase(prompt):
        prompt = f"{prompt.rstrip('.')} wearing {target_with_article}."
    return prompt.strip()


def _target_prompt_source_clothing_conflicts(*, source_prompt: str, target_prompt: str, target_clothing: str) -> list[str]:
    target_key = _normalized_phrase(target_clothing)
    target_text = _normalized_phrase(target_prompt)
    conflicts: list[str] = []
    for source_clothing in _source_clothing_phrases(source_prompt):
        source_key = _normalized_phrase(source_clothing)
        if source_key and source_key != target_key and source_key in target_text:
            conflicts.append(source_clothing)
    return conflicts


def _replacement_source_prompt_for_target(source_prompt: str, *, source_object: str, target_object: str) -> str:
    prompt = str(source_prompt).strip()
    source = _normalized_phrase(source_object)
    target = str(target_object).strip()
    if not prompt or not source or not target:
        return prompt
    source_regex = re.escape(str(source_object).strip())
    target_with_article = target if re.match(r"^(?:a|an|the)\s+", target, flags=re.IGNORECASE) else f"a {target}"
    contact_pattern = re.compile(
        rf"\b(?P<subject>[^.]*?)\b(?P<verb>sits|sit|sitting|seated)\s+(?P<prep>on|in)\s+"
        rf"(?:(?:a|an|the)\s+)?{source_regex}\b",
        flags=re.IGNORECASE,
    )
    match = contact_pattern.search(prompt)
    if match:
        subject = match.group("subject").strip()
        subject = re.sub(r"\b(?:is|are|was|were)$", "", subject, flags=re.IGNORECASE).strip()
        subject = subject if subject else "The subject"
        replacement = f"{subject} is seated on {target_with_article}"
        return (prompt[: match.start()] + replacement + prompt[match.end() :]).strip()
    return re.sub(source_regex, target, prompt, count=1, flags=re.IGNORECASE).strip()


def _video_edit_target_prompt(
    *,
    source_prompt: str,
    edit_text: str,
    difference: dict[str, Any],
    edit_token: str = "",
) -> str:
    difference_type = str(difference.get("type", "")).strip()
    to_value = str(difference.get("to", "")).strip()
    from_value = _video_edit_source_object(difference, edit_text)
    if _is_existing_object_replacement(difference, edit_text):
        target = _video_edit_target_object(difference, edit_text)
        base_prompt = _replacement_source_prompt_for_target(
            source_prompt,
            source_object=from_value,
            target_object=target,
        )
        edit_clause = (
            f"Replace only the {from_value} with {target}. "
            f"The same shot shows {target} in the original {from_value} location; no {from_value} is visible."
        )
    elif _is_object_removal(difference, edit_text):
        edit_clause = (
            f"Remove only the {from_value}. "
            f"The {from_value} area is clean and naturally filled; no {from_value} is visible."
        )
    elif difference_type == "object_presence" and to_value:
        edit_clause = f"Add only {to_value}."
    elif difference_type == "object_count" and to_value:
        edit_clause = f"Change only the count to {to_value}."
    elif _is_clothing_edit(difference, edit_text, edit_token):
        return _clothing_target_prompt(
            source_prompt=source_prompt,
            edit_text=edit_text,
            difference=difference,
            edit_token=edit_token,
        )
    elif difference_type == "attribute":
        edit_clause = f"Change only the specified attribute: {edit_text}."
    elif difference_type == "scene":
        if _is_background_replace_edit(difference, edit_text):
            return _background_replace_target_prompt(source_prompt, difference, edit_text, edit_token)
        target = to_value or str(difference.get("description", "")).strip() or edit_text
        edit_clause = (
            f"The same subject, camera, action timing, and layout are preserved while the background becomes {target}."
        )
    elif difference_type == "action":
        edit_clause = f"Change only the action: {edit_text}."
    elif difference_type == "audio_event":
        edit_clause = f"Change only the audio event: {edit_text}."
    else:
        edit_clause = f"Apply only this edit: {edit_text}."
    return f"{base_prompt if _is_existing_object_replacement(difference, edit_text) else source_prompt} {edit_clause}".strip()


def _video_edit_preserve_tokens(
    annotation: dict[str, Any],
    difference: dict[str, Any],
    edit_token: str,
    edit_text: str = "",
) -> list[str]:
    values: list[str] = []
    values.extend(_normalize_list(annotation.get("subjects", [])))
    values.extend(list(_normalize_object_counts(annotation.get("object_counts", {})).keys()))
    values.extend(_normalize_list(annotation.get("actions", [])))
    scene = str(annotation.get("scene", "")).strip()
    if scene:
        values.append(scene)
    values.extend(["camera motion", "lighting", "timing"])
    preserved = _filter_video_edit_preserve_tokens(
        _dedupe_strings([str(value).strip() for value in values if str(value).strip()]),
        difference=difference,
        edit_token=edit_token,
    )
    if _is_background_replace_edit(difference, edit_text):
        preserved = _filter_background_replace_preserve_tokens(preserved)
        return preserved[:10]
    return preserved[:8]


def _video_edit_risk_assessment(annotation: dict[str, Any], *, difference_type: str) -> dict[str, Any]:
    visible_text = _dedupe_strings(
        _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
    )
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))
    storyline = _dedupe_strings(_normalize_list(annotation.get("storyline", [])))
    events = annotation.get("events", [])
    event_count = len(events) if isinstance(events, list) else 0
    summary_tokens = _tokenize_text(str(annotation.get("summary", "")))
    scene_text = _normalized_phrase(str(annotation.get("scene", "")))
    risk_reasons: list[str] = []
    if visible_text:
        risk_reasons.append("visible_text_present")
    if difference_type != "action" and len(actions) >= 2:
        risk_reasons.append("multiple_actions")
    if difference_type != "action" and event_count >= 2:
        risk_reasons.append("multi_event_timeline")
    if len(subjects) >= 4:
        risk_reasons.append("many_subjects")
    if any(token in summary_tokens for token in {"speaks", "speaking", "talks", "talking", "vlogging", "interview"}):
        risk_reasons.append("speaking_person")
    if any(token in summary_tokens for token in {"transition", "transitions", "followed", "split", "screen", "cut"}):
        risk_reasons.append("scene_or_shot_change")
    if any(token in scene_text for token in ("ui", "screen", "interface", "control room")):
        risk_reasons.append("ui_or_text_heavy_scene")
    if storyline and len(storyline) >= 3 and difference_type != "action":
        risk_reasons.append("long_storyline")

    score = min(1.0, 0.18 * len(risk_reasons))
    allow_generation = not any(
        reason in set(risk_reasons)
        for reason in {
            "visible_text_present",
            "multiple_actions",
            "multi_event_timeline",
            "scene_or_shot_change",
            "ui_or_text_heavy_scene",
        }
    )
    risk_level = "low"
    if score >= 0.55 or not allow_generation:
        risk_level = "high"
    elif score >= 0.25:
        risk_level = "medium"
    locks = ["preserve camera motion, lighting, timing, and layout exactly"]
    if visible_text:
        locks.append("preserve all visible text exactly; do not alter letters, captions, labels, signs, subtitles, or UI text")
    if actions and difference_type != "action":
        locks.append("preserve the exact action and motion timing; do not change gestures, pose, order, or movement")
    if subjects:
        locks.append("preserve all existing people, subjects, and object identities")
    return {
        "score": round(score, 3),
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "allow_generation": allow_generation,
        "locks": locks,
    }


def _merge_video_edit_locks(negative_prompt: str, risk: dict[str, Any] | None = None) -> str:
    prompt = str(negative_prompt).strip()
    locks = [
        str(item).strip()
        for item in (risk or {}).get("locks", [])
        if str(item).strip()
    ]
    for lock in locks:
        if lock.lower() not in prompt.lower():
            prompt = f"{prompt} {lock}." if prompt else f"{lock}."
    return prompt.strip()


def _video_edit_negative_prompt(preserve_tokens: list[str], *, risk: dict[str, Any] | None = None) -> str:
    protected = ", ".join(preserve_tokens[:6]) if preserve_tokens else "the original subject, scene, camera, timing"
    prompt = (
        f"Do not change {protected}. Do not add extra people, change the scene, alter visible text, "
        "reorder shots, or introduce additional edits."
    )
    return _merge_video_edit_locks(prompt, risk)


def _target_prompt_contract_mentions_absence(prompt: str, source_object: str) -> bool:
    source = _normalized_phrase(source_object)
    text = _normalized_phrase(prompt)
    return bool(source and (f"no {source}" in text or f"without {source}" in text))


def _repair_video_edit_prompt_contract(
    *,
    source_prompt: str,
    target_prompt: str,
    edit_text: str,
    difference: dict[str, Any],
    edit_token: str,
    preserve_tokens: list[str],
    negative_prompt: str,
    mask_query: str,
    risk: dict[str, Any] | None,
) -> tuple[str, list[str], str, list[str]]:
    repairs: list[str] = []
    source_object = _video_edit_source_object(difference, edit_text)
    normalized_target = _normalized_phrase(target_prompt)
    if _is_existing_object_replacement(difference, edit_text):
        if "add only" in normalized_target or "replace" not in normalized_target:
            target_prompt = _video_edit_target_prompt(
                source_prompt=source_prompt,
                edit_text=edit_text,
                difference=difference,
                edit_token=edit_token,
            )
            repairs.append("target_prompt_rewritten_for_object_replacement")
        elif not _target_prompt_contract_mentions_absence(target_prompt, source_object):
            target_prompt = f"{target_prompt.rstrip('.')} No {source_object} is visible."
            repairs.append("target_prompt_added_source_absence")
    elif _is_object_removal(difference, edit_text):
        if "add only no" in normalized_target or (
            "remove" not in normalized_target and not _target_prompt_contract_mentions_absence(target_prompt, source_object)
        ):
            target_prompt = _video_edit_target_prompt(
                source_prompt=source_prompt,
                edit_text=edit_text,
                difference=difference,
                edit_token=edit_token,
            )
            repairs.append("target_prompt_rewritten_for_object_removal")
        elif not _target_prompt_contract_mentions_absence(target_prompt, source_object):
            target_prompt = f"{target_prompt.rstrip('.')} No {source_object} is visible."
            repairs.append("target_prompt_added_source_absence")
    elif _is_clothing_edit(difference, edit_text, edit_token):
        target_clothing = _video_edit_target_object(difference, edit_text, edit_token) or edit_token
        if (
            "change only" in normalized_target
            or (
                _is_black_jacket_target(difference, edit_text, edit_token)
                and (
                    _normalized_phrase(VACE_BLACK_JACKET_REQUIRED_PHRASE) not in normalized_target
                    or any(marker in normalized_target for marker in VACE_BLACK_JACKET_FORBIDDEN_PROMPT_MARKERS)
                )
            )
            or _target_prompt_source_clothing_conflicts(
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                target_clothing=target_clothing,
            )
        ):
            target_prompt = _video_edit_target_prompt(
                source_prompt=source_prompt,
                edit_text=edit_text,
                difference=difference,
                edit_token=edit_token,
            )
            repairs.append("target_prompt_rewritten_for_clothing_edit")

    if _is_background_replace_edit(difference, edit_text, mask_query=mask_query, target_prompt=target_prompt):
        repaired_target_prompt = _background_replace_target_prompt(source_prompt, difference, edit_text, edit_token)
        if _normalized_phrase(repaired_target_prompt) != _normalized_phrase(target_prompt):
            target_prompt = repaired_target_prompt
            repairs.append("target_prompt_rewritten_for_background_replace")
        repaired_preserve = _filter_background_replace_preserve_tokens(preserve_tokens)
        if repaired_preserve != preserve_tokens:
            preserve_tokens = repaired_preserve
            repairs.append("preserve_tokens_rewritten_for_background_replace")
        if _normalized_phrase(negative_prompt) != _normalized_phrase(VACE_BACKGROUND_REPLACE_NEGATIVE_PROMPT):
            negative_prompt = VACE_BACKGROUND_REPLACE_NEGATIVE_PROMPT
            repairs.append("negative_prompt_rewritten_for_background_replace")

    filtered_preserve = _filter_video_edit_preserve_tokens(
        preserve_tokens,
        difference=difference,
        edit_token=edit_token,
        mask_query=mask_query,
    )
    if filtered_preserve != preserve_tokens:
        preserve_tokens = filtered_preserve
        repairs.append("preserve_tokens_removed_edit_source")

    source_key = _normalized_phrase(source_object)
    negative_key = _normalized_phrase(negative_prompt)
    if not negative_prompt or (source_key and source_key in negative_key):
        negative_prompt = _video_edit_negative_prompt(preserve_tokens, risk=risk)
        repairs.append("negative_prompt_regenerated_without_edit_source")
    return target_prompt, preserve_tokens, negative_prompt, repairs


def _annotation_mentions_object(annotation: dict[str, Any], object_name: str) -> bool:
    object_key = _normalized_phrase(object_name)
    if not object_key:
        return False
    counts = _normalize_object_counts(annotation.get("object_counts", {}))
    if any(object_key == _normalized_phrase(name) for name in counts):
        return True
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                str(annotation.get("scene", "")),
                " ".join(_normalize_list(annotation.get("subjects", []))),
                " ".join(_normalize_list(annotation.get("actions", []))),
            ]
        )
    )
    return object_key in text


def _reference_has_screen_text_risk(annotation: dict[str, Any], source_object: str) -> bool:
    source_tokens = set(TOKEN_PATTERN.findall(_normalized_phrase(source_object)))
    if not source_tokens & VACE_SCREEN_TEXT_OBJECTS:
        return False
    visible_text = _normalize_list(annotation.get("visible_text", [])) + _normalize_list(annotation.get("on_screen_text", []))
    text = _normalized_phrase(
        " ".join(
            [str(annotation.get("summary", "")), str(annotation.get("scene", "")), " ".join(visible_text)]
        )
    )
    risky_markers = {"webpage", "website", "screen", "browser", "ui", "interface", "text", "logo", "caption"}
    return bool(visible_text or source_tokens & {"laptop", "computer", "screen", "monitor"} and any(marker in text for marker in risky_markers))


def _reference_has_seated_support_conflict(annotation: dict[str, Any], source_object: str) -> bool:
    source_tokens = set(TOKEN_PATTERN.findall(_normalized_phrase(source_object)))
    if not source_tokens & VACE_SEATED_SUPPORT_OBJECTS:
        return False
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                " ".join(_normalize_list(annotation.get("actions", []))),
                " ".join(_normalize_list(annotation.get("subjects", []))),
            ]
        )
    )
    return any(marker in text for marker in ("sit", "sits", "sitting", "seated", "seat", "sits in", "sits on"))


def _target_instance_allows_support_edit(target_instance_description: str) -> bool:
    text = _normalized_phrase(target_instance_description)
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "empty",
            "far right",
            "far left",
            "no one sitting",
            "not occupied",
            "unoccupied",
            "unused",
            "without anyone",
        )
    )


def _reference_has_multiple_visible_instances(annotation: dict[str, Any], source_object: str) -> bool:
    source_key = _normalized_phrase(source_object)
    if not source_key:
        return False
    source_tokens = set(TOKEN_PATTERN.findall(source_key))
    if not source_tokens & VACE_GENERIC_MULTI_INSTANCE_MASK_OBJECTS:
        return False
    counts = _normalize_object_counts(annotation.get("object_counts", {}))
    for raw_name, count in counts.items():
        if count > 1 and (
            _normalized_phrase(raw_name) == source_key
            or _text_mentions_phrase(raw_name, source_object)
            or _text_mentions_phrase(source_object, raw_name)
        ):
            return True
    text = _normalized_phrase(
        " ".join(
            [
                str(annotation.get("summary", "")),
                str(annotation.get("scene", "")),
                " ".join(_normalize_list(annotation.get("subjects", []))),
            ]
        )
    )
    plural_markers = {f"{token}s" for token in source_tokens} | {f"{token}es" for token in source_tokens}
    if source_key.endswith("y"):
        plural_markers.add(source_key[:-1] + "ies")
    return any(marker in text for marker in plural_markers)


def _target_prompt_conflicts_with_replacement_source_state(target_prompt: str, source_object: str) -> bool:
    source_key = _normalized_phrase(source_object)
    if not source_key or not _target_prompt_contract_mentions_absence(target_prompt, source_object):
        return False
    text = _normalized_phrase(target_prompt)
    state_patterns = [
        f"sitting on {source_key}",
        f"sitting in {source_key}",
        f"sits on {source_key}",
        f"sits in {source_key}",
        f"seated on {source_key}",
        f"seated in {source_key}",
        f"standing on {source_key}",
        f"lying on {source_key}",
    ]
    articles = ("a", "an", "the")
    state_patterns.extend(
        f"{prefix} {article} {source_key}"
        for prefix in (
            "sitting on",
            "sitting in",
            "sits on",
            "sits in",
            "seated on",
            "seated in",
            "standing on",
            "lying on",
        )
        for article in articles
    )
    return any(pattern in text for pattern in state_patterns)


def _webvid_style_edit_lint_errors(
    *,
    source_prompt: str,
    target_prompt: str,
    edit_text: str,
    difference: dict[str, Any],
) -> list[str]:
    text = _normalized_phrase(
        " ".join(
            [
                source_prompt,
                target_prompt,
                edit_text,
                str(difference.get("from", "")),
                str(difference.get("to", "")),
                str(difference.get("description", "")),
            ]
        )
    )
    errors: list[str] = []
    if any(marker in text for marker in VACE_TEXT_OR_LOGO_EDIT_MARKERS):
        errors.append("visible_text_or_logo_edit")
    if any(marker in text for marker in VACE_BROAD_SCENE_EDIT_MARKERS):
        errors.append("broad_scene_or_subject_replacement")
    if "shutterstock" in text or "stock clip" in text:
        errors.append("loose_stock_pair_not_vace_editable")
    return errors


def _video_edit_plan_lint(
    *,
    source_prompt: str,
    target_prompt: str,
    edit_text: str,
    difference: dict[str, Any],
    edit_token: str,
    preserve_tokens: list[str],
    negative_prompt: str,
    reference_annotation: dict[str, Any],
    mask_query: str = "",
    preserve_regions: list[str] | None = None,
    risk: dict[str, Any] | None = None,
    target_instance_description: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    target_key = _normalized_phrase(target_prompt)
    source_object = _video_edit_source_object(difference, edit_text)
    source_key = _normalized_phrase(source_object)
    preserve_keys = {_normalized_phrase(item) for item in preserve_tokens}
    negative_key = _normalized_phrase(negative_prompt)

    if "add only no" in target_key:
        errors.append("target_prompt_contains_add_only_no")
    if _is_existing_object_replacement(difference, edit_text) and "add only" in target_key:
        errors.append("replacement_target_prompt_uses_add_instead_of_replace")
    if _is_existing_object_replacement(difference, edit_text) and _target_prompt_conflicts_with_replacement_source_state(
        target_prompt,
        source_object,
    ):
        errors.append("replacement_target_prompt_conflicts_with_source_state")
    if source_key and (source_key in preserve_keys or any(source_key == key for key in preserve_keys)):
        errors.append("preserve_tokens_lock_edit_source")
    if source_key and source_key in negative_key:
        errors.append("negative_prompt_locks_edit_source")
    if _is_background_replace_edit(difference, edit_text, mask_query=mask_query, target_prompt=target_prompt):
        source_background_markers = _background_source_markers(
            {
                "reference_caption": source_prompt,
                "difference": difference,
                "generation": {
                    "source_prompt": source_prompt,
                    "preserve_regions": preserve_regions or [],
                },
            }
        )
        for marker in source_background_markers:
            if _text_mentions_phrase(target_key, marker):
                errors.append("target_prompt_contains_source_background")
                break
        preserve_text = _normalized_phrase(" ".join(preserve_tokens))
        if any(_text_mentions_phrase(preserve_text, marker) for marker in source_background_markers):
            errors.append("preserve_tokens_contain_source_background")
        preserve_region_text = _normalized_phrase(" ".join(preserve_regions or []))
        if any(_text_mentions_phrase(preserve_region_text, marker) for marker in VACE_BACKGROUND_REPLACE_REGION_DENY_MARKERS):
            errors.append("preserve_regions_contain_source_background_region")
        if any(_is_background_replace_lock_denied(item) for item in [negative_prompt] + _normalize_list((risk or {}).get("locks", []))):
            errors.append("background_replace_contains_source_layout_or_lighting_lock")
    if _is_clothing_edit(difference, edit_text, edit_token):
        structural_clothing_reason = _structural_clothing_edit_reason(
            difference,
            edit_text,
            edit_token,
            source_prompt,
        )
        if structural_clothing_reason:
            errors.append(structural_clothing_reason)
        target_clothing = _video_edit_target_object(difference, edit_text, edit_token) or edit_token
        if "change only" in target_key:
            errors.append("clothing_target_prompt_uses_operation_instruction")
        if _target_prompt_source_clothing_conflicts(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            target_clothing=target_clothing,
        ):
            errors.append("target_prompt_preserves_source_clothing")
        if _is_black_jacket_target(difference, edit_text, edit_token):
            if _normalized_phrase(VACE_BLACK_JACKET_REQUIRED_PHRASE) not in target_key:
                errors.append("black_jacket_target_prompt_missing_open_black_long_sleeved_jacket")
            for marker in sorted(VACE_BLACK_JACKET_FORBIDDEN_PROMPT_MARKERS):
                if marker in target_key:
                    errors.append(f"black_jacket_target_prompt_forbidden_marker:{marker}")

    if (_is_existing_object_replacement(difference, edit_text) or _is_object_removal(difference, edit_text)) and source_object:
        if not _annotation_mentions_object(reference_annotation, source_object):
            if _is_existing_object_replacement(difference, edit_text):
                errors.append("object_replacement_source_not_visible")
            else:
                warnings.append("edit_source_not_clearly_present_in_annotation")
    if _is_existing_object_replacement(difference, edit_text) and _reference_has_screen_text_risk(reference_annotation, source_object):
        errors.append("object_replacement_screen_or_visible_text_risk")
    if _is_existing_object_replacement(difference, edit_text) and _reference_has_seated_support_conflict(reference_annotation, source_object):
        if not _target_instance_allows_support_edit(target_instance_description):
            errors.append("object_replacement_breaks_support_contact")
    if _is_object_removal(difference, edit_text) and _reference_has_seated_support_conflict(reference_annotation, source_object):
        errors.append("object_removal_breaks_seated_support")
    if (
        (_is_existing_object_replacement(difference, edit_text) or _is_object_removal(difference, edit_text))
        and source_object
        and _normalized_phrase(mask_query) == _normalized_phrase(source_object)
        and _reference_has_multiple_visible_instances(reference_annotation, source_object)
        and not str(target_instance_description).strip()
    ):
        errors.append("ambiguous_multi_instance_mask_query")
    errors.extend(
        _webvid_style_edit_lint_errors(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_text=edit_text,
            difference=difference,
        )
    )

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _video_edit_control_plan(route: str) -> list[str]:
    if route == "vace_controlled":
        return ["first_frame_reference", "local_roi_mask", "depth_or_lineart_control"]
    if route == "tokenflow_style":
        return ["first_frame_reference", "tokenflow_consistency", "local_roi_mask"]
    if route == "ltx2_retake":
        return ["first_frame_reference", "retake_reference", "motion_consistency_check"]
    return []


def _video_edit_generation_defaults(route: str) -> dict[str, Any]:
    return {
        "gpu_ids": "0,1",
        "offload_model": False,
        "frame_count": 49,
        "steps": 25,
        "resolution": "832x480",
        "postprocess": {"audio_copied_from_reference": True},
    }


def _audio_expected_event(difference: dict[str, Any], edit_text: str) -> str:
    for field_name in ("to", "description", "from"):
        value = str(difference.get(field_name, "")).strip()
        if value and not _absence_like_phrase(value) and not _is_speech_only_audio_phrase(value):
            return value[:120]
    tokens = [token for token in TOKEN_PATTERN.findall(edit_text.lower()) if token in NON_SPEECH_AUDIO_TOKENS]
    return " ".join(tokens[:4])[:120]


def _synthetic_audio_expected_event(record: dict[str, Any]) -> str:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    audio_plan = generation.get("audio_edit_plan", {}) if isinstance(generation.get("audio_edit_plan"), dict) else {}
    expected_event = str(audio_plan.get("expected_event", "")).strip()
    if expected_event:
        return expected_event
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    return _audio_expected_event(difference, str(record.get("edit_text", "")))


def _audio_terms_mention_event(terms: list[str], expected_event: str) -> bool:
    expected_tokens = _tokenize_text(expected_event) - {"audio", "event", "sound", "sounds", "noise", "no"}
    if not expected_tokens:
        return False
    for term in terms:
        if _text_mentions_phrase(term, expected_event):
            return True
        term_tokens = _tokenize_text(term)
        if expected_tokens.issubset(term_tokens):
            return True
        if _jaccard(expected_tokens, term_tokens) >= 0.5:
            return True
    return False


def _audio_edit_route(expected_event: str, annotation: dict[str, Any]) -> str:
    event = _normalized_phrase(expected_event)
    if any(token in event for token in ("footstep", "walking", "scratch", "writing", "whoosh", "splash")):
        return "foleycrafter_temporal"
    return "deterministic_overlay"


def _safe_audio_ideation_candidate(candidate: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any] | None:
    suggestion = _audio_edit_suggestion(annotation)
    if suggestion is None:
        return None
    source_edit_text = str(candidate.get("edit_text", "")).strip()
    event = str(suggestion["expected_event"]).strip()
    edit_text = str(suggestion["edit_text"]).strip()
    proposal_seed = str(candidate.get("proposal_id", "")) or str(candidate.get("reference_video", "")) + edit_text
    revised = dict(candidate)
    revised["proposal_id"] = f"{str(candidate.get('proposal_id', '')).strip() or 'candidate'}__audio_ideation_{_stable_hash(proposal_seed)[:8]}"
    revised["edit_text"] = edit_text
    revised["difference"] = {
        "type": "audio_event",
        "from": f"no {event}",
        "to": event,
        "description": str(suggestion["description"]).strip(),
    }
    revised["source_candidate_edit_text"] = source_edit_text
    revised["source_candidate_difference"] = candidate.get("difference", {})
    revised["candidate_source"] = "safe_audio_ideation_from_reference"
    revised["ideation_reason"] = str(suggestion["reason"]).strip()
    return revised


def _audio_edit_suggestion(annotation: dict[str, Any]) -> dict[str, str] | None:
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))
    scene = str(annotation.get("scene", "")).strip()
    summary = str(annotation.get("summary", "")).strip()
    text = _normalized_phrase(" ".join([summary, scene, " ".join(actions), " ".join(subjects)]))
    suggestions = (
        (
            ("writing", "write", "pen", "pencil"),
            "scratching sound",
            "add a scratching sound synchronized with the writing",
            "A pen scratching sound is synchronized with the visible writing motion.",
            "visible writing motion can support a synchronized Foley sound",
        ),
        (
            ("jumping", "jump", "launched", "launch", "flying", "gliding"),
            "whoosh",
            "add a whoosh sound synchronized with the jump or launch",
            "A short whoosh sound is synchronized with the visible jump or launch.",
            "visible jump or launch motion can support a synchronized Foley sound",
        ),
        (
            ("walking", "walk", "running", "run", "foot", "steps"),
            "footsteps",
            "add footsteps synchronized with the walking or running",
            "Footsteps are synchronized with the visible walking or running.",
            "visible walking or running can support synchronized footsteps",
        ),
        (
            ("clapping", "clap", "applaud", "applause"),
            "applause",
            "add applause to the audio",
            "Applause is added to match the visible clapping or audience context.",
            "visible clapping or audience context can support applause",
        ),
        (
            ("water", "river", "ocean", "waves", "splash"),
            "water splash",
            "add a water splash sound",
            "A water splash sound is added to match the visible water context.",
            "visible water context can support a water Foley sound",
        ),
        (
            ("forest", "trees", "outdoor", "wind"),
            "wind ambience",
            "add soft wind ambience to the audio",
            "Wind ambience is added while preserving the video stream.",
            "outdoor or forest context can support ambient wind",
        ),
    )
    for markers, expected_event, edit_text, description, reason in suggestions:
        if any(marker in text for marker in markers):
            return {
                "expected_event": expected_event,
                "edit_text": edit_text,
                "description": description,
                "reason": reason,
            }
    return None


def _audio_edit_reference_understanding(annotation: dict[str, Any]) -> dict[str, Any]:
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:6]
    subjects = _dedupe_strings(_normalize_list(annotation.get("subjects", [])))[:6]
    audio_events = _dedupe_strings(_non_speech_audio_terms(annotation))[:6]
    visible_text = _dedupe_strings(
        _normalize_list(annotation.get("visible_text", []))
        + _normalize_list(annotation.get("on_screen_text", []))
    )[:6]
    suggestion = _audio_edit_suggestion(annotation)
    suggested_events: list[dict[str, Any]] = []
    if suggestion is not None:
        suggested_events.append(
            {
                "expected_event": suggestion["expected_event"],
                "edit_text": suggestion["edit_text"],
                "reason": suggestion["reason"],
                "route": _audio_edit_route(suggestion["expected_event"], annotation),
                "timing_strategy": _audio_timing_strategy(suggestion["expected_event"], annotation),
            }
        )
    return {
        "main_subjects": subjects,
        "visible_actions": actions,
        "existing_non_speech_audio_events": audio_events,
        "visible_text": visible_text,
        "scene": str(annotation.get("scene", "")).strip(),
        "suggested_non_speech_audio_events": suggested_events,
        "bad_audio_edits": [
            "speech topic change",
            "transcript change",
            "narration-only change",
            "voiceover-only change",
            "unrelated music that conflicts with visible context",
        ],
    }


def _audio_edit_route_suitability(
    *,
    expected_event: str,
    difference: dict[str, Any],
    edit_text: str,
    reference_annotation: dict[str, Any],
) -> dict[str, Any]:
    issues = _speech_content_edit_issues(edit_text=edit_text, difference=difference)
    if issues:
        return {
            "allow_generation": False,
            "reason": "speech_content_or_speech_only_audio",
            "issues": issues,
        }
    if _audio_terms_mention_event(_non_speech_audio_terms(reference_annotation), expected_event):
        return {
            "allow_generation": False,
            "reason": "reference_already_has_expected_audio_event",
        }
    route = _audio_edit_route(expected_event, reference_annotation)
    timing = _audio_timing_strategy(expected_event, reference_annotation)
    priority = "S" if route == "foleycrafter_temporal" and timing == "visual_sync" else "A"
    return {
        "allow_generation": True,
        "reason": "contextual_non_speech_audio_edit",
        "route": route,
        "timing_strategy": timing,
        "priority": priority,
    }


def _audio_edit_prompt(expected_event: str, annotation: dict[str, Any], edit_text: str) -> str:
    actions = _dedupe_strings(_normalize_list(annotation.get("actions", [])))[:3]
    if actions:
        return f"{expected_event}, synchronized with {', '.join(actions)}"
    return f"{expected_event}. {edit_text}".strip()


def _audio_timing_strategy(expected_event: str, annotation: dict[str, Any]) -> str:
    event = _normalized_phrase(expected_event)
    if any(token in event for token in ("ambient", "wind", "rain", "waves", "hum", "music")):
        return "whole_clip_ambience"
    if annotation.get("events") or annotation.get("actions"):
        return "visual_sync"
    return "fixed_timestamp"


def _annotation_for_known_pair(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    pair: dict[str, Any],
    clip_id_field: str,
    video_field: str,
    line_number: int,
) -> dict[str, Any]:
    clip_id = str(pair.get(clip_id_field, "")).strip()
    if clip_id and clip_id in lookup:
        return lookup[clip_id]

    video_path = str(pair.get(video_field, "")).strip()
    if video_path:
        resolved = _resolve_under_root(root, video_path)
        for key in _path_lookup_keys(root, resolved, video_path):
            if key in lookup:
                return lookup[key]
    raise ValueError(f"known pair line {line_number}: cannot resolve {clip_id_field} or {video_field}")


def _known_pair_video_path(
    root: Path,
    pair: dict[str, Any],
    annotation: dict[str, Any],
    field_name: str,
) -> str:
    raw_value = str(pair.get(field_name, "")).strip()
    if raw_value:
        return _display_path(root, _resolve_under_root(root, raw_value))
    return _display_path(root, _resolve_under_root(root, str(annotation.get("output_path", ""))))


def _known_pair_model_fields(
    *,
    pair: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
) -> dict[str, Any]:
    difference = dict(pair.get("difference") or {})
    if not difference:
        difference = _detect_primary_difference(reference_annotation, target_annotation) or {}
        difference.pop("changed_types", None)
    if not difference:
        difference = {
            "type": "attribute",
            "from": "",
            "to": "",
            "description": str(pair.get("edit_text", "")).strip(),
        }
    difference_type = str(difference.get("type", "")).strip()
    edit_text = str(pair.get("edit_text", "")).strip() or _build_fallback_edit_text(difference)
    modalities = pair.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        modalities = _infer_pair_modalities(reference_annotation, target_annotation, difference_type)
    return {
        "edit_text": edit_text,
        "modalities": [str(item).strip() for item in modalities if str(item).strip()],
        "reference_caption": str(pair.get("reference_caption") or reference_annotation.get("summary", "")).strip(),
        "target_caption": str(pair.get("target_caption") or target_annotation.get("summary", "")).strip(),
        "difference": difference,
        "proposal_reason": str(pair.get("proposal_reason", "known pair validation")).strip(),
    }


def _synthetic_pair_source_matches_reference(pair: dict[str, Any]) -> bool:
    if str(pair.get("source_type", "synthetic_edit")).strip() != "synthetic_edit":
        return False
    generation = pair.get("generation")
    if not isinstance(generation, dict):
        return False
    source_video = _normalized_path_text(generation.get("source_video", ""))
    reference_video = _normalized_path_text(pair.get("reference_video", ""))
    if not source_video or not reference_video:
        return False
    return bool(
        source_video == reference_video
        or source_video.endswith("/" + reference_video)
        or reference_video.endswith("/" + source_video)
    )


def _normalized_path_text(value: Any) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


def _known_pair_source_context(pair: dict[str, Any]) -> dict[str, Any]:
    source_context = pair.get("source_context")
    if isinstance(source_context, dict) and source_context:
        normalized = dict(source_context)
        normalized.setdefault("relation", "known_pair")
        normalized.setdefault("score", 0.9)
        return normalized
    if _synthetic_pair_source_matches_reference(pair):
        return {
            "relation": "synthetic_from_reference",
            "score": 0.95,
            "generation_source_video": str(pair.get("generation", {}).get("source_video", "")).strip(),
        }
    return {"relation": "synthetic_edit", "score": 0.9}


def _known_pair_hard_negative_annotations(
    *,
    root: Path,
    lookup: dict[str, dict[str, Any]],
    annotations: list[dict[str, Any]],
    pair: dict[str, Any],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw_value in pair.get("hard_negatives", []) if isinstance(pair.get("hard_negatives"), list) else []:
        raw_path = str(raw_value).strip()
        if not raw_path:
            continue
        resolved = _resolve_under_root(root, raw_path)
        for key in _path_lookup_keys(root, resolved, raw_path):
            if key in lookup and lookup[key].get("clip_id") not in {
                reference_annotation.get("clip_id"),
                target_annotation.get("clip_id"),
            }:
                selected.append(lookup[key])
                break
    if selected:
        unique: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for annotation in selected:
            clip_id = str(annotation.get("clip_id", "")).strip()
            if clip_id and clip_id not in seen_ids:
                unique.append(annotation)
                seen_ids.add(clip_id)
        return unique[:3]
    return _select_hard_negative_annotations(
        reference_annotation=reference_annotation,
        target_annotation=target_annotation,
        annotations=annotations,
        primary_difference=difference,
    )


def _known_pair_hard_negative_paths(
    *,
    root: Path,
    pair: dict[str, Any],
    hard_negative_annotations: list[dict[str, Any]],
) -> list[str]:
    raw_values = pair.get("hard_negatives", [])
    if isinstance(raw_values, list) and raw_values:
        return [_display_path(root, _resolve_under_root(root, str(item).strip())) for item in raw_values if str(item).strip()]
    return [
        _display_path(root, _resolve_under_root(root, str(annotation.get("output_path", ""))))
        for annotation in hard_negative_annotations
        if str(annotation.get("output_path", "")).strip()
    ][:3]


def _known_pair_base_quality(
    *,
    root: Path,
    pair: dict[str, Any],
    annotations: list[dict[str, Any]],
    reference_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    difference: dict[str, Any],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    provided = pair.get("quality") if isinstance(pair.get("quality"), dict) else {}
    semantic_context_score = _same_context_score(reference_annotation, target_annotation)
    same_context_score = _pair_context_score(
        semantic_context_score=semantic_context_score,
        source_context=source_context,
    )
    synthetic_context_override = str(source_context.get("relation", "")).strip() == "synthetic_from_reference"
    if synthetic_context_override:
        same_context_score = max(same_context_score, _score_float(source_context.get("score")))
    detected_difference = _detect_primary_difference(reference_annotation, target_annotation)
    changed_types = list(detected_difference.get("changed_types", [])) if detected_difference else [str(difference.get("type", "")).strip()]
    quality: dict[str, Any] = {
        "same_context_score": _score_float(provided.get("same_context_score", same_context_score)),
        "edit_match_score": _score_float(provided.get("edit_match_score", 0.75)),
        "target_uniqueness_score": _score_float(
            provided.get(
                "target_uniqueness_score",
                _target_uniqueness_score(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    annotations=annotations,
                    primary_difference=difference,
                ),
            )
        ),
        "difference_strength_score": _score_float(
            provided.get(
                "difference_strength_score",
                _difference_strength_score(
                    reference_annotation=reference_annotation,
                    target_annotation=target_annotation,
                    primary_difference=difference,
                    changed_types=changed_types,
                ),
            )
        ),
    }
    visual_score = provided.get("visual_near_duplicate_score")
    if visual_score is None:
        visual_score = _visual_near_duplicate_score(
            _resolve_under_root(root, str(reference_annotation.get("output_path", ""))),
            _resolve_under_root(root, str(target_annotation.get("output_path", ""))),
        )
    if visual_score is not None:
        quality["visual_near_duplicate_score"] = _score_float(visual_score)
    if synthetic_context_override:
        quality["synthetic_context_override"] = 1.0
    return quality


def _synthetic_generation_route(generation: dict[str, Any]) -> str:
    actual_route = str(generation.get("generation_route", "")).strip()
    if actual_route:
        return actual_route
    route = str(generation.get("model_route", "")).strip()
    if route:
        return route
    audio_plan = generation.get("audio_edit_plan", {})
    if isinstance(audio_plan, dict):
        return str(audio_plan.get("route", "")).strip()
    return ""


def _is_audio_synthetic_route(route: str) -> bool:
    return route in SYNTHETIC_AUDIO_ROUTES


def _background_replace_actual_route(generation: dict[str, Any]) -> str:
    explicit = str(generation.get("background_replace_route", "")).strip()
    if explicit:
        return explicit
    policy = generation.get("background_replace_policy")
    if isinstance(policy, dict):
        actual = str(policy.get("actual_route", "")).strip()
        if actual:
            return actual
    return ""


def _plain_background_replacement_vace_issue(record: dict[str, Any], generation: dict[str, Any]) -> str:
    route = _synthetic_generation_route(generation)
    if route != "vace_controlled":
        return ""
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    edit_text = str(record.get("edit_text", "")).strip()
    if not _is_background_replace_edit(
        difference,
        edit_text,
        edit_region=str(generation.get("edit_region", "") or record.get("edit_region", "")).strip(),
        mask_query=str(generation.get("mask_query", "") or record.get("mask_query", "")).strip(),
        target_prompt=str(generation.get("target_prompt", "") or record.get("target_prompt", "")).strip(),
    ):
        return ""
    actual_route = _background_replace_actual_route(generation)
    if actual_route in {
        VACE_BG_REPLACE_COMPOSITE_ROUTE,
        DETERMINISTIC_BG_COMPOSITE_ROUTE,
        GUIDED_COMPOSITE_REFINE_VACE_ROUTE,
    }:
        return ""
    return "full background replacement requires composite-first-frame or deterministic composite route; plain masked VACE is experiment-only"


def _known_pair_generation_issues(record: dict[str, Any]) -> list[str]:
    source_type = str(record.get("source_type", "")).strip() or "natural"
    if source_type not in ALLOWED_SOURCE_TYPES:
        return [f"unsupported source_type: {source_type}"]
    if source_type != "synthetic_edit":
        return []
    generation = record.get("generation")
    if not isinstance(generation, dict) or not generation:
        return ["synthetic_edit pair is missing generation metadata"]
    issues: list[str] = []
    route = _synthetic_generation_route(generation)
    for field_name in ("model", "source_video", "model_route"):
        if not str(generation.get(field_name, "")).strip():
            issues.append(f"generation.{field_name} is required for synthetic_edit pairs")
    if _is_audio_synthetic_route(route):
        audio_plan = generation.get("audio_edit_plan")
        if not isinstance(audio_plan, dict) or not audio_plan:
            issues.append("generation.audio_edit_plan is required for synthetic audio pairs")
        else:
            for field_name in ("audio_prompt", "expected_event"):
                if not str(audio_plan.get(field_name, "")).strip():
                    issues.append(f"generation.audio_edit_plan.{field_name} is required for synthetic audio pairs")
            if not _boolish(audio_plan.get("preserve_video")):
                issues.append("generation.audio_edit_plan.preserve_video=true is required for synthetic audio pairs")
        return issues

    for field_name in ("prompt", "source_prompt", "target_prompt"):
        if not str(generation.get(field_name, "")).strip():
            issues.append(f"generation.{field_name} is required for synthetic visual pairs")
    review_inputs_dir = str(generation.get("review_inputs_dir", "")).strip()
    if not review_inputs_dir:
        issues.append("generation.review_inputs_dir is required for synthetic visual pairs")
    preserve_tokens = generation.get("preserve_tokens")
    if not isinstance(preserve_tokens, list) or not [item for item in preserve_tokens if str(item).strip()]:
        issues.append("generation.preserve_tokens is required for synthetic visual pairs")
    postprocess = generation.get("postprocess")
    if not isinstance(postprocess, dict) or "audio_copied_from_reference" not in postprocess:
        issues.append("generation.postprocess.audio_copied_from_reference is required for synthetic visual pairs")
    elif not str(postprocess.get("raw_generated_video", "")).strip():
        issues.append("generation.postprocess.raw_generated_video is required for synthetic visual pairs")
    duration_metrics = generation.get("duration_metrics")
    if not isinstance(duration_metrics, dict) or not duration_metrics:
        issues.append("generation.duration_metrics is required for synthetic visual pairs")
    else:
        duration_gate = duration_metrics.get("duration_gate")
        if not isinstance(duration_gate, dict):
            issues.append("generation.duration_metrics.duration_gate is required for synthetic visual pairs")
        elif not _boolish(duration_gate.get("passed")):
            issues.append("generation.duration_metrics.duration_gate.passed=true is required for synthetic visual pairs")
    post_vace_verdict = generation.get("post_vace_verdict")
    if not isinstance(post_vace_verdict, dict) or not post_vace_verdict:
        issues.append("generation.post_vace_verdict is required for synthetic visual pairs")
    edit_token = str(generation.get("edit_token") or record.get("edit_token") or "").strip()
    structural_clothing_reason = _structural_clothing_edit_reason(
        record.get("difference", {}) if isinstance(record.get("difference"), dict) else {},
        str(record.get("edit_text", "")).strip(),
        edit_token,
        str(generation.get("source_prompt", "")).strip(),
    )
    if structural_clothing_reason and route == "vace_controlled":
        issues.append("structural clothing edit requires try-on route instead of vace_controlled")
    difference = record.get("difference", {}) if isinstance(record.get("difference"), dict) else {}
    edit_text = str(record.get("edit_text", "")).strip()
    plain_background_issue = _plain_background_replacement_vace_issue(record, generation)
    if plain_background_issue:
        issues.append(plain_background_issue)
    source_object = _video_edit_source_object(difference, edit_text)
    if route == "vace_controlled" and _is_existing_object_replacement(difference, edit_text):
        if _target_prompt_conflicts_with_replacement_source_state(
            str(generation.get("target_prompt", "")).strip(),
            source_object,
        ):
            issues.append("replacement target prompt conflicts with source object state")
        if _reference_has_seated_support_conflict(
            {"summary": str(generation.get("source_prompt", "")).strip(), "actions": []},
            source_object,
        ) and not _target_instance_allows_support_edit(str(generation.get("target_instance_description", "")).strip()):
            issues.append("support-contact object replacement requires a non-VACE route or explicit unoccupied target instance")
    if route == "vace_controlled":
        if not str(generation.get("src_video_for_vace", "")).strip():
            issues.append("generation.src_video_for_vace is required for vace_controlled pairs")
        if not str(generation.get("src_mask", "")).strip():
            issues.append("generation.src_mask is required for vace_controlled pairs")
        if int(generation.get("mask_semantics_version") or 0) < VIDEO_MASK_SEMANTICS_VERSION:
            issues.append("generation.mask_semantics_version is required and must be current for vace_controlled pairs")
        if str(generation.get("mask_polarity", "")).strip() != VIDEO_MASK_POLARITY:
            issues.append("generation.mask_polarity must be white_generate_black_preserve for vace_controlled pairs")
        mask_metrics = generation.get("mask_metrics")
        if not isinstance(mask_metrics, dict) or not mask_metrics:
            issues.append("generation.mask_metrics is required for vace_controlled pairs")
        target_instance_description = str(generation.get("target_instance_description", "")).strip()
        mask_alignment = generation.get("mask_target_instance_alignment")
        if target_instance_description and not (
            isinstance(mask_alignment, dict) and _boolish(mask_alignment.get("passed"))
        ):
            issues.append("generation.mask_target_instance_alignment.passed=true is required for target-instance edits")
    return issues


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"jsonl file not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_number}: expected a JSON object")
        records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_under_root(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _as_non_negative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _parse_sources(raw_sources: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in raw_sources:
        if "=" not in raw:
            raise ValueError(f"source must use dataset=/path form: {raw}")
        dataset_name, raw_path = raw.split("=", 1)
        dataset_name = dataset_name.strip()
        raw_path = raw_path.strip()
        if not dataset_name or not raw_path:
            raise ValueError(f"source must use dataset=/path form: {raw}")
        parsed.append((dataset_name, Path(raw_path)))
    return parsed


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
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


def _first_unique(candidates: list[str], excluded: list[str]) -> str:
    excluded_lower = {item.lower() for item in excluded}
    for candidate in candidates:
        if candidate.lower() not in excluded_lower:
            return candidate
    return ""


def _first_item(values: list[str]) -> str:
    return values[0] if values else ""


def _tokenize_values(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(_tokenize_text(value))
    return tokens


def _tokenize_text(value: str) -> set[str]:
    tokens = set()
    for match in TOKEN_PATTERN.finditer(value.lower()):
        token = match.group(0)
        if token in STOPWORDS or len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Composed Omni Retrieval data helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_layout = subparsers.add_parser("init-layout")
    init_layout.add_argument("--root", default=DEFAULT_DATA_ROOT)

    index_raw = subparsers.add_parser("index-raw")
    index_raw.add_argument("--root", default=DEFAULT_DATA_ROOT)
    index_raw.add_argument(
        "--source",
        action="append",
        default=[],
        help="dataset=/absolute/path. If omitted, discover immediate children under <root>/raw_datasets.",
    )
    index_raw.add_argument("--output-path")

    extract_clips_parser = subparsers.add_parser("extract-clips")
    extract_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    extract_clips_parser.add_argument("--plan-path", required=True)
    extract_clips_parser.add_argument("--raw-index-path")
    extract_clips_parser.add_argument("--output-manifest-path")
    extract_clips_parser.add_argument("--dry-run", action="store_true")
    extract_clips_parser.add_argument("--overwrite", action="store_true")

    plan_detective_parser = subparsers.add_parser("plan-detective-clips")
    plan_detective_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_detective_parser.add_argument("--source-clips-path", required=True)
    plan_detective_parser.add_argument("--clip-plan-output-path")
    plan_detective_parser.add_argument("--clip-groups-output-path")
    plan_detective_parser.add_argument("--max-source-videos", type=int, default=100)
    plan_detective_parser.add_argument("--segment-seconds", type=float, default=8.0)
    plan_detective_parser.add_argument("--min-clip-seconds", type=float, default=3.0)
    plan_detective_parser.add_argument("--max-clip-seconds", type=float, default=15.0)

    select_single_source_parser = subparsers.add_parser("select-single-source-video")
    select_single_source_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    select_single_source_parser.add_argument("--source-clips-path", required=True)
    select_single_source_parser.add_argument("--output-path")
    select_single_source_parser.add_argument("--candidates-output-path")
    select_single_source_parser.add_argument("--selection-annotations-path")
    select_single_source_parser.add_argument("--dataset", default="daily_omni")
    select_single_source_parser.add_argument("--min-duration-seconds", type=float, default=28.0)
    select_single_source_parser.add_argument("--max-duration-seconds", type=float, default=32.0)
    select_single_source_parser.add_argument("--top-k", type=int, default=8)
    select_single_source_parser.add_argument("--max-source-videos-scan", type=int, default=2000)
    select_single_source_parser.add_argument("--max-eligible-candidates", type=int)
    select_single_source_parser.add_argument("--selection-mode", choices=("local_score", "random", "first"), default="local_score")
    select_single_source_parser.add_argument("--random-seed", type=int)
    select_single_source_parser.add_argument("--base-url")
    select_single_source_parser.add_argument("--api-key", default="EMPTY")
    select_single_source_parser.add_argument("--model")
    select_single_source_parser.add_argument("--timeout-seconds", type=float, default=180.0)

    plan_single_source_parser = subparsers.add_parser("plan-single-source-clips")
    plan_single_source_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_single_source_parser.add_argument("--selected-source-path", required=True)
    plan_single_source_parser.add_argument("--clip-plan-output-path")
    plan_single_source_parser.add_argument("--clip-groups-output-path")
    plan_single_source_parser.add_argument("--whole-manifest-output-path")
    plan_single_source_parser.add_argument("--segment-seconds", type=float, default=5.0)
    plan_single_source_parser.add_argument("--min-clip-seconds", type=float, default=3.0)

    stable_clips_parser = subparsers.add_parser("plan-stable-omni-clips")
    stable_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    stable_clips_parser.add_argument("--raw-index-path")
    stable_clips_parser.add_argument("--output-path")
    stable_clips_parser.add_argument("--cache-path")
    stable_clips_parser.add_argument("--max-source-videos", type=int, default=50)
    stable_clips_parser.add_argument("--min-clip-seconds", type=float, default=5.0)
    stable_clips_parser.add_argument("--max-clip-seconds", type=float, default=8.0)
    stable_clips_parser.add_argument("--base-url")
    stable_clips_parser.add_argument("--api-key", default="EMPTY")
    stable_clips_parser.add_argument("--model")
    stable_clips_parser.add_argument("--timeout-seconds", type=float, default=180.0)

    reference_understanding_parser = subparsers.add_parser("cache-reference-understandings")
    reference_understanding_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    reference_understanding_parser.add_argument("--clip-annotations-path", required=True)
    reference_understanding_parser.add_argument("--output-path")

    annotate_clips_parser = subparsers.add_parser("annotate-clips")
    annotate_clips_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    annotate_clips_parser.add_argument("--clips-manifest-path", required=True)
    annotate_clips_parser.add_argument("--output-path")
    annotate_clips_parser.add_argument("--base-url", required=True)
    annotate_clips_parser.add_argument("--api-key", required=True)
    annotate_clips_parser.add_argument("--model", required=True)
    annotate_clips_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    annotate_clips_parser.add_argument("--concurrency", type=int, default=1)
    annotate_clips_parser.add_argument("--overwrite", action="store_true")

    detective_annotate_parser = subparsers.add_parser("detective-annotate-clips")
    detective_annotate_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    detective_annotate_parser.add_argument("--clips-manifest-path", required=True)
    detective_annotate_parser.add_argument("--output-path")
    detective_annotate_parser.add_argument("--base-url", required=True)
    detective_annotate_parser.add_argument("--api-key", required=True)
    detective_annotate_parser.add_argument("--model", required=True)
    detective_annotate_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    detective_annotate_parser.add_argument("--concurrency", type=int, default=1)
    detective_annotate_parser.add_argument("--overwrite", action="store_true")
    detective_annotate_parser.add_argument("--audio-focused", action="store_true")

    mine_pair_candidates_parser = subparsers.add_parser("mine-pair-candidates")
    mine_pair_candidates_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    mine_pair_candidates_parser.add_argument("--clip-annotations-path", required=True)
    mine_pair_candidates_parser.add_argument("--clip-groups-path", required=True)
    mine_pair_candidates_parser.add_argument("--output-path")
    mine_pair_candidates_parser.add_argument("--report-path")
    mine_pair_candidates_parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_MINED_PAIR_CANDIDATES)
    mine_pair_candidates_parser.add_argument("--acceptance-profile", choices=sorted(ACCEPTANCE_PROFILE_NAMES), default=DEFAULT_ACCEPTANCE_PROFILE)

    mine_single_source_parser = subparsers.add_parser("mine-single-source-pairs")
    mine_single_source_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    mine_single_source_parser.add_argument("--clip-annotations-path", required=True)
    mine_single_source_parser.add_argument("--clip-groups-path", required=True)
    mine_single_source_parser.add_argument("--output-path")
    mine_single_source_parser.add_argument("--report-path")
    mine_single_source_parser.add_argument("--acceptance-profile", choices=sorted(ACCEPTANCE_PROFILE_NAMES), default=DEFAULT_ACCEPTANCE_PROFILE)

    propose_single_source_parser = subparsers.add_parser("propose-single-source-pairs")
    propose_single_source_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    propose_single_source_parser.add_argument("--clip-annotations-path", required=True)
    propose_single_source_parser.add_argument("--pair-candidates-path", required=True)
    propose_single_source_parser.add_argument("--whole-annotation-path")
    propose_single_source_parser.add_argument("--output-path")
    propose_single_source_parser.add_argument("--accepted-output-path")
    propose_single_source_parser.add_argument("--base-url", required=True)
    propose_single_source_parser.add_argument("--api-key", required=True)
    propose_single_source_parser.add_argument("--model", required=True)
    propose_single_source_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    propose_single_source_parser.add_argument("--max-accepted-pairs", type=int, default=5)
    propose_single_source_parser.add_argument("--max-proposals", type=int)
    propose_single_source_parser.add_argument("--zero-accepted-stop-after", type=int, default=DEFAULT_ZERO_ACCEPTED_STOP_AFTER)
    propose_single_source_parser.add_argument("--acceptance-profile", choices=sorted(ACCEPTANCE_PROFILE_NAMES), default=DEFAULT_ACCEPTANCE_PROFILE)
    propose_single_source_parser.add_argument("--audio-dataset-line", choices=sorted(AUDIO_DATASET_LINE_NAMES), default=STANDARD_AUDIO_DATASET_LINE)
    propose_single_source_parser.add_argument("--accepted-progress-path")
    propose_single_source_parser.add_argument("--rejected-progress-path")
    propose_single_source_parser.add_argument("--omni-retries", type=int, default=0)
    propose_single_source_parser.add_argument("--fail-on-transient-omni-errors", action="store_true")

    propose_pairs_parser = subparsers.add_parser("propose-pairs")
    propose_pairs_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    propose_pairs_parser.add_argument("--clip-annotations-path", required=True)
    propose_pairs_parser.add_argument("--output-path")
    propose_pairs_parser.add_argument("--raw-index-path")
    propose_pairs_parser.add_argument("--base-url", required=True)
    propose_pairs_parser.add_argument("--api-key", required=True)
    propose_pairs_parser.add_argument("--model", required=True)
    propose_pairs_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    propose_pairs_parser.add_argument("--overwrite", action="store_true")

    propose_group_pairs_parser = subparsers.add_parser("propose-group-pairs")
    propose_group_pairs_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    propose_group_pairs_parser.add_argument("--clip-annotations-path", required=True)
    propose_group_pairs_parser.add_argument("--clip-groups-path", required=True)
    propose_group_pairs_parser.add_argument("--mined-candidates-path")
    propose_group_pairs_parser.add_argument("--output-path")
    propose_group_pairs_parser.add_argument("--accepted-output-path")
    propose_group_pairs_parser.add_argument("--accepted-progress-path")
    propose_group_pairs_parser.add_argument("--rejected-progress-path")
    propose_group_pairs_parser.add_argument("--raw-index-path")
    propose_group_pairs_parser.add_argument("--base-url", required=True)
    propose_group_pairs_parser.add_argument("--api-key", required=True)
    propose_group_pairs_parser.add_argument("--model", required=True)
    propose_group_pairs_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    propose_group_pairs_parser.add_argument("--max-accepted-pairs", type=int, default=10)
    propose_group_pairs_parser.add_argument("--max-proposals", type=int)
    propose_group_pairs_parser.add_argument("--zero-accepted-stop-after", type=int, default=DEFAULT_ZERO_ACCEPTED_STOP_AFTER)
    propose_group_pairs_parser.add_argument("--acceptance-profile", choices=sorted(ACCEPTANCE_PROFILE_NAMES), default=DEFAULT_ACCEPTANCE_PROFILE)
    propose_group_pairs_parser.add_argument("--no-strict-audio-matters-visual-anchor", action="store_true")
    propose_group_pairs_parser.add_argument("--overwrite", action="store_true")

    plan_video_edits_parser = subparsers.add_parser("plan-video-edits")
    plan_video_edits_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_video_edits_parser.add_argument("--pair-candidates-path", required=True)
    plan_video_edits_parser.add_argument("--clip-annotations-path", required=True)
    plan_video_edits_parser.add_argument("--output-path")
    plan_video_edits_parser.add_argument("--max-plans", type=int, default=10)
    plan_video_edits_parser.add_argument("--base-url")
    plan_video_edits_parser.add_argument("--api-key", default="EMPTY")
    plan_video_edits_parser.add_argument("--model")
    plan_video_edits_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    plan_video_edits_parser.add_argument("--planning-mode", choices=("production", "exploration"), default="production")
    plan_video_edits_parser.add_argument("--planner-cache-path")

    plan_audio_edits_parser = subparsers.add_parser("plan-audio-edits")
    plan_audio_edits_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_audio_edits_parser.add_argument("--pair-candidates-path", required=True)
    plan_audio_edits_parser.add_argument("--clip-annotations-path", required=True)
    plan_audio_edits_parser.add_argument("--output-path")
    plan_audio_edits_parser.add_argument("--max-plans", type=int, default=10)

    plan_video_masks_parser = subparsers.add_parser("plan-video-masks")
    plan_video_masks_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    plan_video_masks_parser.add_argument("--video-edit-plan-path", required=True)
    plan_video_masks_parser.add_argument("--output-path")
    plan_video_masks_parser.add_argument("--mask-manifest-path")
    plan_video_masks_parser.add_argument("--max-masks", type=int)

    src_ref_plan_parser = subparsers.add_parser("plan-src-ref-images")
    src_ref_plan_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    src_ref_plan_parser.add_argument("--video-edit-plan-path", required=True)
    src_ref_plan_parser.add_argument("--output-path")
    src_ref_plan_parser.add_argument("--image-root")
    src_ref_plan_parser.add_argument("--num-candidates", type=int, default=4)

    src_ref_select_parser = subparsers.add_parser("select-src-ref-images")
    src_ref_select_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    src_ref_select_parser.add_argument("--src-ref-image-plan-path", required=True)
    src_ref_select_parser.add_argument("--output-path")
    src_ref_select_parser.add_argument("--max-selected", type=int, default=2)
    src_ref_select_parser.add_argument("--base-url")
    src_ref_select_parser.add_argument("--api-key", default="EMPTY")
    src_ref_select_parser.add_argument("--model")
    src_ref_select_parser.add_argument("--timeout-seconds", type=float, default=180.0)

    validate_known_pairs_parser = subparsers.add_parser("validate-known-pairs")
    validate_known_pairs_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    validate_known_pairs_parser.add_argument("--known-pairs-path", required=True)
    validate_known_pairs_parser.add_argument("--clip-annotations-path", required=True)
    validate_known_pairs_parser.add_argument("--output-path")
    validate_known_pairs_parser.add_argument("--accepted-output-path")
    validate_known_pairs_parser.add_argument("--raw-index-path")
    validate_known_pairs_parser.add_argument("--base-url", required=True)
    validate_known_pairs_parser.add_argument("--api-key", required=True)
    validate_known_pairs_parser.add_argument("--model", required=True)
    validate_known_pairs_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    validate_known_pairs_parser.add_argument("--max-accepted-pairs", type=int, default=10)
    validate_known_pairs_parser.add_argument("--overwrite", action="store_true")

    validate_pilot_parser = subparsers.add_parser("validate-pilot")
    validate_pilot_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    validate_pilot_parser.add_argument("--pilot-jsonl-path", required=True)
    validate_pilot_parser.add_argument("--gallery-output-path", required=True)
    validate_pilot_parser.add_argument("--report-output-path", required=True)

    review_bundle_parser = subparsers.add_parser("build-review-bundle")
    review_bundle_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    review_bundle_parser.add_argument("--pairs-path", required=True)
    review_bundle_parser.add_argument("--output-dir", required=True)
    review_bundle_parser.add_argument("--clip-annotations-path")
    review_bundle_parser.add_argument("--limit", type=int)
    review_bundle_parser.add_argument("--no-copy-videos", action="store_true")

    diagnostic_bundle_parser = subparsers.add_parser("build-diagnostic-bundle")
    diagnostic_bundle_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    diagnostic_bundle_parser.add_argument("--pairs-path", required=True)
    diagnostic_bundle_parser.add_argument("--output-dir", required=True)
    diagnostic_bundle_parser.add_argument("--clip-annotations-path")
    diagnostic_bundle_parser.add_argument("--limit-per-bucket", type=int, default=5)
    diagnostic_bundle_parser.add_argument("--no-copy-videos", action="store_true")

    single_source_bundle_parser = subparsers.add_parser("build-single-source-review-bundle")
    single_source_bundle_parser.add_argument("--root", default=DEFAULT_DATA_ROOT)
    single_source_bundle_parser.add_argument("--selected-source-path", required=True)
    single_source_bundle_parser.add_argument("--segments-manifest-path", required=True)
    single_source_bundle_parser.add_argument("--clip-annotations-path", required=True)
    single_source_bundle_parser.add_argument("--ranked-pairs-path", required=True)
    single_source_bundle_parser.add_argument("--accepted-pairs-path", required=True)
    single_source_bundle_parser.add_argument("--output-dir", required=True)
    single_source_bundle_parser.add_argument("--no-copy-videos", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "init-layout":
        result = {name: str(path) for name, path in ensure_layout(args.root).items()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "index-raw":
        sources = _parse_sources(args.source) if args.source else discover_raw_sources(args.root)
        if not sources:
            raise ValueError("no raw sources found; pass --source or create <root>/raw_datasets/<dataset>")
        result = index_raw_sources(root=args.root, sources=sources, output_path=args.output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "extract-clips":
        result = extract_clips(
            root=args.root,
            plan_path=args.plan_path,
            raw_index_path=args.raw_index_path,
            output_manifest_path=args.output_manifest_path,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "annotate-clips":
        result = annotate_clips(
            root=args.root,
            clips_manifest_path=args.clips_manifest_path,
            output_path=args.output_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-detective-clips":
        result = plan_detective_event_clips(
            root=args.root,
            source_clips_path=args.source_clips_path,
            clip_plan_output_path=args.clip_plan_output_path,
            clip_groups_output_path=args.clip_groups_output_path,
            max_source_videos=args.max_source_videos,
            segment_seconds=args.segment_seconds,
            min_clip_seconds=args.min_clip_seconds,
            max_clip_seconds=args.max_clip_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "select-single-source-video":
        result = select_single_source_video(
            root=args.root,
            source_clips_path=args.source_clips_path,
            output_path=args.output_path,
            candidates_output_path=args.candidates_output_path,
            selection_annotations_path=args.selection_annotations_path,
            dataset=args.dataset,
            min_duration_seconds=args.min_duration_seconds,
            max_duration_seconds=args.max_duration_seconds,
            top_k=args.top_k,
            max_source_videos_scan=args.max_source_videos_scan,
            max_eligible_candidates=args.max_eligible_candidates,
            selection_mode=args.selection_mode,
            random_seed=args.random_seed,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-single-source-clips":
        result = plan_single_source_clips(
            root=args.root,
            selected_source_path=args.selected_source_path,
            clip_plan_output_path=args.clip_plan_output_path,
            clip_groups_output_path=args.clip_groups_output_path,
            whole_manifest_output_path=args.whole_manifest_output_path,
            segment_seconds=args.segment_seconds,
            min_clip_seconds=args.min_clip_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-stable-omni-clips":
        result = plan_stable_omni_clips(
            root=args.root,
            raw_index_path=args.raw_index_path,
            output_path=args.output_path,
            cache_path=args.cache_path,
            max_source_videos=args.max_source_videos,
            min_clip_seconds=args.min_clip_seconds,
            max_clip_seconds=args.max_clip_seconds,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "cache-reference-understandings":
        result = cache_reference_understandings(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "detective-annotate-clips":
        result = detective_annotate_clips(
            root=args.root,
            clips_manifest_path=args.clips_manifest_path,
            output_path=args.output_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
            audio_focused=args.audio_focused,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "mine-pair-candidates":
        result = mine_pair_candidates(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            clip_groups_path=args.clip_groups_path,
            output_path=args.output_path,
            report_path=args.report_path,
            max_candidates=args.max_candidates,
            acceptance_profile=args.acceptance_profile,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "mine-single-source-pairs":
        result = mine_single_source_pairs(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            clip_groups_path=args.clip_groups_path,
            output_path=args.output_path,
            report_path=args.report_path,
            acceptance_profile=args.acceptance_profile,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "propose-single-source-pairs":
        result = propose_single_source_pairs(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            pair_candidates_path=args.pair_candidates_path,
            whole_annotation_path=args.whole_annotation_path,
            output_path=args.output_path,
            accepted_output_path=args.accepted_output_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_accepted_pairs=args.max_accepted_pairs,
            max_proposals=args.max_proposals,
            zero_accepted_stop_after=args.zero_accepted_stop_after,
            acceptance_profile=args.acceptance_profile,
            audio_dataset_line=args.audio_dataset_line,
            accepted_progress_path=args.accepted_progress_path,
            rejected_progress_path=args.rejected_progress_path,
            omni_retries=args.omni_retries,
            fail_on_transient_omni_errors=args.fail_on_transient_omni_errors,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "propose-pairs":
        result = propose_pairs(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            raw_index_path=args.raw_index_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "propose-group-pairs":
        print("[propose-group-pairs] cli enter", file=sys.stderr, flush=True)
        result = propose_group_pairs(
            root=args.root,
            clip_annotations_path=args.clip_annotations_path,
            clip_groups_path=args.clip_groups_path,
            mined_candidates_path=args.mined_candidates_path,
            output_path=args.output_path,
            accepted_output_path=args.accepted_output_path,
            accepted_progress_path=args.accepted_progress_path,
            rejected_progress_path=args.rejected_progress_path,
            raw_index_path=args.raw_index_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_accepted_pairs=args.max_accepted_pairs,
            max_proposals=args.max_proposals,
            zero_accepted_stop_after=args.zero_accepted_stop_after,
            acceptance_profile=args.acceptance_profile,
            strict_audio_matters_visual_anchor=not args.no_strict_audio_matters_visual_anchor,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-video-edits":
        result = plan_video_edits(
            root=args.root,
            pair_candidates_path=args.pair_candidates_path,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            max_plans=args.max_plans,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            planning_mode=args.planning_mode,
            planner_cache_path=args.planner_cache_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-audio-edits":
        result = plan_audio_edits(
            root=args.root,
            pair_candidates_path=args.pair_candidates_path,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            max_plans=args.max_plans,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-video-masks":
        result = plan_video_masks(
            root=args.root,
            video_edit_plan_path=args.video_edit_plan_path,
            output_path=args.output_path,
            mask_manifest_path=args.mask_manifest_path,
            max_masks=args.max_masks,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "plan-src-ref-images":
        result = plan_src_ref_images(
            root=args.root,
            video_edit_plan_path=args.video_edit_plan_path,
            output_path=args.output_path,
            image_root=args.image_root,
            num_candidates=args.num_candidates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "select-src-ref-images":
        result = select_src_ref_images(
            root=args.root,
            src_ref_image_plan_path=args.src_ref_image_plan_path,
            output_path=args.output_path,
            max_selected=args.max_selected,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "validate-known-pairs":
        result = validate_known_pairs(
            root=args.root,
            known_pairs_path=args.known_pairs_path,
            clip_annotations_path=args.clip_annotations_path,
            output_path=args.output_path,
            accepted_output_path=args.accepted_output_path,
            raw_index_path=args.raw_index_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_accepted_pairs=args.max_accepted_pairs,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "build-review-bundle":
        result = build_manual_review_bundle(
            root=args.root,
            pairs_path=args.pairs_path,
            output_dir=args.output_dir,
            clip_annotations_path=args.clip_annotations_path,
            limit=args.limit,
            copy_videos=not args.no_copy_videos,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "build-diagnostic-bundle":
        result = build_diagnostic_review_bundle(
            root=args.root,
            pairs_path=args.pairs_path,
            output_dir=args.output_dir,
            clip_annotations_path=args.clip_annotations_path,
            limit_per_bucket=args.limit_per_bucket,
            copy_videos=not args.no_copy_videos,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "build-single-source-review-bundle":
        result = build_single_source_review_bundle(
            root=args.root,
            selected_source_path=args.selected_source_path,
            segments_manifest_path=args.segments_manifest_path,
            clip_annotations_path=args.clip_annotations_path,
            ranked_pairs_path=args.ranked_pairs_path,
            accepted_pairs_path=args.accepted_pairs_path,
            output_dir=args.output_dir,
            copy_videos=not args.no_copy_videos,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = validate_pilot_dataset(
        root=args.root,
        pilot_jsonl_path=args.pilot_jsonl_path,
        gallery_output_path=args.gallery_output_path,
        report_output_path=args.report_output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
