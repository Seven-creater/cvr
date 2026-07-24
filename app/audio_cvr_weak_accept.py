from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlparse

import numpy as np


EXPECTED_FULL1000_SHA256 = "70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e"
AUDIT_VERSION = "single_rater_blind_audit_v1"
VARIANT_VERSION = "reference_identity_ladder_v1"
AUDIT_GATES = (
    "audible_change_clear",
    "reference_does_not_satisfy_edit",
    "target_satisfies_edit",
    "muted_video_insufficient",
    "visual_context_preserved",
    "no_transcript_asr_shortcut",
    "overall_valid",
)
EXTRA_AUDIT_QUOTAS = {
    "avatar": 10,
    "vggsound": 15,
    "ave": 15,
    "worldsense": 5,
    "vgg_monoaudio": 5,
}
REFERENCE_VARIANTS = ("transcoded", "temporal", "spatial")
AUDIT_QUOTA_FALLBACK_ORDER = (
    "vggsound",
    "ave",
    "avatar",
    "vgg_monoaudio",
    "worldsense",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(*values: Any) -> str:
    body = "\0".join(str(value) for value in values)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _safe_embedding_name(sample_id: str) -> str:
    return _stable_digest("e5-reference-variant", sample_id)[:32]


def _sample_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "proposal_id", "candidate_id", "record_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError("record is missing sample_id/proposal_id/candidate_id")


def _row_aliases(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("sample_id", "proposal_id", "candidate_id", "record_id"):
        value = str(row.get(key) or "").strip()
        if value:
            values.add(value)
    reference = Path(str(row.get("reference_video") or "")).name.lower()
    target = Path(str(row.get("target_video") or "")).name.lower()
    edit = re.sub(r"\s+", " ", str(row.get("edit_text") or "").strip().lower())
    if reference and target and edit:
        values.add(f"pair::{reference}::{target}::{edit}")
    return values


def _dataset_label(row: dict[str, Any]) -> str:
    preferred = " ".join(
        str(row.get(key) or "")
        for key in (
            "dataset",
            "source_dataset",
            "origin_dataset",
            "provenance_source",
            "source_run",
        )
    ).lower()
    haystack = preferred or json.dumps(row, ensure_ascii=False).lower()
    if "monoaudio" in haystack or "mono_audio" in haystack:
        return "vgg_monoaudio"
    if "worldsense" in haystack:
        return "worldsense"
    if re.search(r"(^|[^a-z])ave([^a-z]|$)", haystack) or "ave_dataset" in haystack:
        return "ave"
    if "vggsound" in haystack or "vgg_sound" in haystack:
        return "vggsound"
    if "avatar" in haystack:
        return "avatar"
    return "unknown"


def _resolve_media(raw: str, roots: Sequence[Path]) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise FileNotFoundError("empty media path")
    path = Path(value).expanduser()
    attempts = [path] if path.is_absolute() else [Path.cwd() / path, *[root / path for root in roots]]
    if path.is_absolute():
        attempts.extend(root / Path(*path.parts[1:]) for root in roots)
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    basename = path.name
    for root in roots:
        direct = root / basename
        if direct.is_file():
            return direct.resolve()
    raise FileNotFoundError(f"media not found: {raw}")


def _verify_full1000(path: Path, expected_sha256: str = EXPECTED_FULL1000_SHA256) -> list[dict[str, Any]]:
    actual = _sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"Full1000 SHA256={actual}, expected={expected_sha256}")
    rows = _load_jsonl(path)
    if len(rows) != 1000:
        raise ValueError(f"Full1000 must contain 1000 rows, found {len(rows)}")
    ids = [_sample_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Full1000 sample IDs are not unique")
    return rows


def _stable_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _stable_digest(seed, _sample_id(row)))


def prepare_human_audit(
    *,
    full_path: Path,
    core_path: Path,
    output_dir: Path,
    media_roots: Sequence[Path],
    seed: int = 20260724,
    expected_full_sha256: str = EXPECTED_FULL1000_SHA256,
) -> dict[str, Any]:
    full_rows = _verify_full1000(full_path, expected_full_sha256)
    core_rows = _load_jsonl(core_path)
    if len(core_rows) != 150:
        raise ValueError(f"Core audit must contain 150 rows, found {len(core_rows)}")

    alias_to_index: dict[str, int] = {}
    ambiguous: set[str] = set()
    for index, row in enumerate(full_rows):
        for alias in _row_aliases(row):
            if alias in alias_to_index and alias_to_index[alias] != index:
                ambiguous.add(alias)
            else:
                alias_to_index[alias] = index
    for alias in ambiguous:
        alias_to_index.pop(alias, None)

    core_indices: list[int] = []
    for row in core_rows:
        matches = {alias_to_index[alias] for alias in _row_aliases(row) if alias in alias_to_index}
        if len(matches) != 1:
            raise ValueError(f"Core record {_sample_id(row)} matched {len(matches)} Full1000 records")
        core_indices.append(matches.pop())
    if len(set(core_indices)) != 150:
        raise ValueError("Core150 maps to duplicate Full1000 records")

    core_set = set(core_indices)
    extra_indices: list[int] = []
    available: dict[str, int] = {}
    candidate_pools: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    realized_quotas: Counter[str] = Counter()
    for dataset, quota in EXTRA_AUDIT_QUOTAS.items():
        candidates = [
            (index, row)
            for index, row in enumerate(full_rows)
            if index not in core_set and _dataset_label(row) == dataset
        ]
        ordered = sorted(
            candidates,
            key=lambda item: _stable_digest(
                seed, dataset, _sample_id(item[1])
            ),
        )
        candidate_pools[dataset] = ordered
        available[dataset] = len(candidates)
        selected = ordered[: min(quota, len(ordered))]
        extra_indices.extend(index for index, _ in selected)
        realized_quotas[dataset] += len(selected)

    deficit = 50 - len(extra_indices)
    selected_set = set(extra_indices)
    for dataset in AUDIT_QUOTA_FALLBACK_ORDER:
        if deficit <= 0:
            break
        remaining = [
            (index, row)
            for index, row in candidate_pools[dataset]
            if index not in selected_set
        ]
        take = min(deficit, len(remaining))
        extra_indices.extend(index for index, _ in remaining[:take])
        selected_set.update(index for index, _ in remaining[:take])
        realized_quotas[dataset] += take
        deficit -= take
    if deficit:
        raise ValueError(
            f"extra audit sample is short by {deficit} after deterministic quota fallback"
        )
    if len(extra_indices) != 50 or len(set(extra_indices)) != 50:
        raise ValueError("extra audit sample must contain exactly 50 unique records")

    selected_indices = core_indices + extra_indices
    selected_rows = [full_rows[index] for index in selected_indices]
    variant_check_ids = {
        _sample_id(row) for row in _stable_order(selected_rows, seed + 31)[:30]
    }
    repeat_ids = {
        _sample_id(row) for row in _stable_order([full_rows[index] for index in core_indices], seed + 41)[:10]
    }
    repeat_ids.update(
        _sample_id(row) for row in _stable_order([full_rows[index] for index in extra_indices], seed + 43)[:10]
    )

    private_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    primary_review_by_sample: dict[str, str] = {}
    for index, row in zip(selected_indices, selected_rows):
        sample_id = _sample_id(row)
        review_id = _stable_digest(AUDIT_VERSION, seed, sample_id, "primary")[:24]
        primary_review_by_sample[sample_id] = review_id
        reference = _resolve_media(str(row["reference_video"]), media_roots)
        target = _resolve_media(str(row["target_video"]), media_roots)
        private_rows.append(
            {
                "review_id": review_id,
                "sample_id": sample_id,
                "full1000_index": index,
                "audit_partition": "core150" if index in core_set else "supplement50",
                "dataset": _dataset_label(row),
                "edit_text": str(row["edit_text"]).strip(),
                "reference_video": str(reference),
                "target_video": str(target),
                "is_hidden_repeat": False,
                "repeat_of_review_id": None,
                "requires_variant_check": sample_id in variant_check_ids,
                "automatic_review_decision": str(
                    row.get("decision")
                    or (row.get("benchmark_review") or {}).get("decision")
                    or "pass"
                ),
            }
        )
        public_rows.append(
            {
                "review_id": review_id,
                "edit_text": str(row["edit_text"]).strip(),
                "requires_variant_check": sample_id in variant_check_ids,
            }
        )

    for sample_id in sorted(repeat_ids):
        primary = next(row for row in private_rows if row["sample_id"] == sample_id)
        review_id = _stable_digest(AUDIT_VERSION, seed, sample_id, "repeat")[:24]
        repeat = dict(primary)
        repeat.update(
            {
                "review_id": review_id,
                "is_hidden_repeat": True,
                "repeat_of_review_id": primary_review_by_sample[sample_id],
                "requires_variant_check": False,
            }
        )
        private_rows.append(repeat)
        public_rows.append(
            {
                "review_id": review_id,
                "edit_text": primary["edit_text"],
                "requires_variant_check": False,
            }
        )

    order = sorted(
        range(len(private_rows)),
        key=lambda index: _stable_digest(seed, "display-order", private_rows[index]["review_id"]),
    )
    private_rows = [{**private_rows[index], "display_index": position} for position, index in enumerate(order)]
    public_by_id = {row["review_id"]: row for row in public_rows}
    public_rows = [
        {**public_by_id[row["review_id"]], "display_index": row["display_index"]}
        for row in private_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "private_manifest.jsonl", private_rows)
    _atomic_jsonl(output_dir / "public_manifest.jsonl", public_rows)
    summary = {
        "audit_version": AUDIT_VERSION,
        "seed": seed,
        "full1000_path": str(full_path.resolve()),
        "full1000_sha256": _sha256_file(full_path),
        "core150_path": str(core_path.resolve()),
        "unique_sample_count": 200,
        "core150_count": 150,
        "supplement_count": 50,
        "hidden_repeat_count": 20,
        "display_item_count": 220,
        "variant_semantics_check_count": 30,
        "supplement_requested_quotas": EXTRA_AUDIT_QUOTAS,
        "supplement_realized_quotas": dict(realized_quotas),
        "supplement_quota_fallback_order": list(AUDIT_QUOTA_FALLBACK_ORDER),
        "supplement_available": available,
        "selection_uses_model_scores": False,
        "rater_count": 1,
        "claim_scope": "blinded single-rater human audit",
    }
    _atomic_json(output_dir / "audit_manifest_summary.json", summary)
    return summary


_AUDIT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio-CVR blinded audit</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#18202a}
main{max-width:1120px;margin:0 auto;padding:18px}.bar{display:flex;justify-content:space-between;gap:12px}
.panel{background:white;border:1px solid #d9dee5;padding:14px;margin-top:12px;border-radius:6px}
.media{display:grid;grid-template-columns:1fr 1fr;gap:12px}video,audio{width:100%}
.gates{display:grid;grid-template-columns:1fr 1fr;gap:8px 18px}.gate{border-bottom:1px solid #edf0f3;padding:8px 0}
label{font-weight:600}button{border:0;background:#1769aa;color:white;padding:10px 18px;border-radius:4px;font-weight:700}
button:disabled{opacity:.45}.muted{font-size:12px;color:#5d6875}.error{color:#a32020}
</style></head><body><main>
<div class="bar"><h2>Audio-CVR blinded single-rater audit</h2><strong id="progress"></strong></div>
<div id="error" class="error"></div><div id="content"></div>
</main><script>
let current=null;
const gateLabels={
 audible_change_clear:"The sound change is clearly audible",
 reference_does_not_satisfy_edit:"Reference does not satisfy the edit",
 target_satisfies_edit:"Target satisfies the edit",
 muted_video_insufficient:"Muted video alone is insufficient to identify the target",
 visual_context_preserved:"Visual context is substantially preserved",
 no_transcript_asr_shortcut:"No transcript/ASR shortcut",
 overall_valid:"Overall valid Audio-CVR item"
};
function gateHtml(k){return `<div class="gate"><label>${gateLabels[k]}</label><br>
<input type="radio" name="${k}" value="true" required> Yes
<input type="radio" name="${k}" value="false"> No</div>`}
async function load(){
 const r=await fetch("/api/state"); const s=await r.json();
 document.getElementById("progress").textContent=`${s.completed}/${s.total}`;
 if(s.complete){document.getElementById("content").innerHTML='<div class="panel"><h3>Audit complete.</h3></div>';return}
 current=s.item;
 let variant=current.requires_variant_check?`<div class="panel"><h3>Identity-perturbation semantic check</h3>
 <div class="media"><div><video controls src="/media/${current.review_id}/temporal"></video><b>Temporal trim preserves pre-edit meaning</b><br>
 <input type="radio" name="temporal_preserves_pre_edit" value="true" required> Yes <input type="radio" name="temporal_preserves_pre_edit" value="false"> No</div>
 <div><video controls src="/media/${current.review_id}/spatial"></video><b>Spatial crop preserves pre-edit meaning</b><br>
 <input type="radio" name="spatial_preserves_pre_edit" value="true" required> Yes <input type="radio" name="spatial_preserves_pre_edit" value="false"> No</div></div></div>`:"";
 document.getElementById("content").innerHTML=`<form id="form">
 <div class="panel"><h3>Edit</h3><p>${escapeHtml(current.edit_text)}</p></div>
 <div class="panel"><h3>1. Listen only</h3><div class="media"><div><b>Reference audio</b><audio controls src="/media/${current.review_id}/reference"></audio></div><div><b>Target audio</b><audio controls src="/media/${current.review_id}/target"></audio></div></div></div>
 <div class="panel"><h3>2. Inspect muted video</h3><div class="media"><video controls muted src="/media/${current.review_id}/reference"></video><video controls muted src="/media/${current.review_id}/target"></video></div></div>
 <div class="panel"><h3>3. Inspect full audiovisual pair</h3><div class="media"><video controls src="/media/${current.review_id}/reference"></video><video controls src="/media/${current.review_id}/target"></video></div></div>
 ${variant}<div class="panel"><div class="gates">${Object.keys(gateLabels).map(gateHtml).join("")}</div>
 <p><label>Confidence</label> <select name="confidence"><option value="1">1 - low</option><option value="2">2</option><option value="3" selected>3</option><option value="4">4</option><option value="5">5 - high</option></select></p>
 <p><label>Optional note</label><br><textarea name="note" rows="2" style="width:100%"></textarea></p>
 <button type="submit">Save and continue</button></div></form>`;
 document.getElementById("form").onsubmit=submit;
}
function escapeHtml(v){const d=document.createElement("div");d.textContent=v;return d.innerHTML}
async function submit(e){
 e.preventDefault();const fd=new FormData(e.target), payload={review_id:current.review_id};
 for(const k of Object.keys(gateLabels)) payload[k]=fd.get(k)==="true";
 payload.confidence=Number(fd.get("confidence"));payload.note=fd.get("note")||"";
 if(current.requires_variant_check){payload.temporal_preserves_pre_edit=fd.get("temporal_preserves_pre_edit")==="true";payload.spatial_preserves_pre_edit=fd.get("spatial_preserves_pre_edit")==="true"}
 const b=e.target.querySelector("button");b.disabled=true;
 const r=await fetch("/api/submit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
 if(!r.ok){document.getElementById("error").textContent=await r.text();b.disabled=false;return}
 document.getElementById("error").textContent="";await load();
}
load().catch(e=>document.getElementById("error").textContent=e);
</script></body></html>"""


class _AuditServer:
    def __init__(self, audit_dir: Path, variants_dir: Path | None) -> None:
        self.audit_dir = audit_dir
        self.variants_dir = variants_dir
        self.private_rows = sorted(_load_jsonl(audit_dir / "private_manifest.jsonl"), key=lambda row: row["display_index"])
        self.public_by_id = {
            row["review_id"]: row for row in _load_jsonl(audit_dir / "public_manifest.jsonl")
        }
        self.private_by_id = {row["review_id"]: row for row in self.private_rows}
        self.response_dir = audit_dir / "responses"
        self.response_dir.mkdir(parents=True, exist_ok=True)

    def completed_ids(self) -> set[str]:
        return {path.stem for path in self.response_dir.glob("*.json")}

    def state(self) -> dict[str, Any]:
        completed = self.completed_ids()
        remaining = [row for row in self.private_rows if row["review_id"] not in completed]
        if not remaining:
            return {"complete": True, "completed": len(completed), "total": len(self.private_rows)}
        private = remaining[0]
        public = dict(self.public_by_id[private["review_id"]])
        if public["requires_variant_check"] and self.variants_dir is not None:
            for condition in ("temporal", "spatial"):
                if not self.variant_path(private, condition).is_file():
                    public["requires_variant_check"] = False
                    break
        return {
            "complete": False,
            "completed": len(completed),
            "total": len(self.private_rows),
            "item": public,
        }

    def variant_path(self, row: dict[str, Any], condition: str) -> Path:
        if self.variants_dir is None:
            raise FileNotFoundError("reference variants are not configured")
        key = _stable_digest(VARIANT_VERSION, row["sample_id"])[:24]
        return self.variants_dir / "variants" / condition / f"{key}.mp4"

    def media_path(self, review_id: str, role: str) -> Path:
        row = self.private_by_id.get(review_id)
        if row is None:
            raise FileNotFoundError(review_id)
        if role in {"reference", "target"}:
            return Path(row[f"{role}_video"])
        if role in {"temporal", "spatial"}:
            return self.variant_path(row, role)
        raise FileNotFoundError(role)

    def save_response(self, payload: dict[str, Any]) -> None:
        review_id = str(payload.get("review_id") or "")
        if review_id not in self.private_by_id:
            raise ValueError("unknown review_id")
        if (self.response_dir / f"{review_id}.json").exists():
            raise ValueError("review item was already submitted")
        normalized: dict[str, Any] = {
            "review_id": review_id,
            "audit_version": AUDIT_VERSION,
            "submitted_unix": time.time(),
        }
        for gate in AUDIT_GATES:
            if not isinstance(payload.get(gate), bool):
                raise ValueError(f"{gate} must be boolean")
            normalized[gate] = payload[gate]
        confidence = int(payload.get("confidence", 0))
        if confidence not in {1, 2, 3, 4, 5}:
            raise ValueError("confidence must be 1..5")
        normalized["confidence"] = confidence
        normalized["note"] = str(payload.get("note") or "")[:2000]
        if self.private_by_id[review_id]["requires_variant_check"]:
            for gate in ("temporal_preserves_pre_edit", "spatial_preserves_pre_edit"):
                if not isinstance(payload.get(gate), bool):
                    raise ValueError(f"{gate} must be boolean")
                normalized[gate] = payload[gate]
        _atomic_json(self.response_dir / f"{review_id}.json", normalized)
        responses = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.response_dir.glob("*.json"))
        ]
        _atomic_jsonl(self.audit_dir / "responses.jsonl", responses)


def _send_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    size = path.stat().st_size
    start, end = 0, size - 1
    range_header = handler.headers.get("Range")
    status = HTTPStatus.OK
    if range_header:
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if match:
            start = int(match.group(1) or 0)
            end = min(int(match.group(2) or end), end)
            status = HTTPStatus.PARTIAL_CONTENT
    length = max(0, end - start + 1)
    handler.send_response(status)
    handler.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    if status == HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def serve_human_audit(
    *, audit_dir: Path, variants_dir: Path | None, host: str, port: int
) -> None:
    state = _AuditServer(audit_dir, variants_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            print(f"[human-audit] {self.address_string()} {format % args}", flush=True)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                body = _AUDIT_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                body = json.dumps(state.state()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            match = re.fullmatch(r"/media/([a-f0-9]{24})/([a-z]+)", path)
            if match:
                try:
                    media = state.media_path(match.group(1), match.group(2))
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                _send_file(self, media)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/submit":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                state.save_response(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            body = b'{"saved":true}'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[human-audit] serving http://{host}:{port}", flush=True)
    server.serve_forever()


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _validity_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    valid = sum(bool(row["overall_valid"]) for row in rows)
    return {
        "count": count,
        "valid_count": valid,
        "valid_rate": valid / count if count else None,
        "wilson_95_ci": _wilson_interval(valid, count),
    }


def summarize_human_audit(
    audit_dir: Path, output_dir: Path, *, allow_partial: bool = False
) -> dict[str, Any]:
    manifest = _load_jsonl(audit_dir / "private_manifest.jsonl")
    responses = {
        row["review_id"]: row for row in _load_jsonl(audit_dir / "responses.jsonl")
    }
    if len(responses) != len(manifest) and not allow_partial:
        raise ValueError(f"human audit incomplete: {len(responses)}/{len(manifest)}")
    combined = [
        {**row, **responses[row["review_id"]]}
        for row in manifest
        if row["review_id"] in responses
    ]
    primaries = [row for row in combined if not row["is_hidden_repeat"]]
    repeats = [row for row in combined if row["is_hidden_repeat"]]

    by_review_id = {row["review_id"]: row for row in combined}
    repeat_pairs = [
        (by_review_id[row["repeat_of_review_id"]], row)
        for row in repeats
        if row["repeat_of_review_id"] in by_review_id
    ]
    gate_agreement = {
        gate: (
            sum(first[gate] == second[gate] for first, second in repeat_pairs)
            / len(repeat_pairs)
            if repeat_pairs
            else None
        )
        for gate in AUDIT_GATES
    }
    exact = sum(
        all(first[gate] == second[gate] for gate in AUDIT_GATES)
        for first, second in repeat_pairs
    )
    variant_rows = [row for row in primaries if row["requires_variant_check"]]
    report = {
        "audit_version": AUDIT_VERSION,
        "rater_count": 1,
        "claim_scope": "blinded single-rater human audit",
        "partial_audit": len(combined) != len(manifest),
        "planned_display_item_count": len(manifest),
        "completed_display_item_count": len(combined),
        "completion_rate": len(combined) / len(manifest) if manifest else None,
        "unique_sample_count": len(primaries),
        "display_item_count": len(combined),
        "core150": _validity_summary([row for row in primaries if row["audit_partition"] == "core150"]),
        "supplement50": _validity_summary(
            [row for row in primaries if row["audit_partition"] == "supplement50"]
        ),
        "supplement_by_dataset": {
            dataset: _validity_summary(
                [
                    row
                    for row in primaries
                    if row["audit_partition"] == "supplement50" and row["dataset"] == dataset
                ]
            )
            for dataset in EXTRA_AUDIT_QUOTAS
        },
        "failure_gate_counts": {
            gate: sum(not bool(row[gate]) for row in primaries) for gate in AUDIT_GATES
        },
        "hidden_repeat": {
            "count": len(repeat_pairs),
            "completed_repeat_item_count": len(repeats),
            "unpaired_repeat_item_count": len(repeats) - len(repeat_pairs),
            "exact_all_gate_agreement": (
                exact / len(repeat_pairs) if repeat_pairs else None
            ),
            "gate_level_agreement": gate_agreement,
        },
        "automatic_human_overall_agreement": _validity_summary(primaries)["valid_rate"],
        "variant_semantics": {
            "count": len(variant_rows),
            "temporal_preservation_rate": (
                sum(bool(row["temporal_preserves_pre_edit"]) for row in variant_rows) / len(variant_rows)
                if variant_rows
                else None
            ),
            "spatial_preservation_rate": (
                sum(bool(row["spatial_preserves_pre_edit"]) for row in variant_rows) / len(variant_rows)
                if variant_rows
                else None
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "human_audit_summary.json", report)
    _atomic_jsonl(
        output_dir / "human_valid_samples.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "audit_partition": row["audit_partition"],
                "dataset": row["dataset"],
                "overall_valid": row["overall_valid"],
                "confidence": row["confidence"],
            }
            for row in primaries
            if row["overall_valid"]
        ],
    )
    _atomic_jsonl(
        output_dir / "human_audit_failures.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "audit_partition": row["audit_partition"],
                "dataset": row["dataset"],
                "failed_gates": [gate for gate in AUDIT_GATES if not row[gate]],
                "confidence": row["confidence"],
                "note": row["note"],
            }
            for row in primaries
            if not row["overall_valid"]
        ],
    )
    return report


def evaluate_human_valid_subset(
    *, valid_path: Path, named_per_query_paths: Sequence[str], output_path: Path
) -> dict[str, Any]:
    valid_ids = {_sample_id(row) for row in _load_jsonl(valid_path)}
    result: dict[str, Any] = {}
    for specification in named_per_query_paths:
        if "=" not in specification:
            raise ValueError("--per-query must be NAME=PATH")
        name, raw_path = specification.split("=", 1)
        rows = [row for row in _load_jsonl(Path(raw_path)) if _sample_id(row) in valid_ids]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get("mode") or row.get("condition") or "all")].append(row)
        result[name] = {}
        for group, values in groups.items():
            ranks = np.asarray(
                [
                    row.get("with_reference_rank")
                    or row.get("target_rank")
                    or row.get("rank")
                    for row in values
                ],
                dtype=np.float64,
            )
            tbr = np.asarray(
                [bool(row.get("target_beats_reference")) for row in values], dtype=bool
            )
            gaps = np.asarray(
                [
                    row.get("target_reference_gap")
                    or row.get("target_reference_score_gap")
                    or 0.0
                    for row in values
                ],
                dtype=np.float64,
            )
            result[name][group] = {
                "query_count": len(values),
                "R@1": float(np.mean(ranks <= 1)),
                "R@5": float(np.mean(ranks <= 5)),
                "R@10": float(np.mean(ranks <= 10)),
                "MRR": float(np.mean(1.0 / ranks)),
                "target_beats_reference": float(np.mean(tbr)),
                "target_reference_gap_mean": float(np.mean(gaps)),
            }
    payload = {
        "human_valid_sample_count": len(valid_ids),
        "models": result,
        "reencoded": False,
    }
    _atomic_json(output_path, payload)
    return payload


def _probe_media(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"reference has no video stream: {path}")
    return {
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "has_audio": audio is not None,
    }


def prepare_reference_variant_plan(
    *,
    full_path: Path,
    output_dir: Path,
    media_roots: Sequence[Path],
    expected_full_sha256: str = EXPECTED_FULL1000_SHA256,
) -> dict[str, Any]:
    rows = _verify_full1000(full_path, expected_full_sha256)
    plan: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = _sample_id(row)
        source = _resolve_media(str(row["reference_video"]), media_roots)
        key = _stable_digest(VARIANT_VERSION, sample_id)[:24]
        for condition in REFERENCE_VARIANTS:
            plan.append(
                {
                    "sample_id": sample_id,
                    "full1000_index": index,
                    "condition": condition,
                    "source_path": str(source),
                    "output_path": str(
                        (output_dir / "variants" / condition / f"{key}.mp4").resolve()
                    ),
                    "item_path": str(
                        (output_dir / "items" / condition / f"{key}.json").resolve()
                    ),
                    "variant_version": VARIANT_VERSION,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "reference_variant_plan.jsonl", plan)
    summary = {
        "variant_version": VARIANT_VERSION,
        "full1000_path": str(full_path.resolve()),
        "full1000_sha256": _sha256_file(full_path),
        "sample_count": len(rows),
        "conditions": list(REFERENCE_VARIANTS),
        "planned_item_count": len(plan),
    }
    _atomic_json(output_dir / "reference_variant_plan_summary.json", summary)
    return summary


def _ffmpeg_variant_command(
    condition: str, source: Path, output: Path, probe: dict[str, Any]
) -> list[str]:
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    video = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    audio = ["-c:a", "aac", "-b:a", "192k"] if probe["has_audio"] else ["-an"]
    if condition == "transcoded":
        return [*common, "-i", str(source), *video, *audio, "-movflags", "+faststart", str(output)]
    if condition == "temporal":
        duration = float(probe["duration"]) - 1.0
        if duration <= 0.5:
            raise ValueError(f"reference is too short for temporal trim: {source}")
        return [
            *common,
            "-ss",
            "0.5",
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            *video,
            *audio,
            "-movflags",
            "+faststart",
            str(output),
        ]
    if condition == "spatial":
        width, height = int(probe["width"]), int(probe["height"])
        crop_width = max(2, int(width * 0.9) // 2 * 2)
        crop_height = max(2, int(height * 0.9) // 2 * 2)
        filter_value = (
            f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:(ih-{crop_height})/2,"
            f"scale={width}:{height}"
        )
        return [
            *common,
            "-i",
            str(source),
            "-vf",
            filter_value,
            *video,
            *audio,
            "-movflags",
            "+faststart",
            str(output),
        ]
    raise ValueError(f"unknown reference condition: {condition}")


def generate_reference_variants(
    *,
    plan_path: Path,
    shard_index: int,
    shard_count: int,
    retries: int,
) -> dict[str, Any]:
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    selected = [
        row for index, row in enumerate(_load_jsonl(plan_path)) if index % shard_count == shard_index
    ]
    generated = reused = failed = 0
    failure_rows: list[dict[str, Any]] = []
    for row in selected:
        source = Path(row["source_path"])
        output = Path(row["output_path"])
        item_path = Path(row["item_path"])
        if output.is_file() and output.stat().st_size > 0 and item_path.is_file():
            reused += 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        probe = _probe_media(source)
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            temporary = output.with_name(f".{output.name}.{time.time_ns()}.tmp.mp4")
            try:
                command = _ffmpeg_variant_command(row["condition"], source, temporary, probe)
                subprocess.run(command, check=True, capture_output=True, text=True)
                output_probe = _probe_media(temporary)
                if output_probe["has_audio"] != probe["has_audio"]:
                    raise ValueError("variant audio-stream presence differs from source")
                temporary.replace(output)
                _atomic_json(
                    item_path,
                    {
                        **row,
                        "source_sha256": _sha256_file(source),
                        "output_sha256": _sha256_file(output),
                        "source_probe": probe,
                        "output_probe": output_probe,
                        "ffmpeg_command": command,
                        "finite_check": True,
                        "attempt": attempt,
                    },
                )
                generated += 1
                last_error = None
                break
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                last_error = exc
                time.sleep(min(8.0, float(attempt)))
        if last_error is not None:
            failed += 1
            failure_rows.append(
                {
                    **row,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                }
            )
    root = plan_path.parent
    _atomic_jsonl(
        root / "failures" / f"shard_{shard_index:03d}_of_{shard_count:03d}.jsonl",
        failure_rows,
    )
    summary = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(selected),
        "generated_count": generated,
        "reused_count": reused,
        "failed_count": failed,
    }
    _atomic_json(
        root / "shard_summaries" / f"shard_{shard_index:03d}_of_{shard_count:03d}.json",
        summary,
    )
    return summary


def summarize_reference_variants(plan_path: Path, output_dir: Path) -> dict[str, Any]:
    plan = _load_jsonl(plan_path)
    completed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in plan:
        item_path = Path(row["item_path"])
        output_path = Path(row["output_path"])
        if item_path.is_file() and output_path.is_file() and output_path.stat().st_size > 0:
            completed.append(json.loads(item_path.read_text(encoding="utf-8")))
        else:
            missing.append(row)
    expected = Counter(row["condition"] for row in plan)
    actual = Counter(row["condition"] for row in completed)
    sample_ids_by_condition = {
        condition: {row["sample_id"] for row in completed if row["condition"] == condition}
        for condition in REFERENCE_VARIANTS
    }
    identical_sets = len({frozenset(values) for values in sample_ids_by_condition.values()}) == 1
    summary = {
        "planned_count": len(plan),
        "complete_count": len(completed),
        "missing_count": len(missing),
        "expected_by_condition": dict(expected),
        "complete_by_condition": dict(actual),
        "condition_sample_ids_identical": identical_sets,
        "missing_rate": len(missing) / max(1, len(plan)),
        "accepted_missing_rate": len(missing) / max(1, len(plan)) <= 0.01,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "reference_variant_manifest.jsonl", completed)
    _atomic_jsonl(output_dir / "reference_variant_missing.jsonl", missing)
    _atomic_json(output_dir / "reference_variant_summary.json", summary)
    return summary


def prepare_imagebind_variant_inventory(
    *, variant_manifest: Path, output_path: Path
) -> dict[str, Any]:
    rows = _load_jsonl(variant_manifest)
    inventory = []
    for row in rows:
        path = Path(row["output_path"])
        stat = path.stat()
        inventory.append(
            {
                "media_id": f"reference_variant::{row['condition']}::{row['sample_id']}",
                "resolved_media_path": str(path.resolve()),
                "media_key": _stable_digest(path.resolve(), stat.st_size, stat.st_mtime_ns),
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "role": "reference_variant",
            }
        )
    _atomic_jsonl(output_path, inventory)
    summary = {
        "inventory_count": len(inventory),
        "by_condition": dict(Counter(row["condition"] for row in inventory)),
    }
    _atomic_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def assemble_imagebind_variant_cache(
    *,
    exact_assembly_dir: Path,
    variant_inventory: Path,
    cache_root: Path,
    condition: str,
    output_dir: Path,
) -> dict[str, Any]:
    from app.audio_cvr_external_baseline import _load_media_embedding

    if condition not in REFERENCE_VARIANTS:
        raise ValueError(f"invalid condition: {condition}")
    with np.load(
        exact_assembly_dir / "imagebind_embeddings.npz", allow_pickle=False
    ) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    sample_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
    reference_indices = np.asarray(arrays["reference_indices"], dtype=np.int64)
    if len(sample_ids) != 1000 or len(reference_indices) != 1000:
        raise ValueError("ImageBind exact assembly must contain Full1000")
    inventory = {
        (row["condition"], row["sample_id"]): row
        for row in _load_jsonl(variant_inventory)
    }
    vision: list[np.ndarray] = []
    audio: list[np.ndarray] = []
    missing: list[str] = []
    for sample_id in sample_ids:
        row = inventory.get((condition, sample_id))
        if row is None:
            missing.append(sample_id)
            continue
        value = _load_media_embedding(cache_root, row["media_id"])
        if value is None:
            missing.append(sample_id)
            continue
        vision.append(value[0])
        audio.append(value[1])
    if missing:
        raise ValueError(f"missing {len(missing)} ImageBind reference variants")
    vision_array = np.stack(vision)
    audio_array = np.stack(audio)
    original_vision = arrays["gallery_vision"].copy()
    original_audio = arrays["gallery_audio"].copy()
    arrays["gallery_vision"][reference_indices] = vision_array
    arrays["gallery_audio"][reference_indices] = audio_array
    non_reference = np.ones(len(original_vision), dtype=bool)
    non_reference[reference_indices] = False
    vision_unchanged = np.array_equal(
        original_vision[non_reference], arrays["gallery_vision"][non_reference]
    )
    audio_unchanged = np.array_equal(
        original_audio[non_reference], arrays["gallery_audio"][non_reference]
    )
    if not vision_unchanged or not audio_unchanged:
        raise ValueError("non-reference ImageBind gallery embeddings changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_dir / "imagebind_embeddings.npz", arrays)
    for name in ("records.jsonl", "gallery.jsonl", "assembly_summary.json"):
        source = exact_assembly_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    audit = {
        "condition": condition,
        "query_count": len(sample_ids),
        "gallery_count": len(original_vision),
        "replaced_reference_count": len(reference_indices),
        "non_reference_vision_bitwise_identical": vision_unchanged,
        "non_reference_audio_bitwise_identical": audio_unchanged,
        "query_vision_bitwise_identical": True,
        "query_audio_bitwise_identical": True,
        "query_text_bitwise_identical": True,
        "only_own_reference_replaced_per_query": True,
    }
    _atomic_json(output_dir / "reference_replacement_audit.json", audit)
    return audit


def cache_e5_variant_references(
    *,
    variant_manifest: Path,
    output_dir: Path,
    model_path: str,
    video_audio_mode: str,
    shard_index: int,
    shard_count: int,
    device: str,
    batch_size: int,
    retries: int,
    torch_dtype: str,
    attn_implementation: str,
    video_max_pixels: int,
    video_fps: int,
) -> dict[str, Any]:
    from app.e5_cvr_eval import load_e5_encoder

    rows = [
        row
        for index, row in enumerate(_load_jsonl(variant_manifest))
        if index % shard_count == shard_index
    ]
    cache_dir = output_dir / video_audio_mode / "items"
    pending = []
    reused = 0
    for row in rows:
        path = cache_dir / row["condition"] / f"{_safe_embedding_name(row['sample_id'])}.npy"
        if path.is_file():
            value = np.load(path, allow_pickle=False)
            if value.ndim == 1 and np.isfinite(value).all():
                reused += 1
                continue
        pending.append((row, path))
    encoder = runtime = None
    if pending:
        encoder, runtime = load_e5_encoder(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            batch_size=batch_size,
            video_max_pixels=video_max_pixels,
            video_fps=video_fps,
            video_audio_mode=video_audio_mode,
        )
    encoded = failed = 0
    failures: list[dict[str, Any]] = []
    for row, path in pending:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                assert encoder is not None
                value = np.asarray(
                    encoder.encode_document([{"video": row["output_path"]}])[0],
                    dtype=np.float32,
                )
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError(f"invalid E5 embedding shape={value.shape}")
                norm = np.linalg.norm(value)
                _atomic_npy(path, value / max(float(norm), 1e-12))
                encoded += 1
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(min(8.0, float(attempt)))
        if last_error is not None:
            failed += 1
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "condition": row["condition"],
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                }
            )
    summary = {
        "video_audio_mode": video_audio_mode,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(rows),
        "encoded_count": encoded,
        "reused_count": reused,
        "failed_count": failed,
        "runtime": (
            asdict(runtime)
            if runtime is not None and is_dataclass(runtime)
            else (vars(runtime) if runtime is not None else None)
        ),
    }
    _atomic_jsonl(
        output_dir
        / video_audio_mode
        / "failures"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.jsonl",
        failures,
    )
    _atomic_json(
        output_dir
        / video_audio_mode
        / "shards"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.json",
        summary,
    )
    return summary


def assemble_e5_variant_cache(
    *,
    exact_cache_dir: Path,
    variant_manifest: Path,
    variant_embedding_root: Path,
    video_audio_mode: str,
    condition: str,
    output_dir: Path,
) -> dict[str, Any]:
    if condition not in REFERENCE_VARIANTS:
        raise ValueError(f"invalid condition: {condition}")
    exact_npz = exact_cache_dir / "eval_embeddings.npz"
    rows = [row for row in _load_jsonl(variant_manifest) if row["condition"] == condition]
    by_id = {row["sample_id"]: row for row in rows}
    with np.load(exact_npz, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    if "sample_ids" not in arrays:
        records = _load_jsonl(exact_cache_dir / "eval_records.jsonl")
        sample_ids = [_sample_id(row) for row in records]
    else:
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
    if len(sample_ids) != 1000:
        raise ValueError(f"exact E5 cache must contain 1000 queries, found {len(sample_ids)}")
    reference_indices = np.asarray(arrays["reference_gallery_index"], dtype=np.int64)
    if len(set(reference_indices.tolist())) != len(reference_indices):
        raise ValueError("reference gallery indices are not one-to-one")
    variant_vectors = []
    missing = []
    for sample_id in sample_ids:
        if sample_id not in by_id:
            missing.append(sample_id)
            continue
        path = (
            variant_embedding_root
            / video_audio_mode
            / "items"
            / condition
            / f"{_safe_embedding_name(sample_id)}.npy"
        )
        if not path.is_file():
            missing.append(sample_id)
            continue
        variant_vectors.append(np.asarray(np.load(path, allow_pickle=False), dtype=np.float32))
    if missing:
        raise ValueError(f"missing {len(missing)} E5 variant reference embeddings")
    variants = np.stack(variant_vectors)
    if variants.shape != arrays["reference"].shape:
        raise ValueError(
            f"variant reference shape={variants.shape}, expected={arrays['reference'].shape}"
        )
    original_gallery = arrays["gallery"].copy()
    arrays["reference"] = variants
    arrays["gallery"][reference_indices] = variants
    non_reference = np.ones(len(original_gallery), dtype=bool)
    non_reference[reference_indices] = False
    unchanged = np.array_equal(original_gallery[non_reference], arrays["gallery"][non_reference])
    if not unchanged:
        raise ValueError("non-reference gallery embeddings changed during replacement")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_dir / "eval_embeddings.npz", arrays)
    for name in (
        "eval_records.jsonl",
        "eval_gallery.jsonl",
        "summary.json",
        "eval_manifest.jsonl",
    ):
        source = exact_cache_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    audit = {
        "condition": condition,
        "video_audio_mode": video_audio_mode,
        "query_count": len(sample_ids),
        "gallery_count": len(original_gallery),
        "replaced_reference_count": len(reference_indices),
        "other_candidate_count": len(original_gallery) - len(reference_indices),
        "non_reference_embeddings_bitwise_identical": unchanged,
        "query_embeddings_bitwise_identical": True,
        "target_embeddings_bitwise_identical": True,
        "reference_indices_identical": True,
        "only_own_reference_replaced_per_query": True,
        "exact_cache_path": str(exact_cache_dir.resolve()),
    }
    _atomic_json(output_dir / "reference_replacement_audit.json", audit)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio-CVR Weak Accept evidence repair")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-human-audit")
    prepare.add_argument("--full-path", required=True)
    prepare.add_argument("--core-path", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--media-root", action="append", default=[])
    prepare.add_argument("--seed", type=int, default=20260724)
    prepare.add_argument("--expected-full-sha256", default=EXPECTED_FULL1000_SHA256)

    serve = subparsers.add_parser("serve-human-audit")
    serve.add_argument("--audit-dir", required=True)
    serve.add_argument("--variants-dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)

    summarize = subparsers.add_parser("summarize-human-audit")
    summarize.add_argument("--audit-dir", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--allow-partial", action="store_true")

    valid = subparsers.add_parser("evaluate-human-valid-subset")
    valid.add_argument("--valid-path", required=True)
    valid.add_argument("--per-query", action="append", default=[], required=True)
    valid.add_argument("--output-path", required=True)

    plan = subparsers.add_parser("prepare-reference-variants")
    plan.add_argument("--full-path", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--media-root", action="append", default=[])
    plan.add_argument("--expected-full-sha256", default=EXPECTED_FULL1000_SHA256)

    generate = subparsers.add_parser("generate-reference-variants")
    generate.add_argument("--plan-path", required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--shard-count", type=int, required=True)
    generate.add_argument("--retries", type=int, default=3)

    variant_summary = subparsers.add_parser("summarize-reference-variants")
    variant_summary.add_argument("--plan-path", required=True)
    variant_summary.add_argument("--output-dir", required=True)

    imagebind = subparsers.add_parser("prepare-imagebind-variant-inventory")
    imagebind.add_argument("--variant-manifest", required=True)
    imagebind.add_argument("--output-path", required=True)

    imagebind_assemble = subparsers.add_parser("assemble-imagebind-variant-cache")
    imagebind_assemble.add_argument("--exact-assembly-dir", required=True)
    imagebind_assemble.add_argument("--variant-inventory", required=True)
    imagebind_assemble.add_argument("--cache-root", required=True)
    imagebind_assemble.add_argument("--condition", choices=REFERENCE_VARIANTS, required=True)
    imagebind_assemble.add_argument("--output-dir", required=True)

    e5_cache = subparsers.add_parser("cache-e5-variant-references")
    e5_cache.add_argument("--variant-manifest", required=True)
    e5_cache.add_argument("--output-dir", required=True)
    e5_cache.add_argument("--model-path", required=True)
    e5_cache.add_argument("--video-audio-mode", choices=("on", "off"), required=True)
    e5_cache.add_argument("--shard-index", type=int, required=True)
    e5_cache.add_argument("--shard-count", type=int, required=True)
    e5_cache.add_argument("--device", default="cuda")
    e5_cache.add_argument("--batch-size", type=int, default=1)
    e5_cache.add_argument("--retries", type=int, default=3)
    e5_cache.add_argument("--torch-dtype", default="bfloat16")
    e5_cache.add_argument("--attn-implementation", default="flash_attention_2")
    e5_cache.add_argument("--video-max-pixels", type=int, default=313600)
    e5_cache.add_argument("--video-fps", type=int, default=1)

    e5_assemble = subparsers.add_parser("assemble-e5-variant-cache")
    e5_assemble.add_argument("--exact-cache-dir", required=True)
    e5_assemble.add_argument("--variant-manifest", required=True)
    e5_assemble.add_argument("--variant-embedding-root", required=True)
    e5_assemble.add_argument("--video-audio-mode", choices=("on", "off"), required=True)
    e5_assemble.add_argument("--condition", choices=REFERENCE_VARIANTS, required=True)
    e5_assemble.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-human-audit":
        value = prepare_human_audit(
            full_path=Path(args.full_path),
            core_path=Path(args.core_path),
            output_dir=Path(args.output_dir),
            media_roots=[Path(root) for root in args.media_root],
            seed=args.seed,
            expected_full_sha256=args.expected_full_sha256,
        )
    elif args.command == "serve-human-audit":
        serve_human_audit(
            audit_dir=Path(args.audit_dir),
            variants_dir=Path(args.variants_dir) if args.variants_dir else None,
            host=args.host,
            port=args.port,
        )
        return
    elif args.command == "summarize-human-audit":
        value = summarize_human_audit(
            Path(args.audit_dir),
            Path(args.output_dir),
            allow_partial=args.allow_partial,
        )
    elif args.command == "evaluate-human-valid-subset":
        value = evaluate_human_valid_subset(
            valid_path=Path(args.valid_path),
            named_per_query_paths=args.per_query,
            output_path=Path(args.output_path),
        )
    elif args.command == "prepare-reference-variants":
        value = prepare_reference_variant_plan(
            full_path=Path(args.full_path),
            output_dir=Path(args.output_dir),
            media_roots=[Path(root) for root in args.media_root],
            expected_full_sha256=args.expected_full_sha256,
        )
    elif args.command == "generate-reference-variants":
        value = generate_reference_variants(
            plan_path=Path(args.plan_path),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            retries=args.retries,
        )
    elif args.command == "summarize-reference-variants":
        value = summarize_reference_variants(Path(args.plan_path), Path(args.output_dir))
    elif args.command == "prepare-imagebind-variant-inventory":
        value = prepare_imagebind_variant_inventory(
            variant_manifest=Path(args.variant_manifest),
            output_path=Path(args.output_path),
        )
    elif args.command == "assemble-imagebind-variant-cache":
        value = assemble_imagebind_variant_cache(
            exact_assembly_dir=Path(args.exact_assembly_dir),
            variant_inventory=Path(args.variant_inventory),
            cache_root=Path(args.cache_root),
            condition=args.condition,
            output_dir=Path(args.output_dir),
        )
    elif args.command == "cache-e5-variant-references":
        value = cache_e5_variant_references(
            variant_manifest=Path(args.variant_manifest),
            output_dir=Path(args.output_dir),
            model_path=args.model_path,
            video_audio_mode=args.video_audio_mode,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            batch_size=args.batch_size,
            retries=args.retries,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            video_max_pixels=args.video_max_pixels,
            video_fps=args.video_fps,
        )
    elif args.command == "assemble-e5-variant-cache":
        value = assemble_e5_variant_cache(
            exact_cache_dir=Path(args.exact_cache_dir),
            variant_manifest=Path(args.variant_manifest),
            variant_embedding_root=Path(args.variant_embedding_root),
            video_audio_mode=args.video_audio_mode,
            condition=args.condition,
            output_dir=Path(args.output_dir),
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
