from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.composed_data import probe_media


VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}
DEFAULT_SEED = 20260722


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str | None
    license: str
    source_kind: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "existing_vggsound": DatasetSpec(
        name="existing_vggsound",
        repo_id=None,
        license="upstream VGGSound / source-video terms",
        source_kind="existing_vggsound",
    ),
    "avqa_videos": DatasetSpec(
        name="avqa_videos",
        repo_id="juyil/AVQA-videos",
        license="non-commercial research; underlying VGGSound/YouTube rights",
        source_kind="avqa",
    ),
    "ave_dataset": DatasetSpec(
        name="ave_dataset",
        repo_id="UnFaZeD07/AVE-Dataset",
        license="MIT metadata; underlying video rights remain upstream",
        source_kind="ave",
    ),
    "avscapbench": DatasetSpec(
        name="avscapbench",
        repo_id="NJU-LINK/AVSCapBench",
        license="CC-BY-NC-SA-4.0",
        source_kind="avscap",
    ),
}


def parse_key_value_counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw in values:
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            key, separator, value = item.partition("=")
            if not separator or not key.strip():
                raise ValueError(f"expected NAME=COUNT, got {item!r}")
            count = int(value)
            if count < 0:
                raise ValueError(f"count must be non-negative: {item!r}")
            result[key.strip()] = count
    return result


def parse_avqa_video_identity(stem: str) -> str:
    """Remove only a final numeric start-time suffix from an AVQA/VGGSound stem."""
    match = re.match(r"^(.+)_(-?\d+(?:\.\d+)?)$", str(stem).strip())
    return match.group(1) if match else str(stem).strip()


def stable_order_key(dataset: str, path: Path, root: Path, seed: int) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    return hashlib.sha256(f"{seed}|{dataset}|{relative}".encode("utf-8")).hexdigest()


def source_candidate_key(dataset: str, path: Path) -> str:
    return hashlib.sha256(f"{dataset}|{path.resolve().as_posix()}".encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def prepare_mirror_sources(
    *,
    root: str | Path,
    run_root: str | Path,
    datasets: list[str],
    source_targets: dict[str, int],
    exclude_overlap_with: list[str | Path] | None = None,
    hf_endpoint: str = "https://hf-mirror.com",
    min_duration_seconds: float = 6.0,
    seed: int = DEFAULT_SEED,
    probe_workers: int = 16,
    materialize_mode: str = "symlink",
    skip_download: bool = False,
    allow_partial_downloads: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    run_path = Path(run_root).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    download_root = root_path / "raw_datasets" / "hf_mirror_audio_cvr"
    construction_root = run_path / "construction_data"
    staged_raw_root = construction_root / "raw"
    staged_raw_root.mkdir(parents=True, exist_ok=True)

    unknown = sorted(set(datasets) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    for dataset in datasets:
        if dataset not in source_targets:
            raise ValueError(f"missing source target for {dataset}")

    ingest_config = {
        "datasets": datasets,
        "source_targets": source_targets,
        "exclude_overlap_with": sorted(str(Path(value).resolve()) for value in (exclude_overlap_with or [])),
        "hf_endpoint": hf_endpoint,
        "min_duration_seconds": float(min_duration_seconds),
        "seed": int(seed),
        "materialize_mode": materialize_mode,
    }
    ingest_config_path = run_path / "source_ingest_config.json"
    ingest_progress_path = run_path / "source_ingest.progress.jsonl"
    if ingest_config_path.exists():
        existing_config = json.loads(ingest_config_path.read_text(encoding="utf-8"))
        if existing_config != ingest_config:
            raise ValueError(
                "source ingest configuration changed for an existing run root; "
                "use the original options with --resume or choose a new run root"
            )
        if ingest_progress_path.exists() and not resume:
            raise ValueError("source ingest progress already exists; pass --resume to preserve completed work")
    else:
        _write_json(ingest_config_path, ingest_config)

    source_roots: dict[str, Path] = {}
    download_summaries: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        if dataset == "existing_vggsound":
            source_root = root_path / "raw" / "vggsound"
            if not source_root.exists():
                raise FileNotFoundError(f"existing VGGSound root is missing: {source_root}")
            download_summaries[dataset] = {
                "status": "existing_local_source",
                "source_root": str(source_root),
            }
        else:
            source_root = download_root / dataset
            if not skip_download:
                download_summaries[dataset] = _download_dataset(
                    spec,
                    source_root,
                    hf_endpoint=hf_endpoint,
                    resume=resume,
                    allow_partial=allow_partial_downloads,
                )
            else:
                download_summaries[dataset] = {
                    "status": "download_skipped",
                    "source_root": str(source_root),
                }
            if not source_root.exists():
                raise FileNotFoundError(f"downloaded dataset root is missing: {source_root}")
        source_roots[dataset] = source_root

    exclusion_roots = [Path(value).resolve() for value in (exclude_overlap_with or []) if Path(value).exists()]
    exclusion_index = _build_exclusion_index(exclusion_roots, run_path=run_path, resume=resume)
    progress_events = _load_jsonl(ingest_progress_path)
    processed_events = {
        str(event.get("candidate_key") or ""): event
        for event in progress_events
        if str(event.get("candidate_key") or "")
    }
    global_selected_hashes: set[str] = set()
    global_selected_identities: set[str] = set()
    provenance: list[dict[str, Any]] = []
    dataset_summaries: dict[str, dict[str, Any]] = {}
    aggregate_rejections: Counter[str] = Counter()

    for event in processed_events.values():
        if event.get("decision") != "selected" or str(event.get("dataset") or "") not in datasets:
            continue
        row = event.get("record")
        if not isinstance(row, dict):
            continue
        provenance.append(row)
        content_hash = str(row.get("content_sha256") or "")
        if content_hash:
            global_selected_hashes.add(content_hash)
        stem = str(row.get("original_stem") or "")
        identity = str(row.get("source_identity") or "")
        global_selected_identities.update(value.casefold() for value in (stem, identity) if value)

    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        source_root = source_roots[dataset]
        candidates = _eligible_dataset_videos(spec, source_root)
        candidates.sort(key=lambda path: stable_order_key(dataset, path, source_root, seed))
        target = source_targets[dataset]
        selected = [row for row in provenance if str(row.get("dataset") or "") == dataset]
        rejection_counts: Counter[str] = Counter(
            str(event.get("reason") or "unspecified")
            for event in processed_events.values()
            if str(event.get("dataset") or "") == dataset and event.get("decision") == "rejected"
        )

        # Probe in deterministic windows. We intentionally inspect more than the target
        # so invalid media do not silently reduce the selected count.
        window_size = max(64, probe_workers * 8)
        for offset in range(0, len(candidates), window_size):
            if len(selected) >= target:
                break
            window = [
                path
                for path in candidates[offset : offset + window_size]
                if source_candidate_key(dataset, path) not in processed_events
            ]
            if not window:
                continue
            with ThreadPoolExecutor(max_workers=max(1, probe_workers)) as executor:
                media_rows = list(executor.map(probe_media, window))
            for path, media in zip(window, media_rows):
                if len(selected) >= target:
                    break
                candidate_key = source_candidate_key(dataset, path)

                def reject(reason: str) -> None:
                    event = {
                        "event": "source_ingest_decision",
                        "candidate_key": candidate_key,
                        "dataset": dataset,
                        "original_path": str(path.resolve()),
                        "decision": "rejected",
                        "reason": reason,
                        "media_probe": media,
                    }
                    _append_jsonl_durable(ingest_progress_path, event)
                    processed_events[candidate_key] = event
                    rejection_counts[reason] += 1

                if media.get("error"):
                    reject("probe_error")
                    continue
                if not media.get("has_video"):
                    reject("missing_video_stream")
                    continue
                if not media.get("has_audio"):
                    reject("missing_audio_stream")
                    continue
                if float(media.get("duration_seconds") or 0.0) < float(min_duration_seconds):
                    reject("duration_below_minimum")
                    continue

                stem = path.stem
                identity = parse_avqa_video_identity(stem)
                identity_keys = {stem.casefold(), identity.casefold()}
                # Existing VGGSound may overlap AVATAR, but must not reject itself
                # merely because the user also supplied the VGGSound source root.
                applicable_exclusion_keys = exclusion_index["identity_keys"]
                if dataset == "existing_vggsound":
                    applicable_exclusion_keys = exclusion_index["avatar_identity_keys"]
                if identity_keys & applicable_exclusion_keys:
                    reject("overlap_identity")
                    continue
                if identity_keys & global_selected_identities:
                    reject("duplicate_selected_identity")
                    continue

                content_hash = sha256_file(path)
                exclusion_paths = exclusion_index["hash_to_paths"].get(content_hash, set())
                resolved = path.resolve().as_posix()
                different_exclusion_paths = {value for value in exclusion_paths if value != resolved}
                if dataset == "existing_vggsound":
                    different_exclusion_paths = {
                        value for value in different_exclusion_paths if "/raw/avatar/" in value.replace("\\", "/")
                    }
                if different_exclusion_paths:
                    reject("overlap_content_hash")
                    continue
                if content_hash in global_selected_hashes:
                    reject("duplicate_selected_content")
                    continue

                staged_path = _materialize_video(
                    path=path,
                    staged_dataset_root=staged_raw_root / dataset,
                    content_hash=content_hash,
                    mode=materialize_mode,
                )
                metadata = _candidate_metadata(spec, source_root, path)
                row = {
                    "dataset": dataset,
                    "repo_id": spec.repo_id,
                    "license": spec.license,
                    "original_path": str(path.resolve()),
                    "staged_path": str(staged_path.resolve() if staged_path.exists() else staged_path),
                    "construction_relative_path": staged_path.relative_to(construction_root).as_posix(),
                    "source_identity": identity,
                    "original_stem": stem,
                    "content_sha256": content_hash,
                    "selection_order_hash": stable_order_key(dataset, path, source_root, seed),
                    "media_probe": media,
                    "source_metadata": metadata,
                }
                selected.append(row)
                provenance.append(row)
                global_selected_hashes.add(content_hash)
                global_selected_identities.update(identity_keys)
                event = {
                    "event": "source_ingest_decision",
                    "candidate_key": candidate_key,
                    "dataset": dataset,
                    "original_path": str(path.resolve()),
                    "decision": "selected",
                    "reason": "",
                    "record": row,
                }
                _append_jsonl_durable(ingest_progress_path, event)
                processed_events[candidate_key] = event

        dataset_summaries[dataset] = {
            "repo_id": spec.repo_id,
            "license": spec.license,
            "source_root": str(source_root),
            "discovered_video_count": len(candidates),
            "requested_source_count": target,
            "selected_source_count": len(selected),
            "rejection_counts": dict(rejection_counts),
        }
        aggregate_rejections.update(rejection_counts)

    _write_jsonl(run_path / "provenance_manifest.jsonl", provenance)
    selected_dataset_counts = Counter(row["dataset"] for row in provenance)
    media_summary = {
        "selected_count": len(provenance),
        "selected_dataset_counts": dict(selected_dataset_counts),
        "valid_audio_video_count": sum(
            1 for row in provenance if row["media_probe"].get("has_audio") and row["media_probe"].get("has_video")
        ),
        "min_duration_seconds": min_duration_seconds,
        "probe_workers": probe_workers,
        "rejection_counts": dict(aggregate_rejections),
    }
    overlap_summary = {
        "exclude_overlap_roots": [str(path) for path in exclusion_roots],
        "indexed_identity_count": len(exclusion_index["identity_keys"]),
        "indexed_content_hash_count": len(exclusion_index["hash_to_paths"]),
        "overlap_identity_rejected": aggregate_rejections["overlap_identity"],
        "overlap_content_hash_rejected": aggregate_rejections["overlap_content_hash"],
        "within_selection_identity_rejected": aggregate_rejections["duplicate_selected_identity"],
        "within_selection_content_rejected": aggregate_rejections["duplicate_selected_content"],
    }
    summary = {
        "status": "complete",
        "root": str(root_path),
        "run_root": str(run_path),
        "construction_root": str(construction_root),
        "staged_raw_root": str(staged_raw_root),
        "hf_endpoint": hf_endpoint,
        "seed": seed,
        "datasets": datasets,
        "source_targets": source_targets,
        "download_summaries": download_summaries,
        "selected_source_count": len(provenance),
        "selected_dataset_counts": dict(selected_dataset_counts),
        "dataset_summaries": dataset_summaries,
        "selection_uses_model_scores": False,
        "frozen_benchmark_modified": False,
        "provenance_manifest_path": str(run_path / "provenance_manifest.jsonl"),
        "progress_path": str(ingest_progress_path),
        "durable_decision_count": len(processed_events),
    }
    _write_json(run_path / "source_ingest_summary.json", summary)
    _write_json(run_path / "media_probe_summary.json", media_summary)
    _write_json(run_path / "overlap_dedup_summary.json", overlap_summary)
    return summary


def summarize_supplement_run(*, run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    all_rows = _load_jsonl(root / "b_all_audio_cvr_triplets.jsonl")
    ranked_rows = _load_jsonl(root / "b_ranked_single_source_pairs.jsonl")
    rejection_counts: Counter[str] = Counter()
    for row in ranked_rows:
        if row.get("accepted"):
            continue
        reasons = row.get("reject_reasons") or row.get("rejection_reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        if not reasons:
            reasons = [str(row.get("decision_reason") or row.get("reason") or "unspecified")]
        rejection_counts.update(str(value) for value in reasons if str(value).strip())

    crosstab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        dataset = _first_text(row, "dataset", "source_dataset") or "unknown"
        subtype = _first_text(row, "b_subtype", "audio_delta_type") or "unknown"
        crosstab[dataset][subtype] += 1
    crosstab_payload = {
        "accepted_count": len(all_rows),
        "dataset_subtype_counts": {dataset: dict(counter) for dataset, counter in sorted(crosstab.items())},
        "dataset_counts": dict(Counter(_first_text(row, "dataset", "source_dataset") or "unknown" for row in all_rows)),
        "subtype_counts": dict(Counter(_first_text(row, "b_subtype", "audio_delta_type") or "unknown" for row in all_rows)),
    }
    rejection_payload = {
        "ranked_count": len(ranked_rows),
        "accepted_count": len(all_rows),
        "rejected_count": sum(1 for row in ranked_rows if not row.get("accepted")),
        "rejection_reason_counts": dict(rejection_counts),
    }
    _write_json(root / "rejection_breakdown.json", rejection_payload)
    _write_json(root / "subtype_dataset_crosstab.json", crosstab_payload)
    return {"rejection_breakdown": rejection_payload, "subtype_dataset_crosstab": crosstab_payload}


def extend_frozen_test(
    *,
    existing_test_path: str | Path,
    candidate_path: str | Path,
    output_dir: str | Path,
    target_count: int = 1000,
    sound_event_target: int = 800,
    music_target: int = 200,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    existing_path = Path(existing_test_path)
    candidates_path = Path(candidate_path)
    output_root = Path(output_dir)
    existing = _load_jsonl(existing_path)
    candidates = _load_jsonl(candidates_path)
    if not existing:
        raise ValueError(f"existing frozen test is empty: {existing_path}")
    if target_count != sound_event_target + music_target:
        raise ValueError("sound_event_target + music_target must equal target_count")
    if len(existing) >= target_count:
        raise ValueError(f"existing test already has {len(existing)} rows; target is {target_count}")

    existing_ids = _record_identity_sets(existing)
    existing_subtypes = Counter(_record_subtype(row) for row in existing)
    required = {
        "sound_event": sound_event_target - existing_subtypes["sound_event"],
        "music": music_target - existing_subtypes["music"],
    }
    if any(value < 0 for value in required.values()):
        raise ValueError(f"existing test already exceeds a subtype target: {required}")
    if sum(required.values()) != target_count - len(existing):
        raise ValueError(
            "existing test contains unsupported subtypes or does not match the requested extension; "
            f"existing_subtypes={dict(existing_subtypes)} required={required}"
        )

    rejection_counts: Counter[str] = Counter()
    eligible_by_subtype: dict[str, list[dict[str, Any]]] = {"sound_event": [], "music": []}
    for row in candidates:
        reason = _test_extension_rejection_reason(row, existing_ids)
        if reason:
            rejection_counts[reason] += 1
            continue
        normalized = dict(row)
        normalized["source_disjoint_group_id"] = _record_source_id(row)
        normalized["pair_group_id"] = _record_pair_id(row)
        normalized["sample_id"] = _record_sample_id(row)
        normalized["benchmark_extension_origin"] = "audio_cvr_avatar_like_supplement"
        subtype = _record_subtype(row)
        eligible_by_subtype[subtype].append(normalized)

    for subtype, rows in eligible_by_subtype.items():
        rows.sort(key=lambda row: _test_extension_sort_key(row, seed=seed))

    # Music is the scarcer class in the requested 80/20 benchmark. Allocate its
    # exact quota first, then fill sound events without reusing a source or pair.
    additions: list[dict[str, Any]] = []
    selected_ids = {key: set(values) for key, values in existing_ids.items()}
    for subtype in ("music", "sound_event"):
        if required[subtype] == 0:
            continue
        selected_for_subtype = 0
        for row in eligible_by_subtype[subtype]:
            identities = {
                "source": _record_source_id(row),
                "pair": _record_pair_id(row),
                "sample": _record_sample_id(row),
            }
            duplicate_label = next(
                (label for label, value in identities.items() if value in selected_ids[label]),
                "",
            )
            if duplicate_label:
                rejection_counts[f"duplicate_candidate_{duplicate_label}"] += 1
                continue
            additions.append(row)
            for label, value in identities.items():
                selected_ids[label].add(value)
            selected_for_subtype += 1
            if selected_for_subtype >= required[subtype]:
                break
        if selected_for_subtype < required[subtype]:
            raise ValueError(
                f"not enough eligible {subtype} candidates: available={selected_for_subtype} "
                f"required={required[subtype]}; "
                f"rejections={dict(rejection_counts)}"
            )

    selected_sample_ids = {_record_sample_id(row) for row in additions}
    reserve: list[dict[str, Any]] = []
    reserve_ids = {key: set(values) for key, values in selected_ids.items()}
    for subtype in ("music", "sound_event"):
        for row in eligible_by_subtype[subtype]:
            if _record_sample_id(row) in selected_sample_ids:
                continue
            identities = {
                "source": _record_source_id(row),
                "pair": _record_pair_id(row),
                "sample": _record_sample_id(row),
            }
            if any(value in reserve_ids[label] for label, value in identities.items()):
                continue
            reserve.append(row)
            for label, value in identities.items():
                reserve_ids[label].add(value)
    additions.sort(key=lambda row: (_record_subtype(row), _test_extension_sort_key(row, seed=seed)))
    reserve.sort(key=lambda row: (_record_subtype(row), _test_extension_sort_key(row, seed=seed)))
    combined = existing + additions
    if len(combined) != target_count:
        raise AssertionError(f"combined test count mismatch: {len(combined)} != {target_count}")
    audit = _audit_test_records(combined)
    if audit["violation_count"]:
        raise ValueError(f"test extension leakage audit failed: {audit['violations'][:20]}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"test_main_{target_count}.jsonl"
    _write_jsonl(output_path, combined)
    digest = sha256_file(output_path)
    _write_text_atomic(
        output_root / f"test_main_{target_count}.sha256",
        f"{digest}  {output_path.name}\n",
    )
    manifest = {
        "status": "frozen",
        "existing_test_path": str(existing_path.resolve()),
        "candidate_path": str(candidates_path.resolve()),
        "output_path": str(output_path.resolve()),
        "existing_count": len(existing),
        "added_count": len(additions),
        "reserve_count": len(reserve),
        "target_count": target_count,
        "subtype_counts": dict(Counter(_record_subtype(row) for row in combined)),
        "added_subtype_counts": dict(Counter(_record_subtype(row) for row in additions)),
        "dataset_counts": dict(Counter(_record_dataset(row) for row in combined)),
        "sha256": digest,
        "seed": seed,
        "selection_uses_model_scores": False,
        "legacy_test_rows_preserved_first": True,
        "rejection_counts": dict(rejection_counts),
        "audit": audit,
    }
    _write_json(output_root / "frozen_test1000_manifest.json", manifest)
    _write_json(output_root / "test1000_leakage_audit.json", audit)
    _write_jsonl(output_root / "test1000_additions.jsonl", additions)
    _write_jsonl(output_root / "test1000_reserve_candidates.jsonl", reserve)
    return manifest


def _download_dataset(
    spec: DatasetSpec,
    destination: Path,
    *,
    hf_endpoint: str,
    resume: bool,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if spec.repo_id is None:
        return {"status": "not_required", "source_root": str(destination)}
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".audio_cvr_download_complete"
    partial_marker = destination / ".audio_cvr_download_partial.json"
    if resume and marker.exists():
        return {
            "status": "complete_cached",
            "repo_id": spec.repo_id,
            "source_root": str(destination),
        }
    executable = shutil.which("hf") or shutil.which("huggingface-cli")
    if not executable:
        raise RuntimeError("neither `hf` nor `huggingface-cli` is available")
    command = [executable, "download", spec.repo_id, "--repo-type", "dataset", "--local-dir", str(destination)]
    environment = os.environ.copy()
    environment["HF_ENDPOINT"] = hf_endpoint
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        media_files = _iter_videos(destination)
        materialized_media = [path for path in media_files if path.stat().st_size > 4096]
        partial_summary = {
            "status": "partial_after_download_error",
            "repo_id": spec.repo_id,
            "source_root": str(destination),
            "returncode": int(exc.returncode),
            "discovered_media_count": len(media_files),
            "materialized_media_count": len(materialized_media),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(partial_marker, partial_summary)
        if not allow_partial or not materialized_media:
            raise
        return partial_summary
    _write_text_atomic(marker, spec.repo_id + "\n")
    partial_marker.unlink(missing_ok=True)
    return {
        "status": "complete",
        "repo_id": spec.repo_id,
        "source_root": str(destination),
        "discovered_media_count": len(_iter_videos(destination)),
    }


def _eligible_dataset_videos(spec: DatasetSpec, source_root: Path) -> list[Path]:
    videos = _iter_videos(source_root)
    if spec.source_kind == "avqa":
        eligible = _load_avqa_eligible_stems(source_root)
        if eligible:
            videos = [path for path in videos if path.stem in eligible]
    elif spec.source_kind == "avscap":
        eligible = _load_avscap_audio_rich_ids(source_root)
        if eligible:
            videos = [path for path in videos if path.stem in eligible]
    return videos


def _load_avqa_eligible_stems(root: Path) -> set[str]:
    result: set[str] = set()
    for name in ("train_qa.json", "val_qa.json"):
        for path in root.rglob(name):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for row in payload:
                if not isinstance(row, dict):
                    continue
                relation = str(row.get("question_relation") or "").strip().casefold()
                if relation in {"sound", "both"}:
                    value = str(row.get("video_name") or "").strip()
                    if value:
                        result.add(Path(value).stem)
    return result


def _load_avscap_audio_rich_ids(root: Path) -> set[str]:
    result: set[str] = set()
    candidates = list(root.rglob("OmniCaption.json")) + list(root.rglob("metadata.jsonl"))
    for path in candidates:
        try:
            rows = _load_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, dict):
            rows = list(rows.values())
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            event = row.get("event") if isinstance(row.get("event"), dict) else {}
            audio_events = event.get("audio_events") if isinstance(event.get("audio_events"), dict) else {}
            music = audio_events.get("music") or []
            sfx = audio_events.get("sfx") or []
            if not music and not sfx:
                continue
            video_id = str(row.get("video_id") or "").strip()
            video_path = str(row.get("video_path") or row.get("file_name") or "").strip()
            if video_id:
                result.add(video_id)
            if video_path:
                result.add(Path(video_path).stem)
    return result


def _candidate_metadata(spec: DatasetSpec, source_root: Path, path: Path) -> dict[str, Any]:
    return {
        "source_kind": spec.source_kind,
        "relative_path": path.relative_to(source_root).as_posix(),
        "audio_primary_prefilter": spec.source_kind in {"avqa", "ave", "avscap", "existing_vggsound"},
    }


def _build_exclusion_index(roots: list[Path], *, run_path: Path, resume: bool) -> dict[str, Any]:
    cache_path = run_path / "overlap_source_index.jsonl"
    progress_path = run_path / "overlap_source_index.progress.jsonl"
    rows: list[dict[str, Any]] = []
    if resume and cache_path.exists():
        rows = _load_jsonl(cache_path)
    else:
        if resume and progress_path.exists():
            rows = _load_jsonl(progress_path)
        indexed_paths = {str(row.get("path") or "") for row in rows}
        videos = [path for root in roots for path in _iter_videos(root)]
        for path in videos:
            resolved_path = str(path.resolve())
            if resolved_path in indexed_paths:
                continue
            stem = path.stem
            row = {
                "path": resolved_path,
                "stem": stem,
                "source_identity": parse_avqa_video_identity(stem),
                "content_sha256": sha256_file(path),
            }
            rows.append(row)
            indexed_paths.add(resolved_path)
            _append_jsonl_durable(progress_path, row)
        _write_jsonl(cache_path, rows)
    identity_keys: set[str] = set()
    avatar_identity_keys: set[str] = set()
    hash_to_paths: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        path_text = str(row.get("path") or "")
        keys = {str(row.get("stem") or "").casefold(), str(row.get("source_identity") or "").casefold()}
        identity_keys.update(key for key in keys if key)
        if "/raw/avatar/" in path_text.replace("\\", "/"):
            avatar_identity_keys.update(key for key in keys if key)
        content_hash = str(row.get("content_sha256") or "")
        if content_hash:
            hash_to_paths[content_hash].add(path_text)
    return {
        "identity_keys": identity_keys,
        "avatar_identity_keys": avatar_identity_keys,
        "hash_to_paths": hash_to_paths,
    }


def _materialize_video(*, path: Path, staged_dataset_root: Path, content_hash: str, mode: str) -> Path:
    videos_root = staged_dataset_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    destination = videos_root / f"{content_hash[:12]}_{path.name}"
    if destination.exists() or destination.is_symlink():
        return destination
    if mode == "symlink":
        destination.symlink_to(path.resolve())
    elif mode == "hardlink":
        os.link(path, destination)
    elif mode == "copy":
        shutil.copy2(path, destination)
    else:
        raise ValueError(f"unknown materialize mode: {mode}")
    return destination


def _iter_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.as_posix(),
    )


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    payload = row.get("source_payload")
    if isinstance(payload, dict):
        return _first_text(payload, *keys)
    return ""


def _record_subtype(row: dict[str, Any]) -> str:
    value = _first_text(row, "b_subtype", "audio_delta_type").lower().replace("-", "_")
    aliases = {
        "audio_event": "sound_event",
        "sound": "sound_event",
        "sound_event": "sound_event",
        "music": "music",
    }
    return aliases.get(value, value or "unknown")


def _record_dataset(row: dict[str, Any]) -> str:
    explicit = _first_text(row, "dataset", "source_dataset").lower().replace("-", "_")
    if explicit:
        return explicit
    text = " ".join(
        (_first_text(row, "reference_video"), _first_text(row, "target_video"))
    ).lower()
    for dataset in ("existing_vggsound", "avqa_videos", "vggsound", "avatar"):
        if dataset in text:
            return dataset
    return "unknown"


def _record_source_id(row: dict[str, Any]) -> str:
    explicit = _first_text(
        row,
        "source_disjoint_group_id",
        "raw_source_id",
        "source_id",
        "source_clip_id",
        "group_id",
    )
    if explicit:
        return explicit
    path = _first_text(row, "reference_video", "target_video")
    stem = Path(path).stem
    stem = re.sub(r"__single_\d+$", "", stem, flags=re.IGNORECASE)
    if stem:
        return f"{_record_dataset(row)}:{stem}"
    return ""


def _record_pair_id(row: dict[str, Any]) -> str:
    explicit = _first_text(row, "inverse_pair_group_id", "pair_group_id")
    if explicit:
        return explicit
    pair = sorted(
        (
            _first_text(row, "reference_video").replace("\\", "/").lower(),
            _first_text(row, "target_video").replace("\\", "/").lower(),
        )
    )
    if not pair[0] or not pair[1]:
        return ""
    digest = hashlib.sha256(f"{_record_source_id(row)}|{pair[0]}|{pair[1]}".encode("utf-8")).hexdigest()[:24]
    return f"pair_{digest}"


def _record_sample_id(row: dict[str, Any]) -> str:
    explicit = _first_text(row, "sample_id", "proposal_id", "candidate_id")
    if explicit:
        return explicit
    payload = "|".join(
        (
            _first_text(row, "reference_video"),
            _first_text(row, "target_video"),
            _first_text(row, "edit_text", "audio_only_edit_text"),
        )
    )
    return f"audiocvr_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}" if payload.strip("|") else ""


def _record_identity_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result = {"source": set(), "pair": set(), "sample": set()}
    for row in rows:
        for key, value in (
            ("source", _record_source_id(row)),
            ("pair", _record_pair_id(row)),
            ("sample", _record_sample_id(row)),
        ):
            if value:
                result[key].add(value)
    return result


def _test_extension_rejection_reason(row: dict[str, Any], excluded: dict[str, set[str]]) -> str:
    if "accepted" in row and not _truthy(row.get("accepted")):
        return "not_accepted"
    if _truthy(row.get("fallback")) or _truthy(row.get("fallback_used")):
        return "fallback_record"
    if _truthy(row.get("manual_review_required")):
        return "manual_review_required"
    if _truthy(row.get("is_inverse")) or _first_text(row, "direction").lower() == "inverse":
        return "inverse_record"
    subtype = _record_subtype(row)
    if subtype not in {"sound_event", "music"}:
        return f"unsupported_subtype:{subtype}"
    source_id = _record_source_id(row)
    pair_id = _record_pair_id(row)
    sample_id = _record_sample_id(row)
    if not source_id:
        return "missing_source_id"
    if not pair_id:
        return "missing_pair_id"
    if not sample_id:
        return "missing_sample_id"
    if source_id in excluded["source"]:
        return "source_overlap_existing_test"
    if pair_id in excluded["pair"]:
        return "pair_overlap_existing_test"
    if sample_id in excluded["sample"]:
        return "sample_overlap_existing_test"
    tier = _first_text(row, "split_tier", "automatic_tier").lower()
    if tier and tier != "main":
        return f"tier_not_main:{tier}"
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _quality_number(row: dict[str, Any], key: str) -> float:
    for source in (
        row,
        row.get("quality") if isinstance(row.get("quality"), dict) else {},
        row.get("audio_delta_analysis") if isinstance(row.get("audio_delta_analysis"), dict) else {},
        row.get("final_omni_verification") if isinstance(row.get("final_omni_verification"), dict) else {},
    ):
        try:
            if key in source:
                return float(source.get(key) or 0.0)
        except (TypeError, ValueError):
            pass
    return 0.0


def _test_extension_sort_key(row: dict[str, Any], *, seed: int) -> tuple[float, float, str]:
    stable = hashlib.sha256(f"{seed}|{_record_sample_id(row)}".encode("utf-8")).hexdigest()
    return (
        -_quality_number(row, "audio_delta_strength"),
        -_quality_number(row, "video_context_strength"),
        stable,
    )


def _audit_test_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for label, extractor in (
        ("source", _record_source_id),
        ("pair", _record_pair_id),
        ("sample", _record_sample_id),
    ):
        counts = Counter(extractor(row) for row in rows)
        for value, count in counts.items():
            if not value:
                violations.append(f"missing_{label}")
            elif count > 1:
                violations.append(f"duplicate_{label}:{value}:{count}")
    unsupported = Counter(_record_subtype(row) for row in rows if _record_subtype(row) not in {"sound_event", "music"})
    violations.extend(f"unsupported_subtype:{key}:{value}" for key, value in unsupported.items())
    return {
        "row_count": len(rows),
        "unique_source_count": len({_record_source_id(row) for row in rows}),
        "unique_pair_count": len({_record_pair_id(row) for row in rows}),
        "unique_sample_count": len({_record_sample_id(row) for row in rows}),
        "violation_count": len(violations),
        "violations": violations,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    data = path.read_bytes()
    raw_lines = data.split(b"\n")
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_number == len(raw_lines) and not data.endswith(b"\n"):
                _truncate_incomplete_jsonl_tail(path, data)
                break
            raise ValueError(f"{path} line {line_number}: invalid JSON: {exc}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _truncate_incomplete_jsonl_tail(path: Path, data: bytes) -> None:
    prefix_length = data.rfind(b"\n") + 1
    tail = data[prefix_length:]
    if not tail:
        return
    backup = path.with_name(f"{path.name}.incomplete_tail.{os.getpid()}")
    backup.write_bytes(tail)
    with path.open("r+b") as handle:
        handle.truncate(prefix_length)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"failed to append JSONL record to {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare mirrored audio-video sources for Audio-CVR construction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--dataset", action="append", default=[])
    prepare.add_argument("--source-targets", action="append", default=[])
    prepare.add_argument("--exclude-overlap-with", action="append", default=[])
    prepare.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    prepare.add_argument("--min-duration-seconds", type=float, default=6.0)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--probe-workers", type=int, default=16)
    prepare.add_argument("--materialize-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    prepare.add_argument("--skip-download", action="store_true")
    prepare.add_argument("--allow-partial-downloads", action="store_true")
    prepare.add_argument("--resume", action="store_true")

    summarize = subparsers.add_parser("summarize-run")
    summarize.add_argument("--run-root", required=True)

    extend = subparsers.add_parser("extend-frozen-test")
    extend.add_argument("--existing-test", required=True)
    extend.add_argument("--candidate-path", required=True)
    extend.add_argument("--output-dir", required=True)
    extend.add_argument("--target-count", type=int, default=1000)
    extend.add_argument("--sound-event-target", type=int, default=800)
    extend.add_argument("--music-target", type=int, default=200)
    extend.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        datasets = [part.strip() for raw in args.dataset for part in str(raw).split(",") if part.strip()]
        result = prepare_mirror_sources(
            root=args.root,
            run_root=args.run_root,
            datasets=datasets,
            source_targets=parse_key_value_counts(args.source_targets),
            exclude_overlap_with=args.exclude_overlap_with,
            hf_endpoint=args.hf_endpoint,
            min_duration_seconds=args.min_duration_seconds,
            seed=args.seed,
            probe_workers=args.probe_workers,
            materialize_mode=args.materialize_mode,
            skip_download=args.skip_download,
            allow_partial_downloads=args.allow_partial_downloads,
            resume=args.resume,
        )
    elif args.command == "summarize-run":
        result = summarize_supplement_run(run_root=args.run_root)
    elif args.command == "extend-frozen-test":
        result = extend_frozen_test(
            existing_test_path=args.existing_test,
            candidate_path=args.candidate_path,
            output_dir=args.output_dir,
            target_count=args.target_count,
            sound_event_target=args.sound_event_target,
            music_target=args.music_target,
            seed=args.seed,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
