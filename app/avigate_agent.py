from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable

from app.avigate_official import (
    retrieve_texts_from_video_official,
    retrieve_videos_from_text_official,
)
from app.omni_checker import OmniChecker, RetrievalHints
from app.retrieval_types import RetrievalHit


def run_official_agent_partial_eval(
    *,
    mode: str,
    runtime: Any,
    checker: OmniChecker,
    sample_size: int,
    start_index: int = 0,
    topk: int = 10,
    omni_concurrency: int = 4,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
    rerank_window: int = 5,
    recall_ks: tuple[int, ...] = (1, 5, 10),
    output_dir: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    _ = max_iter, submit_threshold
    if mode not in {"t2v", "v2t"}:
        raise ValueError("mode must be 't2v' or 'v2t'")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    recall_ks = tuple(sorted({int(k) for k in recall_ks if int(k) > 0}))
    if not recall_ks:
        raise ValueError("recall_ks must contain at least one positive integer")

    rows = runtime.text_rows if mode == "t2v" else runtime.video_rows
    selected_rows = rows[start_index : start_index + sample_size]
    total = len(selected_rows)
    if total <= 0:
        raise ValueError("no rows available for partial eval")

    summary_path: Path | None = None
    traces_path: Path | None = None
    if output_dir:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "summary.json"
        traces_path = output_root / "traces.jsonl"
        traces_path.write_text("", encoding="utf-8")

    round1_hits = {k: 0 for k in recall_ks}
    final_hits = {k: 0 for k in recall_ks}
    final_top1_correct = 0
    total_omni_calls = 0
    audio_off_runs = 0
    fallback_runs = 0
    query_rewrite_runs = 0

    summary: dict[str, Any] = {}
    for run_index, row in enumerate(selected_rows, start=1):
        label = row.text if mode == "t2v" else row.video_id
        _emit_progress(progress, f"[{mode}] start {run_index}/{total}: {label}")

        if mode == "t2v":
            trace = run_t2v_official_agent_case(
                query_text=row.text,
                runtime=runtime,
                checker=checker,
                target_video_id=row.video_id,
                topk=topk,
                omni_concurrency=omni_concurrency,
                rerank_window=rerank_window,
                progress=progress,
            )
            if trace["retrieval_hints"].get("query_text_override"):
                query_rewrite_runs += 1
            _update_t2v_recall_counts(
                trace=trace,
                target_video_id=row.video_id,
                ks=recall_ks,
                round1_hits=round1_hits,
                final_hits=final_hits,
            )
            if trace["final_result"].get("video_id") == row.video_id:
                final_top1_correct += 1
        else:
            trace = run_v2t_official_agent_case(
                query_video_id=row.video_id,
                runtime=runtime,
                checker=checker,
                topk=topk,
                progress=progress,
            )
            _update_v2t_recall_counts(
                trace=trace,
                target_text_ids=set(runtime.target_text_ids(row.video_id)),
                ks=recall_ks,
                round1_hits=round1_hits,
                final_hits=final_hits,
            )
            if trace["final_result"].get("text_id") in set(runtime.target_text_ids(row.video_id)):
                final_top1_correct += 1

        total_omni_calls += int(trace["omni_calls"])
        if trace["retrieval_hints"].get("audio_mode") == "off":
            audio_off_runs += 1
        if trace.get("fallback_used"):
            fallback_runs += 1

        if traces_path is not None:
            with traces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

        summary = _build_partial_eval_summary(
            mode=mode,
            runs=run_index,
            recall_ks=recall_ks,
            round1_hits=round1_hits,
            final_hits=final_hits,
            final_top1_correct=final_top1_correct,
            total_omni_calls=total_omni_calls,
            audio_off_runs=audio_off_runs,
            fallback_runs=fallback_runs,
            query_rewrite_runs=query_rewrite_runs,
        )
        if summary_path is not None:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        _emit_progress(
            progress,
            f"[{mode}] done {run_index}/{total}: top1={trace['final_result']['item_id']} omni_calls={trace['omni_calls']}",
        )

    result = {"summary": summary}
    if summary_path is not None:
        result["summary_path"] = str(summary_path)
    if traces_path is not None:
        result["traces_path"] = str(traces_path)
    return result


def run_t2v_official_agent_case(
    *,
    query_text: str,
    runtime: Any,
    checker: OmniChecker,
    target_video_id: str | None = None,
    topk: int = 10,
    omni_concurrency: int = 4,
    rerank_window: int = 5,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
    progress: Callable[[str], None] | None = None,
) -> dict:
    _ = max_iter, submit_threshold
    _emit_progress(progress, f"[t2v] understand query={query_text}")
    query_understanding = checker.understand_t2v_query(query_text)
    retrieval_hints = RetrievalHints.from_query_understanding(query_text, query_understanding)
    effective_query = retrieval_hints.query_text_override or query_text
    initial_hits = retrieve_videos_from_text_official(
        effective_query,
        runtime,
        topk=topk,
        audio_mode=retrieval_hints.audio_mode,
    )

    omni_calls = 1
    fallback_used = query_understanding.fallback_used
    fallback_stage = "query_understanding" if fallback_used else None
    candidate_video_descriptions: list[dict] = []
    reranked_hits = _clone_hits(initial_hits)

    if not fallback_used:
        window = min(max(1, int(rerank_window)), len(initial_hits))
        candidate_video_descriptions = _describe_candidate_videos(
            hits=initial_hits[:window],
            runtime=runtime,
            checker=checker,
            omni_concurrency=omni_concurrency,
            progress=progress,
        )
        omni_calls += len(candidate_video_descriptions)
        if any(item["video_description"].get("fallback_used") for item in candidate_video_descriptions):
            fallback_used = True
            fallback_stage = "candidate_video_description"

        if not fallback_used and candidate_video_descriptions:
            candidate_payloads = [
                {
                    "video_id": item["candidate"]["video_id"],
                    "original_rank": item["rank"],
                    "original_score": item["candidate"]["score"],
                    "video_description": item["video_description"],
                }
                for item in candidate_video_descriptions
            ]
            _emit_progress(progress, "[t2v] rerank candidate videos")
            rerank_result = checker.rerank_t2v(query_understanding, candidate_payloads)
            omni_calls += 1
            reranked_hits, repair_used = _rerank_hits(
                initial_hits,
                rerank_result.ordered_video_ids,
                key_name="video_id",
                window=window,
            )
            if rerank_result.fallback_used or repair_used:
                fallback_used = True
                fallback_stage = "t2v_rerank"
                reranked_hits = _clone_hits(initial_hits)

    final_result = _build_final_result(reranked_hits, initial_hits, key_name="video_id")
    return {
        "mode": "t2v-agent",
        "query_text": query_text,
        "target_video_id": target_video_id,
        "query_understanding": query_understanding.to_dict(),
        "retrieval_hints": retrieval_hints.to_dict(),
        "initial_hits": [hit.to_dict() for hit in initial_hits],
        "candidate_video_descriptions": candidate_video_descriptions,
        "reranked_hits": [hit.to_dict() for hit in reranked_hits],
        "final_result": final_result,
        "omni_calls": omni_calls,
        "fallback_used": fallback_used,
        "fallback_stage": fallback_stage,
    }


def run_v2t_official_agent_case(
    *,
    query_video_id: str,
    runtime: Any,
    checker: OmniChecker,
    topk: int = 10,
    max_iter: int = 3,
    submit_threshold: float = 0.7,
    min_inspected_before_submit: int = 6,
    progress: Callable[[str], None] | None = None,
) -> dict:
    _ = max_iter, submit_threshold, min_inspected_before_submit
    query_video = runtime.video_rows[runtime._video_index[query_video_id]]
    _emit_progress(progress, f"[v2t] describe video_id={query_video_id}")
    video_description = checker.describe_video(query_video)
    retrieval_hints = RetrievalHints.from_video_description(video_description)
    initial_hits = retrieve_texts_from_video_official(
        query_video_id,
        runtime,
        topk=topk,
        audio_mode=retrieval_hints.audio_mode,
    )

    omni_calls = 1
    fallback_used = video_description.fallback_used
    fallback_stage = "video_description" if fallback_used else None
    reranked_hits = _clone_hits(initial_hits)

    if not fallback_used and initial_hits:
        candidate_payloads = [
            {
                "text_id": hit.text_id,
                "original_rank": hit.rank,
                "original_score": hit.score,
                "text": hit.text,
            }
            for hit in initial_hits
        ]
        _emit_progress(progress, f"[v2t] rerank {len(candidate_payloads)} candidate texts")
        rerank_result = checker.rerank_v2t(query_video, video_description, candidate_payloads)
        omni_calls += 1
        reranked_hits, repair_used = _rerank_hits(initial_hits, rerank_result.ordered_text_ids, key_name="text_id")
        if rerank_result.fallback_used or repair_used:
            fallback_used = True
            fallback_stage = "v2t_rerank"
            reranked_hits = _clone_hits(initial_hits)

    final_result = _build_final_result(reranked_hits, initial_hits, key_name="text_id")
    return {
        "mode": "v2t-agent",
        "query_video_id": query_video_id,
        "query_video_path": query_video.video_path,
        "video_description": video_description.to_dict(),
        "retrieval_hints": retrieval_hints.to_dict(),
        "initial_hits": [hit.to_dict() for hit in initial_hits],
        "reranked_hits": [hit.to_dict() for hit in reranked_hits],
        "final_result": final_result,
        "omni_calls": omni_calls,
        "fallback_used": fallback_used,
        "fallback_stage": fallback_stage,
    }


def _describe_candidate_videos(
    *,
    hits: list[RetrievalHit],
    runtime: Any,
    checker: OmniChecker,
    omni_concurrency: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    indexed_jobs = []
    for index, hit in enumerate(hits):
        candidate_video = runtime.video_rows[runtime._video_index[hit.video_id or ""]]
        indexed_jobs.append((index, hit, candidate_video))

    max_workers = max(1, min(int(omni_concurrency), len(indexed_jobs)))
    if max_workers == 1:
        described = [_describe_one_candidate(index, hit, video, checker, progress) for index, hit, video in indexed_jobs]
        for item in described:
            item.pop("_order", None)
        return described

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_describe_one_candidate, index, hit, video, checker, progress)
            for index, hit, video in indexed_jobs
        ]
    described = [future.result() for future in futures]
    described.sort(key=lambda item: item["_order"])
    for item in described:
        item.pop("_order", None)
    return described


def _describe_one_candidate(
    index: int,
    hit: RetrievalHit,
    candidate_video: Any,
    checker: OmniChecker,
    progress: Callable[[str], None] | None,
) -> dict:
    _emit_progress(progress, f"[t2v] describe rank={hit.rank} video_id={candidate_video.video_id}")
    description = checker.describe_video(candidate_video)
    return {
        "_order": index,
        "rank": hit.rank,
        "candidate": hit.to_dict(),
        "video_description": description.to_dict(),
    }


def _build_partial_eval_summary(
    *,
    mode: str,
    runs: int,
    recall_ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
    final_top1_correct: int,
    total_omni_calls: int,
    audio_off_runs: int,
    fallback_runs: int,
    query_rewrite_runs: int,
) -> dict:
    summary = {
        "runs": runs,
        "round1_recall": {f"R@{k}": round(round1_hits[k] / runs, 4) for k in recall_ks},
        "final_recall": {f"R@{k}": round(final_hits[k] / runs, 4) for k in recall_ks},
        "final_top1_accuracy": round(final_top1_correct / runs, 4),
        "avg_omni_calls": round(total_omni_calls / runs, 4),
        "audio_off_rate": round(audio_off_runs / runs, 4),
        "fallback_rate": round(fallback_runs / runs, 4),
        "mode": mode,
    }
    if mode == "t2v":
        summary["query_rewrite_rate"] = round(query_rewrite_runs / runs, 4)
    return summary


def _update_t2v_recall_counts(
    *,
    trace: dict,
    target_video_id: str,
    ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
) -> None:
    _update_recall_counts(
        initial_hits=trace["initial_hits"],
        reranked_hits=trace["reranked_hits"],
        ks=ks,
        round1_hits=round1_hits,
        final_hits=final_hits,
        predicate=lambda hit: hit.get("video_id") == target_video_id,
    )


def _update_v2t_recall_counts(
    *,
    trace: dict,
    target_text_ids: set[str],
    ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
) -> None:
    _update_recall_counts(
        initial_hits=trace["initial_hits"],
        reranked_hits=trace["reranked_hits"],
        ks=ks,
        round1_hits=round1_hits,
        final_hits=final_hits,
        predicate=lambda hit: hit.get("text_id") in target_text_ids,
    )


def _update_recall_counts(
    *,
    initial_hits: list[dict],
    reranked_hits: list[dict],
    ks: tuple[int, ...],
    round1_hits: dict[int, int],
    final_hits: dict[int, int],
    predicate: Callable[[dict], bool],
) -> None:
    for k in ks:
        if any(predicate(hit) for hit in initial_hits[:k]):
            round1_hits[k] += 1
        if any(predicate(hit) for hit in reranked_hits[:k]):
            final_hits[k] += 1


def _build_final_result(
    reranked_hits: list[RetrievalHit],
    initial_hits: list[RetrievalHit],
    *,
    key_name: str,
) -> dict:
    if not reranked_hits:
        return {"item_id": None, "rank_in_final_search": 0, "original_rank": 0}
    original_rank_by_id = {
        getattr(hit, key_name): hit.rank
        for hit in initial_hits
        if getattr(hit, key_name) is not None
    }
    top_hit = reranked_hits[0]
    key_value = getattr(top_hit, key_name)
    payload = top_hit.to_dict()
    payload["rank_in_final_search"] = 1
    payload["original_rank"] = int(original_rank_by_id.get(key_value, top_hit.rank))
    return payload


def _rerank_hits(
    hits: list[RetrievalHit],
    ordered_ids: list[str],
    *,
    key_name: str,
    window: int | None = None,
) -> tuple[list[RetrievalHit], bool]:
    if not hits:
        return [], False
    prefix = list(hits[:window]) if window is not None else list(hits)
    suffix = list(hits[window:]) if window is not None else []
    key_to_hit = {
        str(getattr(hit, key_name)): hit
        for hit in prefix
        if getattr(hit, key_name) is not None
    }
    ordered: list[RetrievalHit] = []
    seen: set[str] = set()
    repair_used = False

    for item_id in ordered_ids:
        normalized_id = str(item_id).strip()
        if normalized_id not in key_to_hit or normalized_id in seen:
            repair_used = True
            continue
        ordered.append(key_to_hit[normalized_id])
        seen.add(normalized_id)

    if len(seen) != len(prefix):
        repair_used = True
    for hit in prefix:
        key_value = str(getattr(hit, key_name))
        if key_value not in seen:
            ordered.append(hit)
            seen.add(key_value)

    reranked = ordered + suffix
    return _with_ranks(reranked), repair_used


def _clone_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    return _with_ranks(list(hits))


def _with_ranks(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    reranked: list[RetrievalHit] = []
    for rank, hit in enumerate(hits, start=1):
        reranked.append(
            RetrievalHit(
                rank=rank,
                item_id=hit.item_id,
                score=hit.score,
                video_id=hit.video_id,
                text_id=hit.text_id,
                text=hit.text,
                video_path=hit.video_path,
            )
        )
    return reranked


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
