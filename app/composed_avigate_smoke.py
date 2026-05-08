from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib import request

import numpy as np

from app.avigate_agent import run_official_agent_partial_eval
from app.avigate_official import AvigateRuntimeConfig, load_avigate_runtime
from app.omni_checker import OpenAIOmniChecker
from app.retrieval_types import parse_topk_values


DEFAULT_DATASET_ROOT = "/data02/usr/wangqihao/Demo/test/data"
DEFAULT_OUTPUT_ROOT = "/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs"
DEFAULT_AVIGATE_MODEL_DIR = (
    "/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/avigate/"
    "ckpt_msrvtt_paper_like_4gpu_stable"
)
DEFAULT_AVIGATE_CHECKPOINT = f"{DEFAULT_AVIGATE_MODEL_DIR}/pytorch_model.bin.4"
DEFAULT_CLIP_WEIGHT = "/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/clip/ViT-B-32.pt"
DEFAULT_CHECKER_BASE_URL = "http://127.0.0.1:8092/v1"


@dataclass(frozen=True)
class ComposedTriplet:
    sample_id: str
    sample_dir: str
    reference_video: str
    target_video: str
    edit_text: str
    reference_caption: str
    query_text: str
    source: str
    difference_type: str
    accepted: bool | None
    target_clip_id: str


def load_composed_triplets(dataset_root: str | Path, *, sample_size: int = 20, start_index: int = 0) -> list[ComposedTriplet]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    selected_dirs = sample_dirs[start_index : start_index + sample_size]
    if len(selected_dirs) < sample_size:
        raise ValueError(f"requested {sample_size} samples from {root}, found {len(selected_dirs)}")
    return [_read_triplet(sample_dir) for sample_dir in selected_dirs]


def stage_triplets(
    triplets: list[ComposedTriplet],
    *,
    staged_root: str | Path,
    ffmpeg: str = "ffmpeg",
    extract_audio: bool = True,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    root = Path(staged_root)
    video_root = root / "video_root"
    audio_root = root / "audio_root"
    video_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    audio_failures: list[dict[str, str]] = []
    materialized_videos = 0
    materialized_audios = 0
    for triplet in triplets:
        video_dst = video_root / f"{triplet.sample_id}.mp4"
        if _materialize_file(Path(triplet.target_video), video_dst, mode=link_mode):
            materialized_videos += 1

        audio_dst = audio_root / f"{triplet.sample_id}.wav"
        if extract_audio:
            extracted, error = _extract_audio(Path(triplet.target_video), audio_dst, ffmpeg=ffmpeg)
            if extracted:
                materialized_audios += 1
            elif error:
                audio_failures.append({"sample_id": triplet.sample_id, "error": error})

    split_text = _build_split_csv(triplets)
    triplets_text = "\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in triplets) + "\n"
    data_json_text = "{}\n"
    report = {
        "sample_count": len(triplets),
        "staged_root": str(root),
        "split_csv": str(root / "split.csv"),
        "data_json": str(root / "data.json"),
        "video_root": str(video_root),
        "audio_root": str(audio_root),
        "materialized_videos": materialized_videos,
        "materialized_audios": materialized_audios,
        "audio_failures": audio_failures,
        "link_mode": link_mode,
    }

    _write_text_if_changed(root / "split.csv", split_text)
    _write_text_if_changed(root / "triplets.jsonl", triplets_text)
    _write_text_if_changed(root / "data.json", data_json_text)
    _write_text_if_changed(root / "staging_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def run_baseline(runtime: Any, *, recall_ks: tuple[int, ...], topk: int, output_dir: str | Path) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    max_rank = max(max(recall_ks), int(topk))
    video_index = {row.video_id: index for index, row in enumerate(runtime.video_rows)}

    hit_counts = {k: 0 for k in recall_ks}
    trace_lines: list[str] = []
    for row in runtime.text_rows:
        scores = np.asarray(runtime.score_text_query(row.text), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        target_index = video_index[row.video_id]
        target_rank = int(np.where(order == target_index)[0][0]) + 1
        for k in recall_ks:
            if target_rank <= k:
                hit_counts[k] += 1

        top_hits = []
        for rank, index in enumerate(order[:max_rank], start=1):
            video_row = runtime.video_rows[int(index)]
            top_hits.append(
                {
                    "rank": rank,
                    "video_id": video_row.video_id,
                    "score": round(float(scores[index]), 6),
                    "video_path": video_row.video_path,
                }
            )
        trace_lines.append(
            json.dumps(
                {
                    "sample_id": row.video_id,
                    "query_text": row.text,
                    "target_video_id": row.video_id,
                    "target_rank": target_rank,
                    "topk_hits": top_hits,
                },
                ensure_ascii=False,
            )
        )

    runs = len(runtime.text_rows)
    summary = {
        "mode": "composed-avigate-baseline",
        "runs": runs,
        "video_count": len(runtime.video_rows),
        "text_count": len(runtime.text_rows),
        "audio_available": bool(runtime.audio_available),
        "t2v": {f"R@{k}": round(hit_counts[k] / max(1, runs), 4) for k in recall_ks},
        "traces_path": str(output_root / "baseline_traces.jsonl"),
    }
    _write_text_if_changed(output_root / "baseline_traces.jsonl", "\n".join(trace_lines) + ("\n" if trace_lines else ""))
    _write_text_if_changed(output_root / "baseline_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def run_agent(
    runtime: Any,
    *,
    checker_base_url: str,
    checker_api_key: str,
    checker_model: str,
    checker_timeout_seconds: float,
    recall_ks: tuple[int, ...],
    topk: int,
    omni_concurrency: int,
    rerank_window: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    checker = OpenAIOmniChecker(
        base_url=checker_base_url,
        api_key=checker_api_key,
        model=checker_model,
        timeout_seconds=checker_timeout_seconds,
    )
    result = run_official_agent_partial_eval(
        mode="t2v",
        runtime=runtime,
        checker=checker,
        sample_size=len(runtime.text_rows),
        topk=topk,
        omni_concurrency=omni_concurrency,
        rerank_window=rerank_window,
        recall_ks=recall_ks,
        output_dir=str(output_dir),
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    return dict(result["summary"])


def write_comparison(
    *,
    run_root: str | Path,
    staged_root: str | Path,
    baseline_summary: dict[str, Any],
    agent_summary: dict[str, Any] | None,
    checker_model: str | None,
) -> dict[str, Any]:
    root = Path(run_root)
    baseline = baseline_summary.get("t2v", {})
    agent_round1 = (agent_summary or {}).get("round1_recall", {})
    agent_final = (agent_summary or {}).get("final_recall", {})
    comparison = {
        "run_root": str(root),
        "staged_root": str(staged_root),
        "checker_model": checker_model,
        "sample_count": int(baseline_summary.get("runs", 0)),
        "rows": [
            {"method": "AVIGATE baseline", **_metric_row(baseline)},
            {"method": "AVIGATE round1 in agent", **_metric_row(agent_round1)},
            {"method": "AVIGATE+Qwen2.5-Omni Agent", **_metric_row(agent_final)},
        ],
    }
    _write_text_if_changed(root / "comparison.json", json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    _write_text_if_changed(root / "comparison.md", _comparison_markdown(comparison))
    return comparison


def build_runtime_from_staging(args: argparse.Namespace, staging_report: dict[str, Any]) -> Any:
    config = AvigateRuntimeConfig(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        data_json_path=str(staging_report["data_json"]),
        test_csv_path=str(staging_report["split_csv"]),
        video_root=str(staging_report["video_root"]),
        audio_root=str(staging_report["audio_root"]),
        clip_weight_path=args.clip_weight,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size_val=args.batch_size_val,
        max_words=args.max_words,
        max_frames=args.max_frames,
        sim_header=args.sim_header,
        cross_num_hidden_layers=args.cross_num_hidden_layers,
        audio_query_layers=args.audio_query_layers,
        temperature=args.temperature,
    )
    return load_avigate_runtime(config)


def resolve_checker_model(base_url: str, api_key: str) -> str:
    url = f"{base_url.rstrip('/')}/models"
    req = request.Request(url, headers=_checker_headers(api_key))
    with request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_id = _first_model_id(payload)
    if not model_id:
        raise RuntimeError(f"no model id found in {url} response")
    return model_id


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root or _default_run_root(args.sample_size))
    run_root.mkdir(parents=True, exist_ok=True)
    staged_root = Path(args.staged_root) if args.staged_root else run_root / "staged"

    triplets = load_composed_triplets(args.dataset_root, sample_size=args.sample_size, start_index=args.start_index)
    staging_report = stage_triplets(
        triplets,
        staged_root=staged_root,
        ffmpeg=args.ffmpeg,
        extract_audio=not args.skip_audio_extract,
        link_mode=args.link_mode,
    )

    runtime = build_runtime_from_staging(args, staging_report)
    recall_ks = tuple(parse_topk_values(args.topk))
    baseline_summary = run_baseline(runtime, recall_ks=recall_ks, topk=args.topk_value, output_dir=run_root)

    agent_summary: dict[str, Any] | None = None
    checker_model: str | None = None
    if not args.skip_agent:
        checker_model = args.checker_model or resolve_checker_model(args.checker_base_url, args.checker_api_key)
        agent_summary = run_agent(
            runtime,
            checker_base_url=args.checker_base_url,
            checker_api_key=args.checker_api_key,
            checker_model=checker_model,
            checker_timeout_seconds=args.checker_timeout_seconds,
            recall_ks=recall_ks,
            topk=args.topk_value,
            omni_concurrency=args.omni_concurrency,
            rerank_window=args.rerank_window,
            output_dir=run_root / "agent",
        )

    comparison = write_comparison(
        run_root=run_root,
        staged_root=staged_root,
        baseline_summary=baseline_summary,
        agent_summary=agent_summary,
        checker_model=checker_model,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a composed AVIGATE vs AVIGATE+Omni evaluation slice")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--run-root")
    parser.add_argument("--staged-root")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--skip-audio-extract", action="store_true")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")

    parser.add_argument("--model-dir", default=DEFAULT_AVIGATE_MODEL_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_AVIGATE_CHECKPOINT)
    parser.add_argument("--clip-weight", default=DEFAULT_CLIP_WEIGHT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-val", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--sim-header", default="seqTransf")
    parser.add_argument("--cross-num-hidden-layers", type=int, default=4)
    parser.add_argument("--audio-query-layers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", default="1,5,10")
    parser.add_argument("--topk-value", type=int, default=10)

    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--checker-base-url", default=DEFAULT_CHECKER_BASE_URL)
    parser.add_argument("--checker-api-key", default="EMPTY")
    parser.add_argument("--checker-model")
    parser.add_argument("--checker-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--omni-concurrency", type=int, default=2)
    parser.add_argument("--rerank-window", type=int, default=5)
    return parser


def main() -> None:
    run_smoke(build_parser().parse_args())


def _read_triplet(sample_dir: Path) -> ComposedTriplet:
    reference_video = sample_dir / "reference.mp4"
    target_video = sample_dir / "target.mp4"
    edit_text_path = sample_dir / "edit_text.txt"
    info_path = sample_dir / "info.json"
    for path in (reference_video, target_video, edit_text_path, info_path):
        if not path.exists():
            raise FileNotFoundError(f"required sample file missing: {path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    edit_text = edit_text_path.read_text(encoding="utf-8").strip()
    reference_caption = str(info.get("reference_caption", "")).strip()
    if not reference_caption:
        annotation_path = sample_dir / "reference_annotation.json"
        if annotation_path.exists():
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            reference_caption = str(annotation.get("summary", "")).strip()
    if not reference_caption:
        raise ValueError(f"missing reference_caption for sample {sample_dir.name}")
    if not edit_text:
        raise ValueError(f"missing edit_text for sample {sample_dir.name}")

    return ComposedTriplet(
        sample_id=sample_dir.name,
        sample_dir=str(sample_dir),
        reference_video=str(reference_video),
        target_video=str(target_video),
        edit_text=edit_text,
        reference_caption=reference_caption,
        query_text=_compose_query(reference_caption, edit_text),
        source=str(info.get("source", "")),
        difference_type=str(info.get("difference_type", "")),
        accepted=info.get("accepted") if isinstance(info.get("accepted"), bool) else None,
        target_clip_id=str(info.get("target_clip_id", "")),
    )


def _compose_query(reference_caption: str, edit_text: str) -> str:
    caption = reference_caption.strip()
    if caption and caption[-1] not in ".!?":
        caption = f"{caption}."
    edit = edit_text.strip().rstrip(".")
    return f"{caption} Edit: {edit}."


def _build_split_csv(triplets: list[ComposedTriplet]) -> str:
    rows = ["video_id,sentence"]
    for triplet in triplets:
        rows.append(f"{_csv_cell(triplet.sample_id)},{_csv_cell(triplet.query_text)}")
    return "\n".join(rows) + "\n"


def _csv_cell(value: str) -> str:
    if any(char in value for char in [",", '"', "\n", "\r"]):
        return '"' + value.replace('"', '""') + '"'
    return value


def _materialize_file(src: Path, dst: Path, *, mode: str) -> bool:
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if _materialized_matches(src, dst):
            return False
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return True
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return True
        except OSError:
            shutil.copy2(src, dst)
            return True
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return True


def _materialized_matches(src: Path, dst: Path) -> bool:
    try:
        if dst.is_symlink():
            return dst.resolve() == src
        if dst.stat().st_size == src.stat().st_size:
            return True
    except OSError:
        return False
    return False


def _extract_audio(video_path: Path, audio_path: Path, *, ffmpeg: str) -> tuple[bool, str | None]:
    if audio_path.exists() and audio_path.stat().st_size > 0:
        return False, None
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return False, str(exc)
    if completed.returncode != 0:
        if audio_path.exists():
            audio_path.unlink()
        message = completed.stderr.strip() or completed.stdout.strip() or f"ffmpeg exited {completed.returncode}"
        return False, message[-500:]
    return True, None


def _write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ("R@1", "R@5", "R@10")}


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Composed AVIGATE Comparison",
        "",
        f"- run_root: `{comparison['run_root']}`",
        f"- staged_root: `{comparison['staged_root']}`",
        f"- checker_model: `{comparison.get('checker_model') or 'skipped'}`",
        f"- sample_count: `{comparison['sample_count']}`",
        "",
        "| Method | R@1 | R@5 | R@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(f"| {row['method']} | {_fmt_metric(row.get('R@1'))} | {_fmt_metric(row.get('R@5'))} | {_fmt_metric(row.get('R@10'))} |")
    return "\n".join(lines) + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _checker_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _first_model_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                model_id = _first_model_id(item)
                if model_id:
                    return model_id
        if payload.get("id"):
            return str(payload["id"])
    if isinstance(payload, str):
        return payload
    return None


def _default_run_root(sample_size: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{DEFAULT_OUTPUT_ROOT}/composed_avigate_eval{sample_size}_{stamp}"


if __name__ == "__main__":
    main()
