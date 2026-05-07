#!/usr/bin/env bash
set -euo pipefail

if [ -f /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
  conda activate omni_src
fi

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=${ROOT:-/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/single_source_omni_batch}
MODEL=${MODEL:-qwen3-omni}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
SOURCE_CLIPS=${SOURCE_CLIPS:-$ROOT/metadata/source_clips_all.jsonl}
WORLDSENSE_ROOT=${WORLDSENSE_ROOT:-$ROOT/raw_datasets/worldsense/_extracted}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-6}
DAILY_SOURCE_COUNT=${DAILY_SOURCE_COUNT:-5}
WORLDSENSE_SOURCE_COUNT=${WORLDSENSE_SOURCE_COUNT:-5}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-2}
MAX_ACCEPTED_PAIRS_PER_SOURCE=${MAX_ACCEPTED_PAIRS_PER_SOURCE:-3}
MAX_PROPOSALS=${MAX_PROPOSALS:-10}
ACCEPTANCE_PROFILE=${ACCEPTANCE_PROFILE:-exploration}
ANNOTATION_TIMEOUT_SECONDS=${ANNOTATION_TIMEOUT_SECONDS:-1200}
PROPOSE_TIMEOUT_SECONDS=${PROPOSE_TIMEOUT_SECONDS:-1200}
PAIR_REQUEST_TIMEOUT_SECONDS=${PAIR_REQUEST_TIMEOUT_SECONDS:-120}
RANDOM_SEED=${RANDOM_SEED:-}

usage() {
  cat <<'EOF'
Usage: run_single_source_omni_batch.sh [options]

Options:
  --root PATH
  --run-root PATH
  --model NAME
  --base-url URL
  --source-clips PATH
  --worldsense-root PATH
  --segment-seconds N
  --daily-source-count N
  --worldsense-source-count N
  --max-parallel-jobs N
  --max-accepted-pairs-per-source N
  --max-proposals N
  --acceptance-profile exploration|final
  --annotation-timeout-seconds N
  --propose-timeout-seconds N
  --pair-request-timeout-seconds N
  --random-seed N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --source-clips) SOURCE_CLIPS="$2"; shift 2 ;;
    --worldsense-root) WORLDSENSE_ROOT="$2"; shift 2 ;;
    --segment-seconds) SEGMENT_SECONDS="$2"; shift 2 ;;
    --daily-source-count) DAILY_SOURCE_COUNT="$2"; shift 2 ;;
    --worldsense-source-count) WORLDSENSE_SOURCE_COUNT="$2"; shift 2 ;;
    --max-parallel-jobs) MAX_PARALLEL_JOBS="$2"; shift 2 ;;
    --max-accepted-pairs-per-source) MAX_ACCEPTED_PAIRS_PER_SOURCE="$2"; shift 2 ;;
    --max-proposals) MAX_PROPOSALS="$2"; shift 2 ;;
    --acceptance-profile) ACCEPTANCE_PROFILE="$2"; shift 2 ;;
    --annotation-timeout-seconds) ANNOTATION_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --propose-timeout-seconds) PROPOSE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --pair-request-timeout-seconds) PAIR_REQUEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[single-source-batch] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

jsonl_row_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return
  fi
  grep -cve '^[[:space:]]*$' "$path" || true
}

probe_omni_model() {
  local models_json
  models_json=$(curl -fsS "$BASE_URL/models")
  OMNI_MODELS_JSON="$models_json" python3 - "$MODEL" <<'PY'
import json
import os
import sys

wanted = sys.argv[1]
payload = json.loads(os.environ["OMNI_MODELS_JSON"])
served = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
print("[single-source-batch] served_models=" + ",".join(served))
if wanted not in served:
    raise SystemExit(f"[single-source-batch] model {wanted!r} is not served by {served}; use vLLM registered name")
PY
}

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/sources" "$RUN_ROOT/manual_review/accepted" "$RUN_ROOT/manual_review/diagnostic"
echo "[single-source-batch] start $(date)"
echo "[single-source-batch] run_root=$RUN_ROOT model=$MODEL base_url=$BASE_URL segment_seconds=$SEGMENT_SECONDS"
probe_omni_model

BATCH_SOURCE_MANIFEST="$RUN_ROOT/batch_source_manifest.jsonl"
BATCH_STATUS="$RUN_ROOT/batch_status.jsonl"
BATCH_RANKED="$RUN_ROOT/batch_ranked_pairs.jsonl"
BATCH_ACCEPTED="$RUN_ROOT/batch_accepted_pairs.jsonl"
BATCH_SUMMARY="$RUN_ROOT/batch_summary.md"

ROOT="$ROOT" \
SOURCE_CLIPS="$SOURCE_CLIPS" \
WORLDSENSE_ROOT="$WORLDSENSE_ROOT" \
DAILY_SOURCE_COUNT="$DAILY_SOURCE_COUNT" \
WORLDSENSE_SOURCE_COUNT="$WORLDSENSE_SOURCE_COUNT" \
RANDOM_SEED="$RANDOM_SEED" \
BATCH_SOURCE_MANIFEST="$BATCH_SOURCE_MANIFEST" \
python3 - <<'PY'
import json
import os
import random
from pathlib import Path

from app.composed_data import _display_source_path, probe_media

root = Path(os.environ["ROOT"])
source_clips = Path(os.environ["SOURCE_CLIPS"])
worldsense_root = Path(os.environ["WORLDSENSE_ROOT"])
daily_count = int(os.environ["DAILY_SOURCE_COUNT"])
world_count = int(os.environ["WORLDSENSE_SOURCE_COUNT"])
seed = os.environ.get("RANDOM_SEED")
rng = random.Random(int(seed)) if seed not in (None, "") else random.Random(20260508)
output = Path(os.environ["BATCH_SOURCE_MANIFEST"])


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")[:120] or "source"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_path_from_row(row: dict) -> Path:
    raw = str(row.get("source_path") or row.get("output_path") or "").strip()
    if not raw:
        return root / "__missing__"
    path = Path(raw)
    return path if path.is_absolute() else root / path


def valid_media(path: Path, *, min_duration: float, max_duration=None):
    media = probe_media(path)
    duration = float(media.get("duration_seconds") or 0.0)
    if media.get("error") or not media.get("has_video") or not media.get("has_audio"):
        return False, media
    if duration < min_duration:
        return False, media
    if max_duration is not None and duration > max_duration:
        return False, media
    return True, media


daily = []
seen_daily = set()
for row in load_jsonl(source_clips):
    if row.get("dataset") != "daily_omni":
        continue
    path = source_path_from_row(row)
    display = _display_source_path(root, str(path)).replace("\\", "/")
    if "/clips/" in f"/{display}" or "raw/daily_omni/video/" not in display:
        continue
    key = str(path)
    if key in seen_daily or not path.exists():
        continue
    seen_daily.add(key)
    ok, media = valid_media(path, min_duration=28.0, max_duration=32.0)
    if not ok:
        continue
    source_id = str(row.get("clip_id") or path.stem)
    daily.append(
        {
            "job_id": safe_id(f"daily_omni_{source_id}"),
            "source_clip_id": source_id,
            "dataset": "daily_omni",
            "source_path": str(path),
            "duration_seconds": media["duration_seconds"],
            "source_window_start_seconds": 0.0,
            "source_window_duration_seconds": 30.0,
            "media_probe": media,
            "source_row_ids": list(row.get("source_row_ids", [])),
            "text_fields": row.get("text_fields", {}),
            "selection_notes": ["daily_omni raw 30s video", "6s single-source batch"],
        }
    )

world = []
world_paths = sorted(worldsense_root.glob("videos_chunk_*/videos/*.mp4"))
rng.shuffle(world_paths)
for path in world_paths:
    if len(world) >= world_count:
        break
    if not path.exists():
        continue
    ok, media = valid_media(path, min_duration=30.0)
    if not ok:
        continue
    duration = float(media.get("duration_seconds") or 0.0)
    window_start = min(30.0, max(0.0, duration * 0.25))
    if window_start + 30.0 > duration:
        window_start = max(0.0, duration - 30.0)
    source_id = f"worldsense_{path.stem}"
    world.append(
        {
            "job_id": safe_id(source_id),
            "source_clip_id": source_id,
            "dataset": "worldsense",
            "source_path": str(path),
            "duration_seconds": round(duration, 3),
            "source_window_start_seconds": round(window_start, 3),
            "source_window_duration_seconds": 30.0,
            "media_probe": media,
            "source_row_ids": [],
            "text_fields": {},
            "selection_notes": ["worldsense extracted long video", "30s window sampled for 6s single-source batch"],
        }
    )

rng.shuffle(daily)
selected = daily[:daily_count] + world[:world_count]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"daily_selected": len(daily[:daily_count]), "worldsense_selected": len(world[:world_count]), "output": str(output)}, ensure_ascii=False))
PY

SOURCE_COUNT=$(jsonl_row_count "$BATCH_SOURCE_MANIFEST")
if [ "$SOURCE_COUNT" -eq 0 ]; then
  echo "[single-source-batch] ERROR: no eligible sources selected" >&2
  exit 2
fi
echo "[single-source-batch] selected_sources=$SOURCE_COUNT manifest=$BATCH_SOURCE_MANIFEST"

run_one_source() {
  local line_number="$1"
  local selected_json
  selected_json=$(python3 - "$BATCH_SOURCE_MANIFEST" "$line_number" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
line_number = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(json.dumps(rows[line_number - 1], ensure_ascii=False))
PY
)
  local job_id
  local dataset
  job_id=$(SELECTED_JSON="$selected_json" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SELECTED_JSON"])["job_id"])
PY
)
  dataset=$(SELECTED_JSON="$selected_json" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SELECTED_JSON"])["dataset"])
PY
)
  local job_root="$RUN_ROOT/sources/$job_id"
  mkdir -p "$job_root/logs"
  SELECTED_JSON="$selected_json" JOB_ROOT="$job_root" python3 - <<'PY'
import json
import os
from pathlib import Path

job_root = Path(os.environ["JOB_ROOT"])
selected = json.loads(os.environ["SELECTED_JSON"])
(job_root / "selected_source_video.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  echo "[single-source-batch] job start line=$line_number job_id=$job_id dataset=$dataset"
  local status=0
  set +e
  bash scripts/run_single_source_omni_pair_pilot.sh \
    --root "$ROOT" \
    --run-root "$job_root" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --dataset "$dataset" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --concurrency 1 \
    --max-accepted-pairs "$MAX_ACCEPTED_PAIRS_PER_SOURCE" \
    --max-proposals "$MAX_PROPOSALS" \
    --zero-accepted-stop-after 0 \
    --annotation-timeout-seconds "$ANNOTATION_TIMEOUT_SECONDS" \
    --propose-timeout-seconds "$PROPOSE_TIMEOUT_SECONDS" \
    --pair-request-timeout-seconds "$PAIR_REQUEST_TIMEOUT_SECONDS" \
    --acceptance-profile "$ACCEPTANCE_PROFILE" \
    --start-stage plan \
    2>&1 | sed -u "s/^/[single-source-batch][$job_id] /" | tee "$job_root/logs/job.log"
  status=${PIPESTATUS[0]}
  set -e
  JOB_ROOT="$job_root" SELECTED_JSON="$selected_json" STATUS="$status" python3 - <<'PY'
import json
import os
from pathlib import Path

job_root = Path(os.environ["JOB_ROOT"])
selected = json.loads(os.environ["SELECTED_JSON"])
status = int(os.environ["STATUS"])

def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

payload = {
    "job_id": selected.get("job_id"),
    "dataset": selected.get("dataset"),
    "source_clip_id": selected.get("source_clip_id"),
    "source_path": selected.get("source_path"),
    "source_window_start_seconds": selected.get("source_window_start_seconds"),
    "source_window_duration_seconds": selected.get("source_window_duration_seconds"),
    "run_root": str(job_root),
    "exit_code": status,
    "status": "passed" if status == 0 else "failed",
    "ranked_count": count(job_root / "ranked_single_source_pairs.jsonl"),
    "accepted_count": count(job_root / "accepted_pairs.jsonl"),
    "review_bundle": str(job_root / "single_source_review_bundle"),
}
(job_root / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  echo "[single-source-batch] job done line=$line_number job_id=$job_id status=$status"
  return 0
}

for line_number in $(seq 1 "$SOURCE_COUNT"); do
  run_one_source "$line_number" &
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL_JOBS" ]; do
    sleep 5
  done
done
wait

ROOT_RUN="$RUN_ROOT" \
BATCH_STATUS="$BATCH_STATUS" \
BATCH_RANKED="$BATCH_RANKED" \
BATCH_ACCEPTED="$BATCH_ACCEPTED" \
BATCH_SUMMARY="$BATCH_SUMMARY" \
python3 - <<'PY'
import json
import shutil
from collections import Counter
from pathlib import Path

run_root = Path(__import__("os").environ["ROOT_RUN"])
status_path = Path(__import__("os").environ["BATCH_STATUS"])
ranked_path = Path(__import__("os").environ["BATCH_RANKED"])
accepted_path = Path(__import__("os").environ["BATCH_ACCEPTED"])
summary_path = Path(__import__("os").environ["BATCH_SUMMARY"])
manual_root = run_root / "manual_review"

def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

statuses = []
ranked_rows = []
accepted_rows = []
reject_buckets = Counter()
for status_file in sorted((run_root / "sources").glob("*/status.json")):
    status = json.loads(status_file.read_text(encoding="utf-8"))
    statuses.append(status)
    job_root = Path(status["run_root"])
    for row in load_jsonl(job_root / "ranked_single_source_pairs.jsonl"):
        row["batch_job_id"] = status["job_id"]
        row["batch_dataset"] = status["dataset"]
        ranked_rows.append(row)
        for issue in row.get("single_source_pair_acceptance_issues", []) or []:
            reject_buckets[str(issue).split(":", 1)[0]] += 1
    for row in load_jsonl(job_root / "accepted_pairs.jsonl"):
        row["batch_job_id"] = status["job_id"]
        row["batch_dataset"] = status["dataset"]
        accepted_rows.append(row)
    review = job_root / "single_source_review_bundle" / "pair_review"
    accepted_dir = review / "accepted"
    if accepted_dir.exists():
        for item in accepted_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, manual_root / "accepted" / f"{status['job_id']}__{item.name}", dirs_exist_ok=True)
    diagnostic_dir = review / "diagnostic"
    if diagnostic_dir.exists():
        for item in diagnostic_dir.iterdir():
            if not item.is_dir():
                continue
            metadata_path = item / "metadata.json"
            bucket = "diagnostic"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                issues = metadata.get("single_source_pair_acceptance_issues") or []
                if issues:
                    bucket = str(issues[0]).split(":", 1)[0]
                elif metadata.get("final_omni_verification", {}).get("main_reject_reason"):
                    bucket = str(metadata["final_omni_verification"]["main_reject_reason"]).split(":", 1)[0]
            shutil.copytree(item, manual_root / "diagnostic" / bucket / f"{status['job_id']}__{item.name}", dirs_exist_ok=True)

write_jsonl(status_path, statuses)
write_jsonl(ranked_path, ranked_rows)
write_jsonl(accepted_path, accepted_rows)

dataset_counts = Counter(status.get("dataset", "") for status in statuses)
accepted_by_dataset = Counter(row.get("batch_dataset", "") for row in accepted_rows)
lines = [
    "# Single-source Omni batch summary",
    "",
    f"- source_jobs: {len(statuses)}",
    f"- passed_jobs: {sum(1 for item in statuses if item.get('status') == 'passed')}",
    f"- failed_jobs: {sum(1 for item in statuses if item.get('status') != 'passed')}",
    f"- ranked_pairs: {len(ranked_rows)}",
    f"- accepted_pairs: {len(accepted_rows)}",
    f"- datasets: {dict(dataset_counts)}",
    f"- accepted_by_dataset: {dict(accepted_by_dataset)}",
    f"- top_reject_buckets: {dict(reject_buckets.most_common(12))}",
    "",
    "## Paths",
    "",
    f"- batch_status: `{status_path}`",
    f"- batch_ranked_pairs: `{ranked_path}`",
    f"- batch_accepted_pairs: `{accepted_path}`",
    f"- manual_review: `{manual_root}`",
]
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"status": str(status_path), "ranked": len(ranked_rows), "accepted": len(accepted_rows), "summary": str(summary_path)}, ensure_ascii=False))
PY

echo "[single-source-batch] done $(date)"
echo "[single-source-batch] summary=$BATCH_SUMMARY"
