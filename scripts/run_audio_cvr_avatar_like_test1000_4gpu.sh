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
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/runs/audio_cvr_avatar_like_test1000_$(date +%Y%m%d_%H%M%S)}
BASE_URL=${BASE_URL:-http://127.0.0.1:8093/v1}
MODEL=${MODEL:-qwen3-omni-30b-a3b-instruct}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DATASETS=${DATASETS:-existing_vggsound,avqa_videos,ave_dataset,avscapbench}
SOURCE_TARGETS=${SOURCE_TARGETS:-existing_vggsound=3500,avqa_videos=5000,ave_dataset=2500,avscapbench=1200}
CLIP_SECONDS=${CLIP_SECONDS:-8}
MIN_CLIP_SECONDS=${MIN_CLIP_SECONDS:-6}
MAX_CLIP_SECONDS=${MAX_CLIP_SECONDS:-9}
MAX_CLIPS_PER_SOURCE=${MAX_CLIPS_PER_SOURCE:-2}
MAX_B_CANDIDATES=${MAX_B_CANDIDATES:-7000}
PILOT_CANDIDATES=${PILOT_CANDIDATES:-700}
PILOT_SECOND_CANDIDATES=${PILOT_SECOND_CANDIDATES:-1400}
PILOT_MIN_TOTAL=${PILOT_MIN_TOTAL:-100}
PILOT_MIN_SOUND_EVENT=${PILOT_MIN_SOUND_EVENT:-75}
PILOT_MIN_MUSIC=${PILOT_MIN_MUSIC:-20}
PILOT_BORDERLINE_TOTAL=${PILOT_BORDERLINE_TOTAL:-85}
PILOT_BORDERLINE_SOUND_EVENT=${PILOT_BORDERLINE_SOUND_EVENT:-68}
PILOT_BORDERLINE_MUSIC=${PILOT_BORDERLINE_MUSIC:-17}
PROPOSE_SHARDS=${PROPOSE_SHARDS:-128}
PROPOSE_PARALLEL_JOBS=${PROPOSE_PARALLEL_JOBS:-24}
CONCURRENCY=${CONCURRENCY:-24}
QUALITY_PROFILE=${QUALITY_PROFILE:-b_audio_blind_review_v2_volume}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-240}
SHARD_TIMEOUT_SECONDS=${SHARD_TIMEOUT_SECONDS:-21600}
PROBE_WORKERS=${PROBE_WORKERS:-24}
RESUME=${RESUME:-0}
EXCLUDE_OVERLAP_PATHS=()
EXISTING_TEST_PATH=${EXISTING_TEST_PATH:-}
EXPECTED_EXISTING_TEST_SHA256=${EXPECTED_EXISTING_TEST_SHA256:-}

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_avatar_like_test1000_4gpu.sh [options]

Uses an already-running Qwen3-Omni service. It never starts or stops GPU services.

Options:
  --root PATH
  --run-root PATH
  --base-url URL
  --model NAME
  --hf-endpoint URL
  --datasets NAME[,NAME]
  --source-targets NAME=COUNT[,NAME=COUNT]
  --clip-seconds N
  --min-clip-seconds N
  --max-clip-seconds N
  --max-clips-per-source N
  --max-b-candidates N
  --pilot-candidates N
  --pilot-second-candidates N
  --propose-shards N
  --propose-parallel-jobs N
  --concurrency N
  --quality-profile NAME
  --exclude-overlap-with PATH (repeatable)
  --existing-test-path PATH
  --expected-existing-test-sha256 HASH
  --resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --source-targets) SOURCE_TARGETS="$2"; shift 2 ;;
    --clip-seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --min-clip-seconds) MIN_CLIP_SECONDS="$2"; shift 2 ;;
    --max-clip-seconds) MAX_CLIP_SECONDS="$2"; shift 2 ;;
    --max-clips-per-source) MAX_CLIPS_PER_SOURCE="$2"; shift 2 ;;
    --max-b-candidates) MAX_B_CANDIDATES="$2"; shift 2 ;;
    --pilot-candidates) PILOT_CANDIDATES="$2"; shift 2 ;;
    --pilot-second-candidates) PILOT_SECOND_CANDIDATES="$2"; shift 2 ;;
    --propose-shards) PROPOSE_SHARDS="$2"; shift 2 ;;
    --propose-parallel-jobs) PROPOSE_PARALLEL_JOBS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --quality-profile) QUALITY_PROFILE="$2"; shift 2 ;;
    --exclude-overlap-with) EXCLUDE_OVERLAP_PATHS+=("$2"); shift 2 ;;
    --existing-test-path) EXISTING_TEST_PATH="$2"; shift 2 ;;
    --expected-existing-test-sha256) EXPECTED_EXISTING_TEST_SHA256="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[avatar-like-test1000] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$QUALITY_PROFILE" != "b_audio_blind_review_v2" ] && [ "$QUALITY_PROFILE" != "b_audio_blind_review_v2_volume" ]; then
  echo "[avatar-like-test1000] unsupported quality profile: $QUALITY_PROFILE" >&2
  exit 2
fi
if [ -z "$EXISTING_TEST_PATH" ]; then
  echo "[avatar-like-test1000] --existing-test-path is required" >&2
  exit 2
fi
if [ ! -s "$EXISTING_TEST_PATH" ]; then
  echo "[avatar-like-test1000] existing test is missing or empty: $EXISTING_TEST_PATH" >&2
  exit 2
fi
if [ "$PILOT_CANDIDATES" -le 0 ] || [ "$PILOT_SECOND_CANDIDATES" -lt "$PILOT_CANDIDATES" ] || [ "$MAX_B_CANDIDATES" -lt "$PILOT_SECOND_CANDIDATES" ]; then
  echo "[avatar-like-test1000] require 0 < pilot <= second pilot <= max B candidates" >&2
  exit 2
fi
if [ $((PILOT_SECOND_CANDIDATES % PILOT_CANDIDATES)) -ne 0 ]; then
  echo "[avatar-like-test1000] second pilot candidate count must be an integer multiple of the first pilot" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/protocol_postprocess"
STATUS_PATH="$RUN_ROOT/status.json"
CONSTRUCTION_ROOT="$RUN_ROOT/construction_data"
SINGLE_SOURCE_ROOT="$CONSTRUCTION_ROOT/clips/audio_cvr_avatar_like_6_9s"

write_status() {
  local state="$1"
  local stage="$2"
  local detail="$3"
  STATUS_STATE="$state" STATUS_STAGE="$stage" STATUS_DETAIL="$detail" STATUS_PATH="$STATUS_PATH" \
    python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_PATH"])
payload = {
    "state": os.environ["STATUS_STATE"],
    "stage": os.environ["STATUS_STAGE"],
    "detail": os.environ["STATUS_DETAIL"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "service_owned": False,
    "gpu_ids_used": [0, 1, 2, 3],
    "gpu_ids_forbidden": [4, 5, 6, 7],
}
path.parent.mkdir(parents=True, exist_ok=True)
encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temp_path.open("w", encoding="utf-8") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temp_path, path)
history_path = path.with_name("status_history.jsonl")
with history_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

on_exit() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    write_status "FAILED" "launcher" "exit_code=$status; existing Omni service was not stopped"
  fi
  exit "$status"
}
trap on_exit EXIT

write_status "RUNNING" "preflight" "checking frozen test and existing Omni service"
curl -fsS -m 30 "$BASE_URL/models" > "$RUN_ROOT/logs/omni_models.json"
MODEL=$(python3 - "$RUN_ROOT/logs/omni_models.json" "$MODEL" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
wanted = sys.argv[2]
ids = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
if wanted in ids:
    print(wanted)
elif len(ids) == 1:
    print(ids[0])
else:
    aliases = [item for item in ids if wanted.lower() in item.lower() or item.lower() in wanted.lower()]
    if len(aliases) != 1:
        raise SystemExit(f"cannot resolve Omni model alias {wanted!r}; served models={ids}")
    print(aliases[0])
PY
)
echo "$MODEL" > "$RUN_ROOT/logs/resolved_omni_model.txt"
curl -fsS -m 90 "$BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK only."}],"max_tokens":8}' \
  > "$RUN_ROOT/logs/omni_health_chat.json"

actual_test_sha=$(sha256sum "$EXISTING_TEST_PATH" | awk '{print $1}')
if [ -n "$EXPECTED_EXISTING_TEST_SHA256" ] && [ "$actual_test_sha" != "$EXPECTED_EXISTING_TEST_SHA256" ]; then
  echo "[avatar-like-test1000] frozen test SHA mismatch expected=$EXPECTED_EXISTING_TEST_SHA256 actual=$actual_test_sha" >&2
  exit 2
fi
printf '%s  %s\n' "$actual_test_sha" "$EXISTING_TEST_PATH" > "$RUN_ROOT/existing_test150.sha256"

export HF_ENDPOINT
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}

ingest_args=(
  python3 -m app.audio_cvr_source_ingest prepare
  --root "$ROOT"
  --run-root "$RUN_ROOT"
  --dataset "$DATASETS"
  --source-targets "$SOURCE_TARGETS"
  --hf-endpoint "$HF_ENDPOINT"
  --min-duration-seconds "$MIN_CLIP_SECONDS"
  --probe-workers "$PROBE_WORKERS"
  --allow-partial-downloads
)
for path in "${EXCLUDE_OVERLAP_PATHS[@]}"; do
  ingest_args+=(--exclude-overlap-with "$path")
done
if [ "$RESUME" = "1" ]; then
  ingest_args+=(--resume)
fi

if [ "$RESUME" = "1" ] && python3 - "$RUN_ROOT/source_ingest_summary.json" "$RUN_ROOT/provenance_manifest.jsonl" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
if not summary_path.is_file() or not manifest_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
selected = int(summary.get("selected_source_count") or 0)
rows = sum(1 for line in manifest_path.open("rb") if line.strip())
raise SystemExit(0 if selected > 0 and rows == selected else 1)
PY
then
  write_status "RUNNING" "source_ingest_cached" "freezing completed source ingest; newly downloaded media will not be injected into this run"
  echo "[avatar-like-test1000] reuse completed source ingest from $RUN_ROOT"
else
  write_status "RUNNING" "source_ingest" "downloading and staging VGGSound-family videos"
  "${ingest_args[@]}" 2>&1 | tee -a "$RUN_ROOT/logs/source_ingest.log"
fi

clip_args=(
  python3 -m app.audio_cvr_clips
  --root "$CONSTRUCTION_ROOT"
  --output-root "$SINGLE_SOURCE_ROOT"
  --clip-seconds "$CLIP_SECONDS"
  --min-clip-seconds "$MIN_CLIP_SECONDS"
  --max-clip-seconds "$MAX_CLIP_SECONDS"
  --min-clips-per-source 2
  --max-clips-per-source "$MAX_CLIPS_PER_SOURCE"
  --include-tail-segment
)
IFS=',' read -r -a dataset_items <<< "$DATASETS"
for dataset in "${dataset_items[@]}"; do
  clip_args+=(--dataset "$dataset")
done

CLIP_SUMMARY="$SINGLE_SOURCE_ROOT/_manifests/audio_cvr_avatar_like_6_9s_summary.json"
if [ "$RESUME" = "1" ] && python3 - "$CLIP_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
if not summary_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
segments = int(summary.get("segment_count") or 0)
durable = int(summary.get("durable_clip_count") or 0)
manifest = Path(str(summary.get("manifest_path") or ""))
groups = Path(str(summary.get("groups_path") or ""))
raise SystemExit(0 if segments > 0 and durable == segments and manifest.is_file() and groups.is_file() else 1)
PY
then
  write_status "RUNNING" "clip_build_cached" "reusing the completed clip inventory without rescanning newly extracted sources"
  echo "[avatar-like-test1000] reuse completed clip build summary=$CLIP_SUMMARY"
else
  write_status "RUNNING" "clip_build" "building 6-9 second source-aware clips"
  "${clip_args[@]}" 2>&1 | tee -a "$RUN_ROOT/logs/clip_build.log"
fi

run_bline_phase() {
  local candidate_limit="$1"
  local phase_name="$2"
  local phase_log="$RUN_ROOT/logs/bline_${phase_name}.log"
  local phase_dir="$RUN_ROOT/staged_review"
  local phase_marker="$phase_dir/phase_${candidate_limit}_complete.json"
  mkdir -p "$phase_dir"
  if [ -s "$phase_marker" ]; then
    echo "[avatar-like-test1000] phase candidate_limit=$candidate_limit already complete; continuing from the next cumulative range"
    return 0
  fi
  local args=(
    bash scripts/run_audio_cvr_v1_b_first.sh
    --root "$CONSTRUCTION_ROOT"
    --single-source-root "$SINGLE_SOURCE_ROOT"
    --run-root "$RUN_ROOT"
    --base-url "$BASE_URL"
    --base-url-pool "$BASE_URL"
    --model "$MODEL"
    --max-b-candidates "$candidate_limit"
    --propose-shards "$PROPOSE_SHARDS"
    --propose-parallel-jobs "$PROPOSE_PARALLEL_JOBS"
    --concurrency "$CONCURRENCY"
    --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
    --shard-timeout-seconds "$SHARD_TIMEOUT_SECONDS"
    --target-b-count 1000000
    --quality-profile "$QUALITY_PROFILE"
    --resume
  )
  write_status "RUNNING" "omni_review_${phase_name}" "reviewing cumulative candidate limit=$candidate_limit; existing Omni service remains running"
  "${args[@]}" 2>&1 | tee -a "$RUN_ROOT/logs/bline.log" "$phase_log"
  PHASE_MARKER="$phase_marker" CANDIDATE_LIMIT="$candidate_limit" RUN_ROOT="$RUN_ROOT" python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
ranked = root / "b_ranked_single_source_pairs.jsonl"
candidates = root / "b_candidates.jsonl"

def count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

payload = {
    "state": "COMPLETE",
    "candidate_limit": int(os.environ["CANDIDATE_LIMIT"]),
    "ranked_count": count(ranked),
    "candidate_count": count(candidates),
    "accepted_progress_count": sum(count(path) for path in root.glob("b_shards/accepted_progress_*.jsonl")),
    "rejected_progress_count": sum(count(path) for path in root.glob("b_shards/rejected_progress_*.jsonl")),
    "candidate_sha256": digest(candidates),
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "resume_policy": "proposal_id_checkpoint; only unseen cumulative candidates call Omni",
}
marker = Path(os.environ["PHASE_MARKER"])
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY
}

assess_pilot_phase() {
  local candidate_limit="$1"
  local output_name="$2"
  local scale="$3"
  local assessment_dir="$RUN_ROOT/pilot_assessments/$output_name"
  python3 -m app.audio_cvr_source_ingest assess-pilot \
    --run-root "$RUN_ROOT" \
    --existing-test "$EXISTING_TEST_PATH" \
    --output-dir "$assessment_dir" \
    --requested-candidates "$candidate_limit" \
    --full-candidate-target "$MAX_B_CANDIDATES" \
    --min-total "$((PILOT_MIN_TOTAL * scale))" \
    --min-sound-event "$((PILOT_MIN_SOUND_EVENT * scale))" \
    --min-music "$((PILOT_MIN_MUSIC * scale))" \
    --borderline-total "$((PILOT_BORDERLINE_TOTAL * scale))" \
    --borderline-sound-event "$((PILOT_BORDERLINE_SOUND_EVENT * scale))" \
    --borderline-music "$((PILOT_BORDERLINE_MUSIC * scale))" \
    > "$RUN_ROOT/logs/assess_${output_name}.log"
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' \
    "$assessment_dir/pilot_assessment.json"
}

stop_after_pilot() {
  local state="$1"
  local detail="$2"
  write_status "$state" "pilot_decision" "$detail; all progress preserved; existing Omni service remains running"
  trap - EXIT
  echo "[avatar-like-test1000] $state: $detail"
  exit 0
}

run_bline_phase "$PILOT_CANDIDATES" "pilot_${PILOT_CANDIDATES}"
pilot_decision=$(assess_pilot_phase "$PILOT_CANDIDATES" "pilot_${PILOT_CANDIDATES}" 1)
if [ "$pilot_decision" = "FAIL" ]; then
  stop_after_pilot "PILOT_REJECTED" "first $PILOT_CANDIDATES candidates cannot support the requested Test1000 mix"
fi
if [ "$pilot_decision" = "BORDERLINE" ]; then
  run_bline_phase "$PILOT_SECOND_CANDIDATES" "pilot_${PILOT_SECOND_CANDIDATES}"
  second_scale=$((PILOT_SECOND_CANDIDATES / PILOT_CANDIDATES))
  second_decision=$(assess_pilot_phase "$PILOT_SECOND_CANDIDATES" "pilot_${PILOT_SECOND_CANDIDATES}" "$second_scale")
  if [ "$second_decision" != "GO" ]; then
    stop_after_pilot "PILOT_REJECTED" "cumulative $PILOT_SECOND_CANDIDATES-candidate review did not reach the scaled GO thresholds"
  fi
fi

write_status "RUNNING" "pilot_passed" "pilot yield supports Test1000; expanding review from durable progress without stopping Omni"
run_bline_phase "$MAX_B_CANDIDATES" "full_${MAX_B_CANDIDATES}"

write_status "RUNNING" "postprocess" "building splits, quality reports, and local candidates"
python3 -m app.audio_lines_single_source build-b-splits \
  --run-root "$RUN_ROOT" \
  >> "$RUN_ROOT/logs/build_b_splits.log" 2>&1

python3 -m app.audio_cvr_protocol_eval summarize-data \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/protocol_postprocess" \
  --run-label "Audio-CVR Avatar-like Test1000 Supplement" \
  >> "$RUN_ROOT/logs/summarize_data.log" 2>&1

python3 -m app.audio_cvr_protocol_eval mine-local-same-source \
  --run-root "$RUN_ROOT" \
  --input "$RUN_ROOT/b_main_audio_cvr_triplets.jsonl" \
  --output "$RUN_ROOT/b_main_local_same_source_candidates.jsonl" \
  --max-per-query 5 \
  --manifest-path "$RUN_ROOT/extracted_single_source_clips.jsonl" \
  --summary-output "$RUN_ROOT/local_same_source_candidate_summary.json" \
  --coverage-output "$RUN_ROOT/local_same_source_coverage.md" \
  >> "$RUN_ROOT/logs/mine_local_same_source.log" 2>&1

python3 -m app.audio_cvr_source_ingest summarize-run \
  --run-root "$RUN_ROOT" \
  >> "$RUN_ROOT/logs/summarize_supplement.log" 2>&1

write_status "RUNNING" "freeze_test1000" "preserving test150 and selecting 850 new source-disjoint queries"
python3 -m app.audio_cvr_source_ingest extend-frozen-test \
  --existing-test "$EXISTING_TEST_PATH" \
  --candidate-path "$RUN_ROOT/b_main_audio_cvr_triplets.jsonl" \
  --output-dir "$RUN_ROOT/benchmark_test1000" \
  --target-count 1000 \
  --sound-event-target 800 \
  --music-target 200 \
  >> "$RUN_ROOT/logs/extend_test1000.log" 2>&1

write_status "COMPLETE" "done" "test1000 frozen; existing Omni service remains running and GPU 4-7 were untouched"
trap - EXIT
echo "[avatar-like-test1000] COMPLETE run_root=$RUN_ROOT"
