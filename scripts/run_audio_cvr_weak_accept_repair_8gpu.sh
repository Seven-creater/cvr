#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data02/usr/wangqihao/miniconda3/envs/omni_src/bin/python"
OMNIEMBED_PYTHON="/data02/usr/wangqihao/miniconda3/envs/peft/bin/python"
QWEN_OMNI_UTILS_ROOT="/data02/usr/wangqihao/miniconda3/envs/omni/lib/python3.10/site-packages"
FULL_TEST="runs/audio_cvr_test1000_unified_auditonly_20260723_142000/final_test1000/test_main_1000.jsonl"
CORE_TEST="runs/audiocvr_benchmark150_auto_20260720_164327/benchmark_v1_final150_val28/test_main_150.jsonl"
OMNICVR_RECORDS="runs/omnicvr_reference_diagnostics_20260721_144436/records_audio_center/eval.jsonl"
OMNICVR_GALLERY="runs/omnicvr_reference_diagnostics_20260721_144436/records_audio_center/eval_gallery.jsonl"
E5_MODEL="/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/e5-omni-7B"
E5_EXACT_VT="runs/e5_overlap_avatar_vgg_20260722_204500/cache_V_T"
E5_EXACT_VAT="runs/e5_overlap_avatar_vgg_20260722_204500/cache_V_A_T"
E5_ADAPTER_ROOT="runs/audiocvr_fewshot_bidir_final_20260721_071148/final_forward_bidir"
IMAGEBIND_MODEL="/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/imagebind"
IMAGEBIND_ROOT="runs/imagebind_overlap_pre516_test1000_20260723_010521"
OMNIEMBED_BASE="/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen2.5-omni"
OMNIEMBED_ADAPTER="/data02/pretrained_model/cvr_learn/cvr_model/02_large_multimodal_embedding/omniembed-v0.1-multivent"
OUT_ROOT=""
EXPECTED_SHA256="70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e"
AUDIT_HOST="127.0.0.1"
AUDIT_PORT=8787
GPU_IDS="0,1,2,3,4,5,6,7"
MEDIA_ROOTS=()
FINAL_SEEDS="13,23,42,71,101"
VARIANT_WORKERS=8
ENCODING_RETRIES=4
OMNIEMBED_RETRIES=4

usage() {
  cat <<'EOF'
Usage: run_audio_cvr_weak_accept_repair_8gpu.sh --out-root PATH [options]

Runs the frozen Full1000 Weak Accept evidence repair:
  blinded human-audit preparation, deterministic source variants,
  OmniEmbed-MultiVent, E5 reference perturbations, and ImageBind perturbations.

All item caches are atomic and resumable. Full1000 membership, order, SHA256,
and frozen E5 adapters are never modified.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --omniembed-python) OMNIEMBED_PYTHON="$2"; shift 2 ;;
    --qwen-omni-utils-root) QWEN_OMNI_UTILS_ROOT="$2"; shift 2 ;;
    --full-test) FULL_TEST="$2"; shift 2 ;;
    --core-test) CORE_TEST="$2"; shift 2 ;;
    --omnicvr-records) OMNICVR_RECORDS="$2"; shift 2 ;;
    --omnicvr-gallery) OMNICVR_GALLERY="$2"; shift 2 ;;
    --e5-model) E5_MODEL="$2"; shift 2 ;;
    --e5-exact-vt) E5_EXACT_VT="$2"; shift 2 ;;
    --e5-exact-vat) E5_EXACT_VAT="$2"; shift 2 ;;
    --e5-adapter-root) E5_ADAPTER_ROOT="$2"; shift 2 ;;
    --imagebind-model) IMAGEBIND_MODEL="$2"; shift 2 ;;
    --imagebind-root) IMAGEBIND_ROOT="$2"; shift 2 ;;
    --omniembed-base) OMNIEMBED_BASE="$2"; shift 2 ;;
    --omniembed-adapter) OMNIEMBED_ADAPTER="$2"; shift 2 ;;
    --expected-sha256) EXPECTED_SHA256="$2"; shift 2 ;;
    --audit-host) AUDIT_HOST="$2"; shift 2 ;;
    --audit-port) AUDIT_PORT="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --media-root) MEDIA_ROOTS+=("$2"); shift 2 ;;
    --variant-workers) VARIANT_WORKERS="$2"; shift 2 ;;
    --encoding-retries) ENCODING_RETRIES="$2"; shift 2 ;;
    --omniembed-retries) OMNIEMBED_RETRIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OUT_ROOT" ]] || { echo "--out-root is required" >&2; exit 2; }
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"/{logs,pids,workers}
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
STATUS="$OUT_ROOT/status.json"
HUMAN_AUDIT="$OUT_ROOT/human_audit"
VARIANT_ROOT="$OUT_ROOT/reference_variants"
VARIANT_PLAN="$VARIANT_ROOT/reference_variant_plan.jsonl"
VARIANT_MANIFEST="$VARIANT_ROOT/reference_variant_manifest.jsonl"
OMNIEMBED_ROOT="$OUT_ROOT/omniembed"
OMNIEMBED_INVENTORY="$OMNIEMBED_ROOT/inventory/embedding_inventory.jsonl"
OMNIEMBED_CACHE="$OMNIEMBED_ROOT/cache"
E5_VARIANT_EMBEDDINGS="$OUT_ROOT/e5_variant_embeddings"
E5_VARIANT_CACHES="$OUT_ROOT/e5_variant_caches"
E5_EVALUATION="$OUT_ROOT/e5_evaluation"
IMAGEBIND_VARIANT_INVENTORY="$OUT_ROOT/imagebind_variants/reference_variants.jsonl"
IMAGEBIND_VARIANT_ASSEMBLIES="$OUT_ROOT/imagebind_variants/assemblies"
IMAGEBIND_VARIANT_EVALUATION="$OUT_ROOT/imagebind_variants/evaluation"
IMAGEBIND_CACHE="$IMAGEBIND_ROOT/content_cache"
IMAGEBIND_EXACT_ASSEMBLY="$IMAGEBIND_ROOT/final_assembly"
CHILD_PIDS=()
RUN_STATE="FAILED"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} -eq 8 ]] || { echo "Exactly eight GPUs are required" >&2; exit 2; }
IFS=',' read -r -a SEEDS <<< "$FINAL_SEEDS"

write_status() {
  local state="$1" stage="$2" message="$3"
  "$PYTHON_BIN" - "$STATUS" "$state" "$stage" "$message" "$EXPECTED_SHA256" <<'PY'
import json, os, pathlib, sys, tempfile, time
path, state, stage, message, sha = sys.argv[1:]
payload = {
    "state": state,
    "stage": stage,
    "message": message,
    "full1000_sha256": sha,
    "launcher_pid": os.getppid(),
    "updated_unix": time.time(),
}
target = pathlib.Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".status.", dir=target.parent)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${CHILD_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  if [[ "$RUN_STATE" != "COMPLETE" ]]; then
    write_status "FAILED" "launcher" "exit_code=$code; all atomic caches and audit responses are preserved"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "$$" > "$OUT_ROOT/launcher.pid"
ps -o pgid= -p $$ | tr -d ' ' > "$OUT_ROOT/launcher.pgid"

media_args=()
if [[ ${#MEDIA_ROOTS[@]} -eq 0 ]]; then
  MEDIA_ROOTS=(
    "$REPO_ROOT"
    "/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval"
  )
fi
for root in "${MEDIA_ROOTS[@]}"; do media_args+=(--media-root "$root"); done

run_parallel() {
  local -a pids=() logs=()
  while [[ $# -gt 0 ]]; do
    local log="$1" command="$2"
    shift 2
    (
      set -Eeuo pipefail
      eval "$command"
    ) > "$log" 2>&1 &
    pids+=("$!")
    logs+=("$log")
    CHILD_PIDS+=("$!")
  done
  local failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Worker failed: ${logs[$index]}" >&2
      tail -120 "${logs[$index]}" >&2 || true
      failed=1
    fi
  done
  CHILD_PIDS=()
  [[ "$failed" -eq 0 ]]
}

prepare_evidence() {
  write_status "RUNNING" "PREPARE_HUMAN_AUDIT" "verifying immutable Full1000 and preparing blinded audit"
  "$PYTHON_BIN" -m app.audio_cvr_weak_accept prepare-human-audit \
    --full-path "$FULL_TEST" --core-path "$CORE_TEST" --output-dir "$HUMAN_AUDIT" \
    --expected-full-sha256 "$EXPECTED_SHA256" "${media_args[@]}" \
    > "$OUT_ROOT/logs/prepare_human_audit.log" 2>&1
  "$PYTHON_BIN" -m app.audio_cvr_weak_accept prepare-reference-variants \
    --full-path "$FULL_TEST" --output-dir "$VARIANT_ROOT" \
    --expected-full-sha256 "$EXPECTED_SHA256" "${media_args[@]}" \
    > "$OUT_ROOT/logs/prepare_reference_variants.log" 2>&1
}

generate_variants() {
  write_status "RUNNING" "PREPARE_REFERENCE_VARIANTS" "generating deterministic transcoded, temporal, and spatial references"
  local -a jobs=()
  for ((shard=0; shard<VARIANT_WORKERS; shard++)); do
    jobs+=(
      "$OUT_ROOT/logs/reference_variants_shard${shard}.log"
      "\"$PYTHON_BIN\" -m app.audio_cvr_weak_accept generate-reference-variants --plan-path \"$VARIANT_PLAN\" --shard-index $shard --shard-count $VARIANT_WORKERS --retries $ENCODING_RETRIES"
    )
  done
  run_parallel "${jobs[@]}"
  "$PYTHON_BIN" -m app.audio_cvr_weak_accept summarize-reference-variants \
    --plan-path "$VARIANT_PLAN" --output-dir "$VARIANT_ROOT" \
    > "$OUT_ROOT/logs/summarize_reference_variants.log" 2>&1
}

start_audit_server() {
  local pid_file="$HUMAN_AUDIT/server.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    return
  fi
  mkdir -p "$HUMAN_AUDIT"
  setsid nohup "$PYTHON_BIN" -m app.audio_cvr_weak_accept serve-human-audit \
    --audit-dir "$HUMAN_AUDIT" --variants-dir "$VARIANT_ROOT" \
    --host "$AUDIT_HOST" --port "$AUDIT_PORT" \
    > "$HUMAN_AUDIT/server.log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  sleep 2
  kill -0 "$(cat "$pid_file")" 2>/dev/null || {
    tail -100 "$HUMAN_AUDIT/server.log" >&2
    return 1
  }
}

encode_omniembed() {
  write_status "RUNNING" "ENCODE_OMNIEMBED" "encoding Audio-CVR and OmniCVR with frozen OmniEmbed-MultiVent"
  "$OMNIEMBED_PYTHON" -m app.audio_cvr_omniembed prepare \
    --audio-test "$FULL_TEST" --omnicvr-records "$OMNICVR_RECORDS" \
    --omnicvr-gallery "$OMNICVR_GALLERY" --variant-manifest "$VARIANT_MANIFEST" \
    --output-dir "$OMNIEMBED_ROOT/inventory" --audio-test-sha256 "$EXPECTED_SHA256" \
    "${media_args[@]}" > "$OUT_ROOT/logs/omniembed_prepare.log" 2>&1
  local -a jobs=()
  for shard in {0..7}; do
    gpu="${GPUS[$shard]}"
    jobs+=(
      "$OUT_ROOT/logs/omniembed_shard${shard}.log"
      "CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false QWEN_OMNI_UTILS_ROOT=\"$QWEN_OMNI_UTILS_ROOT\" \"$OMNIEMBED_PYTHON\" -m app.audio_cvr_omniembed encode --inventory-path \"$OMNIEMBED_INVENTORY\" --cache-dir \"$OMNIEMBED_CACHE\" --base-model \"$OMNIEMBED_BASE\" --adapter-model \"$OMNIEMBED_ADAPTER\" --shard-index $shard --shard-count 8 --device cuda --retries $OMNIEMBED_RETRIES --attn-implementation sdpa"
    )
  done
  run_parallel "${jobs[@]}"
  "$OMNIEMBED_PYTHON" -m app.audio_cvr_omniembed audit-cache \
    --inventory-path "$OMNIEMBED_INVENTORY" --cache-dir "$OMNIEMBED_CACHE" \
    --output-path "$OMNIEMBED_ROOT/cache_audit.json" \
    > "$OUT_ROOT/logs/omniembed_cache_audit.log" 2>&1
  "$OMNIEMBED_PYTHON" -m app.audio_cvr_omniembed evaluate \
    --records-dir "$OMNIEMBED_ROOT/inventory" --inventory-path "$OMNIEMBED_INVENTORY" \
    --cache-dir "$OMNIEMBED_CACHE" --output-dir "$OMNIEMBED_ROOT/evaluation" \
    > "$OUT_ROOT/logs/omniembed_evaluate.log" 2>&1
  "$OMNIEMBED_PYTHON" -m app.audio_cvr_omniembed statistics \
    --per-query-path "$OMNIEMBED_ROOT/evaluation/per_query_results.jsonl" \
    --output-dir "$OMNIEMBED_ROOT/statistics" --iterations 20000 \
    > "$OUT_ROOT/logs/omniembed_statistics.log" 2>&1
}

encode_e5_variants() {
  write_status "RUNNING" "ENCODE_REFERENCE_VARIANTS" "encoding only perturbed reference documents with frozen E5-Omni"
  local -a jobs=()
  for shard in {0..3}; do
    jobs+=(
      "$OUT_ROOT/logs/e5_vt_variant_shard${shard}.log"
      "CUDA_VISIBLE_DEVICES=${GPUS[$shard]} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \"$PYTHON_BIN\" -m app.audio_cvr_weak_accept cache-e5-variant-references --variant-manifest \"$VARIANT_MANIFEST\" --output-dir \"$E5_VARIANT_EMBEDDINGS\" --model-path \"$E5_MODEL\" --video-audio-mode off --shard-index $shard --shard-count 4 --device cuda --batch-size 1 --retries $ENCODING_RETRIES"
    )
    jobs+=(
      "$OUT_ROOT/logs/e5_vat_variant_shard${shard}.log"
      "CUDA_VISIBLE_DEVICES=${GPUS[$((shard+4))]} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \"$PYTHON_BIN\" -m app.audio_cvr_weak_accept cache-e5-variant-references --variant-manifest \"$VARIANT_MANIFEST\" --output-dir \"$E5_VARIANT_EMBEDDINGS\" --model-path \"$E5_MODEL\" --video-audio-mode on --shard-index $shard --shard-count 4 --device cuda --batch-size 1 --retries $ENCODING_RETRIES"
    )
  done
  run_parallel "${jobs[@]}"
  for condition in transcoded temporal spatial; do
    "$PYTHON_BIN" -m app.audio_cvr_weak_accept assemble-e5-variant-cache \
      --exact-cache-dir "$E5_EXACT_VT" --variant-manifest "$VARIANT_MANIFEST" \
      --variant-embedding-root "$E5_VARIANT_EMBEDDINGS" --video-audio-mode off \
      --condition "$condition" --output-dir "$E5_VARIANT_CACHES/$condition/V_T" \
      > "$OUT_ROOT/logs/e5_assemble_${condition}_V_T.log" 2>&1
    "$PYTHON_BIN" -m app.audio_cvr_weak_accept assemble-e5-variant-cache \
      --exact-cache-dir "$E5_EXACT_VAT" --variant-manifest "$VARIANT_MANIFEST" \
      --variant-embedding-root "$E5_VARIANT_EMBEDDINGS" --video-audio-mode on \
      --condition "$condition" --output-dir "$E5_VARIANT_CACHES/$condition/V_A_T" \
      > "$OUT_ROOT/logs/e5_assemble_${condition}_V_A_T.log" 2>&1
  done
}

evaluate_e5() {
  write_status "RUNNING" "EVALUATE" "evaluating frozen E5 adapters on exact and reference-perturbation conditions"
  local -a jobs=()
  local slot=0
  for seed in "${SEEDS[@]}"; do
    adapter="$E5_ADAPTER_ROOT/seed_${seed}/adapter"
    [[ -s "$adapter/adapter.pt" ]] || { echo "Missing E5 adapter: $adapter" >&2; return 1; }
    for condition in exact transcoded temporal spatial; do
      for mode in V_T V_A_T; do
        if [[ "$condition" == "exact" && "$mode" == "V_T" ]]; then cache="$E5_EXACT_VT"
        elif [[ "$condition" == "exact" ]]; then cache="$E5_EXACT_VAT"
        else cache="$E5_VARIANT_CACHES/$condition/$mode"
        fi
        for reference_state in with_reference masked_reference; do
          output="$E5_EVALUATION/seed_${seed}/$condition/$mode/$reference_state"
          summary="$output/summary.json"
          if [[ -s "$summary" ]] && "$PYTHON_BIN" - "$summary" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if value.get("eval_count")==1000 else 1)
PY
          then
            continue
          fi
          gpu="${GPUS[$((slot % 8))]}"
          log="$OUT_ROOT/logs/e5_eval_seed${seed}_${condition}_${mode}_${reference_state}.log"
          mask_arg=""
          [[ "$reference_state" == "masked_reference" ]] && mask_arg="--exclude-query-reference"
          jobs+=(
            "$log"
            "CUDA_VISIBLE_DEVICES=$gpu \"$PYTHON_BIN\" -m app.e5_audio_delta_train eval --cache-dir \"$cache\" --adapter-dir \"$adapter\" --output-dir \"$output\" --device cuda --topk 1,5,10 --save-topk 20 $mask_arg"
          )
          slot=$((slot + 1))
          if [[ $(( ${#jobs[@]} / 2 )) -eq 8 ]]; then
            run_parallel "${jobs[@]}"
            jobs=()
          fi
        done
      done
    done
  done
  [[ ${#jobs[@]} -eq 0 ]] || run_parallel "${jobs[@]}"
  "$PYTHON_BIN" -m app.audio_cvr_reference_ladder summarize-e5 \
    --evaluation-root "$E5_EVALUATION" --output-dir "$OUT_ROOT/statistics/e5" \
    --seeds "$FINAL_SEEDS" --iterations 20000 --seed 20260724 \
    > "$OUT_ROOT/logs/e5_reference_ladder_statistics.log" 2>&1
}

encode_and_evaluate_imagebind_variants() {
  write_status "RUNNING" "ENCODE_REFERENCE_VARIANTS" "encoding ImageBind reference variants with content-addressed resume"
  "$PYTHON_BIN" -m app.audio_cvr_weak_accept prepare-imagebind-variant-inventory \
    --variant-manifest "$VARIANT_MANIFEST" --output-path "$IMAGEBIND_VARIANT_INVENTORY" \
    > "$OUT_ROOT/logs/imagebind_prepare_variants.log" 2>&1
  local -a jobs=()
  for shard in {0..7}; do
    jobs+=(
      "$OUT_ROOT/logs/imagebind_variant_shard${shard}.log"
      "CUDA_VISIBLE_DEVICES=${GPUS[$shard]} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \"$PYTHON_BIN\" -m app.audio_cvr_external_baseline cache-imagebind --inventory \"$IMAGEBIND_VARIANT_INVENTORY\" --cache-root \"$IMAGEBIND_CACHE\" --model-dir \"$IMAGEBIND_MODEL\" --inventory-kind media --shard-index $shard --shard-count 8 --device cuda --batch-size 2 --encoding-retries $ENCODING_RETRIES"
    )
  done
  run_parallel "${jobs[@]}"
  for condition in transcoded temporal spatial; do
    assembly="$IMAGEBIND_VARIANT_ASSEMBLIES/$condition"
    evaluation="$IMAGEBIND_VARIANT_EVALUATION/$condition"
    "$PYTHON_BIN" -m app.audio_cvr_weak_accept assemble-imagebind-variant-cache \
      --exact-assembly-dir "$IMAGEBIND_EXACT_ASSEMBLY" \
      --variant-inventory "$IMAGEBIND_VARIANT_INVENTORY" --cache-root "$IMAGEBIND_CACHE" \
      --condition "$condition" --output-dir "$assembly" \
      > "$OUT_ROOT/logs/imagebind_assemble_${condition}.log" 2>&1
    "$PYTHON_BIN" -m app.audio_cvr_external_baseline evaluate \
      --assembly-dir "$assembly" --output-dir "$evaluation" --save-topk 20 \
      > "$OUT_ROOT/logs/imagebind_evaluate_${condition}.log" 2>&1
    "$PYTHON_BIN" -m app.audio_cvr_external_baseline summarize \
      --evaluation-dir "$evaluation" --output-dir "$evaluation/statistics" \
      --iterations 20000 --seed 20260724 \
      > "$OUT_ROOT/logs/imagebind_statistics_${condition}.log" 2>&1
  done
  "$PYTHON_BIN" -m app.audio_cvr_reference_ladder summarize-imagebind \
    --exact-evaluation "$IMAGEBIND_ROOT/evaluation" \
    --variant-evaluation-root "$IMAGEBIND_VARIANT_EVALUATION" \
    --output-dir "$OUT_ROOT/statistics/imagebind" --iterations 20000 --seed 20260724 \
    > "$OUT_ROOT/logs/imagebind_reference_ladder_statistics.log" 2>&1
}

final_audit() {
  write_status "RUNNING" "STATISTICS" "verifying frozen test, replacement invariants, and complete model evidence"
  "$PYTHON_BIN" - "$FULL_TEST" "$EXPECTED_SHA256" "$OUT_ROOT" <<'PY'
import hashlib, json, pathlib, sys
test, expected, root = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
actual = hashlib.sha256(test.read_bytes()).hexdigest()
rows = [json.loads(line) for line in test.read_text(encoding="utf-8").splitlines() if line.strip()]
violations = []
if actual != expected: violations.append(f"sha256:{actual}")
if len(rows) != 1000: violations.append(f"test_count:{len(rows)}")
required = [
    root / "reference_variants/reference_variant_summary.json",
    root / "omniembed/cache_audit.json",
        root / "omniembed/evaluation/results.json",
        root / "omniembed/statistics/paired_comparisons.json",
        root / "statistics/e5/e5_reference_ladder_mean_std.json",
        root / "statistics/e5/e5_paired_comparisons.json",
        root / "statistics/imagebind/imagebind_reference_ladder.json",
        root / "statistics/imagebind/imagebind_paired_comparisons.json",
    ]
for condition in ("transcoded", "temporal", "spatial"):
    required.extend([
        root / f"e5_variant_caches/{condition}/V_T/reference_replacement_audit.json",
        root / f"e5_variant_caches/{condition}/V_A_T/reference_replacement_audit.json",
        root / f"imagebind_variants/assemblies/{condition}/reference_replacement_audit.json",
        root / f"imagebind_variants/evaluation/{condition}/seven_mode_results.json",
    ])
missing = [str(path) for path in required if not path.is_file()]
if missing: violations.append(f"missing_outputs:{len(missing)}")
payload = {
    "state": "PAPER_EVIDENCE_COMPLETE" if not violations else "FAILED",
    "full1000_count": len(rows),
    "full1000_sha256": actual,
    "full1000_membership_unchanged": actual == expected,
    "human_audit_response_count": len(list((root / "human_audit/responses").glob("*.json"))),
    "human_audit_complete": (root / "human_audit_summary/human_audit_summary.json").is_file(),
    "missing_outputs": missing,
    "violation_count": len(violations),
    "violations": violations,
}
(root / "final_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if violations:
    raise SystemExit(str(violations))
PY
}

prepare_evidence
generate_variants
start_audit_server
encode_omniembed
encode_e5_variants
evaluate_e5
encode_and_evaluate_imagebind_variants
final_audit
RUN_STATE="COMPLETE"
write_status "COMPLETE" "PAPER_EVIDENCE_COMPLETE" "automatic evidence complete; blinded human audit server remains available"
trap - EXIT INT TERM
exit 0
