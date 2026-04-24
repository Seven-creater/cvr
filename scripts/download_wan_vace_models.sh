#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone}
WAN_ROOT=${WAN_ROOT:-$MODEL_ROOT/Wan2.1}
MODEL_SIZE=${MODEL_SIZE:-1.3B}
INCLUDE_CODE=${INCLUDE_CODE:-1}

usage() {
  cat <<'EOF'
Usage: download_wan_vace_models.sh [options]

Options:
  --model-root PATH
  --wan-root PATH
  --model-size 1.3B|14B|both
  --skip-code
  -h, --help

Downloads Wan2.1 VACE weights into:
  /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/Wan2.1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root)
      MODEL_ROOT="$2"
      WAN_ROOT="$MODEL_ROOT/Wan2.1"
      shift 2
      ;;
    --wan-root)
      WAN_ROOT="$2"
      shift 2
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --skip-code)
      INCLUDE_CODE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[wan-download] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

download_hf_repo() {
  local repo_id="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --local-dir "$local_dir"
    return
  fi
  python - "$repo_id" "$local_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=local_dir)
PY
}

echo "[wan-download] start $(date)"
echo "[wan-download] wan_root=$WAN_ROOT"
echo "[wan-download] model_size=$MODEL_SIZE"
mkdir -p "$WAN_ROOT"

if [[ "$INCLUDE_CODE" == "1" ]]; then
  if [[ ! -d "$WAN_ROOT/code/.git" ]]; then
    git clone https://github.com/Wan-Video/Wan2.1.git "$WAN_ROOT/code"
  else
    echo "[wan-download] code already exists: $WAN_ROOT/code"
  fi
fi

case "$MODEL_SIZE" in
  1.3B)
    download_hf_repo "Wan-AI/Wan2.1-VACE-1.3B" "$WAN_ROOT/Wan2.1-VACE-1.3B"
    ;;
  14B)
    download_hf_repo "Wan-AI/Wan2.1-VACE-14B" "$WAN_ROOT/Wan2.1-VACE-14B"
    ;;
  both)
    download_hf_repo "Wan-AI/Wan2.1-VACE-1.3B" "$WAN_ROOT/Wan2.1-VACE-1.3B"
    download_hf_repo "Wan-AI/Wan2.1-VACE-14B" "$WAN_ROOT/Wan2.1-VACE-14B"
    ;;
  *)
    echo "[wan-download] unsupported --model-size=$MODEL_SIZE; expected 1.3B, 14B, or both" >&2
    exit 2
    ;;
esac

echo "[wan-download] downloaded dirs"
du -sh "$WAN_ROOT"/* || true
echo "[wan-download] done $(date)"
