#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SAMPLE_SIZE=${SAMPLE_SIZE:-400}

exec "$SCRIPT_DIR/run_composed_avigate_smoke20.sh" "$@"
