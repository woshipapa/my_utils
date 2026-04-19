#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

bash "${NSYS_WRAPPER}" -- \
  python3 -m sglang.launch_server --model-path "${MODEL_PATH}" --host "${HOST}" --port "${PORT}"
