#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ncu_generic_wrap.sh -- <framework-launch-command>

Examples:
  ncu_generic_wrap.sh -- torchrun --nproc_per_node=8 train.py --config cfg.yaml
  ncu_generic_wrap.sh -- vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
USAGE
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "[ncu_generic_wrap] missing command; use --help" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NCU_WRAPPER="${REPO_ROOT}/my_utils/profiling/ncu/run_ncu_quick_yaml.py"
NCU_CONFIG="${REPO_ROOT}/my_utils/profiling/ncu/ncu_full_collection.yaml"

python "${NCU_WRAPPER}" --config "${NCU_CONFIG}" -- "$@"
