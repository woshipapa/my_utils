#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/profile_cli_common.sh"

usage() {
  cat <<'USAGE'
Usage:
  run_nsys_quick.sh [--] <command> [args...]

Examples:
  run_nsys_quick.sh python train.py --config cfg.yaml
  NSYS_START_STEP=10 NSYS_STOP_STEP=20 run_nsys_quick.sh torchrun --nproc_per_node=8 --no_python python train.py
  NSYS_NIC_METRICS_MODE=lf NSYS_SYSCALL=process-tree run_nsys_quick.sh python train.py

Notes:
  - Uses preset: profiling/templates/preset_nsys_default.env by default.
  - Override with PROFILE_PRESET=/path/to/preset.env.
  - Compatibility fallback (nsys 2024/2026 switch differences) is handled automatically.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

PROFILE_PRESET="${PROFILE_PRESET:-${SCRIPT_DIR}/preset_nsys_default.env}"
profile_prepare "${PROFILE_PRESET}"

EXEC_CMD=("$@")
profile_wrap_exec_with_nsys EXEC_CMD

echo "[run_nsys_quick] Launching:"
printf ' %q' "${EXEC_CMD[@]}"
printf '\n'

exec "${EXEC_CMD[@]}"
