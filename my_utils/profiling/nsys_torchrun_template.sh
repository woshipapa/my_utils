#!/usr/bin/env bash
# Nsight Systems 2026.2 reusable template for model training profiling.
# Usage:
#   1) Copy this file and edit APP_CMD / APP_ARGS
#   2) Optional: set env vars before launch
#   3) Run: bash nsys_torchrun_template.sh
#
# Notes:
# - This template is launcher-agnostic. Default launcher is torchrun.
# - For torchrun + wrapped command (nsys profile ... python ...), keep --no_python.

set -euo pipefail

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_is_true() {
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

_nsys_detect_version() {
  local hint="${NSYS_VERSION_HINT:-}"
  if [[ "$hint" =~ ([0-9]{4})\.([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    return 0
  fi
  local raw
  raw="$(nsys --version 2>/dev/null || true)"
  if [[ "$raw" =~ ([0-9]{4})\.([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
  else
    echo "0 0"
  fi
}

_nsys_version_gte() {
  local major="${1:-0}"
  local minor="${2:-0}"
  if (( NSYS_VER_MAJOR > major )); then
    return 0
  fi
  if (( NSYS_VER_MAJOR == major && NSYS_VER_MINOR >= minor )); then
    return 0
  fi
  return 1
}

_nsys_contains_token() {
  local needle="${1:-}"
  shift || true
  local token
  for token in "$@"; do
    if [[ "$token" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# User section: model/framework command
# -----------------------------------------------------------------------------
# Base application command (can be python/deepspeed/accelerate/etc.)
APP_CMD=(python path/to/train.py)

# Application arguments
APP_ARGS=(
  --config path/to/config.yaml
)

# -----------------------------------------------------------------------------
# Environment defaults
# -----------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export ENABLE_NVTX="${ENABLE_NVTX:-1}"

# -----------------------------------------------------------------------------
# Distributed launcher section (default: torchrun)
# -----------------------------------------------------------------------------
USE_DISTRIBUTED_LAUNCHER="${USE_DISTRIBUTED_LAUNCHER:-1}"
DIST_LAUNCHER="${DIST_LAUNCHER:-torchrun}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DIST_EXTRA_ARGS_STR="${DIST_EXTRA_ARGS_STR:-}"

DIST_CMD=()
if [[ "${USE_DISTRIBUTED_LAUNCHER}" == "1" ]]; then
  if [[ "${DIST_LAUNCHER}" == "torchrun" ]]; then
    DIST_CMD=(
      torchrun
      --nnodes="${NNODES}"
      --nproc_per_node="${NPROC_PER_NODE}"
      --node_rank="${NODE_RANK}"
      --master_addr="${MASTER_ADDR}"
      --master_port="${MASTER_PORT}"
      --no_python
    )
  else
    DIST_CMD=("${DIST_LAUNCHER}")
  fi

  if [[ -n "${DIST_EXTRA_ARGS_STR}" ]]; then
    read -r -a _dist_extra <<< "${DIST_EXTRA_ARGS_STR}"
    DIST_CMD+=("${_dist_extra[@]}")
  fi
fi

# -----------------------------------------------------------------------------
# Nsight Systems section (2026.2 style)
# -----------------------------------------------------------------------------
NSYS_PROFILER="${NSYS_PROFILER:-1}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-./logs/nsys}"

# Use %p by default for safety. For distributed rank, set e.g.
# NSYS_OUTPUT_BASENAME='train_rank_%q{RANK}'
NSYS_OUTPUT_BASENAME="${NSYS_OUTPUT_BASENAME:-train_%p}"

NSYS_FORCE_OVERWRITE="${NSYS_FORCE_OVERWRITE:-true}"
NSYS_EXPORT="${NSYS_EXPORT:-none}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"

# capture-range: none | cudaProfilerApi | hotkey | nvtx
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
# capture-range-end: none | stop | stop-shutdown | repeat[:N] | repeat-shutdown:N
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"
NSYS_NVTX_CAPTURE="${NSYS_NVTX_CAPTURE:-}"
NSYS_NVTX_DOMAIN_INCLUDE="${NSYS_NVTX_DOMAIN_INCLUDE:-}"
NSYS_NVTX_DOMAIN_EXCLUDE="${NSYS_NVTX_DOMAIN_EXCLUDE:-}"

NSYS_CUDABACKTRACE="${NSYS_CUDABACKTRACE:-true}"
# sample: process-tree | system-wide | none
NSYS_SAMPLE="${NSYS_SAMPLE:-process-tree}"
NSYS_SHOW_OUTPUT="${NSYS_SHOW_OUTPUT:-true}"
NSYS_NIC_METRICS="${NSYS_NIC_METRICS:-false}"      # legacy bool
NSYS_NIC_METRICS_MODE="${NSYS_NIC_METRICS_MODE:-}" # preferred: lf|hf|none
NSYS_SYSCALL_MODE="${NSYS_SYSCALL_MODE:-}"         # preferred for nsys>=2026
NSYS_VERSION_HINT="${NSYS_VERSION_HINT:-}"         # optional override, e.g. 2026.2
NSYS_GPU_METRICS_DEVICES="${NSYS_GPU_METRICS_DEVICES:-none}"
NSYS_GPU_METRICS_FREQUENCY="${NSYS_GPU_METRICS_FREQUENCY:-}"

RUN_CMD=("${APP_CMD[@]}")

if [[ "${NSYS_PROFILER}" == "1" ]]; then
  mkdir -p "${NSYS_OUTPUT_DIR}"

  read -r NSYS_VER_MAJOR NSYS_VER_MINOR < <(_nsys_detect_version)

  # Normalize trace/syscall behavior across nsys versions.
  IFS=',' read -r -a _trace_in <<< "${NSYS_TRACE}"
  _trace_out=()
  _trace_has_syscall=0
  for _tok in "${_trace_in[@]}"; do
    _tok="${_tok// /}"
    [[ -z "${_tok}" ]] && continue
    if [[ "${_tok}" == "syscall" ]]; then
      _trace_has_syscall=1
      continue
    fi
    _trace_out+=("${_tok}")
  done
  if [[ ${#_trace_out[@]} -eq 0 ]]; then
    _trace_out=(cuda nvtx osrt cublas cudnn)
  fi
  NSYS_TRACE_RESOLVED="$(IFS=,; echo "${_trace_out[*]}")"

  if _nsys_version_gte 2026 0; then
    if [[ -z "${NSYS_SYSCALL_MODE}" && "${_trace_has_syscall}" == "1" ]]; then
      NSYS_SYSCALL_MODE="process-tree"
    fi
  elif [[ -n "${NSYS_SYSCALL_MODE}" && "${NSYS_SYSCALL_MODE}" != "none" ]]; then
    if ! _nsys_contains_token "syscall" "${_trace_out[@]}"; then
      NSYS_TRACE_RESOLVED="${NSYS_TRACE_RESOLVED},syscall"
    fi
  fi

  # Normalize NIC metrics behavior across nsys versions.
  _nic_mode="$(printf '%s' "${NSYS_NIC_METRICS_MODE}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${_nic_mode}" ]]; then
    if _is_true "${NSYS_NIC_METRICS}"; then
      if _nsys_version_gte 2026 0; then
        _nic_mode="lf"
      else
        _nic_mode="true"
      fi
    fi
  fi
  if _nsys_version_gte 2026 0; then
    if [[ "${_nic_mode}" == "true" || "${_nic_mode}" == "1" ]]; then
      _nic_mode="lf"
    elif [[ "${_nic_mode}" == "false" || "${_nic_mode}" == "0" ]]; then
      _nic_mode="none"
    fi
  else
    if [[ "${_nic_mode}" == "lf" || "${_nic_mode}" == "hf" ]]; then
      _nic_mode="true"
    elif [[ "${_nic_mode}" == "none" ]]; then
      _nic_mode="false"
    fi
  fi

  NSYS_CMD=(
    nsys profile
    "--output=${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_BASENAME}"
    "--force-overwrite=${NSYS_FORCE_OVERWRITE}"
    "--export=${NSYS_EXPORT}"
    "--trace=${NSYS_TRACE_RESOLVED}"
    "--capture-range=${NSYS_CAPTURE_RANGE}"
    "--capture-range-end=${NSYS_CAPTURE_RANGE_END}"
    "--cudabacktrace=${NSYS_CUDABACKTRACE}"
    "--sample=${NSYS_SAMPLE}"
    "--show-output=${NSYS_SHOW_OUTPUT}"
    "--gpu-metrics-devices=${NSYS_GPU_METRICS_DEVICES}"
  )

  if [[ -n "${_nic_mode}" ]]; then
    NSYS_CMD+=("--nic-metrics=${_nic_mode}")
  fi

  if _nsys_version_gte 2026 0 && [[ -n "${NSYS_SYSCALL_MODE}" ]]; then
    NSYS_CMD+=("--syscall=${NSYS_SYSCALL_MODE}")
  fi

  if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" ]]; then
    NSYS_CMD+=("--gpu-metrics-frequency=${NSYS_GPU_METRICS_FREQUENCY}")
  fi

  if [[ "${NSYS_CAPTURE_RANGE}" == "nvtx" && -n "${NSYS_NVTX_CAPTURE}" ]]; then
    NSYS_CMD+=("--nvtx-capture=${NSYS_NVTX_CAPTURE}")
  fi

  if [[ -n "${NSYS_NVTX_DOMAIN_INCLUDE}" ]]; then
    NSYS_CMD+=("--nvtx-domain-include=${NSYS_NVTX_DOMAIN_INCLUDE}")
  fi

  if [[ -n "${NSYS_NVTX_DOMAIN_EXCLUDE}" ]]; then
    NSYS_CMD+=("--nvtx-domain-exclude=${NSYS_NVTX_DOMAIN_EXCLUDE}")
  fi

  RUN_CMD=("${NSYS_CMD[@]}" "${RUN_CMD[@]}")
fi

CMD=("${DIST_CMD[@]}" "${RUN_CMD[@]}" "${APP_ARGS[@]}")

printf 'Launching command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
