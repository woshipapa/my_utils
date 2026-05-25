#!/usr/bin/env bash
set -euo pipefail

PLUGIN="${NCCL_PROFILER_PLUGIN:-}"
DUMP_DIR="${NCCL_INSPECTOR_DUMP_DIR:-./nccl-inspector-logs}"
INTERVAL_US="${NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS:-500}"
PROM_DUMP="${NCCL_INSPECTOR_PROM_DUMP:-0}"
ENABLE_P2P="${NCCL_INSPECTOR_ENABLE_P2P:-1}"
VERBOSE="${NCCL_INSPECTOR_DUMP_VERBOSE:-0}"
MIN_SIZE_BYTES="${NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES:-8192}"
REQUIRE_KERNEL_TIMING="${NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING:-1}"

usage() {
  cat <<'EOF'
Usage:
  run_nccl_inspector.sh [options] -- <command> [args...]

Options:
  --plugin PATH              libnccl-profiler-inspector.so path
  --dump-dir DIR             output directory (default: ./nccl-inspector-logs)
  --interval-us N            dump interval in microseconds (default: 500)
  --prometheus               enable Prometheus textfile output
  --enable-p2p 0|1           enable P2P tracking (default: 1)
  --verbose 0|1              include verbose event trace JSON (default: 0)
  --min-size-bytes N         minimum message size to track (default: 8192)
  --require-kernel-timing 0|1 keep only GPU kernel-timed events (default: 1)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plugin)
      PLUGIN="${2:?missing --plugin value}"
      shift 2
      ;;
    --dump-dir)
      DUMP_DIR="${2:?missing --dump-dir value}"
      shift 2
      ;;
    --interval-us)
      INTERVAL_US="${2:?missing --interval-us value}"
      shift 2
      ;;
    --prometheus)
      PROM_DUMP="1"
      shift
      ;;
    --enable-p2p)
      ENABLE_P2P="${2:?missing --enable-p2p value}"
      shift 2
      ;;
    --verbose)
      VERBOSE="${2:?missing --verbose value}"
      shift 2
      ;;
    --min-size-bytes)
      MIN_SIZE_BYTES="${2:?missing --min-size-bytes value}"
      shift 2
      ;;
    --require-kernel-timing)
      REQUIRE_KERNEL_TIMING="${2:?missing --require-kernel-timing value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "[nccl-inspector] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "[nccl-inspector] missing command after --" >&2
  usage >&2
  exit 2
fi

if [[ -z "${PLUGIN}" ]]; then
  echo "[nccl-inspector] NCCL profiler plugin path is required via --plugin or NCCL_PROFILER_PLUGIN" >&2
  exit 2
fi

mkdir -p "${DUMP_DIR}"

export NCCL_PROFILER_PLUGIN="${PLUGIN}"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_DIR="${DUMP_DIR}"
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS="${INTERVAL_US}"
export NCCL_INSPECTOR_PROM_DUMP="${PROM_DUMP}"
export NCCL_INSPECTOR_ENABLE_P2P="${ENABLE_P2P}"
export NCCL_INSPECTOR_DUMP_VERBOSE="${VERBOSE}"
export NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES="${MIN_SIZE_BYTES}"
export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING="${REQUIRE_KERNEL_TIMING}"

echo "[nccl-inspector] dump_dir=${NCCL_INSPECTOR_DUMP_DIR}"
echo "[nccl-inspector] prom_dump=${NCCL_INSPECTOR_PROM_DUMP} interval_us=${NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS}"
echo "[nccl-inspector] command: $*"
exec "$@"
