#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nsys_offline_common.sh"

SQLITE_PATH=${SQLITE_PATH:-}
OUTPUT=${OUTPUT:-./nsys_metrics_out/nsys_analyze.json}
FORMAT=${FORMAT:-json}
PRETTY=${PRETTY:-1}
DEVICE_ID=${DEVICE_ID:--1}
START_NS=${START_NS:--1}
END_NS=${END_NS:--1}
TOP_K=${TOP_K:-20}
LIMIT=${LIMIT:-500000}
ITERATION_MARKER=${ITERATION_MARKER:-sample_0}
MODEL_FLOPS_PER_STEP=${MODEL_FLOPS_PER_STEP:-}
PEAK_TFLOPS=${PEAK_TFLOPS:-}
PEAK_PRECISION=${PEAK_PRECISION:-fp16}

if [[ -z "${SQLITE_PATH}" ]]; then
  echo "[error] set SQLITE_PATH=/abs/path/to/file.sqlite" >&2
  exit 2
fi
require_file "${SQLITE_PATH}"
ensure_dir "$(dirname -- "${OUTPUT}")"

CMD=(
  nsys-analyze
  --sqlite "${SQLITE_PATH}"
  --output "${OUTPUT}"
  --format "${FORMAT}"
  --device-id "${DEVICE_ID}"
  --start-ns "${START_NS}"
  --end-ns "${END_NS}"
  --top-k "${TOP_K}"
  --limit "${LIMIT}"
  --iteration-marker "${ITERATION_MARKER}"
  --peak-precision "${PEAK_PRECISION}"
)

if [[ "${PRETTY}" == "1" ]]; then
  CMD+=(--pretty)
fi
if [[ -n "${MODEL_FLOPS_PER_STEP}" ]]; then
  CMD+=(--model-flops-per-step "${MODEL_FLOPS_PER_STEP}")
fi
if [[ -n "${PEAK_TFLOPS}" ]]; then
  CMD+=(--peak-tflops "${PEAK_TFLOPS}")
fi

echo "[run] myutils-profile ${CMD[*]}"
run_myutils_profile "${CMD[@]}"