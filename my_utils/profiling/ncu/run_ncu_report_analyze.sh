#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/ncu_offline_common.sh"

REPORT_PATH=${REPORT_PATH:-}
METRIC_LIKE=${METRIC_LIKE:-}
KERNEL_LIKE=${KERNEL_LIKE:-%}
TOP_K=${TOP_K:-20}
FORMAT=${FORMAT:-json}
OUTPUT=${OUTPUT:-}
PRETTY=${PRETTY:-1}
INCLUDE_ALL_METRICS=${INCLUDE_ALL_METRICS:-1}
ALL_METRICS_LIMIT=${ALL_METRICS_LIMIT:-20000}

if [[ -z "${REPORT_PATH}" ]]; then
  echo "[error] REPORT_PATH is required" >&2
  exit 2
fi
require_file "${REPORT_PATH}"

CMD=(
  ncu-report-analyze
  --report "${REPORT_PATH}"
  --kernel-like "${KERNEL_LIKE}"
  --top-k "${TOP_K}"
  --format "${FORMAT}"
  --all-metrics-limit "${ALL_METRICS_LIMIT}"
)

if [[ -n "${METRIC_LIKE}" ]]; then
  CMD+=(--metric-like "${METRIC_LIKE}")
fi

if [[ "${INCLUDE_ALL_METRICS}" == "1" ]]; then
  CMD+=(--include-all-metrics)
else
  CMD+=(--no-include-all-metrics)
fi

if [[ "${PRETTY}" == "1" ]]; then
  CMD+=(--pretty)
fi

if [[ -n "${OUTPUT}" ]]; then
  ensure_dir "$(dirname "${OUTPUT}")"
  CMD+=(--output "${OUTPUT}")
fi

run_myutils_profile "${CMD[@]}"
