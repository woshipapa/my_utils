#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/ncu_offline_common.sh"

CSV_PATH=${CSV_PATH:-}
METRIC_LIKE=${METRIC_LIKE:-}
KERNEL_LIKE=${KERNEL_LIKE:-%}
TOP_K=${TOP_K:-20}
FORMAT=${FORMAT:-json}
OUTPUT=${OUTPUT:-}
PRETTY=${PRETTY:-1}

if [[ -z "${CSV_PATH}" ]]; then
  echo "[error] CSV_PATH is required" >&2
  exit 2
fi
require_file "${CSV_PATH}"

CMD=(
  ncu-csv-analyze
  --csv "${CSV_PATH}"
  --kernel-like "${KERNEL_LIKE}"
  --top-k "${TOP_K}"
  --format "${FORMAT}"
)

if [[ -n "${METRIC_LIKE}" ]]; then
  CMD+=(--metric-like "${METRIC_LIKE}")
fi

if [[ "${PRETTY}" == "1" ]]; then
  CMD+=(--pretty)
fi

if [[ -n "${OUTPUT}" ]]; then
  ensure_dir "$(dirname "${OUTPUT}")"
  CMD+=(--output "${OUTPUT}")
fi

run_myutils_profile "${CMD[@]}"
