#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nsys_offline_common.sh"

SQLITE_PATH=${SQLITE_PATH:-}
OUTPUT=${OUTPUT:-./nsys_metrics_out/kernels_flat.json}
FORMAT=${FORMAT:-json}
DEVICE_ID=${DEVICE_ID:--1}
START_NS=${START_NS:--1}
END_NS=${END_NS:--1}
LIMIT=${LIMIT:-500000}
ATTACH_ITERATION=${ATTACH_ITERATION:-0}
ITERATION_MARKER=${ITERATION_MARKER:-sample_0}

if [[ -z "${SQLITE_PATH}" ]]; then
  echo "[error] set SQLITE_PATH=/abs/path/to/file.sqlite" >&2
  exit 2
fi
require_file "${SQLITE_PATH}"
ensure_dir "$(dirname -- "${OUTPUT}")"

CMD=(
  nsys-export
  --sqlite "${SQLITE_PATH}"
  --output "${OUTPUT}"
  --format "${FORMAT}"
  --device-id "${DEVICE_ID}"
  --start-ns "${START_NS}"
  --end-ns "${END_NS}"
  --limit "${LIMIT}"
)

if [[ "${ATTACH_ITERATION}" == "1" ]]; then
  CMD+=(--attach-iteration --iteration-marker "${ITERATION_MARKER}")
fi

echo "[run] myutils-profile ${CMD[*]}"
run_myutils_profile "${CMD[@]}"
