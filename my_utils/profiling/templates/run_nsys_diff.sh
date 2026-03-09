#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nsys_offline_common.sh"

BEFORE_SQLITE=${BEFORE_SQLITE:-}
AFTER_SQLITE=${AFTER_SQLITE:-}
OUTPUT=${OUTPUT:-./nsys_metrics_out/nsys_diff.json}
FORMAT=${FORMAT:-json}
PRETTY=${PRETTY:-1}
DEVICE_ID=${DEVICE_ID:--1}
START_NS=${START_NS:--1}
END_NS=${END_NS:--1}
TOP_K=${TOP_K:-20}

if [[ -z "${BEFORE_SQLITE}" ]]; then
  echo "[error] set BEFORE_SQLITE=/abs/path/to/before.sqlite" >&2
  exit 2
fi
if [[ -z "${AFTER_SQLITE}" ]]; then
  echo "[error] set AFTER_SQLITE=/abs/path/to/after.sqlite" >&2
  exit 2
fi
require_file "${BEFORE_SQLITE}"
require_file "${AFTER_SQLITE}"
ensure_dir "$(dirname -- "${OUTPUT}")"

CMD=(
  nsys-diff
  --before-sqlite "${BEFORE_SQLITE}"
  --after-sqlite "${AFTER_SQLITE}"
  --output "${OUTPUT}"
  --format "${FORMAT}"
  --device-id "${DEVICE_ID}"
  --start-ns "${START_NS}"
  --end-ns "${END_NS}"
  --top-k "${TOP_K}"
)

if [[ "${PRETTY}" == "1" ]]; then
  CMD+=(--pretty)
fi

echo "[run] myutils-profile ${CMD[*]}"
run_myutils_profile "${CMD[@]}"
