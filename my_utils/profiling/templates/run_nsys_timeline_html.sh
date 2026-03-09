#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nsys_offline_common.sh"

SQLITE_PATH=${SQLITE_PATH:-}
OUTPUT=${OUTPUT:-./nsys_metrics_out/timeline.html}
DEVICE_ID=${DEVICE_ID:--1}
START_NS=${START_NS:--1}
END_NS=${END_NS:--1}
LIMIT=${LIMIT:-100000}
WIDTH_PX=${WIDTH_PX:-1800}

if [[ -z "${SQLITE_PATH}" ]]; then
  echo "[error] set SQLITE_PATH=/abs/path/to/file.sqlite" >&2
  exit 2
fi
require_file "${SQLITE_PATH}"
ensure_dir "$(dirname -- "${OUTPUT}")"

CMD=(
  nsys-timeline-html
  --sqlite "${SQLITE_PATH}"
  --output "${OUTPUT}"
  --device-id "${DEVICE_ID}"
  --start-ns "${START_NS}"
  --end-ns "${END_NS}"
  --limit "${LIMIT}"
  --width-px "${WIDTH_PX}"
)

echo "[run] myutils-profile ${CMD[*]}"
run_myutils_profile "${CMD[@]}"
