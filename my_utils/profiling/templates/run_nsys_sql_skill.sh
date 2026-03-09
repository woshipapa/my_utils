#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nsys_offline_common.sh"

SQLITE_PATH=${SQLITE_PATH:-}
SKILL_NAME=${SKILL_NAME:-top_kernels}
SKILL_PARAMS=${SKILL_PARAMS:-"device_id=0 limit=20"}
OUTPUT=${OUTPUT:-}
PRETTY=${PRETTY:-1}
LIST_ONLY=${LIST_ONLY:-0}

if [[ -z "${SQLITE_PATH}" ]]; then
  echo "[error] set SQLITE_PATH=/abs/path/to/file.sqlite" >&2
  exit 2
fi
require_file "${SQLITE_PATH}"

CMD=(nsys-sql-skill --sqlite "${SQLITE_PATH}")
if [[ "${LIST_ONLY}" == "1" ]]; then
  CMD+=(--list-skills)
else
  CMD+=(--skill "${SKILL_NAME}")
  for kv in ${SKILL_PARAMS}; do
    CMD+=(--param "${kv}")
  done
fi
if [[ "${PRETTY}" == "1" ]]; then
  CMD+=(--pretty)
fi
if [[ -n "${OUTPUT}" ]]; then
  ensure_dir "$(dirname -- "${OUTPUT}")"
  CMD+=(--output "${OUTPUT}")
fi

echo "[run] myutils-profile ${CMD[*]}"
run_myutils_profile "${CMD[@]}"

