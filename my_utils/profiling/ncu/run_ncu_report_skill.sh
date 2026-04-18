#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/ncu_offline_common.sh"

REPORT_PATH=${REPORT_PATH:-}
SKILL_NAME=${SKILL_NAME:-}
SKILL_PARAMS=${SKILL_PARAMS:-}
OUTPUT=${OUTPUT:-}
PRETTY=${PRETTY:-1}
LIST_SKILLS=${LIST_SKILLS:-0}

if [[ -z "${REPORT_PATH}" ]]; then
  echo "[error] REPORT_PATH is required" >&2
  exit 2
fi
require_file "${REPORT_PATH}"

CMD=(ncu-report-skill --report "${REPORT_PATH}")
if [[ "${LIST_SKILLS}" == "1" ]]; then
  CMD+=(--list-skills)
else
  if [[ -z "${SKILL_NAME}" ]]; then
    echo "[error] SKILL_NAME is required when LIST_SKILLS=0" >&2
    exit 2
  fi
  CMD+=(--skill "${SKILL_NAME}")
fi

if [[ -n "${SKILL_PARAMS}" ]]; then
  read -r -a _params <<< "${SKILL_PARAMS}"
  for item in "${_params[@]}"; do
    CMD+=(--param "${item}")
  done
fi

if [[ "${PRETTY}" == "1" ]]; then
  CMD+=(--pretty)
fi

if [[ -n "${OUTPUT}" ]]; then
  ensure_dir "$(dirname "${OUTPUT}")"
  CMD+=(--output "${OUTPUT}")
fi

run_myutils_profile "${CMD[@]}"
