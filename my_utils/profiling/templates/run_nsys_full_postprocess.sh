#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nsys_offline_common.sh"

# Required
SQLITE_PATH=${SQLITE_PATH:-}

# Optional
OUT_DIR=${OUT_DIR:-./nsys_metrics_out}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-nsys_report}
DEVICE_ID=${DEVICE_ID:--1}
START_NS=${START_NS:--1}
END_NS=${END_NS:--1}
TOP_K=${TOP_K:-20}
LIMIT=${LIMIT:-500000}
ITERATION_MARKER=${ITERATION_MARKER:-sample_0}
SKILL_NAME=${SKILL_NAME:-top_kernels}
SKILL_PARAMS=${SKILL_PARAMS:-}
ENABLE_ANALYZE=${ENABLE_ANALYZE:-1}
ENABLE_EXPORT=${ENABLE_EXPORT:-1}
ENABLE_TIMELINE=${ENABLE_TIMELINE:-1}
ENABLE_SQL_SKILL=${ENABLE_SQL_SKILL:-1}
EXPORT_JSON=${EXPORT_JSON:-1}
EXPORT_CSV=${EXPORT_CSV:-1}
ATTACH_ITERATION=${ATTACH_ITERATION:-1}

if [[ -z "${SQLITE_PATH}" ]]; then
  echo "[error] set SQLITE_PATH=/abs/path/to/file.sqlite" >&2
  exit 2
fi
require_file "${SQLITE_PATH}"
ensure_dir "${OUT_DIR}"

if [[ -z "${SKILL_PARAMS}" ]]; then
  if [[ "${DEVICE_ID}" == "-1" ]]; then
    SKILL_PARAMS="limit=${TOP_K}"
  else
    SKILL_PARAMS="device_id=${DEVICE_ID} limit=${TOP_K}"
  fi
fi

if [[ "${ENABLE_ANALYZE}" == "1" ]]; then
  ANALYZE_OUT="${OUT_DIR}/${OUTPUT_PREFIX}_analyze.json"
  CMD=(
    nsys-analyze
    --sqlite "${SQLITE_PATH}"
    --output "${ANALYZE_OUT}"
    --format json
    --pretty
    --device-id "${DEVICE_ID}"
    --start-ns "${START_NS}"
    --end-ns "${END_NS}"
    --top-k "${TOP_K}"
    --limit "${LIMIT}"
    --iteration-marker "${ITERATION_MARKER}"
  )
  echo "[run] myutils-profile ${CMD[*]}"
  run_myutils_profile "${CMD[@]}"
fi

if [[ "${ENABLE_EXPORT}" == "1" ]]; then
  if [[ "${EXPORT_JSON}" == "1" ]]; then
    EXPORT_JSON_OUT="${OUT_DIR}/${OUTPUT_PREFIX}_kernels_flat.json"
    CMD=(
      nsys-export
      --sqlite "${SQLITE_PATH}"
      --output "${EXPORT_JSON_OUT}"
      --format json
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
  fi

  if [[ "${EXPORT_CSV}" == "1" ]]; then
    EXPORT_CSV_OUT="${OUT_DIR}/${OUTPUT_PREFIX}_kernels_flat.csv"
    CMD=(
      nsys-export
      --sqlite "${SQLITE_PATH}"
      --output "${EXPORT_CSV_OUT}"
      --format csv
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
  fi
fi

if [[ "${ENABLE_TIMELINE}" == "1" ]]; then
  TIMELINE_OUT="${OUT_DIR}/${OUTPUT_PREFIX}_timeline.html"
  CMD=(
    nsys-timeline-html
    --sqlite "${SQLITE_PATH}"
    --output "${TIMELINE_OUT}"
    --device-id "${DEVICE_ID}"
    --start-ns "${START_NS}"
    --end-ns "${END_NS}"
    --limit "${LIMIT}"
  )
  echo "[run] myutils-profile ${CMD[*]}"
  run_myutils_profile "${CMD[@]}"
fi

if [[ "${ENABLE_SQL_SKILL}" == "1" ]]; then
  SKILL_SAFE_NAME=$(echo "${SKILL_NAME}" | tr ' ' '_' | tr -cd '[:alnum:]_-.')
  SKILL_OUT="${OUT_DIR}/${OUTPUT_PREFIX}_skill_${SKILL_SAFE_NAME}.json"
  CMD=(
    nsys-sql-skill
    --sqlite "${SQLITE_PATH}"
    --skill "${SKILL_NAME}"
    --pretty
    --output "${SKILL_OUT}"
  )
  for kv in ${SKILL_PARAMS}; do
    CMD+=(--param "${kv}")
  done
  echo "[run] myutils-profile ${CMD[*]}"
  run_myutils_profile "${CMD[@]}"
fi

echo "[done] outputs in ${OUT_DIR}"
