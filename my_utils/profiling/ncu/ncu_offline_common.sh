#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for offline NCU CSV post-processing templates.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)}
MY_UTILS_ROOT=${MY_UTILS_ROOT:-${REPO_ROOT}/third_party/my_utils}

if [[ -d "${MY_UTILS_ROOT}" ]]; then
  export PYTHONPATH="${PYTHONPATH:-}:${MY_UTILS_ROOT}"
fi

PYTHON_BIN=${PYTHON_BIN:-python}

function require_file() {
  local file_path="$1"
  if [[ ! -f "${file_path}" ]]; then
    echo "[error] file not found: ${file_path}" >&2
    exit 2
  fi
}

function ensure_dir() {
  local dir_path="$1"
  mkdir -p "${dir_path}"
}

function run_myutils_profile() {
  "${PYTHON_BIN}" -m my_utils.profiling.cli "$@"
}
