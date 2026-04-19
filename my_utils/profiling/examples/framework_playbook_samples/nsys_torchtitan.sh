#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

TORCHTITAN_WORKDIR="${TORCHTITAN_WORKDIR:-/path/to/torchtitan}"
MODULE="${MODULE:-llama3}"
CONFIG="${CONFIG:-llama3_8b}"

(
  cd "${TORCHTITAN_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- env MODULE="${MODULE}" CONFIG="${CONFIG}" ./run_train.sh
)
