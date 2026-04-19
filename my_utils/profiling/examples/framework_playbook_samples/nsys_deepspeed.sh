#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

PROJECT_WORKDIR="${PROJECT_WORKDIR:-/path/to/project}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
DS_CONFIG="${DS_CONFIG:-ds_config.json}"
NUM_GPUS="${NUM_GPUS:-8}"

(
  cd "${PROJECT_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- \
    deepspeed --num_gpus="${NUM_GPUS}" "${TRAIN_SCRIPT}" --deepspeed "${DS_CONFIG}"
)
