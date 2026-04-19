#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

VERL_WORKDIR="${VERL_WORKDIR:-/path/to/verl}"
TRAIN_FILES="${TRAIN_FILES:-~/data/gsm8k/train.parquet}"
VAL_FILES="${VAL_FILES:-~/data/gsm8k/test.parquet}"

(
  cd "${VERL_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- \
    python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    trainer.n_gpus_per_node=8
)
