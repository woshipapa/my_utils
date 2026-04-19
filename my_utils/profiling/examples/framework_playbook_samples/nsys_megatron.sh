#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

MEGATRON_WORKDIR="${MEGATRON_WORKDIR:-/path/to/Megatron-LM}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

(
  cd "${MEGATRON_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --no_python \
    python pretrain_gpt.py --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1
)
