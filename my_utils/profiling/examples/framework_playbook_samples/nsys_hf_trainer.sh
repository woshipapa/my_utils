#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

HF_PROJECT_WORKDIR="${HF_PROJECT_WORKDIR:-/path/to/transformers}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

(
  cd "${HF_PROJECT_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- \
    torchrun --nproc_per_node "${NPROC_PER_NODE}" \
    examples/pytorch/summarization/run_summarization.py --fp16
)
