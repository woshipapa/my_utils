#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

SLIME_WORKDIR="${SLIME_WORKDIR:-/path/to/slime}"
SLIME_SCRIPT="${SLIME_SCRIPT:-scripts/run-glm4-9B.sh}"

(
  cd "${SLIME_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- bash "${SLIME_SCRIPT}"
)
