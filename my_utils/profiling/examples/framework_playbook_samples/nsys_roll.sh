#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NSYS_WRAPPER="${REPO_ROOT}/my_utils/profiling/templates/run_nsys_quick.sh"

ROLL_WORKDIR="${ROLL_WORKDIR:-/path/to/ROLL}"
ROLL_SCRIPT="${ROLL_SCRIPT:-examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh}"

(
  cd "${ROLL_WORKDIR}"
  bash "${NSYS_WRAPPER}" -- bash "${ROLL_SCRIPT}"
)
