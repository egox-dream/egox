#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
EGOX_SIM_ARGS=()
if [[ -n "${SIM_PYTHON:-}" ]]; then
    EGOX_SIM_ARGS=(--python_executable "$SIM_PYTHON")
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Evaluate Cleanup headless. Select --model_path; increase --workers to run trials in parallel.
exec "${PYTHON:-python}" "$REPO_ROOT/scripts/rollout.py" sim \
    --mode headless \
    --benchmark kitchen_cleanup \
    --checkpoint policy_last.ckpt \
    "${EGOX_SIM_ARGS[@]}" \
    --output "$REPO_ROOT/results/G1-Sim-Dex3/kitchen-cleanup/headless" \
    --device cuda:0 \
    --workers 1 \
    --trials all \
    "$@"
