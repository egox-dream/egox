#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
EGOX_SIM_ARGS=()
if [[ -n "${SIM_PYTHON:-}" ]]; then
    EGOX_SIM_ARGS=(--python_executable "$SIM_PYTHON")
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Inspect one Bread trial with a window. Select any trained policy with --model_path.
exec "${PYTHON:-python}" "$REPO_ROOT/scripts/rollout.py" sim \
    --mode interactive \
    --benchmark kitchen_bread \
    --checkpoint policy_last.ckpt \
    "${EGOX_SIM_ARGS[@]}" \
    --output "$REPO_ROOT/results/G1-Sim-Dex3/kitchen-Bread/interactive" \
    --device cuda:0 \
    --workers 1 \
    --trials trial_001 \
    "$@"
