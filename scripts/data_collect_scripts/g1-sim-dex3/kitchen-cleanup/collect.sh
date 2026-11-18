#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data}"
EGOX_SIM_ARGS=()
if [[ -n "${SIM_PYTHON:-}" ]]; then
    EGOX_SIM_ARGS=(--sim_python "$SIM_PYTHON")
fi

exec python "$REPO_ROOT/scripts/data_collect.py" sim \
    --embodiment G1-Sim-Dex3 \
    --teleoperator pico \
    --task kitchen_cleanup \
    --mode headless \
    --seed 0 \
    "${EGOX_SIM_ARGS[@]}" \
    --output "$DATASET_ROOT/g1-sim-dex3/kitchen-cleanup/train" \
    "$@"
