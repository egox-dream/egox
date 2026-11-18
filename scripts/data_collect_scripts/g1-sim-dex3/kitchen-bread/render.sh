#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data}"

# Run in the Isaac environment after collection; restore states without stepping physics.
exec python "$REPO_ROOT/examples/kitchen/render_dataset.py" \
    --input "$DATASET_ROOT/g1-sim-dex3/kitchen-bread/train" \
    --output "$DATASET_ROOT/g1-sim-dex3/kitchen-bread/train/processed" \
    "$@"
