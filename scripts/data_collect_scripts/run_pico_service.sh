#!/usr/bin/env bash
# Foreground, non-GUI XRoboToolkit service from the repository's optional runtime.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PICO_SERVICE_DIR=${EGOX_PICO_SERVICE_DIR:-"$REPO_ROOT/third_party/xrobotoolkit/runtime/opt/apps/roboticsservice"}
if [[ ! -x "$PICO_SERVICE_DIR/RoboticsServiceProcess" ]]; then
    echo "Missing XRoboToolkit service: $PICO_SERVICE_DIR/RoboticsServiceProcess" >&2
    echo 'Install it with: bash scripts/setup.sh --PICO-teleop --name ENV' >&2
    exit 1
fi
PICO_SERVICE_DIR="$(cd -- "$PICO_SERVICE_DIR" && pwd)"
export LD_LIBRARY_PATH="$PICO_SERVICE_DIR:$PICO_SERVICE_DIR/lib:$PICO_SERVICE_DIR/SDK/x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$PICO_SERVICE_DIR"
exec ./RoboticsServiceProcess "$@"
