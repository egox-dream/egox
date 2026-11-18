#!/usr/bin/env bash
set -euo pipefail

# Create a training, inference, collection and visualization environment, optionally including simulation.
usage() {
    echo "Usage: bash scripts/setup.sh [--sim] [--PICO-teleop] [--name ENV_NAME | -n ENV_NAME | ENV_NAME]"
    echo "--name, -n: choose the Conda environment name (also accepts --name=ENV_NAME)."
    echo "Default: egox_policy. Existing environments support --PICO-teleop without --sim."
    echo "Includes training, inference, collection and visualization dependencies with PyTorch 2.14.1 / CUDA 12.6."
    echo "EGOX_TORCH_INDEX_URL overrides the default PyTorch wheel index."
    echo "--sim: adds Isaac Sim and G1 control with PyTorch 2.7 / CUDA 12.8."
    echo "--PICO-teleop: installs the Python SDK/PC service and downloads the verified official headset APK."
    echo "PICO files live under third_party/xrobotoolkit; PC setup supports Ubuntu 22.04/24.04 x86_64."
    echo "Install the APK on the headset with: python third_party/xrobotoolkit/install.py --install-headset DEVICE"
    echo "Simulation requires Linux, an NVIDIA RTX GPU, Git, cmake and build-essential."
    echo "Using --sim accepts NVIDIA's Isaac Sim EULA. Setup does not launch Isaac."
}

EGOX_SIM=0
EGOX_PICO=0
EGOX_ENV_NAME=""
set_env_name() {
    if [[ -z "$1" || "$1" == -* ]]; then
        echo "Provide a nonempty environment name that does not start with '-'." >&2
        usage >&2
        exit 2
    fi
    if [[ -n "$EGOX_ENV_NAME" ]]; then
        echo "Specify only one environment name." >&2
        usage >&2
        exit 2
    fi
    EGOX_ENV_NAME="$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim) EGOX_SIM=1 ;;
        --PICO-teleop) EGOX_PICO=1 ;;
        --name|-n)
            if [[ $# -lt 2 ]]; then
                echo "Missing environment name after $1." >&2
                usage >&2
                exit 2
            fi
            set_env_name "$2"
            shift
            ;;
        --name=*) set_env_name "${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) set_env_name "$1" ;;
    esac
    shift
done
EGOX_ENV_NAME="${EGOX_ENV_NAME:-egox_policy}"
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required. Install Miniconda or Miniforge, then rerun this script." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Existing environments may add PICO without recreating their training/simulation stack.
EGOX_CONDA_BASE="$(conda info --base)"
EGOX_ENV_EXISTS="$(conda info --json | "$EGOX_CONDA_BASE/bin/python" -c '
import json, sys
from pathlib import Path
info, name = json.load(sys.stdin), sys.argv[1]
if name in {"base", "root"}:
    sys.exit("Choose an environment other than base/root.")
exists = any(Path(path).name == name for path in info["envs"])
if not exists and any((Path(folder) / name).exists() for folder in info["envs_dirs"]):
    sys.exit(f"Directory {name!r} already exists but is not a registered Conda environment.")
print(int(exists))
' "$EGOX_ENV_NAME")"
EGOX_PYTHON=(conda run --no-capture-output --name "$EGOX_ENV_NAME" python)
if [[ "$EGOX_PICO" == 1 ]]; then
    "$EGOX_CONDA_BASE/bin/python" "$REPO_ROOT/third_party/xrobotoolkit/install.py" --check
fi
if [[ "$EGOX_ENV_EXISTS" == 1 ]]; then
    if [[ "$EGOX_PICO" != 1 || "$EGOX_SIM" == 1 ]]; then
        echo "Environment '$EGOX_ENV_NAME' already exists. Choose a new name, or use --PICO-teleop without --sim to add PICO." >&2
        exit 1
    fi
    "${EGOX_PYTHON[@]}" -m pip install -e "$REPO_ROOT[collection]"
    "${EGOX_PYTHON[@]}" "$REPO_ROOT/third_party/xrobotoolkit/install.py"
    echo "PICO PC setup and APK download complete. Activate with: conda activate $EGOX_ENV_NAME"
    exit 0
fi

if [[ "$EGOX_SIM" == 1 && ! -f "$REPO_ROOT/third_party/IsaacLab/source/isaaclab/setup.py" ]]; then
    git -C "$REPO_ROOT" submodule update --init third_party/IsaacLab
fi
conda create --yes --name "$EGOX_ENV_NAME" --override-channels --channel conda-forge \
    python=3.11 pip
if [[ "$EGOX_SIM" == 1 ]]; then
    conda env config vars set --name "$EGOX_ENV_NAME" OMNI_KIT_ACCEPT_EULA=YES
    # Isaac pins packaging 23.0; newer wheel releases require packaging >=24.
    "${EGOX_PYTHON[@]}" -m pip install 'setuptools<81' wheel==0.43.0 toml
    "${EGOX_PYTHON[@]}" -m pip install --no-build-isolation flatdict==4.0.1
    "${EGOX_PYTHON[@]}" -m pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    # Core Isaac Lab is sufficient; optional RL packages are not used.
    "${EGOX_PYTHON[@]}" -m pip install --no-deps --no-build-isolation \
        -e "$REPO_ROOT/third_party/IsaacLab/source/isaaclab"
    "${EGOX_PYTHON[@]}" - "$REPO_ROOT" <<'PY'
from importlib.metadata import requires
import subprocess
import sys

# Isaac Lab's unused livestreaming dependency conflicts with Isaac Sim's FastAPI.
# Keep Isaac Sim's Starlette version and resolve all other requirements together.
requirements = [r for r in requires('isaaclab') if not r.startswith('starlette')]
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
    'isaacsim[all,extscache]==5.1.0', '--extra-index-url', 'https://pypi.nvidia.com',
    'torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0',
    '-e', sys.argv[1] + '[training,inference,collection,simulation,visualization,test]', *requirements])
PY
else
    EGOX_TORCH_INDEX_URL="${EGOX_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
    "${EGOX_PYTHON[@]}" -m pip install \
        torch==2.14.1 torchvision==0.29.1 --index-url "$EGOX_TORCH_INDEX_URL"
    "${EGOX_PYTHON[@]}" -m pip install -e "$REPO_ROOT[training,inference,collection,visualization,test]"
fi

if [[ "$EGOX_PICO" == 1 ]]; then
    "${EGOX_PYTHON[@]}" "$REPO_ROOT/third_party/xrobotoolkit/install.py"
fi

"${EGOX_PYTHON[@]}" - "$EGOX_SIM" <<'PY'
from importlib.metadata import version
import subprocess
import sys

check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
print(check.stdout, end='')
print(check.stderr, end='', file=sys.stderr)
if check.returncode:
    if sys.argv[1] != '1':
        raise SystemExit(check.returncode)
    expected = (f'isaaclab {version("isaaclab")} has requirement starlette==0.49.1, '
                f'but you have starlette {version("starlette")}.')
    if check.stderr or check.stdout.strip() != expected:
        raise SystemExit(check.returncode)
    print('Known Isaac Lab livestreaming conflict; local simulation uses Isaac Sim\'s Starlette.')

import torch, torchvision, scipy, cyclonedds, h5py, cv2, zmq, av
from egox.inference import MEDPolicy
from egox.teleoperator.data_collecter import DataCollector
from egox.training.trainer import train
av.CodecContext.create('libx264', 'w')
print('Collection dependencies and H.264 encoder available.')
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "BF16:", torch.cuda.is_bf16_supported())
if sys.argv[1] == '1':
    # Pinocchio's Assimp must load before Isaac's incompatible build.
    import pinocchio, numpy
    from isaaclab.app import AppLauncher
    from egox.robots.g1_sim_dex3.controller import G1Dex3SimController
    controller = G1Dex3SimController()
    for package in ('isaacsim', 'isaaclab', 'numpy', 'pin', 'cyclonedds'):
        print(package + ':', version(package))
    print('G1 controller and WBC weights loaded.')
PY
echo "Setup complete. Activate with: conda activate $EGOX_ENV_NAME"
if [[ "$EGOX_SIM" == 1 ]]; then
    echo "Bundled robot assets: $REPO_ROOT/assets/robots/g1_dex3/usd"
fi
