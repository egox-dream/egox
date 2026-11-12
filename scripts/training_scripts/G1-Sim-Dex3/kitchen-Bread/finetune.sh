#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DATA_DIR="${DATA_DIR:-unknown}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Fine-tune on this task with wrist cameras and pretrained weights.
# Usage: ./finetune.sh PRETRAINED_CHECKPOINT [training options]
if [[ $# -eq 0 || -z "$1" || "$1" == -* ]]; then
    echo "Usage: $0 PRETRAINED_CHECKPOINT [training options]" >&2
    exit 2
fi
pretrained_path="$1"
shift

exec "${PYTHON:-python}" "$REPO_ROOT/scripts/train.py" \
    --dataset_dir "$DATA_DIR/G1-Sim-Dex3/Kitchen-Bread" \
    --main_cam_resolution "(1280,480)" \
    --wrist_cam_resolution "(240,240)" \
    --gpu_id 0 \
    --seed 6 \
    --bf16 \
    --wandb_name egox-release \
    --wandb \
    --auto_resume \
    --task_name G1-Sim-Dex3_kitchen-Bread_finetune2 \
    --pretrained_path "$pretrained_path" \
    --image_sharing encoder \
    --batch_size 32 \
    --lr_tokenizer 5e-5 \
    --lr_action_head 5e-5 \
    --lr_trunk 5e-5 \
    --lr_dream_head 5e-6 \
    --lr_schedule constant \
    --num_steps 100000 \
    --save_every 5000 \
    "$@"
