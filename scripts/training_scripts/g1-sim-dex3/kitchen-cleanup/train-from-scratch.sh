#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/yaru/egox/egox-open-source/verify/egox/data}"

# Train this task from scratch, including wrist cameras.
# Usage: ./train-from-scratch.sh [training options]

exec python "$REPO_ROOT/scripts/train.py" \
    --dataset_dir "$DATA_DIR/g1-sim-dex3/kitchen-cleanup" \
    --main_cam_resolution "(1280,480)" \
    --wrist_cam_resolution "(240,240)" \
    --gpu_id 0 \
    --seed 6 \
    --bf16 \
    --wandb_name egox \
    --wandb \
    --auto_resume \
    --task_name G1-Sim-Dex3_kitchen-cleanup_train-from-scratch \
    --image_sharing none \
    --batch_size 32 \
    --lr_tokenizer 1e-4 \
    --lr_action_head 1e-4 \
    --lr_trunk 1e-4 \
    --lr_dream_head 5e-6 \
    --lr_schedule cosine \
    --num_steps 100000 \
    --save_every 5000 \
    "$@"
