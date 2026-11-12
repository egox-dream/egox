#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DATA_DIR="${DATA_DIR:-unknown}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Pretrain on all data except G1-Sim-Dex3/Kitchen-Cleanup.
# Usage: ./pretrain.sh [training options]; DATA_DIR overrides the data root.

# One directory per embodiment; include only the other simulated task.
DATASET_DIRS=(
    "$DATA_DIR/Human-Dex5"
    "$DATA_DIR/G1-Dex5"
    "$DATA_DIR/xArm7-Dex4"
    "$DATA_DIR/Z1-Tool"
    "$DATA_DIR/LocoMan-Dex2"
    "$DATA_DIR/R1Lite-Gripper"
    "$DATA_DIR/G1-Sim-Dex3/Kitchen-Bread"
)

exec "${PYTHON:-python}" "$REPO_ROOT/scripts/train.py" \
    --dataset_dir "${DATASET_DIRS[@]}" \
    --main_cam_resolution "(1280,480)" \
    --wrist_cam_resolution "(240,240)" \
    --gpu_id 0 \
    --seed 6 \
    --bf16 \
    --wandb_name egox-release \
    --wandb \
    --auto_resume \
    --task_name G1-Sim-Dex3_kitchen-cleanup_pretrain \
    --exclude_signals left_wrist_image right_wrist_image \
    --image_sharing encoder \
    --per_emb_batch_size 48 \
    --lr_tokenizer 1e-4 \
    --lr_action_head 1e-4 \
    --lr_trunk 1e-4 \
    --lr_dream_head 5e-5 \
    --lr_schedule cosine \
    --num_steps 100000 \
    --save_every 5000 \
    "$@"
