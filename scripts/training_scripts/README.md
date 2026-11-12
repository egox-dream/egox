# Training Scripts

MXT+ recipes for G1-Sim-Dex3. Each script calls [scripts/train.py](../train.py)
with task-specific data and training settings.

## Script Structure

```text
scripts/training_scripts/G1-Sim-Dex3/
├── kitchen-Bread/               # Kitchen-Bread target task
│   ├── pretrain.sh              # Cross-embodiment pretraining
│   ├── finetune.sh              # Adapt pretrained weights to the target task
│   └── train-from-scratch.sh    # Train on the target task without pretrained weights
└── kitchen-cleanup/             # Kitchen-Cleanup target task
    ├── pretrain.sh
    ├── finetune.sh
    └── train-from-scratch.sh
```

## Usage

Follow [Installation](../../README.md#installation), activate your environment,
and run from the repository root. The dataset is not yet released; these recipes
require compatible HDF5 demonstrations. The default data root is the placeholder
`unknown`; set `DATA_DIR` to your actual dataset root before training:

```bash
export DATA_DIR=/path/to/dataset
```

Pretraining uses the cross-embodiment mixture, excluding the target robot–task
pair. Fine-tuning and training from scratch use
`$DATA_DIR/G1-Sim-Dex3/Kitchen-Bread` or `Kitchen-Cleanup`. Each episode's
`embodiment` attribute selects its [robot configuration](../../src/egox/configs/README.md).

```bash
# Pretrain on the other embodiments and tasks.
bash scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/pretrain.sh

# Fine-tune; the pretrained checkpoint is the first argument.
bash scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/finetune.sh \
    /path/to/pretrained/policy_last.ckpt

# Train the target task from scratch.
bash scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/train-from-scratch.sh
```

Replace `kitchen-Bread` with `kitchen-cleanup` in these commands for Cleanup.
Append options to override a recipe, for example `--gpu_id 1 --num_steps 10000
--no-wandb`. Recipes enable W&B logging and resume existing runs automatically;
use `--no-auto_resume` for a fresh, timestamped run. Resuming takes precedence
over loading pretrained weights.

See [shared defaults](../../src/egox/training/config.py) and
`python scripts/train.py --help` for all options.

## Saved Outputs

Runs are saved under `ckpt/_<task_name>/` by default. Each recipe sets its own
run name; override it with `--task_name`, or change the directory prefix with
`--ckpt_dir`.

| File | Purpose |
| --- | --- |
| `training_last.ckpt` | Full state for resuming: model, optimizer, step, RNG, data-loader positions, configuration, and normalization statistics |
| `policy_last.ckpt` | Latest snapshot's model weights for fine-tuning or inference |
| `policy_step_<step>_seed_<seed>.ckpt` | Model weights saved at numbered steps |
| `all_configs.json` | Resolved training/model configuration and completed step count |
| `dataset_stats.pkl` | Dataset normalization statistics used by the policy |

`training_last.ckpt`, `policy_last.ckpt`, and both metadata files are saved every
`--snapshot_every` steps (default 500) and at normal completion. Numbered weights
are saved every `--save_every` steps (5000 in these recipes). Resuming in the same
run directory refreshes missing or stale metadata exports.

Keep `all_configs.json` and `dataset_stats.pkl` alongside the policy weights for
[rollout and evaluation](../rollout_scripts/README.md).
