# Training Scripts

MED recipes for G1-Sim-Dex3. Each script calls [scripts/train.py](../train.py)
with task-specific data and training settings.

## Script Structure

```text
scripts/training_scripts/g1-sim-dex3/
├── kitchen-bread/               # Kitchen-Bread target task
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
and run from the repository root. Download the HDF5 demonstrations from the
[EgoX dataset on Hugging Face](https://huggingface.co/datasets/binghong0727/egox):

```bash
# Download all embodiments for cross-embodiment pretraining.
python scripts/download_data.py

# Or download only the simulated robot for fine-tuning or training from scratch.
python scripts/download_data.py --embodiment g1-sim-dex3

# Check file count and total size without downloading files.
python scripts/download_data.py --embodiment g1-sim-dex3 --dry-run
```

Downloads and recipes default to `data/` under the repository root. The downloader
preserves lowercase Hub paths, for example:

```text
data/
├── g1-sim-dex3/
│   ├── kitchen-bread/
│   │   ├── train/
│   │   └── val/
│   └── kitchen-cleanup/
│       ├── train/
│       └── val/
├── human-dex5/
└── ...
```

To use a different disk or directory, give the downloader `--output-dir` and set
the recipes' `DATA_DIR` to the same location:

```bash
python scripts/download_data.py --output-dir /path/to/egox-data
export DATA_DIR=/path/to/egox-data
```

Dataset directories use lowercase embodiment and task names under `DATA_DIR`,
matching the Hub layout. Use this same layout for existing local datasets.

Pretraining uses the cross-embodiment mixture, excluding the target robot–task
pair. Fine-tuning and training from scratch use
`$DATA_DIR/g1-sim-dex3/kitchen-bread` or `kitchen-cleanup`. Each episode's
`embodiment` attribute selects its [robot configuration](../../src/egox/configs/README.md).

```bash
# Pretrain on the other embodiments and tasks.
bash scripts/training_scripts/g1-sim-dex3/kitchen-bread/pretrain.sh

# Fine-tune; the pretrained checkpoint is the first argument.
bash scripts/training_scripts/g1-sim-dex3/kitchen-bread/finetune.sh \
    /path/to/pretrained/policy_last.ckpt

# Train the target task from scratch.
bash scripts/training_scripts/g1-sim-dex3/kitchen-bread/train-from-scratch.sh
```

Replace `kitchen-bread` with `kitchen-cleanup` in these commands for Cleanup.
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
[rollout and evaluation](../rollout_scripts/readme.md).
