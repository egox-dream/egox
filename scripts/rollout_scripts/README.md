# Rollout Scripts

Policy rollout and benchmark evaluation for G1-Sim-Dex3. Each script calls
[scripts/rollout.py](../rollout.py) with the `sim` backend.

## Script Structure

```text
scripts/rollout_scripts/G1-Sim-Dex3/
├── kitchen-Bread/          # Kitchen-Bread target task
│   ├── interactive.sh     # Visible trial_001 with keyboard control
│   └── headless.sh        # Automatic evaluation of all 50 trials
└── kitchen-cleanup/        # Kitchen-Cleanup target task
    ├── interactive.sh
    └── headless.sh
```

## Usage

Follow [Installation](../../README.md#installation) with `--sim`, activate that
environment, and run from the repository root. Set `--model_path` to a
[training run](../training_scripts/README.md#saved-outputs) containing
`policy_last.ckpt`, `all_configs.json`, and `dataset_stats.pkl`.

```bash
# Download robot assets once.
python scripts/download_assets.py

# Interactive rollout; press 2 to start.
bash scripts/rollout_scripts/G1-Sim-Dex3/kitchen-Bread/interactive.sh \
    --model_path /path/to/training/run

# Headless benchmark evaluation.
bash scripts/rollout_scripts/G1-Sim-Dex3/kitchen-Bread/headless.sh \
    --model_path /path/to/training/run
```

Replace `kitchen-Bread` with `kitchen-cleanup` for Cleanup. Interactive mode
requires a desktop session: **1** resets, **2** starts/resumes, **3** pauses policy
commands, and **Q** quits. Each trial waits for **2**.

Append options to override a recipe: `--trials id`, `--trials ood`, or
`--trials trial_001` selects trials; `--video` records camera views. Both modes
default to one worker. Headless mode supports `--workers N`, with a separate
policy and simulator per worker; interactive mode requires one worker.

Use `--checkpoint` to select other weights. Selecting `training_last.ckpt`
uses its embedded configuration and statistics. See
`python scripts/rollout.py sim --help` for all options and
[simulation examples](../../examples/README.md) for scene structure and viewers.
Real-robot deployment requires a
[controller integration](../../src/egox/robots/README.md).

## Saved Outputs

Results are saved under `results/G1-Sim-Dex3/<task>/<mode>/` by default, where
`<mode>` is `interactive` or `headless`. Override the directory with `--output`.

| File | Purpose |
| --- | --- |
| `run.json` | Checkpoint, preprocessing, and evaluation protocol identity |
| `results.json` | Per-trial results, aggregate ID/OOD scores, and missing trials |
| `trial_*/result.json` | Trial scores, success, termination reason, and timing |
| `trial_*/trace.npz` | Recorded simulation states used for scoring |
| `trial_*/simulator.log` | Simulator logs |
| `trial_*/video.mp4` | Camera recording when `--video` is enabled |
| `trial_*/error.json` | Headless execution errors, when present |

Rerunning re-scores completed traces and retries incomplete trials. Choose a new
`--output` when changing the checkpoint, preprocessing, or evaluation protocol.
