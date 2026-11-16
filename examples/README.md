# Simulation Examples

Kitchen-Bread and Kitchen-Cleanup scenes, task logic, and evaluation code for G1-Sim-Dex3.

## Code Structure

```text
examples/
└── kitchen/                    # G1-Sim-Dex3 kitchen tasks
    ├── evaluate.py             # Policy trials, parallel workers, scoring, and results
    ├── run_sim.py              # Isaac Sim process and G1 DDS controller
    ├── launch.py               # Start and stop simulator processes
    ├── scene.py                # Scene selection, trial loading, and object resets
    ├── tasks.py                # Simulator-side Bread and Cleanup task logic
    ├── scoring.py              # Task progress, success, and aggregate metrics
    ├── sensor.py               # OOD camera perturbations
    ├── coating.py              # Spreader-to-bread coating transfer model
    ├── holder.py               # Cleanup tool-placement checks
    ├── view.py                 # Interactive scene and robot viewer
    ├── benchmarks/
    │   ├── kitchen_bread/      # 50 fixed Bread trials and evaluation protocol
    │   ├── kitchen_cleanup/    # 50 fixed Cleanup trials and evaluation protocol
    │   └── assets/             # Benchmark objects, materials, and textures
    └── scenes/
        ├── bread/              # Kitchen-Bread benchmark scene
        ├── bread_supported/    # Bread scene with additional chair supports (authored-scene preview)
        ├── cleanup_base/       # Kitchen-Cleanup scene with the training holder
        ├── cleanup/            # Cleanup benchmark scene with a 64 mm holder opening
        └── assets/             # Shared room, props, milk can, and holder assets
```

Scene manifests retain the original asset-generation hashes. `provenance://`
entries identify source files that are not bundled with this repository.

## Scene Viewers

Use an environment created with `--sim` (see [Installation](../README.md#installation))
and a desktop session. The [G1 robot model](../assets/README.md) is included in the
repository. Run from the repository root:

```bash
# Inspect the first Bread rollout trial.
python examples/kitchen/view.py --robot-at bread

# Inspect the first Cleanup rollout trial, including its utensil holder.
python examples/kitchen/view.py --robot-at cleanup

# Inspect a specific rollout trial.
python examples/kitchen/view.py --trial examples/kitchen/benchmarks/kitchen_cleanup/trial_010
```

By default, the viewer loads the selected task's `trial_001` using the same scene
USD, appearance overlay, saved object and door positions, camera rig, and render
settings as rollout. Cleanup includes the benchmark utensil holder with its
64 mm opening. The viewer holds the robot's base and joints at the recorded pose
for inspection; it does not run the controller, policy, or task scoring. No policy
checkpoint is required. Use `--robot-at none` to hide the robot; combine it with
`--trial` to select a particular task's scene without a robot.

For an authored-scene preview without the trial overlay and saved object layout,
use `--variant diverse` or `--scene-usd PATH`. `--variant baseline` shows the Bread
scene with task props only; Cleanup has no baseline variant. These overrides
cannot be combined with `--trial`.

**Shift + left-drag** moves props and doors. The **EgoX** panel switches cameras,
pauses/resumes physics, and resets the layout. Edits stay in memory; closing the
window exits the viewer.
