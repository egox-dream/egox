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
        ├── bread_supported/    # Bread scene with additional chair supports (viewer default)
        ├── cleanup_base/       # Kitchen-Cleanup scene with the training holder
        ├── cleanup/            # Cleanup benchmark scene with a 64 mm holder opening
        └── assets/             # Shared room, props, milk can, and holder assets
```

Scene manifests retain the original asset-generation hashes. `provenance://`
entries identify source files that are not bundled with this repository.

## Scene Viewers

Use an environment created with `--sim` (see [Installation](../README.md#installation))
and a desktop session. Run from the repository root:

```bash
# Download robot assets once.
python scripts/download_assets.py

# Inspect the kitchen with G1 at the Bread starting pose.
python examples/kitchen/view.py --robot-at bread

# Inspect the kitchen with G1 at the Cleanup starting pose.
python examples/kitchen/view.py --robot-at cleanup
```

The viewer holds the robot's base and joints at the selected pose for scene and
camera inspection; no policy checkpoint is required. Use `--robot-at none` to
view the scene without a robot. Add `--variant baseline` for task props only;
the default `diverse` variant includes extra kitchenware.

**Shift + left-drag** moves props and doors. The **EgoX** panel switches cameras,
pauses/resumes physics, and resets the layout. Edits stay in memory; closing the
window exits the viewer.
