# Assets

Robot files shared by every example. Files that only one task uses live with that example.

## `robots/g1_dex3/`: Unitree G1 (29 DoF) with Dex3 hands

| File | Contents |
|---|---|
| `g1_body29_hand14.urdf` | Kinematic model for IK and proprioception (no meshes needed) |
| `wbc/` | Whole-body-control policy (TorchScript), with its PD gains, default pose and command ranges |
| `camera_rig.json` | Stereo head camera and two wrist cameras of the `G1-Sim-Dex3` dataset: mounting and intrinsics |
| `usd/` | Simulation model; downloaded, not stored in git |

Get the simulation model with:

```bash
python scripts/download_assets.py
```

It downloads Unitree's [Isaac Lab asset archive](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds)
(1.3 GB) once to `~/.cache/egox_sim/`, checks it against the pinned version, and extracts the robot
(78 MB). Without internet access, first copy the archive to `~/.cache/egox_sim/unitree_assets.zip`.
