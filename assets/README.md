# Assets

Robot files shared by every example. Files that only one task uses live with that example.

## `robots/g1_dex3/`: Unitree G1 (29 DoF) with Dex3 hands

| File | Contents |
|---|---|
| `g1_body29_hand14.urdf` | Kinematic model for IK and proprioception (no meshes needed) |
| `wbc/` | Whole-body-control policy (TorchScript), with its PD gains, default pose and command ranges |
| `camera_rig.json` | Stereo head camera and two wrist cameras of the `G1-Sim-Dex3` dataset: mounting and intrinsics |
| `usd/` | USD simulation model included in the repository |

The bundled USD files (about 77 MiB) are used by simulation viewers, policy
rollout, and benchmark evaluation. Dataset-based policy training uses the
recorded observations and actions.

The model comes from Unitree's
[Isaac Lab assets](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds/tree/394cf2448f8a9ed815c77c701a761f3d1ff1c8fb),
using the `assets/robots/g1-29dof_wholebody_dex3/` directory in `assets.zip`.
This is the asset version used for the dataset.

Source archive SHA256:
`06fbf14549be3a81e3dbd4a8a019e7f060f48de1556b70f291a18ac8f9ead4b0`.
