# Robot control

Only G1-Sim-Dex3 is integrated in the current release, for simulation. Collection
and policy rollout share the `RobotClient` interface. Later embodiments, including
G1-Dex5, supply their own controllers. Training code is independent.

| Module | Responsibility |
| --- | --- |
| [channel.py](channel.py) | DDS transport, request IDs and NumPy payloads |
| [service.py](service.py) | Shared `RobotClient` and `ControllerService` |
| [../inference/rollout.py](../inference/rollout.py) | Shared policy/action loop for the general CLI and task evaluation |
| [../sim/scene.py](../sim/scene.py) | Shared scene setup, camera placement and snapshots |
| [g1_sim_dex3/controller.py](g1_sim_dex3/controller.py) | IK, hand mapping, WBC and proprioception |
| [g1_sim_dex3/environment.py](g1_sim_dex3/environment.py) | Robot physics, cameras, timing and state restoration |
| [g1_sim_dex3/description.py](g1_sim_dex3/description.py) | Joint names and robot asset locations |

Scenes, task settings and scoring live in `examples/`. G1's controller runs on
Isaac's main loop, preserving physics/WBC timing. See the [simulation examples](../../../examples/README.md).
