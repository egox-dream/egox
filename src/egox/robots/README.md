# Robot control

Only G1-Sim-Dex3 is integrated in the current release, for simulation. Collection
and policy rollout share the `RobotClient` interface. Later embodiments, including
G1-Dex5, supply their own controllers. Training code is independent.

| Module | Responsibility |
| --- | --- |
| [channel.py](channel.py) | DDS commands, state, request IDs and image references |
| [service.py](service.py) | Shared `RobotClient` and `ControllerService`, including image reconstruction |
| [images.py](images.py) | Exact camera snapshots over local ZeroMQ IPC |
| [../inference/rollout.py](../inference/rollout.py) | Shared policy/action loop for the general CLI and task evaluation |
| [../sim/scene.py](../sim/scene.py) | Shared scene setup, camera placement and snapshots |
| [g1_sim_dex3/controller.py](g1_sim_dex3/controller.py) | IK, hand mapping, WBC and proprioception |
| [g1_sim_dex3/environment.py](g1_sim_dex3/environment.py) | Robot physics, cameras, timing and state restoration |
| [g1_sim_dex3/description.py](g1_sim_dex3/description.py) | Joint names and robot asset locations |

Scenes, task settings and scoring live in `examples/`. G1's controller runs on
Isaac's main loop, preserving physics/WBC timing. See the [simulation examples](../../../examples/README.md).

## Images and robot state

DDS carries commands and robot state. `ControllerService` separates the camera
arrays in `obs/*_image` and the top-level preview `jpeg` into local ZeroMQ
snapshots. `RobotClient` retrieves them and returns the same observation or
preview dictionary to callers. Policy RGB stays uncompressed; collection preview
JPEGs keep their existing encoding.

Each image reference identifies an immutable snapshot by endpoint and sequence.
The client retrieves that exact snapshot; an expired reference raises an error
instead of substituting a newer frame. State-only collection retains its
pre-action snapshots, timestamps, pause/reset behavior, and separate optional
preview. Its action replies contain no images.

For real hardware, pass `camera_source=read_cameras` to `RobotClient`.
`read_cameras(observation)` returns a dictionary of camera arrays, such as
`{"main_image": rgb}`, acquired directly through the camera SDK or OpenCV.
It runs for `observe`, `reset`, and `step` replies containing `obs`; the caller
owns camera startup and cleanup. A local camera source cannot be combined with
controller RGB images; separate JPEG previews remain available. Hardware adapters
should use capture timestamps to pair camera frames with robot state.

PyZMQ is loaded only when simulator images or previews use ZeroMQ. For an
existing environment, install it with:

```bash
python -m pip install "pyzmq>=26"
```

Update the client and controller service together; older clients cannot resolve
the new image references.
