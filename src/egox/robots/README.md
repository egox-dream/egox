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
| [g1_sim_dex3/teleop.py](g1_sim_dex3/teleop.py) | G1 head calibration, hand targets and locomotion mapping for collection |
| [../teleoperator/base.py](../teleoperator/base.py), [pico.py](../teleoperator/pico.py) | Headset interface and PICO tracking adapter |
| [../teleoperator/data_collecter.py](../teleoperator/data_collecter.py) | Collection lifecycle, state/RGB episode writers and preview worker |
| [../teleoperator/pico_video.py](../teleoperator/pico_video.py) | Remote Vision control and H.264 stream to the headset |

Scenes, task settings and scoring live in `examples/`. G1's controller runs on
Isaac's main loop, preserving physics/WBC timing. See the [simulation examples](../../../examples/README.md).

[`scripts/data_collect.py`](../../../scripts/data_collect.py) launches a simulator
with `sim` or uses an existing controller with `connect`. PICO supplies
`TeleopFrame` input; `G1Dex3Teleop` maps it into actions for the same `RobotClient`
used by policy rollout. The asynchronous writer records pre-action state and
commands, while an independent preview client reads the optional 24 Hz JPEG
stream. [`render_dataset.py`](../../../examples/kitchen/render_dataset.py) later
renders full training RGB from the saved states. See the
[collection guide](../../../scripts/data_collect_scripts/readme.md) for setup and controls.

Only G1-Sim-Dex3 has a collection mapper in this release. New robots supply their
own mapper and controller; the Apple Vision Pro headset adapter is reserved and
not implemented.

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
