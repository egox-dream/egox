# PICO data collection

PICO demonstration collection and offline dataset rendering for G1-Sim-Dex3.

## Script Structure

```text
scripts/data_collect_scripts/
├── run_pico_service.sh         # Start the XRoboToolkit PC service
└── g1-sim-dex3/
    ├── kitchen-bread/          # Kitchen-Bread target task
    │   ├── collect.sh         # Collect raw .state.h5 episodes (headless by default)
    │   └── render.sh          # Generate training HDF5 files and camera videos
    └── kitchen-cleanup/        # Kitchen-Cleanup target task
        ├── collect.sh
        └── render.sh
```

`collect.sh` calls [scripts/data_collect.py](../data_collect.py) with the `sim`
backend; `render.sh` calls [render_dataset.py](../../examples/kitchen/render_dataset.py).

## Usage

Create an environment with simulation, recording, headset video and the PICO
SDK/service:

```bash
bash scripts/setup.sh --sim --PICO-teleop --name egox
conda activate egox
```

This installs the collection dependencies, the pinned `xrobotoolkit_sdk` Python
package and the PC service while retaining the existing training and Isaac stack.
The SDK is installed into the selected Python environment; native bindings must
match its Python ABI. The service is downloaded and extracted locally without
`sudo` and is started separately below.

Install the matching headset APK manually from the official
[XRoboToolkit Unity client releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases).

Start the PC service in another terminal:

```bash
bash scripts/data_collect_scripts/run_pico_service.sh
```

The default service directory is
`third_party/xrobotoolkit/runtime/opt/apps/roboticsservice` under this repository.
`EGOX_PICO_SERVICE_DIR` selects another installation. On the headset, set PC
Service to the workstation's LAN IP, enable Head, Controller and Send, and disable
**Switch w/ A Button**. 

Collect either task:

```bash
bash scripts/data_collect_scripts/g1-sim-dex3/kitchen-bread/collect.sh
bash scripts/data_collect_scripts/g1-sim-dex3/kitchen-cleanup/collect.sh
```

Add `--pico_host HEADSET_IP` for the Remote
Vision stream; in Listen, enter the workstation IP and select ZEDMINI/H.264.
Collection uses realtime physics and a cheaper 24 Hz main stereo preview.
Wait for the terminal's `X: reset | A: calibrate...` prompt before pressing Listen;
Isaac startup can take several minutes. A connection alone does not mean images
are ready: `PICO video first frame sent` confirms the sender has produced video.
If the connection was closed, turn Listen off and on to restart Pico's receiver.

| Button | Operation |
| --- | --- |
| X | Reset and sample new object positions |
| A, then A | Calibrate head heading/height, then engage and record |
| B / A | Pause / resume |
| Y | Save the episode |
| Left menu | Discard the episode |

Lost/stale tracking pauses commands; press A after tracking recovers. Ctrl+C
discards unsaved rows and closes the owned simulator. After a fall, save or
discard, then press X to reset. Physics targets 200 Hz between commands, with
WBC at 50 Hz. Pausing freezes the scene. More than 250 ms of simulation lag or
500 ms without a command pauses control instead of accumulating a motion backlog.

Saving publishes `.state.h5` recordings; unfinished files use `.partial`. After closing collection, rendering uses its active `python`:

```bash
conda activate egox
bash scripts/data_collect_scripts/g1-sim-dex3/kitchen-bread/render.sh
bash scripts/data_collect_scripts/g1-sim-dex3/kitchen-cleanup/render.sh
```

`--input FILE_OR_DIRECTORY` and `--output DIRECTORY` override the default task
paths. Raw recordings remain in `train/`; rendered episodes go into
`train/processed/`. The final `.hdf5` files contain training
fields and timing metadata; native replay state stays in the raw recording.

Visualize G1-Sim-Dex3 joint motion, wrist targets and main/wrist camera images with:

```bash
python scripts/data_visualization/vis_g1_sim_motion.py \
    --file data/g1-sim-dex3/kitchen-cleanup/train/processed/episode_NAME.hdf5
```

The viewer also opens raw `.state.h5` recordings before rendering, showing motion
without training camera images.
