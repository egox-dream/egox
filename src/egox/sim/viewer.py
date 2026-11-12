"""View a simulation setup: a scene or empty ground, with the robot and its cameras.

Shift + left-drag moves dynamic props and doors; the panel pauses, restores the starting layout
and switches cameras. Edits stay in memory: the USDs are never saved.

The robot's base is fixed and its joints hold their pose, because standing needs the whole-body
controller. The viewer shows the setup; it does not run the robot.
"""
from contextlib import ExitStack
import math
from pathlib import Path
import time

from .scene import ROOM, ROBOT, RoomState, add_camera, camera_path

FREE_CAMERA = '/OmniverseKit_Persp'
PHYSICS_DT, RENDER_DT = 1 / 120, 1 / 60
HOLD_STIFFNESS, HOLD_DAMPING = 17.5, .9  # per degree: about 1000 N*m/rad and 50 N*m*s/rad


class Deadline:
    """Fires once per period; after a stall it resumes from now instead of catching up."""

    def __init__(self, period):
        self.period, self.next = period, time.monotonic()

    def due(self, now):
        if now < self.next:
            return False
        self.next += self.period
        if self.next <= now:
            self.next = now + self.period
        return True


def _add_robot(stage, robot):
    """Fix the robot's base, hold its joints at the requested pose, and mount its cameras."""
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema, Usd, UsdPhysics

    cfg = sim_utils.UsdFileCfg(usd_path=str(robot.usd), articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        fix_root_link=True, enabled_self_collisions=False))
    cfg.func(ROBOT, cfg, translation=tuple(robot.position), orientation=tuple(robot.orientation))
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        angle = math.degrees(robot.joint_positions.get(prim.GetName(), 0.))
        drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
        drive.CreateTargetPositionAttr().Set(angle)
        drive.CreateStiffnessAttr().Set(HOLD_STIFFNESS)
        drive.CreateDampingAttr().Set(HOLD_DAMPING)
        state = PhysxSchema.JointStateAPI.Apply(prim, 'angular')
        state.CreatePositionAttr().Set(angle)
        state.CreateVelocityAttr().Set(0.)
    # The Dex3 wrist camera brackets would block the wrist cameras.
    for side in ('left', 'right'):
        for branch in ('visuals', 'collisions'):
            prim = stage.GetPrimAtPath(f'{ROBOT}/{side}_hand_camera_base_link/{branch}')
            if prim:
                prim.SetActive(False)
    for view, camera in robot.camera_rig['cameras'].items():
        add_camera(view, camera)


def load_setup(scene_usd=None, robot=None, rendering_mode=None):
    """Open the setup on CPU PhysX, let it settle, and snapshot its start layout."""
    import carb
    from isaacsim.core.api import SimulationContext
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema

    # Scene queries let the viewport's mouse grab raycast into the room.
    sim = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.,
                            physics_prim_path='/World/physicsScene', backend='torch', device='cpu',
                            sim_params={'use_gpu_pipeline': False, 'use_fabric': False,
                                        'enable_scene_query_support': True})
    stage = sim.stage
    # Keep the room's authored collision groups.
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/World/physicsScene')) \
        .CreateInvertCollisionGroupFilterAttr(False)
    if scene_usd:
        room = sim_utils.UsdFileCfg(usd_path=str(scene_usd))
        room.func(ROOM, room)
    else:
        ground = sim_utils.GroundPlaneCfg()
        ground.func('/World/Ground', ground)
    light = sim_utils.DomeLightCfg(intensity=1500.)
    light.func('/World/Light', light)
    if robot:
        _add_robot(stage, robot)

    settings = carb.settings.get_settings()
    reflections = stage.GetPrimAtPath(ROOM).GetAttribute('egox:enableReflections') if scene_usd else None
    if reflections and reflections.Get() is not None and rendering_mode != 'performance':
        settings.set_bool('/rtx/reflections/enabled', bool(reflections.Get()))
    for key in ('mouseInteractionEnabled', 'mouseGrab', 'forceGrab'):
        settings.set_bool(f'/physics/{key}', True)
    settings.set_bool('/physics/mouseGrabIgnoreInvisible', False)
    settings.set_float('/physics/pickingForce', 1.)

    sim.reset()
    for _ in range(round(1 / PHYSICS_DT)):  # one second for the props to settle
        sim.step(render=False)
    sim.render()
    return sim, [RoomState(stage, root) for root, used in ((ROOM, scene_usd), (ROBOT, robot)) if used]


def available_cameras(stage, scene_cameras=(), robot=None):
    """Panel labels and prim paths of the cameras this setup has, the first being the start view."""
    cameras = {name: f'{ROOM}/Cameras/{name}' for name in scene_cameras}
    if robot:
        rig = robot.camera_rig
        for output, views in rig['outputs'].items():  # the first of a stereo pair stands for it
            cameras[output.replace('_', ' ').capitalize()] = camera_path(rig['cameras'][views[0]], views[0])
    cameras['Free'] = FREE_CAMERA
    return {label: path for label, path in cameras.items() if stage.GetPrimAtPath(path)}


class Viewer:
    """Physics and rendering on their own clocks, with panel requests applied between them."""

    def __init__(self, sim, states, cameras):
        self.sim, self.states, self.cameras = sim, states, cameras
        self.pending = {'pause': False, 'reset': False, 'camera': next(iter(cameras), None)}
        self.physics, self.rendering = Deadline(PHYSICS_DT), Deadline(RENDER_DT)

    def apply_requests(self):
        from omni.kit.viewport.utility import get_active_viewport

        if self.pending['pause']:
            if self.sim.is_playing():
                self.sim.pause()
            else:
                self.sim.play()
            self.pending['pause'] = False
        if self.pending['reset']:
            for state in self.states:
                state.restore()
                state.view.update_articulations_kinematic()
            self.pending['reset'] = False
        if self.pending['camera']:
            viewport = get_active_viewport()
            if viewport:
                viewport.camera_path = self.cameras[self.pending['camera']]
            self.pending['camera'] = None

    def update(self):
        self.apply_requests()
        now = time.monotonic()
        if self.sim.is_playing() and self.physics.due(now):
            self.sim.step(render=False)
        if self.rendering.due(now):
            self.sim.render()


def build_panel(viewer):
    import omni.ui as ui

    labels = list(viewer.cameras)
    window = ui.Window('EgoX', width=380, height=110 + 30 * ((len(labels) + 3) // 4))
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label('Shift + left-drag moves props and doors.', word_wrap=True)
            with ui.HStack(spacing=5):
                ui.Button('Pause / Resume', clicked_fn=lambda: viewer.pending.update(pause=True))
                ui.Button('Reset layout', clicked_fn=lambda: viewer.pending.update(reset=True))
            for start in range(0, len(labels), 4):
                with ui.HStack(spacing=5):
                    for label in labels[start:start + 4]:
                        ui.Button(label, clicked_fn=lambda label=label: viewer.pending.update(camera=label))
    return window


def add_arguments(parser):
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device='cpu')  # CPU PhysX; rendering still uses the GPU


def view(args, scene_usd=None, robot=None, scene_cameras=(), free_camera=None):
    """Open the viewer and run until the window closes. `free_camera` is (eye, target) in meters."""
    from isaaclab.app import AppLauncher

    if args.headless:
        raise SystemExit('The viewer needs a display; run it without --headless.')
    if scene_usd and not Path(scene_usd).is_file():
        raise SystemExit(f'Missing scene USD {scene_usd}')
    if robot and not Path(robot.usd).is_file():
        raise SystemExit(f'Missing robot USD {robot.usd}; run python scripts/download_assets.py')
    with ExitStack() as cleanup:
        app = AppLauncher(args).app
        cleanup.callback(app.close)

        from isaacsim.core.utils.viewports import set_camera_view

        sim, states = load_setup(scene_usd, robot, args.rendering_mode)
        cleanup.callback(sim.stop)
        if free_camera:
            set_camera_view(*free_camera)
        viewer = Viewer(sim, states, available_cameras(sim.stage, scene_cameras, robot))
        window = build_panel(viewer)
        cleanup.callback(window.destroy)
        print('Viewer ready: Shift + left-drag moves props and doors; stop the timeline to exit.', flush=True)
        while app.is_running() and not sim.is_stopped():
            viewer.update()
            time.sleep(.001)
