"""Isaac Lab scene of the G1 with Dex3 hands: robot, optional room USD and the EgoX camera rig.

Physics runs at 200 Hz and the whole-body controller at 50 Hz. RGB rollout advances one
30 Hz sample per command; state-only collection follows wall time with a light preview.
A benchmark trial (trial.json, see benchmarks/) fixes the room, its runtime
overlay, the robot, the camera rig, render settings and the initial state, and adds its task.
Construct the environment only after the Isaac Sim app has started.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from .controller import OBS_MASKS, STANDING_HEIGHT, G1Dex3SimController
from .description import ARM_JOINT_NAMES, BODY_JOINT_NAMES, HAND_JOINT_NAMES

PHYSICS_HZ, WBC_DECIMATION, SAMPLE_HZ = 200, 4, 30
WARMUP_RENDERS = 16
EFFORT_LIMITS = {'.*_hip_(pitch|yaw)_joint': 88., '.*_hip_roll_joint': 139., '.*_knee_joint': 139.,
                 'waist_yaw_joint': 88., 'waist_(roll|pitch)_joint': 35., '.*_ankle_.*_joint': 35.,
                 '.*_(shoulder_.*|elbow|wrist_roll)_joint': 25., '.*_wrist_(pitch|yaw)_joint': 5.}
VELOCITY_LIMITS = {'.*_hip_(pitch|yaw)_joint': 32., '.*_(hip_roll|knee)_joint': 20., 'waist_yaw_joint': 32.,
                   'waist_(roll|pitch)_joint': 30., '.*_ankle_.*_joint': 30.,
                   '.*_(shoulder_.*|elbow|wrist_roll)_joint': 37., '.*_wrist_(pitch|yaw)_joint': 22.}
ROBOT = '/World/envs/env_0/Robot'
ROOM = '/World/SceneSmith'  # the benchmark overlays address the room here
PREVIEW_SETTINGS = {
    '/rtx/translucency/enabled': False, '/rtx/reflections/enabled': False,
    '/rtx/ambientOcclusion/enabled': False, '/rtx/indirectDiffuse/enabled': False,
    '/rtx-transient/dldenoiser/enabled': True, '/rtx/directLighting/enabled': True,
    '/rtx/directLighting/sampledLighting/enabled': True,
    '/rtx/directLighting/sampledLighting/samplesPerPixel': 1,
    '/rtx/shadows/enabled': True, '/rtx/raytracing/cached/enabled': True,
    '/rtx-transient/dlssg/enabled': False,
}


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))


def robot_cfg(usd_path, position, yaw, wbc):
    """Dex3 whole-body G1 with the WBC profile's PD gains and default pose."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    kp = dict(zip(BODY_JOINT_NAMES, wbc['lower_kps'])) | {BODY_JOINT_NAMES[i]: v for i, v in zip(wbc['arm_joint2motor_idx'], wbc['arm_kps'])}
    kd = dict(zip(BODY_JOINT_NAMES, wbc['lower_kds'])) | {BODY_JOINT_NAMES[i]: v for i, v in zip(wbc['arm_joint2motor_idx'], wbc['arm_kds'])}
    return ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False,
                linear_damping=0., angular_damping=0., max_linear_velocity=1000., max_angular_velocity=1000.,
                max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(position), rot=yaw_quat(yaw),
            joint_pos=dict(zip(BODY_JOINT_NAMES, wbc['lower_default_joint_pos'] + wbc['arm_default_joint_pos']))),
        soft_joint_pos_limit_factor=.9,
        actuators={
            'body': ImplicitActuatorCfg(joint_names_expr=list(BODY_JOINT_NAMES), effort_limit_sim=EFFORT_LIMITS,
                                        velocity_limit_sim=VELOCITY_LIMITS, stiffness=kp, damping=kd, armature=.01),
            # Dex3 keeps its USD motor limits (1.4 / 2.45 N*m); these gains are stable at 5 ms.
            'hands': ImplicitActuatorCfg(joint_names_expr=['.*_hand_.*_joint'], stiffness=100., damping=10., armature=.1),
        })


class CameraProjection:
    """Render a centered pinhole whose field of view covers the requested intrinsics, then remap.

    Isaac's RGB renderer models only centered square-pixel pinholes; the remap gives the
    exact requested K (as the G1-Sim-Dex3 recordings) at an unchanged resolution.
    """

    def __init__(self, intrinsics):
        k = intrinsics
        self.width, self.height = k['width'], k['height']
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        rays = np.stack([(xx - k['cx']) / k['fx'], (yy - k['cy']) / k['fy']], axis=-1)
        reach = np.max(np.abs(rays), axis=(0, 1))
        self.render_focal = float(min((self.width / 2 - 1) / reach[0], (self.height / 2 - 1) / reach[1]))
        self.map_x = (rays[..., 0] * self.render_focal + self.width / 2).astype(np.float32)
        self.map_y = (rays[..., 1] * self.render_focal + self.height / 2).astype(np.float32)

    def apply(self, image):
        import cv2

        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)


def camera_cfg(name, camera, projection):
    """Pinhole camera on a robot link; rig poses map ROS optical axes into the parent link."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path=f'{{ENV_REGEX_NS}}/Robot/{camera["parent"]}/egox_{name}', height=projection.height,
        width=projection.width, update_period=0., data_types=['rgb'], update_latest_camera_pose=False,
        spawn=sim_utils.PinholeCameraCfg(focal_length=12., horizontal_aperture=12. * projection.width / projection.render_focal,
                                         clipping_range=(.02, 100.)),
        offset=CameraCfg.OffsetCfg(pos=tuple(camera['translation_m']), rot=tuple(camera['rotation_quat_wxyz_ros']),
                                   convention='ros'))


class G1Dex3Env:
    def __init__(self, args, *, task_factory=None, trial=None):
        import carb
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from egox.sim.scene import RoomState

        self.trial = trial
        self.capture_mode = getattr(args, 'capture_mode', 'rgb')
        if self.capture_mode not in ('rgb', 'state_only'):
            raise ValueError('capture_mode must be rgb or state_only')
        self.clock = time.monotonic_ns
        self._anchor_ns = self._last_command_ns = None
        self._anchor_steps = 0
        self._last_recorded_frame = None
        self.stop_reason = None
        self._preview_slot = None
        self._preview_origin = None
        self._preview = {}
        self._preview_sequence = 0
        self.preview_enabled = getattr(args, 'preview', True)
        self.robot_usd = Path(args.robot_usd)
        if trial:
            args.scene_usd, args.scene_position, args.scene_yaw = trial['scene_usd'], trial['scene_position'], trial['scene_yaw']
            args.robot_position, args.robot_yaw = trial['robot_spawn_position'], trial['robot_yaw']
        self.controller = G1Dex3SimController()
        self.rig = trial['camera_rig'] if trial else json.loads(Path(args.camera_rig).read_text())
        self.camera_names = ((tuple(self.rig['outputs']['main']) if self.preview_enabled else ()) if self.capture_mode == 'state_only'
                             else tuple(self.rig['cameras']))
        self.projections = {name: CameraProjection(self.rig['cameras'][name]['intrinsics']) for name in self.camera_names}
        self.device = args.device
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=1 / PHYSICS_HZ, render_interval=WBC_DECIMATION, device=args.device))
        cfg = InteractiveSceneCfg(num_envs=1, env_spacing=5., replicate_physics=False)
        cfg.robot = robot_cfg(args.robot_usd, args.robot_position, args.robot_yaw, self.controller.wbc)
        cfg.light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=1500.))
        if args.scene_usd:
            cfg.room = AssetBaseCfg(prim_path=ROOM, spawn=sim_utils.UsdFileCfg(usd_path=str(args.scene_usd)),
                                    init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(args.scene_position),
                                                                            rot=yaw_quat(args.scene_yaw)))
        else:
            cfg.ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.GroundPlaneCfg())
        for name in self.camera_names:
            setattr(cfg, f'camera_{name}', camera_cfg(name, self.rig['cameras'][name], self.projections[name]))
        self.scene = InteractiveScene(cfg)
        self._edit_stage(bool(args.scene_usd))
        settings = carb.settings.get_settings()
        self.render_settings = trial['render_settings'] if trial else self._scene_render_settings()
        for key, value in self.render_settings.items():
            settings.set(key, value)
        if self.capture_mode == 'state_only':
            for key, value in PREVIEW_SETTINGS.items():
                settings.set(key, value)
        self.task = task_factory(self.scene.stage, trial) if trial and task_factory else None
        self.sim.reset()
        self.scene.update(1 / PHYSICS_HZ)

        self.robot = self.scene['robot']
        names = list(self.robot.data.joint_names)
        self.body_ids = [names.index(n) for n in BODY_JOINT_NAMES]
        self.arm_ids = [names.index(n) for n in ARM_JOINT_NAMES]
        self.hand_ids = [names.index(n) for n in HAND_JOINT_NAMES]
        self.lower_ids = self.body_ids[:15]
        self.body_index = {name: i for i, name in enumerate(self.robot.data.body_names)}
        self.limits = self.robot.data.joint_pos_limits[0]
        self.arm_effort_limits = self.robot.root_physx_view.get_dof_max_forces()[:, self.arm_ids].to(self.device)
        self.targets = self.robot.data.default_joint_pos.clone()
        self.efforts = torch.zeros_like(self.targets)
        self._targets_dirty = True
        self.phase = self.sample = self.physics_steps = 0
        self.room = None
        self._set_arm_effort(self.controller.arm_gravity(self.controller.arm_default))
        if not hasattr(args, 'replay_state'):
            self._advance(args.settle_steps * WBC_DECIMATION)  # stand up and let the room settle
        self.robot_root = self.robot.data.root_state_w.clone()
        self.robot_joints = self.robot.data.joint_pos.clone()
        self.room = RoomState(self.scene.stage, ROOM) if args.scene_usd else None
        if self.task:
            self.task.bind(self)
        if hasattr(args, 'replay_state'):
            self.restore_state(args.replay_state, args.replay_metadata)
        else:
            self.reset()

    def _edit_stage(self, has_room):
        """Session-layer edits before physics starts; the source USDs are untouched."""
        from pxr import PhysxSchema, Usd

        stage = self.scene.stage
        # The Dex3 wrist camera brackets would block the virtual wrist cameras.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for side in ('left', 'right'):
                for branch in ('visuals', 'collisions'):
                    prim = stage.GetPrimAtPath(f'{ROBOT}/{side}_hand_camera_base_link/{branch}')
                    if prim:
                        prim.SetActive(False)
        if has_room:
            # Keep the room's authored collision groups: the environment filter would otherwise
            # make some of them (e.g. refrigerator doors) invisible to the robot.
            stage.RemovePrim('/World/collisions')
            PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(self.scene.physics_scene_path)) \
                .CreateInvertCollisionGroupFilterAttr().Set(False)
        if self.trial:
            # The trial's layout, appearance, lighting and contact settings, strongest of all.
            stage.GetSessionLayer().subLayerPaths.insert(0, str(self.trial['overlay']))

    def _scene_render_settings(self):
        """A room may ask for RTX reflections (`egox:enableReflections`)."""
        prim = self.scene.stage.GetPrimAtPath(ROOM)
        attribute = prim.GetAttribute('egox:enableReflections') if prim else None
        value = attribute.Get() if attribute else None
        return {} if value is None else {'/rtx/reflections/enabled': bool(value)}

    def _numpy(self, value):
        return value.detach().cpu().numpy().astype(np.float64)

    def _set_arm_effort(self, effort):
        """Arm gravity feedforward; the PD drive keeps the remaining torque budget."""
        import torch

        effort = torch.as_tensor(effort, dtype=self.targets.dtype, device=self.device).reshape(1, -1)
        effort = torch.clamp(effort, -self.arm_effort_limits, self.arm_effort_limits)
        self.efforts[:, self.arm_ids] = effort
        self._targets_dirty = True
        self.robot.write_joint_effort_limit_to_sim(self.arm_effort_limits - effort.abs(), joint_ids=self.arm_ids)

    def _advance(self, physics_steps):
        import torch

        data = self.robot.data
        pending_update = 0
        for index in range(physics_steps):
            if self.phase == 0:
                lower = self.controller.lower_body(self._numpy(data.root_ang_vel_b[0]), self._numpy(data.projected_gravity_b[0]),
                                                   self._numpy(data.joint_pos[0, self.lower_ids]),
                                                   self._numpy(data.joint_vel[0, self.lower_ids]))
                lower = torch.as_tensor(lower, dtype=self.targets.dtype, device=self.device)
                self.targets[0, self.lower_ids] = torch.clamp(lower, self.limits[self.lower_ids, 0], self.limits[self.lower_ids, 1])
                self._targets_dirty = True
            if self.robot.has_external_wrench:
                raise RuntimeError('held implicit joint targets do not support external wrenches')
            if self._targets_dirty:
                self.robot.set_joint_position_target(self.targets)
                self.robot.set_joint_velocity_target(torch.zeros_like(self.targets))
                self.robot.set_joint_effort_target(self.efforts)
                self.scene.write_data_to_sim()
                self._targets_dirty = False
            self.sim.step(render=False)
            if self.task and self.room:
                self.task.after_physics_step(1 / PHYSICS_HZ)
            self.phase = (self.phase + 1) % WBC_DECIMATION
            self.physics_steps += 1
            pending_update += 1
            if self.phase == 0 or index == physics_steps - 1:
                self.scene.update(pending_update / PHYSICS_HZ)
                pending_update = 0

    def step(self, action, source_time_ns=None):
        """Fixed-step RGB rollout, or atomic pre-action state capture during realtime collection."""
        if getattr(self, 'capture_mode', 'rgb') == 'state_only':
            return self._collect_step(action, source_time_ns)
        self._apply_action(self.controller.set_action(action))
        self.sample += 1
        self._advance((self.sample * PHYSICS_HZ) // SAMPLE_HZ - ((self.sample - 1) * PHYSICS_HZ) // SAMPLE_HZ)
        return self.observe()

    def _apply_action(self, prepared):
        import torch

        arm, effort, hand = prepared
        self.targets[0, self.arm_ids] = torch.as_tensor(arm, dtype=self.targets.dtype, device=self.device)
        hand = torch.as_tensor(hand, dtype=self.targets.dtype, device=self.device)
        self.targets[0, self.hand_ids] = torch.clamp(hand, self.limits[self.hand_ids, 0], self.limits[self.hand_ids, 1])
        self._set_arm_effort(effort)

    def pause(self, reason=None):
        """Freeze collection; the next command starts a new wall-clock anchor without idle debt."""
        self._anchor_ns = self._last_command_ns = None
        self.stop_reason = reason
        return {}

    def tick(self, *, finish_debt=False):
        """Catch up held targets to wall time, as the old rigid simulator did.

        Work is bounded to four-step chunks and 100 ms per idle poll; more than
        250 ms lag or 500 ms command silence pauses control rather than replaying
        a long backlog. No physics runs while paused or in fixed-step evaluation.
        """
        if self._anchor_ns is None:
            return 0
        started = now = self.clock()
        if now - self._last_command_ns > 500_000_000:
            self.pause('command watchdog expired')
            return 0
        step_ns = 1_000_000_000 // PHYSICS_HZ
        due = (now - self._anchor_ns) // step_ns - (self.physics_steps - self._anchor_steps)
        advanced = 0
        while due > 0:
            represented = self._anchor_ns + (self.physics_steps - self._anchor_steps) * step_ns
            if now - represented > 250_000_000:
                self.pause('realtime simulation lag exceeded 250 ms')
                return advanced
            chunk = min(int(due), WBC_DECIMATION)
            try:
                self._advance(chunk)
            except BaseException:
                self.pause('physics failed')
                raise
            advanced += chunk
            due -= chunk
            now = self.clock()
            represented = self._anchor_ns + (self.physics_steps - self._anchor_steps) * step_ns
            if now - represented > 250_000_000 or now - self._last_command_ns > 500_000_000:
                self.pause('realtime catch-up exceeded lag or command watchdog limit')
                return advanced
            if not finish_debt and now - started >= 100_000_000:
                break
        return advanced

    def _check_source(self, source):
        if source is not None and (type(source) is not int or not 0 <= self.clock() - source <= 500_000_000):
            self.pause('tracking source became stale')
            return False
        return True

    def _collect_step(self, action, source):
        if self.stop_reason or not self._check_source(source):
            return {'paused': self.stop_reason, 'recordable': False}
        # IK updates its seed and the WBC command. Keep those changes staged so
        # elapsed wall time is simulated with the PREVIOUS command, never the new one.
        controller = self.controller
        previous_command, previous_seed = controller.command.copy(), controller.ik_seed.copy()
        try:
            prepared = controller.set_action(action)
            command, seed = controller.command.copy(), controller.ik_seed.copy()
        finally:
            controller.command, controller.ik_seed = previous_command, previous_seed
        if not self._check_source(source):
            return {'paused': self.stop_reason, 'recordable': False}
        armed = self._anchor_ns is not None
        if armed:
            self.tick(finish_debt=True)
            if self._anchor_ns is None:
                return {'paused': self.stop_reason, 'recordable': False}
        observation = self.observe()
        if not self._check_source(source):
            return {'paused': self.stop_reason, 'recordable': False}
        now = self.clock()
        if armed and now - observation['state_time_ns'] > 250_000_000:
            self.pause('observation capture exceeded realtime lag limit')
            return {'paused': self.stop_reason, 'recordable': False}
        try:
            self._apply_action(prepared)
            controller.command, controller.ik_seed = command, seed
        except BaseException:
            self.pause('action installation failed')
            raise
        if not armed:
            self._anchor_ns, self._anchor_steps = now, self.physics_steps
            observation['state_time_ns'] = now
        self._last_command_ns, self.stop_reason = self.clock(), None
        observation['recordable'] = self.physics_steps != self._last_recorded_frame
        self._last_recorded_frame = self.physics_steps
        return observation

    def reset(self):
        """Back to the trial's recorded initial state, or to the settled robot and room."""
        import torch

        self.pause()
        self._last_recorded_frame = None
        self._preview_slot = self._preview_origin = None
        self._preview = {}
        if self.trial:
            self._restore(self.trial['state'])
        else:
            root = self.robot_root.clone()
            root[:, 7:] = 0.
            self.robot.write_root_state_to_sim(root)
            self.robot.write_joint_state_to_sim(self.robot_joints, torch.zeros_like(self.robot_joints))
            self.robot.reset()
            if self.room:
                self.room.restore()
            self.sim.forward()
            self.scene.update(1 / PHYSICS_HZ)
            self.targets = self.robot.data.joint_pos.clone()
            self.controller.reset(self._numpy(self.robot.data.joint_pos[0, self.arm_ids]))
            self._set_arm_effort(self.controller.arm_gravity(self._numpy(self.robot.data.joint_pos[0, self.arm_ids])))
            self.phase = 0
        self.sample = self.physics_steps = 0
        return self.observe()

    def _restore(self, state):
        """Robot, room, whole-body controller, arm IK and task state of a benchmark trial."""
        import torch

        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)  # noqa: E731
        xyzw = lambda pose: np.asarray(pose)[..., [0, 1, 2, 4, 5, 6, 3]]  # noqa: E731
        order = [self.trial['joint_names'].index(name) for name in self.robot.data.joint_names]
        view, ids = self.robot.root_physx_view, torch.tensor([0], device=self.device)
        view.set_root_transforms(tensor(xyzw(state['robot/pose_world_wxyz']))[None], ids)
        view.set_root_velocities(tensor(state['robot/velocity_world'])[None], ids)
        self.robot.write_joint_state_to_sim(tensor(state['robot/joint_position'][order])[None],
                                            tensor(state['robot/joint_velocity'][order])[None])
        articulations = {}
        for index, path in enumerate(self.trial['articulations']):
            prefix = f'articulations/articulation_{index:03d}/'
            articulations[path] = (xyzw(state[prefix + 'pose_world_wxyz']), state[prefix + 'velocity_world'],
                                   state[prefix + 'joint_position'], state[prefix + 'joint_velocity'])
        self.room.set_state(self.trial['rigid_bodies'], xyzw(state['rigid_objects/pose_world_wxyz']),
                            state['rigid_objects/velocity_world'], articulations)
        self.robot.reset()
        controller = self.controller
        controller.history = state['controller/lbc_history'].astype(np.float32).copy()
        controller.previous_action = state['controller/previous_lbc_action'].astype(np.float32).copy()
        controller.command = state['controller/command'].astype(np.float64).copy()
        controller.ik_seed = state['ik/previous'].astype(np.float64).copy()
        self.targets = tensor(state['controller/joint_position_target'][order])[None].clone()
        self._set_arm_effort(state['controller/joint_effort_target'][order][self.arm_ids])
        self.phase = int(state['controller/lbc_substep_phase'][0])
        self.sim.forward()
        self.scene.update(1 / PHYSICS_HZ)
        if self.task:
            self.task.reset(state)
        # Let RTX's temporal accumulation and auto exposure settle on the restored scene, as
        # the first images of the paper's runs were (the state does not change meanwhile).
        if getattr(self, 'capture_mode', 'rgb') == 'rgb':
            for _ in range(WARMUP_RENDERS):
                self.render()

    def render(self):
        if getattr(self, 'capture_mode', 'rgb') == 'state_only':
            now = self.clock()
            if self._preview_origin is None:
                self._preview_origin = now
            slot = (now - self._preview_origin) * 24 // 1_000_000_000
            if slot == self._preview_slot:
                return
            self._preview_slot = slot
        if self.task:
            self.task.before_render()
        self.sim.render()
        if getattr(self, 'capture_mode', 'rgb') == 'state_only' and self.preview_enabled:
            import cv2

            frames = self._images()
            image = np.concatenate([frames[v] for v in self.rig['outputs']['main']], axis=1)
            ok, jpeg = cv2.imencode('.jpg', image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('could not encode headset preview')
            self._preview_sequence += 1
            self._preview = {'jpeg': jpeg.reshape(-1), 'frame_id': self.physics_steps,
                             'preview_sequence': self._preview_sequence}

    def _images(self):
        frames = {}
        for name in self.projections:
            camera = self.scene[f'camera_{name}']
            camera.update(0., force_recompute=True)
            frames[name] = self.projections[name].apply(camera.data.output['rgb'][0, ..., :3].cpu().numpy())
        return frames

    def preview(self):
        """Separate, compressed 24 Hz headset preview; never part of action replies."""
        if self.capture_mode != 'state_only' or not self.preview_enabled:
            return {}
        self.render()
        return self._preview

    def link_pose(self, name):
        """World pose (XYZ + WXYZ) of a robot link."""
        return self._numpy(self.robot.data.body_link_state_w[0, self.body_index[name], :7])

    def link_position_physx(self, name):
        return self._numpy(self.robot.root_physx_view.get_link_transforms()[0, self.body_index[name], :3])

    def dof_positions_physx(self, ids):
        return self._numpy(self.robot.root_physx_view.get_dof_positions()[0, ids])

    def targets_numpy(self, ids):
        return self._numpy(self.targets[0, ids])

    def describe(self):
        """Reuse the controller's hand FK and command limits for teleoperation."""
        description = {'embodiment': 'G1-Sim-Dex3', 'action_hz': SAMPLE_HZ,
                'teleop': {'grasp_curve': self.controller.grasp_curve,
                          'command_bounds': self.controller.command_bounds,
                          'standing_height': STANDING_HEIGHT}}
        if getattr(self, 'capture_mode', 'rgb') == 'state_only':
            description.update(capture_mode='state_only', collection=self.recording_metadata())
        return description

    def observe(self):
        """Images and proprioception under the policy's observation keys, with their masks.

        With a trial, `state` holds the scorer's inputs at this sample (see scoring.py).
        """
        obs = {}
        if getattr(self, 'capture_mode', 'rgb') == 'rgb':
            self.render()
            frames = self._images()
            obs = {f'{output}_image': np.concatenate([frames[v] for v in views], axis=1)
                   for output, views in self.rig['outputs'].items()}
        joints = self.robot.data.joint_pos[0]
        obs.update(self.controller.proprio(self._numpy(joints[self.body_ids]), self._numpy(joints[self.hand_ids])))
        masks = {key: OBS_MASKS.get(key, np.array(True) if 'image' in key else np.ones(len(value), dtype=bool))
                 for key, value in obs.items()}
        observation = {'obs': obs, 'obs_mask': masks, 'sim_time': self.physics_steps / PHYSICS_HZ}
        if getattr(self, 'capture_mode', 'rgb') == 'state_only':
            observation.update(replay_state=self.snapshot_state(), frame_id=self.physics_steps,
                               state_time_ns=(self.clock() if self._anchor_ns is None else self._anchor_ns
                                              + (self.physics_steps - self._anchor_steps) * 1_000_000_000 // PHYSICS_HZ))
        if self.task:
            root = self._numpy(self.robot.root_physx_view.get_root_transforms()[0])[[0, 1, 2, 6, 3, 4, 5]]
            up = 1 - 2 * (root[4] ** 2 + root[5] ** 2)  # z axis of the pelvis, world z component
            tilt = math.degrees(math.acos(float(np.clip(up, -1., 1.))))
            observation['state'] = {'sim_time': observation['sim_time'], 'robot_pose': root, 'robot_tilt_deg': tilt,
                                    **self.task.trace(observation)}
            observation['fell'] = bool(root[2] < .45 or tilt > 60.)
        return observation

    def recording_metadata(self):
        """Immutable scene/camera contract for offline rendering, with portable repository paths."""
        root = Path(__file__).resolve().parents[4]

        def plain(value):
            if isinstance(value, Path):
                return value.resolve().relative_to(root).as_posix() if value.resolve().is_relative_to(root) else str(value)
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value.tolist() if isinstance(value, np.ndarray) else value

        return plain({'trial': {key: value for key, value in (self.trial or {}).items() if key != 'state'},
                      'physics_hz': PHYSICS_HZ, 'action_hz': SAMPLE_HZ,
                      'robot_usd': self.robot_usd, 'camera_rig': self.rig, 'render_settings': self.render_settings,
                      'joint_names': list(self.robot.data.joint_names),
                      'rigid_bodies': self.room.paths if self.room else [],
                      'articulations': self.room.articulation_paths if self.room else [],
                      'articulation_joint_names': [[str(path).rsplit('/', 1)[-1] for path in view.dof_paths[0]]
                                                  for view, *_ in self.room.articulations] if self.room else []})

    def snapshot_state(self):
        """Owned pre-action poses for replay; coordinate/order conventions match old trial states."""
        view = self.robot.root_physx_view
        wxyz = lambda pose: self._numpy(pose)[..., [0, 1, 2, 6, 3, 4, 5]]  # noqa: E731
        state = {'robot/pose_world_wxyz': wxyz(view.get_root_transforms()[0]),
                 'robot/velocity_world': self._numpy(view.get_root_velocities()[0]),
                 'robot/joint_position': self._numpy(view.get_dof_positions()[0]),
                 'robot/joint_velocity': self._numpy(view.get_dof_velocities()[0]),
                 'physics_steps': np.array(self.physics_steps, dtype=np.int64)}
        if self.room:
            if self.room.bodies:
                state['rigid_objects/pose_world_wxyz'] = self.room.rigid_poses()[:, [0, 1, 2, 6, 3, 4, 5]]
                state['rigid_objects/velocity_world'] = self.room.rigid_velocities()
            for index, (view, *_) in enumerate(self.room.articulations):
                prefix = f'articulations/articulation_{index:03d}/'
                state.update({prefix + 'pose_world_wxyz': wxyz(view.get_root_transforms()[0]),
                              prefix + 'velocity_world': self._numpy(view.get_root_velocities()[0]),
                              prefix + 'joint_position': self._numpy(view.get_dof_positions()[0]),
                              prefix + 'joint_velocity': self._numpy(view.get_dof_velocities()[0])})
        if self.task and hasattr(self.task, 'model'):
            state['coating/blade_thickness_full'] = self.task.model.blade_thickness_m.copy()
            state['coating/bread_thickness_full'] = self.task.model.bread_thickness_m.copy()
        return state

    def restore_state(self, state, metadata):
        """Pose-only replay: install each saved state exactly, without integrating physics."""
        import torch

        tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)  # noqa: E731
        xyzw = lambda pose: np.asarray(pose)[..., [0, 1, 2, 4, 5, 6, 3]]  # noqa: E731
        order = [metadata['joint_names'].index(name) for name in self.robot.data.joint_names]
        view, ids = self.robot.root_physx_view, torch.tensor([0], device=self.device)
        view.set_root_transforms(tensor(xyzw(state['robot/pose_world_wxyz']))[None], ids)
        view.set_root_velocities(tensor(state['robot/velocity_world'])[None], ids)
        self.robot.write_joint_state_to_sim(tensor(state['robot/joint_position'][order])[None],
                                            tensor(state['robot/joint_velocity'][order])[None])
        if self.room:
            articulations = {}
            for index, path in enumerate(metadata['articulations']):
                prefix = f'articulations/articulation_{index:03d}/'
                current = self.room.articulations[self.room.articulation_index(path)][0]
                names = [str(p).rsplit('/', 1)[-1] for p in current.dof_paths[0]]
                order = [metadata['articulation_joint_names'][index].index(name) for name in names]
                articulations[path] = (xyzw(state[prefix + 'pose_world_wxyz']), state[prefix + 'velocity_world'],
                                       state[prefix + 'joint_position'][order], state[prefix + 'joint_velocity'][order])
            if self.room.bodies:
                self.room.set_state(metadata['rigid_bodies'], xyzw(state['rigid_objects/pose_world_wxyz']),
                                    state['rigid_objects/velocity_world'], articulations)
            else:
                for path, values in articulations.items():
                    view = self.room.articulations[self.room.articulation_index(path)][0]
                    self.room._set_articulation(view, *(tensor(value)[None] for value in values))
        self.robot.reset()
        self.sim.forward()
        self.scene.update(1 / PHYSICS_HZ)
        self.physics_steps = int(state['physics_steps'])
        self.targets = self.robot.data.joint_pos.clone()
        self._targets_dirty = True
        if self.task:
            self.task.reset(state)

    def close(self):
        """Stop physics; the caller owns the Isaac application lifetime."""
        if self.task and hasattr(self.task, 'close'):
            self.task.close()
        # Remove IsaacLab's wait-for-Play callback before stopping the timeline.
        self.sim.clear_instance()
        self.sim.stop()
