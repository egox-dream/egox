"""Shared robot/camera setup and scene-state restoration for Isaac simulation."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOM = '/World/Scene'
ROBOT = '/World/Robot'


@dataclass(frozen=True)
class RobotSetup:
    usd: Path
    position: tuple
    orientation: tuple  # WXYZ
    joint_positions: dict  # radians by joint name; other joints hold zero
    camera_rig: dict  # see assets/robots/g1_dex3/camera_rig.json


def camera_path(camera, view):
    return f'{ROBOT}/{camera["parent"]}/egox_{view}'


def add_camera(view, camera):
    """A pinhole camera on a robot link; rig poses map ROS optical axes into the parent link."""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    intrinsics = camera['intrinsics']
    spawn = sim_utils.PinholeCameraCfg(focal_length=12., horizontal_aperture=12. * intrinsics['width'] / intrinsics['fx'],
                                       clipping_range=(.02, 100.))
    rotation = convert_camera_frame_orientation_convention(
        torch.tensor(camera['rotation_quat_wxyz_ros']).unsqueeze(0), origin='ros', target='opengl')[0]
    spawn.func(camera_path(camera, view), spawn, translation=tuple(camera['translation_m']),
               orientation=tuple(rotation.tolist()))


class RoomState:
    """The room's rigid bodies and articulations (doors, drawers): their snapshot and restore.

    Rigid bodies outside articulations are ordered by prim path, as in the recorded benchmark
    states. Poses are XYZ + XYZW quaternion (native PhysX), velocities linear then angular.
    """

    def __init__(self, stage, root):
        import omni.physics.tensors as physics
        from pxr import Usd, UsdPhysics

        prims = list(Usd.PrimRange(stage.GetPrimAtPath(root)))
        roots = sorted(str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        self.paths = sorted(str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)
                            and not any(str(p.GetPath()) == r or str(p.GetPath()).startswith(r + '/') for r in roots))
        try:
            from isaaclab.sim.utils.stage import get_current_stage_id
            self.view = physics.create_simulation_view('torch', stage_id=get_current_stage_id())
        except ImportError:
            self.view = physics.create_simulation_view('torch')
        self.view.set_subspace_roots('/')
        self.bodies = None
        if self.paths:
            view = self.view.create_rigid_body_view(self.paths)
            order = [list(view.prim_paths).index(p) for p in self.paths]
            if order != list(range(len(order))):
                raise RuntimeError('rigid-body view does not follow the requested prim order')
            self.bodies = (view, view.get_transforms().clone(), view.get_velocities().clone())
        self.articulation_paths = roots
        self.articulations = []
        for path in roots:
            view = self.view.create_articulation_view(path)
            self.articulations.append((view, view.get_root_transforms().clone(), view.get_dof_positions().clone()))

    def rigid_poses(self):
        return self.bodies[0].get_transforms().detach().cpu().numpy().astype(np.float64)

    def rigid_velocities(self):
        return self.bodies[0].get_velocities().detach().cpu().numpy().astype(np.float64)

    def articulation_index(self, path):
        return self.articulation_paths.index(path)

    def dof_index(self, articulation, joint):
        return [p.rsplit('/', 1)[-1] for p in self.articulations[articulation][0].dof_paths[0]].index(joint)

    def dof_positions(self, articulation):
        return self.articulations[articulation][0].get_dof_positions()[0].detach().cpu().numpy()

    def restore(self):
        """Back to the snapshot taken at construction, at rest."""
        import torch

        if self.bodies:
            view, pose, _ = self.bodies
            self._set_bodies(pose, torch.zeros_like(self.bodies[2]))
        for view, pose, joints in self.articulations:
            self._set_articulation(view, pose, torch.zeros(1, 6, device=pose.device), joints, torch.zeros_like(joints))

    def set_state(self, paths, poses, velocities, articulations):
        """Install recorded states: rigid poses/velocities by prim path, articulations by path.

        `articulations` maps an articulation path to (root pose, root velocity, joint positions,
        joint velocities); joint drives hold the restored positions.
        """
        import torch

        view = self.bodies[0]
        device = self.bodies[1].device
        order = [paths.index(p) for p in self.paths]
        self._set_bodies(torch.as_tensor(np.asarray(poses)[order], dtype=torch.float32, device=device),
                         torch.as_tensor(np.asarray(velocities)[order], dtype=torch.float32, device=device))
        for path, (pose, velocity, joints, joint_velocities) in articulations.items():
            view = self.articulations[self.articulation_index(path)][0]
            tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device).reshape(1, -1)  # noqa: E731
            self._set_articulation(view, tensor(pose), tensor(velocity), tensor(joints), tensor(joint_velocities))

    def _set_bodies(self, pose, velocity):
        import torch

        view = self.bodies[0]
        ids = torch.arange(view.count, device=pose.device)
        view.set_transforms(pose, ids)
        view.set_velocities(velocity, ids)

    @staticmethod
    def _set_articulation(view, pose, velocity, joints, joint_velocities):
        import torch

        ids = torch.arange(view.count, device=pose.device)
        if not view.get_metatype(0).fixed_base:
            view.set_root_transforms(pose, ids)
        view.set_root_velocities(velocity, ids)
        if view.max_dofs:
            view.set_dof_positions(joints, ids)
            view.set_dof_velocities(joint_velocities, ids)
            view.set_dof_position_targets(joints, ids)
            view.set_dof_velocity_targets(torch.zeros_like(joints), ids)
