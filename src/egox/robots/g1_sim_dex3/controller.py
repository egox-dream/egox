"""Controller for the simulated Unitree G1 (29 DoF) with Dex3 hands.

Maps EgoX G1-Sim-Dex3 actions to joint targets, and measured joints to EgoX
proprioception, in the conventions of the G1-Sim-Dex3 dataset:

- lower body (legs, waist): a 50 Hz whole-body-control (WBC) policy tracks the
  velocity, height and torso-orientation commands;
- arms: numerical IK reaches the head-relative wrist poses, with gravity feedforward;
- hands: fingertip targets are projected onto the Dex3 open/close curve.

Simulator independent (NumPy, SciPy, PyTorch, Pinocchio); the simulator owns physics.
"""

from pathlib import Path

import numpy as np
import yaml

from .description import ARM_JOINT_NAMES, BODY_JOINT_NAMES, HAND_JOINT_NAMES, SIDES, URDF, WBC_CONFIG

HAND_CLOSED = np.array([.35, -.55, -1.2, 1., 1.2, 1., 1.2,
                        -.35, .55, 1.2, -1., -1.2, -1., -1.2])  # grasp preset; the open hand is all zeros
# Fingertip points on the distal links, in EgoX order thumb, index, middle (Dex3 has no ring/pinky).
FINGERTIPS = {side: ((f'{side}_hand_thumb_2_link', (0., .05 if side == 'right' else -.05, 0.)),
                     (f'{side}_hand_index_1_link', (.05, 0., 0.)),
                     (f'{side}_hand_middle_1_link', (.05, 0., 0.))) for side in SIDES}
EEF_IN_WRIST = np.array([.05, 0., 0.])        # EgoX end-effector origin in the wrist yaw link
HEAD_TO_ARM_BASE = np.array([.15, 0., .45])   # head-relative wrist commands -> fixed-waist arm frame
STANDING_HEIGHT = .75
# Availability masks of the recorded G1-Sim-Dex3 proprioception; other fields are fully available.
OBS_MASKS = {'body_pose_proprio': np.array([False, False] + [True] * 4),
             'right_eef_link_proprio': np.array([True] * 9 + [False] * 6),
             'left_eef_link_proprio': np.array([True] * 9 + [False] * 6)}


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


EGOX_FROM_WRIST = {'right': rot_x(-np.pi / 2), 'left': rot_x(np.pi / 2)}  # wrist link -> EgoX end-effector axes


def rpy_matrix(rpy):
    """Extrinsic XYZ roll/pitch/yaw, i.e. Rz(yaw) Ry(pitch) Rx(roll)."""
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_rpy(r):
    return np.array([np.arctan2(r[2, 1], r[2, 2]), np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])),
                     np.arctan2(r[1, 0], r[0, 0])])


def pose_vector(transform, side):
    """4x4 wrist-link transform -> EgoX [xyz, rpy] of the end effector."""
    return np.r_[transform[:3, 3], matrix_rpy(transform[:3, :3] @ EGOX_FROM_WRIST[side].T)]


def validate_action(action):
    """Check a single physical-unit action before changing controller state."""
    widths = {'body_pose': 6, 'velocity_commands': 3,
              **{f'{side}_eef_to_body_pose': 6 for side in SIDES},
              **{f'{side}_eef_link_pose': 15 for side in SIDES}}
    result = {}
    for key, width in widths.items():
        if key not in action:
            raise ValueError(f'missing action field: {key}')
        value = np.asarray(action[key], dtype=float)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f'{key} must contain {width} finite values')
        result[key] = value
    return result


class G1Dex3SimController:
    def __init__(self, urdf=URDF, wbc_config=WBC_CONFIG):
        import pinocchio as pin
        import torch

        # Full body + hand model for proprioception, fixed-waist arm model for IK (as on the real G1).
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q_index = [self.model.joints[self.model.getJointId(n)].idx_q for n in BODY_JOINT_NAMES + HAND_JOINT_NAMES]
        self.frames = {name: self.model.getFrameId(name) for name in
                       ('torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link',
                        'right_wrist_yaw_link', *(link for tips in FINGERTIPS.values() for link, _ in tips))}
        locked = [i for i in range(1, self.model.njoints) if self.model.names[i] not in ARM_JOINT_NAMES]
        self.arm = pin.buildReducedModel(self.model, locked, pin.neutral(self.model))
        self.eef = {side: self.arm.addFrame(pin.Frame(
            f'{side}_egox_eef', self.arm.getJointId(f'{side}_wrist_yaw_joint'), self.arm.getFrameId(f'{side}_wrist_yaw_link'),
            pin.SE3(np.eye(3), EEF_IN_WRIST), pin.FrameType.OP_FRAME)) for side in SIDES}
        self.arm_data = self.arm.createData()
        self.arm_index = [self.arm.joints[self.arm.getJointId(n)].idx_q for n in ARM_JOINT_NAMES]
        self.arm_bounds = (self.arm.lowerPositionLimit + 1e-6, self.arm.upperPositionLimit - 1e-6)

        config_path = Path(wbc_config)
        self.wbc = yaml.safe_load(config_path.read_text())
        self.policy = torch.jit.load(str(config_path.parent / self.wbc['policy_path']), map_location='cpu').eval()
        self.lower_default = np.array(self.wbc['lower_default_joint_pos'])
        self.arm_default = np.array(self.wbc['arm_default_joint_pos'])
        ranges = self.wbc['command_range']
        self.command_bounds = np.array([ranges[k] for k in (
            'lin_vel_x', 'lin_vel_y', 'ang_vel_z', 'height', 'body_roll', 'body_pitch', 'body_yaw')]).T

        # Valid fingertips along the nominal open -> closed grasp, to invert fingertip targets.
        grasp = np.linspace(0., 1., 4097)[:, None]
        tips = [self.fingertips(np.zeros(29), g * HAND_CLOSED) for g in grasp]
        self.grasp_curve = {side: np.stack([t[f'{side}_eef_link_proprio'][:9] for t in tips]) for side in SIDES}
        self.reset(self.arm_default)

    def reset(self, arm_q):
        """Neutral standing command, empty WBC history, IK seeded at the measured arm."""
        self.command = np.array([0., 0., 0., STANDING_HEIGHT, 0., 0., 0.])
        self.history = np.zeros((2, 58), dtype=np.float32)
        self.previous_action = np.zeros(15, dtype=np.float32)
        self.ik_seed = np.clip(self._arm_q(arm_q), *self.arm_bounds)

    # ------------------------------------------------------------------ commands

    def set_action(self, action):
        """EgoX action -> (14 arm targets, 14 arm feedforward torques, 14 hand targets).

        `action` holds `body_pose` (6, x/y unused), `velocity_commands` (3), and per side
        `{side}_eef_to_body_pose` (6, head-relative wrist) and `{side}_eef_link_pose` (15 fingertips).
        Also stores the WBC command used by `lower_body`.
        """
        action = validate_action(action)
        import pinocchio as pin

        body = action['body_pose']
        command = np.r_[np.asarray(action['velocity_commands'], dtype=float), STANDING_HEIGHT + body[2], body[3:6]]
        self.command = np.clip(command, *self.command_bounds)
        targets = {}
        for side in SIDES:
            pose = np.asarray(action[f'{side}_eef_to_body_pose'], dtype=float)
            targets[side] = pin.SE3(rpy_matrix(pose[3:]) @ EGOX_FROM_WRIST[side], pose[:3] + HEAD_TO_ARM_BASE)
        arm_target = self.solve_ik(targets)
        grasp = [self.grasp_from_fingertips(np.asarray(action[f'{side}_eef_link_pose'], dtype=float)[:9], side)
                 for side in SIDES]
        hand_target = np.repeat(grasp, 7) * HAND_CLOSED
        return arm_target, self.arm_gravity(arm_target), hand_target

    def solve_ik(self, targets):
        """Bounded least squares on wrist position/rotation, regularized toward zero and the last solution."""
        import pinocchio as pin
        from scipy.optimize import least_squares

        seed = self.ik_seed.copy()

        def residual(q):
            pin.framesForwardKinematics(self.arm, self.arm_data, q)
            errors = []
            for side in ('left', 'right'):  # the solver's iterates depend on this order
                pose, target = self.arm_data.oMf[self.eef[side]], targets[side]
                errors += [np.sqrt(50.) * (pose.translation - target.translation),
                           pin.log3(pose.rotation @ target.rotation.T)]
            return np.concatenate(errors + [np.sqrt(.02) * q, np.sqrt(.1) * (q - seed)])

        result = least_squares(residual, np.clip(seed, *self.arm_bounds), bounds=self.arm_bounds, max_nfev=30,
                               ftol=1e-5, xtol=1e-5, gtol=1e-5)
        self.ik_seed = result.x
        return result.x[self.arm_index]

    def arm_gravity(self, arm_q):
        """Arm gravity torques at `arm_q` (fixed waist; the WBC handles the lower body)."""
        import pinocchio as pin

        zero = np.zeros(self.arm.nv)
        return pin.rnea(self.arm, self.arm_data, self._arm_q(arm_q), zero, zero)[self.arm_index].copy()

    def grasp_from_fingertips(self, tips, side):
        """Closest point on the nominal grasp curve, as a fraction in [0, 1]."""
        start, delta = self.grasp_curve[side][:-1], np.diff(self.grasp_curve[side], axis=0)
        alpha = np.clip(np.sum((tips - start) * delta, axis=1) / np.sum(delta * delta, axis=1), 0., 1.)
        index = np.argmin(np.sum((start + alpha[:, None] * delta - tips) ** 2, axis=1))
        return (index + alpha[index]) / len(start)

    def lower_body(self, base_ang_vel, projected_gravity, lower_q, lower_dq):
        """One 50 Hz WBC update: 15 leg/waist position targets from the base and lower-body state."""
        import torch

        self.history[0] = self.history[1]
        self.history[1] = np.r_[base_ang_vel, projected_gravity, self.command, np.asarray(lower_q) - self.lower_default,
                                lower_dq, self.previous_action]
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(self.history.reshape(1, -1).clip(-100., 100.)))
        self.previous_action = action[0].numpy().clip(-100., 100.)
        return self.lower_default + self.wbc['action_scale'] * self.previous_action

    # ------------------------------------------------------------------ proprioception

    def proprio(self, body_q, hand_q):
        """EgoX proprioception from the 29 measured body joints and 14 hand joints (right, then left)."""
        import pinocchio as pin

        self._forward(body_q, hand_q)
        pose = lambda name: self.data.oMf[self.frames[name]].homogeneous
        # Feet frame: midpoint of the ankles, z from their mean up axis, x from their mean heading.
        left, right = pose('left_ankle_roll_link'), pose('right_ankle_roll_link')
        z = left[:3, 2] + right[:3, 2]
        z /= np.linalg.norm(z)
        x = left[:3, 0] + right[:3, 0]
        x -= x.dot(z) * z
        x /= np.linalg.norm(x)
        feet = np.eye(4)
        feet[:3, :3] = np.stack([x, np.cross(z, x), z])
        feet[:3, 3] = -feet[:3, :3] @ (left[:3, 3] + right[:3, 3]) / 2
        torso = feet @ pose('torso_link')
        eef = np.eye(4)
        eef[:3, 3] = EEF_IN_WRIST
        result = {'body_pose_proprio': np.r_[torso[:3, 3], matrix_rpy(torso[:3, :3])],
                  'eef_pose_proprio': np.concatenate(
                      [pose_vector(feet @ pose(f'{side}_wrist_yaw_link') @ eef, side) for side in SIDES]),
                  'robot_joint_pos_proprio': np.asarray(body_q),
                  'right_eef_joint_pos_proprio': np.asarray(hand_q[:7]),
                  'left_eef_joint_pos_proprio': np.asarray(hand_q[7:])}
        pin.framesForwardKinematics(self.arm, self.arm_data, self._arm_q(np.asarray(body_q)[15:]))
        for side in SIDES:
            transform = self.arm_data.oMf[self.eef[side]].homogeneous.copy()
            transform[:3, 3] -= HEAD_TO_ARM_BASE
            result[f'{side}_eef_to_body_pose_proprio'] = pose_vector(transform, side)
        result.update(self._fingertips())
        return {key: np.asarray(value, dtype=np.float32) for key, value in result.items()}

    def fingertips(self, body_q, hand_q):
        """15 fingertip coordinates per side in the EgoX end-effector frame (ring/pinky are zero)."""
        self._forward(body_q, hand_q)
        return self._fingertips()

    def _fingertips(self):
        result = {}
        for side in SIDES:
            wrist = self.data.oMf[self.frames[f'{side}_wrist_yaw_link']].inverse()
            tips = np.zeros((5, 3))
            for i, (link, offset) in enumerate(FINGERTIPS[side]):
                tips[i] = EGOX_FROM_WRIST[side] @ (wrist * self.data.oMf[self.frames[link]] * np.array(offset) - EEF_IN_WRIST)
            result[f'{side}_eef_link_proprio'] = tips.reshape(15).astype(np.float32)
        return result

    def _forward(self, body_q, hand_q):
        import pinocchio as pin

        q = pin.neutral(self.model)
        q[self.q_index] = np.r_[body_q, hand_q]
        pin.framesForwardKinematics(self.model, self.data, q)

    def _arm_q(self, arm_q):
        q = np.zeros(self.arm.nq)
        q[self.arm_index] = arm_q
        return q
