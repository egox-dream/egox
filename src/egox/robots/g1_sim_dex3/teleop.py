"""Pico's common tracking frame -> G1-Sim-Dex3 training actions, using NumPy only."""

import numpy as np

from .controller import OBS_MASKS, SIDES, matrix_rpy, rpy_matrix


def _rpy(rotation):
    # Preserve the old collector's Euler convention at pitch +/-90 degrees.
    cy = np.hypot(rotation[0, 0], rotation[1, 0])
    if cy < 1e-6:
        return np.array([np.arctan2(-rotation[1, 2], rotation[1, 1]), np.arctan2(-rotation[2, 0], cy), 0.])
    return matrix_rpy(rotation)


class G1Dex3Teleop:
    """Calibrate head heading/height, then produce the controller's six action fields.

    ``config`` comes from the running controller's ``describe`` response: its
    nominal grasp curve, WBC command bounds and standing height. Reusing that
    curve keeps collection and rollout hand targets in the same robot frame,
    without loading a second kinematics model or policy in the collection process.
    The collector owns reset/engage/pause events and rejects stale input.
    """

    action_masks = {'body_pose': np.array([False, False] + [True] * 4),
                    **{f'{side}_eef_link_pose': OBS_MASKS[f'{side}_eef_link_proprio'] for side in SIDES}}

    def __init__(self, config):
        self.grasp_curve = {}
        for side in SIDES:
            curve = np.asarray(config['grasp_curve'][side], dtype=float)
            if curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] != 9 or not np.isfinite(curve).all():
                raise ValueError(f'{side} grasp_curve must contain at least two finite 9-coordinate samples')
            self.grasp_curve[side] = curve.copy()
        self.command_bounds = np.asarray(config['command_bounds'], dtype=float)
        self.standing_height = float(config['standing_height'])
        if (self.command_bounds.shape != (2, 7) or not np.isfinite(self.command_bounds).all()
                or np.any(self.command_bounds[0] > self.command_bounds[1]) or not np.isfinite(self.standing_height)):
            raise ValueError('invalid controller command bounds or standing height')
        self.reset()

    @property
    def calibrated(self):
        return self.origin is not None

    def reset(self):
        self.origin = None
        self.heading = np.eye(3)

    def calibrate(self, frame):
        if not frame.valid:
            raise ValueError('cannot calibrate invalid tracking')
        frame.validate()
        head = np.asarray(frame.head)
        self.heading = rpy_matrix([0., 0., -_rpy(head[:3, :3])[2]])
        self.origin = self.heading @ head[:3, 3]

    @staticmethod
    def tracking_recovery_error(previous_frame, previous, frame, action):
        """The old simulator's continuity limits for resuming after a brief PICO gap."""
        if any(np.max(np.abs(a['velocity_commands'])) > .01 for a in (previous, action)):
            return 'tracking recovered with a locomotion command; center sticks'
        if np.max(np.abs(np.asarray(frame.grasp) - np.asarray(previous_frame.grasp))) > .2:
            return 'hand trigger changed during tracking loss'
        if np.linalg.norm(action['body_pose'][:3] - previous['body_pose'][:3]) > .05:
            return 'head position moved during tracking loss'
        for key, position_limit, angle_limit in (('body_pose', None, 15.),
                ('right_eef_to_body_pose', .08, 20.), ('left_eef_to_body_pose', .08, 20.)):
            a, b = previous[key], action[key]
            relative = rpy_matrix(a[3:]).T @ rpy_matrix(b[3:])
            angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.)))
            if angle > angle_limit or (position_limit is not None and np.linalg.norm(a[:3] - b[:3]) > position_limit):
                return f'{key} moved during tracking loss'
        return None

    def action(self, frame):
        if not self.calibrated:
            raise ValueError('calibration is required')
        if not frame.valid:
            raise ValueError('tracking is invalid')
        frame.validate()
        head = np.asarray(frame.head)
        height = (self.heading @ head[:3, 3] - self.origin)[2]
        command = np.r_[frame.velocity, self.standing_height + height, _rpy(self.heading @ head[:3, :3])]
        command = np.clip(command, *self.command_bounds)
        # Body x/y are unused and masked, as in the old finalized training labels.
        result = {'body_pose': np.r_[0., 0., command[3] - self.standing_height, command[4:]],
                  'velocity_commands': command[:3]}
        for side, grasp in zip(SIDES, frame.grasp):
            wrist = np.asarray(getattr(frame, f'{side}_wrist'))
            position = head[:3, :3].T @ (wrist[:3, 3] - head[:3, 3])
            if np.linalg.norm(position) > 1.5:
                raise ValueError(f'{side} wrist target is outside the supported 1.5 m workspace')
            # Pico already supplies the anatomical wrist axes; IK owns robot offsets.
            result[f'{side}_eef_to_body_pose'] = np.r_[position, _rpy(head[:3, :3].T @ wrist[:3, :3])]
            curve = self.grasp_curve[side]
            location = float(grasp) * (len(curve) - 1)
            index = min(int(location), len(curve) - 2)
            tips = curve[index] + (location - index) * (curve[index + 1] - curve[index])
            result[f'{side}_eef_link_pose'] = np.r_[tips, np.zeros(6)]  # no ring/pinky on Dex3
        return {key: value.astype(np.float32) for key, value in result.items()}
