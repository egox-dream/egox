"""PICO: headset and two controllers, with triggers for grasp, joysticks for velocity, and buttons."""
import importlib
import json
from pathlib import Path
import time

import numpy as np

from .base import INVALID, STALE, SIDES, ButtonEdges, Teleoperator, TeleopFrame, checked_transform

# XRoboToolkit sends native tracking coordinates (x-right, y-up, z-backward), not Unity's z-forward
# display coordinates. See https://github.com/XR-Robotics/XRoboToolkit-PC-Service
OPENXR_TO_EGOX = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
# A controller is not an anatomical wrist. These local offsets make a forward, thumb-up grip face the
# G1's palms inward in the canonical wrist frame.
CONTROLLER_TO_WRIST = {
    'right': np.array([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]),
    'left': np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]]),
}
BUTTONS = {'get_X_button': 'reset', 'get_A_button': 'engage', 'get_B_button': 'pause',
           'get_Y_button': 'save', 'get_left_menu_button': 'discard'}
JOYSTICK_DEADZONE = .15
G1_VELOCITY_LIMITS = (.55, .43, 1.22)  # the command range of assets/robots/g1_dex3/wbc/g1_student.yaml


class _StalePacket(ValueError):
    pass


def pose_from_xyzw(value, name='SDK pose'):
    """An SDK pose [x, y, z, qx, qy, qz, qw] as a transform; rejects untracked zero quaternions."""
    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f'{name} must have seven finite values')
    norm = np.linalg.norm(pose[3:])
    if norm < .5 or norm > 1.5:
        raise ValueError(f'{name} has an invalid quaternion (norm={norm:.3g})')
    x, y, z, w = pose[3:] / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    matrix[:3, 3] = pose[:3]
    return matrix


class Pico(Teleoperator):
    """Read a PICO headset and controllers through XRoboToolkit; needs no body tracker.

    Needs the XRoboToolkit PC service running on this machine, its headset app connected to it, and
    its Python binding `xrobotoolkit_sdk`, all from https://github.com/XR-Robotics

    `calibration` (a JSON path or a dict) may replace `world_basis` and the
    `right_controller_to_wrist` / `left_controller_to_wrist` offsets; an explicit offset replaces the
    default even if it is the identity.

    The SDK's getters have separate locks and no freshness per device, so a stable packet timestamp is
    checked around the reads, and a stale packet or an invalid pose makes the frame invalid. This cannot
    detect one frozen controller while the headset still updates.
    """

    def __init__(self, sdk=None, calibration=None, max_age_s=.5, velocity_scale=G1_VELOCITY_LIMITS,
                 clock_ns=time.monotonic_ns, initialize=True):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        self.velocity_scale = np.asarray(velocity_scale, dtype=float)
        if (self.velocity_scale.shape != (3,) or not np.isfinite(self.velocity_scale).all()
                or np.any(self.velocity_scale < 0)):
            raise ValueError('velocity_scale must contain three finite nonnegative limits')
        if sdk is None:
            try:
                sdk = importlib.import_module('xrobotoolkit_sdk')
            except ImportError as error:
                raise RuntimeError('PICO needs xrobotoolkit_sdk and the XRoboToolkit PC service') from error
        self.sdk, self.clock_ns = sdk, clock_ns
        self.max_age_ns = int(max_age_s * 1e9)
        config = json.loads(Path(calibration).read_text()) if isinstance(calibration, (str, Path)) else (calibration or {})
        self.basis = np.asarray(config.get('world_basis', OPENXR_TO_EGOX), dtype=float)
        if self.basis.shape != (3, 3) or not np.allclose(self.basis.T @ self.basis, np.eye(3), atol=1e-6):
            raise ValueError('world_basis must be an orthogonal 3x3 coordinate conversion')
        self.offsets = {side: checked_transform(config.get(f'{side}_controller_to_wrist', CONTROLLER_TO_WRIST[side]))
                        for side in SIDES}
        self.edges = ButtonEdges()
        self.last_stamp = None
        self.last_received_ns = 0
        self.wait_for_new_stamp = False
        self.closed = False
        if initialize:
            self.sdk.init()

    def _pose(self, raw, offset=None, name='SDK pose'):
        raw = pose_from_xyzw(raw, name)
        pose = np.eye(4)
        pose[:3, 3] = self.basis @ raw[:3, 3]
        pose[:3, :3] = self.basis @ raw[:3, :3] @ self.basis.T
        return pose if offset is None else pose @ offset

    def _snapshot(self):
        """Every getter once, retried until the packet timestamp is the same before and after."""
        sdk = self.sdk
        for _ in range(3):
            stamp = int(sdk.get_time_stamp_ns())
            poses = sdk.get_headset_pose(), sdk.get_right_controller_pose(), sdk.get_left_controller_pose()
            grasp = np.asarray([sdk.get_right_trigger(), sdk.get_left_trigger()], dtype=float)
            axes = np.asarray([sdk.get_left_axis(), sdk.get_right_axis()], dtype=float)
            if grasp.shape != (2,) or not np.isfinite(grasp).all():
                raise ValueError('SDK triggers must contain two finite values')
            if axes.shape != (2, 2) or not np.isfinite(axes).all():
                raise ValueError('SDK joystick axes must each contain two finite values')
            grasp = np.clip(grasp, 0., 1.)
            pressed = {event for getter, event in BUTTONS.items() if bool(getattr(sdk, getter)())}
            raw_hands, hand_valid = np.zeros((2, 26, 7)), np.zeros(2, dtype=bool)
            for i, side in enumerate(SIDES):
                if getattr(sdk, f'get_{side}_hand_is_active', lambda: False)():
                    hand = np.asarray(getattr(sdk, f'get_{side}_hand_tracking_state')(), dtype=float)
                    if hand.shape == (26, 7) and np.isfinite(hand).all():
                        raw_hands[i], hand_valid[i] = hand, True
            if stamp == int(sdk.get_time_stamp_ns()):
                return stamp, poses, grasp, axes, pressed, raw_hands, hand_valid
        raise ValueError('SDK packet changed during every snapshot attempt')

    def _check_stamp(self, stamp, now):
        if stamp <= 0:
            raise ValueError('waiting for the first SDK packet')
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.last_stamp, self.wait_for_new_stamp = stamp, True
            raise ValueError('SDK timestamp restarted; waiting for a new packet')
        if self.wait_for_new_stamp and stamp == self.last_stamp:
            raise ValueError('waiting for an advancing SDK timestamp')
        if stamp != self.last_stamp:
            self.last_stamp, self.last_received_ns = stamp, now
            self.wait_for_new_stamp = False
        if now - self.last_received_ns > self.max_age_ns:
            age_ms = (now - self.last_received_ns) / 1e6
            raise _StalePacket(f'PICO packet timestamp has not advanced for {age_ms:.0f} ms')

    def poll(self):
        now = self.clock_ns()
        if self.closed:
            return TeleopFrame.invalid('pico', now, 'PICO is closed')
        try:
            stamp, poses, grasp, (left_axis, right_axis), pressed, raw_hands, hand_valid = self._snapshot()
            self._check_stamp(stamp, now)
            head = self._pose(poses[0], name='headset pose')
            right = self._pose(poses[1], self.offsets['right'], name='right controller pose')
            left = self._pose(poses[2], self.offsets['left'], name='left controller pose')
            axes = np.array([left_axis[1], -left_axis[0], -right_axis[0]], dtype=float)
            axes[np.abs(axes) < JOYSTICK_DEADZONE] = 0.
            return TeleopFrame(head, right, left, self.last_received_ns, stamp, grasp,
                               np.clip(axes, -1., 1.) * self.velocity_scale, self.edges.update(pressed),
                               hand_valid=hand_valid, raw_hands=raw_hands, source='pico').validate()
        except (ValueError, TypeError, IndexError, AttributeError, RuntimeError, OverflowError) as error:
            self.edges.interrupt()
            stale = isinstance(error, _StalePacket)
            return TeleopFrame.invalid('pico', self.last_received_ns if stale else now, error,
                                       STALE if stale else INVALID, self.last_stamp or 0)

    def close(self):
        if not self.closed:
            self.closed = True
            self.sdk.close()
