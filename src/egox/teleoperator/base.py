"""What every teleoperator provides, whichever headset it comes from."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

SIDES = ('right', 'left')  # the order of every per-hand array
STALE, INVALID = 'stale_packet', 'invalid_tracking'  # TeleopFrame.tracking_status


def checked_transform(value):
    """A copy of a pose as a proper 4x4 rigid transform; ValueError if it is not one."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('pose must be a finite 4x4 transform')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('pose has an invalid homogeneous row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
        raise ValueError('pose rotation must be orthonormal with determinant +1')
    return matrix.copy()


@dataclass
class TeleopFrame:
    """One sample. Poses are 4x4 transforms in the EgoX world: metres, x-forward, y-left, z-up.

    Check `valid` before use: while it is False the poses are identities, not the operator's.
    """

    head: np.ndarray
    right_wrist: np.ndarray
    left_wrist: np.ndarray
    source_time_ns: int  # local monotonic time this sample arrived
    device_time_ns: int = 0  # the headset's own timestamp, if it has one
    grasp: np.ndarray = field(default_factory=lambda: np.zeros(2))  # right, left; 0 open, 1 closed
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # forward m/s, left m/s, yaw rad/s
    events: frozenset = frozenset()  # 'reset', 'engage', 'pause', 'save', 'discard'; reported once per press
    # Right, left (2, 25, 3): wrist, 4 thumb, then 5 per finger, in the EgoX wrist frame; tips are at
    # 4, 9, 14, 19 and 24. None when the headset provides no landmarks.
    hand_landmarks: np.ndarray | None = None
    hand_valid: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    raw_hands: np.ndarray | None = None  # hand tracking in the headset's own layout; never robot targets
    valid: bool = True
    reason: str = ''  # why the frame is invalid, for people
    tracking_status: str = ''  # why the frame is invalid, for code: STALE or INVALID
    source: str = ''

    def validate(self):
        for pose in (self.head, self.right_wrist, self.left_wrist):
            checked_transform(pose)
        for name, shape in (('grasp', (2,)), ('velocity', (3,))):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'{name} must have shape {shape} and finite values')
        if np.any(np.asarray(self.grasp) < 0) or np.any(np.asarray(self.grasp) > 1):
            raise ValueError('grasp values must be within [0, 1]')
        return self

    @classmethod
    def invalid(cls, source, time_ns, reason, tracking_status=INVALID, device_time_ns=0):
        return cls(np.eye(4), np.eye(4), np.eye(4), time_ns, device_time_ns, valid=False, reason=str(reason),
                   tracking_status=tracking_status, source=source)


class ButtonEdges:
    """Report each press once, independent of hand or trigger geometry.

    A press held through a tracking outage must not start anything on recovery, so after
    `interrupt` only stopping inputs are reported, which can still cancel a pending recovery.
    """

    STOPPING = frozenset({'pause', 'save', 'discard', 'reset'})

    def __init__(self):
        self.previous = set()
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True

    def update(self, pressed):
        pressed = set(pressed)
        edges = frozenset(pressed - self.previous)
        self.previous = pressed
        if self.interrupted:
            self.interrupted = False
            return frozenset(pressed & self.STOPPING)
        return edges


class Teleoperator(ABC):
    """A VR input device. `poll` never blocks and never raises on lost tracking."""

    @abstractmethod
    def poll(self) -> TeleopFrame:
        """The latest sample; invalid while tracking is lost or stale."""

    def close(self):
        """Release the device; safe to call twice."""

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
