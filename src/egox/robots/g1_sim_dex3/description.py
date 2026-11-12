"""Asset locations, joint names and default pose of the G1 Dex3; needs no simulator or controller."""
import json
from pathlib import Path

import yaml

ASSETS = Path(__file__).resolve().parents[4] / 'assets/robots/g1_dex3'
URDF = ASSETS / 'g1_body29_hand14.urdf'
USD = ASSETS / 'usd/g1_29dof_with_dex3_rev_1_0.usd'  # from scripts/download_assets.py
WBC_CONFIG = ASSETS / 'wbc/g1_student.yaml'
CAMERA_RIG = ASSETS / 'camera_rig.json'

SIDES = ('right', 'left')
BODY_JOINT_NAMES = tuple(
    f'{side}_{name}_joint' for side in ('left', 'right')
    for name in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
) + ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint') + tuple(
    f'{side}_{name}_joint' for side in ('left', 'right')
    for name in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))
ARM_JOINT_NAMES = BODY_JOINT_NAMES[15:]  # the first 15 (legs, waist) belong to the whole-body controller
HAND_JOINT_NAMES = tuple(f'{side}_hand_{name}_joint' for side in SIDES
                         for name in ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1'))


def default_joint_positions():
    """The whole-body controller's default pose, in radians, by joint name."""
    wbc = yaml.safe_load(WBC_CONFIG.read_text())
    return dict(zip(BODY_JOINT_NAMES, wbc['lower_default_joint_pos'] + wbc['arm_default_joint_pos']))


def camera_rig():
    """The stereo head camera and two wrist cameras of the G1-Sim-Dex3 dataset."""
    return json.loads(CAMERA_RIG.read_text())
