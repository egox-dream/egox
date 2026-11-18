"""Visualize G1-Sim-Dex3 motion and recorded main/wrist cameras without a simulator.

Draw measured joints, wrist targets and fingertips in a common coordinate frame.
Native recordings use their saved world poses; training-only files use the feet
frame. Playback uses saved simulation timestamps when available, otherwise the
recorded nominal frequency. Frames hold saved states without interpolating motion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from egox.robots.g1_sim_dex3.description import BODY_JOINT_NAMES, HAND_JOINT_NAMES, URDF

# Adapted from the original data_collect/data_vis/vis_g1_sim_motion.py.
SIDES = ('right', 'left')


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None = None
    child_frame_inverse: np.ndarray | None = None


def pose6_to_transform(pose):
    """EgoX XYZ + extrinsic XYZ Euler angles, in metres/radians."""
    pose = np.asarray(pose, dtype=np.float64)
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = Rotation.from_euler("xyz", pose[..., 3:]).as_matrix()
    result[..., :3, 3] = pose[..., :3]
    return result


def pose7_to_transform(pose):
    """Recorded native root/object pose: XYZ, quaternion WXYZ."""
    pose = np.asarray(pose, dtype=np.float64)
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(pose[..., [4, 5, 6, 3]]).as_matrix()
    result[..., :3, 3] = pose[..., :3]
    return result


def read_urdf(path):
    """Read the kinematic tree only; this does not load meshes or physics."""
    tree = ET.parse(path).getroot()
    links = {node.attrib["name"] for node in tree.findall("link")}
    joints = []
    for node in tree.findall("joint"):
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        origin = node.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        axis = node.find("axis")
        axis = np.array([1., 0., 0.]) if axis is None else np.fromstring(axis.attrib["xyz"], sep=" ")
        if np.linalg.norm(axis) == 0:
            raise ValueError(f"zero joint axis: {node.attrib['name']}")
        axis /= np.linalg.norm(axis)
        mimic = node.find("mimic")
        mimic = None if mimic is None else (mimic.attrib["joint"], float(mimic.attrib.get("multiplier", 1)), float(mimic.attrib.get("offset", 0)))
        joints.append(Joint(node.attrib["name"], node.find("parent").attrib["link"],
                            node.find("child").attrib["link"], node.attrib["type"],
                            pose6_to_transform(np.r_[xyz, rpy]), axis, mimic))
    roots = links - {joint.child for joint in joints}
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, got {roots}")
    return roots.pop(), joints


def forward_kinematics(root_link, joints, joint_names, joint_position, root_transform):
    """Vectorized named-joint FK, preserving saved ordering and floating root."""
    joint_position = np.asarray(joint_position, dtype=np.float64)
    positions = dict(zip(joint_names, joint_position.T))
    transforms = {root_link: np.asarray(root_transform, dtype=np.float64)}
    pending = list(joints)
    while pending:
        ready = [joint for joint in pending if joint.parent in transforms]
        if not ready:
            raise ValueError("URDF contains an unresolved parent or cycle")
        for joint in ready:
            motion = np.broadcast_to(np.eye(4), root_transform.shape).copy()
            if joint.kind != "fixed":
                if joint.mimic:
                    source, multiplier, offset = joint.mimic
                    values = positions[source] * multiplier + offset
                elif joint.name in positions:
                    values = positions[joint.name]
                else:
                    raise ValueError(f"recording has no position for movable URDF joint {joint.name}")
                if joint.kind in ("revolute", "continuous"):
                    motion[:, :3, :3] = Rotation.from_rotvec(values[:, None] * joint.axis).as_matrix()
                elif joint.kind == "prismatic":
                    motion[:, :3, 3] = values[:, None] * joint.axis
                else:
                    raise ValueError(f"unsupported joint type {joint.kind}")
            transform = transforms[joint.parent] @ joint.origin @ motion
            transforms[joint.child] = transform if joint.child_frame_inverse is None else transform @ joint.child_frame_inverse
            pending.remove(joint)
    return transforms


def recorded_hand_joints(joints, metadata):
    """Replace the IK URDF's hand subtrees with the recorded native hand model.

    USD joints have both parent and child anchor frames, unlike a simple URDF
    joint origin. Older exports include these native geometry declarations.
    """
    scene = metadata.get('scene', {})
    contract = metadata.get('hand_target_contract', scene.get('hand_target_contract'))
    if contract is None:
        return joints
    if 'geometry' not in contract:
        return joints  # Older proxy-hand contracts have no native skeleton.
    geometry = contract['geometry']['hands']
    descendants = {hand['root_link'] for hand in geometry.values()}
    replaced = set()
    while True:
        children = [joint for joint in joints if joint.parent in descendants and joint.name not in replaced]
        if not children:
            break
        replaced.update(joint.name for joint in children)
        descendants.update(joint.child for joint in children)
    result = [joint for joint in joints if joint.name not in replaced]
    for side, hand in geometry.items():
        for joint in hand['joints']:
            parent = pose7_to_transform([*joint['local_pos0_m'], *joint['local_rot0_wxyz']])
            child = pose7_to_transform([*joint['local_pos1_m'], *joint['local_rot1_wxyz']])
            axis = np.eye(3)['XYZ'.index(joint.get('axis', 'X'))] if joint['type'] != 'fixed' else np.array([1.,0.,0.])
            result.append(Joint(joint['name'], joint['parent'], joint['child'], joint['type'], parent,
                                axis, child_frame_inverse=np.linalg.inv(child)))
        for slot, tip in zip(contract['tip_order'], hand['fingertips']):
            if tip['valid']:
                result.append(Joint(f'{side}_{slot}_tip_marker', tip['link'], f'{side}_{slot}_tip_marker', 'fixed',
                                    pose6_to_transform([*tip['position_m'], 0.,0.,0.]), np.array([1.,0.,0.])))
    return result


def link_side(name):
    return 'right' if name.startswith(('right', 'R_')) else 'left' if name.startswith(('left', 'L_')) else 'body'


def sample_rows(timestamps, fps):
    """Hold-last resampling: never display a row from the playback future."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be a nonempty finite vector")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps run backwards; split the episode by reset/epoch first")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    relative = timestamps - timestamps[0]
    # One final output frame includes the last saved state, at most 1/fps later.
    playback = np.arange(int(np.ceil(relative[-1] * fps - 1e-8)) + 1) / fps
    rows = np.searchsorted(relative, playback + 1e-10, side="right") - 1
    return playback, np.clip(rows, 0, len(relative) - 1)


def trail_start(row, times, segments, seconds=1.0):
    """Trails end at every recorded segment boundary, regardless of its cause."""
    start = int(np.searchsorted(times, times[row] - seconds))
    changes = np.flatnonzero(segments[start:row] != segments[row])
    return start + int(changes[-1]) + 1 if len(changes) else start


def recorded_clock(attrs):
    """Legacy EgoX sim collection used wall time; do not infer from nominal Hz."""
    value = attrs.get("sampling_clock", "wall")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value not in ("simulation", "wall", "realtime"):
        raise ValueError(f"unsupported recorded sampling_clock: {value!r}")
    return str(value)


def playback_clock(requested, recorded):
    if requested not in ("recorded", "simulation", "wall"):
        raise ValueError(f"unsupported playback timebase: {requested!r}")
    return ("simulation" if recorded == "realtime" else recorded) if requested == "recorded" else requested


def _feet_frame(transforms):
    """Feet-to-display transform; the reference is rebuilt for every saved row."""
    left, right = (transforms[f'{side}_ankle_roll_link'] for side in ('left', 'right'))
    z = left[:, :3, 2] + right[:, :3, 2]
    z /= np.linalg.norm(z, axis=1)[:, None]
    x = left[:, :3, 0] + right[:, :3, 0]
    x -= np.sum(x * z, axis=1)[:, None] * z
    x /= np.linalg.norm(x, axis=1)[:, None]
    result = np.broadcast_to(np.eye(4), left.shape).copy()
    result[:, :3, :3] = np.stack([x, np.cross(z, x), z], axis=2)
    result[:, :3, 3] = (left[:, :3, 3] + right[:, :3, 3]) / 2
    return result


def _numeric(file, name, count, width=None):
    result = np.asarray(file[name], dtype=float)
    if len(result) != count or not np.isfinite(result).all() or (width is not None and result.shape != (count, width)):
        raise ValueError(f'{name}: invalid numeric rows')
    return result


def load_motion(path, urdf_override=None):
    """Read old native episodes, current state recordings, or training-only HDF5.

    Without a native root pose, draw in the measured feet frame. Never infer
    world travel by integrating commanded velocities or treating body_proprio as odometry.
    """
    with h5py.File(path, 'r') as f:
        embodiment = str(f.attrs.get('embodiment', 'G1-Sim-Dex3'))
        if embodiment not in ('G1-Sim-Dex3', 'g1_sim_dex3'):
            raise ValueError(f'{embodiment}: this viewer supports G1-Sim-Dex3 only')
        metadata = json.loads(f.attrs.get('metadata', '{}'))
        collection = json.loads(f.attrs.get('collection', '{}'))
        profile = json.loads(f.attrs.get('profile', '{}'))
        scene = metadata.get('scene', {})
        contract = json.loads(f.attrs.get('proprio_contract', '{}'))
        declarations = [item['proprio_semantics'] for item in (f.attrs, metadata, scene)
                        if 'proprio_semantics' in item]
        semantics = declarations[0] if declarations else ('g1_feet_kinematic_v1' if collection else 'episode_world_v1')
        if semantics not in ('g1_feet_kinematic_v1', 'episode_world_v1') or any(value != semantics for value in declarations):
            raise ValueError('unknown or inconsistent proprioception coordinate frames')
        urdf = Path(urdf_override or URDF)
        actual_hash = hashlib.sha256(urdf.read_bytes()).hexdigest()
        expected_hash = contract.get('source_urdf_sha256', scene.get('ik_urdf_sha256'))
        if expected_hash and actual_hash != expected_hash:
            raise ValueError('URDF hash differs from the recorded model; provide a matching --urdf')
        root_link, joints = read_urdf(urdf)
        joints = recorded_hand_joints(joints, metadata)
        count = len(f['actions/body'])
        if not count:
            raise ValueError('empty episode')
        native = f.get('replay_state', f.get('observations/sim_state'))
        native_world = native is not None
        if native_world:
            manifest = scene.get('replay_state', {})
            names = collection.get('joint_names', manifest.get('robot', {}).get('joint_names'))
            if names is None:
                raise ValueError('native replay state has no recorded joint-name order')
            q = _numeric(native, 'robot/joint_position', count, len(names))
            root = pose7_to_transform(_numeric(native, 'robot/pose_world_wxyz', count, 7))
            transforms = forward_kinematics(root_link, joints, names, q, root)
            root_velocity = _numeric(native, 'robot/velocity_world', count, 6)
        else:
            if semantics != 'g1_feet_kinematic_v1':
                raise ValueError('legacy world-frame data needs native replay states to reconstruct its frame')
            names = (*BODY_JOINT_NAMES, *HAND_JOINT_NAMES)
            q = np.concatenate([_numeric(f, 'observations/proprioceptions/robot_joint_pos_proprio', count, 29),
                                *[_numeric(f, f'observations/proprioceptions/{side}_eef_joint_pos_proprio', count, 7)
                                  for side in SIDES]], axis=1)
            identity = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            transforms = forward_kinematics(root_link, joints, names, q, identity)
            inverse = np.linalg.inv(_feet_frame(transforms))
            transforms = {name: inverse @ value for name, value in transforms.items()}
            root_velocity = None
        torso = transforms['torso_link']
        neutral = forward_kinematics(root_link, joints, names, np.zeros((1, len(names))), np.eye(4)[None])
        offset = np.array([.15, 0., .45]) - neutral['torso_link'][0, :3, 3]
        targets, measured, tips = {}, {}, {}
        for side in SIDES:
            for group, name, destination in (('actions', f'{side}_eef_to_body', targets),
                    ('observations/proprioceptions', f'{side}_eef_to_body_proprio', measured)):
                local = _numeric(f, f'{group}/{name}', count, 6).copy()
                local[:, :3] += offset
                destination[side] = torso @ pose6_to_transform(local)
            # Saved local/global poses must also agree with the independently
            # reconstructed skeleton, not just with each other.
            wrist = transforms[f'{side}_wrist_yaw_link'] @ pose6_to_transform(
                [.05, 0., 0., np.pi / 2 if side == 'right' else -np.pi / 2, 0., 0.])
            if (np.max(np.linalg.norm(wrist[:, :3, 3] - measured[side][:, :3, 3], axis=1)) > .002
                    or not np.allclose(wrist[:, :3, :3], measured[side][:, :3, :3], atol=.002, rtol=0)):
                raise ValueError(f'{side} saved wrist pose disagrees with measured-joint FK')
            for kind, group, name in (('targets', 'actions', f'{side}_eef_link'),
                                     ('measured', 'observations/proprioceptions', f'{side}_eef_link_proprio')):
                dataset = f'{group}/{name}'
                if dataset in f:
                    local = _numeric(f, dataset, count, 15).reshape(count, 5, 3)[:, :3]
                    transform = (targets if kind == 'targets' else measured)[side]
                    tips[side, kind] = np.einsum('nij,nkj->nki', transform[:, :3, :3], local) + transform[:, None, :3, 3]
        errors = {}
        if semantics == 'g1_feet_kinematic_v1':
            inverse = np.linalg.inv(_feet_frame(transforms))
            expected_body = inverse @ torso
            saved_body = pose6_to_transform(_numeric(f, 'observations/proprioceptions/body_proprio', count, 6))
            if not np.allclose(saved_body, expected_body, atol=2e-5, rtol=0):
                raise ValueError('saved body pose disagrees with measured-joint FK in the feet frame')
            global_eef = _numeric(f, 'observations/proprioceptions/eef_proprio', count, 12)
            for side, start in zip(SIDES, (0, 6)):
                expected = inverse @ measured[side]
                saved = pose6_to_transform(global_eef[:, start:start + 6])
                errors[side] = float(np.max(np.linalg.norm(saved[:, :3, 3] - expected[:, :3, 3], axis=1)))
                if not np.allclose(saved, expected, atol=2e-5, rtol=0):
                    raise ValueError(f'{side} saved local/global EEF poses disagree')
        else:
            frame = scene.get('episode_frame', {})
            origin = frame.get('origin_world_m', np.asarray(scene.get('robot_position', [0., 0., .8])) - [0., 0., .8])
            yaw = frame.get('yaw_world_rad', scene.get('robot_yaw', 0.))
            to_world = pose6_to_transform([*origin, 0., 0., yaw])
            global_eef = _numeric(f, 'observations/proprioceptions/eef_proprio', count, 12)
            for side, start in zip(SIDES, (0, 6)):
                saved = to_world @ pose6_to_transform(global_eef[:, start:start + 6])
                errors[side] = float(np.max(np.linalg.norm(saved[:, :3, 3] - measured[side][:, :3, 3], axis=1)))
                if errors[side] > .002:
                    raise ValueError(f'{side} saved episode-world EEF pose disagrees with replay FK')
        objects = {}
        if native_world and 'rigid_objects/pose_world_wxyz' in native:
            paths = collection.get('rigid_bodies', scene.get('replay_state', {}).get('rigid_objects', {}).get('prim_paths', []))
            poses = native['rigid_objects/pose_world_wxyz'][:]
            # Use the shortest unique path suffix: two plates may share a mesh name.
            for i, path in enumerate(paths):
                parts = path.strip('/').split('/')
                for depth in range(1, len(parts) + 1):
                    name = '/'.join(parts[-depth:])
                    if sum(other == name or other.endswith('/' + name) for other in paths) == 1:
                        break
                objects[name] = pose7_to_transform(poses[:, i])
        time_path = next((key for key in ('sim_time', 'observations/sim_time') if key in f), None)
        timing_source = 'recorded' if time_path else 'nominal'
        if time_path:
            sim_time = _numeric(f, time_path, count)
        else:
            frequency = float(f.attrs.get('frequency', collection.get('action_hz', profile.get('collection_hz', 30.))))
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError('episode frequency must be positive')
            sim_time = np.arange(count) / frequency
        sample_rows(sim_time, 30.)
        wall_path = next((key for key in ('observations/capture_time_ns', 'capture_time_ns') if key in f), None)
        wall_time = _numeric(f, wall_path, count) if wall_path else None
        if wall_time is not None:
            wall_time = (wall_time - wall_time[0]) / 1e9
        segments_path = 'segment_id' if 'segment_id' in f else 'observations/segment_id'
        segments = np.asarray(f[segments_path], dtype=np.int64) if segments_path in f else np.zeros(count, dtype=np.int64)
        if segments.shape != (count,):
            raise ValueError('segment IDs do not match episode length')
        image_names = [name for name in ('main', 'wrist_left', 'wrist_right') if 'observations/images/' + name in f]
        for name in image_names:
            if len(f['observations/images/' + name]) != count:
                raise ValueError(f'{name}: image rows do not match motion rows')
        return {'source': str(Path(path).resolve()), 'embodiment': embodiment,
                'urdf': str(urdf), 'urdf_sha256': actual_hash, 'joints': joints,
                'transforms': transforms, 'targets': targets, 'measured': measured, 'tips': tips,
                'objects': objects, 'sim_time': sim_time - sim_time[0], 'wall_time': wall_time,
                'timing_source': timing_source,
                'sampling_clock': recorded_clock(f.attrs) if metadata else 'realtime',
                'proprio_semantics': str(semantics), 'capture_mode': str(f.attrs.get('capture_mode', 'rgb')),
                'display_frame': 'world' if native_world else 'feet (recomputed each row)',
                'segments': segments, 'hand_names': HAND_JOINT_NAMES,
                'hand_q': q[:, [list(names).index(name) for name in HAND_JOINT_NAMES]],
                'hand_target': _numeric(f, 'actions/hand_joint_target', count, 14) if 'actions/hand_joint_target' in f else None,
                'velocity': _numeric(f, 'actions/velocity_commands', count, 3), 'fk_max_error_m': errors,
                'root_planar_speed': np.linalg.norm(root_velocity[:, :2], axis=1) if native_world else None,
                'appearance': scene.get('episode_randomization', {}).get('appearance', {}),
                'coating': {name: _numeric(f, 'observations/task_state/coating/' + name, count)
                            for name in ('bread_coverage', 'blade_ml') if 'observations/task_state/coating/' + name in f},
                'image_names': image_names}


def decode_image(dataset, row):
    pixels = dataset[row]
    if pixels.ndim == 1:
        import cv2
        result = cv2.imdecode(np.asarray(pixels, np.uint8), cv2.IMREAD_COLOR)
        if result is None:
            raise ValueError(f"invalid encoded image in row {row}")
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return pixels[..., :3]


def camera_notice(data):
    """Make absent training images explicit without blocking saved-state motion."""
    if data.get("image_names"):
        return None
    return ("No training RGB is stored in this episode; showing recorded motion only. "
            "Run the task render.sh first for camera views, then open its RGB output.")


def build_animation(data, image_file, fps=30., clock="simulation", trail_seconds=1., max_frames=None, *, snapshot_row=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    clock = playback_clock(clock, data["sampling_clock"])
    times = data["sim_time" if clock == "simulation" else "wall_time"]
    if times is None:
        raise ValueError('this episode has no recorded wall-clock capture timestamps; use simulation time')
    playback, rows = sample_rows(times, fps)
    if snapshot_row is not None:
        playback, rows = np.array([times[snapshot_row]]), np.array([snapshot_row])
    elif max_frames is not None:
        playback, rows = playback[:max_frames], rows[:max_frames]
    time_label = 'nominal' if clock == 'simulation' and data.get('timing_source') == 'nominal' else clock
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#f4f6fa")
    grid = fig.add_gridspec(4, 4, height_ratios=[1.25, 1.4, .10, .78],
                           left=.04, right=.98, top=.865, bottom=.13, wspace=.13, hspace=.22)
    title = fig.text(.035, .96, "G1 recorded motion", fontsize=21, weight="bold", color="#172336")
    status = fig.text(.035, .925, "", fontsize=11, color="#46566b")
    speed_status = fig.text(.035, .898, "", fontsize=11, color="#172336")
    ax = fig.add_subplot(grid[:2, :2], projection="3d", facecolor="#f4f6fa")
    ax.view_init(elev=23, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_xlabel("X (m)", labelpad=1)
    ax.set_title(data["display_frame"] + " frame", fontsize=10)
    ax.set_ylabel("Y (m)", labelpad=1)
    ax.set_zlabel("Z (m)", labelpad=1)
    transforms = data["transforms"]
    keep = [j for j in data["joints"] if not any(tag in j.child for tag in ("camera", "imu", "contour", "logo"))]
    segments = np.stack([np.stack([transforms[j.parent][:, :3, 3], transforms[j.child][:, :3, 3]], axis=1) for j in keep], axis=1)
    colors = [{"right": "#1675bb", "left": "#dc7731", "body": "#4b596c"}[link_side(j.child)] for j in keep]
    skeleton = Line3DCollection(segments[0], colors=colors, linewidths=[2.0 if "hand" in j.child else 4 for j in keep])
    ax.add_collection3d(skeleton)
    all_points = segments.reshape(-1, 3)
    if data["objects"]:
        all_points = np.concatenate([all_points, *(value[:, :3, 3] for value in data["objects"].values())])
    lower, upper = all_points.min(0), all_points.max(0)
    lower[:2] -= .12
    upper[:2] += .12
    lower[2] -= .1
    upper[2] = max(1.4, upper[2] + .12)
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    ax.set_box_aspect(upper - lower)
    axis_artists = {}
    trail_artists = {}
    for side, color in (("right", "#1675bb"), ("left", "#dc7731")):
        for kind, style in (("measured", "-"), ("targets", "--")):
            axis_artists[side, kind] = Line3DCollection(np.zeros((3, 2, 3)), colors=["#d74743", "#29985d", "#277cd0"], linewidths=1.5, linestyles=style, alpha=.9)
            ax.add_collection3d(axis_artists[side, kind])
            trail_artists[side, kind], = ax.plot([], [], [], color=color, lw=1.5, ls=style, alpha=.65)
    object_artists = {}
    object_colors = {"Bread": "#986538", "Spreader": "#bd416d", "OpenMilkCan": "#768443"}
    for name, transform in data["objects"].items():
        marker, = ax.plot([], [], [], marker="s" if name == "Bread" else "o", ms=7, color=object_colors.get(name, '#768443'), linestyle="none")
        label_name = name.split('/')[0].removeprefix('kitchen_').split('_0')[0].replace('_', ' ')
        label = ax.text(*transform[0, :3, 3], label_name, fontsize=7, color=object_colors.get(name, '#768443'))
        axis = Line3DCollection(np.zeros((1, 2, 3)), colors=object_colors.get(name, '#768443'), linewidths=2.8)
        ax.add_collection3d(axis)
        object_artists[name] = marker, label, axis
    ax.legend(handles=[Line2D([0], [0], color="#1675bb", lw=3, label="Right measured"),
                       Line2D([0], [0], color="#dc7731", lw=3, label="Left measured"),
                       Line2D([0], [0], color="#68758a", ls="--", label="Wrist target / trail")],
              loc="upper left", fontsize=8, frameon=False)
    image_artists = {}
    if camera_notice(data):
        image_ax = fig.add_subplot(grid[:2, 2:])
        image_ax.axis("off")
        image_ax.text(.5, .5, "Recorded motion only\n\nTraining camera images are absent.\n"
                      "Build the offline RGB dataset for camera views:\n"
                      "task render.sh",
                      ha="center", va="center", fontsize=13, color="#46566b",
                      transform=image_ax.transAxes, linespacing=1.6)
    locations = {"main": grid[0, 2:], "wrist_left": grid[1, 2], "wrist_right": grid[1, 3]}
    labels = {"main": "Recorded main camera · left / right stereo", "wrist_left": "Recorded left wrist", "wrist_right": "Recorded right wrist"}
    for name in data["image_names"]:
        image_ax = fig.add_subplot(locations[name])
        image_ax.axis("off")
        image_ax.set_title(labels[name], fontsize=10, loc="left", color="#46566b", pad=5)
        image_artists[name] = image_ax.imshow(decode_image(image_file["observations/images/" + name], 0))
    joint_ax = fig.add_subplot(grid[3, :2], facecolor="#f4f6fa")
    x = np.arange(len(data["hand_names"]))
    bar_colors = ["#1675bb" if link_side(name) == "right" else "#dc7731" for name in data["hand_names"]]
    bars = joint_ax.bar(x, data["hand_q"][0], width=.64, color=bar_colors, alpha=.8, label="Measured")
    target_marks = None
    if data['hand_target'] is not None:
        target_marks, = joint_ax.plot(x, data['hand_target'][0], '_', color='#172336', ms=12, mew=2, label='Current target')
    labels = [name.replace("right_hand_", "R ").replace("left_hand_", "L ").replace("R_", "R ").replace("L_", "L ")
              .replace("_proximal", "").replace("_joint", "").replace("_", " ") for name in data["hand_names"]]
    if len(labels) > 16:
        labels = [label if i % 2 == 0 else "" for i, label in enumerate(labels)]
    joint_ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    limits = [data['hand_q']] + ([] if data['hand_target'] is None else [data['hand_target']])
    low = min(float(value.min()) for value in limits) - .2
    high = max(float(value.max()) for value in limits) + .2
    joint_ax.set_ylim(low, high)
    joint_ax.set_title("Hand joint angles (rad)", loc="left", fontsize=10, color="#46566b")
    joint_ax.axhline(0, color="#8796a5", lw=.6)
    joint_ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    time_ax = fig.add_subplot(grid[3, 2:], facecolor="#f4f6fa")
    for i, (label, color) in enumerate(zip(("vx (m/s)", "vy (m/s)", "yaw (rad/s)"), ("#1675bb", "#dc7731", "#8062b6"))):
        # Break the line at segment transitions; no interpolation through gaps.
        values = data["velocity"][:, i].copy()
        values[np.r_[False, np.diff(data["segments"]) != 0]] = np.nan
        time_ax.plot(times, values, lw=1, label=label, color=color)
    if data['root_planar_speed'] is not None:
        root_speeds = data['root_planar_speed'].copy()
        root_speeds[np.r_[False, np.diff(data['segments']) != 0]] = np.nan
        time_ax.plot(times, root_speeds, lw=1., label='Measured XY speed (m/s)', color='#172336')
    cursor = time_ax.axvline(0, color="#172336", lw=1.6)
    time_ax.set_xlim(0, max(float(times[-1]), 1 / fps))
    time_ax.set_ylim(-1.7, 1.7)
    time_ax.set_xlabel(f"Elapsed {time_label} time (s)", labelpad=3)
    time_ax.set_title("Locomotion command" + (" and measured speed" if data['root_planar_speed'] is not None else ''),
                      loc="left", fontsize=10, color="#46566b")
    time_ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.text(.035, .014, "Solid robot = saved joint state + model FK. Dashed frames = current wrist targets. Objects = saved rigid poses. Hands include recorded tip offsets when available.", fontsize=9, color="#556477")
    tip_artists = {key: ax.plot([], [], [], marker='o' if key[1] == 'measured' else 'x',
                                color='#1675bb' if key[0] == 'right' else '#dc7731', ls='none', ms=4)[0]
                   for key in data['tips']}
    previous_row = [-1]

    def update(frame):
        row = int(rows[frame])
        start = trail_start(row, times, data["segments"], trail_seconds)
        skeleton.set_segments(segments[row])
        for (side, kind), artist in axis_artists.items():
            transform = data[kind][side][row]
            origin = transform[:3, 3]
            artist.set_segments(np.stack([np.broadcast_to(origin, (3, 3)), origin + .09 * transform[:3, :3].T], axis=1))
            points = data[kind][side][start:row + 1, :3, 3]
            trail_artists[side, kind].set_data_3d(points.T)
        for name, (marker, label, axis) in object_artists.items():
            transform = data["objects"][name][row]
            position = transform[:3, 3]
            marker.set_data_3d(position[:, None])
            label.set_position(position[:2] + [.02, .02])
            label.set_3d_properties(position[2] + .035, zdir=None)
            if name == "Spreader":
                bounds = np.asarray(data["appearance"].get("spreader", {}).get("blade_bounds_xyz_m", [[-.05, 0, 0], [.05, 0, 0]]))
                local = np.zeros((2, 3)); local[:, 0] = bounds[:, 0]
                axis.set_segments([(transform[:3, :3] @ local.T).T + position])
        if row != previous_row[0]:
            for name, artist in image_artists.items():
                artist.set_data(decode_image(image_file["observations/images/" + name], row))
            previous_row[0] = row
        for bar, height in zip(bars, data["hand_q"][row]):
            bar.set_height(height)
        if target_marks is not None:
            target_marks.set_ydata(data['hand_target'][row])
        for key, artist in tip_artists.items():
            artist.set_data_3d(data['tips'][key][row].T)
        cursor.set_xdata([playback[frame], playback[frame]])
        contents = "recorded motion and cameras" if image_artists else "recorded motion · RGB absent"
        title.set_text(f"{data['embodiment']} · {time_label.upper()} TIME · {contents}")
        boundary = row > 0 and data["segments"][row] != data["segments"][row - 1]
        coating = data["coating"]
        extra = f"  |  blade {coating['blade_ml'][row]:.2f} mL · bread {100 * coating['bread_coverage'][row]:.0f}% coated" if len(coating) == 2 else ""
        wall = f" | wall {data['wall_time'][row]:.2f} s" if data['wall_time'] is not None else ''
        status.set_text(f"Row {row + 1:,} / {len(times):,} | {time_label} {times[row]:.2f} s{wall} | segment {data['segments'][row]}" + (" · boundary" if boundary else "") + extra)
        vx, vy, yaw = data["velocity"][row]
        speed = f"Measured root XY speed {data['root_planar_speed'][row]:.3f} m/s | " if data['root_planar_speed'] is not None else ''
        speed_status.set_text(f"{speed}Command: vx {vx:+.2f}, vy {vy:+.2f} m/s, yaw {yaw:+.2f} rad/s")
        return []

    animation = FuncAnimation(fig, update, frames=len(rows), interval=1000 / fps, blit=False, repeat=True)
    return fig, animation, update, playback, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--save', '--output', dest='output', type=Path, help='PNG snapshot, GIF or MP4 animation')
    parser.add_argument('--no-show', '--no_show', action='store_true')
    parser.add_argument('--urdf', type=Path, help='Matching robot URDF; defaults to bundled Dex3 model')
    parser.add_argument('--timebase', '--clock', dest='clock', choices=('simulation', 'wall', 'recorded'), default='simulation')
    parser.add_argument('--fps', type=float, default=30., help='Playback rate; duration follows saved timestamps')
    parser.add_argument('--trail-seconds', type=float, default=1.)
    parser.add_argument('--frame', '--preview-row', type=int, default=0, dest='frame', help='Saved row for PNG/initial view')
    parser.add_argument('--max-frames', type=int, help='Limit exported playback frames')
    args = parser.parse_args(argv)
    if args.frame < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('--frame must be nonnegative and --max-frames positive')
    if not np.isfinite(args.fps) or args.fps <= 0 or not np.isfinite(args.trail_seconds) or args.trail_seconds < 0:
        parser.error('--fps must be positive and --trail-seconds nonnegative')
    if args.output and args.output.suffix.lower() not in ('.png', '.gif', '.mp4'):
        parser.error('--save must end in .png, .gif or .mp4')
    if args.no_show or args.output:
        import matplotlib
        matplotlib.use('Agg')
    data = load_motion(args.file, args.urdf)
    if notice := camera_notice(data):
        print(notice)
    print(f"{data['embodiment']}: {len(data['sim_time'])} rows; display frame: {data['display_frame']}")
    if data['timing_source'] == 'nominal':
        print('Simulation timestamps are absent; playback uses the recorded nominal frequency.')
    if args.frame >= len(data['sim_time']):
        parser.error('--frame is outside the episode')
    with h5py.File(args.file, 'r') as file:
        snapshot = args.frame if (args.output and args.output.suffix.lower() == '.png') or (args.no_show and not args.output) else None
        fig, animation, update, playback, rows = build_animation(data, file, args.fps, args.clock,
                                                               args.trail_seconds, args.max_frames, snapshot_row=snapshot)
        import matplotlib.pyplot as plt
        update(int(np.argmin(abs(rows - args.frame))))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() == '.png':
                fig.savefig(args.output, dpi=100)
            else:
                from matplotlib.animation import FFMpegWriter, PillowWriter
                writer = PillowWriter(fps=args.fps) if args.output.suffix.lower() == '.gif' else FFMpegWriter(fps=args.fps)
                animation.save(str(args.output), writer=writer)
            print(f'Saved {args.output}')
        if args.no_show or args.output:
            fig.canvas.draw()
            plt.close(fig)
        else:
            plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
