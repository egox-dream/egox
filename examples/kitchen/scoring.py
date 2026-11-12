"""Scores of the two G1-Sim-Dex3 kitchen tasks (NumPy/SciPy only).

Kitchen-Bread (`kitchen_nine_stage_v3`): nine ordered progress points (open the fridge,
grasp the can, close the fridge, carry the can, place it, pick the spreader, dip it, spread,
place it). Success: at least 5 % newly coated bread at the end and the spreader released
and at rest above the floor at the end.

Kitchen-Cleanup (`spreader_four_stage_v1`): four ordered points (grasp, lift, blade-first
insertion, released supported placement), from the holder evaluator's 200 Hz latches.
Success: the placement was reached at any physics step.

The stage points (`stage_score`) are the paper's task scores; `task_score` adds
SUCCESS_BONUS for a successful trial (at most 10 for Kitchen-Bread, 5 for Kitchen-Cleanup), and
`normalized_score` divides it by that maximum (1 for a perfect trial of either task).
Both read the per-step trace recorded by evaluate.py and the trial's frozen `scoring` data.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

BREAD_STAGES = ('open', 'grasp', 'close', 'locomotion', 'place_can', 'pick_knife', 'insert', 'spread', 'place_knife')
CLEANUP_STAGES = ('grasp', 'lift', 'insert', 'release_place')
INSERT_CONDITIONS = ('inserted', 'inside_opening', 'handle_clear', 'no_bottom_penetration', 'blade_first',
                     'holder_upright')
PLACEMENT_CONDITIONS = ('grasp_observed', 'lift_observed', *INSERT_CONDITIONS, 'supported', 'slow',
                        'open_command', 'released', 'hand_withdrawn')
SUCCESS_BONUS = 1  # points added to the stage points of a successful trial


@dataclass(frozen=True)
class BreadThresholds:
    open_angle_deg: float = 25.
    closed_angle_deg: float = 10.
    grasp_tip_centroid_distance_m: float = .085
    grasp_lift_m: float = .025
    grasp_transport_m: float = .06
    grasp_window_s: float = .20
    grasp_relative_span_m: float = .035
    release_tip_centroid_distance_m: float = .11
    object_rest_speed_m_s: float = .10
    object_rest_angular_speed_rad_s: float = 1.5
    placement_window_s: float = .20
    can_support_tolerance_m: float = .025
    can_upright_deg: float = 25.
    can_plate_distance_m: float = .65
    locomotion_displacement_m: float = .35
    locomotion_approach_m: float = .30
    robot_plate_distance_m: float = .85
    spread_added_coverage: float = .05
    knife_minimum_height_above_floor_m: float = .30
    maximum_event_sampling_gap_s: float = .15


def rotations(pose_wxyz):
    return Rotation.from_quat(np.asarray(pose_wxyz)[..., [4, 5, 6, 3]]).as_matrix()


def lowest_can_surface_z(pose, radius, height, thickness):
    """Lowest authored can surface (bottom cylinder plus 24 wall pieces); tipping is not lifting."""
    pose = np.asarray(pose, dtype=float)
    rotation = rotations(pose)
    radial_z, axis_z = rotation[:, 2, :2], rotation[:, 2, 2]
    cylinder = pose[:, 2] + np.minimum(0., thickness * axis_z) - radius * np.linalg.norm(radial_z, axis=1)
    angles = np.arange(24) * (2 * np.pi / 24)
    corners = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    walls = pose[:, 2] + (radial_z @ corners.T).min(1) + np.minimum(0., height * axis_z)
    return np.minimum(cylinder, walls)


def bread_features(trace, scoring):
    """Scorer inputs from a Kitchen-Bread trace (world poses XYZ+WXYZ) and the trial's scoring data."""
    times = np.asarray(trace['sim_time'], dtype=float)
    can, tool, plate = (np.asarray(trace[f'{k}_pose']) for k in ('can', 'spreader', 'plate'))
    cv, tv = np.asarray(trace['can_velocity']), np.asarray(trace['spreader_velocity'])
    robot = np.asarray(trace['robot_pose'])
    geometry = scoring['can']
    can_center = can[:, :3] + rotations(can)[:, :, 2] * geometry['height_m'] / 2
    handle_center = np.asarray(scoring['spreader']['handle_anchor_xyz_m'], dtype=float).copy()
    handle_center[0] += scoring['spreader']['handle_size_m'][0] / 2
    tool_center = tool[:, :3] + np.einsum('nij,j->ni', rotations(tool), handle_center)
    hand = scoring['hand']
    active = np.asarray(hand['active_tip_mask'], dtype=bool).reshape(5, 3).all(1)
    n = len(times)
    tips, wrists, wrist_rotations = [], [], []
    for side in ('left', 'right'):
        wrist = np.asarray(trace[f'{side}_wrist_pose'])
        wrist_r = rotations(wrist)
        local = (np.asarray(trace[f'{side}_eef_link_proprio']).reshape(n, 5, 3)[:, active]
                 @ np.asarray(hand['wrist_to_egox_rotation'][side]) + hand['eef_origin_in_wrist_m'])
        tips.append(wrist[:, :3] + np.einsum('nij,nj->ni', wrist_r, local.mean(1)))
        wrists.append(wrist[:, :3])
        wrist_rotations.append(wrist_r)
    tips, wrists, wr = np.stack(tips, 1), np.stack(wrists, 1), np.stack(wrist_rotations, 1)
    relative = lambda p: np.einsum('nsji,nsj->nsi', wr, p[:, None] - wrists)  # noqa: E731
    scene_r = Rotation.from_euler('z', float(scoring['scene_yaw'])).as_matrix()
    origin = np.asarray(scoring['scene_position'], dtype=float)
    local = lambda p: (p - origin) @ scene_r  # noqa: E731
    bar_shift = np.asarray(scoring['bar_translation_scene_m'])
    features = dict(
        time=times - times[0], door_deg=np.degrees(np.abs(np.asarray(trace['door_angle']))),
        floor_world_z=float(origin[2]),
        can=can[:, :3], can_local=local(can[:, :3]), can_center=can_center,
        can_lowest_world_z=lowest_can_surface_z(can, geometry['diameter_m'] / 2, geometry['height_m'],
                                                geometry['bottom_thickness_m']),
        can_up=rotations(can)[:, 2, 2], can_speed=np.linalg.norm(cv[:, :3], axis=1),
        can_angular_speed=np.linalg.norm(cv[:, 3:], axis=1),
        knife=tool[:, :3], knife_local=local(tool[:, :3]), knife_handle=tool_center,
        knife_speed=np.linalg.norm(tv[:, :3], axis=1), knife_angular_speed=np.linalg.norm(tv[:, 3:], axis=1),
        can_tip_distance=np.linalg.norm(tips - can_center[:, None], axis=2),
        knife_tip_distance=np.linalg.norm(tips - tool_center[:, None], axis=2),
        can_in_wrist=relative(can_center), knife_in_wrist=relative(tool_center),
        plate=plate[:, :3], robot=robot[:, :3], robot_up=rotations(robot)[:, 2, 2],
        table_bounds=np.asarray(scoring['table_xy_bounds_m']) + bar_shift[[0, 1, 0, 1]],
        table_height=float(scoring['table_height_m']))
    for key in ('bread_coverage', 'bread_ml', 'loaded_ml', 'transferred_ml', 'dip_count'):
        features[key] = np.asarray(trace[f'coating_{key}'], dtype=float)
    return features


def held_evidence(features, kind, cfg):
    """Object lifted and moved, close to a hand, with bounded wrist-relative slip."""
    time, position = features['time'], features[kind]
    lift_height = features['can_lowest_world_z'] if kind == 'can' else position[:, 2]
    lifted = lift_height - lift_height[0] >= cfg.grasp_lift_m
    transported = np.linalg.norm(position - position[0], axis=1) >= cfg.grasp_transport_m
    near = features[kind + '_tip_distance'] <= cfg.grasp_tip_centroid_distance_m
    relative = features[kind + '_in_wrist']
    held = np.zeros((len(time), 2), dtype=bool)
    for i, t in enumerate(time):
        start = int(np.searchsorted(time, t - cfg.grasp_window_s, side='right')) - 1
        if start < 0 or start == i or np.max(np.diff(time[start:i + 1])) > cfg.maximum_event_sampling_gap_s:
            continue
        span = np.linalg.norm(np.ptp(relative[start:i + 1], axis=0), axis=1)
        held[i] = near[start:i + 1].all(0) & (span <= cfg.grasp_relative_span_m) & lifted[i] & transported[i]
    return held


def score_bread(f, cfg=None):
    cfg = cfg or BreadThresholds()
    time = np.asarray(f['time'])
    stages, indices = {}, {}

    def event(name, mask, requires=(), dwell=0.):
        dependencies = [indices.get(k) for k in requires]
        earliest = max((i for i in dependencies if i is not None), default=-1)
        valid = np.asarray(mask, dtype=bool).copy()
        if any(i is None for i in dependencies):
            valid[:] = False
        valid[:earliest + 1] = False
        index = start = None
        for i in np.flatnonzero(valid):
            if start is None or not valid[i - 1] or time[i] - time[i - 1] > cfg.maximum_event_sampling_gap_s:
                start = i
            if time[i] - time[start] >= dwell - 1e-8:
                index = int(i)
                break
        indices[name] = index
        stages[name] = {'points': int(index is not None), 'first_event_sim_s': None if index is None else float(time[index]),
                        'requires': list(requires)}
        return index
    can_held, knife_held = held_evidence(f, 'can', cfg), held_evidence(f, 'knife', cfg)
    event('open', (f['door_deg'] >= cfg.open_angle_deg) & (f['door_deg'] - f['door_deg'][0] >= cfg.open_angle_deg))
    event('grasp', can_held.any(1), ('open',))
    event('close', f['door_deg'] <= cfg.closed_angle_deg, ('open', 'grasp'))
    displacement = np.linalg.norm(f['robot'][:, :2] - f['robot'][0, :2], axis=1)
    plate_distance = np.linalg.norm(f['robot'][:, :2] - f['plate'][0, :2], axis=1)
    approach = plate_distance[0] - plate_distance
    carry = np.linalg.norm(f['can'][:, :2] - f['can'][0, :2], axis=1)
    event('locomotion', (displacement >= cfg.locomotion_displacement_m) & (approach >= cfg.locomotion_approach_m)
          & (plate_distance <= cfg.robot_plate_distance_m) & (carry >= cfg.locomotion_displacement_m)
          & can_held.any(1), ('grasp',))
    bounds = f['table_bounds']
    on_table = ((f['can_local'][:, 0] >= bounds[0]) & (f['can_local'][:, 0] <= bounds[2])
                & (f['can_local'][:, 1] >= bounds[1]) & (f['can_local'][:, 1] <= bounds[3]))
    can_plate = np.linalg.norm(f['can'][:, :2] - f['plate'][:, :2], axis=1)
    can_support = np.abs(f['can_local'][:, 2] - f['table_height']) <= cfg.can_support_tolerance_m
    can_released = (f['can_tip_distance'] > cfg.release_tip_centroid_distance_m).all(1)
    can_rest = (f['can_speed'] <= cfg.object_rest_speed_m_s) & (f['can_angular_speed'] <= cfg.object_rest_angular_speed_rad_s)
    event('place_can', on_table & can_support & can_rest & can_released
          & (f['can_up'] >= np.cos(np.radians(cfg.can_upright_deg))) & (can_plate <= cfg.can_plate_distance_m),
          ('grasp', 'locomotion'), cfg.placement_window_s)
    pickup = event('pick_knife', knife_held.any(1), ('locomotion',))
    dip_base = 0 if pickup is None else pickup
    insertion = event('insert', (f['dip_count'] > f['dip_count'][dip_base])
                      & (f['loaded_ml'] > f['loaded_ml'][dip_base] + 1e-6), ('pick_knife',))
    added = f['bread_coverage'] - f['bread_coverage'][0]
    spread_base = 0 if insertion is None else insertion
    event('spread', (f['bread_coverage'] - f['bread_coverage'][spread_base] >= cfg.spread_added_coverage)
          & (f['transferred_ml'] > f['transferred_ml'][spread_base] + 1e-6), ('insert',))
    height = f['knife'][:, 2] - f['floor_world_z']
    knife_released = (f['knife_tip_distance'] > cfg.release_tip_centroid_distance_m).all(1)
    knife_placed = ((height >= cfg.knife_minimum_height_above_floor_m) & (f['knife_speed'] <= cfg.object_rest_speed_m_s)
                    & (f['knife_angular_speed'] <= cfg.object_rest_angular_speed_rad_s) & knife_released)
    event('place_knife', knife_placed, ('pick_knife', 'spread'), cfg.placement_window_s)
    # Success is independent of the progress points; a later drop invalidates it.
    tail = int(np.searchsorted(time, time[-1] - cfg.placement_window_s, side='right')) - 1
    terminal_placement = bool(tail >= 0 and knife_placed[tail:].all()
                              and np.max(np.diff(time[tail:])) <= cfg.maximum_event_sampling_gap_s)
    coating_success = bool(added[-1] >= cfg.spread_added_coverage)
    score = sum(v['points'] for v in stages.values())
    return {'metric': 'kitchen_nine_stage_v3', 'success': coating_success and terminal_placement,
            'task_score': score, 'maximum_task_score': 9, 'stages': stages, 'thresholds': asdict(cfg),
            'duration_sim_s': float(time[-1] - time[0]), 'num_states': len(time),
            'robot_fell': bool(np.any(f['robot'][:, 2] < .45) or np.any(f['robot_up'] < .5)),
            'final_added_bread_coverage': float(added[-1]), 'final_spreader_placement_verified': terminal_placement}


def cleanup_features(trace):
    """Scorer inputs from a Kitchen-Cleanup trace: the holder evaluator's state at every sample."""
    return {'time': np.asarray(trace['sim_time'], dtype=float),
            'task_state': {key[len('holder_'):]: np.asarray(value) for key, value in trace.items()
                           if key.startswith('holder_')},
            'robot_fell': bool(np.any(np.asarray(trace['robot_tilt_deg']) > 60.)
                               or np.any(np.asarray(trace['robot_pose'])[:, 2] < .45))}


def score_cleanup(features):
    time = np.asarray(features['time'], dtype=float)
    time = time - time[0]
    n = len(time)
    task = features['task_state']
    conditions = {key: np.asarray(task['conditions_' + key], dtype=bool) for key in (*PLACEMENT_CONDITIONS, 'retained')}
    flags = {key: np.asarray(task[key], dtype=bool) for key in
             ('success', 'ever_success', 'ever_grasped', 'ever_lifted', 'right_hand_tool_contact', 'holder_support_contact')}
    measured = {key: np.asarray(task[key], dtype=float) for key in
                ('right_hand_openness', 'right_hand_clearance_m', 'linear_speed_m_s', 'angular_speed_rad_s',
                 'insertion_depth_m')}
    current = np.logical_and.reduce([conditions[key] for key in PLACEMENT_CONDITIONS])
    if not np.array_equal(current, flags['success']):
        raise ValueError('holder success disagrees with its placement conditions')
    reached = flags['ever_success'] | flags['success']
    valid_insert = np.logical_and.reduce([conditions[key] for key in INSERT_CONDITIONS]) & flags['ever_lifted']
    # The 200 Hz success latch proves an insertion between two 30 Hz samples.
    masks = (flags['ever_grasped'], flags['ever_lifted'], valid_insert | reached, reached)
    stages, previous = {}, None
    for i, (name, mask) in enumerate(zip(CLEANUP_STAGES, masks)):
        candidates = (np.flatnonzero(mask & (np.arange(n) >= (previous or 0))) if i == 0 or previous is not None
                      else [])
        index = int(candidates[0]) if len(candidates) else None
        stages[name] = {'points': int(index is not None), 'first_observed_sim_s': None if index is None else float(time[index]),
                        'requires': [] if i == 0 else [CLEANUP_STAGES[i - 1]]}
        previous = index
    score = sum(item['points'] for item in stages.values())
    return {'metric': 'spreader_four_stage_v1', 'success': bool(reached.any()), 'final_success': bool(current[-1]),
            'task_score': score, 'maximum_task_score': 4, 'stages': stages,
            'duration_sim_s': float(time[-1]), 'num_states': n, 'robot_fell': bool(features.get('robot_fell', False)),
            'final_measurements': {key: float(value[-1]) for key, value in measured.items()},
            'final_right_hand_tool_contact': bool(flags['right_hand_tool_contact'][-1])}


def score_trial(task, trace, scoring):
    if task == 'kitchen_bread':
        result = score_bread(bread_features(trace, scoring))
    else:
        result = score_cleanup(cleanup_features(trace))
    result['stage_score'] = result['task_score']
    result['task_score'] += SUCCESS_BONUS * result['success']
    result['maximum_task_score'] += SUCCESS_BONUS
    result['normalized_score'] = result['task_score'] / result['maximum_task_score']
    return result


def summarize(results):
    """Success rate, mean task score and mean normalized score per split (ID / OOD) and overall."""
    def stats(items):
        if not items:
            return None
        return {'trials': len(items), 'successes': sum(r['success'] for r in items),
                'success_rate': float(np.mean([r['success'] for r in items])),
                'mean_task_score': float(np.mean([r['task_score'] for r in items])),
                'task_score_std': float(np.std([r['task_score'] for r in items])),
                'maximum_task_score': items[0]['maximum_task_score'],
                'mean_normalized_score': float(np.mean([r['normalized_score'] for r in items])),
                'normalized_score_std': float(np.std([r['normalized_score'] for r in items])),
                'robot_falls': sum(r['robot_fell'] for r in items)}
    return {'all': stats(results), 'id': stats([r for r in results if r['group'] == 'id']),
            'ood': stats([r for r in results if r['group'] != 'id'])}

