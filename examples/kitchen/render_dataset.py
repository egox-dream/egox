"""Render saved simulation states into training HDF5 episodes and camera MP4s.

"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from egox.configs.loader import load_robot_config
from egox.data.mapping import KEY_MASK_PATH_MAP, KEY_PATH_MAP
from egox.teleoperator.data_collecter import EpisodeWriter, _publish

REPO = Path(__file__).resolve().parents[2]
TIMING_FIELDS = ('segment_id', 'sim_time', 'state_time_ns', 'frame_id')
VIDEO_CAMERAS = {'main_image': 'main_cam', 'left_wrist_image': 'wrist_left_cam',
                 'right_wrist_image': 'wrist_right_cam'}


def _datasets(group):
    values = {}

    def remember(name, item):
        if isinstance(item, h5py.Dataset):
            values[name] = item

    group.visititems(remember)
    return values


def _state_shapes(metadata):
    joints = len(metadata['joint_names'])
    shapes = {'robot/pose_world_wxyz': (7,), 'robot/velocity_world': (6,),
              'robot/joint_position': (joints,), 'robot/joint_velocity': (joints,), 'physics_steps': ()}
    if metadata['rigid_bodies']:
        count = len(metadata['rigid_bodies'])
        shapes.update({'rigid_objects/pose_world_wxyz': (count, 7), 'rigid_objects/velocity_world': (count, 6)})
    if len(metadata['articulations']) != len(metadata['articulation_joint_names']):
        raise ValueError('articulation metadata does not declare each joint order')
    for index, names in enumerate(metadata['articulation_joint_names']):
        prefix = f'articulations/articulation_{index:03d}/'
        shapes.update({prefix + 'pose_world_wxyz': (7,), prefix + 'velocity_world': (6,),
                       prefix + 'joint_position': (len(names),), prefix + 'joint_velocity': (len(names),)})
    if 'coating' in metadata['trial']:
        coating = metadata['trial']['coating']['config']
        shapes['coating/blade_thickness_full'] = (2, *coating['blade_resolution'])
        shapes['coating/bread_thickness_full'] = tuple(coating['bread_resolution'])
    return shapes


def inspect_episode(path):
    """Check the raw episode before allocating a renderer or creating output."""
    path = Path(path)
    if not path.name.endswith('.state.h5'):
        raise ValueError(f'{path}: expected a collected .state.h5 episode')
    with h5py.File(path, 'r') as source:
        if source.attrs.get('capture_mode') != 'state_only':
            raise ValueError(f'{path}: expected state_only collection')
        metadata = json.loads(source.attrs['collection'])
        config = load_robot_config(source.attrs['embodiment'])
        fields = {**config['tokenizer']['proprioceptions'], **config['action']}
        count = len(source[KEY_PATH_MAP[next(iter(config['action']))]])
        if not count:
            raise ValueError(f'{path}: empty recording')
        for key, field in fields.items():
            data = source[KEY_PATH_MAP[key]]
            if data.shape != (count, *field['shape']) or data.dtype != np.float32:
                raise ValueError(f'{path}: invalid recorded shape/dtype for {key}')
            mask = source.get(KEY_MASK_PATH_MAP[key])
            if mask is not None and (mask.shape != tuple(field['shape']) or mask.dtype != np.bool_):
                raise ValueError(f'{path}: invalid static mask for {key}')
        states = _datasets(source['replay_state'])
        for name, shape in _state_shapes(metadata).items():
            if name not in states or states[name].shape != (count, *shape):
                raise ValueError(f'{path}: missing or malformed native replay state {name}')
        for name, data in {**{KEY_PATH_MAP[k]: source[KEY_PATH_MAP[k]] for k in fields}, **states}.items():
            if data.ndim < 1 or data.shape[0] != count or data.dtype.kind not in 'iuf':
                raise ValueError(f'{path}: invalid replay rows for {name}')
            for start in range(0, count, 128):
                values = data[start:start + 128]
                if not np.isfinite(values).all():
                    raise ValueError(f'{path}: nonfinite {name}')
                if name.endswith('pose_world_wxyz') and (values.shape[-1] != 7 or not np.allclose(
                        np.linalg.norm(values[..., 3:], axis=-1), 1., atol=1e-3, rtol=0)):
                    raise ValueError(f'{path}: invalid world WXYZ pose in {name}')
                if name.startswith('coating/') and np.any(values < 0):
                    raise ValueError(f'{path}: negative coating thickness')
        times, segments = source['sim_time'][:], source['segment_id'][:]
        if (times.shape != (count,) or not np.isfinite(times).all() or np.any(times < 0)
                or segments.shape != (count,) or segments.dtype.kind not in 'iu'
                or np.any(segments < 0) or np.any(np.diff(segments) < 0)
                or np.any(np.diff(times) < 0)
                or np.any((np.diff(times) <= 0) & (np.diff(segments) == 0))):
            raise ValueError(f'{path}: invalid sample times or segment boundaries')
        for name in ('state_time_ns', 'frame_id'):
            values = source[name][:]
            if (values.shape != (count,) or values.dtype.kind not in 'iu'
                    or np.any(values < 0) or np.any(values[1:] <= values[:-1])):
                raise ValueError(f'{path}: invalid recorded {name}')
        steps = states['physics_steps'][:]
        hz = metadata['physics_hz']
        if (not np.isfinite(hz) or hz <= 0 or steps.dtype.kind not in 'iu' or np.any(steps < 0)
                or not np.allclose(times, steps / hz, atol=1e-9, rtol=0)):
            raise ValueError(f'{path}: recorded times do not match native physics steps')
        boundaries = np.r_[0, np.flatnonzero(np.diff(segments)) + 1, count]
        return metadata, boundaries


def output_paths(source, output):
    stem = Path(source).name.removesuffix('.state.h5')
    return [Path(output) / (stem + '.hdf5')]


class _RenderedEpisodeWriter(EpisodeWriter):
    def save(self, destination, source):
        """Keep timing and collection metadata in the partial file until publication."""
        if len(self) != len(source['segment_id']):
            raise ValueError('rendered episode must retain every source row')
        for name in TIMING_FIELDS:
            source.copy(name, self._file)
        self._file.attrs.update(capture_mode='rgb', collection=source.attrs['collection'])
        return super().save(destination)


def build_dataset(source_path, output, render_row, *, jpeg_quality=95):
    """Render with render_row(row, flat_state); publish one complete episode.

    The injected renderer returns RGB images keyed by policy observation names.
    Saved numeric values are never recomputed from the renderer or resampled.
    """
    metadata, boundaries = inspect_episode(source_path)
    destinations = output_paths(source_path, output)
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)
    with h5py.File(source_path, 'r') as source:
        states = _datasets(source['replay_state'])
        with _RenderedEpisodeWriter(output, source.attrs['embodiment'], jpeg_quality) as writer:
            obs_keys = writer.observations.keys() - writer.images.keys()
            masks = {key: source[KEY_MASK_PATH_MAP[key]][()] for key in writer.fields
                     if key not in writer.images and KEY_MASK_PATH_MAP[key] in source}
            for row in range(int(boundaries[-1])):
                state = {name: value[row] for name, value in states.items()}
                state['sim_time'] = float(source['sim_time'][row])
                images = render_row(row, state)
                if set(images) != set(writer.images):
                    raise ValueError('offline renderer must return every configured camera')
                rig = metadata['camera_rig']
                for key, image in images.items():
                    views = [rig['cameras'][view]['intrinsics'] for view in rig['outputs'][key.removesuffix('_image')]]
                    shape = (views[0]['height'], sum(view['width'] for view in views), 3)
                    if np.asarray(image).shape != shape:
                        raise ValueError(f'{key}: offline camera shape must be {shape}')
                obs = {key: source[KEY_PATH_MAP[key]][row] for key in obs_keys}
                action = {key: source[KEY_PATH_MAP[key]][row] for key in writer.actions}
                writer.append({'obs': {**obs, **images}, 'obs_mask': masks}, action, masks)
            writer.save(destinations[0], source)
    return destinations


def episode_video_path(episode, camera='main_image'):
    episode = Path(episode)
    return episode.with_name(f'{episode.stem}_{VIDEO_CAMERAS[camera]}_video.mp4')


def export_episode_video(episode, camera='main_image'):
    """Export one camera at its saved resolution, with one video frame per row.

    Nominal collection FPS controls playback; recorded timing and segment gaps
    remain in HDF5. Decode saved images so video and training data show the same rows.
    """
    import cv2

    episode = Path(episode)
    destination = episode_video_path(episode, camera)
    if destination.exists():
        raise FileExistsError(destination)
    temporary, writer = None, None
    try:
        with h5py.File(episode, 'r') as source:
            images = source[KEY_PATH_MAP[camera]]
            if not len(images):
                raise ValueError(f'{episode}: no {camera} frames')
            metadata = json.loads(source.attrs.get('collection', '{}'))
            fps = float(metadata.get('action_hz', source.attrs.get('frequency', 30)))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f'{episode}: invalid collection frequency')
            shape = None
            for row in range(len(images)):
                pixels = images[row]
                if source.attrs.get('compress', False):
                    frame = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
                else:
                    frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f'{episode}: invalid {camera} at row {row}')
                if shape is None:
                    shape = frame.shape
                    if shape[0] % 2 or shape[1] % 2:
                        raise ValueError('MP4 preview requires even image dimensions without cropping')
                    fd, name = tempfile.mkstemp(prefix='.video_', suffix='.mp4', dir=destination.parent)
                    os.close(fd)
                    temporary = Path(name)
                    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (shape[1], shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('OpenCV could not open the MP4 encoder')
                elif frame.shape != shape:
                    raise ValueError(f'{episode}: {camera} shape changed at row {row}')
                writer.write(frame)
            writer.release()
            writer = None
            # Check the finalized container before publication.
            video = cv2.VideoCapture(str(temporary))
            try:
                if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(images):
                    raise RuntimeError('MP4 encoder did not retain every episode frame')
            finally:
                video.release()
        _publish(temporary, destination)
        return destination
    finally:
        if writer is not None:
            writer.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_episode_videos(episode):
    """Create missing main/wrist previews for the cameras present in an episode."""
    with h5py.File(episode, 'r') as source:
        cameras = [camera for camera in VIDEO_CAMERAS if KEY_PATH_MAP[camera] in source]
    if not cameras:
        raise ValueError(f'{episode}: no supported camera images')
    for camera in cameras:
        destination = episode_video_path(episode, camera)
        if destination.exists():
            print(f'Skipped existing {destination}', flush=True)
        else:
            print(f'Saved {export_episode_video(episode, camera)}', flush=True)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _recorded_order(state, actual, recorded):
    """Match readback rows/joints to the named order saved with the recording."""
    result = dict(state)
    for group, order_key, fields in (
            ('robot', 'joint_names', ('joint_position', 'joint_velocity')),
            ('rigid_objects', 'rigid_bodies', ('pose_world_wxyz', 'velocity_world'))):
        if recorded[order_key]:
            order = [actual[order_key].index(name) for name in recorded[order_key]]
            for field in fields:
                key = f'{group}/{field}'
                result[key] = state[key][order]
    for index, path in enumerate(recorded['articulations']):
        native = actual['articulations'].index(path)
        source = f'articulations/articulation_{native:03d}/'
        target = f'articulations/articulation_{index:03d}/'
        order = [actual['articulation_joint_names'][native].index(name)
                 for name in recorded['articulation_joint_names'][index]]
        for field in ('pose_world_wxyz', 'velocity_world', 'joint_position', 'joint_velocity'):
            value = state[source + field]
            result[target + field] = value[order] if field.startswith('joint_') else value
    return result


def render_state(env, state, metadata, render_frames, reset_accumulation):
    """Hold one native state for camera convergence; verify no physics was integrated."""
    env.restore_state(state, metadata)
    steps = env.physics_steps
    reset_accumulation()
    # observe() performs the final render and camera readback.
    for _ in range(render_frames - 1):
        env.render()
    observation = env.observe()
    if env.physics_steps != steps:
        raise RuntimeError('offline rendering unexpectedly advanced physics')
    if not np.isclose(observation['sim_time'], state['sim_time'], atol=1e-9, rtol=0):
        raise RuntimeError('offline rendering returned a different simulation time')
    restored = _recorded_order(env.snapshot_state(), env.recording_metadata(), metadata)
    for key, value in state.items():
        if key != 'sim_time' and not np.allclose(restored[key], value, atol=1e-5, rtol=0):
            raise RuntimeError(f'offline state readback differs from recorded {key}')
    return {key: value for key, value in observation['obs'].items() if key.endswith('_image')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='.state.h5 file or directory (searched recursively)')
    parser.add_argument('--output', required=True, type=Path, help='directory for rendered .hdf5 training episodes')
    parser.add_argument('--video_only', action='store_true',
                        help='create missing main/wrist MP4 previews from --output HDF5 files; no simulator needed')
    parser.add_argument('--jpeg_quality', type=int, default=95)
    parser.add_argument('--render_frames', type=int, default=8, help='stationary RTX passes per saved state')
    parser.add_argument('--device', default='cpu', help='PhysX device; RTX rendering still uses the GPU')
    args = parser.parse_args(argv)
    if args.video_only:
        episodes = sorted(args.output.rglob('*.hdf5'))
        if not episodes:
            parser.error('no rendered .hdf5 episodes found under --output')
        for episode in episodes:
            export_episode_videos(episode)
        return 0
    if args.input is None:
        parser.error('--input is required unless --video_only is set')
    if not 1 <= args.jpeg_quality <= 100 or args.render_frames < 1:
        parser.error('--jpeg_quality must be 1..100 and --render_frames must be positive')
    sources = sorted(args.input.rglob('*.state.h5')) if args.input.is_dir() else [args.input]
    if not sources:
        parser.error('no .state.h5 recordings found')
    metadata, destinations = None, set()
    for source in sources:
        current, _ = inspect_episode(source)
        if metadata is not None and metadata != current:
            parser.error('recordings use different scenes/rigs; render each collection separately')
        metadata = current
        for destination in output_paths(source, args.output):
            if destination.exists() or destination in destinations:
                raise FileExistsError(destination)
            destinations.add(destination)
    trial = deepcopy(metadata['trial'])
    for key in ('scene_usd', 'overlay'):
        trial[key] = _resolve(trial[key])
        if not trial[key].is_file():
            raise FileNotFoundError(trial[key])
    args.robot_usd = _resolve(metadata['robot_usd'])
    if not args.robot_usd.is_file():
        raise FileNotFoundError(args.robot_usd)
    with h5py.File(sources[0], 'r') as source:
        args.replay_state = {name: value[0] for name, value in _datasets(source['replay_state']).items()}
        args.replay_state['sim_time'] = float(source['sim_time'][0])
    args.replay_metadata = metadata
    args.capture_mode, args.settle_steps = 'rgb', 0
    args.headless, args.enable_cameras = True, True
    # Match simulator startup: Pinocchio's Assimp must precede Kit's build.
    import pinocchio  # noqa: F401
    from isaaclab.app import AppLauncher

    with ExitStack() as cleanup:
        app = AppLauncher(args).app
        cleanup.callback(app.close)
        from isaaclab.sim import SimulationContext
        from egox.robots.g1_sim_dex3.environment import G1Dex3Env
        from tasks import make_task
        import omni.usd

        cleanup.callback(SimulationContext.clear_instance)
        env = G1Dex3Env(args, task_factory=make_task, trial=trial)
        cleanup.callback(env.close)

        def render_row(row, state):
            images = render_state(env, state, metadata, args.render_frames,
                                  omni.usd.get_context().reset_renderer_accumulation)
            if row % 100 == 0:
                print(f'Rendered row {row}: recorded simulation time {state["sim_time"]:.3f}s', flush=True)
            return images

        for source in sources:
            for destination in build_dataset(source, args.output, render_row, jpeg_quality=args.jpeg_quality):
                print(f'Saved {destination}', flush=True)
                try:
                    export_episode_videos(destination)
                except Exception as exc:
                    raise RuntimeError(f'{destination} is complete, but video export failed; '
                                       'retry with --video_only to keep the rendered HDF5') from exc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
