"""Collect controller observations and commands in the training HDF5 format."""
import ctypes
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

import cv2
import h5py
import numpy as np

from egox.configs.loader import load_robot_config
from egox.data.mapping import KEY_MASK_PATH_MAP, KEY_PATH_MAP
from .base import STALE


def _publish(source, destination):
    """Atomically rename without overwriting on Linux, including exFAT volumes."""
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. No destination exists until rename succeeds.
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(destination))


class EpisodeWriter:
    """Stream pre-action observations and their commands; publish only on save().

    Images are RGB uint8 HWC; numeric fields follow the robot config. Masks are
    constant throughout an episode because the training reader uses static masks.
    Each append holds only one encoded row in memory. Extra controller/task fields
    are ignored. close() discards an episode that has not been explicitly saved.
    """

    def __init__(self, output_dir, embodiment, jpeg_quality=95):
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be an integer within [1, 100]')
        config = load_robot_config(embodiment)
        self.images = config['tokenizer']['Image']
        self.observations = {**self.images, **config['tokenizer']['proprioceptions']}
        self.actions = config['action']
        self.fields = {**self.observations, **self.actions}
        if not self.actions:
            raise ValueError('Collection requires at least one configured action')
        for key in self.fields:
            if key not in KEY_PATH_MAP or key not in KEY_MASK_PATH_MAP:
                raise ValueError(f'{key}: missing storage mapping')
        for key, field in self.images.items():
            if field.get('layout', 'HWC') != 'HWC' or field.get('color_space', 'RGB') != 'RGB':
                raise ValueError(f'{key}: collection requires RGB HWC images')
        self.output_dir, self.embodiment = Path(output_dir), embodiment
        self.jpeg_quality = jpeg_quality
        self._closed = False
        self._reset()

    def _reset(self):
        self._file = self._partial = None
        self._length, self._failed = 0, False
        self._shapes, self._masks = {}, {}

    def __len__(self):
        return self._length

    def _check(self):
        if self._closed:
            raise RuntimeError('Episode writer is closed')
        if self._failed:
            raise RuntimeError('Episode write failed; discard it before collecting again')

    @staticmethod
    def _mask(value, shape, key):
        mask = np.asarray(value)
        if mask.dtype != np.bool_:
            raise ValueError(f'{key}: mask must be boolean')
        if mask.ndim == 0:
            return np.full(shape, mask.item(), dtype=bool)
        if mask.shape != shape:
            raise ValueError(f'{key}: mask must have static shape {shape}, got {mask.shape}')
        return mask.copy()

    def _row(self, packet, action, action_masks):
        """Validate every field before opening or changing the episode file."""
        observation, obs_masks = packet['obs'], packet.get('obs_mask', {})
        values, shapes, masks = {}, {}, {}
        for key, field in self.fields.items():
            source = observation if key in self.observations else action
            if key not in source:
                raise ValueError(f'{key}: missing configured collection field')
            value = np.asarray(source[key])
            if key in self.images:
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or 0 in value.shape:
                    raise ValueError(f'{key}: expected a nonempty RGB uint8 HWC image')
                if 'shape' in field and value.shape != tuple(field['shape']):
                    raise ValueError(f'{key}: expected configured image shape {tuple(field["shape"])}')
                mask_shape = (1,)
            else:
                if value.shape != tuple(field['shape']) or value.dtype.kind not in 'iuf':
                    raise ValueError(f'{key}: expected numeric shape {tuple(field["shape"])}')
                value = value.astype(np.float32)
                if not np.isfinite(value).all():
                    raise ValueError(f'{key}: all values, including masked padding, must be finite')
                mask_shape = value.shape
            shapes[key] = value.shape
            source_masks = obs_masks if key in self.observations else action_masks
            masks[key] = self._mask(source_masks.get(key, True), mask_shape, key)
            if self._length and (shapes[key] != self._shapes[key]
                                 or not np.array_equal(masks[key], self._masks[key])):
                raise ValueError(f'{key}: shape or mask changed; start a new episode')
            values[key] = value
        # Encode only after the entire uncompressed row has passed validation.
        for key in self.images:
            success, encoded = cv2.imencode('.jpg', cv2.cvtColor(values[key], cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not success:
                raise ValueError(f'{key}: JPEG encoding failed')
            values[key] = encoded.reshape(-1)
        return values, shapes, masks

    def _open(self, shapes, masks):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='episode_', suffix='.partial', dir=self.output_dir)
        os.close(fd)
        self._partial = Path(name)
        self._file = h5py.File(self._partial, 'w')
        self._file.attrs.update(embodiment=self.embodiment, compress=True)
        for key in self.fields:
            shape = () if key in self.images else shapes[key]
            dtype = h5py.vlen_dtype(np.uint8) if key in self.images else np.float32
            self._file.create_dataset(KEY_PATH_MAP[key], shape=(0, *shape),
                                      maxshape=(None, *shape), chunks=(1, *shape), dtype=dtype)
            if not masks[key].all():
                self._file.create_dataset(KEY_MASK_PATH_MAP[key], data=masks[key])
        self._shapes, self._masks = shapes, masks

    def append(self, observation_packet, action, action_masks=None):
        """Append a pre-action observation after its corresponding command succeeds."""
        self._check()
        values, shapes, masks = self._row(observation_packet, action, action_masks or {})
        try:
            if self._file is None:
                self._open(shapes, masks)
            for key, value in values.items():
                dataset = self._file[KEY_PATH_MAP[key]]
                dataset.resize(self._length + 1, axis=0)
                dataset[self._length] = value
            self._length += 1
        except BaseException:
            self._failed = True
            raise

    suffix = '.hdf5'

    def save(self, destination=None):
        """Publish a complete .hdf5 without replacing another file; begin a new episode."""
        self._check()
        if not self._length:
            return None
        destination = (Path(destination) if destination is not None else
                       self.output_dir / f'episode_{datetime.now():%Y%m%d_%H%M%S}{self.suffix}')
        try:
            self._file.flush()
            self._file.close()
            self._file = None
            _publish(self._partial, destination)
        except BaseException:
            self._failed = True
            raise
        self._reset()
        return destination

    def discard(self):
        """Remove only this writer's unfinished episode, including a failed write."""
        try:
            if self._file is not None:
                self._file.close()
        finally:
            if self._partial is not None:
                self._partial.unlink(missing_ok=True)
            self._reset()

    def close(self):
        self.discard()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()


class StateEpisodeWriter(EpisodeWriter):
    """Record numeric training fields and exact pre-action scene state for offline RGB.

    The .state.h5 suffix deliberately excludes these files from training discovery.
    Preview pixels never enter the recording. Only the offline builder publishes
    a training .hdf5 with the original image paths and masks.
    """

    suffix = '.state.h5'

    def __init__(self, output_dir, embodiment, collection):
        super().__init__(output_dir, embodiment)
        self.collection = json.dumps(collection, allow_nan=False)
        self.observations = {key: field for key, field in self.observations.items() if key not in self.images}
        self.images = {}
        self.fields = {**self.observations, **self.actions}

    def append(self, observation_packet, action, action_masks=None):
        self._check()
        values, shapes, masks = self._row(observation_packet, action, action_masks or {})
        replay = observation_packet.get('replay_state')
        if not isinstance(replay, dict) or not replay:
            raise ValueError('State collection requires a nonempty replay_state')
        extra = {}
        for key, value in replay.items():
            if not isinstance(key, str) or not key or any(part in ('', '.', '..') for part in key.split('/')):
                raise ValueError(f'Invalid replay state path: {key!r}')
            value = np.asarray(value)
            if value.dtype.kind not in 'biuf' or 0 in value.shape or not np.isfinite(value).all():
                raise ValueError(f'{key}: replay state must be finite numeric values')
            extra[f'replay_state/{key}'] = value
        extra['sim_time'] = np.asarray(observation_packet['sim_time'], dtype=np.float64)
        for key in ('frame_id', 'state_time_ns', 'segment_id'):
            value = observation_packet.get(key, 0 if key == 'segment_id' else -1)
            if type(value) is not int:
                raise ValueError(f'{key}: expected an integer')
            extra[key] = np.asarray(value, dtype=np.int64)
        if extra['sim_time'].shape or not np.isfinite(extra['sim_time']) or extra['sim_time'] < 0:
            raise ValueError('sim_time must be a finite nonnegative scalar')
        if self._length:
            previous = {name for name in self._extra_shapes}
            if set(extra) != previous or any(value.shape != self._extra_shapes[name] for name, value in extra.items()):
                raise ValueError('Replay state shape changed; start a new episode')
            if extra['sim_time'] <= self._file['sim_time'][-1]:
                raise ValueError('Recorded simulation times must increase')
        try:
            if self._file is None:
                self._open(shapes, masks)
                self._file.attrs.update(capture_mode='state_only', collection=self.collection, compress=False)
                self._extra_shapes = {name: value.shape for name, value in extra.items()}
                for name, value in extra.items():
                    self._file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                              chunks=(1, *value.shape), dtype=value.dtype,
                                              compression='lzf' if name.startswith('replay_state/') else None)
            for name, value in {**{KEY_PATH_MAP[key]: value for key, value in values.items()}, **extra}.items():
                dataset = self._file[name]
                dataset.resize(self._length + 1, axis=0)
                dataset[self._length] = value
            self._length += 1
        except BaseException:
            self._failed = True
            raise


class AsyncEpisodeWriter:
    """Bounded, ordered disk worker; save/discard wait for all accepted rows."""

    def __init__(self, writer, max_pending=8):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        self.writer = writer
        self._queue = queue.Queue(maxsize=max_pending)
        self._error, self._closed, self._length = None, False, 0
        self._closing, self._close_error = threading.Event(), None
        self._thread = threading.Thread(target=self._run, name='egox-recording', daemon=True)
        self._thread.start()

    def __len__(self):
        return self._length

    def _check(self, allow_error=False):
        if self._closed:
            raise RuntimeError('Episode writer is closed')
        if self._error is not None and not allow_error:
            raise RuntimeError(f'Episode writer failed: {self._error}; discard before collecting again') from self._error

    def append(self, observation_packet, action, action_masks=None):
        self._check()
        packet = {key: value for key, value in observation_packet.items()
                  if key not in ('preview', 'preview_image', 'preview_jpeg')}
        args = deepcopy((packet, action, action_masks))
        self._check()
        try:
            self._queue.put_nowait(('append', args, None, None))
        except queue.Full as error:
            raise RuntimeError('Episode writer queue is full; stop collection before sending another command') from error
        self._length += 1

    def _barrier(self, operation, *args):
        self._check(allow_error=operation == 'discard')
        ready, outcome = threading.Event(), {}
        self._queue.put((operation, args, ready, outcome))
        ready.wait()
        if 'error' in outcome:
            raise RuntimeError(f'Episode writer failed: {outcome["error"]}') from outcome['error']
        return outcome.get('result')

    def save(self, destination=None):
        result = self._barrier('save', *(() if destination is None else (destination,)))
        self._length = 0
        return result

    def discard(self):
        self._barrier('discard')
        self._length = 0

    def close(self, timeout=5.):
        """Drain accepted rows and discard unsaved data without hanging shutdown."""
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Shutdown timeout must be finite and positive')
        if not self._closed:
            self._closed = True
            self._closing.set()
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError('Episode writer did not stop; disk I/O is still pending in the background')
            self._length = 0
            if self._close_error is not None:
                raise RuntimeError(f'Episode writer failed to close: {self._close_error}') from self._close_error

    def _run(self):
        while True:
            try:
                operation, args, ready, outcome = self._queue.get(timeout=.1)
            except queue.Empty:
                if not self._closing.is_set():
                    continue
                try:
                    self.writer.close()
                except BaseException as error:
                    self._close_error = error
                return
            try:
                if operation != 'discard' and self._error is not None:
                    raise self._error
                result = getattr(self.writer, operation)(*args)
                if operation == 'discard':
                    self._error = None
                if outcome is not None:
                    outcome['result'] = result
            except BaseException as error:
                self._error = error
                if outcome is not None:
                    outcome['error'] = error
            finally:
                self._queue.task_done()
                if ready is not None:
                    ready.set()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()


class PreviewWorker:
    """Fetch compressed preview on a separate DDS client, never blocking tracking."""

    def __init__(self, client_factory, streamer, hz=24):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError('Preview frequency must be positive and finite')
        self.client_factory, self.streamer, self.period = client_factory, streamer, 1 / hz
        self._stop, self._error = threading.Event(), None
        self._thread = threading.Thread(target=self._run, name='egox-preview', daemon=True)

    def _run(self):
        last_warning = -float('inf')
        while not self._stop.is_set():
            try:
                with self.client_factory() as client:
                    while not self._stop.is_set():
                        started = time.monotonic()
                        result = client.preview()
                        if result.get('jpeg') is not None:
                            image = cv2.imdecode(np.asarray(result['jpeg'], dtype=np.uint8), cv2.IMREAD_COLOR)
                            if image is None:
                                raise ValueError('Simulator returned an invalid preview JPEG')
                            self.streamer.push(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        self._stop.wait(max(0., self.period - (time.monotonic() - started)))
            except TimeoutError:
                # Reset can temporarily occupy the simulation thread. This read
                # is safe to retry on a fresh client; commands are never retried.
                now = time.monotonic()
                if not self._stop.is_set() and now - last_warning >= 5:
                    logging.warning('Simulator preview timed out (possibly resetting); reconnecting in the background')
                    last_warning = now
                self._stop.wait(.1)
            except BaseException as error:
                self._error = error
                return

    def check(self):
        if self._error is not None:
            raise RuntimeError(f'Simulator preview failed: {self._error}') from self._error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.)
        if self._thread.is_alive():
            raise RuntimeError('Preview client did not stop; use a bounded DDS request timeout')

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.close()


class DataCollector:
    """Teleoperate through the same DDS client as rollout, recording accepted commands.

    The device supplies TeleopFrame; the embodiment mapper supplies calibrate/reset,
    action(frame) and action_masks. Realtime collection records the server's atomic
    pre-action state; fixed-step controllers use the preceding observation.
    Inputs and writer are owned by the caller. Lost tracking freezes motion;
    optional brief-gap recovery requires fresh, continuous targets from the mapper.
    """

    def __init__(self, client, teleoperator, mapper, writer, *, preview=None, max_age_s=.5, state_only=False,
                 tracking_recovery_s=0.):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        if not np.isfinite(tracking_recovery_s) or tracking_recovery_s < 0:
            raise ValueError('tracking_recovery_s must be finite and nonnegative')
        if tracking_recovery_s and not callable(getattr(mapper, 'tracking_recovery_error', None)):
            raise ValueError('tracking recovery requires a mapper with continuity checks')
        self.client, self.teleoperator, self.mapper, self.writer = client, teleoperator, mapper, writer
        self.preview, self.max_age_ns = preview, int(max_age_s * 1e9)
        self.state, self.observation, self.last_source_ns = 'idle', None, None
        self.state_only, self.action_hz = state_only, 30
        self.segment_id, self.last_sim_time = 0, None
        self.tracking_recovery_ns = int(tracking_recovery_s * 1e9)
        self.last_frame = self.last_action = None
        self._clear_tracking_wait()

    def _clear_tracking_wait(self):
        self.recovery_first_ns = self.recovery_last_ns = None
        self.recovery_samples = 0

    def _state(self, state, message):
        if self.state_only and self.state == 'active' and state != 'active':
            self.client.pause()
            self.segment_id += 1
            self.last_sim_time = None
        self.state = state
        if state != 'tracking_wait':
            self._clear_tracking_wait()
        print(message, flush=True)

    def _start_tracking_wait(self, now, reason):
        if not self.tracking_recovery_ns or self.last_frame is None:
            self._state('paused', f'Tracking unavailable or stale: {reason}. Recover tracking, then A: resume.')
        elif now - self.last_frame.source_time_ns >= self.tracking_recovery_ns:
            self._state('paused', 'Tracking recovery window expired. Recover tracking, then A: resume.')
        else:
            self._clear_tracking_wait()
            self._state('tracking_wait', f'Tracking wait: {reason}. Control frozen; B: cancel automatic recovery.')

    def _try_tracking_recovery(self, frame, action, now):
        if now - self.last_frame.source_time_ns >= self.tracking_recovery_ns:
            reason = 'tracking recovery window expired'
        elif frame.source != 'pico':
            reason = 'tracking source changed'
        else:
            reason = self.mapper.tracking_recovery_error(self.last_frame, self.last_action, frame, action)
        if reason:
            self._state('paused', f'{reason}. Recover tracking, then A: resume.')
            return False
        if self.recovery_last_ns is None or frame.source_time_ns > self.recovery_last_ns:
            if self.recovery_first_ns is None:
                self.recovery_first_ns = frame.source_time_ns
            self.recovery_last_ns = frame.source_time_ns
            self.recovery_samples += 1
        if self.recovery_samples < 3 or self.recovery_last_ns - self.recovery_first_ns < 100_000_000:
            return False
        elapsed_ms = (now - self.last_frame.source_time_ns) / 1e6
        self._state('active', f'Tracking recovered after {elapsed_ms:.0f} ms; recording resumed in segment {self.segment_id}.')
        return True

    def _preview(self):
        if not self.state_only and self.preview is not None and self.observation is not None:
            self.preview.push(self.observation['obs']['main_image'])

    def step(self, frame, now_ns=None):
        """Handle one input sample; stopping buttons always take priority over motion."""
        events = frame.events
        if 'discard' in events or 'save' in events:
            if self.state_only and self.state == 'active':
                self.client.pause()  # Stop physics before waiting for the disk worker.
                self.state = 'paused'
            count = len(self.writer)
            if 'discard' in events:
                self.writer.discard()
                message = f'Discarded {count} rows. X: reset for the next episode.'
            else:
                path = self.writer.save()
                message = f'Saved {count} rows to {path}.' if path else 'No rows to save.'
            if self.observation is not None:
                self.client.release()
            self.observation, self.last_source_ns = None, None
            self.segment_id, self.last_sim_time = 0, None
            self.last_frame = self.last_action = None
            self.mapper.reset()
            self._state('idle', message)
            return
        if 'reset' in events:
            if len(self.writer):
                self._state('finished' if self.state == 'finished' else 'paused',
                            'Save (Y) or discard (left menu) before resetting the episode.')
                return
            self.observation = self.client.reset()
            self.mapper.reset()
            self.last_source_ns = None
            self.segment_id, self.last_sim_time = 0, None
            self.last_frame = self.last_action = None
            self._state('calibrate', 'Reset. Stand comfortably, then A: calibrate head heading and height.')
            self._preview()
            return
        if 'pause' in events:
            if self.state in ('active', 'tracking_wait'):
                self._state('paused', 'Paused. A: resume; Y: save; left menu: discard.')
            return

        if self.state_only and self.preview is not None:
            try:
                self.preview.check()
            except BaseException:
                self._state('finished', 'Preview failed. Collection stopped.')
                raise

        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        fresh = frame.valid and 0 <= now_ns - frame.source_time_ns <= self.max_age_ns
        if not fresh:
            reason = frame.reason or f'tracking sample age {(now_ns - frame.source_time_ns) / 1e6:.0f} ms'
            if self.state == 'active':
                if frame.source == 'pico' and frame.tracking_status == STALE:
                    self._start_tracking_wait(now_ns, reason)
                else:
                    self._state('paused', f'Tracking unavailable or stale: {reason}. Recover tracking, then A: resume.')
            elif self.state == 'tracking_wait':
                if frame.source != 'pico' or frame.tracking_status != STALE:
                    self._state('paused', f'Invalid tracking during recovery: {reason}. Recover tracking, then A: resume.')
                elif now_ns - self.last_frame.source_time_ns >= self.tracking_recovery_ns:
                    self._state('paused', 'Tracking recovery window expired. Recover tracking, then A: resume.')
                else:
                    self._clear_tracking_wait()
            return
        if (self.state == 'active' and frame.source == 'pico' and self.last_frame is not None
                and now_ns - self.last_frame.source_time_ns > self.max_age_ns):
            # A fresh packet may arrive after the watchdog cutoff between polls.
            self._start_tracking_wait(now_ns, 'gap since the last accepted tracking sample exceeded the freshness limit')
        if 'engage' in events:
            if self.state == 'calibrate':
                self.mapper.calibrate(frame)
                self._state('ready', 'Calibrated. A: engage and start recording.')
                return
            if self.state in ('ready', 'paused', 'tracking_wait'):
                self._state('active', 'Recording. B: pause; Y: save; left menu: discard.')
            elif self.state == 'idle':
                print('X: reset before calibrating.', flush=True)
        if self.state not in ('active', 'tracking_wait'):
            return
        try:
            action = self.mapper.action(frame)
        except ValueError as exc:
            self._state('paused', f'Invalid target: {exc}. Recover tracking, then A: resume.')
            return
        if self.state == 'tracking_wait' and not self._try_tracking_recovery(frame, action, now_ns):
            return
        if self.last_source_ns is not None and frame.source_time_ns <= self.last_source_ns:
            return  # Never execute/record the same tracking packet twice.
        # Realtime physics can move between requests: only the server can capture
        # the atomic pre-action observation. Fixed-step replies remain post-action.
        before = self.observation
        try:
            after = self.client.send(action, **({'source_time_ns': frame.source_time_ns} if self.state_only else {}))
            if self.state_only and after.get('paused'):
                self._state('paused', f'{after["paused"]}. Recover tracking, then A: resume.')
                return
            self.observation, self.last_source_ns = after, frame.source_time_ns
            if self.state_only:
                before = after
                if after.get('recordable', True):
                    sim_time = after['sim_time']
                    if self.last_sim_time is not None and sim_time - self.last_sim_time > 1.5 / self.action_hz + 1e-9:
                        self.segment_id += 1
                    before = {**before, 'segment_id': self.segment_id}
                    self.writer.append(before, action, self.mapper.action_masks)
                    self.last_sim_time = sim_time
            else:
                self.writer.append(before, action, self.mapper.action_masks)
            self.last_frame, self.last_action = deepcopy(frame), deepcopy(action)
        except BaseException:
            try:
                if self.state_only:
                    self.client.pause()
            except Exception:
                pass  # Preserve the original command/write error if the connection also failed.
            self.state = 'finished'  # An uncertain command/write must never be retried.
            raise
        self._preview()
        if after.get('fell', False):
            self._state('finished', 'Robot fell. Save or discard, then X: reset.')

    def run(self, action_hz=30):
        """Poll tracking at the collection rate; realtime physics advances independently."""
        if not np.isfinite(action_hz) or action_hz <= 0:
            raise ValueError('action_hz must be finite and positive')
        print('X: reset | A: calibrate, then engage/resume | B: pause | Y: save | left menu: discard', flush=True)
        self.action_hz = action_hz
        if self.preview is not None and not self.state_only:
            self.preview.push(self.client.observe()['obs']['main_image'])
        try:
            while True:
                started = time.monotonic()
                self.step(self.teleoperator.poll())
                time.sleep(max(0., 1 / action_hz - (time.monotonic() - started)))
        finally:
            if self.state_only and self.state == 'active':
                timeout = getattr(self.client, 'timeout_ms', None)
                try:
                    if timeout is not None:
                        self.client.timeout_ms = min(timeout, 250)
                    self.client.pause()  # Freeze before preview/disk workers are drained by the caller.
                except Exception:
                    pass  # A disconnected controller retains its own command watchdog.
                finally:
                    if timeout is not None:
                        self.client.timeout_ms = timeout
                    self.state = 'paused'
