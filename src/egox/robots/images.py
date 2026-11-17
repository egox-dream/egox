"""Local ZeroMQ image snapshots, referenced by small DDS observation replies.

Snapshots own their bytes and expire after eight newer captures. Reading an expired
reference fails instead of pairing the latest images with an older robot state.
No camera or simulator API runs on the image server's thread.
"""
from collections import OrderedDict
import json
import math
from pathlib import Path
import tempfile
import threading
import time

from .channel import MAX_BYTES, decode, encode


class ImageServer:
    def __init__(self):
        import zmq  # Optional until images are actually transferred.

        self._zmq = zmq
        self._directory = tempfile.TemporaryDirectory(prefix='egox_images_')
        self.endpoint = 'ipc://' + str(Path(self._directory.name) / 'images.sock')
        self._frames, self._sequence = OrderedDict(), 0
        self._lock, self._stop, self._ready = threading.Lock(), threading.Event(), threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._serve, name='egox-images', daemon=True)
        try:
            self._thread.start()
            if not self._ready.wait(5.):
                raise TimeoutError('image server did not start')
            if self._error is not None:
                raise RuntimeError('image server failed to start') from self._error
        except BaseException:
            self.close()
            raise

    def store(self, images):
        """Capture immutable image bytes and return their exact snapshot reference."""
        metadata, payload = encode(images)
        with self._lock:
            if self._stop.is_set() or self._error is not None:
                raise RuntimeError('image server is closed or failed') from self._error
            self._sequence += 1
            self._frames[self._sequence] = (metadata.encode(), payload)
            while len(self._frames) > 8:
                self._frames.popitem(last=False)
            return {'endpoint': self.endpoint, 'sequence': self._sequence}

    def _serve(self):
        zmq = self._zmq
        try:
            # Both socket creation and destruction belong to this thread.
            with zmq.Context() as context, context.socket(zmq.REP) as socket:
                socket.linger, socket.sndtimeo, socket.maxmsgsize = 0, 100, 4096
                socket.bind(self.endpoint)
                self._ready.set()
                while not self._stop.is_set():
                    if not socket.poll(50):
                        continue
                    sequence = None
                    try:
                        request = socket.recv_multipart()
                        if len(request) != 1:
                            raise ValueError('invalid image request')
                        sequence = json.loads(request[0])
                        if type(sequence) is not int or sequence < 1:
                            raise ValueError('invalid image sequence')
                        with self._lock:
                            frame = self._frames.get(sequence)
                        if frame is None:
                            raise ValueError('image snapshot expired or does not exist')
                        parts = [json.dumps({'sequence': sequence}).encode(), *frame]
                    except (ValueError, TypeError, UnicodeError) as exc:
                        parts = [json.dumps({'sequence': sequence, 'error': str(exc)}).encode()]
                    # The buffers are immutable; ZeroMQ retains them until transmission ends.
                    socket.send_multipart(parts, copy=False)
        except Exception as exc:
            self._error = exc
        finally:
            self._ready.set()

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1.)
        if self._thread.is_alive():
            raise RuntimeError('image server did not stop')
        with self._lock:
            self._frames.clear()
        self._directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ImageClient:
    """Read referenced images; use each client on one thread and close it there."""

    def __init__(self, timeout_ms=60000):
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError('image timeout must be finite and positive')
        self.timeout_ms = math.ceil(timeout_ms)
        self._context = self._socket = self._endpoint = None
        self._closed = False

    def read(self, reference):
        import zmq

        if self._closed:
            raise RuntimeError('image client is closed')
        if not isinstance(reference, dict):
            raise ValueError('invalid image reference')
        endpoint, sequence = reference.get('endpoint'), reference.get('sequence')
        if not isinstance(endpoint, str) or not endpoint.startswith('ipc:///'):
            raise ValueError('image endpoint must be a local IPC socket')
        if type(sequence) is not int or sequence < 1:
            raise ValueError('invalid image sequence')
        if self._context is None:
            self._context = zmq.Context()
        if self._endpoint != endpoint:
            self._disconnect()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.linger = 0
            self._socket.immediate = 1
            self._socket.sndtimeo = self.timeout_ms
            self._socket.maxmsgsize = MAX_BYTES
            self._socket.connect(endpoint)
            self._endpoint = endpoint
        deadline = time.monotonic() + self.timeout_ms / 1000
        try:
            self._socket.send_json(sequence)
            remaining = max(0, math.ceil((deadline - time.monotonic()) * 1000))
            if not self._socket.poll(remaining):
                raise TimeoutError('image snapshot request timed out')
            parts = self._socket.recv_multipart()
            header = json.loads(parts[0])
            if not isinstance(header, dict) or header.get('sequence') != sequence:
                raise ValueError('image reply does not match the requested snapshot')
            if 'error' in header:
                raise RuntimeError(header['error'])
            if len(parts) != 3:
                raise ValueError('invalid image reply')
            images = decode(parts[1].decode(), parts[2])
            if not isinstance(images, dict):
                raise ValueError('invalid image snapshot')
            return images
        except zmq.Again as exc:
            self._disconnect()
            raise TimeoutError('image snapshot request timed out') from exc
        except BaseException:
            # A timed-out REQ cannot send again until its socket has been replaced.
            self._disconnect()
            raise

    def _disconnect(self):
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._endpoint = None

    def close(self):
        self._disconnect()
        if self._context is not None:
            self._context.term()
            self._context = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
