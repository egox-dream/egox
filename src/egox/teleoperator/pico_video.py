"""Optional asynchronous XRoboToolkit Remote Vision video.

Connect as a TCP client to PICO port 12345 and send [uint32 big-endian AU
length][Annex-B H.264 access unit]. The companion control listener on PC
port 13579 accepts the headset's Remote Vision OPEN_CAMERA/CLOSE_CAMERA requests.
The head camera is letterboxed to 2560x720 for the receiver's ZEDMINI profile.
Mono input is duplicated; stereo input preserves the saved eye ordering.
This display conversion never changes recorded RGB images.

PyAV with libx264 is optional and checked by start(); GStreamer is not required.
Neither construction nor import opens a network connection. Only start() does.
"""
from __future__ import annotations

from fractions import Fraction
import logging
import ipaddress
import queue
import socket
import struct
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# Verified against the official XR-Robotics/XRoboToolkit-Unity-Client sources:
# Assets/Scripts/Network/{OperatorControlClient,
# NetworkDataProtocolSerializer,CameraRequestSerializer}.cs. The outer TCP
# length is big-endian; command/payload lengths and camera integers are LE.
MAX_CONTROL_BODY = 65536
CONTROL_IDLE_TIMEOUT_S = 5.


def control_packet(command, payload=b""):
    command, payload = command.encode("utf-8"), bytes(payload)
    if not 1 <= len(command) <= 128 or len(payload) > MAX_CONTROL_BODY - 136:
        raise ValueError("invalid control command or payload size")
    body = struct.pack("<I", len(command)) + command + struct.pack("<I", len(payload)) + payload
    return struct.pack(">I", len(body)) + body


def _decode_control_body(body):
    if not 9 <= len(body) <= MAX_CONTROL_BODY:
        raise ValueError("invalid Remote Vision control body size")
    command_size = struct.unpack_from("<I", body)[0]
    if not 1 <= command_size <= 128 or len(body) < command_size + 8:
        raise ValueError("invalid Remote Vision command size")
    size = struct.unpack_from("<I", body, 4 + command_size)[0]
    if size != len(body) - command_size - 8:
        raise ValueError("invalid Remote Vision payload size")
    return body[4:4 + command_size].decode("utf-8"), body[8 + command_size:]


def _decode_camera_request(payload):
    if len(payload) < 33 or payload[:3] != b"\xca\xfe\x01":
        raise ValueError("invalid Remote Vision camera request header")
    values = struct.unpack_from("<7i", payload, 3)
    offset, strings = 31, []
    for _ in range(2):
        if offset >= len(payload):
            raise ValueError("truncated Remote Vision camera request")
        size = payload[offset]
        offset += 1
        if offset + size > len(payload):
            raise ValueError("truncated Remote Vision camera string")
        strings.append(payload[offset:offset + size].decode("utf-8"))
        offset += size
    if offset != len(payload):
        raise ValueError("trailing Remote Vision camera request bytes")
    return (*values, *strings)


def frame_packet(payload):
    payload = bytes(payload)
    if not payload or len(payload) > 32 * 1024 * 1024:
        raise ValueError("encoded access unit must contain1..33554432 bytes")
    return struct.pack(">I", len(payload)) + payload


def stereo_display_frame(rgb, width=2560, height=720, *, stereo_input=False):
    """Letterbox each saved eye; duplicate a mono source for legacy profiles."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) <= 0:
        raise ValueError("display source must be RGB uint8 HxWx3")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("H264 output dimensions must be positive and even")
    if stereo_input:
        if rgb.shape[1] % 2:
            raise ValueError("stereo input must have two equally sized halves")
        views = np.split(rgb, 2, axis=1)
    else:
        views = [rgb]
    eye_width = width // 2
    eyes = []
    for view in views:
        scale = min(eye_width / view.shape[1], height / view.shape[0])
        w, h = max(1, round(view.shape[1] * scale)), max(1, round(view.shape[0] * scale))
        # Display-only nearest-neighbour scaling keeps the optional package minimal.
        xs = np.minimum((np.arange(w) / scale).astype(int), view.shape[1] - 1)
        ys = np.minimum((np.arange(h) / scale).astype(int), view.shape[0] - 1)
        eye = np.zeros((height, eye_width, 3), dtype=np.uint8)
        x, y = (eye_width - w) // 2, (height - h) // 2
        eye[y:y+h, x:x+w] = view[ys[:, None], xs[None, :]]
        eyes.append(eye)
    return np.concatenate(eyes if stereo_input else eyes * 2, axis=1)


class _PyAVEncoder:
    def __init__(self, width, height, fps, bitrate):
        import av
        self.av = av
        self.codec = av.CodecContext.create("libx264", "w")
        self.codec.width, self.codec.height = width, height
        self.codec.pix_fmt = "yuv420p"
        self.codec.framerate, self.codec.time_base = Fraction(fps, 1), Fraction(1, fps)
        self.codec.bit_rate, self.codec.gop_size = bitrate, fps
        self.codec.options = {"preset": "ultrafast", "tune": "zerolatency",
            "x264-params": f"repeat-headers=1:annexb=1:keyint={fps}:min-keyint={fps}:scenecut=0"}
        self.codec.open()

    def encode(self, rgb, pts):
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        frame.pts = pts
        return [bytes(packet) for packet in self.codec.encode(frame) if packet.size]

    def close(self):
        # No draining across reconnect: the next encoder begins with fresh SPS/PPS.
        self.codec = None


class PicoVideoStreamer:
    """Latest-frame video plus an independent bounded Remote Vision control server.

    The IP entered in the headset's Listen dialog is this PC's address. ``host``
    is the headset address. Only that configured headset may use the control
    listener; incoming camera requests can never redirect the video destination.
    ``control_port=None`` opts out for legacy direct-stream receiver clients.
    """
    def __init__(self, host, port=12345, width=2560, height=720, fps=30, bitrate=4_000_000,
                 reconnect_s=1., encoder_factory=None, connector=None,
                 control_host="0.0.0.0", control_port=13579, stereo_input=False):
        if not host:
            raise ValueError("an explicit PICO host is required to enable video")
        if not 1 <= port <= 65535 or fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("invalid video port, rate, or output dimensions")
        if control_port is not None and not 0 <= control_port <= 65535:
            raise ValueError("invalid Remote Vision control port")
        self.host, self.port, self.width, self.height = host, int(port), width, height
        self.stereo_input = bool(stereo_input)
        self.control_host, self.control_port = control_host, control_port
        self.fps, self.bitrate, self.reconnect_s = int(fps), int(bitrate), float(reconnect_s)
        self.encoder_factory = encoder_factory or _PyAVEncoder
        self.connector = connector or socket.create_connection
        self.frames = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.thread = None
        self.control_thread = None
        self.socket = None
        self.control_socket = None
        self.listener = None
        self.allowed_peers = set()
        self.video_enabled = threading.Event()
        if control_port is None:
            self.video_enabled.set()  # Direct receivers have no OPEN_CAMERA handshake.
        self.video_generation = 0
        self.latest_frame = None
        self.error = ""
        self.dropped_frames = 0
        self.sent_frames = 0

    def start(self):
        if self.thread is not None:
            return True
        try:
            encoder = self.encoder_factory(self.width, self.height, self.fps, self.bitrate)
            encoder.close()
        except Exception as exc:
            self.error = (f"PICO video unavailable: PyAV with libx264 is required ({exc}). "
                          "Install the optional 'av' package in this environment, or omit --pico_host. "
                          "GStreamer is not used by this sender.")
            logger.error(self.error)
            return False
        self.stopping.clear()
        if self.control_port is not None:
            try:
                self.allowed_peers = {entry[4][0] for entry in socket.getaddrinfo(
                    self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)}
                self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.listener.bind((self.control_host, self.control_port))
                self.listener.listen(2)
                self.listener.settimeout(.2)
                self.control_port = self.listener.getsockname()[1]
            except OSError as exc:
                if self.listener is not None:
                    self.listener.close()
                    self.listener = None
                self.error = f"PICO Remote Vision control listener unavailable: {exc}"
                logger.error(self.error)
                return False
            self.control_thread = threading.Thread(target=self._control_worker, daemon=True,
                name="egox-pico-control")
            self.control_thread.start()
            logger.info("PICO Remote Vision control listening on %s:%s; in Listen, enter this PC's LAN IP and Confirm",
                self.control_host, self.control_port)
        self.thread = threading.Thread(target=self._worker, daemon=True, name="egox-pico-video")
        self.thread.start()
        return True

    def push(self, rgb):
        """Copy and queue without waiting for the encoder or a socket."""
        if self.stopping.is_set():
            return
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("PICO video requires RGB uint8 images")
        try:
            self.frames.put_nowait(rgb.copy())
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(rgb.copy())
            except queue.Full:
                self.dropped_frames += 1

    @staticmethod
    def _close_socket(sock):
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _handle_control(self, command, payload, sock):
        if command == "PING":
            sock.sendall(control_packet("PONG", payload))
        elif command == "OPEN_CAMERA":
            width, height, fps, bitrate, hevc, mode, port, camera, host = _decode_camera_request(payload)
            if (width, height) != (self.width, self.height) or camera != "ZED" or hevc != 0:
                raise ValueError(f"select ZEDMINI ({self.width}x{self.height} H.264) in Remote Vision")
            if not 0 < fps <= 240 or bitrate <= 0 or mode not in (0, 1, 2):
                raise ValueError("invalid Remote Vision camera parameters")
            # Never resolve or connect to a destination supplied by a request.
            if port != self.port or str(ipaddress.ip_address(host)) not in self.allowed_peers:
                raise ValueError("camera destination does not match the configured PICO host/port")
            # Refresh H.264 headers in place. Closing this socket can stop the
            # headset's video listener, leaving every subsequent reconnect refused.
            self.video_generation += 1
            self.video_enabled.set()
            logger.info("PICO Remote Vision OPEN_CAMERA accepted: %sx%s; video goes to %s:%s",
                width, height, self.host, self.port)
        elif command == "CLOSE_CAMERA":
            self.video_enabled.clear()
            self.video_generation += 1
            self._close_socket(self.socket)
            logger.info("PICO Remote Vision closed; waiting for Listen")
        elif command not in ("PONG", "AUDIO_SESSION"):
            logger.debug("Ignoring unsupported PICO control command %r", command)

    def _control_session(self, sock):
        buffer, partial_since, heartbeat = bytearray(), None, time.monotonic()
        received_at = heartbeat
        supports_heartbeat = False  # Older Pico apps send OPEN_CAMERA but never answer PING.
        while not self.stopping.is_set():
            now = time.monotonic()
            # Require replies only after the client has demonstrated PONG support.
            if supports_heartbeat and now - received_at > CONTROL_IDLE_TIMEOUT_S:
                raise ValueError("Remote Vision control heartbeat timed out; reconnect with Listen")
            # Newer clients disconnect after 5 seconds without a received byte.
            if now >= heartbeat:
                sock.sendall(control_packet("PING"))
                heartbeat = now + 1.
            if partial_since is not None and now - partial_since > 2.:
                raise ValueError("Remote Vision control frame was incomplete for 2 seconds")
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                return
            received_at = time.monotonic()
            if not buffer:
                partial_since = now
            buffer.extend(chunk)
            while len(buffer) >= 4:
                size = struct.unpack_from(">I", buffer)[0]
                if not 9 <= size <= MAX_CONTROL_BODY:
                    raise ValueError("invalid Remote Vision outer frame size")
                if len(buffer) < size + 4:
                    break
                body = bytes(buffer[4:size + 4])
                del buffer[:size + 4]
                command, payload = _decode_control_body(body)
                supports_heartbeat |= command == 'PONG'
                self._handle_control(command, payload, sock)
                partial_since = time.monotonic() if buffer else None

    def _control_worker(self):
        while not self.stopping.is_set():
            sock = None
            try:
                sock, peer = self.listener.accept()
                if peer[0] not in self.allowed_peers:
                    logger.warning("PICO Remote Vision rejected unconfigured peer %s", peer[0])
                    continue
                self.control_socket = sock
                sock.settimeout(.2)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if hasattr(socket, 'TCP_USER_TIMEOUT'):
                    # Detect lost TCP acknowledgements even for apps without PONG.
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, int(CONTROL_IDLE_TIMEOUT_S * 1000))
                logger.info("PICO Remote Vision control connected from %s", peer[0])
                self._control_session(sock)
            except socket.timeout:
                continue
            except (OSError, ValueError) as exc:
                if not self.stopping.is_set():
                    logger.warning("PICO Remote Vision control disconnected: %s", exc)
            finally:
                self._close_socket(sock)
                self.control_socket = None

    def _worker(self):
        while not self.stopping.is_set():
            if not self.video_enabled.is_set():
                self.stopping.wait(.1)
                continue
            # Isaac startup may take minutes. Do not open an empty video stream
            # while the headset is waiting for its first decodable frame.
            while True:
                try:
                    self.latest_frame = self.frames.get_nowait()
                except queue.Empty:
                    break
            if self.latest_frame is None:
                self.stopping.wait(.05)
                continue
            encoder, sock = None, None
            generation = self.video_generation
            try:
                sock = self.connector((self.host, self.port), timeout=1.)
                self.socket = sock
                sock.settimeout(.5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                encoder = self.encoder_factory(self.width, self.height, self.fps, self.bitrate)
                start = time.monotonic()
                period, deadline, previous_pts = 1. / self.fps, start, -1
                display_source, display_frame, first_frame = None, None, True
                logger.info("PICO video connected to %s:%s; waiting for camera frames", self.host, self.port)
                while not self.stopping.is_set() and self.video_enabled.is_set():
                    if self.stopping.wait(max(0., deadline - time.monotonic())):
                        break
                    if generation != self.video_generation:
                        encoder.close()
                        generation = self.video_generation
                        encoder = self.encoder_factory(self.width, self.height, self.fps, self.bitrate)
                        start = deadline = time.monotonic()
                        previous_pts, first_frame = -1, True
                    # Preserve the latest frame across idle periods and reconnects.
                    # Re-encoding it keeps the decoder alive and repeats keyframes.
                    while True:
                        try:
                            self.latest_frame = self.frames.get_nowait()
                        except queue.Empty:
                            break
                    rgb = self.latest_frame
                    if rgb is None:
                        self.stopping.wait(.05)
                        deadline = time.monotonic()
                        continue
                    if rgb is not display_source:
                        display_frame = stereo_display_frame(rgb, self.width, self.height, stereo_input=self.stereo_input)
                        display_source = rgb
                    pts = max(previous_pts + 1, int((time.monotonic() - start) * self.fps))
                    previous_pts = pts
                    for payload in encoder.encode(display_frame, pts):
                        sock.sendall(frame_packet(payload))
                        self.sent_frames += 1
                        self.error = ""
                        if first_frame:
                            logger.info("PICO video first frame sent (%sx%s H.264); headset display still needs visual confirmation",
                                self.width, self.height)
                            first_frame = False
                    # Encoding and sending consume part of the frame period.
                    # Preserve the phase through ordinary work/wake-up jitter;
                    # after an overrun, discard the backlog and encode only the
                    # latest frame instead of replaying missed preview frames.
                    deadline = max(deadline + period, time.monotonic())
            except Exception as exc:
                # Suppress errors from a concurrent camera close/reconfiguration.
                if not self.stopping.is_set() and generation == self.video_generation:
                    self.error = str(exc)
                    logger.warning("PICO video disconnected: %s", exc)
            finally:
                if encoder is not None:
                    encoder.close()
                if sock is not None:
                    sock.close()
                self.socket = None
            self.stopping.wait(self.reconnect_s)

    def stop(self):
        self.stopping.set()
        for sock in (self.socket, self.control_socket, self.listener):
            self._close_socket(sock)
        if self.thread is not None:
            self.thread.join(timeout=2.)
            self.thread = None
        if self.control_thread is not None:
            self.control_thread.join(timeout=2.)
            self.control_thread = None
        self.listener = None
