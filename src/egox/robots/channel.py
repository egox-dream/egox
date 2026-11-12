"""DDS requests and observations shared by controller and simulation processes.

Each robot instance has its own namespace. Requests carry a client UUID and
sequence number; replies are matched to both. A timed-out command is never
retried because it may already have executed. Arrays use raw bytes, not pickle.
CycloneDDS is imported only when opening a connection.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import re
import threading
import time
import uuid

import numpy as np

MAX_BYTES = 64 * 1024 * 1024
BLOCK_BYTES = 65536  # DDS byte arrays stay bytes; octet sequences become Python int lists.


def encode(value):
    """Return JSON metadata and one owned buffer, preserving array shapes/dtypes."""
    buffers, offset = [], 0

    def visit(item):
        nonlocal offset
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in 'biuf':
                raise ValueError('DDS arrays must have a boolean or numeric dtype')
            data = np.ascontiguousarray(item).tobytes()
            descriptor = {'__array__': {'offset': offset, 'shape': list(item.shape), 'dtype': item.dtype.str}}
            offset += len(data)
            if offset > MAX_BYTES:
                raise ValueError('DDS message exceeds size limit')
            buffers.append(data)
            return descriptor
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): visit(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(value) for value in item]
        return item

    metadata = json.dumps(visit(value), allow_nan=False, separators=(',', ':'))
    if len(metadata.encode()) + offset > MAX_BYTES:
        raise ValueError('DDS message exceeds size limit')
    return metadata, b''.join(buffers)


def decode(metadata, payload):
    if len(metadata.encode()) + len(payload) > MAX_BYTES:
        raise ValueError('DDS message exceeds size limit')

    def visit(item):
        if isinstance(item, dict) and '__array__' in item:
            spec = item['__array__']
            dtype, shape, offset = np.dtype(spec['dtype']), spec['shape'], spec['offset']
            if (dtype.kind not in 'biuf' or not isinstance(shape, list) or len(shape) > 8
                    or any(type(n) is not int or n < 0 for n in shape)
                    or type(offset) is not int or offset < 0):
                raise ValueError('invalid DDS array metadata')
            count = math.prod(shape)
            if offset + count * dtype.itemsize > len(payload):
                raise ValueError('DDS array exceeds its buffer')
            return np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        if isinstance(item, dict):
            return {key: visit(value) for key, value in item.items()}
        if isinstance(item, list):
            return [visit(value) for value in item]
        return item

    return visit(json.loads(metadata))


@lru_cache(maxsize=1)
def _packet_type():
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import array, sequence as idl_sequence, uint8

    @dataclass
    class RobotPacket(IdlStruct, typename='egox::RobotPacket'):
        client_id: str
        sequence: int
        metadata: str
        payload_size: int
        blocks: idl_sequence[array[uint8, BLOCK_BYTES]]

    return RobotPacket


def _packet(client_id, sequence, metadata, payload):
    blocks = [payload[start:start + BLOCK_BYTES].ljust(BLOCK_BYTES, b'\0')
              for start in range(0, len(payload), BLOCK_BYTES)]
    return _packet_type()(client_id, sequence, metadata, len(payload), blocks)


def _unpack(sample):
    if not 0 <= sample.payload_size <= MAX_BYTES or len(sample.blocks) != math.ceil(sample.payload_size / BLOCK_BYTES):
        raise ValueError('invalid DDS payload length')
    return decode(sample.metadata, b''.join(sample.blocks)[:sample.payload_size])


class _Connection:
    def __init__(self, namespace, domain_id, server):
        if not isinstance(namespace, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', namespace):
            raise ValueError('DDS namespace must contain only letters, digits, underscores or hyphens')
        if type(domain_id) is not int or not 0 <= domain_id <= 232:
            raise ValueError('DDS domain_id must be an integer within [0, 232]')
        from cyclonedds.core import Listener
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from cyclonedds.util import duration

        self.closed = False
        self.participant = self.reader = self.writer = None
        self.topics = []
        self._received_match, self._sent_match = threading.Event(), threading.Event()
        received, sent = self._received_match, self._sent_match
        self._listener = Listener(
            on_subscription_matched=lambda _, status: received.set() if status.current_count else received.clear(),
            on_publication_matched=lambda _, status: sent.set() if status.current_count else sent.clear())
        qos = Qos(Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                  Policy.History.KeepLast(8), Policy.Durability.Volatile)
        try:
            self.participant = DomainParticipant(domain_id)
            self.topics = [Topic(self.participant, f'egox/{namespace}/{name}', _packet_type())
                           for name in ('request', 'reply')]
            self.reader = DataReader(self.participant, self.topics[0 if server else 1], qos=qos,
                                     listener=self._listener)
            self.writer = DataWriter(self.participant, self.topics[1 if server else 0], qos=qos,
                                     listener=self._listener)
        except BaseException:
            self.close()
            raise

    def _ready(self, deadline):
        for event in (self._sent_match, self._received_match):
            if not event.wait(max(0., deadline - time.monotonic())):
                return False
        return True

    def _take(self, deadline):
        if self.closed:
            raise RuntimeError('DDS connection is closed')
        while True:
            samples = self.reader.take(N=1)
            if samples:
                sample = samples[0]
                if getattr(getattr(sample, 'sample_info', None), 'valid_data', True):
                    return sample
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if not samples:
                time.sleep(min(.002, remaining))

    def close(self):
        self.closed = True
        self.writer = self.reader = None
        self.topics.clear()
        self.participant = None
        self._listener = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Server(_Connection):
    """Single-owner request processing; callers execute handlers on their own thread."""

    def __init__(self, namespace, domain_id=0):
        super().__init__(namespace, domain_id, server=True)
        self._pending = None
        self._sequences = {}

    @property
    def client_id(self):
        return self._pending.client_id if self._pending is not None else None

    def receive(self, timeout_ms=0):
        if self._pending is not None:
            raise RuntimeError('reply to the current DDS request before receiving another')
        deadline = time.monotonic() + max(0., timeout_ms / 1000)
        for _ in range(32):  # Keep invalid/duplicate traffic from starving the controller loop.
            sample = self._take(deadline)
            if sample is None:
                return None
            if sample.sequence <= self._sequences.get(sample.client_id, 0):
                continue  # Duplicate delivery must never execute a command twice.
            self._sequences[sample.client_id] = sample.sequence
            self._pending = sample
            try:
                request = _unpack(sample)
                if not isinstance(request, dict) or not isinstance(request.get('method'), str):
                    raise ValueError('invalid DDS request')
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.reply({'ok': False, 'error': f'invalid DDS request: {exc}'})
                continue
            return request
        return None

    def reply(self, value):
        if self._pending is None:
            raise RuntimeError('no pending DDS request')
        metadata, payload = encode(value)
        sample = self._pending
        self.writer.write(_packet(sample.client_id, sample.sequence, metadata, payload))
        self._pending = None


class Client(_Connection):
    """One outstanding request per client; timeout means its outcome is unknown."""

    def __init__(self, namespace, domain_id=0, timeout_ms=60000):
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError('timeout_ms must be finite and positive')
        self.timeout_ms, self.client_id, self.sequence = timeout_ms, uuid.uuid4().hex, 0
        super().__init__(namespace, domain_id, server=False)

    def request(self, method, **payload):
        if self.closed:
            raise RuntimeError('DDS connection is closed')
        if not isinstance(method, str) or not method:
            raise ValueError('DDS method must be a nonempty string')
        metadata, data = encode({**payload, 'method': method})
        deadline = time.monotonic() + self.timeout_ms / 1000
        if self._ready(deadline) and time.monotonic() < deadline:
            self.sequence += 1
            try:
                self.writer.write(_packet(self.client_id, self.sequence, metadata, data))
            except BaseException:
                self.closed = True
                self.close()  # A transport failure leaves execution uncertain, just like a timeout.
                raise
            while time.monotonic() < deadline:
                sample = self._take(deadline)
                if sample is None:
                    break
                if sample.client_id != self.client_id or sample.sequence != self.sequence:
                    continue
                try:
                    reply = _unpack(sample)
                except (ValueError, TypeError, KeyError, OverflowError):
                    self.closed = True
                    self.close()
                    raise
                if not isinstance(reply, dict) or not reply.get('ok'):
                    error = reply.get('error', 'invalid reply') if isinstance(reply, dict) else 'invalid reply'
                    raise RuntimeError(f'robot {method} failed: {error}')
                return reply
        self.closed = True
        self.close()
        raise TimeoutError(f'robot {method} timed out; outcome unknown, reconnect before another request')
