"""Shared DDS interface for collection and policy clients of an embodiment controller.

The service runs callbacks on its caller's thread. In Isaac, call ``poll`` from
the simulation loop; DDS never runs physics from a subscription callback.
"""

from .channel import Client, Server


class RobotClient(Client):
    """Observe a robot and submit embodiment-specific, physical-unit actions.

    ``send`` waits for execution acknowledgement and its observation. Fixed-step
    evaluation returns the next observation; realtime state-only collection
    returns the atomic pre-action state (advertised by ``describe``).
    ``reset`` explicitly resets an episode and is only available when the server
    supplies that operation. It is not a generic request to move real hardware.
    ``release`` hands over command ownership; normal close also attempts a
    bounded release. A failed connection cannot safely release ownership, so
    an unacknowledged command requires restarting the controller service.
    """

    def __init__(self, namespace='g1_sim_dex3', domain_id=0, timeout_ms=60000):
        self.episode_id = None
        self._owns_control = False
        super().__init__(namespace, domain_id=domain_id, timeout_ms=timeout_ms)

    def request(self, method, **payload):
        if method in ('step', 'reset', 'pause', 'release'):
            if self.episode_id is None:
                self.request('status')
            payload.setdefault('episode_id', self.episode_id)
        if method in ('step', 'reset'):
            # A callback may fail after accepting ownership or changing state.
            self._owns_control = True
        reply = super().request(method, **payload)
        self.episode_id = reply['episode_id']
        if method == 'release':
            self._owns_control = False
        return reply

    def observe(self):
        """Read an observation without taking ownership or advancing simulation."""
        return self.request('observe')

    def describe(self):
        """Read the controller's embodiment and optional teleoperation settings once."""
        return self.request('describe')

    def preview(self):
        """Read a lightweight JPEG preview without taking command ownership."""
        return self.request('preview')

    def send(self, action, *, source_time_ns=None):
        payload = {} if source_time_ns is None else {'source_time_ns': source_time_ns}
        return self.request('step', action=action, **payload)

    def pause(self):
        """Freeze a realtime simulation session while retaining its ownership."""
        return self.request('pause')

    def reset(self):
        return self.request('reset')

    def release(self):
        return self.request('release')

    def close(self):
        try:
            if self._owns_control and not self.closed:
                self._owns_control = False  # A release timeout must not recursively release.
                timeout = self.timeout_ms
                self.timeout_ms = min(timeout, 250)
                try:
                    self.release()
                except Exception:
                    pass  # A disconnected server cannot acknowledge ownership release.
                finally:
                    self.timeout_ms = timeout
        finally:
            super().close()


class ControllerService:
    """Serve an embodiment's observe/send callbacks, with optional episode reset.

    Multiple clients may observe; one client owns commands until it releases
    them. Episode IDs reject queued commands from before a successful reset.
    A lost owner is not replaced automatically: restart this service after an
    unacknowledged command or release to recover control.
    Callbacks return dictionaries; the description advertises the capture mode.
    The caller owns the controller/backend lifetime.
    """

    def __init__(self, *, observe, send, reset=None, describe=None, pause=None, preview=None, namespace='g1_sim_dex3',
                 domain_id=0, transport=None):
        self.observe, self.send, self.reset = observe, send, reset
        self.describe = describe
        self.pause, self.preview = pause, preview
        self.transport = transport if transport is not None else Server(namespace, domain_id=domain_id)
        self.owner = None
        self.episode_id = 0

    def poll(self, timeout_ms=0):
        """Handle at most one request on this thread; return False when idle."""
        request = self.transport.receive(timeout_ms)
        if request is None:
            return False
        try:
            method = request.get('method')
            if method == 'status':
                result = {}
            elif method == 'describe':
                if self.describe is None:
                    raise ValueError('controller description is not available')
                result = self.describe()
            elif method == 'observe':
                result = self.observe()
            elif method == 'preview':
                if self.preview is None:
                    raise ValueError('controller preview is not available')
                result = self.preview()
            elif method in ('step', 'reset', 'pause', 'release'):
                epoch = request.get('episode_id')
                if type(epoch) is not int or epoch != self.episode_id:
                    raise ValueError('episode changed; observe before sending another command')
                client = self.transport.client_id
                if self.owner is not None and self.owner != client:
                    raise RuntimeError('another client owns this controller; it must release control first')
                if method == 'release':
                    if self.pause is not None:
                        self.pause()
                    self.owner, result = None, {}
                elif method == 'pause':
                    if self.pause is None:
                        raise ValueError('controller pause is not supported')
                    result = self.pause()
                else:
                    if method == 'reset' and self.reset is None:
                        raise ValueError('episode reset is not supported by this controller')
                    if method == 'step' and 'action' not in request:
                        raise ValueError('step requires an action')
                    options = {}
                    if method == 'step' and 'source_time_ns' in request:
                        stamp = request['source_time_ns']
                        if type(stamp) is not int or stamp < 0:
                            raise ValueError('source_time_ns must be a nonnegative integer')
                        options['source_time_ns'] = stamp
                    self.owner = client
                    result = self.send(request['action'], **options) if method == 'step' else self.reset()
            else:
                raise ValueError(f'unknown controller method: {method}')
            if not isinstance(result, dict):
                raise TypeError('controller callbacks must return an observation dictionary')
            if method == 'reset':
                self.episode_id += 1
            reply = {**result, 'ok': True, 'episode_id': self.episode_id}
        except Exception as exc:
            reply = {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'episode_id': self.episode_id}
        self.transport.reply(reply)
        return True

    def close(self):
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
