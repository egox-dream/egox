"""Shared policy execution through an embodiment's robot client."""
import math
import time

from .keyboard import ResetRequested


class _ControlGate:
    """Apply queued keyboard events on the rollout thread, including during pacing waits."""

    def __init__(self, controls):
        self.controls = controls
        self.running = self.refresh = False
        print('Ready: 1 reset, 2 start/resume, 3 pause commands, Q quit. Waiting for 2.', flush=True)

    def _handle(self, command):
        if command == 'quit':
            raise KeyboardInterrupt
        if command == 'reset':
            raise ResetRequested
        if command == 'pause':
            self.running = False
            print('Policy commands paused. Press 2 to resume or 1 to reset.', flush=True)
        elif command == 'start' and not self.running:
            self.running = self.refresh = True

    def interrupted(self):
        while (command := self.controls.poll()) is not None:
            self._handle(command)
        return not self.running or self.refresh

    def wait(self):
        self.interrupted()
        while not self.running:
            self._handle(self.controls.poll(timeout=.05))
            self.interrupted()
        refresh, self.refresh = self.refresh, False
        return refresh

    def ready_to_send(self, deadline):
        while not self.interrupted():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._handle(self.controls.poll(timeout=min(.05, remaining)))
        return False


def rollout(client, policy, observation, steps, *, action_hz=None, controls=None, preprocess=None, stop_on=None):
    """Yield the initial observation and each result of sending a policy action.

    The caller owns the connection and robot reset. Each run clears the policy's
    action chunk. Preprocess yielded observations in place before continuing;
    breaking the iteration stops before another action is computed or sent.

    Optional ``action_hz`` caps wall-clock send frequency without catching up
    after delays. Otherwise the controller determines pacing. The embodiment
    controller always owns its low-level control loop.

    With ``controls``, wait for start and handle pause/reset/quit at command
    boundaries. Resume observes again and discards the previous action chunk.
    ``preprocess(observation, step)`` also prepares these refreshed observations.
    ResetRequested tells the caller to discard the attempt and reset its backend.
    An already-sent request must finish before another event can be applied.
    ``stop_on`` also checks refreshed feedback before issuing another action.
    """
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    if action_hz is not None and (not math.isfinite(action_hz) or action_hz <= 0):
        raise ValueError('action_hz must be finite and positive')
    policy.reset()
    gate = _ControlGate(controls) if controls is not None else None
    inferred = False
    next_send = time.monotonic() if action_hz is not None else 0.
    if preprocess is not None:
        preprocess(observation, 0)
    yield observation
    for step in range(steps):
        while True:
            if gate is not None and gate.wait():
                if inferred:
                    policy.reset()
                observation = client.observe()
                if preprocess is not None:
                    preprocess(observation, step)
            if stop_on and observation.get(stop_on, False):
                return
            action = policy.act(observation)
            inferred = True
            if gate is not None:
                if not gate.ready_to_send(next_send):
                    continue  # A pause/reset during inference must not send a stale action.
            elif action_hz is not None:
                time.sleep(max(0., next_send - time.monotonic()))
            break
        if action_hz is not None:
            next_send = time.monotonic() + 1 / action_hz
        observation = client.send(action)
        if preprocess is not None:
            preprocess(observation, step + 1)
        yield observation
