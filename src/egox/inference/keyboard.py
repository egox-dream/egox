"""Keyboard commands for interactive rollout; callbacks never call the robot or policy."""
import queue


class ResetRequested(Exception):
    """End the current attempt so its owner can reset the robot and recording."""


class KeyboardControls:
    """Use the main-branch 1/2/3 keys, plus Q to quit, without importing a GUI in batch jobs."""

    KEYS = {'1': 'reset', '2': 'start', '3': 'pause', 'q': 'quit'}

    def __init__(self):
        self.commands = queue.Queue()
        self.pressed = set()
        self.listener = None

    def _press(self, key):
        char = getattr(key, 'char', None)
        char = char.lower() if char else None
        if char in self.KEYS and char not in self.pressed:
            self.pressed.add(char)
            self.commands.put(self.KEYS[char])

    def _release(self, key):
        char = getattr(key, 'char', None)
        self.pressed.discard(char.lower() if char else None)

    def poll(self, timeout=0.):
        if self.listener is None or not self.listener.is_alive():
            raise RuntimeError('Keyboard listener stopped; policy execution has stopped.')
        try:
            return self.commands.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        try:
            from pynput.keyboard import Listener

            self.listener = Listener(on_press=self._press, on_release=self._release)
            self.listener.start()
        except Exception as exc:
            self.__exit__(None, None, None)
            raise RuntimeError("Keyboard controls require pynput and an accessible desktop keyboard. "
                               "Install pip install -e '.[inference]' and run in a desktop session.") from exc
        return self

    def __exit__(self, *_):
        if self.listener is not None:
            self.listener.stop()
            if self.listener.ident is not None:
                self.listener.join(timeout=1.)
            self.listener = None
