"""Launch/stop the separate Isaac process for collection or evaluation."""
import contextlib
import ctypes
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

EXAMPLE = Path(__file__).resolve().parent


@contextlib.contextmanager
def ignore_shutdown_signals():
    """Let bounded cleanup finish even if Ctrl-C or SIGTERM arrives again."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def guard_parent(parent_pid):
    """On Linux, stop this child if its launcher disappears, including before setup."""
    if sys.platform == 'linux':
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)


def configure_simulator_process():
    """Called before Isaac imports; manual simulator launches need no parent/pipe."""
    parent_pid = os.environ.pop('EGOX_SIM_PARENT_PID', None)
    if parent_pid is not None:
        guard_parent(int(parent_pid))
    ready_fd = os.environ.pop('EGOX_SIM_READY_FD', None)
    if ready_fd is not None:
        ready_fd = int(ready_fd)
        os.set_inheritable(ready_fd, False)
    return ready_fd


def simulator_ready(ready_fd):
    """Notify only this launcher's pipe after the controller is ready."""
    if ready_fd is not None:
        try:
            os.write(ready_fd, b'ready\n')
        finally:
            os.close(ready_fd)


def stop_simulator(process):
    """Close only the simulator process group created for this trial."""
    with ignore_shutdown_signals():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def start_simulator(args, trial_dir, namespace, log, events=None, *, extra_args=()):
    command = [str(args.sim_python), '-u', str(EXAMPLE / 'run_sim.py'), '--trial', str(trial_dir),
               '--namespace', namespace, '--dds-domain', str(args.dds_domain)]
    command.extend(extra_args)
    if not args.visible:
        command.append('--headless')
    ready_read, ready_write = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(ready_write,),
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'EGOX_SIM_PARENT_PID': str(os.getpid()), 'EGOX_SIM_READY_FD': str(ready_write)})
        os.close(ready_write)
        ready_write = None
        if events is not None:
            events.put({'event': 'started', 'pid': process.pid, 'worker': os.getpid()})
        deadline = time.monotonic() + args.startup_timeout
        with selectors.DefaultSelector() as selector:
            selector.register(ready_read, selectors.EVENT_READ)
            message = b''
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'simulator exited with code {process.returncode}; see {log.name}')
                if selector.select(min(.2, max(0., deadline - time.monotonic()))):
                    chunk = os.read(ready_read, 16)
                    if not chunk:
                        raise RuntimeError(f'simulator closed its readiness pipe before ready; see {log.name}')
                    message += chunk
                    if message == b'ready\n':
                        return process
                    if not b'ready\n'.startswith(message):
                        raise RuntimeError(f'invalid simulator readiness signal; see {log.name}')
        raise TimeoutError(f'simulator not ready after {args.startup_timeout} s; see {log.name}')
    except BaseException:
        if process is not None:
            stop_simulator(process)
            if events is not None:
                events.put({'event': 'stopped', 'pid': process.pid})
        raise
    finally:
        os.close(ready_read)
        if ready_write is not None:
            os.close(ready_write)
