"""Launch/stop the separate Isaac process for collection or evaluation."""
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import time

EXAMPLE = Path(__file__).resolve().parent


def stop_simulator(process):
    """Close only the simulator process group created for this trial."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def start_simulator(args, trial_dir, namespace, log, events=None, *, extra_args=()):
    command = [str(args.sim_python), '-u', str(EXAMPLE / 'run_sim.py'), '--trial', str(trial_dir),
               '--namespace', namespace, '--dds-domain', str(args.dds_domain)]
    command.extend(extra_args)
    if not args.visible:
        command.append('--headless')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                               env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if events is not None:
        events.put({'event': 'started', 'pid': process.pid, 'worker': os.getpid()})
    try:
        deadline = time.monotonic() + args.startup_timeout
        with Path(log.name).open() as reader:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'simulator exited with code {process.returncode}; see {log.name}')
                if 'simulation ready' in reader.read():
                    return process
                time.sleep(.2)
        raise TimeoutError(f'simulator not ready after {args.startup_timeout} s; see {log.name}')
    except BaseException:
        stop_simulator(process)
        if events is not None:
            events.put({'event': 'stopped', 'pid': process.pid})
        raise
