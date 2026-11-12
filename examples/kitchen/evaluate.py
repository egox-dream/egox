"""Evaluate current MXT checkpoints on the frozen Kitchen-Bread / Kitchen-Cleanup trials.

The policy and Isaac Sim run in separate processes. Each worker runs one trial
at a time, with simulation-paced actions and rendered cameras in both GUI and headless
modes. Completed traces are re-scored on resume; failed trials are retried.
"""
import contextlib
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import time
import traceback
import uuid

import numpy as np

from egox.inference.rollout import rollout
from egox.inference.keyboard import KeyboardControls, ResetRequested
from egox.robots.service import RobotClient
from egox.training.checkpoints import RUN_CONFIG, RUN_STATS, SNAPSHOT
from examples.kitchen import scoring
from examples.kitchen.launch import start_simulator, stop_simulator

EXAMPLE = Path(__file__).resolve().parent
BENCHMARKS = EXAMPLE / 'benchmarks'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def run_identity(args, protocol):
    """Fingerprint the weights and preprocessing, preserving modality ordering."""
    checkpoint = (args.model_path / args.checkpoint).resolve()
    identity = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'embodiment': args.embodiment, 'benchmark': args.benchmark, 'protocol': protocol,
                'benchmark_sha256': digest(BENCHMARKS / args.benchmark / 'benchmark.json'),
                'sensor_sha256': digest(EXAMPLE / 'sensor.py')}
    if checkpoint.name != SNAPSHOT:
        config = json.loads((args.model_path / RUN_CONFIG).read_text())
        fields = ('policy_config', 'height', 'width', 'main_cam_height', 'main_cam_width',
                  'wrist_cam_height', 'wrist_cam_width', 'bf16')
        selected = {key: config.get(key) for key in fields}
        identity['configuration_sha256'] = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
        identity['statistics_sha256'] = digest(args.model_path / RUN_STATS)
    return identity


def prepare_output(args, protocol):
    """Bind resumed results to the exact checkpoint, preprocessing and protocol."""
    identity = run_identity(args, protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'run.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('output belongs to a different checkpoint/configuration/protocol; choose another --output')
    elif any(args.output.glob('trial_*/result.json')):
        raise ValueError('output contains results without run.json; choose another --output')
    else:
        write_json(manifest, identity)
    args.run_identity = identity


def load_policy(args, protocol):
    from egox.inference import MXTPolicy
    if run_identity(args, protocol) != args.run_identity:
        raise ValueError('checkpoint or metadata changed before policy loading; use a fixed checkpoint and a new output')
    policy = MXTPolicy(args.model_path, args.checkpoint, args.embodiment, args.device,
                       inference_interval=protocol['inference_interval'])
    if run_identity(args, protocol) != args.run_identity:
        raise ValueError('checkpoint or metadata changed during policy loading; use a fixed checkpoint and a new output')
    return policy


def run_episode(client, policy, trial, protocol, video=None, *, controls=None):
    """Trace every 30 Hz observation, including the initial and terminal states."""
    steps = round(protocol['duration_s'] * protocol['action_hz'])
    def preprocess(observation, row):
        if trial.get('sensor') is not None:
            from examples.kitchen import sensor
            for camera in ('main', 'left_wrist', 'right_wrist'):
                key = f'{camera}_image'
                observation['obs'][key] = sensor.postprocess(observation['obs'][key], trial['sensor'], row, camera)

    states, reason = [], 'horizon'
    for observation in rollout(client, policy, client.reset(), steps, controls=controls, preprocess=preprocess):
        states.append(observation['state'])
        if video is not None:
            obs = observation['obs']
            wrists = np.concatenate([obs['left_wrist_image'], obs['right_wrist_image']], axis=1)
            video.write(np.concatenate([obs['main_image'], wrists], axis=0)[..., ::-1])
        if observation['fell']:
            reason = 'robot_fell'
            break
    return {key: np.asarray([state[key] for state in states]) for key in states[0]}, reason


def evaluate_trial(args, policy, task, trial_id, protocol, output, events=None, *, controls=None):
    trial_dir = BENCHMARKS / task / trial_id
    trial = json.loads((trial_dir / 'trial.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    namespace = f'egox_{uuid.uuid4().hex}'
    with (output / 'simulator.log').open('w') as log:
        process = start_simulator(args, trial_dir, namespace, log, events)
        try:
            started = time.monotonic()
            with RobotClient(namespace, domain_id=args.dds_domain, timeout_ms=600000) as client:
                while True:
                    video = None
                    try:
                        if args.video:
                            import cv2
                            video = cv2.VideoWriter(str(output / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                    protocol['action_hz'], (1280, 960))
                            if not video.isOpened():
                                raise RuntimeError('could not open video writer')
                        trace, reason = run_episode(client, policy, trial, protocol, video, controls=controls)
                        break
                    except ResetRequested:
                        print(f'{trial_id}: resetting; discarded the unfinished attempt.', flush=True)
                    finally:
                        if video is not None:
                            video.release()
        finally:
            stop_simulator(process)
            if events is not None:
                events.put({'event': 'stopped', 'pid': process.pid})
    temporary = output / 'trace.tmp.npz'
    np.savez_compressed(temporary, **trace)
    temporary.replace(output / 'trace.npz')
    result = scoring.score_trial(task, trace, trial['scoring'])
    result.update(trial=trial_id, group=trial['group'], termination=reason,
                  wall_time_s=round(time.monotonic() - started, 1))
    write_json(output / 'result.json', result)
    (output / 'error.json').unlink(missing_ok=True)
    return result


def rescore(task, trial_id, output):
    trial = json.loads((BENCHMARKS / task / trial_id / 'trial.json').read_text())
    with np.load(output / 'trace.npz', allow_pickle=False) as trace:
        scores = scoring.score_trial(task, dict(trace), trial['scoring'])
    path = output / 'result.json'
    write_json(path, {**json.loads(path.read_text()), **scores})


def worker(args, protocol, trial_ids, events):
    def terminate(*_):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, terminate)
    try:
        policy = load_policy(args, protocol)
    except Exception:
        error = traceback.format_exc()
        for trial_id in trial_ids:
            events.put({'trial': trial_id, 'error': error})
        return
    for trial_id in trial_ids:
        try:
            result = evaluate_trial(args, policy, args.benchmark, trial_id, protocol, args.output / trial_id, events)
        except Exception:
            result = {'trial': trial_id, 'error': traceback.format_exc()}
        events.put(result)


def run_interactive(args, protocol, pending):
    """Own the keyboard in the foreground process; Isaac remains a separate process."""
    with KeyboardControls() as controls:
        policy = load_policy(args, protocol)
        for trial_id in pending:
            print(f'Interactive trial: {trial_id}', flush=True)
            evaluate_trial(args, policy, args.benchmark, trial_id, protocol,
                           args.output / trial_id, controls=controls)


def run_workers(args, protocol, pending):
    """Bound concurrency and clean up a crashed worker without stopping its peers."""
    context = multiprocessing.get_context('spawn')
    events = context.Queue()
    assignments = [pending[i::args.workers] for i in range(min(args.workers, len(pending)))]
    workers = [context.Process(target=worker, args=(args, protocol, trials, events)) for trials in assignments]
    remaining, simulators, reaped = set(pending), {}, set()

    def check_workers(drained=False):
        for process, trials in zip(workers, assignments):
            if process.pid is None or process.pid in reaped or process.exitcode is None:
                continue
            # Normal exits flush their queue before leaving; consume those results first.
            if process.exitcode == 0 and not drained:
                continue
            reaped.add(process.pid)
            for pid, owner in list(simulators.items()):
                if owner == process.pid:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
                    del simulators[pid]
            for trial_id in trials:
                if trial_id not in remaining:
                    continue
                remaining.remove(trial_id)
                directory = args.output / trial_id
                if (directory / 'result.json').is_file() and (directory / 'trace.npz').is_file():
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                error = f'worker exited without a result (exit code {process.exitcode})'
                write_json(directory / 'error.json', {'trial': trial_id, 'error': error})
                print(f'{trial_id}: {error}', flush=True)

    try:
        for process in workers:
            process.start()
        while remaining:
            try:
                result = events.get(timeout=1)
            except queue.Empty:
                check_workers(drained=True)
                continue
            if 'event' in result:
                if result['event'] == 'started':
                    simulators[result['pid']] = result['worker']
                    # The startup event may arrive just after its owner was reaped.
                    if result['worker'] in reaped:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(result['pid'], signal.SIGKILL)
                        del simulators[result['pid']]
                else:
                    simulators.pop(result['pid'], None)
            else:
                trial_id = result['trial']
                remaining.discard(trial_id)
                if 'error' in result:
                    directory = args.output / trial_id
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory / 'error.json', result)
                    print(f'{trial_id}: {result["error"]}', flush=True)
                else:
                    print(f"{trial_id}: success={result['success']} score={result['task_score']}/"
                          f"{result['maximum_task_score']} ({result['termination']})", flush=True)
            check_workers()
    finally:
        for process in workers:
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        for pid in simulators:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        events.close()
        events.join_thread()


def select_trials(trials, selection):
    known = {trial['id'] for trial in trials}
    unknown = set(selection) - known - {'all', 'id', 'ood'}
    if unknown:
        raise ValueError(f'unknown trial selection: {sorted(unknown)}')
    return [trial['id'] for trial in trials if 'all' in selection or trial['id'] in selection
            or ('id' in selection and trial['group'] == 'id') or ('ood' in selection and trial['group'] != 'id')]


def evaluate(args):
    """Run kitchen trials from the shared rollout CLI's validated arguments."""
    benchmark = json.loads((BENCHMARKS / args.benchmark / 'benchmark.json').read_text())
    if args.embodiment is None:
        args.embodiment = benchmark['embodiment']
    elif args.embodiment != benchmark['embodiment']:
        raise ValueError(f"{args.benchmark} requires {benchmark['embodiment']}")
    protocol = dict(benchmark['protocol'])
    if args.inference_interval is not None:
        protocol['inference_interval'] = args.inference_interval
    selected = select_trials(benchmark['trials'], args.trials)
    if not selected:
        raise ValueError('no trials selected')
    prepare_output(args, protocol)
    pending = []
    for trial_id in selected:
        output = args.output / trial_id
        if (output / 'result.json').is_file() and (output / 'trace.npz').is_file():
            rescore(args.benchmark, trial_id, output)
        else:
            pending.append(trial_id)
    print(f'{args.benchmark}: {len(selected)} trials, {len(pending)} to run, {args.workers} worker(s)', flush=True)
    if pending:
        if getattr(args, 'visible', False):
            run_interactive(args, protocol, pending)
        else:
            run_workers(args, protocol, pending)
    results = [json.loads((args.output / trial / 'result.json').read_text()) for trial in selected
               if (args.output / trial / 'result.json').is_file() and (args.output / trial / 'trace.npz').is_file()]
    completed = {result['trial'] for result in results}
    summary = {'benchmark': args.benchmark, 'model_path': str(args.model_path), 'checkpoint': args.checkpoint,
               'protocol': protocol, 'summary': scoring.summarize(results) if results else None,
               'missing': [trial for trial in selected if trial not in completed],
               'trials': {result['trial']: {key: result[key] for key in ('group', 'success', 'task_score',
                                                                       'normalized_score', 'termination')}
                          for result in results}}
    write_json(args.output / 'results.json', summary)
    print(json.dumps(summary['summary'], indent=1))
    return 1 if summary['missing'] else 0
