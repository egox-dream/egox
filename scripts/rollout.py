"""Run a policy in simulation (headless or interactive) or on a real robot's DDS controller."""
import argparse
import math
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument('--model_path', type=Path, required=True, help='training run directory')
    common.add_argument('--checkpoint', default='policy_last.ckpt')
    common.add_argument('--embodiment', help='robot ID; inferred from the benchmark or single-embodiment checkpoint')
    common.add_argument('--dds_domain', type=int, default=0)
    common.add_argument('--inference_interval', type=int,
                        help='rows per action chunk; default: benchmark protocol or policy horizon')
    common.add_argument('--device', default='cuda:0')

    backends = parser.add_subparsers(dest='backend', required=True)
    sim = backends.add_parser('sim', parents=[common], allow_abbrev=False,
                             help='launch simulation trials and score the policy')
    sim.add_argument('--mode', choices=('headless', 'interactive'), default='headless',
                     help='headless: automatic evaluation; interactive: viewport and 1/2/3 keyboard controls')
    sim.add_argument('--benchmark', required=True, choices=('kitchen_bread', 'kitchen_cleanup'))
    sim.add_argument('--python_executable', dest='sim_python', type=Path, metavar='PATH', default=Path(sys.executable),
                     help='Python executable for simulation (default: current interpreter)')
    sim.add_argument('--sim_python', dest='sim_python', type=Path, help=argparse.SUPPRESS)
    sim.add_argument('--output', type=Path, required=True)
    sim.add_argument('--trials', nargs='+', default=['all'], help='all, id, ood, or trial IDs')
    sim.add_argument('--workers', type=int, default=1, help='parallel policy/simulator pairs')
    sim.add_argument('--video', action='store_true')
    sim.add_argument('--startup_timeout', type=float, default=900.)

    real = backends.add_parser('real', parents=[common], allow_abbrev=False,
                              help='connect to an already-running robot controller with 1/2/3 keyboard controls')
    real.add_argument('--namespace', required=True, help='DDS namespace of the controller service')
    real.add_argument('--episodes', type=int, default=1)
    real.add_argument('--steps', type=int, default=900, help='maximum actions per episode')
    real.add_argument('--reset', action='store_true', help='request a controller reset before each episode')
    real.add_argument('--action_hz', type=float, help='cap wall-clock action send rate; default: controller pacing')
    real.add_argument('--stop_on', metavar='FIELD', help='end an episode when this observation flag is true')
    args = parser.parse_args()
    if args.inference_interval is not None and args.inference_interval <= 0:
        parser.error('--inference_interval must be positive')
    if not 0 <= args.dds_domain <= 232:
        parser.error('--dds_domain must be within [0, 232]')

    if args.backend == 'sim':
        args.visible = args.mode == 'interactive'
        if args.workers < 1 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
            parser.error('--workers and --startup_timeout must be positive')
        if args.visible and args.workers != 1:
            parser.error('interactive mode requires --workers 1')
        if not args.sim_python.is_file():
            parser.error(f'simulator Python does not exist: {args.sim_python}')
        # Examples stay in the checkout; spawn workers must import the same module.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from examples.kitchen.evaluate import evaluate

        try:
            return evaluate(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    if min(args.episodes, args.steps) <= 0:
        parser.error('episodes and steps must be positive')
    if args.action_hz is not None and (not math.isfinite(args.action_hz) or args.action_hz <= 0):
        parser.error('--action_hz must be finite and positive')

    from egox.inference import MXTPolicy, rollout
    from egox.inference.keyboard import KeyboardControls, ResetRequested
    from egox.robots.service import RobotClient

    policy = MXTPolicy(args.model_path, args.checkpoint, args.embodiment, args.device,
                       inference_interval=args.inference_interval)
    with KeyboardControls() as controls, RobotClient(args.namespace, domain_id=args.dds_domain) as client:
        for episode in range(args.episodes):
            started = time.monotonic()
            reset = args.reset
            while True:
                observation = client.reset() if reset else client.observe()
                try:
                    for step, observation in enumerate(rollout(client, policy, observation, args.steps,
                                                               action_hz=args.action_hz, controls=controls,
                                                               stop_on=args.stop_on)):
                        if args.stop_on and observation.get(args.stop_on, False):
                            break
                except ResetRequested:
                    reset = True
                    continue
                else:
                    break
            print(f'episode {episode + 1}: {step} steps, wall time {time.monotonic() - started:.1f} s', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nRollout stopped.', flush=True)
        raise SystemExit(130)
