"""Serve the G1 DDS controller in a kitchen trial or on empty ground.

Evaluation advances one 30 Hz sample per command. State-only collection uses
realtime physics and a lightweight main-camera preview; training RGB is rendered offline.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import signal
import time

from egox.robots.g1_sim_dex3.description import CAMERA_RIG, USD
from scene import CollectionReset, load_trial


def main():
    try:
        import cyclonedds  # noqa: F401 -- fail before allocating the simulator/GPU.
    except ImportError as exc:
        raise RuntimeError('The simulation environment needs cyclonedds==11.0.1. '
                           'Install it with: python -m pip install cyclonedds==11.0.1') from exc
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, help='directory containing trial.json and state.npz')
    parser.add_argument('--randomize', action='store_true',
                        help='collection only: resample object positions on each reset; evaluation stays frozen')
    parser.add_argument('--seed', type=int, default=42, help='collection object-reset random seed')
    parser.add_argument('--capture_mode', choices=('rgb', 'state_only'), default='rgb',
                        help='rgb: fixed-step evaluation; state_only: realtime collection with offline RGB')
    parser.add_argument('--preview', action=argparse.BooleanOptionalAction, default=True,
                        help='allocate main stereo cameras for state-only headset preview')
    parser.add_argument('--robot_usd', type=Path, default=USD)
    parser.add_argument('--scene_usd', type=Path, help='physics-authored room USD; a ground plane otherwise')
    parser.add_argument('--scene_position', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--scene_yaw', type=float, default=0.)
    parser.add_argument('--robot_position', nargs=3, type=float, default=[0., 0., .8])
    parser.add_argument('--robot_yaw', type=float, default=0., help='radians')
    parser.add_argument('--camera_rig', type=Path, default=CAMERA_RIG)
    parser.add_argument('--settle_steps', type=int, default=30, help='50 Hz control steps before the first reset')
    parser.add_argument('--namespace', default='g1_sim_dex3', help='DDS topic namespace for this robot instance')
    parser.add_argument('--dds-domain', type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device='cpu')  # CPU PhysX; RGB cameras still render on the GPU.
    args = parser.parse_args()
    if args.randomize and not args.trial:
        parser.error('--randomize requires --trial')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if not args.robot_usd.is_file():
        parser.error(f'missing robot USD {args.robot_usd}; run python scripts/download_assets.py')
    if args.settle_steps < 0:
        parser.error('--settle_steps must be nonnegative')
    # Pinocchio's Assimp must load before Kit's incompatible build.
    import pinocchio  # noqa: F401

    args.enable_cameras = True
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    with ExitStack() as cleanup:
        app = AppLauncher(args).app
        cleanup.callback(app.close)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous = signal.signal(sig, stop)
            cleanup.callback(signal.signal, sig, previous)

        from egox.robots.g1_sim_dex3.environment import G1Dex3Env
        from egox.robots.service import ControllerService
        from isaaclab.sim import SimulationContext
        from tasks import make_task

        # Also cover a partially constructed environment: app shutdown must not
        # enter IsaacLab's STOP callback, which waits indefinitely for Play.
        cleanup.callback(SimulationContext.clear_instance)
        trial = load_trial(args.trial) if args.trial else None
        env = G1Dex3Env(args, task_factory=make_task, trial=trial)
        cleanup.callback(env.close)
        reset = CollectionReset(env, args.seed) if args.randomize else env.reset
        service = ControllerService(observe=env.observe, send=env.step, reset=reset, describe=env.describe,
                                    **({'pause': env.pause, 'preview': env.preview}
                                       if args.capture_mode == 'state_only' else {}),
                                    namespace=args.namespace, domain_id=args.dds_domain)
        cleanup.callback(service.close)
        print(f'G1 Dex3 simulation ready on DDS {args.dds_domain}/{args.namespace}', flush=True)
        last_viewport = time.monotonic()
        while not stopping and app.is_running():
            if args.capture_mode == 'state_only':
                env.tick()
                service.poll(timeout_ms=2)
                # With no headset, refresh only the optional viewport; no training cameras exist.
                if not args.preview and not args.headless and time.monotonic() - last_viewport >= 1 / 24:
                    env.render()
                    last_viewport = time.monotonic()
            elif not service.poll(timeout_ms=30):
                env.render()  # Keep Kit and the optional viewport responsive while waiting.


if __name__ == '__main__':
    main()
