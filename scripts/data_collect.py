"""Collect Pico demonstrations through the shared robot DDS interface."""
import argparse
from contextlib import ExitStack
import logging
import math
from pathlib import Path
import re
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument('--output', type=Path, required=True,
                        help='accepted episodes; simulation saves .state.h5 for offline RGB generation')
    common.add_argument('--embodiment', default='G1-Sim-Dex3')
    common.add_argument('--teleoperator', choices=('pico', 'apple_vision_pro'), default='pico',
                        help='Pico is implemented; Apple Vision Pro is reserved for a future adapter')
    common.add_argument('--dds_domain', type=int, default=0)
    common.add_argument('--calibration', type=Path, help='optional Pico axis calibration JSON; defaults match old collection')
    common.add_argument('--max_tracking_age', type=float, default=.5, help='pause after this many seconds without fresh tracking')
    common.add_argument('--tracking_recovery_window', type=float,
                        help='brief-gap recovery window from the last accepted sample; default: sim=1s, connect=0 (manual)')
    common.add_argument('--jpeg_quality', type=int, default=95,
                        help='RGB controllers only; offline rendering has its own JPEG quality setting')
    common.add_argument('--pico_host', help='headset IP for optional Remote Vision; recorded images are unchanged')
    common.add_argument('--pico_port', type=int, default=12345)

    backends = parser.add_subparsers(dest='backend', required=True)
    sim = backends.add_parser('sim', parents=[common], allow_abbrev=False,
                             help='launch a kitchen scene with randomized object initialization')
    sim.add_argument('--task', choices=('kitchen_bread', 'kitchen_cleanup'), required=True)
    sim.add_argument('--mode', choices=('interactive', 'headless'), default='headless',
                     help='headless by default; interactive opens the desktop simulator window')
    sim.add_argument('--sim_python', type=Path, default=Path(sys.executable),
                     help='Isaac environment Python; defaults to the active interpreter')
    sim.add_argument('--seed', type=int, default=0, help='object initialization RNG seed; each X reset draws a new placement')
    sim.add_argument('--startup_timeout', type=float, default=900.)
    sim.add_argument('--namespace', help='default: a unique DDS namespace for this session')
    connect = backends.add_parser('connect', parents=[common], allow_abbrev=False,
                                 help='collect from an already-running controller')
    connect.add_argument('--namespace', required=True)
    args = parser.parse_args()
    if args.teleoperator != 'pico':
        parser.error('Apple Vision Pro is not implemented; add a Teleoperator adapter before selecting it')
    if args.embodiment != 'G1-Sim-Dex3':
        parser.error('only G1-Sim-Dex3 has a collection action mapper in this release')
    if not 0 <= args.dds_domain <= 232:
        parser.error('--dds_domain must be within [0, 232]')
    if args.namespace is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', args.namespace):
        parser.error('--namespace must contain only letters, digits, underscores or hyphens')
    if not math.isfinite(args.max_tracking_age) or args.max_tracking_age <= 0:
        parser.error('--max_tracking_age must be finite and positive')
    if args.tracking_recovery_window is None:
        args.tracking_recovery_window = 1. if args.backend == 'sim' else 0.
    if not math.isfinite(args.tracking_recovery_window) or args.tracking_recovery_window < 0:
        parser.error('--tracking_recovery_window must be finite and nonnegative')
    if not 1 <= args.jpeg_quality <= 100 or not 1 <= args.pico_port <= 65535:
        parser.error('--jpeg_quality must be within [1, 100] and --pico_port within [1, 65535]')
    if args.backend == 'sim':
        if not args.sim_python.is_file():
            parser.error(f'simulator Python does not exist: {args.sim_python}')
        if args.seed < 0 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
            parser.error('--seed must be nonnegative and --startup_timeout finite and positive')

    from egox.robots.g1_sim_dex3.teleop import G1Dex3Teleop
    from egox.robots.service import RobotClient
    from egox.teleoperator import Pico
    from egox.teleoperator.data_collecter import DataCollector, EpisodeWriter

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        with ExitStack() as cleanup:
            # Fail on missing tracking SDK before launching the expensive simulator.
            device = cleanup.enter_context(Pico(calibration=args.calibration, max_age_s=args.max_tracking_age))
            preview = None
            if args.pico_host:
                from egox.teleoperator.pico_video import PicoVideoStreamer

                preview = PicoVideoStreamer(args.pico_host, port=args.pico_port, fps=24, stereo_input=True)
                cleanup.callback(preview.stop)
                if not preview.start():
                    raise RuntimeError(preview.error)
            if args.backend == 'sim':
                repo = Path(__file__).resolve().parents[1]
                sys.path.insert(0, str(repo))
                from examples.kitchen.launch import start_simulator, stop_simulator

                args.visible = args.mode == 'interactive'
                args.namespace = args.namespace or 'collect_' + uuid.uuid4().hex
                args.output.mkdir(parents=True, exist_ok=True)
                log = cleanup.enter_context((args.output / f'{args.namespace}.log').open('x'))
                trial = repo / 'examples/kitchen/benchmarks' / args.task / 'trial_001'
                print(f'Starting simulation; see {log.name}', flush=True)
                process = start_simulator(args, trial, args.namespace, log,
                                          extra_args=('--randomize', '--seed', str(args.seed),
                                                      '--capture_mode', 'state_only',
                                                      '--preview' if preview is not None else '--no-preview'))
                cleanup.callback(stop_simulator, process)
            client = cleanup.enter_context(RobotClient(args.namespace, domain_id=args.dds_domain))
            description = client.describe()
            if description['embodiment'] != args.embodiment:
                raise ValueError(f'controller embodiment {description["embodiment"]!r} does not match {args.embodiment!r}')
            mapper = G1Dex3Teleop(description['teleop'])
            options = {}
            if description.get('capture_mode') == 'state_only':
                from egox.teleoperator.data_collecter import AsyncEpisodeWriter, PreviewWorker, StateEpisodeWriter

                writer = cleanup.enter_context(AsyncEpisodeWriter(
                    StateEpisodeWriter(args.output, args.embodiment, description['collection'])))
                if preview is not None:
                    preview = cleanup.enter_context(PreviewWorker(
                        lambda: RobotClient(args.namespace, domain_id=args.dds_domain, timeout_ms=1000), preview))
                options['state_only'] = True
                print('Realtime collection: saving .state.h5; generate training images offline with render.sh.', flush=True)
            else:
                writer = cleanup.enter_context(EpisodeWriter(args.output, args.embodiment, args.jpeg_quality))
            collector = DataCollector(client, device, mapper, writer, preview=preview,
                                      max_age_s=args.max_tracking_age,
                                      tracking_recovery_s=args.tracking_recovery_window, **options)
            collector.run(action_hz=description['action_hz'])
    except KeyboardInterrupt:
        print('\nCollection stopped; the unsaved episode was discarded.', flush=True)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
