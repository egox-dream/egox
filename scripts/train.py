"""Train MXT / MXT+; install with `pip install -e ".[training]"` first."""
import argparse
from dataclasses import InitVar
import json
from typing import get_args, get_origin, get_type_hints

from egox.configs.loader import build_model_config, list_robot_configs
from egox.training.config import TrainConfig


_HELP = {
    'dataset_dir': 'dataset folder(s), required for training; each is a sampling group',
    'default_embodiment': 'explicit embodiment for legacy HDF5 files missing the embodiment attribute',
    'sample_weights': 'sampling weight per dataset_dir (default: uniform)',
    'main_cam_width': 'main camera width after resizing; combined left + right width for stereo',
    'main_cam_height': 'main camera height after resizing',
    'wrist_cam_width': 'wrist camera width after resizing',
    'wrist_cam_height': 'wrist camera height after resizing',
    'main_cam_resolution': 'main camera resize "(width,height)"; width includes both stereo views '
                           f'(default: {TrainConfig.main_cam_width},{TrainConfig.main_cam_height})',
    'wrist_cam_resolution': 'per-wrist camera resize "(width,height)" '
                            f'(default: {TrainConfig.wrist_cam_width},{TrainConfig.wrist_cam_height})',
    'vel_rebalance': 'up-weight timesteps with active velocity commands',
    'robot_config_dir': 'robot YAML folder (default: the packaged configs/robots)',
    'trunk_config_path': 'trunk YAML (default: the packaged configs/model/transformer.yaml)',
    'robots': 'optional dataset filter: load only episodes with these embodiment IDs '
              '(default: discover all IDs in the data); without data, select configs to preview',
    'exclude_signals': 'signals to leave out of the model and data loading on every robot',
    'chunk_size': 'action horizon of every action head',
    'image_sharing': 'share each camera across embodiments: none (default), encoder (ResNet only), '
                     'or tokenizer (ResNet, projection and query pooling)',
    'image_feature_norm': 'LayerNorm on ResNet features before the projection',
    'embodiment_norm_upper_bound': 'override the L2 bound of the soft norm penalty on embodiment tokens',
    'embodiment_norm_upper_bound_weight': 'override the weight of the soft norm penalty; 0 disables it',
    'embodiment_dream_mag_beta': 'override dream_mag_beta of every embodiment_dream_head',
    'per_emb_batch_size': 'batches of this many samples from a single embodiment; one per embodiment per step',
    'batch_size': 'samples per batch (default: 32); pretraining can use per_emb_batch_size',
    'num_steps': 'training step limit (default: 10000)',
    'lr': 'fallback for optimizer groups without their own rate; every group has one now, so it has no effect',
    'gpu_id': 'CUDA device index among visible GPUs (default: 0)',
    'seed': 'random seed for initialization and data sampling (default: 6)',
    'bf16': 'enable bfloat16 autocast (default: on; --no-bf16 disables it)',
    'gradient_checkpoint': 'recompute image encoders in backward',
    'cudnn_benchmark': 'let cuDNN benchmark convolution implementations',
    'max_val_batches': 'maximum number of batches per validation pass',
    'snapshot_every': 'steps between resumable training snapshots',
    'ckpt_dir': 'prefix of the run directory',
    'task_name': 'run name, required for training; task scripts set a distinct name for each stage',
    'pretrained_path': 'initialize from this checkpoint (fine-tuning)',
    'resume_dir': 'manually resume training from this run directory',
    'auto_resume': 'use a fixed run directory and continue its training snapshot (default: on) '
                   '(--no-auto_resume starts a new timestamped directory)',
    'wandb': 'log metrics to Weights & Biases (default: off; --wandb enables it)',
    'wandb_name': 'Weights & Biases project, used only with --wandb (default: MXT); not the run name',
    'print_config': 'print the resolved settings without starting training',
}


def parse_args(argv=None):
    """Build CLI options from TrainConfig and apply them; paths are relative to the working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for name, annotation in get_type_hints(TrainConfig).items():
        annotation = TrainConfig._base_type(annotation.type if isinstance(annotation, InitVar) else annotation)
        options = {'default': getattr(defaults, name), 'help': _HELP.get(name)}
        if annotation is bool:
            options['action'] = argparse.BooleanOptionalAction
        elif get_origin(annotation) is list:
            options.update(nargs='+', type=get_args(annotation)[0])
        else:
            options['type'] = annotation
        if name in TrainConfig.choices:
            options['choices'] = TrainConfig.choices[name]
        flags = [f'--{name}']
        if name == 'print_config':
            flags.append('--print-config')
        parser.add_argument(*flags, **options)
    try:
        return TrainConfig(**vars(parser.parse_args(argv)))
    except ValueError as error:
        parser.error(str(error))


def main():
    args = parse_args()
    if not args.print_config:
        if not args.dataset_dir:
            raise SystemExit('Training requires --dataset_dir; supply the data folders explicitly.')
        if not args.task_name:
            raise SystemExit('Training requires --task_name; choose a run name.')
    try:
        episodes = None
        if args.dataset_dir:
            from egox.data.episodes import discover_episodes

            available = set(list_robot_configs(args.robot_config_dir))
            unknown = set(args.robots or ()) - available
            if unknown:
                raise ValueError(f'No robot YAMLs for filter IDs {sorted(unknown)}')
            episodes = discover_episodes(args.dataset_dir, args.robots, args.default_embodiment)
            for path, embodiment in episodes.embodiments.items():
                if embodiment not in available:
                    raise ValueError(f'{path}: no robot config {embodiment}.yaml; '
                                     'add it to the robot config directory')
        model_cfg = build_model_config(args, episodes.embodiments.values() if episodes is not None else None)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'Invalid configuration: {error}')
    if args.print_config:
        print(json.dumps({**args.to_dict(), 'policy_config': model_cfg}, indent=2))
        return
    from egox.training.trainer import train

    train(args, model_cfg, episodes)


if __name__ == '__main__':
    main()
