"""Run metadata and atomic training snapshots."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle

SNAPSHOT = 'training_last.ckpt'  # training state, configuration and normalization statistics
RUN_CONFIG = 'all_configs.json'  # readable copy of the run configuration
RUN_STATS = 'dataset_stats.pkl'  # readable copy of the normalization statistics


class _StatsUnpickler(pickle.Unpickler):
    """Read NumPy 2 statistics in Isaac Sim's pinned NumPy 1.26.0 environment."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError as exc:
            if exc.name == 'numpy._core' and (module == 'numpy._core' or module.startswith('numpy._core.')):
                return super().find_class('numpy.core' + module[len('numpy._core'):], name)
            raise


def load_snapshot(run_dir):
    """The snapshot is authoritative; readable exports are never used for resume."""
    import torch

    snapshot = torch.load(Path(run_dir) / SNAPSHOT, map_location='cpu', weights_only=True)
    if not isinstance(snapshot, dict) or not all(
            isinstance(snapshot.get(key), dict) for key in ('config', 'norm_stats')):
        raise ValueError(f'{run_dir}/{SNAPSHOT}: missing embedded configuration or normalization statistics. '
                         'Resume requires a checkpoint saved with this metadata; start a new run with '
                         '--pretrained_path pointing to policy_last.ckpt to reuse older weights.')
    return snapshot


def stats_to_tensors(stats):
    """Keep NumPy statistics compatible with torch.load(weights_only=True)."""
    import torch

    if isinstance(stats, dict):
        return {key: stats_to_tensors(value) for key, value in stats.items()}
    return torch.as_tensor(stats).detach().cpu()


def stats_to_numpy(stats):
    """Preserve the existing normalization-statistics format for data and inference."""
    if isinstance(stats, dict):
        return {key: stats_to_numpy(value) for key, value in stats.items()}
    return stats.cpu().numpy()


def export_metadata(snapshot, run_dir):
    """Refresh readable copies from a committed snapshot; either export may safely lag."""
    with _atomic_file(Path(run_dir) / RUN_CONFIG, 'w') as stream:
        json.dump({**snapshot['config'], 'completed_steps': snapshot['step'] + 1}, stream, indent=4)
    with _atomic_file(Path(run_dir) / RUN_STATS, 'wb') as stream:
        pickle.dump(stats_to_numpy(snapshot['norm_stats']), stream)


def load_run(run_dir, checkpoint):
    """Weights, configuration and NumPy normalization statistics of a run, for inference.

    A training snapshot carries its own metadata; any other checkpoint holds weights only
    and is read together with the exports next to it.
    """
    import torch

    run_dir = Path(run_dir)
    weights = torch.load(run_dir / checkpoint, map_location='cpu', weights_only=True)
    if 'model' not in weights:
        with (run_dir / RUN_CONFIG).open() as stream:
            config = json.load(stream)
        with (run_dir / RUN_STATS).open('rb') as stream:
            return weights, config, _StatsUnpickler(stream).load()
    if not all(isinstance(weights.get(key), dict) for key in ('config', 'norm_stats')):
        raise ValueError('Training snapshot needs embedded configuration and normalization statistics')
    return weights['model'], weights['config'], stats_to_numpy(weights['norm_stats'])


def check_same_model(saved, current):
    """A resumed run must keep its model settings, including token order."""
    same = saved == current and all(
        list(config['tokenizer']) == list(saved['embodiment_args'][embodiment]['tokenizer'])
        for embodiment, config in current['embodiment_args'].items())
    if not same:
        raise ValueError('The model settings differ from the saved run. Restore them to resume, '
                         'or start a new run (a new --task_name, with --pretrained_path to reuse its weights).')


def resume_model_config(saved, current):
    """Validate active configs and retain the saved order, including legacy unused entries."""
    active = {emb: cfg for emb, cfg in saved['embodiment_args'].items()
              if emb in current['embodiment_args']}
    check_same_model({**saved, 'embodiment_args': active}, current)
    return saved


@contextmanager
def _atomic_file(path, mode):
    temporary = f'{path}.tmp-{os.getpid()}'
    try:
        with open(temporary, mode) as stream:
            yield stream
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_save(value, path):
    import torch

    with _atomic_file(path, 'wb') as stream:
        torch.save(value, stream)
