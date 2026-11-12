"""Discover episode embodiments without loading arrays or importing PyTorch."""
from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path


@dataclass
class EpisodeIndex:
    groups: list[list[str]]  # Original traversal order, retained for normalization.
    embodiments: dict[str, str]
    group_indices: list[int]  # Positions in the supplied dataset directories/weights.
    explicit_splits: bool = False


def find_all_hdf5(dataset_dir):
    if not Path(dataset_dir).is_dir():
        raise ValueError(f'{dataset_dir}: dataset directory does not exist')

    def raise_error(error):
        raise error

    return [os.path.join(root, name)
            for root, _, names in os.walk(dataset_dir, followlinks=True, onerror=raise_error)
            for name in fnmatch.filter(names, '*.hdf5')]


def read_episode_embodiment(root, default=None):
    """An explicit fallback labels legacy files only when their attribute is absent."""
    value = root.attrs.get('embodiment', default)
    if value is None:
        raise ValueError(f'{root.filename}: missing embodiment attribute; '
                         'supply --default_embodiment for legacy data')
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError(f'{root.filename}: embodiment ID must be UTF-8') from error
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or value in ('.', '..') or '/' in value or '\\' in value):
        raise ValueError(f'{root.filename}: invalid embodiment ID {value!r}')
    return value


def discover_episodes(dataset_dirs, robots=None, default_embodiment=None):
    """Read IDs once, optionally filter episodes, and preserve sampling-group order."""
    import h5py

    selected = set(robots) if robots else None
    groups, embodiments, group_indices, split_groups = [], {}, [], []
    for index, folder in enumerate(dataset_dirs):
        paths = find_all_hdf5(folder)
        if not paths:
            raise ValueError(f'{folder}: no HDF5 episodes found')
        kept = []
        for path in paths:
            with h5py.File(path, 'r') as root:
                embodiment = read_episode_embodiment(root, default_embodiment)
            if selected is None or embodiment in selected:
                kept.append(path)
                embodiments[path] = embodiment
        if kept:
            groups.append(kept)
            group_indices.append(index)
            split_groups.append(all(any(split in os.path.normpath(path).split(os.sep) for path in paths)
                                    for split in ('train', 'val')))
    if not groups:
        raise ValueError('No HDF5 episodes selected; check dataset directories and --robots')
    return EpisodeIndex(groups, embodiments, group_indices, explicit_splits=all(split_groups))
