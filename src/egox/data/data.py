"""Shared image preparation and episodic datasets for multi-embodiment training.

Each episode file has an `embodiment` attribute; observation/action fields are
read through KEY_PATH_MAP. A sample holds the current observation, the action
chunk and, for MXT+, the observations and availability masks at future `dream_offsets`.
"""
import os

import cv2
import numpy as np
import torch

from .sampler import ResumableBatchSampler, ResumableDataLoader
from .episodes import discover_episodes, read_episode_embodiment
from .mapping import KEY_MASK_PATH_MAP, KEY_PATH_MAP, embodiment_keys


def resize_image(image, size):
    """RGB uint8 (H, W, 3) at `size` (height, width); area interpolation when it must change."""
    if image.shape[:2] == tuple(size):
        return image
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def blank_image(size):
    """Black frame standing in for an unavailable camera."""
    return np.zeros((*size, 3), dtype=np.uint8)


def get_episode_metadata(paths, emb_dict, episode_embodiments=None):
    """Episode lengths (of the first action field) and embodiments, without reading data."""
    import h5py

    lengths, embodiments = [], []
    for path in paths:
        with h5py.File(path, 'r') as root:
            embodiment = (episode_embodiments[path] if episode_embodiments is not None
                          else read_episode_embodiment(root))
            lengths.append(int(root[KEY_PATH_MAP[embodiment_keys(emb_dict, embodiment)[1][0]]].shape[0]))
            embodiments.append(embodiment)
    return lengths, embodiments


def get_norm_stats(paths, emb_dict, episode_embodiments=None):
    """Per-embodiment mean/std of every proprioception and action field."""
    import h5py

    data = {}
    for path in paths:
        with h5py.File(path, 'r') as root:
            embodiment = (episode_embodiments[path] if episode_embodiments is not None
                          else read_episode_embodiment(root))
            if embodiment not in data:
                obs_keys, action_keys, _ = embodiment_keys(emb_dict, embodiment)
                data[embodiment] = {'obs': {k: [] for k in obs_keys if 'image' not in k},
                                    'action': {k: [] for k in action_keys}}
            for kind in ('obs', 'action'):
                for key, values in data[embodiment][kind].items():
                    values.append(root[KEY_PATH_MAP[key]][()])
    stats = {}
    for embodiment, kinds in data.items():
        stats[embodiment] = {kind: {} for kind in kinds}
        for kind, fields in kinds.items():
            for key, values in fields.items():
                values = np.concatenate(values, axis=0)
                stats[embodiment][kind][key] = {'mean': np.mean(values, axis=0), 'std': np.std(values, axis=0)}
        print(f'Embodiment {embodiment}: {len(values)} steps')
    return stats


def velocity_rebalance_weights(paths, lengths, x=1.0, y=1.0, yaw=5.0, threshold=0.1):
    """Per-timestep sampling weights 1 + x*[|vx|>t] + y*[|vy|>t] + yaw*[|wz|>t] from velocity commands."""
    import h5py

    weights = np.ones(sum(lengths), dtype=np.float64)
    offset = 0
    for path, length in zip(paths, lengths):
        with h5py.File(path, 'r') as f:
            if '/actions/velocity_commands' in f:
                vel = np.array(f['/actions/velocity_commands'])
                n = min(length, len(vel))
                active = np.abs(vel[:n, :3]) > threshold
                weights[offset:offset + n] += active[:, 0] * x + active[:, 1] * y + active[:, 2] * yaw
        offset += length
    print(f'[vel_rebalance] {(weights > 1).sum()}/{len(weights)} timesteps up-weighted, mean weight {weights.mean():.2f}')
    return weights


class EpisodicDataset(torch.utils.data.Dataset):
    """Index -> (sample dict, embodiment) over the concatenated timesteps of the given episodes."""

    def __init__(self, paths, episode_ids, episode_len, chunk_size, emb_dict,
                 main_cam_size=None, wrist_cam_size=None, episode_embodiments=None):
        super().__init__()
        self.paths = np.array(paths, dtype=np.bytes_)
        self.episode_ids = episode_ids
        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(episode_len)
        self.chunk_size = chunk_size
        self.emb_dict = emb_dict
        self.main_cam_size = main_cam_size
        self.wrist_cam_size = wrist_cam_size
        self.episode_embodiments = episode_embodiments

    def __len__(self):
        return int(self.cumulative_len[-1])

    def _decode_image(self, data, compressed, key):
        image = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if compressed else data
        size = self.wrist_cam_size if 'wrist' in key else self.main_cam_size
        return image if size is None else resize_image(image, size)

    def _blank_image(self, key):
        """Black frame for a camera the episode did not record (its mask is False)."""
        size = self.wrist_cam_size if 'wrist' in key else self.main_cam_size
        if size is None:
            raise ValueError(f'{key} is not recorded in every episode; set its camera size to use blank frames')
        return blank_image(size)

    def __getitem__(self, index):
        import h5py

        episode = np.argmax(self.cumulative_len > index)
        start = index - (self.cumulative_len[episode] - self.episode_len[episode])
        path = str(self.paths[self.episode_ids[episode]], encoding='utf-8')
        with h5py.File(path, 'r') as root:
            compressed = root.attrs.get('compress', False)
            embodiment = (self.episode_embodiments[path] if self.episode_embodiments is not None
                          else read_episode_embodiment(root))
            episode_len = self.episode_len[episode]
            obs_keys, action_keys, dream_keys = embodiment_keys(self.emb_dict, embodiment)

            def mask_or_default(key, default):
                path = KEY_MASK_PATH_MAP.get(key)
                if path is not None and path in root:
                    return torch.tensor(root[path][0] if 'image' in key else root[path][()])
                return default

            obs, obs_mask, action, action_mask = {}, {}, {}, {}
            for key in obs_keys:
                if 'image' in key and KEY_PATH_MAP[key] not in root:  # camera not recorded in this episode
                    obs[key] = self._blank_image(key)
                    obs_mask[key] = torch.tensor(False)
                elif 'image' in key:
                    obs[key] = root[KEY_PATH_MAP[key]][start]
                    if compressed:  # (uncompressed observation frames are used at their stored size)
                        obs[key] = self._decode_image(obs[key], compressed, key)
                    obs_mask[key] = mask_or_default(key, torch.tensor(True))
                else:
                    obs[key] = torch.from_numpy(root[KEY_PATH_MAP[key]][start]).float()
                    obs_mask[key] = mask_or_default(key, torch.ones(obs[key].shape[-1], dtype=torch.bool))
            for key in action_keys:
                chunk = torch.from_numpy(root[KEY_PATH_MAP[key]][start:start + self.chunk_size]).float()
                fill = min(episode_len - start, self.chunk_size)
                action[key] = torch.zeros(self.chunk_size, chunk.shape[-1])
                action[key][:fill] = chunk[:fill]
                action_mask[key] = mask_or_default(key, torch.ones(chunk.shape[-1], dtype=torch.bool))

            dream, dream_mask = {}, {}
            if dream_keys:
                offsets = np.asarray(self.emb_dict[embodiment]['embodiment_dream_head']['dream_offsets'])
                indices = start + offsets
                dream_pad = torch.from_numpy(indices >= episode_len)
                indices = np.minimum(indices, episode_len - 1)
                for key in dream_keys:
                    if 'image' in key and KEY_PATH_MAP[key] not in root:
                        dream[key] = np.stack([self._blank_image(key) for _ in indices])
                    else:
                        frames = [root[KEY_PATH_MAP[key]][i] for i in indices]
                        if 'image' in key:
                            dream[key] = np.stack([self._decode_image(f, compressed, key) for f in frames])
                        else:
                            dream[key] = torch.from_numpy(np.stack(frames)).float()
                    # Current HDF5 masks are static per episode; use the same defaults as observations.
                    mask = obs_mask[key].bool()
                    dream_mask[key] = mask.unsqueeze(0).expand(len(indices), *mask.shape).clone()

        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        is_pad[episode_len - start:] = True
        for images in (obs, dream):
            for key in [k for k in images if 'image' in k]:
                images[key] = torch.from_numpy(images[key]).movedim(-1, -3).contiguous()  # uint8 (..., C, H, W)
        sample = {'obs': obs, 'obs_mask': obs_mask, 'action': action, 'action_mask': action_mask, 'is_pad': is_pad}
        if dream:
            sample['dream'] = dream
            sample['dream_mask'] = dream_mask
            sample['dream_pad'] = dream_pad
        return sample, embodiment


def collate_by_embodiment(samples):
    """Batch samples separately per embodiment: {embodiment: batched sample}."""
    def stack(items):
        if isinstance(items[0], dict):
            return {key: stack([item[key] for item in items]) for key in items[0]}
        return torch.stack(items)
    embodiments = np.array([emb for _, emb in samples], dtype=np.str_)
    return {emb: stack([sample for sample, e in samples if e == emb]) for emb in np.unique(embodiments)}


def load_data(dataset_dirs, emb_dict, batch_size, chunk_size, seed, per_emb_batch_size=None,
              sample_weights=None, train_ratio=0.99, min_val_num=1, main_cam_size=None,
              wrist_cam_size=None, vel_rebalance=None, norm_stats=None, episode_index=None):
    """Build resumable train/validation loaders.

    Each directory in `dataset_dirs` is a sampling group (weighted by `sample_weights`).
    Episodes under `train/` and `val/` folders define the split if every group has
    both; otherwise the last episodes of the first group are held out. With
    `per_emb_batch_size`, each batch holds that many samples of a single embodiment
    (round-robin). `norm_stats` (e.g. from a run being resumed) skips recomputation.
    `episode_index` reuses discovery and applies the same selection everywhere.
    """
    episode_index = episode_index if episode_index is not None else discover_episodes(dataset_dirs)
    groups = [sorted(group) for group in episode_index.groups]
    paths = [p for group in groups for p in group]
    for path, embodiment in episode_index.embodiments.items():
        if embodiment not in emb_dict:
            raise ValueError(f'{path}: no robot configuration for embodiment {embodiment!r}')
    if sample_weights is not None:
        if len(sample_weights) != len(dataset_dirs):
            raise ValueError('sample_weights needs one weight per dataset directory')
        sample_weights = [sample_weights[i] for i in episode_index.group_indices]
        if any(weight < 0 for weight in sample_weights) or sum(sample_weights) <= 0:
            raise ValueError('Selected dataset groups need nonnegative weights with a positive total')
    if norm_stats is not None and set(norm_stats) != set(episode_index.embodiments.values()):
        raise ValueError('Resuming requires the same dataset embodiments as the saved normalization statistics')

    def in_split(path, split):
        return split in os.path.normpath(path).split(os.sep)

    if episode_index.explicit_splits:
        print('Using explicit train/val folders for the dataset split')
        train_ids_l, val_ids_l, start = [], [], 0
        for group in groups:
            train_ids_l.append(np.array([start + i for i, p in enumerate(group) if in_split(p, 'train')], dtype=np.int64))
            val_ids_l.append(np.array([start + i for i, p in enumerate(group) if in_split(p, 'val')], dtype=np.int64))
            start += len(group)
    else:
        num_val = max(min_val_num, int((1 - train_ratio) * len(groups[0])))
        ids = np.arange(len(groups[0]))
        offsets = np.cumsum([len(g) for g in groups])
        train_ids_l = [ids[:-num_val]] + [np.arange(len(g)) + offsets[i] for i, g in enumerate(groups[1:])]
        val_ids_l = [ids[-num_val:]]
    train_ids, val_ids = np.concatenate(train_ids_l), np.concatenate(val_ids_l)
    if any(not len(ids) for ids in train_ids_l + val_ids_l):
        raise ValueError('Selected data leaves an empty train or validation split; '
                         'keep episodes in both splits or adjust --robots')
    print(f'Train on {[len(x) for x in train_ids_l]} episodes, validate on {[len(x) for x in val_ids_l]}')

    lengths, episode_embodiments = get_episode_metadata(paths, emb_dict, episode_index.embodiments)
    embodiments = list(dict.fromkeys(episode_embodiments))
    if norm_stats is None:
        # Statistics use the unsorted directory walk, as in the original reduction order.
        norm_stats = get_norm_stats([p for group in episode_index.groups for p in group],
                                   emb_dict, episode_index.embodiments)
    train_len_l = [[lengths[i] for i in ids] for ids in train_ids_l]
    val_len_l = [[lengths[i] for i in ids] for ids in val_ids_l]
    train_len = [n for g in train_len_l for n in g]
    val_len = [n for g in val_len_l for n in g]

    timestep_weights = None
    if vel_rebalance is not None:
        timestep_weights = velocity_rebalance_weights([paths[i] for i in train_ids], train_len, **vel_rebalance)
    train_sampler = ResumableBatchSampler(
        per_emb_batch_size or batch_size, train_len_l, seed=seed, stream='train',
        sample_weights=sample_weights, timestep_weights=timestep_weights,
        episode_embodiments=[episode_embodiments[i] for i in train_ids] if per_emb_batch_size else None)
    val_sampler = ResumableBatchSampler(batch_size, val_len_l, seed=seed, stream='val')

    train_set = EpisodicDataset(paths, train_ids, train_len, chunk_size, emb_dict,
                               main_cam_size, wrist_cam_size, episode_index.embodiments)
    val_set = EpisodicDataset(paths, val_ids, val_len, chunk_size, emb_dict,
                             main_cam_size, wrist_cam_size, episode_index.embodiments)
    env = lambda name, default: int(os.environ.get(name, default))
    train_workers = env('MXT_TRAIN_NUM_WORKERS', env('MXT_NUM_WORKERS', 8))
    val_workers = env('MXT_VAL_NUM_WORKERS', env('MXT_NUM_WORKERS', 8))

    def loader(dataset, sampler, workers, prefetch):
        return ResumableDataLoader(dataset, batch_sampler=sampler, pin_memory=True, num_workers=workers,
                                   collate_fn=collate_by_embodiment, persistent_workers=workers > 0,
                                   prefetch_factor=prefetch if workers > 0 else None)

    return (loader(train_set, train_sampler, train_workers, env('MXT_TRAIN_PREFETCH_FACTOR', 2)),
            loader(val_set, val_sampler, val_workers, env('MXT_VAL_PREFETCH_FACTOR', 1)),
            norm_stats, embodiments)
