"""Resumable randomness: batch sampling and global RNG state.

Batch `i` is drawn from an RNG seeded by (seed, stream, i), so the sampler's whole
state is the number of batches the trainer has consumed. Resuming at that cursor
reproduces the data sequence regardless of how far DataLoader workers prefetched.
"""
import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def _batch_seed(seed, stream, batch):
    value = json.dumps([int(seed), str(stream), int(batch)]).encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), 'little')


class ResumableBatchSampler:
    """Infinite sampler of timestep indices.

    By default each sample picks a dataset group (with `sample_weights`), then a
    timestep in it. With `episode_embodiments`, batches instead cycle through the
    embodiments (sorted) and draw all samples from one. `timestep_weights` reweights
    timesteps within a group or embodiment.
    """

    def __init__(self, batch_size, episode_len_l, seed, stream, sample_weights=None,
                 timestep_weights=None, episode_embodiments=None):
        self.batch_size, self.seed, self.stream = int(batch_size), int(seed), str(stream)
        self.next_batch = 0
        lengths = np.array([n for group in episode_len_l for n in group], dtype=np.int64)
        bounds = np.cumsum([0] + [sum(group) for group in episode_len_l])
        weights = None if timestep_weights is None else np.asarray(timestep_weights, dtype=np.float64)
        self.per_embodiment = episode_embodiments is not None
        self.group_probs = None
        self.strata = []  # (timestep indices, cumulative sampling probabilities or None)
        if self.per_embodiment:
            starts = np.cumsum(np.r_[0, lengths])
            for emb in sorted(set(episode_embodiments)):
                episodes = [i for i, e in enumerate(episode_embodiments) if e == emb]
                indices = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in episodes])
                self.strata.append((indices, self._cdf(None if weights is None else weights[indices])))
        else:
            if sample_weights is not None:
                self.group_probs = np.asarray(sample_weights, dtype=np.float64)
                self.group_probs = self.group_probs / self.group_probs.sum()
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                self.strata.append((np.arange(lo, hi), self._cdf(None if weights is None else weights[lo:hi])))

    @staticmethod
    def _cdf(weights):
        if weights is None:
            return None
        cdf = np.cumsum(weights / weights.sum())
        cdf[-1] = 1.0
        return cdf

    def __iter__(self):
        batch = self.next_batch
        while True:
            yield self.batch(batch)
            batch += 1

    def batch(self, number):
        rng = np.random.default_rng(_batch_seed(self.seed, self.stream, number))
        if self.per_embodiment:
            groups = [number % len(self.strata)] * self.batch_size
        else:
            groups = rng.choice(len(self.strata), size=self.batch_size, p=self.group_probs)
        indices = []
        for group in groups:
            timesteps, cdf = self.strata[group]
            position = (rng.integers(len(timesteps)) if cdf is None
                        else np.searchsorted(cdf, rng.random(), side='right'))
            indices.append(int(timesteps[position]))
            # Unused draw (an augmentation seed in the original sampler); keeps the original
            # batch sequence for a given seed.
            rng.integers(0, 2 ** 63, dtype=np.int64)
        return indices


class ResumableDataLoader(DataLoader):
    """DataLoader that advances its sampler's cursor only when a batch is delivered.

    A new iteration (e.g. each validation pass) continues after the last delivered batch.
    """

    def __iter__(self):
        for batch in super().__iter__():
            self.batch_sampler.next_batch += 1
            yield batch

    def state_dict(self):
        return {'next_batch': self.batch_sampler.next_batch}

    def load_state_dict(self, state):
        self.batch_sampler.next_batch = state['next_batch']


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    algorithm, keys, *rest = state['numpy']
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), *rest))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'].cpu())
