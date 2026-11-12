"""Load a current EgoX training run and predict or execute physical-unit action chunks.

Observations use the dataset's signal names: RGB uint8 HWC images, float vectors,
and optional boolean availability masks. Robot/environment adapters own coordinates
and masks for padded channels; this module owns resizing and model execution.
"""
from contextlib import nullcontext
from copy import deepcopy

import numpy as np


class MXTPolicy:
    """Inference from policy weights plus exports, or a metadata-bearing training snapshot.

    `embodiment` may be omitted when the run trained a single robot.
    `inference_interval` sets how many chunk rows `act()` executes before replanning;
    it defaults to the full action horizon. Call `reset()` before each new episode.
    """

    def __init__(self, model_path, checkpoint='policy_last.ckpt', embodiment=None, device='cuda:0',
                 inference_interval=None):
        import torch
        from egox.models.mxt.model import MXT
        from egox.training.checkpoints import load_run

        weights, config, stats = load_run(model_path, checkpoint)
        saved = config['policy_config']
        if set(saved) != {'embodiment_args', 'trunk_args', 'image_sharing'}:
            raise ValueError('Expected the current resolved EgoX policy configuration')
        if embodiment is None:
            if len(saved['embodiment_args']) != 1:
                raise ValueError(f'This run trained several robots; choose one: {sorted(saved["embodiment_args"])}')
            embodiment = next(iter(saved['embodiment_args']))
        if embodiment not in saved['embodiment_args'] or embodiment not in stats:
            raise ValueError(f'{embodiment}: not present in this trained policy and its statistics')
        self.config, self.embodiment, self.device = config, embodiment, device
        self.tokenizers = saved['embodiment_args'][embodiment]['tokenizer']
        self.heads = saved['embodiment_args'][embodiment]['action_head']
        self.horizon = min(head['action_horizon'] for head in self.heads.values())
        self.inference_interval = self.horizon if inference_interval is None else inference_interval
        if not isinstance(self.inference_interval, int) or not 1 <= self.inference_interval <= self.horizon:
            raise ValueError(f'inference interval {self.inference_interval} must be within [1, {self.horizon}]')
        self.reset()
        self.image_sizes = {camera: (config[f'{camera}_cam_height'], config[f'{camera}_cam_width'])
                            for camera in ('main', 'wrist')}
        if any(not all(isinstance(v, int) and v > 0 for v in size) for size in self.image_sizes.values()):
            raise ValueError('Saved camera dimensions must be positive integers')
        # Every parameter is restored below; initialization must not download ImageNet weights.
        model_config = deepcopy(saved)
        for spec in model_config['embodiment_args'].values():
            for key, tokenizer in spec['tokenizer'].items():
                if 'image' in key:
                    tokenizer.update(weights=None, gradient_checkpoint=False)
        self.policy = MXT(**model_config, norm_stats=stats)
        self.policy.load_state_dict(weights, strict=True)
        self.policy.to(device).eval()
        self.bf16 = bool(config.get('bf16')) and torch.device(device).type == 'cuda'

    def _prepare_observation(self, observation):
        """Prepare CHW images, proprioception and availability masks for this policy.

        Every configured camera must be present and available. Omitted masks
        default to available; proprioception masks still describe padded channels.
        """
        from egox.data.data import resize_image

        values, availability = observation['obs'], observation.get('obs_mask', {})
        obs, masks = {}, {}
        for key, spec in self.tokenizers.items():
            if 'image' in key:
                size = self.image_sizes['wrist' if 'wrist' in key else 'main']
                value = values.get(key)
                if value is None:
                    raise ValueError(f'{key}: missing required camera for rollout')
                mask = np.asarray(availability.get(key, True), dtype=bool)
                if mask.size != 1:
                    raise ValueError(f'{key}: expected one camera availability flag')
                if not mask.item():
                    raise ValueError(f'{key}: camera unavailable for rollout (obs_mask is False)')
                value = np.asarray(value)
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                    raise ValueError(f'{key}: expected RGB uint8 image with shape (H, W, 3)')
                value = resize_image(value, size)
                obs[key] = np.ascontiguousarray(value.transpose(2, 0, 1))
                masks[key] = np.asarray(True, dtype=bool)
            elif 'proprio' in key:
                if key not in values:
                    raise ValueError(f'{key}: missing required proprioception')
                value = np.asarray(values[key], dtype=np.float32)
                if value.shape != (spec['input_dim'],):
                    raise ValueError(f'{key}: expected shape ({spec["input_dim"]},), got {value.shape}')
                mask = np.asarray(availability.get(key, np.ones(value.shape, dtype=bool)), dtype=bool)
                if mask.shape != value.shape:
                    raise ValueError(f'{key}: mask shape {mask.shape} does not match {value.shape}')
                if not np.isfinite(value[mask]).all():
                    raise ValueError(f'{key}: available proprioception must be finite')
                obs[key], masks[key] = np.ascontiguousarray(value), mask
        return obs, masks

    def __call__(self, observation):
        """Predict a complete action chunk without changing the execution cursor."""
        import torch

        obs, masks = self._prepare_observation(observation)
        obs = {key: torch.from_numpy(value).unsqueeze(0).to(self.device) for key, value in obs.items()}
        masks = {key: torch.from_numpy(value).unsqueeze(0).to(self.device) for key, value in masks.items()}
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.bf16 else nullcontext()
        with torch.inference_mode(), amp:
            result = self.policy(obs, self.embodiment, masks)
        chunks = {key: value[0].float().cpu().numpy() for key, value in result.items()}
        for key, head in self.heads.items():
            if key not in chunks or chunks[key].shape != (head['action_horizon'], head['output_dim']):
                raise ValueError(f'{key}: policy returned an invalid action chunk')
            if not np.isfinite(chunks[key]).all():
                raise ValueError(f'{key}: policy returned non-finite actions')
        return chunks

    def reset(self):
        """Discard the previous episode's action chunk."""
        self._step, self._chunk = 0, None

    def act(self, observation):
        """Return one action row, predicting a new chunk at the configured interval."""
        row = self._step % self.inference_interval
        if row == 0:
            self._chunk = self(observation)
        self._step += 1
        return {key: value[row] for key, value in self._chunk.items()}
