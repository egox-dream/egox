"""MXT: a cross-embodiment transformer policy, with MXT+ embodiment dreaming.

Per embodiment, every observation modality is encoded and pooled into a few
tokens. A shared encoder-decoder trunk turns the tokens into one block of
decoded queries per action field, and per-embodiment heads predict action chunks.

Embodiment dreaming (MXT+): an embodiment tokenizer summarizes the current
observation into a few tokens; a dream head predicts, from the trunk output, the
summary tokens of future frames as computed by an EMA copy of the encoders.
"""
import copy

import torch
from torch import nn

from .decoders import ActionHead, action_loss
from .dreaming import EmbodimentTokenizer, DreamHead, dream_loss
from .tokenizers import ImageTokenizer, ProprioTokenizer, build_resnet
from .transformer import Trunk, sinusoid_table

EMA_DECAY = 0.999


class Normalizer(nn.Module):
    """Fixed per-signal dataset statistics; buffers are not learned parameters."""

    def __init__(self, stats, persistent=False):
        super().__init__()
        # Own the tensors so clamping/loading buffers cannot modify the supplied statistics.
        std = torch.tensor(stats['std'], dtype=torch.float32).clamp(min=1e-6)
        self.register_buffer('mean', torch.tensor(stats['mean'], dtype=torch.float32), persistent)
        self.register_buffer('std', std, persistent)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean


def add_positions(features):
    """Add sinusoidal positions along the token axis of (B, N, D) features."""
    return features + sinusoid_table(features.shape[1], features.shape[-1]).to(features.device)


def to_float(obs):
    """uint8 images -> [0, 1] in the autocast dtype (if active); everything else -> float32."""
    out = {}
    for key, value in obs.items():
        if value.dtype == torch.uint8 and 'image' in key:
            device = value.device.type
            dtype = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else torch.float32
            out[key] = value.to(dtype).div_(255.0)
        else:
            out[key] = value.float()
    return out


def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    return data.to(device, non_blocking=True) if isinstance(data, torch.Tensor) else data


class MXT(nn.Module):
    """
    All tokens have the trunk's `hidden_dim`.

    Args:
        embodiment_args: resolved configs from `egox.configs.loader.build_model_config`:
            per-embodiment `tokenizer`, `action_head` and optional `embodiment_dream_head`.
            The loader expands robot YAML defaults and derives all dimensions from the
            signal shapes and trunk width; the model only constructs the modules.
        trunk_args: Trunk config; `output_len` gives the query budget of each action field.
        norm_stats: per-embodiment dataset statistics; only embodiments present here are built.
        image_sharing: 'none' (private image tokenizers), 'encoder' (one ResNet per camera
            shared across embodiments, with private projections and queries), or 'tokenizer'
            (the full image tokenizer, including projection and query pooling, shared per camera).
    """

    def __init__(self, embodiment_args, trunk_args, norm_stats, image_sharing='none'):
        super().__init__()
        if image_sharing not in ('none', 'encoder', 'tokenizer'):
            raise ValueError(f"image_sharing must be 'none', 'encoder' or 'tokenizer', got {image_sharing!r}")
        self.image_sharing = image_sharing
        self.hidden_dim = trunk_args['hidden_dim']
        # Resumed runs can retain configs for robots absent from the current dataset.
        self.embodiment_args = {emb: cfg for emb, cfg in embodiment_args.items()
                                if emb in norm_stats}
        self.modalities = {emb: tuple(key for key in cfg['tokenizer'] if 'image' in key or 'proprio' in key)
                           for emb, cfg in self.embodiment_args.items()}
        self.image_encoders = nn.ModuleDict()
        self.image_encoder_specs = {}
        self.image_tokenizers = nn.ModuleDict()
        self.image_tokenizer_specs = {}
        self.tokenizers = nn.ModuleDict()
        self.embodiment_tokenizers = nn.ModuleDict()
        self.action_heads = nn.ModuleDict()
        self.dream_heads = nn.ModuleDict()
        # Current and future proprioception share each signal's dataset statistics.
        # Statistics are saved separately from model weights in dataset_stats.pkl.
        self.normalizers = nn.ModuleDict()
        for emb, cfg in self.embodiment_args.items():
            stats = norm_stats[emb]
            tok_cfg = cfg['tokenizer']
            self.tokenizers[emb] = nn.ModuleDict()
            for key in self.modalities[emb]:
                registry = (self.image_tokenizers if self.image_sharing == 'tokenizer' and 'image' in key
                            else self.tokenizers[emb])
                registry[key] = self._build_tokenizer(key, tok_cfg)
            if 'embodiment_tokenizer' in tok_cfg:
                summary_cfg = {name: tok_cfg[name] for name in ('hidden_dim', 'nhead', 'dim_head', 'dropout')}
                summary_cfg.update(tok_cfg['embodiment_tokenizer'])
                self.embodiment_tokenizers[emb] = EmbodimentTokenizer(**summary_cfg)
            self.action_heads[emb] = nn.ModuleDict()
            for field, head in cfg['action_head'].items():
                self.action_heads[emb][field] = ActionHead(**head)
            self.normalizers[emb] = nn.ModuleDict(
                {**{key: Normalizer(stats['obs'][key]) for key in self.modalities[emb] if 'proprio' in key},
                 **{key: Normalizer(stats['action'][key]) for key in cfg['action_head']}})
            if 'embodiment_dream_head' in cfg:
                dream_cfg = cfg['embodiment_dream_head']
                summary = self.embodiment_tokenizers[emb]
                self.dream_heads[emb] = DreamHead(
                    num_tokens=len(dream_cfg['dream_offsets']) * summary.token_num, **dream_cfg)
        self.trunk = Trunk(**trunk_args)
        # Query order comes from the shared trunk, independently of each robot's head order.
        self.action_fields = {emb: tuple(field for field in self.trunk.query_slices if field in heads)
                              for emb, heads in self.action_heads.items()}
        if self.dream_heads:
            self.ema = copy.deepcopy(self._dream_encoders()).requires_grad_(False)
            for summary in self.ema['embodiment_tokenizers'].values():
                summary.norm_upper_bound_weight = 0.0

    def _build_tokenizer(self, key, tok_cfg):
        cfg = tok_cfg[key]
        query_cfg = {name: cfg.get(name, tok_cfg[name])
                     for name in ('hidden_dim', 'nhead', 'dim_head', 'dropout')}
        if 'image' in key:
            if self.image_sharing == 'tokenizer':
                tokenizer_cfg = dict(resnet_model='resnet18', weights='DEFAULT', split_dual=False,
                                     feature_norm=True, gradient_checkpoint=False)
                tokenizer_cfg.update(cfg)
                tokenizer_cfg.update(query_cfg)
                if key in self.image_tokenizers:
                    if self.image_tokenizer_specs[key] != tokenizer_cfg:
                        raise ValueError(f'Shared image tokenizer {key!r} needs the same configuration everywhere')
                    return self.image_tokenizers[key]
                self.image_tokenizer_specs[key] = tokenizer_cfg
            shared = None
            if self.image_sharing == 'encoder':
                encoder_cfg = dict(resnet_model=cfg.get('resnet_model', 'resnet18'),
                                   weights=cfg.get('weights', 'DEFAULT'))
                if key not in self.image_encoders:
                    self.image_encoders[key], dim = build_resnet(**encoder_cfg)
                    self.image_encoder_specs[key] = (encoder_cfg, dim)
                elif self.image_encoder_specs[key][0] != encoder_cfg:
                    raise ValueError(f'Shared image encoder {key!r} needs the same resnet_model/weights everywhere')
                shared = (self.image_encoders[key], self.image_encoder_specs[key][1])
            tokenizer = ImageTokenizer(**cfg, shared_encoder=shared)
        else:
            tokenizer = ProprioTokenizer(**cfg)
        tokenizer.init_queries(cfg['token_num'], **query_cfg)
        return tokenizer

    def _get_tokenizer(self, emb, key, ema=False):
        image_tokenizers = self.ema['image_tokenizers'] if ema else self.image_tokenizers
        tokenizers = self.ema['tokenizers'] if ema else self.tokenizers
        return image_tokenizers[key] if key in image_tokenizers else tokenizers[emb][key]

    # ------------------------------------------------------------------ encoding

    def _features(self, tokenizer, x, mask=None):
        """Positioned features; exclude missing images from ResNet and mask proprio channels."""
        if isinstance(tokenizer, ImageTokenizer):
            if mask is not None:
                present = mask.reshape(x.shape[0], -1).bool().any(dim=1)
                if not present.all():
                    # The caller skips fully absent modalities. Keep placeholders out of BatchNorm.
                    features = add_positions(tokenizer(x[present]))
                    rows = present.nonzero(as_tuple=True)[0]
                    return features.new_zeros((x.shape[0], *features.shape[1:])).index_copy(0, rows, features)
        else:
            x = (x if mask is None else x.masked_fill(~mask.bool(), 0)).unsqueeze(-2)
        return add_positions(tokenizer(x))

    def tokenize(self, obs, obs_mask, emb):
        """Returns tokens, boolean validity (B, N; True = present), and the norm penalty."""
        if not obs:
            raise ValueError(f'{emb}: no valid observations provided')
        sample = next(iter(obs.values()))
        bs, device = sample.shape[0], sample.device
        summary = self.embodiment_tokenizers[emb] if emb in self.embodiment_tokenizers else None
        tokens, masks, summary_feats, summary_masks = [], [], [], []
        for key in self.modalities[emb]:
            tokenizer = self._get_tokenizer(emb, key)
            if key not in obs:
                tokens.append(torch.zeros(bs, tokenizer.token_num, tokenizer.hidden_dim, device=device))
                masks.append(torch.zeros(bs, tokenizer.token_num, dtype=torch.bool, device=device))
                continue
            if not torch.any(obs_mask[key]):
                token, mask = tokenizer.compute_latent(None, obs_mask[key])
            else:
                feat = self._features(tokenizer, obs[key], obs_mask[key])
                token, mask = tokenizer.compute_latent(feat, obs_mask[key])
                if summary is not None and key in summary.modalities:
                    summary_feats.append(feat)
                    present = obs_mask[key].reshape(bs, -1).any(dim=1)
                    summary_masks.append(present[:, None].expand(-1, feat.shape[1]))
            tokens.append(token)
            masks.append(mask)
        penalty = None
        if summary is not None:
            if summary_feats:
                token, mask, penalty = summary.compute_latent(
                    torch.cat(summary_feats, dim=1), torch.cat(summary_masks, dim=1))
            else:
                token, mask, _ = summary.compute_latent(None, None, bs=bs, device=device)
            tokens.insert(0, token)
            masks.insert(0, mask)
        return tokens, torch.cat(masks, dim=1), penalty

    def encode(self, obs, obs_mask, emb):
        """Returns decoded trunk queries per action field, the raw trunk output and the norm penalty."""
        obs = to_float(obs)
        obs = {k: self.normalizers[emb][k].normalize(v) if 'proprio' in k else v for k, v in obs.items()}
        tokens, mask, penalty = self.tokenize(obs, obs_mask, emb)
        tokens = add_positions(torch.cat(tokens, dim=-2))
        fields = self.action_fields[emb]
        trunk_out = self.trunk(tokens, mask, fields)
        features, start = {}, 0
        for field in fields:
            n = self.trunk.query_slices[field].stop - self.trunk.query_slices[field].start
            features[field] = trunk_out[:, start:start + n]
            start += n
        return features, trunk_out, penalty

    # ------------------------------------------------------------------ dreaming

    def _dream_encoders(self):
        """The modules whose EMA copy computes dream targets (built fresh; not registered)."""
        cameras = {key for emb in self.dream_heads for key in self.embodiment_tokenizers[emb].modalities}
        return nn.ModuleDict({
            'image_encoders': nn.ModuleDict({key: encoder for key, encoder in self.image_encoders.items()
                                             if key in cameras}),
            'image_tokenizers': nn.ModuleDict({key: tokenizer for key, tokenizer in self.image_tokenizers.items()
                                               if key in cameras}),
            'tokenizers': nn.ModuleDict({
                emb: nn.ModuleDict({key: self.tokenizers[emb][key]
                                    for key in self.embodiment_tokenizers[emb].modalities
                                    if key not in self.image_tokenizers})
                for emb in self.dream_heads}),
            'embodiment_tokenizers': nn.ModuleDict({emb: self.embodiment_tokenizers[emb]
                                                    for emb in self.dream_heads}),
        })

    @torch.no_grad()
    def update_ema(self, decay=EMA_DECAY):
        """EMA-update the target encoders after an optimizer step; buffers (BatchNorm
        statistics) are copied. `decay=None` copies buffers only."""
        if not self.dream_heads:
            return
        student = self._dream_encoders()
        if decay is not None:
            for target, source in zip(self.ema.parameters(), student.parameters()):
                target.mul_(decay).add_(source, alpha=1 - decay)
        for target, source in zip(self.ema.buffers(), student.buffers()):
            target.copy_(source)

    @torch.no_grad()
    def dream_targets(self, dream, emb, dream_mask=None):
        """EMA future tokens and validity; masks are (B, T) for images or (B, T, D) for proprio."""
        self.ema.eval()
        summary = self.ema['embodiment_tokenizers'][emb]
        dream_mask = dream_mask or {}
        sample = next(iter(dream.values()), None)
        if sample is None:
            sample = next(iter(dream_mask.values()), None)
        if sample is None:
            raise ValueError('Future observations or masks must provide batch and time dimensions')
        B, T = sample.shape[:2]
        feats, masks = [], []
        for key in summary.modalities:
            if key not in dream:
                continue
            data = to_float({key: dream[key]})[key]
            mask = dream_mask.get(key)
            if mask is None:
                shape = data.shape[:2] if 'image' in key else data.shape
                mask = torch.ones(shape, dtype=torch.bool, device=data.device)
            mask = mask.reshape(B * T, -1).bool()
            if not mask.any():
                continue
            feat = self._features(self._get_tokenizer(emb, key, ema=True),
                                  data.reshape(B * T, *data.shape[2:]), mask)
            feats.append(feat)
            masks.append(mask.any(dim=1)[:, None].expand(-1, feat.shape[1]))
        if feats:
            target, valid, _ = summary.compute_latent(torch.cat(feats, dim=1), torch.cat(masks, dim=1))
        else:
            target, valid, _ = summary.compute_latent(None, None, bs=B * T, device=sample.device)
        return target.reshape(B, -1, target.shape[-1]), valid.reshape(B, -1)

    # ------------------------------------------------------------------ training / inference

    def compute_loss(self, batch):
        """batch: {embodiment: {obs, obs_mask, action, action_mask, is_pad[, dream, dream_mask, dream_pad]}}."""
        device = self.trunk.query_pos.device
        losses = {'all_emb_loss': 0}
        for emb, b in batch.items():
            b = _to_device(b, device)
            cfg = self.embodiment_args[emb]
            log = losses[emb] = {'loss': 0}
            features, trunk_out, penalty = self.encode(b['obs'], b['obs_mask'], emb)
            if penalty is not None:
                weighted = penalty * self.embodiment_tokenizers[emb].norm_upper_bound_weight
                log['embodiment_norm_upper_bound_penalty'] = penalty.detach()
                log['embodiment_norm_upper_bound_loss'] = weighted
                log['loss'] += weighted
            for field, action in b['action'].items():
                if field not in features:
                    continue
                label = self.normalizers[emb][field].normalize(action)
                pred = self.action_heads[emb][field](features[field]).view(label.shape)
                log[field] = action_loss(pred, label, b['action_mask'][field], b['is_pad']) \
                    * cfg['action_head'][field].get('loss_weight', 1.0)
                log['loss'] += log[field]
            if emb in self.dream_heads and 'dream' in b:
                dream_cfg = cfg['embodiment_dream_head']
                dream = {k: self.normalizers[emb][k].normalize(v) if 'proprio' in k else v
                         for k, v in b['dream'].items()}
                target, valid = self.dream_targets(dream, emb, b.get('dream_mask'))
                pred = self.dream_heads[emb](trunk_out).view(target.shape)
                pad = b['dream_pad'].repeat_interleave(self.embodiment_tokenizers[emb].token_num, dim=1) | ~valid
                loss, direction, magnitude = dream_loss(pred, target, pad, dream_cfg.get('dream_mag_beta', 0.0))
                log['dream_embodiment_dream_cos_direction_loss'] = direction
                log['dream_embodiment_dream_cos_magnitude_loss'] = magnitude
                log['dream_embodiment_dream'] = loss * dream_cfg.get('loss_weight', 1.0)
                log['loss'] += log['dream_embodiment_dream']
            losses['all_emb_loss'] += log['loss']
        return losses

    def forward(self, obs, embodiment, obs_mask, infer_dream=False):
        """Unnormalized action chunks {field: (B, horizon, dim)}; with `infer_dream`, also the
        predicted future embodiment tokens {'embodiment_dream': (B, frames, tokens * D)}."""
        device = self.trunk.query_pos.device
        features, trunk_out, _ = self.encode(_to_device(obs, device), _to_device(obs_mask, device), embodiment)
        cfg = self.embodiment_args[embodiment]
        actions = {}
        for field, head in self.action_heads[embodiment].items():
            action = head(features[field]).view(len(trunk_out), cfg['action_head'][field]['action_horizon'], -1)
            actions[field] = self.normalizers[embodiment][field].unnormalize(action)
        if not (infer_dream and embodiment in self.dream_heads):
            return actions
        frames = len(cfg['embodiment_dream_head']['dream_offsets'])
        return actions, {'embodiment_dream': self.dream_heads[embodiment](trunk_out).view(len(trunk_out), frames, -1)}

    # ------------------------------------------------------------------ optimization / checkpoints

    def get_extra_state(self):
        """Query-bank order has meaning even when two checkpoints have identical shapes."""
        return {'image_sharing': self.image_sharing,
                'output_len': [(field, span.stop - span.start) for field, span in self.trunk.query_slices.items()]}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError('Checkpoint image_sharing and ordered trunk output_len must match the model')

    def param_groups(self, lr_tokenizer, lr_action_head, lr_dream_head, lr_trunk):
        groups = [(self.image_encoders, lr_tokenizer), (self.tokenizers, lr_tokenizer),
                  (self.embodiment_tokenizers, lr_tokenizer), (self.action_heads, lr_action_head),
                  (self.dream_heads, lr_dream_head), (self.trunk, lr_trunk)]
        if self.image_tokenizers:
            groups.insert(1, (self.image_tokenizers, lr_tokenizer))
        groups = [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr} for module, lr in groups]
        assert sum(len(g['params']) for g in groups) == sum(p.requires_grad for p in self.parameters())
        return groups

    def load_pretrained(self, state_dict):
        """Initialize from another run's checkpoint (`model.state_dict()`).

        Matching tokenizers/heads load strictly, new ones keep their initialization,
        and removed ones are skipped. New EMA components copy their online counterparts
        after loading. Image sharing and the trunk query layout must stay unchanged.
        Returns component names grouped as loaded, initialized, ema_initialized, skipped.
        """
        def components(weights):
            # Number of path segments identifying one independently transferable module.
            depths = {'trunk': 1, 'tokenizers': 3, 'action_heads': 3,
                      'image_encoders': 2, 'image_tokenizers': 2,
                      'embodiment_tokenizers': 2, 'dream_heads': 2}
            result = {}
            for key, value in weights.items():
                if key == '_extra_state':
                    continue
                parts = key.split('.')
                offset = int(parts[0] == 'ema')
                depth = depths.get(parts[offset], 0) if len(parts) > offset else 0
                if not depth or len(parts) <= offset + depth:
                    raise RuntimeError(f'Unexpected checkpoint key: {key}')
                name = '.'.join(parts[:offset + depth])
                result.setdefault(name, {})[key] = value
            return result

        self.set_extra_state(state_dict.get('_extra_state'))
        own = self.state_dict()
        current, saved = components(own), components(state_dict)
        if 'trunk' not in saved:
            raise RuntimeError('Checkpoint is missing the trunk')
        common = current.keys() & saved.keys()
        # Check every overlapping component before changing any model weights.
        for name in sorted(common):
            expected, incoming = current[name], saved[name]
            missing, unexpected = expected.keys() - incoming.keys(), incoming.keys() - expected.keys()
            incompatible = [key for key in expected.keys() & incoming.keys()
                            if not isinstance(incoming[key], torch.Tensor)
                            or incoming[key].shape != expected[key].shape]
            if missing or unexpected or incompatible:
                raise RuntimeError(f'Checkpoint mismatch in {name}. Missing: {sorted(missing)}; '
                                   f'unexpected: {sorted(unexpected)}; incompatible shapes: {sorted(incompatible)}')
        for name in common:
            own.update(saved[name])
        added = current.keys() - saved.keys()
        for name in added:
            if name.startswith('ema.'):
                for key in current[name]:
                    own[key] = own[key.removeprefix('ema.')]
        self.load_state_dict(own)
        report = {'loaded': sorted(common),
                  'initialized': sorted(name for name in added if not name.startswith('ema.')),
                  'ema_initialized': sorted(name for name in added if name.startswith('ema.')),
                  'skipped': sorted(saved.keys() - current.keys())}
        print(f'Pretrained: {len(report["loaded"])} loaded, {len(report["initialized"])} initialized, '
              f'{len(report["ema_initialized"])} EMA copied from online, {len(report["skipped"])} skipped')
        return report

