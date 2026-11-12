"""Per-embodiment action decoders and masked action loss (adapted from HPT)."""
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import CrossAttention, sinusoid_table

_smooth_l1 = partial(F.smooth_l1_loss, beta=0.05, reduction='none')


def action_loss(pred, label, mask, is_pad):
    """Smooth-L1 over valid (unpadded timestep, unmasked channel) entries."""
    valid = ~is_pad[:, :label.shape[1]].unsqueeze(-1).bool() & mask.unsqueeze(1).bool()
    return torch.sum(_smooth_l1(pred, label) * valid) / torch.sum(valid).clamp(min=1)


class ActionHead(nn.Module):
    """One learned query per action timestep cross-attends to the field's trunk tokens."""

    def __init__(self, input_dim, output_dim, action_horizon, crossattn_heads=8, crossattn_dim_head=32,
                 crossattn_modality_dropout=0.1, **kwargs):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, action_horizon, input_dim) * 0.02)
        self.cross_attention = CrossAttention(input_dim, crossattn_heads, crossattn_dim_head,
                                              crossattn_modality_dropout)
        self.action_projection = nn.Linear(input_dim, output_dim)
        self.register_buffer('query_pos', sinusoid_table(action_horizon, input_dim).clone())

    def forward(self, x):
        context = x.reshape(x.shape[0], -1, x.shape[-1])
        queries = (self.tokens + self.query_pos).repeat(len(context), 1, 1)
        return self.action_projection(self.cross_attention(queries, context))
