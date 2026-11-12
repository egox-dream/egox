"""Embodiment summary tokenizer, future-token prediction, and dreaming loss (adapted from HPT)."""
import torch
import torch.nn.functional as F
from torch import nn

from .tokenizers import BaseTokenizer
from .transformer import CrossAttention, sinusoid_table


class EmbodimentTokenizer(BaseTokenizer):
    """Summary tokens of an embodiment: queries attend to the concatenated features of
    several modalities. MXT+ dreams these tokens for future frames.

    A positive `norm_upper_bound_weight` adds the soft penalty
    mean(relu(||token|| - norm_upper_bound) ** 2) on the online tokens.
    """

    def __init__(self, hidden_dim, modalities, token_num=8, norm_upper_bound=100.0,
                 norm_upper_bound_weight=1e-5, nhead=8, dim_head=64, dropout=0.1, **kwargs):
        super().__init__()
        if norm_upper_bound_weight < 0 or norm_upper_bound <= 0:
            raise ValueError('norm_upper_bound must be positive and its weight nonnegative')
        self.modalities = list(modalities)
        self.norm_upper_bound = norm_upper_bound
        self.norm_upper_bound_weight = norm_upper_bound_weight
        self.init_queries(token_num, hidden_dim, nhead, dim_head, dropout)

    def compute_latent(self, feat, mask, bs=None, device=None):
        """feat (B, N, D) with boolean mask (B, N), or None. Returns tokens, trunk mask, penalty."""
        if feat is None:
            return (torch.zeros(bs, self.token_num, self.hidden_dim, device=device),
                    torch.zeros(bs, self.token_num, dtype=torch.bool, device=device), None)
        bs = feat.shape[0]
        mask = mask.bool()
        tokens = self.cross_attn(self.queries(bs), feat, mask)
        valid = mask.reshape(bs, -1).any(dim=1)
        penalty = None
        if self.norm_upper_bound_weight > 0:
            valid_tokens = valid[:, None].expand(bs, self.token_num)
            length = torch.linalg.vector_norm(
                tokens.float().masked_fill(~valid_tokens[..., None], 0.0), dim=-1)
            torch._assert_async(torch.isfinite(length).all(), 'Nonfinite embodiment token norm')
            excess = F.relu(length - self.norm_upper_bound).square().masked_fill(~valid_tokens, 0.0)
            penalty = excess.sum() / valid_tokens.sum().clamp(min=1)
        tokens = tokens.masked_fill(~valid[:, None, None], 0)
        trunk_mask = valid[:, None].expand(-1, self.token_num)
        return tokens, trunk_mask, penalty


def dream_loss(pred, target, pad, mag_beta):
    """Cosine distance plus `mag_beta` * smooth-L1 between norms, averaged over unpadded tokens.

    Returns (loss, direction_loss, magnitude_loss).
    """
    direction = 1.0 - F.cosine_similarity(F.normalize(pred, dim=-1), F.normalize(target, dim=-1), dim=-1)
    magnitude = F.smooth_l1_loss(pred.norm(dim=-1), target.norm(dim=-1), reduction='none')
    valid = ~pad
    count = torch.sum(valid).clamp(min=1)
    loss = torch.sum((direction + mag_beta * magnitude) * valid) / count
    return loss, torch.sum(direction * valid) / count, torch.sum(magnitude * valid) / count


class DreamHead(nn.Module):
    """Predicts the EMA embodiment tokens of future frames from the trunk output.

    `num_tokens` = number of future frames x embodiment tokens per frame.
    """

    def __init__(self, input_dim, output_dim, num_tokens, crossattn_heads=8, crossattn_dim_head=32,
                 crossattn_modality_dropout=0.1, **kwargs):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)
        self.cross_attention = CrossAttention(output_dim, crossattn_heads, crossattn_dim_head,
                                              crossattn_modality_dropout)
        embed_dim = crossattn_heads * crossattn_dim_head
        self.mlp = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, output_dim))
        self.register_buffer('query_pos', sinusoid_table(num_tokens, output_dim).clone())

    def forward(self, x):
        context = self.mlp(x)
        context = context.reshape(context.shape[0], -1, context.shape[-1])
        queries = (self.tokens + self.query_pos).repeat(len(context), 1, 1)
        return self.cross_attention(queries, context)
