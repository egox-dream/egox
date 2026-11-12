# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""MXT trunk and attention blocks.

The trunk is DETR's pre-norm transformer encoder-decoder (Apache 2.0,
https://github.com/facebookresearch/detr). CrossAttention is the Perceiver-style
block from HPT (https://github.com/liruiw/HPT).
"""
import copy
import functools

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


@functools.lru_cache(maxsize=128)
def sinusoid_table(n_position, d_hid):
    """Fixed sinusoidal position encodings, shape (1, n_position, d_hid)."""
    positions = np.arange(n_position)[:, np.newaxis]
    dim_pairs = np.arange(d_hid)[np.newaxis, :]
    angles = positions / np.power(10000, 2 * (dim_pairs // 2) / d_hid)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.FloatTensor(angles).unsqueeze(0)


class CrossAttention(nn.Module):
    """Queries attend to a context sequence; mask is boolean (True = keep)."""

    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(query_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        attn_mask = None if mask is None else rearrange(mask, "b ... -> b (...)")[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


def _distinct(x):
    """A new handle on x. nn.MultiheadAttention projects query/key and value with separate
    matmuls when they are different tensors, preserving the original MXT projection path."""
    return x.view_as(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, key_padding_mask):
        x = self.norm1(src)
        q = k = _distinct(x)
        src = src + self.dropout1(self.self_attn(
            q, k, value=x, key_padding_mask=key_padding_mask, need_weights=False)[0])
        x = self.norm2(src)
        return src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x)))))


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask, query_pos):
        x = self.norm1(tgt)
        q = k = x + query_pos
        tgt = tgt + self.dropout1(self.self_attn(q, k, value=x, need_weights=False)[0])
        x = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(
            query=x + query_pos, key=_distinct(memory), value=memory,
            key_padding_mask=memory_key_padding_mask, need_weights=False)[0])
        x = self.norm3(tgt)
        return tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(x)))))


class Trunk(nn.Module):
    """Encoder over observation tokens; decoder over learned query blocks, one per action field.

    `output_len` maps each action field to its number of queries. An embodiment
    decodes only the blocks of its own action fields, in `output_len` order.
    """

    def __init__(self, output_len, hidden_dim=512, nheads=8, dim_feedforward=1024, enc_layers=6, dec_layers=6,
                 dropout=0.1):
        super().__init__()
        # Layers are copies of one initialized layer, as in DETR (biases start identical).
        layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.encoder_layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(enc_layers))
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        layer = TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.decoder_layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(dec_layers))
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.query_slices, total = {}, 0
        for field, count in output_len.items():
            self.query_slices[field] = slice(total, total + count)
            total += count
        self.register_buffer('query_pos', sinusoid_table(total, hidden_dim).squeeze(0))
        self.query_content = nn.Parameter(torch.randn(total, hidden_dim) * 0.02)

    def forward(self, tokens, mask, fields):
        """tokens (B, N, D), validity mask (B, N; True = present) -> queries (B, Q, D)."""
        key_padding_mask = ~mask.bool()  # MultiheadAttention uses True = ignore.
        invalid = key_padding_mask.all(dim=1)
        if invalid.any():
            rows = invalid.nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f'No valid observations for batch samples {rows}')
        bs = tokens.shape[0]
        query_pos = torch.cat([self.query_pos[self.query_slices[f]] for f in fields])
        query = torch.cat([self.query_content[self.query_slices[f]] for f in fields])
        query_pos = query_pos.unsqueeze(1).repeat(1, bs, 1)
        out = query.unsqueeze(1).repeat(1, bs, 1)
        memory = tokens.permute(1, 0, 2)
        for layer in self.encoder_layers:
            memory = layer(memory, key_padding_mask)
        memory = self.encoder_norm(memory)
        for layer in self.decoder_layers:
            out = layer(out, memory, key_padding_mask, query_pos)
        return self.decoder_norm(out).transpose(0, 1)
