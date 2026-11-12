"""Per-modality tokenizers: an encoder followed by learned queries that cross-attend
to its features (adapted from HPT)."""
import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

from .transformer import CrossAttention, sinusoid_table


class BaseTokenizer(nn.Module):
    """Shared by every tokenizer: pools a feature sequence into `token_num` tokens with learned,
    position-encoded queries."""

    def init_queries(self, token_num, hidden_dim, nhead=8, dim_head=64, dropout=0.1):
        self.token_num = token_num
        self.hidden_dim = hidden_dim
        self.tokens = nn.Parameter(torch.randn(1, token_num, hidden_dim))
        self.cross_attn = CrossAttention(hidden_dim, nhead, dim_head, dropout)
        self.register_buffer('query_pos', sinusoid_table(token_num, hidden_dim).clone())

    def queries(self, bs):
        return (self.tokens + self.query_pos).repeat(bs, 1, 1)

    def compute_latent(self, feat, mask):
        """feat (B, N, D) or None if the modality is absent from the whole batch."""
        bs, device = mask.shape[0], self.tokens.device
        if feat is None:
            return (torch.zeros(bs, self.token_num, self.hidden_dim, device=device),
                    torch.zeros(bs, self.token_num, dtype=torch.bool, device=device))
        present = mask.reshape(bs, -1).bool().any(dim=1)
        valid_tokens = present[:, None].expand(-1, self.token_num)
        tokens = self.cross_attn(self.queries(bs), feat.reshape(bs, -1, feat.shape[-1]))
        return tokens.masked_fill(~valid_tokens[..., None], 0), valid_tokens


def build_resnet(resnet_model='resnet18', weights='DEFAULT'):
    """ImageNet ResNet without its pooling and classifier; returns (module, feature_dim)."""
    resnet = getattr(torchvision.models, resnet_model)(weights=weights)
    return nn.Sequential(*list(resnet.children())[:-2]), resnet.fc.in_features


class ImageTokenizer(BaseTokenizer):
    """ResNet feature map -> optional LayerNorm -> linear projection -> one token per cell.

    `split_dual` encodes the left and right halves of a side-by-side stereo frame
    separately. With `shared_encoder`, the ResNet is owned (and saved) by MXT
    and shared across embodiments; this tokenizer only keeps a reference.
    """

    def __init__(self, output_dim, resnet_model='resnet18', weights='DEFAULT', split_dual=False,
                 feature_norm=True, gradient_checkpoint=False, shared_encoder=None, **kwargs):
        super().__init__()
        if shared_encoder is None:
            self.net, feat_dim = build_resnet(resnet_model, weights)
        else:
            object.__setattr__(self, 'net', shared_encoder[0])  # not registered here
            feat_dim = shared_encoder[1]
        self.split_dual = split_dual
        self.feature_norm = feature_norm
        self.gradient_checkpoint = gradient_checkpoint
        self.norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Linear(feat_dim, output_dim)

    def _encode(self, x):
        if self.gradient_checkpoint and self.training:
            return checkpoint(self.net, x, use_reentrant=False)
        return self.net(x)

    def forward(self, x):
        """x (B, ..., 3, H, W) -> features (B, M, output_dim)."""
        B, *_, H, W = x.shape
        x = x.reshape(-1, 3, H, W)
        if self.split_dual:
            feat = torch.cat([self._encode(x[..., :W // 2]), self._encode(x[..., W // 2:])], dim=-1)
        else:
            feat = self._encode(x)
        feat = feat.flatten(start_dim=-2).transpose(1, 2)
        if self.feature_norm:
            feat = self.norm(feat)
        return self.proj(feat).reshape(B, -1, self.proj.out_features).contiguous()


class ProprioTokenizer(BaseTokenizer):
    """MLP over a proprioceptive vector."""

    def __init__(self, input_dim, output_dim, widths=(512, 512), tanh_end=False, ln=True, **kwargs):
        super().__init__()
        dims = [input_dim, *widths]
        modules = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            modules += [nn.Linear(d_in, d_out), *([nn.LayerNorm(d_out)] if ln else []), nn.SiLU()]
        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
