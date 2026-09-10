"""DiT-S/2 predicting the noise eps in x_t = alpha x_0 + sigma eps, for 32 x 32 x 3.

The image is cut into 2 x 2 patches, 256 tokens of dimension 384, with a
learned positional embedding.  Twelve transformer blocks (6 heads, MLP ratio
4) are conditioned on the time through adaLN-Zero: a sinusoidal embedding of
t passed through an MLP yields, per block, a shift, a scale and a gate for
the attention branch and again for the MLP branch, all from a
zero-initialised linear map, so every block starts as the identity.  A final
adaLN layer and a zero-initialised linear map return the patches, which are
folded back into the image.  About 33 million parameters: the DiT-S/2
configuration of Peebles and Xie, on three channels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(t, dim):
    """Features of t in [0, 1]: sin and cos of t * 1000 at dim/2 frequencies."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = 1000.0 * t[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


def modulate(x, shift, scale):
    return x * (1.0 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    """Multi-head self attention through the fused scaled-dot-product kernel,
    which never materialises the attention matrices (nn.MultiheadAttention
    does in training, and at batch 512 that alone exceeds the GPU memory)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(x).reshape(B, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v)                       # (B, heads, n, d/heads)
        return self.proj(h.transpose(1, 2).reshape(B, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), s1, sc1)
        x = x + g1[:, None] * self.attn(h)
        h = modulate(self.norm2(x), s2, sc2)
        return x + g2[:, None] * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, size=32, channels=3, patch=2, dim=384, depth=12, heads=6, mlp_ratio=4):
        super().__init__()
        assert size % patch == 0
        self.patch, self.dim, self.channels = patch, dim, channels
        n = (size // patch) ** 2
        self.embed = nn.Conv2d(channels, dim, patch, stride=patch)
        self.pos = nn.Parameter(0.02 * torch.randn(1, n, dim))
        self.temb = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * channels)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        c = self.temb(sinusoidal(t, self.dim))
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos          # (B, n, dim)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada(c).chunk(2, dim=1)
        h = self.out(modulate(self.norm(h), shift, scale))              # (B, n, p*p*C)
        B, p, C = x.shape[0], self.patch, self.channels
        g = x.shape[-1] // p
        h = h.reshape(B, g, g, p, p, C)
        return h.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)
