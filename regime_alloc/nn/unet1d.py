"""Conditional 1-D convolutional UNet u_φ(r_n, n, h).

Structure follows lucidrains/denoising-diffusion-pytorch `Unet1D` (ResNet
blocks with GroupNorm and scale-shift time conditioning, conv down/upsampling),
without attention, as the paper specifies a convolutional UNet. The condition
vector h is embedded and added to the time embedding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=n.device) * -(math.log(10000) / (half - 1)))
        e = n.float()[:, None] * freqs[None]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return F.silu(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2))
        self.block1 = Block(dim_in, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, emb):
        scale_shift = self.mlp(emb)[:, :, None].chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res(x)


class Unet1D(nn.Module):
    def __init__(self, cond_dim: int, dim: int = 64, dim_mults=(1, 2, 4), channels: int = 1):
        super().__init__()
        dims = [dim] + [dim * m for m in dim_mults]
        emb_dim = dim * 4
        self.init_conv = nn.Conv1d(channels, dim, 7, padding=3)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, emb_dim),
                                      nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.GELU(),
                                      nn.Linear(emb_dim, emb_dim))
        pairs = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()
        for i, (di, do) in enumerate(pairs):
            last = i == len(pairs) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(di, di, emb_dim), ResnetBlock(di, di, emb_dim),
                nn.Conv1d(di, do, 4, 2, 1) if not last else nn.Conv1d(di, do, 3, padding=1)]))
        mid = dims[-1]
        self.mid1 = ResnetBlock(mid, mid, emb_dim)
        self.mid2 = ResnetBlock(mid, mid, emb_dim)
        self.ups = nn.ModuleList()
        for i, (di, do) in enumerate(reversed(pairs)):
            last = i == len(pairs) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(do + di, do, emb_dim), ResnetBlock(do + di, do, emb_dim),
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv1d(do, di, 3, padding=1)) if not last
                else nn.Conv1d(do, di, 3, padding=1)]))
        self.final_res = ResnetBlock(dim * 2, dim, emb_dim)
        self.final_conv = nn.Conv1d(dim, channels, 1)

    def forward(self, x: torch.Tensor, n: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, ℓ) or (B, C, ℓ); n: (B,); cond: (B, cond_dim)."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x[:, None]
        assert x.shape[-1] % (2 ** (len(self.downs) - 1)) == 0, "ℓ must be divisible by 2^(levels-1)"
        emb = self.time_mlp(n) + self.cond_mlp(cond)
        x = self.init_conv(x)
        r = x.clone()
        skips = []
        for b1, b2, down in self.downs:
            x = b1(x, emb); skips.append(x)
            x = b2(x, emb); skips.append(x)
            x = down(x)
        x = self.mid2(self.mid1(x, emb), emb)
        for b1, b2, up in self.ups:
            x = b1(torch.cat([x, skips.pop()], 1), emb)
            x = b2(torch.cat([x, skips.pop()], 1), emb)
            x = up(x)
        x = self.final_res(torch.cat([x, r], 1), emb)
        x = self.final_conv(x)
        return x[:, 0] if squeeze else x
