"""STABLE network: Conditional Diffusion Generator + Multi-Level Guidance.

  h_m = m_τᵀ W_m,  h_c = c_τᵀ W_c,  h_f = [h_m ‖ h_c]                    (§3.3)
  ε̂_n      = u_φ(r_n, n, h_m)        shared / systematic
  ε̂_n^(s)  = u_φ(r_n, n, h_f)        full condition
  z^(s)    = g_π(h_f) ∈ [0, z_max]   balancing gate
  ε̂ = ε̂_n + z (ε̂_n^(s) − ε̂_n)                                         (§3.4)
  L = ‖ε − ε̂‖² + β‖θ‖²,  θ = {φ, π, W_m, W_c}                           (Eq. 2)

u_φ takes a 2d-dim condition; the macro-only call pads h_m with zeros for
the stock half (null token, as in classifier-free guidance).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from regime_alloc.nn.unet1d import Unet1D


class StableNet(nn.Module):
    def __init__(self, macro_dim: int, stock_dim: int, embed_dim: int = 16,
                 unet_dim: int = 64, unet_dim_mults=(1, 2, 4), z_max: float = 2.0,
                 use_mlg: bool = True):
        super().__init__()
        self.W_m = nn.Linear(macro_dim, embed_dim, bias=False)
        self.W_c = nn.Linear(stock_dim, embed_dim, bias=False)
        self.u_phi = Unet1D(cond_dim=2 * embed_dim, dim=unet_dim, dim_mults=tuple(unet_dim_mults))
        self.g_pi = nn.Linear(2 * embed_dim, 1)
        self.z_max = z_max
        self.use_mlg = use_mlg

    def conditions(self, m: torch.Tensor, c: torch.Tensor):
        h_m = self.W_m(m)
        h_c = self.W_c(c)
        h_f = torch.cat([h_m, h_c], dim=-1)
        h_m_pad = torch.cat([h_m, torch.zeros_like(h_c)], dim=-1)
        return h_m_pad, h_f

    def gate(self, h_f: torch.Tensor) -> torch.Tensor:
        if not self.use_mlg:   # ablation A.7: constant gate 0.5
            return torch.full((h_f.shape[0], 1), 0.5, device=h_f.device)
        return self.z_max * torch.sigmoid(self.g_pi(h_f))

    def eps(self, r_n: torch.Tensor, n: torch.Tensor, h_m_pad: torch.Tensor,
            h_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = r_n.shape[0]
        # both u_φ evaluations in one batched pass
        out = self.u_phi(torch.cat([r_n, r_n]), torch.cat([n, n]), torch.cat([h_m_pad, h_f]))
        eps_shared, eps_full = out[:B], out[B:]
        z = self.gate(h_f)
        return eps_shared + z * (eps_full - eps_shared), z

    def forward(self, r_n, n, m, c):
        h_m_pad, h_f = self.conditions(m, c)
        return self.eps(r_n, n, h_m_pad, h_f)

    def l2_penalty(self) -> torch.Tensor:
        return sum((p ** 2).sum() for p in self.parameters())
