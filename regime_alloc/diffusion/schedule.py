"""Forward process q(r_n | r_0) (DDPM, Ho et al. 2020)."""
from __future__ import annotations

import math

import torch


def make_betas(timesteps: int, schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        # Improved-DDPM `get_named_beta_schedule("linear")`: DDPM's 1e-4..0.02
        # for T=1000, rescaled so any T gives the same total noise.
        scale = 1000.0 / timesteps
        return torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
    if schedule == "cosine":
        s = 0.008
        f = lambda t: math.cos((t / timesteps + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
        return torch.tensor([min(1 - f(i + 1) / f(i), 0.999) for i in range(timesteps)],
                            dtype=torch.float64)
    raise ValueError(schedule)


class DiffusionSchedule:
    def __init__(self, timesteps: int, schedule: str = "linear"):
        self.T = timesteps
        betas = make_betas(timesteps, schedule)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).float()  # ᾱ_n, n = 0..T-1

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, r0: torch.Tensor, n: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[n].view(-1, *([1] * (r0.dim() - 1)))
        return ab.sqrt() * r0 + (1 - ab).sqrt() * eps
