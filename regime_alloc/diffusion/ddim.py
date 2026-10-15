"""DDIM sampler (Song et al. 2021), paper §3.3 / A.2."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from regime_alloc.diffusion.schedule import DiffusionSchedule


def ddim_timesteps(T: int, n_steps: int) -> list[int]:
    return sorted(set(np.linspace(0, T - 1, n_steps).round().astype(int).tolist()))


@torch.no_grad()
def ddim_sample(eps_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                sched: DiffusionSchedule, shape: tuple, n_steps: int, eta: float = 0.0,
                device="cpu", generator: torch.Generator | None = None) -> torch.Tensor:
    """eps_fn(r_n, n) -> ε̂. Returns r̂_0.
    r̂_0 = (r_n - sqrt(1-ᾱ_n) ε̂) / sqrt(ᾱ_n)
    r_{n-1} = sqrt(ᾱ_{n-1}) r̂_0 + sqrt(1-ᾱ_{n-1}-σ²) ε̂ + σ z   (σ = 0 when η = 0)"""
    seq = ddim_timesteps(sched.T, n_steps)
    ab = sched.alpha_bar
    r = torch.randn(shape, device=device, generator=generator)
    r0 = r
    for i in reversed(range(len(seq))):
        n = seq[i]
        a_t = ab[n]
        a_prev = ab[seq[i - 1]] if i > 0 else torch.tensor(1.0, device=device)
        eps = eps_fn(r, torch.full((shape[0],), n, device=device, dtype=torch.long))
        r0 = (r - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if i == 0:
            break
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
        noise = torch.randn(shape, device=device, generator=generator) if eta > 0 else 0.0
        r = a_prev.sqrt() * r0 + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise
    return r0
