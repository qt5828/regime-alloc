"""Black–Litterman MVO (paper §3.5, A.3)."""
from __future__ import annotations

import numpy as np


def prior_moments(simret_window: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """ν-day rolling prior, scaled to the ℓ-day horizon of the views.
    μ_prior = Σ_prior w_eq so that w_eq is the MVO direction under the prior (A.3)."""
    S = simret_window.shape[1]
    sigma = np.cov(simret_window, rowvar=False, ddof=1) * horizon
    w_eq = np.full(S, 1.0 / S)
    return sigma @ w_eq, sigma


def view_moments(paths_logret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """paths_logret: (S, k, ℓ). Cumulative return r̄ = exp(Σ log r) − 1 per path.
    μ_view = sample mean over k (Eq. 3); Σ_view = unbiased sample covariance of
    the cross-asset vectors r^(i) (Eq. 4)."""
    cum = np.expm1(paths_logret.sum(axis=-1))          # (S, k)
    return cum.mean(axis=1), np.cov(cum, rowvar=True, ddof=1)


def precision(sigma: np.ndarray, mode: str = "pinv") -> np.ndarray:
    if mode == "pinv":   # zero certainty along directions with no sampled variation
        return np.linalg.pinv(sigma, hermitian=True)
    if mode == "inv":
        return np.linalg.inv(sigma)
    raise ValueError(mode)


def bl_posterior(mu_p, sigma_p, mu_v, sigma_v, view_precision: str = "pinv"):
    """μ_BL = (Φ+Ω)⁻¹(Φ μ_prior + Ω μ_view), Σ_BL = (Φ+Ω)⁻¹  (Eq. 6-7)."""
    Phi = np.linalg.inv(sigma_p)
    Omega = precision(sigma_v, view_precision)
    H = Phi + Omega
    mu_bl = np.linalg.solve(H, Phi @ mu_p + Omega @ mu_v)
    return mu_bl, np.linalg.inv(H), H


def max_sharpe_weights(mu: np.ndarray, sigma: np.ndarray | None = None,
                       sigma_inv: np.ndarray | None = None) -> np.ndarray:
    """w* = Σ⁻¹μ / 1ᵀΣ⁻¹μ  (Eq. 8, closed form; long/short allowed)."""
    x = sigma_inv @ mu if sigma_inv is not None else np.linalg.solve(sigma, mu)
    return x / x.sum()


def bl_mvo_weights(simret_window: np.ndarray, paths_logret: np.ndarray,
                   view_precision: str = "pinv", use_bl: bool = True) -> tuple[np.ndarray, dict]:
    horizon = paths_logret.shape[-1]
    mu_v, sigma_v = view_moments(paths_logret)
    if not use_bl:   # ablation A.7: plug-in MVO on the diffusion view moments
        w = max_sharpe_weights(mu_v, sigma_inv=precision(sigma_v, view_precision))
        return w, {"mu_view": mu_v}
    mu_p, sigma_p = prior_moments(simret_window, horizon)
    mu_bl, _, H = bl_posterior(mu_p, sigma_p, mu_v, sigma_v, view_precision)
    x = H @ mu_bl                      # Σ_BL⁻¹ μ_BL
    return x / x.sum(), {"mu_view": mu_v, "mu_bl": mu_bl, "denom": float(x.sum())}
