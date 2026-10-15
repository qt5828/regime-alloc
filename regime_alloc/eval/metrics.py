"""Evaluation metrics (paper §4.1)."""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def asr(daily: np.ndarray) -> float:
    """Annualised Sharpe ratio, zero risk-free rate."""
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS))


def rmdd(daily: np.ndarray) -> float:
    """Max peak-to-trough loss divided by the peak, in %."""
    nav = np.concatenate([[1.0], np.cumprod(1.0 + daily)])
    peak = np.maximum.accumulate(nav)
    return float(((peak - nav) / peak).max() * 100)


def avol(daily: np.ndarray) -> float:
    """Annualised volatility of daily returns, in %."""
    return float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)


def portfolio_metrics(daily: np.ndarray) -> dict:
    return {"ASR": asr(daily), "RMDD": rmdd(daily), "AVol": avol(daily)}


def dtw_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """DTW with |·| cost between rows of a and b (N, L), normalised by L."""
    N, L = a.shape
    cost = np.abs(a[:, :, None] - b[:, None, :])
    D = np.full((N, L + 1, L + 1), np.inf)
    D[:, 0, 0] = 0.0
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            D[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(
                np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, L, L] / L


def forecast_metrics(paths: np.ndarray, truth: np.ndarray) -> dict:
    """paths: (N, k, ℓ) sampled log-return paths; truth: (N, ℓ).
    MSE of the mean-of-k forecaster; DTW averaged over the k paths (§4.3)."""
    mse = float(((paths.mean(1) - truth) ** 2).mean())
    N, k, L = paths.shape
    dtw = dtw_batch(paths.reshape(N * k, L), np.repeat(truth, k, axis=0)).mean()
    return {"MSE": mse, "DTW": float(dtw)}
