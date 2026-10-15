"""Temporal stock embedding β_τ^(s) (paper §3.3, Das & Ghoshal 2010).

Time-varying regression  y_τ = α_τ + β_τᵀ m_τ + e_τ  with random-walk states
[α, β]. Filtering is causal: β_τ uses observations up to τ only.
Measurement noise R is estimated adaptively from recent innovations
(innovation-based adaptive filter, Das & Ghoshal 2010).
"""
from __future__ import annotations

import numpy as np


def kalman_beta(y: np.ndarray, X: np.ndarray, delta: float = 1e-4,
                r_window: int = 120) -> np.ndarray:
    """y: (T,), X: (T, d). Returns β: (T, d), NaN where not yet estimable.
    Rows with NaN in y or X are skipped (state carried forward)."""
    T, d = X.shape
    k = d + 1
    out = np.full((T, d), np.nan)
    Q = delta / (1.0 - delta) * np.eye(k)
    theta = np.zeros(k)
    P = np.eye(k)
    innov2: list[float] = []
    R = None
    started = False
    for t in range(T):
        if np.isnan(y[t]) or np.isnan(X[t]).any():
            if started:
                out[t] = theta[1:]
            continue
        h = np.concatenate(([1.0], X[t]))
        if R is None:
            innov2.append(y[t] ** 2)
            if len(innov2) >= min(r_window, 20):
                R = float(np.mean(innov2))
                started = True
            else:
                continue
        P = P + Q
        e = y[t] - h @ theta
        s = h @ P @ h + R
        K = P @ h / s
        theta = theta + K * e
        P = P - np.outer(K, h) @ P
        # adaptive R: windowed innovation variance minus state-uncertainty part
        innov2.append(e * e)
        if len(innov2) > r_window:
            innov2.pop(0)
        R = max(float(np.mean(innov2)) - float(h @ P @ h), 1e-8)
        out[t] = theta[1:]
    return out
