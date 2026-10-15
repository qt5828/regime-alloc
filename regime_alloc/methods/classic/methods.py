"""Classical allocators used as baselines in paper §4.1."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from regime_alloc.data.dataset import MarketData
from regime_alloc.methods.base import Method


class CRP(Method):
    """Constant rebalanced portfolio (Kelly 1956): equal weights at every decision."""
    name = "crp"

    def allocate(self, data: MarketData, tau: int) -> np.ndarray:
        return np.full(data.S, 1.0 / data.S)


class MVO(Method):
    """Markowitz (1952) max-Sharpe on the last `lookback` days, long-only."""
    name = "mvo"

    def allocate(self, data: MarketData, tau: int) -> np.ndarray:
        R = data.simret[tau - self.cfg["lookback"] + 1: tau + 1]
        mu, sigma = R.mean(0), np.cov(R, rowvar=False)
        S = data.S

        def neg_sharpe(w):
            return -(w @ mu) / np.sqrt(w @ sigma @ w + 1e-12)

        res = minimize(neg_sharpe, np.full(S, 1.0 / S), method="SLSQP",
                       bounds=[(0.0, 1.0)] * S,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                       options={"maxiter": 500, "ftol": 1e-10})
        w = np.clip(res.x, 0, None)
        return w / w.sum()


class MOM(Method):
    """Momentum (Jegadeesh & Titman 1993): equal-weight the top past winners."""
    name = "mom"

    def allocate(self, data: MarketData, tau: int) -> np.ndarray:
        lb, skip = self.cfg["lookback"], self.cfg["skip"]
        score = data.px[tau - skip] / data.px[tau - lb] - 1.0
        n_top = max(1, int(round(self.cfg["top_frac"] * data.S)))
        w = np.zeros(data.S)
        w[np.argsort(-score)[:n_top]] = 1.0 / n_top
        return w
