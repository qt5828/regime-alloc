"""Rebalancing backtest shared by all methods.

At each rebalance close τ the method returns w (1ᵀw = 1); the book is held
buy-and-hold over τ+1 .. next rebalance (no transaction costs; the paper
mentions none)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime_alloc.data.dataset import MarketData
from regime_alloc.eval.metrics import forecast_metrics, portfolio_metrics
from regime_alloc.methods.base import Method


def run_backtest(data: MarketData, method: Method) -> dict:
    taus = data.rebalance_taus
    end = data.test_end_idx
    daily, days, weights, paths_all, truth_all = [], [], [], [], []
    for i, tau in enumerate(taus):
        w = method.allocate(data, int(tau))
        assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-6, f"bad weights at τ={tau}"
        weights.append(pd.Series(w, index=data.tickers, name=data.dates[tau]))
        stop = min(int(taus[i + 1]) if i + 1 < len(taus) else end, end)
        gross = np.cumprod(1.0 + data.simret[tau + 1: stop + 1], axis=0)       # (h, S)
        nav = np.concatenate([[1.0], gross @ w])
        daily.append(nav[1:] / nav[:-1] - 1.0)
        days.extend(data.dates[tau + 1: stop + 1])

        if tau + data.seg_len <= end:       # full ℓ-day target inside the test window
            p = method.forecast(data, int(tau))
            if p is not None:
                paths_all.append(p)
                truth_all.append(data.future_logret(int(tau)))

    daily = np.concatenate(daily)
    out = {"daily": pd.Series(daily, index=pd.DatetimeIndex(days), name="ret"),
           "weights": pd.DataFrame(weights),
           "portfolio": portfolio_metrics(daily)}
    if paths_all:
        out["forecast"] = forecast_metrics(np.concatenate(paths_all), np.concatenate(truth_all))
    return out
