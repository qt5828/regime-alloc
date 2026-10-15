"""Method interface shared by baselines, STABLE and future proposed methods.

A method sees only `MarketData` and a rebalance index τ; everything it uses
must come from rows <= τ (the backtester never passes future rows)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from regime_alloc.data.dataset import MarketData


class Method:
    name = "base"

    def __init__(self, cfg: dict, market: str):
        self.cfg = cfg
        self.market = market

    def fit(self, data: MarketData, out_dir: Path) -> None:
        """Train on data.train_taus. Default: nothing to train."""

    def load(self, out_dir: Path, data: MarketData | None = None) -> None:
        """Load trained state written by `fit`."""

    def allocate(self, data: MarketData, tau: int) -> np.ndarray:
        """Portfolio weights (S,) decided at the close of day τ, 1ᵀw = 1."""
        raise NotImplementedError

    def forecast(self, data: MarketData, tau: int) -> np.ndarray | None:
        """Optional: (S, k, ℓ) sampled future daily log-return paths for τ+1..τ+ℓ.
        Used for the Table 4 forecast metrics."""
        return None
