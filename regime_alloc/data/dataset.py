"""MarketData: one aligned, leakage-free bundle per (market, split), shared by
every method. Arrays are indexed [t] over the market calendar and [s] over
the filtered universe."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from regime_alloc.data.download import DOWNLOAD_START, load_raw
from regime_alloc.data.universe import filter_universe, market_calendar
from regime_alloc.features.kalman_beta import kalman_beta
from regime_alloc.features.macro import build_macro, rolling_z
from regime_alloc.utils import PROJECT_ROOT, load_yaml

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class MarketData:
    market: str
    split: dict
    dates: pd.DatetimeIndex
    tickers: list[str]
    sector_of: dict[str, str]
    dropped: dict[str, str]
    px: np.ndarray          # (T,S) adjusted close, ffilled after listing
    logret: np.ndarray      # (T,S) daily log return r_t = log P_t - log P_{t-1}
    simret: np.ndarray      # (T,S) daily simple return
    px_z: np.ndarray        # (T,S) ν-day rolling z-score of price level
    macro: np.ndarray       # (T,dm) m_τ
    macro_names: list[str]
    beta: np.ndarray        # (T,S,dm) Kalman β_τ^(s)
    ret_scale: np.ndarray   # (S,) std of daily log returns over the train window
    nu: int
    seg_len: int
    train_taus: np.ndarray = field(default_factory=lambda: np.array([], int))
    rebalance_taus: np.ndarray = field(default_factory=lambda: np.array([], int))
    test_start_idx: int = 0
    test_end_idx: int = 0

    @property
    def S(self) -> int:
        return len(self.tickers)

    def future_logret(self, tau: int, length: int | None = None) -> np.ndarray:
        """(S, ℓ) realised log returns over τ+1 .. τ+ℓ."""
        L = length or self.seg_len
        return self.logret[tau + 1: tau + 1 + L].T


def _date_idx(dates: pd.DatetimeIndex, d, side: str) -> int:
    """side='left': first idx with date >= d; 'right': last idx with date <= d."""
    d = pd.Timestamp(d)
    return int(dates.searchsorted(d, "left")) if side == "left" \
        else int(dates.searchsorted(d, "right")) - 1


def build_market_data(market: str, split: dict, nu: int = 120, seg_len: int = 20,
                      kalman_delta: float = 1e-4, kalman_r_window: int = 120,
                      use_cache: bool = True) -> MarketData:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED_DIR / f"{market}_{split['name']}_nu{nu}_l{seg_len}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    ucfg = load_yaml(PROJECT_ROOT / "configs/data/universes.yaml")["markets"][market]
    prices, macro_raw = load_raw(market)
    index_px = prices[ucfg["index"]]
    stock_px = prices.drop(columns=[ucfg["index"]])

    # Calendar runs from the download start to the end of the test window only:
    # nothing after test_end can influence any feature.
    dates = market_calendar(index_px, DOWNLOAD_START, split["test_end"])
    tickers, dropped = filter_universe(stock_px, dates, split["train_start"], split["test_end"])
    sector_of = {t: sec for sec, ts in ucfg["sectors"].items() for t in ts}

    px_df = stock_px[tickers].reindex(stock_px.index.union(dates)).ffill().reindex(dates)
    # no back-fill: before listing stays NaN
    for t in tickers:
        first = stock_px[t].first_valid_index()
        px_df.loc[px_df.index < first, t] = np.nan
    px = px_df.to_numpy(float)
    logret = np.vstack([np.full((1, len(tickers)), np.nan), np.diff(np.log(px), axis=0)])
    simret = np.expm1(logret)
    px_z = rolling_z(px_df, nu).to_numpy(float)

    macro_df = build_macro(index_px, macro_raw, dates, nu, ucfg["foreign_macro_lag"])
    macro = macro_df.to_numpy(float)

    beta = np.stack([kalman_beta(logret[:, s], macro, kalman_delta, kalman_r_window)
                     for s in range(len(tickers))], axis=1)

    tr0 = _date_idx(dates, split["train_start"], "left")
    tr1 = _date_idx(dates, split["train_end"], "right")
    te0 = _date_idx(dates, split["test_start"], "left")
    te1 = _date_idx(dates, split["test_end"], "right")
    ret_scale = np.nanstd(logret[tr0:tr1 + 1], axis=0)

    # training anchors: target window τ+1..τ+ℓ must stay inside the train window
    train_taus = np.arange(tr0, tr1 - seg_len + 1)
    # rebalance at the close before the test start, then every ℓ trading days
    rebalance_taus = np.arange(te0 - 1, te1, seg_len)

    data = MarketData(market=market, split=split, dates=dates, tickers=tickers,
                      sector_of=sector_of, dropped=dropped, px=px, logret=logret,
                      simret=simret, px_z=px_z, macro=macro,
                      macro_names=list(macro_df.columns), beta=beta,
                      ret_scale=ret_scale, nu=nu, seg_len=seg_len,
                      train_taus=train_taus, rebalance_taus=rebalance_taus,
                      test_start_idx=te0, test_end_idx=te1)
    tmp = cache.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, cache)   # atomic: parallel jobs never read a half-written file
    return data


def load_split(path) -> dict:
    return load_yaml(Path(PROJECT_ROOT) / path)
