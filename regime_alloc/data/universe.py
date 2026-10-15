"""Trading calendar and survivorship filter (paper §4.1: drop stocks whose
history does not span the whole sample window)."""
from __future__ import annotations

import pandas as pd


def market_calendar(index_series: pd.Series, start, end) -> pd.DatetimeIndex:
    """Trading days of the region = days on which the region index traded."""
    s = index_series.dropna()
    return s.loc[pd.Timestamp(start):pd.Timestamp(end)].index


def filter_universe(prices: pd.DataFrame, calendar: pd.DatetimeIndex,
                    window_start, window_end, list_tolerance_days: int = 10,
                    max_missing: float = 0.05) -> tuple[list[str], dict[str, str]]:
    """Keep tickers listed by `window_start` (+tolerance) and alive until
    `window_end`, with at most `max_missing` missing calendar days in between."""
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    cal = calendar[(calendar >= ws) & (calendar <= we)]
    keep, dropped = [], {}
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty:
            dropped[t] = "no data"
            continue
        if s.index[0] > ws + pd.Timedelta(days=list_tolerance_days):
            dropped[t] = f"listed {s.index[0].date()}"
            continue
        if s.index[-1] < we - pd.Timedelta(days=list_tolerance_days):
            dropped[t] = f"ends {s.index[-1].date()}"
            continue
        miss = 1.0 - s.reindex(cal).notna().mean()
        if miss > max_missing:
            dropped[t] = f"missing {miss:.1%}"
            continue
        keep.append(t)
    return keep, dropped
