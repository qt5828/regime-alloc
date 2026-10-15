"""Macro condition m_τ (paper Table 1): market index, dollar index, U.S. term
spread (10y-3m), VIX, gold. Each series contributes its ν-day rolling-normalised
level and its log difference ("raw and processed values")."""
from __future__ import annotations

import numpy as np
import pandas as pd


def align_to_calendar(s: pd.Series, calendar: pd.DatetimeIndex, lag: bool) -> pd.Series:
    """As-of alignment. With `lag`, a value stamped on day d becomes usable only
    from d+1, because US closes happen after Asian/European closes."""
    s = s.dropna()
    if lag:
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(s.index.union(calendar)).ffill().reindex(calendar)


def rolling_z(x: pd.Series | pd.DataFrame, nu: int):
    mu = x.rolling(nu, min_periods=nu).mean()
    sd = x.rolling(nu, min_periods=nu).std()
    return (x - mu) / sd.replace(0.0, np.nan)


def build_macro(index_px: pd.Series, macro_raw: pd.DataFrame,
                calendar: pd.DatetimeIndex, nu: int, foreign_lag: bool) -> pd.DataFrame:
    levels = {"market": align_to_calendar(index_px, calendar, lag=False)}
    a = {k: align_to_calendar(macro_raw[k], calendar, lag=foreign_lag) for k in macro_raw}
    levels["dollar"] = a["dollar"]
    levels["term_spread"] = a["tnx"] - a["irx"]
    levels["vix"] = a["vix"]
    levels["gold"] = a["gold"]

    feats = {}
    for name, lv in levels.items():
        feats[f"{name}_z"] = rolling_z(lv, nu)
        # term spread can be <= 0 -> plain difference instead of log difference
        feats[f"{name}_d"] = lv.diff() if name == "term_spread" else np.log(lv).diff()
    return pd.DataFrame(feats, index=calendar)
