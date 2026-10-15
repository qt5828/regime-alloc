"""Download adjusted-close prices from yfinance and cache them as parquet."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from regime_alloc.utils import PROJECT_ROOT, load_yaml

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_START = "2012-01-01"   # ν-day warm-up before the 2013-01 train start
DOWNLOAD_END = "2026-10-15"     # inclusive; extra data kept for future splits

yf.set_tz_cache_location(str(PROJECT_ROOT / ".cache" / "yf"))


def _fetch(ticker: str, start: str, end: str, retries: int = 3) -> pd.Series:
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(retries):
        df = yf.download(ticker, start=start, end=end_excl, auto_adjust=False,
                         progress=False, threads=False)
        if len(df):
            s = df["Adj Close"].iloc[:, 0].dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            return s.rename(ticker)
        time.sleep(2 * (attempt + 1))
    return pd.Series(dtype=float, name=ticker)


def download_tickers(tickers: list[str], start: str = DOWNLOAD_START,
                     end: str = DOWNLOAD_END) -> pd.DataFrame:
    series = []
    for t in tickers:
        s = _fetch(t, start, end)
        print(f"  {t:<12} {len(s):>5} rows" + ("" if len(s) else "  <-- FAILED"))
        series.append(s)
    return pd.concat(series, axis=1).sort_index()


def download_market(market: str, universe_cfg: dict | None = None) -> Path:
    cfg = universe_cfg or load_yaml(PROJECT_ROOT / "configs/data/universes.yaml")
    mcfg = cfg["markets"][market]
    tickers = [mcfg["index"]] + [t for ts in mcfg["sectors"].values() for t in ts]
    print(f"[{market}] downloading {len(tickers)} tickers")
    df = download_tickers(tickers)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{market}_prices.parquet"
    df.to_parquet(path)
    return path


def download_macro(universe_cfg: dict | None = None) -> Path:
    cfg = universe_cfg or load_yaml(PROJECT_ROOT / "configs/data/universes.yaml")
    names = cfg["macro"]
    print(f"[macro] downloading {len(names)} tickers")
    df = download_tickers(list(names.values()))
    df.columns = list(names.keys())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "macro.parquet"
    df.to_parquet(path)
    return path


def load_raw(market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(RAW_DIR / f"{market}_prices.parquet"),
            pd.read_parquet(RAW_DIR / "macro.parquet"))
