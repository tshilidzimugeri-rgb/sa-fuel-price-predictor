"""Daily market inputs: product futures and the rand/dollar rate.

SA's basic fuel price follows international *product* prices, so petrol is
tracked with RBOB gasoline futures and diesel with NY Harbor ULSD futures,
both quoted in US dollars per gallon. Brent is kept for context.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TICKERS = {"RB=F": "gasoline_usd_gal", "HO=F": "diesel_usd_gal", "BZ=F": "brent_usd_bbl", "USDZAR=X": "usdzar"}
SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "market_snapshot.csv"
LITRES_PER_GALLON = 3.785411784


def download(start: str = "2011-09-01") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(list(TICKERS), start=start, progress=False, auto_adjust=False)["Close"]
    frame = raw.rename(columns=TICKERS)[list(TICKERS.values())]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.ffill().dropna()


def load(live: bool = True) -> tuple[pd.DataFrame, str]:
    """Live data from Yahoo Finance, falling back to the bundled snapshot."""
    if live:
        try:
            frame = download()
            if len(frame) > 1000:
                return add_rand_prices(frame), "live"
        except Exception:
            pass
    frame = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)
    return add_rand_prices(frame), "snapshot"


def add_rand_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Product prices converted to SA cents per litre."""
    out = frame.copy()
    out["petrol_proxy"] = out["gasoline_usd_gal"] * out["usdzar"] / LITRES_PER_GALLON * 100
    out["diesel_proxy"] = out["diesel_usd_gal"] * out["usdzar"] / LITRES_PER_GALLON * 100
    return out
