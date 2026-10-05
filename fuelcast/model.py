"""Regression of the monthly price change on the change in window-average
product prices (in rand), with walk-forward backtesting and a nowcast for the
windows that are still open."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from .market import LITRES_PER_GALLON
from .windows import Window, window_for

PRICES = Path(__file__).resolve().parents[1] / "data" / "fuel_prices.csv"

FUELS = {
    "petrol": {"name": "Petrol 95 ULP", "price": "petrol_95", "proxy": "petrol_proxy", "usd": "gasoline_usd_gal"},
    "diesel": {"name": "Diesel 0.05%", "price": "diesel_005", "proxy": "diesel_proxy", "usd": "diesel_usd_gal"},
}
FEATURES = ["d_proxy", "april"]


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICES, parse_dates=["effective_date"])
    prices["month"] = prices["effective_date"].dt.to_period("M").dt.to_timestamp()
    return prices


def window_means(market: pd.DataFrame, window: Window, until: pd.Timestamp | None = None) -> pd.Series:
    end = window.end if until is None else min(window.end, until)
    days = market.loc[window.start:end]
    return days.mean(numeric_only=True) if len(days) else pd.Series(dtype=float)


def build_table(prices: pd.DataFrame, market: pd.DataFrame, fuel: str) -> pd.DataFrame:
    """One row per announced adjustment, with its features and outcome."""
    spec = FUELS[fuel]
    rows = []
    for month, price in zip(prices["month"], prices[spec["price"]]):
        means = window_means(market, window_for(month))
        if means.empty:
            continue
        rows.append({"month": month, "price": price, "proxy": means[spec["proxy"]],
                     "usd": means[spec["usd"]], "usdzar": means["usdzar"]})
    table = pd.DataFrame(rows)
    table["change"] = table["price"].diff()
    table["d_proxy"] = table["proxy"].diff()
    table["april"] = (table["month"].dt.month == 4).astype(int)
    return table.dropna().reset_index(drop=True)


def fit(table: pd.DataFrame) -> HuberRegressor:
    # Huber loss keeps one-off policy changes (levy relief, slate levy) from
    # dragging the fit around.
    return HuberRegressor(epsilon=1.5, max_iter=2000).fit(table[FEATURES], table["change"])


def backtest(table: pd.DataFrame, start: str = "2016-01-01", min_train: int = 36) -> pd.DataFrame:
    """Walk-forward: each month is predicted by a model fitted only on the
    months before it."""
    out = []
    for i in range(len(table)):
        row = table.iloc[i]
        if row["month"] < pd.Timestamp(start) or i < min_train:
            continue
        model = fit(table.iloc[:i])
        out.append({"month": row["month"], "actual": row["change"],
                    "predicted": float(model.predict(table.iloc[[i]][FEATURES])[0])})
    result = pd.DataFrame(out)
    result["error"] = result["actual"] - result["predicted"]
    return result


def metrics(bt: pd.DataFrame) -> dict:
    actual, pred = bt["actual"], bt["predicted"]
    moved = actual != 0
    return {
        "mae": float(bt["error"].abs().mean()),
        "naive_mae": float(actual.abs().mean()),
        "direction": float((np.sign(pred[moved]) == np.sign(actual[moved])).mean()),
        "r2": float(1 - bt["error"].var() / actual.var()),
        "n": int(len(bt)),
    }


@dataclass
class Forecast:
    fuel: str
    window: Window
    change: float  # c/l
    low: float
    high: float
    progress: float
    oil_effect: float  # c/l from the dollar product price
    rand_effect: float  # c/l from the exchange rate
    usd_now: float
    usdzar_now: float
    usd_window: float
    usdzar_window: float
    current_price: float


def nowcast(prices: pd.DataFrame, market: pd.DataFrame, table: pd.DataFrame, bt: pd.DataFrame,
            fuel: str, today: pd.Timestamp, oil_shift: float = 0.0, rand_level: float | None = None
            ) -> list[Forecast]:
    """Forecasts for every adjustment whose window has started but whose price
    is not yet known. Days still to come are filled with the latest prices,
    optionally shifted by the what-if inputs."""
    spec = FUELS[fuel]
    model = fit(table)
    coef = dict(zip(FEATURES, model.coef_))
    lo_q, hi_q = np.quantile(bt["error"], [0.1, 0.9])

    last = market.iloc[-1]
    future_usd = last[spec["usd"]] * (1 + oil_shift)
    future_fx = rand_level if rand_level else last["usdzar"]

    month = prices["month"].max() + pd.offsets.MonthBegin(1)
    prev_usd, prev_fx = table.iloc[-1]["usd"], table.iloc[-1]["usdzar"]
    price = float(prices[spec["price"]].iloc[-1])
    forecasts = []
    while True:
        window = window_for(month)
        if window.start > today:
            break
        days = pd.bdate_range(window.start, window.end)
        seen = market.loc[window.start:min(window.end, today)]
        n_seen = len(seen)
        n_future = max(0, len(days) - n_seen)
        usd = _blend(seen[spec["usd"]], future_usd, n_future)
        fx = _blend(seen["usdzar"], future_fx, n_future)

        to_cents = 100 / LITRES_PER_GALLON
        oil_effect = coef["d_proxy"] * (usd - prev_usd) * prev_fx * to_cents
        rand_effect = coef["d_proxy"] * usd * (fx - prev_fx) * to_cents
        change = model.intercept_ + oil_effect + rand_effect + coef["april"] * (month.month == 4)
        # Wider band while much of the window is still unknown.
        spread = 1 + n_future / max(len(days), 1)
        forecasts.append(Forecast(
            fuel=fuel, window=window, change=float(change),
            low=float(change + lo_q * spread), high=float(change + hi_q * spread),
            progress=window.progress(today), oil_effect=float(oil_effect), rand_effect=float(rand_effect),
            usd_now=float(last[spec["usd"]]), usdzar_now=float(last["usdzar"]),
            usd_window=float(usd), usdzar_window=float(fx), current_price=price,
        ))
        price += change
        prev_usd, prev_fx = usd, fx
        month += pd.offsets.MonthBegin(1)
    return forecasts


def _blend(seen: pd.Series, future_value: float, n_future: int) -> float:
    total = seen.sum() + future_value * n_future
    count = len(seen) + n_future
    return float(total / count) if count else float(future_value)
