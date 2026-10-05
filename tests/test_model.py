import pandas as pd
import pytest

from fuelcast import market, model


@pytest.fixture(scope="module")
def inputs():
    mkt, _ = market.load(live=False)
    return model.load_prices(), mkt


def test_prices_are_clean():
    p = model.load_prices()
    assert p["effective_date"].is_monotonic_increasing
    assert p[["petrol_95", "diesel_005"]].min().min() > 900  # c/l, no lost digits
    assert p["petrol_95"].diff().abs().max() < 400


@pytest.mark.parametrize("fuel", list(model.FUELS))
def test_backtest_beats_naive(inputs, fuel):
    prices, mkt = inputs
    bt = model.backtest(model.build_table(prices, mkt, fuel))
    m = model.metrics(bt)
    assert m["mae"] < 0.7 * m["naive_mae"]
    assert m["direction"] > 0.75


def test_nowcast_effects_add_up(inputs):
    prices, mkt = inputs
    table = model.build_table(prices, mkt, "diesel")
    bt = model.backtest(table)
    today = mkt.index.max()
    for fc in model.nowcast(prices, mkt, table, bt, "diesel", today):
        assert fc.low < fc.change < fc.high
        assert 0 <= fc.progress <= 1
