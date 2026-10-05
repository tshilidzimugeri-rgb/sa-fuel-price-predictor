import pandas as pd

from fuelcast.windows import window_for


def test_october_2026_window():
    # Last Friday of August 2026 to last Thursday of September 2026.
    w = window_for("2026-10-01")
    assert w.start == pd.Timestamp("2026-08-28")
    assert w.end == pd.Timestamp("2026-09-24")
    assert w.effective == pd.Timestamp("2026-10-07")


def test_windows_are_contiguous():
    for month in pd.date_range("2015-01-01", "2026-12-01", freq="MS"):
        this, nxt = window_for(month), window_for(month + pd.offsets.MonthBegin(1))
        assert nxt.start - this.end == pd.Timedelta(days=1)


def test_progress_bounds():
    w = window_for("2026-11-01")
    assert w.progress(w.start - pd.Timedelta(days=5)) == 0
    assert w.progress(w.end + pd.Timedelta(days=5)) == 1
