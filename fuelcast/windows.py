"""The DMPR review window behind each monthly fuel price adjustment.

An adjustment that takes effect on the first Wednesday of month M is based on
average international prices and the rand/dollar rate from the day after the
last Thursday of month M-2 (normally the last Friday) up to the last Thursday
of month M-1, so consecutive windows never overlap or leave a gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

THURSDAY, WEDNESDAY = 3, 2


def last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return end - pd.Timedelta(days=(end.weekday() - weekday) % 7)


def first_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    start = pd.Timestamp(year, month, 1)
    return start + pd.Timedelta(days=(weekday - start.weekday()) % 7)


@dataclass(frozen=True)
class Window:
    month: pd.Timestamp  # first day of the month the new price applies to
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def effective(self) -> pd.Timestamp:
        return first_weekday(self.month.year, self.month.month, WEDNESDAY)

    @property
    def label(self) -> str:
        return self.month.strftime("%B %Y")

    def progress(self, today: pd.Timestamp) -> float:
        """Share of the window's calendar days that have passed."""
        total = (self.end - self.start).days + 1
        done = (min(today, self.end) - self.start).days + 1
        return max(0.0, min(1.0, done / total))


def window_for(month) -> Window:
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    before = month - pd.offsets.MonthBegin(1)
    two_before = month - pd.offsets.MonthBegin(2)
    return Window(
        month=month,
        start=last_weekday(two_before.year, two_before.month, THURSDAY) + pd.Timedelta(days=1),
        end=last_weekday(before.year, before.month, THURSDAY),
    )
