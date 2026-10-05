"""Builds data/fuel_prices.csv: the official monthly Gauteng fuel prices.

Sources
- Fuels Industry Association of SA (fuelsindustry.org.za): yearly tables,
  2012 onwards.
- Petrol Pulse (petrolpulse.co.za): the last 14 months, used for months the
  FISA page has not published yet.

Run:  python scripts/build_prices.py
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

FISA_URL = "https://fuelsindustry.org.za/consumer-information/fuel-prices-current-past/"
PULSE_URL = "https://www.petrolpulse.co.za/fuel-price-history"
OUT = Path(__file__).resolve().parents[1] / "data" / "fuel_prices.csv"
# Cells the sources got wrong, checked against the DMPR media statement for
# that month: (effective date, column) -> cents per litre.
CORRECTIONS = {
    # Petrol Pulse repeats May's 95 ULP price; DMPR: +143 c/l from 3 June 2026.
    ("2026-06-03", "petrol_95"): 2806.0,
}
HEADERS = {"User-Agent": "Mozilla/5.0 (sa-fuel-price-predictor)"}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def to_cents(value) -> float | None:
    """FISA cells come as 1552, '1 376.00' or 215900 (the last is R21.59 with the
    decimal point lost). Everything is returned in cents per litre."""
    if pd.isna(value):
        return None
    text = re.sub(r"[^\d.]", "", str(value))
    if not text:
        return None
    cents = float(text)
    while cents > 10_000:  # no SA fuel has cost R100/l; the decimal was dropped
        cents /= 100
    while cents < 500:  # nor less than R5/l; a digit landed after the decimal
        cents *= 10
    return round(cents, 2)


def parse_fisa(html: str) -> pd.DataFrame:
    rows = []
    for table in pd.read_html(io.StringIO(html)):
        year = int(re.search(r"(\d{4})", str(table.columns[0])).group(1))
        labels = table.iloc[:, 0].astype(str).str.strip()
        gauteng_at = labels[labels.str.upper() == "GAUTENG"].index[0]
        section = table.loc[gauteng_at:]
        lab = section.iloc[:, 0].astype(str)
        petrol = section[lab.str.startswith("95 ULP")].iloc[0]
        diesel = section[lab.str.startswith("Diesel 0.05%")].iloc[0]
        for col in table.columns[1:]:
            match = re.match(r"(\d{1,2})\s+([A-Za-z]{3})", str(col))
            if not match:
                continue
            day, mon = int(match.group(1)), MONTHS[match.group(2).lower()]
            rows.append({
                "effective_date": pd.Timestamp(year, mon, day),
                "petrol_95": to_cents(petrol[col]),
                "diesel_005": to_cents(diesel[col]),
            })
    return pd.DataFrame(rows)


def parse_pulse(html: str) -> pd.DataFrame:
    table = pd.read_html(io.StringIO(html))[0]
    rows = []
    for _, r in table.iterrows():
        when = pd.to_datetime(re.sub(r"(\D+)(\d{4})", r"\1 \2", r["Effective Date"]), format="%B %Y")
        price = lambda cell: float(re.search(r"R(\d+\.\d+)", str(cell)).group(1)) * 100
        rows.append({"month": when, "petrol_95": price(r["95 ULP"]), "diesel_005": price(r["Diesel 0.05% *"])})
    return pd.DataFrame(rows)


def first_wednesday(month: pd.Timestamp) -> pd.Timestamp:
    day = month.replace(day=1)
    return day + pd.Timedelta(days=(2 - day.weekday()) % 7)


def main() -> None:
    fisa = parse_fisa(requests.get(FISA_URL, headers=HEADERS, timeout=60).text)
    pulse = parse_pulse(requests.get(PULSE_URL, headers=HEADERS, timeout=60).text)

    fisa["month"] = fisa["effective_date"].dt.to_period("M").dt.to_timestamp()
    newer = pulse[~pulse["month"].isin(fisa["month"])].copy()
    # Adjustments take effect on the first Wednesday of the month.
    newer["effective_date"] = newer["month"].map(first_wednesday)

    data = (pd.concat([fisa, newer], ignore_index=True)
              .dropna(subset=["petrol_95", "diesel_005"])
              .sort_values("effective_date")
              .drop_duplicates("month", keep="last"))

    # The two sources must agree where they overlap, or one has been misread.
    overlap = fisa.merge(pulse, on="month", suffixes=("", "_pulse"))
    gap = (overlap["petrol_95"] - overlap["petrol_95_pulse"]).abs().max()
    if gap > 1:
        raise SystemExit(f"FISA and Petrol Pulse disagree by {gap:.0f} c/l on 95 ULP")

    for (when, column), cents in CORRECTIONS.items():
        data.loc[data["effective_date"] == pd.Timestamp(when), column] = cents

    data["effective_date"] = data["effective_date"].dt.date
    OUT.parent.mkdir(exist_ok=True)
    data[["effective_date", "petrol_95", "diesel_005"]].to_csv(OUT, index=False)
    print(f"{len(data)} months, {data['effective_date'].min()} to {data['effective_date'].max()} -> {OUT}")


if __name__ == "__main__":
    main()
