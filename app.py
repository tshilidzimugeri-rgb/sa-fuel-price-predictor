"""SA Fuel Price Predictor - Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

# One BLAS thread is plenty here and avoids OpenBLAS running out of memory
# on small hosts.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fuelcast import market, model

st.set_page_config(page_title="SA Fuel Price Predictor", page_icon="⛽", layout="wide")

INK, MUTED, UP, DOWN, ACCENT, GRID = "#0f172a", "#64748b", "#dc2626", "#16a34a", "#1d4ed8", "#e2e8f0"

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.5rem; max-width: 1200px;}}
  h1, h2, h3 {{color: {INK}; letter-spacing: -0.01em;}}
  .eyebrow {{color: {ACCENT}; font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;}}
  .lead {{color: {MUTED}; font-size: 1.05rem; margin-top: -.5rem;}}
  .card {{border: 1px solid {GRID}; border-radius: 12px; padding: 1.1rem 1.25rem; background: #fff; height: 100%;}}
  .card .label {{color: {MUTED}; font-size: .85rem;}}
  .card .value {{font-size: 2.1rem; font-weight: 700; line-height: 1.2;}}
  .card .sub {{color: {MUTED}; font-size: .85rem;}}
  .pill {{display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem; font-weight: 600;
          background: #eff6ff; color: {ACCENT}; margin-right: .3rem;}}
  .pill.done {{background: #f1f5f9; color: {MUTED};}}
  .bar {{height: 6px; background: {GRID}; border-radius: 3px; margin: .6rem 0 .3rem;}}
  .bar > div {{height: 6px; background: {ACCENT}; border-radius: 3px;}}
  footer, #MainMenu {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching oil and rand prices…")
def load_market():
    return market.load()


@st.cache_data(show_spinner=False)
def load_model_inputs(_market: pd.DataFrame, stamp: str):
    prices = model.load_prices()
    out = {}
    for fuel in model.FUELS:
        table = model.build_table(prices, _market, fuel)
        bt = model.backtest(table)
        out[fuel] = (table, bt, model.metrics(bt))
    return prices, out


mkt, source = load_market()
prices, inputs = load_model_inputs(mkt, f"{mkt.index.max()}-{len(mkt)}")
today = mkt.index.max()

# ---- Sidebar: what-if -------------------------------------------------------
with st.sidebar:
    st.markdown("### What-if scenario")
    st.caption("Fills the rest of each open review window with these prices instead of today's.")
    oil_shift = st.slider("International product prices vs today", -30, 30, 0, format="%d%%") / 100
    rand_now = float(mkt["usdzar"].iloc[-1])
    rand_level = st.slider("Rand per US dollar", round(rand_now * 0.85, 2), round(rand_now * 1.15, 2),
                           round(rand_now, 2), step=0.05)
    st.divider()
    st.caption(f"Market data: **{source}**, last close {today:%d %b %Y}.  \n"
               f"Official prices up to {prices['effective_date'].max():%B %Y}.")

# ---- Header -----------------------------------------------------------------
st.markdown('<div class="eyebrow">South Africa · Inland (Gauteng) prices</div>', unsafe_allow_html=True)
st.title("Fuel Price Predictor")
st.markdown('<p class="lead">Estimates the next official petrol and diesel adjustment from international '
            'product prices and the rand, weeks before the Department of Mineral and Petroleum Resources '
            'announces it.</p>', unsafe_allow_html=True)

forecasts = {fuel: model.nowcast(prices, mkt, inputs[fuel][0], inputs[fuel][1], fuel, today,
                                 oil_shift=oil_shift, rand_level=rand_level if rand_level != round(rand_now, 2) else None)
             for fuel in model.FUELS}


def rands(cents: float, sign: bool = True) -> str:
    return f"{'+' if sign and cents >= 0 else '−' if sign else ''}R{abs(cents) / 100:.2f}"


def card(fc: model.Forecast) -> str:
    colour = UP if fc.change > 0 else DOWN
    status = ('<span class="pill done">Window closed · awaiting announcement</span>' if fc.progress >= 1
              else f'<span class="pill">Window {fc.progress:.0%} complete</span>')
    return f"""
    <div class="card">
      <div class="label">{model.FUELS[fc.fuel]['name']} · from {fc.window.effective:%a %d %b %Y}</div>
      <div class="value" style="color:{colour}">{rands(fc.change)}<span style="font-size:1rem;color:{MUTED}"> /litre</span></div>
      <div class="sub">80% range {rands(fc.low)} to {rands(fc.high)}</div>
      <div class="bar"><div style="width:{fc.progress * 100:.0f}%"></div></div>
      {status}
      <div class="sub" style="margin-top:.6rem">Pump price would move from
        <b>R{fc.current_price / 100:.2f}</b> to <b>R{(fc.current_price + fc.change) / 100:.2f}</b></div>
    </div>"""


# ---- Forecast cards ---------------------------------------------------------
months = [fc.window for fc in forecasts["petrol"]]
for i, window in enumerate(months):
    st.subheader(f"{window.label} adjustment")
    st.caption(f"Review window {window.start:%d %b} – {window.end:%d %b %Y}")
    cols = st.columns(2)
    for col, fuel in zip(cols, model.FUELS):
        col.markdown(card(forecasts[fuel][i]), unsafe_allow_html=True)
    st.write("")

tab_why, tab_perf, tab_hist, tab_how = st.tabs(["What's driving it", "Model accuracy", "Price history", "How it works"])

layout = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=380,
              font=dict(family="Inter, Segoe UI, sans-serif", color=INK),
              legend=dict(orientation="h", y=-0.15))

with tab_why:
    left, right = st.columns([1, 1])
    with left:
        rows = []
        for fuel in model.FUELS:
            for fc in forecasts[fuel]:
                rows.append({"fuel": model.FUELS[fuel]["name"], "month": fc.window.label,
                             "Oil / product price": fc.oil_effect / 100, "Rand / dollar": fc.rand_effect / 100,
                             "Other (levies, seasonal)": (fc.change - fc.oil_effect - fc.rand_effect) / 100})
        parts = pd.DataFrame(rows)
        fig = go.Figure()
        for name, colour in [("Oil / product price", "#0f766e"), ("Rand / dollar", "#7c3aed"),
                             ("Other (levies, seasonal)", "#94a3b8")]:
            fig.add_bar(name=name, x=[f"{f}<br>{m}" for f, m in zip(parts.fuel, parts.month)], y=parts[name],
                        marker_color=colour, hovertemplate="%{y:+.2f} R/l<extra>" + name + "</extra>")
        fig.update_layout(**layout, barmode="relative", title="Where the expected change comes from (R/litre)")
        st.plotly_chart(fig, width="stretch")
    with right:
        recent = mkt.loc[mkt.index >= today - pd.Timedelta(days=150)]
        fig = go.Figure()
        fig.add_scatter(x=recent.index, y=recent["brent_usd_bbl"], name="Brent (US$/bbl)", line=dict(color="#0f766e"))
        fig.add_scatter(x=recent.index, y=recent["usdzar"], name="Rand per US$", yaxis="y2", line=dict(color="#7c3aed"))
        for window in months:
            fig.add_vrect(x0=window.start, x1=min(window.end, today), fillcolor=ACCENT, opacity=0.06, line_width=0)
        fig.update_layout(**layout, title="Brent and the rand (shaded: open review windows)",
                          yaxis=dict(title="US$/bbl"), yaxis2=dict(title="R/$", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(fig, width="stretch")

with tab_perf:
    st.markdown("Every month since 2016 was predicted by a model trained **only on earlier months**, "
                "so these numbers are what the model would really have achieved.")
    for fuel in model.FUELS:
        _, bt, m = inputs[fuel]
        st.markdown(f"#### {model.FUELS[fuel]['name']}")
        c = st.columns(4)
        c[0].metric("Average error", f"{m['mae']:.0f} c/l")
        c[1].metric("Naive guess error", f"{m['naive_mae']:.0f} c/l", help="Error if you always guessed 'no change'.")
        c[2].metric("Direction right", f"{m['direction']:.0%}")
        c[3].metric("Variance explained (R²)", f"{m['r2']:.2f}")
        fig = go.Figure()
        fig.add_bar(x=bt["month"], y=bt["actual"] / 100, name="Actual change", marker_color="#cbd5e1")
        fig.add_scatter(x=bt["month"], y=bt["predicted"] / 100, name="Predicted", mode="lines+markers",
                        line=dict(color=ACCENT, width=2), marker=dict(size=4))
        fig.update_layout(**layout, title="Monthly change, R/litre", yaxis_title="R/litre")
        st.plotly_chart(fig, width="stretch")
    st.info("The biggest misses line up with policy decisions rather than markets: the 2022 and 2026 temporary "
            "fuel levy relief, slate levy changes and the annual April levy increase.")

with tab_hist:
    fig = go.Figure()
    fig.add_scatter(x=prices["effective_date"], y=prices["petrol_95"] / 100, name="Petrol 95 ULP",
                    line=dict(color=UP, width=2), line_shape="hv")
    fig.add_scatter(x=prices["effective_date"], y=prices["diesel_005"] / 100, name="Diesel 0.05% (wholesale)",
                    line=dict(color=INK, width=2), line_shape="hv")
    fig.update_layout(**layout, title="Official Gauteng fuel prices since 2012 (R/litre)", yaxis_title="R/litre")
    st.plotly_chart(fig, width="stretch")
    show = prices[["effective_date", "petrol_95", "diesel_005"]].sort_values("effective_date", ascending=False).copy()
    show.columns = ["Effective", "Petrol 95 (c/l)", "Diesel 0.05% (c/l)"]
    st.dataframe(show, hide_index=True, width="stretch", height=320)

with tab_how:
    st.markdown("""
**The rule.** South Africa's fuel prices are regulated. On the first Wednesday of each month the price moves
by the change in the *Basic Fuel Price*, which tracks international product prices converted to rand and averaged
over a review window running from the last Friday of one month to the last Thursday of the next.

**The model.** For every adjustment since 2012 the app averages two daily series over that window:

* RBOB gasoline futures (petrol) and NY Harbor ULSD futures (diesel), in US$ per gallon
* the rand/dollar exchange rate

and converts them to cents per litre. A robust (Huber) regression links the month-on-month change in that
rand price to the actual change at the pump, with a term for April, when the annual fuel levy increase lands.

**The nowcast.** For a review window that is still open, the days that have passed use real prices and the days
still to come use today's price, or your scenario from the sidebar. The 80% range comes from the backtest errors and
widens while more of the window is unknown.

**Limits.** Policy changes (levy relief, slate levy, margins) are not in the market data, so the model cannot see
them coming. Coastal prices are usually about 80c/l lower than inland.

**Data.** Official prices: Fuels Industry Association of SA and DMPR announcements. Market data: Yahoo Finance.
""")
    st.caption("Not financial advice. Built by Tshilidzi Mugeri.")
