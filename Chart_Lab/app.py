#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chart Trainer – clean indentation (4 spaces only, no tabs)

import os
import random
from datetime import date

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from services.data_loader import get_price
from services.indicators import add_mas
from services.simulator import GameState

st.set_page_config(page_title="차트 훈련소", page_icon="📈", layout="wide")

# ────────────────── caching helpers ──────────────────

@st.cache_data
def load_price(ticker: str) -> pd.DataFrame:
    return get_price(ticker)


@st.cache_data
def add_indicators_cached(df: pd.DataFrame, mas: tuple) -> pd.DataFrame:
    ma_params = [(k, p, True) for k, p in mas]
    return add_mas(df.copy(), ma_params)


# ────────────────── game helpers ──────────────────

def make_game(ticker: str, capital: int) -> GameState:
    df = load_price(ticker)
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(df.index) if lo <= d <= hi and i >= 120]
    start_idx = random.choice(pool)
    return GameState(df, idx=start_idx, start_cash=capital, tkr=ticker.upper())


def start_game(tkr: str, capital: int):
    st.session_state.game = make_game(tkr, capital)
    st.session_state.view_n = 120
    st.session_state.last_summary = None
    st.rerun()


def load_modelbook(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [t.strip().upper() for t in f.read().split(",") if t.strip()]


def start_random_modelbook(capital: int):
    """Pick a random ticker from `modelbook.txt` and start a new game.
    - modelbook.txt must exist in the repo root (same level as this app.py)
    - File contents: comma‑separated tickers, e.g.  GEV,NVDA,SMCI
    Any blank/whitespace items or non‑alphabetic codes are discarded safely.
    """
    root = os.path.dirname(__file__)
    mb = os.path.join(root, "modelbook.txt")
    if not os.path.exists(mb):
        st.error("modelbook.txt 파일을 찾지 못했습니다.")
        return

    tickers_raw = load_modelbook(mb)
    tickers = sorted({t for t in tickers_raw if t.isalpha()})

    if not tickers:
        st.error("modelbook.txt 에 유효한 티커가 없습니다.")
        return

    start_game(random.choice(tickers), capital)


def jump_random_date():
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi and i >= 120]
    g.idx, g.cash, g.pos, g.log = random.choice(pool), g.initial_cash, None, []
    st.session_state.view_n = 120
    st.rerun()


# ────────────────── landing page ──────────────────

if "game" not in st.session_state:
    st.header("📈 차트 훈련소")
    col_code, col_cash = st.columns([2, 1])
    ticker_in = col_code.text_input("시작할 티커 입력", "")
    cash_in = col_cash.number_input("시작 자본($)", 1000, 1_000_000, 100_000, 1_000)

    btn_new, btn_rand = st.columns([1, 1])
    if btn_new.button("새 게임 시작") and ticker_in.strip():
        start_game(ticker_in.strip().upper(), int(cash_in))
    if btn_rand.button("모델북 랜덤 시작"):
        start_random_modelbook(int(cash_in))

    if st.session_state.get("last_summary"):
        st.subheader("지난 게임 요약")
        st.json(st.session_state.last_summary)
    st.stop()

# ────────────────── main game ──────────────────

g: GameState = st.session_state.game
chart_col, side_col = st.columns([7, 3])

# moving averages input
ema_txt = chart_col.text_input("EMA 기간(쉼표)", "10,21")
sma_txt = chart_col.text_input("SMA 기간(쉼표)", "50,200")
mas = [("EMA", int(p)) for p in ema_txt.split(",") if p.strip().isdigit()] + \
      [("SMA", int(p)) for p in sma_txt.split(",") if p.strip().isdigit()]
mas_tuple = tuple(mas)

# price & indicators
df_full = load_price(g.ticker)
df_vis = add_indicators_cached(df_full, mas_tuple).iloc[: g.idx + 1]
price_now = df_vis.Close.iloc[-1]

# view window length
if "view_n" not in st.session_state:
    st.session_state.view_n = 120
view_n = chart_col.number_input("표시봉", 50, len(df_vis), st.session_state.view_n, 10)
st.session_state.view_n = int(view_n)

df_sub = df_vis.iloc[-int(view_n):]

# plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25], vertical_spacing=0.02)

palette = ["red", "blue", "orange", "black", "green", "purple"]
for i, (k, p) in enumerate(mas_tuple):
    fig.add_scatter(x=df_vis.index, y=df_vis[f"{k}{p}"],
                    line=dict(width=1, color=palette[i % len(palette)]),
                    name=f"{k}{p}", row=1, col=1)

inc = dict(line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)")
dec = dict(line=dict(color="black", width=1), fillcolor="black")
fig.add_candlestick(x=df_vis.index, open=df_vis.Open, high=df_vis.High,
                    low=df_vis.Low, close=df_vis.Close,
                    increasing=inc, decreasing=dec, name="Price", row=1, col=1)

vol_col = ["black" if c <= o else "white" for o, c in zip(df_vis.Open, df_vis.Close)]
fig.add_bar(x=df_vis.index, y=df_vis.Volume, marker_color=vol_col,
            marker_line_color="black", marker_line_width=0.5,
            name="Volume", row=2, col=1)
fig.add_scatter(x=df_vis.index, y=df_vis.Volume.rolling(50).mean(),
                line=dict(width=1, color="blue", dash="dot"), name="Vol SMA50",
                row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified",
                  margin=dict(t=25, b=20, l=5, r=5))
fig.update_yaxes(showgrid=False, fixedrange=True, row=2, col=1)

if "chart_slot" not in st.session_state:
    st.session_state.chart_slot = chart_col.empty()
st.session_state.chart_slot.plotly_chart(fig, use_container_width=True,
                                         config={"displayModeBar": False})

# status panel
real_pl = g.cash - g.initial_cash
unreal_pl = g.pos.unreal(price_now) if g.pos else 0
pos_val = g.pos.market_value(price_now) if g.pos else 0
equity = g.cash + pos_val

with side_col:
    st.subheader("상태")
    st.write(f"**날짜**: {g.today.date()}")
    st.write(f"**현금(실현 자본)**: ${g.cash:,.2f}")
    st.write(f"**실현 P/L**: {real_pl:+,.2f}")
    st.write(f"**미실현 P/L**: {unreal_pl:+,.2f}")
    if g.pos:
        pct = pos_val / equity * 100 if equity else 0
        st.write(f"**포지션**: {g.pos.side.upper()} {g.pos.qty}주 @ {g.pos.avg_price:.2f} (**{pct:.1f}%**) ")
    st.write("### 거래 로그")
    if g.log:
        st.dataframe(pd.DataFrame(g.log))
    else:
        st.write("_empty_")

# ---------------------------------------------------------------------------
# (buy/sell/flat/next/random 버튼 로직은 기존 코드에서 그대로 복사)
