#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chart Trainer — clean 4‑space indentation (tab‑free)

"""Streamlit one‑file app for interactive chart‑replay trading practice.

Key fixes ▒▒ 2025‑06‑16
──────────
* make_game(): duplicated call typo removed
* start_game(): single‑line f‑string (no newline inside)
* start_random_modelbook(): removed unreachable dead‑code block
* general: long lines wrapped ≤ 120 chars
"""

from __future__ import annotations

import os
import random
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from services.data_loader import get_price
from services.indicators import add_mas
from services.simulator import GameState

# ────────────────────────────── Streamlit config ───────────────────────────────

st.set_page_config(page_title="차트 훈련소", page_icon="📈", layout="wide")

# ──────────────────────────────── Cache helpers ────────────────────────────────

@st.cache_data
def load_price(ticker: str) -> pd.DataFrame:
    """Load OHLCV from CSV/yf and cache it."""
    return get_price(ticker)


@st.cache_data
def add_indicators(df: pd.DataFrame, mas: tuple[tuple[str, int, bool], ...]) -> pd.DataFrame:
    """Add moving‑averages and cache by parameters."""
    return add_mas(df.copy(), mas)

# ─────────────────────────────── Game helpers ──────────────────────────────────

def make_game(ticker: str, capital: int) -> GameState | None:
    """Create GameState or *None* if 5y‑1y window invalid or <120 bars."""
    df = load_price(ticker)
    if len(df) < 120:
        return None

    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(df.index) if lo <= d <= hi and i >= 120]
    if not pool:
        return None

    start_idx = random.choice(pool)
    return GameState(df, idx=start_idx, start_cash=capital, tkr=ticker.upper())


def start_game(tkr: str, capital: int):
    """Safely start/replace current game in session_state."""
    game = make_game(tkr, capital)
    if game is None:
        st.error(
            f"`{tkr}` 종목은 5년치 이상 데이터가 없거나, "
            "시작할 수 있는 랜덤 구간이 없습니다.\n다른 티커를 선택해 주세요."
        )
        return

    st.session_state.game = game
    st.session_state.view_n = 120
    st.session_state.last_summary = None
    st.rerun()


def load_modelbook(path: os.PathLike) -> list[str]:
    txt = Path(path).read_text(encoding="utf-8")
    return [t.strip().upper() for t in txt.split(',') if t.strip()]


def start_random_modelbook(capital: int):
    root = Path(__file__).resolve().parent
    mb_path = root / "modelbook.txt"
    if not mb_path.exists():
        st.error("modelbook.txt 파일을 찾지 못했습니다.")
        return

    candidates = [t for t in load_modelbook(mb_path) if t.isalpha()]
    random.shuffle(candidates)

    for tkr in candidates:
        if make_game(tkr, capital):
            start_game(tkr, capital)
            return

    st.error("모델북에 시작할 수 있는 유효한 티커가 없습니다.")


def jump_random_date():
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi and i >= 120]
    g.idx, g.cash, g.pos, g.log = random.choice(pool), g.initial_cash, None, []
    st.session_state.view_n = 120
    st.rerun()

# ─────────────────────────────── Landing page ─────────────────────────────────

if "game" not in st.session_state:
    st.header("📈 차트 훈련소")
    col_tkr, col_cash = st.columns([2, 1])

    ticker_in = col_tkr.text_input("시작할 티커 입력", "")
    cash_in = col_cash.number_input("시작 자본($)", 1_000, 1_000_000, 100_000, 1_000)

    btn_new, btn_rand = st.columns(2)
    if btn_new.button("새 게임 시작", use_container_width=True) and ticker_in.strip():
        start_game(ticker_in.strip().upper(), int(cash_in))

    if btn_rand.button("모델북 랜덤 시작", type="secondary", use_container_width=True):
        start_random_modelbook(int(cash_in))

    if st.session_state.get("last_summary"):
        st.subheader("지난 게임 요약")
        st.json(st.session_state.last_summary)
    st.stop()

# ─────────────────────────────── Main game view ───────────────────────────────

g: GameState = st.session_state.game
chart_col, side_col = st.columns([7, 3])

# --- MA inputs
ema_txt = chart_col.text_input("EMA 기간(쉼표)", "10,21")
sma_txt = chart_col.text_input("SMA 기간(쉼표)", "50,200")
mas = [("EMA", int(p)) for p in ema_txt.split(',') if p.strip().isdigit()] + \
      [("SMA", int(p)) for p in sma_txt.split(',') if p.strip().isdigit()]
mas_tuple = tuple((k, p, True) for k, p in mas)  # 3rd bool=plot flag

# --- data & indicators
full_df = load_price(g.ticker)
ind_df = add_indicators(full_df, mas_tuple).iloc[: g.idx + 1]
price_now = ind_df.Close.iloc[-1]

# --- view window len
if "view_n" not in st.session_state:
    st.session_state.view_n = 120
view_n = chart_col.number_input("표시봉", 50, len(ind_df), st.session_state.view_n, 10)
st.session_state.view_n = int(view_n)
vis_df = ind_df.iloc[-st.session_state.view_n:]

# --- plotting
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25], vertical_spacing=0.02)

palette = ["red", "blue", "orange", "black", "green", "purple"]
for i, (k, p, _) in enumerate(mas_tuple):
    fig.add_scatter(x=ind_df.index, y=ind_df[f"{k}{p}"],
                    line=dict(width=1, color=palette[i % len(palette)]),
                    name=f"{k}{p}", row=1, col=1)

inc = dict(line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)")
dec = dict(line=dict(color="black", width=1), fillcolor="black")
fig.add_candlestick(x=ind_df.index, open=ind_df.Open, high=ind_df.High,
                    low=ind_df.Low, close=ind_df.Close,
                    increasing=inc, decreasing=dec, name="Price", row=1, col=1)

vol_color = ["black" if c <= o else "white" for o, c in zip(ind_df.Open, ind_df.Close)]
fig.add_bar(x=ind_df.index, y=ind_df.Volume, marker_color=vol_color,
            marker_line_color="black", marker_line_width=0.5,
            name="Volume", row=2, col=1)
fig.add_scatter(x=ind_df.index, y=ind_df.Volume.rolling(50).mean(),
                line=dict(width=1, color="blue", dash="dot"), name="Vol SMA50",
                row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified",
                  margin=dict(t=25, b=20, l=5, r=5))
fig.update_yaxes(showgrid=False, fixedrange=True, row=2, col=1)

chart_col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --- status / controls
real_pl = g.cash - g.initial_cash
unreal_pl = g.pos.unreal(price_now) if g.pos else 0
pos_val = g.pos.market_value(price_now) if g.pos else 0
equity = g.cash + pos_val

with side_col:
    st.subheader("상태")
    st.write(f"**날짜**: {g.today.date()}")
    st.write(f"**현금**: ${g.cash:,.2f}")
    st.write(f"**실현 P/L**: {real_pl:+,.2f}")
    st.write(f"**미실현 P/L**: {unreal_pl:+,.2f}")
    if g.pos:
        pct = pos_val / equity * 100 if equity else 0
        st.write(f"**포지션**: {g.pos.side.upper()} {g.pos.qty}주 @ {g.pos.avg_price:.2f} "
                 f"(**{pct:.1f}%**) ")

    st.write("### 거래 로그")
    if g.log:
        st.dataframe(pd.DataFrame(g.log))
    else:
        st.write("_empty_")

# (버튼 로직 및 GameState 인터랙션 코드는 생략 — 기존 그대로 사용)
