#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chart Trainer — clean 4‑space indentation (tab‑free)
"""Streamlit one‑file app for interactive chart‑replay trading practice.

Key fixes ▒▒ 2025‑06‑16 ②
──────────
* **TypeError on start_random_modelbook** – now caches the first *valid* GameState, no double make_game()
* f‑string back‑tick → plain quote (SyntaxError)
* extra safety: load_price() returns **None** → graceful error
* start_cash & equity stats clarified
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
def load_price(ticker: str) -> pd.DataFrame | None:
    """Load OHLCV; *None* if not found / empty."""
    try:
        df = get_price(ticker)
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except FileNotFoundError:
        return None


@st.cache_data
def add_indicators(df: pd.DataFrame, mas: tuple[tuple[str, int, bool], ...]) -> pd.DataFrame:
    """Add moving‑averages and cache by parameters."""
    return add_mas(df.copy(), mas)

# ─────────────────────────────── Game helpers ──────────────────────────────────

def make_game(ticker: str, capital: int) -> GameState | None:
    """Create GameState or *None* if data window invalid."""
    df = load_price(ticker)
    if df is None or len(df) < 120:
        return None

    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(df.index) if lo <= d <= hi and i >= 120]
    if not pool:
        return None

    start_idx = random.choice(pool)
    return GameState(df, idx=start_idx, start_cash=capital, tkr=ticker.upper())


def _cannot_start(tkr: str):
    st.error(
        f"'{tkr}' 종목은 5년치 이상 데이터가 없거나, 시작할 수 있는 랜덤 구간이 없습니다."  # noqa: E501
    )


def start_game(tkr: str, capital: int):
    """Safely start/replace current game in session_state."""
    game = make_game(tkr, capital)
    if game is None:
        _cannot_start(tkr)
        return False

    st.session_state.game = game
    st.session_state.view_n = 120
    st.session_state.last_summary = None
    st.rerun()
    return True  # pragma: no cover – only reached in dev/CLI


def load_modelbook(path: os.PathLike) -> list[str]:
    txt = Path(path).read_text(encoding="utf-8")
    return [t.strip().upper() for t in txt.split(',') if t.strip()]


def start_random_modelbook(capital: int):
    root = Path(__file__).resolve().parent
    mb_path = root / "modelbook.txt"
    if not mb_path.exists():
        st.error("📄 modelbook.txt 파일이 없습니다 – 업로드 또는 추가해 주세요.")
        return

    tickers = load_modelbook(mb_path)
    random.shuffle(tickers)

    for tkr in tickers:
        game = make_game(tkr, capital)
        if game:
            st.session_state.game = game
            st.session_state.view_n = 120
            st.session_state.last_summary = None
            st.rerun()
            return

    st.error("모델북에 시작할 수 있는 유효한 티커가 없습니다.")


def jump_random_date():
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi and i >= 120]
    if pool:
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

g: GameState = st.session_state.game  # type: ignore[attr-defined]
chart_col, side_col = st.columns([7, 3])

# --- MA inputs
ema_txt = chart_col.text_input("EMA 기간(쉼표)", "10,21")
sma_txt = chart_col.text_input("SMA 기간(쉼표)", "50,200")
mas = [("EMA", int(p)) for p in ema_txt.split(',') if p.strip().isdigit()] + \
      [("SMA", int(p)) for p in sma_txt.split(',') if p.strip().isdigit()]
mas_tuple = tuple((k, p, True) for k, p in mas)  # 3rd bool=plot flag

# --- data & indicators
df_price = load_price(g.ticker)
if df_price is None:
    st.error("데이터 로드 실패 – 다시 시작해 주세요.")
    st.stop()

ind_df = add_indicators(df_price, mas_tuple).iloc[: g.idx + 1]
price_now = ind_df.Close.iloc[-1]

# --- view window len
if "view_n" not in st.session_state:
    st.session_state.view_n = 120
view_n = chart_col.number_input("표시봉", 50, len(ind_df), st.session_state.view_n, 10)
st.session_state.view_n = int(view_n)
vis_df = ind_df.iloc[-st.session_state.view_n:]

# --- plotting (unchanged) ...
# (이하 기존 plot / 버튼 로직 동일 – 생략)
