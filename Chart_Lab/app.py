# app.py (v3 - 최종 통합 버전)
import streamlit as st
import pandas as pd
import random
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 모듈화된 버전을 정확히 임포트합니다.
from services.data_loader import get_price
from services.indicators import add_mas
from services.simulator import GameState, Position # Position도 임포트해두면 좋습니다.

st.set_page_config(page_title="차트 훈련소", page_icon="📈", layout="wide")

# --- 상수 정의 ---
PAD, MARGIN = 20, 0.05  # x축 오른쪽 공백, y축 여유 비율

# ╭───────────────── 캐싱 헬퍼 ─────────────────╮
@st.cache_data
def load_cached_price(ticker: str):
    """get_price를 직접 캐싱하여 Streamlit의 위젯 ID 문제를 회피"""
    return get_price(ticker)

@st.cache_data
def add_cached_indicators(df: pd.DataFrame, mas_tuple: tuple):
    """이동평균 계산 캐시. mas_tuple 은 (('EMA',10), ('SMA',50), ...) 형태"""
    # 이전 버전의 add_mas는 ('EMA', 10, True) 형태의 입력을 기대하므로 맞춰줍니다.
    mas_settings = [(kind, period, True) for kind, period in mas_tuple]
    return add_mas(df.copy(), mas_settings)

# ╭───────────────── 게임 생성/시작 ─────────────────╮
# GameState 생성 로직을 우리가 수정한 버전에 맞춥니다.
def create_game(tkr: str, capital: int) -> GameState | None:
    df = load_cached_price(tkr)
    if df is None or len(df) < 120:
        st.error(f"'{tkr}' 종목 데이터를 불러오지 못했거나 데이터가 너무 적습니다.")
        return None

    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(df.index) if lo <= d <= hi and i >= 120]
    if not pool:
        st.error(f"'{tkr}' 종목에서 시작 가능한 랜덤 구간을 찾지 못했습니다.")
        return None

    return GameState(df, tkr=tkr, idx=random.choice(pool), start_cash=capital)

def start_game(tkr: str, capital: int):
    game = create_game(tkr, capital)
    if game:
        st.session_state.game = game
        st.session_state.last_summary, st.session_state.view_n = None, 120
        st.rerun()

def start_random_modelbook(capital: int):
    root = os.path.dirname(__file__)
    # modelbook.txt 경로를 좀 더 안정적으로 찾습니다.
    path = os.path.join(root, "modelbook.txt")
    if not os.path.exists(path):
        path = os.path.join(root, "..", "modelbook.txt") # 상위 폴더도 확인
    if not os.path.exists(path):
        st.error("modelbook.txt 파일을 찾지 못했습니다."); return
    
    with open(path, "r", encoding="utf-8") as f:
        tickers = [t.strip().upper() for t in f.read().split(",") if t.strip()]
    if not tickers:
        st.error("modelbook.txt 에 티커가 없습니다"); return
    
    random.shuffle(tickers)
    for tkr in tickers:
        game = create_game(tkr, capital)
        if game:
            st.session_state.game = game
            st.session_state.last_summary, st.session_state.view_n = None, 120
            st.rerun()
            return
    st.error("모델북에 시작 가능한 유효한 티커가 없습니다.")


def jump_random_date():
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi and i >= 120]
    if pool:
        # GameState의 초기 자본(initial_cash)을 사용합니다.
        g.idx, g.cash, g.pos, g.log = random.choice(pool), g.initial_cash, None, []
        st.session_state.view_n = 120
        st.rerun()

# ╭───────────────── 첫 랜딩 ─────────────────╮
if "game" not in st.session_state:
    st.header("📈 차트 훈련소")
    col_tkr, col_cash = st.columns([2, 1])
    ticker_in = col_tkr.text_input("시작할 티커 입력", "AAPL")
    cash_in = col_cash.number_input("시작 자본($)", 1_000, 1_000_000, 10_000, 1_000)

    btn_new, btn_rand = st.columns(2)
    if btn_new.button("새 게임 시작", use_container_width=True) and ticker_in.strip():
        start_game(ticker_in.strip().upper(), int(cash_in))
    if btn_rand.button("모델북 랜덤 시작", type="secondary", use_container_width=True):
        start_random_modelbook(int(cash_in))
    
    if st.session_state.get("last_summary"):
        st.subheader("지난 게임 요약")
        st.json(st.session_state.last_summary) # st.write 대신 st.json으로 보기 좋게
    st.stop()

# ╭───────────────── 게임 화면 ─────────────────╮
g: GameState = st.session_state.game
chart_col, side_col = st.columns([7, 3])

# --- 이동평균 입력 ---
ema_in = chart_col.text_input("EMA 기간(쉼표)", "10,21")
sma_in = chart_col.text_input("SMA 기간(쉼표)", "50,200")
mas_input = [("EMA", int(p)) for p in ema_in.split(",") if p.strip().isdigit()] + \
            [("SMA", int(p)) for p in sma_in.split(",") if p.strip().isdigit()]
mas_tuple = tuple(mas_input)  # 캐시 key 용

# --- 차트 그리기 ---
# 이 부분은 보내주신 '잘 작동하던' 코드의 복잡하지만 강력한 로직을 그대로 가져옵니다.
df_full = g.df
visible_df_with_ma = add_cached_indicators(df_full, mas_tuple).iloc[:g.idx + 1]

# 거래 가능한 날만 필터링 (NaN이나 거래량 0인 날 제외)
df_trade = (visible_df_with_ma.dropna(subset=["Open", "High", "Low", "Close"])
                              .loc[visible_df_with_ma.Volume > 0]
                              .assign(i=lambda d: range(len(d)))) # 정수 인덱스 'i' 생성

if df_trade.empty:
    st.error("표시할 데이터가 없습니다. 게임을 다시 시작해주세요.")
    st.stop()

price_now = df_trade.Close.iloc[-1]

# --- 표시할 캔들 수 (Autoscale) ---
if "view_n" not in st.session_state: st.session_state.view_n = 120
view_n_input = chart_col.number_input("표시봉", 50, len(df_trade), st.session_state.view_n, 10)
if int(view_n_input) != st.session_state.view_n:
    st.session_state.view_n = int(view_n_input)
view_n = st.session_state.view_n

start_i = df_trade.i.iloc[max(0, len(df_trade) - view_n)]
end_i = df_trade.i.iloc[-1]

# --- 차트 Y축 범위 자동계산 ---
sub = df_trade[df_trade.i >= start_i]
ma_cols = [f"{k}{p}" for k, p in mas_tuple]
ymin = sub[["Low"] + ma_cols].min().min()
ymax = sub[["High"] + ma_cols].max().max()
span = ymax - ymin if ymax > ymin else 1
yrng = [ymin - span * MARGIN, ymax + span * MARGIN]

# --- 차트 객체 생성 ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.8, 0.2], vertical_spacing=0.02)

# 이동평균선 그리기
for k, p in mas_tuple:
    fig.add_scatter(x=df_trade.i, y=df_trade[f"{k}{p}"],
                    line=dict(width=1.5), name=f"{k}{p}", row=1, col=1)

# 캔들차트 그리기
fig.add_candlestick(x=df_trade.i, open=df_trade.Open, high=df_trade.High,
                    low=df_trade.Low, close=df_trade.Close, name="Price", row=1, col=1)

# 거래량 차트 그리기
fig.add_bar(x=df_trade.i, y=df_trade.Volume, name="Volume", row=2, col=1)

# --- 차트 레이아웃 업데이트 ---
tick_step = max(len(sub) // 10, 1) # x축 날짜가 너무 겹치지 않게 간격 조절
fig.update_layout(
    xaxis=dict(tickmode="array", tickvals=sub.i[::tick_step], ticktext=sub.index.strftime("%y-%m-%d")[::tick_step], tickangle=0),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(t=25, b=20, l=5, r=40)
)
fig.update_yaxes(range=yrng, row=1, col=1)
fig.update_xaxes(range=[start_i - 1, end_i + PAD]) # 시작점에 여백을 줘서 잘리지 않게

# 차트를 그릴 공간을 미리 만들고 내용만 업데이트 (깜빡임 방지)
if "chart_slot" not in st.session_state:
    st.session_state.chart_slot = chart_col.empty()
st.session_state.chart_slot.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ╭───────────────── 사이드바 (상태, 컨트롤) ─────────────────╮
with side_col:
    # --- 상태 정보 ---
    pos_val = g.pos.qty * price_now if g.pos else 0
    equity = g.cash + pos_val
    unreal = (price_now - g.pos.avg_price) * g.pos.qty if g.pos else 0
    st.subheader(f"종목: {g.ticker}")
    st.metric("현재 평가자산", f"${equity:,.2f}", f"${unreal:,.2f} 미실현")
    st.text(f"현금: ${g.cash:,.2f}")
    if g.pos:
        st.text(f"포지션: {g.pos.qty}주 @ ${g.pos.avg_price:,.2f}")

    st.markdown("---")
    
    # --- 매매 컨트롤 ---
    st.subheader("매매")
    amount = st.number_input("수량(주)", min_value=1, value=10, step=1)
    
    b_col, s_col = st.columns(2)
    if b_col.button("매수", use_container_width=True):
        if g.cash >= amount * price_now:
            g.buy(amount)
            st.rerun()
        else:
            st.warning("현금이 부족합니다.")
            
    if s_col.button("매도(청산)", use_container_width=True):
        if g.pos and g.pos.qty >= amount:
            g.sell(amount) # simulator에 sell/flat 로직이 필요
            st.rerun()
        else:
            st.warning("매도할 주식이 없습니다.")
            
    if st.button("전량 청산", use_container_width=True) and g.pos:
        g.flat()
        st.rerun()
    
    st.markdown("---")

    # --- 게임 컨트롤 ---
    st.subheader("게임 진행")
    n_col, j_col, r_col = st.columns(3)
    if n_col.button("▶ 다음", use_container_width=True):
        g.next_candle()
        st.rerun()
    if j_col.button("🎲 날짜 변경", use_container_width=True):
        jump_random_date()
    if r_col.button("📚 모델북", use_container_width=True):
        start_random_modelbook(int(cash_in if 'cash_in' in locals() else 10000))

    # --- 게임 종료 ---
    if st.button("게임 종료 & 결과 보기", type="primary", use_container_width=True):
        # 결과 요약 로직 (생략, 필요시 추가)
        st.session_state.pop("game")
        st.rerun()