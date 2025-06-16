# app.py (최종 통합본)
# 기능: 종목명 숨기기, 현재가 표시, 주문 비중(%) 표시 기능 추가

import streamlit as st
import pandas as pd
import random
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# services 폴더의 모듈들을 정확히 임포트합니다.
from services.data_loader import get_price
from services.indicators import add_mas
from services.simulator import GameState, Position

# ------------------------------ Streamlit 페이지 설정 ------------------------------
st.set_page_config(page_title="차트 훈련소", page_icon="📈", layout="wide")

# ----------------------------------- 상수 정의 -----------------------------------
PAD, MARGIN = 20, 0.05  # x축 오른쪽 공백, y축 여유 비율

# ---------------------------------- 캐싱 헬퍼 ----------------------------------
@st.cache_data
def load_cached_price(ticker: str) -> pd.DataFrame | None:
    """get_price 결과를 캐시하여 반복적인 파일 로딩 방지"""
    return get_price(ticker)

@st.cache_data
def add_cached_indicators(df: pd.DataFrame, mas_tuple: tuple) -> pd.DataFrame:
    """이동평균 계산 결과를 캐시. mas_tuple은 (('EMA',10), ('SMA',50), ...) 형태"""
    mas_settings = [(kind, period, True) for kind, period in mas_tuple]
    return add_mas(df.copy(), mas_settings)

# --------------------------------- 게임 생성/시작 ---------------------------------
def create_game(tkr: str, capital: int) -> GameState | None:
    """GameState 객체를 생성하고, 실패 시 None을 반환"""
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

    return GameState(df=df, tkr=tkr, idx=random.choice(pool), start_cash=capital)

def start_game(tkr: str, capital: int):
    """새 게임을 시작하고 세션 상태를 설정"""
    game = create_game(tkr, capital)
    if game:
        st.session_state.game = game
        st.session_state.last_summary = None
        st.session_state.view_n = 120
        st.rerun()

def start_random_modelbook(capital: int):
    """modelbook.txt에서 랜덤 티커로 게임 시작"""
    # modelbook.txt 경로를 좀 더 안정적으로 찾습니다.
    root = os.path.dirname(__file__)
    path = os.path.join(root, "..", "modelbook.txt")
    if not os.path.exists(path):
        path = os.path.join(root, "modelbook.txt")
    if not os.path.exists(path):
        st.error("modelbook.txt 파일을 찾지 못했습니다."); return
    
    with open(path, "r", encoding="utf-8") as f:
        tickers = [t.strip().upper() for t in f.read().split(",") if t.strip()]
    if not tickers:
        st.error("modelbook.txt에 티커가 없습니다"); return
    
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
    """같은 티커 내에서 랜덤한 날짜로 점프 (리셋)"""
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi and i >= 120]
    if pool:
        g.idx, g.cash, g.pos, g.log = random.choice(pool), g.initial_cash, None, []
        st.session_state.view_n = 120
        st.rerun()

# --------------------------------- 첫 랜딩 페이지 ---------------------------------
if "game" not in st.session_state:
    st.header("📈 차트 훈련소")
    
    # --- 결과 표시 부분 ---
    if st.session_state.get("last_summary"):
        st.markdown("---")
        summary = st.session_state.last_summary
        # [수정] 결과 분석에 종목명 표시
        st.subheader(f"📊 지난 게임 성과 분석: **{summary.get('종목', '???')}**")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("최종 순손익", summary.get("최종 순손익", "N/A"))
        m_col2.metric("승률", summary.get("승률", "N/A"))
        m_col3.metric("손익비", summary.get("손익비 (Profit Factor)", "N/A"))
        
        with st.expander("상세 결과 보기"):
            st.json(summary)
        st.markdown("---")
    
    # --- 새 게임 시작 UI ---
    col_tkr, col_cash = st.columns([2, 1])
    ticker_in = col_tkr.text_input("시작할 티커 입력", "AAPL")
    cash_in = col_cash.number_input("시작 자본($)", 1_000, 1_000_000, 10_000, 1_000)

    btn_new, btn_rand = st.columns(2)
    if btn_new.button("새 게임 시작", use_container_width=True) and ticker_in.strip():
        start_game(ticker_in.strip().upper(), int(cash_in))
    if btn_rand.button("모델북 랜덤 시작", type="secondary", use_container_width=True):
        start_random_modelbook(int(cash_in))
    
    st.stop()

# ---------------------------------- 메인 게임 화면 ----------------------------------
g: GameState = st.session_state.game
chart_col, side_col = st.columns([7, 3])

# --- 이동평균 입력 ---
with chart_col:
    ma_cols = st.columns(2)
    ema_in = ma_cols[0].text_input("EMA 기간(쉼표)", "10,21")
    sma_in = ma_cols[1].text_input("SMA 기간(쉼표)", "50,200")
mas_input = [("EMA", int(p)) for p in ema_in.split(",") if p.strip().isdigit()] + \
            [("SMA", int(p)) for p in sma_in.split(",") if p.strip().isdigit()]
mas_tuple = tuple(mas_input)

# --- 데이터 준비 ---
df_full = g.df
visible_df_with_ma = add_cached_indicators(df_full, mas_tuple).iloc[:g.idx + 1]

df_trade = (visible_df_with_ma.dropna(subset=["Open", "High", "Low", "Close"])
                              .loc[visible_df_with_ma.Volume > 0]
                              .assign(i=lambda d: range(len(d))))

if df_trade.empty:
    st.error("표시할 데이터가 없습니다. 게임을 다시 시작해주세요.")
    st.stop()

price_now = df_trade.Close.iloc[-1]

# --- 표시할 캔들 수 ---
if "view_n" not in st.session_state: st.session_state.view_n = 120
view_n_input = chart_col.number_input("표시봉", 50, len(df_trade), st.session_state.view_n, 10, label_visibility="collapsed")
if int(view_n_input) != st.session_state.view_n:
    st.session_state.view_n = int(view_n_input)
view_n = st.session_state.view_n

start_i = df_trade.i.iloc[max(0, len(df_trade) - view_n)]
end_i = df_trade.i.iloc[-1]

# --- 차트 Y축 범위 ---
sub = df_trade[df_trade.i >= start_i]
ma_cols_for_range = [f"{k}{p}" for k, p in mas_tuple]
ymin = sub[["Low"] + ma_cols_for_range].min().min()
ymax = sub[["High"] + ma_cols_for_range].max().max()
span = ymax - ymin if ymax > ymin else 1
yrng = [ymin - span * MARGIN, ymax + span * MARGIN]

# --- 차트 객체 생성 ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.8, 0.2], vertical_spacing=0.02)
# (차트 그리는 나머지 부분은 동일)
for k, p in mas_tuple:
    fig.add_scatter(x=df_trade.i, y=df_trade[f"{k}{p}"], line=dict(width=1.5), name=f"{k}{p}", row=1, col=1)
fig.add_candlestick(x=df_trade.i, open=df_trade.Open, high=df_trade.High, low=df_trade.Low, close=df_trade.Close, name="Price", row=1, col=1, increasing=dict(line=dict(color="black", width=1), fillcolor="white"), decreasing=dict(line=dict(color="black", width=1), fillcolor="black"))
fig.add_bar(x=df_trade.i, y=df_trade.Volume, name="Volume", row=2, col=1, marker_color='rgba(128,128,128,0.5)')
tick_step = max(len(sub) // 10, 1)
fig.update_layout(xaxis=dict(tickmode="array", tickvals=sub.i[::tick_step], ticktext=sub.index.strftime("%y-%m-%d")[::tick_step], tickangle=0), xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=25, b=20, l=5, r=40))
fig.update_yaxes(range=yrng, row=1, col=1)
fig.update_xaxes(range=[start_i - 1, end_i + PAD])
if "chart_slot" not in st.session_state:
    st.session_state.chart_slot = chart_col.empty()
st.session_state.chart_slot.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------ 사이드바 (상태, 컨트롤) ------------------------------
with side_col:
    # --- 상태 정보 ---
    pos_val = g.pos.qty * price_now if g.pos else 0
    equity = g.cash + pos_val
    unreal = (price_now - g.pos.avg_price) * g.pos.qty if g.pos and g.pos.side == 'long' else (g.pos.avg_price - price_now) * g.pos.qty if g.pos and g.pos.side == 'short' else 0
    
    st.subheader("종목: ???") # [수정] 종목명 숨기기
    st.metric("현재 평가자산", f"${equity:,.2f}", f"${unreal:,.2f} 미실현")
    st.text(f"현금: ${g.cash:,.2f}")
    st.text(f"현재가(종가): ${price_now:,.2f}") # [추가] 현재가 표시
    if g.pos:
        st.text(f"포지션: {g.pos.side.upper()} {g.pos.qty}주 @ ${g.pos.avg_price:,.2f}")

    st.markdown("---")
    
    # --- 매매 컨트롤 ---
    st.subheader("매매")
    amount = st.number_input("수량(주)", min_value=1, value=10, step=1)
    
    # [추가] 주문 금액 및 자산 대비 비중 표시
    order_value = amount * price_now
    position_pct = (order_value / equity) * 100 if equity > 0 else 0
    st.caption(f"주문 금액: ${order_value:,.2f} (자산의 {position_pct:.1f}%)")

    b_col, s_col = st.columns(2)
    if b_col.button("매수", use_container_width=True):
        if g.cash >= order_value:
            g.buy(amount)
            st.rerun()
        else:
            st.warning("현금이 부족합니다.")
            
    if s_col.button("매도/공매도", use_container_width=True):
        g.sell(amount)
        st.rerun()
            
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
        start_random_modelbook(g.initial_cash)

    st.markdown("---")
    
    # --- 결과 요약 생성 함수 ---
    def create_summary(log: list, ticker: str) -> dict: # [수정] ticker 인자 추가
        trades = [x for x in log if "pnl" in x]
        summary = {"종목": ticker} # [추가] 결과에 종목명 추가
        if not trades:
            summary["총 거래 횟수"] = 0
            return summary

        total_pnl = sum(x["pnl"] for x in trades)
        total_fees = sum(x.get("fee", 0) for x in trades)
        net_pnl = total_pnl + total_fees

        wins = [x for x in trades if x["pnl"] > 0]
        losses = [x for x in trades if x["pnl"] <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t['pnl'] for t in losses)) / len(losses) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        summary.update({
            "최종 순손익": f"${net_pnl:,.2f}",
            "총 거래 횟수": f"{len(trades)}회",
            "승률": f"{win_rate:.2f}%",
            "손익비 (Profit Factor)": f"{profit_factor:.2f}",
            "평균 수익": f"${avg_win:,.2f}",
            "평균 손실": f"${avg_loss:,.2f}",
            "총 수수료": f"${abs(total_fees):,.2f}",
        })
        return summary

    # --- 게임 종료 버튼 ---
    if st.button("게임 종료 & 결과 보기", type="primary", use_container_width=True):
        if g.pos:
            g.flat()
        
        # [수정] 결과 생성 시 g.ticker 전달
        st.session_state.last_summary = create_summary(g.log, g.ticker)
        st.session_state.pop("game", None)
        st.rerun()

