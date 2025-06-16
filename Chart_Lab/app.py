# app.py (Ver. 2.1)
# 기능: 사용자가 입력한 손절매 가격을 차트에 수평 점선으로 표시

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
MA_COLORS = {
    ("EMA", 10): "red",
    ("EMA", 21): "blue",
    ("SMA", 50): "orange",
    ("SMA", 200): "black",
}

# ---------------------------------- 캐싱 헬퍼 ----------------------------------
@st.cache_data
def load_cached_price(ticker: str) -> pd.DataFrame | None:
    """yfinance로 가격 데이터를 로드하고 캐시합니다."""
    return get_price(ticker)

@st.cache_data
def add_cached_indicators(df: pd.DataFrame, mas_tuple: tuple) -> pd.DataFrame:
    """계산된 이동평균선을 데이터프레임에 추가하고 캐시합니다."""
    mas_settings = [(kind, period, True) for kind, period in mas_tuple]
    return add_mas(df.copy(), mas_settings)

# --------------------------------- 게임 생성/시작 ---------------------------------
def reset_session_state():
    """게임과 관련된 세션 상태를 초기화합니다."""
    st.session_state.view_n = 120
    st.session_state.stop_loss_price = 0.0
    st.session_state.chart_height = 800
    st.session_state.ema_input = "10,21"
    st.session_state.sma_input = "50,200"

def create_game(tkr: str, capital: int) -> GameState | None:
    """새로운 게임 인스턴스를 생성합니다."""
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
    """지정된 티커로 새 게임을 시작합니다."""
    game = create_game(tkr, capital)
    if game:
        st.session_state.game = game
        st.session_state.last_summary = None
        reset_session_state()
        st.rerun()

def start_random_modelbook(capital: int):
    """modelbook.txt에서 랜덤 티커로 새 게임을 시작합니다."""
    # 경로 탐색 로직 (상대 경로 문제 해결)
    root = os.path.dirname(__file__)
    path = os.path.join(root, "..", "modelbook.txt")
    if not os.path.exists(path):
        path = os.path.join(root, "modelbook.txt")
        
    if not os.path.exists(path):
        st.error("modelbook.txt 파일을 찾지 못했습니다."); return
    
    with open(path, "r", encoding="utf-8") as f:
        tickers = [t.strip().upper() for t in f.read().split(",")]
    if not tickers:
        st.error("modelbook.txt에 티커가 없습니다"); return
    
    random.shuffle(tickers)
    for tkr in tickers:
        game = create_game(tkr, capital)
        if game:
            st.session_state.game = game
            st.session_state.last_summary = None
            reset_session_state()
            st.rerun()
            return # 유효한 게임을 찾으면 즉시 종료
    st.error("모델북에 시작 가능한 유효한 티커가 없습니다.")

def jump_random_date():
    """게임 내에서 랜덤한 날짜로 점프합니다."""
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi and i >= 120]
    if pool:
        g.idx, g.cash, g.pos, g.log = random.choice(pool), g.initial_cash, None, []
        reset_session_state()
        st.rerun()

# --------------------------------- 첫 랜딩 페이지 ---------------------------------
if "game" not in st.session_state:
    st.header("📈 차트 훈련소")
    
    # 지난 게임 결과가 있으면 표시
    if st.session_state.get("last_summary"):
        st.markdown("---")
        summary = st.session_state.last_summary
        st.subheader(f"📊 지난 게임 성과 분석: **{summary.get('종목', '???')}**")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("최종 순손익", summary.get("최종 순손익", "N/A"))
        m_col2.metric("승률", summary.get("승률", "N/A"))
        m_col3.metric("손익비", summary.get("손익비 (Profit Factor)", "N/A"))
        
        with st.expander("상세 결과 보기"):
            st.json(summary)
        st.markdown("---")
    
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

# stop_loss_price 초기화 (안정성)
if 'stop_loss_price' not in st.session_state:
    st.session_state.stop_loss_price = 0.0

# -------------- 사이드바 UI --------------
with side_col:
    # 계좌 상태 표시
    price_now = g.df.Close.iloc[g.idx]
    pos_val = g.pos.qty * price_now if g.pos else 0
    equity = g.cash + pos_val
    unreal = (price_now - g.pos.avg_price) * g.pos.qty if g.pos and g.pos.side == 'long' else (g.pos.avg_price - price_now) * g.pos.qty if g.pos and g.pos.side == 'short' else 0
    
    st.subheader("종목: ???")
    st.metric("현재 평가자산", f"${equity:,.2f}", f"${unreal:,.2f} 미실현")
    st.text(f"현금: ${g.cash:,.2f}")
    st.text(f"현재가(종가): ${price_now:,.2f}")
    if g.pos:
        st.text(f"포지션: {g.pos.side.upper()} {g.pos.qty}주 @ ${g.pos.avg_price:,.2f}")

    # 게임 진행 컨트롤
    st.markdown("---")
    st.subheader("게임 진행")
    n_col, j_col, r_col = st.columns(3)
    if n_col.button("▶ 다음", use_container_width=True):
        g.next_candle()
        if not g.pos: st.session_state.stop_loss_price = 0.0
        st.rerun()
    if j_col.button("🎲 날짜 변경", use_container_width=True):
        jump_random_date()
    if r_col.button("📚 모델북", use_container_width=True):
        start_random_modelbook(g.initial_cash)
    
    # 게임 종료
    if st.button("게임 종료 & 결과 보기", type="primary", use_container_width=True):
        if g.pos: g.flat()
        trades = [x for x in g.log if "pnl" in x]
        summary = {"종목": g.ticker}
        if not trades:
            summary["총 거래 횟수"] = 0
        else:
            total_pnl = sum(x["pnl"] for x in trades)
            total_fees = sum(x.get("fee", 0) for x in trades)
            net_pnl = total_pnl + total_fees
            wins = [x for x in trades if x["pnl"] > 0]
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            summary.update({
                "최종 순손익": f"${net_pnl:,.2f}",
                "총 거래 횟수": f"{len(trades)}회",
                "승률": f"{win_rate:.2f}%",
            })
        st.session_state.last_summary = summary
        st.session_state.pop("game", None)
        st.rerun()

    # 매매 컨트롤
    st.markdown("---")
    st.subheader("매매")
    amount = st.number_input("수량(주)", min_value=1, value=10, step=1)
    
    order_value = amount * price_now
    position_pct = (order_value / equity) * 100 if equity > 0 else 0
    st.caption(f"주문 금액: ${order_value:,.2f} (자산의 {position_pct:.1f}%)")
    
    temp_stop_loss = st.session_state.get("stop_loss_price", 0.0)
    
    # 베팅 리스크 계산 및 표시
    if temp_stop_loss > 0:
        # 매수 시 리스크
        risk_per_share_long = price_now - temp_stop_loss
        if risk_per_share_long > 0:
            total_risk_long = risk_per_share_long * amount
            risk_pct_long = (total_risk_long / equity) * 100 if equity > 0 else 0
            st.caption(f"↳ 베팅 리스크 (매수): ${total_risk_long:,.2f} ({risk_pct_long:.2f}%)")
        
        # 매도(공매도) 시 리스크
        risk_per_share_short = temp_stop_loss - price_now
        if risk_per_share_short > 0:
            total_risk_short = risk_per_share_short * amount
            risk_pct_short = (total_risk_short / equity) * 100 if equity > 0 else 0
            st.caption(f"↳ 베팅 리스크 (매도): ${total_risk_short:,.2f} ({risk_pct_short:.2f}%)")

    st.number_input("손절매 가격", key="stop_loss_price", format="%.2f", step=0.01)

    # 매매 버튼
    b_col, s_col = st.columns(2)
    if b_col.button("매수", use_container_width=True):
        if g.cash >= order_value: g.buy(amount); st.rerun()
        else: st.warning("현금이 부족합니다.")
            
    if s_col.button("매도/공매도", use_container_width=True):
        g.sell(amount); st.rerun()
            
    if st.button("전량 청산", use_container_width=True) and g.pos:
        g.flat()
        st.session_state.stop_loss_price = 0.0 # 청산 시 손절가 초기화
        st.rerun()
    
    # 차트 설정
    st.markdown("---")
    st.subheader("차트 설정")
    chart_height = st.slider("차트 높이", min_value=400, max_value=1200, value=800, step=50, key="chart_height")


# -------------- 차트 UI --------------
with chart_col:
    # 이동평균선 설정
    ma_cols = st.columns(2)
    ema_input = ma_cols[0].text_input("EMA 기간(쉼표)", st.session_state.ema_input)
    sma_input = ma_cols[1].text_input("SMA 기간(쉼표)", st.session_state.sma_input)
    mas_input = [("EMA", int(p.strip())) for p in ema_input.split(",") if p.strip().isdigit()] + \
                [("SMA", int(p.strip())) for p in sma_input.split(",") if p.strip().isdigit()]
    mas_tuple = tuple(mas_input)

    # 차트 데이터 준비
    df_full = g.df
    visible_df_with_ma = add_cached_indicators(df_full, mas_tuple).iloc[:g.idx + 1]
    df_trade = (visible_df_with_ma.dropna(subset=["Open", "High", "Low", "Close"])
                                  .loc[visible_df_with_ma.Volume > 0]
                                  .assign(i=lambda d: range(len(d))))

    if df_trade.empty:
        st.error("표시할 데이터가 없습니다."); st.stop()

    # 표시할 캔들 수 설정
    view_n = st.number_input("표시봉", 50, len(df_trade), st.session_state.view_n, 10, label_visibility="collapsed", key="view_n")

    start_i = df_trade.i.iloc[max(0, len(df_trade) - view_n)]
    end_i = df_trade.i.iloc[-1]
    sub = df_trade[df_trade.i >= start_i]
    
    # Y축 범위 계산
    ma_cols_for_range = [f"{k}{p}" for k, p in mas_tuple]
    ymin = sub[["Low"] + ma_cols_for_range].min().min()
    ymax = sub[["High"] + ma_cols_for_range].max().max()

    span = ymax - ymin if ymax > ymin else 1
    price_yrange = [ymin - span * MARGIN, ymax + span * MARGIN]
    volume_yrange = [0, sub['Volume'].max() * 1.2]

    # 차트 객체 생성
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
    
    # 이동평균선 그리기
    for i, (k, p) in enumerate(mas_tuple):
        color = MA_COLORS.get((k, p))
        fig.add_scatter(x=df_trade.i, y=df_trade[f"{k}{p}"], line=dict(width=1.5, color=color), name=f"{k}{p}", row=1, col=1)
    
    # 캔들스틱 및 볼륨 차트 그리기
    fig.add_candlestick(x=df_trade.i, open=df_trade.Open, high=df_trade.High, low=df_trade.Low, close=df_trade.Close, name="Price", row=1, col=1, increasing=dict(line=dict(color="black", width=1), fillcolor="white"), decreasing=dict(line=dict(color="black", width=1), fillcolor="black"))
    fig.add_bar(x=df_trade.i, y=df_trade.Volume, name="Volume", row=2, col=1, marker_color='rgba(128,128,128,0.5)')
    
    # 매매 기록(화살표) 표시
    log_df = pd.DataFrame(g.log)
    if not log_df.empty:
        log_df = log_df[log_df.action.str.contains("ENTER")]
        merged = log_df.merge(df_trade.reset_index(), on='date', how='inner')
        if not merged.empty:
            buy_df = merged[merged.action.str.contains("LONG")]
            sell_df = merged[merged.action.str.contains("SHORT")]
            if not buy_df.empty:
                fig.add_scatter(x=buy_df['i'], y=buy_df['Low'] - span * 0.03, mode="markers", marker=dict(symbol="triangle-up", color="green", size=10), name="Buy", row=1, col=1)
            if not sell_df.empty:
                fig.add_scatter(x=sell_df['i'], y=sell_df['High'] + span * 0.03, mode="markers", marker=dict(symbol="triangle-down", color="red", size=10), name="Sell", row=1, col=1)

    # ==================================================================
    # ✨ 새로운 기능: 손절매 가격을 차트에 수평선으로 표시
    # ==================================================================
    stop_loss_price = st.session_state.get("stop_loss_price", 0.0)
    if stop_loss_price > 0:
        fig.add_hline(
            y=stop_loss_price,
            line_dash="dash",      # 점선 스타일
            line_color="red",      # 선 색상
            line_width=2,          # 선 두께
            annotation_text="손절 라인",  # 라인 옆에 표시될 텍스트
            annotation_position="bottom right", # 텍스트 위치
            annotation_font_size=12,
            annotation_font_color="red",
            row=1, col=1
        )

    # 차트 레이아웃 설정
    fig.update_layout(height=st.session_state.chart_height, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=25, b=20, l=5, r=40), spikedistance=-1)
    fig.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#999999", spikemode="across", spikesnap="cursor", range=[start_i - 1, end_i + PAD])
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#999999", spikemode="across", spikesnap="cursor")
    fig.update_yaxes(range=price_yrange, row=1, col=1)
    fig.update_yaxes(range=volume_yrange, row=2, col=1)
    
    # 차트 출력
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

