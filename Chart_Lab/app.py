# app.py (Ver. 2.9)
# 기능: 1. on_click 콜백을 사용하여 버튼 클릭 로직을 안정화하고 StreamlitAPIException 해결.
#      2. 현금 부족 시 '매수' 버튼 비활성화 기능 추가.

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

def initialize_state_if_missing():
    """세션 상태의 키가 비정상적으로 사라졌을 경우를 대비해 기본값으로 다시 초기화합니다."""
    defaults = {
        "view_n": 120,
        "stop_loss_price": 0.0,
        "chart_height": 800,
        "ema_input": "10,21",
        "sma_input": "50,200",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
            return
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
initialize_state_if_missing() 
chart_col, side_col = st.columns([7, 3])

# -------------- 사이드바 UI --------------
with side_col:
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

    with st.expander("📖 모델북 차트 게임 설명서"):
        help_text = """
        **1) 이동평균선 (EMA, SMA)**
        - 쉼표(,)로 구분하여 여러 기간을 추가할 수 있습니다. (예: `10,21,60`)
        - 기본값은 EMA(10, 21), SMA(50, 200) 입니다.

        **2) 표시 캔들 수 (표시봉)**
        - 차트 하단에 한 번에 표시할 캔들의 개수를 조절합니다.
        - 기본값은 120개입니다.

        **3) 줌인 / 줌아웃**
        - 차트 영역을 더블클릭하면 전체 기간이 표시됩니다.
        - 다시 더블클릭하면 설정한 표시 캔들 수로 돌아옵니다. 장기 추세 판단에 유용합니다.

        **4) 블라인드 테스트**
        - 훈련의 집중도를 높이기 위해, 게임이 진행되는 동안에는 종목명이 가려집니다.
        - 게임 종료 후 결과 분석 화면에서 실제 종목명을 확인할 수 있습니다.

        **5) 게임 진행 버튼**
        - **▶ 다음**: 다음 날의 캔들로 이동합니다.
        - **🎲 날짜 변경**: 현재 종목 내에서 다른 시작 시점으로 무작위 이동합니다.
        - **📚 모델북**: `modelbook.txt`에 등록된 다른 종목으로 새로운 게임을 시작합니다.

        **6) 매매 기능**
        - **수량**: 매수/매도할 주식 수를 정합니다. (기본 10주)
        - **주문 금액**: 설정한 수량에 따른 주문 금액과 현재 자산 대비 비중이 표시됩니다.
        - **손절매 가격**: 리스크 관리를 위해 손절 가격을 설정할 수 있습니다.
            - 가격을 입력하면 차트에 빨간 점선이 표시되어 시각적으로 손절 라인을 확인할 수 있습니다.
            - 이 기능은 실제 손절 주문을 실행하지는 않으며, 베팅 규모를 가늠하는 보조 도구입니다.
            - 손절 가격을 입력하면, 해당 주문이 손절될 경우 예상되는 **베팅 리스크**(손실 금액 및 자산 대비 %)를 미리 계산해 보여줍니다.

        **7) 차트 높이 조절**
        - 슬라이더를 이용해 차트의 세로 크기를 조절할 수 있습니다.
        """
        st.markdown(help_text, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("게임 진행")
    
    # ==================================================================
    # ✨ 버그 수정 및 기능 개선: 버튼 로직을 on_click 콜백으로 변경
    # ==================================================================
    def on_click_next():
        g.next_candle()
        if not g.pos:
            st.session_state.stop_loss_price = 0.0

    def on_click_jump():
        jump_random_date()

    def on_click_modelbook():
        start_random_modelbook(g.initial_cash)
        
    n_col, j_col, r_col = st.columns(3)
    n_col.button("▶ 다음", use_container_width=True, on_click=on_click_next)
    j_col.button("🎲 날짜 변경", use_container_width=True, on_click=on_click_jump)
    r_col.button("📚 모델북", use_container_width=True, on_click=on_click_modelbook)
    
    if st.button("게임 종료 & 결과 보기", type="primary", use_container_width=True):
        if g.pos: g.flat()
        trades = [x for x in g.log if "pnl" in x]
        summary = {"종목": g.ticker}
        if not trades: summary["총 거래 횟수"] = 0
        else:
            total_pnl, total_fees = sum(x["pnl"] for x in trades), sum(x.get("fee", 0) for x in trades)
            net_pnl, wins = total_pnl + total_fees, [x for x in trades if x["pnl"] > 0]
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            summary.update({"최종 순손익": f"${net_pnl:,.2f}", "총 거래 횟수": f"{len(trades)}회", "승률": f"{win_rate:.2f}%"})
        st.session_state.last_summary = summary
        st.session_state.pop("game", None)
        st.rerun()

    st.markdown("---")
    st.subheader("매매")
    amount = st.number_input("수량(주)", min_value=1, value=10, step=1)
    
    order_value = amount * price_now
    position_pct = (order_value / equity) * 100 if equity > 0 else 0
    st.caption(f"주문 금액: ${order_value:,.2f} (자산의 {position_pct:.1f}%)")
    
    if st.session_state.stop_loss_price > 0:
        risk_per_share_long = price_now - st.session_state.stop_loss_price
        if risk_per_share_long > 0:
            total_risk_long = risk_per_share_long * amount
            risk_pct_long = (total_risk_long / equity) * 100 if equity > 0 else 0
            st.caption(f"↳ 베팅 리스크 (매수): ${total_risk_long:,.2f} ({risk_pct_long:.2f}%)")
        
        risk_per_share_short = st.session_state.stop_loss_price - price_now
        if risk_per_share_short > 0:
            total_risk_short = risk_per_share_short * amount
            risk_pct_short = (total_risk_short / equity) * 100 if equity > 0 else 0
            st.caption(f"↳ 베팅 리스크 (매도): ${total_risk_short:,.2f} ({risk_pct_short:.2f}%)")

    st.number_input("손절매 가격", key="stop_loss_price", format="%.2f", step=0.01)
    
    def on_click_buy(qty):
        g.buy(qty)

    def on_click_sell(qty):
        g.sell(qty)
        
    def on_click_flat():
        g.flat()
        st.session_state.stop_loss_price = 0.0

    b_col, s_col = st.columns(2)
    can_buy = g.cash >= order_value
    b_col.button("매수", use_container_width=True, on_click=on_click_buy, args=(amount,), disabled=not can_buy)
    s_col.button("매도/공매도", use_container_width=True, on_click=on_click_sell, args=(amount,))
    if not can_buy:
        b_col.caption("현금이 부족합니다.")
            
    st.button("전량 청산", use_container_width=True, on_click=on_click_flat, disabled=not g.pos)
    
    st.markdown("---")
    st.subheader("차트 설정")
    st.slider("차트 높이", min_value=400, max_value=1200, step=50, key="chart_height")

# -------------- 차트 UI --------------
with chart_col:
    ma_cols = st.columns(2)
    st.text_input("EMA 기간(쉼표)", key="ema_input")
    st.text_input("SMA 기간(쉼표)", key="sma_input")
    mas_input = [("EMA", int(p.strip())) for p in st.session_state.ema_input.split(",") if p.strip().isdigit()] + \
                [("SMA", int(p.strip())) for p in st.session_state.sma_input.split(",") if p.strip().isdigit()]
    mas_tuple = tuple(mas_input)

    df_full = g.df
    visible_df_with_ma = add_cached_indicators(df_full, mas_tuple).iloc[:g.idx + 1]
    df_trade = (visible_df_with_ma.dropna(subset=["Open", "High", "Low", "Close"])
                                  .loc[visible_df_with_ma.Volume > 0]
                                  .assign(i=lambda d: range(len(d))))

    if df_trade.empty: st.error("표시할 데이터가 없습니다."); st.stop()

    st.number_input(
        "표시봉",
        min_value=50,
        max_value=len(df_trade),
        step=10,
        key="view_n",
        label_visibility="collapsed"
    )
    view_n = st.session_state.view_n

    start_i = df_trade.i.iloc[max(0, len(df_trade) - view_n)]
    end_i = df_trade.i.iloc[-1]
    sub = df_trade[df_trade.i >= start_i]
    
    ma_cols_for_range = [f"{k}{p}" for k, p in mas_tuple]
    ymin = sub[["Low"] + ma_cols_for_range].min().min()
    ymax = sub[["High"] + ma_cols_for_range].max().max()

    span = ymax - ymin if ymax > ymin else 1
    price_yrange = [ymin - span * MARGIN, ymax + span * MARGIN]
    volume_yrange = [0, sub['Volume'].max() * 1.2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
    
    for i, (k, p) in enumerate(mas_tuple):
        color = MA_COLORS.get((k, p))
        fig.add_scatter(x=df_trade.i, y=df_trade[f"{k}{p}"], line=dict(width=1.5, color=color), name=f"{k}{p}", row=1, col=1)
    
    fig.add_candlestick(x=df_trade.i, open=df_trade.Open, high=df_trade.High, low=df_trade.Low, close=df_trade.Close, name="Price", row=1, col=1, increasing=dict(line=dict(color="black", width=1), fillcolor="white"), decreasing=dict(line=dict(color="black", width=1), fillcolor="black"))

    volume_colors = ['white' if c >= o else 'black' for o, c in zip(df_trade['Open'], df_trade['Close'])]
    fig.add_bar(x=df_trade.i, y=df_trade.Volume, name="Volume", row=2, col=1, marker=dict(color=volume_colors, line=dict(color='black', width=1)))

    log_df = pd.DataFrame(g.log)
    if not log_df.empty:
        log_df = log_df[log_df.action.str.contains("ENTER")]
        merged = pd.merge(log_df, df_trade.reset_index(), left_on='date', right_on='Date', how='inner')
        if not merged.empty:
            buy_df = merged[merged.action.str.contains("LONG")]
            sell_df = merged[merged.action.str.contains("SHORT")]
            if not buy_df.empty:
                fig.add_scatter(x=buy_df['i'], y=buy_df['Low'] - span * 0.03, mode="markers", marker=dict(symbol="triangle-up", color="green", size=10), name="Buy", row=1, col=1)
            if not sell_df.empty:
                fig.add_scatter(x=sell_df['i'], y=sell_df['High'] + span * 0.03, mode="markers", marker=dict(symbol="triangle-down", color="red", size=10), name="Sell", row=1, col=1)

    shapes = []
    separator_line = dict(
        type='line', xref='paper', yref='paper',
        x0=0, y0=0.3, x1=1, y1=0.3,
        line=dict(color='black', width=1)
    )
    shapes.append(separator_line)

    if st.session_state.stop_loss_price > 0:
        stop_loss_line = dict(
            type='line', xref='paper', yref='y',
            x0=0, y0=st.session_state.stop_loss_price,
            x1=1, y1=st.session_state.stop_loss_price,
            line=dict(color='black', width=2, dash='dash')
        )
        shapes.append(stop_loss_line)
        fig.add_annotation(
            x=end_i + PAD, y=st.session_state.stop_loss_price,
            text="손절 라인", showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(color="black", size=12)
        )
    
    fig.update_layout(
        height=st.session_state.chart_height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(t=25, b=20, l=5, r=40),
        spikedistance=-1,
        shapes=shapes
    )

    fig.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#999999", spikemode="across", spikesnap="cursor", range=[start_i - 1, end_i + PAD])
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#999999", spikemode="across", spikesnap="cursor")
    fig.update_yaxes(range=price_yrange, row=1, col=1)
    fig.update_yaxes(range=volume_yrange, row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
