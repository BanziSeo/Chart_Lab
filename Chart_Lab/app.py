# app.py (Ver. 4.1)
# 기능: 1. '표시봉' 라벨을 '캔들 수'로 변경.

import streamlit as st
import pandas as pd
import random
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# services 폴더의 모듈들을 정확히 임포트합니다.
from services.data_loader import get_price
from services.indicators import add_mas

# ------------------------------ Streamlit 페이지 설정 ------------------------------
st.set_page_config(page_title="차트 훈련소", page_icon="📈", layout="wide")

# ----------------------------------- 상수 정의 -----------------------------------
PAD, MARGIN = 20, 0.05
DEFAULT_VIEW_CANDLES = 120
MIN_HISTORY_DAYS = 30
GAME_MAX_CANDLES = 80 # 게임당 최대 진행 캔들 수
MA_COLORS = {
    ("EMA", 10): "red",
    ("EMA", 21): "blue",
    ("SMA", 50): "orange",
    ("SMA", 200): "black",
}

# --------------------------------- 핵심 로직 클래스 ---------------------------------
class Position:
    """매매 포지션을 관리하는 클래스"""
    def __init__(self, side: str, entry_qty: int, entry_price: float, current_pos=None):
        self.side = side
        if current_pos and current_pos.side == self.side:
            current_value = current_pos.avg_price * current_pos.qty
            entry_value = entry_price * entry_qty
            total_qty = current_pos.qty + entry_qty
            self.avg_price = (current_value + entry_value) / total_qty
            self.qty = total_qty
        else:
            self.avg_price = entry_price
            self.qty = entry_qty

    def close(self, exit_price: float) -> float:
        """포지션 청산 시 손익(PNL)을 계산합니다."""
        if self.side == "long":
            pnl = (exit_price - self.avg_price) * self.qty
        else: # short
            pnl = (self.avg_price - exit_price) * self.qty
        return pnl

class GameState:
    """게임의 모든 상태를 관리하는 클래스"""
    def __init__(self, df, idx: int, start_cash: int, tkr: str, max_duration: int = GAME_MAX_CANDLES):
        self.df, self.ticker = df, tkr
        self.initial_cash, self.cash = start_cash, start_cash
        self.pos: Position | None = None
        self.log = []
        self.start_idx = idx
        self.idx = idx
        self.max_duration = max_duration
        self.start_date = self.df.index[self.start_idx]

    @property
    def today(self): return self.df.index[self.idx]
    @property
    def candles_passed(self): return self.idx - self.start_idx
    @property
    def is_over(self):
        end_of_duration = self.candles_passed >= self.max_duration
        end_of_data = self.idx >= len(self.df) - 1
        return end_of_duration or end_of_data
    def next_candle(self):
        if not self.is_over: self.idx += 1
    def buy(self, qty: int):
        price_now = self.df.Close.iloc[self.idx]
        self.cash -= (price_now * qty)
        self.pos = Position("long", qty, price_now, self.pos)
        self.log.append({"date": self.today, "action": "ENTER LONG", "price": price_now, "qty": qty})
    def sell(self, qty: int):
        price_now = self.df.Close.iloc[self.idx]
        self.pos = Position("short", qty, price_now, self.pos)
        self.log.append({"date": self.today, "action": "ENTER SHORT", "price": price_now, "qty": qty})
    def flat(self):
        if not self.pos: return
        price_now = self.df.Close.iloc[self.idx]
        pnl = self.pos.close(price_now)
        trade_value = self.pos.qty * price_now
        fee = trade_value * 0.0014
        if self.pos.side == 'long':
            self.cash += trade_value - fee
        else: # short
            self.cash += pnl - fee
        self.log.append({"date": self.today, "action": "EXIT", "price": price_now, "pnl": pnl, "fee": -fee})
        self.pos = None

# ---------------------------------- 캐싱 헬퍼 ----------------------------------
@st.cache_data
def load_cached_price(ticker: str) -> pd.DataFrame | None:
    return get_price(ticker)

@st.cache_data
def add_cached_indicators(df: pd.DataFrame, mas_tuple: tuple) -> pd.DataFrame:
    mas_settings = [(kind, period, True) for kind, period in mas_tuple]
    return add_mas(df.copy(), mas_settings)

# --------------------------------- 게임 생성/시작 ---------------------------------
def reset_session_state():
    st.session_state.view_n = DEFAULT_VIEW_CANDLES
    st.session_state.stop_loss_price = 0.0
    st.session_state.chart_height = 800
    st.session_state.ema_input = "10,21"
    st.session_state.sma_input = "50,200"

def initialize_state_if_missing():
    defaults = { "view_n": DEFAULT_VIEW_CANDLES, "stop_loss_price": 0.0, "chart_height": 800, "ema_input": "10,21", "sma_input": "50,200",}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def create_game(tkr: str, capital: int) -> GameState | None:
    df = load_cached_price(tkr)
    if df is None or len(df) < MIN_HISTORY_DAYS:
        st.error(f"'{tkr}' 종목 데이터를 불러오지 못했거나 데이터가 너무 적습니다. (최소 {MIN_HISTORY_DAYS}일 필요)")
        return None
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(df.index) if lo <= d <= hi]
    if not pool:
        st.error(f"'{tkr}' 종목에서 시작 가능한 랜덤 구간을 찾지 못했습니다. (훈련 구간: 1년~5년 전)")
        return None
    return GameState(df=df, tkr=tkr, idx=random.choice(pool), start_cash=capital)

def start_game(tkr: str, capital: int):
    game = create_game(tkr, capital)
    if game:
        st.session_state.game = game
        st.session_state.last_summary = None
        reset_session_state()
        st.rerun()

def start_random_modelbook(capital: int):
    root = os.path.dirname(__file__)
    path = os.path.join(root, "..", "modelbook.txt")
    if not os.path.exists(path): path = os.path.join(root, "modelbook.txt")
    if not os.path.exists(path): st.error("modelbook.txt 파일을 찾지 못했습니다."); return
    with open(path, "r", encoding="utf-8") as f:
        tickers = [t.strip().upper() for t in f.read().split(",")]
    if not tickers: st.error("modelbook.txt에 티커가 없습니다"); return
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
    g: GameState = st.session_state.game
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i, d in enumerate(g.df.index) if lo <= d <= hi]
    if pool:
        g.idx = random.choice(pool)
        g.start_idx = g.idx
        g.start_date = g.df.index[g.start_idx]
        g.cash, g.pos, g.log = g.initial_cash, None, []
        st.session_state.stop_loss_price = 0.0
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
        with st.expander("상세 결과 보기"): st.json(summary)
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
    if g.pos: st.text(f"포지션: {g.pos.side.upper()} {g.pos.qty}주 @ ${g.pos.avg_price:,.2f}")

    st.markdown("---")
    
    # ✨ UI 개편: 매매 섹션 재구성
    st.subheader("매매 및 진행")
    st.progress(g.candles_passed / (g.max_duration -1) if g.max_duration > 1 else 1, text=f"{g.candles_passed + 1} / {g.max_duration} 캔들")

    amount = st.number_input("수량(주)", min_value=1, value=10, step=1)
    st.number_input("손절매 가격", key="stop_loss_price", format="%.2f", step=0.01)
    
    st.caption(f"주문 금액: ${amount * price_now:,.2f} (자산의 {(amount * price_now / equity) * 100 if equity > 0 else 0:.1f}%)")
    if st.session_state.stop_loss_price > 0:
        risk_per_share_long = price_now - st.session_state.stop_loss_price
        if risk_per_share_long > 0:
            total_risk = risk_per_share_long * amount
            st.caption(f"↳ 베팅 리스크 (매수): ${total_risk:,.2f} (자산의 {(total_risk / equity) * 100 if equity > 0 else 0:.2f}%)")
        risk_per_share_short = st.session_state.stop_loss_price - price_now
        if risk_per_share_short > 0:
            total_risk = risk_per_share_short * amount
            st.caption(f"↳ 베팅 리스크 (매도): ${total_risk:,.2f} (자산의 {(total_risk / equity) * 100 if equity > 0 else 0:.2f}%)")
    
    def on_click_buy(qty): g.buy(qty)
    def on_click_sell(qty): g.sell(qty)
    def on_click_flat(): g.flat(); st.session_state.stop_loss_price = 0.0
    def on_click_next():
        g.next_candle()
        if not g.pos: st.session_state.stop_loss_price = 0.0
        
    b_col, s_col = st.columns(2)
    can_buy = g.cash >= (amount * price_now)
    b_col.button("매수 (Long)", use_container_width=True, on_click=on_click_buy, args=(amount,), disabled=not can_buy)
    s_col.button("매도 (Short)", use_container_width=True, on_click=on_click_sell, args=(amount,))
    if not can_buy: b_col.caption("현금이 부족합니다.")

    n_col, f_col = st.columns(2)
    n_col.button("▶ 다음 캔들", use_container_width=True, on_click=on_click_next, disabled=g.is_over, type="primary")
    f_col.button("전량 청산", use_container_width=True, on_click=on_click_flat, disabled=not g.pos)
    
    st.markdown("---")

    # ✨ UI 개편: 게임 관리 섹션 신설
    st.subheader("게임 관리")
    def on_click_jump(): jump_random_date()
    def on_click_modelbook(): start_random_modelbook(g.initial_cash)
    
    j_col, m_col = st.columns(2)
    j_col.button("🎲 날짜 변경", use_container_width=True, on_click=on_click_jump)
    m_col.button("📚 모델북", use_container_width=True, on_click=on_click_modelbook)
    
    if g.is_over: st.info("최대 캔들 수에 도달했습니다. 게임을 종료하고 결과를 확인하세요.")

    if st.button("게임 종료 & 결과 보기", type="secondary", use_container_width=True):
        if g.pos: g.flat()
        trades = [x for x in g.log if "pnl" in x]
        year = g.start_date.year
        summary = {"종목": f"{g.ticker} ({year}년)"}
        if not trades: summary["총 거래 횟수"] = 0
        else:
            total_pnl = sum(x["pnl"] for x in trades)
            total_fees = sum(x.get("fee", 0) for x in trades)
            net_pnl = total_pnl + total_fees
            wins = [x for x in trades if x["pnl"] > 0]
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            summary.update({"최종 순손익": f"${net_pnl:,.2f}", "총 거래 횟수": f"{len(trades)}회", "승률": f"{win_rate:.2f}%"})
        st.session_state.last_summary = summary
        st.session_state.pop("game", None)
        st.rerun()

    with st.expander("📖 게임 설명서"):
        st.markdown("""
        **1) 매매 및 진행**
        - **▶ 다음 캔들**: 다음 날의 캔들로 이동합니다. (게임당 최대 80개)
        - **전량 청산**: 보유한 모든 포지션을 즉시 종료합니다.
        
        **2) 게임 관리**
        - **🎲 날짜 변경**: 현재 종목 내에서 다른 시작 시점으로 무작위 이동합니다.
        - **📚 모델북**: `modelbook.txt`에 등록된 다른 종목으로 새로운 게임을 시작합니다.
        - **게임 종료 & 결과**: 게임을 끝내고 성과를 분석합니다.
        """)

# -------------- 차트 UI --------------
with chart_col:
    # ✨ UI 개편: 차트 설정 UI 변경
    with st.expander("차트 설정"):
        c1, c2, c3 = st.columns([2,2,1])
        c1.text_input("EMA 기간 (쉼표)", key="ema_input")
        c2.text_input("SMA 기간 (쉼표)", key="sma_input")
        c3.slider("차트 높이", min_value=400, max_value=1200, step=50, key="chart_height")

    mas_input = [("EMA", int(p.strip())) for p in st.session_state.ema_input.split(",") if p.strip().isdigit()] + \
                [("SMA", int(p.strip())) for p in st.session_state.sma_input.split(",") if p.strip().isdigit()]
    mas_tuple = tuple(mas_input)
    df_full = g.df
    visible_df_with_ma = add_cached_indicators(df_full, mas_tuple).iloc[:g.idx + 1]
    df_trade = (visible_df_with_ma.dropna(subset=["Open", "High", "Low", "Close"])
                                  .loc[visible_df_with_ma.Volume > 0]
                                  .assign(i=lambda d: range(len(d))))
    if df_trade.empty: st.error("표시할 데이터가 없습니다."); st.stop()

    max_candles = len(df_trade)
    min_candles = 10 
    if max_candles < min_candles: min_candles = max_candles
    if 'view_n' not in st.session_state or st.session_state.view_n is None: st.session_state.view_n = DEFAULT_VIEW_CANDLES
    st.session_state.view_n = max(min_candles, min(st.session_state.view_n, max_candles))
    
    st.number_input(
        "캔들 수", min_value=min_candles, max_value=max_candles, # ✨ UI 수정: '표시봉' -> '캔들 수'
        step=10, key="view_n"
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
            if not buy_df.empty: fig.add_scatter(x=buy_df['i'], y=buy_df['Low'] - span * 0.03, mode="markers", marker=dict(symbol="triangle-up", color="green", size=10), name="Buy", row=1, col=1)
            if not sell_df.empty: fig.add_scatter(x=sell_df['i'], y=sell_df['High'] + span * 0.03, mode="markers", marker=dict(symbol="triangle-down", color="red", size=10), name="Sell", row=1, col=1)
    shapes = []
    separator_line = dict(type='line', xref='paper', yref='paper', x0=0, y0=0.3, x1=1, y1=0.3, line=dict(color='black', width=1))
    shapes.append(separator_line)
    if st.session_state.stop_loss_price > 0:
        stop_loss_line = dict(type='line', xref='paper', yref='y', x0=0, y0=st.session_state.stop_loss_price, x1=1, y1=st.session_state.stop_loss_price, line=dict(color='black', width=2, dash='dash'))
        shapes.append(stop_loss_line)
        fig.add_annotation(x=end_i + PAD, y=st.session_state.stop_loss_price, text="손절 라인", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="black", size=12))
    
    fig.update_layout(height=st.session_state.chart_height, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=25, b=20, l=5, r=40), spikedistance=-1, shapes=shapes)
    fig.update_xaxes(showspikes=True, spikethickness=1, spikecolor="#999999", spikemode="across", spikesnap="cursor", range=[start_i - 1, end_i + PAD])
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#999999", spikemode="across", spikesnap="cursor")
    fig.update_yaxes(range=price_yrange, row=1, col=1)
    fig.update_yaxes(range=volume_yrange, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
