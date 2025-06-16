# ───────────────────────────── Chart_Lab/app.py ─────────────────────────────
"""Interactive Trading Simulator – Streamlit app
   깜빡임 최소화 + 시작 자본 입력 + 거래 로그 복원 버전
"""
import os, random, streamlit as st, pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from services.data_loader import get_price
from services.indicators import add_mas
from services.simulator import GameState

# ────────────────────────── Page config ──────────────────────────
st.set_page_config(page_title="차트 훈련소", page_icon="📈", layout="wide")

# ────────────────────── 캐싱: 가격·지표 계산 ──────────────────────
@st.cache_data
def cache_price(tkr: str):
    return get_price(tkr)

@st.cache_data
def cache_indicators(df: pd.DataFrame, mas_tuple: tuple):
    mas = [(k, p, True) for k, p in mas_tuple]  # indicator util expects 3‑tuple
    return add_mas(df.copy(), mas)

# ───────────────────────── 게임 생성 함수 ─────────────────────────

def create_game(tkr: str, capital: int) -> GameState:
    df = cache_price(tkr)
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    idx_pool = [i for i, d in enumerate(df.index) if lo <= d <= hi and i >= 120]
    start_idx = random.choice(idx_pool)
    return GameState(df, idx=start_idx, start_cash=capital)


def start_game(tkr: str, capital: int):
    st.session_state.game = create_game(tkr, capital)
    st.session_state.last_summary = None
    st.session_state.view_n = 120
    st.rerun()


def load_modelbook(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [t.strip().upper() for t in f.read().split(",") if t.strip()]


def start_random_modelbook(capital: int):
    root = os.path.dirname(__file__)
    mb_path = os.path.join(root, "modelbook.txt")
    if not os.path.exists(mb_path):
        st.error("modelbook.txt 파일을 찾지 못했습니다.")
        return
    tickers = load_modelbook(mb_path)
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

# ───────────────────────── 첫 랜딩 화면 ─────────────────────────
if "game" not in st.session_state:
    st.header("📈 차트 훈련소")

    col_code, col_cash = st.columns([2, 1])
    code_in = col_code.text_input("시작할 티커 입력", "")
    cash_in = col_cash.number_input("시작 자본($)", 1_000, 1_000_000, 100_000, step=1_000)

    c1, c2 = st.columns([1, 1])
    if c1.button("새 게임 시작") and code_in.strip():
        start_game(code_in.strip().upper(), int(cash_in))
    if c2.button("모델북 랜덤 시작"):
        start_random_modelbook(int(cash_in))

    if st.session_state.get("last_summary"):
        st.subheader("지난 게임 요약")
        st.json(st.session_state.last_summary)
    st.stop()

# ───────────────────────── 메인 게임 화면 ─────────────────────────
g: GameState = st.session_state.game
chart_col, side_col = st.columns([7, 3])

# ── 이동평균 입력
ema_txt = chart_col.text_input("EMA 기간(쉼표)", "10,21")
sma_txt = chart_col.text_input("SMA 기간(쉼표)", "50,200")
mas_input = [("EMA", int(p)) for p in ema_txt.split(",") if p.strip().isdigit()] + \
            [("SMA", int(p)) for p in sma_txt.split(",") if p.strip().isdigit()]
mas_tuple = tuple(mas_input)

# ── 데이터 준비
full_df = cache_price(g.ticker)
vis_df = cache_indicators(full_df, mas_tuple).iloc[: g.idx + 1]
price_now = vis_df.Close.iloc[-1]

# ── 뷰 윈도우 길이 조절
if "view_n" not in st.session_state:
    st.session_state.view_n = 120
view_n = st.session_state.view_n
view_n = chart_col.number_input("표시봉", 50, len(vis_df), view_n, step=10, key="view_n")

start_i = max(0, len(vis_df) - view_n)
sub_df = vis_df.iloc[start_i:]

# ── 차트 생성 (placeholder 재사용)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
# …(캔들, MA, 볼륨, 화살표 그리는 기존 코드 그대로)…
# 권장: 기존 차트 생성 로직 붙여 넣기 (지면 관계상 생략)

if "chart_slot" not in st.session_state:
    st.session_state.chart_slot = chart_col.empty()

st.session_state.chart_slot.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── 측면 상태 패널
realized_pl = g.cash - g.initial_cash
unreal_pl = g.pos.unreal(price_now) if g.pos else 0
pos_val = g.pos.market_value(price_now) if g.pos else 0
equity = g.cash + pos_val

with side_col:
    st.subheader("상태")
    st.write(f"**날짜**: {g.today.date()}")
    st.write(f"**현금(실현 자본)**: ${g.cash:,.2f}")
    st.write(f"**실현 P/L**: {realized_pl:+,.2f}")
    st.write(f"**미실현 P/L**: {unreal_pl:+,.2f}")
    if g.pos:
        pct = pos_val / equity * 100 if equity else 0
        st.write(f"**포지션**: {g.pos.side.upper()} {g.pos.qty}주 @ {g.pos.avg_price:.2f} (**{pct:.1f}%**) ")

    st.write("### 거래 로그")
    if g.log:
        st.dataframe(pd.DataFrame(g.log))
    else:
        st.write("_empty_")

# ── 매매 & 컨트롤 버튼 (기존 로직 그대로) …
# (buy/sell/flat/next_candle/jump_random/modelbook_random 구현 생략)

# ───────────────────────────── END app.py ─────────────────────────────