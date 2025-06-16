# app.py ───────────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, random, os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from services.data_loader import get_price
from services.indicators import add_mas
from services.simulator import GameState

st.set_page_config(layout="wide")
PAD, MARGIN = 20, 0.05        # x축 오른쪽 공백, y축 여유 비율

# ╭───────────────── 캐싱 헬퍼 ─────────────────╮
@st.cache_data
def add_indicators(df: pd.DataFrame, mas_tuple: tuple):
    """이동평균 계산 캐시. mas_tuple 은 (('EMA',10),('SMA',50)...) 형태"""
    mas = [(kind, per, True) for kind, per in mas_tuple]
    return add_mas(df.copy(), mas)

# ╭───────────────── 게임 생성/시작 ─────────────────╮
def create_game(tkr: str) -> GameState:
    df = get_price(tkr)
    today = pd.Timestamp.today().normalize()
    lo, hi = today - pd.DateOffset(years=5), today - pd.DateOffset(years=1)
    pool = [i for i,d in enumerate(df.index) if lo <= d <= hi and i >= 120]
    return GameState(df, idx=random.choice(pool))

def start_game(tkr: str):
    st.session_state.game = create_game(tkr)
    st.session_state.last_summary, st.session_state.view_n = None, 120
    st.rerun()

def start_random_modelbook():
    root = os.path.dirname(__file__)
    path = next((p for p in (os.path.join(root,"modelbook.txt"),
                             os.path.join(root,"Chart_Lab","modelbook.txt"))
                 if os.path.exists(p)), None)
    if not path:
        st.error("modelbook.txt 파일을 찾지 못했습니다."); return
    with open(path,"r",encoding="utf-8") as f:
        tickers=[t.strip().upper() for t in f.read().split(",") if t.strip()]
    if not tickers: st.error("modelbook.txt 에 티커가 없습니다"); return
    start_game(random.choice(tickers))

def jump_random_date():
    g=st.session_state.game
    today=pd.Timestamp.today().normalize()
    lo,hi=today-pd.DateOffset(years=5),today-pd.DateOffset(years=1)
    pool=[i for i,d in enumerate(g.df.index) if lo<=d<=hi and i>=120]
    g.idx, g.cash, g.pos, g.log = random.choice(pool), 10_000, None, []
    st.session_state.view_n = 120
    st.rerun()

# ╭───────────────── 첫 랜딩 ─────────────────╮
if "game" not in st.session_state:
    st.header("📈 차트 훈련소")
    code=st.text_input("시작할 티커 입력","")
    c1,c2=st.columns([1,1])
    if c1.button("새 게임 시작") and code.strip():
        start_game(code.strip().upper())
    if c2.button("모델북 랜덤 시작"):
        start_random_modelbook()
    if st.session_state.get("last_summary"):
        st.subheader("지난 게임 요약"); st.write(st.session_state.last_summary)
    st.stop()

# ╭───────────────── 게임 화면 ─────────────────╮
g: GameState = st.session_state.game
chart_col, side_col = st.columns([7,3])

# ── 이동평균 입력
ema_in=chart_col.text_input("EMA 기간(쉼표)","10,21")
sma_in=chart_col.text_input("SMA 기간(쉼표)","50,200")
mas_input=[("EMA",int(p)) for p in ema_in.split(",") if p.strip().isdigit()] + \
          [("SMA",int(p)) for p in sma_in.split(",") if p.strip().isdigit()]
mas_tuple=tuple(mas_input)   # 캐시 key 용

clr={("EMA",10):"red",("EMA",21):"blue",("SMA",50):"orange",("SMA",200):"black"}
pal=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

# ── 데이터 준비 + 캐시 MA
df_full = g.df
visible = add_indicators(df_full, mas_tuple).iloc[:g.idx+1]
df_trade=(visible.dropna(subset=["Open","High","Low","Close"])
                 .loc[visible.Volume>0]
                 .assign(i=lambda d: range(len(d))))
price_now=df_trade.Close.iloc[-1]

# ── AUTOSCALE 봉 수
if "view_n" not in st.session_state: st.session_state.view_n=120
view_n=st.columns([9,1])[1].number_input("표시봉",50,len(df_trade),
                                          st.session_state.view_n,10,
                                          label_visibility="collapsed")
if int(view_n)!=st.session_state.view_n:
    st.session_state.view_n=int(view_n)
view_n=st.session_state.view_n

start_i=df_trade.i.iloc[max(0,len(df_trade)-view_n)]
end_i=df_trade.i.iloc[-1]

sub=df_trade[df_trade.i>=start_i]
ymin=sub[["Low"]+[f"{k}{p}" for k,p in mas_tuple]].min().min()
ymax=sub[["High"]+[f"{k}{p}" for k,p in mas_tuple]].max().max()
span=ymax-ymin if ymax>ymin else 1
yrng=[ymin-span*MARGIN, ymax+span*MARGIN]

# ── 차트
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                  row_heights=[0.75,.25],vertical_spacing=.02)
for i,(k,p) in enumerate(mas_tuple):
    fig.add_scatter(x=df_trade.i,y=df_trade[f"{k}{p}"],
                    line=dict(width=1,color=clr.get((k,p),pal[i%len(pal)])),
                    name=f"{k}{p}",row=1,col=1)

fig.add_candlestick(x=df_trade.i,open=df_trade.Open,high=df_trade.High,
                    low=df_trade.Low,close=df_trade.Close,
                    increasing=dict(line=dict(color="black",width=1),
                                    fillcolor="rgba(0,0,0,0)"),
                    decreasing=dict(line=dict(color="black",width=1),
                                    fillcolor="black"),
                    name="Price",row=1,col=1)

if g.pos:
    avg=g.pos.avg_price
    fig.add_hline(y=avg,line=dict(color=("green" if g.pos.side=="long" else "red"),
                                  width=1,dash="dash"),
                  annotation_text=f"Avg {avg:.2f}",annotation_position="top left")

# ── 매매 화살표
log_df=pd.DataFrame(g.log)
if not log_df.empty:
    buy_df=log_df[log_df.action.str.contains("ENTER") & log_df.action.str.contains("LONG")]
    sell_df=log_df[log_df.action.str.contains("ENTER") & log_df.action.str.contains("SHORT")]
    if not buy_df.empty:
        bx=[df_trade.loc[d].i for d in buy_df.date]
        by=[df_trade.loc[d].Low - span*0.03 for d in buy_df.date]
        fig.add_scatter(x=bx,y=by,mode="markers",
                        marker=dict(symbol="triangle-up",color="green",size=8),
                        name="Buy",row=1,col=1)
    if not sell_df.empty:
        sx=[df_trade.loc[d].i for d in sell_df.date]
        sy=[df_trade.loc[d].High + span*0.03 for d in sell_df.date]
        fig.add_scatter(x=sx,y=sy,mode="markers",
                        marker=dict(symbol="triangle-down",color="red",size=8),
                        name="Sell",row=1,col=1)

vol_c=["black" if c<=o else "white" for o,c in zip(df_trade.Open,df_trade.Close)]
fig.add_bar(x=df_trade.i,y=df_trade.Volume,marker_color=vol_c,
            marker_line_color="black",marker_line_width=.5,
            name="Volume",row=2,col=1)
fig.add_scatter(x=df_trade.i,y=df_trade.Volume.rolling(50).mean(),
                line=dict(width=1,color="blue",dash="dot"),
                name="Vol SMA50",row=2,col=1)

gap=max(len(df_trade)//8,1)
fig.update_layout(xaxis=dict(tickmode="array",
                             tickvals=df_trade.i[::gap],
                             ticktext=df_trade.index.strftime("%Y-%m-%d")[::gap],
                             tickangle=-45),
                  xaxis_rangeslider_visible=False,
                  hovermode="x unified",
                  margin=dict(t=25,b=20,l=5,r=5))
fig.update_yaxes(range=yrng,fixedrange=True,showgrid=False,row=1,col=1)
fig.update_yaxes(showgrid=False,fixedrange=True,row=2,col=1)
fig.update_xaxes(range=[start_i,end_i+PAD])

# ── placeholder 재사용 → 깜빡임 최소화
if "chart_slot" not in st.session_state:
    st.session_state.chart_slot = chart_col.empty()
st.session_state.chart_slot.plotly_chart(fig,use_container_width=True,
                                         config={"displayModeBar":False})

# ── 상태
with side_col:
    pos_val=g.pos.qty*price_now if g.pos else 0
    equity=g.cash+pos_val
    unreal=(price_now-g.pos.avg_price)*g.pos.qty*(1 if g.pos and g.pos.side=="long" else -1) if g.pos else 0
    st.subheader("상태")
    st.write(f"**날짜**: {g.today.date()}")
    st.write(f"**자본(실현)**: ${g.cash:,.2f}")
    st.write(f"**미실현 P/L**: {'+' if unreal>=0 else ''}{unreal:,.2f}")
    if g.pos:
        pct=pos_val/equity*100 if equity else 0
        st.write(f"**포지션**: {g.pos.side.upper()} {g.pos.qty}주 "
                 f"@ {g.pos.avg_price:.2f} (**{pct:.1f}%**)")

# ── 매수/매도
buy_cols=st.columns([1,1,1,10],gap="small")
for col,p in zip(buy_cols[:3],[.25,.50,1]):
    if col.button(f"매수 {int(p*100)}%"):
        qty=int((g.cash*p)//price_now)
        if qty>0:
            g.buy(qty); st.rerun()

sell_cols=st.columns([1,1,1,10],gap="small")
for col,p in zip(sell_cols[:3],[.25,.50,1]):
    if col.button(f"매도 {int(p*100)}%"):
        qty=int((g.cash*p)//price_now)
        if qty>0:
            g.sell(qty); st.rerun()

# ── 컨트롤
ctl=st.columns([1,1,1,1,10],gap="small")
if ctl[0].button("전량 청산") and g.pos:
    g.flat(); st.rerun()
if ctl[1].button("다음 캔들 ▶"):
    g.next_candle(); st.rerun()
if ctl[2].button("티커 랜덤날짜"):
    jump_random_date()
if ctl[3].button("모델북 랜덤"):
    start_random_modelbook()

# ── 종료 & 새 게임
st.markdown("---")
new_col,end_btn,start_btn=st.columns([2,1,1])
new_code=new_col.text_input("새 티커","")
if end_btn.button("게임 종료/결과"):
    realized=sum(x.get("pnl",0) for x in g.log)
    wins=[x for x in g.log if x.get("pnl",0)>0]
    trades=[x for x in g.log if "pnl" in x]
    equity=g.cash+(g.pos.qty*price_now if g.pos else 0)
    st.session_state.last_summary={
        "최종 자본":f"${equity:,.2f}",
        "총 실현 P/L":f"{realized:+,.2f}",
        "거래 횟수":len(trades),
        "승률":f"{(len(wins)/len(trades)*100):.1f}%" if trades else "N/A"}
    st.session_state.pop("game"); st.rerun()
if start_btn.button("새 게임 시작") and new_code.strip():
    start_game(new_code.strip().upper())
