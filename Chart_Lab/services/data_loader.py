import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE = Path(__file__).parent.parent / "data"

def get_price(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    단일 티커라도 yfinance가 멀티인덱스(열레벨 2개)로 줄 수 있음.
    첫 레벨(O/H/L/C/V)만 사용하도록 평탄화 후 CSV 캐시.
    """
    CACHE.mkdir(exist_ok=True)
    csv = CACHE / f"{ticker}_{years}y.csv"

    if csv.exists():
        return pd.read_csv(csv, index_col="Date", parse_dates=True)

    # ① 다운로드 (auto_adjust False 추천)
    df = yf.download(ticker, period=f"{years}y", auto_adjust=False, progress=False)

    # ② 멀티인덱스 → 단일 컬럼 이름
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ③ 인덱스 이름 통일 & 저장
    df.index.name = "Date"
    df.to_csv(csv)        # index_label 자동으로 index.name 사용
    return df
