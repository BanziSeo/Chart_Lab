import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Chart_Lab.services.simulator import GameState, trade_summary


def make_df():
    data = {
        "Open": [100, 105, 110],
        "High": [101, 106, 111],
        "Low": [99, 104, 109],
        "Close": [100, 105, 110],
        "Volume": [1000, 1000, 1000],
    }
    idx = pd.date_range("2020-01-01", periods=3)
    return pd.DataFrame(data, index=idx)


def test_trade_summary_metrics():
    df = make_df()
    g = GameState(df, idx=0, start_cash=1000, tkr="ABC")

    g.buy(1)
    g.next_candle()
    g.flat()  # +5 pnl

    g.sell(1)
    g.next_candle()
    g.flat()  # -5 pnl

    summary = trade_summary(g)

    assert summary["ticker"] == "ABC"
    assert summary["total_pnl"] == 0.0
    assert summary["max_trade_pnl"] == 5.0
    assert summary["win_rate"] == 50.0

