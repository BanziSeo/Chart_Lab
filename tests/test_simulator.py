import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from Chart_Lab.services.simulator import GameState


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


def test_buy_flat_updates_cash():
    df = make_df()
    g = GameState(df, idx=0, start_cash=1000)
    g.buy(1)
    g.next_candle()
    g.next_candle()
    g.flat()
    assert g.cash == 1010
    assert g.pos is None


def test_sell_flat_updates_cash():
    df = make_df()
    g = GameState(df, idx=0, start_cash=1000)
    g.sell(1)
    g.next_candle()
    g.next_candle()
    g.flat()
    assert g.cash == 990
    assert g.pos is None


def test_next_candle_and_today():
    df = make_df()
    g = GameState(df, idx=0, start_cash=1000)
    first_day = g.today
    g.next_candle()
    assert g.today > first_day


def make_big_df(n=210):
    data = {
        "Open": list(range(100, 100 + n)),
        "High": list(range(101, 101 + n)),
        "Low": list(range(99, 99 + n)),
        "Close": list(range(100, 100 + n)),
        "Volume": [1000] * n,
    }
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame(data, index=idx)


def test_start_idx_and_limit():
    df = make_big_df()
    g = GameState(df, idx=5, start_cash=1000)
    assert g.start_idx == 5
    for _ in range(199):
        assert g.next_candle()
    assert not g.next_candle()
    assert g.idx - g.start_idx == 199
