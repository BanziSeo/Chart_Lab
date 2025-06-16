"""Simple trading simulator components used by the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Position:
    """Represents a single long or short position."""

    side: str  # "long" or "short"
    qty: int
    entry: float

    def market_value(self, price: float) -> float:
        """Return the directional market value of the position."""
        direction = 1 if self.side == "long" else -1
        return self.qty * price * direction

    def unreal(self, price: float) -> float:
        """Calculate unrealized PnL at the given price."""
        direction = 1 if self.side == "long" else -1
        return (price - self.entry) * self.qty * direction

    def close(self, price: float) -> float:
        """Close the position and return realized PnL."""
        return self.unreal(price)


class GameState:
    """Holds the state for a trading simulation."""

    def __init__(self, df: pd.DataFrame, idx: int, start_cash: int = 100_000, tkr: Optional[str] = None) -> None:
        self.df = df
        self.idx = idx
        self.initial_cash = start_cash
        self.cash = float(start_cash)
        self.pos: Optional[Position] = None
        self.log: List[Dict[str, Any]] = []
        self.ticker = tkr.upper() if tkr else ""

    @property
    def today(self) -> pd.Timestamp:
        """Current date in the price data."""
        return self.df.index[self.idx]

    def next_candle(self) -> bool:
        """Advance to the next bar. Returns False if at the end."""
        if self.idx + 1 >= len(self.df):
            return False
        self.idx += 1
        return True

    def _price(self) -> float:
        return float(self.df.Close.iloc[self.idx])

    def buy(self, qty: int) -> None:
        """Enter a long position at the current price."""
        price = self._price()
        self.pos = Position("long", qty, price)
        self.log.append({"date": self.today, "action": "ENTER LONG", "price": price, "qty": qty})

    def sell(self, qty: int) -> None:
        """Enter a short position at the current price."""
        price = self._price()
        self.pos = Position("short", qty, price)
        self.log.append({"date": self.today, "action": "ENTER SHORT", "price": price, "qty": qty})

    def flat(self) -> None:
        """Exit any open position and realise PnL."""
        if not self.pos:
            return
        price = self._price()
        pnl = self.pos.close(price)
        self.cash += pnl
        self.log.append({"date": self.today, "action": "EXIT", "price": price, "pnl": pnl})
        self.pos = None

    @property
    def equity(self) -> float:
        """Return cash plus unrealised PnL."""
        unreal = self.pos.unreal(self._price()) if self.pos else 0.0
        return self.cash + unreal


def trade_summary(game: GameState) -> Dict[str, float | str]:
    """Return basic statistics for the finished game."""

    # Gather PnL from every exit in the log
    trade_pnls = [e["pnl"] for e in game.log if e.get("action") == "EXIT" and "pnl" in e]

    total_pnl = float(sum(trade_pnls))
    win_rate = float(sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) * 100) if trade_pnls else 0.0
    max_trade_pnl = float(max(trade_pnls)) if trade_pnls else 0.0

    return {
        "ticker": game.ticker,
        "total_pnl": total_pnl,
        "max_trade_pnl": max_trade_pnl,
        "win_rate": win_rate,
    }
