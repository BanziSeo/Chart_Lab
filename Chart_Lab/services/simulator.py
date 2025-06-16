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
        self.start_idx = idx
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
        """Advance to the next bar. Returns False if at the end or limit."""
        if self.idx + 1 >= len(self.df):
            return False
        if self.idx - self.start_idx >= 199:
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
