# ──────────────────── services/simulator.py (핵심 변경) ────────────────────
"""Only the modified parts are shown – rest of the class stays the same."""
class GameState:
    def __init__(self, df, idx: int, start_cash: int = 100_000, tkr: str | None = None):
        self.df = df
        self.idx = idx
        self.initial_cash = start_cash
        self.cash = start_cash
        self.pos = None  # Position object or None
        self.log = []
        # store the ticker symbol if provided
        self.ticker = tkr if tkr is not None else ""

    # ... today, next_candle methods unchanged ...

    def buy(self, qty: int):
        px = self.df.Close.iloc[self.idx]
        self.pos = Position("long", qty, px, self.pos)
        self.log.append({"date": self.today, "action": "ENTER LONG", "price": px, "qty": qty})

    def sell(self, qty: int):
        px = self.df.Close.iloc[self.idx]
        self.pos = Position("short", qty, px, self.pos)
        self.log.append({"date": self.today, "action": "ENTER SHORT", "price": px, "qty": qty})

    def flat(self):
        if not self.pos:
            return
        px = self.df.Close.iloc[self.idx]
        pnl = self.pos.close(px)
        self.cash += pnl
        self.log.append({"date": self.today, "action": "EXIT", "price": px, "pnl": pnl})
        self.pos = None

# Position class는 market_value(), unreal() 헬퍼 메서드를 갖고 있다고 가정