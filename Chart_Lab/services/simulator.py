# services/simulator.py
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


# ────────────────── 포지션 객체 ──────────────────
@dataclass
class Position:
    side: str            # "long" 또는 "short"
    qty: int
    avg_price: float
    stop: float | None = None

    def add(self, qty: int, price: float):
        """
        추가 진입(피라미딩) 시 가중 평균 단가로 재계산.
        """
        new_qty = self.qty + qty
        if new_qty == 0:        # (이론상) 전량 청산되는 경우
            self.qty = 0
            self.avg_price = 0
            return
        self.avg_price = (self.avg_price * self.qty + price * qty) / new_qty
        self.qty = new_qty


# ────────────────── 게임 상태 ──────────────────
@dataclass
class GameState:
    df: pd.DataFrame            # OHLC 데이터프레임
    idx: int                    # 현재 인덱스 번호
    cash: float = 10_000.0
    pos: Position | None = None
    log: List[Dict] = field(default_factory=list)

    # 현재 날짜·가격 -------------------------------------------------
    @property
    def today(self):
        return self.df.index[self.idx]

    @property
    def price(self) -> float:
        return float(self.df.Close.iloc[self.idx])

    # 매수 ---------------------------------------------------------
    def buy(self, qty: int, stop: float = 0.0):
        cost = self.price * qty
        if cost > self.cash:
            raise ValueError("현금이 부족합니다.")
        self.cash -= cost

        if self.pos is None:
            self.pos = Position("long", qty, self.price, stop or None)
        elif self.pos.side == "long":
            self.pos.add(qty, self.price)
            self.pos.stop = stop or self.pos.stop
        else:
            raise NotImplementedError("숏 포지션 보유 상태에서의 롱 진입은 미구현")

        self.log.append({"date": self.today, "action": "ENTER LONG",
                         "price": self.price})

    # 매도(공매도) --------------------------------------------------
    def sell(self, qty: int, stop: float = 0.0):
        cost = self.price * qty
        if cost > self.cash:
            raise ValueError("현금이 부족합니다.")
        self.cash -= cost

        if self.pos is None:
            self.pos = Position("short", qty, self.price, stop or None)
        elif self.pos.side == "short":
            self.pos.add(qty, self.price)
            self.pos.stop = stop or self.pos.stop
        else:
            raise NotImplementedError("롱 포지션 보유 상태에서의 숏 진입은 미구현")

        self.log.append({"date": self.today, "action": "ENTER SHORT",
                         "price": self.price})

    # 전량 청산 -----------------------------------------------------
    def flat(self):
        if self.pos is None:
            return

        pnl = 0.0
        if self.pos.side == "long":
            pnl = (self.price - self.pos.avg_price) * self.pos.qty
        else:  # short
            pnl = (self.pos.avg_price - self.price) * self.pos.qty

        self.cash += self.pos.qty * self.price + pnl
        self.log.append({"date": self.today, "action": "EXIT",
                         "price": self.price, "pnl": pnl})
        self.pos = None

    # 다음 캔들로 이동 --------------------------------------------
    def next_candle(self):
        if self.idx < len(self.df) - 1:
            self.idx += 1
