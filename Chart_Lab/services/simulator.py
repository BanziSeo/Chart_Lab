# services/simulator.py (v2 - Position 클래스 포함 최종본)

class Position:
    """
    개별 포지션의 상태를 관리하는 클래스.
    - side: 'long' 또는 'short'
    - qty: 보유 수량
    - avg_price: 평균 진입 단가
    """
    def __init__(self, side: str, entry_qty: int, entry_price: float, current_pos=None):
        self.side = side
        
        # 기존에 같은 방향의 포지션이 있다면, 물타기/불타기(평단가 조절)를 합니다.
        if current_pos and current_pos.side == self.side:
            current_value = current_pos.avg_price * current_pos.qty
            entry_value = entry_price * entry_qty
            total_qty = current_pos.qty + entry_qty
            
            self.avg_price = (current_value + entry_value) / total_qty
            self.qty = total_qty
        else:
            # 신규 진입 또는 반대 포지션 진입의 경우 (기존 포지션 덮어쓰기)
            self.avg_price = entry_price
            self.qty = entry_qty

    def close(self, exit_price: float) -> float:
        """
        포지션을 청산하고 실현 손익(PnL)을 계산하여 반환합니다.
        """
        if self.side == "long":
            pnl = (exit_price - self.avg_price) * self.qty
        else:  # short
            pnl = (self.avg_price - exit_price) * self.qty
        
        return pnl

class GameState:
    """
    전체 게임의 상태를 관리하는 클래스.
    """
    def __init__(self, df, idx: int, start_cash: int, tkr: str):
        self.df = df
        self.ticker = tkr
        self.idx = idx
        self.initial_cash = start_cash
        self.cash = start_cash
        self.pos: Position | None = None  # 포지션 상태 (Position 객체 또는 None)
        self.log = []

    @property
    def today(self):
        """현재 시점의 날짜(인덱스)를 반환합니다."""
        return self.df.index[self.idx]

    def next_candle(self):
        """다음 캔들로 이동합니다."""
        if self.idx < len(self.df) - 1:
            self.idx += 1

    def buy(self, qty: int):
        """매수 주문을 처리합니다."""
        price_now = self.df.Close.iloc[self.idx]
        self.pos = Position("long", qty, price_now, self.pos)
        self.log.append({"date": self.today, "action": "ENTER LONG", "price": price_now, "qty": qty})

    def sell(self, qty: int):
        """
        매도 주문을 처리합니다.
        (주: 현재 로직에서는 공매도(short) 진입으로 동작합니다. 
        단순 청산을 원하면 flat()을 사용해야 합니다.)
        """
        price_now = self.df.Close.iloc[self.idx]
        # 보유 수량보다 많이 팔면, 그만큼 공매도로 전환됩니다.
        # 이 부분의 로직은 원하는 전략에 따라 수정할 수 있습니다.
        if self.pos and self.pos.side == "long":
            # 보유 중인 long 포지션 청산
            self.flat() # 단순화: 일단 전량 청산 후 신규 short 진입
            self.pos = Position("short", qty, price_now)
        else:
            # 신규 short 진입 또는 short 포지션에 추가
            self.pos = Position("short", qty, price_now, self.pos)
        
        self.log.append({"date": self.today, "action": "ENTER SHORT", "price": price_now, "qty": qty})

    def flat(self):
        """보유 포지션을 전량 청산합니다."""
        if not self.pos:
            return
            
        price_now = self.df.Close.iloc[self.idx]
        pnl = self.pos.close(price_now)
        self.cash += pnl
        self.log.append({"date": self.today, "action": "EXIT", "price": price_now, "pnl": pnl})
        self.pos = None