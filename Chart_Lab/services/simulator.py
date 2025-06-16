# services/simulator.py (최종 점검 완료)

class Position:
    """
    개별 포지션의 상태를 관리하는 클래스.
    - side: 'long' 또는 'short'
    - qty: 보유 수량
    - avg_price: 평균 진입 단가
    """
    def __init__(self, side: str, entry_qty: int, entry_price: float, current_pos=None):
        self.side = side
        
        # 같은 방향으로 추가 진입 시, 평단가와 수량을 업데이트 (물타기/불타기)
        if current_pos and current_pos.side == self.side:
            current_value = current_pos.avg_price * current_pos.qty
            entry_value = entry_price * entry_qty
            total_qty = current_pos.qty + entry_qty
            
            self.avg_price = (current_value + entry_value) / total_qty
            self.qty = total_qty
        else: # 신규 진입
            self.avg_price = entry_price
            self.qty = entry_qty

    def close(self, exit_price: float) -> float:
        """포지션을 청산하고 실현 손익(PnL)을 계산하여 반환합니다."""
        if self.side == "long":
            pnl = (exit_price - self.avg_price) * self.qty
        else:  # short
            pnl = (self.avg_price - exit_price) * self.qty
        
        return pnl

class GameState:
    """전체 게임의 상태를 관리하는 클래스."""
    def __init__(self, df, idx: int, start_cash: int, tkr: str):
        self.df = df
        self.ticker = tkr
        self.idx = idx
        self.initial_cash = start_cash
        self.cash = start_cash
        self.pos: Position | None = None
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
        self.cash -= (price_now * qty) # 매수 시 현금 차감
        self.pos = Position("long", qty, price_now, self.pos)
        self.log.append({"date": self.today, "action": "ENTER LONG", "price": price_now, "qty": qty})

    def sell(self, qty: int):
        """매도(공매도) 주문을 처리합니다."""
        price_now = self.df.Close.iloc[self.idx]
        self.pos = Position("short", qty, price_now, self.pos)
        # 공매도 시 증거금 관련 로직은 여기서는 단순화합니다.
        self.log.append({"date": self.today, "action": "ENTER SHORT", "price": price_now, "qty": qty})

    def flat(self):
        """보유 포지션을 전량 청산하고 수수료를 차감합니다."""
        if not self.pos:
            return
            
        price_now = self.df.Close.iloc[self.idx]
        pnl = self.pos.close(price_now)
        
        trade_value = self.pos.qty * price_now
        fee = trade_value * 0.0014 # 수수료 0.14%

        self.cash += (pnl - fee)
        self.log.append({"date": self.today, "action": "EXIT", "price": price_now, "pnl": pnl, "fee": -fee})
        self.pos = None

