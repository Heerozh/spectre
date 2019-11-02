from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .basic import MA, EMA
from .statistical import STDDEV
from .engine import OHLCV
import numpy as np
import pandas as pd


class NormalizedBollingerBands(CustomFactor):
    inputs = (OHLCV.close, 2)
    win = 20
    _min_win = 2

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        super().__init__(win, inputs)
        comm_inputs = (self.inputs[0],)
        k = self.inputs[1]
        self.inputs = (self.inputs[0],
                       MA(win=self.win, inputs=comm_inputs),
                       STDDEV(win=self.win, inputs=comm_inputs),
                       k)
        self.win = 1

    def compute(self, close, ma, std, k):
        return (close - ma) / (k * std)


class MovingAverageConvergenceDivergenceSignal(EMA):
    """
    engine.add( MACD(win=sign, inputs=(EMA(win=fast), EMA(win=slow))) )
    or
    engine.add( MACD().normalized() )
    """
    inputs = (OHLCV.close,)
    win = 9
    _min_win = 2

    def __init__(self, fast=12, slow=26, sign=9, inputs: Optional[Sequence[BaseFactor]] = None,
                 adjust=False):
        super().__init__(sign, inputs, adjust)
        self.inputs = (EMA(inputs=inputs, win=fast) - EMA(inputs=inputs, win=slow),)

    def normalized(self):
        """In order not to double the calculation, reuse `inputs` factor here"""
        macd = self.inputs[0]
        sign = self
        return macd - sign


class TrueRange(CustomFactor):
    """ATR = MA(inputs=(TrueRange(),))"""
    inputs = (OHLCV.high, OHLCV.low, OHLCV.close)
    win = 2
    _min_win = 2

    def compute(self, high, low, close):
        high_to_low = high - low
        high_to_prev_close = (high - close.shift(1)).abs()
        low_to_prev_close = (low - close.shift(1)).abs()
        max1 = high_to_low.where(high_to_low > high_to_prev_close, high_to_prev_close)
        return max1.where(max1 > low_to_prev_close, low_to_prev_close)


class RSI(CustomFactor):
    inputs = (OHLCV.close,)
    win = 144
    _min_win = 2

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        super().__init__(win, inputs)
        self.win = self.win + 1  # +1 for 1 day diff

    def compute(self, close):
        diffs = close.diff(1)
        up = diffs.clip_lower(0)
        up = up.rolling(self.win-1).mean()  # Cutler's RSI, more stable, independent to data length
        # up = up.ewm(com=14-1, adjust=False).mean()  # Wilder's RSI
        down = diffs.clip_upper(0)
        down = down.rolling(self.win-1).mean().abs()
        # down = down.ewm(com=14-1, adjust=False).mean().abs()  # Wilder RSI

        return 100 - (100 / (1 + up / down))


BBANDS = NormalizedBollingerBands
MACD = MovingAverageConvergenceDivergenceSignal
TRANGE = TrueRange
