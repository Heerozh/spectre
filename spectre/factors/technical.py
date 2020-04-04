"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .basic import MA, EMA
from .statistical import STDDEV
from .engine import OHLCV
from ..parallel import nanmean
import numpy as np
import torch


class BollingerBands(CustomFactor):
    """ usage: BBANDS(win, inputs=[OHLCV.close, k]), k is constant normally 2 """
    inputs = (OHLCV.close, 2)
    win = 20
    _min_win = 2

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        super().__init__(win, inputs)
        if len(self.inputs) < 2:
            raise ValueError("BollingerBands's inputs needs 2 inputs, "
                             "inputs=[OHLCV.close, k]), k is constant normally 2.")
        comm_inputs = (self.inputs[0],)
        k = self.inputs[1]
        self.inputs = (self.inputs[0],
                       MA(win=self.win, inputs=comm_inputs),
                       STDDEV(win=self.win, inputs=comm_inputs),
                       k)
        self.win = 1

    def compute(self, closes, ma, std, k):
        d = k * std
        up = ma + d
        down = ma - d
        return torch.cat([up.unsqueeze(-1), ma.unsqueeze(-1), down.unsqueeze(-1)], dim=-1)

    def normalized(self):
        return NormalizedBollingerBands(self.win, self.inputs)


class NormalizedBollingerBands(CustomFactor):
    def compute(self, closes, ma, std, k):
        return (closes - ma) / (k * std)


class MovingAverageConvergenceDivergenceSignal(EMA):
    """
    engine.add( MACD(fast, slow, sign, inputs=[OHLCV.close]) )
    or
    engine.add( MACD().normalized() )
    """
    inputs = (OHLCV.close,)
    win = 9
    _min_win = 2

    def __init__(self, fast=12, slow=26, sign=9, inputs: Optional[Sequence[BaseFactor]] = None,
                 adjust=False):
        super().__init__(sign, inputs, adjust)
        self.inputs = (EMA(inputs=self.inputs, span=fast) - EMA(inputs=self.inputs, span=slow),)

    def normalized(self):
        # In order not to double the calculation, reuse `inputs` factor here
        macd = self.inputs[0]
        sign = self
        return macd - sign


class TrueRange(CustomFactor):
    """ATR = MA(14, inputs=(TrueRange(),))"""
    inputs = (OHLCV.high, OHLCV.low, OHLCV.close)
    win = 2
    _min_win = 2

    def compute(self, highs, lows, closes):
        high_to_low = highs.last() - lows.last()
        high_to_prev_close = (highs.last() - closes.first()).abs()
        low_to_prev_close = (lows.last() - closes.first()).abs()
        max1 = high_to_low.where(high_to_low > high_to_prev_close, high_to_prev_close)
        return max1.where(max1 > low_to_prev_close, low_to_prev_close)


class RSI(CustomFactor):
    """ usage: RSI(win, inputs=[OHLCV.close]) """
    inputs = (OHLCV.close,)
    win = 14
    _min_win = 2

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        super().__init__(win, inputs)
        self.win = self.win + 1  # +1 for 1 day diff

    def compute(self, closes):
        def _rsi(_closes):
            shift = _closes.roll(1, dims=2)
            shift = shift.contiguous()
            shift[:, :, 0] = np.nan
            diff = _closes - shift
            up = diff.clamp(min=0)
            down = diff.clamp(max=0)
            # Cutler's RSI, more stable, independent to data length
            up = nanmean(up[:, :, 1:], dim=2)
            down = nanmean(down[:, :, 1:], dim=2).abs()
            return 100 - (100 / (1 + up / down))
            # Wilder's RSI
            # up = up.ewm(com=14-1, adjust=False).mean()
            # down = down.ewm(com=14-1, adjust=False).mean().abs()
        return closes.agg(_rsi)

    def normalized(self):
        return self / 50 - 1


class FastStochasticOscillator(CustomFactor):
    """ usage: STOCHF(win, inputs=[OHLCV.high, OHLCV.low, OHLCV.close]) """
    inputs = (OHLCV.high, OHLCV.low, OHLCV.close)
    win = 14
    _min_win = 2

    def compute(self, highs, lows, closes):
        highest_highs = highs.nanmax()
        lowest_lows = lows.nanmin()
        k = (closes.last() - lowest_lows) / (highest_highs - lowest_lows)

        return k * 100

    def normalized(self):
        return self / 100 - 0.5


BBANDS = BollingerBands
MACD = MovingAverageConvergenceDivergenceSignal
TRANGE = TrueRange
STOCHF = FastStochasticOscillator
