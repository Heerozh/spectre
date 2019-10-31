from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .engine import OHLCV
import numpy as np
import pandas as pd


class Returns(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, close):
        return (close.shift(1) - close) / close


class SimpleMovingAverage(CustomFactor):
    inputs = [OHLCV.close]

    def compute(self, data):
        # cuda object should also have shift, rolling and mean
        return data.rolling(self.win).mean()


class WeightedAverageValue(CustomFactor):
    def compute(self, base, weight):
        return (base * weight).rolling(self.win).sum() / weight.rolling(self.win).sum()


class VWAP(WeightedAverageValue):
    inputs = (OHLCV.close, OHLCV.volume)


class ExponentialWeightedMovingAverage(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None,
                 adjust=False):
        super().__init__(win, inputs)
        self.span = self.win
        self.alpha = (2.0 / (1.0 + self.span))
        self.r_alpha = (1.0 - self.alpha)
        self.adjust = adjust
        # Length required to achieve 99.97% accuracy, np.log(1-99.97/100) / np.log(alpha)
        # simplification to 4 * (span+1). 3.45 achieve 99.90%, 2.26 99.00%
        self.win = int(4.5 * (self.span + 1))
        # window greater than 200 produces a very small and negligible weight
        self.win = min(self.win, 200)
        if self.adjust:
            self.weight = np.full(self.win, self.alpha) ** np.arange(self.win + 1, 1, -1)
            self.weight = self.weight / sum(self.weight)  # to sum one
        else:
            self.weight = np.full(self.win, self.r_alpha) ** np.arange(self.win - 1, -1, -1)

    def compute(self, data):
        # numpy.lib.stride_tricks.as_strided won't increase any performance.
        # for future compatibility, choose not to use df.ewm
        # return data.ewm(span=self.span, min_periods=self.win, adjust=True).mean()
        weighted_mean = data.rolling(self.win).apply(lambda x: (x * self.weight).sum())
        if self.adjust:
            return weighted_mean
        else:
            return self.alpha * weighted_mean \
                   + (data.shift(self.win - 1) * self.r_alpha ** self.win)


class AverageDollarVolume(CustomFactor):
    inputs = [OHLCV.close, OHLCV.volume]

    def compute(self, close, volume):
        if self.win == 1:
            return close * volume
        else:
            return (close * volume).rolling(self.win).sum() / self.win


class AnnualizedVolatility(CustomFactor):
    inputs = [Returns(win=2)]
    params = (252,)
    window_length = 252

    def compute(self, returns, annualization_factor):
        return returns.rolling(self.win).std() * (annualization_factor ** .5)


MA = SimpleMovingAverage
SMA = SimpleMovingAverage
EMA = ExponentialWeightedMovingAverage
