from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .engine import OHLCV
import numpy as np
import pandas as pd


class Returns(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, closes):
        return closes.pct_change(self.win-1)


class LogReturns(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, closes):
        return closes.pct_change(self.win-1).log()


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
        self.adjust = adjust
        # Length required to achieve 99.97% accuracy, np.log(1-99.97/100) / np.log(alpha)
        # simplification to 4 * (span+1). 3.45 achieve 99.90%, 2.26 99.00%
        self.win = int(4.5 * (self.span + 1))
        # For GPU efficiency here, weight is not float64 type, EMA 50+ will leading to inaccurate,
        # and so window greater than 200 produces a very small and negligible weight
        self.win = min(self.win, 200)
        self.weight = np.full(self.win, 1 - self.alpha) ** np.arange(self.win - 1, -1, -1)
        if self.adjust:
            self.weight = self.weight / sum(self.weight)  # to sum one

    def compute(self, data):
        if isinstance(data, pd.DataFrame):
            return data.ewm(span=self.span, min_periods=self.win, adjust=self.adjust).mean()
        else:
            # todo for cuda
            # dat_rows, dat_cols = data.shape
            # stride_x, stride_y = data.strides
            # new_shape = (dat_rows - self.win + 1, dat_cols, self.win)
            # new_stride = (stride_x, stride_y, stride_x)
            # for i in range(self.win):
            #       index = (row, col, i) * new_stride
            #       sum += data[index] * self.weight[i]
            weighted_mean = data.rolling(self.win).apply(lambda x: (x * self.weight).sum())
            if self.adjust:
                return weighted_mean
            else:
                alpha = self.alpha
                return alpha * weighted_mean + (data.shift(self.win - 1) * (1 - alpha) ** self.win)


class AverageDollarVolume(CustomFactor):
    inputs = [OHLCV.close, OHLCV.volume]

    def compute(self, closes, volumes):
        if self.win == 1:
            return closes * volumes
        else:
            return (closes * volumes).rolling(self.win).sum() / self.win


class AnnualizedVolatility(CustomFactor):
    inputs = [Returns(win=2), 252]
    window_length = 20

    def compute(self, returns, annualization_factor):
        return returns.rolling(self.win).std(ddof=0) * (annualization_factor ** .5)


MA = SimpleMovingAverage
SMA = SimpleMovingAverage
EMA = ExponentialWeightedMovingAverage
