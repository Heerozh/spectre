"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from ..parallel import nansum, nanmean
from .engine import OHLCV
import numpy as np
import torch


class Returns(CustomFactor):
    """ Returns by tick (not by time) """
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, closes):
        # missing data considered as delisted, calculated on the last day's data.
        return closes.last_nonnan() / closes.first() - 1


class LogReturns(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, closes):
        return (closes.last_nonnan() / closes.first()).log()


class SimpleMovingAverage(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2

    def compute(self, data):
        return data.nanmean()


class WeightedAverageValue(CustomFactor):
    _min_win = 2

    def compute(self, base, weight):
        def _weight_mean(_base, _weight):
            return nansum(_base * _weight, dim=2) / nansum(_weight, dim=2)

        return base.agg(_weight_mean, weight)


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
        self.weight = np.full(self.win, 1 - self.alpha) ** np.arange(self.win - 1, -1, -1)
        if self.adjust:
            self.weight = self.weight / sum(self.weight)  # to sum one

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        if not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight, dtype=torch.float32, device=engine.device)

    def compute(self, data):
        weighted_mean = data.agg(lambda x: nansum(x * self.weight, dim=2))
        if self.adjust:
            return weighted_mean
        else:
            shifted = data.last().roll(self.win - 1, dims=1)
            shifted[:, 0:self.win - 1] = 0
            alpha = self.alpha
            return alpha * weighted_mean + (shifted * (1 - alpha) ** self.win)


class AverageDollarVolume(CustomFactor):
    inputs = [OHLCV.close, OHLCV.volume]

    def compute(self, closes, volumes):
        if self.win == 1:
            return closes * volumes
        else:
            return closes.agg(lambda c, v: nanmean(c * v, dim=2), volumes)


class AnnualizedVolatility(CustomFactor):
    inputs = [Returns(win=2), 252]
    window_length = 20
    _min_win = 2

    def compute(self, returns, annualization_factor):
        return returns.nanstd() * (annualization_factor ** .5)


MA = SimpleMovingAverage
SMA = SimpleMovingAverage
EMA = ExponentialWeightedMovingAverage
