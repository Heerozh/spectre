"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Sequence
import numpy as np
import torch
import math
from .factor import BaseFactor, CustomFactor
from ..parallel import nansum, nanmean
from .engine import OHLCV
from ..config import Global


class Returns(CustomFactor):
    """ Returns by tick (not by time) """
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, closes):
        # missing data considered as delisted, calculated on the last day's data.
        return closes.last_nonnan(offset=1) / closes.first() - 1


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


class LinearWeightedAverage(CustomFactor):
    _min_win = 2

    def __init__(self, win=None, inputs=None):
        super().__init__(win, inputs)
        self.weight = torch.arange(1, self.win + 1, dtype=Global.float_type)
        self.weight = self.weight / self.weight.sum()

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        self.weight = self.weight.to(device=engine.device, copy=False)

    def compute(self, base):
        def _weight_mean(_base):
            return nansum(_base * self.weight, dim=2)

        return base.agg(_weight_mean)


class VWAP(WeightedAverageValue):
    inputs = (OHLCV.close, OHLCV.volume)


class ExponentialWeightedMovingAverage(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def __init__(self, span: int = None, inputs: Sequence[BaseFactor] = None,
                 adjust=False, half_life: float = None):
        if span is not None:
            self.alpha = (2.0 / (1.0 + span))
            # Length required to achieve 99.97% accuracy, np.log(1-99.97/100) / np.log(alpha)
            # simplification to 4 * (span+1). 3.45 achieve 99.90%, 2.26 99.00%
            self.win = int(4.5 * (span + 1))
        else:
            self.alpha = 1 - math.exp(math.log(0.5) / half_life)
            self.win = 15 * half_life

        super().__init__(None, inputs)
        self.adjust = adjust
        self.weight = np.full(self.win, 1 - self.alpha) ** np.arange(self.win - 1, -1, -1)
        if self.adjust:
            self.weight = self.weight / sum(self.weight)  # to sum one

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        if not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight, dtype=Global.float_type, device=engine.device)

    def compute(self, data):
        self.weight = self.weight.to(device=data.device)
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


class ElementWiseMax(CustomFactor):
    _min_win = 1

    def __init__(self, win=None, inputs=None):
        super().__init__(win, inputs)
        assert self.win == 1

    @classmethod
    def binary_fill_na(cls, a, b, value):
        a = a.clone()
        b = b.clone()
        if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64, torch.float16):
            a = a.to(Global.float_type)
            b = b.to(Global.float_type)

        a.masked_fill_(torch.isnan(a), value)
        b.masked_fill_(torch.isnan(b), value)
        return a, b

    def compute(self, a, b):
        ret = torch.max(*ElementWiseMax.binary_fill_na(a, b, -np.inf))
        ret.masked_fill_(torch.isinf(ret), np.nan)
        return ret


class ElementWiseMin(CustomFactor):
    _min_win = 1

    def __init__(self, win=None, inputs=None):
        super().__init__(win, inputs)
        assert self.win == 1

    def compute(self, a, b):
        ret = torch.min(*ElementWiseMax.binary_fill_na(a, b, np.inf))
        ret.masked_fill_(torch.isinf(ret), np.nan)
        return ret


class RollingArgMax(CustomFactor):
    _min_win = 2

    def compute(self, data):
        def _argmax(_data):
            ret = (_data.argmax(dim=2) + 1.) / self.win
            return ret.to(Global.float_type)

        return data.agg(_argmax)


class RollingArgMin(CustomFactor):
    _min_win = 2

    def compute(self, data):
        def _argmin(_data):
            ret = (_data.argmin(dim=2) + 1.) / self.win
            return ret.to(Global.float_type)

        return data.agg(_argmin)


class ConstantsFactor(CustomFactor):
    def __init__(self, value, like=OHLCV.open):
        self.value = value
        super().__init__(1, inputs=[like])

    def compute(self, x):
        return torch.full(x.shape, self.value, device=x.device, dtype=x.dtype)


MA = SimpleMovingAverage
SMA = SimpleMovingAverage
EMA = ExponentialWeightedMovingAverage
