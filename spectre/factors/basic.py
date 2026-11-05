"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import math
from collections.abc import Sequence

import numpy as np
import torch

from ..config import Global
from ..parallel import nanaverage, nanmax, nanmean, nanmin
from .engine import OHLCV
from .factor import BaseFactor, CustomFactor, _same_scale


class Returns(CustomFactor):
    """Returns by tick (not by time)"""

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
        return (closes.last_nonnan(offset=1) / closes.first()).log()


class SimpleMovingAverage(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2
    _own_bar_scale = True

    def compute(self, data):
        return data.nanmean()


class WeightedAverageValue(CustomFactor):
    _min_win = 2
    _own_bar_scale = True

    def compute(self, base, weight):
        return base.agg(lambda _base, _weight: nanaverage(_base, _weight, dim=2), weight)


class LinearWeightedAverage(CustomFactor):
    _min_win = 2
    _own_bar_scale = True

    def __init__(self, win=None, inputs=None):
        super().__init__(win, inputs)
        self.weight = torch.arange(1, self.win + 1, dtype=Global.float_type)
        self.weight = self.weight / self.weight.sum()

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        self.weight = self.weight.to(device=engine.device, copy=False)

    def compute(self, base):
        self.weight = self.weight.to(device=base.device)
        return base.agg(lambda x: nanaverage(x, self.weight, dim=2))


class VWAP(WeightedAverageValue):
    inputs = (OHLCV.close, OHLCV.volume)


class ExponentialWeightedMovingAverage(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2
    _own_bar_scale = True

    def __init__(
        self,
        span: int | None = None,
        inputs: Sequence[BaseFactor] | None = None,
        adjust=False,
        half_life: float | None = None,
    ):
        if span is not None:
            self.alpha = 2.0 / (1.0 + span)
            # Length required to achieve 99.97% accuracy, np.log(1-99.97/100) / np.log(alpha)
            # simplification to 4 * (span+1). 3.45 achieve 99.90%, 2.26 99.00%
            self.win = int(4.5 * (span + 1))
        else:
            self.alpha = 1 - math.exp(math.log(0.5) / half_life)
            self.win = max(math.ceil(15 * half_life), self._min_win)

        super().__init__(None, inputs)
        self.adjust = adjust
        self.weight = np.full(self.win, 1 - self.alpha) ** np.arange(self.win - 1, -1, -1)
        if self.adjust:
            self.weight = self.weight / sum(self.weight)  # to sum one
        else:
            self.weight[1:] *= self.alpha  # the first bar of a window is the seed

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        if not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight, dtype=Global.float_type, device=engine.device)

    def _partial_windows(self, data):
        """
        `ewm(adjust=False)` of windows with missing bars. From the first valid bar on, every valid
        bar does `y = share * x + (1 - share) * y`, this turns that into the weight of each bar.
        """
        bars = torch.arange(1, self.win + 1, device=data.device, dtype=torch.int32)
        valid = ~torch.isnan(data)
        previous = (bars * valid).cummax(dim=-1).values.roll(1, dims=-1)
        previous[..., 0] = 0
        # older bars keep decaying over the missing bars, so `share` depends on the distance
        decay = (1 - self.alpha) ** torch.arange(self.win + 1, device=data.device, dtype=data.dtype)
        share = (self.alpha / (decay + self.alpha))[bars - previous]
        share = share.masked_fill(previous == 0, 1).masked_fill(~valid, 0)
        later = (1 - share).flip(-1).cumprod(dim=-1).flip(-1).roll(-1, dims=-1)
        later[..., -1] = 1
        return nanaverage(data, share * later, dim=-1)

    def _recursive_mean(self, data):
        ret = nanaverage(data, self.weight, dim=2)
        nans = torch.isnan(data)
        partial = nans.any(dim=2) & ~nans.all(dim=2)
        if partial.any():
            ret[partial] = self._partial_windows(data[partial])
        return ret

    def compute(self, data):
        self.weight = self.weight.to(device=data.device)
        # with alpha = 1 the weights of the older bars are 0, and a NaN last bar would leave nothing
        if not self.adjust or self.alpha == 1:
            return data.agg(self._recursive_mean)
        return data.agg(lambda x: nanaverage(x, self.weight, dim=2))


class AverageDollarVolume(CustomFactor):
    inputs = [OHLCV.close, OHLCV.volume]

    def compute(self, closes, volumes):
        if self.win == 1:
            return closes * volumes
        else:
            return closes.agg(lambda c, v: nanmean(c * v, dim=2), volumes)


class AnnualizedVolatility(CustomFactor):
    inputs = [Returns(win=2), 252]
    win = 20
    _min_win = 2

    def compute(self, returns, annualization_factor):
        return returns.nanstd() * (annualization_factor**0.5)


class ElementWiseMax(CustomFactor):
    _min_win = 1

    def __init__(self, win=None, inputs=None):
        super().__init__(win, inputs)
        assert self.win == 1

    def _find_scale(self) -> tuple | None:
        return _same_scale(*self.inputs)

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

    def _find_scale(self) -> tuple | None:
        return _same_scale(*self.inputs)

    def compute(self, a, b):
        ret = torch.min(*ElementWiseMax.binary_fill_na(a, b, np.inf))
        ret.masked_fill_(torch.isinf(ret), np.nan)
        return ret


def _first_position(data, value):
    """Position in the window of the first bar equal to `value`, from 1 / win to 1, or NaN."""
    ret = ((data == value.unsqueeze(-1)).char().argmax(dim=2) + 1.0) / data.shape[2]
    return ret.to(Global.float_type).masked_fill_(torch.isnan(value), np.nan)


class RollingArgMax(CustomFactor):
    _min_win = 2

    def compute(self, data):
        return data.agg(lambda x: _first_position(x, nanmax(x, dim=2)))


class RollingArgMin(CustomFactor):
    _min_win = 2

    def compute(self, data):
        return data.agg(lambda x: _first_position(x, nanmin(x, dim=2)))


class ConstantsFactor(CustomFactor):
    def __init__(self, value, like=OHLCV.open):
        self.value = value
        super().__init__(1, inputs=[like])

    def compute(self, x):
        return torch.full(x.shape, self.value, device=x.device, dtype=x.dtype)


MA = SimpleMovingAverage
SMA = SimpleMovingAverage
EMA = ExponentialWeightedMovingAverage
