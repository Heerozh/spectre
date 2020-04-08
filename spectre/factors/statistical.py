"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import torch
import math
from .factor import CustomFactor, CrossSectionFactor
from .engine import OHLCV
from ..parallel import (linear_regression_1d, quantile, pearsonr, masked_mean, masked_sum,
                        nanmean, nanstd)


class StandardDeviation(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2

    def compute(self, data):
        return data.nanstd()


class RollingHigh(CustomFactor):
    inputs = (OHLCV.close,)
    win = 5
    _min_win = 2

    def compute(self, data):
        return data.nanmax()


class RollingLow(CustomFactor):
    inputs = (OHLCV.close,)
    win = 5
    _min_win = 2

    def compute(self, data):
        return data.nanmin()


class RollingLinearRegression(CustomFactor):
    _min_win = 2

    def __init__(self, win, x, y):
        super().__init__(win=win, inputs=[x, y])

    def compute(self, x, y):
        def lin_reg(_y, _x=None):
            if _x is None:
                _x = torch.arange(_y.shape[2], device=_y.device, dtype=_y.dtype)
                _x = _x.repeat(_y.shape[0], _y.shape[1], 1)
            m, b = linear_regression_1d(_x, _y, dim=2)
            return torch.cat([m.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
        if x is None:
            return y.agg(lin_reg)
        else:
            return y.agg(lin_reg, y)

    @property
    def coef(self):
        return self[0]

    @property
    def intercept(self):
        return self[1]


class RollingMomentum(CustomFactor):
    inputs = (OHLCV.close,)
    win = 20
    _min_win = 2

    def compute(self, prices):
        def polynomial_reg(_y):
            x = torch.arange(_y.shape[2], device=_y.device, dtype=_y.dtype)
            ones = torch.ones(x.shape[0], device=_y.device, dtype=_y.dtype)
            x = torch.stack([ones, x, x ** 2]).T.repeat(_y.shape[0], _y.shape[1], 1, 1)

            xt = x.transpose(-2, -1)
            ret = (xt @ x).inverse() @ xt @ _y.unsqueeze(-1)
            return ret.squeeze(-1)

        return prices.agg(polynomial_reg)

    @property
    def gain(self):
        """gain>0 means stock gaining, otherwise is losing."""
        return self[1]

    @property
    def accelerate(self):
        """accelerate>0 means stock accelerating, otherwise is decelerating."""
        return self[2]

    @property
    def intercept(self):
        return self[0]


class RollingQuantile(CustomFactor):
    inputs = (OHLCV.close, 5)
    _min_win = 2

    def compute(self, data, bins):
        def _quantile(_data):
            return quantile(_data, bins, dim=2)[:, :, -1]
        return data.agg(_quantile)


class HalfLifeMeanReversion(CustomFactor):
    _min_win = 2

    def __init__(self, win, data, mean, mask=None):
        lag = data.shift(1) - mean
        diff = data - data.shift(1)
        lag.set_mask(mask)
        diff.set_mask(mask)
        super().__init__(win=win, inputs=[lag, diff, math.log(2)])

    def compute(self, lag, diff, ln2):
        def calc_h(_x, _y):
            _lambda, _ = linear_regression_1d(_x, _y, dim=2)
            return -ln2 / _lambda
        return lag.agg(calc_h, diff)


class RollingCorrelation(CustomFactor):
    _min_win = 2

    def compute(self, x, y):
        def _corr(_x, _y):
            return pearsonr(_x, _y, dim=2, ddof=1)
        return x.agg(_corr, y)


class InformationCoefficient(CrossSectionFactor):
    def __init__(self, x, y, mask=None):
        super().__init__(win=1, inputs=[x, y], mask=mask)

    def compute(self, x, y):
        ic = pearsonr(x, y, dim=1, ddof=1)
        return ic.unsqueeze(-1).repeat(1, y.shape[1])

    def to_ir(self, win):
        class RollingIC2IR(CustomFactor):
            def compute(self, ic):
                def _to_ir(_ic):
                    # Fundamental Law of Active Management: ir = ic * sqrt(b), 1/sqrt(b) = std(ic)
                    return nanmean(_ic, dim=2) / nanstd(_ic, dim=2, ddof=1)
                return ic.agg(_to_ir)
        return RollingIC2IR(win=win, inputs=[self])


class CrossSectionR2(CrossSectionFactor):
    def __init__(self, y_hat, y, mask=None):
        super().__init__(win=1, inputs=[y_hat, y], mask=mask)

    def compute(self, y_hat, y):
        mask = torch.isnan(y_hat) | torch.isnan(y)
        y_bar = masked_mean(y, mask, dim=1).unsqueeze(-1)
        ss_tot = masked_sum((y - y_bar) ** 2, mask, dim=1)
        ss_reg = masked_sum((y_hat - y_bar) ** 2, mask, dim=1)
        return (ss_reg / ss_tot).unsqueeze(-1).repeat(1, y.shape[1])


STDDEV = StandardDeviation
MAX = RollingHigh
MIN = RollingLow
