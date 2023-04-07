"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import torch
import math
import numpy as np
from .factor import CustomFactor, CrossSectionFactor
from .engine import OHLCV
from ..parallel import (linear_regression_1d, quantile, pearsonr, unmasked_mean, unmasked_sum,
                        nanmean, nanstd, covariance, nanvar)
from ..parallel import DeviceConstant
from ..config import Global


class StandardDeviation(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2
    ddof = 0

    def compute(self, data):
        return data.nanstd(ddof=self.ddof)


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
                _x = DeviceConstant.get(_y.device).arange(_y.shape[2], dtype=_y.dtype)
                _x = _x.expand(_y.shape[0], _y.shape[1], _x.shape[0])
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
            x = DeviceConstant.get(_y.device).arange(_y.shape[2], dtype=_y.dtype)
            ones = torch.ones(x.shape[0], device=_y.device, dtype=_y.dtype)
            x = torch.stack([ones, x, x ** 2]).T
            x = x.expand(_y.shape[0], _y.shape[1], x.shape[0], x.shape[1])

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


class RollingCovariance(CustomFactor):
    _min_win = 2

    def compute(self, x, y):
        def _cov(_x, _y):
            return covariance(_x, _y, dim=2, ddof=1)
        return x.agg(_cov, y)


class XSMaxCorrCoef(CrossSectionFactor):
    """
    Returns the maximum correlation coefficient for each x to others
    """

    def compute(self, *xs):
        x = torch.stack(xs, dim=1)
        x_bar = nanmean(x, dim=2).unsqueeze(-1)
        demean = x - x_bar
        demean.masked_fill_(torch.isnan(demean), 0)
        cov = demean @ demean.transpose(-2, -1)
        cov = cov / (x.shape[-1] - 1)
        diag = cov[:, range(len(xs)), range(len(xs)), None]
        std = diag ** 0.5
        corr = cov / std / std.transpose(-2, -1)
        # set auto corr to zero
        corr[:, range(len(xs)), range(len(xs))] = 0
        max_corr = corr.max(dim=2).values.unsqueeze(-2)
        return max_corr.expand(x.shape[0], x.shape[2], x.shape[1])


class InformationCoefficient(CrossSectionFactor):
    """
    Cross-Section IC, the ic value of all assets is the same.
    """
    def __init__(self, x, y, mask=None, weight=None):
        super().__init__(win=1, inputs=[x, y, weight], mask=mask)

    def compute(self, x, y, weight):
        if weight is None:
            ic = pearsonr(x, y, dim=1, ddof=1)
        else:
            xy = x * y
            mask = torch.isnan(x * y)
            w = weight / unmasked_sum(weight, mask=mask, dim=1).unsqueeze(-1)
            x_bar = unmasked_sum(w * x, mask=mask, dim=1)
            y_bar = unmasked_sum(w * y, mask=mask, dim=1)
            cov_xy = unmasked_sum(w * xy, mask=mask, dim=1) - x_bar * y_bar
            var_x = unmasked_sum(w * x ** 2, mask=mask, dim=1) - x_bar ** 2
            var_y = unmasked_sum(w * y ** 2, mask=mask, dim=1) - y_bar ** 2
            ic = cov_xy / (var_x * var_y) ** 0.5
        return ic.unsqueeze(-1).expand(ic.shape[0], y.shape[1])

    def to_ir(self, win):
        # Use CrossSectionFactor and unfold by self, because if use CustomFactor, the ir value
        # will inconsistent when some assets have no data (like newly listed), the ir value should
        # not be related to assets.
        class RollingIC2IR(CrossSectionFactor):
            def __init__(self, win_, inputs):
                super().__init__(1, inputs)
                self.rolling_win = win_

            def compute(self, ic):
                x = ic[:, 0]
                nan_stack = x.new_full((self.rolling_win - 1,), np.nan)
                new_x = torch.cat((nan_stack, x), dim=0)
                rolling_ic = new_x.unfold(0, self.rolling_win, 1)

                # Fundamental Law of Active Management: ir = ic * sqrt(b), 1/sqrt(b) = std(ic)
                ir = nanmean(rolling_ic, dim=1) / nanstd(rolling_ic, dim=1, ddof=1)
                return ir.unsqueeze(-1).expand(ic.shape)
        return RollingIC2IR(win_=win, inputs=[self])


class RollingInformationCoefficient(RollingCorrelation):
    """
    Rolling IC, Calculate IC between 2 historical data for each asset.
    """
    def to_ir(self, win):
        std = StandardDeviation(win=win, inputs=(self,))
        std.ddof = 1
        mean = self.sum(win) / win

        return mean / std


class RankWeightedInformationCoefficient(InformationCoefficient):
    def __init__(self, x, y, half_life, mask=None):
        alpha = np.exp((np.log(0.5) / half_life))
        y_rank = y.rank(ascending=False, mask=mask) - 1
        weight = alpha ** y_rank
        super().__init__(x, y, mask=mask, weight=weight)


class TTest1Samp(CustomFactor):
    _min_win = 2

    def compute(self, a, pop_mean):
        def _ttest(_x):
            d = nanmean(_x, dim=2) - pop_mean
            v = nanvar(_x, dim=2, ddof=1)
            denom = torch.sqrt(v / self._min_win)
            t = d / denom
            return t
        return a.agg(_ttest)


class StudentCDF(CrossSectionFactor):
    """
    Note!! For performance, This factor assumes that all assets have the
           same t-value in same time!!
    """
    DefaultPrecision = 0.001

    def compute(self, t, dof, precision):
        reduced_t = nanmean(t, dim=1)
        p = torch.zeros_like(reduced_t)
        dof = torch.tensor(dof, dtype=torch.float64, device=t.device)
        for i, v in enumerate(reduced_t.cpu()):
            if np.isnan(v):
                p[i] = torch.nan
            elif np.isinf(v):
                p[i] = 1
            elif v < -9:
                p[i] = 0
            else:
                x = torch.arange(-9, v, precision, device=t.device)
                p[i] = torch.e ** torch.lgamma((dof + 1) / 2) / (
                            torch.sqrt(dof * torch.pi) * torch.e ** torch.lgamma(dof / 2)) * (
                        torch.trapezoid((1 + x ** 2 / dof) ** (-dof / 2 - 1 / 2), x)
                       )
        return p.unsqueeze(-1).expand(t.shape)


class CrossSectionR2(CrossSectionFactor):
    def __init__(self, y, y_pred, mask, total_r2=False):
        super().__init__(win=1, inputs=[y, y_pred], mask=mask)
        self.total_r2 = total_r2

    def compute(self, y, y_pred):
        mask = torch.isnan(y_pred) | torch.isnan(y)
        ss_err = unmasked_sum((y - y_pred) ** 2, mask, dim=1)
        if self.total_r2:
            # 按市场总回报来算r2的话，用这个。不然就是相对回报。
            ss_tot = unmasked_sum(y ** 2, mask, dim=1)
        else:
            y_bar = unmasked_mean(y, mask, dim=1).unsqueeze(-1)
            ss_tot = unmasked_sum((y - y_bar) ** 2, mask, dim=1)
        r2 = -ss_err / ss_tot + 1
        r2[(~mask).to(Global.float_type).sum(dim=1) < 2] = np.nan
        return r2.unsqueeze(-1).expand(r2.shape[0], y.shape[1])


class FactorWiseKthValue(CrossSectionFactor):
    """ The kth value of all factors sorted in ascending order, grouped by each datetime """
    def __init__(self, kth, inputs=None):
        super().__init__(1, inputs)
        self.kth = kth

    def compute(self, *data):
        mx = torch.stack([nanmean(x, dim=1) for x in data], dim=-1)
        nans = torch.isnan(mx)
        mx.masked_fill_(nans, -np.inf)
        ret = torch.kthvalue(mx, self.kth, dim=1, keepdim=True).values
        return ret.expand(ret.shape[0], data[0].shape[1])


class FactorWiseZScore(CrossSectionFactor):
    def compute(self, *data):
        mx = torch.stack([nanmean(x, dim=1) for x in data], dim=-1)
        ret = (mx - nanmean(mx, dim=1).unsqueeze(-1)) / nanstd(mx, dim=1).unsqueeze(-1)
        return ret.unsqueeze(-2).repeat(1, data[0].shape[1], 1)


STDDEV = StandardDeviation
MAX = RollingHigh
MIN = RollingLow
