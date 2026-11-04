"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import itertools
import math

import numpy as np
import torch

from ..config import Global
from ..parallel import (
    DeviceConstant,
    Rolling,
    covariance,
    linear_regression_1d,
    nanmax,
    nanmean,
    nanstd,
    nanvar,
    pearsonr,
    quantile,
    unmasked_mean,
    unmasked_sum,
)
from .basic import MA
from .engine import OHLCV
from .factor import CrossSectionFactor, CustomFactor


class StandardDeviation(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2
    ddof = 0

    def compute(self, data):
        return data.nanstd(ddof=self.ddof)


class XSStandardDeviation(CrossSectionFactor):
    inputs = [OHLCV.close]
    ddof = 0

    def compute(self, data):
        return nanstd(data, ddof=self.ddof).unsqueeze(-1).expand(data.shape[0], data.shape[1])


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
            return y.agg(lin_reg, x)

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
            x = torch.stack([ones, x, x**2]).T
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


class _OwnBarScale(CustomFactor):
    """
    A price in the scale of its own bar, made from the window [t-1, t] of `data`, which Rolling
    has adjusted to the scale of bar t. With the multipliers of `data`, a Rolling over this factor
    adjusts its windows in the same way.
    """

    win = 2

    @property
    def adjustments(self):
        return self.inputs[0].adjustments


class _LagSpread(_OwnBarScale):
    """`data` of the previous bar minus `mean`, which has to be in the scale of its own bar."""

    def compute(self, data, mean):
        return data.first() - (mean.last() if isinstance(mean, Rolling) else mean)


class _BarDiff(_OwnBarScale):
    def compute(self, data):
        return data.last() - data.first()


class HalfLifeMeanReversion(CustomFactor):
    _min_win = 2

    def __init__(self, win, data, mean, mask=None):
        lag = _LagSpread(inputs=[data, mean])
        diff = _BarDiff(inputs=[data])
        super().__init__(win=win, inputs=[lag, diff, math.log(2)])
        # not on `lag` and `diff`, that would also hide the previous bar of their windows
        self.set_mask(mask)

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
        dates, assets = xs[0].shape
        # the diagonal stays NaN, a factor is not one of the others
        corr = xs[0].new_full((dates, len(xs), len(xs)), np.nan)
        for i, j in itertools.combinations(range(len(xs)), 2):
            mask = torch.isnan(xs[i] * xs[j])
            # pearsonr rounds the mean, which leaves a constant with a tiny variance, and the noise
            # from that would pass as a correlation. From its lowest value a constant is exactly 0.
            x = xs[i] - xs[i].masked_fill(mask, np.inf).amin(dim=1, keepdim=True)
            y = xs[j] - xs[j].masked_fill(mask, np.inf).amin(dim=1, keepdim=True)
            corr[:, i, j] = corr[:, j, i] = pearsonr(x, y, dim=1, ddof=1)
        max_corr = nanmax(corr, dim=2).unsqueeze(-2)
        return max_corr.expand(dates, assets, len(xs))


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
            mask = torch.isnan(xy) | torch.isnan(weight)
            w = weight / unmasked_sum(weight, mask=mask, dim=1).unsqueeze(-1)
            x_bar = unmasked_sum(w * x, mask=mask, dim=1)
            y_bar = unmasked_sum(w * y, mask=mask, dim=1)
            cov_xy = unmasked_sum(w * xy, mask=mask, dim=1) - x_bar * y_bar
            var_x = unmasked_sum(w * x**2, mask=mask, dim=1) - x_bar**2
            var_y = unmasked_sum(w * y**2, mask=mask, dim=1) - y_bar**2
            ic = cov_xy / (var_x * var_y) ** 0.5
        return ic.unsqueeze(-1).expand(ic.shape[0], y.shape[1])

    def to_ir(self, win, ddof=1):
        # Use CrossSectionFactor and unfold by self, because if use CustomFactor, the ir value
        # will inconsistent when some assets have no data (like newly listed), the ir value should
        # not be related to assets.
        class RollingIC2IR(CrossSectionFactor):
            def __init__(self, win_, inputs):
                super().__init__(1, inputs)
                self.rolling_win = win_

            def get_total_backwards_(self) -> int:
                return super().get_total_backwards_() + self.rolling_win - 1

            def compute(self, ic):
                x = ic[:, 0]
                nan_stack = x.new_full((self.rolling_win - 1,), np.nan)
                new_x = torch.cat((nan_stack, x), dim=0)
                rolling_ic = new_x.unfold(0, self.rolling_win, 1)

                # Fundamental Law of Active Management: ir = ic * sqrt(b), 1/sqrt(b) = std(ic)
                ir = nanmean(rolling_ic, dim=1) / nanstd(rolling_ic, dim=1, ddof=ddof)
                return ir.unsqueeze(-1).expand(ic.shape)

        return RollingIC2IR(win_=win, inputs=[self])


class RollingInformationCoefficient(RollingCorrelation):
    """
    Rolling IC, Calculate IC between 2 historical data for each asset.
    """

    def to_ir(self, win):
        std = StandardDeviation(win=win, inputs=(self,))
        std.ddof = 1
        mean = MA(win=win, inputs=(self,))

        return mean / std


class RankWeightedInformationCoefficient(InformationCoefficient):
    def __init__(self, x, y, half_life, mask=None):
        alpha = np.exp(np.log(0.5) / half_life)
        y_rank = y.rank(ascending=False, mask=mask) - 1
        weight = alpha**y_rank
        super().__init__(x, y, mask=mask, weight=weight)


class TTest1Samp(CustomFactor):
    _min_win = 2

    def compute(self, a, pop_mean):
        def _ttest(_x):
            d = nanmean(_x, dim=2) - pop_mean
            v = nanvar(_x, dim=2, ddof=1)
            n = (~torch.isnan(_x)).sum(dim=2)
            denom = torch.sqrt(v / n)
            t = d / denom
            return t

        return a.agg(_ttest)


def _beta_fraction(a, b, x, tol):
    """
    Continued fraction of the regularized incomplete beta function, by the modified Lentz method.
    It needs `x` below (a + 1) / (a + b + 2). The denominators stay positive for the `a` and `b`
    of a t distribution, so the guards against 0 of that method are left out.
    """
    c = torch.ones_like(x)
    d = 1 / (1 - (a + b) * x / (a + 1))
    fraction = d
    for m in range(1, 10000):
        even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        for term in (even, odd):
            d = 1 / (1 + term * d)
            c = 1 + term / c
            step = c * d
            fraction = fraction * step
        if not ((step - 1).abs() > tol).any():
            break
    return fraction


class StudentCDF(CrossSectionFactor):
    """
    Note!! For performance, This factor uses Cross-section mean of t-value.

    `precision` works like the step of an integration grid, the error falls with its square:
    0.001 gives a relative error of about 1e-6, and 1e-7 reaches the limit of float64.
    """

    DefaultPrecision = 0.001

    def compute(self, t, dof, precision):
        reduced_t = nanmean(t, dim=1).to(torch.float64)
        dof = torch.tensor(dof, dtype=torch.float64, device=t.device)
        half = dof.new_tensor(0.5)
        # The lower tail is betainc(dof / 2, 1 / 2, x) / 2 with x = dof / (dof + t ** 2). Near t=0
        # x is too high for the fraction, there use betainc(a, b, x) = 1 - betainc(b, a, 1 - x).
        ratio = reduced_t**2 / dof
        flip = ratio * (dof + 2) < 3
        a = torch.where(flip, half, dof / 2)
        b = torch.where(flip, dof / 2, half)
        odds = torch.where(flip, ratio, 1 / ratio)
        log_front = -a * torch.log1p(1 / odds) - b * torch.log1p(odds)
        log_front = log_front + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)
        # rounding keeps the steps of the fraction from getting closer to 1 than this
        tol = max(precision**2, 1e-14)
        beta = torch.exp(log_front) / a * _beta_fraction(a, b, odds / (1 + odds), tol)
        tail = torch.where(flip, 1 - beta, beta) / 2
        p = torch.where(reduced_t < 0, tail, 1 - tail)
        return p.to(t.dtype).unsqueeze(-1).expand(t.shape)


class CrossSectionR2(CrossSectionFactor):
    def __init__(self, y, y_pred, mask, total_r2=False):
        super().__init__(win=1, inputs=[y, y_pred], mask=mask)
        self.total_r2 = total_r2

    def compute(self, y, y_pred):
        mask = torch.isnan(y_pred) | torch.isnan(y)
        ss_err = unmasked_sum((y - y_pred) ** 2, mask, dim=1)
        if self.total_r2:
            # 按市场总回报来算r2的话，用这个。不然就是相对回报。
            ss_tot = unmasked_sum(y**2, mask, dim=1)
        else:
            y_bar = unmasked_mean(y, mask, dim=1).unsqueeze(-1)
            ss_tot = unmasked_sum((y - y_bar) ** 2, mask, dim=1)
        r2 = -ss_err / ss_tot + 1
        r2[(~mask).to(Global.float_type).sum(dim=1) < 2] = np.nan
        return r2.unsqueeze(-1).expand(r2.shape[0], y.shape[1])


class FactorWiseKthValue(CrossSectionFactor):
    """The kth value of all factors sorted in ascending order, grouped by each datetime"""

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
