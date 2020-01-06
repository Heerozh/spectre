"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from .factor import CustomFactor
from .engine import OHLCV
from ..parallel import linear_regression_1d
import torch


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

    def __init__(self, x, y, win):
        super().__init__(win=win, inputs=[x, y])

    def compute(self, x, y):
        def lin_reg(x_, y_):
            m, b = linear_regression_1d(x_, y_, dim=2)
            return torch.cat([m.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
        return x.agg(lin_reg, y)


STDDEV = StandardDeviation
MAX = RollingHigh
MIN = RollingLow
