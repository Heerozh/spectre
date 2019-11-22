"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from .factor import CustomFactor
from .engine import OHLCV


class StandardDeviation(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2

    def compute(self, data):
        return data.std()


class RollingHigh(CustomFactor):
    inputs = (OHLCV.close,)
    win = 5
    _min_win = 2

    def compute(self, data):
        return data.max()


class RollingLow(CustomFactor):
    inputs = (OHLCV.close,)
    win = 5
    _min_win = 2

    def compute(self, data):
        return data.min()


STDDEV = StandardDeviation
MAX = RollingHigh
MIN = RollingLow
