from .factor import CustomFactor
from .engine import OHLCV


class StandardDeviation(CustomFactor):
    inputs = [OHLCV.close]
    _min_win = 2

    def compute(self, data):
        return data.rolling(self.win).std(ddof=0)


class RollingHigh(CustomFactor):
    inputs = (OHLCV.close,)
    win = 5
    _min_win = 2

    def compute(self, data):
        return data.rolling(self.win).max()


class RollingLow(CustomFactor):
    inputs = (OHLCV.close,)
    win = 5
    _min_win = 2

    def compute(self, data):
        return data.rolling(self.win).min()


STDDEV = StandardDeviation
MAX = RollingHigh
MIN = RollingLow
