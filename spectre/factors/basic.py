
from .factor import BaseFactor, CustomFactor
from .engine import OHLCV


class Returns(CustomFactor):
    inputs = [OHLCV.close]
    win = 2
    _min_win = 2

    def compute(self, close):
        return (close.groupby(level=1).shift(1) - close) / close


class SMA(CustomFactor):
    inputs = [OHLCV.close]

    def compute(self, data):
        return data.rolling(self.win).mean()


class WeightedAverageValue(CustomFactor):
    def compute(self, base, weight):
        return (base * weight).rolling(self.win).nanmean() / weight.rolling(self.win).nansum()


class VWAP(WeightedAverageValue):
    inputs = (OHLCV.close, OHLCV.volume)