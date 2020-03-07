"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import warnings
from .datafactor import DatetimeDataFactor
from .factor import TimeGroupFactor, CustomFactor
from .basic import Returns
from ..parallel import nanstd, nanmean


# ----------- Common Market Features -----------


class MarketDispersion(TimeGroupFactor):
    """Cross-section standard deviation of universe stocks returns."""
    inputs = (Returns(), )
    win = 1

    def compute(self, returns):
        ret = nanstd(returns, dim=1).unsqueeze(-1)
        return ret.repeat(1, returns.shape[1])


class MarketReturn(TimeGroupFactor):
    """Cross-section mean returns of universe stocks."""
    inputs = (Returns(), )
    win = 1

    def compute(self, returns):
        ret = nanmean(returns, dim=1).unsqueeze(-1)
        return ret.repeat(1, returns.shape[1])


class MarketVolatility(CustomFactor):
    """MarketReturn Rolling standard deviation."""
    inputs = (MarketReturn(), 252)
    win = 252
    _min_win = 2

    def compute(self, returns, annualization_factor):
        return (returns.nanvar() * annualization_factor) ** 0.5


# ----------- Asset-specific data -----------


class AssetData(CustomFactor):
    def __init__(self, asset, factor):
        self.asset = asset
        self.asset_ind = None
        super().__init__(win=1, inputs=[factor])

    def pre_compute_(self, engine, start, end):
        super().pre_compute_(engine, start, end)
        if not engine.align_by_time:
            warnings.warn("Make sure your data is aligned by time, otherwise will cause data "
                          "disorder. Or set engine.align_by_time = True.",
                          RuntimeWarning)
        self.asset_ind = engine.dataframe_index[1].unique().categories.get_loc(self.asset)

    def compute(self, data):
        ret = data[self.asset_ind]
        return ret.repeat(data.shape[0], 1)


# ----------- Common Calendar Features -----------


MONTH = DatetimeDataFactor('month')
WEEKDAY = DatetimeDataFactor('weekday')
QUARTER = DatetimeDataFactor('quarter')

IS_JANUARY = MONTH == 1
IS_DECEMBER = MONTH == 12
# Because the future data is used in IS_MONTH_END and IS_QUARTER_END factors, it will fail the
# test_lookahead_bias, but because it's != operation, so only a very low probability will fail the
# test. And this method is the fastest, so be it.
IS_MONTH_END = MONTH.shift(-1) != MONTH
IS_MONTH_START = MONTH.shift(1) != MONTH
IS_QUARTER_END = QUARTER.shift(-1) != QUARTER
IS_QUARTER_START = QUARTER.shift(1) != QUARTER

