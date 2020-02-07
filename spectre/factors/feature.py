"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from .factor import TimeGroupFactor, CustomFactor
from .basic import Returns
from ..parallel import nanstd, nanmean


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
