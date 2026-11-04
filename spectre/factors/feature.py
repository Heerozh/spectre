"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import warnings

import pandas as pd
import torch

from ..config import Global
from ..parallel import nanmean, nanstd, nansum
from .basic import Returns
from .datafactor import DatetimeDataFactor
from .factor import CrossSectionFactor, CustomFactor
from .filter import FilterFactor

# ----------- Common Market Features -----------


class MarketDispersion(CrossSectionFactor):
    """Cross-section standard deviation of universe stocks returns."""

    inputs = (Returns(),)
    win = 1

    def compute(self, returns):
        ret = nanstd(returns, dim=1).unsqueeze(-1)
        return ret.expand(ret.shape[0], returns.shape[1])


class MarketReturn(CrossSectionFactor):
    """Cross-section mean returns of universe stocks."""

    inputs = (Returns(),)
    win = 1

    def compute(self, returns):
        ret = nanmean(returns, dim=1).unsqueeze(-1)
        return ret.expand(ret.shape[0], returns.shape[1])


class MarketVolatility(CustomFactor):
    """MarketReturn Rolling standard deviation."""

    inputs = (MarketReturn(), 252)
    win = 252
    _min_win = 2

    def compute(self, returns, annualization_factor):
        return (returns.nanvar() * annualization_factor) ** 0.5


class AdvanceDeclineRatio(CrossSectionFactor):
    """Need to work with MA, and could be applied to volume too"""

    inputs = (Returns(),)
    win = 1

    def compute(self, returns):
        advancing = nansum(returns > 0, dim=1).to(Global.float_type)
        declining = nansum(returns < 0, dim=1).to(Global.float_type)
        ratio = (advancing / declining).unsqueeze(-1)
        return ratio.expand(ratio.shape[0], returns.shape[1])


# ----------- Asset-specific data -----------


class AssetData(CustomFactor):
    def __init__(self, asset, factor):
        self.asset = asset
        self.asset_ind = None
        super().__init__(win=1, inputs=[factor])

    def pre_compute_(self, engine, start, end):
        super().pre_compute_(engine, start, end)
        if not engine.align_by_time:
            warnings.warn(
                "Make sure your data is aligned by time, otherwise will cause data "
                "disorder. Or set engine.align_by_time = True.",
                RuntimeWarning,
            )
        # data has a row for each asset with rows, the categories also keep the ones without
        assets = engine.dataframe_index[1].remove_unused_categories().categories
        if self.asset not in assets:
            raise KeyError(
                f"{self.asset} has no rows from {start} to {end}, so AssetData can't read it. "
                "It may not exist in that range, or StaticAssets screened it out."
            )
        self.asset_ind = assets.get_loc(self.asset)

    def compute(self, data):
        ret = data[self.asset_ind]
        return ret.expand(data.shape[0], ret.shape[0])


# ----------- Common Calendar Features -----------


class PeriodEdge(FilterFactor):
    """
    Is this date the first or last trading date of its period? The same for all assets and all
    bars of the date, so the last bar of an asset says nothing. The calendar is the dates of the
    loaded data, in `engine.timezone`. A date at the end of it has no neighbour, so it is decided
    by business days, a calendar is known in advance, but a holiday there is not seen.
    """

    def __init__(self, freq: str, last: bool):
        # only the first date of a period needs the date before `start`
        super().__init__(win=1 if last else 2)
        self.freq = freq
        self.last = last

    def compute(self) -> torch.Tensor:
        # naive, because some timezones have no midnight on the date daylight saving starts
        dates = self._engine.dataframe_index[0].tz_localize(None).normalize()
        rows, calendar = pd.factorize(dates, sort=True)
        if self.last:
            neighbours = calendar[1:].append(calendar[-1:] + pd.offsets.BDay())
        else:
            neighbours = (calendar[:1] - pd.offsets.BDay()).append(calendar[:-1])
        edges = neighbours.to_period(self.freq) != calendar.to_period(self.freq)
        return self._regroup(edges[rows])


MONTH = DatetimeDataFactor("month")
WEEKDAY = DatetimeDataFactor("weekday")
QUARTER = DatetimeDataFactor("quarter")
TIME = DatetimeDataFactor("hour") + DatetimeDataFactor("minute") / 60.0

IS_JANUARY = MONTH == 1
IS_DECEMBER = MONTH == 12
IS_MONTH_END = PeriodEdge("M", last=True)
IS_MONTH_START = PeriodEdge("M", last=False)
IS_QUARTER_END = PeriodEdge("Q", last=True)
IS_QUARTER_START = PeriodEdge("Q", last=False)
