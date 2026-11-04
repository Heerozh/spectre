"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import warnings
from abc import ABC

import torch

from ..parallel import Rolling
from .factor import BaseFactor, CustomFactor, MultiRetSelector


class FilterFactor(CustomFactor, ABC):
    def __getitem__(self, key):
        return FilterMultiRetSelector(inputs=(self, key))

    def _numeric(self):
        # a regroup pads a filter with False, as a float the missing rows stay NaN
        return self.float()

    def shift(self, periods=1):
        if periods == 0:
            return self
        factor = FilterRawShiftFactor(inputs=(self,))
        factor.periods = periods
        return factor

    def sum(self, win):
        raise ValueError(
            "FilterFactor does not support `.sum()` method, "
            "please convert to float by using `filter_factor.float()`"
        )

    def filter(self, mask):
        raise ValueError(
            "FilterFactor does not support local filtering `.filter()` method, "
            "please convert to float by using `filter_factor.float()`"
        )

    def ts_any(self, win):
        """Return True if Rolling window contains any True"""
        return AnyFilter(win, inputs=(self,))

    def ts_all(self, win):
        """Return True if Rolling window all are True"""
        return AllFilter(win, inputs=(self,))


class FilterMultiRetSelector(MultiRetSelector, FilterFactor):
    """MultiRetSelector returns CustomFactor, we're override here as FilterFactor"""


class PlaceHolderFilter(FilterFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data


class FilterRawShiftFactor(FilterFactor):
    """For "roll_cuda" not implemented for 'Bool'"""

    periods = 1

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        shift = data.char().roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0 : self.periods] = 0
        else:
            shift[:, self.periods :] = 0

        return shift.bool()


class StaticAssets(FilterFactor):
    """Useful for remove specific outliers or debug some assets"""

    def __init__(self, assets: set[str]):
        from .engine import OHLCV

        super().__init__(win=1, inputs=[OHLCV.open])
        self.assets = assets

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        s = self._revert_to_series(data)
        ret = s.index.isin(self.assets, level=1)
        return self._regroup(ret)


class OneHotEncoder(FilterFactor):
    def __init__(self, input_: BaseFactor):
        super().__init__(1, [input_])

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        # sorted, so the order of the outputs is deterministic
        classes = torch.unique(data, sorted=True)
        classes = classes[~torch.isnan(classes)]
        one_hot = []
        if classes.shape[0] > 1000:
            warnings.warn(
                f"One hot encoding with too many features: ({classes.shape[0]}). ", RuntimeWarning
            )
        for i in range(classes.shape[0]):
            one_hot.append((data == classes[i]).unsqueeze(-1))
        return torch.cat(one_hot, dim=-1)


class AnyFilter(FilterFactor):
    _min_win = 2
    # no data before the first bar, it's not True
    _rolling_fill = False

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.values.any(dim=2)


class AllFilter(FilterFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.values.all(dim=2)


class AnyNonNaNFactor(FilterFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return (~torch.isnan(data.values)).any(dim=2)


class AllNonNaNFactor(FilterFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return (~torch.isnan(data.values)).all(dim=2)


class InvertFactor(FilterFactor):
    def compute(self, left) -> torch.Tensor:
        return ~left


class OrFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left | right


class XorFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left ^ right


class AndFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left & right


class LtFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return torch.lt(left, right)


class LeFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return torch.le(left, right)


class GtFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return torch.gt(left, right)


class GeFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return torch.ge(left, right)


class EqFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return torch.eq(left, right)


class NeFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return torch.ne(left, right)
