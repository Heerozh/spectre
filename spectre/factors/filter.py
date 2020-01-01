"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from abc import ABC
from typing import Sequence
from .factor import CustomFactor
import torch


class FilterFactor(CustomFactor, ABC):
    def shift(self, periods=1):
        factor = FilterShiftFactor(inputs=(self,))
        factor.periods = periods
        return factor


class FilterShiftFactor(CustomFactor):
    """For "roll_cuda" not implemented for 'Bool' """
    periods = 1

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        shift = data.char().roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0:self.periods] = 0
        else:
            shift[:, self.periods:] = 0

        return shift.bool()


class StaticAssets(FilterFactor):
    """Useful for remove specific outliers or debug some assets"""
    def __init__(self, assets: Sequence[str]):
        from .engine import OHLCV
        super().__init__(win=1, inputs=[OHLCV.open])
        self.assets = assets

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        s = self._revert_to_series(data)
        ret = s.index.isin(self.assets, level=1)
        return self._regroup(ret)


class InvertFactor(FilterFactor):
    def compute(self, left) -> torch.Tensor:
        return ~left


class OrFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left | right


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
