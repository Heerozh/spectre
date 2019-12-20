"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Sequence
from .factor import FilterFactor
from .engine import OHLCV
import torch


class StaticAssets(FilterFactor):
    win = 1
    inputs = [OHLCV.open]

    def __init__(self, assets: Sequence[str]):
        super().__init__()
        self.assets = assets

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        s = self._revert_to_series(data)
        ret = s.index.isin(self.assets, level=1)
        return self._regroup(ret)
