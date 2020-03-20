"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from .factor import CustomFactor
from ..parallel import masked_first


class RollingFirst(CustomFactor):
    win = 2
    _min_win = 2

    def __init__(self, win, data, mask):
        super().__init__(win, inputs=(data, mask))

    def compute(self, data, mask):
        def _first_filter(_data, _mask):
            first_signal_price = masked_first(_data, _mask, dim=2)
            return first_signal_price

        return data.agg(_first_filter, mask)


class ForwardSignalData(RollingFirst):
    """Data in future window periods where signal = True. Lookahead biased."""
    def __init__(self, win, data, signal):
        super().__init__(win, data.shift(-win+1), signal.shift(-win+1))
