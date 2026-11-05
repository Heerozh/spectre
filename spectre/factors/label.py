"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

from ..parallel import Rolling, masked_first, pad_2d
from .factor import CustomFactor


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
    """
    Data in future window periods where signal = True, in the scale of the current bar, so it
    can be compared with the data of that bar. Lookahead biased.
    """

    def __init__(self, win, data, signal):
        super().__init__(win, data.shift(-win + 1), signal.shift(-win + 1))
        self.data = data

    def compute(self, data, mask):
        multi = self.data.adjustments
        if multi is None:
            return super().compute(data, mask)

        # The rows added by engine.align_by_time have no multiplier. Give them what the loader
        # would have: the one of the bar before them, or of the first bar of the asset.
        multi = pad_2d(pad_2d(multi).flip(1)).flip(1)
        multi = self.inputs[0].compute(multi).to(data.device)

        def _first_filter(_data, _mask, _multi):
            return masked_first(_data * _multi, _mask, dim=2) / _multi[:, :, 0]

        return data.agg(_first_filter, mask, Rolling(multi, self.win))
