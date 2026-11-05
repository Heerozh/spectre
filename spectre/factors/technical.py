"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

from collections.abc import Sequence

import numpy as np
import torch

from ..parallel import nanmean
from .basic import EMA, MA
from .engine import OHLCV
from .factor import BaseFactor, CustomFactor, RollingZScoreFactor
from .statistical import STDDEV


class BollingerBands(CustomFactor):
    """usage: BBANDS(win, inputs=[OHLCV.close, k]), k is constant normally 2"""

    inputs = (OHLCV.close, 2)
    win = 20
    _min_win = 2
    _own_bar_scale = True

    def __init__(self, win: int | None = None, inputs: Sequence[BaseFactor] | None = None):
        super().__init__(win, inputs)
        if len(self.inputs) < 2:
            raise ValueError(
                "BollingerBands's inputs needs 2 inputs, "
                "inputs=[OHLCV.close, k]), k is constant normally 2."
            )
        comm_inputs = (self.inputs[0],)
        k = self.inputs[1]
        self.inputs = (
            self.inputs[0],
            MA(win=self.win, inputs=comm_inputs),
            STDDEV(win=self.win, inputs=comm_inputs),
            k,
        )
        self.win = 1

    def compute(self, closes, ma, std, k):
        d = k * std
        up = ma + d
        down = ma - d
        return torch.cat([up.unsqueeze(-1), ma.unsqueeze(-1), down.unsqueeze(-1)], dim=-1)

    def normalized(self):
        closes, ma, _, k = self.inputs
        return RollingZScoreFactor(ma.win, inputs=[closes]) / k


class _MACDLine(CustomFactor):
    """EMA(fast) - EMA(slow). It is a price, so a rolling window over it needs adjusting."""

    _own_bar_scale = True

    def __init__(self, fast, slow, inputs, adjust):
        super().__init__(inputs=[EMA(span, inputs, adjust) for span in (fast, slow)])

    def compute(self, fast, slow):
        return fast - slow


class MovingAverageConvergenceDivergenceSignal(EMA):
    """
    engine.add( MACD(fast, slow, sign, inputs=[OHLCV.close]) )
    or
    engine.add( MACD().normalized() )
    """

    inputs = (OHLCV.close,)
    win = 9
    _min_win = 2

    def __init__(
        self, fast=12, slow=26, sign=9, inputs: Sequence[BaseFactor] | None = None, adjust=False
    ):
        super().__init__(sign, inputs, adjust)
        self.inputs = (_MACDLine(fast, slow, self.inputs, adjust),)

    def normalized(self):
        # In order not to double the calculation, reuse `inputs` factor here
        macd = self.inputs[0]
        sign = self
        return macd - sign


class TrueRange(CustomFactor):
    """Wilder's ATR(14) = EMA(2 * 14 - 1, inputs=(TrueRange(),)), which is alpha = 1 / 14"""

    inputs = (OHLCV.high, OHLCV.low, OHLCV.close)
    win = 2
    _min_win = 2
    _own_bar_scale = True

    def compute(self, highs, lows, closes):
        high_to_low = highs.last() - lows.last()
        high_to_prev_close = (highs.last() - closes.first()).abs()
        low_to_prev_close = (lows.last() - closes.first()).abs()
        max1 = high_to_low.where(high_to_low > high_to_prev_close, high_to_prev_close)
        return max1.where(max1 > low_to_prev_close, low_to_prev_close)


class _RelativeStrength(CustomFactor):
    """100 - 100 / (1 + RS), RS is the smoothed gains of the bars over their smoothed losses."""

    inputs = (OHLCV.close,)
    win = 14
    _min_win = 2

    def _smooth(self, changes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")

    def compute(self, closes):
        def _rsi(_closes):
            diff = _closes[:, :, 1:] - _closes[:, :, :-1]
            up = self._smooth(diff.clamp(min=0))
            down = self._smooth(-diff.clamp(max=0))
            return 100 - (100 / (1 + up / down))

        return closes.agg(_rsi)

    def normalized(self):
        return self / 50 - 1


class RSI(_RelativeStrength):
    """
    Wilder's RSI, usage: RSI(win, inputs=[OHLCV.close])
    His smoothing is an EMA with alpha = 1 / win, seeded with the first change of a finite window
    of `EMA`, so it doesn't depend on the data before that window.
    This is the RSI of TA-Lib and of chart software. To mine factors, see `CutlerRSI`.
    """

    def __init__(self, win: int | None = None, inputs: Sequence[BaseFactor] | None = None):
        super().__init__(win, inputs)
        self._smoothing = EMA(2 * self.win - 1, self.inputs)
        self.win = self._smoothing.win + 1  # +1 for 1 day diff

    def _smooth(self, changes: torch.Tensor) -> torch.Tensor:
        return self._smoothing.smooth(changes)


class CutlerRSI(_RelativeStrength):
    """
    Cutler's RSI, usage: CutlerRSI(win, inputs=[OHLCV.close])
    Simple means of the gains and losses of the last `win` bars, as in zipline.

    Prefer this one for mining factors, where a factor is computed thousands of times. It is about
    15 times faster than `RSI` on a GPU and 6 times on a CPU, and it reads `win + 1` bars where
    `RSI` reads `9 * win + 1`, so it loads less history, and a new listing has a full window sooner.

    It is another indicator, not a faster way to get `RSI`. With `win=14` on the WIKI prices of
    2014-2017 the values differ by 7 on average, and 18% of the bars are on another side of 70 or
    30. For the levels of TA-Lib or chart software, use `RSI`.
    """

    def __init__(self, win: int | None = None, inputs: Sequence[BaseFactor] | None = None):
        super().__init__(win, inputs)
        self.win = self.win + 1  # +1 for 1 day diff

    def _smooth(self, changes: torch.Tensor) -> torch.Tensor:
        return nanmean(changes, dim=2)


class FastStochasticOscillator(CustomFactor):
    """usage: STOCHF(win, inputs=[OHLCV.high, OHLCV.low, OHLCV.close])"""

    inputs = (OHLCV.high, OHLCV.low, OHLCV.close)
    win = 14
    _min_win = 2

    def compute(self, highs, lows, closes):
        highest_highs = highs.nanmax()
        lowest_lows = lows.nanmin()
        price_range = highest_highs - lowest_lows
        k = (closes.last() - lowest_lows) / price_range.masked_fill(price_range == 0, np.nan)

        return k * 100

    def normalized(self):
        return self / 100 - 0.5


BBANDS = BollingerBands
MACD = MovingAverageConvergenceDivergenceSignal
TRANGE = TrueRange
STOCHF = FastStochasticOscillator
