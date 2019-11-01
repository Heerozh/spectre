from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .basic import MA, EMA
from .statistical import STDDEV
from .engine import OHLCV
import numpy as np
import pandas as pd


class NormalizedBollingerBands(CustomFactor):
    inputs = (OHLCV.close, MA(win=20), STDDEV(win=20), 2)

    def compute(self, close, ma, std, k):
        return (close - ma) / (k * std)


class MovingAverageConvergenceDivergenceSignal(EMA):
    """
    engine.add( MACD(win=sign, inputs=(EMA(win=fast), EMA(win=slow))) )
    or
    engine.add( MACD().normalized() )
    """
    inputs = (EMA(win=12) - EMA(win=26),)
    win = 9

    def normalized(self):
        """In order not to double the calculation, reuse `inputs` factor here"""
        macd = self.inputs[0]
        sign = self
        return macd - sign


BBANDS = NormalizedBollingerBands
MACD = MovingAverageConvergenceDivergenceSignal
