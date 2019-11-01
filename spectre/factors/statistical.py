from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .engine import OHLCV
import numpy as np
import pandas as pd


class StandardDeviation(CustomFactor):
    inputs = [OHLCV.close]

    def compute(self, data):
        return data.rolling(self.win).std()


STDDEV = StandardDeviation
