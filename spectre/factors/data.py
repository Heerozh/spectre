from typing import Optional, Iterable

import pandas as pd

from .factor import BaseFactor


class DataFactor(BaseFactor):

    def compute(self, out: pd.Series, start):
        end = self._engine.last_range[1]

        # 想办法读文件然后只读范围内的