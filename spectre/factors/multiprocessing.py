"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import sys
from collections.abc import Sequence
from multiprocessing import Pool, cpu_count, get_start_method
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

from ..parallel import Rolling
from .datafactor import ColumnDataFactor
from .factor import BaseFactor, CustomFactor


class CPFCaller:
    inputs = None
    win = None
    callback = None

    def split_call(self, splits):
        split_inputs = [[]] * len(self.inputs)
        for i, data in enumerate(self.inputs):
            if isinstance(data, pd.DataFrame):
                split_inputs[i] = [data.iloc[beg:end] for beg, end in splits]
            else:
                split_inputs[i] = [data] * len(splits)
        return np.array([self.callback(*params) for params in zip(*split_inputs)])


class CPUParallelFactor(CustomFactor):
    """
    Use CPU multi-process/thread instead of GPU to process each window of data.
    Useful when your calculations can only be done in the CPU.

    The performance of this method is not so ideal, definitely not as fast as
    using the vectorization library directly.
    """

    def __init__(
        self,
        win: int | None = None,
        inputs: Sequence[BaseFactor] | None = None,
        multiprocess=False,
        core=None,
    ):
        """
        `multiprocess=True` needs the child processes to import `mp_compute`, so write the
        CPUParallelFactor code in a file. In a notebook cell it raises ValueError, unless you
        choose the `fork` start method, which no platform uses by default since Python 3.14.
        """
        super().__init__(win, inputs)

        for data in inputs:
            if isinstance(data, ColumnDataFactor):
                raise ValueError(
                    "Cannot use ColumnDataFactor in CPUParallelFactor, "
                    "please use AdjustedColumnDataFactor instead."
                )
        if multiprocess:
            self.pool = Pool
        else:
            self.pool = ThreadPool
        if core is None:
            self.core = cpu_count()
        else:
            self.core = core

    def _check_importable(self) -> None:
        """
        Only `fork` children inherit `mp_compute`, the others have to import it. If they can't,
        each one dies, the pool keeps replacing them and `map` never returns.
        """
        in_main = type(self).mp_compute.__module__ == "__main__"
        no_file = not hasattr(sys.modules["__main__"], "__file__")
        if in_main and no_file and get_start_method() != "fork":
            raise ValueError(
                f"multiprocess=True can't import {type(self).__name__}.mp_compute, because "
                "it is defined in a notebook cell or another `__main__` without a file. "
                "Move it to a .py file, or use multiprocess=False."
            )

    def compute(self, *inputs):
        n_cores = self.core
        origin_input = None
        date_count = 0

        converted_inputs = []
        for data in inputs:
            if isinstance(data, Rolling):
                s = self._revert_to_series(data.last())
                unstacked = s.unstack(level=1)
                converted_inputs.append(unstacked)
                if origin_input is None:
                    origin_input = s
                    date_count = len(unstacked)
            else:
                converted_inputs.append(data)

        backwards = self.get_total_backwards_()
        first_win_beg = backwards - self.win + 1
        first_win_end = backwards + 1
        windows = date_count - backwards
        ranges = list(
            zip(range(first_win_beg, first_win_beg + windows), range(first_win_end, date_count + 1))
        )
        caller = CPFCaller()
        caller.inputs = converted_inputs
        caller.callback = type(self).mp_compute

        n_cores = min(n_cores, len(ranges))
        split_range = np.array_split(ranges, n_cores)

        if self.pool is Pool:
            self._check_importable()
        with self.pool(n_cores) as p:
            pool_ret = p.map(caller.split_call, split_range)

        pool_ret = np.concatenate(pool_ret)
        ret = pd.Series(index=origin_input.index, dtype="float64").unstack(level=1)
        if pool_ret.shape != ret.iloc[backwards:].shape:
            raise ValueError(
                f"return value shape {pool_ret.shape} != original {ret.iloc[backwards:].shape}"
            )

        ret.iloc[backwards:] = pool_ret
        ret = ret.stack()[origin_input.index]

        return self._regroup(ret)

    @staticmethod
    def mp_compute(*inputs) -> np.array:
        """
        You will receive a window of the input data, type is pd.DataFrame.
        The table below is how it looks when win=3
        |    date    |   A  |   AA   | ... |   ZET  |
        |------------|------|--------|-----|--------|
        | 2020-01-01 | 11.1 | xxx.xx | ... | 123.45 |
        | 2020-01-02 | ...  | ...    | ... | ...    |
        | 2020-01-03 | 22.2 | xxx.xx | ... | 234.56 |

        You should return an np.array of length `input.shape[1]`
        """
        raise NotImplementedError("abstractmethod")
