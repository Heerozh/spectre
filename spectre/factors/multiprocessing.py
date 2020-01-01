"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Optional, Sequence
from .factor import BaseFactor, CustomFactor
from .datafactor import DataFactor
from ..parallel import Rolling
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool


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

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None,
                 multiprocess=False, core=None):
        """
        `multiprocess=True` may not working on windows If your code is written in a notebook cell.
         So it is recommended that you write the CPUParallelFactor code in a file.
         """
        super().__init__(win, inputs)

        for data in inputs:
            if isinstance(data, DataFactor):
                raise ValueError('Cannot use DataFactor in CPUParallelFactor, '
                                 'please use AdjustedDataFactor instead.')
        if multiprocess:
            self.pool = Pool
        else:
            self.pool = ThreadPool
        if core is None:
            self.core = cpu_count()
        else:
            self.core = core

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
        ranges = list(zip(range(first_win_beg, first_win_beg + windows),
                          range(first_win_end, date_count + 1)))
        caller = CPFCaller()
        caller.inputs = converted_inputs
        caller.callback = type(self).mp_compute

        if len(ranges) < n_cores:
            n_cores = len(ranges)
        split_range = np.array_split(ranges, n_cores)

        with self.pool(n_cores) as p:
            pool_ret = p.map(caller.split_call, split_range)

        pool_ret = np.concatenate(pool_ret)
        ret = pd.Series(index=origin_input.index).unstack(level=1)
        if pool_ret.shape != ret.iloc[backwards:].shape:
            raise ValueError('return value shape {} != original {}'.format(
                pool_ret.shape, ret.iloc[backwards:].shape))

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
