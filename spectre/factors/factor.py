from abc import ABC
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd
import torch


class BaseFactor:
    is_timegroup = False  # indicates inputs and return value of this factor are grouped by time
    _engine = None

    # --------------- overload ops ---------------

    def __add__(self, other):
        return AddFactor(inputs=(self, other))

    def __sub__(self, other):
        return SubFactor(inputs=(self, other))

    def __mul__(self, other):
        return MulFactor(inputs=(self, other))

    def __div__(self, other):
        return DivFactor(inputs=(self, other))

    # __mod__

    def __pow__(self, other):
        return PowFactor(inputs=(self, other))

    def __neg__(self):
        return NegFactor(inputs=(self,))

    # and or xor

    def __lt__(self, other):
        return LtFactor(inputs=(self, other))

    def __le__(self, other):
        return LeFactor(inputs=(self, other))

    def __gt__(self, other):
        return LtFactor(inputs=(self, other))

    def __ge__(self, other):
        return LeFactor(inputs=(self, other))

    def __eq__(self, other):
        return EqFactor(inputs=(self, other))

    def __ne__(self, other):
        return NeFactor(inputs=(self, other))

    # --------------- helper functions ---------------

    def top(self, n):
        """This method will interrupt the parallelism of cuda, please use it at the last step."""
        return self.rank(ascending=False) <= n

    def bottom(self, n):
        """This method will interrupt the parallelism of cuda, please use it at the last step."""
        return self.rank(ascending=True) <= n

    def rank(self, ascending=True):
        """This method will interrupt the parallelism of cuda, please use it at the last step."""
        fact = RankFactor(inputs=(self,))
        # fact.method = method
        fact.ascending = ascending
        return fact

    def zscore(self):
        """This method will interrupt the parallelism of cuda, please use it at the last step."""
        fact = ZScoreFactor(inputs=(self,))
        return fact

    def demean(self, groupby=None):
        """
        This method will interrupt the parallelism of cuda, please use it at the last step.
        groupby={'name':group_id}
        """
        fact = DemeanFactor(inputs=(self,))
        fact.groupby = groupby
        return fact

    # --------------- main methods ---------------

    def _regroup_by_time(self, data):
        return self._engine.regroup_by_time_(data)

    def _regroup_by_asset(self, data):
        return self._engine.regroup_by_asset_(data)

    def get_total_backward_(self) -> int:
        raise NotImplementedError("abstractmethod")

    # def _unstack(self, data: torch.Tensor) -> torch.Tensor:
    #     s = self._engine.revert_to_series_(data)
    #     self._unstack_df = s.unstack(level=1)
    #     return data.new_tensor(self._unstack_df.values.T)
    #
    # def _stack(self, data: torch.Tensor) -> torch.Tensor:
    #     df = self._unstack_df
    #     self._unstack_df = None
    #     data = data.cpu().numpy().astype(np.float).T
    #     df[~df.isnull()] = data
    #     # 不行，还是会报长度不符合的错误，原因是如果原来有数据，计算后变nan，那么stack后就是0了
    #     return df.stack().values

    def pre_compute_(self, engine, start, end) -> None:
        self._engine = engine

    def compute_(self, stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")

    def __init__(self, win: Optional[int] = None,
                 inputs: Optional[Sequence[any]] = None) -> None:
        pass

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")


class CustomFactor(BaseFactor):
    # settable member variables
    win = 1         # determine include how many previous data
    inputs = None   # any values in `inputs` list will pass to `compute` function by order
    _min_win = None # assert when `win` less than `_min_win`, prevent user error.

    # internal member variables
    _cache = None
    _cache_hit = 0

    def get_total_backward_(self) -> int:
        backward = 0
        if self.inputs:
            backward = max([up.get_total_backward_() for up in self.inputs
                            if isinstance(up, BaseFactor)] or (0,))
        return backward + self.win - 1

    def pre_compute_(self, engine, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        super().pre_compute_(engine, start, end)
        self._cache = None
        self._cache_hit = 0
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream.pre_compute_(engine, start, end)

    def compute_(self, down_stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        if self._cache is not None:
            self._cache_hit += 1
            return self._cache

        # create self stream
        self_stream = None
        if down_stream:
            self_stream = torch.cuda.Stream(device=down_stream.device)
            down_stream.wait_stream(self_stream)

        # Calculate inputs
        inputs = []
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream_out = upstream.compute_(self_stream)
                    # 如果input是timegroup，且自己不是，就convert到asset group
                    if upstream.is_timegroup and not self.is_timegroup:
                        upstream_out = self._regroup_by_asset(upstream_out)
                    elif not upstream.is_timegroup and self.is_timegroup:
                        upstream_out = self._regroup_by_time(upstream_out)
                    inputs.append(upstream_out)
                else:
                    inputs.append(upstream)

        if self_stream:
            with torch.cuda.stream(self_stream):
                out = self.compute(*inputs)
        else:
            out = self.compute(*inputs)
        self._cache = out
        return out

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        """
        :param win:  Optional[int]
            Including additional past data with 'window length' in `input`
            when passed to the `compute` function.
            **If not specified, use `self.win` instead.**
        :param inputs: Optional[Iterable[OHLCV|BaseFactor]]
            Input factors, will all passed to the `compute` function.
            **If not specified, use `self.inputs` instead.**
        """
        super().__init__(win, inputs)
        if win:
            self.win = win
        if inputs:
            self.inputs = inputs

        assert (self.win >= (self._min_win or 1))

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Abstractmethod, do the actual factor calculation here.
        Unlike zipline, here calculate all data at once. Does not guarantee Look-Ahead Bias.
        Data structure:
                | index    | stock1 | ... | stockN |
                |----------|--------|-----|--------|
                | datetime | input  | ... | input  |
        :param inputs: All input factors data, including all data from `start(minus win)` to `end`.
        :return: your factor values, length should be same as the inputs`
        """
        raise NotImplementedError("abstractmethod")


class DataFactor(BaseFactor):
    def get_total_backward_(self) -> int:
        return 0

    def __init__(self, inputs: Optional[Sequence[str]] = None) -> None:
        super().__init__(inputs)
        if inputs:
            self.inputs = inputs
        assert (len(self.inputs) == 1), "DataFactor's `inputs` can only contains on column"
        self._data = None

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        self._data = engine.get_tensor_groupby_asset_(self.inputs[0])

    def compute_(self, stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        return self._data

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        pass


class FilterFactor(CustomFactor, ABC):
    pass


class TimeGroupFactor(CustomFactor, ABC):
    """Class that inputs and return value is grouped by time"""
    is_timegroup = True


# --------------- helper factors ---------------


class RankFactor(TimeGroupFactor):
    method = 'first'
    ascending = True,

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        # todo: 测试下zipline在0.23版本下是否确实慢了
        if not self.ascending:
            data = data.clone()
            data[torch.isnan(data)] = -np.inf
        _, indices = torch.sort(data, dim=1, descending=not self.ascending)
        _, rank = torch.sort(indices, dim=1)
        return torch.take(torch.arange(1, data.shape[0]+1, device=data.device), rank)
        # 试一下直接拿series来groupby做
        # s = self._revert_to_series(data)
        # return s.groupby(level=0).rank(method=self.method, ascending=self.ascending)


class DemeanFactor(CustomFactor):
    groupby = None

    # todo 可以iex ref-data/sectors 先获取行业列表，然后Collections获取股票？
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Recommended to set groupby, otherwise you only need use rank."""
        if self.groupby:
            g = data.groupby(self.groupby, axis=1)
            return g.transform(lambda x: x - x.mean())
        else:
            return data.sub(data.mean(axis=1), axis=0)


class ZScoreFactor(CustomFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)


# --------------- op factors ---------------


class SubFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left - right


class AddFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left + right


class MulFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left * right


class DivFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left / right


class PowFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left ** right


class NegFactor(CustomFactor):
    def compute(self, left) -> torch.Tensor:
        return -left


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

