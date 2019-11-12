from abc import ABC
from typing import Optional, Sequence, Union
import torch


class BaseFactor:
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
        return LtFactor(inputs=(other, self))

    def __ge__(self, other):
        return LeFactor(inputs=(other, self))

    def __eq__(self, other):
        return EqFactor(inputs=(self, other))

    def __ne__(self, other):
        return NeFactor(inputs=(self, other))

    # --------------- helper functions ---------------

    def top(self, n):
        return self.rank(ascending=False) <= n

    def bottom(self, n):
        return self.rank(ascending=True) <= n

    def rank(self, method='first', ascending=True):
        fact = RankFactor(inputs=(self,))
        fact.method = method
        fact.ascending = ascending
        return fact

    def zscore(self):
        fact = ZScoreFactor(inputs=(self,))
        return fact

    def demean(self, groupby=None):
        """groupby={'name':group_id}"""
        fact = DemeanFactor(inputs=(self,))
        fact.groupby = groupby
        return fact

    # --------------- main methods ---------------

    def get_total_backward_(self) -> int:
        raise NotImplementedError("abstractmethod")

    def pre_compute_(self, engine, start, end) -> None:
        raise NotImplementedError("abstractmethod")

    def compute_(self, stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")

    def __init__(self, win: Optional[int] = None,
                 inputs: Optional[Sequence[any]] = None) -> None:
        pass

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")


class CustomFactor(BaseFactor):
    win = 1
    inputs = None

    _min_win = None
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
        self._data = engine.get_data_tensor_(self.inputs[0])

    def compute_(self, stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        # todo 等待自己的数据复制完成
        return self._data

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        pass


class FilterFactor(CustomFactor, ABC):
    pass


# --------------- helper factors ---------------


class RankFactor(CustomFactor):
    method = 'first'
    ascending = True,

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.rank(axis=1, method=self.method, ascending=self.ascending)


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
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left - right


class AddFactor(CustomFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


class MulFactor(CustomFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * right


class DivFactor(CustomFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left / right


class PowFactor(CustomFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left ** right


class NegFactor(CustomFactor):
    def compute(self, left: torch.Tensor) -> torch.Tensor:
        return -left


class LtFactor(FilterFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left.lt(right)


class LeFactor(FilterFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left.le(right)


class EqFactor(FilterFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left.eq(right)


class NeFactor(FilterFactor):
    def compute(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left.ne(right)

