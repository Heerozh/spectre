from abc import ABC
from typing import Optional, Sequence, Union
import pandas as pd


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

    def demean(self):
        fact = DemeanFactor(inputs=(self,))
        return fact

    # --------------- main methods ---------------

    def _get_total_backward(self) -> int:
        raise NotImplementedError("abstractmethod")

    def _pre_compute(self, engine, start, end) -> None:
        raise NotImplementedError("abstractmethod")

    def _compute(self) -> any:
        raise NotImplementedError("abstractmethod")

    def __init__(self, win: Optional[int] = None,
                 inputs: Optional[Sequence[any]] = None) -> None:
        pass

    def compute(self, *inputs) -> any:
        raise NotImplementedError("abstractmethod")


class CustomFactor(BaseFactor):
    win = 1
    inputs = None

    _min_win = None
    _cache = None
    _cache_hit = 0

    def _get_total_backward(self) -> int:
        backward = 0
        if self.inputs:
            backward = max([up._get_total_backward()
                            for up in self.inputs if isinstance(up, BaseFactor)] or (0,))
        return backward + self.win - 1

    def _pre_compute(self, engine, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        self._cache = None
        self._cache_hit = 0
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream._pre_compute(engine, start, end)

    def _compute(self) -> Union[Sequence, pd.DataFrame]:
        if self._cache is not None:
            self._cache_hit += 1
            return self._cache

        # Calculate inputs
        if self.inputs:
            inputs = []
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream_out = upstream._compute()
                    inputs.append(upstream_out)
                else:
                    inputs.append(upstream)
            out = self.compute(*inputs)
        else:
            out = self.compute()
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

    def compute(self, *inputs) -> Union[Sequence, pd.DataFrame]:
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
    def _get_total_backward(self) -> int:
        return 0

    def __init__(self, inputs: Optional[Sequence[str]] = None) -> None:
        super().__init__(inputs)
        if inputs:
            self.inputs = inputs
        assert (len(self.inputs) == 1), "DataFactor's `inputs` can only contains on column"

    def _pre_compute(self, engine, start, end) -> None:
        df = engine.get_loader_data()
        self._data = df[self.inputs[0]].unstack(level=1)

    def _compute(self) -> pd.DataFrame:
        return self._data

    def compute(self, *inputs) -> any:
        pass


class FilterFactor(CustomFactor, ABC):
    pass


# --------------- helper factors ---------------


class RankFactor(CustomFactor):
    method = 'first'
    ascending = True,

    def compute(self, data) -> Union[Sequence, pd.DataFrame]:
        return data.rank(axis=1, method=self.method, ascending=self.ascending)


class DemeanFactor(CustomFactor):
    # todo 不按sector demean没有意义，和rank一样
    # 可以iex ref-data/sectors 先获取行业列表，然后Collections获取股票？
    def compute(self, data) -> Union[Sequence, pd.DataFrame]:
        return data.sub(data.mean(axis=1), axis=0)


class ZScoreFactor(CustomFactor):

    def compute(self, data) -> Union[Sequence, pd.DataFrame]:
        return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)


# --------------- op factors ---------------


class SubFactor(CustomFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left - right


class AddFactor(CustomFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left + right


class MulFactor(CustomFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left * right


class DivFactor(CustomFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left / right


class PowFactor(CustomFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left ** right


class NegFactor(CustomFactor):
    def compute(self, left) -> Union[Sequence, pd.DataFrame]:
        return -left


class LtFactor(FilterFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left.lt(right)


class LeFactor(FilterFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left.le(right)


class EqFactor(FilterFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left.eq(right)


class NeFactor(FilterFactor):
    def compute(self, left, right) -> Union[Sequence, pd.DataFrame]:
        return left.ne(right)
