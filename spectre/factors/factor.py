"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from abc import ABC
from typing import Optional, Sequence, Union
import numpy as np
import torch
from ..parallel import nansum, nanmean, nanstd, nanlast, Rolling
from .plotting import plot_factor_diagram


class BaseFactor:
    """ Basic factor class, only helper methods """
    groupby = 'asset'  # indicates inputs and return value of this factor are grouped by what
    _engine = None

    # --------------- overload ops ---------------

    def __add__(self, other):
        return AddFactor(inputs=(self, other))

    def __sub__(self, other):
        return SubFactor(inputs=(self, other))

    def __mul__(self, other):
        return MulFactor(inputs=(self, other))

    def __truediv__(self, other):
        return DivFactor(inputs=(self, other))

    # __mod__

    def __pow__(self, other):
        return PowFactor(inputs=(self, other))

    def __neg__(self):
        return NegFactor(inputs=(self,))

    def __and__(self, other):
        return AndFactor(inputs=(self, other))

    def __or__(self, other):
        return OrFactor(inputs=(self, other))

    def __lt__(self, other):
        return LtFactor(inputs=(self, other))

    def __le__(self, other):
        return LeFactor(inputs=(self, other))

    def __gt__(self, other):
        return GtFactor(inputs=(self, other))

    def __ge__(self, other):
        return GeFactor(inputs=(self, other))

    def __eq__(self, other):
        return EqFactor(inputs=(self, other))

    def __ne__(self, other):
        return NeFactor(inputs=(self, other))

    def __invert__(self):
        return InvertFactor(inputs=(self,))

    def __getitem__(self, key):
        return MultipleReturnSelector(inputs=(self, key))

    # --------------- helper functions ---------------

    def top(self, n, mask: 'BaseFactor' = None):
        return self.rank(ascending=False, mask=mask) <= n

    def bottom(self, n, mask: 'BaseFactor' = None):
        return self.rank(ascending=True, mask=mask) <= n

    def rank(self, ascending=True, mask: 'BaseFactor' = None):
        factor = RankFactor(inputs=(self,))
        # factor.method = method
        factor.ascending = ascending
        factor.set_mask(mask)
        return factor

    def zscore(self, axis_asset=False, mask: 'BaseFactor' = None):
        if axis_asset:
            factor = AssetZScoreFactor(inputs=(self,))
        else:
            factor = ZScoreFactor(inputs=(self,))
        factor.set_mask(mask)
        return factor

    def demean(self, groupby: Union[str, dict] = None, mask: 'BaseFactor' = None):
        """
        Set `groupby` to the name of a column, like 'sector'.
        `groupby` also can be a dictionary like groupby={'name': group_id}, `group_id` must > 0
        dict groupby will interrupt the parallelism of cuda, it is recommended to add group key to
        the Dataloader as a column, or use it only in the last step.
        """
        factor = DemeanFactor(inputs=(self,))
        if isinstance(groupby, str):
            factor.groupby = groupby
        elif isinstance(groupby, dict):
            factor.group_dict = groupby
        elif groupby is not None:
            raise ValueError()
        factor.set_mask(mask)
        return factor

    def quantile(self, bins=5, mask: 'BaseFactor' = None):
        factor = QuantileFactor(inputs=(self,))
        factor.bins = bins
        factor.set_mask(mask)
        return factor

    def to_weight(self, demean=True, mask: 'BaseFactor' = None):
        factor = ToWeightFactor(inputs=(self,))
        factor.set_mask(mask)
        factor.demean = demean
        return factor

    def shift(self, periods=1):
        factor = ShiftFactor(inputs=(self,))
        factor.periods = periods
        return factor

    def abs(self):
        return AbsFactor(inputs=(self,))

    def filter(self, mask):
        mf = DoNothingFactor(inputs=(self,))
        mf.set_mask(mask)
        return mf

    # --------------- main methods ---------------

    def _regroup_by_other(self, factor, factor_out):
        if factor.groupby != self.groupby:
            ret = self._engine.revert_(factor_out, factor.groupby, type(factor).__name__)
            return self._regroup(ret)
        else:
            return factor_out

    def _regroup(self, data):
        return self._engine.group_by_(data, self.groupby)

    def _revert_to_series(self, data):
        return self._engine.revert_to_series_(data, self.groupby, type(self).__name__)

    def get_total_backwards_(self) -> int:
        raise NotImplementedError("abstractmethod")

    def include_close_data(self) -> bool:
        return False

    def pre_compute_(self, engine: 'FactorEngine', start, end) -> None:
        self._engine = engine
        engine.column_to_parallel_groupby_(self.groupby)

    def clean_up_(self) -> None:
        self._engine = None

    def compute_(self, stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")

    def __init__(self) -> None:
        pass

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")


class CustomFactor(BaseFactor):
    """ Common base class for factor, that can contain child factors and handle
    hierarchical relationships """
    # settable member variables
    win = 1          # determine include how many previous data
    inputs = None    # any values in `inputs` list will pass to `compute` function by order
    _min_win = None  # assert when `win` less than `_min_win`, prevent user error.

    # internal member variables
    _cache = None
    _ref_count = 0
    _cache_stream = None
    _mask = None

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
        super().__init__()
        if win:
            self.win = win
        if inputs:
            self.inputs = inputs

        assert (self.win >= (self._min_win or 1))

    def set_mask(self, mask: BaseFactor = None):
        """ Mask fill all **INPUT** data to NaN """
        self._mask = mask

    def get_total_backwards_(self) -> int:
        backwards = 0
        if self.inputs:
            backwards = max([up.get_total_backwards_() for up in self.inputs
                            if isinstance(up, BaseFactor)] or (0,))
        backwards = backwards + self.win - 1

        if self._mask:
            mask_backward = self._mask.get_total_backwards_()
            return max(mask_backward, backwards)
        else:
            return backwards

    def include_close_data(self) -> bool:
        ret = super().include_close_data()
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    up_ret = upstream.include_close_data()
                    ret = max(ret, up_ret)
        return ret

    def show_graph(self):
        plot_factor_diagram(self)

    def clean_up_(self) -> None:
        super().clean_up_()
        self._cache = None
        self._cache_stream = None
        self._ref_count = 0

        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream.clean_up_()

        if self._mask is not None:
            self._mask.clean_up_()

    def pre_compute_(self, engine, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        super().pre_compute_(engine, start, end)
        self._cache = None
        self._cache_stream = None

        self._ref_count += 1
        if self._ref_count > 1:  # already pre_compute_ed, skip child
            return
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream.pre_compute_(engine, start, end)

        if self._mask is not None:
            self._mask.pre_compute_(engine, start, end)

    def _format_input(self, upstream, upstream_out, mask_factor, mask_out):
        # If input.groupby not equal self.groupby, convert it
        ret = self._regroup_by_other(upstream, upstream_out)

        if mask_out is not None:
            mask = self._regroup_by_other(mask_factor, mask_out)
            ret = ret.masked_fill(~mask, np.nan)

        # if need rolling and adjustment
        if self.win > 1:
            adj_multi = None
            if isinstance(upstream, DataFactor):
                adj_multi = upstream.adjustments
            ret = Rolling(ret, self.win, adj_multi)
        return ret

    def compute_(self, down_stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        # return cached result
        self._ref_count -= 1
        assert self._ref_count >= 0
        if self._cache is not None:
            if down_stream:
                down_stream.wait_event(self._cache_stream.record_event())
            ret = self._cache
            if self._ref_count == 0:
                self._cache = None
                self._cache_stream = None
            return ret

        # create self stream
        self_stream = None
        if down_stream:
            self_stream = torch.cuda.Stream(device=down_stream.device)
            if self._ref_count > 0:
                self._cache_stream = self_stream

        # Calculate mask
        mask_out = None
        if self._mask:
            mask_out = self._mask.compute_(self_stream)
        # Calculate inputs
        inputs = []
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    out = upstream.compute_(self_stream)
                    if self_stream:
                        with torch.cuda.stream(self_stream):
                            out = self._format_input(upstream, out, self._mask, mask_out)
                    else:
                        out = self._format_input(upstream, out, self._mask, mask_out)
                    inputs.append(out)
                else:
                    inputs.append(upstream)

        if self_stream:
            with torch.cuda.stream(self_stream):
                out = self.compute(*inputs)
            down_stream.wait_event(self_stream.record_event())
        else:
            out = self.compute(*inputs)

        if self._ref_count > 0:
            self._cache = out
        return out

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Abstractmethod, do the actual factor calculation here.
        Unlike zipline, here calculate all data at once. Does not guarantee Look-Ahead Bias.

        `inputs` Data structure:
        For parallel, the data structure is designed for optimal performance.
        * Groupby `asset`(default):
            set N = individual asset tick count, Max = Max tick count of all asset
            win = 1:
                | asset id | price(t+0) | ... | price(t+N) | price(t+N+1) | ... | price(t+Max) |
                |----------|------------|-----|------------|--------------|-----|--------------|
                |     0    | 123.45     | ... | 234.56     | NaN          | ... | Nan          |
                The price is sorted by tick, not by time, so it won't be aligned by time and got NaN
                values in the middle of prices (unless tick value itself is NaN), NaNs all put at
                the end of the row.
            win > 1:
                Gives you a rolling object `r`, you can `r.mean()`, `r.sum()`, or `r.agg()`
                `r.agg(callback)` gives you raw data structure as:
                | asset id | Rolling tick | price(t+0) | ... | price(t+Win) |
                |----------|--------------|------------|-----|--------------|
                |     0    | 0            | NaN        | ... | 123.45       |
                |          | ...          | ...        | ... | ...          |
                |          | N            | xxx.xx     | ... | 234.56       |
                If this table too big, it will split to multiple tables and call the callback
                function separately.
        * Groupby `date` or others (set `groupby = 'date'`):
            set N = asset count, Max = Max asset count in all time
                | time id  | price(t+0) | ... | price(t+N) | price(t+N+1) | ... | price(t+Max) |
                |----------|------------|-----|------------|--------------|-----|--------------|
                |     0    | 100.00     | ... | 200.00     | NaN          | ... | Nan          |
                The prices is all asset price in same tick time, this is useful for calculations
                such as rank, quantile.
                But the order of assets in each row (time) is random, so the column cannot be
                considered as a particular asset.
        * Custom:
            Use `series = self._revert_to_series(input)` you can get `pd.Series` data type, and
            manipulate by your own. Remember to call `return self._regroup(series)` when returning.
            WARNING: This method will be very inefficient and break parallel.
        :param inputs: All input factors data, including all data from `start(minus win)` to `end`.
        :return: your factor values, length should be same as the `inputs`
        """
        raise NotImplementedError("abstractmethod")


class FilterFactor(CustomFactor, ABC):
    def shift(self, periods=1):
        factor = FilterShiftFactor(inputs=(self,))
        factor.periods = periods
        return factor


class DataFactor(BaseFactor):
    def __init__(self, inputs: Optional[Sequence[str]] = None,
                 is_data_after_market_close=True) -> None:
        super().__init__()
        if inputs:
            self.inputs = inputs
        assert (3 > len(self.inputs) > 0), \
            "DataFactor's `inputs` can only contains one data column and corresponding " \
            "adjustments column"
        self._data = None
        self._multi = None
        self.is_data_after_market_close = is_data_after_market_close

    @property
    def adjustments(self):
        return self._multi

    def get_total_backwards_(self) -> int:
        return 0

    def include_close_data(self) -> bool:
        return self.is_data_after_market_close

    def pre_compute_(self, engine: 'FactorEngine', start, end) -> None:
        super().pre_compute_(engine, start, end)
        self._data = engine.column_to_tensor_(self.inputs[0])
        self._data = engine.group_by_(self._data, self.groupby)
        if len(self.inputs) > 1 and self.inputs[1] in engine.dataframe_:
            self._multi = engine.column_to_tensor_(self.inputs[1])
            self._multi = engine.group_by_(self._multi, self.groupby)
        else:
            self._multi = None

    def clean_up_(self) -> None:
        super().clean_up_()
        self._data = None
        self._multi = None

    def compute_(self, stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        return self._data

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        pass


class AdjustedDataFactor(CustomFactor):
    def __init__(self, data: DataFactor):
        super().__init__(1, (data,))
        self.parent = data

    def compute(self, data) -> torch.Tensor:
        multi = self.parent.adjustments
        if multi is None:
            return data
        return data * multi / nanlast(multi, dim=1)[:, None]


class TimeGroupFactor(CustomFactor, ABC):
    """Class that inputs and return value is grouped by datetime"""
    groupby = 'date'
    win = 1

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        super().__init__(win, inputs)
        assert self.win == 1, 'TimeGroupFactor can only be win=1'


# --------------- helper factors ---------------

class MultipleReturnSelector(CustomFactor):

    def compute(self, data: torch.Tensor, key) -> torch.Tensor:
        return data[:, :, key]


class ShiftFactor(CustomFactor):
    periods = 1

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        shift = data.roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0:self.periods] = np.nan
        else:
            shift[:, self.periods:] = np.nan
        return shift


class AbsFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.abs()


class DoNothingFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data


class FilterShiftFactor(CustomFactor):
    """For "roll_cuda" not implemented for 'Bool' """
    periods = 1

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        shift = data.char().roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0:self.periods] = 0
        else:
            shift[:, self.periods:] = 0

        return shift.bool()


class RankFactor(TimeGroupFactor):
    ascending = True,

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if not self.ascending:
            filled = data.clone()
            filled.masked_fill_(torch.isnan(data), -np.inf)
        else:
            filled = data
        _, indices = torch.sort(filled, dim=1, descending=not self.ascending)
        _, indices = torch.sort(indices, dim=1)
        rank = indices.float() + 1.
        rank[torch.isnan(data)] = np.nan
        return rank


class DemeanFactor(TimeGroupFactor):
    group_dict = None

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if self.group_dict is not None:
            s = self._revert_to_series(data)
            d = dict.fromkeys(s.index.levels[1], -1)
            d.update(self.group_dict)
            g = s.groupby([d, 'date'], level=1)
            ret = g.transform(lambda x: x - x.mean())
            return self._regroup(ret)
        else:
            return data - nanmean(data)[:, None]


class ZScoreFactor(TimeGroupFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return (data - nanmean(data)[:, None]) / nanstd(data)[:, None]


class AssetZScoreFactor(CustomFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return (data - nanmean(data)[:, None]) / nanstd(data)[:, None]


class QuantileFactor(TimeGroupFactor):
    """return the quantile that factor belongs to each tick"""
    bins = 5

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        x, _ = torch.sort(data, dim=1)
        mask = torch.isnan(data)
        act_size = data.shape[1] - mask.sum(dim=1)
        q = np.linspace(0, 1, self.bins + 1, dtype=np.float32)
        q = torch.tensor(q[:, None], device=data.device)
        q_index = q * (act_size - 1)
        q_weight = q % 1
        q_index = q_index.long()
        q_next = q_index + 1
        q_next[-1] = act_size - 1

        rows = torch.arange(data.shape[0], device=data.device)
        b_start = x[rows, q_index]
        b = b_start + (x[rows, q_next] - b_start) * q_weight
        b[0] -= 1
        b = b[:, :, None]

        ret = data.new_full(data.shape, np.nan, dtype=torch.float32)
        for start, end, tile in zip(b[:-1], b[1:], range(self.bins)):
            ret[(data > start) & (data <= end)] = tile + 1.
        return ret


class ToWeightFactor(TimeGroupFactor):
    demean = True

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if self.demean:
            data = data - nanmean(data)[:, None]
        return data / nansum(data.abs(), dim=1)[:, None]


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


class InvertFactor(FilterFactor):
    def compute(self, left) -> torch.Tensor:
        return ~left


class OrFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left | right


class AndFactor(FilterFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left & right


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
