"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from abc import ABC
from typing import Optional, Sequence, Union
import numpy as np
import torch
from ..parallel import nansum, nanmean, nanstd, pad_2d, Rolling, quantile
from ..plotting import plot_factor_diagram


class BaseFactor:
    """ Basic factor class, only helper methods """
    groupby = 'asset'  # indicates inputs and return value of this factor are grouped by what
    _engine = None

    # --------------- overload ops ---------------

    # op: +
    def __add__(self, other):
        return AddFactor(inputs=(self, other))

    def __radd__(self, other):
        return AddFactor(inputs=(other, self))

    # op: -
    def __sub__(self, other):
        return SubFactor(inputs=(self, other))

    def __rsub__(self, other):
        return SubFactor(inputs=(other, self))

    # op: *
    def __mul__(self, other):
        return MulFactor(inputs=(self, other))

    def __rmul__(self, other):
        return MulFactor(inputs=(other, self))

    # op: /
    def __truediv__(self, other):
        return DivFactor(inputs=(self, other))

    def __rtruediv__(self, other):
        return DivFactor(inputs=(other, self))

    # op: %
    def __mod__(self, other):
        return ModFactor(inputs=(self, other))

    def __rmod__(self, other):
        return ModFactor(inputs=(other, self))

    # op: **
    def __pow__(self, other):
        return PowFactor(inputs=(self, other))

    def __rpow__(self, other):
        return PowFactor(inputs=(other, self))

    # op: negative
    def __neg__(self):
        return NegFactor(inputs=(self,))

    # op: and
    def __and__(self, other):
        from .filter import AndFactor
        return AndFactor(inputs=(self, other))

    def __rand__(self, other):
        from .filter import AndFactor
        return AndFactor(inputs=(other, self))

    # op: or
    def __or__(self, other):
        from .filter import OrFactor
        return OrFactor(inputs=(self, other))

    def __ror__(self, other):
        from .filter import OrFactor
        return OrFactor(inputs=(other, self))

    # op: <=>==!=
    def __lt__(self, other):
        from .filter import LtFactor
        return LtFactor(inputs=(self, other))

    def __le__(self, other):
        from .filter import LeFactor
        return LeFactor(inputs=(self, other))

    def __gt__(self, other):
        from .filter import GtFactor
        return GtFactor(inputs=(self, other))

    def __ge__(self, other):
        from .filter import GeFactor
        return GeFactor(inputs=(self, other))

    def __eq__(self, other):
        from .filter import EqFactor
        return EqFactor(inputs=(self, other))

    def __ne__(self, other):
        from .filter import NeFactor
        return NeFactor(inputs=(self, other))

    # op: ~
    def __invert__(self):
        from .filter import InvertFactor
        return InvertFactor(inputs=(self,))

    # op: []
    def __getitem__(self, key):
        return MultiRetSelector(inputs=(self, key))

    # --------------- helper functions ---------------

    def top(self, n, mask: 'BaseFactor' = None):
        """Cross-section top values"""
        return self.rank(ascending=False, mask=mask) <= n

    def bottom(self, n, mask: 'BaseFactor' = None):
        return self.rank(ascending=True, mask=mask) <= n

    def rank(self, ascending=True, mask: 'BaseFactor' = None):
        """Cross-section rank"""
        factor = RankFactor(inputs=(self,), mask=mask)
        # factor.method = method
        factor.ascending = ascending
        return factor

    def zscore(self, groupby: str = 'date', mask: 'BaseFactor' = None, weight: 'BaseFactor' = None):
        """Cross-section zscore"""
        inputs = [self]
        if weight is not None:
            inputs.append(weight)
        factor = ZScoreFactor(inputs=inputs)
        factor.set_mask(mask)
        factor.groupby = groupby

        return factor

    def demean(self, groupby: Union[str, dict] = None, mask: 'BaseFactor' = None):
        """
        Cross-section demean.
        Set `groupby` to the name of a column, like 'sector'.
        `groupby` also can be a dictionary like groupby={'name': group_id}, `group_id` must > 0.
        Group by dict is implemented by pandas, not in GPU.
        """
        factor = DemeanFactor(inputs=(self,), mask=mask)
        if isinstance(groupby, str):
            factor.groupby = groupby
        elif isinstance(groupby, dict):
            factor.group_dict = groupby
        elif groupby is not None:
            raise ValueError()
        return factor

    def quantile(self, bins=5, mask: 'BaseFactor' = None, groupby: str = 'date'):
        """Cross-section quantiles to which the factor belongs"""
        factor = QuantileClassifier(inputs=(self,))
        factor.set_mask(mask)
        factor.groupby = groupby
        factor.bins = bins
        return factor

    def to_weight(self, demean=True, mask: 'BaseFactor' = None):
        """factor value to portfolio weight, which sum(abs(weight)) == 1"""
        factor = ToWeightFactor(inputs=(self,), mask=mask)
        factor.demean = demean
        return factor

    def shift(self, periods=1):
        factor = ShiftFactor(inputs=(self,))
        factor.periods = periods
        return factor

    def abs(self):
        return AbsFactor(inputs=(self,))

    def log(self):
        return LogFactor(inputs=(self,))

    def sum(self, win):
        return SumFactor(win, inputs=(self,))

    def filter(self, mask):
        """Local filter, fills elements of self with NaN where mask is False."""
        mf = DoNothingFactor(inputs=(self,))
        mf.set_mask(mask)
        return mf

    def one_hot(self):
        from .filter import OneHotEncoder
        factor = OneHotEncoder(self)
        return factor

    def fill_na(self, value=None, ffill=None):
        if value is not None:
            factor = FillNANFactor(inputs=(self, value))
        elif ffill:
            factor = PadFactor(inputs=(self,))
        else:
            raise ValueError('Either `value=number` or `ffill=True` must be specified.')
        return factor

    fill_nan = fill_na

    def masked_fill(self, mask, fill):
        return MaskedFillFactor(inputs=(self, mask, fill))

    def any(self, win):
        return AnyFactor(win, inputs=(self,))

    def all(self, win):
        return AllFactor(win, inputs=(self,))

    def clamp(self, left, right):
        return ClampFactor(self, left, right)

    def mad_clamp(self, z, mask: 'BaseFactor' = None):
        factor = MADClampFactor(inputs=(self,))
        factor.z = z
        factor.set_mask(mask)
        return factor

    def float32(self):
        factor = TypeCastFactor(inputs=(self,))
        factor.dtype = torch.float32
        return factor

    float = float32

    # --------------- main methods ---------------
    @property
    def adjustments(self):
        """Returning adjustments multipliers"""
        return None

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

    def should_delay(self) -> bool:
        """Is this factor should be delayed?"""
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
    _force_delay = None
    _keep_cache = False

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
        assert isinstance(self.inputs, (list, tuple, type(None))), '`factor.inputs` must be a list.'
        assert isinstance(self.win, int), '`factor.win` must be a integer.'

        assert (self.win >= (self._min_win or 1))

    @classmethod
    def sequential(cls, *args):
        """ Helper method for fast initialization """
        if isinstance(args[0], int):
            win = max(args[0], cls._min_win)
            return cls(win, inputs=[*args[1:]])
        else:
            return cls(inputs=[*args[0:]])

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

    def should_delay(self) -> bool:
        ret = super().should_delay()
        if self._force_delay is not None:
            return self._force_delay
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    up_ret = upstream.should_delay()
                    ret = max(ret, up_ret)
        return ret

    def set_delay(self, delay):
        """None: Auto delay. False: Force not to delay. True: Force to delay."""
        self._force_delay = delay

    def show_graph(self):
        plot_factor_diagram(self)

    def clean_up_(self) -> None:
        super().clean_up_()
        if not self._keep_cache:
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

        self._ref_count += 1
        if self._ref_count > 1:  # already pre_compute_ed, skip child
            return
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream.pre_compute_(engine, start, end)

        if self._mask is not None:
            self._mask.pre_compute_(engine, start, end)

        if self._keep_cache:
            # ref count +1 so this factor's cache will not cleanup
            self._ref_count += 1

    def _format_input(self, upstream, upstream_out, mask_factor, mask_out):
        # If input.groupby not equal self.groupby, convert it
        ret = self._regroup_by_other(upstream, upstream_out)

        if mask_out is not None:
            mask = self._regroup_by_other(mask_factor, mask_out)
            ret = ret.masked_fill(~mask, np.nan)

        # if need rolling and adjustment
        if self.win > 1:
            if len(ret.shape) >= 3:
                raise ValueError("upstream factor `{}` has multiple outputs ({}), "
                                 "rolling win > 1 only supports one output, "
                                 "use slice to select a value before using it, for example: "
                                 "`factor[0]`.".format(str(upstream), ret.shape[2]))
            ret = Rolling(ret, self.win, upstream.adjustments)
        return ret

    def compute_(self, down_stream: Union[torch.cuda.Stream, None]) -> torch.Tensor:
        # return cached result
        self._ref_count -= 1
        if self._ref_count < 0:
            raise ValueError('Reference count error: Maybe you override `pre_compute_`, '
                             'but did not call super() method.')
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
        Unlike zipline, here we will receive all data, you need to be careful to prevent
        Look-Ahead Bias.

        `inputs` Data structure:
        The data structure is designed for optimal performance in parallel.
        * Groupby `asset`(default):
            set N = individual asset bar count, Max = Max bar count across all asset
            win = 1:
                | asset id | price(t+0) | ... | price(t+N) | price(t+N+1) | ... | price(t+Max) |
                |----------|------------|-----|------------|--------------|-----|--------------|
                |     0    | 123.45     | ... | 234.56     | NaN          | ... | Nan          |
                If `align_by_time=False` then the time represented by each bar is different.
            win > 1:
                Gives you a rolling object `r`, you can `r.mean()`, `r.sum()`, or `r.agg()`.
                `r.agg(callback)` gives you raw data structure as:
                | asset id | Rolling bar  | price(t+0) | ... | price(t+Win) |
                |----------|--------------|------------|-----|--------------|
                |     0    | 0            | NaN        | ... | 123.45       |
                |          | ...          | ...        | ... | ...          |
                |          | N            | xxx.xx     | ... | 234.56       |
                If this table too big, it will split to multiple tables by axis 0, and call the
                callback function separately.
        * Groupby `date` or others (set `groupby = 'date'`):
            set N = asset count, Max = Max asset count in all time
                | time id  | price(t+0) | ... | price(t+N) | price(t+N+1) | ... | price(t+Max) |
                |----------|------------|-----|------------|--------------|-----|--------------|
                |     0    | 100.00     | ... | 200.00     | NaN          | ... | Nan          |
                The prices is all asset prices in same time, this is useful for calculations
                such as rank, quantile.
                If `align_by_time=False` then the order of assets in each row (time) is not fixed,
                in that way the column cannot be considered as a particular asset.
        * Custom:
            Use `series = self._revert_to_series(input)` you can get `pd.Series` data type, and
            manipulate by your own. Remember to call `return self._regroup(series)` when returning.
            WARNING: This method will be very inefficient and break parallel.
        :param inputs: All input factors data, including all data from `start(minus win)` to `end`.
        :return: your factor values, length should be same as the `inputs`
        """
        raise NotImplementedError("abstractmethod")


class CrossSectionFactor(CustomFactor, ABC):
    """Class that inputs and return value is grouped by datetime"""
    groupby = 'date'
    win = 1

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None,
                 mask: Optional[BaseFactor] = None):
        super().__init__(win, inputs)
        self.set_mask(mask)
        assert self.win == 1, 'CrossSectionFactor.win can only be 1'

    def __getitem__(self, key):
        return MultiRetSelectorCS(inputs=(self, key))


# --------------- helper factors ---------------


class MultiRetSelector(CustomFactor):

    def compute(self, data: torch.Tensor, key) -> torch.Tensor:
        if len(data.shape) < 3:
            raise KeyError('This factor has only one return value, cannot slice.')
        elif data.shape[2] <= key:
            raise KeyError('OutOfBounds: factor has only {} return values, and slice is [{}].'.
                           format(data.shape[2], key))
        return data[:, :, key]


class MultiRetSelectorCS(MultiRetSelector, CrossSectionFactor):
    pass


class ShiftFactor(CustomFactor):
    periods = 1

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if data.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError('factor.shift() does not support `int` type, '
                             'please convert to float by using `factor.float()`, upstreams: {}'
                             .format(self.inputs))

        shift = data.roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0:self.periods] = np.nan
        else:
            shift[:, self.periods:] = np.nan
        return shift


class AbsFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.abs()


class SumFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.nansum()


class LogFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.log()


class AnyFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return ~torch.isnan(data.values).any(dim=2)


class AllFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return ~torch.isnan(data.values).all(dim=2)


class TypeCastFactor(CustomFactor):
    _min_win = 1
    dtype = torch.float32

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.type(self.dtype)


class PadFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return pad_2d(data)


class FillNANFactor(CustomFactor):
    def compute(self, data: torch.Tensor, value) -> torch.Tensor:
        mask = torch.isnan(data)
        return data.masked_fill(mask, value)


class MaskedFillFactor(CustomFactor):
    def compute(self, data, mask, fill) -> torch.Tensor:
        if isinstance(fill, (int, float, bool)):
            return data.masked_fill(mask, fill)
        ret = data.clone()
        ret[mask] = fill[mask]
        return ret


class DoNothingFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data


class RankFactor(CrossSectionFactor):
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
        rank.masked_fill_(torch.isnan(data), np.nan)
        return rank


class DemeanFactor(CrossSectionFactor):
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
            return data - nanmean(data).unsqueeze(-1)


class ZScoreFactor(CrossSectionFactor):

    def compute(self, data: torch.Tensor, weight=None) -> torch.Tensor:
        if weight is None:
            mean = nanmean(data)
        else:
            mean = nansum(data * weight) / nansum(weight)
        return (data - mean.unsqueeze(-1)) / nanstd(data).unsqueeze(-1)


class QuantileClassifier(CrossSectionFactor):
    """Returns the quantile of the factor at each datetime"""
    bins = 5

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return quantile(data, self.bins, dim=1)


class ToWeightFactor(CrossSectionFactor):
    demean = True

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if self.demean:
            data = data - nanmean(data).unsqueeze(-1)
        return data / nansum(data.abs(), dim=1).unsqueeze(-1)


class ClampFactor(CustomFactor):
    """ Simple Winsorizing """
    _min_win = 1

    def __init__(self, data, left, right):
        super().__init__(win=1, inputs=[data, left, right])

    def compute(self, data, left, right):
        return data.clamp(left, right)


class MADClampFactor(CustomFactor):
    """ Mean Absolute Deviation Clamping """
    z = 5

    def compute(self, data):
        median = data.median(dim=1).values.unsqueeze(-1)
        mad = (data - median).abs().median(dim=1).values.unsqueeze(-1)
        ret = data.clone()
        upper = median + self.z * mad
        lower = median - self.z * mad
        upper_mask = data > upper
        lower_mask = data < lower
        upper = upper.repeat(1, data.shape[1])
        lower = lower.repeat(1, data.shape[1])

        ret[upper_mask] = upper[upper_mask]
        ret[lower_mask] = lower[lower_mask]
        return ret


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


class ModFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left % right


class PowFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        return left ** right


class NegFactor(CustomFactor):
    def compute(self, left) -> torch.Tensor:
        return -left
