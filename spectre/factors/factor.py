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
from ..parallel import (nansum, nanmean, nanstd, nanmax, nanmin, pad_2d, Rolling, quantile,
                        masked_kth_value_1d, clamp_1d_, rankdata)
from ..plotting import plot_factor_diagram
from ..config import Global


class BaseFactor:
    """ Basic factor class, only helper methods """
    groupby = 'asset'  # indicates inputs and return value of this factor are grouped by what
    _engine = None
    _clean_required = None  # If your factor caches data, this should be set to True when caching

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

    # op: or
    def __or__(self, other):
        from .filter import OrFactor
        return OrFactor(inputs=(self, other))

    def __xor__(self, other):
        from .filter import XorFactor
        return XorFactor(inputs=(self, other))

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

    def top(self, n: Union[int, float], mask: 'BaseFactor' = None):
        """ Cross-section top values """
        rank_data = self.rank(ascending=False, mask=mask, method='ordinal')
        if type(n) is float:
            assert 0 < n < 1
            return rank_data <= (XSMax(inputs=[rank_data]) * n)
        else:
            return rank_data <= n

    def bottom(self, n: Union[int, float], mask: 'BaseFactor' = None):
        rank_data = self.rank(ascending=True, mask=mask, method='ordinal')
        if type(n) is float:
            assert 0 < n < 1
            return rank_data <= (XSMax(inputs=[rank_data]) * n)
        else:
            return rank_data <= n

    def rank(self, ascending=True, mask: 'BaseFactor' = None, normalize=False, method='average'):
        """ Cross-section rank """
        factor = RankFactor(inputs=(self,), mask=mask)
        factor.method = method
        factor.ascending = ascending
        factor.normalize = normalize
        return factor

    def zscore(self, groupby: str = 'date', mask: 'BaseFactor' = None, weight: 'BaseFactor' = None):
        """ Cross-section zscore """
        factor = ZScoreFactor(data=self, mask=mask, groupby=groupby, weight=weight)
        return factor

    def demean(self, groupby: Union[str, dict] = None, mask: 'BaseFactor' = None,
               beta: 'BaseFactor' = None):
        """
        Cross-section demean.
        Set `groupby` to the name of a column, like 'sector'.
        `groupby` also can be a dictionary like groupby={'name': group_id}, `group_id` must > 0.
        Group by dict is implemented by pandas, not in GPU.
        """
        factor = DemeanFactor(inputs=(self, beta), mask=mask)
        if isinstance(groupby, str):
            factor.groupby = groupby
        elif isinstance(groupby, dict):
            factor.group_dict = groupby
        elif groupby is not None:
            raise ValueError()
        return factor

    def mean(self, groupby: str = 'date', mask: Union['BaseFactor', str] = None):
        """
        Cross-section mean.
        """
        factor = MeanFactor(inputs=(self,), mask=mask)
        factor.groupby = groupby
        return factor

    def median(self, mask: 'BaseFactor' = None, groupby: str = 'date'):
        factor = MedianFactor(inputs=(self,), mask=mask)
        factor.groupby = groupby
        return factor

    def quantile(self, bins=5, mask: 'BaseFactor' = None, groupby: str = 'date'):
        """ Cross-section quantiles to which the factor belongs """
        factor = QuantileClassifier(inputs=(self,))
        factor.set_mask(mask)
        factor.groupby = groupby
        factor.bins = bins
        return factor

    def std(self, groupby: str = 'date', mask: 'BaseFactor' = None):
        from .statistical import XSStandardDeviation
        factor = XSStandardDeviation(inputs=(self,), mask=mask)
        factor.groupby = groupby
        return factor

    def to_weight(self, demean=True, mask: 'BaseFactor' = None):
        """ factor value to portfolio weight, which sum(abs(weight)) == 1 """
        factor = ToWeightFactor(inputs=(self,), mask=mask)
        factor.demean = demean
        return factor

    def shift(self, periods=1):
        if periods == 0:
            return self
        return RawShiftFactor(self, periods)

    def abs(self):
        return AbsFactor(inputs=(self,))

    def log(self):
        return LogFactor(inputs=(self,))

    def sign(self):
        return SignFactor(inputs=(self,))

    def ts_sum(self, win: int, mask: 'BaseFactor' = None):
        ret = SumFactor(win, inputs=(self,))
        ret.set_mask(mask)
        return ret

    def xs_sum(self, mask: 'BaseFactor' = None, groupby='date'):
        # todo future name will change to sum
        ret = XSSumFactor(inputs=(self,), mask=mask)
        ret.groupby = groupby
        return ret

    def ts_prod(self, win: int):
        return ProdFactor(win, inputs=(self,))

    def round(self, decimal):
        return RoundFactor(inputs=(self, decimal))

    def tanh(self):
        return TanhFactor(inputs=(self,))

    def filter(self, mask: 'BaseFactor'):
        """ Local filter, fill unmasked elements with NaNs. """
        mf = DoNothingFactor(inputs=(self,))
        mf.set_mask(mask)
        return mf

    def one_hot(self):
        from .filter import OneHotEncoder
        factor = OneHotEncoder(self)
        return factor

    def fill_na(self, value: float = None, ffill: bool = None, inf=False, nan=True):
        if value is not None:
            factor = FillNANFactor(inputs=(self, value))
        elif ffill:
            factor = PadFactor(inputs=(self,))
        else:
            raise ValueError('Either `value=number` or `ffill=True` must be specified.')
        factor.inf = inf
        factor.nan = nan
        return factor

    fill_nan = fill_na

    def masked_fill(self, mask: 'BaseFactor', fill: Union['BaseFactor', float, int, bool]):
        return MaskedFillFactor(inputs=(self, mask, fill))

    def ts_any(self, win: int):
        """ Return True if Rolling window contains any Non-NaNs """
        from .filter import AnyNonNaNFactor
        return AnyNonNaNFactor(win, inputs=(self,))

    def ts_all(self, win: int):
        """ Return True if Rolling window all are Non-NaNs """
        from .filter import AllNonNaNFactor
        return AllNonNaNFactor(win, inputs=(self,))

    def ts_count(self, win: int):
        """ Return Non-NaN data count """
        return NonNaNCountFactor(win, inputs=(self,))

    def clamp(self, left: Union[float, int], right: Union[float, int]):
        return ClampFactor(self, left, right)

    def mad_clamp(self, z: float, mask: 'BaseFactor' = None, groupby='date', fill=None, mean=False):
        """ Cross-section MAD clamp """
        factor = MADClampFactor(inputs=(self, fill))
        factor.groupby = groupby
        factor.z = z
        factor.set_mask(mask)
        factor.m_is_mean = mean
        return factor

    def winsorizing(self, z: float = 0.05, mask: 'BaseFactor' = None):
        """ Cross-section winsorizing clamp """
        factor = WinsorizingFactor(inputs=(self,))
        factor.groupby = CrossSectionFactor.groupby
        assert 0 < z < 1
        factor.z = z
        factor.set_mask(mask)
        return factor

    def half(self):
        factor = TypeCastFactor(inputs=(self,))
        factor.dtype = torch.float16
        return factor

    def float(self):
        factor = TypeCastFactor(inputs=(self,))
        factor.dtype = Global.float_type
        return factor

    def double(self):
        factor = TypeCastFactor(inputs=(self,))
        factor.dtype = torch.float64
        return factor

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

    def clean_up_(self, force=False) -> None:
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
    _mask_out = None
    _force_delay = None
    _keep_cache = False
    _total_backwards = None
    _cache_hash = None

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

        if not (self.win >= (self._min_win or 1)):
            raise ValueError('factor({}) win ({}) must  >= [_min_win({}) and 1]'.format(
                str(self), self.win, self._min_win))

        if self.inputs:
            self.inputs = list(ipt() if callable(ipt) else ipt for ipt in self.inputs)
            for ipt in self.inputs:
                if isinstance(ipt, BaseFactor):
                    if ipt._clean_required is not None:
                        self._clean_required = ipt._clean_required
                        break

    @classmethod
    def sequential(cls, *args):
        """ Helper method for fast initialization """
        if isinstance(args[0], int):
            win = args[0]
            if cls._min_win is not None:
                win = max(args[0], cls._min_win)
            return cls(win, inputs=[*args[1:]])
        else:
            return cls(inputs=[*args[0:]])

    def set_mask(self, mask: Union[BaseFactor, str] = None):
        """ Input data mask, fill all unmasked **INPUT** data to NaN """
        self._mask = mask
        self._total_backwards = None
        if isinstance(mask, BaseFactor) and mask._clean_required is not None:
            self._clean_required = max(self._clean_required or False, mask._clean_required)
        return self

    def _get_mask_factor(self):
        # todo add a placeholder factor for universe
        if type(self._mask) is str and (self._mask == 'universe') is True:
            return self._engine.get_filter()
        else:
            return self._mask

    def _get_computed_mask(self):
        """ Allow subclass get the current mask result in compute() """
        if self._mask_out is None:
            return ~self._engine.get_group_padding_mask(self.groupby)
        else:
            return self._regroup_by_other(self._get_mask_factor(), self._mask_out)

    def get_total_backwards_(self) -> int:
        if self._total_backwards is not None:
            return self._total_backwards

        backwards = 0
        if self.inputs:
            backwards = max([up.get_total_backwards_() for up in self.inputs
                             if isinstance(up, BaseFactor)] or (0,))
        backwards = backwards + self.win - 1

        if isinstance(self._mask, BaseFactor):
            # 'universe' str mask backward already calculated in engine
            mask_backward = self._get_mask_factor().get_total_backwards_()
            self._total_backwards = max(mask_backward, backwards)
        else:
            self._total_backwards = backwards
        return self._total_backwards

    def should_delay(self) -> bool:
        ret = super().should_delay()
        if self._force_delay is not None:
            return self._force_delay
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    up_ret = upstream.should_delay()
                    ret = max(ret, up_ret)
                    if ret is True:
                        return True
        if isinstance(self._mask, BaseFactor):
            ret = max(ret, self._mask.should_delay())
        return ret

    def set_delay(self, delay):
        """None: Auto delay. False: Force not to delay. True: Force to delay."""
        self._force_delay = delay

    def is_keep_cache(self) -> bool:
        if self._keep_cache:
            return True

        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, CustomFactor):
                    if upstream.is_keep_cache():
                        return True

        if isinstance(self._mask, CustomFactor):
            if self._mask.is_keep_cache():
                return True
        return False

    def show_graph(self):
        plot_factor_diagram(self)

    def clean_up_(self, force=False) -> None:
        super().clean_up_(force)
        # brand-new factor, no need to clean up
        if self._clean_required is None:
            return
        # if not brand new, then maybe some child kept cache. force tell us cache expired.
        # _clean_required: None=not to clean, True=require clean, False=only force clean
        # todo build a dependencies tree for loop factor tree, prevent circular references and
        #  reduce the number of iterations
        if not self._clean_required and not force:
            return

        if force:
            self._clean_required = None
        else:
            self._clean_required = False

        if not self._keep_cache or force:
            self._cache = None
            self._cache_stream = None
        self._ref_count = 0

        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream.clean_up_(force)

        if isinstance(self._mask, BaseFactor):
            # if mask is 'universe', engine will do clean up
            self._get_mask_factor().clean_up_(force)

    def pre_compute_(self, engine, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        # TODO If the same factor added to 2 different engines, the _keep_cache should not able to
        #  be enabled. alert or rise.
        if engine.cache_hash != self._cache_hash and self._cache is not None:
            print('The factor({}) data cached for the original engine is invalidated due to engine '
                  'changes.'.format(self))
            self._cache = None
            self._cache_stream = None

        super().pre_compute_(engine, start, end)
        self._clean_required = True

        self._ref_count += 1
        if self._ref_count > 1:  # already pre_compute_ed, skip child
            return
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream.pre_compute_(engine, start, end)

        mask_factor = self._get_mask_factor()
        if mask_factor is not None:
            mask_factor.pre_compute_(engine, start, end)

        if self._keep_cache:
            # ref count +1 so this factor's cache will not clean up
            self._ref_count += 1
            self._cache_hash = engine.cache_hash

    def _format_input(self, upstream, upstream_out, mask_factor, mask_out):
        # If input.groupby not equal self.groupby, convert it
        ret = self._regroup_by_other(upstream, upstream_out)

        if mask_out is not None:
            mask = self._regroup_by_other(mask_factor, mask_out)
            if ret.dtype == torch.bool:
                ret = ret.masked_fill(~mask, False)
            else:
                ret = ret.masked_fill(~mask, np.nan)

        # if need rolling and adjustment
        if self.win > 1:
            if len(ret.shape) >= 3:
                raise ValueError("upstream factor `{}` has multiple outputs ({}), "
                                 "rolling win > 1 only supports one output, "
                                 "use slice to select a value before using it, for example: "
                                 "`factor[0]`.".format(str(upstream), ret.shape[2]))
            # todo: If the indirect children is a column factor, also need to adjust.
            #  for now, only adjust direct children.
            #  refactor: if any shift occurs, adjust them.
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
        mask_factor = self._get_mask_factor()
        if mask_factor is not None:
            mask_out = mask_factor.compute_(self_stream)
            self._mask_out = mask_out
        # Calculate inputs
        inputs = []
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    out = upstream.compute_(self_stream)
                    if self_stream:
                        with torch.cuda.stream(self_stream):
                            out = self._format_input(upstream, out, mask_factor, mask_out)
                    else:
                        out = self._format_input(upstream, out, mask_factor, mask_out)
                    inputs.append(out)
                else:
                    inputs.append(upstream)

        if self_stream:
            with torch.cuda.stream(self_stream):
                out = self.compute(*inputs)
            down_stream.wait_event(self_stream.record_event())
        else:
            try:
                out = self.compute(*inputs)
            except RuntimeError as e:
                if 'CUDA out of memory' in repr(e):
                    out = self.compute(*[
                        t.cpu() if isinstance(t, (torch.Tensor, Rolling)) else t
                        for t in inputs])
                    out = out.to(self._engine.device)
                else:
                    raise e

        self._mask_out = None
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
                | time     | price(0) | ... | price(N) | price(N+1) | ... | price(N+Max) |
                |----------|----------|-----|----------|------------|-----|--------------|
                |     t    | 100.00   | ... | 200.00   | NaN        | ... | Nan          |
                |    ...   | ...      | ... | ...      | ...        | ... | ...          |
                |   t+Max  | 123.45   | ... | 234.56   | NaN        | ... | Nan          |
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
        return MultiRetSelectorXS(inputs=(self, key))


# --------------- helper factors ---------------


class MultiRetSelector(CustomFactor):
    def pre_compute_(self, engine, start, end):
        super().pre_compute_(engine, start, end)
        # consistent with upstream groupby, otherwise it will regroup before compute
        self.groupby = self.inputs[0].groupby

    def compute(self, data: torch.Tensor, key) -> torch.Tensor:
        if len(data.shape) < 3:
            raise KeyError('This factor({}) has only one return value, cannot slice.'.format(
                type(self.inputs[0]).__name__
            ))
        elif data.shape[2] <= key:
            raise KeyError('OutOfBounds: factor has only {} return values, and slice is [{}].'.
                           format(data.shape[2], key))
        return data[:, :, key]


class MultiRetSelectorXS(MultiRetSelector, CrossSectionFactor):
    pass


class RawShiftFactor(CustomFactor):
    """
    Shift the input data directly, note that this method will not process adjustment.
    """
    def __init__(self, data, periods=1):
        super().__init__(inputs=[data])
        self.periods = periods

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if data.dtype in {torch.bool, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError('RawShiftFactor does not support `int` or `bool` type, '
                             'please convert to float by using `factor.float()`, upstreams: {}'
                             .format(self.inputs))

        shift = data.roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0:self.periods] = np.nan
        elif self.periods < 0:
            shift[:, self.periods:] = np.nan
        else:
            raise ValueError("Cannot shift with 0")
        return shift


class AbsFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.abs()


class SumFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.nansum()


class UniqueTSSumFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        def _uni_sum(x):
            s = x.roll(1, dims=2)
            s[:, :, 0:1] = np.nan
            u = x.masked_fill(x == s, np.nan)
            return nansum(u, dim=2)
        return data.agg(_uni_sum)


class XSSumFactor(CrossSectionFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.nansum(dim=-1).unsqueeze(-1).expand(data.shape[0], data.shape[1])


class ProdFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.nanprod()


class LogFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.log()


class SignFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.sign()


class RoundFactor(CustomFactor):
    def compute(self, data: torch.Tensor, decimal) -> torch.Tensor:
        digits = 10 ** decimal
        return (data * digits).round() / digits


class TanhFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return torch.tanh(data)


class NonNaNCountFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        def _nan_count(_data):
            return (~torch.isnan(_data)).to(Global.float_type).sum(dim=2)
        return data.agg(_nan_count)


class TypeCastFactor(CustomFactor):
    _min_win = 1
    dtype = None

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.type(self.dtype)


class PadFactor(CustomFactor):
    inf = False
    nan = True

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return pad_2d(data, including_inf=self.inf, including_nan=self.nan)


class FillNANFactor(CustomFactor):
    inf = False
    nan = True

    def compute(self, data: torch.Tensor, value) -> torch.Tensor:
        mask = None
        if self.nan:
            mask = torch.isnan(data)
        if self.inf:
            mask = (mask | torch.isinf(data)) if mask is not None else torch.isinf(data)
        return data.masked_fill(mask, value)


class MaskedFillFactor(CustomFactor):
    def compute(self, data, mask, fill) -> torch.Tensor:
        if isinstance(fill, (int, float, bool)):
            return data.masked_fill(mask, fill)
        if data.dtype != fill.dtype:
            fill = fill.type(data.dtype)
        return data.masked_scatter(mask, fill.masked_select(mask))


class DoNothingFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data


PlaceHolderFactor = DoNothingFactor


class RankFactor(CrossSectionFactor):
    ascending = True
    normalize = False
    method = 'average'

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        ret = rankdata(data, 1, ascending=self.ascending, method=self.method)
        if self.normalize:
            ret /= nanmax(ret, dim=1).unsqueeze(-1)
        return ret


class RollingRankFactor(CustomFactor):
    """ rank over rolling period, scaled to the interval [1/days, 1] """
    ascending = True,
    _min_win = 2

    def compute(self, data) -> torch.Tensor:
        def _rolling_rank(_data):
            ret = rankdata(_data, 2, ascending=self.ascending, method='average', normalize=True)
            return ret[:, :, -1]

        return data.agg(_rolling_rank)


class DemeanFactor(CrossSectionFactor):
    group_dict = None

    def compute(self, data: torch.Tensor, beta) -> torch.Tensor:
        if self.group_dict is not None:
            s = self._revert_to_series(data)
            d = dict.fromkeys(s.index.levels[1], -1)
            d.update(self.group_dict)
            g = s.groupby([d, 'date'], level=1)
            ret = g.transform(lambda x: x - x.mean())
            print('Deprecated, group_dict will be removed in the future')
            return self._regroup(ret)
        else:
            # Why double?
            # Assuming that all values are same, after demean should be zero,
            # and then rank it, results should be the same value.
            # But if you demean by sector, and due to the float precision,
            # demean results will slightly different, so you got a random ranking results.
            double = data.to(torch.float64)
            if beta is not None:
                mean = nanmean(double).unsqueeze(-1)
                double -= beta * mean
            else:
                double -= nanmean(double).unsqueeze(-1)
            return double.to(data.dtype)


class MeanFactor(CrossSectionFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return nanmean(data).unsqueeze(-1).expand(data.shape[0], data.shape[1])


class MedianFactor(CrossSectionFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        universe_mask = self._get_computed_mask()
        half_universe = 0.5 * ((universe_mask & ~torch.isnan(data)).sum(dim=1) - 1)
        median_ks_odd = half_universe.long().unsqueeze(-1)
        median_ks_even = (half_universe + 0.6).long().unsqueeze(-1)

        median1, sorted_data = masked_kth_value_1d(
            data, universe_mask, median_ks_odd, treat_nan_as=np.nan)
        median2 = sorted_data.gather(1, median_ks_even)
        median = (median1 + median2) / 2

        return median.expand(data.shape[0], data.shape[1])


class XSMax(CrossSectionFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return nanmax(data).unsqueeze(-1).expand(data.shape[0], data.shape[1])


class XSMin(CrossSectionFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return nanmin(data).unsqueeze(-1).expand(data.shape[0], data.shape[1])


class ZScoreFactor(CrossSectionFactor):

    def __init__(self, data, mask=None, groupby: str = 'date', weight=None):
        super().__init__(inputs=[data, weight], mask=mask)
        self.groupby = groupby

    def compute(self, data: torch.Tensor, weight=None) -> torch.Tensor:
        if weight is None:
            mean = nanmean(data).unsqueeze(-1)
        elif weight == 0:
            mean = 0
        else:
            mean = nansum(data * weight) / nansum(weight)
            mean = mean.to(Global.float_type).unsqueeze(-1)
        return (data - mean) / nanstd(data).unsqueeze(-1)


class RollingZScoreFactor(CustomFactor):
    """ ZScore over rolling period """
    _min_win = 2

    def compute(self, data) -> torch.Tensor:
        def _rolling_zscore(_data):
            mean = nanmean(_data, dim=2).unsqueeze(-1)
            ret = (_data - mean) / nanstd(_data, dim=2).unsqueeze(-1)
            return ret[:, :, -1]

        return data.agg(_rolling_zscore)


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
    """
    Median Absolute Deviation Clamping
    median of the absolute deviations from the data's median
    """
    z = 5
    treat_nan_as = 0  # or inf
    m_is_mean = False  # switch to mean abs deviation

    def compute(self, data, fill):
        ret = data.clone()

        universe_mask = self._get_computed_mask()
        if self.m_is_mean:
            center = nanmean(data).unsqueeze(-1)
            diff = (data - center).abs()
            mad = nanmean(diff).unsqueeze(-1)
        else:
            half_universe = 0.5 * (universe_mask.sum(dim=1) - 1)
            median_ks_odd = half_universe.long().unsqueeze(-1)
            median_ks_even = (half_universe + 0.6).long().unsqueeze(-1)

            median1, sorted_data = masked_kth_value_1d(
                data, universe_mask, median_ks_odd, treat_nan_as=self.treat_nan_as)
            median2 = sorted_data.gather(1, median_ks_even)
            center = (median1 + median2) / 2

            diff = (data - center).abs()
            # sort mad
            sorted_data, _ = diff.sort(dim=1)
            mad1 = sorted_data.gather(1, median_ks_odd)
            mad2 = sorted_data.gather(1, median_ks_even)
            mad = (mad1 + mad2) / 2

        upper = center + self.z * mad
        lower = center - self.z * mad
        if fill is None:
            clamp_1d_(ret, lower, upper)
        else:
            ret.masked_fill_((ret <= lower) | (ret >= upper), fill)
        return ret


class WinsorizingFactor(CustomFactor):
    z = 0.05
    treat_nan_as = 0  # or inf

    def compute(self, data):
        universe_mask = self._get_computed_mask()
        n = universe_mask.sum(dim=1)
        upper_k = n - (self.z * n).long() - 1
        lower_k = (self.z * n).long()

        upper_k.clamp_(0, data.shape[1]).unsqueeze_(-1)
        lower_k.clamp_(0, data.shape[1]).unsqueeze_(-1)

        upper, sorted_data = masked_kth_value_1d(
            data, universe_mask, upper_k, treat_nan_as=self.treat_nan_as)
        lower = sorted_data.gather(1, lower_k)

        ret = data.clone()
        clamp_1d_(ret, lower, upper)
        return ret


# --------------- op factors ---------------


class SubFactor(CustomFactor):
    def compute(self, left, right) -> torch.Tensor:
        if type(right) is torch.Tensor and right.dtype is torch.bool:
            return left - right.float()
        else:
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
