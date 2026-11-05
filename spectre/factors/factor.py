"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import builtins
import warnings
from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..config import Global
from ..parallel import (
    Rolling,
    clamp_1d_,
    masked_kth_value_1d,
    nanaverage,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
    nansum,
    pad_2d,
    quantile,
    rankdata,
)
from ..parallel.algorithmic import nanprod
from ..plotting import plot_factor_diagram

if TYPE_CHECKING:
    from .engine import FactorEngine


class BaseFactor:
    """Basic factor class, only helper methods"""

    groupby = "asset"  # indicates inputs and return value of this factor are grouped by what
    _engine = None
    _clean_required = None  # If your factor caches data, this should be set to True when caching

    # --------------- overload ops ---------------

    @staticmethod
    def _numeric_inputs(*inputs):
        """In arithmetic a filter is 0 or 1, on a bool tensor True + True would be True."""
        return tuple(i._numeric() if isinstance(i, BaseFactor) else i for i in inputs)

    # op: +
    def __add__(self, other):
        return AddFactor(inputs=self._numeric_inputs(self, other))

    def __radd__(self, other):
        return AddFactor(inputs=self._numeric_inputs(other, self))

    # op: -
    def __sub__(self, other):
        return SubFactor(inputs=self._numeric_inputs(self, other))

    def __rsub__(self, other):
        return SubFactor(inputs=self._numeric_inputs(other, self))

    # op: *
    def __mul__(self, other):
        return MulFactor(inputs=self._numeric_inputs(self, other))

    def __rmul__(self, other):
        return MulFactor(inputs=self._numeric_inputs(other, self))

    # op: /
    def __truediv__(self, other):
        return DivFactor(inputs=self._numeric_inputs(self, other))

    def __rtruediv__(self, other):
        return DivFactor(inputs=self._numeric_inputs(other, self))

    # op: %
    def __mod__(self, other):
        return ModFactor(inputs=self._numeric_inputs(self, other))

    # op: **
    def __pow__(self, other):
        return PowFactor(inputs=self._numeric_inputs(self, other))

    def __rpow__(self, other):
        return PowFactor(inputs=self._numeric_inputs(other, self))

    # op: negative
    def __neg__(self):
        return NegFactor(inputs=self._numeric_inputs(self))

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

    @staticmethod
    def _rank_of_fraction(rank_data, n):
        """
        The rank where the fraction `n` of each date ends. Ranks are whole, so it floors: 50% of 1
        asset is no asset. A product that should be whole can round to just under it in any float
        type, 100 * 0.29 is 28.999999999999996 in float64, so it goes up by a few rounding errors.
        """
        rounding = 2 * float(np.finfo(type(n)).eps)
        return XSMax(inputs=[rank_data]).double() * (float(n) * (1 + rounding))

    def top(self, n: float, mask: BaseFactor = None):
        """Cross-section top values"""
        rank_data = self.rank(ascending=False, mask=mask, method="ordinal")
        if isinstance(n, (float, np.floating)):
            assert 0 < n < 1
            return rank_data <= self._rank_of_fraction(rank_data, n)
        else:
            return rank_data <= n

    def bottom(self, n: float, mask: BaseFactor = None):
        rank_data = self.rank(ascending=True, mask=mask, method="ordinal")
        if isinstance(n, (float, np.floating)):
            assert 0 < n < 1
            return rank_data <= self._rank_of_fraction(rank_data, n)
        else:
            return rank_data <= n

    def rank(self, ascending=True, mask: BaseFactor = None, normalize=False, method="average"):
        """Cross-section rank"""
        factor = RankFactor(inputs=(self._numeric(),), mask=mask)
        factor.method = method
        factor.ascending = ascending
        factor.normalize = normalize
        return factor

    def zscore(self, groupby: str = "date", mask: BaseFactor = None, weight: BaseFactor = None):
        """Cross-section zscore"""
        factor = ZScoreFactor(data=self._numeric(), mask=mask, groupby=groupby, weight=weight)
        return factor

    def demean(
        self,
        groupby: str | dict | None = None,
        mask: BaseFactor = None,
        beta: BaseFactor = None,
    ):
        """
        Cross-section demean.
        Set `groupby` to the name of a column, like 'sector', to demean each sector on each date.
        'date' is added to it, unless it's there already, or `groupby` is 'asset', which demeans
        each asset over all loaded dates, including the future ones.
        `groupby` also can be a dictionary like groupby={'name': group_id}, `group_id` must > 0.
        Group by dict is implemented by pandas, not in GPU.
        """
        factor = DemeanFactor(inputs=(self._numeric(), beta), mask=mask)
        if isinstance(groupby, str):
            if groupby != "asset" and "date" not in groupby.split(","):
                groupby = "date," + groupby
            factor.groupby = groupby
        elif isinstance(groupby, dict):
            factor.group_dict = groupby
        elif groupby is not None:
            raise ValueError()
        return factor

    def demedian(self, groupby: str = "date", mask: BaseFactor = None):
        factor = DemedianFactor(inputs=(self._numeric(),), mask=mask)
        factor.groupby = groupby
        return factor

    def mean(
        self,
        groupby: str = "date",
        mask: BaseFactor | str = None,
        weight: BaseFactor = None,
        weight_sum: builtins.float | None = None,
    ):
        """
        Cross-section mean.
        """
        factor = MeanFactor(inputs=(self._numeric(), weight, weight_sum), mask=mask)
        factor.groupby = groupby
        return factor

    def median(self, mask: BaseFactor = None, groupby: str = "date"):
        factor = MedianFactor(inputs=(self._numeric(),), mask=mask)
        factor.groupby = groupby
        return factor

    def max(self, mask: BaseFactor = None, groupby: str = "date"):
        factor = XSMax(inputs=(self._numeric(),), mask=mask)
        factor.groupby = groupby
        return factor

    def min(self, mask: BaseFactor = None, groupby: str = "date"):
        factor = XSMin(inputs=(self._numeric(),), mask=mask)
        factor.groupby = groupby
        return factor

    def ts_max(self, win=2):
        from .statistical import RollingHigh

        return RollingHigh(win=win, inputs=(self._numeric(),))

    def ts_min(self, win=2):
        from .statistical import RollingLow

        return RollingLow(win=win, inputs=(self._numeric(),))

    def quantile(self, bins=5, mask: BaseFactor = None, groupby: str = "date"):
        """Cross-section quantiles to which the factor belongs"""
        if not mask:
            warnings.warn(
                "quantile() mask parameter not set, normally set mask=universe, is it expected?",
                RuntimeWarning,
            )
        factor = QuantileClassifier(inputs=(self._numeric(),))
        factor.set_mask(mask)
        factor.groupby = groupby
        factor.bins = bins
        return factor

    def std(self, groupby: str = "date", mask: BaseFactor = None):
        from .statistical import XSStandardDeviation

        factor = XSStandardDeviation(inputs=(self._numeric(),), mask=mask)
        factor.groupby = groupby
        return factor

    def to_weight(self, demean=True, mask: BaseFactor = None):
        """factor value to portfolio weight, which sum(abs(weight)) == 1"""
        factor = ToWeightFactor(inputs=(self._numeric(),), mask=mask)
        factor.demean_weight = demean
        return factor

    def shift(self, periods=1):
        if periods == 0:
            return self
        return RawShiftFactor(self, periods)

    def abs(self):
        return AbsFactor(inputs=(self._numeric(),))

    def log(self):
        return LogFactor(inputs=(self,))

    def sign(self):
        return SignFactor(inputs=(self._numeric(),))

    def ts_sum(self, win: int, mask: BaseFactor = None):
        ret = SumFactor(win, inputs=(self._numeric(),))
        ret.set_mask(mask)
        return ret

    def xs_sum(self, mask: BaseFactor = None, groupby="date"):
        # todo future name will change to sum
        ret = XSSumFactor(inputs=(self._numeric(),), mask=mask)
        ret.groupby = groupby
        return ret

    def ts_prod(self, win: int):
        return ProdFactor(win, inputs=(self._numeric(),))

    def round(self, decimal):
        return RoundFactor(inputs=(self, decimal))

    def tanh(self):
        return TanhFactor(inputs=(self,))

    def filter(self, mask: BaseFactor):
        """Local filter, fill unmasked elements with NaNs."""
        mf = DoNothingFactor(inputs=(self,))
        mf.set_mask(mask)
        return mf

    def one_hot(self):
        from .filter import OneHotEncoder

        factor = OneHotEncoder(self._numeric())
        return factor

    def fill_na(
        self, value: builtins.float | None = None, ffill: bool | None = None, inf=False, nan=True
    ):
        if value is not None:
            factor = FillNANFactor(inputs=(self, value))
        elif ffill:
            factor = PadFactor(inputs=(self,))
        else:
            raise ValueError("Either `value=number` or `ffill=True` must be specified.")
        factor.inf = inf
        factor.nan = nan
        return factor

    fill_nan = fill_na

    def masked_fill(self, mask: BaseFactor, fill: BaseFactor | builtins.float | bool):
        return MaskedFillFactor(inputs=(self, mask, fill))

    def ts_any(self, win: int):
        """Return True if Rolling window contains any Non-NaNs"""
        from .filter import AnyNonNaNFactor

        return AnyNonNaNFactor(win, inputs=(self,))

    def ts_all(self, win: int):
        """Return True if Rolling window all are Non-NaNs"""
        from .filter import AllNonNaNFactor

        return AllNonNaNFactor(win, inputs=(self,))

    def ts_count(self, win: int):
        """Return Non-NaN data count"""
        return NonNaNCountFactor(win, inputs=(self._numeric(),))

    def clamp(self, left: float, right: float, fill=None):
        return ClampFactor(self._numeric(), left, right, fill)

    def mad_clamp(self, z: float, mask: BaseFactor = None, groupby="date", fill=None, mean=False):
        """Cross-section MAD clamp"""
        factor = MADClampFactor(inputs=(self._numeric(), fill))
        factor.groupby = groupby
        factor.z = z
        factor.set_mask(mask)
        factor.m_is_mean = mean
        return factor

    def winsorizing(self, z: float = 0.05, mask: BaseFactor = None):
        """Cross-section winsorizing clamp"""
        factor = WinsorizingFactor(inputs=(self._numeric(),))
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

    def _numeric(self):
        """This factor as the input of a helper that calculates with numbers."""
        return self

    def _regroup_by_other(self, factor, factor_out):
        if factor.groupby != self.groupby:
            ret = self._engine.revert_(factor_out, factor.groupby, type(factor).__name__)
            return self._regroup(ret)
        else:
            return factor_out

    def _regroup(self, data):
        return self._engine.group_by_(data, self.groupby)

    def _revert_to_series(self, data):
        data = data.to(self._engine.device)  # it's on cpu after running out of CUDA memory
        return self._engine.revert_to_series_(data, self.groupby, type(self).__name__)

    def get_total_backwards_(self) -> int:
        raise NotImplementedError("abstractmethod")

    def should_delay(self) -> bool:
        """Is this factor should be delayed?"""
        return False

    # quoted, because engine.py imports this module and inspect.signature() evaluates annotations
    def pre_compute_(self, engine: "FactorEngine", start, end) -> None:  # noqa: UP037
        self._engine = engine
        engine.column_to_parallel_groupby_(self.groupby)

    def clean_up_(self, force=False) -> None:
        self._engine = None

    def compute_(self, stream: torch.cuda.Stream | None) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")

    def __init__(self) -> None:
        pass

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("abstractmethod")


class CustomFactor(BaseFactor):
    """Common base class for factor, that can contain child factors and handle
    hierarchical relationships"""

    # settable member variables
    win = 1  # determine include how many previous data
    inputs = None  # any values in `inputs` list will pass to `compute` function by order
    _min_win = None  # assert when `win` less than `_min_win`, prevent user error.
    _rolling_fill = np.nan  # value of the rolling window before the first data
    # Set if `compute` reads across the slots of a group, like a mean of each date. Then the padding
    # slots of an input in the same group are cleared. Not by default, because `shift(-n)` needs
    # them: they hold the returns up to the last close of a delisted asset. It also clears the
    # dates that don't have all their assets, see `FactorEngine.incomplete_rows_`.
    _reads_across_group = False

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
    _computing_on_cpu = False

    def __init__(self, win: int | None = None, inputs: Sequence[BaseFactor] | None = None):
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
        assert isinstance(self.inputs, (list, tuple, type(None))), "`factor.inputs` must be a list."
        assert isinstance(self.win, int), "`factor.win` must be a integer."

        if not (self.win >= (self._min_win or 1)):
            raise ValueError(
                f"factor({self!s}) win ({self.win}) must  >= [_min_win({self._min_win}) and 1]"
            )

        if self.inputs:
            self.inputs = [ipt() if callable(ipt) else ipt for ipt in self.inputs]
            for ipt in self.inputs:
                if isinstance(ipt, BaseFactor) and ipt._clean_required is not None:
                    self._clean_required = ipt._clean_required
                    break

    @classmethod
    def sequential(cls, *args):
        """Helper method for fast initialization"""
        if isinstance(args[0], int):
            win = args[0]
            if cls._min_win is not None:
                win = max(args[0], cls._min_win)
            return cls(win, inputs=[*args[1:]])
        else:
            return cls(inputs=[*args[0:]])

    def set_mask(self, mask: BaseFactor | str = None):
        """Input data mask, fill all unmasked **INPUT** data to NaN"""
        self._mask = mask
        self._total_backwards = None
        if isinstance(mask, BaseFactor) and mask._clean_required is not None:
            self._clean_required = max(self._clean_required or False, mask._clean_required)
        return self

    def get_mask(self):
        return self._mask

    # What mask='universe' takes from the filter. The engine decides the delay and the history of
    # a run before `pre_compute_` binds it to the factors, so it leaves them here while it does.
    _universe_delay = False
    _universe_backwards = 0
    # Counted by the engine. The memo of `get_total_backwards_` saves walking a large graph many
    # times in one run. It can't last longer, the graph can change after a run in ways that a
    # factor can't tell the ones downstream of it, like a `set_mask` on an input.
    _run_number = 0

    def _get_mask_factor(self):
        # todo add a placeholder factor for universe
        if type(self._mask) is str and (self._mask == "universe") is True:
            return self._engine.get_filter()
        else:
            return self._mask

    def _get_computed_mask(self):
        """Allow subclass get the current mask result in compute()"""
        if self._mask_out is None:
            mask = ~self._engine.get_group_padding_mask(self.groupby)
        else:
            mask = self._regroup_by_other(self._get_mask_factor(), self._mask_out)
        return mask.cpu() if self._computing_on_cpu else mask

    def get_total_backwards_(self) -> int:
        # mask='universe' anywhere upstream makes the result depend on the filter of the run
        universe = self._universe_backwards
        if (
            self._total_backwards is not None
            and self._memo_universe == universe
            and self._memo_run == self._run_number
        ):
            return self._total_backwards

        from .filter import FilterRawShiftFactor

        backwards = 0
        if self.inputs:
            backwards = max(
                [up.get_total_backwards_() for up in self.inputs if isinstance(up, BaseFactor)]
                or (0,)
            )

        # the mask applies to the inputs, so it is rolled and shifted with them
        if isinstance(self._mask, BaseFactor):
            backwards = max(self._mask.get_total_backwards_(), backwards)
        elif self._mask is not None:
            backwards = max(universe, backwards)

        backwards = backwards + self.win - 1
        if isinstance(self, (RawShiftFactor, FilterRawShiftFactor)):
            backwards += max(self.periods, 0)
        self._total_backwards = backwards
        self._memo_universe = universe
        self._memo_run = self._run_number
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
        elif self._mask is not None:
            ret = max(ret, self._universe_delay)
        return ret

    def set_delay(self, delay):
        """None: Auto delay. False: Force not to delay. True: Force to delay."""
        self._force_delay = delay

    def is_keep_cache(self) -> bool:
        if self._keep_cache:
            return True

        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, CustomFactor) and upstream.is_keep_cache():
                    return True

        return isinstance(self._mask, CustomFactor) and self._mask.is_keep_cache()

    def show_graph(self):
        plot_factor_diagram(self)

    def nodes_count(self):
        ret = 1

        def cum(_):
            nonlocal ret
            ret += 1

        self.iter(cum)

        return ret

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

    def iter(self, func):
        func(self)

        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    func(upstream)
                if isinstance(upstream, CustomFactor):
                    upstream.iter(func)

        if self._engine is not None:
            mask = self._get_mask_factor()
        else:
            mask = self._mask
        if mask is not None:
            if isinstance(mask, BaseFactor):
                func(mask)
            if isinstance(mask, CustomFactor):
                mask.iter(func)

    def pre_compute_(self, engine, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        # TODO If the same factor added to 2 different engines, the _keep_cache should not able to
        #  be enabled. alert or rise.
        if engine.cache_hash != self._cache_hash and self._cache is not None:
            print(
                f"The factor({self}) data cached for the original engine is invalidated "
                "due to engine changes."
            )
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

    @staticmethod
    def _clear_slots(data: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if data.ndim > slots.ndim:  # multiple outputs
            slots = slots.unsqueeze(-1)
        return data.masked_fill(slots, False if data.dtype == torch.bool else np.nan)

    def _format_input(self, upstream, upstream_out, mask_factor, mask_out):
        # If input.groupby not equal self.groupby, convert it
        ret = self._regroup_by_other(upstream, upstream_out)

        if mask_out is not None:
            mask = self._regroup_by_other(mask_factor, mask_out)
            if ret.dtype == torch.bool:
                ret = ret.masked_fill(~mask, False)
            else:
                ret = ret.masked_fill(~mask, np.nan)

        if self._reads_across_group:
            # only an input in the same group, a regroup has cleared the padding of the others
            if upstream.groupby == self.groupby:
                ret = self._clear_slots(ret, self._engine.get_group_padding_mask(self.groupby))
            incomplete = self._engine.incomplete_rows_(self.groupby, upstream, mask_factor)
            if incomplete is not None:
                ret = self._clear_slots(ret, incomplete)

        # if need rolling and adjustment
        if self.win > 1:
            if len(ret.shape) >= 3:
                raise ValueError(
                    f"upstream factor `{upstream!s}` has multiple outputs ({ret.shape[2]}), "
                    "rolling win > 1 only supports one output, "
                    "use slice to select a value before using it, for example: "
                    "`factor[0]`."
                )
            # todo: If the indirect children is a column factor, also need to adjust.
            #  for now, only adjust direct children.
            #  refactor: if any shift occurs, adjust them.
            ret = Rolling(ret, self.win, upstream.adjustments, self._rolling_fill)
        return ret

    def compute_(self, down_stream: torch.cuda.Stream | None) -> torch.Tensor:
        # return cached result
        self._ref_count -= 1
        if self._ref_count < 0:
            raise ValueError(
                "Reference count error: possible circular reference! "
                'Do not set_mask="universe" in global filter.'
                "Or maybe you override `pre_compute_`, "
                "but did not call super() method."
            )
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
                if "CUDA out of memory" in repr(e):
                    out = self._compute_on_cpu(inputs)
                else:
                    raise

        self._mask_out = None
        if self._ref_count > 0:
            self._cache = out
        return out

    def _compute_on_cpu(self, inputs) -> torch.Tensor:
        self._computing_on_cpu = True
        try:
            out = self.compute(
                *[t.cpu() if isinstance(t, (torch.Tensor, Rolling)) else t for t in inputs]
            )
        finally:
            self._computing_on_cpu = False
        return out.to(self._engine.device)

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Abstractmethod, do the actual factor calculation here.
        Unlike zipline, here we will receive all data, you need to be careful to prevent
        Look-Ahead Bias.

        `inputs` Data structure:
        The data structure is designed for optimal performance in parallel.
        * Groupby `asset`(default):
            set T = data bar
            set N = individual asset bar count - 1,  Max = Max bar count across all assets - 1
            win = 1:
                | asset idx| price(T+0) | ... | price(T+N) | price(T+N+1) | ... | price(T+Max) |
                |----------|------------|-----|------------|--------------|-----|--------------|
                |     0    | 123.45     | ... | 234.56     | NaN          | ... | Nan          |
                It's 2D matrix like data[asset idx, T idx], Bars after T+N will fill with NANs.
                If `align_by_time=False` then the time represented by each bar is different.
            win > 1:
                Gives you a rolling object `r`, you can call `r.mean()`, `r.sum()`, or `r.agg()`.
                `r.agg(callback)` gives you raw data structure as:
                | asset idx;time idx |price(t-win+1)| ... | price(t+0) |
                |----------;---------|--------------|-----|--------------|
                |     0    ; T+0     | NaN          | ... | 123.45       |
                |          ; ...     | ...          | ... | ...          |
                |          ; T+N     | xxx.xx       | ... | 234.56       |
                |     1    ; T+0     | NaN          | ... | 123.45       |
                |          ; ...     | ...          | ... | ...          |
                |          ; T+N     | xxx.xx       | ... | 234.56       |
                It's 3D matrix like: data[asset idx, T idx, win idx]
                If this table too big, it will split to multiple tables by axis 0, and call the
                callback function separately.
        * Groupby `date` or Cross-Section Factors (set `groupby = 'date'`):
            set A = asset
            set N = asset count - 1, Max = Max asset count in all time - 1
                | time idx |price(A+0)| ... |price(A+N)|price(A+N+1)| ... | price(A+Max) |
                |----------|----------|-----|----------|------------|-----|--------------|
                |     T    | 100.00   | ... | 200.00   | NaN        | ... | Nan          |
                |    ...   | ...      | ... | ...      | ...        | ... | ...          |
                |   T+Max  | 123.45   | ... | 234.56   | NaN        | ... | Nan          |
                This is the price of all assets at same time T, useful for calculations such as
                every day market means.
                If `align_by_time=False` then the order of assets in each row (time) is not fixed,
                in that way the column cannot be considered as a particular asset.
        * Custom:
            Use `series = self._revert_to_series(input)` you can get `pd.Series` data type, and
            manipulate by your own. Remember to call `return self._regroup(series)` when returning.
            WARNING: This CPU method will be very inefficient and break parallel.
        :param inputs: All input factors data, including all data from `start(minus win)` to `end`.
        :return: your factor values, length should be the same as the `inputs`
        """
        raise NotImplementedError("abstractmethod")


class CrossSectionFactor(CustomFactor, ABC):
    """Class that inputs and return value is grouped by datetime"""

    groupby = "date"
    win = 1
    _reads_across_group = True

    def __init__(
        self,
        win: int | None = None,
        inputs: Sequence[BaseFactor] | None = None,
        mask: BaseFactor | None = None,
    ):
        super().__init__(win, inputs)
        self.set_mask(mask)
        assert self.win == 1, "CrossSectionFactor.win can only be 1"

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
            raise KeyError(
                f"This factor({type(self.inputs[0]).__name__}) has only one return value, "
                "cannot slice."
            )
        elif data.shape[2] <= key:
            raise KeyError(
                f"OutOfBounds: factor has only {data.shape[2]} return values, and slice is [{key}]."
            )
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
            raise ValueError(
                "RawShiftFactor does not support `int` or `bool` type, "
                f"please convert to float by using `factor.float()`, upstreams: {self.inputs}"
            )

        shift = data.roll(self.periods, dims=1)
        if self.periods > 0:
            shift[:, 0 : self.periods] = np.nan
        elif self.periods < 0:
            shift[:, self.periods :] = np.nan
        else:
            raise ValueError("Cannot shift with 0")
        return shift


class AbsFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.abs()


def _nan_if_empty(kernel, data: torch.Tensor, dim: int) -> torch.Tensor:
    """`nansum` and `nanprod` give 0 and 1 on a group with no valid value, a factor gives NaN."""
    return kernel(data, dim=dim).masked_fill_(torch.isnan(data).all(dim=dim), np.nan)


class SumFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.agg(lambda x: _nan_if_empty(nansum, x, dim=2))


class UniqueTSSumFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        def _uni_sum(x):
            s = x.roll(1, dims=2)
            s[:, :, 0:1] = np.nan
            u = x.masked_fill(x == s, np.nan)
            return _nan_if_empty(nansum, u, dim=2)

        return data.agg(_uni_sum)


class XSSumFactor(CrossSectionFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        total = _nan_if_empty(nansum, data, dim=1).unsqueeze(-1)
        return total.expand(data.shape[0], data.shape[1])


class ProdFactor(CustomFactor):
    _min_win = 2

    def compute(self, data: Rolling) -> torch.Tensor:
        return data.agg(lambda x: _nan_if_empty(nanprod, x, dim=2))


class LogFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.log()


class SignFactor(CustomFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return data.sign().masked_fill(torch.isnan(data), np.nan)


class RoundFactor(CustomFactor):
    def compute(self, data: torch.Tensor, decimal) -> torch.Tensor:
        digits = 10**decimal
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
        if isinstance(value, (int, float, bool)):
            return data.masked_fill(mask, value)
        elif data.dtype != value.dtype:
            value = value.type(data.dtype)
        return data.masked_scatter(mask, value.masked_select(mask))


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
    method = "average"

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        ret = rankdata(data, 1, ascending=self.ascending, method=self.method)
        if self.normalize:
            ret /= nanmax(ret, dim=1).unsqueeze(-1)
        return ret


class RollingRankFactor(CustomFactor):
    """rank over rolling period, scaled to the interval [1/days, 1]"""

    ascending = (True,)
    _min_win = 2

    def compute(self, data) -> torch.Tensor:
        def _rolling_rank(_data):
            ret = rankdata(_data, 2, ascending=self.ascending, method="average", normalize=True)
            return ret[:, :, -1]

        return data.agg(_rolling_rank)


class DemeanFactor(CrossSectionFactor):
    group_dict = None

    def compute(self, data: torch.Tensor, beta) -> torch.Tensor:
        if self.group_dict is not None:
            s = self._revert_to_series(data)
            d = dict.fromkeys(s.index.levels[1], -1)
            d.update(self.group_dict)
            g = s.groupby([d, "date"], level=1, observed=False)
            ret = g.transform(lambda x: x - x.mean())
            print("Spectre: Deprecated, group_dict will be removed in the future")
            return self._regroup(ret)
        else:
            # Why double?
            # Assuming that all values are same, after demean should be zero,
            # and then rank it, results should be the same value.
            # But if you demean by sector, and due to the float precision,
            # demean results will slightly different, so you got a random ranking results.
            double = data.to(torch.float64)
            mean = nanmean(double).unsqueeze(-1)
            if beta is not None:
                mean = beta * mean
            return (double - mean).to(data.dtype)


class DemedianFactor(CrossSectionFactor):
    nan_policy = "omit"

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        universe_mask = self._get_computed_mask()
        median, _ = masked_kth_value_1d(data, universe_mask, [0.5], nan_policy=self.nan_policy)

        return data - median[0].expand(data.shape[0], data.shape[1])


class MeanFactor(CrossSectionFactor):
    def compute(self, data: torch.Tensor, weight=None, weight_sum=None) -> torch.Tensor:
        if weight is None:
            return nanmean(data).unsqueeze(-1).expand(data.shape[0], data.shape[1])
        else:
            mean = nanaverage(data, weight, weight_sum=weight_sum).to(Global.float_type)
            return mean.unsqueeze(-1).expand(data.shape[0], data.shape[1])


class MedianFactor(CrossSectionFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        universe_mask = self._get_computed_mask()
        median, _ = masked_kth_value_1d(data, universe_mask, [0.5], nan_policy="omit")
        return median[0].expand(data.shape[0], data.shape[1])


class XSMax(CrossSectionFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return nanmax(data).unsqueeze(-1).expand(data.shape[0], data.shape[1])


class XSMin(CrossSectionFactor):
    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return nanmin(data).unsqueeze(-1).expand(data.shape[0], data.shape[1])


class ZScoreFactor(CrossSectionFactor):
    def __init__(self, data, mask=None, groupby: str = "date", weight=None):
        super().__init__(inputs=[data, weight], mask=mask)
        self.groupby = groupby

    def compute(self, data: torch.Tensor, weight=None) -> torch.Tensor:
        if not isinstance(weight, torch.Tensor) and weight == 0:
            return data / nanstd(data).unsqueeze(-1)

        samples = data
        if weight is not None:
            # a row without a weight still gets a score, but it is not a sample of mean and std
            samples = data.masked_fill(torch.isnan(data * weight), np.nan)
        # from the lowest value, so rounding doesn't give equal values a spread
        lowest = nanmin(samples).unsqueeze(-1)
        samples = samples - lowest
        mean = nanmean(samples) if weight is None else nanaverage(samples, weight)
        std = nanstd(samples).unsqueeze(-1)
        std = std.masked_fill(std == 0, np.nan)  # a value off the mean would be inf
        return (data - lowest - mean.unsqueeze(-1)) / std


class RollingZScoreFactor(CustomFactor):
    """ZScore over rolling period"""

    _min_win = 2

    def compute(self, data) -> torch.Tensor:
        def _rolling_zscore(_data):
            # from the lowest value, so rounding doesn't give equal values a spread
            samples = _data - nanmin(_data, dim=2).unsqueeze(-1)
            std = nanstd(samples, dim=2)
            std = std.masked_fill(std == 0, np.nan)
            return (samples[:, :, -1] - nanmean(samples, dim=2)) / std

        return data.agg(_rolling_zscore)


class QuantileClassifier(CrossSectionFactor):
    """Returns the quantile of the factor at each datetime"""

    bins = 5

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        return quantile(data, self.bins, dim=1)


class ToWeightFactor(CrossSectionFactor):
    demean_weight = True

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        if self.demean_weight:
            # from the lowest value, so rounding doesn't give equal values a spread
            data = data - nanmin(data).unsqueeze(-1)
            data = data - nanmean(data).unsqueeze(-1)
        return data / nansum(data.abs(), dim=1).unsqueeze(-1)


class ClampFactor(CustomFactor):
    """Simple Winsorizing"""

    _min_win = 1

    def __init__(self, data, left, right, fill=None):
        super().__init__(win=1, inputs=[data, left, right, fill])

    def compute(self, data, left, right, fill):
        if fill is None:
            return data.clamp(left, right)
        else:
            return data.masked_fill((data < left) | (data > right), fill)


class MADClampFactor(CustomFactor):
    """
    Median Absolute Deviation Clamping
    median of the absolute deviations from the data's median
    """

    z = 5
    nan_policy = "omit"
    m_is_mean = False  # switch to mean abs deviation
    _reads_across_group = True

    def compute(self, data, fill):
        ret = data.clone()

        universe_mask = self._get_computed_mask()
        if self.m_is_mean:
            # from the lowest value, so the mean of equal values doesn't round away from them
            lowest = nanmin(data).unsqueeze(-1)
            center = lowest + nanmean(data - lowest).unsqueeze(-1)
            diff = (data - center).abs()
            mad = nanmean(diff).unsqueeze(-1)
        else:
            center, kth = masked_kth_value_1d(
                data, universe_mask, [0.5], nan_policy=self.nan_policy
            )
            center, kth = center[0], kth[0]
            diff = (data - center).abs()
            # sort mad
            sorted_data, _ = diff.sort(dim=1)
            mad1 = sorted_data.gather(1, kth[0])
            mad2 = sorted_data.gather(1, kth[1])
            mad = (mad1 + mad2) / 2

        upper = center + self.z * mad
        lower = center - self.z * mad
        if fill is None:
            clamp_1d_(ret, lower, upper)
        elif fill == "center":
            # masked_scatter_ takes source values in order, select the centers of masked first
            fill_mask = (ret <= lower) | (ret >= upper)
            ret.masked_scatter_(
                fill_mask, center.expand(data.shape[0], data.shape[1]).masked_select(fill_mask)
            )
        else:
            # if MAD is 0, both bounds are the center, which is not an outlier
            ret.masked_fill_(((ret <= lower) | (ret >= upper)) & (ret != center), fill)
        return ret


class WinsorizingFactor(CustomFactor):
    z = 0.05
    nan_policy = "omit"
    _reads_across_group = True

    def compute(self, data):
        universe_mask = self._get_computed_mask()
        k_values, _ = masked_kth_value_1d(
            data, universe_mask, [self.z, -self.z], even_mean=False, nan_policy=self.nan_policy
        )
        lower, upper = k_values

        ret = data.clone()
        clamp_1d_(ret, lower, upper)
        return ret


class IQRNormalityFactor(CrossSectionFactor):
    """
    0 = most likely be normal dist
    > 0: how many times away from the standard deviation
    """

    nan_policy = "omit"

    def compute(self, data):
        universe_mask = self._get_computed_mask()
        k_values, _ = masked_kth_value_1d(
            data, universe_mask, [-0.25, 0.25], even_mean=False, nan_policy=self.nan_policy
        )
        q3, q1 = k_values
        iqr = q3 - q1
        # from the lowest value, so rounding doesn't give equal values a spread
        std = nanstd(data - nanmin(data).unsqueeze(-1)).unsqueeze(-1)
        #  interquartile range of a normal random variable is 1.34898 times its standard deviation
        normality = ((iqr / std) - 1.34898).abs()

        return normality.expand(data.shape[0], data.shape[1])


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
        return left**right


class NegFactor(CustomFactor):
    def compute(self, left) -> torch.Tensor:
        return -left
