"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import uuid
import warnings
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch

from ..data import DataLoader
from ..parallel import DummyParallelGroupBy, ParallelGroupBy
from ..plotting import plot_chart, plot_quantile_and_cumulative_returns
from .datafactor import AdjustedColumnDataFactor, ColumnDataFactor
from .factor import BaseFactor, CustomFactor
from .filter import FilterFactor, StaticAssets


class OHLCV:
    open = ColumnDataFactor(inputs=("",), should_delay=False)
    high = ColumnDataFactor(inputs=("",))
    low = ColumnDataFactor(inputs=("",))
    close = ColumnDataFactor(inputs=("",))
    volume = ColumnDataFactor(inputs=("",))


class FactorEngine:
    """
    Engine for compute factors, used for back-testing and alpha-research both.
    """

    # friend private:

    @property
    def dataframe_(self):
        return self._dataframe

    @property
    def loader_(self):
        return self._loader

    def get_group_(self, group_name):
        return self._groups[group_name]

    @property
    def cache_hash(self):
        return self._cache_hash

    def column_to_tensor_(self, data_column) -> torch.Tensor:
        # cache data with column prevent double copying
        if data_column in self._column_cache:
            return self._column_cache[data_column]

        values = self._dataframe[data_column].to_numpy()
        if not values.flags.writeable:
            # Copy-on-Write of pandas 3 returns read-only array, which torch doesn't support
            values = values.copy()
        data = torch.from_numpy(values).to(self._device, non_blocking=True)
        self._column_cache[data_column] = data
        return data

    def _column_to_group_codes(self, column: str) -> np.ndarray:
        """Integer codes of a column of any type, in the order of its values."""
        series = self._dataframe[column]
        assert not series.isna().any(), column
        if column == self._loader.time_category:
            # already ids, and every run comes here
            codes = series.to_numpy().astype(np.int64)
            return codes - codes.min()
        if series.dtype.name == "category":
            return series.cat.codes.to_numpy().astype(np.int64)
        return pd.factorize(series, sort=True)[0]

    def column_to_parallel_groupby_(self, group_column: str, as_group_name=None):
        # todo refactor: group_column change to ClassifierFactor type
        if isinstance(group_column, torch.Tensor):
            return
        if as_group_name is None:
            as_group_name = group_column
        if as_group_name in self._groups:
            return

        cols = group_column.split(",")

        if len(self._dataframe.index.levels[1]) == 1 and (
            cols[0] == "date" or cols[0] == self._loader.time_category
        ):
            col = self._loader.time_category
            series = self._dataframe[col]
            self._groups[as_group_name] = DummyParallelGroupBy(series.shape, self._device)
            return

        keys = None
        for col in cols:
            if col:
                if col == "date":
                    col = self._loader.time_category
                codes = self._column_to_group_codes(col)
                if keys is None:
                    keys = codes
                else:
                    keys = keys * (codes.max() + 1) + codes
                    # there can't be more groups than rows, so this keeps the keys in int32
                    if keys.max() >= len(keys):
                        keys = pd.factorize(keys, sort=True)[0]
        keys = torch.tensor(keys, device=self._device, dtype=torch.int32)
        self._groups[as_group_name] = ParallelGroupBy(keys)

    def revert_(self, data: torch.Tensor, group: str, factor_name: str) -> torch.Tensor:
        if isinstance(group, torch.Tensor):
            g = ParallelGroupBy(group)
            return g.revert(data, factor_name)
        return self._groups[group].revert(data, factor_name)

    def revert_to_series_(self, data: torch.Tensor, group: str, factor_name: str) -> pd.Series:
        array = self.revert_(data, group, factor_name).cpu()
        return pd.Series(array.numpy(), index=self._dataframe.index)

    def group_by_(self, data: torch.Tensor | pd.Series, group: str) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            if isinstance(group, torch.Tensor):
                g = ParallelGroupBy(group)
                return g.split(data)
            else:
                return self._groups[group].split(data)
        elif isinstance(data, pd.Series):
            data = torch.tensor(data.values, device=self._device)
            return self._groups[group].split(data)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data, device=self._device)
            return self._groups[group].split(data)
        else:
            raise ValueError("Invalid data type, should be tensor or series.")

    def get_group_padding_mask(self, group: str) -> torch.Tensor:
        if isinstance(group, torch.Tensor):
            g = ParallelGroupBy(group)
            return g.padding_mask
        return self._groups[group].padding_mask

    def incomplete_rows_(self, group: str, *factors: BaseFactor | None) -> torch.Tensor | None:
        """
        A date needs all its assets to be read across. An asset with gaps gets older bars of its
        own from the loader, no other asset has rows on those dates. These `factors` also need
        their look-back: on the first complete dates only that asset has a value.
        :return: bool tensor in the shape of `group`, True on those rows. None if there are none.
        """
        if self._first_complete_date == 0 or not isinstance(group, str):
            return None
        columns = group.split(",")
        if "date" not in columns and self._loader.time_category not in columns:
            return None

        look_back = 0
        # cut by max_backwards, the look-back is short for all assets
        if not self._history_cut:
            with self._masks_know_the_filter():
                look_back = max(f.get_total_backwards_() for f in factors if f is not None)
        key = (group, look_back)
        if key not in self._incomplete_rows:
            dates = self._dataframe.index.codes[0]
            rows = torch.zeros(len(dates), dtype=torch.bool, device=self._device)
            rows[: dates.searchsorted(self._first_complete_date + look_back)] = True
            self._incomplete_rows[key] = self._groups[group].split(rows)
            if self._enable_stream and self._device.type == "cuda":
                # the other streams will read it without waiting for this one
                torch.cuda.current_stream().synchronize()
        return self._incomplete_rows[key]

    # private:

    def _prepare_tensor(self, start, end, backwards, delayed=False):
        # Check cache, just in case, if use some ML techniques, engine may be called repeatedly
        # with same date range.
        self._history_cut = backwards > self.max_backwards
        if self._history_cut:
            warnings.warn(
                f"engine.max_backwards is {self.max_backwards}, but {backwards} bars of historical "
                "data are required, so the first bars of the results will use partial windows.",
                RuntimeWarning,
            )
        backwards = min(backwards, self.max_backwards)
        if start == self._last_loaded[0] and end == self._last_loaded[1]:
            if backwards <= self._last_loaded[2]:
                return False
            else:
                print(
                    "Due to Factor Rolling window increase, extend the length of historical data "
                    "in GPU memory, "
                    f"{self._last_loaded[2]} -> {backwards}"
                )
        self._groups = dict()

        # Get data
        df = self._loader.load(start, end, backwards).copy()
        # If possible, pre-screen
        if isinstance(self._filter, StaticAssets):
            # not `.loc[]`, it raises KeyError if any asset is not found since pandas 2.0
            df = df[df.index.get_level_values(1).isin(list(self._filter.assets))]
            if df.shape[0] == 0:
                raise ValueError(
                    f"The assets {self._filter.assets} specified by StaticAssets filter, "
                    "was not found in DataLoader."
                )
        # check history data is insufficient
        dates_of_data = df.index.levels[0]
        df.index = df.index.remove_unused_levels()
        history_win = df.index.levels[0].get_indexer([start], method="bfill")[0]
        # a delayed run drops its first bar, which needs only `backwards - 1` of them
        required = backwards - 1 if delayed else backwards
        if history_win < required:
            warnings.warn(
                "Historical data seems insufficient. "
                f"{required} rows of historical data are required, "
                f"but only {history_win} rows are obtained. "
                "It is also possible that `calender_asset` of the loader is not set, "
                "some out of trading hours data will cause indexing problems.",
                RuntimeWarning,
            )
        # post processing data
        # on the last `backwards` dates all assets have their rows, see `incomplete_rows_`
        self._first_complete_date = max(history_win - backwards, 0)
        static = isinstance(self._filter, StaticAssets)
        if static and len(df.index.levels[1]) < len(self._filter.assets):
            # an asset was gone before those dates, nothing shows it had rows on the older ones
            start_loc = dates_of_data.get_indexer([start], method="bfill")[0]
            first_date = dates_of_data[max(start_loc - backwards, 0)]
            first_complete = df.index.levels[0].searchsorted(first_date)
            self._first_complete_date = max(self._first_complete_date, first_complete)
        self._incomplete_rows = {}
        if self._align_by_time:
            # every asset gets a row on every date, so a bar is a date, and the older bars the
            # loader gives to an asset with gaps would only add empty rows to all the others
            df = df.loc[df.index.levels[0][self._first_complete_date] :]
            self._first_complete_date = 0
            # since pandas 0.23, MultiIndex reindex is slow, so using a alternative way here,
            # but still very slow.
            # df = df.reindex(pd.MultiIndex.from_product(df.index.levels))
            df = df.unstack(level=1).stack()
            df[self._loader.time_category] = (
                df[self._loader.time_category]
                .groupby(level=0)
                .transform(lambda x: x.fillna(x.value_counts().index[0]))
            )
        if self.timezone != "UTC":
            df = df.reset_index("asset").tz_convert(self.timezone).set_index(["asset"], append=True)

        self._dataframe = df
        self._dataframe_index = [df.index.get_level_values(i) for i in range(len(df.index.levels))]

        # asset group
        cat = self._dataframe_index[1].codes
        # cat.copy() for suppress torch readonly numpy array warning
        keys = torch.tensor(cat.copy(), device=self._device, dtype=torch.int32)
        self._groups["asset"] = ParallelGroupBy(keys)

        # time group prepare
        self.column_to_parallel_groupby_(self._loader.time_category, "date")
        # change engine cache id
        # print('_cache length changed', max_backwards, self._last_loaded[2])
        self._cache_hash = uuid.uuid4()

        # if not self.align_by_time and not self.loader_.is_align_by_time:
        #     warnings.warn("Date Misalignment!!! Either the dataloader is marked as align_by_time"
        #                   ", or turned on the engine's align_by_time.",
        #                   RuntimeWarning)

        self._column_cache = {}
        if isinstance(self._filter, StaticAssets):
            # if pre-screened, don't cache data, only cache full data.
            self._last_loaded = [None, None, None]
        else:
            self._last_loaded = [start, end, backwards]
        return True

    def _compute_and_revert(self, f: BaseFactor, name) -> torch.Tensor:
        stream = None
        if self._device.type == "cuda" and self._enable_stream:
            stream = torch.cuda.current_stream()
        data = f.compute_(stream)
        return self._groups[f.groupby].revert(data, name)

    # public:

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._dataframe_index = None
        self._groups = dict()
        self._last_loaded = [None, None, None]
        self._column_cache = {}
        self._factors = {}
        self._filter = None
        self._device = torch.device("cpu")
        self._enable_stream = False
        self._align_by_time = False
        self._timezone = "UTC"
        self._cache_hash = uuid.uuid4()
        self._first_complete_date = 0
        self._incomplete_rows = {}
        self._history_cut = False
        self.max_backwards = np.inf

    @property
    def device(self):
        return self._device

    @property
    def dataframe_index(self):
        return self._dataframe_index

    def create_tensor(self, group: str, dtype, values, nan_values) -> torch.Tensor:
        return self._groups[group].create(dtype, values, nan_values)

    @property
    def align_by_time(self):
        return self._align_by_time

    @align_by_time.setter
    def align_by_time(self, enable: bool):
        """
        If `enable` is `True`, df index will be the product of 'date' and 'asset'.
        This method is slow, recommended to do it in your DataLoader in advance.
        """
        if enable != self._align_by_time:
            # cached data was prepared with the old setting
            self.empty_cache()
        self._align_by_time = enable

    @property
    def timezone(self):
        return self._timezone

    @timezone.setter
    def timezone(self, tz: str):
        """Timezone of the datetime index, affects DatetimeDataFactor like WEEKDAY"""
        if tz != self._timezone:
            # cached data was prepared with the old timezone
            self.empty_cache()
        self._timezone = tz

    def add(
        self, factor: Iterable[BaseFactor] | BaseFactor, name: Iterable[str] | str, replace=False
    ) -> None:
        """
        Add factor or filter to engine, as a column.
        """
        assert factor is not None
        if isinstance(factor, Iterable):
            for i, fct in enumerate(factor):
                self.add(fct, name[i], replace)
        else:
            if name in self._factors and not replace:
                raise KeyError(
                    f"A factor with the name {name} already exists."
                    "please specify a new name by engine.add(factor, new_name)"
                )
            self._factors[name] = factor

    def set_filter(self, factor: FilterFactor | None) -> None:
        static_assets = isinstance(self._filter, StaticAssets) or isinstance(factor, StaticAssets)
        if static_assets and self._dataframe is not None:
            print("Cache clean due to the StaticAssets filter changed.")
            self.empty_cache()
        self._filter = factor

    def get_filter(self):
        return self._filter

    def get_factor(self, name):
        return self._factors[name]

    @property
    def factors(self):
        return self._factors.copy()

    def clear(self):
        self.remove_all_factors()
        self.set_filter(None)

    def empty_cache(self):
        self._last_loaded = [None, None, None]
        self._column_cache = {}
        self._groups = dict()
        self._dataframe = None
        self._dataframe_index = None
        self._cache_hash = uuid.uuid4()
        self._first_complete_date = 0
        self._incomplete_rows = {}

    def remove_all_factors(self) -> None:
        self._factors = {}

    def to_cuda(self, enable_stream=False, gpu_id=0) -> None:
        """
        Set enable_stream to True allows pipeline branches to calculation simultaneously.
        However, this will lead to more VRAM usage and may affect performance.
        """
        self._device = torch.device(f"cuda:{gpu_id}")
        self._enable_stream = enable_stream
        self.empty_cache()

    def to_cpu(self) -> None:
        self._device = torch.device("cpu")
        self.empty_cache()

    @staticmethod
    def _to_utc(time):
        # to_datetime makes None a NaT, and None means the last bar of the data to the loader
        return None if time is None else pd.to_datetime(time, utc=True)

    def test_lookahead_bias(self, start, end):
        """Check all factors, if there are look-ahead bias"""
        if self._loader.ohlcv is None:
            raise ValueError(
                "test_lookahead_bias randomizes the OHLCV columns, but the loader has no ohlcv."
            )
        start, end = self._to_utc(start), self._to_utc(end)
        # get results
        df_expected = self.run(start, end)
        # modify future data
        dt_index = self._dataframe[start:].index.get_level_values(0).unique()
        mid = int(len(dt_index) / 2)
        mid_left = dt_index[mid - 1]
        mid_right = dt_index[mid]
        length = self._dataframe.loc[mid_right:].shape[0]
        try:
            for col in self._loader.ohlcv:
                dtype = self._dataframe[col].dtype
                random = np.random.randn(length)
                if pd.api.types.is_integer_dtype(dtype):
                    # or it casts to 0 and 1
                    random = np.abs(random) * 1000
                self._dataframe.loc[mid_right:, col] = random.astype(dtype)
            self._column_cache = {}
            # hack to disable reload _dataframe
            self._last_loaded = [start, end, np.inf]
            # check if results are consistent
            df = self.run(start, end)
        finally:
            # the data in memory is random now, whatever happened
            self.empty_cache()

        try:
            pd.testing.assert_frame_equal(df_expected[:mid_left], df[:mid_left])
        except AssertionError:
            raise RuntimeError("A look-ahead bias was detected, please check your factors code")
        return "No assertion raised."

    @contextmanager
    def _masks_know_the_filter(self):
        """
        mask='universe' reaches the filter through the engine, which `pre_compute_` binds. Lets
        `should_delay` and `get_total_backwards_` of the factors use it before that. The values
        are read here, a filter can't use itself as a mask, `compute_` refuses that.
        """
        if self._filter is not None:
            CustomFactor._universe_delay = self._filter.should_delay()
            CustomFactor._universe_backwards = self._filter.get_total_backwards_()
        try:
            yield
        finally:
            CustomFactor._universe_delay = False
            CustomFactor._universe_backwards = 0

    def _run(self, start, end, delay_factor):
        if len(self._factors) == 0:
            raise ValueError("Please add at least one factor to engine, then run again.")

        if delay_factor == "all":
            delays = self._factors
        else:
            with self._masks_know_the_filter():
                delays = {col for col, fct in self._factors.items() if fct.should_delay()}
        if delay_factor is False and len(delays) > 0:
            warnings.warn(
                "Warning!! delay_factor is set to False, "
                f"but {delays!s} factors uses data that is only available "
                "after the market is closed.",
                RuntimeWarning,
            )
            delays = {}

        # make columns to data factors.
        if self._loader.ohlcv is not None:
            OHLCV.open.inputs = (self._loader.ohlcv[0], self._loader.adjustment_multipliers[0])
            OHLCV.high.inputs = (self._loader.ohlcv[1], self._loader.adjustment_multipliers[0])
            OHLCV.low.inputs = (self._loader.ohlcv[2], self._loader.adjustment_multipliers[0])
            OHLCV.close.inputs = (self._loader.ohlcv[3], self._loader.adjustment_multipliers[0])
            OHLCV.volume.inputs = (self._loader.ohlcv[4], self._loader.adjustment_multipliers[1])

        # shift factors if necessary
        filter_ = self._filter
        if filter_ and delay_factor and (filter_.should_delay() or delay_factor == "all"):
            filter_ = filter_.shift(1)
        factors = {col: col in delays and fct.shift(1) or fct for col, fct in self._factors.items()}
        delayed = len(delays) > 0 or filter_ is not self._filter

        # calculate how much historical data is needed
        with self._masks_know_the_filter():
            max_backwards = max([f.get_total_backwards_() for f in factors.values()])
            if filter_:
                max_backwards = max(max_backwards, filter_.get_total_backwards_())

        # copy data to tensor, if any data copied, return True
        force_cleanup = self._prepare_tensor(start, end, max_backwards, delayed)

        # clean up before start, and if _prepare_tensor returns new data,
        # then force clean up any caches in sub factors
        # For heavily nested factors, iterate factor tree will very slow, so here we try not to
        # clean up new factors.
        if filter_:
            filter_.clean_up_(force_cleanup)
        for f in factors.values():
            f.clean_up_(force_cleanup)

        try:
            # some pre-work
            if filter_:
                filter_.pre_compute_(self, start, end)
            for f in factors.values():
                # print(_)
                f.pre_compute_(self, start, end)

            # schedule possible gpu work first
            results = {col: self._compute_and_revert(fct, col) for col, fct in factors.items()}
            shifted_mask = None
            if filter_:
                shifted_mask = self._compute_and_revert(filter_, "filter")
        except BaseException:
            # Reset reference counts and cached data, otherwise the factors (some of them are
            # shared, like OHLCV) will use the stale data next time.
            if filter_:
                filter_.clean_up_(force=True)
            for f in factors.values():
                f.clean_up_(force=True)
            raise

        # do clean up again
        if filter_:
            filter_.clean_up_()
        for f in factors.values():
            f.clean_up_()

        return results, shifted_mask, delayed

    def run(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp | None, delay_factor=True
    ) -> pd.DataFrame:
        """
        Compute factors and filters, return a df contains all.
        If `end` is None, up to the last bar of the data.
        """
        start, end = self._to_utc(start), self._to_utc(end)

        results, shifted_mask, delayed = self._run(start, end, delay_factor)
        # do cpu work and synchronize will automatically done by torch
        ret = pd.DataFrame(index=self._dataframe.index.copy())
        # pandas will rise DataFrame is highly fragmented on this, no sense.
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            ret = ret.assign(**{col: t.cpu().numpy() for col, t in results.items()})
        if shifted_mask is not None:
            ret = ret[shifted_mask.cpu().numpy()]

        # if any factors delayed, return df also should be delayed
        if delayed:
            index = ret.index.levels[0]
            start_ind = index.get_indexer([start], method="bfill")[0]
            if (start_ind + 1) >= len(index):
                raise ValueError("There is no data between start and end.")
            start = index[start_ind + 1]
        ret = ret.sort_index()
        return ret.loc[start:]

    def run_raw(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
        delay_factor=True,
        return_index=False,
    ) -> dict[str, torch.Tensor] | tuple:
        """
        Compute factors and filters, return a dict contains factor_name = torch.Tensor
        If `end` is None, up to the last bar of the data.
        """
        start, end = self._to_utc(start), self._to_utc(end)

        results, shifted_mask, delayed = self._run(start, end, delay_factor)

        index = self._dataframe.index.levels[0]
        start_ind = index.get_indexer([start], method="bfill")[0]
        if delayed:  # if any factors delayed, return df also should be delayed
            unique_date_index = self._dataframe.index.levels[0]
            start_unique_ind = unique_date_index.get_indexer([start], "bfill")[0]
            if (start_unique_ind + 1) >= len(unique_date_index):
                raise ValueError("There is no data between start and end.")
            start = unique_date_index[start_unique_ind + 1]

        index = self._dataframe_index[0]
        start_ind = index.searchsorted(start)

        if start_ind >= len(index):
            raise ValueError("There is no data between start and end.")
        if shifted_mask is not None:
            shifted_mask = shifted_mask[start_ind:]
            results = {k: v[start_ind:][shifted_mask] for k, v in results.items()}
        else:
            results = {k: v[start_ind:] for k, v in results.items()}

        if return_index:
            index = index[start_ind:]
            if shifted_mask is not None:
                index = index[shifted_mask.cpu().numpy()]
            return results, index
        else:
            return results

    def get_factors_raw_value(self):
        stream = None
        if self._device.type == "cuda":
            stream = torch.cuda.current_stream()
        return {c: f.compute_(stream) for c, f in self._factors.items()}

    def get_price_matrix(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        prices: ColumnDataFactor = OHLCV.close,
    ) -> pd.DataFrame:
        """
        Get the price data for Factor Return Analysis.
        :param start: same as run
        :param end: should be longer than the `end` time of `run`, for forward returns calculations.
        :param prices: prices data factor. If you traded at the opening, you should set it
                       to OHLCV.open.
        """
        factors_backup, filter_backup = self._factors, self._filter
        self._factors = {"price": AdjustedColumnDataFactor(prices)}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # get tickers first
                if self._filter is not None:
                    assets_ret = self.run(start, end, delay_factor=False)
                    assets = assets_ret.index.get_level_values(1).unique()
                    self._filter = StaticAssets(assets)
                ret = self.run(start, end, delay_factor=False)
        finally:
            self._factors = factors_backup
            self._filter = filter_backup

        ret = ret["price"].unstack(level=[1])
        return ret

    def plot_chart(
        self, start, end, trace_types=None, styles=None, delay_factor=True, inline=True, ohlcv=None
    ):
        """
        Plotting common stock price chart for researching.
        :param start: same as engine.run()
        :param end: same as engine.run()
        :param delay_factor: same as engine.run()
        :param trace_types: dict(factor_name=plotly_trace_type), default is 'Scatter'
        :param styles: dict(factor_name=plotly_trace_styles)
        :param inline: display plot immediately
        :param ohlcv: open high low close volume column names, open high low can be None.

        Usage::

            engine = factors.FactorEngine(loader)
            engine.timezone = 'America/New_York'
            engine.set_filter(factors.StaticAssets({'NVDA', 'MSFT'}))
            engine.add(factors.MA(20), 'MA20')
            engine.add(factors.RSI(), 'RSI')
            engine.to_cuda()
            engine.plot_chart('2017', '2018', styles={
                'MA20': {
                          'line': {'dash': 'dash'}
                       },
                'RSI': {
                          'yaxis': 'y3',
                          'line': {'width': 1}
                       }
            })

        """
        assert ohlcv is not None or self.loader_.ohlcv is not None, (
            "parameter ohlcv cannot be None."
        )
        df = self.run(start, end, delay_factor)
        figs = plot_chart(
            self._dataframe,
            ohlcv or self.loader_.ohlcv,
            df,
            trace_types=trace_types,
            styles=styles,
            inline=inline,
        )
        return figs, df

    def full_run(
        self,
        start,
        end,
        trade_at="close",
        periods=(1, 4, 9),
        quantiles=5,
        filter_zscore=20,
        demean=True,
        preview=True,
        to_weight=True,
        xs_quantile=True,
        demean_weight=True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return this:
        |    	                    |    	|  Returns  |      factor_name          	|
        |date	                    |asset	|10D	    |factor	    |factor_quantile	|
        |---------------------------|-------|-----------|-----------|-------------------|
        |2014-01-08 00:00:00+00:00	|ARNC	|0.070159	|0.215274	|5                  |
        |                           |BA	    |-0.038556	|-1.638784	|1                  |
        For alphalens analysis, you can use this:
        factor_data = full_run_return[['factor_name', 'Returns']].droplevel(0, axis=1)
        al.tears.create_returns_tear_sheet(factor_data)
        :param str, pd.Timestamp start: Factor analysis start time
        :param str, pd.Timestamp end: Factor analysis end time
        :param trade_at: Which price for forward returns. 'open', or 'close.
                         With 'open', a delayed factor enters at the open of its row, and a factor
                         that isn't delayed at the open of the next row. All factors of a run share
                         one return column, so if only some of them are delayed, all are delayed.
                         If is a factor, its values are the prices, and all factors are delayed.
                         If is 'current_close', same as run engine with delay_factor=False,
                         Be sure that no any high,low,close data is used in factor, otherwise will
                         cause lookahead bias.
        :param periods: Forward return periods
        :param quantiles: Number of quantile
        :param xs_quantile: Calculate quantile based on cross-section
        :param filter_zscore: Drop extreme factor return, for stability of the analysis.
        :param demean: demean quantile returns
        :param preview: Display a preview chart of the result
        :param to_weight: Normalize the factor value to a weight on the cross-section
        :param demean_weight: If to_weight is True, will the factor weight be converted into a
                              hedged weight: sum(weight) = 0
        """
        factors = self._factors.copy()
        universe = self.get_filter()

        column_names = {}
        try:
            # add quantile factor of all factors
            for c, f in factors.items():
                column_names[c] = (c, "factor")
                quantile_fct = f.quantile(quantiles, mask=universe)
                if not xs_quantile:
                    quantile_fct.groupby = "asset"
                self.add(quantile_fct, c + "_q_")
                column_names[c + "_q_"] = (c, "factor_quantile")

                if to_weight:
                    self.add(f.to_weight(mask=universe, demean=demean_weight), c + "_w_")
                else:
                    self.add(f, c + "_w_")
                column_names[c + "_w_"] = (c, "factor_weight")

            # add the rolling returns of each period, use AdjustedColumnDataFactor for best
            # performance
            # `shift` puts on row t the return of entering on t + 1, or on t if that row is
            # delayed. The delay of the return columns is set here, or a universe would decide it
            # by its mask.
            shift = -1
            delay_factor = True
            delay_returns = True
            # entering a bar later needs a bar more at the end
            extra_tail = 0
            inputs = (AdjustedColumnDataFactor(OHLCV.close),)
            if isinstance(trade_at, BaseFactor):
                inputs = (trade_at,)
                delay_factor = "all"
            elif trade_at == "open":
                inputs = (AdjustedColumnDataFactor(OHLCV.open),)
                # On row t a delayed factor knows close[t - 1], so it can enter at open[t]. A factor
                # that isn't delayed knows open[t], so it can only enter at open[t + 1].
                with self._masks_know_the_filter():
                    delays = [f.should_delay() for f in self._factors.values()]
                delay_returns = any(delays)
                if not delay_returns:
                    extra_tail = 1
                elif not all(delays):
                    delay_factor = "all"
            elif trade_at == "current_close":
                shift = 0
                delay_factor = False
                delay_returns = None

            from .basic import Returns

            for n in periods:
                # Different: returns here diff by bar, which alphalens diff by time
                ret = Returns(win=n + 1, inputs=inputs).shift(-n + shift)
                mask = universe
                returns = ret
                if filter_zscore is not None:
                    # Different: The zscore here contains all backward data which alphalens not
                    # counted.
                    zscore_factor = ret.zscore(groupby="asset", mask=universe)
                    zscore_filter = zscore_factor.abs() <= filter_zscore
                    if mask is not None:
                        mask = mask & zscore_filter
                    else:
                        mask = zscore_filter
                    returns = ret.filter(mask)
                returns.set_delay(delay_returns)
                self.add(returns, str(n) + "_r_")
                if demean:
                    demeaned = ret.demean(mask=mask)
                    demeaned.set_delay(delay_returns)
                    self.add(demeaned, str(n) + "_d_")

            # run and get df
            factor_data = self.run(start, end, delay_factor)
        finally:
            self._factors = factors
        factor_data.index = factor_data.index.remove_unused_levels()
        # factor_data = factor_data.sort_index()  # 140 ms
        # Count on the loaded dates, a filter can leave out any of them. A value in the column
        # can't tell it either, on their last rows the short assets read padding.
        dates = self._dataframe.index.levels[0]
        tail = max(periods) + extra_tail
        no_data = "No enough data for forward returns, please expand the end date"
        assert len(dates) > tail, no_data
        factor_data = factor_data.loc[: dates[-tail - 1]].copy()
        assert len(factor_data) > 0, no_data

        # infer freq
        delta = min(dates[1:] - dates[:-1])
        unit = delta.resolution_string
        freq = int(delta / pd.Timedelta(1, unit))
        # change columns name
        period_cols = {n: str(n * freq) + unit for n in periods}
        for n, period_col in period_cols.items():
            column_names[str(n) + "_r_"] = ("Returns", period_col)
            if demean:
                column_names[str(n) + "_d_"] = ("Demeaned", period_col)
        new_cols = pd.MultiIndex.from_tuples([column_names[c] for c in factor_data.columns])
        factor_data.columns = new_cols
        factor_data = factor_data.sort_index(axis=1)

        # mean return, return std err
        mean_returns = []
        for fact_name in factors:
            mean_return = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], []]))
            group = [(fact_name, "factor_quantile"), "date"]
            if demean:
                returns_col = "Demeaned"
            else:
                returns_col = "Returns"
            grouped_mean = factor_data[[returns_col, fact_name]].groupby(group).agg("mean")
            for period_col in period_cols.values():
                demean_col = (returns_col, period_col)
                mean_col = (fact_name, period_col)
                mean_return[mean_col] = grouped_mean[demean_col]
            if mean_return.empty:
                continue
            mean_return.index = mean_return.index.set_names("quantile", level=0)
            mean_return = mean_return.groupby(level=0).agg(["mean", "sem"])
            mean_return = mean_return.sort_index(axis=1)
            mean_returns.append(mean_return)
        mean_returns = pd.concat(mean_returns, axis=1)

        # plot
        if preview:
            fig = plot_quantile_and_cumulative_returns(factor_data, mean_returns, dates)
            fig.show()

        return factor_data, mean_returns
