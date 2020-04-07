"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Iterable, Tuple, Dict
import warnings
from .factor import BaseFactor
from .filter import FilterFactor, StaticAssets
from .datafactor import ColumnDataFactor, AdjustedColumnDataFactor
from ..plotting import plot_quantile_and_cumulative_returns, plot_chart
from ..data import DataLoader
from ..parallel import ParallelGroupBy
import pandas as pd
import numpy as np
import torch


class OHLCV:
    open = ColumnDataFactor(inputs=('',), should_delay=False)
    high = ColumnDataFactor(inputs=('',))
    low = ColumnDataFactor(inputs=('',))
    close = ColumnDataFactor(inputs=('',))
    volume = ColumnDataFactor(inputs=('',))


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

    def column_to_tensor_(self, data_column) -> torch.Tensor:
        # cache data with column prevent double copying
        if data_column in self._column_cache:
            return self._column_cache[data_column]

        series = self._dataframe[data_column]
        data = torch.from_numpy(series.values).to(self._device, non_blocking=True)
        self._column_cache[data_column] = data
        return data

    def column_to_parallel_groupby_(self, group_column: str, as_group_name=None):
        if as_group_name is None:
            as_group_name = group_column
        if as_group_name in self._groups:
            return

        series = self._dataframe[group_column]
        if series.dtype.name == 'category':
            cat = series.cat.codes
        else:
            cat = series.values
        keys = torch.tensor(cat, device=self._device, dtype=torch.int32)
        self._groups[as_group_name] = ParallelGroupBy(keys)

    def revert_(self, data: torch.Tensor, group: str, factor_name: str) -> torch.Tensor:
        return self._groups[group].revert(data, factor_name)

    def revert_to_series_(self, data: torch.Tensor, group: str, factor_name: str) -> pd.Series:
        array = self.revert_(data, group, factor_name).cpu()
        return pd.Series(array, index=self._dataframe.index)

    def group_by_(self, data: Union[torch.Tensor, pd.Series], group: str) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return self._groups[group].split(data)
        elif isinstance(data, pd.Series):
            data = torch.tensor(data.values, device=self._device)
            return self._groups[group].split(data)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data, device=self._device)
            return self._groups[group].split(data)
        else:
            raise ValueError('Invalid data type, should be tensor or series.')

    # private:

    def _prepare_tensor(self, start, end, max_backwards):
        # Check cache, just in case, if use some ML techniques, engine may be called repeatedly
        # with same date range.
        if start == self._last_load[0] and end == self._last_load[1] \
                and max_backwards <= self._last_load[2]:
            return
        self._groups = dict()

        # Get data
        df = self._loader.load(start, end, max_backwards).copy()
        # If possible, pre-screen
        if isinstance(self._filter, StaticAssets):
            df = df.loc[(slice(None), self._filter.assets), :]
            if df.shape[0] == 0:
                raise ValueError("The assets {} specified by StaticAssets filter, was not found in "
                                 "DataLoader.".format(self._filter.assets))
        # check history data is insufficient
        df.index = df.index.remove_unused_levels()
        history_win = df.index.levels[0].get_loc(start, 'bfill')
        if history_win < max_backwards:
            warnings.warn("Historical data seems insufficient. "
                          "{} rows of historical data are required, but only {} rows are obtained. "
                          "It is also possible that `calender_asset` of the loader is not set, "
                          "some out of trading hours data will cause indexing problems."
                          .format(max_backwards, history_win),
                          RuntimeWarning)
        # post processing data
        if self._align_by_time:
            # since pandas 0.23, MultiIndex reindex is slow, so using a alternative way here,
            # but still very slow.
            # df = df.reindex(pd.MultiIndex.from_product(df.index.levels))
            df = df.unstack(level=1).stack(dropna=False)
        if self.timezone != 'UTC':
            df = df.reset_index('asset').tz_convert(self.timezone)\
                .set_index(['asset'], append=True)

        self._dataframe = df
        self._dataframe_index = [df.index.get_level_values(i) for i in range(len(df.index.levels))]

        # asset group
        cat = self._dataframe_index[1].codes
        keys = torch.tensor(cat, device=self._device, dtype=torch.int32)
        self._groups['asset'] = ParallelGroupBy(keys)

        # time group prepare
        self.column_to_parallel_groupby_(self._loader.time_category, 'date')

        self._column_cache = {}
        if isinstance(self._filter, StaticAssets):
            # if pre-screened, don't cache data, only cache full data.
            self._last_load = [None, None, None]
        else:
            self._last_load = [start, end, max_backwards]

    def _compute_and_revert(self, f: BaseFactor, name) -> torch.Tensor:
        stream = None
        if self._device.type == 'cuda' and self._enable_stream:
            stream = torch.cuda.current_stream()
        data = f.compute_(stream)
        return self._groups[f.groupby].revert(data, name)

    # public:

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._dataframe_index = None
        self._groups = dict()
        self._last_load = [None, None, None]
        self._column_cache = {}
        self._factors = {}
        self._filter = None
        self._device = torch.device('cpu')
        self._enable_stream = False
        self._align_by_time = False
        self.timezone = 'UTC'

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
        self._align_by_time = enable

    def add(self,
            factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Union[Iterable[str], str],
            replace=False) -> None:
        """
        Add factor or filter to engine, as a column.
        """
        if isinstance(factor, Iterable):
            for i, fct in enumerate(factor):
                self.add(fct, name and name[i] or None)
        else:
            if name in self._factors and not replace:
                raise KeyError('A factor with the name {} already exists.'
                               'please specify a new name by engine.add(factor, new_name)'
                               .format(name))
            self._factors[name] = factor

    def set_filter(self, factor: Union[FilterFactor, None]) -> None:
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
        self._last_load = [None, None, None]
        self._column_cache = {}
        self._groups = dict()
        self._dataframe = None
        self._dataframe_index = None

    def remove_all_factors(self) -> None:
        self._factors = {}

    def to_cuda(self, enable_stream=False) -> None:
        """
        Set enable_stream to True allows pipeline branches to calculation simultaneously.
        However, this will lead to more VRAM usage and may affect performance.
        """
        self._device = torch.device('cuda')
        self._enable_stream = enable_stream
        self.empty_cache()

    def to_cpu(self) -> None:
        self._device = torch.device('cpu')
        self.empty_cache()

    def test_lookahead_bias(self, start, end):
        """Check all factors, if there are look-ahead bias"""
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        # get results
        df_expected = self.run(start, end)
        # modify future data
        dt_index = self._dataframe[start:].index.get_level_values(0).unique()
        mid = int(len(dt_index) / 2)
        mid_left = dt_index[mid-1]
        mid_right = dt_index[mid]
        length = self._dataframe.loc[mid_right:].shape[0]
        for col in self._loader.ohlcv:
            self._dataframe.loc[mid_right:, col] = np.random.randn(length)
        self._column_cache = {}
        # hack to disable reload _dataframe
        max_backwards = max([f.get_total_backwards_() for f in self._factors.values()])
        if self._filter:
            max_backwards = max(max_backwards, self._filter.get_total_backwards_())
        self._last_load = [start, end, max_backwards]
        # check if results are consistent
        df = self.run(start, end)
        # clean
        self.empty_cache()

        try:
            pd.testing.assert_frame_equal(df_expected[:mid_left], df[:mid_left])
        except AssertionError:
            raise RuntimeError('A look-ahead bias was detected, please check your factors code')
        return 'No assertion raised.'

    def _run(self, start, end, delay_factor):
        if len(self._factors) == 0:
            raise ValueError('Please add at least one factor to engine, then run again.')

        delays = {col for col, fct in self._factors.items() if fct.should_delay()}
        if not delay_factor and len(delays) > 0:
            warnings.warn("Warning!! delay_factor is set to False, "
                          "but {} factors uses data that is only available "
                          "after the market is closed.".format(str(delays)),
                          RuntimeWarning)
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
        if filter_ and filter_.should_delay() and delay_factor:
            filter_ = filter_.shift(1)
        factors = {col: col in delays and fct.shift(1) or fct
                   for col, fct in self._factors.items()}

        # calculate how much historical data is needed
        max_backwards = max([f.get_total_backwards_() for f in factors.values()])
        if filter_:
            max_backwards = max(max_backwards, filter_.get_total_backwards_())

        # copy data to tensor
        self._prepare_tensor(start, end, max_backwards)

        # clean up before start (may be keyboard interrupted)
        if filter_:
            filter_.clean_up_()
        for f in factors.values():
            f.clean_up_()

        # some pre-work
        if filter_:
            filter_.pre_compute_(self, start, end)
        for f in factors.values():
            f.pre_compute_(self, start, end)

        # schedule possible gpu work first
        results = {col: self._compute_and_revert(fct, col) for col, fct in factors.items()}
        shifted_mask = None
        if filter_:
            shifted_mask = self._compute_and_revert(filter_, 'filter')

        # do clean up again
        if filter_:
            filter_.clean_up_()
        for f in factors.values():
            f.clean_up_()

        return results, shifted_mask, len(delays) > 0

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp],
            delay_factor=True) -> pd.DataFrame:
        """
        Compute factors and filters, return a df contains all.
        """
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

        results, shifted_mask, delayed = self._run(start, end, delay_factor)
        # do cpu work and synchronize will automatically done by torch
        ret = pd.DataFrame(index=self._dataframe.index.copy())
        ret = ret.assign(**{col: t.cpu().numpy() for col, t in results.items()})
        if shifted_mask is not None:
            ret = ret[shifted_mask.cpu().numpy()]

        # if any factors delayed, return df also should be delayed
        if delayed:
            index = ret.index.levels[0]
            start_ind = index.get_loc(start, 'bfill')
            if (start_ind + 1) >= len(index):
                raise ValueError('There is no data between start and end.')
            start = index[start_ind + 1]
        return ret.loc[start:]

    def run_raw(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp],
                delay_factor=True) -> Dict[str, torch.Tensor]:
        """
        Compute factors and filters, return a dict contains factor_name = torch.Tensor
        """
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

        results, shifted_mask, delayed = self._run(start, end, delay_factor)

        index = self._dataframe.index.levels[0]
        start_ind = index.get_loc(start, 'bfill')
        if delayed:  # if any factors delayed, return df also should be delayed
            start_ind += 1
        if start_ind >= len(index):
            raise ValueError('There is no data between start and end.')
        if shifted_mask is not None:
            shifted_mask = shifted_mask[start_ind:]
            results = {k: v[start_ind:][shifted_mask] for k, v in results.items()}
        else:
            results = {k: v[start_ind:] for k, v in results.items()}
        return results

    def get_factors_raw_value(self):
        stream = None
        if self._device.type == 'cuda':
            stream = torch.cuda.current_stream()
        return {c: f.compute_(stream) for c, f in self._factors.items()}

    def get_price_matrix(self,
                         start: Union[str, pd.Timestamp],
                         end: Union[str, pd.Timestamp],
                         prices: ColumnDataFactor = OHLCV.close,
                         ) -> pd.DataFrame:
        """
        Get the price data for Factor Return Analysis.
        :param start: same as run
        :param end: should be longer than the `end` time of `run`, for forward returns calculations.
        :param prices: prices data factor. If you traded at the opening, you should set it
                       to OHLCV.open.
        """
        factors_backup = self._factors
        self._factors = {'price': AdjustedColumnDataFactor(prices)}

        # get tickers first
        assets = None
        if self._filter is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assets_ret = self.run(start, end, delay_factor=False)
            assets = assets_ret.index.get_level_values(1).unique()

        filter_backup = self._filter
        self._filter = StaticAssets(assets)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = self.run(start, end, delay_factor=False)
        self._factors = factors_backup
        self._filter = filter_backup

        ret = ret['price'].unstack(level=[1])
        return ret

    def plot_chart(self, start, end, trace_types=None, styles=None, delay_factor=True,
                   inline=True):
        """
        Plotting common stock price chart for researching.
        :param start: same as engine.run()
        :param end: same as engine.run()
        :param delay_factor: same as engine.run()
        :param trace_types: dict(factor_name=plotly_trace_type), default is 'Scatter'
        :param styles: dict(factor_name=plotly_trace_styles)
        :param inline: display plot immediately

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
        df = self.run(start, end, delay_factor)
        figs = plot_chart(self._dataframe, self.loader_.ohlcv, df, trace_types=trace_types,
                          styles=styles, inline=inline)
        return figs, df

    def full_run(self, start, end, trade_at='close', periods=(1, 4, 9),
                 quantiles=5, filter_zscore=20, demean=True, preview=True
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                         If is 'current_close', same as run engine with delay_factor=False,
                         Be sure that no any high,low,close data is used in factor, otherwise will
                         cause lookahead bias.
        :param periods: Forward return periods
        :param quantiles: Number of quantile
        :param filter_zscore: Drop extreme factor return, for stability of the analysis.
        :param demean: Whether the factor is converted into a hedged weight: sum(weight) = 0
        :param preview: Display a preview chart of the result
        """
        factors = self._factors.copy()
        universe = self.get_filter()

        column_names = {}
        # add quantile factor of all factors
        for c, f in factors.items():
            self.add(f.quantile(quantiles, mask=universe), c + '_q_')
            self.add(f.to_weight(mask=universe, demean=demean), c + '_w_')
            column_names[c] = (c, 'factor')
            column_names[c + '_q_'] = (c, 'factor_quantile')
            column_names[c + '_w_'] = (c, 'factor_weight')

        # add the rolling returns of each period, use AdjustedColumnDataFactor for best performance
        shift = -1
        inputs = (AdjustedColumnDataFactor(OHLCV.close),)
        if trade_at == 'open':
            inputs = (AdjustedColumnDataFactor(OHLCV.open),)
        elif trade_at == 'current_close':
            shift = 0
        from .basic import Returns
        for n in periods:
            # Different: returns here diff by bar, which alphalens diff by time
            ret = Returns(win=n + 1, inputs=inputs).shift(-n + shift)
            mask = universe
            if filter_zscore is not None:
                # Different: The zscore here contains all backward data which alphalens not counted.
                zscore_factor = ret.zscore(groupby='asset', mask=universe)
                zscore_filter = zscore_factor.abs() <= filter_zscore
                if mask is not None:
                    mask = mask & zscore_filter
                else:
                    mask = zscore_filter
                self.add(ret.filter(mask), str(n) + '_r_')
            else:
                self.add(ret, str(n) + '_r_')
            self.add(ret.demean(mask=mask), str(n) + '_d_')

        # run and get df
        factor_data = self.run(start, end, trade_at != 'current_close')
        self._factors = factors
        factor_data.index = factor_data.index.remove_unused_levels()
        # factor_data.sort_index(inplace=True)  # 140 ms
        assert len(factor_data.index.levels[0]) > max(periods) - shift, \
            'No enough data for forward returns, please expand the end date'
        last_date = factor_data.index.levels[0][-max(periods) + shift - 1]
        factor_data = factor_data.loc[:last_date]

        # infer freq
        delta = min(factor_data.index.levels[0][1:] - factor_data.index.levels[0][:-1])
        unit = delta.resolution_string
        freq = int(delta / pd.Timedelta(1, unit))
        # change columns name
        period_cols = {n: str(n * freq) + unit for n in periods}
        for n, period_col in period_cols.items():
            column_names[str(n) + '_r_'] = ('Returns', period_col)
            column_names[str(n) + '_d_'] = ('Demeaned', period_col)
        new_cols = pd.MultiIndex.from_tuples([column_names[c] for c in factor_data.columns])
        factor_data.columns = new_cols
        factor_data.sort_index(axis=1, inplace=True)

        # mean return, return std err
        mean_return = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], []]))
        for fact_name, _ in factors.items():
            group = [(fact_name, 'factor_quantile'), 'date']
            grouped_mean = factor_data[['Demeaned', fact_name]].groupby(group).agg('mean')
            for n, period_col in period_cols.items():
                demean_col = ('Demeaned', period_col)
                mean_col = (fact_name, period_col)
                mean_return[mean_col] = grouped_mean[demean_col]
        mean_return.index.set_names('quantile', level=0)
        mean_return = mean_return.groupby(level=0).agg(['mean', 'sem'])
        mean_return.sort_index(axis=1, inplace=True)

        # plot
        if preview:
            plot_quantile_and_cumulative_returns(factor_data, mean_return)

        return factor_data, mean_return
