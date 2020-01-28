"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Iterable, Tuple
import warnings
from .factor import BaseFactor
from .filter import FilterFactor, StaticAssets
from .datafactor import DataFactor, AdjustedDataFactor
from .plotting import plot_quantile_and_cumulative_returns
from ..data import DataLoader
from ..parallel import ParallelGroupBy
import pandas as pd
import numpy as np
import torch


class OHLCV:
    open = DataFactor(inputs=('',), is_data_after_market_close=False)
    high = DataFactor(inputs=('',))
    low = DataFactor(inputs=('',))
    close = DataFactor(inputs=('',))
    volume = DataFactor(inputs=('',))


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

    def create_tensor(self, group: str, dtype, values, nan_values) -> torch.Tensor:
        return self._groups[group].create(dtype, values, nan_values)

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
        df.index = df.index.remove_unused_levels()
        history_win = df.index.levels[0].get_loc(start, 'bfill')
        if history_win < max_backwards:
            warnings.warn("Historical data seems insufficient. "
                          "{} rows of historical data are required, but only {} rows are obtained. "
                          "It is also possible that `calender_asset` of the loader is not set, "
                          "some out of trading hours data will cause indexing problems."
                          .format(max_backwards, history_win),
                          RuntimeWarning)
        if isinstance(self._filter, StaticAssets):
            df = df.loc[(slice(None), self._filter.assets), :]
            if df.shape[0] == 0:
                raise ValueError("The asset specified by StaticAssets filter, was not found in "
                                 "DataLoader.")
        if self._align_by_time:
            # since pandas 0.23, MultiIndex reindex is slow, so using a alternative way here,
            # but still very slow.
            # df = df.reindex(pd.MultiIndex.from_product(df.index.levels))
            df = df.unstack(level=1).stack(dropna=False)
        self._dataframe = df

        # asset group
        cat = self._dataframe.index.get_level_values(1).codes
        keys = torch.tensor(cat, device=self._device, dtype=torch.int32)
        self._groups['asset'] = ParallelGroupBy(keys)

        # time group prepare
        self.column_to_parallel_groupby_(self._loader.time_category, 'date')

        self._column_cache = {}
        self._last_load = [start, end, max_backwards]

    def _compute_and_revert(self, f: BaseFactor, name) -> torch.Tensor:
        stream = None
        if self._device.type == 'cuda':
            stream = torch.cuda.current_stream()
        data = f.compute_(stream)
        return self._groups[f.groupby].revert(data, name)

    # public:

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._groups = dict()
        self._last_load = [None, None, None]
        self._column_cache = {}
        self._factors = {}
        self._filter = None
        self._device = torch.device('cpu')
        self._align_by_time = False

    @property
    def device(self):
        return self._device

    def set_align_by_time(self, enable: bool):
        """
        If `enable` is `True`, df index will be the product of 'date' and 'asset'.
        This method is slow, recommended to do it in your DataLoader in advance.
        """
        self._align_by_time = enable

    def add(self,
            factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Union[Iterable[str], str],
            ) -> None:
        """
        Add factor or filter to engine, as a column.
        """
        if isinstance(factor, Iterable):
            for i, fct in enumerate(factor):
                self.add(fct, name and name[i] or None)
        else:
            if name in self._factors:
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

    def get_asset_names(self):
        return self._dataframe.index.get_level_values(1).unique().values

    def clear(self):
        self.remove_all_factors()
        self.set_filter(None)

    def remove_all_factors(self) -> None:
        self._factors = {}

    def to_cuda(self) -> None:
        self._device = torch.device('cuda')
        self._last_load = [None, None, None]

    def to_cpu(self) -> None:
        self._device = torch.device('cpu')
        self._last_load = [None, None, None]

    def test_lookahead_bias(self, start, end):
        """Check all factors, if there are look-ahead bias"""
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        # get results
        df_expected = self.run(start, end)
        # modify future data
        mid = int(self._dataframe[start:].shape[0] / 2)
        mid_time = self._dataframe[start:].index[mid][0]
        length = self._dataframe.loc[mid_time:].shape[0]
        for c in self._loader.ohlcv:
            self._dataframe.loc[mid_time:, c] = np.random.randn(length)
        self._column_cache = {}
        # check if results are consistent
        df = self.run(start, end)
        # clean
        self._column_cache = {}
        self._last_load = [None, None, None]

        try:
            pd.testing.assert_frame_equal(df_expected[:mid_time], df[:mid_time])
        except AssertionError as e:
            raise RuntimeError('A look-ahead bias was detected, please check your factors code')
        return 'No assertion raised.'

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp],
            delay_factor=True) -> pd.DataFrame:
        """
        Compute factors and filters, return a df contains all.
        """
        if len(self._factors) == 0:
            raise ValueError('Please add at least one factor to engine, then run again.')

        if not delay_factor:
            for c, f in self._factors.items():
                if f.is_close_data_used():
                    warnings.warn("Warning!! delay_factor is set to False, "
                                  "but {} factor uses data that is only available "
                                  "after the market is closed.".format(c),
                                  RuntimeWarning)

        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        # make columns to data factors.
        if self._loader.ohlcv is not None:
            OHLCV.open.inputs = (self._loader.ohlcv[0], self._loader.adjustment_multipliers[0])
            OHLCV.high.inputs = (self._loader.ohlcv[1], self._loader.adjustment_multipliers[0])
            OHLCV.low.inputs = (self._loader.ohlcv[2], self._loader.adjustment_multipliers[0])
            OHLCV.close.inputs = (self._loader.ohlcv[3], self._loader.adjustment_multipliers[0])
            OHLCV.volume.inputs = (self._loader.ohlcv[4], self._loader.adjustment_multipliers[1])

        # get factor
        filter_ = self._filter
        if filter_ and delay_factor:
            filter_ = filter_.shift(1)
        factors = {c: delay_factor and f.shift(1) or f for c, f in self._factors.items()}

        # Calculate data that requires backwards in tree
        max_backwards = max([f.get_total_backwards_() for f in factors.values()])
        if filter_:
            max_backwards = max(max_backwards, filter_.get_total_backwards_())
        # Get data
        self._prepare_tensor(start, end, max_backwards)

        # clean up before start / may be keyboard interrupt
        if filter_:
            filter_.clean_up_()
        for f in factors.values():
            f.clean_up_()

        # ready to compute
        if filter_:
            filter_.pre_compute_(self, start, end)
        for f in factors.values():
            f.pre_compute_(self, start, end)

        # schedule possible gpu work first
        results = {col: self._compute_and_revert(fct, col) for col, fct in factors.items()}
        shift_mask = None
        if filter_:
            shift_mask = self._compute_and_revert(filter_, 'filter')
        # do cpu work and synchronize will automatically done by torch
        ret = pd.DataFrame(index=self._dataframe.index.copy())
        ret = ret.assign(**{col: t.cpu().numpy() for col, t in results.items()})
        if filter_:
            ret = ret[shift_mask.cpu().numpy()]

        # do clean up again
        if filter_:
            filter_.clean_up_()
        for f in factors.values():
            f.clean_up_()

        index = ret.index.levels[0]
        start = index.get_loc(start, 'bfill')
        if delay_factor:
            start += 1
        return ret.loc[index[start]:]

    def get_factors_raw_value(self):
        stream = None
        if self._device.type == 'cuda':
            stream = torch.cuda.current_stream()
        return {c: f.compute_(stream) for c, f in self._factors.items()}

    def get_price_matrix(self,
                         start: Union[str, pd.Timestamp],
                         end: Union[str, pd.Timestamp],
                         prices: DataFactor = OHLCV.close,
                         ) -> pd.DataFrame:
        """
        Get the price data for Factor Return Analysis.
        :param start: same as run
        :param end: should be longer than the `end` time of `run`, for forward returns calculations.
        :param prices: prices data factor. If you traded at the opening, you should set it
                       to OHLCV.open.
        """
        factors_backup = self._factors
        self._factors = {'price': AdjustedDataFactor(prices)}

        # get tickers first
        assets = None
        if self._filter is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assets_ret = self.run(start, end, delay_factor=False)
            assets = assets_ret.index.get_level_values(1).unique()

        filter_backup = self._filter
        self._filter = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = self.run(start, end, delay_factor=False)
        self._factors = factors_backup
        self._filter = filter_backup

        ret = ret['price'].unstack(level=[1])
        if assets is not None:
            ret = ret[assets]
        return ret

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
        for alphalens analysis, you can use this:
        factor_data = full_run_return[['factor_name', 'Returns']].droplevel(0, axis=1)
        al.tears.create_returns_tear_sheet(factor_data)
        :param str, pd.Timestamp start: factor analysis start time
        :param str, pd.Timestamp end: factor analysis end time
        :param trade_at: which price for forward returns. 'open', or 'close.
                         If is 'current_close', same as run engine with delay_factor=False,
                         Meaning use the factor to trade on the same day it generated. Be sure that
                         no any high,low,close data is used in factor, otherwise will cause
                         lookahead bias.
        :param periods: forward return periods
        :param quantiles: number of quantile
        :param filter_zscore: drop extreme factor return, for stability of the analysis.
        :param demean: Whether the factor is converted into a hedged weight: sum(weight) = 0
        :param preview: display a preview chart of the result
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

        # add the rolling returns of each period, use AdjustedDataFactor for best performance
        shift = -1
        inputs = (AdjustedDataFactor(OHLCV.close),)
        if trade_at == 'open':
            inputs = (AdjustedDataFactor(OHLCV.open),)
        elif trade_at == 'current_close':
            shift = 0
        from .basic import Returns
        for n in periods:
            # Different: returns here diff by bar, which alphalens diff by time
            ret = Returns(win=n + 1, inputs=inputs).shift(-n + shift)
            mask = universe
            if filter_zscore is not None:
                # Different: The zscore here contains all backward data which alphalens not counted.
                zscore_factor = ret.zscore(axis_asset=True, mask=universe)
                zscore_filter = zscore_factor.abs() <= filter_zscore
                mask = mask & zscore_filter
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
        mean_return.index.levels[0].name = 'quantile'
        mean_return = mean_return.groupby(level=0).agg(['mean', 'sem'])
        mean_return.sort_index(axis=1, inplace=True)

        # plot
        if preview:
            plot_quantile_and_cumulative_returns(factor_data, mean_return)

        return factor_data, mean_return
