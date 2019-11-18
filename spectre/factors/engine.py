"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Iterable
from .factor import BaseFactor, DataFactor, FilterFactor
from .dataloader import DataLoader
from ..parallel import ParallelGroupBy
import pandas as pd
import numpy as np
import torch


class OHLCV:
    open = DataFactor(inputs=('',))
    high = DataFactor(inputs=('',))
    low = DataFactor(inputs=('',))
    close = DataFactor(inputs=('',))
    volume = DataFactor(inputs=('',))


class FactorEngine:
    """
    Engine for compute factors, used for back-testing and alpha-research both.
    """

    # friend private:

    def get_dataframe_(self):
        return self._dataframe

    def get_assetgroup_(self):
        return self._assetgroup

    def get_timegroup_(self):
        return self._timegroup

    def get_tensor_groupby_asset_(self, column) -> torch.Tensor:
        # todo cache data with column prevent double copying
        series = self._dataframe[column]
        data = torch.from_numpy(series.values).pin_memory().to(self._device, non_blocking=True)
        data = self._assetgroup.split(data)
        return data

    def regroup_by_asset_(self, data: Union[torch.Tensor, pd.Series]) -> torch.Tensor:
        if isinstance(data, pd.Series):
            data = torch.tensor(data.values, device=self._device)
        else:
            data = self._timegroup.revert(data, 'regroup_by_asset_')
        data = self._assetgroup.split(data)
        return data

    def regroup_by_time_(self, data: Union[torch.Tensor, pd.Series]) -> torch.Tensor:
        if isinstance(data, pd.Series):
            data = torch.tensor(data.values, device=self._device)
        else:
            data = self._assetgroup.revert(data, 'regroup_by_time_')
        data = self._timegroup.split(data)
        return data

    def revert_to_series_(self, data: torch.Tensor, is_timegroup: bool) -> pd.Series:
        if is_timegroup:
            ret = self._timegroup.revert(data)
        else:
            ret = self._assetgroup.revert(data)
        return pd.Series(ret, index=self._dataframe.index)

    # private:

    def _prepare_tensor(self, start, end, max_backward):
        # todo if unchanging, return
        # Get data
        self._dataframe = self._loader.load(start, end, max_backward)
        from datetime import datetime, timezone
        assert self._dataframe.index.is_lexsorted(), \
            "In the df returned by DateLoader, the index must be sorted, " \
            "try using df.sort_index(level=0, inplace=True)"
        assert str(self._dataframe.index.levels[0].tzinfo) == 'UTC', \
            "In the df returned by DateLoader, the date index must be UTC timezone."
        assert self._dataframe.index.levels[-1].ordered, \
            "In the df returned by DateLoader, the asset index must ordered categorical."
        cat = self._dataframe.index.get_level_values(1).codes
        keys = torch.tensor(cat, device=self._device, dtype=torch.int32)
        self._assetgroup = ParallelGroupBy(keys)

        # time group prepare
        cat = self._dataframe.time_cat_id.values
        keys = torch.tensor(cat, device=self._device, dtype=torch.int32)
        self._timegroup = ParallelGroupBy(keys)

    def _compute_and_revert(self, f: BaseFactor, name) -> Union[np.array, pd.Series]:
        """Returning pd.Series will cause very poor performance, please avoid it at 99% costs"""
        data = f.compute_(None)
        if f.is_timegroup:
            return self._timegroup.revert(data, name).cpu().numpy()
        else:
            return self._assetgroup.revert(data, name).cpu().numpy()

    # public:

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._assetgroup = None
        self._timegroup = None
        self._factors = {}
        self._filter = None
        self._device = torch.device('cpu')

    def get_device(self):
        return self._device

    def add(self,
            factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Union[Iterable[str], str]) -> None:
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

    def remove_all_factors(self) -> None:
        self._factors = {}

    def to_cuda(self) -> None:
        self._device = torch.device('cuda')

    def to_cpu(self) -> None:
        self._device = torch.device('cpu')

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """
        Compute factors and filters, return a df contains all.
        """
        if len(self._factors) == 0:
            raise ValueError('Please add at least one factor to engine, then run again.')
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        # make columns to data factors.
        OHLCV.open.inputs = (self._loader.get_ohlcv_names()[0], 'price_multi')
        OHLCV.high.inputs = (self._loader.get_ohlcv_names()[1], 'price_multi')
        OHLCV.low.inputs = (self._loader.get_ohlcv_names()[2], 'price_multi')
        OHLCV.close.inputs = (self._loader.get_ohlcv_names()[3], 'price_multi')
        OHLCV.volume.inputs = (self._loader.get_ohlcv_names()[4], 'vol_multi')
        # todo: 1 刚启动时很慢的问题，2数据加载cache功能，

        # Calculate data that requires backward in tree
        max_backward = max([f.get_total_backward_() for f in self._factors.values()])
        if self._filter:
            max_backward = max(max_backward, self._filter.get_total_backward_())
        # Get data
        self._prepare_tensor(start, end, max_backward)

        # ready to compute
        if self._filter:
            self._filter.pre_compute_(self, start, end)
        for f in self._factors.values():
            f.pre_compute_(self, start, end)

        # if cuda, Compute factors and sync
        if self._device.type == 'cuda':
            stream = torch.cuda.Stream(device=self._device)
            for col, fct in self._factors.items():
                fct.compute_(stream)
            if self._filter:
                self._filter.compute_(stream)
            torch.cuda.synchronize(device=self._device)

        # compute factors from cpu or read cache
        ret = pd.DataFrame(index=self._dataframe.index.copy())
        ret = ret.assign(**{c: self._compute_and_revert(f, c)
                            for c, f in self._factors.items()})

        # Remove filter False rows
        if self._filter:
            filter_data = self._compute_and_revert(self._filter, 'filter')
            ret = ret[filter_data]

        return ret.loc[start:end]

    def get_factors_raw_value(self):
        return {c: f.compute_(None) for c, f in self._factors.items()}

    def get_price_matrix(self,
                         start: Union[str, pd.Timestamp],
                         end: Union[str, pd.Timestamp],
                         prices: BaseFactor = OHLCV.close,
                         forward: int = 1) -> pd.DataFrame:
        """
        Get the price data for Factor Return Analysis.
        :param start: same as run
        :param end: should long than factor end time, for forward returns calculations.
        :param prices: prices data factor. If you traded at the opening, you should set it
                       to OHLCV.open.
        :param forward: int To prevent lookahead bias, please set it carefully.
                        You can only able to trade after factor calculation, so if you use
                        'close' price data to calculate the factor, the forward should be set
                        to 1 or greater, this means that the price of the trade due to the factor,
                        is the next day price in the future.
                        If you only use Open data, and trade at the close price, you can set it
                        to 0.
        """
        factors_backup = self._factors
        filter_backup = self._filter
        self._factors = {'price': prices}
        self._filter = None
        ret = self.run(start, end)
        # todo 需要按end日期full adjust, 也许弄个full adjust data factor?
        self._factors = factors_backup
        self._filter = filter_backup
        return ret['price'].unstack(level=[1]).shift(-forward)

    def get_factor_return(self, period=(1, 5, 10), quantiles=5, filter_zscore=20) -> pd.DataFrame:
        """
        Return this:
        |date	                    |asset	|11D	    |factor	    |factor_quantile	|
        |---------------------------|-------|-----------|-----------|-------------------|
        |2014-01-08 00:00:00+00:00	|ARNC	|0.070159	|0.215274	|5                  |
        |                           |BA	    |-0.038556	|-1.638784	|1                  |
        :param period: forward return periods
        :param quantiles: number of quantile
        :param filter_zscore: drop extreme factor return, for stability of the analysis.
        """
        pass
