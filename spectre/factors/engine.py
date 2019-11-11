from typing import Union, Optional, Iterable
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
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._datagroup = None
        self._factors = {}
        self._filter = None
        self._device = None

    def get_data_tensor(self, column) -> torch.Tensor:
        series = self._dataframe[column]
        data = torch.tensor(series.values).pin_memory()
        data = self._datagroup.split(data)
        return data.to(self._device, non_blocking=True)

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

    def remove_all(self) -> None:
        self._factors = {}

    def get_device(self):
        return self._device

    def to_cuda(self) -> None:
        self._device = torch.device('cpu')

    def to_cpu(self) -> None:
        self._device = torch.device('cuda')

    def _paper_tensor(self, start, end, max_backward):
        # Get data
        self._dataframe = self._loader.load(start, end, max_backward)
        from datetime import datetime, timezone
        assert self._dataframe.index.is_lexsorted(), \
            "In the df returned by DateLoader, the index must be sorted, "\
            "try using df.sort_index(level=0, inplace=True)"
        assert str(self._dataframe.index.levels[0].tzinfo) == 'UTC', \
            "In the df returned by DateLoader, the date index must be UTC timezone."
        assert self._dataframe.index.levels[-1].ordered, \
            "In the df returned by DateLoader, the asset index must ordered categorical."
        # todo if cuda, copy _dataframe to gpu, and return object
        cat = self._dataframe .index.get_level_values(1).codes
        cat = cat + np.arange(0, len(cat))/len(cat)
        key = torch.tensor(cat, device=self._device).pin_memory()
        self._datagroup = ParallelGroupBy(key)

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """
        Compute factors and filters, return a df contains all.
        """
        if len(self._factors) == 0:
            raise ValueError('Please add at least one factor to engine, then run again.')
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        # make columns to data factors.
        OHLCV.open.inputs = (self._loader.get_ohlcv_names()[0],)
        OHLCV.high.inputs = (self._loader.get_ohlcv_names()[1],)
        OHLCV.low.inputs = (self._loader.get_ohlcv_names()[2],)
        OHLCV.close.inputs = (self._loader.get_ohlcv_names()[3],)
        OHLCV.volume.inputs = (self._loader.get_ohlcv_names()[4],)

        # Calculate data that requires backward in tree
        max_backward = max([f.get_total_backward_() for f in self._factors.values()])
        if self._filter:
            max_backward = max(max_backward, self._filter.get_total_backward_())
        # Get data
        self._paper_tensor(start, end, max_backward)

        # compute
        if self._filter:
            self._filter.pre_compute_(self, start, end)
        for f in self._factors.values():
            f.pre_compute_(self, start, end)

        # Compute factors
        ret = pd.DataFrame(index=self._dataframe.index.copy())
        ret = ret.assign(**{c: f._stack_compute() for c, f in self._factors.items()})

        # Remove filter False rows
        if self._filter:
            filter_data = self._filter._stack_compute()
            filter_data = filter_data.reindex_like(ret)
            ret = ret[filter_data]

        return ret.loc[start:end]

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
        self._factors = factors_backup
        self._filter = filter_backup
        return ret['price'].unstack(level=[1]).shift(-forward)

    def get_factor_return(self, period=(1, 5, 10), quantiles=5, filter_zscore=20):
        """
        :param filter_zscore: drop extreme factor return, for stability of the analysis.
        Return format:
        |date	                    |asset	|11D	    |factor	    |factor_quantile	|
        |---------------------------|-------|-----------|-----------|-------------------|
        |2014-01-08 00:00:00+00:00	|ARNC	|0.070159	|0.215274	|5                  |
        |                           |BA	    |-0.038556	|-1.638784	|1                  |
        """
        pass
