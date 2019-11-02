from typing import Union, Optional, Iterable
from .factor import BaseFactor, DataFactor, FilterFactor
from .dataloader import DataLoader
import pandas as pd
import numpy as np


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
        self._factors = {}
        self._filter = None
        self._cuda = False

    def get_loader_data(self) -> pd.DataFrame:
        return self._dataframe

    def add(self, factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Union[Iterable[str], str]) -> None:
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

    def to_cuda(self) -> None:
        self._cuda = True

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> pd.DataFrame:
        if len(self._factors) == 0:
            raise ValueError('Please add at least one factor to engine, then run again.')
        start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
        # make columns to data factors.
        OHLCV.open.inputs = (self._loader.get_ohlcv_names()[0],)
        OHLCV.high.inputs = (self._loader.get_ohlcv_names()[1],)
        OHLCV.low.inputs = (self._loader.get_ohlcv_names()[2],)
        OHLCV.close.inputs = (self._loader.get_ohlcv_names()[3],)
        OHLCV.volume.inputs = (self._loader.get_ohlcv_names()[4],)

        # Calculate data that requires backward in tree
        max_backward = max([f._get_total_backward() for f in self._factors.values()])
        if self._filter:
            max_backward = max(max_backward, self._filter._get_total_backward())
        # Get data
        self._dataframe = self._loader.load(start, end, max_backward)
        # todo if cuda, copy _dataframe to gpu, and return object

        # compute
        if self._filter:
            self._filter._pre_compute(self, start, end)
        for f in self._factors.values():
            f._pre_compute(self, start, end)

        # Compute factors
        ret = pd.DataFrame(index=self._dataframe.index.copy())
        for c, f in self._factors.items():
            factor_data = f._compute()
            if isinstance(factor_data, pd.DataFrame):
                factor_data = factor_data.stack()
            else:
                factor_data = np.hstack(factor_data)
            ret[c] = factor_data

        # Remove filter False rows
        if self._filter:
            filter_data = self._filter._compute()
            if isinstance(filter_data, pd.DataFrame):
                filter_data = filter_data.stack()
            else:
                filter_data = np.hstack(filter_data)
            filter_data = filter_data.reindex_like(ret)
            ret = ret[filter_data]

        return ret.loc[start:end]

    def get_price_matrix(self, start: Union[str, pd.Timestamp],
                         end: Union[str, pd.Timestamp], prices=OHLCV.close) -> pd.DataFrame:
        backup = self._factors
        self._factors = {'price': prices}
        ret = self.run(start, end)
        self._factors = backup
        return ret['price'].unstack(level=[1])
