from typing import Union, Optional, Iterable
from .factor import BaseFactor, DataFactor, FilterFactor
from .dataloader import DataLoader
import pandas as pd


class OHLCV:
    open = DataFactor()
    high = DataFactor()
    low = DataFactor()
    close = DataFactor()
    volume = DataFactor()


class FactorEngine:
    """
    Engine for compute factors, used for back-testing and alpha-research both.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._factors = {}
        self._filters = {}

    def get_loader_data(self):
        return self._dataframe

    def add(self, factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Union[Iterable[str], str]) -> None:
        if isinstance(factor, Iterable):
            for i, fct in enumerate(factor):
                self.add(fct, name and name[i] or None)
        else:
            if isinstance(factor, FilterFactor):
                if name in self._factors:
                    raise KeyError('A filter with the name {} already exists.'
                                   'please specify a new name by engine.add(factor, new_name)'
                                   .format(name))
                self._filters[name] = factor
            else:
                if name in self._factors:
                    raise KeyError('A factor with the name {} already exists.'
                                   'please specify a new name by engine.add(factor, new_name)'
                                   .format(name))
                self._factors[name] = factor

    def run(self, start: Optional[any], end: Optional[any]) -> pd.DataFrame:
        # make columns to data factors.
        OHLCV.open.inputs = (self._loader.get_ohlcv_names()[0],)
        OHLCV.high.inputs = (self._loader.get_ohlcv_names()[1],)
        OHLCV.low.inputs = (self._loader.get_ohlcv_names()[2],)
        OHLCV.close.inputs = (self._loader.get_ohlcv_names()[3],)
        OHLCV.volume.inputs = (self._loader.get_ohlcv_names()[4],)

        # Calculate data that requires backward in tree
        max_backward = max(max([f._get_total_backward() for f in self._factors.values()] or (0,)),
                           max([f._get_total_backward() for f in self._filters.values()] or (0,)))
        # Get data
        self._dataframe = self._loader.load(start, end, max_backward)

        # compute
        for f in self._factors.values():
            f._pre_compute(self, start, end)
        for f in self._filters.values():
            f._pre_compute(self, start, end)

        # todo filter
        # remove false rows

        # Compute factors
        rtn = pd.DataFrame(index=self._dataframe.index.copy())
        for c, f in self._factors.items():
            rtn[c] = f._compute()

        return rtn
