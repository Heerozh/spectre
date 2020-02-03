"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import pandas as pd
import os
import warnings
from .dataloader import DataLoader


class ArrowLoader(DataLoader):
    """ Read from persistent data. """

    def __init__(self, path: str = None, keep_in_memory: bool = True) -> None:
        # pandas 0.22 has the fastest MultiIndex
        if pd.__version__.startswith('0.22'):
            import feather
            cols = feather.read_dataframe(path + '.meta')
        else:
            cols = pd.read_feather(path + '.meta')

        ohlcv = cols.ohlcv.values
        adjustments = cols.adjustments.values[:2]
        if adjustments[0] is None:
            adjustments = None
        super().__init__(path, ohlcv, adjustments)
        self.keep_in_memory = keep_in_memory
        self._cache = None

    @classmethod
    def _last_modified(cls, file_path) -> float:
        if not os.path.isfile(file_path):
            return 0
        else:
            return os.path.getmtime(file_path)

    @property
    def last_modified(self) -> float:
        return self._last_modified(self._path)

    @classmethod
    def ingest(cls, source: DataLoader, save_to, force: bool = False) -> None:
        if not force and (source.last_modified <= cls._last_modified(save_to)):
            warnings.warn("You called `ingest()`, but `source` seems unchanged, "
                          "no ingestion required. Set `force=True` to re-ingest.",
                          RuntimeWarning)
            return

        df = source.test_load()
        df.reset_index(inplace=True)
        df.to_feather(save_to)

        meta = pd.DataFrame(columns=['ohlcv', 'adjustments'])
        meta.ohlcv = source.ohlcv
        meta.adjustments[:2] = source.adjustments
        meta.to_feather(save_to + '.meta')

    def _load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache

        if pd.__version__.startswith('0.22'):
            import feather
            df = feather.read_dataframe(self._path)
        else:
            df = pd.read_feather(self._path)
        df.set_index(['date', 'asset'], inplace=True)

        if self.keep_in_memory:
            self._cache = df
        return df
