"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import os
import warnings

import pandas as pd

from .dataloader import DataLoader


class ArrowLoader(DataLoader):
    """Read from persistent data."""

    def __init__(self, path: str | None = None, keep_in_memory: bool = True) -> None:
        if not os.path.exists(path + ".meta"):
            raise FileNotFoundError(os.path.abspath(path + ".meta"))

        # pandas 0.22 has the fastest MultiIndex
        if pd.__version__.startswith("0.22"):
            import feather

            cols = feather.read_dataframe(path + ".meta")
        else:
            cols = pd.read_feather(path + ".meta")

        ohlcv = tuple(cols.ohlcv)
        adjustments = tuple(cols.adjustments.values[:2])
        # missing values may be read as None or NaN, depends on pandas version
        if pd.isna(adjustments[0]):
            adjustments = None
        super().__init__(path, ohlcv, adjustments)
        self.keep_in_memory = keep_in_memory
        self._cache = None
        self._filter = None

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
            warnings.warn(
                "You called `ingest()`, but `source` seems unchanged, "
                "no ingestion required. Set `force=True` to re-ingest.",
                RuntimeWarning,
            )
            return

        df = source.test_load()
        df = df.reset_index()
        df.to_feather(save_to)

        # don't use chained assignment like `meta.adjustments[:2] = ...` here, it does nothing
        # under the Copy-on-Write of pandas 3.
        meta = pd.DataFrame({"ohlcv": list(source.ohlcv)})
        adjustments = list(source.adjustments or [])
        meta["adjustments"] = adjustments + [None] * (len(meta) - len(adjustments))
        meta.to_feather(save_to + ".meta")

    def filter(self, func):
        self._filter = func
        self._cache = None

    def _load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache

        if pd.__version__.startswith("0.22"):
            import feather

            df = feather.read_dataframe(self._path)
        else:
            df = pd.read_feather(self._path)
        df = df.set_index(["date", "asset"])

        if self._filter is not None:
            df = self._filter(df)

        if self.keep_in_memory:
            self._cache = df
        return df
