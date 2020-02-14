"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import numpy as np
import pandas as pd
import os
import glob
import warnings
from .dataloader import DataLoader


class CsvDirLoader(DataLoader):
    def __init__(self, prices_path: str, prices_by_year=False, earliest_date: pd.Timestamp = None,
                 dividends_path=None, splits_path=None, file_pattern='*.csv',
                 calender_asset: str = None, align_by_time=False,
                 ohlcv=('open', 'high', 'low', 'close', 'volume'), adjustments=None,
                 split_ratio_is_inverse=False, split_ratio_is_fraction=False,
                 prices_index='date', dividends_index='exDate', splits_index='exDate', **read_csv):
        """
        Load data from csv dir
        :param prices_path: prices csv folder, structured as one csv per stock.
            When encountering duplicate indexes data in `prices_index`, Loader will keep the last,
            drop others.
        :param prices_by_year: If price file name like 'spy_2017.csv', set this to True
        :param earliest_date: Data before this date will not be saved to memory. Note: Use the
            same time zone as in the csv files.
        :param dividends_path: dividends csv folder, structured as one csv per stock.
            For duplicate data, loader will first drop the exact same rows, and then for the same
            `dividends_index` but different 'dividend amount' rows, loader will sum them up.
            If `dividends_path` not set, the `adjustments[0]` column is considered to be included
            in the prices csv.
        :param splits_path: splits csv folder, structured as one csv per stock,
            When encountering duplicate indexes data in `splits_index`, Loader will use the last
            non-NaN 'split ratio', drop others.
            If `splits_path` not set, the `adjustments[1]` column is considered to be included
            in the prices csv.
        :param file_pattern: csv file name pattern, default is '*.csv'.
        :param calender_asset: asset name as trading calendar, like 'SPY', for clean up non-trading
            time data.
        :param align_by_time: if True and `calender_asset` not None, df index will be the product of
            'date' and 'asset'.
        :param ohlcv: Required, OHLCV column names. When you don't need to use `adjustments` and
            `factors.OHLCV`, you can set this to None.
        :param adjustments: Optional, `dividend amount` and `splits ratio` column names.
        :param split_ratio_is_inverse: If split ratio calculated by to/from, set to True.
            For example, 2-for-1 split, to/form = 2, 1-for-15 Reverse Split, to/form = 0.6666...
        :param split_ratio_is_fraction: If split ratio in csv is fraction string, set to True.
        :param prices_index: `index_col`for csv in prices_path
        :param dividends_index: `index_col`for csv in dividends_path.
        :param splits_index: `index_col`for csv in splits_path.
        :param read_csv: Parameters for all csv when calling pd.read_csv.
        """
        if adjustments is None:
            super().__init__(prices_path, ohlcv, None)
        else:
            super().__init__(prices_path, ohlcv, ('ex-dividend', 'split_ratio'))

        assert 'index_col' not in read_csv, \
            "`index_col` cannot be used here. Use `prices_index` and `dividends_index` and " \
            "`splits_index` instead."
        if 'dtype' not in read_csv:
            warnings.warn("It is recommended to set the `dtype` parameter and use float32 whenever "
                          "possible. Example: dtype = {'Open': np.float32, 'High': np.float32, "
                          "'Low': np.float32, 'Close': np.float32, 'Volume': np.float64}",
                          RuntimeWarning)
        self._adjustment_cols = adjustments
        self._split_ratio_is_inverse = split_ratio_is_inverse
        self._split_ratio_is_fraction = split_ratio_is_fraction
        self._prices_by_year = prices_by_year
        self._earliest_date = earliest_date
        self._dividends_path = dividends_path
        self._splits_path = splits_path
        self._file_pattern = file_pattern
        self._calender = calender_asset
        self._prices_index = prices_index
        self._dividends_index = dividends_index
        self._splits_index = splits_index
        self._read_csv = read_csv
        self._align_by_time = align_by_time

    @property
    def last_modified(self) -> float:
        pattern = os.path.join(self._path, self._file_pattern)
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError("Dir '{}' does not contains any csv files.".format(self._path))
        return max([os.path.getmtime(fn) for fn in files])

    def _walk_split_by_year_dir(self, csv_path, index_col):
        years = set(pd.date_range(self._earliest_date or 0, pd.Timestamp.now()).year)
        pattern = os.path.join(csv_path, self._file_pattern)
        files = glob.glob(pattern)
        assets = {}
        for fn in files:
            base = os.path.basename(fn)
            symbol, year = base[:-9].upper(), int(base[-8:-4])  # like 'spy_2011.csv'
            if year in years:
                if symbol in assets:
                    assets[symbol].append(fn)
                else:
                    assets[symbol] = [fn, ]

        def multi_read_csv(file_list):
            df = pd.concat([pd.read_csv(_fn, index_col=index_col, **self._read_csv)
                            for _fn in file_list])
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "df must index by datetime, set correct `read_csv`, "
                    "for example index_col='date', parse_dates=True. "
                    "For mixed-timezone like daylight saving time, "
                    "set date_parser=lambda col: pd.to_datetime(col, utc=True)")

            return df[~df.index.duplicated(keep='last')]

        dfs = {symbol: multi_read_csv(file_list) for symbol, file_list in assets.items()}
        return dfs

    def _walk_dir(self, csv_path, index_col):
        pattern = os.path.join(csv_path, self._file_pattern)
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError("There are no files is {}".format(csv_path))

        def symbol(file):
            return os.path.basename(file)[:-4].upper()

        def read_csv(file):
            df = pd.read_csv(file, index_col=index_col, **self._read_csv)
            if len(df.index.dropna()) == 0:
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "df must index by datetime, set correct `read_csv`, "
                    "for example parse_dates=True. "
                    "For mixed-timezone like daylight saving time, "
                    "set date_parser=lambda col: pd.to_datetime(col, utc=True)")
            return df[self._earliest_date:]

        dfs = {symbol(fn): read_csv(fn) for fn in files}
        return dfs

    def _load(self):
        if self._prices_by_year:
            dfs = self._walk_split_by_year_dir(self._path, self._prices_index)
        else:
            dfs = self._walk_dir(self._path, self._prices_index)
        dfs = {k: v[~v.index.duplicated(keep='last')] for k, v in dfs.items() if v is not None}
        df = pd.concat(dfs, sort=False)
        df = df.rename_axis(['asset', 'date'])

        if self._dividends_path is not None:
            dfs = self._walk_dir(self._dividends_path, self._dividends_index)
            ex_div_col = self._adjustment_cols[0]
            div_index = self._dividends_index

            def _agg_duplicated(_df):
                if _df is None or ex_div_col not in _df:
                    return None
                _df = _df.reset_index().drop_duplicates()
                _df = _df.dropna(subset=[ex_div_col])
                _df = _df.set_index(div_index)[ex_div_col]
                return _df.groupby(level=0).agg('sum')

            dfs = {k: _agg_duplicated(v) for k, v in dfs.items()}
            div = pd.concat(dfs, sort=False)
            div = div.reindex(df.index)
            div = div.fillna(0)
            div.name = self._adjustments[0]
            # div = df.rename_axis(['asset', 'date'])
            df = pd.concat([df, div], axis=1)

        if self._splits_path is not None:
            dfs = self._walk_dir(self._splits_path, self._splits_index)
            sp_rto_col = self._adjustment_cols[1]

            def _drop_na_and_duplicated(_df):
                if _df is None or sp_rto_col not in _df:
                    return None
                _df = _df.dropna(subset=[sp_rto_col])[sp_rto_col]
                return _df[~_df.index.duplicated(keep='last')]

            dfs = {k: _drop_na_and_duplicated(v) for k, v in dfs.items()}
            splits = pd.concat(dfs, sort=False)
            if self._split_ratio_is_fraction:
                from fractions import Fraction

                def fraction_2_float(x):
                    try:
                        return float(Fraction(x))
                    except (ValueError, ZeroDivisionError):
                        return np.nan

                splits = splits.apply(fraction_2_float)
            splits = splits.reindex(df.index)
            splits = splits.fillna(1)
            splits.name = self._adjustments[1]
            df = pd.concat([df, splits], axis=1)

        df = df.swaplevel(0, 1).sort_index(level=0)

        df = self._format(df, self._split_ratio_is_inverse)
        if self._calender:
            # drop the data of the non-trading day by calender,
            # because there may be some one-line junk data in non-trading day,
            # causing extra row of nan to all others assets.
            df = self._align_to(df, self._calender, self._align_by_time)
        return df
