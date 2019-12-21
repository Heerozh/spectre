"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Optional
import pandas as pd
import numpy as np
import os
import glob
from zipfile import ZipFile
import warnings


class DataLoader:
    def __init__(self, path: str, ohlcv=('open', 'high', 'low', 'close', 'volume'),
                 adjustments=('ex-dividend', 'split_ratio')) -> None:
        self._path = path
        self._ohlcv = ohlcv
        self._adjustments = adjustments

    @property
    def ohlcv(self):
        return self._ohlcv

    @property
    def adjustments(self):
        return self._adjustments

    @property
    def adjustment_multipliers(self):
        return ['price_multi', 'vol_multi']

    @property
    def time_category(self):
        return '_time_cat_id'

    @property
    def last_modified(self) -> float:
        """ data source last modification time """
        raise NotImplementedError("abstractmethod")

    @classmethod
    def _align_to(cls, df, calender_asset):
        """ helper method for align index """
        index = df.loc[(slice(None), calender_asset), :].index.get_level_values(0)
        df = df[df.index.get_level_values(0).isin(index)]
        df.index = df.index.remove_unused_levels()
        return df

    def _format(self, df, split_ratio_is_inverse=False) -> pd.DataFrame:
        """
        Format the data as we want it. df index must be in order [datetime, asset_name]
        * change index name to ['date', 'asset']
        * change asset column type to category
        * covert date index to utc timezone
        * create time_cat column
        * create adjustment multipliers columns
        """
        df = df.rename_axis(['date', 'asset'])
        # speed up asset index search time
        df = df.reset_index()
        asset_type = pd.api.types.CategoricalDtype(categories=pd.unique(df.asset), ordered=True)
        df.asset = df.asset.astype(asset_type)
        # format index and convert to utc timezone-aware
        df.set_index(['date', 'asset'], inplace=True)
        if df.index.levels[0].tzinfo is None:
            df = df.tz_localize('UTC', level=0, copy=False)
        else:
            df = df.tz_convert('UTC', level=0, copy=False)
        df.sort_index(level=[0, 1], inplace=True)
        # generate time key for parallel
        date_index = df.index.get_level_values(0)
        unique_date = date_index.unique()
        time_cat = dict(zip(unique_date, range(len(unique_date))))
        cat = np.fromiter(map(lambda x: time_cat[x], date_index), dtype=np.int)
        df[self.time_category] = cat

        # Process dividends and split
        if self.adjustments is not None:
            div_col = self.adjustments[0]
            spr_col = self.adjustments[1]
            close_col = self.ohlcv[3]
            price_multi_col = self.adjustment_multipliers[0]
            vol_multi_col = self.adjustment_multipliers[1]
            if split_ratio_is_inverse:
                df[spr_col] = 1 / df[spr_col]

            # move ex-div up 1 row
            groupby = df.groupby(level=1)
            last = pd.DataFrame.last_valid_index
            ex_div = groupby[div_col].shift(-1)
            ex_div.loc[groupby.apply(last)] = 0
            sp_rto = groupby[spr_col].shift(-1)
            sp_rto.loc[groupby.apply(last)] = 1

            df[div_col] = ex_div
            df[spr_col] = sp_rto

            # generate dividend multipliers
            price_multi = (1 - ex_div / df[close_col]) * sp_rto
            price_multi = price_multi[::-1].groupby(level=1).cumprod()[::-1]
            df[price_multi_col] = price_multi.astype(np.float32)
            vol_multi = (1 / sp_rto)[::-1].groupby(level=1).cumprod()[::-1]
            df[vol_multi_col] = vol_multi.astype(np.float32)

        return df

    def _load(self) -> pd.DataFrame:
        """
        Return dataframe with multi-index ['date', 'asset']

        You need to call `self.test_load()` in your test case to check if the format
        you returned is correct.
        """
        raise NotImplementedError("abstractmethod")

    def test_load(self):
        """
        Basic test for the format returned by _load(),
        If you write your own Loader, call this method at your test case.
        """
        df = self._load()

        assert df.index.names == ['date', 'asset'], \
            "df.index.names should be ['date', 'asset'] "
        assert not any(df.index.duplicated()), \
            "There are duplicate indexes in df, you need handle them up."
        assert df.index.is_lexsorted(), \
            "df.index must be sorted, try using df.sort_index(level=0, inplace=True)"
        assert str(df.index.levels[0].tzinfo) == 'UTC', \
            "df.index.date must be UTC timezone."
        assert df.index.levels[-1].ordered, \
            "df.index.asset must ordered categorical."
        assert self.time_category in df, \
            "You must create a time_category column, convert time to category id"

        if self.adjustments:
            assert all(x in df for x in self.adjustments), \
                "Adjustments columns `{}` not found.".format(self.adjustments)
            assert all(x in df for x in self.adjustment_multipliers), \
                "Adjustment multipliers columns `{}` not found.".format(self.adjustment_multipliers)
            assert not any(df[self.adjustments[0]].isna()), \
                "There is nan value in ex-dividend column, should be filled with 0."
            assert not any(df[self.adjustments[1]].isna()), \
                "There is nan value in split_ratio column, should be filled with 1."

        return df

    def load(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp],
             backwards: int) -> pd.DataFrame:
        df = self._load()

        index = df.index.levels[0]

        if start is None:
            start = index[0]
        if end is None:
            end = index[-1]

        if index[0] > start:
            raise ValueError("`start` time cannot less than earliest time of data: {}."
                             .format(index[0]))
        if index[-1] < end:
            raise ValueError("`end` time cannot greater than latest time of data: {}."
                             .format(index[-1]))

        start_loc = index.get_loc(start, 'bfill')
        backward_loc = max(start_loc - backwards, 0)
        end_loc = index.get_loc(end, 'ffill')
        assert end_loc >= start_loc, 'There is no data between `start` and `end` date.'

        backward_start = index[backward_loc]
        return df.loc[backward_start:end]


class ArrowLoader(DataLoader):
    """ Read from persistent data. """

    def __init__(self, path: str = None, keep_in_memory: bool=True) -> None:
        cols = pd.read_feather(path + '.meta')
        ohlcv = cols.ohlcv.values
        adjustments = cols.adjustments.values[:2]
        super().__init__(path, ohlcv, adjustments)
        self.keep_in_memory = keep_in_memory
        self._cache = None

    @classmethod
    def _last_modified(cls, filepath) -> float:
        if not os.path.isfile(filepath):
            return 0
        else:
            return os.path.getmtime(filepath)

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

        df = pd.read_feather(self._path)
        df.set_index(['date', 'asset'], inplace=True)

        if self.keep_in_memory:
            self._cache = df
        return df


class CsvDirLoader(DataLoader):
    def __init__(self, prices_path: str, prices_by_year=False, earliest_date: pd.Timestamp = None,
                 dividends_path=None, splits_path=None, calender_asset: str = None,
                 ohlcv=('open', 'high', 'low', 'close', 'volume'), adjustments=None,
                 split_ratio_is_inverse=False,
                 prices_index='date', dividends_index='exDate', splits_index='exDate', **read_csv):
        """
        Load data from csv dir
        :param prices_path: prices csv folder, structured as one csv per stock.
            When encountering duplicate indexes data in `prices_index`, Loader will keep the last,
            drop others.
        :param prices_by_year: If price file name like 'spy_2017.csv', set this to True
        :param earliest_date: Data before this date will not be read, save memory
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
        :param calender_asset: asset name as trading calendar, like 'SPY', for clean up non-trading
            time data.
        :param ohlcv: Required, OHLCV column names. When you don't need to use `adjustments` and
            `factors.OHLCV`, you can set this to None.
        :param adjustments: Optional, `dividend amount` and `splits ratio` column names.
        :param split_ratio_is_inverse: If split ratio calculated by to/from, set to True.
            For example, 2-for-1 split, to/form = 2, 1-for-15 Reverse Split, to/form = 0.6666...
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
            "`index_col` for which csv? Use `prices_index` and `dividends_index` and " \
            "`splits_index` instead."
        self._adjustment_cols = adjustments
        self._split_ratio_is_inverse = split_ratio_is_inverse
        self._prices_by_year = prices_by_year
        self._earliest_date = earliest_date
        self._dividends_path = dividends_path
        self._splits_path = splits_path
        self._calender = calender_asset
        self._prices_index = prices_index
        self._dividends_index = dividends_index
        self._splits_index = splits_index
        self._read_csv = read_csv

    @property
    def last_modified(self) -> float:
        pattern = os.path.join(self._path, '*.csv')
        files = glob.glob(pattern)
        return max([os.path.getmtime(fn) for fn in files])

    def _walk_split_by_year_dir(self, csv_path, index_col):
        years = set(pd.date_range(self._earliest_date or 0, pd.Timestamp.now()).year)
        pattern = os.path.join(csv_path, '*.csv')
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
            return df[~df.index.duplicated(keep='last')]

        dfs = {symbol: multi_read_csv(file_list) for symbol, file_list in assets.items()}
        return dfs

    def _walk_dir(self, csv_path, index_col):
        pattern = os.path.join(csv_path, '*.csv')
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError("There are no files is {}".format(csv_path))

        def symbol(file):
            return os.path.basename(file)[:-4].upper()

        def read_csv(file):
            df = pd.read_csv(file, index_col=index_col, **self._read_csv)
            if len(df.index.dropna()) == 0:
                return None
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

        assert isinstance(df.index.levels[1], pd.DatetimeIndex), \
            "data must index by datetime, set correct `read_csv`, " \
            "for example index_col='date', parse_dates=True"

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
            df = df = self._align_to(df, self._calender)
        return df


class QuandlLoader(DataLoader):
    @property
    def last_modified(self) -> float:
        """ the quandl data is no longer updated, so return a fixed value """
        return 1

    def __init__(self, file: str, calender_asset='AAPL') -> None:
        """
        Usage:
        download data first:
        https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?qopts.export=true&api_key=[yourapi_key]
        then:
        loader = factors.QuandlLoader('./quandl/WIKI_PRICES.zip')
        """
        super().__init__(file,
                         ohlcv=('open', 'high', 'low', 'close', 'volume'),
                         adjustments=('ex-dividend', 'split_ratio'))
        self._calender = calender_asset

    def _load(self) -> pd.DataFrame:
        with ZipFile(self._path) as pkg:
            with pkg.open(pkg.namelist()[0]) as csv:
                df = pd.read_csv(csv, parse_dates=['date'],
                                 usecols=['ticker', 'date', 'open', 'high', 'low', 'close',
                                          'volume', 'ex-dividend', 'split_ratio', ],
                                 dtype={
                                     'open': np.float32, 'high': np.float32, 'low': np.float32,
                                     'close': np.float32, 'volume': np.float64,
                                     'ex-dividend': np.float64, 'split_ratio': np.float64
                                 })

        df.set_index(['date', 'ticker'], inplace=True)
        df.split_ratio.loc[("2001-09-12", 'GMT')] = 1  # fix nan
        df = self._format(df, split_ratio_is_inverse=True)
        if self._calender:
            df = self._align_to(df, self._calender)

        return df
