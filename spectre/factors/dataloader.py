"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import pandas as pd
import numpy as np
from os import path
import glob
from zipfile import ZipFile


class DataLoader:
    def __init__(self, calender_assert=None,
                 ohlcv=('open', 'high', 'low', 'close', 'volume')) -> None:
        self._ohlcv = ohlcv
        self._calender = calender_assert

    def get_ohlcv_names(self):
        return self._ohlcv

    @classmethod
    def _backward_date(cls, index, start, end, backward):
        start_loc = index.get_loc(start, 'bfill')
        backward_loc = max(start_loc - backward, 0)
        end_loc = index.get_loc(end, 'ffill')
        assert end_loc >= start_loc, 'There is no data between `start` and `end` date.'
        return index[backward_loc]

    @classmethod
    def _generate_time_key(cls, df):
        date_index = df.index.get_level_values(0)
        unique_date = date_index.unique()
        time_cat = dict(zip(unique_date, range(len(unique_date))))
        cat = np.fromiter(map(lambda x: time_cat[x], date_index), dtype=np.int)
        df['time_cat_id'] = cat

    def load(self, start: pd.Timestamp, end: pd.Timestamp, backward: int) -> pd.DataFrame:
        """
        If for back-testing, `start` `end` parameter has no meaning,
        because should be same as 'now'.
        """
        raise NotImplementedError("abstractmethod")


class CsvDirLoader(DataLoader):
    def __init__(self, csv_path: str, calender_assert: str = None,
                 split_by_year=False, dividends_path='',
                 ohlcv=('open', 'high', 'low', 'close', 'volume'),
                 **read_csv):
        """
        Load data from csv dir, structured as xxx.csv per stock..
        :param calender_assert: assert name as trading calendar, like 'SPY'
        :param split_by_year: If file name like 'spy_2017.csv', set this to True
        :param ohlcv: OHLCV column names, If set, you can access those data like
                      `spectre.factors.OHLCV.open`, also
        :param read_csv: Parameters for pd.read_csv.
        """
        super().__init__(calender_assert, ohlcv)
        self._csv_dir = csv_path
        # dividends: self._div_dir = dividends_path #dividends_path='',
        self._split_by_year = split_by_year
        self._read_csv = read_csv

        self._cache = (None, None, None)

    def _load_split_by_year(self, start, end):
        years = set(pd.date_range(start, end).year)
        pattern = path.join(self._csv_dir, '*.csv')
        files = glob.glob(pattern)
        assets = {}
        for fn in files:
            base = path.basename(fn)
            symbol, year = base[:-9].upper(), int(base[-8:-4])  # like './spy_2011.csv'
            if year in years:
                if symbol in assets:
                    assets[symbol].append(fn)
                else:
                    assets[symbol] = [fn, ]

        def multi_read_csv(file_list):
            df = pd.concat([pd.read_csv(_fn, **self._read_csv) for _fn in file_list])
            return df[~df.index.duplicated(keep='last')]

        dfs = {symbol: multi_read_csv(file_list) for symbol, file_list in assets.items()}
        return dfs

    def _load(self):
        pattern = path.join(self._csv_dir, '*.csv')
        files = glob.glob(pattern)
        dfs = {path.basename(fn)[:-4].upper(): pd.read_csv(fn, **self._read_csv) for fn in files}
        return dfs

    def _load_div_split(self):
        # todo load div/split data from csv folder
        pass

    def _load_from_cache(self, start, end, backward):
        if self._cache[0] and start >= self._cache[0] and end <= self._cache[1]:
            index = self._cache[2].index.levels[0]
            start_slice = self._backward_date(index, start, end, backward)
            return self._cache[2].loc[start_slice:end]
        else:
            return None

    def load(self, start, end, backward) -> pd.DataFrame:
        ret = self._load_from_cache(start, end, backward)
        if ret is not None:
            return ret

        if self._split_by_year:
            dfs = self._load_split_by_year(start, end)
        else:
            dfs = self._load()

        df = pd.concat(dfs)
        assert isinstance(df.index.levels[1], pd.DatetimeIndex), \
            "data must index by datetime, set correct `read_csv`, " \
            "for example index_col='date', parse_dates=True"

        df = df.rename_axis(['asset', 'date'])
        # speed up string index column search time
        df = df.reset_index()
        asset_type = pd.api.types.CategoricalDtype(categories=pd.unique(df.asset), ordered=True)
        df.asset = df.asset.astype(asset_type)
        df.set_index(['date', 'asset'], inplace=True)
        if df.index.levels[0].tzinfo is None:
            df = df.tz_localize('UTC', level=0, copy=False)
        else:
            df = df.tz_convert('UTC', level=0, copy=False)
        df.sort_index(level=[0, 1], inplace=True)
        if self._calender:
            # drop the data of the non-trading day by calender,
            # because there may be some one-line junk data in non-trading day,
            # causing extra row of nan to all others assets.
            calender = df.loc[(slice(None), self._calender), :].index.get_level_values(0)
            df = df[df.index.get_level_values(0).isin(calender)]
            # if not remove will cause backward search get wrong result.
            df.index = df.index.remove_unused_levels()

        # pre time group id
        self._generate_time_key(df)

        times = df.index.get_level_values(0)
        self._cache = (times[0], times[-1], df)
        assert times[0] <= start, "`start` date less than actual data's first line."
        assert times[-1] >= end, "`end` date greater than actual data's last line."

        return self._load_from_cache(start, end, backward)


class QuandlLoader(DataLoader):
    @classmethod
    def _make_hdf(cls, file):
        with ZipFile(file) as pkg:
            with pkg.open(pkg.namelist()[0]) as csv:
                df = pd.read_csv(csv, parse_dates=['date'],
                                 usecols=['ticker', 'date', 'open', 'high', 'low', 'close',
                                          'volume', 'ex-dividend', 'split_ratio', ],
                                 dtype={
                                     'open': np.float32, 'high': np.float32, 'low': np.float32,
                                     'close': np.float32, 'volume': np.float64,
                                     'ex-dividend': np.float64, 'split_ratio': np.float64
                                 })
        df = df.rename(columns={'ticker': 'asset'})

        # speed up string index column search time
        asset_type = pd.api.types.CategoricalDtype(categories=pd.unique(df.asset), ordered=True)
        df.asset = df.asset.astype(asset_type)

        df.set_index(['date', 'asset'], inplace=True)
        df.sort_index(level=[0, 1], inplace=True)

        # move ex-div up 1 row
        groupby = df.groupby(level=1)
        last = pd.DataFrame.last_valid_index

        ex_div = groupby['ex-dividend'].shift(-1)
        ex_div.loc[groupby.apply(last)] = 0
        sp_rto = groupby['split_ratio'].shift(-1)
        sp_rto.loc[groupby.apply(last)] = 1

        # get dividend multipliers
        price_multi = (1 - ex_div / df['close']) * (1 / sp_rto)
        price_multi = price_multi[::-1].groupby(level=1).cumprod()[::-1]
        df['price_multi'] = price_multi.astype(np.float32)
        vol_multi = sp_rto[::-1].groupby(level=1).cumprod()[::-1]
        df['vol_multi'] = vol_multi.astype(np.float32)
        # drop raw dividend columns
        df.drop(['ex-dividend', 'split_ratio'], axis=1, inplace=True)

        # pre time group id
        cls._generate_time_key(df)

        df.to_hdf(file + '.cache.hdf', 'WIKI_PRICES', format='table')  # complevel=1 slow 3x
        return df

    def __init__(self, file: str, calender_assert='AAPL') -> None:
        """
        Usage:
        download data first:
        https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?qopts.export=true&api_key=[yourapi_key]
        then:
        loader = factors.QuandlLoader('./quandl/WIKI_PRICES.zip')
        """
        super().__init__(calender_assert, ohlcv=('open', 'high', 'low', 'close', 'volume'))
        try:
            df = pd.read_hdf(file + '.cache.hdf', 'WIKI_PRICES')
        except FileNotFoundError:
            df = self._make_hdf(file)

        df = df.tz_localize('UTC', level=0, copy=False)
        if self._calender:
            calender = df.loc[(slice(None), self._calender), :].index.get_level_values(0)
            df = df[df.index.get_level_values(0).isin(calender)]
            df.index = df.index.remove_unused_levels()
        self._cache = df

    def load(self, start, end, backward: int) -> pd.DataFrame:
        start_slice = self._backward_date(self._cache.index.levels[0], start, end, backward)
        return self._cache.loc[start_slice:end]
