import pandas as pd
import os
from zipfile import ZipFile
import numpy as np
from numba import jit, njit, prange, vectorize


class DataLoader:
    def __init__(self, calender_assert=None,
                 ohlcv=('open', 'high', 'low', 'close', 'volume')) -> None:
        self._ohlcv = ohlcv
        self._calender = calender_assert

    def get_ohlcv_names(self):
        return self._ohlcv

    def load(self, start: pd.Timestamp, end: pd.Timestamp, backward: int) -> pd.DataFrame:
        """
        If for back-testing, `start` `end` parameter has no meaning,
        because should be same as 'now'.
        """
        raise NotImplementedError("abstractmethod")

    # 应该有通用adj和sid方法


class CsvDirLoader(DataLoader):
    def __init__(self, path: str, calender_assert: str = None,
                 split_by_year=False, dividends_path='',
                 ohlcv=('open', 'high', 'low', 'close', 'volume'),
                 **read_csv):
        """
        Load data from csv dir, structured as xxx.csv per stock..
        :param calender_assert: assert name as trading calendar, like 'SPY'
        :param split_by_year: If file name like 'spy_2017.csv', set this to True
        :param ohlcv: OHLCV column names, If set, you can access those data like
                      `spectre.factors.OHLCV.open`, also
                      TODO all those column will be adjust by Dividends/Splits(anchor at `end` time)
        :param read_csv: Parameters for pd.read_csv.
        """
        super().__init__(calender_assert, ohlcv)
        self._csv_dir = path
        # dividends: self._div_dir = dividends_path #dividends_path='',
        self._split_by_year = split_by_year
        self._read_csv = read_csv

        self._cache = (None, None, None)

    def _load_split_by_year(self, start, end):
        years = set(pd.date_range(start, end).year)
        dfs = {}
        for entry in os.scandir(self._csv_dir):
            if entry.name.endswith(".csv") and entry.is_file():
                year, name = entry.name[-8:-4], entry.name[:-9]  # like 'spy_2011.csv'
                if int(year) not in years or name in dfs:
                    continue
                df = pd.DataFrame()
                for year in years:
                    try:
                        csv_path = '{}{}_{}.csv'.format(self._csv_dir, name, year)
                        df_year = pd.read_csv(csv_path, **self._read_csv)
                        df = pd.concat([df_year, df])
                    except OSError:
                        pass
                df.sort_index(inplace=True)
                assert isinstance(df.index, pd.DatetimeIndex), \
                    "data must index by datetime, set correct `read_csv`, " \
                    "for example index_col='date', parse_dates=True"
                df = df[~df.index.duplicated(keep='last')]
                if df.index.tzinfo is None:
                    df = df.tz_localize('UTC')
                else:
                    df = df.tz_convert('UTC')
                dfs[name] = df
        return dfs

    def _load(self, start, end):
        dfs = {}
        for entry in os.scandir(self._csv_dir):
            if entry.name.endswith(".csv") and entry.is_file():
                df = pd.read_csv(self._csv_dir + entry.name, **self._read_csv)
                assert isinstance(df.index, pd.DatetimeIndex), \
                    "data must index by datetime, set correct `read_csv`, " \
                    "for example index_col='date', parse_dates=True"
                if df.index.tzinfo is None:
                    df = df.tz_localize('UTC')
                else:
                    df = df.tz_convert('UTC')
                dfs[entry.name[:-4]] = df
        return dfs

    def _load_from_cache(self, start, end, backward):
        if self._cache[0] and start >= self._cache[0] and end <= self._cache[1]:
            index = self._cache[2].index.get_level_values(0).unique()
            start_slice = index.get_loc(start, 'bfill')
            start_slice = max(start_slice - backward, 0)
            start_slice = index[start_slice]
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
            dfs = self._load(start, end)

        df_concat = pd.concat(dfs).swaplevel(0, 1).sort_index(level=0)
        df_concat = df_concat.rename_axis(['date', 'asset'])
        if self._calender:
            # drop the data of the non-trading day by calender,
            # because there may be some one-line junk data in non-trading day,
            # causing extra row of nan to all others assets.
            calender = df_concat.loc[(slice(None), self._calender), :].index.get_level_values(0)
            df_concat = df_concat[df_concat.index.get_level_values(0).isin(calender)]
        times = df_concat.index.get_level_values(0)
        self._cache = (times[0], times[-1], df_concat)

        return self._load_from_cache(start, end, backward)


class QuandlLoader(DataLoader):
    def __init__(self, file: str, calender_assert='AAPL',
                 ohlcv=('open', 'high', 'low', 'close', 'volume')) -> None:
        """
        Usage:
        download data first:
        https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?qopts.export=true&api_key=[yourapi_key]
        then:
        loader = factors.QuandlLoader('./quandl/WIKI_PRICES.zip')
        """
        super().__init__(calender_assert, ohlcv)
        try:
            df = pd.read_hdf(file + '.cache.hdf', 'WIKI_PRICES')
        except FileNotFoundError:
            with ZipFile(file) as zip:
                with zip.open(zip.namelist()[0]) as csv:
                    df = pd.read_csv(csv, parse_dates=['date'],
                                     index_col=['date', 'ticker'],
                                     usecols=['ticker', 'date', 'open', 'high', 'low', 'close',
                                              'volume',
                                              'ex-dividend', 'split_ratio', ],
                                     )
            df = df.rename_axis(['date', 'asset'])
            df.sort_index(level=0, inplace=True)

            # move ex-div up 1 row
            ex_div = df.groupby(level=1)['ex-dividend'].shift(-1)
            ex_div.loc[ex_div.index.get_level_values(0)[-1]] = 0
            sp_rto = df.groupby(level=1)['split_ratio'].shift(-1)
            sp_rto.loc[sp_rto.index.get_level_values(0)[-1]] = 1

            # get dividend multipliers
            price_multi = (1 - ex_div / df['close']) * (1 / sp_rto)
            price_multi = price_multi[::-1].groupby(level=1).cumprod()[::-1]
            df['price_multi'] = price_multi
            vol_multi = sp_rto[::-1].groupby(level=1).cumprod()[::-1]
            df['vol_multi'] = vol_multi

            df.drop(['ex-dividend', 'split_ratio'], axis=1, inplace=True)

            df.to_hdf(file + '.cache.hdf', 'WIKI_PRICES')  # complevel=1 slow 3x

        df.tz_localize('UTC', level=0, copy=False)
        if self._calender:
            calender = df.loc[(slice(None), self._calender), :].index.get_level_values(0)
            df = df[df.index.get_level_values(0).isin(calender)]
        self._cache = df

    @staticmethod
    @njit(parallel=True)
    def _group_div(values, keys, last):
        for key in prange(last.shape[0]):
            values[keys == key] /= last[key]
        return values

    @staticmethod
    def adjust_prices(df):
        import time
        s = time.time()

        price_multi = df.price_multi.groupby(level=1).apply(lambda x: x / x[-1])
        vol_multi = df.vol_multi.groupby(level=1).apply(lambda x: x / x[-1])

        print(time.time() - s)

        # adjust price
        rtn = df[['open', 'high', 'low', 'close']].mul(price_multi, axis=0)
        rtn['volume'] = df['volume'].mul(vol_multi, axis=0)
        print(time.time() - s)
        return rtn

    def load(self, start, end, backward: int) -> pd.DataFrame:
        index = self._cache.index.get_level_values(0).unique()
        start_slice = index.get_loc(start, 'bfill')
        start_slice = max(start_slice - backward, 0)
        start_slice = index[start_slice]
        return self.adjust_prices(self._cache.loc[start_slice:end])
