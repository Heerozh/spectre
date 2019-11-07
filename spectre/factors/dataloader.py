import pandas as pd
from os import path
import glob
from zipfile import ZipFile
# from numba import jit, njit, prange, vectorize


class DataLoader:
    def __init__(self, calender_assert=None,
                 ohlcv=('open', 'high', 'low', 'close', 'volume')) -> None:
        self._ohlcv = ohlcv
        self._calender = calender_assert

    def get_ohlcv_names(self):
        return self._ohlcv

    @classmethod
    def _backward_date(cls, index, date, backward):
        start_slice = index.get_loc(date, 'bfill')
        start_slice = max(start_slice - backward, 0)
        start_slice = index[start_slice]
        return start_slice

    # @staticmethod
    # @njit(parallel=True)
    # def _np_div(v1, v2):
    #     return v1 / v2

    def _adjust_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'price_multi' not in df:
            return df
        # 这也就稍微慢一点2.19
        price_multi = df.price_multi / df.price_multi.groupby(level=1).transform('last')
        vol_multi = df.vol_multi / df.vol_multi.groupby(level=1).transform('last')
        # 下面的代码2.16
        # price_multi = QuandlLoader._np_div(
        #     df.price_multi.values,  df.price_multi.groupby(level=1).transform('last').values)
        # vol_multi = QuandlLoader._np_div(
        #     df.vol_multi.values, df.vol_multi.groupby(level=1).transform('last').values)

        # adjust price
        rtn = df[list(self._ohlcv[:4])].mul(price_multi, axis=0)
        rtn[self._ohlcv[4]] = df[self._ohlcv[4]].mul(vol_multi, axis=0)
        # print(time.time() - s)
        return rtn

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
                      TODO all those column will be adjust by Dividends/Splits(anchor at `end` time)
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

    def _load(self, start, end):
        pattern = path.join(self._csv_dir, '*.csv')
        files = glob.glob(pattern)
        dfs = {path.basename(fn)[:-4].upper(): pd.read_csv(fn, **self._read_csv) for fn in files}
        return dfs

    def _load_div_split(self):
        # todo
        pass

    def _load_from_cache(self, start, end, backward):
        if self._cache[0] and start >= self._cache[0] and end <= self._cache[1]:
            start_slice = self._backward_date(self._cache[2].index.levels[0], start, backward)
            return self._adjust_prices(self._cache[2].loc[start_slice:end])
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
        df.sort_index(inplace=True)
        if self._calender:
            # drop the data of the non-trading day by calender,
            # because there may be some one-line junk data in non-trading day,
            # causing extra row of nan to all others assets.
            calender = df.loc[(slice(None), self._calender), :].index.get_level_values(0)
            df = df[df.index.get_level_values(0).isin(calender)]

        times = df.index.get_level_values(0)
        self._cache = (times[0], times[-1], df)
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
            with ZipFile(file) as pkg:
                with pkg.open(pkg.namelist()[0]) as csv:
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
            # drop raw dividend columns
            df.drop(['ex-dividend', 'split_ratio'], axis=1, inplace=True)

            df.to_hdf(file + '.cache.hdf', 'WIKI_PRICES')  # complevel=1 slow 3x
        # speed up string index column search time
        df = df.reset_index()
        asset_type = pd.api.types.CategoricalDtype(categories=pd.unique(df.asset), ordered=True)
        df.asset = df.asset.astype(asset_type)
        df.set_index(['date', 'asset'], inplace=True)
        df.sort_index(level=0, inplace=True)
        df = df.tz_localize('UTC', level=0, copy=False)
        if self._calender:
            calender = df.loc[(slice(None), self._calender), :].index.get_level_values(0)
            df = df[df.index.get_level_values(0).isin(calender)]
        self._cache = df

    def load(self, start, end, backward: int) -> pd.DataFrame:
        start_slice = self._backward_date(self._cache.index.levels[0], start, backward)
        return self._adjust_prices(self._cache.loc[start_slice:end])
