"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Optional
import pandas as pd
import numpy as np


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
    def _align_to(cls, df, calender_asset, align_by_time=False):
        """ helper method for align index """
        index = df.loc[(slice(None), calender_asset), :].index.get_level_values(0)
        df = df[df.index.get_level_values(0).isin(index)]
        df.index = df.index.remove_unused_levels()
        if align_by_time:
            df = df.reindex(pd.MultiIndex.from_product(df.index.levels))
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
        asset_type = pd.api.types.CategoricalDtype(categories=pd.unique(df.asset).sort(),
                                                   ordered=True)
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

    def load(self, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
             backwards: int = 0) -> pd.DataFrame:
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
        assert end_loc >= start_loc, 'There is no data between `start` and `end`.'

        backward_start = index[backward_loc]
        return df.loc[backward_start:end]


class DataLoaderFastGetter:
    """Fast get method for dataloader's DataFrame"""
    class DictLikeCursor:
        def __init__(self, parent, row_slice, column_id):
            self.parent = parent
            self.row_slice = row_slice
            self.data = parent.raw_data[row_slice, column_id]
            self.index = parent.asset_index[row_slice]
            self.length = len(self.index)

        def get_datetime_index(self):
            return self.parent.indexes[0][self.row_slice]

        def __getitem__(self, asset):
            asset_id = self.parent.asset_to_code[asset]
            cursor_index = self.index
            i = cursor_index.searchsorted(asset_id)
            if i >= self.length:
                raise KeyError('{} not found'.format(asset))
            if cursor_index[i] != asset_id:
                raise KeyError('{} not found'.format(asset))
            return self.data[i]

        def items(self):
            idx = self.index
            code_to_asset = self.parent.code_to_asset
            for i in range(self.length):
                code = idx[i]
                name = code_to_asset[code]
                yield name, self.data[i]

        def get(self, asset, default=None):
            try:
                return self[asset]
            except KeyError:
                return default

    def __init__(self, df):
        cat = df.index.get_level_values(1)

        self.source = df
        self.raw_data = df.values
        self.columns = df.columns
        self.indexes = [df.index.get_level_values(0), cat]
        self.asset_index = cat.codes
        self.asset_to_code = {v: k for k, v in enumerate(cat.categories)}
        self.code_to_asset = dict(enumerate(cat.categories))
        self.last_row_slice = None

    def get_slice(self, start, stop):
        if isinstance(start, slice):
            return start
        idx = self.indexes[0]
        stop = stop or start
        row_slice = slice(idx.searchsorted(start), idx.searchsorted(stop, side='right'))
        return row_slice

    def get_as_dict(self, start, stop=None, column_id=slice(None)):
        row_slice = self.get_slice(start, stop)
        cur = self.DictLikeCursor(self, row_slice, column_id)
        self.last_row_slice = row_slice
        return cur

    def get_as_df(self, start, stop=None):
        """550x faster than .loc[], 3x faster than .iloc[]"""
        row_slice = self.get_slice(start, stop)
        data = self.raw_data[row_slice]
        index = self.indexes[1][row_slice]
        self.last_row_slice = row_slice
        return pd.DataFrame(data, index=index, columns=self.columns)
