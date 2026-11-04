"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(
        self,
        path: str,
        ohlcv=("open", "high", "low", "close", "volume"),
        adjustments=("ex-dividend", "split_ratio"),
    ) -> None:
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
        return ["price_multi", "vol_multi"]

    @property
    def time_category(self):
        return "_time_cat_id"

    @property
    def last_modified(self) -> float:
        """data source last modification time"""
        raise NotImplementedError("abstractmethod")

    @property
    def min_timedelta(self) -> pd.Timedelta:
        """Minimum time delta of date index"""
        date_idx = self.load().index.levels[0]
        return min(date_idx[1:] - date_idx[:-1])

    @staticmethod
    def _attach_events(events: pd.Series, bars: pd.MultiIndex, how: str) -> pd.Series:
        """
        Reindex `events`, indexed by (asset, ex-date), to `bars`, indexed by (date, asset) in the
        same timezone. An ex-date has no bar on a halt, or when the bars are intraday and the
        ex-dates are days, so an event goes to the first bar of its asset on or after the ex-date,
        moving it to an earlier bar would be look-ahead. Events sharing a bar are merged by `how`,
        a name of a groupby aggregation. An event after the last bar of its asset is dropped, no
        bars are after it.
        """
        dates = bars.get_level_values(0)
        left = pd.DataFrame(
            {
                "asset": events.index.get_level_values(0),
                "ex_date": events.index.get_level_values(1).as_unit(dates.unit),
                "value": events.to_numpy(),
            }
        )
        left = left.dropna(subset="ex_date").sort_values("ex_date")
        right = pd.DataFrame({"asset": bars.get_level_values(1), "date": dates})
        attached = pd.merge_asof(
            left,
            right.sort_values("date"),
            left_on="ex_date",
            right_on="date",
            by="asset",
            direction="forward",
        )
        return attached.groupby(["date", "asset"])["value"].agg(how).reindex(bars)

    def _align_to(self, df, calender_asset, align_by_time=False):
        """
        helper method for align index. A dividend or a split on a row dropped by the calendar still
        happened, so it goes to the next kept bar of its asset.
        """
        index = df.loc[(slice(None), calender_asset), :].index.get_level_values(0)
        kept = df.index.get_level_values(0).isin(index)
        dropped, df = df[~kept], df[kept]
        df.index = df.index.remove_unused_levels()
        if self.adjustments is not None:
            div_col, spr_col = self.adjustments
            if div_col in df:
                moved = self._attach_events(dropped[div_col].swaplevel(), df.index, "sum")
                df[div_col] = df[div_col].add(moved, fill_value=0)
            if spr_col in df:
                moved = self._attach_events(dropped[spr_col].swaplevel(), df.index, "prod")
                df[spr_col] = df[spr_col].mul(moved, fill_value=1)
        if align_by_time:
            df = df.reindex(pd.MultiIndex.from_product(df.index.levels))
            # df = df.unstack(level=1).stack(dropna=False) 这个速度快，但新版本取消掉了

            def trim_nans(x):
                dts = x.index.get_level_values(0)
                first = x.first_valid_index()
                last = x.last_valid_index()
                if first is None:
                    return None
                mask = (dts >= first[0]) & (dts <= last[0])
                return x[mask]

            df = df.groupby(level=1, group_keys=False).apply(trim_nans)
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
        # print(pd.Timestamp.now(), 'Formatting index...')
        df = df.rename_axis(["date", "asset"])
        # speed up asset index search time
        df = df.reset_index()
        # categories=None: categories are inferred from the sorted unique values
        asset_type = pd.api.types.CategoricalDtype(categories=None, ordered=True)
        df.asset = df.asset.astype(asset_type)
        # format index and convert to utc timezone-aware
        df = df.set_index(["date", "asset"])
        if df.index.levels[0].tzinfo is None:
            df = df.tz_localize("UTC", level=0)
        else:
            df = df.tz_convert("UTC", level=0)
        df = df.sort_index(level=[0, 1])
        # generate time key for parallel
        # print(pd.Timestamp.now(), 'Formatting time key for gpu sorting...')
        date_index = df.index.get_level_values(0)
        unique_date = date_index.unique()
        time_cat = dict(zip(unique_date, range(len(unique_date))))
        # cat = np.fromiter(map(lambda x: time_cat[x], date_index), dtype=int)
        df[self.time_category] = date_index.map(time_cat)
        # print(pd.Timestamp.now(), 'Done.')

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
            groupby = df.groupby(level=1, observed=False)
            last = pd.DataFrame.last_valid_index
            ex_div = groupby[div_col].shift(-1)
            ex_div.loc[groupby.apply(last)] = 0
            sp_rto = groupby[spr_col].shift(-1)
            sp_rto.loc[groupby.apply(last)] = 1

            df[div_col] = ex_div
            df[spr_col] = sp_rto

            # generate dividend multipliers
            vol_multi = (1 / sp_rto)[::-1].groupby(level=1, observed=False).cumprod()[::-1]
            # A dividend is divided by the last close before its ex-date. That close can be some
            # bars back, so bring it to the scale of the ex-date, there may be a split in between.
            split_adjusted = df[close_col].where(df[close_col] > 0) / vol_multi
            last_close = split_adjusted.groupby(level=1, observed=False).ffill() * vol_multi
            # no last close means no earlier price to adjust
            price_multi = (1 - ex_div / last_close).fillna(1) * sp_rto
            price_multi = price_multi[::-1].groupby(level=1, observed=False).cumprod()[::-1]
            df[price_multi_col] = price_multi.astype(np.float32)
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

        assert df.index.names == ["date", "asset"], "df.index.names should be ['date', 'asset'] "
        assert not any(df.index.duplicated()), (
            "There are duplicate indexes in df, you need handle them up."
        )
        assert df.index.is_monotonic_increasing, (
            "df.index must be sorted, try using df = df.sort_index(level=0)"
        )
        assert str(df.index.levels[0].tzinfo) == "UTC", "df.index.date must be UTC timezone."
        assert df.index.levels[-1].ordered, "df.index.asset must ordered categorical."
        assert self.time_category in df, (
            "You must create a time_category column, convert time to category id"
        )

        if self.adjustments:
            assert all(x in df for x in self.adjustments), (
                f"Adjustments columns `{self.adjustments}` not found."
            )
            assert all(x in df for x in self.adjustment_multipliers), (
                f"Adjustment multipliers columns `{self.adjustment_multipliers}` not found."
            )
            assert not any(df[self.adjustments[0]].isna()), (
                "There is nan value in ex-dividend column, should be filled with 0."
            )
            assert not any(df[self.adjustments[1]].isna()), (
                "There is nan value in split_ratio column, should be filled with 1."
            )
            assert not df[self.adjustment_multipliers].isna().any(axis=None), (
                "There is nan value in adjustment multipliers columns, the prices before it would "
                "lose the dividends and splits after it."
            )
            assert not any(df[self.time_category].isna()), (
                "There is nan value in time_category column, should be filled with time id."
            )
        return df

    def load(
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, backwards: int = 0
    ) -> pd.DataFrame:
        """
        Rows from `start` to `end`, and the `backwards` bars of each asset before `start`. A window
        counts the bars of its asset, so an asset with gaps reaches back further than the dates.
        Only that asset has rows on those older dates, a cross-section of them is not complete.
        An asset without a row in the last `backwards` dates or in the range is left out.
        """
        df = self._load()

        index = df.index.levels[0]

        if start is None:
            start = index[0]
        if end is None:
            end = index[-1]

        if index[0] > start:
            raise ValueError(
                f"`start` time ({start}) cannot be less than earliest time of data: {index[0]}."
            )

        if index[-1] < end:
            raise ValueError(
                f"`end` time ({end}) cannot be greater than latest time of data: {index[-1]}."
            )

        start_loc = index.get_indexer([start], method="bfill")[0]
        backward_loc = max(start_loc - backwards, 0)
        end_loc = index.get_indexer([end], method="ffill")[0]
        assert end_loc >= start_loc, "There is no data between `start` and `end`."

        backward_start = index[backward_loc]
        rows = df.loc[backward_start:end]
        if backwards == 0 or backward_loc == 0:
            return rows

        dates, assets = df.index.codes
        first_row, start_row = dates.searchsorted(backward_loc), dates.searchsorted(start_loc)
        end_row = first_row + len(rows)
        asset_count = len(df.index.levels[1])
        in_rows = np.bincount(assets[first_row:end_row], minlength=asset_count) > 0
        bars = np.bincount(assets[first_row:start_row], minlength=asset_count)
        missing = np.where(in_rows, backwards - bars, 0)
        older = np.flatnonzero(missing[assets[:first_row]] > 0)
        if len(older) == 0:
            return rows
        older_assets = assets[older]
        from_last = pd.Series(older_assets).groupby(older_assets).cumcount(ascending=False)
        older = older[from_last.to_numpy() < missing[older_assets]]
        return df.iloc[np.concatenate((older, np.arange(first_row, end_row)))]


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
                raise KeyError(f"{asset} not found")
            if cursor_index[i] != asset_id:
                raise KeyError(f"{asset} not found")
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
        row_slice = slice(idx.searchsorted(start), idx.searchsorted(stop, side="right"))
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
