"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

from .dataloader import DataLoader

_UTC_OFFSET = "_utc_offset"


def _ffill_zeros(df: pd.DataFrame, groups) -> pd.DataFrame:
    """
    Replace every 0 with the previous value of the same group (column by column), NaNs are kept
    as is, and leading zeros of each group stay 0. Rows of the same group must be contiguous, and
    oldest first, otherwise a value is filled from the future.
    Like `df.replace(to_replace=0, method='ffill')` (removed in pandas 3.0), but it doesn't fill
    values across groups.
    """
    df = df.copy()
    row_ids = np.arange(len(df))
    groups = np.asarray(groups)
    group_start = np.r_[True, groups[1:] != groups[:-1]]
    group_start = np.maximum.accumulate(np.where(group_start, row_ids, 0))
    for col in df.columns:
        values = df[col].to_numpy()
        # index of the last non-zero row
        last_valid = np.maximum.accumulate(np.where(values != 0, row_ids, 0))
        # no non-zero row in the current group yet, keep it
        last_valid = np.where(last_valid < group_start, row_ids, last_valid)
        df[col] = values[last_valid]
    return df


def _written_offsets(index: pd.Index) -> np.ndarray:
    """The UTC offsets at the end of each timestamp of `index`, 0 for one without an offset."""
    # few rows differ in their last characters, and a regex on every row of a file is slow
    rows, endings = pd.factorize(index.astype(str).str[-9:])
    parts = endings.str.extract(r"^[^+-]*([+-]\d\d)(?::?(\d\d))?$")
    offsets = pd.to_timedelta(parts[0] + ":" + parts[1].fillna("00") + ":00")
    return offsets.fillna(pd.Timedelta(0)).to_numpy()[rows]


def _local_times(df: pd.DataFrame) -> pd.DatetimeIndex:
    """The times of the index of `df` as they are written in its csv file."""
    if _UTC_OFFSET in df:
        return df.index.tz_localize(None) + df[_UTC_OFFSET].to_numpy()
    return df.index.tz_localize(None)


def _bar_offsets(df: pd.DataFrame) -> pd.Series:
    """
    The UTC offsets of the bars of an asset, for reading a date in its local time. A file holds
    more than one if it crosses a change of the daylight saving time. They are indexed by the local
    time of the last bar with that offset.
    """
    local_times = _local_times(df)
    offsets = pd.Series(local_times - df.index.tz_convert(None), index=local_times).sort_index()
    return offsets[offsets != offsets.shift(-1)]


def _concat_events(events: dict, tz, bar_offsets: dict) -> pd.Series:
    """
    Join {asset: events by ex-date, or None}, to a series indexed by (asset, ex-date). The ex-dates
    of each asset are converted to `tz`, the timezone of the bars, because the files can be in
    different UTC offsets, or have no timezone. `tz` None keeps them as they are, see `_walk_dir`.
    An ex-date without a timezone is a local date of its asset, so the UTC offset comes from
    `bar_offsets` of `_bar_offsets`: the one of its first bar on or after that date.
    """

    def in_tz(asset, series):
        ex_dates = series.index
        if tz is None:
            return series
        if ex_dates.tz is not None:
            return series.set_axis(ex_dates.tz_convert(tz))
        offsets = bar_offsets[asset]
        in_force = offsets.index.searchsorted(ex_dates).clip(max=len(offsets) - 1)
        utc_times = ex_dates - pd.TimedeltaIndex(offsets.to_numpy()[in_force])
        return series.set_axis(utc_times.tz_localize("UTC").tz_convert(tz))

    # the events of an asset without bars have nowhere to go
    events = {k: v for k, v in events.items() if v is not None and (tz is None or k in bar_offsets)}
    events = {asset: in_tz(asset, series) for asset, series in events.items()}
    if not events:
        index = pd.MultiIndex.from_arrays([pd.Index([], dtype=str), pd.DatetimeIndex([], tz=tz)])
        return pd.Series(index=index, dtype=np.float64)
    return pd.concat(events, sort=False)


class CsvDirLoader(DataLoader):
    def __init__(
        self,
        prices_path: str,
        prices_by_year=False,
        earliest_date: pd.Timestamp = None,
        dividends_path=None,
        splits_path=None,
        file_pattern="*.csv",
        calender_asset: str | None = None,
        align_by_time=False,
        ohlcv=("open", "high", "low", "close", "volume"),
        adjustments=None,
        split_ratio_is_inverse=False,
        split_ratio_is_fraction=False,
        prices_index="date",
        dividends_index="exDate",
        splits_index="exDate",
        **read_csv,
    ):
        """
        Load data from csv dir
        :param prices_path: prices csv folder, structured as one csv per stock.
            When encountering duplicate indexes data in `prices_index`, Loader will keep the last,
            drop others.
        :param prices_by_year: If price file name like 'spy_2017.csv', set this to True
        :param earliest_date: Data before this date will not be saved to memory. A date without
            timezone is on the clock of the prices csv of each stock.
        :param dividends_path: dividends csv folder, structured as one csv per stock.
            For duplicate data, loader will first drop the exact same rows, and then for the same
            `dividends_index` but different 'dividend amount' rows, loader will sum them up.
            A dividend goes to the first bar of the stock on or after `dividends_index`, and the
            ones sharing a bar are summed up. A `dividends_index` without timezone is a local date
            of that stock, read in the UTC offset of its prices around that date.
            If `dividends_path` not set, the `adjustments[0]` column is considered to be included
            in the prices csv.
        :param splits_path: splits csv folder, structured as one csv per stock,
            When encountering duplicate indexes data in `splits_index`, Loader will use the last
            non-NaN 'split ratio', drop others.
            A split goes to the first bar of the stock on or after `splits_index`, and the ones
            sharing a bar are multiplied. A `splits_index` without timezone is a local date of that
            stock, read in the UTC offset of its prices around that date.
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
            super().__init__(prices_path, ohlcv, ("ex-dividend", "split_ratio"))

        assert "index_col" not in read_csv, (
            "`index_col` cannot be used here. Use `prices_index` and `dividends_index` and "
            "`splits_index` instead."
        )
        if "dtype" not in read_csv:
            warnings.warn(
                "It is recommended to set the `dtype` parameter and use float32 whenever "
                "possible. Example: dtype = {'Open': np.float32, 'High': np.float32, "
                "'Low': np.float32, 'Close': np.float32, 'Volume': np.float64}",
                RuntimeWarning,
            )
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
            raise ValueError(f"Dir '{self._path}' does not contains any csv files.")
        return max([os.path.getmtime(fn) for fn in files])

    def _index_by_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        if not self._read_csv.get("parse_dates"):
            raise ValueError(
                "df must index by datetime, set correct `read_csv`, "
                "for example index_col='date', parse_dates=True."
            )
        # pandas leaves a column with several UTC offsets as it is, a file gets them by crossing a
        # change of the daylight saving time. Only UTC holds those rows, so keep their offsets.
        offsets = _written_offsets(df.index)
        return df.set_axis(pd.to_datetime(df.index, utc=True)).assign(**{_UTC_OFFSET: offsets})

    def _walk_split_by_year_dir(self, csv_path, index_col):
        years = range(pd.Timestamp(self._earliest_date or 0).year, pd.Timestamp.now().year + 1)
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
                    assets[symbol] = [
                        fn,
                    ]

        def multi_read_csv(file_list):
            df = pd.concat(
                [pd.read_csv(_fn, index_col=index_col, **self._read_csv) for _fn in file_list]
            )
            df = self._index_by_datetime(df)
            return df[~df.index.duplicated(keep="last")]

        dfs = {symbol: multi_read_csv(file_list) for symbol, file_list in assets.items()}
        return dfs

    def _walk_dir(self, csv_path, index_col, as_written=False):
        # `as_written` is for ex-dates on bars without a timezone, which have no instant to compare
        # with, so the index of each file becomes the local time written in it.
        pattern = os.path.join(csv_path, self._file_pattern)
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError(f"There are no files is {csv_path}")

        def symbol(file):
            return os.path.basename(file)[:-4].upper()

        def read_csv(file):
            df = pd.read_csv(file, index_col=index_col, **self._read_csv)
            if len(df.index.dropna()) == 0:
                return None
            df = self._index_by_datetime(df)
            return df.set_axis(_local_times(df)) if as_written else df

        dfs = {symbol(fn): read_csv(fn) for fn in files}
        return dfs

    def _from_earliest_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The rows of the prices of an asset from `earliest_date`, by a mask, because the result of a
        slice depends on the order of the rows. Ex-dates don't need this, an event before the first
        bar adjusts nothing, and they can have another timezone, or none.
        """
        if self._earliest_date is None:
            return df
        earliest_date = pd.Timestamp(self._earliest_date)
        if earliest_date.tz is None or df.index.tz is None:
            # one side has no instant, so it's the time as written, on the clock of the prices
            return df[_local_times(df) >= earliest_date.tz_localize(None)]
        return df[df.index >= earliest_date]

    def _clear_fills_over_events(self, df: pd.DataFrame, filled: pd.DataFrame) -> pd.DataFrame:
        """
        A 0 filled by `_ffill_zeros` holds a value in the scale before any dividend or split since
        that value, so from the bar of the event it's wrong, and it becomes NaN, like a bar without
        data. Only a split changes the scale of the volume. `filled` tells the filled cells, and
        is indexed by (asset, date). The result is sorted by asset and date.
        """
        div_col, spr_col = self.adjustments
        df = df.sort_index(level=[1, 0])
        filled = filled.reindex(df.index.swaplevel(), fill_value=False)
        rows = np.arange(len(df))
        assets = df.index.codes[1]
        first_rows = np.r_[True, assets[1:] != assets[:-1]]
        splits = (df[spr_col] != 1).to_numpy()
        events = splits | (df[div_col] != 0).to_numpy()

        def last_true(mask):
            return np.maximum.accumulate(np.where(mask, rows, -1))

        for col in self.ohlcv:
            is_filled = filled[col].to_numpy()
            # a row with its own value, the value of a filled row comes from the last one of them
            sources = last_true((df[col].notna().to_numpy() & ~is_filled) | first_rows)
            col_events = splits if col in self.ohlcv[4:] else events
            df[col] = df[col].mask(is_filled & (last_true(col_events) > sources))
        return df

    def _load(self):
        if self._prices_by_year:
            dfs = self._walk_split_by_year_dir(self._path, self._prices_index)
        else:
            dfs = self._walk_dir(self._path, self._prices_index)
        dfs = {k: self._from_earliest_date(v) for k, v in dfs.items() if v is not None}
        dfs = {k: v[~v.index.duplicated(keep="last")] for k, v in dfs.items()}
        # the local time of the bars is gone after the join, and a naive ex-date needs it
        bar_offsets = {k: _bar_offsets(v) for k, v in dfs.items() if v.index.tz and len(v)}
        dfs = {k: v.drop(columns=_UTC_OFFSET, errors="ignore") for k, v in dfs.items()}
        if len({v.index.tz for v in dfs.values()}) > 1:
            # concat can't keep several timezones in one index
            dfs = {k: v.tz_convert("UTC") if v.index.tz else v for k, v in dfs.items()}
        df = pd.concat(dfs, sort=False)
        # the rows of a file, and the files of `prices_by_year`, can be in any order
        df = df.rename_axis(["asset", "date"]).sort_index()

        filled = pd.DataFrame()
        if self.ohlcv is not None:
            # 这里把0当成nan进行ffill（按asset分别填充）
            zeros = df[list(self.ohlcv)] == 0
            df[list(self.ohlcv)] = _ffill_zeros(df[list(self.ohlcv)], df.index.get_level_values(0))
            filled = zeros & (df[list(self.ohlcv)] != 0)
            filled = filled[filled.any(axis=1)]

        df = df.swaplevel(0, 1).sort_index(level=0)

        if self._calender:
            # drop the data of the non-trading day by calender,
            # because there may be some one-line junk data in non-trading day,
            # causing extra row of nan to all others assets.
            df = self._align_to(df, self._calender, self._align_by_time)
            df = df.sort_index(level=[0, 1])
            if self._align_by_time and self.adjustments is not None:
                # the rows added by the alignment have no events, they are not missing data
                df = df.fillna({self.adjustments[0]: 0, self.adjustments[1]: 1})

        tz = df.index.levels[0].tz
        if self._dividends_path is not None:
            dfs = self._walk_dir(self._dividends_path, self._dividends_index, as_written=tz is None)
            ex_div_col = self._adjustment_cols[0]
            div_index = self._dividends_index

            def _agg_duplicated(_df):
                if _df is None or ex_div_col not in _df:
                    return None
                _df = _df.reset_index().drop_duplicates()
                _df = _df.dropna(subset=[ex_div_col])
                _df = _df.set_index(div_index)[ex_div_col]
                return _df.groupby(level=0).agg("sum")

            dfs = {k: _agg_duplicated(v) for k, v in dfs.items()}
            div = _concat_events(dfs, tz, bar_offsets)
            div = self._attach_events(div, df.index, "sum")
            div = div.fillna(0)
            div.name = self._adjustments[0]
            # div = df.rename_axis(['asset', 'date'])
            df = pd.concat([df, div], axis=1)

        if self._splits_path is not None:
            dfs = self._walk_dir(self._splits_path, self._splits_index, as_written=tz is None)
            sp_rto_col = self._adjustment_cols[1]

            def _drop_na_and_duplicated(_df):
                if _df is None or sp_rto_col not in _df:
                    return None
                _df = _df.dropna(subset=[sp_rto_col])[sp_rto_col]
                return _df[~_df.index.duplicated(keep="last")]

            dfs = {k: _drop_na_and_duplicated(v) for k, v in dfs.items()}
            splits = _concat_events(dfs, tz, bar_offsets)
            if self._split_ratio_is_fraction:
                from fractions import Fraction

                def fraction_2_float(x):
                    try:
                        return float(Fraction(x))
                    except ValueError, ZeroDivisionError:
                        return np.nan

                splits = splits.apply(fraction_2_float)
            splits = self._attach_events(splits, df.index, "prod")
            splits = splits.fillna(1)
            splits.name = self._adjustments[1]
            df = pd.concat([df, splits], axis=1)

        if self.adjustments is not None and not filled.empty:
            df = self._clear_fills_over_events(df, filled)
        df = self._format(df, self._split_ratio_is_inverse)
        return df
