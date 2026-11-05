import glob
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import spectre
from tests.conftest import (
    DAILY_OHLCV,
    DATA_DIR,
    DIRTY_DATES,
    DIRTY_SECTORS,
    INTRADAY_OHLCV,
    NANSPLIT_ON,
    make_adjusted_daily_loader,
)

nan = np.nan

START, END = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-01-15", tz="UTC")
MEMORY_OHLCV = ("open", "high", "low", "close", "volume")


def read_raw(asset, folder="daily", index_col="date"):
    return pd.read_csv(DATA_DIR / folder / f"{asset}.csv", index_col=index_col, parse_dates=True)


def first_last(df, asset, col):
    values = df.loc[(slice(None), asset), col].values
    return values[0], values[-1]


def dates_of(df):
    return df.index.get_level_values(0).unique()


def values_of(df, asset, col):
    return df.loc[(slice(None), asset), col].values


@pytest.mark.parametrize("kwargs", [{}, {"prices_index": "date"}])
def test_index_must_be_datetime(kwargs):
    loader = spectre.data.CsvDirLoader(str(DATA_DIR / "daily"), **kwargs)
    with pytest.raises(ValueError, match="df must index by datetime.*"):
        loader.load(START, END, 0)


def test_empty_dir(tmp_path):
    loader = spectre.data.CsvDirLoader(str(tmp_path), prices_index="date", parse_dates=True)
    with pytest.raises(ValueError, match="There are no files"):
        loader.load()


def test_format(daily_loader):
    df = daily_loader.test_load()
    assert df.index.names == ["date", "asset"]
    assert str(df.index.levels[0].tz) == "UTC"
    assert list(df.index.levels[1]) == ["AAPL", "MSFT"]
    # time category is the id of unique datetime
    dates = df.index.get_level_values(0)
    assert_array_equal(df[daily_loader.time_category].values, pd.factorize(dates, sort=True)[0])


def test_calender_asset(daily_loader, daily_loader_no_calendar):
    # MSFT has a junk row on 2019-01-05 (Saturday), dropped by calender_asset='AAPL'
    junk = pd.Timestamp("2019-01-05", tz="UTC")
    assert junk not in daily_loader.load().index.levels[0]
    assert junk in daily_loader_no_calendar.load().index.levels[0]


def test_load_values():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / "daily"), calender_asset="AAPL", prices_index="date", parse_dates=True
    )

    df = loader.load(START, END, 0)
    aapl, msft = read_raw("AAPL"), read_raw("MSFT")
    assert first_last(df, "AAPL", "close") == (160.35, 158.09)
    assert first_last(df, "MSFT", "close") == (100.1, 105.36)
    assert_array_equal(values_of(df, "AAPL", "close"), aapl.close["2019-01-01":"2019-01-15"].values)
    assert_array_equal(
        values_of(df, "MSFT", "close"), msft.close["2019-01-01":"2019-01-15"].dropna().values
    )

    # 2019-01-12 is Saturday
    df = loader.load(pd.Timestamp("2019-01-11", tz="UTC"), pd.Timestamp("2019-01-12", tz="UTC"))
    assert first_last(df, "MSFT", "close") == (104.5, 104.5)


def test_load_backwards():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / "daily"), calender_asset="AAPL", prices_index="date", parse_dates=True
    )
    df = loader.load(START, END, 11)
    # 11 trading days before START
    dates = dates_of(df)
    assert dates[0] == pd.Timestamp("2018-12-14", tz="UTC")
    assert len(dates[dates < START]) == 11
    assert first_last(df, "AAPL", "close") == (173.43, 158.09)
    assert first_last(df, "MSFT", "close") == (106.57, 105.36)


def history_of(df, start):
    return df[df.index.get_level_values("date") < start]


def test_load_backwards_counts_the_bars_of_each_asset(dirty_loader, dirty_panel):
    start, end = DIRTY_DATES[35], DIRTY_DATES[-1]
    df = dirty_loader.load(start, end, 10)

    expected = history_of(dirty_panel, start).groupby(level="asset").tail(10)
    assert sorted(history_of(df, start).index) == sorted(expected.index)
    # the 3 bars after the halt, and the 7 bars before it
    halted = history_of(df, start).xs("HALTED", level="asset")
    assert list(halted.index) == [*DIRTY_DATES[13:20], *DIRTY_DATES[32:35]]
    # listed 5 bars before start, it has no more bars
    assert len(history_of(df, start).xs("LISTED", level="asset")) == 5
    # and all rows of the range, in the order of the data
    assert df.index.is_monotonic_increasing
    assert sorted(df.loc[start:end].index) == sorted(dirty_panel.loc[start:end].index)


def test_load_backwards_skips_the_assets_that_are_gone(dirty_loader, dirty_panel):
    """An asset without a row in the range of a date slice can't affect anything in the range."""
    start, end = DIRTY_DATES[50], DIRTY_DATES[-1]
    df = dirty_loader.load(start, end, 3)
    assert "DELISTED" not in df.index.get_level_values("asset")
    # the rows are the 3 dates before start, and there's nobody with a gap in them
    assert sorted(df.index) == sorted(dirty_panel.loc[DIRTY_DATES[47] : end].index)


def test_load_backwards_of_no_gaps_is_a_date_slice(dirty_loader):
    start, end = DIRTY_DATES[15], DIRTY_DATES[-1]
    df = dirty_loader.load(start, end, 5)
    pd.testing.assert_frame_equal(df, dirty_loader.load().loc[DIRTY_DATES[10] : end])


def test_load_without_backwards_is_the_range(dirty_loader):
    start, end = DIRTY_DATES[35], DIRTY_DATES[-1]
    df = dirty_loader.load(start, end, 0)
    pd.testing.assert_frame_equal(df, dirty_loader.load().loc[start:end])


def test_load_backwards_past_the_first_date(dirty_loader):
    start, end = DIRTY_DATES[2], DIRTY_DATES[-1]
    df = dirty_loader.load(start, end, 10)
    pd.testing.assert_frame_equal(df, dirty_loader.load().loc[DIRTY_DATES[0] : end])


def test_load_out_of_range(daily_loader):
    with pytest.raises(ValueError, match="cannot be less than earliest time"):
        daily_loader.load(pd.Timestamp("2000-01-01", tz="UTC"), END)
    with pytest.raises(ValueError, match="cannot be greater than latest time"):
        daily_loader.load(START, pd.Timestamp("2030-01-01", tz="UTC"))
    with pytest.raises(AssertionError, match="There is no data between"):
        daily_loader.load(
            pd.Timestamp("2019-01-05", tz="UTC"), pd.Timestamp("2019-01-06", tz="UTC")
        )


def test_earliest_date():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / "daily"),
        prices_index="date",
        parse_dates=True,
        earliest_date=pd.Timestamp("2019-01-02"),
    )
    assert loader.load().index.levels[0][0] == pd.Timestamp("2019-01-02", tz="UTC")


def test_zero_prices_are_forward_filled(tmp_path):
    index = ["2019-01-01", "2019-01-02", "2019-01-03"]
    ohlcv = ("open", "high", "low", "close", "volume")
    a = pd.DataFrame({"date": index, **{c: [1.0, 0, 3] for c in ohlcv}})
    b = pd.DataFrame({"date": index, **{c: [0.0, 2, 0] for c in ohlcv}})
    a.to_csv(tmp_path / "A.csv", index=False)
    b.to_csv(tmp_path / "B.csv", index=False)
    loader = spectre.data.CsvDirLoader(str(tmp_path), prices_index="date", parse_dates=True)
    df = loader.load()
    for col in ohlcv:
        assert_array_equal(values_of(df, "A", col), [1, 1, 3])
        # the leading zero of B is not filled by the price of A
        assert_array_equal(values_of(df, "B", col), [0, 2, 2])


def test_split_by_year(intraday_loader):
    # data across the year
    start = pd.Timestamp("2018-12-31 14:50:00", tz="America/New_York").tz_convert("UTC")
    end = pd.Timestamp("2019-01-02 10:00:00", tz="America/New_York").tz_convert("UTC")
    df = intraday_loader.load(start, end, 0)
    assert first_last(df, "AAPL", "Open") == (157.45, 155.17)
    assert first_last(df, "MSFT", "Open") == (101.44, 99.55)
    assert dates_of(df)[0] == start
    assert dates_of(df)[-1] == end

    df = intraday_loader.test_load()
    raw = pd.concat([read_raw(f"AAPL_{year}", "5mins", "Date") for year in (2018, 2019)])
    assert len(df.loc[(slice(None), "AAPL"), :]) == len(raw.index.unique())


def test_dividends_and_splits(adjusted_daily_loader):
    df = adjusted_daily_loader.test_load()
    # Two MSFT dividends on 2019-01-11 are summed up, and moved up to the last trading day
    # before the ex-date (MSFT has no data on 2019-01-10).
    assert df.loc[("2019-01-09", "MSFT"), "ex-dividend"] == pytest.approx(0.46 + 0.11)
    assert df.loc[("2019-01-11", "MSFT"), "ex-dividend"] == 0
    # 1-for-15 reverse split on 2019-01-14 (last non-NaN ratio is used)
    assert df.loc[("2019-01-11", "MSFT"), "split_ratio"] == 15
    assert df.loc[("2019-01-14", "MSFT"), "split_ratio"] == 1


def expected_msft_adjusted(start="2019-01-02"):
    """MSFT adjusted to the price of 2019-01-15."""
    msft = read_raw("MSFT").loc[start:"2019-01-15"].dropna()
    split = pd.Series(1.0, index=msft.index)
    split[:"2019-01-11"] = 15
    dividend = pd.Series(1.0, index=msft.index)
    dividend[:"2019-01-09"] = 1 - (0.46 + 0.11) / msft.uClose["2019-01-09"]
    return msft.uOpen * split * dividend, msft.uVolume / split


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_adjusted_column(adjusted_daily_loader):
    start = pd.Timestamp("2019-01-02", tz="UTC")
    engine = spectre.factors.FactorEngine(adjusted_daily_loader)
    engine.add(spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.volume), "vol")
    engine.add(spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.open), "open")
    df = engine.run(start, END, delay_factor=False)

    expected_open, expected_vol = expected_msft_adjusted()
    assert_almost_equal(values_of(df, "MSFT", "open"), expected_open.values, decimal=4)
    assert_almost_equal(values_of(df, "MSFT", "vol"), expected_vol.values, decimal=0)
    assert_almost_equal(expected_open.values,
                        [1526.24849, 1548.329113, 1536.244448, 1541.16783, 1563.696033,
                         1585.47827, 1569.750105, 104.9, 103.19], decimal=4)  # fmt: skip

    # AAPL has no dividends and splits in this period
    aapl = read_raw("AAPL").loc["2019-01-02":"2019-01-15"]
    assert_almost_equal(values_of(df, "AAPL", "open"), aapl.uOpen.values, decimal=4)
    assert_almost_equal(values_of(df, "AAPL", "vol"), aapl.uVolume.values, decimal=4)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_rolling_window_is_adjusted(adjusted_daily_loader):
    """Values in a rolling window are adjusted to the last row of the window."""
    windows = {}

    class LastWindow(spectre.factors.CustomFactor):
        win = 10

        def __init__(self, name, inputs):
            super().__init__(inputs=inputs)
            self.name = name

        def compute(self, data):
            # data.agg gives the adjusted values: [asset, time, window], keep the last window
            windows[self.name] = data.agg(lambda x: x[:, -1]).numpy()
            return data.last()

    engine = spectre.factors.FactorEngine(adjusted_daily_loader)
    engine.add(LastWindow("vol", inputs=[spectre.factors.OHLCV.volume]), "vol")
    engine.add(LastWindow("open", inputs=[spectre.factors.OHLCV.open]), "open")
    engine.run(END, END, delay_factor=False)

    aapl = read_raw("AAPL").loc["2019-01-02":"2019-01-15"]
    assert_almost_equal(windows["vol"][0], aapl.uVolume.values, decimal=4)
    assert_almost_equal(windows["open"][0], aapl.uOpen.values, decimal=4)
    # MSFT has no row on 2019-01-10, so its 10 bars start a bar earlier than those of AAPL
    expected_open, expected_vol = expected_msft_adjusted("2018-12-31")
    assert_almost_equal(windows["vol"][1], expected_vol.values, decimal=0)
    assert_almost_equal(windows["open"][1], expected_open.values, decimal=4)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_no_ohlcv():
    start = pd.Timestamp("2019-01-02")
    loader = spectre.data.CsvDirLoader(
        prices_path=str(DATA_DIR / "daily"),
        earliest_date=start,
        calender_asset="AAPL",
        ohlcv=None,
        adjustments=None,
        prices_index="date",
        parse_dates=True,
    )
    engine = spectre.factors.FactorEngine(loader)
    engine.add(spectre.factors.ColumnDataFactor(inputs=["uOpen"]), "open")
    df = engine.run(start, pd.Timestamp("2019-01-15"), delay_factor=False)
    assert_array_equal(
        values_of(df, "AAPL", "open"), read_raw("AAPL").uOpen["2019-01-02":"2019-01-15"].values
    )


def test_align_by_time_loader():
    loader = make_adjusted_daily_loader(calender_asset="AAPL", align_by_time=True)
    df = loader.load(START, END)
    # MSFT has a nan row on 2019-01-10
    assert len(df) == 2 * 10
    assert df.loc[("2019-01-10", "MSFT"), DAILY_OHLCV].isna().all()


# ---------------------------------- adjustments ----------------------------------


def memory_loader(closes, dividends=None, splits=None):
    """One asset 'A', `dividends` and `splits` are {bar: value}, on the bar of the ex-date."""
    bars = len(closes)
    dates = pd.date_range("2020-01-06", periods=bars, freq="D", tz="UTC")
    df = pd.DataFrame({col: closes for col in MEMORY_OHLCV}, index=dates, dtype=float)
    df["ex-dividend"] = pd.Series(dividends or {}, index=range(bars), dtype=float).fillna(0).values
    df["split_ratio"] = pd.Series(splits or {}, index=range(bars), dtype=float).fillna(1).values
    df = df.assign(asset="A").set_index("asset", append=True)
    return spectre.data.MemoryLoader(df, MEMORY_OHLCV, ("ex-dividend", "split_ratio"))


@pytest.mark.parametrize(
    "closes, dividends, splits, price_multi, vol_multi",
    [
        pytest.param([20, nan, 10, 10], {}, {2: 0.5}, [0.5, 0.5, 1, 1], [2, 2, 1, 1], id="after"),
        pytest.param([20, 20, nan, 10], {}, {2: 0.5}, [0.5, 0.5, 1, 1], [2, 2, 1, 1], id="on"),
        pytest.param([9, nan, 9, 9], {}, {}, [1, 1, 1, 1], [1, 1, 1, 1], id="no_event"),
        pytest.param([9, 9, nan], {}, {}, [1, 1, 1], [1, 1, 1], id="last_bar"),
        # 1 - 2 / 100, the last close before the ex-date
        pytest.param(
            [100, nan, nan, 98, 98], {3: 2}, {}, [0.98, 0.98, 0.98, 1, 1], [1] * 5, id="dividend"
        ),
        # The dividend is after the split, so 1 is compared with 100 in the scale of the split.
        pytest.param(
            [100, nan, nan, 49, 49],
            {3: 1},
            {2: 0.5},
            [0.49, 0.49, 0.98, 1, 1],
            [2, 2, 1, 1, 1],
            id="dividend_after_split",
        ),
        # A dividend on the ex-date of a split is paid on the shares before it, see `yahoo.py`.
        pytest.param(
            [100, nan, 50, 50], {2: 1}, {2: 0.5}, [0.495, 0.495, 1, 1], [2, 2, 1, 1], id="same_date"
        ),
        # No close before the ex-date, so there is no ratio, and no price to adjust.
        pytest.param([nan, nan, 9, 9], {1: 2}, {}, [1, 1, 1, 1], [1, 1, 1, 1], id="no_last_close"),
        pytest.param([nan, nan, 9, 9], {1: 2}, {1: 0.5}, [0.5, 1, 1, 1], [2, 1, 1, 1], id="split"),
        # the csv loader keeps leading zeros, see `_ffill_zeros`
        pytest.param([0, 0, 9, 9], {}, {}, [1, 1, 1, 1], [1, 1, 1, 1], id="zero_close"),
        pytest.param([0, 0, 9, 9], {1: 2}, {}, [1, 1, 1, 1], [1, 1, 1, 1], id="zero_last_close"),
    ],
)
def test_nan_close_keeps_events(closes, dividends, splits, price_multi, vol_multi):
    df = memory_loader(closes, dividends, splits).load()
    assert_allclose(df.price_multi, price_multi, rtol=1e-6)
    assert_allclose(df.vol_multi, vol_multi, rtol=1e-6)
    assert df.price_multi.dtype == np.float32
    assert df.vol_multi.dtype == np.float32


class FrameLoader(spectre.data.DataLoader):
    """Returns the frame as is, for frames that `_format` doesn't produce."""

    def __init__(self, df):
        super().__init__("", MEMORY_OHLCV, ("ex-dividend", "split_ratio"))
        self.df = df

    @property
    def last_modified(self):
        return 1

    def _load(self):
        return self.df


@pytest.mark.parametrize("column", ["price_multi", "vol_multi"])
def test_test_load_finds_nan_multipliers(tmp_path, column):
    df = memory_loader([10, 10, 10]).load().copy()
    df.loc[df.index[1], column] = nan
    with pytest.raises(AssertionError, match="nan value in adjustment multipliers"):
        FrameLoader(df).test_load()

    # so it can't be written to disk
    path = tmp_path / "prices.feather"
    with pytest.raises(AssertionError, match="nan value in adjustment multipliers"):
        spectre.data.ArrowLoader.ingest(FrameLoader(df), str(path), force=True)
    assert not path.exists()


def test_test_load_passes_nan_prices(dirty_loader):
    assert dirty_loader.test_load()["close"].isna().any()


def test_dirty_panel_multipliers(dirty_loader, dirty_panel):
    df = dirty_loader.load()
    assert not df[["price_multi", "vol_multi"]].isna().any(axis=None)

    # an adjusted close moves like `clean_close`, in the scale of the last bar
    assert list(df.index.levels[1]) == sorted(DIRTY_SECTORS)
    for asset in DIRTY_SECTORS:
        raw = dirty_panel.xs(asset, level="asset")
        scale = (raw.close / raw.clean_close).dropna()
        expected = scale.iloc[-1] / scale
        assert_allclose(df.price_multi.xs(asset, level="asset")[scale.index], expected, rtol=1e-6)

    multi = df.xs("NANSPLIT", level="asset")
    assert_allclose(multi.price_multi.iloc[:NANSPLIT_ON], 0.5)
    assert_allclose(multi.price_multi.iloc[NANSPLIT_ON:], 1)
    assert_allclose(multi.vol_multi.iloc[:NANSPLIT_ON], 2)
    assert_allclose(multi.vol_multi.iloc[NANSPLIT_ON:], 1)


def write_prices(folder, asset, closes):
    """`closes` is {date: close}."""
    folder.mkdir(exist_ok=True)
    frame = pd.DataFrame({col: closes.values() for col in MEMORY_OHLCV}, index=list(closes))
    frame.rename_axis("date").to_csv(folder / f"{asset}.csv")


def write_events(folder, asset, column, events):
    """`events` is {ex-date: value}."""
    folder.mkdir(exist_ok=True)
    frame = pd.DataFrame({column: events.values()}, index=list(events))
    frame.rename_axis("exDate").to_csv(folder / f"{asset}.csv")


def events_loader(tmp_path, prices, dividends, splits, **kwargs):
    """
    A loader of `prices`, {asset: {date: close}}. `dividends` and `splits` are {ex-date: value},
    they belong to the first asset.
    """
    for asset, closes in prices.items():
        write_prices(tmp_path / "prices", asset, closes)
    asset = next(iter(prices))
    write_events(tmp_path / "dividends", asset, "amount", dividends)
    write_events(tmp_path / "splits", asset, "ratio", splits)
    return spectre.data.CsvDirLoader(
        str(tmp_path / "prices"),
        ohlcv=MEMORY_OHLCV,
        dividends_path=str(tmp_path / "dividends"),
        splits_path=str(tmp_path / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
        **kwargs,
    )


DAYS = list(pd.date_range("2020-01-06", periods=8).strftime("%Y-%m-%d"))
NO_DIVIDEND = NO_SPLIT = {}


def test_event_without_a_bar_goes_to_the_next_bar(tmp_path):
    closes = {DAYS[0]: 100, DAYS[2]: 49, DAYS[3]: 49}
    df = events_loader(tmp_path, {"A": closes}, {DAYS[1]: 1.0}, {DAYS[1]: 0.5}).test_load()
    # moved up one row by `_format`, like every event
    assert list(df["ex-dividend"]) == [1, 0, 0]
    assert list(df["split_ratio"]) == [0.5, 1, 1]
    assert_allclose(df.price_multi, [(1 - 1 / 100) * 0.5, 1, 1], rtol=1e-6)
    assert list(df.vol_multi) == [2, 1, 1]


def test_event_is_not_moved_before_its_ex_date(tmp_path):
    closes = dict.fromkeys(DAYS[:4], 100)
    # after the bar of DAYS[1], so the first bar on or after it is DAYS[2]
    splits = {f"{DAYS[1]} 12:00:00": 0.5}
    df = events_loader(tmp_path, {"A": closes}, NO_DIVIDEND, splits).test_load()
    assert list(df["split_ratio"]) == [1, 0.5, 1, 1]
    assert list(df.price_multi) == [0.5, 0.5, 1, 1]


def test_events_sharing_a_bar_are_merged(tmp_path):
    closes = {DAYS[0]: 100, DAYS[5]: 16, DAYS[6]: 16}
    dividends = {DAYS[1]: 1.0, DAYS[3]: 2.0}
    splits = {DAYS[2]: 0.5, DAYS[4]: 1 / 3}
    df = events_loader(tmp_path, {"A": closes}, dividends, splits).test_load()
    assert_allclose(df["ex-dividend"], [3, 0, 0])
    assert_allclose(df["split_ratio"], [1 / 6, 1, 1])
    assert_allclose(df.price_multi, [(1 - 3 / 100) / 6, 1, 1], rtol=1e-6)
    assert_allclose(df.vol_multi, [6, 1, 1])


@pytest.mark.parametrize(
    "splits, kwargs",
    [
        ({DAYS[1]: 2, DAYS[2]: 3}, {"split_ratio_is_inverse": True}),
        ({DAYS[1]: "1/2", DAYS[2]: "1/3"}, {"split_ratio_is_fraction": True}),
    ],
)
def test_split_formats_of_events_sharing_a_bar(tmp_path, splits, kwargs):
    closes = {DAYS[0]: 60, DAYS[3]: 10, DAYS[4]: 10}
    df = events_loader(tmp_path, {"A": closes}, NO_DIVIDEND, splits, **kwargs).test_load()
    assert_allclose(df["split_ratio"], [1 / 6, 1, 1])
    assert_allclose(df.vol_multi, [6, 1, 1])


def test_events_outside_of_the_bars(tmp_path):
    closes = dict.fromkeys(DAYS[2:5], 100)
    dividends = {DAYS[0]: 1.0, DAYS[7]: 2.0}
    splits = {DAYS[0]: 0.5, DAYS[7]: 0.5}
    df = events_loader(tmp_path, {"A": closes}, dividends, splits).test_load()
    assert (df["ex-dividend"] == 0).all()
    assert (df["split_ratio"] == 1).all()
    assert (df[["price_multi", "vol_multi"]] == 1).all(axis=None)


def test_event_only_goes_to_the_bars_of_its_asset(tmp_path):
    prices = {"A": dict.fromkeys(DAYS[:2], 100), "B": dict.fromkeys(DAYS[:4], 100)}
    df = events_loader(tmp_path, prices, NO_DIVIDEND, {DAYS[2]: 0.5}).test_load()
    assert (df["split_ratio"] == 1).all()
    assert (df[["price_multi", "vol_multi"]] == 1).all(axis=None)


def test_no_valid_event(tmp_path):
    closes = dict.fromkeys(DAYS[:3], 100)
    df = events_loader(tmp_path, {"A": closes}, {DAYS[1]: nan}, {DAYS[1]: nan}).test_load()
    assert (df["ex-dividend"] == 0).all()
    assert (df["split_ratio"] == 1).all()


@pytest.mark.parametrize(
    "kwargs", [{}, {"split_ratio_is_fraction": True}, {"align_by_time": True}], ids=str
)
def test_event_folders_without_events(tmp_path, kwargs):
    closes = dict.fromkeys(DAYS[:3], 100.0)
    if "align_by_time" in kwargs:
        kwargs = {**kwargs, "calender_asset": "A"}
    df = events_loader(tmp_path, {"A": closes}, {}, {}, **kwargs).test_load()
    assert len(df) == 3
    assert (df["ex-dividend"] == 0).all()
    assert (df["split_ratio"] == 1).all()
    assert (df[["price_multi", "vol_multi"]] == 1).all(axis=None)


def test_event_files_without_the_event_column(tmp_path):
    loader = events_loader(tmp_path, {"A": dict.fromkeys(DAYS[:3], 100.0)}, {}, {})
    for folder in ("dividends", "splits"):
        frame = pd.DataFrame({"exDate": DAYS[1:2], "description": ["other data"]})
        frame.to_csv(tmp_path / folder / "A.csv", index=False)
    df = loader.test_load()
    assert (df["ex-dividend"] == 0).all()
    assert (df["split_ratio"] == 1).all()


def test_event_folder_without_files_is_an_error(tmp_path):
    """A wrong path, it must not read as a stock without events."""
    loader = events_loader(tmp_path, {"A": dict.fromkeys(DAYS[:3], 100.0)}, {}, {})
    (tmp_path / "splits" / "A.csv").unlink()
    with pytest.raises(ValueError, match="There are no files"):
        loader.load()


def test_event_on_a_row_dropped_by_the_calendar(tmp_path):
    prices = {
        "A": {DAYS[0]: 100, DAYS[1]: 100, DAYS[2]: 50, DAYS[3]: 50},
        "CALENDAR": dict.fromkeys([DAYS[0], DAYS[2], DAYS[3]], 1),
    }
    loader = events_loader(
        tmp_path, prices, {DAYS[1]: 1.0}, {DAYS[1]: 0.5}, calender_asset="CALENDAR"
    )
    df = loader.test_load().xs("A", level=1)
    assert list(df["ex-dividend"]) == [1, 0, 0]
    assert list(df["split_ratio"]) == [0.5, 1, 1]


def test_unsorted_csv_files(tmp_path):
    closes = {DAYS[3]: 50, DAYS[0]: 100, DAYS[2]: 50}
    dividends = {DAYS[1]: 1.0, DAYS[0]: 2.0}
    df = events_loader(tmp_path, {"A": closes}, dividends, {DAYS[1]: 0.5}).test_load()
    assert list(df.close) == [100, 50, 50]
    assert list(df["ex-dividend"]) == [1, 0, 0]
    assert list(df["split_ratio"]) == [0.5, 1, 1]


@pytest.mark.parametrize(
    "bar_time, split_time, price_multi",
    [
        # without a timezone it's the time of the other side, so it's on the bar of DAYS[1]
        ("00:00:00-05:00", "00:00:00", [0.5, 1, 1, 1]),
        ("00:00:00", "00:00:00-05:00", [0.5, 1, 1, 1]),
        # the same instant as the bar of DAYS[1]
        ("00:00:00-05:00", "05:00:00+00:00", [0.5, 1, 1, 1]),
        # after the bar of DAYS[1]
        ("00:00:00-05:00", "07:00:00+00:00", [0.5, 0.5, 1, 1]),
        ("00:00:00+08:00", "00:00:00+00:00", [0.5, 0.5, 1, 1]),
    ],
)
def test_ex_date_timezones(tmp_path, bar_time, split_time, price_multi):
    closes = {f"{day} {bar_time}": 100 for day in DAYS[:4]}
    splits = {f"{DAYS[1]} {split_time}": 0.5}
    df = events_loader(tmp_path, {"A": closes}, NO_DIVIDEND, splits).test_load()
    assert list(df.price_multi) == price_multi


@pytest.mark.parametrize(
    "bar_time, price_multi_a, price_multi_b",
    [
        # without a timezone on the bars it's the time as written, so on the bar of that day
        ("00:00:00", [0.5, 0.5, 1, 1], [0.5, 1, 1, 1]),
        # 00:00-04:00 is 23:00-05:00 of the day before, the bar after it is the bar of DAYS[1]
        ("00:00:00-05:00", [0.5, 0.5, 1, 1], [0.5, 1, 1, 1]),
        # 04:00 and 05:00 UTC, after the bar of that day
        ("00:00:00+00:00", [0.5, 0.5, 0.5, 1], [0.5, 0.5, 1, 1]),
    ],
)
def test_ex_date_files_in_different_utc_offsets(tmp_path, bar_time, price_multi_a, price_multi_b):
    closes = {f"{day} {bar_time}": 100.0 for day in DAYS[:4]}
    prices = {"A": closes, "B": closes}
    # the two files are on both sides of a change of the daylight saving time
    loader = events_loader(tmp_path, prices, {}, {f"{DAYS[2]} 00:00:00-05:00": 0.5})
    write_events(tmp_path / "splits", "B", "ratio", {f"{DAYS[1]} 00:00:00-04:00": 0.5})
    df = loader.test_load()
    assert list(df.price_multi.xs("A", level=1)) == price_multi_a
    assert list(df.price_multi.xs("B", level=1)) == price_multi_b


def test_ex_date_files_with_and_without_a_timezone(tmp_path):
    closes = {f"{day} 00:00:00-05:00": 100.0 for day in DAYS[:4]}
    prices = {"A": closes, "B": closes}
    dividends = {f"{DAYS[1]} 00:00:00-05:00": 1.0}
    loader = events_loader(tmp_path, prices, dividends, {f"{DAYS[1]} 00:00:00-05:00": 0.5})
    write_events(tmp_path / "dividends", "B", "amount", {DAYS[2]: 1.0})
    write_events(tmp_path / "splits", "B", "ratio", {DAYS[2]: 0.5})
    df = loader.test_load()
    assert list(df["split_ratio"].xs("A", level=1)) == [0.5, 1, 1, 1]
    assert list(df["split_ratio"].xs("B", level=1)) == [1, 0.5, 1, 1]
    assert list(df["ex-dividend"].xs("A", level=1)) == [1, 0, 0, 0]
    assert list(df["ex-dividend"].xs("B", level=1)) == [0, 1, 0, 0]


def test_files_across_a_change_of_daylight_saving_time(tmp_path):
    """pandas reads a column of two UTC offsets as strings, in prices and in ex-dates."""
    closes = {f"{day} 09:30:00-05:00": 100.0 for day in DAYS[:2]}
    closes |= {f"{day} 09:30:00-04:00": 100.0 for day in DAYS[2:4]}
    splits = {f"{DAYS[1]} 00:00:00-05:00": 0.5, f"{DAYS[3]} 00:00:00-04:00": 0.5}
    loader = events_loader(tmp_path, {"A": closes, "B": closes}, {}, splits)
    df = loader.test_load()
    expected = pd.to_datetime(list(closes), utc=True)
    assert list(df.xs("A", level=1).index) == list(expected)
    assert list(df.price_multi.xs("A", level=1)) == [0.25, 0.5, 0.5, 1]
    assert list(df.price_multi.xs("B", level=1)) == [1, 1, 1, 1]


def test_price_files_in_different_utc_offsets(tmp_path):
    a = {f"{day} 09:30:00-05:00": 100.0 for day in DAYS[:3]}
    b = {f"{day} 09:30:00-04:00": 100.0 for day in DAYS[:3]}
    across = {f"{DAYS[0]} 09:30:00-05:00": 100.0, f"{DAYS[1]} 09:30:00-04:00": 100.0}
    prices = {"A": a, "B": b, "C": across}
    df = events_loader(tmp_path, prices, {}, {}).test_load()
    for asset, closes in prices.items():
        expected = pd.to_datetime(list(closes), utc=True)
        assert list(df.xs(asset, level=1).index) == list(expected), asset


def local_bars(zone, times, days=None):
    """Timestamps of the bars at `times` of the business days, on the clock of `zone`."""
    days = pd.bdate_range("2019-03-04", "2019-04-12") if days is None else days
    return pd.DatetimeIndex(
        [pd.Timestamp(f"{day.date()} {time}", tz=zone) for day in days for time in times]
    )


def split_prices(bars, ex_date):
    """{bar: close}, the price halves on the local date `ex_date`, in the text of a csv."""
    return {
        bar.isoformat(): 50.0 if bar.tz_localize(None) >= pd.Timestamp(ex_date) else 100.0
        for bar in bars
    }


# the clocks change in this range: New York on 03-10, Berlin on 03-31, Sydney on 04-07
DAILY, INTRADAY = ("00:00:00",), ("10:00:00", "12:30:00", "15:55:00")


@pytest.mark.parametrize("times", [DAILY, INTRADAY], ids=["daily", "intraday"])
@pytest.mark.parametrize(
    "zone, ex_date",
    [
        ("America/New_York", "2019-03-08"),
        ("America/New_York", "2019-03-11"),
        ("Europe/Berlin", "2019-03-29"),
        ("Europe/Berlin", "2019-04-01"),
        ("Australia/Sydney", "2019-04-05"),
        ("Australia/Sydney", "2019-04-08"),
    ],
)
def test_naive_ex_date_on_both_sides_of_a_change_of_the_clock(tmp_path, zone, ex_date, times):
    bars = local_bars(zone, times)
    loader = events_loader(tmp_path, {"A": split_prices(bars, ex_date)}, {}, {ex_date: 0.5})
    df = loader.test_load()
    assert list(df.index.get_level_values(0)) == list(bars.tz_convert("UTC"))
    before = bars.tz_localize(None) < pd.Timestamp(ex_date)
    assert list(df.price_multi) == list(np.where(before, 0.5, 1))
    assert list(df.vol_multi) == list(np.where(before, 2, 1))
    assert (df["split_ratio"] != 1).sum() == 1


@pytest.mark.parametrize("times", [DAILY, INTRADAY], ids=["daily", "intraday"])
@pytest.mark.parametrize("ex_date", ["2019-03-08", "2019-04-01", "2019-04-08"])
def test_naive_ex_dates_of_files_in_different_fixed_offsets(tmp_path, ex_date, times):
    tokyo = local_bars("Asia/Tokyo", times)
    new_york = local_bars("Etc/GMT+5", times)
    prices = {"A": split_prices(tokyo, ex_date), "B": split_prices(new_york, ex_date)}
    loader = events_loader(tmp_path, prices, {}, {ex_date: 0.5})
    write_events(tmp_path / "splits", "B", "ratio", {ex_date: 0.5})
    df = loader.test_load()
    for asset, bars in (("A", tokyo), ("B", new_york)):
        before = bars.tz_localize(None) < pd.Timestamp(ex_date)
        assert list(df.price_multi.xs(asset, level=1)) == list(np.where(before, 0.5, 1)), asset


def test_naive_ex_date_of_a_file_next_to_a_file_across_a_change_of_the_clock(tmp_path):
    ex_date = "2019-04-01"
    tokyo = local_bars("Asia/Tokyo", DAILY)
    berlin = local_bars("Europe/Berlin", DAILY)
    prices = {"A": split_prices(berlin, ex_date), "B": split_prices(tokyo, ex_date)}
    loader = events_loader(tmp_path, prices, {}, {ex_date: 0.5})
    write_events(tmp_path / "splits", "B", "ratio", {ex_date: 0.5})
    df = loader.test_load()
    for asset, bars in (("A", berlin), ("B", tokyo)):
        before = bars.tz_localize(None) < pd.Timestamp(ex_date)
        assert list(df.price_multi.xs(asset, level=1)) == list(np.where(before, 0.5, 1)), asset


def test_naive_ex_date_in_the_gap_over_a_change_of_the_clock(tmp_path):
    bars = local_bars("Europe/Berlin", DAILY)
    # no bars on the weekend of the change, the first bar on or after both days is the Monday
    closes = split_prices(bars, "2019-04-01")
    loader = events_loader(tmp_path, {"A": closes}, {"2019-03-30": 1.0}, {"2019-03-31": 0.5})
    df = loader.test_load()
    friday = bars.tz_localize(None).get_loc(pd.Timestamp("2019-03-29"))
    assert list(np.flatnonzero(df["split_ratio"] != 1)) == [friday]
    assert list(np.flatnonzero(df["ex-dividend"] != 0)) == [friday]


def test_naive_ex_date_is_not_early_on_an_evening_bar(tmp_path):
    """20:00 in New York is the next day in UTC, the bars of the evening are still before it."""
    ex_date = "2019-04-02"
    bars = local_bars("America/New_York", ("09:30:00", "20:00:00"))
    loader = events_loader(tmp_path, {"A": split_prices(bars, ex_date)}, {}, {ex_date: 0.5})
    df = loader.test_load()
    before = bars.tz_localize(None) < pd.Timestamp(ex_date)
    assert list(df.price_multi) == list(np.where(before, 0.5, 1))


def test_naive_ex_date_of_year_files_across_a_change_of_the_clock(tmp_path):
    """Each file is in one UTC offset or in two, and the years have to join."""
    (tmp_path / "prices").mkdir()
    winter = local_bars("Europe/Berlin", DAILY, pd.bdate_range("2018-12-24", "2018-12-31"))
    spring = local_bars("Europe/Berlin", DAILY, pd.bdate_range("2019-03-25", "2019-04-12"))
    bars = winter.append(spring)
    ex_date = "2019-04-01"
    closes = pd.Series(split_prices(bars, ex_date), name="close")
    frame = pd.DataFrame({col: closes for col in MEMORY_OHLCV}).rename_axis("date")
    frame.iloc[: len(winter)].to_csv(tmp_path / "prices" / "A_2018.csv")
    frame.iloc[len(winter) :].to_csv(tmp_path / "prices" / "A_2019.csv")
    write_events(tmp_path / "dividends", "A", "amount", {})
    write_events(tmp_path / "splits", "A", "ratio", {ex_date: 0.5})
    df = spectre.data.CsvDirLoader(
        str(tmp_path / "prices"),
        prices_by_year=True,
        ohlcv=MEMORY_OHLCV,
        dividends_path=str(tmp_path / "dividends"),
        splits_path=str(tmp_path / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
    ).test_load()
    before = bars.tz_localize(None) < pd.Timestamp(ex_date)
    assert list(df.price_multi) == list(np.where(before, 0.5, 1))


@pytest.mark.parametrize(
    "ending", ["+02:00", "+0200", "+02"], ids=["with_colon", "without_colon", "only_hours"]
)
def test_naive_ex_date_with_other_formats_of_the_offset(tmp_path, ending):
    closes = {f"{DAYS[0]} 00:00:00+01:00": 100.0}
    closes |= {f"{day} 00:00:00{ending}": 50.0 for day in DAYS[1:3]}
    df = events_loader(tmp_path, {"A": closes}, {}, {DAYS[1]: 0.5}).test_load()
    assert list(df.price_multi) == [0.5, 1, 1]


@pytest.mark.parametrize("order", ["newest_first", "shuffled"])
def test_naive_ex_date_in_any_row_order_across_a_change_of_the_clock(tmp_path, order):
    ex_date = "2019-04-01"
    bars = local_bars("Europe/Berlin", DAILY)
    closes = pd.Series(split_prices(bars, ex_date))
    rows = pd.DataFrame({col: closes for col in MEMORY_OHLCV}).rename_axis("date")
    (tmp_path / "prices").mkdir()
    ROW_ORDERS[order](rows).to_csv(tmp_path / "prices" / "A.csv")
    write_events(tmp_path / "dividends", "A", "amount", {})
    write_events(tmp_path / "splits", "A", "ratio", {ex_date: 0.5})
    df = spectre.data.CsvDirLoader(
        str(tmp_path / "prices"),
        ohlcv=MEMORY_OHLCV,
        dividends_path=str(tmp_path / "dividends"),
        splits_path=str(tmp_path / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
    ).test_load()
    before = bars.tz_localize(None) < pd.Timestamp(ex_date)
    assert list(df.price_multi) == list(np.where(before, 0.5, 1))


def test_ex_dates_with_a_timezone_of_a_file_across_a_change_of_the_clock(tmp_path):
    """They have their instant, so the local time of the bars doesn't matter."""
    ex_date = pd.Timestamp("2019-04-01", tz="Europe/Berlin")
    bars = local_bars("Europe/Berlin", DAILY)
    closes = split_prices(bars, "2019-04-01")
    splits = {ex_date.isoformat(): 0.5}
    df = events_loader(tmp_path, {"A": closes}, {}, splits).test_load()
    before = bars < ex_date
    assert list(df.price_multi) == list(np.where(before, 0.5, 1))


def test_ex_dates_across_a_change_of_the_clock_on_bars_without_a_timezone(tmp_path):
    """The bars have no instant, so an ex-date is read as written, not in UTC."""
    days = [f"2019-03-{day:02}" for day in (7, 8, 11, 12, 13)]
    closes = dict(zip(days, (100.0, 50.0, 50.0, 25.0, 25.0)))
    splits = {f"{days[1]} 00:00:00-05:00": 0.5, f"{days[3]} 00:00:00-04:00": 0.5}
    df = events_loader(tmp_path, {"A": closes}, {}, splits).test_load()
    assert list(df.price_multi) == [0.25, 0.5, 0.5, 1, 1]


@pytest.mark.parametrize("bar_time", ["00:00:00", "00:00:00-05:00"], ids=["naive", "aware"])
def test_dividend_file_across_a_change_of_the_clock(tmp_path, bar_time):
    days = [f"2019-03-{day:02}" for day in (7, 8, 11, 12, 13)]
    closes = {f"{day} {bar_time}": 100.0 for day in days}
    dividends = {f"{days[1]} 00:00:00-05:00": 1.0, f"{days[3]} 00:00:00-04:00": 2.0}
    df = events_loader(tmp_path, {"A": closes}, dividends, {}).test_load()
    # 00:00-04:00 is 23:00-05:00 of the day before, the first bar after it is the same
    assert list(df["ex-dividend"]) == [1, 0, 2, 0, 0]


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_returns_over_a_split_across_a_change_of_the_clock(tmp_path, device):
    ex_date = "2019-04-01"
    bars = local_bars("Europe/Berlin", DAILY)
    loader = events_loader(tmp_path, {"A": split_prices(bars, ex_date)}, {}, {ex_date: 0.5})
    engine = spectre.factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(spectre.factors.Returns(), "returns")
    dates = bars.tz_convert("UTC")
    result = engine.run(dates[1], dates[-1], delay_factor=False)
    assert_allclose(result.returns, 0, atol=1e-6)


def test_daily_ex_date_of_intraday_bars(tmp_path, device):
    closes = {f"{DAYS[0]} {time}": 100.0 for time in ("09:30:00", "09:35:00", "09:40:00")}
    closes |= {f"{DAYS[1]} {time}": 50.0 for time in ("09:30:00", "09:35:00", "09:40:00")}
    loader = events_loader(tmp_path, {"A": closes}, NO_DIVIDEND, {DAYS[1]: 0.5})
    df = loader.test_load()
    # it applies once, between the two days
    assert list(df["split_ratio"]) == [1, 1, 0.5, 1, 1, 1]
    assert list(df.price_multi) == [0.5, 0.5, 0.5, 1, 1, 1]
    assert list(df.vol_multi) == [2, 2, 2, 1, 1, 1]

    engine = spectre.factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(spectre.factors.Returns(win=2), "returns")
    dates = dates_of(df)
    result = engine.run(dates[1], dates[-1], delay_factor=False)
    assert_allclose(result.returns, 0, atol=1e-6)


def test_events_of_the_intraday_test_data():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / "5mins"),
        prices_by_year=True,
        prices_index="Date",
        ohlcv=INTRADAY_OHLCV,
        dividends_path=str(DATA_DIR / "dividends"),
        splits_path=str(DATA_DIR / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
    )
    df = loader.test_load()
    aapl, msft = df.xs("AAPL", level=1), df.xs("MSFT", level=1)
    assert (aapl[["price_multi", "vol_multi"]] == 1).all(axis=None)

    # The bars are in New York time and the ex-dates are days, 2019-01-11 and 2019-01-14. Each
    # event is on the last bar of the day before its ex-date.
    last_bar_01_10 = pd.Timestamp("2019-01-10 15:55", tz="America/New_York")
    last_bar_01_11 = pd.Timestamp("2019-01-11 15:55", tz="America/New_York")
    dividend = msft["ex-dividend"][msft["ex-dividend"] != 0]
    split = msft["split_ratio"][msft["split_ratio"] != 1]
    assert list(dividend.index) == [last_bar_01_10]
    assert list(split.index) == [last_bar_01_11]
    assert dividend.iloc[0] == pytest.approx(0.46 + 0.11)
    assert split.iloc[0] == 15

    close = read_raw("MSFT_2019", "5mins", "Date").Close[last_bar_01_10]
    assert close == 103.58
    expected = pd.Series(1.0, index=msft.index)
    expected[:last_bar_01_11] = 15
    expected[:last_bar_01_10] = 15 * (1 - (0.46 + 0.11) / close)
    assert_allclose(msft.price_multi, expected, rtol=1e-6)
    expected = pd.Series(1.0, index=msft.index)
    expected[:last_bar_01_11] = 1 / 15
    assert_allclose(msft.vol_multi, expected, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_adjusted_column_of_aligned_bars():
    """The dividend of MSFT moves up onto the NaN row of 2019-01-10, the split onto the next."""
    loader = make_adjusted_daily_loader(calender_asset="AAPL", align_by_time=True)
    df = loader.test_load()
    assert not df[["price_multi", "vol_multi"]].isna().any(axis=None)

    engine = spectre.factors.FactorEngine(loader)
    engine.add(spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.volume), "vol")
    engine.add(spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.open), "open")
    df = engine.run(pd.Timestamp("2019-01-02", tz="UTC"), END, delay_factor=False)

    expected_open, expected_vol = expected_msft_adjusted()
    msft = df.xs("MSFT", level=1).drop(pd.Timestamp("2019-01-10", tz="UTC"))
    assert_almost_equal(msft.open.values, expected_open.values, decimal=4)
    assert_almost_equal(msft.vol.values, expected_vol.values, decimal=0)


def inline_events_loader(tmp_path, prices, **kwargs):
    """Like `events_loader`, but the columns of the events are in the prices csv."""
    folder = tmp_path / "prices"
    for asset, closes in prices.items():
        write_prices(folder, asset, closes)
        frame = pd.read_csv(folder / f"{asset}.csv")
        frame = frame.assign(**{"ex-dividend": 0.0, "split_ratio": 1.0})
        frame.to_csv(folder / f"{asset}.csv", index=False)
    return spectre.data.CsvDirLoader(
        str(folder),
        ohlcv=MEMORY_OHLCV,
        adjustments=("ex-dividend", "split_ratio"),
        parse_dates=True,
        **kwargs,
    )


def joined_events_loader(tmp_path, prices, **kwargs):
    return events_loader(tmp_path, prices, NO_DIVIDEND, NO_SPLIT, **kwargs)


@pytest.mark.parametrize("make_loader", [inline_events_loader, joined_events_loader])
@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_moving_average_over_an_aligned_gap(tmp_path, device, make_loader):
    close = pd.Series(100.0 + np.arange(8) ** 2, index=DAYS)
    gap = close.drop(DAYS[3])
    prices = {"GAP": gap.to_dict(), "FULL": close.to_dict()}
    loader = make_loader(tmp_path, prices, calender_asset="FULL", align_by_time=True)

    engine = spectre.factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(spectre.factors.MA(3), "ma")
    result = engine.run(DAYS[2], DAYS[-1], delay_factor=False)
    # a window with a missing bar is the mean of its valid bars
    expected = gap.reindex(DAYS).rolling(3, min_periods=1).mean()[DAYS[2] :]
    assert_allclose(result.ma.xs("GAP", level=1), expected, rtol=1e-6)

    df = loader.test_load()
    assert (df[["price_multi", "vol_multi"]] == 1).all(axis=None)
    assert (df["ex-dividend"] == 0).all()
    assert (df["split_ratio"] == 1).all()


def test_event_in_the_prices_csv_after_an_aligned_gap(tmp_path):
    prices = tmp_path / "prices"
    prices.mkdir()
    full = pd.DataFrame({col: [100.0, 100, 50, 50] for col in MEMORY_OHLCV}, index=DAYS[:4])
    full = full.assign(**{"ex-dividend": 0.0, "split_ratio": 1.0}).rename_axis("date")
    full.to_csv(prices / "FULL.csv")
    gap = full.drop(DAYS[1])
    gap.loc[DAYS[2], ["ex-dividend", "split_ratio"]] = (1.0, 0.5)
    gap.to_csv(prices / "GAP.csv")

    loader = spectre.data.CsvDirLoader(
        str(prices),
        ohlcv=MEMORY_OHLCV,
        adjustments=("ex-dividend", "split_ratio"),
        calender_asset="FULL",
        align_by_time=True,
        parse_dates=True,
    )
    df = loader.test_load().xs("GAP", level=1)
    assert list(df["ex-dividend"]) == [0, 1, 0, 0]
    assert list(df["split_ratio"]) == [1, 0.5, 1, 1]
    assert_allclose(df.price_multi, [(1 - 1 / 100) * 0.5] * 2 + [1, 1], rtol=1e-6)
    assert list(df.vol_multi) == [2, 2, 1, 1]


# --------------------------- events on rows the calendar drops ---------------------------


def write_inline(folder, asset, closes, dividends=None, splits=None):
    """`closes`, `dividends` and `splits` are {date: value}, the events go in the prices csv."""
    folder.mkdir(exist_ok=True)
    rows = pd.DataFrame({col: closes.values() for col in MEMORY_OHLCV}, index=list(closes))
    rows["ex-dividend"] = pd.Series(dividends or {}, index=rows.index, dtype=float).fillna(0)
    rows["split_ratio"] = pd.Series(splits or {}, index=rows.index, dtype=float).fillna(1)
    rows.rename_axis("date").to_csv(folder / f"{asset}.csv")


def calendar_loader(folder, calendar, **kwargs):
    """Loader of the inline events in `folder`, with an extra asset for the `calendar` dates."""
    write_inline(folder, "CALENDAR", dict.fromkeys(calendar, 1.0))
    return spectre.data.CsvDirLoader(
        str(folder),
        ohlcv=MEMORY_OHLCV,
        adjustments=("ex-dividend", "split_ratio"),
        calender_asset="CALENDAR",
        parse_dates=True,
        **kwargs,
    )


def test_inline_event_on_a_dropped_row(tmp_path):
    closes = {DAYS[0]: 100.0, DAYS[1]: 100.0, DAYS[2]: 50.0, DAYS[3]: 50.0}
    write_inline(tmp_path / "prices", "A", closes, {DAYS[1]: 1.0}, {DAYS[1]: 0.5})
    loader = calendar_loader(tmp_path / "prices", [DAYS[0], DAYS[2], DAYS[3]])
    df = loader.test_load().xs("A", level=1)
    assert list(df.close) == [100, 50, 50]
    # on the next kept bar, and moved up one row by `_format`, like every event
    assert list(df["ex-dividend"]) == [1, 0, 0]
    assert list(df["split_ratio"]) == [0.5, 1, 1]
    assert_allclose(df.price_multi, [(1 - 1 / 100) * 0.5, 1, 1], rtol=1e-6)
    assert list(df.vol_multi) == [2, 1, 1]


def test_inline_events_merge_on_the_next_kept_bar(tmp_path):
    closes = {DAYS[0]: 100.0, DAYS[1]: 100.0, DAYS[2]: 100.0, DAYS[3]: 16.0, DAYS[4]: 16.0}
    dividends = {DAYS[1]: 1.0, DAYS[2]: 2.0, DAYS[3]: 3.0}
    splits = {DAYS[1]: 0.5, DAYS[3]: 1 / 3}
    write_inline(tmp_path / "prices", "A", closes, dividends, splits)
    loader = calendar_loader(tmp_path / "prices", [DAYS[0], DAYS[3], DAYS[4]])
    df = loader.test_load().xs("A", level=1)
    assert_allclose(df["ex-dividend"], [6, 0, 0])
    assert_allclose(df["split_ratio"], [1 / 6, 1, 1])
    assert_allclose(df.price_multi, [(1 - 6 / 100) / 6, 1, 1], rtol=1e-6)


def test_inline_event_is_not_moved_before_the_dropped_row(tmp_path):
    closes = dict.fromkeys(DAYS[:5], 100.0)
    write_inline(tmp_path / "prices", "A", closes, splits={DAYS[2]: 0.5})
    loader = calendar_loader(tmp_path / "prices", [DAYS[0], DAYS[1], DAYS[3], DAYS[4]])
    df = loader.test_load().xs("A", level=1)
    # on DAYS[3], then moved up one row, onto the last bar before the event
    assert list(df["split_ratio"]) == [1, 0.5, 1, 1]
    assert list(df.price_multi) == [0.5, 0.5, 1, 1]


def test_inline_event_after_the_last_kept_bar(tmp_path):
    write_inline(tmp_path / "prices", "A", dict.fromkeys(DAYS[:3], 100.0), splits={DAYS[2]: 0.5})
    write_inline(tmp_path / "prices", "B", dict.fromkeys(DAYS[:4], 100.0))
    loader = calendar_loader(tmp_path / "prices", [DAYS[0], DAYS[1], DAYS[3]])
    df = loader.test_load()
    # no bar is after it, and it doesn't go to the bars of another asset
    assert (df["split_ratio"] == 1).all()
    assert (df[["price_multi", "vol_multi"]] == 1).all(axis=None)


def test_inline_event_on_a_dropped_row_with_align_by_time(tmp_path):
    closes = {DAYS[0]: 100.0, DAYS[1]: 100.0, DAYS[3]: 50.0}
    write_inline(tmp_path / "prices", "A", closes, splits={DAYS[1]: 0.5})
    loader = calendar_loader(tmp_path / "prices", [DAYS[0], DAYS[2], DAYS[3]], align_by_time=True)
    df = loader.test_load().xs("A", level=1)
    assert_array_equal(df.close, [100, nan, 50])
    assert list(df["split_ratio"]) == [1, 0.5, 1]
    assert list(df.price_multi) == [0.5, 0.5, 1]


def test_junk_row_without_values(tmp_path):
    """A row of NaNs on a dropped row holds no event, like MSFT on 2019-01-05."""
    closes = {DAYS[0]: 100.0, DAYS[1]: nan, DAYS[2]: 100.0}
    write_inline(tmp_path / "prices", "A", closes)
    csv = tmp_path / "prices" / "A.csv"
    rows = pd.read_csv(csv, index_col="date")
    rows.loc[DAYS[1], ["ex-dividend", "split_ratio"]] = nan
    rows.to_csv(csv)
    df = calendar_loader(tmp_path / "prices", [DAYS[0], DAYS[2]]).test_load().xs("A", level=1)
    assert list(df["ex-dividend"]) == [0, 0]
    assert list(df["split_ratio"]) == [1, 1]


def test_quandl_event_on_a_dropped_row(tmp_path):
    """Rows of the quandl file are sorted by ticker, not by date, and its ratio is to/from."""
    aapl = ["2001-09-10", "2001-09-12", "2001-09-13"]
    rows = [(date, "AAPL", 10.0, 0.0, 1.0) for date in aapl]
    rows += [("2001-09-12", "GMT", 10.0, 0.0, nan)]  # `QuandlLoader` has a fix for this row
    rows += [
        ("2001-09-10", "ZZZ", 100.0, 0.0, 1.0),
        ("2001-09-11", "ZZZ", 100.0, 1.0, 2.0),
        ("2001-09-12", "ZZZ", 50.0, 0.0, 1.0),
        ("2001-09-13", "ZZZ", 50.0, 0.0, 1.0),
    ]
    columns = ["date", "ticker", "close", "ex-dividend", "split_ratio"]
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.assign(open=frame.close, high=frame.close, low=frame.close, volume=1000.0)
    with ZipFile(tmp_path / "prices.zip", "w") as package:
        package.writestr("prices.csv", frame.to_csv(index=False))

    df = spectre.data.QuandlLoader(str(tmp_path / "prices.zip")).test_load().xs("ZZZ", level=1)
    assert list(df.close) == [100, 50, 50]
    assert list(df["ex-dividend"]) == [1, 0, 0]
    assert list(df["split_ratio"]) == [0.5, 1, 1]
    assert_allclose(df.price_multi, [(1 - 1 / 100) * 0.5, 1, 1], rtol=1e-6)
    assert list(df.vol_multi) == [2, 1, 1]


# ------------------------------- zeros next to an event -------------------------------


BARS = pd.bdate_range("2020-01-06", periods=14, name="date")
BAR_DAYS = list(BARS.strftime("%Y-%m-%d"))


def flat_rows(closes, volumes=1000.0):
    rows = pd.DataFrame({col: closes for col in MEMORY_OHLCV}, index=BARS[: len(closes)])
    return rows.assign(volume=volumes).astype(float)


def rows_loader(folder, rows, dividends, splits, calendar=None, **kwargs):
    """`dividends` and `splits` are {ex-date: value}, `calendar` is the rows of a calendar asset."""
    (folder / "prices").mkdir(parents=True)
    rows.to_csv(folder / "prices" / "A.csv")
    if calendar is not None:
        calendar.to_csv(folder / "prices" / "CALENDAR.csv")
        kwargs["calender_asset"] = "CALENDAR"
    write_events(folder / "dividends", "A", "amount", dividends)
    write_events(folder / "splits", "A", "ratio", splits)
    return spectre.data.CsvDirLoader(
        str(folder / "prices"),
        ohlcv=MEMORY_OHLCV,
        dividends_path=str(folder / "dividends"),
        splits_path=str(folder / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "dividends, splits, after",
    [
        ({}, {BAR_DAYS[7]: 0.5}, 50.0),
        ({BAR_DAYS[7]: 10.0}, {}, 90.0),
        ({BAR_DAYS[7]: 10.0}, {BAR_DAYS[7]: 0.5}, 45.0),
    ],
    ids=["split", "dividend", "both"],
)
@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_zeros_on_an_ex_date_are_missing_prices(tmp_path, device, dividends, splits, after):
    rows = flat_rows([100.0] * 7 + [after] * 7)
    zeros, nans = rows.copy(), rows.copy()
    zeros.iloc[7:9] = 0.0
    nans.iloc[7:9] = nan
    loader = rows_loader(tmp_path / "zeros", zeros, dividends, splits)

    engine = spectre.factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(spectre.factors.Returns(), "returns")
    engine.add(spectre.factors.MA(3), "ma")
    dates = rows.index.tz_localize("UTC")
    result = engine.run(dates[2], dates[-1], delay_factor=False)
    # the value of a share never moves, so a window is flat in the scale of its last bar
    assert_allclose(result.ma, [100] * 5 + [after] * 7, rtol=1e-6)
    adjusted = pd.Series(after, index=rows.index).where(zeros.close != 0)
    assert_allclose(result.returns, adjusted.pct_change(fill_method=None).iloc[2:], atol=1e-6)

    columns = [*MEMORY_OHLCV[:4], "ex-dividend", "split_ratio", "price_multi", "vol_multi"]
    df = loader.test_load()[columns]
    pd.testing.assert_frame_equal(
        df, rows_loader(tmp_path / "nans", nans, dividends, splits).test_load()[columns]
    )


def test_zeros_without_an_event_are_filled(tmp_path):
    rows = flat_rows([100, 101, 0, 0, 104, 52, 53, 0, 55])
    df = rows_loader(tmp_path, rows, {}, {BAR_DAYS[5]: 0.5}).test_load()
    assert list(df.close) == [100, 101, 101, 101, 104, 52, 53, 53, 55]


def test_zeros_up_to_an_ex_date_are_filled_until_it(tmp_path):
    rows = flat_rows([100, 100, 0, 0, 0, 50, 50])
    df = rows_loader(tmp_path, rows, {}, {BAR_DAYS[3]: 0.5}).test_load()
    assert_array_equal(df.close, [100, 100, 100, nan, nan, 50, 50])


def test_zeros_after_an_ex_date_are_filled_in_the_new_scale(tmp_path):
    rows = flat_rows([100, 100, 50, 0, 0, 50])
    df = rows_loader(tmp_path, rows, {}, {BAR_DAYS[2]: 0.5}).test_load()
    assert list(df.close) == [100, 100, 50, 50, 50, 50]


def test_zero_on_the_first_bar_after_an_ex_date_without_a_bar(tmp_path):
    rows = flat_rows([100, 100, 0, 50, 50]).drop(BARS[1])
    df = rows_loader(tmp_path, rows, {}, {BAR_DAYS[1]: 0.5}).test_load()
    assert_array_equal(df.close, [100, nan, 50, 50])


def test_zero_volume_on_an_ex_date(tmp_path):
    """A split scales the volume, a dividend doesn't. The prices of that bar are valid."""
    rows = flat_rows([100.0, 100, 100, 50, 50], [1000.0, 1000, 1000, 0, 2000])
    df = rows_loader(tmp_path / "split", rows, {}, {BAR_DAYS[3]: 0.5}).test_load()
    assert_array_equal(df.volume, [1000, 1000, 1000, nan, 2000])
    assert list(df.close) == [100, 100, 100, 50, 50]

    df = rows_loader(tmp_path / "dividend", rows, {BAR_DAYS[3]: 1.0}, {}).test_load()
    assert list(df.volume) == [1000, 1000, 1000, 1000, 2000]


def test_zeros_are_judged_by_column(tmp_path):
    rows = flat_rows([100.0, 100, 50, 50])
    rows.iloc[2, rows.columns.get_loc("close")] = 0.0
    df = rows_loader(tmp_path, rows, {}, {BAR_DAYS[2]: 0.5}).test_load()
    assert_array_equal(df.close, [100, 100, nan, 50])
    assert list(df.open) == [100, 100, 50, 50]


def test_zero_next_to_an_event_in_the_prices_csv(tmp_path):
    rows = flat_rows([100.0, 100, 0, 0, 50]).assign(**{"ex-dividend": 0.0, "split_ratio": 1.0})
    rows.iloc[2, rows.columns.get_loc("split_ratio")] = 0.5
    (tmp_path / "prices").mkdir()
    rows.to_csv(tmp_path / "prices" / "A.csv")
    df = spectre.data.CsvDirLoader(
        str(tmp_path / "prices"),
        ohlcv=MEMORY_OHLCV,
        adjustments=("ex-dividend", "split_ratio"),
        parse_dates=True,
    ).test_load()
    assert_array_equal(df.close, [100, 100, nan, nan, 50])


@pytest.mark.parametrize("ex_bar", [1, 2], ids=["on_the_gap", "on_the_zero"])
def test_zero_next_to_an_event_with_align_by_time(tmp_path, ex_bar):
    rows = flat_rows([100, 100, 0, 50, 50]).drop(BARS[1])
    loader = rows_loader(
        tmp_path, rows, {}, {BAR_DAYS[ex_bar]: 0.5}, flat_rows([1] * 5), align_by_time=True
    )
    df = loader.test_load().xs("A", level=1)
    assert_array_equal(df.close, [100, nan, nan, 50, 50])


def test_zero_filled_from_a_row_the_calendar_drops(tmp_path):
    """It's filled as before, because no event is between that row and the zero."""
    rows = flat_rows([100, 50, 0, 50])
    loader = rows_loader(tmp_path, rows, {}, {BAR_DAYS[0]: 0.5}, flat_rows([1] * 4).drop(BARS[1]))
    assert list(loader.test_load().close.xs("A", level=1)) == [100, 50, 50]


def test_zero_of_another_asset_is_not_next_to_an_event(tmp_path):
    """Each asset has its own events, the rows of the next asset don't see them."""
    prices = {"A": dict.fromkeys(DAYS[:3], 100.0), "B": {DAYS[0]: 100.0, DAYS[1]: 0.0}}
    df = events_loader(tmp_path, prices, {}, {DAYS[2]: 0.5}).test_load()
    assert list(df.close.xs("B", level=1)) == [100, 100]


# ------------------------------- row order of the files -------------------------------

ROW_ORDERS = {
    "oldest_first": lambda rows: rows,
    "newest_first": lambda rows: rows.iloc[::-1],
    "shuffled": lambda rows: rows.sample(frac=1, random_state=0),
}


def random_bars(dates, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(dates)))
    bars = pd.DataFrame({col: closes for col in MEMORY_OHLCV}, index=dates.rename("date"))
    bars["open"] *= 1 + rng.normal(0, 0.01, len(dates))
    bars["volume"] = rng.integers(1000, 2000, len(dates)).astype(float)
    return bars


def load_in_order(folder, order, assets, **kwargs):
    """Write `assets`, {name: rows}, one csv each with the rows in `order`, and load them."""
    folder.mkdir()
    for name, rows in assets.items():
        ROW_ORDERS[order](rows).to_csv(folder / f"{name}.csv")
    return spectre.data.CsvDirLoader(
        str(folder), ohlcv=MEMORY_OHLCV, parse_dates=True, **kwargs
    ).load()


@pytest.mark.parametrize("order", ["newest_first", "shuffled"])
def test_zeros_are_filled_from_the_past_in_any_row_order(tmp_path, order):
    dates = pd.bdate_range("2023-01-02", periods=60)
    rows = random_bars(dates)
    rows.iloc[20] = 0.0  # a day without data
    rows.iloc[[30, 31]] = 0.0  # a halt of two days
    rows.iloc[40, rows.columns.get_loc("volume")] = 0.0  # valid prices, no volume
    assets = {"A": rows, "B": random_bars(dates, seed=1)}

    oldest_first = load_in_order(tmp_path / "oldest", "oldest_first", assets)
    other = load_in_order(tmp_path / "other", order, assets)
    pd.testing.assert_frame_equal(other, oldest_first)

    expected = rows.replace(0, nan).ffill()
    assert_allclose(other.xs("A", level=1)[list(MEMORY_OHLCV)], expected)
    assert_allclose(other.xs("B", level=1)[list(MEMORY_OHLCV)], assets["B"])


def test_zeros_are_filled_from_the_past_in_any_order_of_year_files(tmp_path, monkeypatch):
    dates = pd.bdate_range("2021-12-27", "2023-01-13")
    rows = random_bars(dates)
    rows.iloc[int(np.flatnonzero(dates.year == 2022)[0])] = 0.0
    for year in (2021, 2022, 2023):
        rows[dates.year == year].to_csv(tmp_path / f"A_{year}.csv")

    real_glob = glob.glob
    monkeypatch.setattr(glob, "glob", lambda pattern: sorted(real_glob(pattern), reverse=True))
    df = spectre.data.CsvDirLoader(
        str(tmp_path), prices_by_year=True, ohlcv=MEMORY_OHLCV, parse_dates=True
    ).load()
    assert_allclose(df[list(MEMORY_OHLCV)], rows.replace(0, nan).ffill())


@pytest.mark.parametrize("order", ["newest_first", "shuffled"])
@pytest.mark.parametrize(
    "cut, bars", [("2023-02-01", 38), ("2023-02-04", 35)], ids=["on_a_bar", "between_bars"]
)
def test_earliest_date_in_any_row_order(tmp_path, order, cut, bars):
    dates = pd.bdate_range("2023-01-02", periods=60)
    assets = {"A": random_bars(dates)}
    earliest_date = pd.Timestamp(cut)

    oldest_first = load_in_order(
        tmp_path / "oldest", "oldest_first", assets, earliest_date=earliest_date
    )
    other = load_in_order(tmp_path / "other", order, assets, earliest_date=earliest_date)
    pd.testing.assert_frame_equal(other, oldest_first)
    assert list(other.index.get_level_values(0)) == list(dates[-bars:].tz_localize("UTC"))


# ------------------------------ earliest_date and event files ------------------------------


def test_tz_aware_earliest_date_with_naive_ex_dates(tmp_path):
    closes = {f"{day} 09:30:00-05:00": 100.0 for day in DAYS[:5]}
    dividends = {DAYS[0]: 9.0}  # before `earliest_date`
    splits = {DAYS[3]: 0.5}
    earliest_date = pd.Timestamp(DAYS[1], tz="America/New_York")
    df = events_loader(
        tmp_path, {"A": closes}, dividends, splits, earliest_date=earliest_date
    ).test_load()
    assert len(df) == 4
    assert (df["ex-dividend"] == 0).all()
    assert list(df["split_ratio"]) == [1, 0.5, 1, 1]


def test_naive_earliest_date_with_tz_aware_ex_dates(tmp_path):
    closes = dict.fromkeys(DAYS[:5], 100.0)
    splits = {f"{DAYS[3]} 00:00:00-05:00": 0.5}
    df = events_loader(
        tmp_path, {"A": closes}, NO_DIVIDEND, splits, earliest_date=pd.Timestamp(DAYS[1])
    ).test_load()
    assert list(df["split_ratio"]) == [1, 0.5, 1, 1]


def test_tz_aware_earliest_date_with_year_files_and_naive_ex_dates():
    """The prices of the test data have a timezone, and its ex-dates don't."""
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / "5mins"),
        prices_by_year=True,
        prices_index="Date",
        ohlcv=INTRADAY_OHLCV,
        dividends_path=str(DATA_DIR / "dividends"),
        splits_path=str(DATA_DIR / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
        earliest_date=pd.Timestamp("2019-01-01", tz="America/New_York"),
    )
    df = loader.test_load()
    assert df.index.get_level_values(0)[0] == pd.Timestamp("2019-01-02 09:30", tz="US/Eastern")
    msft = df.xs("MSFT", level=1)
    assert (msft["split_ratio"] != 1).sum() == 1
    assert (msft["ex-dividend"] != 0).sum() == 1


def test_earliest_date_keeps_the_events_of_kept_rows(tmp_path):
    """The year files are kept whole, so bars can be before `earliest_date`, and the event too."""
    closes = pd.Series({DAYS[0]: 100.0, DAYS[1]: 100.0, DAYS[2]: 50.0, DAYS[3]: 50.0})
    frame = pd.DataFrame({col: closes for col in MEMORY_OHLCV}).rename_axis("date")
    (tmp_path / "prices").mkdir()
    frame.to_csv(tmp_path / "prices" / "A_2020.csv")
    write_events(tmp_path / "dividends", "A", "amount", {DAYS[2]: 1.0})
    write_events(tmp_path / "splits", "A", "ratio", {DAYS[2]: 0.5})
    df = spectre.data.CsvDirLoader(
        str(tmp_path / "prices"),
        prices_by_year=True,
        ohlcv=MEMORY_OHLCV,
        dividends_path=str(tmp_path / "dividends"),
        splits_path=str(tmp_path / "splits"),
        adjustments=("amount", "ratio"),
        parse_dates=True,
        earliest_date=pd.Timestamp(DAYS[3]),
    ).test_load()
    assert list(df.close) == [100, 100, 50, 50]
    assert list(df["ex-dividend"]) == [0, 1, 0, 0]
    assert list(df["split_ratio"]) == [1, 0.5, 1, 1]
