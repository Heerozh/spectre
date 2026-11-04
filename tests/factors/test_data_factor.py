import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

import spectre
from spectre import factors
from tests.conftest import DIRTY_ADJUSTMENTS, DIRTY_DATES, DIRTY_OHLCV

START, END = "2019-01-11", "2019-01-15"


def values(df, asset, col):
    return df.loc[(slice(None), asset), col].values


def test_column_data(engine_no_calendar):
    engine = engine_no_calendar
    engine.add(factors.OHLCV.volume, "volume")
    df = engine.run(START, END)
    # delayed, volume of 2019-01-11 and 2019-01-14
    assert_array_equal(values(df, "AAPL", "volume"), (28065422, 33834032))
    assert_array_equal(values(df, "MSFT", "volume"), (28627674, 28720936))

    engine.add(factors.ColumnDataFactor(inputs=("changePercent",)), "chg")
    df = engine.run(START, END)
    assert_array_equal(values(df, "AAPL", "chg"), (-0.9835, -1.5724))
    assert_array_equal(values(df, "MSFT", "chg"), (-0.8025, -0.7489))


@pytest.mark.parametrize("delay", [False, True])
def test_open_not_delayed(engine_no_calendar, delay):
    engine_no_calendar.add(factors.OHLCV.open, "open")
    df = engine_no_calendar.run(START, END, delay_factor=delay)
    assert_array_equal(values(df, "AAPL", "open"), (155.72, 155.19, 150.81))
    assert_array_equal(values(df, "MSFT", "open"), (104.65, 104.9, 103.19))


def test_column_data_dtype(engine):
    engine.add(factors.ColumnDataFactor(inputs=("uClose",), dtype=torch.float32), "close")
    df = engine.run(START, END)
    assert df.close.dtype == np.float32


@pytest.mark.parametrize(
    "fill_na, expected",
    [
        (None, [1.0, np.nan, 3.0]),
        ("ffill", [1.0, 1.0, 3.0]),
        ("bfill", [1.0, 3.0, 3.0]),
    ],
)
def test_series_data(engine, fill_na, expected):
    dates = pd.to_datetime(["2019-01-11", "2019-01-15"], utc=True)
    index = pd.MultiIndex.from_arrays([dates, ["AAPL", "AAPL"]], names=["date", "asset"])
    series = pd.Series([1.0, 3.0], index=index, name="my_data")
    engine.add(factors.SeriesDataFactor(series, fill_na=fill_na, should_delay=False), "test")
    df = engine.run(START, END)
    assert_array_equal(values(df, "AAPL", "test"), expected)
    assert np.isnan(values(df, "MSFT", "test")).all()


def test_series_data_invalid_fill_na(engine):
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2019-01-11"], utc=True), ["AAPL"]], names=["date", "asset"]
    )
    series = pd.Series([1.0], index=index, name="my_data")
    engine.add(factors.SeriesDataFactor(series, fill_na="zero"), "test")
    with pytest.raises(ValueError, match="`fill_na` must be one of"):
        engine.run(START, END)
    # the data of the first run must not let the second one through
    with pytest.raises(ValueError, match="`fill_na` must be one of"):
        engine.run(START, END)


# ------------------------------------ series and the frame ----------------------------------

SERIES_RANGE = DIRTY_DATES[10], DIRTY_DATES[14]
OTHER_RANGE = DIRTY_DATES[40], DIRTY_DATES[44]


def series_factor(series, **kwargs):
    return factors.SeriesDataFactor(series, should_delay=False, **kwargs)


def read(engine, factor, dates=SERIES_RANGE, asset="FULL"):
    engine.add(factor, "test", replace=True)
    return engine.run(*dates, delay_factor=False).test.xs(asset, level="asset")


def rows(series, dates=SERIES_RANGE, asset="FULL"):
    return series.xs(asset, level="asset").loc[dates[0] : dates[1]].values


def test_series_named_like_a_column(dirty_engine, dirty_panel):
    """The operations of pandas keep the name, so it says nothing about the data."""
    doubled = dirty_panel.close * 2
    assert doubled.name == "close"
    assert_allclose(read(dirty_engine, series_factor(doubled)), rows(doubled))


def test_series_derived_from_a_column(dirty_engine, dirty_panel):
    change = dirty_panel.close.groupby(level="asset").pct_change()
    assert change.name == "close"
    result = read(dirty_engine, series_factor(change))
    assert_allclose(result, rows(change))
    assert abs(result).max() < 0.1


def test_series_without_a_name(dirty_engine, dirty_panel):
    doubled = (dirty_panel.close * 2).rename(None)
    assert_allclose(read(dirty_engine, series_factor(doubled)), rows(doubled))


def test_series_with_one_name_in_one_run(dirty_engine, dirty_panel):
    double = (dirty_panel.close * 2).rename("signal")
    triple = (dirty_panel.close * 3).rename("signal")
    dirty_engine.add(series_factor(double), "double")
    dirty_engine.add(series_factor(triple), "triple")
    result = dirty_engine.run(*SERIES_RANGE, delay_factor=False).xs("FULL", level="asset")
    assert_allclose(result.double, rows(double))
    assert_allclose(result.triple, rows(triple))


def test_series_with_one_name_in_two_runs(dirty_engine, dirty_panel):
    """These runs share the frame, it's not loaded again."""
    double = (dirty_panel.close * 2).rename("signal")
    triple = (dirty_panel.close * 3).rename("signal")
    assert_allclose(read(dirty_engine, series_factor(double)), rows(double))
    loaded = dirty_engine.cache_hash
    assert_allclose(read(dirty_engine, series_factor(triple)), rows(triple))
    assert dirty_engine.cache_hash == loaded


def with_a_gap(dirty_panel, name="signal"):
    """A series on the close of each row, but FULL has no value on bars 11 and 12."""
    gap = [(date, "FULL") for date in DIRTY_DATES[11:13]]
    return dirty_panel.close.drop(index=gap).rename(name)


@pytest.mark.parametrize(
    "fill_na, method",
    [("ffill", "ffill"), ("pad", "ffill"), ("bfill", "bfill"), ("backfill", "bfill")],
)
def test_series_fill_na_of_a_series_named_like_a_column(dirty_engine, dirty_panel, fill_na, method):
    series = with_a_gap(dirty_panel, name="close")
    result = read(dirty_engine, series_factor(series, fill_na=fill_na))
    full = series.xs("FULL", level="asset").reindex(DIRTY_DATES[10:15])
    expected = getattr(full, method)()
    assert full.isna().sum() == 2
    assert_allclose(result, expected.values)


def test_series_fill_na_of_each_instance(dirty_engine, dirty_panel):
    """On the same series, and on the same frame."""
    series = with_a_gap(dirty_panel)
    full = series.xs("FULL", level="asset").reindex(DIRTY_DATES[10:15])
    raw = read(dirty_engine, series_factor(series))
    assert_array_equal(np.isnan(raw.values), full.isna().values)
    filled = read(dirty_engine, series_factor(series, fill_na="ffill"))
    assert_allclose(filled, full.ffill().values)


def test_series_keeps_its_data_in_repeated_runs(dirty_engine, dirty_panel):
    doubled = dirty_panel.close * 2
    factor = series_factor(doubled)
    for _ in range(3):
        assert_allclose(read(dirty_engine, factor), rows(doubled))


def test_series_after_the_frame_is_reloaded(dirty_engine, dirty_panel):
    doubled = dirty_panel.close * 2
    factor = series_factor(doubled)
    assert_allclose(read(dirty_engine, factor), rows(doubled))
    loaded = dirty_engine.cache_hash
    assert_allclose(read(dirty_engine, factor, OTHER_RANGE), rows(doubled, OTHER_RANGE))
    assert dirty_engine.cache_hash != loaded
    assert_allclose(read(dirty_engine, factor), rows(doubled))


def test_series_fill_na_after_the_frame_is_reloaded(dirty_engine, dirty_panel):
    series = with_a_gap(dirty_panel)
    factor = series_factor(series, fill_na="ffill")
    full = series.xs("FULL", level="asset").reindex(DIRTY_DATES[10:15]).ffill()
    assert_allclose(read(dirty_engine, factor), full.values)
    read(dirty_engine, factor, OTHER_RANGE)
    assert_allclose(read(dirty_engine, factor), full.values)


def test_series_shared_by_two_engines(dirty_engine, dirty_panel, device):
    """The second engine has other rows. Runs of the two take turns, one frame is still loaded."""
    doubled = dirty_panel.close * 2
    factor = series_factor(doubled)

    loader = spectre.data.MemoryLoader(dirty_panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    other = factors.FactorEngine(loader)
    other.set_filter(factors.StaticAssets({"FULL", "SPLIT"}))
    if device == "cuda":
        other.to_cuda()

    for _ in range(2):
        assert_allclose(read(dirty_engine, factor), rows(doubled))
        assert_allclose(read(other, factor, OTHER_RANGE), rows(doubled, OTHER_RANGE))
        split = read(other, factor, OTHER_RANGE, "SPLIT")
        assert_allclose(split, rows(doubled, OTHER_RANGE, "SPLIT"))


def test_series_in_two_places_of_a_graph(dirty_engine, dirty_panel):
    doubled = dirty_panel.close * 2
    factor = series_factor(doubled)
    dirty_engine.add(factor, "series")
    dirty_engine.add(factors.MA(3, inputs=[factor]), "ma")
    result = dirty_engine.run(*SERIES_RANGE, delay_factor=False).xs("FULL", level="asset")
    assert_allclose(result.series, rows(doubled))
    expected = doubled.xs("FULL", level="asset").rolling(3).mean()
    # the history of the window
    assert_allclose(result.ma, expected.loc[SERIES_RANGE[0] : SERIES_RANGE[1]].values)


def test_series_on_rows_added_by_align_by_time(dirty_engine, dirty_panel):
    """LISTED has no rows in the range, and it has nothing to read there."""
    dirty_engine.align_by_time = True
    doubled = dirty_panel.close * 2
    dates = DIRTY_DATES[28], DIRTY_DATES[32]
    result = read(dirty_engine, series_factor(doubled), dates, "LISTED")
    expected = doubled.xs("LISTED", level="asset").reindex(DIRTY_DATES[28:33])
    assert expected.isna().sum() == 2
    assert_allclose(result, expected.values)


def test_series_in_another_timezone(dirty_engine, dirty_panel):
    dirty_engine.timezone = "Asia/Tokyo"
    doubled = dirty_panel.close * 2
    assert_allclose(read(dirty_engine, series_factor(doubled)), rows(doubled))


def test_series_has_the_fill_na_method_of_all_factors(dirty_engine, dirty_panel):
    """The argument of the constructor picks the fill on the rows of the series, it's no method."""
    series = with_a_gap(dirty_panel)
    full = series.xs("FULL", level="asset").reindex(DIRTY_DATES[10:15])

    result = read(dirty_engine, series_factor(series).fill_na(-1.0))
    assert_allclose(result, full.fillna(-1.0).values)
    result = read(dirty_engine, series_factor(series).fill_na(ffill=True))
    assert_allclose(result, full.ffill().values)
    # both, the method has nothing left to fill
    result = read(dirty_engine, series_factor(series, fill_na="ffill").fill_na(-1.0))
    assert_allclose(result, full.ffill().values)


@pytest.mark.parametrize("new_instances", [False, True], ids=["one_instance", "new_instances"])
def test_series_does_not_grow_the_frame(dirty_engine, dirty_panel, new_instances):
    """
    The frame and the column cache live until the next load. A loop that makes a new series on
    every run of one range must not leave anything for each of them.
    """
    doubled = dirty_panel.close * 2
    factor = series_factor(doubled)
    read(dirty_engine, factor)
    columns = dirty_engine.dataframe_.columns.copy()
    cached = len(dirty_engine._column_cache)
    loaded = dirty_engine.cache_hash
    for i in range(5):
        factor = series_factor(doubled.rename(f"signal{i}")) if new_instances else factor
        assert_allclose(read(dirty_engine, factor), rows(doubled))
        assert dirty_engine.cache_hash == loaded
        assert dirty_engine.dataframe_.columns.equals(columns)
        assert len(dirty_engine._column_cache) == cached


@pytest.mark.parametrize("fill_na", [None, "ffill"])
def test_series_with_duplicated_rows_raises(dirty_engine, dirty_panel, fill_na):
    """One row can't hold two values, and more rows than the frame would shift all the others."""
    doubled = dirty_panel.close * 2
    duplicated = pd.concat([doubled, doubled.iloc[:1]])
    dirty_engine.add(series_factor(duplicated, fill_na=fill_na), "test")
    with pytest.raises(ValueError, match="non-unique"):
        dirty_engine.run(*SERIES_RANGE, delay_factor=False)


# ------------------------------------ fill on the series ------------------------------------


def values_of(*points, name="eps"):
    """A series from (date, asset, value), with values on a few rows, like reports of companies."""
    index = pd.MultiIndex.from_tuples([p[:2] for p in points], names=["date", "asset"])
    return pd.Series([p[2] for p in points], index=index, name=name, dtype=float)


def filled(series, dates, fill_na, asset="FULL"):
    """The fill of pandas of one asset on `dates`, its values on all of the dates of the series."""
    values = series.xs(asset, level="asset").dropna().sort_index()
    return values.reindex(dates, method="ffill" if fill_na == "ffill" else "bfill")


@pytest.mark.parametrize(
    "start, other_factor",
    [(0, False), (10, False), (10, True)],
    ids=["from_bar_0", "from_bar_10", "from_bar_10_with_MA"],
)
def test_series_fill_does_not_depend_on_the_loaded_rows(dirty_engine, start, other_factor):
    series = values_of((DIRTY_DATES[3], "FULL", 1.0), (DIRTY_DATES[30], "FULL", 2.0))
    if other_factor:
        dirty_engine.add(factors.MA(10), "ma")
    dates = DIRTY_DATES[start], DIRTY_DATES[-1]
    result = read(dirty_engine, series_factor(series, fill_na="ffill"), dates)
    assert result.loc[DIRTY_DATES[15]] == 1.0
    assert result.loc[DIRTY_DATES[31]] == 2.0
    assert_allclose(result, filled(series, DIRTY_DATES[start:], "ffill").values)


def test_series_fill_from_a_date_without_a_bar(dirty_engine):
    """2020-01-04 is a Saturday, so it's on no row of the frame."""
    saturday = pd.Timestamp("2020-01-04", tz="UTC")
    assert saturday not in DIRTY_DATES
    series = values_of((saturday, "FULL", 1.0))
    result = read(
        dirty_engine, series_factor(series, fill_na="ffill"), (DIRTY_DATES[0], DIRTY_DATES[9])
    )
    assert_array_equal(result.values, filled(series, DIRTY_DATES[:10], "ffill").values)
    assert result.notna().sum() == 7

    result = read(
        dirty_engine, series_factor(series, fill_na="bfill"), (DIRTY_DATES[0], DIRTY_DATES[9])
    )
    assert_array_equal(result.values, filled(series, DIRTY_DATES[:10], "bfill").values)
    assert result.notna().sum() == 3


@pytest.mark.parametrize("fill_na", ["ffill", "bfill"])
def test_series_fill_past_the_last_date_of_the_series(dirty_engine, fill_na):
    """The last value goes on over the later rows, there is nothing to take for the other way."""
    series = values_of((DIRTY_DATES[5], "FULL", 1.0), (DIRTY_DATES[20], "FULL", 2.0))
    result = read(
        dirty_engine, series_factor(series, fill_na=fill_na), (DIRTY_DATES[0], DIRTY_DATES[-1])
    )
    assert_array_equal(result.values, filled(series, DIRTY_DATES, fill_na).values)
    assert result.iloc[-1] == 2.0 if fill_na == "ffill" else np.isnan(result.iloc[-1])


def test_series_bfill_from_a_value_after_the_loaded_rows(dirty_engine):
    series = values_of((DIRTY_DATES[30], "FULL", 2.0))
    result = read(dirty_engine, series_factor(series, fill_na="bfill"))
    assert (result == 2.0).all()


def test_series_fill_skips_the_nans_of_the_series(dirty_engine):
    """As a fill does, a NaN is no value."""
    series = values_of(
        (DIRTY_DATES[3], "FULL", 1.0),
        (DIRTY_DATES[5], "FULL", np.nan),
        (DIRTY_DATES[8], "FULL", 2.0),
    )
    dates = DIRTY_DATES[0], DIRTY_DATES[9]
    result = read(dirty_engine, series_factor(series, fill_na="ffill"), dates)
    assert result.loc[DIRTY_DATES[6]] == 1.0
    result = read(dirty_engine, series_factor(series, fill_na="bfill"), dates)
    assert result.loc[DIRTY_DATES[4]] == 2.0


def test_series_fill_without_a_value(dirty_engine):
    series = values_of((DIRTY_DATES[3], "FULL", np.nan))
    assert read(dirty_engine, series_factor(series, fill_na="ffill")).isna().all()


def test_series_fill_is_for_each_asset(dirty_engine):
    series = values_of(
        (DIRTY_DATES[3], "FULL", 1.0),
        (DIRTY_DATES[5], "SPLIT", 2.0),
        # not in the data at all
        (DIRTY_DATES[1], "NOT_AN_ASSET", 3.0),
    )
    dirty_engine.add(series_factor(series, fill_na="ffill"), "test")
    result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[9], delay_factor=False).test
    assert_array_equal(result.xs("FULL", level="asset").notna(), DIRTY_DATES[:10] >= DIRTY_DATES[3])
    assert_array_equal(
        result.xs("SPLIT", level="asset").notna(), DIRTY_DATES[:10] >= DIRTY_DATES[5]
    )
    assert result.xs("HALTED", level="asset").isna().all()


@pytest.mark.parametrize("fill_na", ["ffill", "bfill"])
@pytest.mark.parametrize("start", [0, 22, 34], ids=lambda bar: f"bar_{bar}")
def test_series_fill_of_all_assets(dirty_engine, dirty_panel, fill_na, start):
    """Bar 22 and 34 have the halt of HALTED before them, and rows of a listing after them."""
    rng = np.random.default_rng(0)
    series = dirty_panel.close[rng.random(len(dirty_panel)) < 0.2].dropna()
    dirty_engine.add(series_factor(series, fill_na=fill_na), "test")
    result = dirty_engine.run(DIRTY_DATES[start], DIRTY_DATES[-1], delay_factor=False).test
    for asset, rows_of_asset in result.groupby(level="asset", observed=True):
        dates = rows_of_asset.index.get_level_values("date")
        assert_allclose(rows_of_asset.values, filled(series, dates, fill_na, asset).values)


def test_series_fill_of_a_series_from_a_loader(dirty_engine, dirty_loader):
    """Its asset level is categorical, and with an asset the rows don't have."""
    series = dirty_loader.load().close.dropna()
    series = series.iloc[::7].rename("signal")
    assert series.index.levels[1].dtype == "category"
    dirty_engine.set_filter(factors.StaticAssets({"FULL", "SPLIT"}))
    dates = DIRTY_DATES[10], DIRTY_DATES[-1]
    result = read(dirty_engine, series_factor(series, fill_na="ffill"), dates, "SPLIT")
    assert_allclose(result, filled(series, DIRTY_DATES[10:], "ffill", "SPLIT").values)


def test_series_fill_ignores_a_row_without_an_asset(dirty_engine):
    """It's later than the value of SPLIT, the last label of the level, so it would win there."""
    series = values_of((DIRTY_DATES[4], "SPLIT", 2.0), (DIRTY_DATES[6], np.nan, 1.0))
    assert (read(dirty_engine, series_factor(series, fill_na="ffill"), asset="SPLIT") == 2.0).all()


def test_series_fill_on_rows_added_by_align_by_time(dirty_engine):
    """The last value of DELISTED goes on after its last bar, on rows it never had."""
    dirty_engine.align_by_time = True
    series = values_of((DIRTY_DATES[30], "DELISTED", 1.0))
    result = read(
        dirty_engine,
        series_factor(series, fill_na="ffill"),
        (DIRTY_DATES[35], DIRTY_DATES[-1]),
        "DELISTED",
    )
    assert len(result) == len(DIRTY_DATES[35:])
    assert (result == 1.0).all()


def test_series_fill_on_intraday_rows(intraday_loader, device):
    """A value a day, stamped before the session, is on no bar. It reaches them by the fill."""
    engine = factors.FactorEngine(intraday_loader)
    if device == "cuda":
        engine.to_cuda()
    stamps = intraday_loader.load().index.levels[0]
    days = stamps.normalize().unique()
    series = values_of(*[(day, "AAPL", float(i)) for i, day in enumerate(days)])

    start, end = stamps[stamps >= days[12]][0], stamps[-1]
    assert read(engine, series_factor(series), (start, end), "AAPL").isna().all()

    result = read(engine, series_factor(series, fill_na="ffill"), (start, end), "AAPL")
    assert_array_equal(result.values, filled(series, result.index, "ffill", "AAPL").values)
    # each day has its value, on all of its bars
    assert result.nunique() == result.index.normalize().nunique()


def test_series_fill_shared_by_two_engines(dirty_engine, device):
    """They have other rows, so what was read for one is not what the other wants."""
    series = values_of((DIRTY_DATES[3], "FULL", 1.0), (DIRTY_DATES[30], "FULL", 2.0))
    factor = series_factor(series, fill_na="ffill")

    closes = pd.DataFrame(
        {"close": 100.0},
        index=pd.MultiIndex.from_product(
            [DIRTY_DATES[10:], ["FULL", "OTHER"]], names=["date", "asset"]
        ),
    )
    for col in ("open", "high", "low"):
        closes[col] = closes.close
    closes["volume"] = 1.0
    ohlcv = ("open", "high", "low", "close", "volume")
    other = factors.FactorEngine(spectre.data.MemoryLoader(closes, ohlcv=ohlcv))
    if device == "cuda":
        other.to_cuda()

    for _ in range(2):
        dates = DIRTY_DATES[0], DIRTY_DATES[-1]
        assert_allclose(read(dirty_engine, factor, dates), filled(series, DIRTY_DATES, "ffill"))
        dates = DIRTY_DATES[10], DIRTY_DATES[-1]
        expected = filled(series, DIRTY_DATES[10:], "ffill")
        assert_allclose(read(other, factor, dates), expected)


def test_series_fill_of_the_same_frame_is_read_once(dirty_engine, monkeypatch):
    """Runs of one range share a frame, and the rows of the frame are all a fill depends on."""
    calls = []
    merge_asof = pd.merge_asof

    def counted(*args, **kwargs):
        calls.append(args)
        return merge_asof(*args, **kwargs)

    monkeypatch.setattr(pd, "merge_asof", counted)
    series = values_of((DIRTY_DATES[3], "FULL", 1.0), (DIRTY_DATES[30], "FULL", 2.0))
    factor = series_factor(series, fill_na="ffill")
    for _ in range(3):
        read(dirty_engine, factor)
    assert len(calls) == 1

    # a new frame has other rows
    result = read(dirty_engine, factor, OTHER_RANGE)
    assert len(calls) == 2
    assert (result == 2.0).all()
