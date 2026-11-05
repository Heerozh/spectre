from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import spectre
from spectre import factors, parallel
from tests.conftest import (
    DIRTY_ADJUSTMENTS,
    DIRTY_DATES,
    DIRTY_OHLCV,
    DIRTY_SECTORS,
    DIVIDEND_ON,
    LISTED_ON,
    NAN_BAR,
    NANSPLIT_ON,
    SPLIT_ON,
)
from tests.factors import reference as ref
from tests.factors.helpers import (
    aligned_bars,
    assert_factor,
    delete,
    nansplit_bars,
    run_dirty,
    run_factor,
    run_late_listed,
)

nan = np.nan
SOLO = 6  # MSFT has no data on 2019-01-10


def returns(df):
    return ref.returns(df.close)


def engine_of(closes, device="cpu", **columns):
    """
    An engine on a table of closes, dates by assets, where NaN means the asset has no row.
    `columns` are more columns of the data, each a dict with a value for each asset.
    """
    df = closes.stack().dropna().rename("close").rename_axis(["date", "asset"]).to_frame()
    for col in ("open", "high", "low"):
        df[col] = df.close
    df["volume"] = 1.0
    for name, values in columns.items():
        df[name] = df.index.get_level_values("asset").map(values).astype(float)
    ohlcv = ("open", "high", "low", "close", "volume")
    engine = factors.FactorEngine(spectre.data.MemoryLoader(df, ohlcv=ohlcv))
    if device == "cuda":
        engine.to_cuda()
    return engine


# ------------------------------------ market features ------------------------------------


def returns_of_assets():
    return pd.DataFrame({a: returns(ref.daily_prices(a)) for a in ref.ASSETS})


def assert_market_number(engine, factor, values):
    """
    `values` has a number for each date, and both assets must read it, delayed by a date of the
    market. MSFT has no 2019-01-10, and on the other dates it reads what AAPL reads, 01-11 is its
    bar after the gap.
    """
    # a delayed run drops the first date
    dates = ref.daily_prices("AAPL").loc[ref.START : ref.END].index[1:]
    msft_dates = dates[dates.isin(ref.daily_prices("MSFT").index)]
    delayed = values.shift(1)
    assert_factor(engine, factor, delayed[dates].values, delayed[msft_dates].values)

    aapl, msft = run_factor(engine, factor, check_bias=False)
    assert len(msft) == len(aapl) - 1
    assert_array_equal(msft, aapl[dates.isin(msft_dates)])


def test_market_dispersion(engine):
    dispersion = returns_of_assets().std(axis=1, ddof=0)
    assert_market_number(engine, factors.MarketDispersion(), dispersion)
    # only AAPL has a return on that date
    assert dispersion.loc["2019-01-10":"2019-01-10"].item() == 0


def test_market_return(engine):
    assert_market_number(engine, factors.MarketReturn(), returns_of_assets().mean(axis=1))


def test_market_volatility(engine):
    """A number of the market: it rolls over the dates of the market, and is delayed by them."""
    market_return = returns_of_assets().mean(axis=1)
    volatility = (market_return.rolling(252).var(ddof=0) * 252) ** 0.5
    assert_market_number(engine, factors.MarketVolatility(), volatility)


MARKET_DATES = pd.date_range("2020-01-01", periods=60, freq="B", tz="UTC")
MARKET_WIN = 10
LISTED_BAR, DELISTED_BAR = 30, 40
HALT = slice(20, 32)  # the first bar after the halt is 32
EDGES_OF_ASSETS = [("LISTED", LISTED_BAR), ("HALTED", HALT.stop), ("DELISTED", DELISTED_BAR)]
each_delay = pytest.mark.parametrize("delay", [False, True], ids=["not_delayed", "delayed"])
ALIGNED = pytest.mark.parametrize("align", [False, True], ids=["not_aligned", "aligned"])
# each factor with one number for a date, and that number from the returns of the assets
MARKET_NUMBERS = {
    "return": (factors.MarketReturn, lambda r: r.mean(axis=1)),
    "dispersion": (factors.MarketDispersion, lambda r: r.std(axis=1, ddof=0)),
    "advance_decline": (
        factors.AdvanceDeclineRatio,
        lambda r: (r > 0).sum(axis=1) / (r < 0).sum(axis=1),
    ),
    "volatility": (
        lambda: factors.MarketVolatility(win=MARKET_WIN),
        lambda r: (r.mean(axis=1).rolling(MARKET_WIN, min_periods=1).var(ddof=0) * 252) ** 0.5,
    ),
}
each_market_number = pytest.mark.parametrize("name", MARKET_NUMBERS)


@pytest.fixture(scope="module")
def market_closes():
    """No splits or NaN prices, so the returns are simple. Each defect is an asset of its own."""
    rng = np.random.default_rng(1)
    assets = ("DELISTED", "FULL", "HALTED", "LISTED", "OTHER")
    bars = len(MARKET_DATES)
    closes = pd.DataFrame(
        {asset: 100 * np.cumprod(1 + rng.normal(0, 0.02, bars)) for asset in assets},
        index=MARKET_DATES,
    )
    closes.loc[: MARKET_DATES[LISTED_BAR - 1], "LISTED"] = nan
    closes.loc[MARKET_DATES[HALT], "HALTED"] = nan
    closes.loc[MARKET_DATES[DELISTED_BAR + 1] :, "DELISTED"] = nan
    return closes


def asset_returns(closes, aligned=False):
    """The return of a row is from the bar before it, and if aligned, every date is a bar."""
    if aligned:
        by_asset = closes.pct_change(fill_method=None)
    else:
        by_asset = pd.DataFrame({a: closes[a].dropna().pct_change() for a in closes})
    return by_asset.reindex(closes.index)


def market_volatility(closes, win=MARKET_WIN, annualization=252, aligned=False):
    """Rolls over the market return of every date, the mean of the assets with a return on it."""
    market = asset_returns(closes, aligned).mean(axis=1)
    return (market.rolling(win, min_periods=1).var(ddof=0) * annualization) ** 0.5


def run_market(engine, factor, start=0, delay=False, name="test"):
    engine.add(factor, name, replace=True)
    return engine.run(MARKET_DATES[start], MARKET_DATES[-1], delay_factor=delay)[name]


def assert_rows(result, expected):
    """`expected` has a number for a date, each row of `result` must be that number."""
    assert_allclose(result.values, expected[result.index.get_level_values("date")].values)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_delay
@pytest.mark.parametrize("start", [0, 25, 34], ids=lambda bar: f"bar_{bar}")
def test_market_volatility_is_one_number_for_a_date(market_closes, device, delay, start):
    """The start next to the halt loads older rows of the halted asset, and it reads none."""
    engine = engine_of(market_closes, device)
    result = run_market(engine, factors.MarketVolatility(win=MARKET_WIN), start, delay)
    expected = market_volatility(market_closes)
    assert_rows(result, expected.shift(1) if delay else expected)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_market_number
@each_delay
@ALIGNED
@pytest.mark.parametrize("start", [0, 25, 34], ids=lambda bar: f"bar_{bar}")
def test_market_number_is_the_same_for_all_assets(market_closes, device, name, delay, align, start):
    make, reference = MARKET_NUMBERS[name]
    engine = engine_of(market_closes, device)
    engine.align_by_time = align
    result = run_market(engine, make(), start, delay)
    expected = reference(asset_returns(market_closes, align))
    assert_rows(result, expected.shift(1) if delay else expected)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_market_number
@each_delay
@pytest.mark.parametrize("asset, bar", EDGES_OF_ASSETS, ids=[a for a, _ in EDGES_OF_ASSETS])
def test_market_number_on_the_edges_of_an_asset(market_closes, device, name, delay, asset, bar):
    """Its first bar, its first bar after a halt, and its last bar read what the market reads."""
    engine = engine_of(market_closes, device)
    result = run_market(engine, MARKET_NUMBERS[name][0](), delay=delay).unstack()
    assert result.loc[MARKET_DATES[bar], asset] == result.loc[MARKET_DATES[bar], "FULL"]
    assert not np.isnan(result.loc[MARKET_DATES[bar], asset])


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_delay
def test_market_volatility_with_align_by_time(market_closes, device, delay):
    engine = engine_of(market_closes, device)
    engine.align_by_time = True
    result = run_market(engine, factors.MarketVolatility(win=MARKET_WIN), delay=delay)
    expected = market_volatility(market_closes, aligned=True)
    assert_rows(result, expected.shift(1) if delay else expected)


def test_market_volatility_of_windows_that_are_not_full(market_closes, device):
    """At the start of the data, on the valid dates of the window. The first has no returns."""
    engine = engine_of(market_closes, device)
    with pytest.warns(RuntimeWarning, match="delay_factor is set to False"):
        result = run_market(engine, factors.MarketVolatility(win=MARKET_WIN))
    first_dates = result.loc[MARKET_DATES[:3]].unstack()
    assert np.isnan(first_dates.iloc[0]).all()
    # one return, its variance is 0 as in pandas
    assert (first_dates.iloc[1].dropna() == 0).all()
    assert (first_dates.iloc[2].dropna() > 0).all()


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_market_volatility_keeps_its_arguments(market_closes, device):
    engine = engine_of(market_closes, device)
    result = run_market(engine, factors.MarketVolatility(win=5))
    assert_rows(result, market_volatility(market_closes, win=5))

    result = run_market(engine, factors.MarketVolatility(inputs=[factors.MarketReturn(), 365]))
    assert_rows(result, market_volatility(market_closes, win=252, annualization=365))

    result = run_market(engine, factors.MarketVolatility.sequential(5, factors.MarketReturn(), 12))
    assert_rows(result, market_volatility(market_closes, win=5, annualization=12))

    class ShortVolatility(factors.MarketVolatility):
        win = 5

    assert_rows(run_market(engine, ShortVolatility()), market_volatility(market_closes, win=5))


def test_market_volatility_needs_a_window():
    with pytest.raises(ValueError, match="win"):
        factors.MarketVolatility(win=1)


def test_market_volatility_counts_its_window_in_the_history():
    # 9 older dates for the window, and one for the first return
    assert factors.MarketVolatility(win=10).get_total_backwards_() == 10
    assert factors.MarketVolatility(win=10).shift(2).get_total_backwards_() == 12


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_market_number
@pytest.mark.parametrize("periods", [1, 3, -2])
def test_market_number_shift_is_by_dates(market_closes, device, name, periods):
    """So a shift doesn't read different dates on the bars of different assets."""
    make, reference = MARKET_NUMBERS[name]
    engine = engine_of(market_closes, device)
    expected = reference(asset_returns(market_closes)).shift(periods)
    assert_rows(run_market(engine, make().shift(periods), 5), expected)


@each_market_number
def test_market_number_shift_by_0_is_itself(name):
    factor = MARKET_NUMBERS[name][0]()
    assert factor.shift(0) is factor


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_delay
def test_market_return_of_each_sector_stays_on_the_bars_of_an_asset(market_closes, device, delay):
    """Grouped by more than the date it has a number for each sector, so it's not one for a date."""
    sectors = {"DELISTED": 1, "FULL": 1, "HALTED": 2, "LISTED": 2, "OTHER": 2}
    engine = engine_of(market_closes, device, sector=sectors)
    factor = factors.MarketReturn()
    factor.groupby = "date,sector"
    result = run_market(engine, factor, 5, delay)

    by_sector = asset_returns(market_closes).T.groupby(sectors).mean().T
    for asset, rows_of_asset in result.groupby(level="asset", observed=True):
        expected = by_sector[sectors[asset]].reindex(market_closes[asset].dropna().index)
        expected = expected.shift(1) if delay else expected
        dates = rows_of_asset.index.get_level_values("date")
        assert_allclose(rows_of_asset.values, expected[dates].values, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_market_volatility_as_an_input(market_closes, device):
    """Other factors go on with it by the bars of an asset, as they do with all inputs."""
    engine = engine_of(market_closes, device)
    factor = factors.MA(3, inputs=[factors.MarketVolatility(win=MARKET_WIN)])
    result = run_market(engine, factor, 5).xs("FULL", level="asset")
    expected = market_volatility(market_closes).rolling(3).mean()
    assert_allclose(result.values, expected[result.index].values)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_market_volatility_with_the_first_asset_masked(market_closes, device):
    """The market return of a date is on all its rows, the others still have it."""
    engine = engine_of(market_closes, device)
    # DELISTED is the first asset of the dates it has rows on
    others = factors.AssetClassifierDataFactor({"DELISTED": 1}, 0) == 0
    factor = factors.MarketVolatility(win=MARKET_WIN).set_mask(others)
    assert_rows(run_market(engine, factor), market_volatility(market_closes))


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@each_market_number
def test_market_number_of_one_asset(market_closes, device, name):
    """With one asset the date group is another class."""
    make, reference = MARKET_NUMBERS[name]
    engine = engine_of(market_closes, device)
    engine.set_filter(factors.StaticAssets({"FULL"}))
    expected = reference(asset_returns(market_closes[["FULL"]]))
    assert_rows(run_market(engine, make()), expected)
    assert_rows(run_market(engine, make(), 5, True), expected.shift(1))


def test_market_volatility_on_the_dirty_panel(dirty_engine):
    """Splits and NaN prices change the market return, not that it's one number for a date."""
    factor = factors.MarketVolatility(win=MARKET_WIN)
    dirty_engine.add(factor, "test")
    result = dirty_engine.run(DIRTY_DATES[5], DIRTY_DATES[-1]).test
    assert result.notna().all()
    assert (result.groupby(level="date").nunique() == 1).all()
    dirty_engine.test_lookahead_bias(DIRTY_DATES[5], DIRTY_DATES[-1])


def test_advance_decline_ratio(engine):
    by_asset = returns_of_assets()
    ratio = (by_asset > 0).sum(axis=1) / (by_asset < 0).sum(axis=1)
    assert_market_number(engine, factors.AdvanceDeclineRatio(), ratio)


def test_asset_data(engine):
    """AssetData broadcasts the data of an asset to all assets, data must be aligned by time."""
    engine.align_by_time = True
    msft = ref.daily_prices("MSFT").close.reindex(ref.daily_prices("AAPL").index)
    expected = msft.shift(1).loc["2019-01-03" : ref.END].values
    assert np.isnan(expected[SOLO])
    assert_factor(engine, factors.AssetData("MSFT", factors.OHLCV.close), expected, expected)


def test_asset_data_warns_not_aligned(engine):
    with pytest.warns(RuntimeWarning, match="Make sure your data is aligned by time"):
        run_factor(engine, factors.AssetData("MSFT", factors.OHLCV.close), check_bias=False)


ASSET_DATES = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
ASSET_START, ASSET_END = ASSET_DATES[15], ASSET_DATES[19]


@pytest.fixture
def asset_closes():
    """Each asset has its own hundreds. A is gone before the range, and E comes after it."""
    bars = np.arange(len(ASSET_DATES), dtype=float)
    closes = pd.DataFrame(
        {asset: i * 100 + bars for i, asset in enumerate("ABCDE", start=1)}, index=ASSET_DATES
    )
    closes.loc[ASSET_DATES[10] :, "A"] = nan
    closes.loc[: ASSET_DATES[24], "E"] = nan
    return closes


def run_asset_data(engine, target, align, start=ASSET_START):
    engine.align_by_time = align
    engine.add(factors.AssetData(target, factors.OHLCV.close), "test", replace=True)
    return engine.run(start, ASSET_END, delay_factor=False).test.unstack("asset")


@pytest.mark.filterwarnings("ignore:Make sure your data is aligned by time")
@ALIGNED
@pytest.mark.parametrize("target", ["B", "C", "D"])
@pytest.mark.parametrize("timezone", ["UTC", "Asia/Tokyo"])
def test_asset_data_next_to_an_asset_out_of_range(asset_closes, device, target, align, timezone):
    """Categories keep the assets that are not in the range, the rows of a factor don't."""
    engine = engine_of(asset_closes, device)
    engine.timezone = timezone
    result = run_asset_data(engine, target, align)
    assert list(result.columns) == ["B", "C", "D"]
    expected = asset_closes[target].loc[ASSET_START:ASSET_END].values
    for asset in result:
        assert_array_equal(result[asset].values, expected)


@pytest.mark.filterwarnings("ignore:Make sure your data is aligned by time")
@ALIGNED
@pytest.mark.parametrize("target", ["B", "D"])
def test_asset_data_of_static_assets(asset_closes, device, target, align):
    """StaticAssets screens the assets out before the rows are made."""
    engine = engine_of(asset_closes, device)
    engine.set_filter(factors.StaticAssets({"B", "D"}))
    result = run_asset_data(engine, target, align)
    assert list(result.columns) == ["B", "D"]
    expected = asset_closes[target].loc[ASSET_START:ASSET_END].values
    for asset in result:
        assert_array_equal(result[asset].values, expected)


@pytest.mark.filterwarnings("ignore:Make sure your data is aligned by time")
@ALIGNED
@pytest.mark.parametrize(
    "target", ["A", "E", "NOT_AN_ASSET"], ids=["delisted", "not_listed_yet", "unknown"]
)
def test_asset_data_of_asset_without_rows(asset_closes, device, target, align):
    """It raises, as an unknown name always did, and doesn't read the asset next to it."""
    engine = engine_of(asset_closes, device)
    with pytest.raises(KeyError, match=f"{target} has no rows"):
        run_asset_data(engine, target, align)

    # and the engine is not broken by it
    result = run_asset_data(engine, "C", align)
    assert_array_equal(result.B.values, asset_closes.C.loc[ASSET_START:ASSET_END].values)


@pytest.mark.filterwarnings("ignore:Make sure your data is aligned by time")
@ALIGNED
def test_asset_data_of_asset_screened_out(asset_closes, device, align):
    engine = engine_of(asset_closes, device)
    engine.set_filter(factors.StaticAssets({"B", "D"}))
    with pytest.raises(KeyError, match="C has no rows"):
        run_asset_data(engine, "C", align)


@pytest.mark.filterwarnings("ignore:Make sure your data is aligned by time")
@ALIGNED
def test_asset_data_of_asset_only_in_the_history(asset_closes, device, align):
    """A has rows in the history that MA loads. It reads them, and nothing in the range."""
    engine = engine_of(asset_closes, device)
    engine.add(factors.MA(5), "ma")
    result = run_asset_data(engine, "A", align, start=ASSET_DATES[12])
    assert np.isnan(result.values).all()


# ------------------------------------ calendar features ------------------------------------


@pytest.mark.parametrize(
    "factor, attr",
    [
        (factors.WEEKDAY, "day_of_week"),
        (factors.MONTH, "month"),
        (factors.QUARTER, "quarter"),
        (factors.DatetimeDataFactor("day"), "day"),
    ],
)
def test_datetime_factor(engine, factor, attr):
    # date features are available before market open, not delayed
    expected = ref.expected(
        lambda df: pd.Series(getattr(df.index, attr), index=df.index), delayed=False
    )
    assert_factor(engine, factor, *expected)


def test_weekday_values(engine):
    expected = np.array([2, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0])
    assert_factor(engine, factors.WEEKDAY, expected, delete(expected, SOLO))


def test_is_january(engine):
    assert_factor(engine, factors.IS_JANUARY, [True] * 10, [True] * 9)
    assert_factor(engine, factors.IS_DECEMBER, [False] * 10, [False] * 9)


def test_timezone(engine):
    engine.timezone = "America/New_York"

    def weekday_in_ny(df):
        ny = df.index.tz_localize("UTC").tz_convert("America/New_York")
        return pd.Series(ny.day_of_week, index=df.index)

    # 2019-01-02 00:00 UTC is 2019-01-01 19:00 in New York
    expected = ref.expected(weekday_in_ny, delayed=False)
    assert_array_equal(expected[0], [1.0, 2, 3, 6, 0, 1, 2, 3, 6, 0])
    assert_factor(engine, factors.WEEKDAY, *expected)

    # change back, cached data should be invalidated
    engine.timezone = "UTC"
    expected = np.array([2, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0])
    assert_factor(engine, factors.WEEKDAY, expected, delete(expected, SOLO))


# ------------------------------------ month and quarter flags ----------------------------------

# factor, period, last date of the period or first
PERIOD_FLAGS = {
    "month_end": (factors.IS_MONTH_END, "M", True),
    "month_start": (factors.IS_MONTH_START, "M", False),
    "quarter_end": (factors.IS_QUARTER_END, "Q", True),
    "quarter_start": (factors.IS_QUARTER_START, "Q", False),
}
each_flag = pytest.mark.parametrize("flag", PERIOD_FLAGS)


def local_dates(stamps, timezone="UTC"):
    return stamps.tz_convert(timezone).tz_localize(None).normalize()


def is_business_edge(date, freq, last):
    """No business day of the period of `date` comes after it, or before it."""
    period = date.to_period(freq)
    business_days = pd.bdate_range(period.start_time, period.end_time)
    return date >= business_days[-1] if last else date <= business_days[0]


def expected_flag(flag, rows, calendar, timezone="UTC"):
    """
    The flag of each of `rows`, a date index with duplicates or not. In `calendar`, the dates of
    the frame, it's the last date of a period, or the first. The date at the end of the calendar
    has no neighbour, it goes by the business days.
    """
    _, freq, last = PERIOD_FLAGS[flag]
    dates = local_dates(calendar, timezone).unique().sort_values()
    by_period = dates.to_series().groupby(dates.to_period(freq))
    edges = by_period.max() if last else by_period.min()
    end = dates[-1] if last else dates[0]
    if not is_business_edge(end, freq, last):
        edges = edges[edges != end]
    return local_dates(rows, timezone).isin(edges)


def flag_of_rows(result, flag, calendar, timezone="UTC"):
    return expected_flag(flag, result.index.get_level_values("date"), calendar, timezone)


@each_flag
@pytest.mark.parametrize("start", [0, 23, 24, 33, 40], ids=lambda bar: f"bar_{bar}")
def test_period_flag_is_of_the_date_not_of_the_bars_of_an_asset(dirty_engine, flag, start):
    """2020-02-03 (bar 23) is the first trading date of February, in the history of bar 24."""
    result = run_dirty(dirty_engine, PERIOD_FLAGS[flag][0], start)
    assert result.dtype == bool
    assert_array_equal(result.values, flag_of_rows(result, flag, DIRTY_DATES))


def dates_of(result, asset):
    return list(result[result].xs(asset, level="asset").index.strftime("%m-%d"))


def test_month_flags_of_the_dirty_panel(dirty_engine):
    """Written out, for the bars an asset starts and stops on."""
    ends = run_dirty(dirty_engine, factors.IS_MONTH_END)
    starts = run_dirty(dirty_engine, factors.IS_MONTH_START)
    # the first and the last bar of a run are not special
    assert dates_of(ends, "FULL") == ["01-31", "02-28"]
    assert dates_of(starts, "FULL") == ["01-01", "02-03", "03-02"]
    # delisted on 02-26, in the middle of a month
    assert dates_of(ends, "DELISTED") == ["01-31"]
    # halted from 01-29 to 02-13, over the end of January
    assert dates_of(ends, "HALTED") == ["02-28"]
    assert dates_of(starts, "HALTED") == ["01-01", "03-02"]
    # listed on 02-12
    assert dates_of(starts, "LISTED") == ["03-02"]


@pytest.mark.parametrize(
    "last_date, month_end, quarter_end",
    [
        ("2020-03-30", False, False),
        ("2020-03-31", True, True),
        ("2020-04-30", True, False),
        # the 29th is a Saturday
        ("2020-02-28", True, False),
        ("2020-02-27", False, False),
        # the 30th and the 31st are the weekend
        ("2020-05-29", True, False),
        ("2020-05-28", False, False),
    ],
)
def test_period_end_on_the_last_date_of_the_data(device, last_date, month_end, quarter_end):
    """Nothing comes after it to compare with, a calendar is known in advance."""
    dates = pd.bdate_range(end=last_date, periods=10, tz="UTC")
    engine = engine_of(pd.DataFrame({"A": 100.0}, index=dates), device)
    for factor, expected in (
        (factors.IS_MONTH_END, month_end),
        (factors.IS_QUARTER_END, quarter_end),
    ):
        engine.add(factor, "test", replace=True)
        result = engine.run(dates[0], dates[-1], delay_factor=False).test
        assert result.iloc[-1] == expected


@pytest.mark.parametrize(
    "first_date, month_start, quarter_start",
    [
        ("2020-02-03", True, False),
        ("2020-02-04", False, False),
        ("2020-04-01", True, True),
        ("2020-04-02", False, False),
        # the 1st is a Sunday
        ("2020-03-02", True, False),
        ("2020-03-03", False, False),
        # a holiday, but only a calendar of a market would know
        ("2020-01-02", False, False),
    ],
)
def test_period_start_on_the_first_date_of_the_data(device, first_date, month_start, quarter_start):
    dates = pd.bdate_range(first_date, periods=10, tz="UTC")
    engine = engine_of(pd.DataFrame({"A": 100.0}, index=dates), device)
    flags = (factors.IS_MONTH_START, month_start), (factors.IS_QUARTER_START, quarter_start)
    for factor, expected in flags:
        engine.add(factor, "test", replace=True)
        result = engine.run(dates[0], dates[-1], delay_factor=False).test
        assert result.iloc[0] == expected


def test_period_flags_on_a_holiday_next_to_the_end_of_a_month(device):
    """The dates of the data are the calendar, so the last weekday can be closed."""
    dates = pd.bdate_range("2020-03-23", "2020-04-10", tz="UTC")
    closed = pd.to_datetime(["2020-03-31", "2020-04-01"], utc=True)
    dates = dates.drop(closed)
    engine = engine_of(pd.DataFrame({"A": 100.0}, index=dates), device)

    engine.add(factors.IS_MONTH_END, "end")
    engine.add(factors.IS_MONTH_START, "start")
    engine.add(factors.IS_QUARTER_END, "quarter_end")
    engine.add(factors.IS_QUARTER_START, "quarter_start")
    result = engine.run(dates[0], dates[-1], delay_factor=False).xs("A", level="asset")
    assert list(result.index[result.end].strftime("%m-%d")) == ["03-30"]
    assert list(result.index[result.start].strftime("%m-%d")) == ["04-02"]
    assert list(result.index[result.quarter_end].strftime("%m-%d")) == ["03-30"]
    assert list(result.index[result.quarter_start].strftime("%m-%d")) == ["04-02"]

    # a run from that date reads the date before it, by business days it would be no month start
    april_2 = pd.Timestamp("2020-04-02", tz="UTC")
    assert engine.run(april_2, dates[-1], delay_factor=False).start.iloc[0]


def test_period_flags_tell_years_apart(device):
    """No dates between 2 Januaries, it's the same month number."""
    dates = pd.to_datetime(["2019-01-14", "2019-01-15", "2020-01-14", "2020-01-15"], utc=True)
    engine = engine_of(pd.DataFrame({"A": 100.0}, index=dates), device)
    engine.add(factors.IS_MONTH_END, "end")
    engine.add(factors.IS_MONTH_START, "start")
    result = engine.run(dates[0], dates[-1], delay_factor=False).xs("A", level="asset")
    assert_array_equal(result.end, [False, True, False, False])
    assert_array_equal(result.start, [False, False, True, False])


@each_flag
@pytest.mark.parametrize(
    "timezone, hour",
    [
        # 00:00 is the evening before, and Monday is a Sunday
        ("America/New_York", 0),
        ("America/New_York", 15),
        # 16:00 is the morning after
        ("Asia/Tokyo", 16),
        # the midnight of 2020-09-06 doesn't exist there, and bars of that date do
        ("America/Santiago", 0),
    ],
)
def test_period_flag_in_a_timezone(device, flag, timezone, hour):
    dates = pd.bdate_range("2020-03-02", "2020-09-30", tz="UTC") + pd.Timedelta(hours=hour)
    closes = pd.DataFrame({"A": 100.0, "B": 200.0, "C": 300.0}, index=dates)
    closes.loc[dates[50:70], "B"] = nan
    closes.loc[dates[100] :, "C"] = nan

    engine = engine_of(closes, device)
    engine.timezone = timezone
    engine.add(PERIOD_FLAGS[flag][0], "test")
    result = engine.run(dates[0], dates[-1], delay_factor=False).test
    assert_array_equal(result.values, flag_of_rows(result, flag, dates, timezone))


def test_period_flags_differ_by_the_timezone(device):
    """April starts on a Wednesday in UTC, which is the evening of March 31 in New York."""
    dates = pd.bdate_range("2020-03-02", "2020-05-29", tz="UTC")
    engine = engine_of(pd.DataFrame({"A": 100.0}, index=dates), device)
    engine.add(factors.IS_MONTH_END, "test")
    utc = engine.run(dates[0], dates[-1], delay_factor=False).test.xs("A", level="asset")
    assert utc[utc].index.strftime("%m-%d").tolist() == ["03-31", "04-30", "05-29"]

    engine.timezone = "America/New_York"
    utc_rows = engine.run(dates[0], dates[-1], delay_factor=False).test.xs("A", level="asset")
    utc_rows.index = utc_rows.index.tz_convert("UTC")
    # stamped 04-01 and 05-01, the dates in New York are 03-31 and 04-30
    assert utc_rows[utc_rows].index.strftime("%m-%d").tolist() == ["04-01", "05-01"]


@each_flag
@pytest.mark.parametrize("timezone", ["UTC", "America/New_York", "Asia/Tokyo"])
def test_period_flag_on_all_bars_of_a_date(intraday_loader, device, flag, timezone):
    """5 minutes bars. In Tokyo a date changes in the middle of a session."""
    engine = factors.FactorEngine(intraday_loader)
    if device == "cuda":
        engine.to_cuda()
    engine.timezone = timezone
    engine.add(PERIOD_FLAGS[flag][0], "test")
    stamps = intraday_loader.load().index.levels[0]
    result = engine.run(stamps[0], stamps[-1], delay_factor=False).test
    assert_array_equal(result.values, flag_of_rows(result, flag, stamps, timezone))


def test_period_flag_of_a_day_of_bars(intraday_loader):
    engine = factors.FactorEngine(intraday_loader)
    engine.add(factors.IS_MONTH_END, "test")
    stamps = intraday_loader.load().index.levels[0]
    result = engine.run(stamps[0], stamps[-1], delay_factor=False).test
    flagged = result[result].groupby(level="asset", observed=True).size()
    # 78 bars of 5 minutes, of each of AAPL and MSFT on 2018-12-31, the last date of the month
    assert flagged.to_dict() == {"AAPL": 78, "MSFT": 78}


def test_period_flags_with_static_assets(dirty_engine):
    """Screened out rows are not in the frame, FULL brings the dates of the market."""
    dirty_engine.set_filter(factors.StaticAssets({"DELISTED", "FULL"}))
    result = run_dirty(dirty_engine, factors.IS_MONTH_END)
    assert dates_of(result, "FULL") == ["01-31", "02-28"]
    assert dates_of(result, "DELISTED") == ["01-31"]


def test_period_flags_of_a_delisted_asset_alone(dirty_engine):
    """The last date of the frame is its last bar, in the middle of the month."""
    dirty_engine.set_filter(factors.StaticAssets({"DELISTED"}))
    assert dates_of(run_dirty(dirty_engine, factors.IS_MONTH_END), "DELISTED") == ["01-31"]


@each_flag
def test_period_flag_on_rows_added_by_align_by_time(dirty_engine, flag):
    dirty_engine.align_by_time = True
    result = run_dirty(dirty_engine, PERIOD_FLAGS[flag][0])
    expected = flag_of_rows(result, flag, DIRTY_DATES)
    assert len(result) == len(DIRTY_DATES) * len(DIRTY_SECTORS)
    assert_array_equal(result.values, expected)


@each_flag
def test_period_flag_is_a_filter(dirty_engine, flag):
    factor = PERIOD_FLAGS[flag][0]
    assert isinstance(factor, factors.FilterFactor)
    assert not factor.should_delay()

    flags = run_dirty(dirty_engine, factor)
    assert_array_equal(run_dirty(dirty_engine, ~factor).values, ~flags.values)
    assert_array_equal(run_dirty(dirty_engine, factor.float()).values, flags.values.astype(float))
    mondays = factors.WEEKDAY == 0
    expected = flags.values & (flags.index.get_level_values("date").day_of_week == 0)
    assert_array_equal(run_dirty(dirty_engine, factor & mondays).values, expected)
    shifted = flags.groupby(level="asset", observed=True).shift(1, fill_value=False)
    assert_array_equal(run_dirty(dirty_engine, factor.shift(1)).values, shifted.values)

    # as a mask of a cross-section
    opens = run_dirty(dirty_engine, factors.OHLCV.open)
    ranked = run_dirty(dirty_engine, factors.OHLCV.open.rank(mask=factor))
    assert_array_equal(ranked.notna().values, flags.values & opens.notna().values)
    # as the filter of the engine
    dirty_engine.set_filter(factor)
    result = run_dirty(dirty_engine, factors.OHLCV.open)
    assert result.index.equals(flags[flags].index)


def test_period_flags_are_shared_by_two_engines(dirty_engine, device):
    """They are objects of the module, and these engines have different calendars."""
    dates = pd.bdate_range("2021-06-01", "2021-08-31", tz="UTC")
    other = engine_of(pd.DataFrame({"A": 100.0}, index=dates), device)
    for _ in range(2):
        ours = run_dirty(dirty_engine, factors.IS_MONTH_END).xs("FULL", level="asset")
        assert ours[ours].index.strftime("%m-%d").tolist() == ["01-31", "02-28"]

        other.add(factors.IS_MONTH_END, "test", replace=True)
        theirs = other.run(dates[0], dates[-1], delay_factor=False).test.xs("A", level="asset")
        assert theirs[theirs].index.strftime("%m-%d").tolist() == ["06-30", "07-30", "08-31"]


@each_flag
def test_period_flag_has_no_lookahead(dirty_engine, flag):
    dirty_engine.add(PERIOD_FLAGS[flag][0], "test")
    dirty_engine.test_lookahead_bias(DIRTY_DATES[5], DIRTY_DATES[-1])


def test_period_flag_history():
    """Only the first date of a period needs the date before `start`."""
    assert factors.IS_MONTH_END.get_total_backwards_() == 0
    assert factors.IS_QUARTER_END.get_total_backwards_() == 0
    assert factors.IS_MONTH_START.get_total_backwards_() == 1
    assert factors.IS_QUARTER_START.get_total_backwards_() == 1


# ------------------------------------ classifier ------------------------------------


def test_asset_classifier(engine):
    classifier = factors.AssetClassifierDataFactor({"AAPL": 2}, -1)
    assert_factor(engine, classifier, [2] * 10, [-1] * 9)


def test_one_hot(engine, device):
    if device == "cuda":
        engine.to_cuda()
    one_hot = factors.AssetClassifierDataFactor({"AAPL": 2}, -1).one_hot()
    # classes are sorted: -1, 2
    assert_factor(engine, one_hot[0], [False] * 10, [True] * 9)
    assert_factor(engine, one_hot[1], [True] * 10, [False] * 9)
    with pytest.raises(KeyError, match="OutOfBounds"):
        run_factor(engine, one_hot[2])


# ------------------------------------ label ------------------------------------


def test_forward_signal_data(engine):
    """Close price of the first signal in the next 4 bars (lookahead biased)."""
    signal = factors.OHLCV.close > 150
    signal_price = factors.ForwardSignalData(4, factors.OHLCV.close, signal)
    signal_return = signal_price / factors.OHLCV.close - 1

    def forward_signal_return(df):
        close = df.close.loc[: ref.END]  # no future data after `end`
        window = pd.concat([close.shift(-i) for i in range(4)], axis=1)
        first = window.where(window > 150).bfill(axis=1).iloc[:, 0]
        return first / close - 1

    expected = ref.expected(forward_signal_return)
    assert_almost_equal(
        expected[0], [0, 151.55 / 145.23 - 1, 151.55 / 149 - 1, 151.55 / 149 - 1, 0, 0, 0, 0, 0]
    )
    # MSFT never > 150
    assert np.isnan(expected[1]).all()
    assert_factor(engine, signal_return, *expected, check_bias=False)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_of_new_listed_asset(late_listing_engine):
    """
    ForwardSignalData shifts the data backward, so the first `win - 1` bars of a newly listed
    asset are unknown (nan), the shifted out data is not in the window.
    """
    close = factors.OHLCV.close
    result = run_late_listed(late_listing_engine, factors.ForwardSignalData(3, close, close > 15))
    # closes: [10, 20, 11, 30, 12, 13, 14]
    assert_almost_equal(result, [nan, nan, 30, 30, nan, nan, nan])


FORWARD_WIN = 4
# Bars 36 and 46, the bars after the split and the dividend, so rows before them cross them.
FORWARD_SIGNAL = factors.WEEKDAY == 3


def forward_signal_values(values):
    """Value of the first signal in bars t to t + 3. The first 3 bars are unknown, as above."""
    signal = pd.Series(values.index.day_of_week == 3, index=values.index)
    window = pd.concat([values.shift(-i) for i in range(FORWARD_WIN)], axis=1)
    masks = pd.concat([signal.shift(-i, fill_value=False) for i in range(FORWARD_WIN)], axis=1)
    first = window.where(masks.values).bfill(axis=1).iloc[:, 0]
    first.iloc[: FORWARD_WIN - 1] = nan
    return first


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_across_split(dirty_engine, dirty_panel):
    """It is compared with the close of its row, so it must be in the scale of that row."""
    close = factors.OHLCV.close
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    first = forward_signal_values(split.clean_close)

    factor = factors.ForwardSignalData(FORWARD_WIN, close, FORWARD_SIGNAL)
    result = run_dirty(dirty_engine, factor).xs("SPLIT", level="asset")
    assert_allclose(result.values, (first * scale).values, rtol=1e-6)

    result = run_dirty(dirty_engine, factor / close - 1).xs("SPLIT", level="asset")
    assert_allclose(result.values, (first / split.clean_close - 1).values, rtol=0, atol=1e-6)
    # rows that cross the split and the dividend, both ex-dates are days before the signal
    for bar in (SPLIT_ON - 2, SPLIT_ON - 1, DIVIDEND_ON - 2, DIVIDEND_ON - 1):
        assert abs(result.iloc[bar]) < 0.1


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_across_split_after_nan_bar(dirty_engine, dirty_panel):
    close = factors.OHLCV.close
    bars, scale = nansplit_bars(dirty_panel)
    first = forward_signal_values(bars.clean_close)

    factor = factors.ForwardSignalData(FORWARD_WIN, close, FORWARD_SIGNAL)
    result = run_dirty(dirty_engine, factor).xs("NANSPLIT", level="asset")
    assert_allclose(result.values, (first * scale).values, rtol=1e-6)

    result = run_dirty(dirty_engine, factor / close - 1).xs("NANSPLIT", level="asset")
    assert_allclose(result.values, (first / bars.clean_close - 1).values, rtol=0, atol=1e-6)
    # rows that cross the split, the signal of the bar on the split is in reach of both
    for bar in (NANSPLIT_ON - 3, NANSPLIT_ON - 2):
        assert abs(result.iloc[bar]) < 0.1


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_of_volume_across_split(dirty_engine, dirty_panel):
    """Volume has its own multipliers, more shares after a split are not more volume."""
    split = dirty_panel.xs("SPLIT", level="asset")
    shares = pd.Series(np.where(split.index < DIRTY_DATES[SPLIT_ON], 1, 2.0), index=split.index)
    first = forward_signal_values(split.volume / shares)

    factor = factors.ForwardSignalData(FORWARD_WIN, factors.OHLCV.volume, FORWARD_SIGNAL)
    result = run_dirty(dirty_engine, factor).xs("SPLIT", level="asset")
    assert_allclose(result.values, (first * shares).values, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_on_rows_added_by_align_by_time(added_rows_engine, added_rows_panel):
    """
    The rows before a split are in the scale of the bar before them, as the loader would have made
    them. So on the rows of NANSPLIT in front of its split, the signal on it is at the old scale.
    """
    close = factors.OHLCV.close
    factor = factors.ForwardSignalData(FORWARD_WIN, close, FORWARD_SIGNAL)
    result = run_dirty(added_rows_engine, factor).unstack("asset")
    for asset in ("FULL", "NANSPLIT", "LISTED", "DELISTED", "HALTED"):
        bars, scale = aligned_bars(added_rows_panel, asset)
        expected = forward_signal_values(bars.clean_close) * scale
        assert_allclose(result[asset], expected, rtol=1e-6, err_msg=asset)
    # the signal of NANSPLIT on the first bar of the split, seen from the 3 added rows
    assert result.NANSPLIT.iloc[NANSPLIT_ON - 3 : NANSPLIT_ON].notna().all()
    # LISTED is listed on bar 30 and has a signal on bar 31, seen from 2 added rows
    assert result.LISTED.iloc[LISTED_ON - 2 : LISTED_ON].notna().all()


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "make_factor",
    [
        lambda: factors.ForwardSignalData(FORWARD_WIN, factors.OHLCV.close, FORWARD_SIGNAL),
        lambda: factors.ForwardSignalData(FORWARD_WIN, factors.OHLCV.volume, FORWARD_SIGNAL),
        lambda: factors.MACD(5, 10, 3),
    ],
    ids=["ForwardSignalData close", "ForwardSignalData volume", "MACD"],
)
def test_rows_added_by_the_engine_match_rows_from_the_loader(
    added_rows_engine, added_rows_panel, device, make_factor
):
    """
    A loader with align_by_time adds the rows before it makes the multipliers, the engine after.
    What is filled in for the engine has to be what the loader would have made.
    """
    index = pd.MultiIndex.from_product(
        [DIRTY_DATES, added_rows_panel.index.get_level_values("asset").unique()],
        names=added_rows_panel.index.names,
    )
    aligned = added_rows_panel.reindex(index).sort_index()
    aligned["ex-dividend"] = aligned["ex-dividend"].fillna(0)
    aligned["split_ratio"] = aligned["split_ratio"].fillna(1)
    engine = factors.FactorEngine(
        spectre.data.MemoryLoader(aligned, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    )
    if device == "cuda":
        engine.to_cuda()

    from_loader = run_dirty(engine, make_factor()).unstack("asset")
    from_engine = run_dirty(added_rows_engine, make_factor()).unstack("asset")
    assert_allclose(from_engine.values, from_loader.values, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_on_bar_without_price(dirty_engine, dirty_panel):
    """A bar with no close still gets the data of the bars after it."""
    close = factors.OHLCV.close
    factor = factors.ForwardSignalData(FORWARD_WIN, close, FORWARD_SIGNAL)
    result = run_dirty(dirty_engine, factor).xs("NANBAR", level="asset")
    expected = forward_signal_values(dirty_panel.xs("NANBAR", level="asset").close)
    assert not np.isnan(expected.iloc[NAN_BAR])
    assert_allclose(result.values, expected.values, rtol=1e-6)


@pytest.mark.cuda
def test_forward_signal_data_on_cpu_with_multipliers_on_cuda():
    """When CUDA runs out of memory `compute` runs again on cpu, the column stays on cuda."""
    close = factors.OHLCV.close
    factor = factors.ForwardSignalData(2, close, close > 0)
    factor.data = SimpleNamespace(adjustments=torch.tensor([[0.5, 0.5, 1, 1]], device="cuda"))
    # raw closes [100, 110, 60, 70] with a 2:1 split on the third bar, and the signal on it
    shifted_data = torch.tensor([[110, 60, 70, nan]])
    shifted_signal = torch.tensor([[False, True, False, False]])
    rolling = parallel.Rolling(shifted_data, 2), parallel.Rolling(shifted_signal, 2)
    result = factor.compute(*rolling)
    assert result.device.type == "cpu"
    assert_almost_equal(result[0].numpy(), [nan, 120, 60, nan])


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_forward_signal_data_of_other_data_is_not_adjusted(dirty_engine, dirty_panel):
    """The rule for factors without multipliers is unchanged, they are only shifted."""
    closes = factors.OHLCV.close * 1
    factor = factors.ForwardSignalData(FORWARD_WIN, closes, FORWARD_SIGNAL)
    result = run_dirty(dirty_engine, factor).xs("SPLIT", level="asset")
    expected = forward_signal_values(dirty_panel.xs("SPLIT", level="asset").close)
    assert_allclose(result.values, expected.values, rtol=1e-6)
