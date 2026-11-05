import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import spectre
from spectre import factors
from tests.conftest import DELISTED_ON, DIRTY_ADJUSTMENTS, DIRTY_DATES, DIRTY_OHLCV
from tests.factors import reference as ref

close = factors.OHLCV.close


def test_run_raw(engine):
    engine.add(factors.OHLCV.open, "open")
    df = engine.run("2018-01-01", ref.END, False)
    raw = engine.run_raw("2018-01-01", ref.END, False)
    assert_almost_equal(df.open.values, raw["open"].numpy())


def test_run_raw_with_filter(engine):
    engine.add(factors.SMA(5), "ma5")
    engine.set_filter(factors.OHLCV.volume.top(1))
    df = engine.run(ref.START, ref.END)
    raw, index = engine.run_raw(ref.START, ref.END, return_index=True)
    assert_almost_equal(df.ma5.values, raw["ma5"].numpy())
    assert_array_equal(df.index.get_level_values(0), index)


def test_run_raw_index_without_filter(engine):
    engine.add(factors.SMA(5), "ma5")
    df = engine.run(ref.START, ref.END)
    raw, index = engine.run_raw(ref.START, ref.END, return_index=True)
    assert_almost_equal(df.ma5.values, raw["ma5"].numpy())
    assert_array_equal(df.index.get_level_values(0), index)


def test_run_up_to_the_last_bar(dirty_engine):
    start, last = DIRTY_DATES[30], DIRTY_DATES[-1]
    dirty_engine.add(factors.SMA(5), "ma5")
    expected = dirty_engine.run(start, last)
    pd.testing.assert_frame_equal(dirty_engine.run(start, None), expected)

    raw = dirty_engine.run_raw(start, None)
    assert_almost_equal(raw["ma5"].cpu().numpy(), dirty_engine.run_raw(start, last)["ma5"].cpu())
    assert dirty_engine.test_lookahead_bias(start, None) == "No assertion raised."


class GrowingLoader(spectre.data.MemoryLoader):
    """Has the bars up to `last_bar`, which moves on as a live loader gets new bars."""

    last_bar = DIRTY_DATES[-1]

    def _load(self):
        df = self.df.loc[: self.last_bar].copy()
        df.index = df.index.remove_unused_levels()
        return df


def growing_engine(panel, device):
    loader = GrowingLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    loader.last_bar = DIRTY_DATES[50]
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    return loader, engine


def test_run_up_to_the_last_bar_reads_the_new_bars(dirty_panel, device):
    start = DIRTY_DATES[30]
    loader, engine = growing_engine(dirty_panel, device)
    engine.add(factors.MA(3), "f")
    assert engine.run(start, None).index.get_level_values(0).max() == DIRTY_DATES[50]

    loader.last_bar = DIRTY_DATES[55]
    result = engine.run(start, None)
    assert result.index.get_level_values(0).max() == DIRTY_DATES[55]
    fresh = factors.FactorEngine(loader)
    fresh.add(factors.MA(3), "f")
    pd.testing.assert_frame_equal(result, fresh.run(start, None))

    raw = engine.run_raw(start, None)["f"]
    assert len(raw) == len(result)


def test_run_up_to_the_last_bar_reads_a_changed_bar(dirty_panel, device):
    """A live bar changes until it closes, the range of the data stays the same."""
    start, last = DIRTY_DATES[30], DIRTY_DATES[50]
    loader, engine = growing_engine(dirty_panel, device)
    engine.add(factors.OHLCV.open, "open")
    assert engine.run(start, None).open[(last, "FULL")] != 1234.5
    loader.df.loc[(last, "FULL"), "open"] = 1234.5
    assert engine.run(start, None).open[(last, "FULL")] == 1234.5


def test_run_with_an_end_still_reads_the_cache(dirty_panel, device):
    start, end = DIRTY_DATES[30], DIRTY_DATES[50]
    _, engine = growing_engine(dirty_panel, device)
    engine.add(factors.MA(3), "f")
    engine.run(start, end)
    loaded = engine.cache_hash
    engine.run(start, end)
    assert engine.cache_hash == loaded


def test_lookahead_bias_up_to_the_last_bar_still_sees_a_leak(dirty_engine):
    """It needs the cached data it has changed, a reload would give the same results twice."""
    dirty_engine.add(close.shift(-2), "biased")
    with pytest.raises(RuntimeError, match="look-ahead bias"):
        dirty_engine.test_lookahead_bias(DIRTY_DATES[30], None)


@pytest.mark.parametrize("method", ["run", "run_raw"])
def test_delayed_run_on_the_last_bar_has_no_data(dirty_engine, method):
    """The only bar of the range is dropped by the delay."""
    dirty_engine.add(close, "close")
    with pytest.raises(ValueError, match="There is no data between start and end"):
        getattr(dirty_engine, method)(DIRTY_DATES[-1], DIRTY_DATES[-1])


def test_no_trading_day(engine):
    engine.add(factors.SMA(2), "test")
    with pytest.raises(AssertionError, match="There is no data between.*"):
        engine.run("2019-01-05", "2019-01-06")


def test_no_factors(engine):
    with pytest.raises(ValueError, match="Please add at least one factor"):
        engine.run(ref.START, ref.END)


def test_delay_factor(engine):
    engine.add(close, "close")
    engine.add(factors.OHLCV.open, "open")
    df = engine.run(ref.START, ref.END)
    # close is available after market closed, so it's delayed, open is not
    aapl = ref.daily_prices("AAPL").loc["2019-01-03" : ref.END]
    previous = ref.daily_prices("AAPL").loc["2019-01-02":"2019-01-14"]
    assert_array_equal(df.xs("AAPL", level=1).close.values, previous.close.values)
    assert_array_equal(df.xs("AAPL", level=1).open.values, aapl.open.values)

    with pytest.warns(RuntimeWarning, match="delay_factor is set to False"):
        df = engine.run(ref.START, ref.END, delay_factor=False)
    assert df.index.get_level_values(0)[0] == pd.Timestamp("2019-01-02", tz="UTC")

    df = engine.run(ref.START, ref.END, delay_factor="all")
    assert_array_equal(df.xs("AAPL", level=1).open.values, previous.open.values)


def test_insufficient_history_warning(engine):
    engine.add(factors.SMA(20), "test")
    with pytest.warns(RuntimeWarning, match="Historical data seems insufficient"):
        engine.run("2014-10-27", "2014-11-30")


def test_max_backwards_warns_if_it_cuts_the_history(dirty_engine):
    dirty_engine.max_backwards = 5
    dirty_engine.add(factors.MA(20), "f")
    with pytest.warns(RuntimeWarning, match="max_backwards is 5, but 20 bars"):
        dirty_engine.run(DIRTY_DATES[40], DIRTY_DATES[-1])


def test_max_backwards_is_silent_if_the_history_fits(dirty_engine):
    dirty_engine.max_backwards = 20
    dirty_engine.add(factors.MA(20), "f")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dirty_engine.run(DIRTY_DATES[40], DIRTY_DATES[-1])


def test_delayed_run_needs_no_bar_before_the_data(dirty_engine):
    """The first bar of a delayed run is dropped, so it can start on the first bar of the data."""
    dirty_engine.add(close, "close")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[5])
    assert result.index.get_level_values(0).min() == DIRTY_DATES[1]


# ------------------------------------ history ------------------------------------


def run_on(engine, start, *others, delay_factor=True):
    """Run the factor in column `f`, with `others` next to it, each of them may load more data."""
    for i, other in enumerate(others):
        engine.add(other, f"other{i}", replace=True)
    return engine.run(start, DIRTY_DATES[-1], delay_factor=delay_factor).f


def assert_same_rows(result, expected):
    """Same values on the rows of both, the first bar is dropped if only one run is delayed."""
    common = result.index.intersection(expected.index)
    assert len(common.get_level_values(0).unique()) >= len(DIRTY_DATES) - 41
    pd.testing.assert_series_equal(result.loc[common], expected.loc[common], check_exact=False)


@pytest.mark.parametrize(
    "make_factor",
    [
        lambda: factors.MA(5, inputs=[close.shift(3)]),
        lambda: factors.RSI(14).shift(1),
        lambda: factors.MA(5).shift(4),
        lambda: factors.MA(5, inputs=[(close > 100).float().shift(3)]),
        lambda: factors.OHLCV.open.shift(2),
    ],
    ids=["ma_of_shift", "shift_of_rsi", "shift_of_ma", "bool_shift", "open"],
)
def test_result_does_not_depend_on_start(dirty_engine, make_factor):
    start = DIRTY_DATES[40]
    dirty_engine.add(make_factor(), "f")
    result = run_on(dirty_engine, start)
    assert result.notna().any()
    # all of the data
    from_the_beginning = run_on(dirty_engine, DIRTY_DATES[0])
    assert_same_rows(result, from_the_beginning)
    # MA(30) loads more history than the factor
    with_other_factors = run_on(dirty_engine, start, factors.MA(30))
    assert_same_rows(result, with_other_factors)


def test_shift_of_data_matches_pandas(dirty_engine, dirty_panel):
    start = DIRTY_DATES[40]
    full = dirty_panel.xs("FULL", level="asset")
    dirty_engine.add(factors.MA(5, inputs=[close.shift(3)]), "f")
    result = dirty_engine.run(start, DIRTY_DATES[-1]).f.xs("FULL", level=1)
    # delayed by one more bar, and the delay drops the first bar
    expected = full.close.shift(3).rolling(5).mean().shift(1).loc[DIRTY_DATES[41] :]
    assert_almost_equal(result.values, expected.values)


def test_month_start_on_a_start_in_the_middle_of_a_month(dirty_engine):
    start = pd.Timestamp("2020-02-11", tz="UTC")
    dirty_engine.add(factors.IS_MONTH_START, "month_start")
    result = dirty_engine.run(start, DIRTY_DATES[-1]).month_start.xs("FULL", level=1)
    dates = result.index[result]
    assert list(dates) == [pd.Timestamp("2020-03-02", tz="UTC")]


@pytest.mark.parametrize("start", [22, 31, 32, 33, 34, 40], ids=lambda bar: f"bar_{bar}")
def test_window_of_a_halted_asset_counts_its_own_bars(dirty_engine, dirty_panel, start):
    """The halt is on bars 20 to 31, so on a start next to it, the window goes back over it."""
    closes = dirty_panel.xs("HALTED", level="asset").close
    dirty_engine.add(factors.MA(5), "f")
    result = dirty_engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f.xs("HALTED", level=1)
    expected = closes.rolling(5).mean().shift(1).loc[result.index]
    assert_almost_equal(result.values, expected.values)


def test_shift_of_a_halted_asset_counts_its_own_bars(dirty_engine, dirty_panel):
    opens = dirty_panel.xs("HALTED", level="asset").open
    dirty_engine.add(factors.OHLCV.open.shift(3), "f")
    result = dirty_engine.run(DIRTY_DATES[30], DIRTY_DATES[-1]).f.xs("HALTED", level=1)
    assert_almost_equal(result.values, opens.shift(3).loc[result.index].values)


def test_history_warning_counts_the_bars_of_the_assets(dirty_engine):
    dirty_engine.set_filter(factors.StaticAssets({"HALTED"}))
    dirty_engine.add(factors.MA(10), "f")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dirty_engine.run(DIRTY_DATES[35], DIRTY_DATES[-1])


def test_cached_run_keeps_the_bars_of_a_halted_asset(dirty_engine, dirty_panel):
    closes = dirty_panel.xs("HALTED", level="asset").close
    start, end = DIRTY_DATES[35], DIRTY_DATES[-1]
    dirty_engine.add(factors.MA(10), "f")
    dirty_engine.run(start, end)
    loaded = dirty_engine.cache_hash

    # needs less history, so it reads what is loaded
    dirty_engine.add(factors.MA(5), "f", replace=True)
    result = dirty_engine.run(start, end).f.xs("HALTED", level=1)
    assert dirty_engine.cache_hash == loaded
    expected = closes.rolling(5).mean().shift(1).loc[result.index]
    assert_almost_equal(result.values, expected.values)


def test_align_by_time_loads_no_more_dates_for_a_halt(dirty_engine):
    """Every asset has a row on every date then, so the older bars of an asset are of no use."""
    start, window = 40, 15
    dirty_engine.align_by_time = True
    dirty_engine.add(factors.MA(window), "f")
    dirty_engine.run(DIRTY_DATES[start], DIRTY_DATES[-1])
    index = dirty_engine.dataframe_.index
    # one more for the delay
    first_bar = start - window
    assert index.get_level_values(0).min() == DIRTY_DATES[first_bar]
    assert len(index) == (len(DIRTY_DATES) - first_bar) * len(index.levels[1])


def test_align_by_time_does_not_depend_on_start(dirty_engine):
    dirty_engine.align_by_time = True
    dirty_engine.add(factors.MA(5), "f")
    result = dirty_engine.run(DIRTY_DATES[40], DIRTY_DATES[-1]).f
    from_the_beginning = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1]).f
    assert_same_rows(result, from_the_beginning)


def test_filter_delays_a_run_on_its_own(dirty_engine, dirty_panel):
    """Open needs no delay, but the filter does, and that delays the run."""
    start = DIRTY_DATES[40]
    full = dirty_panel.xs("FULL", level="asset")
    level = full.close.rolling(10).mean().iloc[40] - 1
    dirty_engine.set_filter(factors.MA(10) > level)
    dirty_engine.add(factors.OHLCV.open, "open")
    result = dirty_engine.run(start, DIRTY_DATES[-1]).open
    # the rule of a delayed run: its first bar is dropped
    assert result.index.get_level_values(0).min() == DIRTY_DATES[41]

    kept = (full.close.rolling(10).mean() > level).shift(1, fill_value=False)
    expected = full.open[kept].loc[DIRTY_DATES[41] :]
    assert_almost_equal(result.xs("FULL", level=1).values, expected.values)


# ------------------------------------ complete dates ------------------------------------
# An asset with gaps gets older bars of its own, so only that asset has rows on those dates. What
# reads across a date counts them as missing, or it would see a market of one asset.

COMPLETE_STARTS = [22, 25, 28, 31, 33, 36, 40, 45]


class DateSliceLoader(spectre.data.MemoryLoader):
    """Overrides `load` like a live loader can, so an asset with gaps gets no older bars."""

    def load(self, start=None, end=None, backwards=0):
        df = self._load()
        index = df.index.levels[0]
        start_loc = index.get_indexer([start], method="bfill")[0]
        return df.loc[index[max(start_loc - backwards, 0)] : end]


def plain_panel(panel):
    """No NaN bars and no splits, so pandas needs no rules for them."""
    return panel[~panel.index.get_level_values("asset").isin(["NANBAR", "NANSPLIT", "SPLIT"])]


def plain_engine(panel, device, loader_type=spectre.data.MemoryLoader):
    engine = factors.FactorEngine(loader_type(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS))
    if device == "cuda":
        engine.to_cuda()
    return engine


def from_date(values, first_complete):
    """`values` of (date, asset), the older dates count as missing."""
    return values.where(values.index.get_level_values("date") >= DIRTY_DATES[first_complete])


def mean_of_bars(values, win):
    """Mean of the last `win` bars of each asset, of the ones with a value."""
    means = values.groupby(level="asset", group_keys=False).apply(
        lambda bars: bars.rolling(win, min_periods=1).mean()
    )
    return means.loc[values.index]


def delayed(values):
    return values.groupby(level="asset").shift(1)


def market_returns(panel):
    returns = panel.close.groupby(level="asset").pct_change(fill_method=None)
    return returns.groupby(level="date").transform("mean")


def assert_rows(result, expected):
    assert result.notna().any()
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_window_over_a_column_keeps_its_full_window(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[close]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(panel.close, 10)))


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_window_over_a_cross_section_reads_complete_dates(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    ranks = panel.close.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    # 9 bars before the window, and the delay
    assert_rows(result, delayed(mean_of_bars(from_date(ranks, start - 10), 10)))


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_window_over_market_return(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[factors.MarketReturn()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    # 11 bars are loaded. The assets without gaps have no return on the first of them, so it isn't
    # a market yet, even with all its rows.
    expected = delayed(mean_of_bars(from_date(market_returns(panel), start - 10), 10))
    assert_rows(result, expected)


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_window_over_a_cross_section_of_windows(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    ranks = mean_of_bars(panel.close, 3).groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(factors.MA(5, inputs=[factors.MA(3).rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    # the 2 bars of MA(3), the 4 bars of MA(5), and the delay
    assert_rows(result, delayed(mean_of_bars(from_date(ranks, start - 5), 5)))


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_window_over_a_masked_cross_section(dirty_panel, device, start):
    """The look-back of the mask counts too, it decides who is in the cross-section."""
    panel = plain_panel(dirty_panel)
    above = mean_of_bars(panel.close, 3) > 100
    ranks = panel.close.where(above).groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(factors.MA(5, inputs=[close.rank(mask=factors.MA(3) > 100)]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(from_date(ranks, start - 5), 5)))


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_filter_input_of_a_cross_section_is_false_on_older_dates(dirty_panel, device, start):
    class Selected(factors.CrossSectionFactor):
        def compute(self, selected):
            assert selected.dtype == torch.bool
            return selected.float()

    panel = plain_panel(dirty_panel)
    selected = panel.close > 100
    older = panel.index.get_level_values("date") < DIRTY_DATES[start - 10]
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[Selected(inputs=[close > 100])]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(selected.mask(older, False).astype(float), 10)))


@pytest.mark.parametrize(
    "make_factor",
    [
        lambda: close.rank(),
        lambda: close.zscore(),
        lambda: close.demean(),
        lambda: close.mean(),
        lambda: close.median(),
        lambda: close.std(),
        lambda: close.max(),
        lambda: close.quantile(2, mask=close > 0),
        lambda: close.to_weight(),
        lambda: close.winsorizing(0.2),
        lambda: close.mad_clamp(2),
        lambda: close.top(2).float(),
        lambda: factors.MA(10, inputs=[close.rank()]),
        lambda: factors.MA(10, inputs=[close.zscore()]),
        lambda: factors.MA(5, inputs=[factors.OHLCV.open]).set_mask(close.top(2)),
        lambda: factors.InformationCoefficient(factors.OHLCV.open, close).to_ir(3),
    ],
    ids=[
        "rank",
        "zscore",
        "demean",
        "mean",
        "median",
        "std",
        "max",
        "quantile",
        "to_weight",
        "winsorizing",
        "mad_clamp",
        "top",
        "window_of_rank",
        "window_of_zscore",
        "masked_window",
        "to_ir",
    ],
)
@pytest.mark.parametrize("start", [25, 28, 33])
def test_older_bars_of_an_asset_with_gaps_change_no_other_asset(
    dirty_panel, device, make_factor, start
):
    def run(loader_type):
        engine = plain_engine(dirty_panel, device, loader_type)
        engine.add(make_factor(), "f")
        result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
        return result.drop("HALTED", level=1)

    pd.testing.assert_series_equal(run(spectre.data.MemoryLoader), run(DateSliceLoader))


def test_a_loader_without_older_bars_has_nothing_to_do(dirty_panel, device):
    engine = plain_engine(dirty_panel, device, DateSliceLoader)
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    engine.run(DIRTY_DATES[28], DIRTY_DATES[-1])
    assert engine.incomplete_rows_("date", close) is None


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_cross_section_in_the_range_does_not_change(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    ranks = panel.open.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(factors.OHLCV.open.rank(), "f")
    assert_rows(engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f, ranks)


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_delayed_cross_section_on_the_first_row_after_a_halt(dirty_panel, device, start):
    """The delay reads the bar before it, which can be older than the complete dates."""
    panel = plain_panel(dirty_panel)
    ranks = panel.close.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(close.rank(), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(from_date(ranks, start - 1)))


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_window_over_a_date_and_sector_group(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    dates = panel.index.get_level_values("date")
    means = panel.close.groupby([dates, panel.sector]).transform("mean")
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[close.mean(groupby="date,sector")]), "means")
    engine.add(factors.MA(10, inputs=[close.demean(groupby="sector")]), "demeaned")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1])
    assert_rows(result.means, delayed(mean_of_bars(from_date(means, start - 10), 10)))
    demeaned = delayed(mean_of_bars(from_date(panel.close - means, start - 10), 10))
    assert_rows(result.demeaned, demeaned)


@pytest.mark.parametrize("groupby", ["asset", "sector"])
def test_groups_without_a_date_read_all_their_rows(dirty_panel, device, groupby):
    panel, start = plain_panel(dirty_panel), 25
    engine = plain_engine(panel, device)
    engine.add(factors.OHLCV.open.mean(groupby=groupby), "f")
    engine.add(factors.MA(10), "loads_history")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert engine.incomplete_rows_(groupby, close) is None
    # all the rows of the group, also the ones on older dates
    loaded = engine.dataframe_
    assert loaded.xs("HALTED", level=1).index[0] < DIRTY_DATES[start - 10]
    keys = loaded.index.get_level_values(1) if groupby == "asset" else loaded[groupby]
    expected = loaded.open.groupby(keys, observed=True).transform("mean")
    assert_rows(result, expected)


class TwoRanks(factors.CrossSectionFactor):
    inputs = [close]

    def compute(self, data):
        ranks = spectre.parallel.rankdata(data, 1)
        return torch.stack((ranks, -ranks), dim=2)


@pytest.mark.parametrize("start", COMPLETE_STARTS)
def test_multiple_outputs_of_a_cross_section(dirty_panel, device, start):
    panel = plain_panel(dirty_panel)
    ranks = panel.close.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[TwoRanks()[1]]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(from_date(-ranks, start - 10), 10)))


@pytest.mark.filterwarnings("ignore:engine.max_backwards")
def test_complete_dates_with_a_clamped_history(dirty_panel, device):
    panel, start, clamped = plain_panel(dirty_panel), 25, 5
    ranks = panel.close.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.max_backwards = clamped
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(from_date(ranks, start - clamped), 10)))


@pytest.mark.filterwarnings("ignore:engine.max_backwards")
@pytest.mark.parametrize("max_backwards", [np.inf, 4], ids=["all_the_history", "clamped"])
@pytest.mark.parametrize(
    "make_factor",
    [
        lambda: factors.MA(5, inputs=[factors.MA(3).rank()]),
        lambda: factors.MA(3, inputs=[factors.MA(5).demean()]),
        lambda: factors.MA(10, inputs=[factors.MarketReturn()]),
        lambda: factors.MA(5, inputs=[close.rank(mask=factors.MA(3) > 100)]),
        lambda: factors.MA(5).zscore(),
        lambda: factors.InformationCoefficient(factors.MA(3), close).to_ir(3),
    ],
    ids=["window_of_rank", "window_of_demean", "market", "mask", "zscore", "to_ir"],
)
def test_incomplete_rows_change_no_other_asset(
    dirty_panel, device, monkeypatch, make_factor, max_backwards
):
    """
    Those rows include the look-back of an input, and the assets without gaps read it all. If it's
    clamped they are on partial windows all the same, that's no reason to clear them.
    """

    def run():
        engine = plain_engine(dirty_panel, device)
        engine.max_backwards = max_backwards
        engine.add(make_factor(), "f")
        return engine.run(DIRTY_DATES[28], DIRTY_DATES[-1]).f.drop("HALTED", level=1)

    with monkeypatch.context() as patch:
        patch.setattr(factors.FactorEngine, "incomplete_rows_", lambda *_: None)
        without_clearing = run()
    pd.testing.assert_series_equal(run(), without_clearing)


@pytest.mark.parametrize("delay_factor", [True, False])
def test_complete_dates_of_a_run_without_delay(dirty_panel, device, delay_factor):
    panel, start = plain_panel(dirty_panel), 28
    opens = panel.open.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    # not delayed by itself, on a delay_factor='all' run it loads one bar more
    engine.add(factors.MA(10, inputs=[factors.OHLCV.open.rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1], "all" if delay_factor else True).f
    expected = mean_of_bars(from_date(opens, start - (10 if delay_factor else 9)), 10)
    assert_rows(result, delayed(expected) if delay_factor else expected)


@pytest.mark.parametrize(
    "assets, first_complete",
    [
        # a complete asset has a row on all the dates, so these are the same dates as without
        ({"FULL", "HALTED", "DELISTED"}, 18),
        # the older bars of the only asset are a complete cross-section of it
        ({"HALTED"}, None),
    ],
    ids=["with_a_full_asset", "alone"],
)
def test_complete_dates_of_static_assets(dirty_panel, device, assets, first_complete):
    panel, start = dirty_panel, 28
    selected = panel[panel.index.get_level_values("asset").isin(list(assets))]
    means = selected.close.groupby(level="date").transform("mean")
    engine = plain_engine(panel, device)
    engine.set_filter(factors.StaticAssets(assets))
    engine.add(factors.MA(10, inputs=[close.mean()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    known = means if first_complete is None else from_date(means, first_complete)
    assert_rows(result, delayed(mean_of_bars(known, 10)))


def test_complete_dates_of_static_assets_without_a_gone_asset(dirty_panel, device):
    """
    DELISTED ends before the dates of the history, so it isn't loaded, and on the older bars of
    HALTED nothing shows that it had rows. Only the dates of the history are known to be complete.
    """
    panel, start = dirty_panel, 28
    panel = panel.drop(index=[(date, "DELISTED") for date in DIRTY_DATES[16:41]])
    assets = {"HALTED", "DELISTED"}
    selected = panel[panel.index.get_level_values("asset").isin(list(assets))]
    means = selected.close.groupby(level="date").transform("mean")
    engine = plain_engine(panel, device)
    engine.set_filter(factors.StaticAssets(assets))
    engine.add(factors.MA(10, inputs=[close.mean()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(from_date(means, start - 10), 10)))


def test_complete_dates_of_static_assets_with_their_own_gaps(dirty_panel, device):
    """
    Each asset has 4 bars before bar 28: FULL on 7, 8, 9, 27 and HALTED on 16 to 19. Of all those
    dates the last 4 have the rows of both, HALTED has a row on 7 to 9, and it's not loaded.
    """
    panel = dirty_panel.drop(index=[(date, "FULL") for date in DIRTY_DATES[10:27]])
    start, assets = 28, {"FULL", "HALTED"}
    selected = panel[panel.index.get_level_values("asset").isin(list(assets))]
    ranks = selected.close.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.set_filter(factors.StaticAssets(assets))
    engine.add(factors.MA(4, inputs=[close.rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert_rows(result, delayed(mean_of_bars(from_date(ranks, 17), 4)))


def test_complete_dates_of_a_cached_run(dirty_panel, device):
    panel, start = plain_panel(dirty_panel), 28
    ranks = panel.close.groupby(level="date").rank()
    engine = plain_engine(panel, device)
    engine.add(factors.MA(20), "history")
    engine.run(DIRTY_DATES[start], DIRTY_DATES[-1])
    loaded = engine.cache_hash

    engine.remove_all_factors()
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    assert engine.cache_hash == loaded
    # the dates of the longer load are complete
    assert_rows(result, delayed(mean_of_bars(from_date(ranks, start - 20), 10)))


def test_complete_dates_in_another_timezone(dirty_panel, device):
    panel, start = plain_panel(dirty_panel), 28
    ranks = panel.close.groupby(level="date").rank()
    expected = delayed(mean_of_bars(from_date(ranks, start - 10), 10))
    engine = plain_engine(panel, device)
    engine.timezone = "America/New_York"
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    result = engine.run(DIRTY_DATES[start], DIRTY_DATES[-1]).f
    in_utc = result.tz_convert("UTC", level=0)
    assert_rows(in_utc, expected)


def test_align_by_time_has_no_incomplete_dates(dirty_panel, device):
    engine = plain_engine(dirty_panel, device)
    engine.align_by_time = True
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    engine.run(DIRTY_DATES[28], DIRTY_DATES[-1])
    assert engine.incomplete_rows_("date", close) is None


def test_incomplete_rows(dirty_panel, device):
    panel, start = plain_panel(dirty_panel), 28
    engine = plain_engine(panel, device)
    engine.add(factors.MA(10, inputs=[close.rank()]), "f")
    engine.run(DIRTY_DATES[start], DIRTY_DATES[-1], delay_factor=False)

    # 9 bars of history, the rows before them are older bars of HALTED
    loaded_dates = engine.dataframe_.index.levels[0]
    first_complete = list(loaded_dates).index(DIRTY_DATES[start - 9])
    rows = engine.incomplete_rows_("date", close)
    assert rows.shape == engine.get_group_padding_mask("date").shape
    assert rows.any(dim=1).tolist() == [i < first_complete for i in range(len(loaded_dates))]
    # an input with a look-back has a complete cross-section later
    rows = engine.incomplete_rows_("date", factors.MA(3))
    assert rows.any(dim=1).tolist() == [i < first_complete + 2 for i in range(len(loaded_dates))]
    # the groups without a date and the group of a tensor have nothing to do
    assert engine.incomplete_rows_("asset", close) is None
    assert engine.incomplete_rows_(torch.zeros(1), close) is None


# ------------------------------------ mask='universe' ------------------------------------


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
UNIVERSE_DATES = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
UNIVERSE_BAR = 20
SPIKE_START = 26
open_price = factors.OHLCV.open


def universe_panel():
    """
    The closes of A > B > C > D, so `close.top(3)` is {A, B, C}. D has the highest open.
    Opens rise a little on every bar, so a window can tell which of its bars are masked.
    """
    rise = 0.01 * np.arange(len(UNIVERSE_DATES))
    frames = {
        asset: pd.DataFrame({"open": o + rise, "close": c}, index=UNIVERSE_DATES)
        for asset, o, c in [("A", 3.0, 40.0), ("B", 2.0, 30.0), ("C", 1.0, 20.0), ("D", 7.0, 10.0)]
    }
    df = pd.concat(frames, names=["asset", "date"]).swaplevel().sort_index()
    df["high"], df["low"], df["volume"] = df.close, df.close, 1.0
    return df[list(OHLCV_COLUMNS)]


def universe_engine(panel, device, universe):
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, ohlcv=OHLCV_COLUMNS))
    if device == "cuda":
        engine.to_cuda()
    engine.set_filter(universe)
    return engine


def changed_panel():
    """C falls out of the universe on `UNIVERSE_BAR`, by its close of that bar and nothing else."""
    panel = universe_panel()
    panel.loc[(UNIVERSE_DATES[UNIVERSE_BAR], "C"), "close"] = 1.0
    return panel


MASKED_FORMS = {
    "rank": lambda mask: open_price.rank(mask=mask),
    "zscore": lambda mask: open_price.zscore(mask=mask),
    "demean": lambda mask: open_price.demean(mask=mask),
    "mean": lambda mask: open_price.mean(mask=mask),
    "median": lambda mask: open_price.median(mask=mask),
    "std": lambda mask: open_price.std(mask=mask),
    "quantile": lambda mask: open_price.quantile(2, mask=mask),
    "to_weight": lambda mask: open_price.to_weight(mask=mask),
    "filter": lambda mask: open_price.filter(mask),
    "weekday": lambda mask: factors.WEEKDAY.rank(mask=mask),
    "window": lambda mask: factors.MA(3, inputs=[open_price]).set_mask(mask),
    "nested": lambda mask: open_price.rank(mask=mask) + 1,
}


@pytest.mark.parametrize("make_factor", MASKED_FORMS.values(), ids=MASKED_FORMS.keys())
def test_universe_mask_delays_like_the_filter_object(device, make_factor):
    bar = UNIVERSE_DATES[UNIVERSE_BAR]
    start, end = UNIVERSE_DATES[5], UNIVERSE_DATES[-1]

    def run(panel, mask=None):
        universe = factors.OHLCV.close.top(3)
        engine = universe_engine(panel, device, universe)
        engine.add(make_factor(mask or universe), "f")
        return engine.run(start, end).f

    result = run(changed_panel(), "universe")
    pd.testing.assert_series_equal(result, run(changed_panel()))
    # the close of `bar` is known after the close, so nothing on `bar` can depend on it
    pd.testing.assert_series_equal(result.loc[:bar], run(universe_panel(), "universe").loc[:bar])


def test_universe_mask_rank_by_hand(device):
    bar = UNIVERSE_DATES[UNIVERSE_BAR]
    universe = factors.OHLCV.close.top(3)
    engine = universe_engine(changed_panel(), device, universe)
    engine.add(open_price.rank(mask="universe"), "f")
    result = engine.run(UNIVERSE_DATES[5], UNIVERSE_DATES[-1]).f
    # bar is the last row with the old universe, opens of A, B, C are 3, 2, 1
    assert result.loc[bar].to_dict() == {"A": 3, "B": 2, "C": 1}
    # the next row knows the close of bar: opens of A, B, D are 3, 2, 7
    next_bar = UNIVERSE_DATES[UNIVERSE_BAR + 1]
    assert result.loc[next_bar].to_dict() == {"A": 2, "B": 1, "D": 3}


def test_universe_mask_of_a_filter_on_open_is_not_delayed(device):
    engine = universe_engine(universe_panel(), device, open_price.top(3))
    engine.add(open_price.rank(mask="universe"), "f")
    start = UNIVERSE_DATES[5]
    result = engine.run(start, UNIVERSE_DATES[-1]).f
    assert result.index.get_level_values(0).min() == start


def test_universe_mask_warns_when_delay_is_off(device):
    engine = universe_engine(universe_panel(), device, factors.OHLCV.close.top(3))
    engine.add(open_price.rank(mask="universe"), "f")
    with pytest.warns(RuntimeWarning, match="delay_factor is set to False"):
        engine.run(UNIVERSE_DATES[5], UNIVERSE_DATES[-1], delay_factor=False)


def test_universe_mask_follows_the_filter_of_each_run(device):
    factor = open_price.rank(mask="universe") + 1
    start, end = UNIVERSE_DATES[5], UNIVERSE_DATES[-1]
    with_filter = universe_engine(universe_panel(), device, factors.OHLCV.close.top(3))
    with_filter.add(factor, "f")
    no_filter = universe_engine(universe_panel(), device, None)
    no_filter.add(factor, "f")

    assert with_filter.run(start, end).index.get_level_values(0).min() == UNIVERSE_DATES[6]
    assert no_filter.run(start, end).index.get_level_values(0).min() == start
    # nothing is left behind for the next run, or outside of a run
    assert with_filter.run(start, end).index.get_level_values(0).min() == UNIVERSE_DATES[6]
    assert not factor.should_delay()


def spike_panel():
    """
    Prices of `SPIKE_START - 13` to `- 11` are out of reach of a look-back that is too short, and
    they decide MA(10) > 150 and MA(20) > 250 on the first bars of the last window of `MA(5)`.
    The last two bars keep the filters True on the rows of the results.
    """
    start = SPIKE_START
    prices = np.full(len(UNIVERSE_DATES), 100.0)
    prices[start - 13 : start - 10] = 1000
    prices[start - 1 : start + 1] = 1000
    # every bar has a different open, so MA(5) tells which ones are masked
    df = pd.DataFrame(
        {"close": prices, "open": prices + np.arange(len(prices))}, index=UNIVERSE_DATES
    )
    df["high"], df["low"], df["volume"] = prices, prices, 1.0
    panel = pd.concat({"A": df, "B": df}, names=["asset", "date"]).swaplevel().sort_index()
    return df, panel[list(OHLCV_COLUMNS)]


def masked_ma5(df, win, level):
    """MA(5) of open masked by MA(win) > level, on the rows of the delayed run."""
    universe = df.close.rolling(win).mean() > level
    ma5 = df.open.where(universe).rolling(5, min_periods=1).mean()
    return ma5.shift(1).loc[UNIVERSE_DATES[SPIKE_START + 1] :]


def test_mask_under_a_window_matches_pandas(device):
    """Every bar of the window is masked, so the first one needs the whole look-back of the mask."""
    df, panel = spike_panel()
    engine = universe_engine(panel, device, None)
    factor = factors.MA(5, inputs=[open_price]).set_mask(factors.MA(10) > 150)
    engine.add(factor, "f")
    result = engine.run(UNIVERSE_DATES[SPIKE_START], UNIVERSE_DATES[-1]).f.xs("A", level=1)
    assert_almost_equal(result.values, masked_ma5(df, 10, 150).values)


def test_universe_mask_loads_the_history_of_the_filter_under_a_window(device):
    df, panel = spike_panel()
    engine = universe_engine(panel, device, factors.MA(10) > 150)
    factor = factors.MA(5, inputs=[open_price]).set_mask("universe") + 0
    engine.add(factor, "f")
    result = engine.run(UNIVERSE_DATES[SPIKE_START], UNIVERSE_DATES[-1]).f.xs("A", level=1)
    assert_almost_equal(result.values, masked_ma5(df, 10, 150).values)

    # the history of a factor with this mask changes with the filter
    engine.set_filter(factors.MA(20) > 250)
    result = engine.run(UNIVERSE_DATES[SPIKE_START], UNIVERSE_DATES[-1]).f.xs("A", level=1)
    assert_almost_equal(result.values, masked_ma5(df, 20, 250).values)


def test_universe_mask_of_the_filter_itself_is_refused(device):
    universe = factors.OHLCV.close.top(3, mask="universe")
    engine = universe_engine(universe_panel(), device, universe)
    engine.add(open_price, "f")
    with pytest.raises(ValueError, match="possible circular reference"):
        engine.run(UNIVERSE_DATES[5], UNIVERSE_DATES[-1])


def test_lookahead_bias(engine):
    engine.add(factors.SMA(5), "test")
    assert engine.test_lookahead_bias(ref.START, ref.END) == "No assertion raised."
    # shift(-1) is compensated by the delay, -2 uses tomorrow's data
    engine.add(close.shift(-2), "biased")
    with pytest.raises(RuntimeError, match="look-ahead bias"):
        engine.test_lookahead_bias(ref.START, ref.END)


def typed_engine(panel, device, dtypes):
    """An engine on `panel` with these column types, csv loaders recommend float32."""
    ohlcv = ("open", "high", "low", "close", "volume")
    loader = spectre.data.MemoryLoader(
        panel.astype(dtypes), ohlcv=ohlcv, adjustments=("ex-dividend", "split_ratio")
    )
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    return engine


PRICES_FLOAT32 = {col: np.float32 for col in ("open", "high", "low", "close", "volume")}


@pytest.mark.parametrize(
    "dtypes", [PRICES_FLOAT32, {"volume": np.int64}], ids=["float32", "int64_volume"]
)
def test_lookahead_bias_of_any_column_type(dirty_panel, device, dtypes):
    start, end = DIRTY_DATES[5], DIRTY_DATES[-1]
    engine = typed_engine(dirty_panel, device, dtypes)
    engine.add(factors.SMA(5), "test")
    expected = engine.run(start, end)

    assert engine.test_lookahead_bias(start, end) == "No assertion raised."
    engine.add(close.shift(-2), "biased")
    with pytest.raises(RuntimeError, match="look-ahead bias"):
        engine.test_lookahead_bias(start, end)

    # the data of the test is gone
    engine.remove_all_factors()
    engine.add(factors.SMA(5), "test")
    pd.testing.assert_frame_equal(engine.run(start, end), expected)


def test_lookahead_bias_of_a_loader_without_ohlcv(dirty_panel):
    loader = spectre.data.MemoryLoader(dirty_panel, None, None)
    engine = factors.FactorEngine(loader)
    engine.add(factors.ColumnDataFactor(inputs=["close"]), "test")
    with pytest.raises(ValueError, match="loader has no ohlcv"):
        engine.test_lookahead_bias(DIRTY_DATES[5], DIRTY_DATES[-1])


def test_lookahead_bias_restores_the_data_after_an_error(dirty_engine):
    start, end = DIRTY_DATES[5], DIRTY_DATES[-1]

    class FailsOnRandomData(factors.CustomFactor):
        inputs = [close]

        def compute(self, data):
            if data[~data.isnan()].min() < 0:
                raise ZeroDivisionError()
            return data

    dirty_engine.add(FailsOnRandomData(), "test")
    expected = dirty_engine.run(start, end)
    with pytest.raises(ZeroDivisionError):
        dirty_engine.test_lookahead_bias(start, end)
    pd.testing.assert_frame_equal(dirty_engine.run(start, end), expected)


@pytest.mark.parametrize("loader_aligned", [False, True])
def test_align_by_time(loader_aligned):
    if loader_aligned:
        loader = spectre.data.CsvDirLoader(
            str(ref.DATA_DIR / "daily"),
            calender_asset="AAPL",
            ohlcv=ref.DAILY_OHLCV,
            prices_index="date",
            parse_dates=True,
            align_by_time=True,
        )
    else:
        loader = spectre.data.CsvDirLoader(
            str(ref.DATA_DIR / "daily"),
            calender_asset="AAPL",
            ohlcv=ref.DAILY_OHLCV,
            prices_index="date",
            parse_dates=True,
        )
    engine = spectre.factors.FactorEngine(loader)
    engine.align_by_time = not loader_aligned
    engine.add(close, "close")
    engine.add(factors.SMA(2), "ma")
    df = engine.run(ref.START, ref.END)

    # MSFT has a nan row on 2019-01-10, so all assets have the same number of rows
    assert len(df.xs("MSFT", level=1)) == len(df.xs("AAPL", level=1)) == 9
    assert np.isnan(df.loc[("2019-01-11", "MSFT"), "close"])
    # mean of 2019-01-09 and nan
    assert df.loc[("2019-01-11", "MSFT"), "ma"] == df.loc[("2019-01-10", "MSFT"), "close"]


def test_same_factor_in_two_engines(daily_loader):
    engine = factors.FactorEngine(daily_loader)
    engine2 = factors.FactorEngine(daily_loader)
    f = factors.MA(5)
    universe = factors.OHLCV.volume.top(1)
    engine.add(f, "f")
    engine.add(universe, "mask")
    engine2.add(f, "f")
    engine2.set_filter(universe)

    result = engine.run(ref.START, ref.END)
    result2 = engine2.run(ref.START, ref.END)
    assert_array_equal(result.f[result["mask"]], result2.f)


def check_ref_count_cleared(factor):
    if isinstance(factor, factors.CustomFactor):
        assert factor._ref_count == 0, factor
        if factor._mask is not None:
            check_ref_count_cleared(factor._mask)
    if factor.inputs:
        for upstream in factor.inputs:
            if isinstance(upstream, factors.BaseFactor):
                check_ref_count_cleared(upstream)


def test_ref_count_after_run(engine_no_calendar):
    t = factors.OHLCV.volume.top(1)
    b = factors.OHLCV.volume.bottom(1)
    engine_no_calendar.add([t, t & b, t | b, ~t], ["t", "t&b", "t|b", "~t"])
    engine_no_calendar.run("2019-01-01", "2019-01-05")
    for f in engine_no_calendar.factors.values():
        check_ref_count_cleared(f)


def test_reused_factor_computed_once(engine):
    f1 = factors.BBANDS(win=20, inputs=[close, 2]).normalized()
    f2 = factors.EMA(span=10, inputs=[f1])
    fa = factors.STDDEV(win=15, inputs=[f2])
    fb = factors.MACD(12, 26, 9, inputs=[f2])
    engine.add([f2, fa, fb], ["f2", "fa", "fb"])

    engine.run("2019-01-11", ref.END)  # prepare data, so pre_compute_ can be called
    for f in engine.factors.values():
        f.pre_compute_(engine, "2019-01-11", ref.END)
    # referenced by engine, fa, and 2 EMAs of fb
    assert f2._ref_count == 4
    for f in engine.factors.values():
        f.clean_up_(force=True)

    result = engine.run(ref.START, ref.END)
    assert f2._ref_count == 0

    # nested windows should load enough data
    def expected_f2(df):
        upper, middle, _ = ref.bbands(df.close, 20, 2)
        return ((df.close - middle) / (upper - middle)).ewm(span=10, adjust=False).mean()

    aapl, msft = ref.expected(expected_f2)
    assert_almost_equal(result.xs("AAPL", level=1).f2.values, aapl, decimal=3)
    assert_almost_equal(result.xs("MSFT", level=1).f2.values, msft, decimal=3)
    aapl, msft = ref.expected(lambda df: expected_f2(df).rolling(15).std(ddof=0))
    assert_almost_equal(result.xs("AAPL", level=1).fa.values, aapl, decimal=3)


def test_keep_cache(engine_no_calendar):
    engine = engine_no_calendar
    ma5 = factors.MA(5)
    ma5._keep_cache = True

    engine.add(ma5, "a")
    engine.run("2019-01-01", "2019-01-05")
    shape1 = ma5._cache.shape

    # cache is kept when ma5 is not used
    engine.remove_all_factors()
    engine.add(factors.MA(10), "a")
    engine.run("2019-01-01", "2019-01-05")
    assert ma5._cache.shape == shape1

    # longer history loaded by MA(10), so cache invalidated and recalculated
    engine.remove_all_factors()
    engine.add(ma5, "a")
    engine.run("2019-01-01", "2019-01-05")
    assert ma5._cache.shape[0] == shape1[0]
    assert ma5._cache.shape[1] > shape1[1]
    assert tuple(ma5._cache.shape) == (2, 14)


def test_device(engine_no_calendar, device):
    engine = engine_no_calendar
    if device == "cuda":
        engine.to_cuda()
    assert engine.device.type == device
    f1 = factors.OHLCV.volume + 1
    f2 = f1 + 1
    engine.add(f2, "f2")
    engine.add(factors.OHLCV.volume + 2, "fv")
    engine.set_filter(factors.OHLCV.volume > 1)
    result = engine.run(ref.START, ref.END)
    assert_array_equal(result.f2, result.fv)
    engine.to_cpu()
    assert engine.device.type == "cpu"


def test_get_price_matrix(engine):
    engine.set_filter(factors.StaticAssets({"MSFT"}))
    prices = engine.get_price_matrix(ref.START, ref.END)
    assert list(prices.columns) == ["MSFT"]
    msft = ref.daily_prices("MSFT").close.loc["2019-01-02" : ref.END]
    assert_array_equal(prices.MSFT.dropna().values, msft.values)


def test_get_price_matrix_without_a_filter(engine):
    engine.add(factors.SMA(5), "ma5")
    expected = engine.run(ref.START, ref.END)

    prices = engine.get_price_matrix(ref.START, ref.END)
    assert list(prices.columns) == ["AAPL", "MSFT"]
    aapl = ref.daily_prices("AAPL").close.loc["2019-01-02" : ref.END]
    msft = ref.daily_prices("MSFT").close.loc["2019-01-02" : ref.END]
    assert_array_equal(prices.AAPL.values, aapl.values)
    assert_array_equal(prices.MSFT.dropna().values, msft.values)

    # the engine goes on as before
    assert list(engine.factors) == ["ma5"]
    assert engine.get_filter() is None
    pd.testing.assert_frame_equal(engine.run(ref.START, ref.END), expected)


def test_get_price_matrix_keeps_the_factors_and_the_filter(engine):
    universe = factors.OHLCV.volume.top(1)
    engine.add(factors.SMA(5), "ma5")
    engine.set_filter(universe)
    expected = engine.run(ref.START, ref.END)

    prices = engine.get_price_matrix(ref.START, ref.END)
    assert list(prices.columns) == ["AAPL", "MSFT"]

    assert list(engine.factors) == ["ma5"]
    assert engine.get_filter() is universe
    pd.testing.assert_frame_equal(engine.run(ref.START, ref.END), expected)


def test_get_price_matrix_restores_the_engine_after_an_error(engine):
    universe = factors.StaticAssets({"no such asset"})
    engine.add(factors.SMA(5), "ma5")
    engine.set_filter(universe)
    with pytest.raises(ValueError, match="was not found in DataLoader"):
        engine.get_price_matrix(ref.START, ref.END)
    assert list(engine.factors) == ["ma5"]
    assert engine.get_filter() is universe


def test_full_run(engine):
    engine.add(factors.MA(5) - factors.MA(10), "ma_cross")
    with pytest.warns(RuntimeWarning, match="mask parameter not set"):
        factor_data, mean_returns = engine.full_run(
            "2018-06-01", ref.END, periods=(1, 5), quantiles=2, filter_zscore=None, preview=False
        )

    assert set(factor_data.columns) == {
        ("Demeaned", "1D"),
        ("Demeaned", "5D"),
        ("Returns", "1D"),
        ("Returns", "5D"),
        ("ma_cross", "factor"),
        ("ma_cross", "factor_quantile"),
        ("ma_cross", "factor_weight"),
    }
    # forward returns at the close of the day
    aapl = ref.daily_prices("AAPL").close
    returns = factor_data.xs("AAPL", level=1)[("Returns", "5D")]
    expected = (aapl.shift(-5) / aapl - 1).loc[returns.index.tz_localize(None)]
    assert_almost_equal(returns.values, expected.values)
    # last 5 days have no forward returns, dropped
    assert returns.index[-1] == pd.Timestamp("2019-01-08", tz="UTC")

    assert list(mean_returns.index) == [0, 1]
    assert ("ma_cross", "1D", "mean") in mean_returns.columns


FORWARD_DATES = pd.date_range("2020-01-01", periods=40, freq="B", tz="UTC")
FORWARD_START, FORWARD_END = FORWARD_DATES[5], FORWARD_DATES[-1]


def forward_data():
    """
    Random walks, returns the panel and its wide opens and closes. Opens and closes are unrelated,
    so a return can't be mistaken for another. Two columns have a signal planted in them:

    * `after_close` is known after the close of a bar, and tells the return of holding from the
      open of the next bar. It is delayed, so on row t it is open[t + 1] / open[t] - 1.
    * `at_open` is known at the open of a bar, and tells the return of holding from the open of the
      next bar. It's not delayed, so on row t it is open[t + 2] / open[t + 1] - 1.
    """
    rng = np.random.default_rng(11)
    assets = list("ABCDEF")
    shape = len(FORWARD_DATES), len(assets)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0))
    opens = pd.DataFrame(walk, index=FORWARD_DATES, columns=assets)
    closes = opens * np.exp(rng.normal(0, 0.01, shape))
    columns = {
        "open": opens,
        "close": closes,
        "high": np.maximum(opens, closes),
        "low": np.minimum(opens, closes),
        "volume": pd.DataFrame(1000.0, index=FORWARD_DATES, columns=assets),
        "after_close": (opens.shift(-1) / opens - 1).shift(-1),
        "at_open": opens.shift(-2) / opens.shift(-1) - 1,
    }
    panel = pd.concat({col: df.stack() for col, df in columns.items()}, axis=1)
    return panel.rename_axis(["date", "asset"]), opens, closes


def forward_engine(panel, device, universe=None):
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, ohlcv=OHLCV_COLUMNS))
    if device == "cuda":
        engine.to_cuda()
    engine.set_filter(universe)
    return engine


def after_close():
    return factors.ColumnDataFactor(inputs=("after_close",))


def at_open():
    return factors.ColumnDataFactor(inputs=("at_open",), should_delay=False)


def forward_return(prices, n, entry):
    """The return of buying at `prices[t + entry]` and selling `n` bars later, by hand."""
    return prices.shift(-entry - n) / prices.shift(-entry) - 1


def assert_returns(factor_data, column, prices, entry, n=3):
    returns = factor_data[column][f"{n}D"].unstack()
    expected = forward_return(prices, n, entry).reindex_like(returns).where(returns.notna())
    assert returns.notna().to_numpy().sum() > 0
    pd.testing.assert_frame_equal(returns, expected, check_names=False, check_freq=False)


# on row t a delayed factor can buy at open[t], and one that isn't delayed at open[t + 1]
TRADE_AT_OPEN_CASES = {
    "delayed": (after_close, None, None, 0),
    "delayed_universe_of_all": (after_close, None, lambda: close > 0, 0),
    "delayed_universe_on_open": (after_close, None, lambda: open_price > 0, 0),
    "delayed_adv_universe": (after_close, None, lambda: factors.AverageDollarVolume(5).top(3), 0),
    "not_delayed": (at_open, None, None, 1),
    "not_delayed_universe_on_open": (at_open, None, lambda: open_price > 0, 1),
    "not_delayed_adv_on_open": (at_open, None, lambda: open_price.top(4), 1),
    "mixed_factors": (after_close, at_open, None, 0),
    "mixed_by_universe": (at_open, None, lambda: close > 0, 0),
    "mixed_by_adv_universe": (at_open, None, lambda: factors.AverageDollarVolume(5).top(3), 0),
}


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.parametrize("filter_zscore", [None, 20])
@pytest.mark.parametrize(
    "factor, other, universe, entry", TRADE_AT_OPEN_CASES.values(), ids=TRADE_AT_OPEN_CASES.keys()
)
def test_full_run_trade_at_open(device, factor, other, universe, entry, filter_zscore):
    panel, opens, _ = forward_data()
    engine = forward_engine(panel, device, universe and universe())
    engine.add(factor(), "f")
    if other:
        engine.add(other(), "other")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at="open",
        periods=(1, 3),
        quantiles=2,
        filter_zscore=filter_zscore,
        preview=False,
    )

    for n in (1, 3):
        assert_returns(factor_data, "Returns", opens, entry, n)
        # the same rows and the same bars as Returns
        returns = factor_data["Returns"][f"{n}D"]
        demeaned = returns - returns.groupby(level="date").transform("mean")
        pd.testing.assert_series_equal(
            factor_data["Demeaned"][f"{n}D"], demeaned, check_names=False
        )

    # the signal is planted on the bar of its entry
    correlation = factor_data["f"]["factor"].corr(factor_data["Returns"]["1D"])
    assert correlation > 0.999

    # a run that isn't delayed keeps its first bar
    dates = factor_data.index.get_level_values(0)
    assert dates.min() == FORWARD_DATES[6 if entry == 0 else 5]
    # the last bar with a return of the longest period, and no bar after it
    assert dates.max() == FORWARD_DATES[-1 - 3 - entry]
    assert factor_data["Returns"]["3D"].loc[dates.max()].notna().all()


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_full_run_without_demean_or_zscore_is_aligned_too(device):
    panel, opens, _ = forward_data()
    engine = forward_engine(panel, device, close > 0)
    engine.add(at_open(), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at="open",
        periods=(1, 3),
        quantiles=2,
        filter_zscore=None,
        demean=False,
        preview=False,
    )
    assert_returns(factor_data, "Returns", opens, 0)


def test_full_run_universe_mask_by_name(device):
    """The raw factor is delayed by its mask too, so this is not a mix."""
    panel, opens, _ = forward_data()
    engine = forward_engine(panel, device, close > 0)
    engine.add(open_price.rank(mask="universe"), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START, FORWARD_END, trade_at="open", periods=(1, 3), quantiles=2, preview=False
    )
    assert_returns(factor_data, "Returns", opens, 0)
    ranks = opens.rank(axis=1).shift(1)
    result = factor_data["f"]["factor"].unstack()
    pd.testing.assert_frame_equal(
        result, ranks.reindex_like(result), check_names=False, check_freq=False, check_dtype=False
    )


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.parametrize("trade_at", ["close", "open"])
def test_full_run_mixed_factors_are_all_delayed(device, trade_at):
    """Factors of one run share one return column, so the ones that don't need it are delayed."""
    panel, opens, _ = forward_data()
    engine = forward_engine(panel, device)
    engine.add(after_close(), "delayed")
    engine.add(open_price, "open")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at=trade_at,
        periods=(1, 3),
        quantiles=2,
        preview=False,
    )
    first_row = factor_data.xs("A", level=1).iloc[0]
    assert first_row[("open", "factor")] == pytest.approx(opens["A"].iloc[5])
    # no factors are left delayed
    assert list(engine.factors) == ["delayed", "open"]
    assert not open_price.should_delay()


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.parametrize("make_factor", [after_close, at_open], ids=["delayed", "not_delayed"])
def test_full_run_trade_at_a_factor(device, make_factor):
    panel, opens, closes = forward_data()
    engine = forward_engine(panel, device)
    engine.add(make_factor(), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at=(open_price + close) / 2,
        periods=(1, 3),
        quantiles=2,
        preview=False,
    )
    assert_returns(factor_data, "Returns", (opens + closes) / 2, 0)
    returns = factor_data["Returns"]["3D"]
    demeaned = returns - returns.groupby(level="date").transform("mean")
    pd.testing.assert_series_equal(factor_data["Demeaned"]["3D"], demeaned, check_names=False)
    # all factors are delayed
    dates = factor_data.index.get_level_values(0)
    assert dates.min() == FORWARD_DATES[6]
    assert dates.max() == FORWARD_DATES[-1 - 3]
    planted = panel[make_factor().inputs[0]].unstack().shift(1)
    result = factor_data["f"]["factor"].unstack()
    pd.testing.assert_frame_equal(
        result, planted.reindex_like(result), check_names=False, check_freq=False
    )


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.parametrize(
    "make_factor, first_bar", [(after_close, 6), (at_open, 5)], ids=["delayed", "not_delayed"]
)
def test_full_run_trade_at_close(device, make_factor, first_bar):
    panel, _, closes = forward_data()
    engine = forward_engine(panel, device)
    engine.add(make_factor(), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START, FORWARD_END, periods=(1, 3), quantiles=2, preview=False
    )
    assert_returns(factor_data, "Returns", closes, 0)
    # a run that isn't delayed keeps its first bar
    assert factor_data.index.get_level_values(0).min() == FORWARD_DATES[first_bar]
    assert factor_data.index.get_level_values(0).max() == FORWARD_DATES[-1 - 3]


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_full_run_trade_at_current_close(device):
    panel, _, closes = forward_data()
    engine = forward_engine(panel, device)
    engine.add(after_close(), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at="current_close",
        periods=(1, 3),
        quantiles=2,
        preview=False,
    )
    assert_returns(factor_data, "Returns", closes, 0)
    dates = factor_data.index.get_level_values(0)
    assert dates.min() == FORWARD_DATES[5]
    assert dates.max() == FORWARD_DATES[-1 - 3]
    # nothing is delayed
    planted = panel["after_close"].unstack()
    result = factor_data["f"]["factor"].unstack()
    pd.testing.assert_frame_equal(
        result, planted.reindex_like(result), check_names=False, check_freq=False
    )


def rows_by_hand(values, universe, prices, entry, delayed, n=3):
    """
    What the rows of full_run must hold, by hand, in wide frames with a column for each asset. A
    delayed row decides on the bar before it, with the universe of that bar. Every column of the
    row is about that decision, and the return is from `entry` bars after the row.
    """
    if delayed:
        values, universe = values.shift(1), universe.shift(1, fill_value=False)
    bars = FORWARD_DATES[6 if delayed else 5 : len(FORWARD_DATES) - n - entry]
    universe = universe.loc[bars]
    in_universe = values.loc[bars].where(universe)
    centered = in_universe.sub(in_universe.mean(axis=1), axis=0)
    returns = forward_return(prices, n, entry).loc[bars].where(universe)
    return {
        ("f", "factor"): in_universe,
        ("f", "factor_quantile"): in_universe.apply(
            lambda row: pd.qcut(row, 2, labels=False) if row.notna().any() else row, axis=1
        ),
        ("f", "factor_weight"): centered.div(centered.abs().sum(axis=1), axis=0),
        ("Returns", f"{n}D"): returns,
        ("Demeaned", f"{n}D"): returns.sub(returns.mean(axis=1), axis=0),
    }


def mondays():
    return factors.WEEKDAY == 0


def on_mondays(opens, closes):
    return opens.apply(lambda bars: bars.index.day_of_week == 0)


def everyone(opens, closes):
    return opens.notna()


def first_four(opens, closes):
    return opens.notna() & opens.columns.isin(list("ABCD"))


def opens_above(opens, closes):
    return opens > 100


def closes_above(opens, closes):
    return closes > 100


# the universe, the same by hand, the factor, and if the run is delayed
ALIGNED_CASES = {
    # mixed, the universe is delayed with the factor
    "mondays_and_a_delayed_factor": (mondays, on_mondays, after_close, True),
    "open_universe_and_a_delayed_factor": (
        lambda: open_price > 100,
        opens_above,
        after_close,
        True,
    ),
    "static_assets_and_a_delayed_factor": (
        lambda: factors.StaticAssets(set("ABCD")),
        first_four,
        after_close,
        True,
    ),
    # mixed, the factor is delayed with the universe, and with its quantile and weight columns
    "close_universe_and_a_factor_on_the_open": (lambda: close > 100, closes_above, at_open, True),
    # none is delayed
    "mondays_and_a_factor_on_the_open": (mondays, on_mondays, at_open, False),
    "open_universe_and_a_factor_on_the_open": (
        lambda: open_price > 100,
        opens_above,
        at_open,
        False,
    ),
    "no_universe_and_a_factor_on_the_open": (lambda: None, everyone, at_open, False),
    # all are delayed
    "close_universe_and_a_delayed_factor": (lambda: close > 100, closes_above, after_close, True),
    "no_universe_and_a_delayed_factor": (lambda: None, everyone, after_close, True),
}


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.parametrize("filter_zscore", [None, 20])
@pytest.mark.parametrize("trade_at", ["close", "open"])
@pytest.mark.parametrize(
    "universe, universe_by_hand, factor, delayed", ALIGNED_CASES.values(), ids=ALIGNED_CASES.keys()
)
def test_full_run_columns_of_a_row_are_about_one_decision(
    device, universe, universe_by_hand, factor, delayed, trade_at, filter_zscore
):
    panel, opens, closes = forward_data()
    engine = forward_engine(panel, device, universe())
    engine.add(factor(), "f")
    factor_data, mean_returns = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at=trade_at,
        periods=(3,),
        quantiles=2,
        filter_zscore=filter_zscore,
        preview=False,
    )

    prices = opens if trade_at == "open" else closes
    # a delayed row knows a close, so it can enter on its own bar. One that isn't knows the open
    # of its bar: the close of that bar is still to come, the open of that bar has gone.
    entry = 1 if trade_at == "open" and not delayed else 0
    values = panel["at_open" if factor is at_open else "after_close"].unstack()
    expected = rows_by_hand(values, universe_by_hand(opens, closes), prices, entry, delayed)

    rows = expected[("Returns", "3D")].stack().dropna().index
    assert len(rows) > 0
    assert sorted(factor_data.index) == sorted(rows)
    assert factor_data.notna().all(axis=None)
    for column, by_hand in expected.items():
        values = by_hand.stack().reindex(factor_data.index)
        assert_allclose(factor_data[column], values, rtol=1e-5, atol=1e-6, err_msg=str(column))
    assert mean_returns.notna().all(axis=None)


# trade_at, factor, prices, entry bar, first bar of the run, and if the filter is delayed in it
FULL_RETURN_CASES = {
    "close": ("close", at_open, "close", 0, 5, False),
    "close_of_a_delayed_factor": ("close", after_close, "close", 0, 6, True),
    "open_of_a_delayed_factor": ("open", after_close, "open", 0, 6, True),
    "open_not_delayed": ("open", at_open, "open", 1, 5, False),
    "current_close": ("current_close", after_close, "close", 0, 5, False),
    "a_factor": ("a_factor", at_open, "middle", 0, 6, True),
}


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("sparse_filter", [False, True], ids=["every_day", "mondays"])
@pytest.mark.parametrize(
    "trade_at, factor, prices, entry, first_bar, delayed_filter",
    FULL_RETURN_CASES.values(),
    ids=FULL_RETURN_CASES.keys(),
)
def test_full_run_keeps_the_rows_with_a_full_return(
    device, trade_at, factor, prices, entry, first_bar, delayed_filter, sparse_filter
):
    panel, opens, closes = forward_data()
    n = 3
    prices = {"open": opens, "close": closes, "middle": (opens + closes) / 2}[prices]
    universe = None
    if sparse_filter:
        universe = factors.WEEKDAY == 0
        universe.set_delay(delayed_filter)
    engine = forward_engine(panel, device, universe)
    engine.add(factor(), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at=(open_price + close) / 2 if trade_at == "a_factor" else trade_at,
        periods=(n,),
        quantiles=2,
        filter_zscore=None,
        demean=False,
        preview=False,
    )

    # by hand: the rows of this run that have all of their n bars
    expected = forward_return(prices, n, entry).iloc[first_bar:].stack().dropna()
    if sparse_filter:
        # a delayed filter on mondays keeps the row after it
        day_of_week = 1 if delayed_filter else 0
        expected = expected[expected.index.get_level_values(0).day_of_week == day_of_week]
    result = factor_data["Returns"].iloc[:, 0]
    assert sorted(result.index) == sorted(expected.index)
    assert_allclose(result.reindex(expected.index), expected, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
@pytest.mark.parametrize("trade_at", ["close", "open"])
def test_full_run_static_assets_have_nothing_to_delay(device, dirty_loader, dirty_panel, trade_at):
    """A fixed set of assets is the same on every bar, so naming all of them changes no row."""

    def run(universe):
        engine = factors.FactorEngine(dirty_loader)
        if device == "cuda":
            engine.to_cuda()
        if universe is not None:
            engine.set_filter(universe)
        engine.add(factors.MA(3), "f")
        return engine.full_run(
            DIRTY_DATES[5],
            DIRTY_DATES[-1],
            trade_at=trade_at,
            periods=(2,),
            quantiles=2,
            filter_zscore=None,
            preview=False,
        )[0]

    everyone = set(dirty_panel.index.get_level_values("asset"))
    pd.testing.assert_frame_equal(run(None), run(factors.StaticAssets(everyone)))


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_full_run_labels_periods_in_bars_of_the_data(device):
    """A return counts the bars of the data, a filter that keeps mondays doesn't stretch it."""
    panel, _, _ = forward_data()
    engine = forward_engine(panel, device, factors.WEEKDAY == 0)
    engine.add(factors.MA(2), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        periods=(3,),
        quantiles=2,
        filter_zscore=None,
        demean=False,
        preview=False,
    )
    assert list(factor_data["Returns"].columns) == ["3D"]


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_full_run_takes_its_columns_back_if_the_run_fails(device):
    class Fails(factors.CustomFactor):
        inputs = [close]

        def compute(self, data):
            raise ZeroDivisionError()

    panel, _, _ = forward_data()
    engine = forward_engine(panel, device)
    engine.add(Fails(), "f")
    for _ in range(2):
        # the second time would find the columns of the first
        with pytest.raises(ZeroDivisionError):
            engine.full_run(FORWARD_START, FORWARD_END, periods=(1,), preview=False)
        assert list(engine.factors) == ["f"]


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_full_run_takes_its_columns_back_if_a_name_is_taken(device):
    panel, _, _ = forward_data()
    engine = forward_engine(panel, device)
    engine.add(after_close(), "f")
    engine.add(after_close(), "f_w_")
    with pytest.raises(KeyError, match="already exists"):
        engine.full_run(FORWARD_START, FORWARD_END, periods=(1,), preview=False)
    assert list(engine.factors) == ["f", "f_w_"]


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_full_run_without_a_row_with_a_full_return(device):
    """MA(10) loads history, so there are dates to count, but none of them is in the range."""
    panel, _, _ = forward_data()
    engine = forward_engine(panel, device)
    engine.add(factors.MA(10), "f")
    with pytest.raises(AssertionError, match="No enough data for forward returns"):
        engine.full_run(FORWARD_DATES[-3], FORWARD_END, periods=(3,), preview=False)


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_full_run_does_not_drop_a_full_return_for_a_short_asset(device):
    """Gaps and early ends are per asset, the last row is the same for all assets."""
    panel, opens, _ = forward_data()
    n = 3
    last_bar = len(FORWARD_DATES) - 1
    ended, gap = "F", "E"
    panel = panel.drop(index=[(date, ended) for date in FORWARD_DATES[31:]])
    panel = panel.drop(index=[(FORWARD_DATES[last_bar - 1], gap)])
    engine = forward_engine(panel, device)
    engine.add(after_close(), "f")
    factor_data, _ = engine.full_run(
        FORWARD_START,
        FORWARD_END,
        trade_at="open",
        periods=(n,),
        quantiles=2,
        filter_zscore=None,
        demean=False,
        preview=False,
    )
    result = factor_data["Returns"].iloc[:, 0]
    assert result.index.get_level_values(0).max() == FORWARD_DATES[last_bar - n]

    for asset in opens:
        own_bars = panel.open.xs(asset, level="asset")
        expected = (own_bars.shift(-n) / own_bars - 1).dropna().iloc[6:]
        assert_allclose(result.xs(asset, level="asset").loc[expected.index], expected, rtol=1e-5)

    # a row whose asset has a gap after it ends on the last bar, like the other assets of the row
    own_bars = panel.open.xs(gap, level="asset")
    row = FORWARD_DATES[last_bar - n]
    assert result.loc[(row, gap)] == pytest.approx(own_bars.iloc[-1] / own_bars[row] - 1, rel=1e-5)


@pytest.mark.filterwarnings("ignore:quantile\\(\\) mask parameter not set")
@pytest.mark.parametrize("filter_zscore", [None, 20])
def test_full_run_keeps_forward_returns_of_delisted_asset(dirty_engine, dirty_panel, filter_zscore):
    """Near the delisting, the return is to the last close, it is read from the padding slots."""
    dirty_engine.add(factors.MA(2), "ma")
    factor_data, _ = dirty_engine.full_run(
        DIRTY_DATES[5],
        DIRTY_DATES[-1],
        periods=(3,),
        quantiles=2,
        filter_zscore=filter_zscore,
        preview=False,
    )
    returns = factor_data[("Returns", "3D")].xs("DELISTED", level="asset")

    closes = dirty_panel.close.xs("DELISTED", level="asset").to_numpy()
    last = len(closes) - 1
    assert last == DELISTED_ON
    forward = [closes[min(bar + 3, last)] / closes[bar] - 1 for bar in range(last)] + [np.nan]
    assert_allclose(returns, forward[-len(returns) :], rtol=1e-5, atol=1e-6)
    assert returns.iloc[-2] == pytest.approx(closes[last] / closes[last - 1] - 1, rel=1e-5)


def out_of_memory_once(monkeypatch, cls):
    """The next `cls.compute` raises a CUDA OOM, as a tensor too big for the GPU would."""
    real_compute = cls.compute
    calls = []

    def compute(self, *inputs):
        calls.append([t.device.type for t in inputs if hasattr(t, "device")])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
        return real_compute(self, *inputs)

    monkeypatch.setattr(cls, "compute", compute)
    return calls


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "cls, build",
    [
        (factors.factor.MedianFactor, lambda: close.median()),
        (factors.factor.MedianFactor, lambda: close.median(mask=close > 100)),
        (factors.factor.DemedianFactor, lambda: close.demedian()),
        (factors.MADClampFactor, lambda: close.mad_clamp(3)),
        (factors.WinsorizingFactor, lambda: close.winsorizing(0.1)),
        (factors.IQRNormalityFactor, lambda: factors.IQRNormalityFactor(inputs=[close])),
        (factors.factor.DemeanFactor, lambda: close.demean(groupby={"FULL": 1, "SPLIT": 2})),
        (factors.StaticAssets, lambda: ~factors.StaticAssets({"FULL"})),
        (factors.factor.RankFactor, lambda: close.rank()),
        (factors.MA, lambda: factors.MA(5)),
    ],
    ids=[
        "median",
        "median_mask",
        "demedian",
        "mad_clamp",
        "winsorizing",
        "iqr_normality",
        "demean_group_dict",
        "static_assets",
        "rank",
        "rolling",
    ],
)
def test_out_of_memory_falls_back_to_cpu(dirty_engine, monkeypatch, cls, build):
    dirty_engine.add(build(), "test")
    start, end = DIRTY_DATES[0], DIRTY_DATES[-1]
    expected = dirty_engine.run(start, end, delay_factor=False)

    calls = out_of_memory_once(monkeypatch, cls)
    result = dirty_engine.run(start, end, delay_factor=False)

    assert calls[-1] == ["cpu"] * len(calls[-1])
    assert len(calls) == 2
    assert_allclose(result.test, expected.test)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_out_of_memory_fallback_does_not_stick(dirty_engine, monkeypatch):
    start, end = DIRTY_DATES[0], DIRTY_DATES[-1]
    # the mask is in another group, so it is reverted on the device of the engine first
    dirty_engine.add(close.median(mask=close.rank() > 2), "test")
    expected = dirty_engine.run(start, end, delay_factor=False).test

    calls = out_of_memory_once(monkeypatch, factors.factor.MedianFactor)
    assert_allclose(dirty_engine.run(start, end, delay_factor=False).test, expected)
    assert_allclose(dirty_engine.run(start, end, delay_factor=False).test, expected)
    assert calls[-1] == [dirty_engine.device.type]


def test_failed_out_of_memory_fallback_does_not_stick(dirty_engine, monkeypatch):
    factor = close.median()
    dirty_engine.add(factor, "test")
    start, end = DIRTY_DATES[0], DIRTY_DATES[-1]

    def compute(self, data):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")

    with monkeypatch.context() as patch:
        patch.setattr(factors.factor.MedianFactor, "compute", compute)
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            dirty_engine.run(start, end)
    assert factor._computing_on_cpu is False
    assert not dirty_engine.run(start, end).test.isna().all()


def test_factors_reset_after_error(engine):
    """Shared factors (like OHLCV.close) should not keep the data of a failed run."""

    class Fail(factors.CustomFactor):
        inputs = [factors.MarketReturn()]

        def compute(self, market_return):
            raise ZeroDivisionError()

    engine.add(Fail(), "fail")
    with pytest.raises(ZeroDivisionError):
        engine.run("2018-01-01", ref.END)
    assert factors.OHLCV.close._data is None

    # different data shape
    engine.remove_all_factors()
    engine.add(factors.MarketReturn(), "market_return")
    result = engine.run(ref.START, ref.END)
    expected = ref.daily_prices("AAPL").close / ref.daily_prices("AAPL").close.shift(1) - 1
    assert result.xs("AAPL", level=1).market_return.iloc[6] == pytest.approx(
        expected.loc["2019-01-10"]
    )


def test_settings_invalidate_cache(engine):
    engine.add(factors.WEEKDAY, "weekday")
    utc = engine.run(ref.START, ref.END).weekday.values

    # 00:00 UTC is the day before in New York
    engine.timezone = "America/New_York"
    new_york = engine.run(ref.START, ref.END).weekday.values
    assert_array_equal(new_york, (utc - 1) % 7)

    engine.timezone = "UTC"
    assert_array_equal(engine.run(ref.START, ref.END).weekday.values, utc)

    # MSFT gets a row on 2019-01-10
    engine.align_by_time = True
    assert len(engine.run(ref.START, ref.END)) == len(utc) + 1
