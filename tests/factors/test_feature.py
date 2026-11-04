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


def broadcast(series, df):
    return pd.DataFrame({asset: series for asset in df})


# ------------------------------------ market features ------------------------------------


def test_market_dispersion(engine):
    expected = ref.expected_xs(lambda df: broadcast(df.std(axis=1, ddof=0), df), values=returns)
    assert_factor(engine, factors.MarketDispersion(), *expected)
    assert expected[0][SOLO] == 0


def test_market_return(engine):
    expected = ref.expected_xs(lambda df: broadcast(df.mean(axis=1), df), values=returns)
    assert_factor(engine, factors.MarketReturn(), *expected)


def test_market_volatility(engine):
    market_return = pd.DataFrame({a: returns(ref.daily_prices(a)) for a in ref.ASSETS}).mean(axis=1)

    def volatility(df):
        # rolling on the market returns of each asset's own trading days
        ret = market_return[df.index]
        return (ret.rolling(252).var(ddof=0) * 252) ** 0.5

    expected = ref.expected(volatility)
    assert_factor(engine, factors.MarketVolatility(), *expected)
    # MSFT missed one day of the market returns
    assert expected[0][SOLO] != expected[1][SOLO]


def test_advance_decline_ratio(engine):
    def ratio(df):
        return broadcast((df > 0).sum(axis=1) / (df < 0).sum(axis=1), df)

    expected = ref.expected_xs(ratio, values=returns)
    assert_factor(engine, factors.AdvanceDeclineRatio(), *expected)


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
