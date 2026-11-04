import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

import spectre
from spectre import factors
from tests.conftest import DELISTED_ON, DIRTY_ADJUSTMENTS, DIRTY_OHLCV
from tests.factors import reference as ref
from tests.factors.helpers import aligned_bars, assert_factor, nansplit_bars, run_dirty

nan = np.nan


# The EMAs used by MACD are truncated (see test_basic.test_ema), the error is about 1e-3
@pytest.mark.parametrize("params", [(12, 26, 9), (10, 15, 5)])
def test_macd_signal(engine, params):
    expected = ref.expected(lambda df: ref.macd(df.close, *params)[1])
    assert_factor(engine, factors.MACD(*params), *expected, decimal=3)


def test_macd_histogram(engine):
    expected = ref.expected(lambda df: ref.macd(df.close)[2])
    assert_factor(engine, factors.MACD().normalized(), *expected, decimal=3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_of_dirty_assets(dirty_engine, dirty_panel, params):
    signal = run_dirty(dirty_engine, factors.MACD(*params))
    histogram = run_dirty(dirty_engine, factors.MACD(*params).normalized())
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        close = dirty_panel.xs(asset, level="asset").close
        _, expected_signal, expected_histogram = ref.macd(close, *params)
        got = signal.xs(asset, level="asset").values
        assert_allclose(got, expected_signal.values, rtol=0, atol=2e-3, err_msg=asset)
        got = histogram.xs(asset, level="asset").values
        assert_allclose(got, expected_histogram.values, rtol=0, atol=2e-3, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_adjust(dirty_engine, dirty_panel, params):
    """`adjust` is for the fast and slow EMAs as well, all three match ewm(adjust=True)."""
    signal = run_dirty(dirty_engine, factors.MACD(*params, adjust=True))
    histogram = run_dirty(dirty_engine, factors.MACD(*params, adjust=True).normalized())
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        close = dirty_panel.xs(asset, level="asset").close
        _, expected_signal, expected_histogram = ref.macd(close, *params, adjust=True)
        got = signal.xs(asset, level="asset").values
        assert_allclose(got, expected_signal.values, rtol=0, atol=2e-3, err_msg=asset)
        got = histogram.xs(asset, level="asset").values
        assert_allclose(got, expected_histogram.values, rtol=0, atol=2e-3, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_across_split(dirty_engine, dirty_panel, params, adjust):
    """MACD is linear in the price, so it is the MACD of clean_close in the scale of each bar."""
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    _, signal, histogram = ref.macd(split.clean_close, *params, adjust=adjust)

    factor = factors.MACD(*params, adjust=adjust)
    result = run_dirty(dirty_engine, factor).xs("SPLIT", level="asset")
    assert_allclose(result.values, (signal * scale).values, rtol=0, atol=2e-3)
    result = run_dirty(dirty_engine, factors.MACD(*params, adjust=adjust).normalized())
    result = result.xs("SPLIT", level="asset")
    assert_allclose(result.values, (histogram * scale).values, rtol=0, atol=2e-3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_across_split_after_nan_bar(dirty_engine, dirty_panel, params, adjust):
    bars, scale = nansplit_bars(dirty_panel)
    _, signal, histogram = ref.macd(bars.clean_close, *params, adjust=adjust)

    result = run_dirty(dirty_engine, factors.MACD(*params, adjust=adjust))
    result = result.xs("NANSPLIT", level="asset")
    assert_allclose(result.values, (signal * scale).values, rtol=0, atol=2e-3)
    result = run_dirty(dirty_engine, factors.MACD(*params, adjust=adjust).normalized())
    result = result.xs("NANSPLIT", level="asset")
    assert_allclose(result.values, (histogram * scale).values, rtol=0, atol=2e-3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_on_rows_added_by_align_by_time(added_rows_engine, added_rows_panel, params, adjust):
    """EMA keeps its value on these rows, so the line is valid there and stays in the window."""
    signal = run_dirty(added_rows_engine, factors.MACD(*params, adjust=adjust)).unstack("asset")
    for asset in ("FULL", "NANSPLIT", "LISTED", "DELISTED", "HALTED"):
        bars, scale = aligned_bars(added_rows_panel, asset)
        expected = ref.macd(bars.clean_close, *params, adjust=adjust)[1] * scale
        # later on, the windows of DELISTED run out of valid bars and leave `ewm` behind
        rows = DELISTED_ON + 10 if asset == "DELISTED" else None
        got = signal[asset].iloc[:rows]
        # a window with a gap has fewer bars to decay the truncation error
        assert_allclose(got, expected.iloc[:rows], rtol=0, atol=5e-3, err_msg=asset)


@pytest.mark.parametrize("band", [0, 1, 2])
def test_bbands(engine, band):
    expected = ref.expected(lambda df: ref.bbands(df.close)[band])
    assert_factor(engine, factors.BBANDS()[band], *expected)


def bbands_normalized(win, k):
    def _normalized(df):
        upper, middle, _ = ref.bbands(df.close, win, k)
        return (df.close - middle) / (upper - middle)

    return _normalized


def test_bbands_normalized(engine):
    assert_factor(engine, factors.BBANDS().normalized(), *ref.expected(bbands_normalized(20, 2)))
    factor = factors.BBANDS(win=50, inputs=(factors.OHLCV.close, 3)).normalized()
    assert_factor(engine, factor, *ref.expected(bbands_normalized(50, 3)))


def test_bbands_needs_k():
    with pytest.raises(ValueError, match="needs 2 inputs"):
        factors.BBANDS(inputs=[factors.OHLCV.close])


def test_true_range(engine):
    assert_factor(engine, factors.TRANGE(), *ref.expected(ref.true_range))


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_of_dirty_assets(dirty_engine, dirty_panel):
    win = 14
    result = run_dirty(dirty_engine, factors.MA(win, inputs=(factors.TRANGE(),)))
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        df = dirty_panel.xs(asset, level="asset")
        expected = ref.true_range(df).rolling(win, min_periods=1).mean()
        got = result.xs(asset, level="asset").values
        assert_allclose(got, expected.values, rtol=1e-6, err_msg=asset)


def check_true_range_of_scaled_bars(engine, asset, bars, scale):
    """Range and ATR of `asset` are the ones of its clean prices, in the scale of each bar."""
    win = 14
    clean = pd.DataFrame(
        {"high": bars.high / scale, "low": bars.low / scale, "close": bars.clean_close}
    )
    true_range = ref.true_range(clean)

    result = run_dirty(engine, factors.TRANGE()).xs(asset, level="asset")
    assert_allclose(result.values, (true_range * scale).values, rtol=1e-5)

    atr = factors.MA(win, inputs=(factors.TRANGE(),))
    result = run_dirty(engine, atr).xs(asset, level="asset")
    expected = true_range.rolling(win, min_periods=1).mean() * scale
    assert_allclose(result.values, expected.values, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_across_split(dirty_engine, dirty_panel):
    """ATR from the docstring of TrueRange. Each bar's range is in the scale of that bar."""
    split = dirty_panel.xs("SPLIT", level="asset")
    check_true_range_of_scaled_bars(dirty_engine, "SPLIT", split, split.close / split.clean_close)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_across_split_after_nan_bar(dirty_engine, dirty_panel):
    bars, scale = nansplit_bars(dirty_panel)
    check_true_range_of_scaled_bars(dirty_engine, "NANSPLIT", bars, scale)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_on_rows_added_by_align_by_time(added_rows_engine, added_rows_panel):
    bars, scale = aligned_bars(added_rows_panel, "NANSPLIT")
    check_true_range_of_scaled_bars(added_rows_engine, "NANSPLIT", bars, scale)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "scale",
    [[1] * 7, [1] * 4 + [0.5] * 3],
    ids=["no split", "split on the next bar"],
)
def test_atr_keeps_a_bar_with_only_the_close_missing(device, scale):
    """
    The bar has a high, a low and a close before it, so it has a range. Clean prices by hand:
    true range [NaN, 3, 2, 5, NaN, 4, 2], the NaN after it has no close before it.
    """
    scale = np.array(scale)
    dates = pd.date_range("2020-01-01", periods=7, freq="B", tz="UTC")
    bars = pd.DataFrame(
        {
            "high": np.array([11.0, 13, 12, 16, 15, 18, 17]) * scale,
            "low": np.array([9.0, 10, 10, 12, 13, 14, 15]) * scale,
            "close": np.array([10.0, 12, 11, nan, 14, 17, 16]) * scale,
            "volume": 1000.0,
            "ex-dividend": 0.0,
            "split_ratio": np.where(np.diff(scale, prepend=1) == 0, 1.0, 0.5),
        },
        index=dates,
    )
    bars["open"] = bars.close
    panel = pd.concat({"A": bars, "B": bars}, names=["asset", "date"]).swaplevel().sort_index()
    loader = spectre.data.MemoryLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(factors.MA(3, inputs=(factors.TRANGE(),)), "atr")
    result = engine.run(dates[0], dates[-1], delay_factor=False).xs("A", level="asset")
    expected = np.array([nan, 3, 5 / 2, 10 / 3, 7 / 2, 9 / 2, 3]) * scale
    assert_allclose(result.atr.values, expected, rtol=1e-6)


def test_rsi(engine):
    expected = ref.expected(lambda df: ref.rsi(df.close))
    assert_factor(engine, factors.RSI(), *expected)
    # Cutler's RSI calculated by excel
    assert_almost_equal(expected[0], [40.1814301, 33.36385487, 37.37511353, 36.31220413,
                                      41.84100418, 39.19197118, 48.18441452, 44.30411404,
                                      50.05167959])  # fmt: skip
    assert_almost_equal(expected[1], [38.5647217, 42.0627596, 37.9693676, 43.8641553,
                                      48.3458438, 47.095672, 46.7363662, 46.127465])  # fmt: skip


def test_rsi_normalized(engine):
    expected = ref.expected(lambda df: ref.rsi(df.close) / 50 - 1)
    assert_factor(engine, factors.RSI().normalized(), *expected)


def test_stochf(engine):
    assert_factor(engine, factors.STOCHF(), *ref.expected(ref.stochf))


def test_stochf_normalized(engine):
    assert_factor(
        engine, factors.STOCHF().normalized(), *ref.expected(lambda df: ref.stochf(df) / 100 - 0.5)
    )
