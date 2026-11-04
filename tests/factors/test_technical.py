import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from spectre import factors
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, run_dirty


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
def test_macd_across_split(dirty_engine, dirty_panel, params):
    """MACD is linear in the price, so it is the MACD of clean_close in the scale of each bar."""
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    _, signal, histogram = ref.macd(split.clean_close, *params)

    result = run_dirty(dirty_engine, factors.MACD(*params)).xs("SPLIT", level="asset")
    assert_allclose(result.values, (signal * scale).values, rtol=0, atol=2e-3)
    result = run_dirty(dirty_engine, factors.MACD(*params).normalized())
    result = result.xs("SPLIT", level="asset")
    assert_allclose(result.values, (histogram * scale).values, rtol=0, atol=2e-3)


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


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_across_split(dirty_engine, dirty_panel):
    """ATR from the docstring of TrueRange. Each bar's range is in the scale of that bar."""
    win = 14
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    clean = pd.DataFrame(
        {"high": split.high / scale, "low": split.low / scale, "close": split.clean_close}
    )
    true_range = ref.true_range(clean)

    result = run_dirty(dirty_engine, factors.TRANGE()).xs("SPLIT", level="asset")
    assert_allclose(result.values, (true_range * scale).values, rtol=1e-5)

    atr = factors.MA(win, inputs=(factors.TRANGE(),))
    result = run_dirty(dirty_engine, atr).xs("SPLIT", level="asset")
    expected = true_range.rolling(win, min_periods=1).mean() * scale
    assert_allclose(result.values, expected.values, rtol=1e-5)


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
