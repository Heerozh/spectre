import pytest
from numpy.testing import assert_almost_equal

from spectre import factors
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor


# The EMAs used by MACD are truncated (see test_basic.test_ema), the error is about 1e-3
@pytest.mark.parametrize('params', [(12, 26, 9), (10, 15, 5)])
def test_macd_signal(engine, params):
    expected = ref.expected(lambda df: ref.macd(df.close, *params)[1])
    assert_factor(engine, factors.MACD(*params), *expected, decimal=3)


def test_macd_histogram(engine):
    expected = ref.expected(lambda df: ref.macd(df.close)[2])
    assert_factor(engine, factors.MACD().normalized(), *expected, decimal=3)


@pytest.mark.parametrize('band', [0, 1, 2])
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
    with pytest.raises(ValueError, match='needs 2 inputs'):
        factors.BBANDS(inputs=[factors.OHLCV.close])


def test_true_range(engine):
    assert_factor(engine, factors.TRANGE(), *ref.expected(ref.true_range))


def test_rsi(engine):
    expected = ref.expected(lambda df: ref.rsi(df.close))
    assert_factor(engine, factors.RSI(), *expected)
    # Cutler's RSI calculated by excel
    assert_almost_equal(expected[0], [40.1814301, 33.36385487, 37.37511353, 36.31220413,
                                      41.84100418, 39.19197118, 48.18441452, 44.30411404,
                                      50.05167959])
    assert_almost_equal(expected[1], [38.5647217, 42.0627596, 37.9693676, 43.8641553,
                                      48.3458438, 47.095672, 46.7363662, 46.127465])


def test_rsi_normalized(engine):
    expected = ref.expected(lambda df: ref.rsi(df.close) / 50 - 1)
    assert_factor(engine, factors.RSI().normalized(), *expected)


def test_stochf(engine):
    assert_factor(engine, factors.STOCHF(), *ref.expected(ref.stochf))


def test_stochf_normalized(engine):
    assert_factor(engine, factors.STOCHF().normalized(),
                  *ref.expected(lambda df: ref.stochf(df) / 100 - 0.5))
