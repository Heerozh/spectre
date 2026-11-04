import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_almost_equal

from spectre import factors, parallel
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, run_factor

nan = np.nan


def weekday(df):
    return pd.Series(df.index.day_of_week, index=df.index, dtype=float)


def day(df):
    return pd.Series(df.index.day, index=df.index, dtype=float)


# ------------------------------------ returns ------------------------------------


def test_returns(engine):
    expected = ref.expected(lambda df: ref.returns(df.close))
    assert_factor(engine, factors.Returns(), *expected)
    # close of 2019-01-02 / close of 2018-12-31 - 1, delayed to 2019-01-03
    assert_almost_equal(expected[0][:2], [158.61 / 163.71 - 1, 145.23 / 158.61 - 1])


def test_returns_window(engine):
    expected = ref.expected(lambda df: ref.returns(df.close, win=5))
    assert_factor(engine, factors.Returns(win=5), *expected)


def test_returns_with_nan(device):
    """missing data considered as delisted, returns are calculated by the last valid data."""
    data = torch.tensor([[1, 2, nan, 4, 8, nan],
                         [3, 4, 4.5, 6, nan, nan]], device=device)
    result = factors.Returns().compute(parallel.Rolling(data, win=3)).cpu().numpy()
    expected = [[nan, nan, 1, 1, nan, 1],
                [nan, nan, 0.5, 0.5, 1 / 3, nan]]
    assert_almost_equal(result, expected)


def test_log_returns(engine):
    assert_factor(engine, factors.LogReturns(),
                  *ref.expected(lambda df: ref.log_returns(df.close)))


# ------------------------------------ moving averages ------------------------------------


@pytest.mark.parametrize('win', [3, 11])
def test_sma(engine, win):
    assert_factor(engine, factors.SMA(win), *ref.expected(lambda df: ref.sma(df.close, win)))


def test_sma_of_open_not_delayed(engine):
    # open price is available at market open, should not be delayed
    expected = ref.expected(lambda df: ref.sma(df.open, 11), delayed=False)
    assert_factor(engine, factors.SMA(11, inputs=[factors.OHLCV.open]), *expected, delay=False)
    assert len(expected[0]) == 10


def ema_truncation_bound(span, prices):
    """
    spectre's EMA only uses the last `4.5 * (span + 1)` bars, the truncation error is
    (1 - alpha) ^ win * (EMA(t - win) - X(t - win + 1)), which is bounded by the price range.
    """
    alpha = 2 / (span + 1)
    win = int(4.5 * (span + 1))
    return (1 - alpha) ** win * (prices.max() - prices.min())


@pytest.mark.parametrize('span', [11, 50])
@pytest.mark.parametrize('adjust', [False, True])
def test_ema(engine, span, adjust):
    aapl, msft = run_factor(engine, factors.EMA(span, adjust=adjust))
    expected = ref.expected(lambda df: df.close.ewm(span=span, adjust=adjust).mean())
    for asset, result, exp in zip(ref.ASSETS, (aapl, msft), expected):
        bound = ema_truncation_bound(span, ref.daily_prices(asset).close)
        assert bound < 0.02
        assert np.abs(result - exp).max() <= bound, asset


def test_ema_half_life(engine):
    aapl, msft = run_factor(engine, factors.EMA(half_life=5))
    expected = ref.expected(lambda df: df.close.ewm(halflife=5, adjust=False).mean())
    # window is 15 half-lives, truncation error is 0.5^15 * price range
    assert_almost_equal(aapl, expected[0], decimal=2)
    assert_almost_equal(msft, expected[1], decimal=2)


def test_vwap(engine):
    def vwap(df):
        return (df.close * df.volume).rolling(3).sum() / df.volume.rolling(3).sum()

    expected = ref.expected(vwap)
    assert_factor(engine, factors.VWAP(3), *expected)
    assert_almost_equal(expected[1][:3], [103.3029256, 102.5066541, 102.5480467])


def test_average_dollar_volume(engine):
    expected = ref.expected(lambda df: (df.close * df.volume).rolling(3).mean())
    assert_factor(engine, factors.AverageDollarVolume(3), *expected, decimal=2)
    # win=1 is close * volume
    assert_factor(engine, factors.AverageDollarVolume(1),
                  *ref.expected(lambda df: df.close * df.volume), decimal=2)


def test_annualized_volatility(engine):
    def volatility(df):
        return ref.returns(df.close).rolling(3).std(ddof=0) * np.sqrt(252)

    assert_factor(engine, factors.AnnualizedVolatility(3), *ref.expected(volatility))


def test_linear_weighted_average(engine):
    weights = np.arange(1, 6) / 15

    def lwa(df):
        return weekday(df).rolling(5).apply(lambda x: (x * weights).sum(), raw=True)

    expected = ref.expected(lwa, delayed=False)
    assert_factor(engine, factors.LinearWeightedAverage(5, inputs=[factors.WEEKDAY]), *expected,
                  decimal=6)
    assert_almost_equal(expected[0][:3], [2., 2.2666667, 2.8])


# ------------------------------------ element wise ------------------------------------


def test_element_wise_max(engine):
    factor = factors.ElementWiseMax(inputs=[factors.WEEKDAY, factors.DatetimeDataFactor('day')])
    expected = ref.expected(lambda df: np.maximum(weekday(df), day(df)), delayed=False)
    assert_factor(engine, factor, *expected)
    assert_almost_equal(expected[0], [2., 3, 4, 7, 8, 9, 10, 11, 14, 15])


def test_element_wise_min(engine):
    factor = factors.ElementWiseMin(
        inputs=[factors.WEEKDAY, factors.DatetimeDataFactor('day') - 1])
    expected = ref.expected(lambda df: np.minimum(weekday(df), day(df) - 1), delayed=False)
    assert_factor(engine, factor, *expected)
    assert_almost_equal(expected[0], [1., 2, 3, 0, 1, 2, 3, 4, 0, 1])


def test_element_wise_max_nan():
    a = torch.tensor([[1., nan, nan, 4]])
    b = torch.tensor([[2., 1., nan, 3]])
    assert_almost_equal(factors.ElementWiseMax().compute(a, b), [[2, 1, nan, 4]])
    assert_almost_equal(factors.ElementWiseMin().compute(a, b), [[1, 1, nan, 3]])


# weekday has same values in the window, use day as a decimal point to break the ties
WEEKDAY_DAY = factors.WEEKDAY + factors.DatetimeDataFactor('day') / 100


def rolling_arg(func):
    def _arg(df):
        return (weekday(df) + day(df) / 100).rolling(5).apply(lambda x: (func(x) + 1) / 5,
                                                              raw=True)
    return _arg


def test_rolling_argmax(engine):
    expected = ref.expected(rolling_arg(np.argmax), delayed=False)
    assert_factor(engine, factors.RollingArgMax(5, inputs=[WEEKDAY_DAY]), *expected)
    assert_almost_equal(expected[0], [3 / 5, 2 / 5, 1 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 5 / 5,
                                      4 / 5, 3 / 5])


def test_rolling_argmin(engine):
    expected = ref.expected(rolling_arg(np.argmin), delayed=False)
    assert_factor(engine, factors.RollingArgMin(5, inputs=[WEEKDAY_DAY]), *expected)
    assert_almost_equal(expected[1], [4 / 5, 3 / 5, 2 / 5, 5 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5,
                                      4 / 5])


def test_constants_factor(engine):
    aapl, msft = run_factor(engine, factors.ConstantsFactor(3.5), delay=False)
    assert_almost_equal(aapl, [3.5] * 10)
    assert_almost_equal(msft, [3.5] * 9)


def test_min_win():
    with pytest.raises(ValueError, match='win'):
        factors.SMA(1)
