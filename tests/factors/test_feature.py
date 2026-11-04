import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from spectre import factors
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, delete, run_factor

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
    market_return = pd.DataFrame({a: returns(ref.daily_prices(a)) for a in ref.ASSETS}).mean(
        axis=1)

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
    msft = ref.daily_prices('MSFT').close.reindex(ref.daily_prices('AAPL').index)
    expected = msft.shift(1).loc['2019-01-03':ref.END].values
    assert np.isnan(expected[SOLO])
    assert_factor(engine, factors.AssetData('MSFT', factors.OHLCV.close), expected, expected)


def test_asset_data_warns_not_aligned(engine):
    with pytest.warns(RuntimeWarning, match='Make sure your data is aligned by time'):
        run_factor(engine, factors.AssetData('MSFT', factors.OHLCV.close), check_bias=False)


# ------------------------------------ calendar features ------------------------------------


@pytest.mark.parametrize('factor, attr', [
    (factors.WEEKDAY, 'day_of_week'),
    (factors.MONTH, 'month'),
    (factors.QUARTER, 'quarter'),
    (factors.DatetimeDataFactor('day'), 'day'),
])
def test_datetime_factor(engine, factor, attr):
    # date features are available before market open, not delayed
    expected = ref.expected(lambda df: pd.Series(getattr(df.index, attr), index=df.index),
                            delayed=False)
    assert_factor(engine, factor, *expected)


def test_weekday_values(engine):
    expected = np.array([2, 3., 4., 0., 1., 2., 3., 4., 0., 1.])
    assert_factor(engine, factors.WEEKDAY, expected, delete(expected, SOLO))


def test_is_january(engine):
    assert_factor(engine, factors.IS_JANUARY, [True] * 10, [True] * 9)
    assert_factor(engine, factors.IS_DECEMBER, [False] * 10, [False] * 9)


def test_timezone(engine):
    engine.timezone = 'America/New_York'

    def weekday_in_ny(df):
        ny = df.index.tz_localize('UTC').tz_convert('America/New_York')
        return pd.Series(ny.day_of_week, index=df.index)

    # 2019-01-02 00:00 UTC is 2019-01-01 19:00 in New York
    expected = ref.expected(weekday_in_ny, delayed=False)
    assert_array_equal(expected[0], [1., 2, 3, 6, 0, 1, 2, 3, 6, 0])
    assert_factor(engine, factors.WEEKDAY, *expected)

    # change back, cached data should be invalidated
    engine.timezone = 'UTC'
    expected = np.array([2, 3., 4., 0., 1., 2., 3., 4., 0., 1.])
    assert_factor(engine, factors.WEEKDAY, expected, delete(expected, SOLO))


# ------------------------------------ classifier ------------------------------------


def test_asset_classifier(engine):
    classifier = factors.AssetClassifierDataFactor({'AAPL': 2}, -1)
    assert_factor(engine, classifier, [2] * 10, [-1] * 9)


def test_one_hot(engine):
    one_hot = factors.AssetClassifierDataFactor({'AAPL': 2}, -1).one_hot()
    # classes are sorted: -1, 2
    assert_factor(engine, one_hot[0], [False] * 10, [True] * 9)
    assert_factor(engine, one_hot[1], [True] * 10, [False] * 9)
    with pytest.raises(KeyError, match='OutOfBounds'):
        run_factor(engine, one_hot[2])


# ------------------------------------ label ------------------------------------


def test_forward_signal_data(engine):
    """Close price of the first signal in the next 4 bars (lookahead biased)."""
    signal = factors.OHLCV.close > 150
    signal_price = factors.ForwardSignalData(4, factors.OHLCV.close, signal)
    signal_return = signal_price / factors.OHLCV.close - 1

    def forward_signal_return(df):
        close = df.close.loc[:ref.END]  # no future data after `end`
        window = pd.concat([close.shift(-i) for i in range(4)], axis=1)
        first = window.where(window > 150).bfill(axis=1).iloc[:, 0]
        return first / close - 1

    expected = ref.expected(forward_signal_return)
    assert_almost_equal(expected[0], [0, 151.55 / 145.23 - 1, 151.55 / 149 - 1,
                                      151.55 / 149 - 1, 0, 0, 0, 0, 0])
    # MSFT never > 150
    assert np.isnan(expected[1]).all()
    assert_factor(engine, signal_return, *expected, check_bias=False)
