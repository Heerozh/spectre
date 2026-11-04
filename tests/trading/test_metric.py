import pandas as pd
import pytest
from numpy import nan
from numpy.testing import assert_almost_equal

from spectre import trading

DAILY_RETURNS = [0.0022, 0.0090, -0.0067, 0.0052, 0.0030, -0.0012, -0.0091, 0.0082, -0.0071, 0.0093]


@pytest.fixture
def returns():
    return pd.Series(DAILY_RETURNS, index=pd.date_range('2040-01-01', periods=10))


@pytest.mark.parametrize('risk_free, expected', [(0., 2.9101144), (0.04, 2.5492371)])
def test_sharpe_ratio(returns, risk_free, expected):
    # sqrt(252) * mean(r - rf / 252) / std(r - rf / 252, ddof=1)
    assert trading.sharpe_ratio(returns, risk_free) == pytest.approx(expected)


def test_annual_volatility(returns):
    # sqrt(252) * std(r, ddof=1)
    assert trading.annual_volatility(returns) == pytest.approx(0.110841, abs=1e-6)


def test_drawdown(returns):
    dd, duration = trading.drawdown((returns + 1).cumprod())

    assert_almost_equal(dd.values, [0, 0, -0.0067, -0.00153, 0, -0.0012, -0.0102891,
                                    -0.00217, -0.00926, -0.00004], decimal=5)
    assert dd.abs().max() == pytest.approx(0.0102891, abs=1e-7)
    # bars since the last high
    assert duration.tolist() == [0, 0, 1, 2, 0, 1, 2, 3, 4, 5]


def test_turnover():
    txn = pd.DataFrame([['AAPL', 384, 155.92, 157.09960, 1.92],
                        ['AAPL', -384, 158.61, 157.41695, 1.92]],
                       columns=['symbol', 'amount', 'price', 'fill_price', 'commission'],
                       index=pd.to_datetime(['2040-01-01', '2040-01-02']))
    pos = pd.DataFrame([[384, 384 * 155.92, 10000],
                        [nan, nan, 10000 + 384 * 158.61]],
                       columns=pd.MultiIndex.from_tuples(
                           [('shares', 'AAPL'), ('value', 'AAPL'), ('value', 'cash')]),
                       index=pd.to_datetime(['2040-01-01', '2040-01-02']))

    # traded value / portfolio value
    expected = [384 * 157.09960 / (384 * 155.92 + 10000),
                384 * 157.41695 / (10000 + 384 * 158.61)]
    assert_almost_equal(trading.turnover(pos, txn).values, expected)
    assert_almost_equal(expected, [0.8633665, 0.8525076])


def test_turnover_without_transactions():
    txn = pd.DataFrame(columns=['symbol', 'amount', 'price', 'fill_price', 'commission'])
    pos = pd.DataFrame({('value', 'cash'): [10000.]}, index=pd.to_datetime(['2040-01-01']))
    assert len(trading.turnover(pos, txn)) == 0
