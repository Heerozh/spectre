import unittest
import spectre
import pandas as pd
from numpy import nan
from numpy.testing import assert_almost_equal


class TestMetric(unittest.TestCase):

    def test_metrics(self):
        ret = pd.Series([0.0022, 0.0090, -0.0067, 0.0052, 0.0030, -0.0012, -0.0091, 0.0082,
                         -0.0071, 0.0093],
                        index=pd.date_range('2040-01-01', periods=10))

        self.assertAlmostEqual(2.9101144, spectre.trading.sharpe_ratio(ret, 0.00))
        self.assertAlmostEqual(2.5492371, spectre.trading.sharpe_ratio(ret, 0.04))

        dd, ddu = spectre.trading.drawdown((ret+1).cumprod())
        vol = spectre.trading.annual_volatility(ret)

        self.assertAlmostEqual(0.0102891, dd.abs().max())
        self.assertAlmostEqual(5, ddu.max())
        self.assertAlmostEqual(0.110841, vol)

        txn = pd.DataFrame([['AAPL', 384, 155.92, 157.09960, 1.92],
                            ['AAPL', -384, 158.61, 157.41695, 1.92]],
                           columns=['symbol', 'amount', 'price',
                                    'fill_price', 'commission'],
                           index=['2040-01-01', '2040-01-02'])
        txn.index = pd.to_datetime(txn.index)

        pos = pd.DataFrame([[384, 384*155.92, 10000],
                            [nan, nan, 10000+384*158.61]],
                           columns=pd.MultiIndex.from_tuples(
                               [('shares', 'AAPL'), ('value', 'AAPL'), ('value', 'cash')]),
                           index=['2040-01-01', '2040-01-02'])
        pos.index = pd.to_datetime(pos.index)
        assert_almost_equal([0.8633665, 0.8525076], spectre.trading.turnover(pos, txn))
