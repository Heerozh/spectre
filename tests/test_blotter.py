import unittest
import spectre
import pandas as pd
from os.path import dirname
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy import nan

data_dir = dirname(__file__) + '/data/'


class TestBlotter(unittest.TestCase):

    def test_portfolio(self):
        pf = spectre.trading.Portfolio()
        pf.set_date("2019-01-01")
        pf.update_cash(10000)

        pf.set_date("2019-01-03")
        pf.update('AAPL', 10)
        pf.update('MSFT', -25)
        pf.update_cash(-5000)
        pf.process_split('AAPL', 2.0)

        pf.set_date("2019-01-04")
        pf.update('AAPL', -10)

        pf.set_date("2019-01-05 23:00:00")
        pf.update('MSFT', 30)
        pf.update_cash(2000)
        pf.process_dividends('AAPL', 2.0)
        pf.process_dividends('MSFT', 2.0)
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)

        self.assertEqual({'AAPL': 10, 'MSFT': 5}, pf.positions)
        self.assertEqual(7030, pf.cash)
        assert_array_equal([[10000, nan, nan, nan],
                            [5000, 20, -25, nan],
                            [5000, 10, -25, nan],
                            [7030, 10, 5, 7055]], pf.history.values)
        self.assertEqual(str(pf),
                         """<Portfolio>:
               cash  AAPL  MSFT   value
2019-01-01  10000.0   NaN   NaN     NaN
2019-01-03   5000.0  20.0 -25.0     NaN
2019-01-04   5000.0  10.0 -25.0     NaN
2019-01-05   7030.0  10.0   5.0  7055.0""")
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)
        self.assertEqual(7030 + 15 + 10, pf.value)
        self.assertEqual(25 / 7055, pf.leverage)

        pf.set_date("2019-01-06")
        pf.update_cash(-6830)
        pf.update('AAPL', -100)
        pf.update('MSFT', 50)
        self.assertEqual(200, pf.cash)
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)
        self.assertEqual(200 + -135 + 110, pf.value)
        self.assertEqual(245 / 175, pf.leverage)

        pf.set_date("2019-01-07")
        pf.update_cash(-200)
        pf.update('AAPL', 400)
        pf.update('MSFT', 50)
        pf.update_cash(-337.5)
        pf.update_value(lambda x: None)
        self.assertEqual(2, pf.leverage)

    def test_blotter(self):
        pass
