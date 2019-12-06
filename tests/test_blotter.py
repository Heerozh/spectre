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

        pf.set_date("2019-01-05")
        pf.update('MSFT', 30)
        pf.update_cash(2000)
        pf.process_dividends('AAPL', 2.0)
        pf.process_dividends('MSFT', 2.0)

        self.assertEqual(pf.positions, {'AAPL': 10, 'MSFT': 5, 'cash': 7030})
        assert_array_equal(pf.get_history_positions().values,
                           [[10000, nan, nan],
                            [5000, 20, -25],
                            [5000, 10, -25],
                            [7030, 10, 5]])

    def test_blotter(self):
        pass



