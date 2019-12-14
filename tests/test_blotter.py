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
        pf.update('AAPL', 10, None)
        pf.update('MSFT', -25, None)
        pf.update_cash(-5000)
        pf.process_split('AAPL', 2.0)

        pf.set_date("2019-01-04")
        pf.update('AAPL', -10, None)

        pf.set_date("2019-01-05 23:00:00")
        pf.update('MSFT', 30, None)
        pf.update_cash(2000)
        pf.process_dividends('AAPL', 2.0)
        pf.process_dividends('MSFT', 2.0)
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)

        self.assertEqual({'AAPL': 10, 'MSFT': 5}, pf.positions)
        self.assertEqual(7030, pf.cash)
        expected = pd.DataFrame([[nan, nan, nan, nan, 10000.],
                                 [20., -25, nan, nan, 5000.],
                                 [10., -25, nan, nan, 5000.],
                                 [10., 5, 15, 10, 7030.]],
                                columns=pd.MultiIndex.from_tuples(
                                    [('amount', 'AAPL'), ('amount', 'MSFT'),
                                     ('value', 'AAPL'), ('value', 'MSFT'),
                                     ('value', 'cash')]),
                                index=pd.DatetimeIndex(["2019-01-01", "2019-01-03",
                                                        "2019-01-04", "2019-01-05"]))
        pd.testing.assert_frame_equal(expected, pf.history)
        self.assertEqual(str(pf),
                         """<Portfolio>amount       value               
             AAPL  MSFT  AAPL  MSFT     cash
2019-01-01    NaN   NaN   NaN   NaN  10000.0
2019-01-03   20.0 -25.0   NaN   NaN   5000.0
2019-01-04   10.0 -25.0   NaN   NaN   5000.0
2019-01-05   10.0   5.0  15.0  10.0   7030.0""")
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)
        self.assertEqual(7030 + 15 + 10, pf.value)
        self.assertEqual(25 / 7055, pf.leverage)

        pf.set_date("2019-01-06")
        pf.update_cash(-6830)
        pf.update('AAPL', -100, None)
        pf.update('MSFT', 50, None)
        self.assertEqual(200, pf.cash)
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)
        self.assertEqual(200 + -135 + 110, pf.value)
        self.assertEqual(245 / 175, pf.leverage)

        pf.set_date("2019-01-07")
        pf.update_cash(-200)
        pf.update('AAPL', 400, None)
        pf.update('MSFT', 50, None)
        pf.update_cash(-337.5)
        pf.update_value(lambda x: None)
        self.assertEqual(2, pf.leverage)

    def test_blotter(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            dividends_path=data_dir + '/dividends/', splits_path=data_dir + '/splits/',
            adjustments=('amount', 'ratio'), split_ratio_is_inverse=True,
            prices_index='date', dividends_index='exDate', splits_index='exDate', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter(loader, cash=200000)
        blotter.set_commission(0, 0.005, 1)
        blotter.set_slippage(0.005, 0.4)

        # t+0 and double purchase
        date = pd.Timestamp("2019-01-02", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open()
        blotter.set_price("open")
        blotter.order_target_percent('AAPL', 0.3)
        blotter.order_target_percent('AAPL', 0.3)
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', 0)
        blotter.market_close()

        # check transactions
        expected = pd.DataFrame([['AAPL', 384, 155.92, 157.09960, 1.92],
                                 ['AAPL', -384, 158.61, 157.41695, 1.92]],
                                columns=['symbol', 'amount', 'price',
                                         'fill_price', 'commission'],
                                index=[date, date])
        expected.index.name = 'date'
        pd.testing.assert_frame_equal(expected, blotter.get_transactions())

        value = 200000 - 157.0996 * 384 - 1.92 + 157.41695 * 384 - 1.92
        expected = pd.DataFrame([[0.0,  0.0,  value]],
                                columns=pd.MultiIndex.from_tuples(
                                    [('amount', 'AAPL'),
                                     ('value', 'AAPL'),
                                     ('value', 'cash')]),
                                index=[date])
        pd.testing.assert_frame_equal(expected, blotter.get_history_positions())

        # no data day
        date = pd.Timestamp("2019-01-10", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open()
        blotter.set_price("close")
        self.assertRaisesRegex(KeyError, '.*tradable.*', blotter.order_target_percent, 'MSFT', 0.5)
        # test curb
        blotter.daily_curb = 0.01
        blotter.order('AAPL', 1)
        self.assertEqual(0, blotter.positions['AAPL'])
        blotter.daily_curb = 0.033
        blotter.order('AAPL', 1)
        self.assertEqual(1, blotter.positions['AAPL'])
        blotter.order('AAPL', -1)
        blotter.market_close()
        blotter.daily_curb = None
        cash = blotter.portfolio.cash

        # overnight neutralized portfolio
        date = pd.Timestamp("2019-01-11", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open()
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', -0.5)
        blotter.order_target_percent('MSFT', 0.5)
        blotter.market_close()

        # test 01-11 div
        div = 0.46 + 0.11
        cash = cash + 651 * 152.52155 - 3.255 - 969 * 104.116 - 4.845 + div * 969
        self.assertAlmostEqual(cash, blotter.portfolio.cash)

        # test 01-14 splits, use 01-15 to test jump a day
        date = pd.Timestamp("2019-01-15", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open()
        blotter.market_close()
        self.assertEqual(969*15, blotter.positions['MSFT'])
        # test over night value
        print(blotter)
        expected = pd.DataFrame([[-651.0, 969*15, -651*156.94, 969*15 * 108.85, cash]],
                                columns=pd.MultiIndex.from_tuples(
                                    [('amount', 'AAPL'), ('amount', 'MSFT'),
                                     ('value', 'AAPL'), ('value', 'MSFT'),
                                     ('value', 'cash')]),
                                index=[date])
        pd.testing.assert_series_equal(expected.iloc[-1], blotter.get_history_positions().iloc[-1])
