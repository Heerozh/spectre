import unittest
import spectre
import pandas as pd
from os.path import dirname
from numpy import nan

data_dir = dirname(__file__) + '/data/'


class TestBlotter(unittest.TestCase):

    def test_portfolio(self):
        pf = spectre.trading.Portfolio()
        pf.set_datetime("2019-01-01")
        pf.update_cash(10000)

        pf.set_datetime("2019-01-03")
        pf.update('AAPL', 10, 9, 10)
        pf.update('MSFT', -25, 9, 25)
        pf.update_cash(-5000)
        pf.process_split('AAPL', 2.0, 3)

        pf.set_datetime("2019-01-04")
        pf.update('AAPL', -10, 12, 10)

        self.assertEqual(60, pf.positions['AAPL'].realized)

        pf.set_datetime("2019-01-05 23:00:00")
        pf.update('MSFT', 30, 7, 30)
        pf.update_cash(2000)

        self.assertEqual(0, pf.positions['MSFT'].realized)

        pf.process_dividends('AAPL', 2.0)
        pf.process_dividends('MSFT', 2.0)
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)

        position_shares = {k: pos.shares for k, pos in pf.positions.items()}
        self.assertEqual({'AAPL': 10, 'MSFT': 5}, position_shares)
        self.assertEqual(7030, pf.cash)
        expected = pd.DataFrame([[nan, nan, nan, nan, nan,  nan, 10000.],
                                 [05.,  8., 20., -25,  30, -225, 5000.],
                                 [-1.,  8., 10., -25, 120, -225, 5000.],
                                 [-3.,  6., 10.,   5,  15,   10, 7030.]],
                                columns=pd.MultiIndex.from_tuples(
                                    [('avg_px', 'AAPL'), ('avg_px', 'MSFT'),
                                     ('shares', 'AAPL'), ('shares', 'MSFT'),
                                     ('value', 'AAPL'), ('value', 'MSFT'),
                                     ('value', 'cash')]),
                                index=pd.DatetimeIndex(["2019-01-01", "2019-01-03",
                                                        "2019-01-04", "2019-01-05"]))
        expected.index.name = 'index'
        pd.testing.assert_frame_equal(expected, pf.history)
        self.assertEqual(str(pf),
                         """<Portfolio>avg_px      shares        value                
             AAPL MSFT   AAPL  MSFT   AAPL   MSFT     cash
index                                                     
2019-01-01    NaN  NaN    NaN   NaN    NaN    NaN  10000.0
2019-01-03    5.0  8.0   20.0 -25.0   30.0 -225.0   5000.0
2019-01-04   -1.0  8.0   10.0 -25.0  120.0 -225.0   5000.0
2019-01-05   -3.0  6.0   10.0   5.0   15.0   10.0   7030.0""")
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)
        self.assertEqual(7030 + 15 + 10, pf.value)
        self.assertEqual(25 / 7055, pf.leverage)

        pf.set_datetime("2019-01-06")
        pf.update_cash(-6830)
        pf.update('AAPL', -100, 1, 0)
        pf.update('MSFT', 50, 1, 0)
        self.assertEqual(200, pf.cash)
        pf.update_value(lambda x: x == 'AAPL' and 1.5 or 2)
        self.assertEqual(200 + -135 + 110, pf.value)
        self.assertEqual(245 / 175, pf.leverage)

        pf.set_datetime("2019-01-07")
        pf.update_cash(-200)
        pf.update('AAPL', 400, 1.5, 0)
        pf.update('MSFT', 50, 2, 0)
        pf.update_cash(-337.5)
        pf.update_value(lambda x: None)
        self.assertEqual(2, pf.leverage)

        # test realized
        pf = spectre.trading.Portfolio()
        pf.set_datetime("2019-01-03")
        self.assertEqual(0, pf.update('AAPL', 10, 10, 0))
        pos = pf.positions['AAPL']
        self.assertEqual(5, pf.update('AAPL', -5, 11, 0))
        self.assertEqual(-5, pf.update('AAPL', -5, 9, 0))
        self.assertEqual(0, pos.realized)

    def test_simulation_blotter(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            dividends_path=data_dir + '/dividends/', splits_path=data_dir + '/splits/',
            adjustments=('amount', 'ratio'),
            prices_index='date', dividends_index='exDate', splits_index='exDate', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter(loader, capital_base=200000)
        blotter.set_commission(0, 0.005, 1)
        blotter.set_slippage(0.005, 0.4)

        # t+0 and double purchase
        date = pd.Timestamp("2019-01-02", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open(self)
        blotter.set_price("open")
        blotter.order_target_percent('AAPL', 0.3)
        blotter.order_target_percent('AAPL', 0.3)
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', 0.0)
        blotter.market_close(self)
        blotter.update_portfolio_value()

        # check transactions
        rzd = (157.41695 - 157.09960) * 384 - 1.92 * 2
        expected = pd.DataFrame([['AAPL', 384, 155.92, 157.09960, 1.92, 0.0],
                                 ['AAPL', -384, 158.61, 157.41695, 1.92, rzd]],
                                columns=['symbol', 'amount', 'price',
                                         'fill_price', 'commission', 'realized'],
                                index=[date, date])
        expected.index.name = 'index'
        pd.testing.assert_frame_equal(expected, blotter.get_transactions())

        value = 200000 - 157.0996 * 384 - 1.92 + 157.41695 * 384 - 1.92
        expected = pd.DataFrame([[value]],
                                columns=pd.MultiIndex.from_tuples(
                                    [('value', 'cash')]),
                                index=[date])
        expected.index.name = 'index'
        pd.testing.assert_frame_equal(expected, blotter.get_historical_positions())

        # test on 01-09, 01-11 div
        date = pd.Timestamp("2019-01-09", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open(self)
        blotter.set_price("close")
        blotter.order('MSFT', 2)
        cash = blotter.portfolio.cash
        blotter.market_close(self)
        blotter.update_portfolio_value()
        div = 0.46 + 0.11
        self.assertAlmostEqual(cash + div * 2, blotter.portfolio.cash)

        # no data day
        date = pd.Timestamp("2019-01-10", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open(self)
        blotter.set_price("close")
        blotter.order_target_percent('MSFT', 0.5)
        # test curb
        blotter.daily_curb = 0.01
        blotter.order('AAPL', 1)
        self.assertEqual(0, blotter.portfolio.shares('AAPL'))
        self.assertEqual(False, 'AAPL' in blotter.positions)
        blotter.daily_curb = 0.033
        blotter.order('AAPL', 1)
        self.assertEqual(1, blotter.positions['AAPL'].shares)
        blotter.order('AAPL', -1)
        blotter.market_close(self)
        blotter.update_portfolio_value()
        blotter.daily_curb = None
        cash = blotter.portfolio.cash

        # overnight neutralized portfolio
        date = pd.Timestamp("2019-01-11", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open(self)
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', -0.5)
        blotter.order_target_percent('MSFT', 0.5)
        blotter.market_close(self)
        blotter.update_portfolio_value()

        # test on 01-11, 01-14 split
        self.assertEqual(int(969 / 15), blotter.positions['MSFT'].shares)
        cash = cash + 651 * 152.52155 - 3.255 - 967 * 104.116 - 4.835
        cash += (969 - int(969 / 15) * 15) * 103.2  # remaining split to cash
        self.assertAlmostEqual(cash, blotter.portfolio.cash)

        #
        date = pd.Timestamp("2019-01-15", tz='UTC')
        blotter.set_datetime(date)
        blotter.market_open(self)
        blotter.set_price("close")
        blotter.market_close(self)
        blotter.update_portfolio_value()

        # test over night value
        aapl_basis = 152.52155 + 3.255/-651
        msft_basis = (104.11600*967 + 4.835 + 106.93000*2+1-div*2)/969
        msft_basis *= 15
        expected = pd.DataFrame(
            [[aapl_basis, msft_basis, -651.0, int(969 / 15), -651 * 156.94, int(969 / 15) * 108.85,
              cash]],
            columns=pd.MultiIndex.from_tuples(
                [('avg_px', 'AAPL'), ('avg_px', 'MSFT'),
                 ('shares', 'AAPL'), ('shares', 'MSFT'),
                 ('value', 'AAPL'), ('value', 'MSFT'),
                 ('value', 'cash')]),
            index=[date])
        pd.testing.assert_series_equal(expected.iloc[-1],
                                       blotter.get_historical_positions().iloc[-1])

    def test_simulation_blotter_intraday(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/5mins/', prices_by_year=True, prices_index='Date',
            ohlcv=('Open', 'High', 'Low', 'Close', 'Volume'), parse_dates=True, )
        blotter = spectre.trading.SimulationBlotter(loader, capital_base=200000)
        blotter.set_commission(0, 0.005, 1)
        blotter.set_slippage(0.005, 0.4)

        # test not raise Out of market hours
        datetime1 = pd.Timestamp("2019-01-02 14:35:00", tz='UTC')
        blotter.set_datetime(datetime1)
        blotter.market_open(self)
        blotter.set_price("open")
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', 0.3)

        datetime2 = pd.Timestamp("2019-01-02 14:40:00", tz='UTC')
        blotter.set_datetime(datetime2)
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', 0.)
        blotter.set_price("close")
        blotter.market_close(self)
        blotter.update_portfolio_value()

        self.assertAlmostEqual(154.65, blotter.get_transactions().loc[datetime1].price)
        self.assertAlmostEqual(155.0, blotter.get_transactions().loc[datetime2].price)

        # test stop
        blotter.portfolio.set_stop_model(spectre.trading.TrailingStopModel(0.002, blotter.order))
        datetime3 = pd.Timestamp("2019-01-02 14:45:00", tz='UTC')
        blotter.set_datetime(datetime3)
        blotter.market_open(self)
        blotter.set_price("close")
        blotter.order_target_percent('AAPL', 1.)
        blotter.set_price("close")
        blotter.update_portfolio_value()
        blotter.portfolio.check_stop_trigger()

        datetime4 = pd.Timestamp("2019-01-02 14:50:00", tz='UTC')
        blotter.set_datetime(datetime4)
        blotter.set_price("close")
        blotter.update_portfolio_value()
        blotter.portfolio.check_stop_trigger()
        blotter.market_close(self)

        self.assertAlmostEqual(0, len(blotter.portfolio.positions))

    def test_trailing_stop(self):
        def init_model(ratio):
            m = spectre.trading.TrailingStopModel(ratio, True).new_tracker(9, False)
            m.update_price(8)
            m.update_price(7)
            m.update_price(6)
            return m

        model = init_model(-0.1)
        self.assertEqual(9*0.9, model.stop_price)
        model = init_model(0.1)
        self.assertEqual(6*1.1, model.stop_price)

        changes = [0.8, 0.9, 0.91, 1, 1.09, 1.1, 1.2]
        # test stop high bound
        model = init_model(0.1)
        result = []
        for p in changes:
            model.last_price = 6 * p
            result.append(model.check_trigger())
        self.assertListEqual([False, False, False, False, False, True, True], result)

        # test stop low bound
        model = init_model(-0.1)
        result = []
        for p in changes:
            model.last_price = 9 * p
            result.append(model.check_trigger())
        self.assertListEqual([True, True, False, False, False, False, False], result)

        # test DecayTrailingStopModel
        # test short stop loss
        pos = spectre.trading.Position(-10, 9, 0, None, None)
        model = spectre.trading.PnLDecayTrailingStopModel(0.1, 0.1, True).new_tracker(
            pos.last_price, False)
        model.tracking_position = pos
        self.assertEqual(9 * 1.1, model.stop_price)
        pos.last_price = 8
        model.update_price(pos.last_price)
        self.assertAlmostEqual(8 * (1 + 0.1 * 0.05 ** (1/9/0.1)), model.stop_price)
        pos.last_price = 7
        model.update_price(pos.last_price)
        pos.last_price = 6
        model.update_price(pos.last_price)

        model.last_price = 5.99
        self.assertFalse(model.check_trigger())
        model.last_price = 6.01
        self.assertTrue(model.check_trigger())

        # test long stop loss
        pos = spectre.trading.Position(10, 9, 0, None, None)
        model = spectre.trading.PnLDecayTrailingStopModel(-0.1, 0.1, True).new_tracker(
            pos.last_price, False)
        model.tracking_position = pos
        self.assertEqual(9 * 0.9, model.stop_price)
        pos.last_price = 8
        model.update_price(pos.last_price)
        self.assertTrue(model.check_trigger())

        # test long stop gain
        pos = spectre.trading.Position(10, 9, 0, None, None)
        model = spectre.trading.PnLDecayTrailingStopModel(0.1, -0.1, True).new_tracker(
            pos.last_price, False)
        model.tracking_position = pos
        self.assertEqual(9 * 1.1, model.stop_price)
        pos.last_price = 8
        model.update_price(pos.last_price)
        self.assertAlmostEqual(8 * (1 + 0.1 * 0.05 ** (1/9/0.1)), model.stop_price)
        pos.last_price = 7
        model.update_price(pos.last_price)
        pos.last_price = 6
        model.update_price(pos.last_price)

        model.last_price = 5.99
        self.assertFalse(model.check_trigger())
        model.last_price = 6.01
        self.assertTrue(model.check_trigger())

        # test time decay
        pos = spectre.trading.Position(10, 9, 0, pd.Timestamp('2014-01-02', tz='UTC'), None)
        model = spectre.trading.TimeDecayTrailingStopModel(-0.1, pd.Timedelta(days=5), True)\
            .new_tracker(pos.last_price, False)
        model.tracking_position = pos
        self.assertEqual(9 * 0.9, model.stop_price)
        pos.last_price = 8
        model.update_price(pos.last_price)
        self.assertTrue(model.check_trigger())

        pos.current_dt = pd.Timestamp('2014-01-05', tz='UTC')
        self.assertEqual(9 * (1 - 0.1 * 0.05 ** 0.6), model.stop_price)
