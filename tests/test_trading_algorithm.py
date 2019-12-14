import unittest
import spectre
import pandas as pd
from os.path import dirname
from numpy import nan
from numpy.testing import assert_array_equal

data_dir = dirname(__file__) + '/data/'


class TestTradingAlgorithm(unittest.TestCase):

    def test_simulation_event_manager(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        parent = self

        class MockTestAlg(spectre.trading.CustomAlgorithm):
            _data = None
            # test the elapsed time is correct
            _bar_dates = []
            # test if the sequence of events per day is correct
            _seq = 0

            def __init__(self):
                self.blotter = spectre.trading.SimulationBlotter(loader)

            def clear(self):
                pass

            def run_engine(self, start, end):
                engine = spectre.factors.FactorEngine(loader)
                f = spectre.factors.MA(5)
                engine.add(f, 'f')
                return engine.run(start, end)

            def initialize(self):
                self.schedule(spectre.trading.event.EveryBarData(
                    lambda x: self.test_every_bar(self._data)
                ))
                self.schedule(spectre.trading.event.MarketOpen(self.test_before_open, -1000))
                self.schedule(spectre.trading.event.MarketOpen(self.test_open, 0))
                self.schedule(spectre.trading.event.MarketClose(self.test_before_close, -1000))
                self.schedule(spectre.trading.event.MarketClose(self.test_close, 1000))

            def _run_engine(self, source):
                self._data = self.run_engine(None, None)

            def on_run(self):
                self.schedule(spectre.trading.event.EveryBarData(
                   self._run_engine
                ))
                self.initialize()

            def test_every_bar(self, data):
                self._seq += 1
                parent.assertEqual(1, self._seq)

                today = data.index.get_level_values(0)[-1]
                self._bar_dates.append(data.index.get_level_values(0)[-1])
                if today > pd.Timestamp("2019-01-10", tz='UTC'):
                    self.stop_event_manager()

            def test_before_open(self, source):
                self._seq += 1
                parent.assertEqual(2, self._seq)

            def test_open(self, source):
                self._seq += 1
                parent.assertEqual(3, self._seq)

            def test_before_close(self, source):
                self._seq += 1
                parent.assertEqual(4, self._seq)

            def test_close(self, source):
                self._seq += 1
                parent.assertEqual(5, self._seq)
                self._seq = 0

        rcv = MockTestAlg()

        evt_mgr = spectre.trading.SimulationEventManager()
        evt_mgr.subscribe(rcv)
        evt_mgr.run("2019-01-01", "2019-01-15")

        self.assertEqual(rcv._bar_dates[0], pd.Timestamp("2019-01-03", tz='UTC'))
        self.assertEqual(rcv._bar_dates[1], pd.Timestamp("2019-01-04", tz='UTC'))
        # test stop event is correct
        self.assertEqual(rcv._bar_dates[-1], pd.Timestamp("2019-01-11", tz='UTC'))

    def test_one_engine_algorithm(self):
        class OneEngineAlg(spectre.trading.CustomAlgorithm):
            def initialize(self):
                engine = self.get_factor_engine()
                ma5 = spectre.factors.MA(5)
                engine.add(ma5, 'ma5')
                engine.set_filter(ma5.top(5))

                self.schedule_rebalance(spectre.trading.event.MarketOpen(self.rebalance))

                self.blotter.long_ony = True
                self.blotter.set_commission(0, 0, 0)
                self.blotter.set_slippage(0, 0)

            def rebalance(self, data, history):
                weights = data.ma5 / data.ma5.sum()
                assets = data.index
                for asset, weight in zip(assets, weights):
                    self.blotter.order_target_percent(asset, weight)

            def terminate(self):
                pass

        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            dividends_path=data_dir + '/dividends/', splits_path=data_dir + '/splits/',
            adjustments=('amount', 'ratio'),
            prices_index='date', dividends_index='exDate', splits_index='exDate', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter(loader)
        evt_mgr = spectre.trading.SimulationEventManager()
        alg = OneEngineAlg(blotter, main=loader)
        evt_mgr.subscribe(alg)
        evt_mgr.subscribe(blotter)
        evt_mgr.run("2019-01-11", "2019-01-15")

        # test factor delay, order correct
        # --- day1 ---
        aapl_amount1 = int(1e5/155.19)
        aapl_cost1 = aapl_amount1*155.19
        cash1 = 1e5-aapl_cost1
        aapl_value_eod1 = aapl_amount1 * 157
        # --- day2 ---
        aapl_weight2 = 155.854 / (155.854+1268.466)
        msft_weight2 = 1268.466 / (155.854+1268.466)
        value_bod2 = aapl_amount1 * 150.81 + cash1
        aapl_amount2 = aapl_weight2 * value_bod2 / 150.81
        aapl_amount2 = aapl_amount1 + int(aapl_amount2 - aapl_amount1) + 0.0
        aapl_value2 = aapl_amount2 * 156.94
        msft_amount2 = int(msft_weight2 * value_bod2 / 103.19)
        msft_value2 = msft_amount2 * 108.85
        cash2 = 1e5-aapl_cost1 + (aapl_amount1-aapl_amount2) * 150.81 - msft_amount2 * 103.19
        expected = pd.DataFrame([[aapl_amount1,  nan,  aapl_value_eod1, nan, cash1],
                                 [aapl_amount2,  msft_amount2, aapl_value2, msft_value2, cash2]],
                                columns=pd.MultiIndex.from_tuples(
                                    [('amount', 'AAPL'), ('amount', 'MSFT'),
                                     ('value', 'AAPL'), ('value', 'MSFT'),
                                     ('value', 'cash')]),
                                index=[pd.Timestamp("2019-01-14", tz='UTC'),
                                       pd.Timestamp("2019-01-15", tz='UTC')])
        pd.testing.assert_frame_equal(expected, blotter.get_history_positions())

    def test_two_engine_algorithm(self):
        class TwoEngineAlg(spectre.trading.CustomAlgorithm):
            def initialize(self):
                engine_main = self.get_factor_engine('main')
                engine_test = self.get_factor_engine('test')

                ma5 = spectre.factors.MA(5)
                ma4 = spectre.factors.MA(4)
                engine_main.add(ma5, 'ma5')
                engine_test.add(ma4, 'ma4')

                self.schedule_rebalance(spectre.trading.event.MarketClose(
                    self.rebalance, offset_ns=-10000))

                self.blotter.set_commission(0, 0.005, 1)
                self.blotter.set_slippage(0, 0.4)

            def rebalance(self, data, history):
                mask = data['test'].ma4 > data['main'].ma5
                masked_test = data['test'][mask]
                assets = masked_test.index
                weights = masked_test.ma4 / masked_test.ma4.sum()
                for asset, weight in zip(assets, weights):
                    self.blotter.order_target_percent(asset, weight)

            def terminate(self):
                pass

        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            dividends_path=data_dir + '/dividends/', splits_path=data_dir + '/splits/',
            adjustments=('amount', 'ratio'),
            prices_index='date', dividends_index='exDate', splits_index='exDate', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter(loader)
        evt_mgr = spectre.trading.SimulationEventManager()
        alg = TwoEngineAlg(blotter, main=loader, test=loader)
        evt_mgr.subscribe(alg)
        evt_mgr.subscribe(blotter)

        evt_mgr.run("2019-01-10", "2019-01-15")
        first_run = str(blotter)
        evt_mgr.run("2019-01-10", "2019-01-15")

        # test two result should be the same.
        self.assertEqual(first_run, str(blotter))
        assert_array_equal(['AAPL', 'AAPL', 'MSFT', 'AAPL'],
                           blotter.get_transactions().symbol.values)

    def test_with_zipline(self):
        # todo
        pass
