import unittest
import spectre
import pandas as pd
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestTradingAlgorithm(unittest.TestCase):

    def test_simulation_event_manager(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )

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
                assert self._seq == 3

                today = data.index.get_level_values(0)[-1]
                self._bar_dates.append(data.index.get_level_values(0)[-1])
                if today > pd.Timestamp("2019-01-10", tz='UTC'):
                    self.stop_event_manager()

            def test_before_open(self):
                self._seq += 1
                assert self._seq == 1

            def test_open(self):
                self._seq += 1
                assert self._seq == 2

            def test_before_close(self):
                self._seq += 1
                assert self._seq == 4

            def test_close(self):
                self._seq += 1
                assert self._seq == 5
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

                self.schedule_rebalance(spectre.trading.event.EveryBarData(self.rebalance))

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
            prices_index='date', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter(loader)
        evt_mgr = spectre.trading.SimulationEventManager()
        alg = OneEngineAlg(blotter, main=loader)
        evt_mgr.subscribe(alg)
        evt_mgr.subscribe(blotter)
        evt_mgr.run("2019-01-11", "2019-01-15")

        print(blotter)
        # 测试几点，数据是否正确，然后延后factor值是否对，用excel算一下，测试分红等

    def test_two_engine_algorithm(self):
        class TwoEngineAlg(spectre.trading.CustomAlgorithm):
            def initialize(self):
                engine_main = self.get_factor_engine('main')
                engine_test = self.get_factor_engine('test')

                ma5 = spectre.factors.MA(5)
                ma20 = spectre.factors.MA(20)
                engine_main.add(ma5, 'ma5')
                engine_test.add(ma20, 'ma20')

                self.schedule_rebalance(spectre.trading.event.EveryBarData(self.rebalance))

                self.blotter.set_commission(0, 0.005, 1)
                self.blotter.set_slippage(0, 0.4)

            def rebalance(self, data, history):
                mask = data['test'].ma20 > data['main'].ma5
                masked_test = data['test'][mask]
                assets = masked_test.index
                weights = masked_test.ma20 / masked_test.ma20.sum()
                for asset, weight in zip(assets, weights):
                    self.blotter.order_target_percent(asset, weight)

            def terminate(self):
                pass

        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter(loader)
        evt_mgr = spectre.trading.SimulationEventManager()
        alg = TwoEngineAlg(blotter, main=loader, test=loader)
        evt_mgr.subscribe(alg)
        evt_mgr.subscribe(blotter)
        evt_mgr.run("2019-01-10", "2019-01-15")
        print(blotter)

        evt_mgr.run("2019-01-10", "2019-01-15")
        print(blotter)

