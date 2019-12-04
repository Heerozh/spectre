import unittest
import spectre
import pandas as pd
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestTradingAlgorithm(unittest.TestCase):

    def test_simulation_event_manager(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )

        class TestEventReceiver(spectre.trading.EventReceiver):
            _data = None
            _test = []
            _seq = 0

            def create_data(self, start, end):
                engine = spectre.factors.FactorEngine(loader)
                f = spectre.factors.MA(5)
                engine.add(f, 'f')
                return engine.run(start, end)

            def initialize(self):
                self.schedule(spectre.trading.event.EveryBarData(
                    lambda: self.test_every_bar(self._data)
                ))
                self.schedule(spectre.trading.event.MarketOpen(self.test_before_open, -1000))
                self.schedule(spectre.trading.event.MarketOpen(self.test_open, 0))
                self.schedule(spectre.trading.event.MarketClose(self.test_before_close, -1000))
                self.schedule(spectre.trading.event.MarketClose(self.test_close, 1000))

            def on_subscribe(self):
                pass

            def test_every_bar(self, data):
                self._seq += 1
                assert self._seq == 3

                today = data.index.get_level_values(0)[-1]
                self._test.append(data.index.get_level_values(0)[-1])
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

        rcv = TestEventReceiver()

        evt_mgr = spectre.trading.SimulationEventManager()
        evt_mgr.subscribe(rcv)
        evt_mgr.run("2019-01-01", "2019-01-15")

        self.assertEquals(rcv._test[0], pd.Timestamp("2019-01-03", tz='UTC'))
        self.assertEquals(rcv._test[1], pd.Timestamp("2019-01-04", tz='UTC'))
        self.assertEquals(rcv._test[-1], pd.Timestamp("2019-01-11", tz='UTC'))

    def test_one_engine_algorithm(self):
        class OneEngineAlg(spectre.trading.CustomAlgorithm):
            def initialize(self):
                engine = self.get_factor_engine()
                ma5 = spectre.factors.MA(5)
                engine.add(ma5, 'ma5')
                engine.set_filter(ma5.top(5))

                self.schedule(spectre.trading.event.EveryBarData(self.rebalance))

                self.blotter.set_commission()
                self.blotter.set_slippage()

            def rebalance(self, data):
                weight = data.ma5 / data.ma5.sum()
                self.order_to_percent(data.index, weight)

            def analyze(self):
                pass

        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', 'AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        blotter = spectre.trading.SimulationBlotter()
        evt_mgr = spectre.trading.SimulationEventManager()
        alg = OneEngineAlg(blotter, loader)
        evt_mgr.subscribe(alg)
        evt_mgr.run()

    def test_two_engine_algorithm(self):
        class TwoEngineAlg(spectre.trading.CustomAlgorithm):
            def initialize(self, blotter):
                engine_5mins = self.get_factor_engine('5mins')
                engine_daily = self.get_factor_engine('daily')

                ma5 = spectre.factors.MA(5)
                ma20 = spectre.factors.MA(20)
                engine_daily.add(ma5, 'ma5')
                engine_5mins.add(ma20, 'ma20')

                # 现在的问题是如何结合多种不同时间线的数据源， factor是个追求效率没有算时间的功能，所以还是多个factor?
                self.schedule(spectre.trading.event.EveryBarData(self.rebalance))

                self.blotter.set_commission()
                self.blotter.set_slippage()

            def rebalance(self, data):
                mask = data['5mins'].ma20 > data['daily'].ma5
                assets = data['5mins'][mask].index
                weight = assets.ma20 / assets.ma20.sum()
                self.order_to_percent(assets, weight)

            def analyze(self):
                pass

