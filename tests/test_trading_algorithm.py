import unittest
import spectre
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestTradingAlgorithm(unittest.TestCase):

    def test_one_engine_algorithm(self):
        class OneEngineAlg(spectre.trading.CustomAlgorithm):
            def initialize(self):
                engine = self.get_factor_engine()
                ma5 = spectre.factors.MA(5)
                engine.add(ma5, 'ma5')
                engine.set_filter(ma5.top(5))

                self.schedule(spectre.trading.event.EveryTick(self.rebalance))

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

                self.blotter.set_commission()
                self.blotter.set_slippage()

            def handle_data(self, data):
                mask = data['5mins'].ma20 > data['daily'].ma5
                assets = data['5mins'][mask].index
                weight = assets.ma20 / assets.ma20.sum()
                self.order_to_percent(assets, weight)

            def analyze(self):
                pass

