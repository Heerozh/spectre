"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""

from spectre import factors, trading
from spectre.data import YahooDownloader, ArrowLoader
import pandas as pd


class SmartBeta(trading.CustomAlgorithm):
    def initialize(self):
        # setup engine
        engine = self.get_factor_engine()
        engine.to_cuda()

        universe = factors.AverageDollarVolume(win=120).top(500)
        engine.set_filter(universe)

        # SP500 factor
        sp500 = factors.AverageDollarVolume(win=63)
        # our alpha is put more weight on NVDA! StaticAssets return True(1) on NVDA
        # and False(0) on others
        alpha = sp500 * (factors.StaticAssets({'NVDA'})*5 + 1)
        engine.add(alpha.to_weight(demean=False), 'weight')

        # schedule rebalance before market close
        self.schedule_rebalance(trading.event.MarketClose(self.rebalance, offset_ns=-10000))

        # simulation parameters
        self.blotter.capital_base = 1000000
        self.blotter.set_commission(percentage=0, per_share=0.005, minimum=1)

    def rebalance(self, data: pd.DataFrame, history: pd.DataFrame):
        self.blotter.batch_order_target_percent(data.index, data.weight)
        # closing asset position that are no longer in our universe.
        removes = self.blotter.portfolio.positions.keys() - set(data.index)
        self.blotter.batch_order_target_percent(removes, [0] * len(removes))

    def terminate(self, records: 'pd.DataFrame'):
        # plotting results
        self.plot(benchmark='SPY')


if __name__ == '__main__':
    import plotly.io as pio
    pio.renderers.default = "browser"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", help="download yahoo data")
    args = parser.parse_args()

    if args.download:
        YahooDownloader.ingest(
            start_date="2001", save_to="./yahoo",
            symbols=None, skip_exists=True)

    loader = ArrowLoader('./yahoo/yahoo.feather')
    results = trading.run_backtest(loader, SmartBeta, '2013-01-01', '2018-01-01')
    print(results.transactions)
