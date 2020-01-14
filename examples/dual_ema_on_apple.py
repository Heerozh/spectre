"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""

from spectre import factors, trading
from spectre.data import YahooDownloader, ArrowLoader
import pandas as pd


class AppleDualEma(trading.CustomAlgorithm):
    invested = False
    asset = 'AAPL'

    def initialize(self):
        # setup engine
        engine = self.get_factor_engine()
        engine.to_cuda()

        universe = factors.StaticAssets({self.asset})
        engine.set_filter(universe)

        # add your factors
        fast_ema = factors.EMA(20)
        slow_ema = factors.EMA(40)
        engine.add(fast_ema, 'fast_ema')
        engine.add(slow_ema, 'slow_ema')
        engine.add(fast_ema > slow_ema, 'buy_signal')
        engine.add(fast_ema < slow_ema, 'sell_signal')
        engine.add(factors.OHLCV.close, 'price')

        # schedule rebalance before market close
        self.schedule_rebalance(trading.event.MarketClose(self.rebalance, offset_ns=-10000))

        # simulation parameters
        self.blotter.capital_base = 10000
        self.blotter.set_commission(percentage=0, per_share=0.005, minimum=1)

    def rebalance(self, data: pd.DataFrame, history: pd.DataFrame):
        asset_data = data.loc[self.asset]
        buy, sell = False, False
        if asset_data.buy_signal and not self.invested:
            self.blotter.order(self.asset, 100)
            self.invested = True
            buy = True
        elif asset_data.sell_signal and self.invested:
            self.blotter.order(self.asset, -100)
            self.invested = False
            sell = True

        self.record(AAPL=asset_data.price,
                    short_ema=asset_data.fast_ema,
                    long_ema=asset_data.slow_ema,
                    buy=buy,
                    sell=sell)

    def terminate(self, records: 'pd.DataFrame'):
        # plotting results
        self.plot(benchmark='SPY')

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel('Price (USD)')

        records[['AAPL', 'short_ema', 'long_ema']].plot(ax=ax1)

        ax1.plot(
            records.index[records.buy],
            records.loc[records.buy, 'long_ema'],
            '^',
            markersize=10,
            color='m',
        )
        ax1.plot(
            records.index[records.sell],
            records.loc[records.sell, 'short_ema'],
            'v',
            markersize=10,
            color='k',
        )
        plt.legend(loc=0)
        plt.gcf().set_size_inches(18, 8)

        plt.show()


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
    results = trading.run_backtest(loader, AppleDualEma, '2013-01-01', '2018-01-01')
    print(results.transactions)
