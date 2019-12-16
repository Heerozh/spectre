[![Coverage Status](https://coveralls.io/repos/github/Heerozh/spectre/badge.svg?branch=master)](https://coveralls.io/github/Heerozh/spectre?branch=master)

Progress: 8/10  🔳🔳🔳🔳🔳🔳🔳🔳⬜⬜\
~~5/10: CUDA support~~\
~~6/10: Dividends/Splits~~\
~~7/10: Back-test architecture~~\
~~8/10: Event/Blotter/Trading Algorithm~~\
9/10: Optimization\
10/10: Back-test Analysis\
~~11/10: Factor Return Analysis~~

# ||spectre

spectre is a **GPU-accelerated Parallel** quantitative trading library, focused on **performance**.

  * Fast, really fast, see below [Benchmarks](#benchmarks)
  * Pure python code, using PyTorch for parallelize. 
  * Low CUDA memory usage
  * Compatible with `alphalens` and `pyfolio`
  * Python 3.7, pandas 0.25 recommended


## Status

In development.

## Installation

```bash
pip install --no-deps git+git://github.com/Heerozh/spectre.git
```

Dependencies: 

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install pyarrow pandas
```

## Benchmarks

My Machine：
- i9-7900X @ 3.30GHz, 20 Cores
- DDR4 3800MHz
- RTX 2080Ti Founders

Running on Quandl 5 years, 3196 Assets, total 3,637,344 bars.

|                |       spectre (CUDA)         |       spectre (CPU)        |       zipline         |
|----------------|------------------------------|----------------------------|-----------------------|
|SMA(100)        | 87.4 ms ± 745 µs (**34.1x**) | 2.68 s ± 36.1 ms (1.11x)   | 2.98 s ± 14.4 ms (1x) |
|EMA(50) win=200 | 144 ms ± 1.14 ms (**52.8x**) | 4.37 s ± 46.4 ms (1.74x)   | 7.6 s ± 15.4 ms (1x) |
|(MACD+RSI+STOCHF).rank.zscore | 375 ms ± 65.4 ms (**38.1x**) | 6.01 s ± 28.1 (2.38x)   | 14.3 s ± 277 ms (1x) |


* The CUDA memory used in the spectre benchmark is 1.3G, returned by cuda.max_memory_allocated().
* Benchmarks exclude the initial run (no copy data to VRAM, about saving 300ms).




## Quick Start

### Factor and FactorEngine

```python
from spectre import factors

loader = factors.CsvDirLoader(
    './tests/data/daily/',  # Contains fake dataset: AAPL.csv MSFT.csv
    ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
    prices_index='date', parse_dates=True,
)
engine = factors.FactorEngine(loader)
engine.to_cuda()
engine.add(factors.SMA(5), 'ma5')
engine.add(factors.OHLCV.close, 'close')
df = engine.run('2019-01-11', '2019-01-15')
df
```


|                         |         |        ma5|	 close|
|-------------------------|---------|-----------|---------|
|**date**                 |**asset**|           |	      |
|2019-01-14 00:00:00+00:00|     AAPL|    154.254|	153.69|
|                         |     MSFT|        NaN|	103.20|
|2019-01-15 00:00:00+00:00|     AAPL|    155.854|	157.00|
|                         |     MSFT|    104.202|	103.39|


### Factor Analysis

First use `ArrowLoader` to ingest the data in order to improve performance. \
`ArrowLoader` can ingest any `DataLoader` including `CsvDirLoader`.

```python
from spectre import factors
# WIKI_PRICES.zip can be found at: 
# https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?qopts.export=true&api_key=[yourapi_key]
factors.ArrowLoader.ingest(source=factors.QuandlLoader('WIKI_PRICES.zip'),
                           save_to='wiki_prices.feather')
```

Then use the ingested data:

```python
from spectre import factors
loader = factors.ArrowLoader('wiki_prices.feather')
engine = factors.FactorEngine(loader)
universe = factors.AverageDollarVolume(win=120).top(500)
engine.set_filter( universe )

f1 = -(factors.MA(5)-factors.MA(10)-factors.MA(30))
bb = -factors.BBANDS(win=5)
f2 = bb.filter(bb < 0.5)

engine.add( f1.rank(mask=universe).zscore(), 'ma_cross' )
engine.add( f2.rank(mask=universe).zscore(), 'bb' )

engine.to_cuda()
%time factor_data = engine.full_run("2013-01-02", "2018-01-19", periods=(1,5,10,)) 
```

<img src="https://github.com/Heerozh/spectre/raw/media/full_run.png" width="811" height="600">

#### Compatible with alphalens

The return value of `full_run` is compatible with `alphalens`:
```python
import alphalens as al
...
factor_data = engine.full_run("2013-01-02", "2018-01-19") 
clean_data = factor_data[['factor_name', 'Returns']].droplevel(0, axis=1)
al.tears.create_full_tear_sheet(clean_data)
```


### Back-testing

Back-testing uses FactorEngine's results as data, market events as triggers:

```python
from spectre import factors, trading
import pandas as pd


class MyAlg(trading.CustomAlgorithm):
    def initialize(self):
        # setup engine
        engine = self.get_factor_engine()
        engine.to_cuda()
        universe = factors.AverageDollarVolume(win=120).top(500)
        engine.set_filter( universe )

        # add your factors
        f1 = -(factors.MA(5)-factors.MA(10)-factors.MA(30))
        engine.add( f1.rank(mask=universe).zscore().to_weight(), 'ma_cross_weight' )

        # schedule rebalance before market close
        self.schedule_rebalance(trading.event.MarketClose(self.rebalance, offset_ns=-10000))

        # simulation parameters
        self.blotter.capital_base = 100000
        self.blotter.set_commission(percentage=0, per_share=0.005, minimum=1)
        # self.blotter.set_slippage(percentage=0, per_share=0.4)

    def rebalance(self, data: 'pd.DataFrame', history: 'pd.DataFrame'):
        # data is FactorEngine's results for current bar
        self.blotter.order_target_percent(data.index, data.ma_cross_weight)

        # closing asset position that are no longer in our universe.
        # if some asset is delisted then those order will fail, the asset will remain in the 
        # portfolio, the portfolio leverage will become a little higher.
        removes = self.blotter.portfolio.positions.keys() - set(data.index)
        self.blotter.order_target_percent(removes, [0] * len(removes))

        # record data for debugging / plotting
        self.record(aapl_weight=data.loc['AAPL', 'ma_cross_weight'],
                    aapl_price=self.blotter.get_price('AAPL'))

    def terminate(self, records):
        # plotting results
        spy = pd.read_csv('SPY.csv', index_col='date', parse_dates=True).close.pct_change()
        self.plot(benchmark=spy)

        # plotting the relationship between AAPL price and weight
        ax1 = records.aapl_price.plot()
        ax2 = ax1.twinx()
        records.aapl_weight.plot(ax=ax2, style='g-', secondary_y=True)
        
loader = factors.ArrowLoader('wiki_prices.feather')
%time ret, txn, pos = trading.run_backtest(loader, MyAlg, '2013-01-01', '2018-01-01')
```

The return value of `run_backtest` is compatible with `pyfolio`:
```python
import pyfolio as pf
pf.create_full_tear_sheet(ret, positions=pos, transactions=txn,
                          live_start_date='2017-01-03', round_trips=True)
```

BTW, the same strategy took 18 minutes to backtest in zipline.

## API

### Note

#### Differences with zipline:
* spectre's `QuandlLoader` using float32 datatype for GPU performance.
* For performance, spectre's arranges the data to be flattened by bars without time 
  information so may be differences, such as `Return(win=10)` may actually be more than 10 days
  if some stock not open trading in period.
* When encounter NaN, spectre returns NaN, zipline uses `nan*` so still give you a number.
* If an asset has no data on the day, spectre will filter it out, no matter what value you return.


#### Differences with common chart:
* The data is re-adjusted every day, so the factor you got, like the MA, will be different 
  from the stock chart software which only adjusted according to last day. 
  If you want adjusted by last day, use like 'AdjustedDataFactor(OHLCV.close)' as input data. 
  This will speeds up a lot because it only needs to be adjusted once, but brings Look-Ahead Bias.


### Factor lists

```python
# All technical factors passed comparison test with TA-Lib
Returns(inputs=[OHLCV.close])
LogReturns(inputs=[OHLCV.close])
SimpleMovingAverage = MA = SMA(win=5, inputs=[OHLCV.close])
VWAP(inputs=[OHLCV.close, OHLCV.volume])
ExponentialWeightedMovingAverage = EMA(win=5, inputs=[OHLCV.close])
AverageDollarVolume(win=5, inputs=[OHLCV.close, OHLCV.volume])
AnnualizedVolatility(win=20, inputs=[Returns(win=2), 252])
NormalizedBollingerBands = BBANDS(win=20, inputs=[OHLCV.close, 2])
MovingAverageConvergenceDivergenceSignal = MACD(12, 26, 9, inputs=[OHLCV.close])
TrueRange = TRANGE(inputs=[OHLCV.high, OHLCV.low, OHLCV.close])
RSI(win=14, inputs=[OHLCV.close])
FastStochasticOscillator = STOCHF(win=14, inputs=[OHLCV.high, OHLCV.low, OHLCV.close])

StandardDeviation = STDDEV(win=5, inputs=[OHLCV.close])
RollingHigh = MAX(win=5, inputs=[OHLCV.close])
RollingLow = MIN(win=5, inputs=[OHLCV.close])
```

### Factors Common Methods

```python
# Standardization
new_factor = factor.rank()
new_factor = factor.demean(groupby=dict)
new_factor = factor.zscore()

# Quick computation
new_factor = factor1 + factor1

# To filter (Comparison operator):
new_filter = factor1 < factor2
# Rank filter
new_filter = factor.top(n)
new_filter = factor.bottom(n)
```

## How to write your own factor

Inherit from `CustomFactor`, write `compute` function.

Use `torch.Tensor` to write parallel computing code.

### win = 1
When `win = 1`, the `inputs` data is tensor type, the first dimension of data is the 
asset, the second dimension is each bar price data. Note that the bars are not aligned 
across all assets, they are specific to each asset.

        +-----------------------------------+
        |            bar_t1    bar_t3       |
        |               |         |         |
        |               v         v         |
        | asset 1--> [[1.1, 1.2, 1.3, ...], |
        | asset 2-->  [  5,   6,   7, ...]] |
        +-----------------------------------+
Example of LogReturns:      
```python
class LogReturns(CustomFactor):
    inputs = [Returns(OHLCV.close)]
    win = 1

    def compute(self, change: torch.Tensor) -> torch.Tensor:
        return change.log()
```

### win > 1
If rolling windows is required(`win > 1`), all `inputs` data will be wrapped into 
`spectre.parallel.Rolling`

This is just an unfolded `tensor` data, but because the data is very large after unfolded, 
the rolling class automatically splits the data into multiple small portions. You need to use 
the `agg` method to operating `tensor`.
```python
class OvernightReturn(CustomFactor):
    inputs = [OHLCV.open, OHLCV.close]
    win = 2

    def compute(self, opens: parallel.Rolling, closes: parallel.Rolling) -> torch.Tensor:
        ret = opens.last() / closes.first() - 1
        return ret
```
Where `Rolling.first()` is just a helper method for `rolling.agg(lambda x: x[:, :, 0])`, 
where `x[:, :, 0]` return the first element of rolling window. The first dimension of `x` is the 
asset, the second dimension is each bar, and the third dimension is the price date containing 
the bar price and historical price with `win` length, and `Rolling.agg` runs on 
all the portions and combines them.

        +------------------win=3-------------------+
        |          history_t-2 curr_bar_value      |
        |              |          |                |
        |              v          v                |
        | asset 1-->[[[nan, nan, 1.1],  <--bar_t1  |
        |             [nan, 1.1, 1.2],  <--bar_t2  |
        |             [1.1, 1.2, 1.3]], <--bar_t3  |
        |                                          |
        | asset 2--> [[nan, nan,   5],  <--bar_t1  |
        |             [nan,   5,   6],  <--bar_t2  |
        |             [  5,   6,   7]]] <--bar_t3  |
        +------------------------------------------+

`Rolling.agg` can carry multiple `Rolling` objects, such as
```python
weighted_mean = lambda _close, _volume: (_close * _volume).sum(dim=2) / _volume.sum(dim=2)
close.agg(weighted_mean, volume)
```

### Using Pandas Series

For performance, spectre's tensor data is a flattened matrix which grouped by assets, 
there is no DataFrame's index information.
If you need index, or not familiar with pytorch, here is a another way:

```python
class YourFactor(CustomFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        # convert to pd.Series data
        pd_series = self._revert_to_series(data)
        # ...
        # convert back to grouped tensor
        return self._regroup(pd_series)
```
This method is completely non-parallel and inefficient, but easy to write.

# Copyright 
Copyright (C) 2019-2020, by Zhang Jianhao (heeroz@gmail.com), All rights reserved.

------------
> *A spectre is haunting Market — the spectre of capitalism.*
