[![Coverage Status](https://coveralls.io/repos/github/Heerozh/spectre/badge.svg?branch=master)](https://coveralls.io/github/Heerozh/spectre?branch=master)

Progress: 5/10  🔳🔳🔳🔳🔳⬜⬜⬜⬜⬜\
~~5/10: CUDA support~~\
6/10: Dividends/Splits\
7/10: Back-test architecture\
8/10: Portfolio\
9/10: Transaction\
10/10: Back-test\
~~11/10: Factor Return Analysis~~

# ||spectre

spectre is a **GPU-accelerated Parallel** quantitative trading library, focused on **performance**.

  * Fast, really fast, see below [Benchmarks](#benchmarks)
  * Pure python code, using PyTorch for parallelize. 
  * Low CUDA memory usage
  * Python 3.7, pandas 0.25 recommended

[Under construction]

## Status

In development.

## Installation

```bash
pip install git+git://github.com/Heerozh/spectre.git

```

Dependencies: 

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install pytables pandas
```

## Benchmarks

My Machine：
- i9-7900X @ 3.30GHz, 20 Cores
- DDR4 3800MHz
- RTX 2080Ti Founders

Running on Quandl 5 years, 3196 Assets, total 3,637,344 ticks.

|                |       spectre (CUDA)         |       spectre (CPU)        |       zipline         |
|----------------|------------------------------|----------------------------|-----------------------|
|SMA(100)        | 87.4 ms ± 745 µs (**34.1x**) | 2.68 s ± 36.1 ms (1.11x)   | 2.98 s ± 14.4 ms (1x) |
|EMA(50) win=200 | 144 ms ± 1.14 ms (**52.8x**) | 4.37 s ± 46.4 ms (1.74x)   | 7.6 s ± 15.4 ms (1x) |
|(MACD+RSI+STOCHF).rank.zscore | 375 ms ± 65.4 ms (**38.1x**) | 6.01 s ± 28.1 (2.38x)   | 14.3 s ± 277 ms (1x) |


* The CUDA memory used in the spectre benchmark is 1.3G, returned by cuda.max_memory_allocated().
* Benchmarks exclude the initial run (no copy data to VRAM, about saving 300ms).




## Quick Start

### Factor and FactorEngine

#### Factor
```python
from spectre import factors

loader = factors.CsvDirLoader(
    './tests/data/daily/',  # Contains fake dataset: AAPL.csv MSFT.csv
    ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
    index_col='date', parse_dates=True,
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
|2019-01-11 00:00:00+00:00|     AAPL|    154.254|	153.69|
|                         |     MSFT|        NaN|	103.20|
|2019-01-14 00:00:00+00:00|     AAPL|    155.854|	157.00|
|                         |     MSFT|    104.202|	103.39|
|2019-01-15 00:00:00+00:00|     AAPL|    156.932|	156.94|
|                         |     MSFT|    105.332|	108.85|

#### Factor Analysis

```python
from spectre import factors
loader = factors.QuandlLoader('WIKI_PRICES.zip')
engine = factors.FactorEngine(loader)
universe = factors.AverageDollarVolume(win=120).top(500)
engine.set_filter( universe )

f1 = -(factors.MA(5)-factors.MA(10)-factors.MA(30))
f2 = -factors.BBANDS(win=5)

engine.add( f1.rank(mask=universe).zscore(), 'ma_cross' )
engine.add( f2.filter(f2 > -0.5).rank(mask=universe).zscore(), 'bb' )

engine.to_cuda()
%time factor_data = engine.full_run("2013-01-02", "2018-01-19", periods=(1,5,10,)) 
```

<img src="https://github.com/Heerozh/spectre/raw/media/full_run.png">

#### Compatible with alphalens

The return value of `full_run` is compatible with alphalens:
```python
import alphalens as al
...
factor_data = engine.full_run("2013-01-02", "2018-01-19") 
clean_data = factor_data[['factor_name', 'Returns']].droplevel(0, axis=1)
al.tears.create_returns_tear_sheet(clean_data)
```


###  Portfolio and Backtesting

[Under construction]



## API

### Note

#### Differences with zipline:
* spectre's `QuandlLoader` using float32 datatype for GPU performance.
* For performance, spectre's arranges the data to be flattened by ticks without time 
  information so may be differences such as `Return(win=10)` may actually be more than 10 days/time.
* When encounter NaN, spectre returns NaN, zipline uses `nan*` so still give you a number.
* If an asset has no data on the day, spectre will filter it out, no matter what value you return.


#### Differences with common chart:
* The data is re-adjusted every day, so the factor you got, like the MA, will be different 
    from the stock chart software which only adjusted according to last day. 
    If you want adjusted by last day, use like 'AdjustedDataFactor(OHLCV.close)' as input data. 


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

You have to use `torch.Tensor` to write parallel code yourself.

If you can't, here is a simple way:

### Using Pandas Series
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
