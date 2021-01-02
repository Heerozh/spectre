[![Coverage Status](https://coveralls.io/repos/github/Heerozh/spectre/badge.svg?branch=master)](https://coveralls.io/github/Heerozh/spectre?branch=master)

# ||spectre

spectre is a **GPU-accelerated Parallel** quantitative trading library, focused on **performance**.

  * Fast GPU Factor Engine, see below [Benchmarks](#benchmarks)
  * Pure python code, based on PyTorch, so it can integrate DL model very smoothly.
  * Compatible with `alphalens` and `pyfolio`

Python 3.7+, PyTorch 1.3+, Pandas 1.0+ recommended


# Installation

```bash
pip install --no-deps git+git://github.com/Heerozh/spectre.git
```

Dependencies:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install pyarrow pandas tqdm plotly requests
```

# Benchmarks

My Machine：
- i9-7900X @ 3.30GHz, 20 Cores
- DDR4 3800MHz
- 3090: GIGABYTE GeForce RTX 3090 GAMING OC 24G
- 2080Ti: RTX 2080Ti Founders

Running on Quandl 5 years, 3196 Assets, total 3,637,344 bars.

|                |     spectre (CUDA/3090)       |     spectre (CUDA/2080Ti)     |       spectre (CPU)        |   zipline.pipeline    |
|----------------|-------------------------------|-------------------------------|----------------------------|-----------------------|
|SMA(100)        | 87.9 ms ± 3.35 ms (**33.9x**) | 144 ms ± 974 µs (**20.7x**)   | 2.68 s ± 36.1 ms (1.11x)   | 2.98 s ± 14.4 ms (1x) |
|EMA(50) win=229 | 166 ms ± 3.25 ms (**50.5x**)  | 270 ms ± 1.89 ms (**31.0x**)  | 4.37 s ± 46.4 ms (1.74x)   | 8.38 s ± 56.8 ms (1x) |
|(MACD+RSI+STOCHF).rank.zscore | 184 ms ± 7.83 ms (**77.7x**) | 282 ms ± 1.33 ms (**50.7x**) | 6.01 s ± 28.1 (2.38x)   | 14.3 s ± 277 ms (1x) |

* The CUDA memory used in the spectre benchmark is 1.8G, returned by cuda.max_memory_allocated().
* Benchmarks excluded the initial run (no copy data to VRAM, about saving 300ms).

# Quick Start

## DataLoader

First of all is data, you can use [CsvDirLoader](#csvdirloader) read your csv files.

spectre also has built-in Yahoo downloader, `symbols=None` will download all SP500 components.

```python
from spectre.data import YahooDownloader
YahooDownloader.ingest(start_date="2001", save_to="./prices/yahoo", symbols=None, skip_exists=True)
```

You can use `spectre.data.ArrowLoader('./prices/yahoo/yahoo.feather')` load those data now.

## Factor and FactorEngine

```python
from spectre import factors
from spectre.data import ArrowLoader
loader = ArrowLoader('./prices/yahoo/yahoo.feather')
engine = factors.FactorEngine(loader)
engine.to_cuda()
engine.add(factors.SMA(5), 'ma5')
engine.add(factors.OHLCV.close, 'close')
df = engine.run('2019-01-11', '2019-01-15')
df
```


|                         |         |        ma5|	   close|
|-------------------------|---------|-----------|-----------|
|**date**                 |**asset**|           |	        |
|2019-01-14 00:00:00+00:00|        A|  68.842003|  70.379997|
|                         |     AAPL| 151.615997| 152.289993|
|                         |      ABC|  75.835999|  76.559998|
|                         |      ABT|  69.056000|  69.330002|
|                         |     ADBE| 234.537994| 237.550003|
|                      ...|      ...|        ...|	     ...|
|2019-01-15 00:00:00+00:00|      XYL|  68.322006|  69.160004|
|                         |      YUM|  91.010002|  90.000000|
|                         |      ZBH| 102.932007| 102.690002|
|                         |     ZION|  43.760002|  44.320000|
|                         |      ZTS|  85.846001|  84.500000|


## Factor Analysis


```python
from spectre import factors
import math

risk_free_rate = 0.04 / 252
excess_logret = factors.LogReturns() - math.log(1 + risk_free_rate)
universe = factors.AverageDollarVolume(win=120).top(100)

# Barra MOMENTUM
ema126 = factors.EMA(half_life=126, inputs=[excess_logret])
rstr = ema126.shift(11).sum(252)
MOMENTUM = rstr

# Barra Volatility
ema42 = factors.EMA(half_life=42, inputs=[excess_logret])
dastd = factors.STDDEV(252, inputs=[ema42])
VOLATILITY = dastd

# run engine
from spectre.data import ArrowLoader
loader = ArrowLoader('./prices/yahoo/yahoo.feather')
engine = factors.FactorEngine(loader)

engine.set_filter( universe )
engine.add( MOMENTUM, 'MOMENTUM' )
engine.add( VOLATILITY, 'VOLATILITY' )

engine.to_cuda()
%time factor_data, mean_return = engine.full_run("2013-01-02", "2018-01-19", periods=(1,5,10,))
```

<img src="https://github.com/Heerozh/spectre/raw/media/full_run.png" width="800" height="600">

### Diagram

You can also view your factor structure graphically:

```python
factors.BBANDS(win=5).normalized().rank().zscore().show_graph()
```

<img src="https://github.com/Heerozh/spectre/raw/media/factor_diagram.png" width="800" height="360">

The thickness of the line represents the length of the Rolling Window, kind of like "bandwidth".

If `engine.to_cuda(enable_stream=True)`, the calculation of the branches will be performed
simultaneously, but the VRAM usage will increase proportionally. 

### Compatible with alphalens

The return value of `full_run` is compatible with `alphalens`:
```python
import alphalens as al
...
factor_data, _ = engine.full_run("2013-01-02", "2018-01-19")
clean_data = factor_data[['{factor_name}', 'Returns']].droplevel(0, axis=1)
al.tears.create_full_tear_sheet(clean_data)
```


## Back-testing

Back-testing uses FactorEngine's results as data, market event as triggers.

You can find other examples in the `./examples` directory.

```python
from spectre import factors, trading
from spectre.data import ArrowLoader
import pandas as pd, math


class MyAlg(trading.CustomAlgorithm):
    def initialize(self):
        # your factors
        risk_free_rate = 0.04 / 252
        excess_logret = factors.LogReturns() - math.log(1 + risk_free_rate)
        universe = factors.AverageDollarVolume(win=120).top(100)

        # Barra MOMENTUM Risk Factor
        ema126 = factors.EMA(half_life=126, inputs=[excess_logret])
        rstr = ema126.shift(11).sum(252)
        MOMENTUM = rstr.zscore(mask=universe)

        # Barra Volatility Risk Factor
        ema42 = factors.EMA(half_life=42, inputs=[excess_logret])
        dastd = factors.STDDEV(252, inputs=[ema42])
        VOLATILITY = dastd.zscore(mask=universe)

        # setup engine
        engine = self.get_factor_engine()
        engine.to_cuda()
        engine.set_filter( universe )
        engine.add( (MOMENTUM + VOLATILITY).to_weight(), 'alpha_weight' )

        # schedule rebalance before market close
        self.schedule_rebalance(trading.event.MarketClose(self.rebalance, offset_ns=-10000))

        # simulation parameters
        self.blotter.capital_base = 1000000
        self.blotter.set_commission(percentage=0, per_share=0.005, minimum=1)
        # self.blotter.set_slippage(percentage=0, per_share=0.4)

    def rebalance(self, data: 'pd.DataFrame', history: 'pd.DataFrame'):
        data = data.fillna(0)
        self.blotter.batch_order_target_percent(data.index, data.alpha_weight)

        # closing asset position that are no longer in our universe.
        removes = self.blotter.portfolio.positions.keys() - set(data.index)
        self.blotter.batch_order_target_percent(removes, [0] * len(removes))

        # record data for debugging / plotting
        self.record(aapl_weight=data.loc['AAPL', 'alpha_weight'],
                    aapl_price=self.blotter.get_price('AAPL'))

    def terminate(self, records: 'pd.DataFrame'):
        # plotting results
        self.plot(benchmark='SPY')

        # plotting the relationship between AAPL price and weight
        ax1 = records.aapl_price.plot()
        ax2 = ax1.twinx()
        records.aapl_weight.plot(ax=ax2, style='g-')

loader = ArrowLoader('./prices/yahoo/yahoo.feather')
%time results = trading.run_backtest(loader, MyAlg, '2014-01-01', '2019-01-01')
```

<img src="https://github.com/Heerozh/spectre/raw/media/backtest.png" width="800" height="630">

It awful but you get the idea.

The return value of `run_backtest` is compatible with `pyfolio`:
```python
import pyfolio as pf
pf.create_full_tear_sheet(results.returns, positions=results.positions.value, transactions=results.transactions,
                          live_start_date='2017-01-03')
```

# API

## Note

### Differences to zipline:
* In order to GPU optimize, the `CustomFactor.compute` function calculates the results of all bars 
  at once, so you need to be careful to prevent Look-Ahead Bias, because the inputs are not just 
  historical data. Also using `engine.test_lookahead_bias` do some tests.
* spectre's normally using float32 data type for GPU performance.
* spectre FactorEngine arranges data by bars, so `Return(win=10)` means 10 bars return, may 
  actually be more than 10 days if some assets not open trading in period. You can change this 
  behavior by aligning data: filling missing bars with NaNs in your DataLoader, please refer to the 
  `align_by_time` parameter of `CsvDirLoader`.


### Differences to common chart:
* If there is adjustments data, the prices is re-adjusted every day, so the factor you got, like MA, 
  will be different from the stock chart software which only adjusted according to last day.
  If you want adjusted by last day, use like 'AdjustedColumnDataFactor(OHLCV.close)' as input data.
  This will speeds up a lot because it only needs to be adjusted once, but brings Look-Ahead Bias.
* Factors that uses the close data will be delayed by 1 bar.
* spectre's `EMA` uses the algorithm same as `zipline` and `Dataframe.ewm(span=...)`, when `span` is 
  greater than 100, it will be slightly different from common EMA.
* spectre's `RSI` uses the algorithm same as `zipline`, for consistency in benchmarks. 


## Factors

## Built-in Technical Indicator Factors list

```python
Returns(inputs=[OHLCV.close])
LogReturns(inputs=[OHLCV.close])
SimpleMovingAverage = MA = SMA(win=5, inputs=[OHLCV.close])
VWAP(inputs=[OHLCV.close, OHLCV.volume])
ExponentialWeightedMovingAverage = EMA(span=5, inputs=[OHLCV.close])
AverageDollarVolume(win=5, inputs=[OHLCV.close, OHLCV.volume])
AnnualizedVolatility(win=20, inputs=[Returns(win=2), 252])
BollingerBands = BBANDS(win=20, inputs=[OHLCV.close, 2])
MovingAverageConvergenceDivergenceSignal = MACD(12, 26, 9, inputs=[OHLCV.close])
TrueRange = TRANGE(inputs=[OHLCV.high, OHLCV.low, OHLCV.close])
RSI(win=14, inputs=[OHLCV.close])
FastStochasticOscillator = STOCHF(win=14, inputs=[OHLCV.high, OHLCV.low, OHLCV.close])

StandardDeviation = STDDEV(win=5, inputs=[OHLCV.close])
RollingHigh = MAX(win=5, inputs=[OHLCV.close])
RollingLow = MIN(win=5, inputs=[OHLCV.close])
```

## Factors Common Methods

```python
# Standardization
new_factor = factor.rank(mask=filter)   
new_factor = factor.demean(mask=filter, groupby: 'dict or column_name'=None)
new_factor = factor.zscore(mask=filter)
new_factor = factor.to_weight(mask=filter, demean=True)  # return a weight that sum(abs(weight)) = 1

# Quick computation
new_factor = factor1 + factor1
new_factor = factor.abs()
new_factor = factor.sum()

# To filter (Comparison operator):
new_filter = (factor1 < factor2) | (factor1 > 0)
new_filter[n_features] = factor.one_hot()  # one-hot encoding
new_filter = factor.any(win=5)
new_filter = factor.all(win=5)
# Rank filter
new_filter = factor.top(n)
new_filter = factor.bottom(n)
# Specific assets
new_filter = StaticAssets({'AAPL', 'MSFT'})

# Local filter
new_factor = factor.filter(some_filter)   # fills elements of self with NaN where mask is False

# Multiple returns selecting
new_factor = factor[0]

# Others
new_factor = factor.shift(1)
new_factor = factor.quantile(bins=5)  # factor value quantile groupby datetime
new_factor = factor.fill_na(0)
new_factor = factor.fill_na(ffill=True)  # propagate last valid observation forward to next valid

```


## Dataloader

### CsvDirLoader
`loader = spectre.data.CsvDirLoader(prices_path: str, prices_by_year=False, earliest_date: pd.Timestamp = None,
              dividends_path=None, splits_path=None, file_pattern='*.csv', calender_asset: str = None, align_by_time=False,
              ohlcv=('open', 'high', 'low', 'close', 'volume'), adjustments=None,
              split_ratio_is_inverse=False, split_ratio_is_fraction=False,
              prices_index='date', dividends_index='exDate', splits_index='exDate', **read_csv)`

Read CSV files in the directory, each file represents an asset.

Reading csv is very slow, so you also need to use [ArrowLoader](#arrowloader).

**prices_path:** Prices csv folder. When encountering duplicate datetime in `prices_index`, 
    Loader will keep the last, drop others.\
**prices_index:** `index_col`for csv in `prices_path`\
**prices_by_year:** If prices file name like 'spy_2017.csv', set this to True\
**ohlcv:** Required, OHLCV column names. When you don't need to use `adjustments` and
    `factors.OHLCV`, you can set this to None.\
**adjustments:** Optional, list, `dividend amount` and `splits ratio` column names.\
**dividends_path:** Dividends csv folder, structured as one csv per asset.
    For duplicate data, loader will first drop the exact same rows, and then for the same
    `dividends_index` but different 'dividend amount(`adjustments[0]`)' rows, loader will sum them up.
    If `dividends_path` not set, the `adjustments[0]` column is considered to be included
    in the prices csv.\
**dividends_index:** `index_col`for csv in `dividends_path`.\
**splits_path:** Splits csv folder, structured as one csv per asset.
    When encountering duplicate datetime in `splits_index`, Loader will use the last
    non-NaN 'split ratio', drop others.
    If `splits_path` not set, the `adjustments[1]` column is considered to be included
    in the prices csv.\
**splits_index:** `index_col`for csv in `splits_path`.\
**split_ratio_is_inverse:** If split ratio calculated by to/from, set to True.
    For example, 2-for-1 split, to/form = 2, 1-for-15 Reverse Split, to/form = 0.6666...\
**split_ratio_is_fraction:** If split ratio in csv is fraction string, like `1/3`, set to True.\
**file_pattern:** csv file name pattern, default is '*.csv'.\
**earliest_date:** Data before this date will not be read, save memory.\
**calender_asset:** Asset name as trading calendar, like 'SPY', for clean up non-trading
    time data.\
**align_by_time:** If True and `calender_asset` is not None, the index of datetime will be the same 
   for all assets, if some assets have no data at that time, NaNs will be filled. The benefit is 
   that the columns of data matrix in `CustomFactor.compute` will also be aligned.\
**\*\*read_csv:** Parameters for all csv when calling `pd.read_csv`.
 `parse_dates` or `date_parser` is required.

Example for load [IEX](https://github.com/Heerozh/iex_fetcher) CSV files:

```python
usecols = {'date', 'uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume', 'exDate', 'amount', 'ratio'}
csv_loader = spectre.data.CsvDirLoader(
    './iex/daily/', calender_asset='SPY', 
    dividends_path='./iex/dividends/', 
    splits_path='./iex/splits/',
    ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'), adjustments=('amount', 'ratio'),
    prices_index='date', dividends_index='exDate', splits_index='exDate', 
    parse_dates=True, usecols=lambda x: x in usecols,
    dtype={'uOpen': np.float32, 'uHigh': np.float32, 'uLow': np.float32, 'uClose': np.float32, 
           'uVolume': np.float64, 'amount': np.float64, 'ratio': np.float64})
```

### ArrowLoader

Ingest data from other DataLoader into a feather file, speed up reading speed a lot.

3GB data takes about 7 seconds on initial load.

**Ingest**
`spectre.data.ArrowLoader.ingest(source=CsvDirLoader(...), save_to='./filename.feather')`

**Read**
`loader = spectre.data.ArrowLoader('./filename.feather')`

### QuandlLoader

**no longer updated, only contain prices before 2018**

Download 'WIKI_PRICES.zip' (You need an account):
`https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?qopts.export=true&api_key=[yourapi_key]`

```python
from spectre.data import ArrowLoader, QuandlLoader
ArrowLoader.ingest(source=QuandlLoader('WIKI_PRICES.zip'),
                   save_to='wiki_prices.feather')
```

### How to write your own DataLoader

Inherit from `DataLoader`, overriding the `_load` method, read data into a large `DataFrame`, 
index is `MultiIndex ['date', 'asset']`, where date is `Datetime` type, `asset` is `string` type, 
and then call `self._format(df, split_ratio_is_inverse)` to format the data. 
Also call `test_load` in your test case to do basic format testing.

For example, suppose you have a csv file that contains data for all assets:
```python
class YourLoader(spectre.data.DataLoader):
    @property
    def last_modified(self) -> float:
        return os.path.getmtime(self._path)

    def __init__(self, file: str, calender_asset='SPY') -> None:
        super().__init__(file,
                         ohlcv=('open', 'high', 'low', 'close', 'volume'),
                         adjustments=('ex-dividend', 'split_ratio'))
        self._calender = calender_asset

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path, parse_dates=['date'],
                         usecols=['asset', 'date', 'open', 'high', 'low', 'close',
                                  'volume', 'ex-dividend', 'split_ratio', ],
                         dtype={
                             'open': np.float32, 'high': np.float32, 'low': np.float32,
                             'close': np.float32, 'volume': np.float64,
                             'ex-dividend': np.float64, 'split_ratio': np.float64
                         })

        df.set_index(['date', 'asset'], inplace=True)
        df = self._format(df, split_ratio_is_inverse=True)
        if self._calender:
            df = self._align_to(df, self._calender)

        return df
```

## FactorEngine

A fast factor calculation pipeline.

### FactorEngine.__init__

`engine = FactorEngine(loader: DataLoader)`

### FactorEngine.add

`engine.add(factor, column_name)`

Add a factor to engine.


### FactorEngine.set_filter

`engine.set_filter(factor: FilterFactor or None)`

Set the Global Filter, engine deletes rows which Global Filter returns as False at the last step, 
affect all factors.


### FactorEngine.align_by_time

`engine.align_by_time = bool`

Same as `CsvDirLoader(align_by_time=True)`, but it's dynamic. Notes: Very slow on large amounts of 
data, and if the data source is already aligned, this method cannot make it return to unaligned. 


### FactorEngine.clear

`engine.clear()`

Remove global filter, and all factors.


### FactorEngine.to_cuda

`engine.to_cuda(enable_stream=False)`

Switch to GPU mode.

Set enable_stream to True allows pipeline branches to calculation simultaneously.
However, this will lead to more VRAM usage and may also affect performance.

### FactorEngine.to_cpu

`engine.to_cpu()`

Switch to CPU mode.


### FactorEngine.run

`df = engine.run(start_time, end_time, delay_factor=True)`

Run the engine to calculate the factor data, return a DataFrame. The column is each added factor.

#### *Auto Delay
By default, `delay_factor` is True, it means enable auto-delay. If 'high, low, close, volume' data 
is used by a terminal factor (including its upstream), that factor will be delayed by `shift(1)` 
in the last step, because in theory you can't trade on this factor before it generated. Others
will not be delayed, in order to provide the latest data as much as possible.

Set to `False` to force engine not delay any factors.


### FactorEngine.plot_chart

`engine.plot_chart(start_time, end_time, trace_types=None, styles=None, delay_factor=True)`
    
Plotting common stock price chart for researching.

`trace_types`: `dict(factor_name=plotly_trace_type)`, trace type can be 'Bar', or 'Scatter', 
default is 'Scatter'.

`styles`: `dict(factor_name=plotly_trace_styles)`, add the trace styles, please refer 
to plotly documentation: [Scatter traces](https://plot.ly/python/reference/#scatter)


```python
rsi = factors.RSI()
buy_signal = (rsi.shift(1) < 30) & (rsi > 30)

engine = factors.FactorEngine(loader)
engine.timezone = 'America/New_York'
engine.set_filter(factors.StaticAssets({'NVDA', 'MSFT'}))
engine.add(factors.MA(20), 'MA20')
engine.add(rsi, 'RSI')
engine.add(factors.OHLCV.close.filter(buy_signal), 'Buy')
engine.to_cuda()
_ = engine.plot_chart('2017', '2018', styles={
    'MA20': {
              'line': {'dash': 'dash'}
            },
    'RSI': {
              'yaxis': 'y3',  # y1: price axis, y2: volume axis, yN: add new y-axis
              'line': {'width': 1}
           },
    'Buy': { 
              'mode': 'markers', 
              'marker': { 'symbol': 'triangle-up', 'size': 10, 'color': 'rgba(0, 0, 255, 0.5)' }
           }
})
```

<img src="https://github.com/Heerozh/spectre/raw/media/chart.png" width="730" height="1000">


### FactorEngine.full_run

`factor_data, mean_returns = engine.full_run(
    start_time, end_time, trade_at='close', periods=(1, 4, 9),
    quantiles=5, filter_zscore=20, demean=True, preview=True)`
    
Not only run the engine, but also run factor analysis.


### FactorEngine.get_price_matrix

`df_prices = engine.get_price_matrix(start_time, end_time, prices: ColumnDataFactor = OHLCV.close)`

Get the adjusted historical prices matrix which columns is all assets.

If global filter is setted, all unfiltered assets from `start_time` to `end_time` will be included.


### FactorEngine.test_lookahead_bias

`engine.test_lookahead_bias(start_time, end_time)`

Run the engine to test if there is a lookahead bias.

Fill random values to second half of the ohlcv data, and then check if there are differences between
the two runs in the first half.

## ColumnDataFactor

You can use `ColumnDataFactor` to represents data from any column in the `DataLoader`, for example:

`spectre.factors.ColumnDataFactor(inputs=['col_name'])`

`factors.OHLCV.close` is just a sugar way to write 
`spectre.factors.ColumnDataFactor (inputs = [data_loader.ohlcv[3]])`.


## How to write your own factor

Inherit from `factors.CustomFactor`, write `compute` function.

All `inputs` will pass to compute function.

### win = 1
When `win = 1`, the `inputs` data is tensor type, the first dimension of data is the asset, the 
second dimension is each bar price data. Note that if the data is `align_by_time=False`, the number 
of bars for each asset is different and not aligned (for example, the time for each price in bar_t3 
column may be inconsistent).

        +-----------------------------------+
        |            bar_t1    bar_t3       |
        |               |         |         |
        |               v         v         |
        | asset 1--> [[1.1, 1.2, 1.3, ...], |
        | asset 2-->  [  5,   6,   7, ...]] |
        +-----------------------------------+
Example of LogReturns:
```python
from spectre import factors 
import torch
class LogReturns(factors.CustomFactor):
    inputs = [factors.Returns(2, inputs=[factors.OHLCV.close])]
    win = 1

    def compute(self, change: torch.Tensor) -> torch.Tensor:
        return (change + 1).log()
```

### win > 1
If rolling window is required(`win > 1`), all `inputs` data will be wrapped into
`spectre.parallel.Rolling`.

This is just an unfolded `tensor` data, but because the data is very large after unfolded, for
better performance and saving VRAM, the rolling class automatically splits the data into multiple
small chunks. You need to use the `agg` method to operating `tensor`.
```python
from spectre import factors, parallel
class OvernightReturn(factors.CustomFactor):
    inputs = [factors.OHLCV.open, factors.OHLCV.close]
    win = 2

    def compute(self, opens: parallel.Rolling, closes: parallel.Rolling) -> torch.Tensor:
        ret = opens.last() / closes.first() - 1
        return ret
```
The `closes.first()` above is just a helper method for `closes.agg(lambda x: x[:, :, 0])`,
where `x[:, :, 0]` return the first element of rolling window. The first dimension of `x` is the
asset, the second dimension is each bar, and the third dimension is the bar price and historical 
price with `win` length, and `Rolling.agg` runs on all the chunks and combines them.

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

CustomFactor's inputs data is a matrix without DataFrame's Index information.
If you need index, or not familiar with PyTorch, here is a another way:

```python
from spectre import factors
class YourFactor(factors.CustomFactor):

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        # convert to pd.Series data
        pd_series = self._revert_to_series(data)
        # ...
        # convert back to grouped tensor
        return self._regroup(pd_series)
```
This method is completely non-parallel and inefficient, but easy to write.

## Back-testing

[Quick Start](#back-testing) contains easy-to-understand examples, please read first.

The `spectre.trading.CustomAlgorithm` currently does not supports live trading,
will implement it in the future.

### CustomAlgorithm.initialize

`alg.initialize(self)` **Callback**

Called when back-testing starts, at least you need use `get_factor_engine` to add factors
and call `schedule_rebalance` here.


### CustomAlgorithm.terminate

`alg.terminate(self, records: pd.DataFrame)` **Callback**

Called when back-testing ends.


### rebalance callback

`rebalance(self, data: pd.DataFrame, history: pd.DataFrame)` **Callback**

The function name does not have to be 'rebalance', it can be specified in `schedule_rebalance`.
`data` is the factors data of last bar returned by `FactorEngine`; 
`history` same as `data`, but contains previous data, please refer to `set_history_window`.

Put calculations into the `FactorEngine` as much as possible can improve backtest performance.


### CustomAlgorithm.get_factor_engine

`self.get_factor_engine(name: str = None)`
**context:** *initialize, rebalance, terminate*

Get the factor engine of this trading algorithm. But note that you can add factors or filter only 
during `initialize`, otherwise it will cause unexpected effects.

The algorithm has a default engine, `name` can be None.
But if you created multiple engines using `create_factor_engine`, you need to specify which one.


### CustomAlgorithm.create_factor_engine

`self.create_factor_engine(name: str, loader: DataLoader = None)`
**context:** *initialize*

Create another engine, generally used when you need multiple data sources.


### CustomAlgorithm.set_history_window

`self.set_history_window(offset: pd.DateOffset=None)`
**context:** *initialize*

Set the length of historical data passed to each `rebalance` call. **SLOW**

Default: If None, pass all available historical data, so there will be no historical data on the
first day, one historical row on the next day, and so on.


### CustomAlgorithm.schedule_rebalance

`self.schedule_rebalance(event: Event)`
**context:** *initialize*

Schedule `rebalance` to be called when an event occurs.
Events are: `MarketOpen`, `MarketClose`, `EveryBarData`,
For example:
```python
alg.schedule_rebalance(trading.event.MarketClose(self.any_function))
```

The `Market*` events has `offset_ns` parameter `MarketClose(self.any_function, offset_ns=-1000)`, 
a negative value of `offset_ns` means 'before', in backtest mode, the magnitude of the value has no
effect.


### CustomAlgorithm.schedule

`self.schedule(event: Event)`
**context:** *initialize*
   
Schedule an event, callback is `callback(source: "Any class who fired this event")`


### CustomAlgorithm.empty_cache_after_run

`self.empty_cache_after_run = True`
**context:** *initialize*

Empty engine's cache after factor calculation. 
If you need more VRMA in rebalance context, or wanna play 3D game when backtesting, set it to 
True will help.


### CustomAlgorithm.stop_event_manager

`alg.stop_event_manager()`
**context:** *all*

Stop backtesting or live trading.


### CustomAlgorithm.fire_event

`alg.fire_event(event_type: Type[Event])`
**context:** *all*

Trigger a type of event (any subclasses that inherit from `Event`）, 
for example: `alg.fire_event(MarketClose)`, (do not do this, do not fire built-in events)


### CustomAlgorithm.results

`self.results`
**context:** *terminate*

Get back-test results, same as the return value of [trading.run_backtest](#spectretradingrun_backtest)


### CustomAlgorithm.plot

`self.plot(annual_risk_free=0.04, benchmark: Union[pd.Series, str] = None)`
**context:** *terminate*

Plot a simple portfolio cumulative return chart.\
`benchmark`: `pd.Series` of benchmark daily return, or an asset name.



### CustomAlgorithm.current

`self.current`
**context:** *rebalance*

Current datetime, Read-Only.


### CustomAlgorithm.get_price_matrix

`self.get_price_matrix(length: pd.DateOffset, name: str = None, prices=OHLCV.close)`
**context:** *rebalance*

Help method for calling `engine.get_price_matrix`, `name` specifies which engine.

Returns the historical asset prices, adjusted and filtered by the current time.
**Slow**


### CustomAlgorithm.record

`self.record(**kwargs)`
**context:** *rebalance*

Record the data and pass all when calling `terminate`, use `column = value` format.


### SimulationBlotter.set_commission

`self.blotter.set_commission(percentage=0, per_share=0.005, minimum=1)`
**context:** *initialize*

percentage: percentage part, calculated by `percentage * price * shares`\
per_share: calculated by `per_share * shares`\
minimum: minimum commission if above sum does not exceed

commission = max(percentage_part + per_share_part, minimum)


### SimulationBlotter.set_slippage

`self.blotter.set_slippage(percentage=0, per_share=0.01)`
**context:** *initialize, rebalance*

Market impact add to the price.


### SimulationBlotter.set_short_fee

`self.blotter.set_short_fee(percentage=0)`
**context:** *initialize*

Set the transaction fees which only charged for sell orders.


### SimulationBlotter.daily_curb

`self.blotter.daily_curb = float` 
**context:** *initialize, rebalance*

Limit on trading a specific asset if today to previous day return >= ±value. **SLOW**


### SimulationBlotter.order_target

`self.blotter.order_target(asset: str, target: number)` 
**context:** *rebalance*

Place an order on an asset to target number of shares in position, negative number means short.

If asset cannot be traded or limited by `daily_curb`, it will return False.


### SimulationBlotter.batch_order_target

`self.blotter.batch_order_target(asset: Iterable[str], target: Iterable[float])` 
**context:** *rebalance*

Same as `SimulationBlotter.order_target`, but for multiple assets.

Return value is a list of skipped assets, which indicate that they cannot be traded or limited by 
`daily_curb`.


### SimulationBlotter.order_target_percent

`self.blotter.order_target_percent(asset: str, pct: float)` 
**context:** *rebalance*

Place an order on an asset to target percentage of portfolio net value, negative number means short.

If asset cannot be traded or limited by `daily_curb`, it will return False.


### SimulationBlotter.batch_order_target_percent

`self.blotter.batch_order_target_percent(asset: Iterable[str], pct: Iterable[float])` 
**context:** *rebalance*

Same as `SimulationBlotter.order_target_percent`, but for multiple assets and better performance.

Return value is a list of skipped assets, which indicate that they cannot be traded or limited by 
`daily_curb`.


### SimulationBlotter.order

`self.blotter.order(asset: str, amount: int)` 
**context:** *rebalance*

Order a certain amount of an asset, negative number means short.

If asset cannot be traded or limited by `daily_curb`, it will return False.


### SimulationBlotter.get_price

`float = self.blotter.get_price(asset: Union[str, Iterable])` 
**context:** *rebalance*

Get current price of assert. 
*Notice: Batch calls are slow, You can add prices as factor to get the price,
like: `engine.add(OHLCV.close, 'prices')`*


### SimulationBlotter.portfolio.set_stop_model

`self.blotter.portfolio.set_stop_model(model: StopModel)` 
**context:** *initialize*

Set stop tracking model for positions, models are:

`trading.StopModel(ratio, callback)`\
`trading.TrailingStopModel(ratio, callback)`\
`trading.PnLDecayTrailingStopModel` and `trading.TimeDecayTrailingStopModel`

Stop loss example:
```python
class Backtester(trading.CustomAlgorithm):
    def initialize(self):
        ...
        self.blotter.portfolio.set_stop_model(trading.TrailingStopModel(-0.1, self.stop))

    def stop(self, asset, amount):
        self.blotter.order(asset, amount)
        self.record(...)

    def rebalance(self, data, history):
        self.blotter.portfolio.check_stop_trigger()
        ...
```

#### PnLDecayTrailingStopModel
`trading.PnLDecayTrailingStopModel(ratio, pnl_target, callback, decay_rate=0.05, max_decay=0)`

This is a model that can stop gain and stop loss at the same time.

Exponential decay to the stop ratio: `ratio * decay_rate ^ (PnL% / PnL_target%)`, 
So `PnLDecayTrailingStopModel(-0.1, 0.1, callback)` means initial stop loss is -10%, and the 
`ratio` will decrease when profit% approaches the target +10%. If recorded high profit% exceeds 10%, 
any drawdown will trigger a stop loss.

#### TimeDecayTrailingStopModel
`trading.TimeDecayTrailingStopModel(ratio, period_target: pd.Timedelta, callback, decay_rate=0.05, 
max_decay=0)`

Same as `PnLDecayTrailingStopModel`, but target is time period.


### SimulationBlotter.get_returns

`self.blotter.get_returns()`
**context:** *rebalance, terminate*

Get the portfolio returns, use `(self.blotter.get_returns() + 1).prod()` to get current cumulative
return.


### SimulationBlotter.portfolio Read Only Properties

**context:** *rebalance, terminate*

`self.blotter.portfolio.positions` Current positions, `Dict[asset, Position]` type.
```python
class Position:
    shares = None
    average_price = None
    last_price = None
    unrealized = None
    realized = None
```

`self.blotter.portfolio.value` Current portfolio value

`self.blotter.portfolio.cash` Current portfolio cash

`self.blotter.portfolio.leverage` Current portfolio leverage


### spectre.trading.run_backtest

`results = trading.run_backtest(loader: DataLoader, alg_type: Type[CustomAlgorithm], start, end)`

Run backtest, return value is namedtuple:

**results.returns:** daily return

**results.positions:** daily positions

**results.transactions:** full transactions with all orders



# Copyright & Thanks
Copyright (C) 2019-2020, by Zhang Jianhao (heeroz@gmail.com), All rights reserved.

Thanks to [JetBrains](https://www.jetbrains.com/?from=spectre)'s support.

------------
> *A spectre is haunting Market — the spectre of capitalism.*
