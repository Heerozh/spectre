
Progress: 4/10  🔳🔳🔳🔳⬜⬜⬜⬜⬜⬜\
~~1/10: FactorEngine architecture~~\
~~2/10: FactorEngine~~\
~~3/10: Filters~~\
~~4/10: All factors~~\
5/10: CUDA support\
6/10: Dividends/Splits\
7/10: Back-test architecture\
8/10: Portfolio\
9/10: Transaction\
10/10: Back-test\
11/10: Factor Return Analysis

# ||spectre

spectre is a **GPU-accelerated Parallel** quantitative trading library, focused on **performance**.

spectre is a **GPU Enabled Parallel** quantitative trading library,
totally focused on **performance** and **clean**.

  * Pure python code
  * Using **PyTorch** for parallel computation. So yes, spectre can return a `torch.Tensor` type, use it in PyTorch Model directly without any performance loss.
  * zipline limiting pandas version at 0.22 for performance, spectre don't have this limitation.

[Under construction]

# Status

In development.

## Chapter I. Factor and FactorEngine

### Quick Start
```python
from spectre import factors

loader = factors.CsvDirLoader(
    './tests/data/daily/',  # Contains fake dataset: AAPL.csv MSFT.csv
    ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
    index_col='date', parse_dates=True,
)
engine = factors.FactorEngine(loader)
engine.add(factors.OHLCV.close, 'close')
engine.add(factors.SMA(5), 'ma5')
df = engine.run('2019-01-11', '2019-01-15')
df
```


|                         |         |        ma5|	 close|
|-------------------------|---------|-----------|---------|
|**date**                 |**asset**|           |	      |
|2019-01-11 00:00:00+00:00|     AAPL|    154.254|	153.69|
|                         |     MSFT|    104.402|	103.20|
|2019-01-14 00:00:00+00:00|     AAPL|    155.854|	157.00|
|                         |     MSFT|    104.202|	103.39|
|2019-01-15 00:00:00+00:00|     AAPL|    156.932|	156.94|
|                         |     MSFT|    105.332|	108.85|

## Chapter II. Portfolio and Backtesting

[Under construction]

## Chapter III. Benchmarking

|         |      SMA100      | Ratio | EMA50 (win=229)  | Ratio   | MACD(12,26,9)    | Ratio   |
|---------|------------------|-------|------------------|---------|------------------|---------|
|zipline  | 1.89 s ± 23.1 ms |   1   | 5.45 s ± 14.6 ms |	  1   | 5.28 s ± 14 ms   |	   1   |
|spectre  | 1.79 s ± 7.02 ms | 1.06x | 2.06 s ± 7.68 ms |**2.65x**| 3.05 s ± 24.8 ms |**1.73x**|

Using quandl data, compute factor between '2014-01-02' to '2016-12-30'

As of now (unfinished)

[Under construction]

## Chapter IV. Full Example

```python
from spectre import factors
# ------------- get data -------------
loader = factors.QuandlLoader('WIKI_PRICES.zip')
engine = factors.FactorEngine(loader)
# ------------- set factors -------------
engine.set_filter( factors.AverageDollarVolume(win=120).top(500) )
engine.add( (factors.MA(5)-factors.MA(10)-factors.MA(30)).rank().zscore(), 'ma_cross' )
# ------ get factors value and prices ------
df_factors = engine.run('2014-01-02', '2016-12-10')
df_prices = engine.get_price_matrix('2014-01-02', '2016-12-30')
# ------ analysis with alphalens ------
import alphalens as al
al_clean_data = al.utils.get_clean_factor_and_forward_returns(
    factor=df_factors['ma_cross'], prices=df_prices, periods=[11])
al.performance.mean_return_by_quantile(al_clean_data)[0].plot.bar()
(al.performance.factor_returns(al_clean_data) + 1).cumprod().plot()
```

<img src="https://github.com/Heerozh/spectre/raw/media/quantile_return.png" width="50%" height="50%">
<img src="https://github.com/Heerozh/spectre/raw/media/cumprod_return.png" width="50%" height="50%">

[Under construction]

## Chapter V. API

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


------------
> *A spectre is haunting Market — the spectre of capitalism.*
