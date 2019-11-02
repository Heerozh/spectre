
Progress: 4/10  🔳🔳🔳🔳⬜⬜⬜⬜⬜⬜  
~~1/10: FactorEngine architecture~~  
~~2/10: FactorEngine~~  
~~3/10: Filters~~  
~~4/10: All factors~~  
5/10: Factor returns and analysis    
6/10: Back-test architecture  
7/10: Portfolio  
8/10: Transaction  
9/10: Back-test  
10/10: Analysis  
11/10: CUDA support  
12/10: All CUDA factors  

# spectre 

spectre is an quantitative trading library, 
targets performance, GPU support(TODO) and keep it simple. 

[Under construction]

## Chapter I. Factor and FactorEngine

### Quick Start
```python
from spectre import factors

loader = factors.CsvDirLoader(
    './tests/data/daily/',  # Contains AAPL.csv MSFT.csv
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

### Factor lists

```python
    # All technical factors passed TA-Lib Comparison test
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

## Chapter II. Portfolio and Backtesting

[Under construction]

## Chapter III. Benchmarking

|                 |      SMA100      | Ratio | EMA50 (win=229) | Ratio   | MACD(12,26,9)    | Ratio   |
|-----------------|------------------|-------|-----------------|---------|------------------|---------|
|zipline          | 1.93 s ± 34.3 ms |	 1   | 5.6 s ± 68.7 ms |	1    | 5.49 s ± 72.1 ms |	 1    |
|spectre (CPU)    | 1.46 s ± 6.52 ms | 1.32x | 1.67 s ± 9.14 ms|**3.35x**| 2.67 s ± 23.6 ms |**2.06x**|
|spectre (**GPU**)| TODO             |**  ** |	                |**   **|	                 |      |

Using quandl data, compute factor between '2014-01-02' to '2016-12-30'  
 
[Under construction]

## Chapter IV. Full Example

```python
from spectre import factors
loader = factors.QuandlLoader('WIKI_PRICES.zip')
universe = factors.AverageDollarVolume(win=120).top(500) 
engine = factors.FactorEngine(loader)
engine.set_filter(universe)
engine.add((-factors.MA(100)).rank().zscore(), 'ma')
df_factors = engine.run('2014-01-02', '2016-12-30')
df_prices = engine.get_price_matrix('2014-01-02', '2016-12-30')

import alphalens as al
al_clean_data = al.utils.get_clean_factor_and_forward_returns(factor=df_factors['ma'], prices=df_prices, periods=[11])
al.performance.mean_return_by_quantile(al_clean_data)[0].plot.bar()
# al.tears.create_full_tear_sheet(al_clean_data)
```

![matplot](data:image/png;base64,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)

[Under construction]

------------
> *A spectre is haunting Market — the spectre of capitalism.*
