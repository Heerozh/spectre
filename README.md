# spectre 

> *A spectre is haunting Market — the spectre of capitalism.*

spectre is an quantitative trading library, 
targets performance, GPU support(TODO) and still keep it simple. 

[Under construction]

## Chapter I. Factor and FactorEngine

```python
from spectre import factors

loader = factors.CsvDirLoader(
    './tests/data/daily/', 
    ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
    index_col='date', parse_dates=True,
)
engine = factors.FactorEngine(loader)
engine.add(factors.OHLCV.close, 'close')
engine.add(factors.SMA(5), 'ma5')
df = engine.run('2019-01-11', '2019-01-15')
df
```
		

|date                     |    |        ma5|	 close|	
|-------------------------|----|-----------|----------|
|2019-01-11 00:00:00+00:00|AAPL|    154.254|	153.69|
|                         |MSFT|    104.402|	103.20|
|2019-01-14 00:00:00+00:00|AAPL|    155.854|	157.00|
|                         |MSFT|    104.202|	103.39|
|2019-01-15 00:00:00+00:00|AAPL|    156.932|	156.94|
|                         |MSFT|    105.332|	108.85|


## Chapter II. Portfolio and Backtesting

[Under construction]

## Chapter III. Benchmarking

|  time      |      MA100       |	... |
|------------|------------------|------|
|zipline     | 767 ms ± 10.4 ms |	 |
|spectre     | 650 ms ± 5.08 ms |	 |
|spectre(GPU)| TODO |		 |

Code:
```python
start, end = pd.Timestamp('2017-01-03', tz='UTC'), pd.Timestamp('2019-01-03', tz='UTC')
bundle_data = zipline.data.bundles.load('quandl')
engine = SimplePipelineEngine(...)
ma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=100)
pipeline = Pipeline()
pipeline.add( ma, 'ma')
%timeit engine.run_pipeline(pipeline, start, end)

loader = factors.QuandlLoader('../../historical_data/us/prices/quandl/WIKI_PRICES.zip')
engine = factors.FactorEngine(loader)
engine.add(factors.SMA(100), 'ma')
%timeit engine.run(start, end)
```

[Under construction]