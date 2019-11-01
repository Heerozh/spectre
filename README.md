
Progress: 2/10  🔳🔳🔳⬜⬜⬜⬜⬜⬜⬜  
~~1/10: FactorEngine architecture~~  
~~2/10: FactorEngine~~  
~~3/10: Filters~~  
4/10: All factors  
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

|                 |      SMA100      | Ratio | EMA50 (win=229) | Ratio   | MACD(12,26,9)    | Ratio   |
|-----------------|------------------|-------|-----------------|---------|------------------|---------|
|zipline          | 1.93 s ± 34.3 ms |	 1   | 5.6 s ± 68.7 ms |	1    | 5.49 s ± 72.1 ms |	 1    |
|spectre (CPU)    | 1.46 s ± 6.52 ms | 1.32x | 1.67 s ± 9.14 ms|**3.35x**| 2.67 s ± 23.6 ms |**2.06x**|
|spectre (**GPU**)| TODO             |**  ** |	                |**   **|	                 |      |

Using quandl data, compute factor between '2014-01-02' to '2016-12-30'  
 
[Under construction]

------------
> *A spectre is haunting Market — the spectre of capitalism.*
