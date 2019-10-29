# spectre 

> *A spectre is haunting Market — the spectre of capitalism.*

spectre is an quantitative trading library, 
targets performance, GPU support(TODO) and keep it simple. 

[Under construction]

## Chapter I. Factor and FactorEngine

### Quick Start
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

### How to use

Factor system has 4 parts:

1. DataLoader  
    For reading time series data.  
    `factors.CsvDirLoader`: Read all csv files in the directory, one csv file per asset.  
    `factors.QuandlLoader`: Read the quandl wiki prices zip. Currently stopped updating.
    Example:
    `loader = factors.QuandlLoader('./WIKI_PRICES.zip')`

2. FactorEngine  
    For integrating data. The `DataLoader` is called first to read the basic data, 
    then all the added Factors are computed and integrated.

3. Factor  
    All data is a `Factor`, including each column read by the `DataLoader`.  
    **DataFactor:**  
    Get the data in the `DataLoader`, only support one column, example:  
    `volume = factors.DataFactor(inputs=('adj.volume',))`  
    Or use `factors.OHLCV.volume` placeholder to get same`DataFactor` 

    **Indicator:**  
    SMA: `factors.SMA(win=5, inputs=(factors.OHLCV.close,))`
    
    **CustomFactor:**  
    You can inherit `CustomFactor` to write your own Factor, example:
    ```python
    class OverNightReturn(CustomFactor):
        inputs = [OHLCV.close]     # Default input data, if not specified when instantiating this class
        win = 2                    # Default window length, if not specified when instantiating this class
        _min_win = 2               # Allowed minimum window length
    
        def compute(self, close):
            return (close.shift(self.win) - close) / close
    ```

4. Filter  
   The filter is a `Factor` with only `True` and `False` values.
   If pass to the `FactorEngine.set_filter` function, all rows whose results are False will be excluded.

   
   
   
## Chapter II. Portfolio and Backtesting

[Under construction]

## Chapter III. Benchmarking

|                 |      SMA100      | Ratio |  BBAMDS( MACD( SMA(5) ) )  | Ratio |
|-----------------|------------------|-------|---------------------------|-------|
|zipline          | 2.04 s ± 15.3 ms |	 1   |                      |	 1   |
|spectre (CPU)    | 1.34 s ± 6.97 ms    | 1.52x |                      |**1.52x**|
|spectre (**GPU**)| TODO |		 |**  **|

Using quandl data, compute factors between '2014-01-02' to '2016-12-30'  
 

[Under construction]


----------------------------------------------------------------

# Chinese Version:
# spectre 

spectre 是一个量化投资库，为因子研究和交易回测。主要目标为运行速度并保持简化。

[Under construction]

## Chapter I. Factor and FactorEngine

### How to use

Factor系统有4个部分

1. DataLoader
   负责读取时间序列数据  
   `factors.CsvDirLoader`： 读取目录下的所有csv文件，每个资产一个csv文件  
   `factors.QuandlLoader`： 读取quandl wiki prices的zip文件，目前qunaldl已不更新数据，
   仅做参考。 示例:  
   `loader = factors.QuandlLoader('./WIKI_PRICES.zip')`
    
2. FactorEngine  
   负责整合数据。会先调用DataLoader读取基本数据，然后计算所有添加的Factor并整合。

3. Factor  
   所有数据都是一个Factor，包括DataLoader读取的每列。  
   **DataFactor:**  
   获取DataLoader里数据的Factor，示例：  
   `factors.DataFactor(inputs=('column_name',))`  
   也可以使用比如 `factors.OHLCV.volume` 通用占位符来获取`DataFactor` 
    
   **指标：**  
   SMA: `factors.SMA(win=5, inputs=(factors.OHLCV.close,))`
   
   **CustomFactor:**  
   你可以继承CustomFactor来写自己的Factor，示例
   ```python
    class OverNightReturn(CustomFactor):
        inputs = [OHLCV.close]     # 默认输入数据，用户也可以用OverNightReturn(inputs=...)修改
        win = 2                    # 默认窗口期，用户也可以用OverNightReturn(win=...)修改
        _min_win = 2               # 允许的最小窗口期
    
        def compute(self, close):
            return (close.shift(self.win) - close) / close
   ```

4. Filter  
   过滤器是计算结果只有True和False的Factor，但可以传入FactorEngine.set_filter函数，
   之后所有结果为False的Row将不包含在最终结果里。
   
   
   
## Chapter II. Portfolio and Backtesting

[Under construction]

