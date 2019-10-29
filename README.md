
Progress: 2/10  🔳🔳⬜⬜⬜⬜⬜⬜⬜⬜  
~~1/10: FactorEngine architecture~~  
~~2/10: FactorEngine~~  
3/10: All factors  
4/10: Factor returns and analysis  
5/10: Back-test architecture  
6/10: Transaction  
7/10: Portfolio  
8/10: Back-test  
9/10: CUDA support  
10/10: All CUDA factors  

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

Factor system only has 3 parts:

1. DataLoader  
    For reading time series data.  
    `factors.CsvDirLoader`: Read all csv files in the directory, one csv file per asset.  
    `factors.QuandlLoader`: Read the quandl wiki prices zip. Currently stopped updating.
    Example:
    `loader = factors.QuandlLoader('./WIKI_PRICES.zip')`  
    The `ohlcv` parameter indicates the column name used by the placeholder `factors.OHLCV.oepn`

2. Factor  
    All data is a `Factor`, including each column read by the `DataLoader`.  
    **DataFactor:**  
    Get the data in the `DataLoader`, only support one column, example:  
    `volume = factors.DataFactor(inputs=('adj.volume',))`  
    Or use `factors.OHLCV.volume` placeholder to get same`DataFactor` 

    **Indicator**  
    SMA: `factors.SMA(win=5, inputs=(factors.OHLCV.close,))`
    
    **CustomFactor**  
    You can inherit `CustomFactor` to write your own Factor, example:
    ```python
    class OverNightReturn(CustomFactor):
        inputs = [OHLCV.oepn, OHLCV.close]  # Default input data, if not specified when instantiating 
        win = 2                             # Default window length, if not specified when instantiating
        _min_win = 2                        # Allowed minimum window length
    
        def compute(self, open, close):     # All `inputs` will be passed in order
            return (open.shift(self.win) - close) / close
    ```

   **Filter**  
   The filter is a `Factor` with only `True` and `False` values.
   If pass to the `FactorEngine.set_filter` function, all rows whose results are `False` will be excluded.

3. FactorEngine  
    For integrating data. The `DataLoader` is called first to read the basic data, 
    then all the added Factors are computed and integrated.   
   
   
## Chapter II. Portfolio and Backtesting

[Under construction]

## Chapter III. Benchmarking

|                 |      SMA100      | Ratio |  BBAMDS( MACD( SMA(5) ) )  | Ratio |
|-----------------|------------------|-------|---------------------------|-------|
|zipline          | 2.04 s ± 15.3 ms |	 1   |                      |	 1   |
|spectre (CPU)    | 1.34 s ± 6.97 ms    | 1.52x |     TODO                 |****|
|spectre (**GPU**)| TODO |		 |**  **|

Using quandl data, compute factor between '2014-01-02' to '2016-12-30'  
 

[Under construction]


----------------------------------------------------------------

# Chinese Version:
# spectre 

spectre 是一个量化投资库，为因子研究和交易回测。主要目标为运行速度并保持简化。

[Under construction]

## Chapter I. Factor and FactorEngine

### Quick Start
见英文版

### How to use

Factor系统只有3个部分

1. DataLoader
   负责读取时间序列数据  
   `factors.CsvDirLoader`： 读取目录下的所有csv文件，每个资产一个csv文件  
   `factors.QuandlLoader`： 读取quandl wiki prices的zip文件，目前quandl已不更新数据，
   仅做参考。 示例:  
   `loader = factors.QuandlLoader('./WIKI_PRICES.zip')`  
   `ohlcv` 参数指明通用占位符 `factors.OHLCV.oepn` 使用的列名

2. Factor  
   所有数据都是一个`Factor`，包括`DataLoader`读取的每列。  
   **DataFactor**  
   获取`DataLoader`里数据的`Factor`，输入数据只能传入一列，示例：  
   `factors.DataFactor(inputs=('column_name',))`  
   也可以使用比如 `factors.OHLCV.volume` 通用占位符来获取`DataFactor` 
    
   **指标**  
   SMA: `factors.SMA(win=5, inputs=(factors.OHLCV.close,))`
   
   **CustomFactor**  
   你可以继承`CustomFactor`来写自己的`Factor`，示例
   ```python
    class OverNightReturn(CustomFactor):
        inputs = [OHLCV.oepn, OHLCV.close]  # 默认输入数据，用户也可以用OverNightReturn(inputs=...)修改
        win = 2                             # 默认窗口期，用户也可以用OverNightReturn(win=...)修改
        _min_win = 2                        # 允许的最小窗口期
    
        def compute(self, oepn, close):     # 所有 `inputs` 数据将按顺序传入
            return (oepn.shift(self.win) - close) / close
   ```

   **Filter**   
   过滤器是计算结果只有`True`和`False`的`Factor`，但可以传入`FactorEngine.set_filter`函数，
   之后所有结果为`False`的行将不包含在最终结果里。
   
3. FactorEngine  
   负责整合数据。会先调用`DataLoader`读取基本数据，然后计算所有添加的`Factor`并整合。
   
   
## Chapter II. Portfolio and Backtesting

[Under construction]

## Chapter III. Benchmarking

见英文版