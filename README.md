# spectre 

A spectre is haunting Market — the spectre of capitalism.

[Under construction]

## Chapter I. Factor and FactorEngine

    from spectre import CsvDirLoader, factors
    
    loader = CsvDirLoader(
        './tests/data/daily/', 
        ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
        index_col='date', parse_dates=True,
    )
    engine = factors.FactorEngine(loader)
    engine.add(factors.OHLCV.close, 'close')
    df = engine.run('2019-01-11', '2019-01-15')

		

|date                     |    |     close|	
|-------------------------|----|----------|
|2019-01-11 00:00:00+00:00|AAPL|	153.69|
|                         |MSFT|	103.20|
|2019-01-14 00:00:00+00:00|AAPL|	157.00|
|                         |MSFT|	103.39|
|2019-01-15 00:00:00+00:00|AAPL|	156.94|
|                         |MSFT|	108.85|
## Chapter II. Portfolio and Backtesting

## Chapter III. Benchmarking