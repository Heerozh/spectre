import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_array_equal

from spectre import factors

START, END = '2019-01-11', '2019-01-15'


def values(df, asset, col):
    return df.loc[(slice(None), asset), col].values


def test_column_data(engine_no_calendar):
    engine = engine_no_calendar
    engine.add(factors.OHLCV.volume, 'volume')
    df = engine.run(START, END)
    # delayed, volume of 2019-01-11 and 2019-01-14
    assert_array_equal(values(df, 'AAPL', 'volume'), (28065422, 33834032))
    assert_array_equal(values(df, 'MSFT', 'volume'), (28627674, 28720936))

    engine.add(factors.ColumnDataFactor(inputs=('changePercent',)), 'chg')
    df = engine.run(START, END)
    assert_array_equal(values(df, 'AAPL', 'chg'), (-0.9835, -1.5724))
    assert_array_equal(values(df, 'MSFT', 'chg'), (-0.8025, -0.7489))


@pytest.mark.parametrize('delay', [False, True])
def test_open_not_delayed(engine_no_calendar, delay):
    engine_no_calendar.add(factors.OHLCV.open, 'open')
    df = engine_no_calendar.run(START, END, delay_factor=delay)
    assert_array_equal(values(df, 'AAPL', 'open'), (155.72, 155.19, 150.81))
    assert_array_equal(values(df, 'MSFT', 'open'), (104.65, 104.9, 103.19))


def test_column_data_dtype(engine):
    engine.add(factors.ColumnDataFactor(inputs=('uClose',), dtype=torch.float32), 'close')
    df = engine.run(START, END)
    assert df.close.dtype == np.float32


@pytest.mark.parametrize(
    'fill_na, expected',
    [
        (None, [1.0, np.nan, 3.0]),
        ('ffill', [1.0, 1.0, 3.0]),
        ('bfill', [1.0, 3.0, 3.0]),
    ],
)
def test_series_data(engine, fill_na, expected):
    dates = pd.to_datetime(['2019-01-11', '2019-01-15'], utc=True)
    index = pd.MultiIndex.from_arrays([dates, ['AAPL', 'AAPL']], names=['date', 'asset'])
    series = pd.Series([1.0, 3.0], index=index, name='my_data')
    engine.add(factors.SeriesDataFactor(series, fill_na=fill_na, should_delay=False), 'test')
    df = engine.run(START, END)
    assert_array_equal(values(df, 'AAPL', 'test'), expected)
    assert np.isnan(values(df, 'MSFT', 'test')).all()


def test_series_data_invalid_fill_na(engine):
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(['2019-01-11'], utc=True), ['AAPL']], names=['date', 'asset']
    )
    series = pd.Series([1.0], index=index, name='my_data')
    engine.add(factors.SeriesDataFactor(series, fill_na='zero'), 'test')
    with pytest.raises(ValueError, match='`fill_na` must be one of'):
        engine.run(START, END)
