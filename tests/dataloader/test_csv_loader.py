import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

import spectre
from tests.conftest import DAILY_OHLCV, DATA_DIR, make_adjusted_daily_loader

START, END = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')


def read_raw(asset, folder='daily', index_col='date'):
    return pd.read_csv(DATA_DIR / folder / f'{asset}.csv', index_col=index_col, parse_dates=True)


def first_last(df, asset, col):
    values = df.loc[(slice(None), asset), col].values
    return values[0], values[-1]


def dates_of(df):
    return df.index.get_level_values(0).unique()


def values_of(df, asset, col):
    return df.loc[(slice(None), asset), col].values


@pytest.mark.parametrize('kwargs', [{}, {'prices_index': 'date'}])
def test_index_must_be_datetime(kwargs):
    loader = spectre.data.CsvDirLoader(str(DATA_DIR / 'daily'), **kwargs)
    with pytest.raises(ValueError, match='df must index by datetime.*'):
        loader.load(START, END, 0)


def test_empty_dir(tmp_path):
    loader = spectre.data.CsvDirLoader(str(tmp_path), prices_index='date', parse_dates=True)
    with pytest.raises(ValueError, match='There are no files'):
        loader.load()


def test_format(daily_loader):
    df = daily_loader.test_load()
    assert df.index.names == ['date', 'asset']
    assert str(df.index.levels[0].tz) == 'UTC'
    assert list(df.index.levels[1]) == ['AAPL', 'MSFT']
    # time category is the id of unique datetime
    dates = df.index.get_level_values(0)
    assert_array_equal(df[daily_loader.time_category].values, pd.factorize(dates, sort=True)[0])


def test_calender_asset(daily_loader, daily_loader_no_calendar):
    # MSFT has a junk row on 2019-01-05 (Saturday), dropped by calender_asset='AAPL'
    junk = pd.Timestamp('2019-01-05', tz='UTC')
    assert junk not in daily_loader.load().index.levels[0]
    assert junk in daily_loader_no_calendar.load().index.levels[0]


def test_load_values():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / 'daily'), calender_asset='AAPL', prices_index='date', parse_dates=True
    )

    df = loader.load(START, END, 0)
    aapl, msft = read_raw('AAPL'), read_raw('MSFT')
    assert first_last(df, 'AAPL', 'close') == (160.35, 158.09)
    assert first_last(df, 'MSFT', 'close') == (100.1, 105.36)
    assert_array_equal(values_of(df, 'AAPL', 'close'), aapl.close['2019-01-01':'2019-01-15'].values)
    assert_array_equal(
        values_of(df, 'MSFT', 'close'), msft.close['2019-01-01':'2019-01-15'].dropna().values
    )

    # 2019-01-12 is Saturday
    df = loader.load(pd.Timestamp('2019-01-11', tz='UTC'), pd.Timestamp('2019-01-12', tz='UTC'))
    assert first_last(df, 'MSFT', 'close') == (104.5, 104.5)


def test_load_backwards():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / 'daily'), calender_asset='AAPL', prices_index='date', parse_dates=True
    )
    df = loader.load(START, END, 11)
    # 11 trading days before START
    dates = dates_of(df)
    assert dates[0] == pd.Timestamp('2018-12-14', tz='UTC')
    assert len(dates[dates < START]) == 11
    assert first_last(df, 'AAPL', 'close') == (173.43, 158.09)
    assert first_last(df, 'MSFT', 'close') == (106.57, 105.36)


def test_load_out_of_range(daily_loader):
    with pytest.raises(ValueError, match='cannot be less than earliest time'):
        daily_loader.load(pd.Timestamp('2000-01-01', tz='UTC'), END)
    with pytest.raises(ValueError, match='cannot be greater than latest time'):
        daily_loader.load(START, pd.Timestamp('2030-01-01', tz='UTC'))
    with pytest.raises(AssertionError, match='There is no data between'):
        daily_loader.load(
            pd.Timestamp('2019-01-05', tz='UTC'), pd.Timestamp('2019-01-06', tz='UTC')
        )


def test_earliest_date():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / 'daily'),
        prices_index='date',
        parse_dates=True,
        earliest_date=pd.Timestamp('2019-01-02'),
    )
    assert loader.load().index.levels[0][0] == pd.Timestamp('2019-01-02', tz='UTC')


def test_zero_prices_are_forward_filled(tmp_path):
    index = ['2019-01-01', '2019-01-02', '2019-01-03']
    ohlcv = ('open', 'high', 'low', 'close', 'volume')
    a = pd.DataFrame({'date': index, **{c: [1.0, 0, 3] for c in ohlcv}})
    b = pd.DataFrame({'date': index, **{c: [0.0, 2, 0] for c in ohlcv}})
    a.to_csv(tmp_path / 'A.csv', index=False)
    b.to_csv(tmp_path / 'B.csv', index=False)
    loader = spectre.data.CsvDirLoader(str(tmp_path), prices_index='date', parse_dates=True)
    df = loader.load()
    for col in ohlcv:
        assert_array_equal(values_of(df, 'A', col), [1, 1, 3])
        # the leading zero of B is not filled by the price of A
        assert_array_equal(values_of(df, 'B', col), [0, 2, 2])


def test_split_by_year(intraday_loader):
    # data across the year
    start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York').tz_convert('UTC')
    end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York').tz_convert('UTC')
    df = intraday_loader.load(start, end, 0)
    assert first_last(df, 'AAPL', 'Open') == (157.45, 155.17)
    assert first_last(df, 'MSFT', 'Open') == (101.44, 99.55)
    assert dates_of(df)[0] == start
    assert dates_of(df)[-1] == end

    df = intraday_loader.test_load()
    raw = pd.concat([read_raw(f'AAPL_{year}', '5mins', 'Date') for year in (2018, 2019)])
    assert len(df.loc[(slice(None), 'AAPL'), :]) == len(raw.index.unique())


def test_dividends_and_splits(adjusted_daily_loader):
    df = adjusted_daily_loader.test_load()
    # Two MSFT dividends on 2019-01-11 are summed up, and moved up to the last trading day
    # before the ex-date (MSFT has no data on 2019-01-10).
    assert df.loc[('2019-01-09', 'MSFT'), 'ex-dividend'] == pytest.approx(0.46 + 0.11)
    assert df.loc[('2019-01-11', 'MSFT'), 'ex-dividend'] == 0
    # 1-for-15 reverse split on 2019-01-14 (last non-NaN ratio is used)
    assert df.loc[('2019-01-11', 'MSFT'), 'split_ratio'] == 15
    assert df.loc[('2019-01-14', 'MSFT'), 'split_ratio'] == 1


def expected_msft_adjusted():
    """MSFT adjusted to the price of 2019-01-15."""
    msft = read_raw('MSFT').loc['2019-01-02':'2019-01-15'].dropna()
    split = pd.Series(1.0, index=msft.index)
    split[:'2019-01-11'] = 15
    dividend = pd.Series(1.0, index=msft.index)
    dividend[:'2019-01-09'] = 1 - (0.46 + 0.11) / msft.uClose['2019-01-09']
    return msft.uOpen * split * dividend, msft.uVolume / split


@pytest.mark.filterwarnings('ignore:Warning!! delay_factor is set to False')
def test_adjusted_column(adjusted_daily_loader):
    start = pd.Timestamp('2019-01-02', tz='UTC')
    engine = spectre.factors.FactorEngine(adjusted_daily_loader)
    engine.add(spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.volume), 'vol')
    engine.add(spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.open), 'open')
    df = engine.run(start, END, delay_factor=False)

    expected_open, expected_vol = expected_msft_adjusted()
    assert_almost_equal(values_of(df, 'MSFT', 'open'), expected_open.values, decimal=4)
    assert_almost_equal(values_of(df, 'MSFT', 'vol'), expected_vol.values, decimal=0)
    assert_almost_equal(expected_open.values,
                        [1526.24849, 1548.329113, 1536.244448, 1541.16783, 1563.696033,
                         1585.47827, 1569.750105, 104.9, 103.19], decimal=4)  # fmt: skip

    # AAPL has no dividends and splits in this period
    aapl = read_raw('AAPL').loc['2019-01-02':'2019-01-15']
    assert_almost_equal(values_of(df, 'AAPL', 'open'), aapl.uOpen.values, decimal=4)
    assert_almost_equal(values_of(df, 'AAPL', 'vol'), aapl.uVolume.values, decimal=4)


@pytest.mark.filterwarnings('ignore:Warning!! delay_factor is set to False')
def test_rolling_window_is_adjusted(adjusted_daily_loader):
    """Values in a rolling window are adjusted to the last row of the window."""
    windows = {}

    class LastWindow(spectre.factors.CustomFactor):
        win = 10

        def __init__(self, name, inputs):
            super().__init__(inputs=inputs)
            self.name = name

        def compute(self, data):
            # data.agg gives the adjusted values: [asset, time, window], keep the last window
            windows[self.name] = data.agg(lambda x: x[:, -1]).numpy()
            return data.last()

    engine = spectre.factors.FactorEngine(adjusted_daily_loader)
    engine.add(LastWindow('vol', inputs=[spectre.factors.OHLCV.volume]), 'vol')
    engine.add(LastWindow('open', inputs=[spectre.factors.OHLCV.open]), 'open')
    engine.run(END, END, delay_factor=False)

    expected_open, expected_vol = expected_msft_adjusted()
    aapl = read_raw('AAPL').loc['2019-01-02':'2019-01-15']
    assert_almost_equal(windows['vol'][0], aapl.uVolume.values, decimal=4)
    assert_almost_equal(windows['open'][0], aapl.uOpen.values, decimal=4)
    # MSFT has 9 rows, the last value of the window is a padding
    assert_almost_equal(windows['vol'][1], list(expected_vol) + [np.nan], decimal=0)
    assert_almost_equal(windows['open'][1], list(expected_open) + [np.nan], decimal=4)


@pytest.mark.filterwarnings('ignore:Warning!! delay_factor is set to False')
def test_no_ohlcv():
    start = pd.Timestamp('2019-01-02')
    loader = spectre.data.CsvDirLoader(
        prices_path=str(DATA_DIR / 'daily'),
        earliest_date=start,
        calender_asset='AAPL',
        ohlcv=None,
        adjustments=None,
        prices_index='date',
        parse_dates=True,
    )
    engine = spectre.factors.FactorEngine(loader)
    engine.add(spectre.factors.ColumnDataFactor(inputs=['uOpen']), 'open')
    df = engine.run(start, pd.Timestamp('2019-01-15'), delay_factor=False)
    assert_array_equal(
        values_of(df, 'AAPL', 'open'), read_raw('AAPL').uOpen['2019-01-02':'2019-01-15'].values
    )


def test_align_by_time_loader():
    loader = make_adjusted_daily_loader(calender_asset='AAPL', align_by_time=True)
    df = loader.load(START, END)
    # MSFT has a nan row on 2019-01-10
    assert len(df) == 2 * 10
    assert df.loc[('2019-01-10', 'MSFT'), DAILY_OHLCV].isna().all()
