import unittest
import spectre
import os
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestDataLoaderLib(unittest.TestCase):
    def _assertDFFirstLastEqual(self, tdf, col, expected_first, expected_last):
        self.assertAlmostEqual(tdf.loc[tdf.index[0], col], expected_first)
        self.assertAlmostEqual(tdf.loc[tdf.index[-1], col], expected_last)

    def test_required_parameters(self):
        loader = spectre.factors.CsvDirLoader(data_dir + '/daily/')
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)
        loader = spectre.factors.CsvDirLoader(data_dir + '/daily/', prices_index='date', )
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)

    def test_csv_loader_value(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL', prices_index='date', parse_dates=True, )
        start, end = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')

        # test backward
        df = loader.load(start, end, 11)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'AAPL'), :], 'close', 173.43, 158.09)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'close', 106.57, 105.36)

        # test value
        df = loader.load(start, end, 0)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'AAPL'), :], 'close', 160.35, 158.09)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'close', 100.1, 105.36)
        self._assertDFFirstLastEqual(df.loc[(slice('2019-01-11', '2019-01-12'), 'MSFT'), :],
                                     'close', 104.5, 104.5)
        start, end = pd.Timestamp('2019-01-11', tz='UTC'), pd.Timestamp('2019-01-12', tz='UTC')
        df = loader.load(start, end, 0)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'close', 104.5, 104.5)

        loader.test_load()

    def test_csv_split_loader_value(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/5mins/', prices_by_year=True, prices_index='Date', parse_dates=True, )
        start = pd.Timestamp('2019-01-02 14:30:00', tz='UTC')
        end = pd.Timestamp('2019-01-15', tz='UTC')
        loader.load(start, end, 0)

        start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York').tz_convert('UTC')
        end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York').tz_convert('UTC')
        df = loader.load(start, end, 0)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'AAPL'), :], 'Open', 157.45, 155.17)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'Open', 101.44, 99.55)

        loader.test_load()

    def test_csv_div_split(self):
        start, end = pd.Timestamp('2019-01-02', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        loader = spectre.factors.CsvDirLoader(
            prices_path=data_dir + '/daily/', earliest_date=start,
            dividends_path=data_dir + '/dividends/', splits_path=data_dir + '/splits/',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'), adjustments=('amount', 'ratio'),
            prices_index='date', dividends_index='exDate', splits_index='exDate',
            parse_dates=True, )
        loader.test_load()

        df = loader.load(start, end, 0)

        # test value
        self.assertAlmostEqual(df.loc[('2019-01-11', 'MSFT'), 'ex-dividend'].values[-1], 0.57)

    @unittest.skipUnless(os.getenv('COVERAGE_RUNNING'), "too slow, run manually")
    def test_QuandlLoader(self):
        quandl_path = data_dir + '../../../historical_data/us/prices/quandl/'
        try:
            os.remove(quandl_path + 'wiki_prices.feather')
            os.remove(quandl_path + 'wiki_prices.feather.meta')
        except FileNotFoundError:
            pass

        spectre.factors.ArrowLoader.ingest(
            spectre.factors.QuandlLoader(quandl_path + 'WIKI_PRICES.zip'),
            quandl_path + 'wiki_prices.feather'
        )

        loader = spectre.factors.ArrowLoader(quandl_path + 'wiki_prices.feather')

        spectre.parallel.Rolling._split_multi = 80
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.MA(100), 'ma')
        engine.to_cuda()
        df = engine.run("2014-01-02", "2014-01-02", delay_factor=False)
        # expected result comes from zipline
        # AAOI only got 68 tick, so it's nan
        assert_almost_equal(df.head().values.T,
                            [[51.388700, 49.194407, 599.280580, 28.336585, np.nan]], decimal=4)
        assert_almost_equal(df.tail().values.T,
                            [[86.087988, 3.602880, 7.364000, 31.428209, 27.605950]], decimal=4)

        # test last line bug
        df = engine.run("2016-12-15", "2017-01-02")
        df = engine._dataframe.loc[(slice('2016-12-15', '2017-12-15'), 'STJ'), :]
        assert df.price_multi.values[-1] == 1
