import unittest
import spectre
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal


class TestDataLoaderLib(unittest.TestCase):
    def _assertDFFirstLastEqual(self, tdf, col, expected_first, expected_last):
        self.assertAlmostEqual(tdf.loc[tdf.index[0], col], expected_first)
        self.assertAlmostEqual(tdf.loc[tdf.index[-1], col], expected_last)

    def test_required_parameters(self):
        loader = spectre.factors.CsvDirLoader('./data/daily/')
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)
        loader = spectre.factors.CsvDirLoader('./data/daily/', index_col='date', )
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)

    def test_csv_loader_value(self):
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', 'AAPL', index_col='date', parse_dates=True, )
        # test cache
        start, end = pd.Timestamp('2015-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        self.assertIsNone(loader._load_from_cache(start, end, 0))
        start, end = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        df = loader.load(start, end, 0)
        self.assertIsNotNone(loader._load_from_cache(start, end, 0))

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

    def test_csv_split_loader_value(self):
        loader = spectre.factors.CsvDirLoader(
            './data/5mins/', split_by_year=True, index_col='Date', parse_dates=True, )
        start, end = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        loader.load(start, end, 0)

        start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York').tz_convert('UTC')
        end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York').tz_convert('UTC')
        df = loader.load(start, end, 0)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'AAPL'), :], 'Open', 157.45, 155.17)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'Open', 101.44, 99.55)

    @unittest.skipUnless(False, "too slow, run manually")
    def test_QuandlLoader(self):
        loader = spectre.factors.QuandlLoader(
            '../../historical_data/us/prices/quandl/WIKI_PRICES.zip')
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.MA(100), 'ma')
        engine.to_cuda()
        df = engine.run("2014-01-02", "2014-01-02")
        # expected result comes from zipline
        # AAOI only got 68 tick, so it's nan
        assert_almost_equal(df.head().values.T,
                            [[51.388700, 49.194407, 599.280580, 28.336585, np.nan]], decimal=4)
        assert_almost_equal(df.tail().values.T,
                            [[86.087988, 3.602880, 7.364000, 31.428209, 27.605950]], decimal=4)


