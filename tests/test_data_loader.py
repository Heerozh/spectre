import unittest
import spectre
import pandas as pd


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
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'close', 103.45, 105.36)
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

        start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York')
        end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York')
        df = loader.load(start, end, 0)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'AAPL'), :], 'Open', 157.45, 155.17)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'Open', 101.44, 99.55)