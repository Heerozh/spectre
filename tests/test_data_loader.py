import unittest
import spectre
import pandas as pd
import numpy as np


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

        start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York').tz_convert('UTC')
        end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York').tz_convert('UTC')
        df = loader.load(start, end, 0)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'AAPL'), :], 'Open', 157.45, 155.17)
        self._assertDFFirstLastEqual(df.loc[(slice(None), 'MSFT'), :], 'Open', 101.44, 99.55)

    def test_adjust(self):
        df = pd.read_csv('./data/adjustment.csv',
                         parse_dates=['date'], index_col=['date', 'asset'],)
        dl = spectre.factors.DataLoader(ohlcv=('open', 'high', 'low', 'close', 'volume'))
        df = dl._adjust_prices(df).loc[(slice(None), 'CMCSA'), :]

        expected = np.array([
            [37.337898, 37.482291, 37.243296, 37.417563, 12684104.0],
            [37.442458, 37.701369, 37.392668, 37.696390, 20421818.0],
            [37.641620, 37.955300, 37.566934, 37.885594, 15477494.0],
            [37.716306, 38.169399, 37.686432, 38.137036, 13849182.0],
            [38.064839, 38.064839, 37.681453, 37.835803, 16599476.0],
            [37.810908, 37.860698, 37.352835, 37.502207, 19599952.0],
            [37.870656, 37.870656, 37.253254, 37.731243, 13556274.0],
            [37.711327, 38.059860, 37.661536, 37.781034, 13126051.0],
            [37.960279, 38.000112, 37.402626, 37.492249, 13258356.0],
            [37.561955, 37.741201, 37.313003, 37.731243, 11557826.0],
        ])
        np.testing.assert_almost_equal(df.values[:10], expected, decimal=5)


