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
        loader = spectre.data.CsvDirLoader(data_dir + '/daily/')
        self.assertRaisesRegex(ValueError, "df must index by datetime.*",
                               loader.load, '2019-01-01', '2019-01-15', 0)
        loader = spectre.data.CsvDirLoader(data_dir + '/daily/', prices_index='date', )
        self.assertRaisesRegex(ValueError, "df must index by datetime.*",
                               loader.load, '2019-01-01', '2019-01-15', 0)

    def test_csv_loader_value(self):
        loader = spectre.data.CsvDirLoader(
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
        loader = spectre.data.CsvDirLoader(
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
        loader = spectre.data.CsvDirLoader(
            prices_path=data_dir + '/daily/', earliest_date=start.tz_convert(None),
            calender_asset='AAPL',
            dividends_path=data_dir + '/dividends/', splits_path=data_dir + '/splits/',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'), adjustments=('amount', 'ratio'),
            prices_index='date', dividends_index='exDate', splits_index='exDate',
            parse_dates=True, )
        loader.test_load()

        df = loader.load(start, end, 0)

        # test value
        self.assertAlmostEqual(df.loc[('2019-01-09', 'MSFT'), 'ex-dividend'].values[-1], 0.57)

        # test adjustments in engine
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.AdjustedDataFactor(spectre.factors.OHLCV.volume), 'vol')
        engine.add(spectre.factors.AdjustedDataFactor(spectre.factors.OHLCV.open), 'open')
        df = engine.run(start, end, delay_factor=False)

        expected_msft_open = [1526.24849, 1548.329113, 1536.244448, 1541.16783, 1563.696033,
                              1585.47827, 1569.750105, 104.9, 103.19]
        expected_msft_vol = [2947962.0000, 3067160.6000, 2443784.2667, 2176777.6000,
                             2190846.8000, 2018093.5333, 1908511.6000, 28720936.0000, 32882983.0000]
        expected_aapl_open = [155.9200, 147.6300, 148.8400, 148.9000, 150.0000, 157.4400, 154.1000,
                              155.7200, 155.1900, 150.8100]
        expected_aapl_vol = [37932561, 92707401, 59457561, 56974905, 42839940, 45105063,
                             35793075, 28065422, 33834032, 29426699]

        assert_almost_equal(df.loc[(slice(None), 'MSFT'), 'open'], expected_msft_open, decimal=4)
        assert_almost_equal(df.loc[(slice(None), 'AAPL'), 'open'], expected_aapl_open, decimal=4)
        assert_almost_equal(df.loc[(slice(None), 'MSFT'), 'vol'], expected_msft_vol, decimal=0)
        assert_almost_equal(df.loc[(slice(None), 'AAPL'), 'vol'], expected_aapl_vol, decimal=4)

        # rolling adj test
        result = []

        class RollingAdjTest(spectre.factors.CustomFactor):
            win = 10

            def compute(self, data):
                result.append(data.agg(lambda x: x[:, -1]))
                return data.last()

        engine = spectre.factors.FactorEngine(loader)
        engine.add(RollingAdjTest(inputs=[spectre.factors.OHLCV.volume]), 'vol')
        engine.add(RollingAdjTest(inputs=[spectre.factors.OHLCV.open]), 'open')
        engine.run(end, end, delay_factor=False)

        assert_almost_equal(result[0][0], expected_aapl_vol, decimal=4)
        assert_almost_equal(result[0][1], expected_msft_vol+[np.nan], decimal=0)
        assert_almost_equal(result[1][0], expected_aapl_open, decimal=4)
        assert_almost_equal(result[1][1], expected_msft_open+[np.nan], decimal=4)

    def test_no_ohlcv(self):
        start, end = pd.Timestamp('2019-01-02', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        loader = spectre.data.CsvDirLoader(
            prices_path=data_dir + '/daily/', earliest_date=start, calender_asset='AAPL',
            ohlcv=None, adjustments=None,
            prices_index='date',
            parse_dates=True, )
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.DataFactor(inputs=['uOpen']), 'open')
        engine.run(start, end, delay_factor=False)

    @unittest.skipUnless(os.getenv('COVERAGE_RUNNING'), "too slow, run manually")
    def test_yahoo(self):
        yahoo_path = data_dir + '/yahoo/'
        try:
            os.remove(yahoo_path + 'yahoo.feather')
            os.remove(yahoo_path + 'yahoo.feather.meta')
        except FileNotFoundError:
            pass

        spectre.data.YahooDownloader.ingest("2011", yahoo_path, ['IBM', 'AAPL'], skip_exists=False)
        loader = spectre.data.ArrowLoader(yahoo_path + 'yahoo.feather')
        df = loader._load()
        self.assertEqual(['AAPL', 'IBM'], list(df.index.levels[1]))

    @unittest.skipUnless(os.getenv('COVERAGE_RUNNING'), "too slow, run manually")
    def test_QuandlLoader(self):
        quandl_path = data_dir + '../../../historical_data/us/prices/quandl/'
        try:
            os.remove(quandl_path + 'wiki_prices.feather')
            os.remove(quandl_path + 'wiki_prices.feather.meta')
        except FileNotFoundError:
            pass

        spectre.data.ArrowLoader.ingest(
            spectre.data.QuandlLoader(quandl_path + 'WIKI_PRICES.zip'),
            quandl_path + 'wiki_prices.feather'
        )

        loader = spectre.data.ArrowLoader(quandl_path + 'wiki_prices.feather')

        spectre.parallel.Rolling._split_multi = 80
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.MA(100), 'ma')
        engine.to_cuda()
        df = engine.run("2014-01-02", "2014-01-02", delay_factor=False)
        # expected result comes from zipline
        assert_almost_equal(df.head().values.T,
                            [[51.388700, 49.194407, 599.280580, 28.336585, 12.7058]], decimal=4)
        assert_almost_equal(df.tail().values.T,
                            [[86.087988, 3.602880, 7.364000, 31.428209, 27.605950]], decimal=4)

        # test last line bug
        engine.run("2016-12-15", "2017-01-02")
        df = engine._dataframe.loc[(slice('2016-12-15', '2017-12-15'), 'STJ'), :]
        assert df.price_multi.values[-1] == 1

    def test_fast_get(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', prices_index='date', parse_dates=True, )
        df = loader.load()[list(loader.ohlcv)]
        getter = spectre.data.DataLoaderFastGetter(df)

        table = getter.get_as_dict(pd.Timestamp('2018-01-02', tz='UTC'), column_id=3)
        self.assertAlmostEqual(df.loc[("2018-01-02", 'MSFT')].close, table['MSFT'])
        self.assertAlmostEqual(df.loc[("2018-01-02", 'AAPL')].close, table['AAPL'])
        self.assertRaises(KeyError, table.__getitem__, 'A')
        table = dict(table.items())
        self.assertAlmostEqual(df.loc[("2018-01-02", 'MSFT')].close, table['MSFT'])
        self.assertAlmostEqual(df.loc[("2018-01-02", 'AAPL')].close, table['AAPL'])

        table = getter.get_as_dict(pd.Timestamp('2018-01-02', tz='UTC'))
        np.testing.assert_array_almost_equal(df.loc[("2018-01-02", 'MSFT')].values, table['MSFT'])
        np.testing.assert_array_almost_equal(df.loc[("2018-01-02", 'AAPL')].values, table['AAPL'])

        result_df = getter.get_as_df(pd.Timestamp('2018-01-02', tz='UTC'))
        expected = df.xs("2018-01-02")
        pd.testing.assert_frame_equal(expected, result_df)

        table = getter.get_as_dict(pd.Timestamp('2019-01-05', tz='UTC'), column_id=3)
        self.assertTrue(np.isnan(table['MSFT']))
        self.assertRaises(KeyError, table.__getitem__, 'AAPL')

        table = getter.get_as_dict(pd.Timestamp('2019-01-10', tz='UTC'), column_id=3)
        self.assertRaises(KeyError, table.__getitem__, 'MSFT')

        # test 5mins
        loader = spectre.data.CsvDirLoader(
            data_dir + '/5mins/', prices_by_year=True, prices_index='Date', parse_dates=True, )
        df = loader.load()
        getter = spectre.data.DataLoaderFastGetter(df)
        table = getter.get_as_dict(
            pd.Timestamp('2018-12-20 00:00:00+00:00', tz='UTC'),
            pd.Timestamp('2018-12-20 23:59:59+00:00', tz='UTC'),
            column_id=0)
        self.assertTrue(len(table.get_datetime_index().normalize().unique()) == 1)
