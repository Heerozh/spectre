import unittest
import spectre
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from zipline.pipeline.factors import Returns


class TestFactorLib(unittest.TestCase):
    def test_1_CsvDirDataLoader(self):
        # test index type check
        loader = spectre.factors.CsvDirLoader('./data/daily/')
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)
        loader = spectre.factors.CsvDirLoader('./data/daily/', index_col='date', )
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)

        def test_df_first_last(tdf, col, expected_first, expected_last):
            self.assertAlmostEqual(tdf.loc[tdf.index[0], col], expected_first)
            self.assertAlmostEqual(tdf.loc[tdf.index[-1], col], expected_last)

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
        test_df_first_last(df.loc[(slice(None), 'AAPL'), :], 'close', 173.43, 158.09)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'close', 106.57, 105.36)

        # test value
        df = loader.load(start, end, 0)
        test_df_first_last(df.loc[(slice(None), 'AAPL'), :], 'close', 160.35, 158.09)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'close', 103.45, 105.36)
        test_df_first_last(df.loc[(slice('2019-01-11', '2019-01-12'), 'MSFT'), :],
                           'close', 104.5, 104.5)
        start, end = pd.Timestamp('2019-01-11', tz='UTC'), pd.Timestamp('2019-01-12', tz='UTC')
        df = loader.load(start, end, 0)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'close', 104.5, 104.5)

        loader = spectre.factors.CsvDirLoader(
            './data/5mins/', split_by_year=True, index_col='Date', parse_dates=True, )
        start, end = pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        loader.load(start, end, 0)

        start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York')
        end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York')
        df = loader.load(start, end, 0)
        test_df_first_last(df.loc[(slice(None), 'AAPL'), :], 'Open', 157.45, 155.17)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'Open', 101.44, 99.55)

    def test_2_datafactor(self):
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.OHLCV.volume, 'CpVol')
        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'CpVol'].values,
                           (28065422, 33834032, 29426699))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'CpVol'].values,
                           (28627674, 28720936, 32882983))

        engine.add(spectre.factors.DataFactor(inputs=('changePercent',)), 'Chg')
        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'Chg'].values,
                           (-0.9835, -1.5724, 2.1235))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'Chg'].values,
                           (-0.8025, -0.7489, 3.0232))

    def test_3_CustomFactor(self):
        # test backward tree
        a = spectre.factors.CustomFactor(win=2)
        b = spectre.factors.CustomFactor(win=3, inputs=(a,))
        c = spectre.factors.CustomFactor(win=3, inputs=(b,))
        self.assertEqual(c._get_total_backward(), 5)

        a1 = spectre.factors.CustomFactor(win=10)
        a2 = spectre.factors.CustomFactor(win=5)
        b1 = spectre.factors.CustomFactor(win=20, inputs=(a1, a2))
        b2 = spectre.factors.CustomFactor(win=100, inputs=(a2,))
        c1 = spectre.factors.CustomFactor(win=100, inputs=(b1,))
        self.assertEqual(a1._get_total_backward(), 9)
        self.assertEqual(a2._get_total_backward(), 4)
        self.assertEqual(b1._get_total_backward(), 28)
        self.assertEqual(b2._get_total_backward(), 103)
        self.assertEqual(c1._get_total_backward(), 127)

        # test inheritance
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)

        class TestFactor(spectre.factors.CustomFactor):
            inputs = [spectre.factors.OHLCV.close]

            def compute(self, close):
                return np.arange(np.prod(close.shape)).reshape(close.shape)

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = [TestFactor]

            def compute(self, test_input):
                return np.cumsum(test_input)

        engine.add(TestFactor2(), 'test2')
        self.assertRaisesRegex(ValueError, "Length.*",
                               engine.run, '2019-01-11', '2019-01-15')
        engine.remove_all()
        test_f1 = TestFactor()

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = [test_f1]

            def compute(self, test_input):
                return np.cumsum(test_input)

        engine.add(test_f1, 'test1')
        self.assertRaisesRegex(KeyError, ".*exists.*",
                               engine.add, TestFactor(), 'test1')

        engine.add(TestFactor2(), 'test2')
        df = engine.run('2019-01-11', '2019-01-15')
        self.assertEqual(test_f1._cache_hit, 1)
        assert_array_equal(df['test1'].values, [0, 1, 2, 3, 4, 5])
        assert_array_equal(df['test2'].values, [0, 1, 3, 6, 10, 15])

    def test_factors(self):
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', 'AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        total_rows = 10
        engine.add(spectre.factors.OHLCV.high, 'high')
        engine.add(spectre.factors.OHLCV.low, 'low')
        engine.add(spectre.factors.OHLCV.close, 'close')
        df = engine.run('2018-01-01', '2019-01-15')
        df_aapl_close = df.loc[(slice(None), 'AAPL'), 'close']
        df_msft_close = df.loc[(slice(None), 'MSFT'), 'close']
        df_aapl_high = df.loc[(slice(None), 'AAPL'), 'high']
        df_msft_high = df.loc[(slice(None), 'MSFT'), 'high']
        df_aapl_low = df.loc[(slice(None), 'AAPL'), 'low']
        df_msft_low = df.loc[(slice(None), 'MSFT'), 'low']

        def test_expected(factor, _expected_aapl, _expected_msft, _len=10, decimal=7):
            engine.remove_all()
            engine.add(factor, 'test')
            result = engine.run('2019-01-01', '2019-01-15')
            result_aapl = result.loc[(slice(None), 'AAPL'), 'test'].values
            result_msft = result.loc[(slice(None), 'MSFT'), 'test'].values
            assert_almost_equal(result_aapl[-_len:], _expected_aapl[-_len:], decimal=decimal)
            assert_almost_equal(result_msft[-_len:], _expected_msft[-_len:], decimal=decimal)

        # test VWAP
        expected_aapl = [149.0790384, 147.3288365, 149.6858806, 151.9418349,
                         155.9166044, 157.0598718, 157.5146325, 155.9634716]
        expected_msft = [101.6759377, 102.5480467, 103.2112277, 104.2766662,
                         104.7779232, 104.8471192, 104.2296381, 105.3194997]
        test_expected(spectre.factors.VWAP(3), expected_aapl, expected_msft, 8)

        # test AverageDollarVolume
        expected_aapl = [9.44651864548e+09, 1.027077776041e+10, 7.946943447e+09, 7.33979891063e+09,
                         6.43094032063e+09, 5.70460092069e+09, 5.129334268727e+09,
                         4.747847957413e+09]
        expected_msft = [4.25596116072e+09, 4.337221881827e+09, 3.967296370427e+09,
                         3.551354941067e+09, 3.345411315747e+09, 3.206986059747e+09,
                         3.04420074928e+09, 3.167715409797e+09]
        test_expected(spectre.factors.AverageDollarVolume(3), expected_aapl, expected_msft, 8, 2)

        # AnnualizedVolatility
        expected_aapl = [0.3194176, 0.5959649, 0.761103 , 0.8010694, 0.1669258, 0.1869935,
                         0.0980222, 0.5790352, 0.5539618, 0.4551639]
        expected_msft = [0.1916108, 0.2170383, 0.2677953, 0.1348424, 0.2094273, 0.2338434,
                         0.2008854, 0.3119827, 0.2401252, 0.5061202]
        test_expected(spectre.factors.AnnualizedVolatility(3), expected_aapl, expected_msft, 10)

        # test rank
        _expected_aapl = [2.]*10
        _expected_msft = [1]*10
        test_expected(spectre.factors.OHLCV.close.rank(),
                      _expected_aapl, _expected_msft, total_rows)

        # test zscore
        _expected_aapl = [0.707106781, 0.707106781, 0.707106781, 0.707106781, 0.707106781,
                          0.707106781, 0.707106781, 0.707106781, 0.707106781, 0.707106781]
        _expected_msft = -np.array(_expected_aapl)
        test_expected(spectre.factors.OHLCV.close.zscore(),
                      _expected_aapl, _expected_msft, total_rows)

        # test demean
        _expected_aapl = [28.625, 21.965, 23.36, 22.305, 24.175, 25.405, 27.5, 25.245, 26.805,
                          24.045]
        _expected_msft = -np.array(_expected_aapl)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 1}),
                      _expected_aapl, _expected_msft, total_rows)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 2}),
                      [0]*10, [0]*10, total_rows)

        import talib

        # test MA
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=3)
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=3)
        test_expected(spectre.factors.SMA(3), expected_aapl, expected_msft)
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=11)
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=11)
        test_expected(spectre.factors.SMA(11), expected_aapl, expected_msft)

        # test ema
        expected_aapl = talib.EMA(df_aapl_close.values, timeperiod=11)
        expected_msft = talib.EMA(df_msft_close.values, timeperiod=11)
        test_expected(spectre.factors.EMA(11), expected_aapl, expected_msft, decimal=3)
        expected_aapl = talib.EMA(df_aapl_close.values, timeperiod=50)
        expected_msft = talib.EMA(df_msft_close.values, timeperiod=50)
        test_expected(spectre.factors.EMA(50), expected_aapl, expected_msft, decimal=3)

        # test MACD
        expected = talib.MACD(df_aapl_close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        expected_aapl_signal = expected[1][-total_rows:]
        expected_aapl_normal = expected[2][-total_rows:]
        expected = talib.MACD(df_msft_close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        expected_msft_signal = expected[1][-total_rows:]
        expected_msft_normal = expected[2][-total_rows:]
        test_expected(spectre.factors.MACD(), expected_aapl_signal, expected_msft_signal, decimal=3)
        test_expected(spectre.factors.MACD().normalized(), expected_aapl_normal,
                      expected_msft_normal, decimal=3)
        #  #
        expected = talib.MACD(df_aapl_close.values, fastperiod=10, slowperiod=15, signalperiod=5)
        expected_aapl_signal = expected[1][-total_rows:]
        expected = talib.MACD(df_msft_close.values, fastperiod=10, slowperiod=15, signalperiod=5)
        expected_msft_signal = expected[1][-total_rows:]
        test_expected(spectre.factors.MACD(10, 15, 5), expected_aapl_signal, expected_msft_signal,
                      decimal=3)

        # test BBANDS
        expected = talib.BBANDS(df_aapl_close.values, timeperiod=20)
        expected_aapl_normal = (df_aapl_close.values - expected[1]) / (expected[0] - expected[1])
        expected = talib.BBANDS(df_msft_close.values, timeperiod=20)
        expected_msft_normal = (df_msft_close.values - expected[1]) / (expected[0] - expected[1])
        test_expected(spectre.factors.BBANDS(), expected_aapl_normal, expected_msft_normal)
        expected = talib.BBANDS(df_aapl_close.values, timeperiod=50, nbdevup=3, nbdevdn=3)
        expected_aapl_normal = (df_aapl_close.values - expected[1]) / (expected[0] - expected[1])
        expected = talib.BBANDS(df_msft_close.values, timeperiod=50, nbdevup=3, nbdevdn=3)
        expected_msft_normal = (df_msft_close.values - expected[1]) / (expected[0] - expected[1])
        test_expected(spectre.factors.BBANDS(win=50, inputs=(spectre.factors.OHLCV.close, 3)),
                      expected_aapl_normal, expected_msft_normal)

        # test TRANGE
        expected_aapl = talib.TRANGE(df_aapl_high.values, df_aapl_low.values, df_aapl_close.values)
        expected_msft = talib.TRANGE(df_msft_high.values, df_msft_low.values, df_msft_close.values)
        test_expected(spectre.factors.TRANGE(), expected_aapl, expected_msft)

        # test rsi
        # expected_aapl = talib.RSI(df_aapl_close.values, timeperiod=14)
        expected_aapl = [46.7915095, 46.2348644, 46.6612279, 46.4354528, 46.5263158,
                         47.2506481, 47.4633086, 46.6735717, 47.1775162, 47.3809596]
        # expected_msft = talib.RSI(df_msft_close.values, timeperiod=14)
        expected_msft = [49.4040005, 49.8240779, 49.5206631, 50.3420499, 50.0310697,
                         49.9674093, 50.225292 , 50.3566546, 50.1283889, 50.4479649]
        # expected_aapl += 7
        test_expected(spectre.factors.RSI(), expected_aapl, expected_msft)


        # todo 测试是否已算过的重复factor不会算2遍
        # todo 测试错位嵌套factor

        # test cuda result eq cup

    def test_filter_factor(self):
        print('test filter:')
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        universe = spectre.factors.OHLCV.volume.top(1)
        engine.add(spectre.factors.OHLCV.volume, 'not_used')
        engine.set_filter(universe)

        result = engine.run("2019-01-01", "2019-01-15")
        assert_array_equal(result.index.get_level_values(1).values,
                           ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT',
                            'AAPL', 'MSFT'])

        # test ma5 with filter
        import talib
        total_rows = 10
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', 'AAPL', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        # get filtered ma5
        engine = spectre.factors.FactorEngine(loader)
        universe = spectre.factors.OHLCV.volume.top(1)
        engine.set_filter(universe)
        engine.add(spectre.factors.SMA(5), 'ma5')
        df = engine.run('2019-01-01', '2019-01-15')
        result_aapl = df.loc[(slice(None), 'AAPL'), 'ma5'].values
        result_msft = df.loc[(slice(None), 'MSFT'), 'ma5'].values
        # get not filtered close value
        engine.remove_all()
        engine.set_filter(None)
        engine.add(spectre.factors.OHLCV.close, 'c')
        df = engine.run('2018-01-01', '2019-01-15')
        df_aapl_close = df.loc[(slice(None), 'AAPL'), 'c']
        df_msft_close = df.loc[(slice(None), 'MSFT'), 'c']
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=5)[-total_rows:]
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=5)[-total_rows:]
        expected_aapl = np.delete(expected_aapl, [7, 9])
        expected_msft = [expected_msft[7], expected_msft[9]]
        # test
        assert_almost_equal(result_aapl, expected_aapl)
        assert_almost_equal(result_msft, expected_msft)


    # def test_cuda_factors(self):
    #     spectre.to_cuda()
    #     self.test_factors()
    #     pass


if __name__ == '__main__':
    unittest.main()
