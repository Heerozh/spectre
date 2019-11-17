import unittest
import spectre
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal


class TestFactorLib(unittest.TestCase):

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

        def test_expected(factor, _expected_aapl, _expected_msft, _len=9, decimal=7):
            engine.remove_all_factors()
            engine.add(factor, 'test')
            result = engine.run('2019-01-01', '2019-01-15')
            result_aapl = result.loc[(slice(None), 'AAPL'), 'test'].values
            result_msft = result.loc[(slice(None), 'MSFT'), 'test'].values
            assert_almost_equal(result_aapl[-_len:], _expected_aapl[-_len:], decimal=decimal)
            assert_almost_equal(result_msft[-_len:], _expected_msft[-_len:], decimal=decimal)

        # test select no trading day
        engine.add(spectre.factors.SMA(2), 'test')
        self.assertRaisesRegex(AssertionError, "There is no data between.*",
                               engine.run, '2019-01-01', '2019-01-01')
        # test remove unused level bug:
        self.assertRaises(KeyError, engine._dataframe.index.levels[0].get_loc,
                          '2019-01-05 00:00:00+00:00')

        # test VWAP
        expected_aapl = [149.0790384, 147.3288365, 149.6858806, 151.9418349,
                         155.9166044, 157.0598718, 157.5146325, 155.9634716]
        expected_msft = [102.5066541, 102.5480467, 103.2112277, 104.2766662,
                         104.7779232, 104.8471192, 104.2296381, 105.3194997]
        test_expected(spectre.factors.VWAP(3), expected_aapl, expected_msft, 8)

        # test AverageDollarVolume
        expected_aapl = [9.44651864548e+09, 1.027077776041e+10, 7.946943447e+09, 7.33979891063e+09,
                         6.43094032063e+09, 5.70460092069e+09, 5.129334268727e+09,
                         4.747847957413e+09]
        expected_msft = [4222040618.907, 4337221881.827, 3967296370.427, 3551354941.067,
                         3345411315.747, 3206986059.747, 3044200749.280, 3167715409.797]
        test_expected(spectre.factors.AverageDollarVolume(3), expected_aapl, expected_msft, 8, 2)

        # AnnualizedVolatility
        expected_aapl = [0.3141548, 0.5426118, 0.7150832, 0.7475805, 0.1710541, 0.1923727,
                         0.1027987, 0.5697543, 0.5436627, 0.4423527]
        expected_msft = [0.189534377, 0.263729893, 0.344381405, 0.210997343, 0.235832738,
                         0.202266499, 0.308870901, 0.235088127, 0.520421161, ]
        test_expected(spectre.factors.AnnualizedVolatility(3), expected_aapl, expected_msft, 10)

        # test rank
        _expected_aapl = [2.]*10
        _expected_aapl[6] = 1  # because msft was nan this day
        _expected_msft = [1]*9
        test_expected(spectre.factors.OHLCV.close.rank(),
                      _expected_aapl, _expected_msft, total_rows)
        _expected_aapl = [1.]*10
        _expected_msft = [2]*9
        test_expected(spectre.factors.OHLCV.close.rank(ascending=False),
                      _expected_aapl, _expected_msft, total_rows)
        # test rank bug #98a0bdc
        engine.remove_all_factors()
        engine.add(spectre.factors.OHLCV.close.rank(), 'test')
        result = engine.run('2019-01-01', '2019-01-02')
        assert_array_equal([[2.0], [1.0]], result.values)

        # test zscore
        _expected_aapl = [1.]*10
        # aapl has prices data, but we only have two stocks, so one data zscore = 0/0 = nan
        _expected_aapl[6] = np.nan
        _expected_msft = [-1.]*9
        test_expected(spectre.factors.OHLCV.close.zscore(),
                      _expected_aapl, _expected_msft, total_rows)

        # test demean
        _expected_aapl = [28.655, 21.475, 22.305, 22.9, 23.165, 25.015, 0, 25.245, 26.805, 24.045]
        _expected_msft = -np.array(_expected_aapl)
        _expected_msft = np.delete(_expected_msft, 6)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 1}),
                      _expected_aapl, _expected_msft, total_rows, decimal=3)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 2}),
                      [0]*10, [0]*9, total_rows)
        test_expected(spectre.factors.OHLCV.close.demean(),
                      _expected_aapl, _expected_msft, total_rows)

        import talib  # pip install --no-deps d:\doc\Download\TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl

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
        test_expected(spectre.factors.EMA(50), expected_aapl, expected_msft, decimal=2)

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
        # calculate at excel
        expected_aapl = [40.1814301, 33.36385487, 37.37511353, 36.31220413, 41.84100418,
                         39.19197118, 48.18441452, 44.30411404, 50.05167959, 56.47230321]
        # expected_msft = talib.RSI(df_msft_close.values, timeperiod=14)
        expected_msft = [38.5647217, 42.0627596, 37.9693676, 43.8641553, 48.3458438,
                         47.095672 , 46.7363662, 46.127465 , 64.8259304]
        # expected_aapl += 7
        test_expected(spectre.factors.RSI(), expected_aapl, expected_msft)

        # test stochf
        expected_aapl = talib.STOCHF(df_aapl_high.values, df_aapl_low.values, df_aapl_close.values,
                                     fastk_period=14)[0]
        expected_msft = talib.STOCHF(df_msft_high.values, df_msft_low.values, df_msft_close.values,
                                     fastk_period=14)[0]
        test_expected(spectre.factors.STOCHF(), expected_aapl, expected_msft)

        # test same factor only compute once, and nest factor window
        f1 = spectre.factors.BBANDS(win=20, inputs=[spectre.factors.OHLCV.close, 2])
        f2 = spectre.factors.EMA(win=10, inputs=[f1])
        fa = spectre.factors.STDDEV(win=15, inputs=[f2])
        fb = spectre.factors.MACD(12, 26, 9, inputs=[f2])
        engine.remove_all_factors()
        engine.add(f2, 'f2')
        engine.add(fa, 'fa')
        engine.add(fb, 'fb')
        result = engine.run('2019-01-01', '2019-01-15')
        self.assertEqual(f2._cache_hit, 3)

        # test cuda result eq cup

    def test_filter_factor(self):
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
                           ['MSFT', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT',
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
        engine.remove_all_factors()
        engine.set_filter(None)
        engine.add(spectre.factors.OHLCV.close, 'c')
        df = engine.run('2018-01-01', '2019-01-15')
        df_aapl_close = df.loc[(slice(None), 'AAPL'), 'c']
        df_msft_close = df.loc[(slice(None), 'MSFT'), 'c']
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=5)[-total_rows:]
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=5)[-total_rows:]
        expected_aapl = np.delete(expected_aapl, [0, 7, 9])
        expected_msft = [expected_msft[1], expected_msft[7], expected_msft[9]]
        # test
        assert_almost_equal(result_aapl, expected_aapl)
        assert_almost_equal(result_msft, expected_msft)

    def test_cuda(self):
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.to_cuda()
        universe = spectre.factors.OHLCV.volume > 1
        f1 = spectre.factors.OHLCV.volume + 1
        f2 = f1 + 1
        engine.add(f2, 'f2')
        engine.add(spectre.factors.OHLCV.volume + 2, 'fv')
        engine.set_filter(universe)
        result = engine.run("2019-01-01", "2019-01-15")

        assert_array_equal(result.f2, result.fv)

