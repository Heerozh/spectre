import unittest
import spectre
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_equal
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestFactorLib(unittest.TestCase):

    def test_factors(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', 'AAPL',
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
        _expected_aapl = [2.] * 10
        _expected_aapl[6] = 1  # because msft was nan this day
        _expected_msft = [1] * 9
        test_expected(spectre.factors.OHLCV.close.rank(),
                      _expected_aapl, _expected_msft, total_rows)
        _expected_aapl = [1.] * 10
        _expected_msft = [2] * 9
        test_expected(spectre.factors.OHLCV.close.rank(ascending=False),
                      _expected_aapl, _expected_msft, total_rows)
        # test rank bug #98a0bdc
        engine.remove_all_factors()
        engine.add(spectre.factors.OHLCV.close.rank(), 'test')
        result = engine.run('2019-01-01', '2019-01-02')
        assert_array_equal([[2.0], [1.0]], result.values)
        # test rank with filter
        engine.remove_all_factors()
        engine.set_filter(spectre.factors.OHLCV.volume.top(1))
        engine.add(spectre.factors.OHLCV.close.rank(), 'test')
        result = engine.run('2019-01-01', '2019-01-15')
        assert_array_equal([1.] * 10, result.test.values)
        engine.set_filter(None)

        # test zscore
        _expected_aapl = [1.] * 10
        # aapl has prices data, but we only have two stocks, so one data zscore = 0/0 = nan
        _expected_aapl[6] = np.nan
        _expected_msft = [-1.] * 9
        test_expected(spectre.factors.OHLCV.close.zscore(),
                      _expected_aapl, _expected_msft, total_rows)

        # test demean
        _expected_aapl = [28.655, 21.475, 22.305, 22.9, 23.165, 25.015, 0, 25.245, 26.805, 24.045]
        _expected_msft = -np.array(_expected_aapl)
        _expected_msft = np.delete(_expected_msft, 6)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 1}),
                      _expected_aapl, _expected_msft, total_rows, decimal=3)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 2}),
                      [0] * 10, [0] * 9, total_rows)
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
                         47.095672, 46.7363662, 46.127465, 64.8259304]
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
            data_dir + 'daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
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
            data_dir + '/daily/', 'AAPL', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
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
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
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

    def test_quantile(self):
        f = spectre.factors.QuantileFactor()
        f.bins = 5
        import torch
        result = f.compute(torch.tensor([[1, 1, np.nan, 1.01, 1.01, 2],
                                         [3, 4, 5, 1.01, np.nan, 1.01]]))
        expected = [[1, 1, np.nan, 3, 3, 5], [3, 4, 5, 1, np.nan, 1]]
        assert_array_equal(result, expected)

        result = f.compute(torch.tensor([[-1, 1, np.nan, 1.01, 1.02, 2],
                                         [3, -4, 5, 1.01, np.nan, 1.01]]))
        expected = [[1, 2, np.nan, 3, 4, 5], [4, 1, 5, 2, np.nan, 2]]
        assert_array_equal(result, expected)

        data = [[-1.01318216e+00, -6.03849769e-01, -1.57474554e+00, -1.72021079e+00,
                 -9.00418401e-01, -1.26915586e+00, -4.82064962e-01, -1.55332041e+00,
                 -1.37628138e+00, -1.06167054e+00, -8.49674761e-01, -6.39934182e-01,
                 -1.39206827e+00, -1.70104098e+00, -7.75250673e-01, -5.85807621e-01,
                 -7.69612491e-01, -1.22405028e+00, -1.21277392e+00, -1.67059469e+00,
                 4.44852918e-01, -8.59823465e-01, -7.45932102e-01, -9.70331907e-01,
                 -2.32857108e-01, -1.62887216e+00, 6.21891975e-01, 1.58714950e+00,
                 -1.68750930e+00, -1.59617066e+00, -1.58376670e+00, -1.37289846e+00,
                 -1.71457255e+00, -3.32089186e-01, 1.39545119e+00, -1.50032151e+00,
                 -1.42928028e+00, -1.48791742e+00, -1.43830144e+00, -1.58489430e+00,
                 -1.46310949e+00, 1.50595963e+00, 1.15751970e+00, 5.74531198e-01,
                 -1.60744703e+00, -7.98931062e-01, 5.79041779e-01, -1.45408833e+00,
                 -1.71682787e+00, -1.64353144e+00, 7.47059762e-01, -1.23307145e+00],
                [-1.01656508e+00, -6.47827625e-01, -1.57361794e+00, -1.71908307e+00,
                 -9.08311903e-01, -1.27141106e+00, -4.88830775e-01, -1.55332041e+00,
                 -1.36726034e+00, -1.05941534e+00, -8.50802362e-01, -6.41061842e-01,
                 -1.39432359e+00, -1.70104098e+00, -7.70740151e-01, -5.82424700e-01,
                 -7.74123013e-01, -1.22517800e+00, -1.21615684e+00, -1.67059469e+00,
                 4.38087106e-01, -8.59823465e-01, -7.44804442e-01, -9.72587228e-01,
                 -1.08196807e+00, -1.08084035e+00, -1.40447235e+00, -1.38981307e+00,
                 -7.05337167e-01, -1.06279814e+00, -1.65931833e+00, -1.12707353e+00,
                 8.13590348e-01, -7.12103009e-01, -4.07640904e-01, -1.39206827e+00,
                 6.46700025e-01, -1.86623976e-01, -1.67848814e+00, -1.69145607e-03,
                 -1.54880989e+00, -6.03285991e-02, -6.99698985e-01, -1.53753352e+00,
                 1.04137313e+00, -1.17894483e+00, -5.27170479e-01, -1.33455884e+00,
                 -1.50483203e+00, -1.50595963e+00, 1.53978884e+00, -2.41878211e-01]]
        result = f.compute(torch.tensor(data))
        expected = pd.qcut(data[1], 5, labels=False) + 1
        assert_array_equal(result[-1], expected)
