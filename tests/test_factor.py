import unittest
import spectre
import os
import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_almost_equal, assert_array_equal
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestMultiProcessing(spectre.factors.CPUParallelFactor):

    @staticmethod
    def mp_compute(a, b) -> np.array:
        return (a * b).mean(axis=0).values


class TestFactorLib(unittest.TestCase):

    def test_factors(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
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
        engine.remove_all_factors()
        engine.add(spectre.factors.OHLCV.open, 'open')
        df = engine.run('2018-01-01', '2019-01-15', False)
        df_aapl_open = df.loc[(slice(None), 'AAPL'), 'open']
        df_msft_open = df.loc[(slice(None), 'MSFT'), 'open']

        def test_expected(factor, _expected_aapl, _expected_msft, _len=8, decimal=7,
                          delay=True, check_bias=True):
            engine.remove_all_factors()
            engine.add(factor, 'test')
            _result = engine.run('2019-01-01', '2019-01-15', delay)
            if check_bias:
                engine.test_lookahead_bias('2019-01-01', '2019-01-15')
            result_aapl = _result.loc[(slice(None), 'AAPL'), 'test'].values
            result_msft = _result.loc[(slice(None), 'MSFT'), 'test'].values
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
        expected_aapl = [152.3003451, 149.0790384, 147.3288365, 149.6858806,
                         151.9418349, 155.9166044, 157.0598718, 157.5146325]
        expected_msft = [103.3029256, 102.5066541, 102.5480467, 103.2112277,
                         104.2766662, 104.7779232, 104.8471192, 104.2296381]
        test_expected(spectre.factors.VWAP(3), expected_aapl, expected_msft, 8)

        # test AverageDollarVolume
        expected_aapl = [8.48373212946e+09, 9.44651864548e+09, 1.027077776041e+10, 7.946943447e+09,
                         7.33979891063e+09, 6.43094032063e+09, 5.70460092069e+09,
                         5.129334268727e+09]
        expected_msft = [4022824628.837, 4222040618.907, 4337221881.827, 3967296370.427,
                         3551354941.067, 3345411315.747, 3206986059.747, 3044200749.280]
        test_expected(spectre.factors.AverageDollarVolume(3), expected_aapl, expected_msft, 8, 2)

        # AnnualizedVolatility
        expected_aapl = [0.3141548, 0.5426118, 0.7150832, 0.7475805, 0.1710541,
                         0.1923727, 0.1027987, 0.5697543, 0.5436627]
        expected_msft = [0.189534377, 0.263729893, 0.344381405, 0.210997343,
                         0.235832738, 0.202266499, 0.308870901, 0.235088127]
        test_expected(spectre.factors.AnnualizedVolatility(3), expected_aapl, expected_msft, 10)

        # test LogReturn
        expected_aapl = np.log(df_aapl_close) - np.log(df_aapl_close.shift(1))
        expected_msft = np.log(df_msft_close) - np.log(df_msft_close.shift(1))
        expected_aapl = expected_aapl[-9:]
        expected_msft = expected_msft[-8:]
        test_expected(spectre.factors.LogReturns(), expected_aapl, expected_msft, 10)

        # test rank
        _expected_aapl = [2.] * 9
        _expected_aapl[6] = 1  # because msft was nan this day
        _expected_msft = [1] * 8
        test_expected(spectre.factors.OHLCV.close.rank(),
                      _expected_aapl, _expected_msft, total_rows)
        _expected_aapl = [1.] * 9
        _expected_msft = [2] * 8
        test_expected(spectre.factors.OHLCV.close.rank(ascending=False),
                      _expected_aapl, _expected_msft, total_rows)
        # test rank bug #98a0bdc
        engine.remove_all_factors()
        engine.add(spectre.factors.OHLCV.close.rank(), 'test')
        result = engine.run('2019-01-01', '2019-01-03')
        assert_array_equal([[2.0], [1.0]], result.values)
        # test rank with filter cuda
        engine.remove_all_factors()
        engine.set_filter(spectre.factors.OHLCV.volume.top(1))
        engine.add(spectre.factors.OHLCV.close.rank(mask=engine.get_filter()), 'test')
        engine.to_cuda()
        result = engine.run('2019-01-01', '2019-01-15')
        assert_array_equal([1.] * 9, result.test.values)
        # test rank with filter
        engine.remove_all_factors()
        engine.set_filter(spectre.factors.OHLCV.volume.top(1))
        engine.add(spectre.factors.OHLCV.close.rank(mask=engine.get_filter()), 'test')
        engine.to_cpu()
        result = engine.run('2019-01-01', '2019-01-15')
        assert_array_equal([1.] * 9, result.test.values)
        engine.set_filter(None)

        # test zscore
        expected_aapl = [1.] * 9
        # aapl has prices data, but we only have two stocks, so one data zscore = 0/0 = nan
        expected_aapl[6] = np.nan
        expected_msft = [-1.] * 8
        test_expected(spectre.factors.OHLCV.close.zscore(),
                      expected_aapl, expected_msft, total_rows)

        # test demean
        expected_aapl = [28.655, 21.475, 22.305, 22.9, 23.165, 25.015, 0, 25.245, 26.805]
        expected_msft = -np.array(expected_aapl)
        expected_msft = np.delete(expected_msft, 6)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 1}),
                      expected_aapl, expected_msft, total_rows, decimal=3)
        test_expected(spectre.factors.OHLCV.close.demean(groupby={'AAPL': 1, 'MSFT': 2}),
                      [0] * 9, [0] * 8, total_rows)
        test_expected(spectre.factors.OHLCV.close.demean(),
                      expected_aapl, expected_msft, total_rows)

        # test shift
        expected_aapl = df_aapl_close.shift(2)[-total_rows + 1:]
        expected_aapl[0:2] = np.nan
        expected_msft = df_msft_close.shift(2)[-total_rows + 2:]
        expected_msft[0:2] = np.nan
        test_expected(spectre.factors.OHLCV.close.shift(2),
                      expected_aapl, expected_msft, total_rows)

        expected_aapl = [149, 149, 151.55, 156.03, 161, 153.69, 157, 156.94, np.nan]
        expected_msft = [104.39, 103.2, 105.22, 106, 103.2, 103.39, 108.85, np.nan]
        # bcuz delay=True, so it is shift(-2) and then shift(1)
        self.assertRaisesRegex(RuntimeError, '.*look-ahead bias.*', test_expected,
                               spectre.factors.OHLCV.close.shift(-2), expected_aapl, expected_msft,
                               total_rows)
        test_expected(spectre.factors.OHLCV.close.shift(-2),
                      expected_aapl, expected_msft, total_rows, check_bias=False)

        # test zscore and shift, mask bug
        expected_aapl = [1.] * 10
        expected_aapl[0] = np.nan
        expected_aapl[7] = np.nan
        expected_msft = [-1.] * 9
        expected_msft[0] = np.nan
        engine.set_filter(spectre.factors.OHLCV.open.top(2))
        test_expected(spectre.factors.OHLCV.open.zscore().shift(1),
                      expected_aapl, expected_msft, total_rows, delay=False)

        # test quantile get factors from engine._factors nan bug
        expected_aapl = [4.] * 9
        expected_aapl[6] = np.nan
        expected_msft = [0.] * 8
        engine.set_filter(spectre.factors.OHLCV.close.top(2))
        engine.add(spectre.factors.OHLCV.close.zscore(), 'pre')
        f = engine.get_factor('pre')
        test_expected(f.quantile(), expected_aapl, expected_msft, total_rows)

        import talib  # pip install --no-deps d:\doc\Download\TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl

        # test MA
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=3)
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=3)
        test_expected(spectre.factors.SMA(3), expected_aapl, expected_msft)
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=11)
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=11)
        test_expected(spectre.factors.SMA(11), expected_aapl, expected_msft)
        with self.assertWarns(RuntimeWarning):
            engine.run('2019-01-01', '2019-01-15', False)

        # test open ma not shift
        expected_aapl = talib.SMA(df_aapl_open.values, timeperiod=11)
        expected_msft = talib.SMA(df_msft_open.values, timeperiod=11)
        test_expected(spectre.factors.SMA(11, inputs=[spectre.factors.OHLCV.open]),
                      expected_aapl, expected_msft, _len=9, delay=False)

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
        expected_aapl = talib.BBANDS(df_aapl_close.values, timeperiod=20)
        expected_msft = talib.BBANDS(df_msft_close.values, timeperiod=20)
        test_expected(spectre.factors.BBANDS()[0], expected_aapl[0], expected_msft[0])
        # normalized
        expected_aapl_normal = (df_aapl_close.values - expected_aapl[1]) /\
                               (expected_aapl[0] - expected_aapl[1])
        expected_msft_normal = (df_msft_close.values - expected_msft[1]) / \
                               (expected_msft[0] - expected_msft[1])
        test_expected(spectre.factors.BBANDS().normalized(), expected_aapl_normal,
                      expected_msft_normal)

        expected = talib.BBANDS(df_aapl_close.values, timeperiod=50, nbdevup=3, nbdevdn=3)
        expected_aapl_normal = (df_aapl_close.values - expected[1]) / (expected[0] - expected[1])
        expected = talib.BBANDS(df_msft_close.values, timeperiod=50, nbdevup=3, nbdevdn=3)
        expected_msft_normal = (df_msft_close.values - expected[1]) / (expected[0] - expected[1])
        test_expected(spectre.factors.BBANDS(win=50, inputs=(spectre.factors.OHLCV.close, 3))
                      .normalized(),
                      expected_aapl_normal, expected_msft_normal)

        # test TRANGE
        expected_aapl = talib.TRANGE(df_aapl_high.values, df_aapl_low.values, df_aapl_close.values)
        expected_msft = talib.TRANGE(df_msft_high.values, df_msft_low.values, df_msft_close.values)
        test_expected(spectre.factors.TRANGE(), expected_aapl, expected_msft)

        # test rsi
        # expected_aapl = talib.RSI(df_aapl_close.values, timeperiod=14)
        # calculate at excel
        expected_aapl = [40.1814301, 33.36385487, 37.37511353, 36.31220413, 41.84100418,
                         39.19197118, 48.18441452, 44.30411404, 50.05167959]
        # expected_msft = talib.RSI(df_msft_close.values, timeperiod=14)
        expected_msft = [38.5647217, 42.0627596, 37.9693676, 43.8641553, 48.3458438,
                         47.095672, 46.7363662, 46.127465]
        # expected_aapl += 7
        test_expected(spectre.factors.RSI(), expected_aapl, expected_msft)
        # normalized
        expected_aapl = np.array(expected_aapl) / 50 - 1
        expected_msft = np.array(expected_msft) / 50 - 1
        test_expected(spectre.factors.RSI().normalized(), expected_aapl, expected_msft)

        # test stochf
        expected_aapl = talib.STOCHF(df_aapl_high.values, df_aapl_low.values, df_aapl_close.values,
                                     fastk_period=14)[0]
        expected_msft = talib.STOCHF(df_msft_high.values, df_msft_low.values, df_msft_close.values,
                                     fastk_period=14)[0]
        test_expected(spectre.factors.STOCHF(), expected_aapl, expected_msft)

        # test MarketDispersion features
        expected_aapl = [0.0006367, 0.047016, 0.0026646, 0.0056998, 0.0012298, 0.0110741,
                         0., 0.0094943, 0.0098479]
        expected_msft = expected_aapl.copy()
        del expected_msft[6]
        test_expected(spectre.factors.MarketDispersion(), expected_aapl, expected_msft, 10)

        # test MarketReturn features
        expected_aapl = [-0.030516, -0.0373418,  0.0232942, -0.0056998,  0.0183439,
                         0.0184871,  0.0318528, -0.0359094,  0.011689]
        expected_msft = expected_aapl.copy()
        del expected_msft[6]
        test_expected(spectre.factors.MarketReturn(), expected_aapl, expected_msft, 10)

        # test MarketVolatility features
        expected_aapl = [0.341934, 0.3439502, 0.344212, 0.3442616, 0.3447192, 0.3451061,
                         0.3462575, 0.3471314, 0.3469935]
        expected_msft = [0.341934, 0.3439502, 0.344212, 0.3442616, 0.3447192, 0.3451061,
                         0.3467437, 0.3458821]
        test_expected(spectre.factors.MarketVolatility(), expected_aapl, expected_msft, 10)

        # test AssetData features
        expected_msft = df_msft_close[-8:].values
        expected_aapl = df_msft_close[-8:].values
        expected_aapl = np.insert(expected_aapl, 6, np.nan)
        engine.align_by_time = True
        test_expected(spectre.factors.AssetData('MSFT', spectre.factors.OHLCV.close),
                      expected_aapl, expected_msft, 10)
        engine.align_by_time = False

        # test IS_JANUARY,DatetimeDataFactor,etc features
        expected_aapl = [True] * 9
        expected_msft = expected_aapl.copy()
        del expected_msft[6]
        test_expected(spectre.factors.IS_JANUARY, expected_aapl, expected_msft, 10)

        expected_aapl = [False] * 9
        expected_msft = expected_aapl.copy()
        del expected_msft[6]
        test_expected(spectre.factors.IS_DECEMBER, expected_aapl, expected_msft, 10)

        expected_aapl = np.array([3., 4., 0., 1., 2., 3., 4., 0., 1.])
        expected_msft = np.delete(expected_aapl, 5)  # 5 because DatetimeDataFactor no delay
        test_expected(spectre.factors.WEEKDAY, expected_aapl, expected_msft, 10)

        # test timezone
        engine.timezone = 'America/New_York'
        expected_aapl -= 1
        expected_msft -= 1
        expected_aapl[expected_aapl < 0] = 6
        expected_msft[expected_msft < 0] = 6
        test_expected(spectre.factors.WEEKDAY, expected_aapl, expected_msft, 10)
        engine.timezone = 'UTC'

        # test AssetClassifierDataFactor and one_hot features
        test_sector = {'AAPL': 2, }
        expected_aapl = [2] * 9
        expected_msft = [-1] * 8
        test_expected(spectre.factors.AssetClassifierDataFactor(test_sector, -1),
                      expected_aapl, expected_msft, 10)

        one_hot = spectre.factors.AssetClassifierDataFactor(test_sector, -1).one_hot()
        expected_aapl = [True] * 9
        expected_msft = [False] * 8
        test_expected(one_hot[0], expected_aapl, expected_msft, 10)
        expected_aapl = [False] * 9
        expected_msft = [True] * 8
        test_expected(one_hot[1], expected_aapl, expected_msft, 10)

        # test ffill_na
        mask = spectre.factors.WEEKDAY >= 3
        factor = spectre.factors.WEEKDAY.filter(mask)
        expected_aapl = np.array([3., 4., np.nan, np.nan, np.nan, 3., 4., np.nan, np.nan])
        expected_msft = np.delete(expected_aapl, 5)
        test_expected(factor, expected_aapl, expected_msft, 10)

        expected_aapl = np.array([3., 4., 4, 4, 4, 3., 4., 4, 4])
        expected_msft = np.delete(expected_aapl, 5)
        engine.to_cuda()
        test_expected(factor.fill_na(ffill=True), expected_aapl, expected_msft, 10)
        engine.to_cpu()

        # test reused factor only compute once, and nest factor window
        engine.run('2019-01-11', '2019-01-15')  # let pre_compute_ test executable
        f1 = spectre.factors.BBANDS(win=20, inputs=[spectre.factors.OHLCV.close, 2]).normalized()
        f2 = spectre.factors.EMA(win=10, inputs=[f1])
        fa = spectre.factors.STDDEV(win=15, inputs=[f2])
        fb = spectre.factors.MACD(12, 26, 9, inputs=[f2])
        engine.remove_all_factors()
        engine.add(f2, 'f2')
        engine.add(fa, 'fa')
        engine.add(fb, 'fb')

        for f in engine._factors.values():
            f.pre_compute_(engine, '2019-01-11', '2019-01-15')
        self.assertEqual(4, f2._ref_count)

        # reset _ref_count change caused by test above
        def reset(_f):
            _f._ref_count = 0
            if _f.inputs:
                for upstream in _f.inputs:
                    if isinstance(upstream, spectre.factors.BaseFactor):
                        reset(upstream)

        for f in engine._factors.values():
            reset(f)

        result = engine.run('2019-01-01', '2019-01-15')
        self.assertEqual(0, f2._ref_count)

        # test cuda result eq cup

    def test_filter_factor(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + 'daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        universe = spectre.factors.OHLCV.volume.top(1)
        engine.add(spectre.factors.OHLCV.volume, 'not_used')
        engine.set_filter(universe)

        result = engine.run("2019-01-01", "2019-01-15")
        assert_array_equal(result.index.get_level_values(1).values,
                           ['MSFT', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT',
                            'AAPL'])

        # test ma5 with filter
        import talib
        total_rows = 10
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
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
        expected_aapl = np.delete(expected_aapl, [0, 1, 8])
        expected_msft = [expected_msft[2], expected_msft[8]]
        # test
        assert_almost_equal(result_aapl, expected_aapl)
        assert_almost_equal(result_msft, expected_msft)

        aapl_filter = spectre.factors.StaticAssets(['AAPL'])
        engine.remove_all_factors()
        engine.set_filter(aapl_filter)
        engine.add(spectre.factors.OHLCV.close, 'c')
        df = engine.run('2018-01-01', '2019-01-15')
        assert_array_equal(['AAPL'], df.index.get_level_values(1).unique())

    def test_cuda(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
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

    def test_return(self):
        f = spectre.factors.Returns()
        import torch
        data = torch.tensor([
            [1, 2, np.nan, 4, 8, np.nan],
            [3, 4, 4.5, 6, np.nan, np.nan]
        ])
        result = f.compute(spectre.parallel.Rolling(data, win=3))
        expected = [[np.nan, np.nan, 1, 1, np.nan, 1],
                    [np.nan, np.nan, 0.5, 0.5, 1 / 3, 0]]
        assert_almost_equal(result, expected)
        # test on cuda
        data = data.cuda()
        result = f.compute(spectre.parallel.Rolling(data, win=3)).cpu()
        assert_almost_equal(result, expected)

    def test_quantile(self):
        f = spectre.factors.QuantileClassifier()
        f.bins = 5
        import torch
        result = f.compute(torch.tensor([[1, 1, np.nan, 1.01, 1.01, 2],
                                         [3, 4, 5, 1.01, np.nan, 1.01]]))
        expected = [[0, 0, np.nan, 2, 2, 4], [2, 3, 4, 0, np.nan, 0]]
        assert_array_equal(result, expected)

        result = f.compute(torch.tensor([[-1, 1, np.nan, 1.01, 1.02, 2],
                                         [3, -4, 5, 1.01, np.nan, 1.01]]))
        expected = [[0, 1, np.nan, 2, 3, 4], [3, 0, 4, 1, np.nan, 1]]
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
        expected = pd.qcut(data[1], 5, labels=False)
        assert_array_equal(result[-1], expected)

    def test_align_by_time(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.align_by_time = True
        engine.add(spectre.factors.OHLCV.close, 'close')
        engine.add(spectre.factors.SMA(2), 'ma')
        df = engine.run("2019-01-01", "2019-01-15")

        self.assertEqual(df.loc[("2019-01-11", 'MSFT'), 'ma'].values,
                         df.loc[("2019-01-10", 'MSFT'), 'close'].values)

        # dataloader
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', calender_asset='AAPL',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True, align_by_time=True
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.align_by_time = False
        engine.add(spectre.factors.OHLCV.close, 'close')
        engine.add(spectre.factors.SMA(2), 'ma')
        df = engine.run("2019-01-01", "2019-01-15")

        self.assertEqual(df.loc[("2019-01-11", 'MSFT'), 'ma'].values,
                         df.loc[("2019-01-10", 'MSFT'), 'close'].values)

    def test_linear_regression(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)

        class ARange(spectre.factors.CustomFactor):
            def compute(self, y):
                row = torch.arange(y.shape[-1], dtype=torch.float32, device=y.device)
                return row.expand(y.shape)

        x = ARange(inputs=[spectre.factors.OHLCV.close])
        f = spectre.factors.RollingLinearRegression(x, spectre.factors.OHLCV.close, 10)
        engine.add(f[0], 'slope')
        engine.add(f[1], 'intcp')

        df = engine.run("2019-01-01", "2019-01-15")
        result = df.loc[(slice(None), 'AAPL'), 'slope']
        assert_almost_equal(
            [-0.555879, -0.710545, -0.935697, -1.04103, -1.232, -1.704182,
             -0.873212, -0.640606,  0.046424], result, decimal=5)
        assert_array_equal(['slope', 'intcp'], df.columns)

    def test_engine_cross_factor(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine2 = spectre.factors.FactorEngine(loader)
        f = spectre.factors.MA(5)
        universe = spectre.factors.OHLCV.volume.top(1)
        engine.add(f, 'f')
        engine.add(universe, 'mask')
        engine2.add(f, 'f')
        engine2.set_filter(universe)

        result = engine.run("2019-01-01", "2019-01-15")
        result2 = engine2.run("2019-01-01", "2019-01-15")

        assert_array_equal(result.f[result['mask']], result2.f)

    def test_ref_count(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        t = spectre.factors.OHLCV.volume.top(1)
        b = spectre.factors.OHLCV.volume.bottom(1)
        engine.add(t, 't')
        engine.add(t & b, 't&b')
        engine.add(t | b, 't|b')
        engine.add(~t, '~t')
        engine.run("2019-01-01", "2019-01-05")

        def test_count(k):
            if isinstance(k, spectre.factors.CustomFactor):
                if k._ref_count != 0:
                    print(k, k._ref_count)
                    raise ValueError("k._ref_count != 0")
                if k._mask is not None:
                    test_count(k._mask)

            if k.inputs:
                for upstream in k.inputs:
                    if isinstance(upstream, spectre.factors.BaseFactor):
                        test_count(upstream)

        for _, k in engine._factors.items():
            test_count(k)

    def test_ops(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        f = spectre.factors.OHLCV.close
        f2 = f ** 2
        engine.add(f, 'f')
        engine.add(f2, 'f^2')
        engine.add(-f, '-f')
        engine.add(f + f2, 'f+f2')
        engine.add(f - f2, 'f-f2')
        engine.add(f * f2, 'f*f2')
        engine.add(f / f2, 'f/f2')

        engine.add(f > f2, 'f>f2')
        engine.add(f < f2, 'f<f2')
        engine.add(f >= f2, 'f>=f2')
        engine.add(f <= f2, 'f<=f2')
        engine.add(f == f2, 'f==f2')
        engine.add(f != f2, 'f!=f2')

        t = spectre.factors.OHLCV.volume.top(1)
        b = spectre.factors.OHLCV.volume.bottom(1)
        engine.add(t, 't')
        engine.add(t & b, 't&b')
        engine.add(t | b, 't|b')
        engine.add(~t, '~t')

        result = engine.run("2019-01-01", "2019-01-05")

        f = np.array([158.61, 101.30, 145.23, 102.28, 104.39])
        f2 = f ** 2
        assert_array_equal(result['f^2'], f2)
        assert_array_equal(result['-f'], -f)
        assert_array_equal(result['f+f2'], f + f2)
        assert_array_equal(result['f-f2'], f - f2)
        assert_array_equal(result['f*f2'], f * f2)
        assert_array_equal(result['f/f2'], f / f2)

        assert_array_equal(result['f>f2'], f > f2)
        assert_array_equal(result['f<f2'], f < f2)
        assert_array_equal(result['f>=f2'], f >= f2)
        assert_array_equal(result['f<=f2'], f <= f2)
        assert_array_equal(result['f==f2'], f == f2)
        assert_array_equal(result['f!=f2'], f != f2)

        t = np.array([False, True, True, False, False])
        b = ~t
        assert_array_equal(result['t&b'], t & b)
        assert_array_equal(result['t|b'], t | b)
        assert_array_equal(result['~t'], b)

    def test_multiprocess(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.to_cuda()

        self.assertRaises(ValueError, TestMultiProcessing,
                          inputs=[spectre.factors.OHLCV.open])

        engine.add(TestMultiProcessing(
            win=10,
            core=3,
            inputs=[spectre.factors.AdjustedDataFactor(spectre.factors.OHLCV.open),
                    spectre.factors.AdjustedDataFactor(spectre.factors.OHLCV.close)],
            multiprocess=True
        ), 'f')
        engine.add(spectre.factors.MA(
            win=10,
            inputs=[spectre.factors.AdjustedDataFactor(spectre.factors.OHLCV.open) *
                    spectre.factors.AdjustedDataFactor(spectre.factors.OHLCV.close)],
        ), 'f2')
        result = engine.run("2018-12-25", "2019-01-05")
        assert_almost_equal(result.f.values, result.f2.values)
        # test one row
        engine.run("2019-01-04", "2019-01-05")

        # test nested window
        engine.clear()
        engine.add(TestMultiProcessing(
            win=10,
            core=3,
            inputs=[spectre.factors.MA(3), spectre.factors.MA(4)],
            multiprocess=False
        ), 'f')
        engine.run("2019-01-04", "2019-01-05")

    def test_memory_leak(self):
        quandl_path = data_dir + '../../../historical_data/us/prices/quandl/'
        loader = spectre.data.ArrowLoader(quandl_path + 'wiki_prices.feather')
        engine = spectre.factors.FactorEngine(loader)

        engine.to_cuda()
        universe = spectre.factors.AverageDollarVolume(win=252).top(500)
        engine.set_filter(universe)

        df_prices = engine.get_price_matrix('2014-01-02', '2014-01-12')
        loader = None
        universe = None
        df_prices = None
        engine = None

        import gc
        import torch
        gc.collect(2)
        gc.collect(2)
        torch.cuda.empty_cache()
        self.assertEqual(0, torch.cuda.memory_allocated())

    @unittest.skipUnless(os.getenv('COVERAGE_RUNNING'), "too slow, run manually")
    def test_full_run(self):
        quandl_path = data_dir + '../../../historical_data/us/prices/quandl/'
        loader = spectre.data.ArrowLoader(quandl_path + 'wiki_prices.feather')
        engine = spectre.factors.FactorEngine(loader)

        engine.to_cuda()
        universe = spectre.factors.AverageDollarVolume(win=120).top(500)
        engine.set_filter(universe)
        f = spectre.factors.MA(5) - spectre.factors.MA(10) - spectre.factors.MA(30)
        engine.add(f.rank(mask=universe).zscore(), 'ma_cross')
        df_prices = engine.get_price_matrix('2014-01-02', '2017-01-18')  # .shift(-1)
        factor_data, mean_return = engine.full_run(
            '2014-01-02', '2017-01-19',
            periods=(10,), filter_zscore=None, preview=False)

        import alphalens as al
        al_clean_data = al.utils.get_clean_factor_and_forward_returns(
            factor=factor_data[('ma_cross', 'factor')], prices=df_prices, periods=[10],
            filter_zscore=None)
        al_mean, al_std = al.performance.mean_return_by_quantile(al_clean_data)
        assert_almost_equal(mean_return[('ma_cross', '10D', 'mean')].values,
                            al_mean['10D'].values, decimal=5)
        assert_almost_equal(mean_return[('ma_cross', '10D', 'sem')].values,
                            al_std['10D'].values, decimal=5)

        # check last line bug, nanlast function
        stj_10d_return = factor_data.loc[(slice('2016-12-15', '2017-01-03'), 'STJ'),
                                         ('Returns', '10D')]
        expected = [0.003880821, 0.01881325, 0.01290882, 0.01762784, 0.0194248, 0.01405275,
                    0.01240134, 0.00835931, 0.01265514, 0.01240134, 0.00785625,
                    0.00161111]
        assert_almost_equal(stj_10d_return.values, expected)

        # test zscore filter
        factor_data, mean_return = engine.full_run(
            '2014-01-02', '2017-01-19',
            periods=(10,), filter_zscore=0.5, preview=False)

        al_clean_data = al.utils.get_clean_factor_and_forward_returns(
            factor=factor_data[('ma_cross', 'factor')], prices=df_prices, periods=[10],
            filter_zscore=0.5, max_loss=1)
        al_mean, al_std = al.performance.mean_return_by_quantile(al_clean_data)
        # only test sign of data, because alphalens filter zscore only uses returned data.
        # we use include backward data.
        assert_almost_equal(mean_return[('ma_cross', '10D', 'mean')].values,
                            al_mean['10D'].values, decimal=3)
        assert_almost_equal(mean_return[('ma_cross', '10D', 'sem')].values,
                            al_std['10D'].values, decimal=3)
