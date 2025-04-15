import unittest
import spectre
import os
import numpy as np
import pandas as pd
import torch
import scipy.stats
from numpy.testing import assert_almost_equal, assert_array_equal
from os.path import dirname
import warnings


data_dir = dirname(__file__) + '/data/'


class TestMultiProcessing(spectre.factors.CPUParallelFactor):

    @staticmethod
    def mp_compute(a, b) -> np.array:
        return (a * b).mean(axis=0).values


class TestFactorLib(unittest.TestCase):

    def test_factors(self):
        warnings.filterwarnings("ignore", module='spectre')
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

        # test raw data
        raw_tensors = engine.run_raw('2018-01-01', '2019-01-15', False)
        assert_almost_equal(df.open.values, raw_tensors['open'])

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
        # test rank with tier
        x = torch.tensor([[2, 3, 6, 8, 3, 6, 6], [2, 3, 6, np.nan, 3, 2, 9]])
        result = spectre.parallel.rankdata(x)
        assert_array_equal(scipy.stats.rankdata(x[0]), result[0])
        expected = scipy.stats.rankdata(x[1], nan_policy='omit')
        expected[3] = np.nan
        assert_array_equal(expected, result[1])

        # test mean
        expected_aapl = [129.955, 123.755, 126.695, 126.1, 128.385, 131.015, 161.,
                         128.445, 130.195]
        expected_msft = [129.955, 123.755, 126.695, 126.1, 128.385, 131.015,
                         128.445, 130.195]
        test_expected(spectre.factors.OHLCV.close.mean(),
                      expected_aapl, expected_msft, total_rows)
        expected_aapl = [127.76211, 130.98482, 131.98633, 132.3147 , 131.43803, 135.9378 ,
                         161.  , 128.19464, 132.38597]
        expected_msft = [127.76211, 130.98482, 131.98633, 132.3147 , 131.43803, 135.9378 ,
                         128.19464, 132.38597]
        test_expected(spectre.factors.OHLCV.close.mean(weight=spectre.factors.OHLCV.volume),
                      expected_aapl, expected_msft, total_rows, decimal=4)

        # test XSStandardDeviation
        expected_aapl = [28.655, 21.475, 22.305, 22.900, 23.165,
                         25.015, 0.000, 25.245, 26.805, ]
        expected_msft = [28.655, 21.475, 22.305, 22.900, 23.165,
                         25.015, 25.245, 26.805]
        test_expected(spectre.factors.OHLCV.close.std(),
                      expected_aapl, expected_msft, total_rows)

        # test zscore
        expected_aapl = [1.] * 9
        # aapl has prices data, but we only have two stocks, so one data zscore = 0/0 = nan
        expected_aapl[6] = np.nan
        expected_msft = [-1.] * 8
        test_expected(spectre.factors.OHLCV.close.zscore(),
                      expected_aapl, expected_msft, total_rows)

        expected_aapl = [5.5351597, 6.7627474, 6.6801166, 6.5065502, 6.5421973, 6.2374575,
                         np.inf, 6.0879382, 5.8571162]
        expected_msft = [3.5351597, 4.7627474, 4.6801166, 4.5065502, 4.5421973, 4.2374575,
                         4.0879382, 3.8571162]
        test_expected(spectre.factors.OHLCV.close.zscore(weight=0),
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
        test_expected(spectre.factors.OHLCV.close.demedian(),
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
        engine.set_filter(None)

        # test quantile get factors from engine._factors nan bug
        expected_aapl = [4.] * 9
        expected_aapl[6] = np.nan
        expected_msft = [0.] * 8
        engine.set_filter(spectre.factors.OHLCV.close.top(2))
        engine.add(spectre.factors.OHLCV.close.zscore(), 'pre')
        f = engine.get_factor('pre')
        test_expected(f.quantile(), expected_aapl, expected_msft, total_rows)
        engine.set_filter(None)

        # download: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
        # pip install --no-deps d:\doc\Download\TA_Lib-0.4.18-cp37-cp37m-win_amd64.whl
        import talib

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
        expected_aapl_normal = (df_aapl_close.values - expected_aapl[1]) / \
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
        expected_aapl = [-0.030516, -0.0373418, 0.0232942, -0.0056998, 0.0183439,
                         0.0184871, 0.0318528, -0.0359094, 0.011689]
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
        # because align_by_time = True, They will all have data
        expected_aapl = np.insert(expected_aapl, 6, np.nan)
        expected_msft = np.insert(expected_msft, 6, np.nan)
        engine.align_by_time = True
        test_expected(spectre.factors.AssetData('MSFT', spectre.factors.OHLCV.close),
                      expected_aapl, expected_msft, 10)
        engine.align_by_time = False

        # test IS_JANUARY,DatetimeDataFactor,etc features
        # DatetimeDataFactors are un-delayed, so there is 10 data points
        expected_aapl = [True] * 10
        expected_msft = expected_aapl.copy()
        del expected_msft[7]
        test_expected(spectre.factors.IS_JANUARY, expected_aapl, expected_msft, 10)

        expected_aapl = [False] * 10
        expected_msft = expected_aapl.copy()
        del expected_msft[7]
        test_expected(spectre.factors.IS_DECEMBER, expected_aapl, expected_msft, 10)

        expected_aapl = np.array([2, 3., 4., 0., 1., 2., 3., 4., 0., 1.])
        expected_msft = np.delete(expected_aapl, 6)
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
        expected_aapl = [2] * 10
        expected_msft = [-1] * 9
        test_expected(spectre.factors.AssetClassifierDataFactor(test_sector, -1),
                      expected_aapl, expected_msft, 10)

        one_hot = spectre.factors.AssetClassifierDataFactor(test_sector, -1).one_hot()
        expected_aapl = [True] * 10
        expected_msft = [False] * 9
        test_expected(one_hot[0], expected_aapl, expected_msft, 10)
        expected_aapl = [False] * 10
        expected_msft = [True] * 9
        test_expected(one_hot[1], expected_aapl, expected_msft, 10)

        # test ffill_na
        mask = spectre.factors.WEEKDAY >= 3
        factor = spectre.factors.WEEKDAY.filter(mask)
        expected_aapl = np.array([np.nan, 3., 4., np.nan, np.nan, np.nan, 3., 4., np.nan, np.nan])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        expected_aapl = np.array([np.nan, 3., 4., 4, 4, 4, 3., 4., 4, 4])
        expected_msft = np.delete(expected_aapl, 6)
        engine.to_cuda()
        test_expected(factor.fill_na(ffill=True), expected_aapl, expected_msft, 10)

        expected_aapl = np.array([np.nan, 1., np.nan, np.nan, np.nan, np.nan,
                                  1., np.nan, np.nan, np.nan], dtype=np.float32)
        expected_msft = np.delete(expected_aapl, 6)
        test_expected((1e10 ** factor).fill_na(np.nan, nan=False, inf=True)/1e30, expected_aapl,
                      expected_msft, 10, decimal=5)

        expected_aapl = np.array([0, 3., 4., 0, 0, 0, 3., 4., 0, 0])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor.fill_na(0, nan=True, inf=True), expected_aapl,
                      expected_msft, 10)
        engine.to_cpu()
        spectre.factors.WEEKDAY.filter(None)

        # test masked fill
        factor = spectre.factors.WEEKDAY.masked_fill(mask, spectre.factors.WEEKDAY * 2)
        expected_aapl = np.array([2, 6., 8., 0, 1, 2, 6., 8., 0, 1])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.WEEKDAY.masked_fill(mask, -1)
        expected_aapl = np.array([2, -1., -1., 0, 1, 2, -1., -1., 0, 1])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test prod
        factor = spectre.factors.WEEKDAY.ts_prod(3)
        expected_aapl = np.array([0, 0, 24., 0, 0, 0, 6., 24., 0, 0])
        expected_msft = np.array([0, 0, 24., 0, 0, 0, 8., 0, 0])
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test most (pytorch mode has bug)
        # factor = spectre.factors.Returns().sign().mode(groupby='date')
        # # sign values:
        # # -1., -1.,  1.,  0.,  1.,  1.,  1., -1.,  1.
        # # -1.,  1.,  1., -1.,  1.,  1., -1.,  1.
        # expected_aapl = np.array([-1., -1.,  1.,  -1,  1.,  1.,  1., -1.,  1.])
        # expected_msft = np.array([-1., -1.,  1.,  -1,  1.,  1., -1.,  1.])
        # test_expected(factor, expected_aapl, expected_msft, 10)
        # factor = spectre.factors.Returns().round(3).mode(groupby='asset')
        # # sign values:
        # # [[nan, -0.0310, -0.0840, 0.0260, 0.0000, 0.0170, 0.0300, 0.0320,
        # #  -0.0450, 0.0220, -0.0000],
        # # [nan, -0.0300, 0.0100, 0.0210, -0.0110, 0.0200, 0.0070, -0.0260,
        # #  0.0020, 0.0530, nan]]
        # expected_aapl = np.array([-0., -0.,  -0.,  -0,  -0.,  -0.,  -0., -0.,  -0.])
        # expected_msft = np.array([-0.03, -0.03,  -0.03,  -0.03,  -0.03,  -0.03, -0.03,  -0.03])
        # test_expected(factor, expected_aapl, expected_msft, 10, check_bias=False)

        # test Median
        factor = spectre.factors.Returns().round(3).median(groupby='asset')
        # sign values:
        # [[nan, -0.0310, -0.0840, 0.0260, 0.0000, 0.0170, 0.0300, 0.0320,
        #  -0.0450, 0.0220, -0.0000],
        # [nan, -0.0300, 0.0100, 0.0210, -0.0110, 0.0200, 0.0070, -0.0260,
        #  0.0020, 0.0530, nan]]
        expected_aapl = np.array([0.0085] * 9)
        expected_msft = np.array([0.007] * 8)
        test_expected(factor, expected_aapl, expected_msft, 10, check_bias=False)

        # test ForwardSignalData
        signal = spectre.factors.OHLCV.close > 150
        signal_price = spectre.factors.ForwardSignalData(4, spectre.factors.OHLCV.close, signal)
        signal_ret = signal_price / spectre.factors.OHLCV.close - 1

        expected_aapl = np.array([0, 0.04351718, 0.017114094, 0.017114094, 0, 0, 0, 0, 0])
        expected_msft = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        test_expected(signal_ret, expected_aapl, expected_msft, 10)

        # test mad_clamp
        factor = spectre.factors.WEEKDAY.mad_clamp(1)
        factor.groupby = 'asset'
        expected_aapl = np.array([2, 3., 3., 1, 1, 2, 3., 3., 1, 1])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.OHLCV.close.mad_clamp(2)
        factor.groupby = 'asset'
        expected_aapl = np.array([158.61, 147.8, 149., 149., 151.55, 156.03, 161, 153.69, 157.])
        expected_msft = np.array([101.3, 102.28, 104.39, 103.2, 105.22, 105.61, 103.2, 103.39])
        test_expected(factor, expected_aapl, expected_msft, 10, check_bias=False)

        factor = spectre.factors.OHLCV.close.mad_clamp(2, mean=True)
        factor.groupby = 'asset'
        expected_aapl = np.array([158.61, 145.583, 149., 149., 151.55, 156.03, 161, 153.69, 157.])
        expected_msft = np.array([101.3, 102.28, 104.39, 103.2, 105.22, 106, 103.2, 103.39])
        test_expected(factor, expected_aapl, expected_msft, 10, check_bias=False)

        # test nans bug
        x = torch.tensor([[-999, np.nan, np.nan, 2, np.nan, 3, np.nan, np.nan, 999, np.nan],
                          [1, np.nan, np.nan, 2, np.nan, 3, np.nan, np.nan, 4, np.nan]],
                         dtype=torch.float64)
        if torch.cuda.is_available():
            x = x.cuda()
        mad_fct = spectre.factors.MADClampFactor()
        mad_fct.z = 1.5
        mad_fct._mask = mad_fct
        mad_fct._mask_out = ~torch.isnan(x)
        result = mad_fct.compute(x, fill=None)
        expected = x.cpu().numpy()
        expected[0, 0] = -745.25
        expected[0, -2] = 750.25
        assert_almost_equal(expected, result.cpu().numpy())

        # test winsorizing
        factor = spectre.factors.WEEKDAY.winsorizing(0.2)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            np.array([2., 3., 4., 0., 1., 2., 3., 4., 0., 1.]), [0.2, 0.2])
        expected_msft = scipy.stats.mstats.winsorize(
            np.array([2., 3., 4., 0., 1., 2., 4., 0., 1.]), [0.2, 0.2])
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.WEEKDAY.winsorizing(0.01)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            np.array([2., 3., 4., 0., 1., 2., 3., 4., 0., 1.]), [0.01, 0.01])
        expected_msft = scipy.stats.mstats.winsorize(
            np.array([2., 3., 4., 0., 1., 2., 4., 0., 1.]), [0.01, 0.01])
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.WEEKDAY.winsorizing(0.001)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            np.array([2., 3., 4., 0., 1., 2., 3., 4., 0., 1.]), [0.001, 0.001])
        expected_msft = scipy.stats.mstats.winsorize(
            np.array([2., 3., 4., 0., 1., 2., 4., 0., 1.]), [0.001, 0.001])
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.OHLCV.close.winsorizing(0.2)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            np.array([158.6100, 145.2300, 149.0000, 149.0000, 151.5500, 156.0300, 161.0000,
             153.6900, 157.0000, 156.9400]), [0.2, 0.2])[:-1]
        expected_msft = scipy.stats.mstats.winsorize(
            np.array([101.3000, 102.2800, 104.3900, 103.2000, 105.2200, 106.0000, 103.2000,
             103.3900, 108.8500]), [0.2, 0.2])[:-1]
        test_expected(factor, expected_aapl, expected_msft, 10, check_bias=False)

        # test IQRNormalityFactor
        factor = spectre.factors.IQRNormalityFactor(inputs=[spectre.factors.Returns()])
        expected_aapl = np.array([0.65102, 0.65102, 0.65102, 0.65102, 0.65102, 0.65102, np.nan,
                                  0.65102, 0.65102])
        expected_msft = np.array([0.65102, 0.65102, 0.65102, 0.65102, 0.65102, 0.65102, 0.65102,
                                  0.65102])
        test_expected(factor, expected_aapl, expected_msft, 10, check_bias=False)

        # test LinearWeightedAverage
        factor = spectre.factors.LinearWeightedAverage(5, inputs=[spectre.factors.WEEKDAY])
        expected_aapl = np.array([2., 2.2666667, 2.8000002, 1.9333334, 1.6666667, 1.6666667, 2,
                                  2.666667, 2, 1.6666667])
        expected_msft = np.array([2., 2.2666667, 2.8000002, 1.9333334, 1.6666667, 1.6666667,
                                  2.3333335, 1.6000001, 1.4666667])
        test_expected(factor, expected_aapl, expected_msft, 10, decimal=6)

        # test ElementWiseMax
        factor = spectre.factors.ElementWiseMax(inputs=[spectre.factors.WEEKDAY,
                                                        spectre.factors.DatetimeDataFactor('day')])
        expected_aapl = np.array([2., 3, 4, 7, 8, 9, 10, 11, 14, 15])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.ElementWiseMin(inputs=[
            spectre.factors.WEEKDAY, spectre.factors.DatetimeDataFactor('day') - 1])  # -1 for utc
        expected_aapl = np.array([1., 2, 3, 0, 1, 2, 3, 4, 0, 1])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test RollingArgMax, weekday have same value, use day as a decimal point
        weekday = spectre.factors.WEEKDAY + spectre.factors.DatetimeDataFactor('day') / 100
        factor = spectre.factors.RollingArgMax(5, inputs=[weekday])
        expected_aapl = np.array([3/5, 2/5, 1/5, 4/5, 3/5, 2/5, 1/5, 5/5, 4/5, 3/5])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.RollingArgMin(5, inputs=[weekday])
        expected_aapl = np.array([4/5, 3/5, 2/5, 5/5, 4/5, 3/5, 2/5, 1/5, 5/5, 4/5])
        expected_msft = np.array([4/5, 3/5, 2/5, 5/5, 4/5, 3/5, 2/5, 1/5, 4/5])
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test log abs etc...
        factor = spectre.factors.WEEKDAY.log()
        expected_aapl = np.log([2, 3., 4., 0, 1, 2, 3., 4., 0, 1])
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.Returns()
        expected_aapl = np.array([-0.0311526, -0.0843579,  0.0259588,  0.,  0.0171141,
                                  0.0295612,  0.0318528, -0.0454037,  0.0215369])
        expected_msft = np.array([-0.0298793,  0.0096742,  0.0206296, -0.0113996,  0.0195736,
                                  0.007413, -0.0264151,  0.0018411])
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.Returns().abs()
        test_expected(factor, np.abs(expected_aapl), np.abs(expected_msft), 10)

        factor = spectre.factors.Returns().sign()
        test_expected(factor, np.sign(expected_aapl), np.sign(expected_msft), 10)

        factor = (spectre.factors.DatetimeDataFactor('day') / 1000).round(2)
        expected_aapl = np.array([2.,  3.,  4.,  7.,  8.,  9., 10., 11., 14., 15.]) / 1000
        expected_aapl = np.round(expected_aapl, 2)
        expected_msft = np.delete(expected_aapl, 6)
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test sum prod
        factor = spectre.factors.LogReturns().ts_sum(2)
        expected_aapl = np.log(df_aapl_close) - np.log(df_aapl_close.shift(1))
        expected_msft = np.log(df_msft_close) - np.log(df_msft_close.shift(1))
        expected_aapl = expected_aapl.rolling(2).sum()
        expected_msft = expected_msft.rolling(2).sum()
        expected_aapl = expected_aapl[-9:]
        expected_msft = expected_msft[-8:]
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.OHLCV.close.ts_prod(2).log()
        expected_aapl = np.log(df_aapl_close) + np.log(df_aapl_close.shift(1))
        expected_msft = np.log(df_msft_close) + np.log(df_msft_close.shift(1))
        expected_aapl = expected_aapl[-9:]
        expected_msft = expected_msft[-8:]
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.LogReturns().xs_sum()
        expected_aapl = np.array([-0.061983 , -0.0785019,  0.0460473, -0.011465 ,  0.0363538,
        0.0365184,  0.0313561, -0.073237 ,  0.0231476])
        expected_msft = np.array([-0.061983 , -0.0785019,  0.0460473, -0.011465 ,  0.0363538,
        0.0365184, -0.073237 ,  0.0231476])
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test RollingRankFactor
        factor = spectre.factors.RollingRankFactor(5, inputs=[spectre.factors.WEEKDAY])
        expected_aapl = np.array([2.5/5, 3.5/5, 4.5/5, 1.5/5, 2/5, 3/5, 4/5, 5/5, 1/5, 2/5])
        expected_msft = np.array([2.5/5, 3.5/5, 4.5/5, 1.5/5, 2/5, 3/5, 4.5/5, 1.5/5, 2.5/5])
        test_expected(factor, expected_aapl, expected_msft, 10)

        # test count...
        data = torch.tensor([[np.nan, 1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4, 5]])
        data = spectre.parallel.Rolling(data, win=3)
        f = spectre.factors.CustomFactor().ts_count(3)
        result = f.compute(data)
        assert_almost_equal([[0, 1, 2, ] + [3] * 3, [1, 2, ] + [3] * 4], result)

        # -- inf bug
        wsz = spectre.factors.WinsorizingFactor()
        wsz.z = 0.2
        test_data = torch.tensor([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                   5, 2, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan,
                                   3, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                                  ])
        test_data = test_data.repeat(2, 1)
        wsz._mask = wsz
        wsz._mask_out = ~torch.isnan(test_data)
        ret = wsz.compute(test_data)
        assert_almost_equal(test_data.numpy(), ret.numpy())
        self.assertFalse(torch.isinf(ret).any())

        # test RollingCovariance
        factor = spectre.factors.RollingCovariance(
            win=3, inputs=[spectre.factors.WEEKDAY, spectre.factors.DatetimeDataFactor('day')])
        # cov = np.diag(np.cov(np.vstack(x), np.vstack(y), rowvar=True), k=24)
        expected_aapl = np.array([-3., -23.5, 1., -3.83333333, -3.83333333, 1.,
                                  1., 1., -3.83333333, -3.83333333])
        expected_msft = np.array([-3., -23.5, 1., -3.83333333, -3.83333333, 1.,
                                  2.33333333, -3., -3.83333333])
        test_expected(factor, expected_aapl, expected_msft, 10, decimal=6)

        # test RollingCorrelation
        factor = spectre.factors.RollingCorrelation(
            win=3, inputs=[spectre.factors.WEEKDAY, spectre.factors.DatetimeDataFactor('day')])
        expected_aapl = np.array([-0.094057, -0.934533,  1., -0.884615, -0.884615,  1.,
                                  1,  1, -0.884615, -0.884615])
        expected_msft = np.array([-0.094057, -0.934533,  1, -0.884615, -0.884615,  1,
                                  1, -0.59604, -0.884615])
        test_expected(factor, expected_aapl, expected_msft, 10, decimal=6)

        # test CrossSectionR2
        y = torch.tensor([[1., 3, 2, 4, 6, 5, 7, 9],
                          [1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        y_hat = torch.tensor([1., 2, 3, 4, 5, 6, 7, 8]).repeat(2, 1)
        xsr2 = spectre.factors.CrossSectionR2(None, None, None)
        ret = xsr2.compute(y, y_hat)
        from sklearn.metrics import r2_score
        expected = r2_score(y[0], y_hat[0])
        assert_almost_equal(expected, ret[0, 0])
        expected = r2_score(y[1, :1], y_hat[1, :1])
        assert_almost_equal(expected, ret[1, 0])

        # test max corr coef
        xs_corr = spectre.factors.XSMaxCorrCoef()
        xs = [torch.tensor([[1., 2, 3, 4, 5], [1., 2, 3, 4, 5]]),
              torch.tensor([[1., 2, 3, 3, 4], [2., 1, 0, 4, 5]])]
        corr = xs_corr.compute(*xs)
        expected1 = np.corrcoef(torch.stack([xs[0][0], xs[1][0]]))
        expected2 = np.corrcoef(torch.stack([xs[0][1], xs[1][1]]))
        assert_almost_equal(expected1[0, 1], corr[0, 0, 0])
        assert_almost_equal(expected2[0, 1], corr[1, 0, 0])

        # test FactorWiseKthValue
        xs_k = spectre.factors.FactorWiseKthValue(2, [])
        xs = [torch.tensor([[1., 2, 2, 4, 5], [1., 3, 3, 4, 5]]),
              torch.tensor([[1., 3, 3, 3, 4], [2., 1, 0, 5, 5]]),
              torch.tensor([[1., 1, 3, 3, 4], [2., 2, 0, 6, 6]])]
        kv = xs_k.compute(*xs)
        expected = np.array([2.8, 3.2]).repeat(5, 0).reshape(2, 5)
        assert_almost_equal(expected, kv)

        # test FactorWiseKthValue
        xs_z = spectre.factors.FactorWiseZScore()
        zv = xs_z.compute(*xs)
        expected = np.array([[0.7071064,  0.7071064, -1.414214],
                             [0.7071064, -1.414214,  0.7071064]]).repeat(5, 0).reshape(2, 5, 3)
        assert_almost_equal(expected, zv, decimal=4)

        # =========================

        # test reused factor only compute once, and nest factor window
        engine.run('2019-01-11', '2019-01-15')  # let pre_compute_ test executable
        f1 = spectre.factors.BBANDS(win=20, inputs=[spectre.factors.OHLCV.close, 2]).normalized()
        f2 = spectre.factors.EMA(span=10, inputs=[f1])
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
        assert_array_equal(['MSFT', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT',
                            'AAPL'],
                           result.index.get_level_values(1).values,)

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
        engine.add(spectre.factors.OHLCV.close, 'close')
        df = engine.run('2018-01-01', '2019-01-15')
        df_aapl_close = df.loc[(slice(None), 'AAPL'), 'close']
        df_msft_close = df.loc[(slice(None), 'MSFT'), 'close']
        expected_aapl = talib.SMA(df_aapl_close.values, timeperiod=5)[-total_rows:]
        expected_msft = talib.SMA(df_msft_close.values, timeperiod=5)[-total_rows:]
        expected_aapl = np.delete(expected_aapl, [0, 1, 8])
        expected_msft = [expected_msft[2], expected_msft[8]]
        assert_almost_equal(expected_aapl, result_aapl)
        assert_almost_equal(expected_msft, result_msft)

        # test StaticAssets/pre-screen
        aapl_filter = spectre.factors.StaticAssets({'AAPL'})
        engine.remove_all_factors()
        engine.set_filter(aapl_filter)
        engine.add(spectre.factors.OHLCV.close, 'close')
        df = engine.run('2018-01-01', '2019-01-15')
        assert_array_equal(['AAPL'], df.index.get_level_values(1).unique())
        # test StaticAssets/ none pre-screen
        aapl_filter = ~spectre.factors.StaticAssets({'AAPL'})
        engine.set_filter(aapl_filter)
        df = engine.run('2018-01-01', '2019-01-15')
        assert_array_equal(['MSFT'], df.index.get_level_values(1).unique())

        # test filtering a filter
        engine.remove_all_factors()
        mask = spectre.factors.OHLCV.close > 177
        self.assertRaisesRegex(ValueError, '.*does not support local filtering.*',
                               mask.filter, mask)

        # test any
        data = torch.tensor([[np.nan, 1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4, 5]])
        data = spectre.parallel.Rolling(data, win=3)
        f = spectre.factors.CustomFactor().ts_any(3)
        result = f.compute(data)
        assert_almost_equal([[False] + [True] * 5, [True] * 6], result)

        f = spectre.factors.CustomFactor().ts_all(3)
        result = f.compute(data)
        assert_almost_equal([[False] * 3 + [True] * 3, [False] * 2 + [True] * 4], result)

        # test filter in xs factor
        aapl_filter = spectre.factors.StaticAssets(['AAPL'])
        engine.remove_all_factors()

        class TestFilterXS(spectre.factors.CrossSectionFactor):
            def compute(self, x) -> torch.Tensor:
                x_sum = x.sum(dim=1)
                assert (x_sum == 1).all()
                return x

        engine.add(TestFilterXS(inputs=[aapl_filter]), 'aapl_filter')
        engine.set_filter(None)
        engine.run('2019-01-01', '2019-01-15')

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
                    [np.nan, np.nan, 0.5, 0.5, 1 / 3, np.nan]]
        assert_almost_equal(result, expected)
        # test on cuda
        if torch.cuda.is_available():
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

        data = spectre.parallel.Rolling(torch.tensor(data)[:, :3], 3)
        f = spectre.factors.RollingQuantile(10)
        result = f.compute(data, 5)
        expected = [
            np.nan,
            pd.qcut(data.values[0, 1], 5, labels=False)[-1],
            pd.qcut(data.values[0, 2], 5, labels=False)[-1]]
        assert_array_equal(result[0], expected)

        # 测试实际数据
        data = [[ 9.29029644e-01, -2.96781480e-01, -5.45036376e-01, -1.16110361e+00,
       -4.86112952e-01,  6.92346811e-01,  1.71807766e+00, -4.62027401e-01,
       -6.85547173e-01,  1.13849366e+00,  5.02368510e-01, -8.51391196e-01,
        5.32224238e-01,  8.17808568e-01,  8.64651948e-02, -1.29505002e+00,
        1.38863719e+00,  5.14719784e-01, -4.18532550e-01, -1.10055041e+00,
       -1.85373455e-01,  1.22018611e+00, -1.99054897e-01, -7.50840724e-01,
        1.91888034e+00,  3.53980869e-01, -1.00494242e+00,  6.18480630e-02,
        1.60038903e-01, -4.56019342e-01,  1.61706179e-01,  3.92669261e-01,
       -3.34435433e-01, -1.47416985e+00, -1.55765581e+00,  5.90604544e-01,
       -1.18954957e+00,  6.97175145e-01, -5.06783612e-02, -4.47618693e-01,
        9.41468656e-01,  2.52137533e-09,  1.08429658e+00,  1.56121671e+00,
        1.16313076e+00, -1.42207515e+00,  1.76214206e+00, -2.61509866e-01,
       -8.52138162e-01,  9.91557479e-01, -4.51777905e-01,  4.37957913e-01,
        1.85241842e+00,  1.79867971e+00, -1.23124731e+00, -2.39105016e-01,
        3.55268776e-01,  1.07000184e+00, -3.18124563e-01, -7.21607506e-01,
       -1.07886767e+00, -4.63961847e-02,  4.56352323e-01,  5.00709891e-01,
       -1.13155401e+00,  3.15614104e-01, -1.09267139e+00,  6.23904586e-01,
        3.64024967e-01,  1.22176814e+00, -2.79984921e-01,  1.27578270e+00,
       -7.81615913e-01,  8.99420500e-01,  4.45060760e-01, -4.77538794e-01,
        4.68120873e-01, -9.71201897e-01,  1.57270098e+00,  1.06251574e+00,
       -3.64419788e-01,  7.98408315e-02,  7.20673561e-01, -8.65130246e-01,
       -8.84654596e-02, -5.56420028e-01,  1.89160723e-02,  4.91679877e-01,
        1.24546134e+00,  1.73730588e+00,  1.65901756e+00, -1.91093892e-01,
       -1.77692533e-01, -9.61918831e-01,  1.10681987e+00,  6.69407904e-01,
        8.60281587e-01,  6.50695324e-01, -1.65847674e-01, -6.53587162e-01,
        6.10252004e-03,  1.20320928e+00,  3.52604955e-01, -7.30400085e-01,
       -1.12211156e+00,  7.29268968e-01,  2.73594111e-02, -1.17042685e+00,
        7.88800776e-01, -4.28741798e-03, -8.86362255e-01, -1.36520326e-01,
        1.38339472e+00,  8.80989909e-01, -1.41763771e+00, -2.12396041e-01,
        7.83286154e-01,  5.67251980e-01,  1.49849248e+00, -1.62652445e+00,
        2.41854653e-01, -9.56367373e-01,  3.53749752e-01,  1.98986626e+00,
        6.46392524e-01,  3.22151631e-01,  1.98921192e+00, -2.33380127e+00,
        7.66819537e-01,  2.14252591e+00, -4.47613955e-01,  7.38505721e-01,
       -6.81742728e-01, -2.29699039e+00, -7.89720118e-01,  1.04677355e+00,
        1.48091745e+00,  9.62465823e-01,  8.93415868e-01, -4.05416220e-01,
        1.20368981e+00, -8.01701546e-01, -1.04033804e+00,  1.54275775e+00,
        6.96745634e-01,  6.76261261e-02, -4.12975103e-01, -7.90328860e-01,
        9.66707051e-01, -6.79539740e-01, -1.21652520e+00,  9.18928683e-02,
       -4.80283767e-01, -8.01525295e-01,  5.86520672e-01,  1.08422175e-01,
        1.22225869e+00,  6.50619686e-01, -5.18573403e-01, -1.19194604e-01,
        6.71912432e-01, -2.33913660e-01,  1.16661465e+00, -3.76907885e-01,
       -8.65164757e-01,  1.72409940e+00,  1.70007837e+00, -3.23385559e-02,
       -2.66000718e-01, -1.06181133e+00, -6.83412626e-02,  1.27144051e+00,
        7.45215416e-01, -1.91550374e+00, -8.48351002e-01, -3.40340525e-01,
        3.16898555e-01,  9.97029364e-01,  6.26838207e-01, -4.58528876e-01,
        5.22373557e-01,  3.93712759e-01, -7.39046633e-01,  1.97067869e+00,
       -4.91013616e-01, -1.58629096e+00,  8.73231769e-01, -1.46362102e+00,
        4.37433878e-03,  6.14954293e-01, -4.87866431e-01, -1.24889076e+00,
       -9.78501916e-01,  1.37116766e+00, -3.24208468e-01,  2.02993250e+00,
        2.08185121e-01, -4.19294447e-01,  2.73573875e-01,  1.30237865e+00,
       -1.23445177e+00,  1.92309475e+00, -1.86288249e+00, -1.20414364e+00,
       -2.08869600e+00, -1.04289734e+00,  2.61447579e-01, -1.00906575e+00,
       -2.40074247e-01, -3.71027917e-01,  5.28723478e-01,  3.99157524e-01,
        1.70707214e+00, -6.45640790e-02,  6.50075614e-01, -6.06373370e-01,
       -1.96583605e+00,  4.97690350e-01, -4.74653363e-01,  1.05897725e+00,
        1.40658617e+00, -2.03584060e-01, -1.05674662e-01, -1.36719835e+00,
       -1.89006582e-01,  2.07244349e+00, -3.91815186e-01, -8.85053873e-01,
        8.69290054e-01,  7.82633483e-01, -1.30489028e+00,  1.56648204e-01,
       -9.33829024e-02, -1.08663440e+00,  1.68229616e+00, -1.91073418e-01,
        5.94026446e-01,  1.27340162e+00,  5.05373441e-02, -1.39342561e-01,
        2.44499259e-02, -2.25620389e-01,  2.87434191e-01,  8.00414830e-02,
        7.08093983e-04, -2.89859682e-01,  1.10124993e+00, -2.75332928e-01,
       -6.30707681e-01,  6.11561537e-01, -1.55861521e+00,  1.19737554e+00,
       -1.37741518e+00,  2.21781000e-01, -7.51664102e-01, -4.67635900e-01,
        2.97017515e-01,  2.85108298e-01, -4.40221637e-01, -7.33036458e-01,
        2.03584060e-01,  9.87294078e-01,  1.56807423e+00, -5.93704998e-01,
        1.32792282e+00,  4.96067315e-01,  5.19495364e-03, -9.53168631e-01,
       -1.32709280e-01, -5.46912074e-01, -2.73118287e-01, -2.24272132e+00,
        1.67894137e+00,  1.95055580e+00,  6.07518554e-01,  8.68694305e-01,
        2.27361903e-01, -1.85788408e-01,  1.49198580e+00, -3.52911562e-01,
        9.46843565e-01, -9.26284552e-01, -4.84978370e-02, -2.14433700e-01,
       -1.99873194e-01,  1.12959242e+00,  6.94619656e-01,  1.01784229e+00,
        1.14298499e+00, -3.79862338e-01,  1.03247726e+00, -4.11657095e-01,
       -6.31651998e-01, -1.07468069e+00,  5.57753742e-01,  1.39400077e+00,
        1.46586433e-01, -1.58556497e+00,  1.68748057e+00, -1.35271645e+00,
       -7.10914731e-01, -2.72472310e+00,  1.32844162e+00,  8.30109835e-01,
       -1.19094813e+00, -1.18386328e+00,  1.34554684e+00, -2.12416220e+00,
       -3.89704466e-01,  9.01075184e-01, -6.38658345e-01,  6.57866180e-01,
        1.35939515e+00,  1.40272474e+00,  4.32722002e-01,  2.52137533e-09,
        6.57940626e-01,  5.24901338e-02, -1.28308445e-01, -9.91167724e-02,
       -4.44172263e-01,  2.43137980e+00,  1.61036193e+00, -1.28337234e-01,
        8.27327609e-01, -6.39792323e-01, -4.83033001e-01, -1.68000534e-01,
       -1.27624303e-01, -4.01563793e-01,  4.10766691e-01, -7.62258172e-02,
       -1.19568002e+00,  2.10100546e-01,  1.69408113e-01,  1.81544080e-01,
        9.65923309e-01,  8.87268722e-01,  1.19650590e+00, -5.67359328e-01,
       -5.42515934e-01, -1.76821733e+00,  9.71487463e-01, -6.92320287e-01,
        1.79917976e-01,  4.22323644e-01,  6.11761451e-01, -6.62067235e-01,
       -4.73884195e-01, -8.60845625e-01,  8.08786869e-01,  3.89572792e-02,
       -8.49466741e-01, -3.99571210e-01,  1.07462668e+00,  2.73367018e-01,
       -9.71444786e-01,  7.80574858e-01,  6.22046590e-01,  3.10192585e-01,
       -7.54574776e-01, -7.56407619e-01, -2.00627518e+00, -1.45629704e+00,
       -5.99947572e-01, -1.21989146e-01, -1.58163285e+00,  1.40455401e+00,
        1.48306200e-02,  1.56752789e+00,  1.17404985e+00,  2.54474375e-02,
       -1.12292993e+00,  1.49388683e+00, -1.13272560e+00, -6.83507264e-01,
       -8.48879665e-02, -1.06219518e+00, -1.09566212e+00, -7.87017465e-01,
       -1.65318680e+00, -1.01016723e-01,  1.87400293e+00,  2.34432846e-01,
        2.86617458e-01, -3.73461992e-01,  7.76222110e-01, -1.47273874e+00,
        1.01847574e-01,  4.34573233e-01, -3.41897666e-01,  1.50054193e+00,
        1.01206434e+00, -1.42216849e+00, -3.72394860e-01,  3.85771155e-01,
        1.01047921e+00,  7.46897757e-01, -1.98713052e+00, -1.02863503e+00,
        1.34436381e+00, -8.93415868e-01,  2.17498565e+00,  3.84472191e-01,
       -8.38692427e-01, -6.76183462e-01, -8.10572624e-01,  1.13130474e+00,
       -1.03053343e+00,  1.38310230e+00,  1.32660937e+00,  1.11861670e+00,
        3.88253838e-01,  3.82673889e-02, -7.60803938e-01, -1.32223308e+00,
       -4.34339553e-01,  5.20976365e-01,  9.88796473e-01, -4.53283429e-01,
       -1.74038196e+00, -4.64319557e-01,  7.66822517e-01, -7.03449547e-01,
       -1.90755785e-01,  1.21993637e+00,  1.39465225e+00,  8.42393264e-02,
       -1.39114630e+00, -9.09686685e-01,  1.91643268e-01,  7.72297680e-01,
        1.01626992e+00, -1.06257617e+00,  1.73053086e-01,  1.40734911e-01,
       -5.96609473e-01,  1.30832934e+00, -4.04351890e-01, -3.06608438e-01,
       -4.39797491e-01, -2.39943579e-01,  2.52055913e-01,  1.91110599e+00,
       -1.24006140e+00,  1.08708751e+00,  1.39582932e+00, -8.91650617e-01,
       -4.24971104e-01,  1.56938255e+00, -1.57642889e+00, -6.59392416e-01,
        1.30999708e+00,  5.54467380e-01,  1.73480082e+00, -7.94191182e-01,
        6.68237984e-01, -5.45984864e-01,  3.61900061e-01, -4.68523353e-01,
       -9.13197577e-01,  8.97931814e-01,  3.22844356e-01,  2.75422812e-01,
        6.87347293e-01,  9.39118385e-01, -7.27274001e-01,  3.80891323e-01,
       -6.28375888e-01, -6.14962339e-01,  6.33010983e-01, -4.87274170e-01,
       -8.39437664e-01, -1.47330952e+00, -1.10377789e+00, -1.39620876e+00,
       -5.32369077e-01, -7.71480799e-01, -9.91297886e-02, -7.60024667e-01,
        2.13649821e+00, -9.33478415e-01, -1.06567733e-01, -1.32163048e+00,
       -1.42974830e+00, -4.14772868e-01, -6.16311014e-01, -1.06981528e+00,
       -5.18608928e-01,  7.14960873e-01,  2.76600868e-01,  8.42292607e-01,
        2.60536164e-01,  1.83572543e+00, -9.72616613e-01, -3.70325208e-01,
       -1.17939258e+00,  4.19086784e-01,  1.04441035e+00,  1.28736511e-01,
       -1.25334406e+00, -7.43266523e-01,  1.20102644e+00, -1.21717203e+00,
       -1.50207663e+00,  6.07059896e-01,  3.88141155e-01,  2.61299402e-01,
        7.68335342e-01, -7.13861108e-01,  1.93729734e+00,  1.71357059e+00,
       -1.67841840e+00, -7.11296141e-01, -7.56470621e-01,  1.16134095e+00,
        6.13936603e-01, -3.35138381e-01, -1.03637636e+00,  9.43655372e-01,
        1.23864256e-01,  8.43876064e-01,  1.83305711e-01, -8.90030682e-01,
       -1.40898681e+00, -7.02044010e-01, -7.36973763e-01,  9.40429807e-01,
       -6.87081963e-02,  7.86132932e-01,  4.78387952e-01,  1.43077183e+00,
       -1.37428987e+00,  7.78530419e-01, -9.21964526e-01,  6.16950095e-01,
       -1.40051591e+00, -1.80042613e+00,  8.23574126e-01, -1.10526824e+00,
       -1.54978484e-01, -4.40705955e-01, -5.51060975e-01,  7.74940073e-01,
       -1.79169869e+00, -1.07235265e+00, -1.79856515e+00, -7.79466033e-01,
        7.69459724e-01,  8.47521186e-01, -2.97479808e-01,  7.06101477e-01,
       -1.54622233e+00, -9.60346997e-01,  6.18801601e-02,  1.26950192e+00,
        2.91045368e-01, -6.67546391e-01,  1.82559443e+00,  4.80283767e-01,
        1.00229371e+00, -6.30301476e-01, -4.61052716e-01,  1.21982582e-02,
        9.38549459e-01, -2.82906532e-01, -1.39076006e+00,  1.08093388e-01,
        7.32643127e-01, -9.74726081e-01,  1.63696945e+00,  7.66153395e-01,
       -7.71359503e-01, -7.88217068e-01,  9.87768099e-02, -3.92417014e-01,
       -1.15280712e+00,  8.48577738e-01, -6.72452748e-01,  1.87538278e+00,
        1.92983675e+00,  1.27980128e-01, -1.27468383e+00, -1.50374994e-01,
        6.76050112e-02,  2.70631164e-01, -2.28686199e-01,  5.97063184e-01,
        1.80500424e+00, -1.59903598e+00, -5.83325863e-01, -2.30937555e-01,
       -9.53022465e-02,  1.53510526e-01,  9.32832897e-01,  9.19884503e-01,
        1.01383366e-01, -3.33649844e-01,  1.05256939e+00,  1.10105550e+00,
       -3.06786865e-01, -9.73337352e-01, -7.58008122e-01,  2.52137533e-09,
        5.65284133e-01,  4.59943175e-01,  1.23413908e+00,  2.50653058e-01,
       -1.07443571e+00, -8.11547413e-02,  1.17373243e-01,  7.57325530e-01,
        5.70903897e-01, -5.79287589e-01, -1.35882103e+00, -1.22426760e+00,
        7.69660056e-01, -2.30913773e-01,  1.79681003e+00, -8.07382286e-01,
       -4.38165605e-01,  8.94738734e-01,  1.28103459e+00,  1.42940834e-01,
        6.28441811e-01, -8.62172306e-01,  2.02921510e+00, -1.03559923e+00,
       -7.99331725e-01,  1.21214926e+00,  2.04540372e-01, -4.65577036e-01,
       -1.23831344e+00,  1.21941315e-02,  4.96517360e-01, -6.16276801e-01,
        1.21805489e+00,  2.12005067e+00, -9.96730626e-01,  2.37623557e-01,
       -1.87166607e+00, -1.00939977e+00, -1.22155988e+00, -2.21534586e+00,
        1.51415551e+00,  4.64669794e-01, -7.96880662e-01, -1.43165708e+00,
       -1.27455011e-01, -8.56585577e-02, -9.52992976e-01,  3.81986350e-02,
       -1.36501217e+00, -4.95504916e-01,  7.29158044e-01,  2.09153876e-01,
        1.23825872e+00, -1.89663514e-01,  2.10301504e-01,  4.93996553e-02,
       -1.40057039e+00,  1.01606679e+00, -3.33498776e-01, -1.09128058e+00,
        7.50222683e-01, -3.29400867e-01,  1.60903513e+00, -5.37219107e-01,
       -9.71373141e-01,  1.02582979e+00, -4.96517360e-01,  6.56314373e-01,
       -7.81372607e-01,  3.00452769e-01, -1.16926290e-01, -5.39406061e-01,
       -1.12428129e+00, -1.02614832e+00, -5.87392986e-01,  1.43108702e+00,
       -1.00736773e+00, -1.32132983e+00, -5.93403764e-02, -1.38692093e+00,
        1.03887558e+00,  1.80946469e+00, -3.55395138e-01,  1.11066961e+00,
        1.55765188e+00, -1.38732374e+00, -1.28554428e+00,  7.99046218e-01,
        1.66410521e-01,  4.72658545e-01, -6.13722205e-01,  7.53459156e-01,
        1.86942130e-01,  1.32297921e+00, -6.36301041e-01, -8.32042456e-01,
        1.12831879e+00,  3.47629249e-01, -9.84715343e-01, -2.67124861e-01,
        5.00294209e-01, -6.40121341e-01, -1.18598413e+00, -2.76778579e-01,
        3.60107362e-01, -1.56824517e+00,  1.94866300e-01, -1.87486827e+00,
        2.04695866e-01,  7.36973763e-01, -1.08730662e+00, -1.12365615e+00,
        7.61430740e-01,  1.38897061e+00, -1.24734199e+00,  5.60505390e-01,
        1.19221175e+00,  1.25482991e-01,  2.19139171e+00,  1.28450370e+00,
       -3.28325897e-01,  1.37599511e-02, -1.50993884e+00, -3.69926393e-01,
        1.61841953e+00, -3.80891323e-01, -6.61474466e-01, -1.66341454e-01,
        7.91800737e-01,  4.78368998e-01,  4.10621986e-02,  8.14153254e-01,
        1.01903224e+00, -3.13268065e-01, -5.58776975e-01,  1.76393473e+00,
       -7.98203349e-01, -4.83804792e-01,  7.61069953e-01,  9.34295133e-02,
       -4.40386355e-01,  7.64521599e-01, -1.09234774e+00, -5.09303272e-01,
        1.05168350e-01, -8.76170874e-01, -7.41394758e-01, -9.69903886e-01,
        1.99873537e-01, -1.46009177e-01, -1.55439687e+00,  1.60654044e+00,
        5.23835957e-01,  3.42687041e-01,  1.17933476e+00,  4.46100146e-01,
       -1.34031630e+00, -8.56591538e-02, -1.21445513e+00, -2.95975447e-01,
        5.62448323e-01, -1.86134353e-01, -5.39551675e-01, -6.86343491e-01,
        1.13705553e-01,  6.22355118e-02,  2.04929709e+00,  3.59426774e-02,
        1.56671894e+00,  1.30790007e+00, -1.84081030e+00,  1.16850829e+00,
        8.11337829e-01, -4.27494437e-01,  2.73865402e-01, -2.78885737e-02,
        1.50343978e+00,  1.38005924e+00, -3.57569277e-01, -2.19392085e+00,
       -4.28298980e-01,  1.28040755e+00, -1.22802831e-01,  4.51830000e-01,
        7.49542952e-01,  1.13292980e+00,  3.26725572e-01,  1.51147389e+00,
        1.86941605e-02, -1.25613511e+00, -9.52118695e-01, -1.67831910e+00,
       -4.21364814e-01,  4.82135952e-01, -1.56615531e+00, -1.25180209e+00,
       -1.56772971e+00, -7.04354465e-01, -4.72433209e-01,  3.16067994e-01,
        1.26016128e+00,  1.20902993e-01, -1.37396193e+00, -2.94350773e-01,
        9.09175515e-01, -2.09273711e-01, -4.34690297e-01, -3.14300179e-01,
        1.55258393e+00,  4.56765443e-01,  9.69465137e-01,  2.06343189e-01,
       -4.69175339e-01, -1.00966728e+00, -1.63062012e+00, -2.95143902e-01,
       -8.39709640e-01, -1.32868922e+00, -2.14902282e+00,  7.12350965e-01,
       -9.60360020e-02,  9.35510039e-01,  1.95925152e+00, -1.91930637e-01,
        5.13249576e-01, -6.37024164e-01,  9.13259983e-01, -1.65947652e+00,
       -1.76909715e-01, -3.12831938e-01, -2.34614030e-01, -2.96241611e-01,
        1.25815713e+00,  1.15078855e+00, -1.49277732e-01,  2.00224519e+00,
       -1.80729711e+00,  1.41807222e+00,  1.24911511e+00,  1.49324238e+00,
        6.94692135e-01, -9.72604930e-01,  1.75089747e-01,  1.68543088e+00,
       -1.61991823e+00,  8.75356495e-02, -7.72535279e-02, -9.09061313e-01,
        5.12528598e-01, -1.06278455e+00,  8.42967164e-03, -1.19374716e+00,
       -2.40721032e-01, -1.58518791e-01,  4.23762053e-01, -1.00777471e+00,
        2.75780052e-01, -5.89945734e-01, -8.49596143e-01, -6.77206293e-02,
       -1.02325690e+00, -1.22652829e+00, -5.92825949e-01, -1.08018458e+00,
       -1.85351574e+00,  1.34593761e+00, -1.70255387e+00, -1.42495739e+00,
        1.32417941e+00, -2.45606408e-01,  1.22619939e+00, -7.74424016e-01,
       -5.70608556e-01,  8.33589077e-01,  9.69559968e-01,  3.11801612e-01,
       -5.02412081e-01,  9.25775230e-01, -7.20843494e-01,  4.25211117e-02,
        2.36986458e-01,  1.95014313e-01, -3.17343533e-01, -3.12469542e-01,
       -3.11442874e-02,  1.50903881e-01, -6.19958341e-01, -1.38689339e+00,
       -9.72231984e-01, -2.26023614e-01,  8.89268637e-01,  7.13640451e-01,
        1.27289939e+00,  6.23278856e-01, -4.40722778e-02,  1.26340127e+00,
        1.75388765e+00,  1.03240752e+00,  1.11186780e-01, -3.38335574e-01,
       -1.07879484e+00, -6.45456433e-01, -1.49673343e+00,  9.07228470e-01,
       -2.94247717e-01, -2.07667303e+00, -7.05533683e-01,  4.15494233e-01,
       -4.27221954e-01,  1.44536066e+00,  5.63044906e-01, -1.22646284e+00,
        8.71360525e-02,  2.88182735e-01, -1.36590886e+00, -5.41221142e-01,
       -1.72955906e+00,  2.57743150e-01,  2.49879360e-01,  2.33264476e-01,
       -1.09926629e+00, -5.66725791e-01,  1.06002378e+00, -6.33653879e-01,
        1.18701494e+00, -1.53277266e+00,  1.12933493e+00, -9.15147811e-02,
        6.23570442e-01,  1.30483556e+00,  1.42284250e+00,  5.91295846e-02,
        2.05287814e-01,  7.80858934e-01, -3.91004622e-01,  4.41873968e-01,
        4.89571154e-01, -1.03943908e+00, -2.38054261e-01, -1.61883205e-01,
        1.79205000e+00,  6.53807104e-01,  2.31280074e-01, -2.81337082e-01,
       -9.34220135e-01,  1.32083476e+00,  1.43017256e+00, -1.89836934e-01,
        3.51908207e-01,  1.35908473e+00, -3.74445826e-01,  8.75102162e-01,
        8.01368117e-01,  1.22596645e+00, -1.58413363e+00, -6.22219980e-01,
       -7.77619362e-01,  5.48208594e-01, -4.73354995e-01,  5.39273202e-01,
       -7.51822472e-01,  4.13870424e-01, -1.17825359e-01,  1.63643286e-01,
       -4.37617838e-01,  5.93171775e-01,  1.20742285e+00,  9.64689493e-01,
       -8.71612012e-01, -4.65887427e-01, -5.13930060e-02, -1.56422961e+00,
       -3.00297409e-01,  7.96011865e-01,  1.74516618e+00, -1.38133907e+00,
       -6.91094100e-01, -1.02842677e+00, -6.26869500e-01,  1.23359537e+00,
       -2.45405704e-01, -7.30352402e-01,  7.52378345e-01, -1.76479280e-01,
        5.66897273e-01,  1.08439267e+00,  4.82830167e-01, -2.14675725e-01,
        1.68481219e+00, -3.50991547e-01,  7.48334527e-01,  6.30331516e-01,
       -1.82824075e-01, -1.16762318e-01,  1.99803579e+00, -7.04604685e-01,
       -5.13325572e-01, -4.68544036e-01,  3.77331078e-01,  1.28821361e+00,
       -8.16071749e-01,  1.79621851e+00,  1.32637405e+00,  7.08267927e-01,
       -2.02180624e+00, -1.60852408e+00, -1.04875135e+00, -5.14498174e-01,
        9.94248569e-01, -1.37341022e+00,  7.38073528e-01, -2.81746030e-01,
       -2.60739505e-01, -1.51937914e+00, -5.38891077e-01, -3.38691562e-01,
        2.52137533e-09,  7.64771283e-01,  1.20003903e+00,  1.35346127e+00,
        6.81184471e-01, -1.02797174e+00,  3.00488561e-01, -2.02917352e-01,
       -5.93274593e-01,  1.72539622e-01, -5.63898802e-01, -1.18521535e+00,
       -1.04775369e+00, -8.61937523e-01, -4.27950144e-01,  4.47821438e-01,
       -1.70627344e+00,  3.34618300e-01, -2.99728811e-01, -1.70639381e-01,
        8.59170318e-01,  8.08233440e-01, -1.22529840e+00, -8.53946507e-01,
       -1.34822750e+00,  3.75247031e-01, -1.74735415e+00,  9.19162512e-01,
       -5.92612922e-01,  1.30170774e+00,  3.75523120e-01, -5.01554608e-01,
        5.26772499e-01, -3.08137327e-01, -6.08872294e-01, -1.13891304e+00,
        9.12167907e-01,  3.25011194e-01,  7.75326192e-01,  1.54247594e+00,
        1.07496345e+00, -1.78820574e+00,  5.50826430e-01, -1.08739018e+00,
        8.54848206e-01, -3.69314641e-01, -5.76644957e-01, -4.45993185e-01,
        1.97153783e+00, -1.51359022e-01,  1.63441896e+00,  2.57989049e-01,
       -2.01768175e-01,  2.20795557e-01, -6.06169820e-01,  1.68754089e+00,
       -1.48674715e+00,  1.06582737e+00, -1.57248998e+00, -9.60268259e-01,
       -7.17420220e-01, -8.08424234e-01, -1.14256418e+00,  1.62472761e+00,
       -8.56979549e-01,  2.52137533e-09,  1.35689902e+00,  3.80499139e-02,
       -7.65642375e-02, -7.56098211e-01,  1.36435950e+00, -1.30997157e+00,
        4.03890818e-01,  4.02796358e-01, -5.80672622e-01, -5.08536041e-01,
       -6.72696710e-01,  1.85972536e+00, -3.07124376e-01,  6.31844044e-01,
        1.27908516e+00, -1.44503546e+00, -7.00861067e-02, -1.06607270e+00,
        7.06281424e-01, -3.95565301e-01,  7.80673921e-01,  8.88644814e-01,
       -8.33828390e-01,  8.42301667e-01,  5.13023138e-01, -7.10727572e-01,
        3.36549580e-02, -8.67198884e-01, -1.34879768e+00, -1.00036347e+00,
        1.15944588e+00, -1.57123074e-01,  8.72430503e-01,  3.73197049e-01,
       -1.04866636e+00,  3.60411108e-01,  5.89785755e-01, -1.48734999e+00,
        1.47141623e+00,  1.37244892e+00, -9.91001904e-01, -8.94571900e-01,
       -1.80002308e+00, -1.07706606e-01, -6.81540728e-01, -5.96411109e-01,
        1.05616689e+00, -4.55463767e-01, -1.46750972e-01, -1.23859370e+00,
        6.56638741e-01,  5.42227805e-01, -1.96000886e+00,  5.58929145e-01,
        7.16381788e-01,  9.16898847e-01,  9.19419348e-01, -6.28950417e-01,
       -9.74351645e-01,  4.34106201e-01,  1.05484700e+00, -6.89301908e-01,
       -7.53849745e-01, -1.37168062e+00, -2.06811819e-02,  1.62036192e+00,
       -2.73980439e-01, -1.03315496e+00, -5.53131402e-02, -1.79852915e+00,
       -1.54189301e+00, -2.80739713e+00,  1.40058064e+00,  1.08997202e+00,
        1.89215994e+00, -3.65801692e-01, -2.12816238e-01, -1.70869255e+00,
        6.57252729e-01,  1.98648560e+00,  7.77367175e-01,  1.79529920e-01,
       -7.08482504e-01, -1.52647957e-01,  1.34879768e+00, -2.87171215e-01,
        5.14398515e-01, -4.77587044e-01,  1.09819090e+00, -1.21136749e+00,
       -1.54445040e+00,  1.39100552e+00, -7.41029859e-01, -1.03943944e+00,
       -6.90301299e-01, -1.69584394e+00, -1.08777058e+00, -1.18586469e+00,
        1.03080308e+00,  7.90045023e-01,  2.52137533e-09, -3.73356640e-01,
        3.39504242e-01,  5.44356182e-02, -1.06359911e+00, -1.30113113e+00,
       -1.12070225e-01, -7.44702637e-01, -7.82387145e-03,  9.31870401e-01,
       -6.26720846e-01,  1.64215946e+00,  1.80033594e-01, -4.57963437e-01,
       -3.60107183e-01, -6.21151745e-01, -2.07312679e+00, -1.36727977e+00,
       -3.58968079e-01,  1.12677805e-01, -8.82881641e-01, -9.75978851e-01,
       -2.06484509e+00, -7.65627384e-01, -1.03066063e+00, -6.41879976e-01,
       -7.02432930e-01, -1.42700708e+00, -5.63465655e-01, -6.11587763e-01,
       -9.96861637e-01, -6.05307281e-01, -8.61894608e-01,  3.78292054e-01,
        1.67642295e-01,  9.19867992e-01,  5.80587089e-01,  7.60153413e-01,
        5.18008947e-01,  9.81388211e-01, -4.39531624e-01, -3.94996675e-03,
        9.38302994e-01, -1.32030404e+00, -3.45922977e-01,  1.13231802e+00,
        7.11793900e-01, -5.47742210e-02, -2.05015093e-01,  1.62700891e+00,
       -2.46298504e+00, -6.19052708e-01, -9.03051794e-01, -9.18585241e-01,
       -3.76096547e-01,  1.13488090e+00,  6.74891889e-01,  1.31943214e+00,
       -5.49009562e-01,  4.74665835e-02, -2.20394111e+00,  3.88361156e-01,
       -1.48467407e-01, -1.54608858e+00, -4.76676017e-01, -8.33802462e-01,
       -5.06036103e-01, -4.47821438e-01, -1.31402993e+00,  1.04972088e+00,
        1.43230271e+00, -4.00611788e-01, -5.40147424e-01, -1.08059096e+00,
        1.44786417e+00,  4.15749336e-03, -2.32039645e-01, -1.12299931e+00,
        1.16792047e+00, -1.19620860e+00, -6.62989616e-01,  1.78847194e+00,
        2.32876554e-01, -9.07112122e-01, -1.43101156e+00, -1.38557565e+00,
       -1.18593895e+00, -5.36159754e-01, -5.63507736e-01,  1.01713145e+00,
       -1.57589746e+00, -2.34393775e-01,  1.90027758e-01,  3.52401048e-01,
       -9.79026318e-01, -6.64661884e-01,  2.05584240e+00,  1.29698086e+00,
        2.01306134e-01,  5.11175334e-01, -7.59625793e-01, -8.37291121e-01,
        4.80083972e-01, -6.56825960e-01,  3.13129783e-01,  4.22098815e-01,
        7.62755632e-01,  2.52137533e-09, -6.82698250e-01, -1.81277621e+00,
       -4.57722783e-01,  8.12475562e-01, -1.31415367e-01,  1.05358779e+00,
       -1.03041828e+00,  6.66020811e-01, -3.75327528e-01,  1.69508302e+00,
        2.68695891e-01, -6.71973526e-01,  9.11068320e-02,  9.25761580e-01,
       -1.89611465e-01, -1.52592027e+00, -6.12076819e-01,  9.34861839e-01,
       -4.20293301e-01,  3.45621884e-01, -4.06363815e-01,  5.35497904e-01,
        2.83845663e-01,  1.23077631e+00, -1.52525806e+00,  7.14736342e-01,
       -1.06419027e+00, -2.55727625e+00, -8.13365638e-01,  8.71836022e-02,
        1.67845976e+00, -3.20554823e-01,  1.39882457e+00, -2.62392759e-01,
        9.32150900e-01,  8.63386512e-01, -6.73764408e-01, -8.56932580e-01,
       -1.04681003e+00, -3.03545922e-01, -1.43289161e+00,  3.20554823e-01,
        5.62811852e-01,  5.64149499e-01, -5.78824580e-01, -4.81930435e-01,
       -4.86631662e-01,  1.30004370e+00, -9.01500523e-01,  2.75183827e-01,
       -1.65582883e+00, -1.65598676e-01, -8.87101769e-01, -2.90399700e-01,
       -1.58510542e+00,  7.42404103e-01, -1.06356668e+00, -8.20811450e-01,
        3.97303551e-01,  4.83008415e-01,  8.35965395e-01,  1.58935595e+00,
        1.17464042e+00, -9.19852734e-01, -8.21945548e-01, -9.52466249e-01,
        1.81416333e+00,  1.25019741e+00, -3.64775479e-01,  1.26464081e+00,
        1.29834735e+00, -4.05186743e-01, -8.47227812e-01,  1.49420524e+00,
       -1.96111247e-01,  8.08183730e-01, -1.04872096e+00,  1.80928421e+00,
       -1.79976511e+00,  1.15994477e+00, -7.82989860e-01, -1.92388308e+00,
        2.43441626e-01, -8.80867958e-01, -1.50765315e-01, -7.61663795e-01,
        2.28690791e+00, -8.84295523e-01,  2.15432674e-01,  1.31726575e+00,
       -8.82757843e-01,  9.95354295e-01,  5.10053150e-02,  2.10795835e-01,
       -6.69595718e-01, -9.40424263e-01, -9.42565948e-02,  9.61918831e-01,
        3.55264962e-01, -1.50919652e+00,  1.34886518e-01,  1.07878816e+00,
       -5.23665190e-01,  1.38591361e+00, -1.51242042e+00,  7.68737316e-01,
        7.68001735e-01,  3.73843014e-02,  2.97522396e-01,  1.53741091e-01,
        4.47543710e-01,  1.03823102e+00, -2.00171232e-01, -9.50062752e-01,
        1.37369919e+00,  2.52137533e-09,  2.70042658e-01,  2.74624452e-02,
        1.23735714e+00,  2.52137533e-09, -5.62175512e-01,  5.10393918e-01,
        1.48222864e+00,  1.15918791e+00,  1.13756394e+00, -6.17364287e-01,
        1.32432020e+00,  1.69035816e+00,  2.22038627e+00,  9.80960667e-01,
       -1.01105344e+00,  1.39485192e+00, -9.94905055e-01,  9.10575569e-01,
        1.04225194e+00, -2.99304128e-02, -1.00568724e+00, -2.98751974e+00,
        6.43408358e-01, -4.13142145e-01, -9.83847558e-01,  1.56703627e+00,
       -2.98318386e-01,  8.31577003e-01, -4.63743836e-01, -2.17252076e-01,
        1.53902268e+00, -6.45496726e-01,  8.94415677e-02, -3.90127152e-01,
        1.21487522e+00, -1.51980531e+00,  1.55995023e+00, -2.32086033e-01,
        1.87830478e-01, -5.05724669e-01,  1.90590763e+00,  1.67372906e+00,
       -7.40520775e-01, -1.61214221e+00, -2.17689741e-02, -6.07489586e-01,
        1.00380098e-02, -2.60536164e-01, -2.49423146e-01, -1.04605186e+00,
       -1.09281194e+00, -1.18242729e+00, -2.07915857e-01,  1.29653347e+00,
        1.08557081e+00,  7.06843793e-01, -9.50665325e-02, -2.61813402e+00,
       -1.47604609e+00, -4.71200019e-01,  6.91891670e-01, -1.68844946e-02,
       -6.30938590e-01,  7.15323865e-01,  3.48805875e-01, -9.90456104e-01,
        9.63526666e-01, -2.24763203e+00,  1.09992158e+00, -1.03232253e+00,
        2.91580409e-01,  2.88339019e-01, -6.05597973e-01, -5.33178866e-01,
       -1.18826520e+00,  1.17651451e+00,  2.94871420e-01,  1.05401897e+00,
       -7.85494447e-01, -1.74482918e+00,  1.47630513e-01, -9.33403194e-01,
       -4.63406026e-01,  3.85908931e-02, -1.38383329e+00, -3.57709289e-01,
        9.24619079e-01,  7.42582083e-01,  1.50893211e+00, -2.69298702e-01,
        9.84052658e-01, -1.08218513e-01, -8.70470345e-01,  6.89740062e-01,
       -2.77906239e-01, -8.93361509e-01,  1.06267416e+00, -9.79807675e-01,
        4.71109718e-01,  1.15032494e+00,  1.24443364e+00, -1.04108727e+00,
       -1.44320357e+00, -1.07233977e+00,  9.85843360e-01,  1.67290986e+00,
        1.93136132e+00,  1.37258482e+00,  1.55455768e-01,  3.06016562e-04,
       -7.06004143e-01,  8.66742969e-01,  1.07001829e+00, -1.33751667e+00,
       -1.34338582e+00,  3.70408475e-01,  1.10710406e+00, -8.25841352e-02,
        1.64177507e-01,  1.91560888e+00,  7.10087359e-01,  9.11585331e-01,
       -2.60465562e-01,  1.56963453e-01, -5.51039219e-01, -3.92088354e-01,
       -4.49697822e-01, -1.78949380e+00, -3.96679968e-01, -1.51844490e+00,
        5.73006392e-01,  1.20717692e+00, -1.12774348e+00, -1.41257310e+00,
        2.63867378e-01,  9.35478032e-01, -4.36721742e-01, -4.90889370e-01,
        1.15701354e+00, -3.93074304e-01,  3.03206563e-01,  8.59137416e-01,
       -2.01791263e+00,  1.41723645e+00,  1.32289290e+00,  1.06145537e+00,
        6.27787590e-01, -2.97142174e-02,  1.75878668e+00, -8.64966452e-01,
        1.32582355e+00, -1.20520067e+00,  1.55465949e+00,  3.86635125e-01,
        1.41060865e+00, -1.25307810e+00, -1.89845335e+00,  1.65743947e+00,
       -1.46729279e+00, -4.78900015e-01,  2.19212079e+00, -1.33150041e+00,
       -3.91295463e-01,  6.23176455e-01, -9.89343584e-01,  1.28662872e+00,
        1.90007365e+00,  6.23710454e-01, -8.93550038e-01, -4.34426159e-01,
       -1.00610125e+00, -9.27216470e-01, -1.09868646e+00, -9.70357418e-01,
       -4.86170202e-01, -5.99199295e-01, -1.21882260e+00, -4.71046001e-01,
       -1.40058964e-01,  1.22086513e+00, -2.83515781e-01,  7.30692223e-02,
        7.28320554e-02,  1.15976202e+00, -3.01799297e-01, -5.29032528e-01,
        1.15214109e+00,  1.26242423e+00,  1.48039794e+00,  1.64252281e-01,
        2.44784757e-01,  1.16989160e+00,  1.04605186e+00, -1.02699828e+00,
       -1.09618537e-01, -5.98306775e-01,  1.26259995e+00,  1.03559303e+00,
        3.21301877e-01, -9.43944275e-01,  1.36829793e+00, -5.22859216e-01,
       -1.66556990e+00,  1.38976431e+00,  2.23127437e+00,  1.73480189e+00,
       -3.24540424e+00,  2.24105436e-02, -8.51745084e-02, -7.15636760e-02,
        8.26741397e-01,  6.79848909e-01, -1.60527301e+00, -1.96219273e-02,
        1.08408511e+00,  8.77167106e-01,  1.12766623e+00,  8.37679803e-01,
       -1.24913931e+00,  7.49664783e-01, -1.82079583e-01,  1.32449973e+00,
        7.57331908e-01,  1.09194231e+00,  1.10922980e+00, -2.09884092e-01,
       -6.71199024e-01, -9.62716997e-01,  7.10765541e-01,  1.04050732e+00,
       -1.13338685e+00,  1.18170476e+00, -5.54341823e-02,  1.21836245e+00,
       -1.19435930e+00, -1.12505698e+00,  8.83226812e-01,  9.84128177e-01,
        8.11258197e-01,  9.64196265e-01, -6.73346460e-01,  1.32643580e-01,
       -1.68627405e+00, -1.57446671e+00, -1.35043287e+00,  4.20305401e-01,
        9.52866599e-02,  9.16416466e-01,  6.24948204e-01,  5.80543756e-01,
        1.25488803e-01, -4.76707637e-01,  1.00403214e+00,  4.60109472e-01,
        5.26790559e-01,  6.14854276e-01, -1.00072300e+00, -5.89517772e-01,
        1.14558458e+00,  1.50351465e+00, -2.45356679e+00, -5.39436400e-01,
        7.36056328e-01,  1.05232143e+00, -6.62603736e-01,  4.82251912e-01,
       -3.79745513e-01,  8.48825634e-01, -3.82982105e-01, -1.11680365e+00,
        3.84607941e-01,  1.62536287e+00,  1.19932890e+00,  1.11127019e+00,
        5.45282125e-01, -5.95415592e-01,  5.97706378e-01, -1.25929639e-01,
        5.95708728e-01, -1.81222546e+00, -3.93338576e-02, -1.59144926e+00,
       -1.28143048e+00, -1.59114277e+00,  1.03746438e+00,  6.70871496e-01,
        1.17617035e+00,  7.85235524e-01,  3.26991498e-01,  8.43598917e-02,
       -1.70849383e-01, -4.81264532e-01, -3.53561997e-01, -7.74838552e-02,
        1.05192614e+00,  6.38887346e-01,  8.65582526e-01, -7.35550940e-01,
        7.07318932e-02, -3.79751652e-01, -1.01735818e+00,  3.01582366e-01,
       -1.50776938e-01, -1.74263075e-01,  6.68328881e-01, -4.78855610e-01,
       -1.51704681e+00,  7.04590619e-01,  1.12878466e+00, -1.66436911e-01,
        1.05515957e+00,  1.09587538e+00,  1.04705489e+00, -1.80826914e+00,
        7.88280070e-01,  1.92021161e-01, -1.33564723e+00,  1.18468177e+00,
       -1.04666939e-02, -4.65179920e-01, -2.87374884e-01,  1.38287640e+00,
       -8.46773237e-02,  1.06677675e+00, -5.32345951e-01, -1.21552765e+00,
        5.46418309e-01, -1.24371648e+00, -2.03825402e+00,  2.14710808e+00,
       -1.19081306e+00,  7.74185777e-01, -5.37803710e-01,  3.15751702e-01,
       -1.54463756e+00, -4.23871070e-01,  3.49203646e-02, -6.31100118e-01,
        3.96284908e-02, -7.91099370e-01, -1.30605257e+00, -1.12637095e-01,
       -8.22605431e-01,  1.70985162e-01, -7.17300177e-01, -1.17086366e-01,
        1.00024724e+00,  1.25269449e+00, -6.31458342e-01, -7.92750239e-01,
       -2.19676065e+00, -3.42527479e-01, -2.35140711e-01,  1.69404578e+00,
        9.27684009e-01,  1.43727624e+00,  5.75484037e-01,  1.75928223e+00,
       -3.86806391e-02,  1.53416014e+00, -4.48809713e-01,  1.58640727e-01,
       -1.68289268e+00, -1.36830211e+00,  8.97010714e-02,  1.61046576e+00,
       -9.05691087e-01,  9.77867782e-01, -1.46750045e+00, -7.24388599e-01,
        1.00918865e+00, -1.66396439e+00,  1.48970354e+00, -1.03793061e+00,
        8.29516649e-01, -1.42887637e-01,  1.08324635e+00, -2.69016743e+00,
        9.90137458e-01,  4.22681272e-01,  1.09976196e+00,  9.98056293e-01,
        1.31415367e-01,  1.53821516e+00, -1.21724701e+00, -2.52079391e+00,
       -1.96387148e+00,  5.94193578e-01, -2.35916018e+00,  1.36894214e+00,
       -1.47843242e+00, -1.14402390e+00, -7.12621033e-01,  1.40210688e+00,
       -9.78757143e-01,  1.63805038e-01, -5.94802320e-01,  4.53494698e-01,
        1.47464073e+00, -1.13974321e+00, -1.10433459e+00, -3.27958703e-01,
       -1.66704929e+00, -7.05140412e-01, -8.10373366e-01,  7.94320166e-01,
        1.81436077e-01, -1.04372966e+00,  1.58977121e-01, -1.01251669e-01,
       -5.87420881e-01,  4.87192631e-01, -8.42253506e-01, -1.15519285e+00,
        6.05616033e-01, -4.71283384e-02, -1.04423475e+00,  1.74068436e-01,
       -1.48987985e+00, -1.31198919e+00, -8.09531689e-01, -1.86024070e-01,
       -1.88453868e-01,  2.07986295e-01, -3.87510985e-01,  8.45691562e-01,
        5.50570548e-01,  1.60196018e+00,  9.32386696e-01, -3.44829381e-01,
       -1.47469997e+00, -1.61202788e-01,  1.45197630e+00, -1.81570196e+00,
        7.07132459e-01, -1.52653742e+00,  1.36939466e+00, -1.92040813e+00,
        5.71622908e-01, -8.13832104e-01,  1.17977679e+00,  3.88038456e-02,
       -8.63386691e-01,  1.00664020e+00, -1.22097576e+00, -1.05356514e+00,
       -9.30101156e-01, -1.01671100e+00,  7.08667755e-01,  4.36895967e-01,
        7.99032927e-01,  6.71973526e-01,  3.38335574e-01,  1.17888427e+00,
        1.20386732e+00,  7.55146984e-03,  1.56923163e+00, -9.19785678e-01,
        5.87921798e-01,  6.32254243e-01, -8.04029405e-01,  1.52381730e+00,
       -8.23259354e-01, -6.99305177e-01, -5.35282075e-01,  1.02914286e+00,
       -1.06834388e+00, -1.49267924e+00, -7.22999752e-01,  1.71395767e+00,
       -1.49730414e-01,  1.05767334e+00,  2.21540123e-01,  5.98999202e-01,
       -6.21402152e-02, -5.36324680e-01, -2.34000707e+00,  1.09728467e+00,
        5.37863299e-02,  1.09377408e+00,  1.26532054e+00, -1.58165896e+00,
        5.90835035e-01, -9.24415708e-01,  2.52330583e-02,  2.52717555e-01,
       -2.40615159e-01,  2.06734872e+00,  3.77153575e-01, -1.68570685e+00,
        8.86383593e-01,  1.65437356e-01, -1.29821992e+00,  1.00640357e+00,
        1.73906183e+00, -1.06012642e+00,  1.48679642e-02,  1.31331611e+00,
        1.06802046e-01, -1.25041461e+00, -2.00611186e+00, -1.18624616e+00,
        1.34182322e+00,  1.40216768e-01, -1.19545531e+00, -1.41777948e-01,
       -5.14450669e-01, -3.76263440e-01,  1.27485946e-01,  1.40933478e+00,
        2.09506229e-01, -1.06033766e+00, -7.77095497e-01,  1.09987974e+00,
        1.53003538e+00, -1.26843452e+00, -1.38271129e+00, -8.34932685e-01,
        1.34680247e+00, -1.45076528e-01, -4.80496675e-01,  6.61809668e-02,
        2.36083293e+00,  6.03369653e-01, -5.67744374e-01,  4.48862553e-01,
       -1.74031043e+00, -6.99295461e-01,  8.32677484e-02,  1.02616355e-01,
       -6.66195214e-01,  1.02647758e+00, -1.20616281e+00, -7.77790844e-01,
        1.68779790e+00,  8.35557938e-01,  1.06931674e+00, -1.41513801e+00,
       -1.29653347e+00,  1.31649292e+00,  1.11874592e+00, -5.24664164e-01,
        1.09750915e+00,  1.58541843e-01,  1.02439833e+00,  2.02234507e+00,
       -1.58131325e+00,  2.42985815e-01,  2.68121690e-01, -3.10362607e-01,
        2.65918940e-01,  8.08955967e-01,  1.64685452e+00,  1.58277774e+00,
       -1.72063482e+00,  1.52051032e+00,  1.53890967e-01,  1.39499545e+00,
       -1.95657384e+00,  1.08527921e-01, -6.36709869e-01,  8.33687708e-02,
        1.90950837e-02,  3.44079465e-01,  1.24781024e+00, -5.45628250e-01,
       -8.28011990e-01,  1.44165218e+00, -1.92033267e+00,  3.61864150e-01,
        3.52296203e-01,  8.26111436e-01, -3.09774399e-01,  3.83598000e-01,
       -1.84456110e-01, -6.17244959e-01, -6.73857450e-01,  1.43733191e+00,
       -9.70358729e-01, -6.55066669e-01, -3.00707042e-01,  1.22789156e+00,
        7.24590898e-01,  1.32037127e+00, -1.12413394e+00,  1.12678814e+00,
       -7.13380456e-01, -1.31088722e+00, -2.19013882e+00, -3.42720479e-01,
       -1.04006362e+00, -1.41894150e+00,  1.55679643e-01, -8.51504624e-01,
       -3.47501248e-01,  1.80868697e+00, -1.63079000e+00, -2.30088279e-01,
       -3.10645175e+00,  9.42274928e-01,  6.64239824e-01, -1.28878582e+00,
        8.22671950e-01,  4.72791821e-01, -1.51589429e+00,  1.09594536e+00,
        8.19128633e-01, -5.77966809e-01,  4.35196787e-01,  6.21057034e-01,
       -1.13561714e+00, -6.88798904e-01, -9.02629793e-01,  4.20893669e-01,
       -1.43060243e+00,  7.70455599e-01, -2.04515934e+00,  1.20439539e-02,
        5.79566538e-01, -7.45664835e-01,  7.77896404e-01,  1.77606985e-01,
       -1.68478024e+00,  1.46008706e+00,  1.31531000e+00, -7.31352568e-01,
       -1.70942023e-01, -4.23811346e-01, -4.44654286e-01,  1.31443188e-01,
       -4.37325507e-01,  1.25585926e+00, -6.41858578e-01,  1.22219849e+00,
       -1.72879314e+00,  1.22188675e+00, -8.74977708e-01, -1.07678032e+00,
        3.89332995e-02,  1.08420205e+00, -7.87643790e-01, -5.10079324e-01,
       -8.85076299e-02,  4.27090526e-01, -2.52304286e-01, -1.08271110e+00,
       -2.10220075e+00,  1.38699472e+00,  1.16223820e-01, -5.12057424e-01,
       -9.60886180e-01, -7.63412535e-01, -1.60391736e+00, -6.11964464e-01,
       -1.77905285e+00, -1.35165823e+00,  1.02982593e+00, -1.09999275e+00,
        1.55627936e-01,  1.24493390e-01,  1.71079826e+00,  2.81660676e-01,
       -1.91783309e+00, -1.67782831e+00, -3.03214192e+00,  1.23134732e-01,
        1.61786914e+00, -6.68112814e-01,  1.31924772e+00,  1.08161759e+00,
        4.25597839e-02, -1.86022723e+00,  5.47547877e-01,  6.03059709e-01,
       -2.24297667e+00, -1.48913935e-02,  2.46873528e-01, -3.54680032e-01,
        3.77584010e-01,  1.87827075e+00,  9.43021894e-01,  1.84356582e+00,
       -5.40178001e-01,  7.49384940e-01,  3.59145612e-01, -2.19903302e+00,
        1.69737649e+00,  6.64439440e-01,  3.99581969e-01, -6.55513108e-01,
       -2.24812555e+00, -7.63272107e-01,  6.04550183e-01,  1.11851227e+00,
       -1.38486886e+00, -1.32965386e+00,  1.42201591e+00, -1.75359941e+00,
        3.30627151e-02,  1.71604872e+00,  6.22347534e-01,  8.37437153e-01,
       -1.13914108e+00,  7.38647819e-01, -2.89151967e-01,  2.52137533e-09,
        3.70985121e-01, -1.38869607e+00, -3.98573220e-01, -1.06778383e+00,
        1.88135505e-01,  3.73598754e-01, -9.57273304e-01, -5.45105577e-01,
        2.69376963e-01,  5.65912187e-01,  1.82815790e-02,  9.25478458e-01,
       -1.59710407e+00,  8.42576087e-01,  9.51506555e-01, -1.04233146e-01,
       -1.48585796e+00,  1.11032510e+00,  8.26465428e-01,  9.13452685e-01,
        1.18971300e+00, -4.38678026e-01, -2.21442890e+00, -1.18462873e+00,
        5.67162037e-01,  1.16355217e+00,  5.30776799e-01, -7.64084220e-01,
        1.18154988e-01, -4.13757890e-01, -5.43789148e-01,  1.07170391e+00,
       -1.28730178e+00, -1.43739712e+00,  1.19910985e-01, -1.12631309e+00,
        7.14133441e-01,  4.25445318e-01, -1.05034351e+00, -8.31570923e-01,
        7.91501462e-01, -1.83356225e+00, -4.42254186e-01, -1.20030534e+00,
        6.36986375e-01, -1.78748250e+00,  7.84954280e-02, -8.49813282e-01,
       -1.48648334e+00,  2.24131390e-01,  1.19951391e+00,  1.09300780e+00,
       -4.79151934e-01, -9.47771490e-01,  9.89095628e-01,  9.33232486e-01,
        5.49811006e-01,  7.90316403e-01, -5.27104080e-01, -6.34336352e-01,
       -9.17706788e-01,  1.70317161e+00,  8.65908861e-01, -1.16776133e+00,
       -2.11502886e+00,  6.45787299e-01, -9.93061364e-01, -8.38747203e-01,
        2.09629321e+00,  1.49358773e+00,  1.15153961e-01, -9.50218678e-01,
        7.41128385e-01,  2.09609896e-01, -1.83746803e+00,  3.92671913e-01,
        9.01648939e-01, -1.34932506e+00,  1.03236592e+00, -1.08626270e+00,
        1.02136016e+00,  9.62635279e-01,  1.73978269e-01,  4.79927719e-01,
        9.03584898e-01,  2.52657030e-02,  1.67895627e+00,  2.52137533e-09,
       -9.28847492e-01, -8.05352211e-01, -7.38984644e-01, -6.30618155e-01,
       -1.65142894e+00,  1.58393967e+00, -4.01575178e-01,  1.45998687e-01,
       -1.10433385e-01,  1.23094273e+00, -1.92767370e+00,  4.36933264e-02,
       -2.82156259e-01, -1.46622375e-01,  6.33767843e-02,  4.27051127e-01,
        1.09368645e-01, -3.69364113e-01,  3.59612167e-01,  1.15158713e+00,
       -4.44749177e-01,  2.52137533e-09, -3.42755288e-01, -9.69354451e-01,
        8.04039598e-01,  3.33360404e-01,  1.43782914e+00,  1.50998488e-01,
        4.54437971e-01, -1.67899156e+00, -1.07612419e+00, -5.66811502e-01,
        4.92610484e-01, -4.27006036e-01,  4.50238079e-01, -4.08593625e-01,
       -1.37257087e+00,  7.11907983e-01, -6.20467365e-01, -1.71606886e+00,
        4.97227192e-01, -1.40784645e+00,  1.07347572e+00,  1.15600801e+00,
       -2.07006001e+00,  1.49254650e-01,  1.65996492e+00,  2.04905108e-01,
        8.92257869e-01,  1.28741026e+00,  1.23790371e+00,  6.28179610e-01,
        4.82719451e-01,  8.83748353e-01, -1.45454407e+00,  1.39128983e+00,
        1.30663955e+00, -1.16638505e+00, -4.96581018e-01,  1.40848792e+00,
        1.74868190e+00,  2.89135754e-01, -6.11833692e-01,  1.15347838e+00,
       -1.06176555e+00, -8.00306618e-01,  1.15398467e+00, -2.34927654e-01,
        1.04493713e+00,  4.94058251e-01, -1.78932026e-01, -1.04740787e+00,
        5.65191686e-01, -4.39132303e-01,  1.18773282e+00,  9.54493999e-01,
        1.21811235e+00, -1.97794855e+00, -2.98703372e-01,  5.09864569e-01,
        5.25464177e-01, -4.34993058e-01, -1.04334402e+00, -2.01033711e+00,
       -7.18979314e-02,  1.77916408e+00, -8.72567713e-01,  2.13550776e-01,
       -1.65596652e+00, -1.26668602e-01,  1.40753317e+00,  9.50573921e-01,
       -1.47904253e+00,  9.61169958e-01, -9.36556518e-01,  1.04660141e+00,
        8.97715330e-01, -5.38874328e-01, -1.50943086e-01,  1.22131324e+00,
       -2.11558640e-01,  1.25262821e-02,  1.13638628e+00,  1.25657654e+00,
       -2.91762084e-01, -8.84009480e-01,  4.73902106e-01,  1.14728761e+00,
        1.15145817e-01,  2.39149189e+00,  6.81324005e-01,  3.33127409e-01,
        1.51810312e+00, -1.48467410e+00,  9.91054296e-01, -1.33371770e+00,
       -1.03584337e+00,  5.50529957e-01,  2.29535675e+00,  8.40221882e-01,
        1.18883288e+00, -1.76787293e+00,  7.74146914e-01,  7.80805409e-01,
        2.60816664e-01,  7.88129210e-01,  2.00489342e-01,  5.42054772e-01,
        2.79881321e-02, -1.34305370e+00, -1.23454642e+00,  2.02978468e+00,
        8.48335147e-01, -1.26788306e+00,  6.62306249e-01,  2.78483480e-01,
        5.01643538e-01,  2.33175099e-01,  1.00958562e+00, -8.38705719e-01,
       -3.44101310e-01, -3.35420918e-04,  7.19860494e-01,  1.50139904e+00,
        1.08314252e+00, -1.98216185e-01,  2.18900621e-01,  1.57313555e-01,
       -6.74505591e-01, -9.00798678e-01, -7.36284673e-01,  6.32444620e-01,
        9.32666063e-02,  1.69812292e-01, -1.52469361e+00, -1.00567901e+00,
       -5.08951917e-02, -1.22749531e+00,  7.00032711e-01, -1.30818278e-01,
        1.10554051e+00, -1.22422862e+00, -1.07022953e+00,  3.84515852e-01,
        1.34942997e+00, -2.42310271e-01, -7.11228251e-01, -8.09917569e-01,
        9.28378284e-01,  2.68625617e-01,  1.82660055e+00, -1.97193372e+00,
       -6.27566338e-01,  3.28090578e-01,  9.98934388e-01,  1.94145754e-01,
       -2.18175203e-01, -9.22394872e-01,  1.14049757e+00, -1.68448734e+00,
       -6.82984948e-01,  6.17462434e-02, -1.91442871e+00,  2.76083350e-01,
       -1.62384033e+00,  2.04018760e+00,  1.66106975e+00, -1.02266240e+00,
        5.93444467e-01, -1.73926878e+00,  4.26543027e-01,  4.21062738e-01,
        6.89322948e-01,  8.09327781e-01,  1.43962193e+00,  5.47223270e-01,
       -8.47804487e-01,  6.25287414e-01,  1.27176893e+00, -8.66352975e-01,
       -1.35730612e+00, -6.44038320e-01, -5.29928505e-01,  7.69917667e-01,
        1.00551426e+00, -1.18782973e+00,  2.52137533e-09,  1.70206368e+00,
        1.31907248e+00,  7.66933501e-01, -1.47250026e-01, -1.03615701e+00,
        7.27661252e-01, -6.27312541e-01,  1.13435698e+00, -1.91933548e+00,
       -1.11598921e+00, -8.65189552e-01,  2.42887735e-01,  3.42273355e-01,
        1.36091220e+00,  1.78379107e+00,  1.07082760e+00, -1.28792036e+00,
        5.93511105e-01,  7.86373556e-01, -9.11418736e-01,  6.19874239e-01,
        1.22324765e+00, -1.25153407e-01,  9.02397513e-01,  9.07830834e-01,
        5.94602287e-01, -4.54112858e-01,  7.23754466e-01, -9.43479717e-01,
        4.63120371e-01,  3.15636635e-01, -1.91079438e+00,  1.78855085e+00,
       -6.45128012e-01, -1.07813883e+00,  3.14483792e-01,  1.45545378e-01,
       -5.76353908e-01, -9.00947988e-01, -2.31198281e-01,  1.08704793e+00,
        3.53296667e-01,  9.93906185e-02,  2.78700262e-01, -1.13426797e-01,
        1.11337590e+00, -4.07081872e-01,  7.81501532e-01, -5.55807769e-01,
        2.30748534e-01,  6.53929412e-01,  1.88625848e+00, -1.60251534e+00,
        5.30545592e-01,  4.67784345e-01, -2.58586478e+00,  3.67107302e-01,
       -4.58197892e-02, -2.36320901e+00, -3.61806303e-01,  2.00125635e-01,
        8.95249844e-01,  1.90663978e-01, -5.24951220e-01,  2.26062608e+00,
       -3.61278802e-01,  1.14498758e+00, -1.13815475e+00, -4.45179760e-01,
       -6.80013239e-01, -3.41766961e-02, -1.97064734e+00,  3.86230290e-01,
       -1.49179244e+00,  8.20911646e-01, -6.23967528e-01, -1.23640811e+00,
        9.14072931e-01, -6.54998779e-01,  1.26121557e+00,  7.05134094e-01,
        9.34592128e-01,  5.50522566e-01,  8.77176464e-01,  9.39901173e-01,
        1.31021559e+00,  2.72225738e-01, -5.21915793e-01,  3.52922916e-01,
       -3.49220514e-01,  1.16673028e+00,  1.98647571e+00,  6.03070915e-01,
       -3.71085882e-01,  8.99148643e-01, -1.01416838e+00,  5.58782101e-01,
       -5.88442922e-01,  1.16553807e+00, -3.83189768e-01,  1.94141114e+00,
       -1.12911665e+00, -1.71205437e+00,  2.14903906e-01,  3.40899348e-01,
        1.27830040e+00, -3.55857998e-01, -9.19587493e-01,  6.22174323e-01,
       -1.13651288e+00, -1.73426783e+00,  8.98277760e-01,  1.28753185e+00,
        9.11374569e-01, -1.44899142e+00,  1.55931830e+00,  5.78450203e-01,
        7.94759691e-02, -4.18495387e-01,  2.30905935e-01, -9.91679549e-01,
       -1.13720703e+00, -2.97687680e-01, -8.50041568e-01, -2.02972859e-01,
        1.29800832e+00,  1.19777787e+00, -9.45872962e-02,  7.82854855e-02,
        1.16648960e+00, -7.66074002e-01,  6.62469447e-01, -5.43227732e-01,
        3.85212809e-01,  4.43739891e-02, -6.84951782e-01, -4.87346947e-01,
        1.50484371e+00,  2.86701828e-01, -9.92241561e-01,  5.66014707e-01,
        1.67383242e+00,  5.21045923e-01, -5.87139070e-01,  7.37455964e-01,
       -2.08220035e-02,  4.53204572e-01,  9.00847495e-01, -1.97009611e+00,
        9.02446151e-01,  5.08228779e-01, -9.81342569e-02,  9.35920715e-01,
       -1.10255218e+00, -6.13101959e-01,  1.59213436e+00,  1.10501885e+00,
       -8.26702595e-01, -4.66450185e-01,  7.08460331e-01, -4.29631770e-01,
        4.20150250e-01, -5.67207634e-01,  8.68267417e-01,  9.05443847e-01,
        1.74731922e+00, -8.06930482e-01,  1.17990541e+00,  1.22297692e+00,
        1.06319964e+00,  5.06601930e-01, -1.48759997e+00, -1.42530620e-01,
        1.32544661e+00, -1.36063886e+00,  4.54429567e-01,  1.42661679e+00,
       -7.75735736e-01,  1.91679609e+00,  2.57461518e-01, -2.49520734e-01,
        3.34763348e-01, -1.39861000e+00,  5.80862939e-01, -1.07741988e+00,
        1.37151551e+00,  8.76352429e-01, -5.71200311e-01, -9.99775708e-01,
       -1.68543291e+00, -8.43033850e-01,  5.56330323e-01,  3.32359344e-01,
        6.10891163e-01, -6.51752055e-01, -3.47633988e-01,  2.52137533e-09,
       -8.79209220e-01, -6.75288439e-02, -2.28809357e+00, -1.28418756e+00,
        7.84773946e-01, -1.15608290e-01, -7.06322640e-02, -6.47600889e-01,
       -1.03990889e+00, -1.41981173e+00,  9.12676215e-01, -1.11221826e+00,
        1.71390367e+00, -2.63887141e-02, -2.77875245e-01,  1.40136674e-01,
        2.07026005e+00, -3.30641687e-01, -2.75478095e-01, -7.26277590e-01,
       -4.45011109e-01,  5.06712675e-01,  1.06164336e+00,  9.27989125e-01,
        7.54836619e-01, -1.54057038e+00,  2.37677097e-01,  1.26860321e+00,
        2.44723216e-01,  1.01641023e+00,  2.34466776e-01, -2.79618204e-01,
       -8.87161791e-01, -4.38705117e-01,  2.28002071e-01,  7.07252860e-01,
        6.17514670e-01,  1.44378150e+00, -8.71473968e-01,  9.62307036e-01,
        2.08484221e+00,  1.18666220e+00, -5.51974654e-01,  5.01024842e-01,
        5.68920672e-01,  1.19094633e-01,  1.05595458e+00,  1.80843747e+00,
       -7.27501869e-01, -9.43890929e-01, -8.70165408e-01, -1.01949356e-01,
        8.05582583e-01, -4.74405795e-01, -1.44022000e+00, -7.52650082e-01,
       -1.09904692e-01,  5.40177166e-01, -1.06835234e+00, -1.52714953e-01,
       -8.84254634e-01,  2.86161937e-02,  1.07291913e+00,  1.01216555e+00,
        6.16686642e-01, -2.20204085e-01, -7.68325925e-02, -1.38174665e+00,
       -6.68051779e-01,  1.49322200e+00,  1.82079583e-01,  1.26490963e+00,
        7.95798361e-01, -1.44060433e+00,  1.50335222e-01, -1.02983248e+00,
       -7.11279958e-02,  7.78945506e-01, -6.55722201e-01, -7.69940138e-01,
       -1.27461159e+00, -9.73057389e-01,  1.50014126e+00, -1.13806792e-01,
       -6.56868219e-01, -9.51799631e-01,  1.81950912e-01, -4.40159664e-02,
        3.78695726e-01,  2.24824691e+00, -1.10210145e+00, -1.80617556e-01,
       -1.41611969e+00, -1.14704323e+00, -1.08095622e+00, -1.40328848e+00,
        2.63262272e-01, -9.53345120e-01, -3.57140750e-01]]
        f = spectre.factors.QuantileClassifier()
        f.bins = 5
        result = f.compute(torch.tensor(data))
        expected = pd.qcut(data[0], 5).codes
        assert_array_equal(result[0], expected)

    def test_ic(self):
        # test InformationCoefficient
        data = np.array([
            [0.7, 0.3, 0, 0.1, 0.2, -0.1, -0.3, 0.4, -0.5, -0.3],
            [0.1, 0.4, 0.3, 0.4, -0.2, 0.2, 0.5, -0.3, -0.7, -0.9],
            [-0.1, -0.4, -0.3, -0.4, 0.2, -0.2, -0.5, 0.3, 0.7, 0.9],
            [5, 4, 3, 2, 1, 0, -1, -2, -3, -4]
        ])
        # shuffle data
        idx = np.tile(np.random.rand(data.shape[1]).argsort(), (4, 1))
        data = np.take_along_axis(data, idx, axis=1)
        # create data loader
        now = pd.Timestamp.now(tz='UTC').normalize()
        index = [[now]*data.shape[1], list('abcdefghij')]
        index = pd.MultiIndex.from_arrays(index, names=('date', 'asset'))
        df_ab = pd.DataFrame(data.T, columns=['a', 'b', 'c', 'r'], index=index)
        temp_loader = spectre.data.MemoryLoader(df_ab)
        # calc ic
        engine = spectre.factors.FactorEngine(temp_loader)
        a = spectre.factors.ColumnDataFactor(inputs=['a'])
        b = spectre.factors.ColumnDataFactor(inputs=['b'])
        c = spectre.factors.ColumnDataFactor(inputs=['c'])
        r = spectre.factors.ColumnDataFactor(inputs=['r'])
        ica = spectre.factors.InformationCoefficient(a, r)
        icb = spectre.factors.InformationCoefficient(b, r)
        icc = spectre.factors.InformationCoefficient(c, r)
        ica_weighted = spectre.factors.RankWeightedInformationCoefficient(a, r, 5)
        icb_weighted = spectre.factors.RankWeightedInformationCoefficient(b, r, 5)
        icc_weighted = spectre.factors.RankWeightedInformationCoefficient(c, r, 5)
        engine.add([ica, icb, icc, ica_weighted, icb_weighted, icc_weighted],
                   ['ica', 'icb', 'icc', 'ica_weighted', 'icb_weighted', 'icc_weighted'])
        df_ret = engine.run(now, now, delay_factor=False)
        self.assertAlmostEqual(0.707, df_ret.ica.iloc[0], 3)
        self.assertAlmostEqual(0.716, df_ret.icb.iloc[0], 3)
        self.assertAlmostEqual(-0.716, df_ret.icc.iloc[0], 3)
        self.assertAlmostEqual(0.747, df_ret.ica_weighted.iloc[0], 3)
        self.assertAlmostEqual(0.633, df_ret.icb_weighted.iloc[0], 3)
        self.assertAlmostEqual(-0.633, df_ret.icc_weighted.iloc[0], 3)

        # test nans
        ica_weighted = spectre.factors.RankWeightedInformationCoefficient(a, r, 3, mask=a > 0)
        engine.remove_all_factors()
        engine.add(ica_weighted, 'ica_weighted')
        df_ret = engine.run(now, now, delay_factor=False)
        self.assertAlmostEqual(0.518, df_ret.ica_weighted.iloc[0], 3)

        # test IR
        ir_fct = ica.to_ir(3)
        x = torch.tensor([
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1],
        ])
        ir = ir_fct.compute(x)

        expected = np.array([np.nan, 2.1213, 2.1213, 3.5355, 1.4142])[:, None].repeat(5, 1)
        assert_almost_equal(expected, ir, decimal=4)

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

        self.assertEqual(df.loc[("2019-01-11", 'MSFT'), 'ma'],
                         df.loc[("2019-01-10", 'MSFT'), 'close'])

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

        self.assertEqual(df.loc[("2019-01-11", 'MSFT'), 'ma'],
                         df.loc[("2019-01-10", 'MSFT'), 'close'])

    def test_linear_regression(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)

        f = spectre.factors.RollingLinearRegression(10, None, spectre.factors.OHLCV.close)
        engine.add(f.coef, 'slope')
        engine.add(f.intercept, 'intcp')

        df = engine.run("2019-01-01", "2019-01-15")
        result = df.loc[(slice(None), 'AAPL'), 'slope']
        assert_almost_equal(
            [-0.555879, -0.710545, -0.935697, -1.04103, -1.232, -1.704182,
             -0.873212, -0.640606, 0.046424], result, decimal=5)
        assert_array_equal(['slope', 'intcp'], df.columns)

        engine.remove_all_factors()
        f = spectre.factors.RollingLinearRegression(
            10, x=spectre.factors.OHLCV.close, y=spectre.factors.OHLCV.open)
        engine.add(f.coef, 'slope')
        df = engine.run("2019-01-01", "2019-01-15")
        result = df.loc[(slice(None), 'AAPL'), 'slope']
        assert_almost_equal(
            [0.72021, 0.78893, 0.73185, 0.74825, 0.74774, 0.74796, 0.86587,
             0.8005 , 0.74089], result, decimal=5)


        # test RollingMomentum
        engine.remove_all_factors()
        f = spectre.factors.RollingMomentum(5)
        engine.add(f.gain, 'gain')
        engine.add(f.accelerate, 'accelerate')
        engine.add(f.intercept, 'intcp')
        engine.add(spectre.factors.OHLCV.close, 'close')

        df = engine.run("2019-01-01", "2019-01-15")
        result = df.xs('AAPL', level=1)

        from sklearn.linear_model import LinearRegression
        x = np.stack([np.arange(5), np.arange(5) ** 2]).T
        reg = LinearRegression().fit(x, result.close[-5:])
        assert_almost_equal(reg.coef_, result[['gain', 'accelerate']].iloc[-1])
        self.assertAlmostEqual(reg.intercept_, result.intcp.iloc[-1])

        reg = LinearRegression().fit(x, result.close[-6:-1])
        assert_almost_equal(reg.coef_, result[['gain', 'accelerate']].iloc[-2])
        self.assertAlmostEqual(reg.intercept_, result.intcp.iloc[-2])

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
        rf2 = 2 ** f
        engine.add(f, 'f')
        engine.add(f2, 'f^2')
        engine.add(-f, '-f')
        engine.add(f + f2, 'f+f2')
        engine.add(f - f2, 'f-f2')
        engine.add(f * f2, 'f*f2')
        engine.add(f / f2, 'f/f2')
        engine.add(f % f2, 'f%f2')

        engine.add(f > f2, 'f>f2')
        engine.add(f < f2, 'f<f2')
        engine.add(f >= f2, 'f>=f2')
        engine.add(f <= f2, 'f<=f2')
        engine.add(f == f2, 'f==f2')
        engine.add(f != f2, 'f!=f2')

        engine.add(2 + f, '2+f')
        engine.add(2 - f, '2-f')
        engine.add(2 * f, '2*f')
        engine.add(2 / rf2, '2/rf2')

        import operator
        self.assertRaises(TypeError, operator.mod, 2., f)

        engine.add(2 > f, '2>f')
        engine.add(2 < f, '2<f')
        engine.add(2 >= f, '2>=f')
        engine.add(2 <= f, '2<=f')
        engine.add(2 == f, '2==f')
        engine.add(2 != f, '2!=f')

        t = spectre.factors.OHLCV.volume.top(1)
        b = spectre.factors.OHLCV.volume.bottom(1)
        engine.add(t, 't')
        engine.add(t & b, 't&b')
        engine.add(t | b, 't|b')
        engine.add(~t, '~t')

        self.assertRaises(TypeError, operator.and_, 2., b)
        self.assertRaises(TypeError, operator.or_, 2., b)

        result = engine.run("2019-01-01", "2019-01-05")

        f = np.array([158.61, 101.30, 145.23, 102.28, 104.39])
        f2 = f ** 2
        rf2 = 2 ** f
        assert_array_equal(result['f^2'], f2)
        assert_array_equal(result['-f'], -f)
        assert_array_equal(result['f+f2'], f + f2)
        assert_array_equal(result['f-f2'], f - f2)
        assert_array_equal(result['f*f2'], f * f2)
        assert_array_equal(result['f/f2'], f / f2)
        assert_array_equal(result['f%f2'], f % f2)

        assert_array_equal(result['2+f'], 2 + f)
        assert_array_equal(result['2-f'], 2 - f)
        assert_array_equal(result['2*f'], 2 * f)
        assert_almost_equal(result['2/rf2'], 2 / rf2)

        assert_array_equal(result['f>f2'], f > f2)
        assert_array_equal(result['f<f2'], f < f2)
        assert_array_equal(result['f>=f2'], f >= f2)
        assert_array_equal(result['f<=f2'], f <= f2)
        assert_array_equal(result['f==f2'], f == f2)
        assert_array_equal(result['f!=f2'], f != f2)

        assert_array_equal(result['2>f'], 2 > f)
        assert_array_equal(result['2<f'], 2 < f)
        assert_array_equal(result['2>=f'], 2 >= f)
        assert_array_equal(result['2<=f'], 2 <= f)
        assert_array_equal(result['2==f'], 2 == f)
        assert_array_equal(result['2!=f'], 2 != f)

        t = np.array([False, True, True, False, False])
        b = ~t
        assert_array_equal(result['t&b'], t & b)
        assert_array_equal(result['t|b'], t | b)
        assert_array_equal(result['~t'], b)

    def test_cache(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        # cache failure test
        ma5 = spectre.factors.MA(5)
        ma10 = f =spectre.factors.MA(10)
        ma5._keep_cache = True

        engine = spectre.factors.FactorEngine(loader)

        engine.remove_all_factors()
        engine.add(ma5, 'a')
        df = engine.run("2019-01-01", "2019-01-05")
        shape1 = ma5._cache.shape

        engine.remove_all_factors()
        engine.add(ma10, 'a')
        df = engine.run("2019-01-01", "2019-01-05")
        shape2 = ma5._cache.shape
        # engine.empty_cache()

        engine.remove_all_factors()
        engine.add(ma5, 'a')
        df = engine.run("2019-01-01", "2019-01-05")
        shape3 = ma5._cache.shape

        assert_array_equal(shape1, shape2)
        assert_array_equal(shape1 <= shape3, [True, True])
        assert_array_equal(shape3, [2, 13])


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
            inputs=[spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.open),
                    spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.close)],
            multiprocess=True
        ), 'f')
        engine.add(spectre.factors.MA(
            win=10,
            inputs=[spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.open) *
                    spectre.factors.AdjustedColumnDataFactor(spectre.factors.OHLCV.close)],
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
        spectre.parallel.DeviceConstant.clean()

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
