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
        expected = scipy.stats.rankdata(x[1])
        expected[3] = np.nan
        assert_array_equal(expected, result[1])

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
        result = mad_fct.compute(x)
        expected = x.cpu().numpy()
        expected[0, 0] = -745.25
        expected[0, -2] = 750.25
        assert_almost_equal(expected, result.cpu().numpy())

        # test winsorizing
        factor = spectre.factors.WEEKDAY.winsorizing(0.2)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            [2., 3., 4., 0., 1., 2., 3., 4., 0., 1.], [0.2, 0.2])
        expected_msft = scipy.stats.mstats.winsorize(
            [2., 3., 4., 0., 1., 2., 4., 0., 1.], [0.2, 0.2])
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.WEEKDAY.winsorizing(0.001)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            [2., 3., 4., 0., 1., 2., 3., 4., 0., 1.], [0.001, 0.001])
        expected_msft = scipy.stats.mstats.winsorize(
            [2., 3., 4., 0., 1., 2., 4., 0., 1.], [0.001, 0.001])
        test_expected(factor, expected_aapl, expected_msft, 10)

        factor = spectre.factors.OHLCV.close.winsorizing(0.2)
        factor.groupby = 'asset'
        expected_aapl = scipy.stats.mstats.winsorize(
            [158.6100, 145.2300, 149.0000, 149.0000, 151.5500, 156.0300, 161.0000,
             153.6900, 157.0000, 156.9400], [0.2, 0.2])[:-1]
        expected_msft = scipy.stats.mstats.winsorize(
            [101.3000, 102.2800, 104.3900, 103.2000, 105.2200, 106.0000, 103.2000,
             103.3900, 108.8500], [0.2, 0.2])[:-1]
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
        self.assertAlmostEqual(0.707, df_ret.ica[0], 3)
        self.assertAlmostEqual(0.716, df_ret.icb[0], 3)
        self.assertAlmostEqual(-0.716, df_ret.icc[0], 3)
        self.assertAlmostEqual(0.747, df_ret.ica_weighted[0], 3)
        self.assertAlmostEqual(0.633, df_ret.icb_weighted[0], 3)
        self.assertAlmostEqual(-0.633, df_ret.icc_weighted[0], 3)

        # test nans
        ica_weighted = spectre.factors.RankWeightedInformationCoefficient(a, r, 3, mask=a > 0)
        engine.remove_all_factors()
        engine.add(ica_weighted, 'ica_weighted')
        df_ret = engine.run(now, now, delay_factor=False)
        self.assertAlmostEqual(0.518, df_ret.ica_weighted[0], 3)

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
