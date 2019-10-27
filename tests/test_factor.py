import unittest
import spectre
import pandas as pd
import numpy as np
from zipline.pipeline.factors import Returns


class TestFactorLib(unittest.TestCase):
    def test_1_CsvDirDataLoader(self):
        # test index type check
        loader = spectre.factors.CsvDirLoader('./data/daily/')
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)
        loader = spectre.factors.CsvDirLoader('./data/daily/', index_col='date', )
        self.assertRaises(AssertionError, loader.load, '2019-01-01', '2019-01-15', 0)

        def test_df_first_last(tdf, col, vf, vl):
            self.assertAlmostEqual(tdf.loc[tdf.index[0], col], vf)
            self.assertAlmostEqual(tdf.loc[tdf.index[-1], col], vl)

        loader = spectre.factors.CsvDirLoader(
            './data/daily/', index_col='date', parse_dates=True, )
        # test cache
        start, end = pd.Timestamp('2015-01-01', tz='UTC'), pd.Timestamp('2019-01-15', tz='UTC')
        self.assertIsNone(loader._load_from_cache(start, end, 0))
        df = loader.load('2019-01-01', '2019-01-15', 0)
        self.assertIsNotNone(loader._load_from_cache(start, end, 0))

        # test backward
        df = loader.load('2019-01-01', '2019-01-15', 11)
        test_df_first_last(df.loc[(slice(None), 'AAPL'), :], 'close', 173.43, 158.09)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'close', 106.57, 105.36)

        # test value
        df = loader.load('2019-01-01', '2019-01-15', 0)
        test_df_first_last(df.loc[(slice(None), 'AAPL'), :], 'close', 160.35, 158.09)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'close', 103.45, 105.36)
        test_df_first_last(df.loc[(slice('2019-01-11', '2019-01-12'), 'MSFT'), :],
                           'close', 104.5, 104.5)
        df = loader.load('2019-01-11', '2019-01-12', 0)
        test_df_first_last(df.loc[(slice(None), 'MSFT'), :], 'close', 104.5, 104.5)

        loader = spectre.factors.CsvDirLoader(
            './data/5mins/', split_by_year=True, index_col='Date', parse_dates=True, )
        loader.load('2019-01-01', '2019-01-15', 0)

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
        np.testing.assert_array_equal(
            df.loc[(slice(None), 'AAPL'), 'CpVol'].values, (28065422, 33834032, 29426699))
        np.testing.assert_array_equal(
            df.loc[(slice(None), 'MSFT'), 'CpVol'].values, (28627674, 28720936, 32882983))

        engine.add(spectre.factors.DataFactor(inputs=('changePercent',)), 'Chg')
        df = engine.run('2019-01-11', '2019-01-15')
        np.testing.assert_array_equal(
            df.loc[(slice(None), 'AAPL'), 'Chg'].values, (-0.9835, -1.5724, 2.1235))
        np.testing.assert_array_equal(
            df.loc[(slice(None), 'MSFT'), 'Chg'].values, (-0.8025, -0.7489, 3.0232))

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
                return range(len(close))

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = [TestFactor]

            def compute(self, test_input):
                return np.cumsum(test_input)

        self.assertRaisesRegex(TypeError, ".*BaseFactors.*", TestFactor2)

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
        np.testing.assert_array_equal(df['test1'].values, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(df['test2'].values, [0, 1, 3, 6, 10, 15])

    def test_factors(self):
        ma = spectre.factors.MovingAverage(win=5)
        # 测试forward
        # 测试是否已算过的重复factor不会算2遍
        # 测试结果是否正确
        pass

    def test_filter_factor(self):
        loader = spectre.dataloader.form_csvdir('./test/data/')
        stdcol = spectre.StandardColumnNames.OHLCV(
            'uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume')
        engine = spectre.factor_engine(loader, stdcol)

        class testFactor2(spectre.BaseFactor):
            inputs = (spectre.StandardColumnNames.OHLCV.volume)

            def compute(self, out, volume):
                self.assertEqual(len(input), 120 + 3)
                out[:] = input['']

        universe = testFactor2(win=120).top(1)
        engine.add(universe)

        data = engine.run("2017-01-01", "2019-01-05")
        self.assertEqual(len(data.index.get_level_values(1).values), 1)
        self.assertEqual(data.index.get_level_values(1).values[0], 'AAPL')

    # def test_cuda_factors(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
