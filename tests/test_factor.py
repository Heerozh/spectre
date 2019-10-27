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

    def test_2_BaseFactor(self):
        a = spectre.factors.BaseFactor(win=2)
        b = spectre.factors.BaseFactor(win=3, inputs=(a,))
        c = spectre.factors.BaseFactor(win=3, inputs=(b,))
        self.assertEqual(c._get_total_backward(), 5)

        a1 = spectre.factors.BaseFactor(win=10)
        a2 = spectre.factors.BaseFactor(win=5)
        b1 = spectre.factors.BaseFactor(win=20, inputs=(a1, a2))
        b2 = spectre.factors.BaseFactor(win=100, inputs=(a2,))
        c1 = spectre.factors.BaseFactor(win=100, inputs=(b1,))
        self.assertEqual(a1._get_total_backward(), 9)
        self.assertEqual(a2._get_total_backward(), 4)
        self.assertEqual(b1._get_total_backward(), 28)
        self.assertEqual(b2._get_total_backward(), 103)
        self.assertEqual(c1._get_total_backward(), 127)

    def test_3_datafactor(self):
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

    def test_4_custom_factor(self):
        loader = spectre.factors.CsvDirLoader(
            './data/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)

        class TestFactor(spectre.factors.BaseFactor):
            inputs = [spectre.factors.OHLCV.close]

            def compute(self, close):
                return range(len(close))

        class TestFactor2(spectre.factors.BaseFactor):
            inputs = [TestFactor]

            def compute(self, test_input):
                return np.cumsum(test_input)

        self.assertRaisesRegex(TypeError, ".*BaseFactor.*", TestFactor2)

        test_f1 = TestFactor()

        class TestFactor2(spectre.factors.BaseFactor):
            inputs = [test_f1]

            def compute(self, test_input):
                return np.cumsum(test_input)

        engine.add(test_f1, 'test')
        self.assertRaisesRegex(KeyError, ".*exists.*",
                               engine.add, TestFactor(), 'test')

        engine.add(TestFactor2(), 'test2')
        df = engine.run('2019-01-11', '2019-01-15')
        np.testing.assert_array_equal(df.values,
                                      [[0,  0],
                                       [1,  1],
                                       [2,  3],
                                       [3,  6],
                                       [4, 10],
                                       [5, 15]])

    def test_factor_tree(self):
        # 测试forward
        # 测试是否已算过的重复factor不会算2遍
        # 测试结果是否正确
        pass

    def test_data_factor(self):
        # 测试复用的factor是否没有重新计算
        def test_df(df, col, vf, vl):
            self.assertAlmostEqual(df.loc[df.index[0], col], vf)
            self.assertAlmostEqual(df.loc[df.index[-1], col], vl)

        # test data_factors
        data_factors = spectre.factor.DataFactor.form_csvdir(
            './test/data/5mins/', split_by_year=True,
            read_csv={index_col: 'Date'})

        # should 0 index[Date, symbols], 1 Open, 2 High ...
        self.assertIsInstance(data_factors[0], spectre.factor.IndexFactor)
        self.assertEqual(data_factors[0].name, 'Date')
        self.assertEqual(data_factors[1].name, 'Open')
        close = data_factors[4]

        # test dataframes to bigdf
        df = spectre.factor.DataFactor.mergedf({
            'AAPL': pd.read_csv('./test/data/daily/AAPL.csv', index_col='date'),
            'MSFT': pd.read_csv('./test/data/daily/MSFT.csv', index_col='date'),
        })
        test_df(df.loc[(slice(None), 'AAPL'), :], 'close', 108.88, 243.94)
        test_df(df.loc[(slice(None), 'MSFT'), :], 'close', 46.24, 141.24)

        # test datafactors
        # @expect_types(dfs=dataframe.DataFrame, OHLCV=list
        data_factors = spectre.factor.DataFactor({
            'AAPL': pd.read_csv('./test/data/AAPL.csv', index_col='date'),
            'MSFT': pd.read_csv('./test/data/MSFT.csv', index_col='date'),
        }, OHLCV=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'))
        # test index

        # factor
        engine = spectre.factor.Engine()
        engine.add(data_factors)

        test_df(loader.get('AAPL'), 'close', 108.88, 243.94)
        test_df(loader.get('MSFT'), 'close', 46.24, 141.24)

        # test csvdir loader, only read on run
        loader = spectre.dataloader.form_csvdir('./test/data/', split_by_year=True,
                                                read_csv={
                                                    index_col: 'date',
                                                    dtype: {
                                                        'changeOverTime': np.float32,
                                                    }
                                                })
        test_df(loader.get('AAPL'), 'close', 108.88, 243.94)
        test_df(loader.get('MSFT'), 'close', 46.24, 141.24)
        test_df(loader.get('AAPL'), 'changeOverTime', 0, 1.349802)
        test_df(loader.get('MSFT'), 'changeOverTime', 0, 2.02193)

    def test_add_factor(self):
        loader = spectre.dataloader.form_csvdir('./test/data/')
        stdcol = spectre.standard_column_names.standardColumnNamesBase()
        engine = spectre.factor.Engine(loader, stdcol)

        class testFactor(spectre.BaseFactor):
            inputs = (spectre.factor.DataFactor.OHLCV.volume, 'uVolume')

            def compute(self, out, volume, uVolume):
                pass

        facotr = testFactor(win=1)
        self.assertRaisesRegex(
            AttributeError,
            "This factor requires specify StandardColumnName to OHLCV, but you're using: spectre.StandardColumnNames.Base",
            engine.add(facotr))

        stdcol = spectre.StandardColumnNames.OHLCV()
        engine = spectre.factor_engine(loader, stdcol)
        self.assertRaisesRegex(
            AttributeError,
            "This factor requires specify StandardColumnName to OHLCV, exsample: spectre.factor_engine(loader, OHLCV=()).",
            engine.add(facotr))

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

    def test_engine_select_factor(self):
        loader = spectre.dataloader.form_csvdir('./test/data/')
        engine = spectre.factor_engine(loader)

        adv = spectre.AverageDollarVolume(win=120)
        ma = spectre.MovingAverage(win=5)
        engine.add(adv, 'adv')
        engine.add(ma)  # default MA5
        engine.run()

    def test_factors(self):
        ma = spectre.factors.MovingAverage(win=5)

    def test_cuda_factors(self):
        pass


if __name__ == '__main__':
    unittest.main()
