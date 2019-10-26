import unittest
import spectre
import pandas as pd
import numpy as np
from zipline.pipeline.factors import Returns


class TestFactorLib(unittest.TestCase):
    def test_1_engine_dataloader(self):
        def test_df_first_last(tdf, col, vf, vl):
            self.assertAlmostEqual(tdf.loc[tdf.index[0], col], vf)
            self.assertAlmostEqual(tdf.loc[tdf.index[-1], col], vl)

        loader = spectre.factors.CsvDirDataLoader(
            './data/daily/',
            index_col='date',
            parse_dates=True,
        )
        df = loader.load('2019-01-01', '2019-01-15')
        test_df_first_last(df.loc['AAPL', :], 'close', 160.35, 158.09)
        test_df_first_last(df.loc['MSFT', :], 'close', 103.45, 105.36)
        test_df_first_last(df.loc['MSFT', '2019-01-11':'2019-01-12', :], 'close', 104.5, 104.5)

        loader = spectre.factors.CsvDirDataLoader(
            './data/5mins/',
            split_by_year=True,
            index_col='Date',
            parse_dates=True,
        )
        loader.load('2019-01-01', '2019-01-15')

        start = pd.Timestamp('2018-12-31 14:50:00', tz='America/New_York')
        end = pd.Timestamp('2019-01-02 10:00:00', tz='America/New_York')
        df = loader.load(start, end)
        test_df_first_last(df.loc['AAPL', :], 'Open', 157.45, 155.17)
        test_df_first_last(df.loc['MSFT', :], 'Open', 101.44, 99.55)

    def test_base_factor_assert(self):
        engine = spectre.factors.FactorEngine()
        self.assertRaisesRegex(IndexError, '.*IndexFactor.*',
                               engine.run
                               )


        class TestIndex(spectre.factors.IndexFactor):
            self.data = None

            def pre_compute(self, start, end):
                self.data = range(start, end)

            def compute(self, out):
                out[:] = self.data

        index = TestIndex()

        class TestFactor(spectre.factors.BaseFactor):
            inputs = (index,)

            def compute(self, out, index):
                out[:] = np.cumsum(index)

        engine.add(TestFactor(win=3), 'test')

        self.assertRaisesRegex(KeyError, ".*exists.*",
                               engine.add, TestFactor(win=3), 'test'
                               )

        engine.add(index, 'index')
        self.assertRaises(NotImplementedError, engine.run)
        # df = engine.run()
        # self.assertEqual(df.index.values, np.array([0, 1, 2, 3, 4]))
        # self.assertEqual(df['test3'], np.array([0, 1, 3, 6, 9]))

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
