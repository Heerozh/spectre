import unittest
import spectre
import pandas as pd
import numpy as np
from zipline.pipeline.factors import Returns


class TestFactorLib(unittest.TestCase):
    def test_base_factor_assert(self):
        engine = spectre.factors.FactorEngine()
        self.assertRaisesRegex(
            ValueError,
            'Factor does not have `.name` attribute.*',
            engine.add, spectre.factors.BaseFactor()
        )
        self.assertRaisesRegex(
            IndexError,
            '.*IndexFactor.*',
            engine.run
        )

        class TestIndex(spectre.factors.IndexFactor):
            name = 'index'

            def compute(self, out, start):
                out[:] = range(5)

        index = TestIndex()

        class TestFactor(spectre.factors.BaseFactor):
            name = 'test{}'
            inputs = (index,)

            def compute(self, out, start, index):
                out[:] = sum(index)

        engine.add(TestFactor(win=3))

        self.assertRaisesRegex(
            KeyError,
            ".*exists.*",
            engine.add, TestFactor(win=3)
        )

        engine.add(index)
        # df = engine.run()
        # self.assertEqual(df.index.values, np.array([0, 1, 2, 3, 4]))
        # self.assertEqual(df['test3'], np.array([0, 1, 3, 6, 9]))

    def test_factor_tree(self):
        #测试forward
        #测试是否已算过的重复factor不会算2遍
        #测试结果是否正确
        pass

    def test_data_factor(self):
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
