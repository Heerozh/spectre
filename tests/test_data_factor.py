import unittest
import spectre
from numpy.testing import assert_array_equal
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestDataFactorLib(unittest.TestCase):
    def test_datafactor_value(self):
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.OHLCV.volume, 'CpVol')
        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'CpVol'].values,
                           (28065422, 33834032))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'CpVol'].values,
                           (28627674, 28720936))

        engine.add(spectre.factors.DataFactor(inputs=('changePercent',)), 'Chg')
        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'Chg'].values,
                           (-0.9835, -1.5724))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'Chg'].values,
                           (-0.8025, -0.7489))

        engine.remove_all_factors()
        engine.add(spectre.factors.OHLCV.open, 'open')
        df = engine.run('2019-01-11', '2019-01-15', delay_factor=False)
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'open'].values,
                           (155.72, 155.19, 150.81))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'open'].values,
                           (104.65, 104.9, 103.19))

        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'open'].values,
                           (155.72, 155.19, 150.81))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'open'].values,
                           (104.65, 104.9, 103.19))