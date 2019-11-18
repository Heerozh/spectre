import unittest
import spectre
from numpy.testing import assert_array_equal
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestDataFactorLib(unittest.TestCase):
    def test_datafactor_value(self):
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/',
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.OHLCV.volume, 'CpVol')
        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'CpVol'].values,
                           (28065422, 33834032, 29426699))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'CpVol'].values,
                           (28627674, 28720936, 32882983))

        engine.add(spectre.factors.DataFactor(inputs=('changePercent',)), 'Chg')
        df = engine.run('2019-01-11', '2019-01-15')
        assert_array_equal(df.loc[(slice(None), 'AAPL'), 'Chg'].values,
                           (-0.9835, -1.5724, 2.1235))
        assert_array_equal(df.loc[(slice(None), 'MSFT'), 'Chg'].values,
                           (-0.8025, -0.7489, 3.0232))
