import unittest
import spectre
import numpy as np
from numpy.testing import assert_array_equal
import torch
from os.path import dirname

data_dir = dirname(__file__) + '/data/'


class TestCustomFactorLib(unittest.TestCase):

    def test_custom_factor(self):
        # test backward tree
        a = spectre.factors.CustomFactor(win=2)
        b = spectre.factors.CustomFactor(win=3, inputs=(a,))
        c = spectre.factors.CustomFactor(win=3, inputs=(b,))
        self.assertEqual(c.get_total_backward_(), 5)

        a1 = spectre.factors.CustomFactor(win=10)
        a2 = spectre.factors.CustomFactor(win=5)
        b1 = spectre.factors.CustomFactor(win=20, inputs=(a1, a2))
        b2 = spectre.factors.CustomFactor(win=100, inputs=(a2,))
        c1 = spectre.factors.CustomFactor(win=100, inputs=(b1,))
        self.assertEqual(a1.get_total_backward_(), 9)
        self.assertEqual(a2.get_total_backward_(), 4)
        self.assertEqual(b1.get_total_backward_(), 28)
        self.assertEqual(b2.get_total_backward_(), 103)
        self.assertEqual(c1.get_total_backward_(), 127)

        # test inheritance
        loader = spectre.factors.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            index_col='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)

        class TestFactor(spectre.factors.CustomFactor):
            inputs = [spectre.factors.OHLCV.close]

            def compute(self, close):
                return torch.tensor(np.arange(np.prod(close.shape)).reshape(close.shape))

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = []

            def compute(self):
                return torch.tensor([1])

        engine.add(TestFactor2(), 'test2')
        self.assertRaisesRegex(ValueError, "The return data shape.*test2.*",
                               engine.run, '2019-01-11', '2019-01-15')
        engine.remove_all_factors()
        test_f1 = TestFactor()

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = [test_f1]

            def compute(self, test_input):
                return torch.tensor(np.cumsum(test_input.numpy(), axis=1))

        engine.add(test_f1, 'test1')
        self.assertRaisesRegex(KeyError, ".*exists.*",
                               engine.add, TestFactor(), 'test1')

        engine.add(TestFactor2(), 'test2')
        df = engine.run('2019-01-11', '2019-01-15')
        self.assertEqual(test_f1._cache_hit, 1)
        assert_array_equal(df['test1'].values, [0, 3, 1, 4, 2, 5])
        assert_array_equal(df['test2'].values, [0, 3, 1, 7, 3, 12])
