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
        self.assertEqual(5, c.get_total_backwards_())
        m = spectre.factors.CustomFactor(win=10)
        c.set_mask(m)
        self.assertEqual(9, c.get_total_backwards_())

        a1 = spectre.factors.CustomFactor(win=10)
        a2 = spectre.factors.CustomFactor(win=5)
        b1 = spectre.factors.CustomFactor(win=20, inputs=(a1, a2))
        b2 = spectre.factors.CustomFactor(win=100, inputs=(a2,))
        c1 = spectre.factors.CustomFactor(win=100, inputs=(b1,))
        self.assertEqual(9, a1.get_total_backwards_())
        self.assertEqual(4, a2.get_total_backwards_())
        self.assertEqual(28, b1.get_total_backwards_())
        self.assertEqual(103, b2.get_total_backwards_())
        self.assertEqual(127, c1.get_total_backwards_())

        # test inheritance
        loader = spectre.data.CsvDirLoader(
            data_dir + '/daily/', ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'),
            prices_index='date', parse_dates=True,
        )
        engine = spectre.factors.FactorEngine(loader)

        class TestFactor(spectre.factors.CustomFactor):
            inputs = [spectre.factors.OHLCV.open]

            def compute(self, close):
                return torch.tensor(np.arange(close.nelement()).reshape(close.shape))

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = []

            def compute(self):
                return torch.tensor([1])

        engine.add(TestFactor2(), 'test2')
        self.assertRaisesRegex(ValueError, "The return data shape.*test2.*",
                               engine.run, '2019-01-11', '2019-01-15', False)
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

        for f in engine._factors.values():
            f.pre_compute_(engine, '2019-01-11', '2019-01-15')
        self.assertEqual(2, test_f1._ref_count)
        for f in engine._factors.values():
            f._ref_count = 0

        df = engine.run('2019-01-11', '2019-01-15', delay_factor=False)
        self.assertEqual(0, test_f1._ref_count)
        assert_array_equal([0, 3, 1, 4, 2, 5], df['test1'].values)
        assert_array_equal([0, 3, 1, 7, 3, 12], df['test2'].values)
