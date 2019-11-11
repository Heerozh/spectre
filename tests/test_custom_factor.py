import unittest
import spectre
import numpy as np
from numpy.testing import assert_array_equal


class TestCustomFactorLib(unittest.TestCase):

    def test_custom_factor(self):
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
                return np.arange(np.prod(close.shape)).reshape(close.shape)

        class TestFactor2(spectre.factors.CustomFactor):
            inputs = [TestFactor]

            def compute(self, test_input):
                return np.cumsum(test_input)

        engine.add(TestFactor2(), 'test2')
        self.assertRaisesRegex(ValueError, "Length.*",
                               engine.run, '2019-01-11', '2019-01-15')
        engine.remove_all()
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
        assert_array_equal(df['test1'].values, [0, 1, 2, 3, 4, 5])
        assert_array_equal(df['test2'].values, [0, 1, 3, 6, 10, 15])

    def test_level_tree(self):
        class Lv4a(spectre.factors.CustomFactor):
            inputs = []
            pass
        class Lv4b(spectre.factors.CustomFactor):
            inputs = []
            pass

        class Lv3a(spectre.factors.CustomFactor):
            inputs = [Lv4a()]
        class Lv3b(spectre.factors.CustomFactor):
            inputs = [Lv4b()]
        class Lv3c(spectre.factors.CustomFactor):
            inputs = [Lv4b()]

        class Lv2a(spectre.factors.CustomFactor):
            inputs = [Lv3a()]
        class Lv2b(spectre.factors.CustomFactor):
            inputs = [Lv3a()]
        class Lv2c(spectre.factors.CustomFactor):
            inputs = [Lv3b()]

        class Lv1a(spectre.factors.CustomFactor):
            inputs = [Lv2a(), Lv2b(), Lv2c()]

        level_tree_1 = Lv1a()._build_level_tree()
        level_tree_1 = {k: [type(f).__name__ for f in v] for k, v in level_tree_1.items()}
        self.assertEqual(level_tree_1, {0: ['Lv1a'], 1: ['Lv2a', 'Lv2b', 'Lv2c'],
                                        2: ['Lv3a', 'Lv3a', 'Lv3b'], 3: ['Lv4a', 'Lv4a', 'Lv4b']})

        level_tree_2 = Lv3c()._build_level_tree()
        level_tree_2 = {k: [type(f).__name__ for f in v] for k, v in level_tree_2.items()}
        self.assertEqual(level_tree_2, {0: ['Lv3c'], 1: ['Lv4b']})
