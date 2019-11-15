import unittest
import torch
import spectre
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np


class TestParallelAlgorithm(unittest.TestCase):
    def test_groupby(self):
        test_x = torch.tensor([1, 2, 10, 3, 11, 20, 4, 21, 5, 12, 13, 14, 15], dtype=torch.float32)
        test_k = torch.tensor([1, 1,  2, 1,  2,  3, 1,  3, 1,  2,  2,  2,  2])

        groupby = spectre.parallel.ParallelGroupBy(test_k)
        groups = groupby.split(test_x)
        self.assertEqual(groups[0].tolist(), [1.,   2.,  3.,  4.,  5.,  0.])
        self.assertEqual(groups[1].tolist(), [10., 11., 12., 13., 14., 15.])
        self.assertEqual(groups[2].tolist(), [20., 21.,  0.,  0.,  0.,  0.])

        revert_x = groupby.revert(groups)
        assert_array_equal(revert_x.tolist(), test_x.tolist())

    def test_rolling(self):
        x = torch.tensor([[164.0000, 163.7100, 158.6100, 145.230],
                          [104.6100, 104.4200, 101.3000, 102.280]])
        expected = torch.tensor(
            [[np.nan,     np.nan, 486.3200, 467.5500],
             [np.nan,     np.nan, 310.3300, 308.0000]])

        s = spectre.parallel.Rolling(x, 3).sum()
        assert_almost_equal(s.numpy(), expected.numpy(), decimal=4)

    def test_nan(self):
        data = [[1, 2, 1], [4, np.nan, 2], [7, 8, 1]]
        result = spectre.parallel.nanmean(torch.tensor(data, dtype=torch.float))
        expected = np.nanmean(data, axis=1)
        assert_almost_equal(result, expected, decimal=6)

        result = spectre.parallel.nanstd(torch.tensor(data, dtype=torch.float))
        expected = np.nanstd(data, axis=1)
        assert_almost_equal(result, expected, decimal=6)
