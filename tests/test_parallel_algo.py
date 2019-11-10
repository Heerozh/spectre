import unittest
import torch
import spectre
from  numpy.testing import assert_array_equal


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

