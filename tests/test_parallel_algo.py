import unittest
import torch
import spectre
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
import pandas as pd


class TestParallelAlgorithm(unittest.TestCase):
    def test_groupby(self):
        test_x = torch.tensor([1, 2, 10, 3, 11, 20, 4, 21, 5, 12, 13, 14, 15], dtype=torch.float32)
        test_k = torch.tensor([1, 1, 2, 1, 2, 3, 1, 3, 1, 2, 2, 2, 2])

        groupby = spectre.parallel.ParallelGroupBy(test_k)
        groups = groupby.split(test_x)
        assert_array_equal([1., 2., 3., 4., 5., np.nan], groups[0].tolist())
        assert_array_equal([10., 11., 12., 13., 14., 15.], groups[1].tolist())
        assert_array_equal([20., 21., np.nan, np.nan, np.nan, np.nan], groups[2].tolist())

        revert_x = groupby.revert(groups)
        assert_array_equal(revert_x.tolist(), test_x.tolist())

    def test_rolling(self):
        x = torch.tensor([[164.0000, 163.7100, 158.6100, 145.230],
                          [104.6100, 104.4200, 101.3000, 102.280]])
        expected = torch.tensor(
            [[np.nan, np.nan, 486.3200, 467.5500],
             [np.nan, np.nan, 310.3300, 308.0000]])

        self.assertRegex(str(spectre.parallel.Rolling(x, 3)),
                         "spectre.parallel.Rolling object(.|\n)*tensor(.|\n)*")
        s = spectre.parallel.Rolling(x, 3).sum()
        assert_almost_equal(expected.numpy(), s.numpy(), decimal=4)

        # test adjustment
        y = torch.tensor([[0.25, 0.25, 0.5, 1],
                          [0.6, 0.75, 0.75, 1]])
        s = spectre.parallel.Rolling(x, 3, y).sum()
        expected = torch.tensor([
            [
                np.nan, np.nan,
                sum([164.0000 / 2, 163.7100 / 2, 158.6100]),
                sum([163.7100 / 4, 158.6100 / 2, 145.230]),
            ],
            [
                np.nan, np.nan,
                sum([104.6100 * (0.6 / 0.75), 104.4200, 101.3000]),
                sum([104.4200 * 0.75, 101.3000 * 0.75, 102.280]),
            ]
        ])
        assert_almost_equal(expected.numpy(), s.numpy(), decimal=4)

        x = torch.zeros([1024, 102400], dtype=torch.float64)
        spectre.parallel.Rolling(x, 252).sum()

    def test_nan(self):
        # dim=1
        data = [[1, 2, 1], [4, np.nan, 2], [7, 8, 1]]
        result = spectre.parallel.nanmean(torch.tensor(data, dtype=torch.float))
        expected = np.nanmean(data, axis=1)
        assert_almost_equal(expected, result, decimal=6)

        result = spectre.parallel.nanstd(torch.tensor(data, dtype=torch.float))
        expected = np.nanstd(data, axis=1)
        assert_almost_equal(expected, result, decimal=6)

        result = spectre.parallel.nanstd(torch.tensor(data, dtype=torch.float), ddof=1)
        expected = np.nanstd(data, axis=1, ddof=1)
        assert_almost_equal(expected, result, decimal=6)

        # dim=2
        data = [[[np.nan, 1, 2], [1, 2, 1]], [[np.nan, 4, np.nan], [4, np.nan, 2]],
                [[np.nan, 7, 8], [7, 8, 1]]]
        result = spectre.parallel.nanmean(torch.tensor(data, dtype=torch.float), dim=2)
        expected = np.nanmean(data, axis=2)
        assert_almost_equal(expected, result, decimal=6)

        result = spectre.parallel.nanstd(torch.tensor(data, dtype=torch.float), dim=2)
        expected = np.nanstd(data, axis=2)
        assert_almost_equal(expected, result, decimal=6)

        # last
        data = [[1, 2, np.nan], [4, np.nan, 2], [7, 8, 1]]
        result = spectre.parallel.nanlast(torch.tensor(data, dtype=torch.float).cuda())
        expected = [2., 2., 1.]
        assert_almost_equal(expected, result.cpu(), decimal=6)

        data = [[[1, 2, np.nan], [4, np.nan, 2], [7, 8, 1]]]
        result = spectre.parallel.nanlast(torch.tensor(data, dtype=torch.float).cuda(), dim=2)
        expected = [[2., 2., 1.]]
        assert_almost_equal(expected, result.cpu(), decimal=6)

        data = [1, 2, np.nan, 4, np.nan, 2, 7, 8, 1]
        result = spectre.parallel.nanlast(torch.tensor(data, dtype=torch.float), dim=0)
        expected = [1.]
        assert_almost_equal(expected, result, decimal=6)

        data = [1, 2, np.nan]
        mask = [False, True, True]
        result = spectre.parallel.masked_first(
            torch.tensor(data, dtype=torch.float), torch.tensor(mask, dtype=torch.bool), dim=0)
        expected = [2.]
        assert_almost_equal(expected, result, decimal=6)

        mask = [False, False, True]
        result = spectre.parallel.masked_first(
            torch.tensor(data, dtype=torch.float), torch.tensor(mask, dtype=torch.bool), dim=0)
        expected = [np.nan]
        assert_almost_equal(expected, result, decimal=6)

        # nanmin/max
        data = [[1, 2, -14, np.nan, 2], [99999, 8, 1, np.nan, 2]]
        result = spectre.parallel.nanmax(torch.tensor(data, dtype=torch.float))
        expected = np.nanmax(data, axis=1)
        assert_almost_equal(expected, result, decimal=6)

        result = spectre.parallel.nanmin(torch.tensor(data, dtype=torch.float))
        expected = np.nanmin(data, axis=1)
        assert_almost_equal(expected, result, decimal=6)

    def test_stat(self):
        x = torch.tensor([[1., 2, 3, 4, 5], [10, 12, 13, 14, 16], [2, 2, 2, 2, 2, ]])
        y = torch.tensor([[-1., 2, 3, 4, -5], [11, 12, -13, 14, 15], [2, 2, 2, 2, 2, ]])
        result = spectre.parallel.covariance(x, y, ddof=1)
        expected = np.cov(x, y, ddof=1)
        expected = expected[:x.shape[0], x.shape[0]:]
        assert_almost_equal(np.diag(expected), result, decimal=6)

        coef, intcp = spectre.parallel.linear_regression_1d(x, y)
        from sklearn.linear_model import LinearRegression
        for i in range(3):
            reg = LinearRegression().fit(x[i, :, None], y[i, :, None])
            assert_almost_equal(reg.coef_, coef[i], decimal=6)

        # test pearsonr
        result = spectre.parallel.pearsonr(x, y)
        from scipy import stats
        for i in range(3):
            expected, _ = stats.pearsonr(x[i], y[i])
            assert_almost_equal(expected, result[i], decimal=6)

        # test quantile
        x = torch.tensor([[1, 2, np.nan, 3, 4, 5, 6], [3, 4, 5, 1.01, np.nan, 1.02, 1.03]])
        result = spectre.parallel.quantile(x, 5, dim=1)
        expected = pd.qcut(x[0], 5, labels=False)
        assert_array_equal(expected, result[0])
        expected = pd.qcut(x[1], 5, labels=False)
        assert_array_equal(expected, result[1])

        x = torch.tensor([[[1, 2, np.nan, 3, 4, 5, 6], [3, 4, 5, 1.01, np.nan, 1.02, 1.03]],
                          [[1, 2, 2.1, 3, 4, 5, 6], [3, 4, 5, np.nan, np.nan, 1.02, 1.03]]
                          ])
        result = spectre.parallel.quantile(x, 5, dim=2)
        expected = pd.qcut(x[0, 0], 5, labels=False)
        assert_array_equal(expected, result[0, 0])
        expected = pd.qcut(x[0, 1], 5, labels=False)
        assert_array_equal(expected, result[0, 1])
        expected = pd.qcut(x[1, 0], 5, labels=False)
        assert_array_equal(expected, result[1, 0])
        expected = pd.qcut(x[1, 1], 5, labels=False)
        assert_array_equal(expected, result[1, 1])

    def test_pad2d(self):
        x = torch.tensor([[np.nan, 1, 1, np.nan, 1, np.nan, np.nan, 0, np.nan, 0, np.nan, np.nan, 0,
                           np.nan, -1, np.nan, - 1, np.nan, np.nan, np.nan, 1],
                          [np.nan, 1, 0, np.nan, 1, np.nan, np.nan, 1, np.nan, -1, np.nan, -1, 0,
                           np.nan, -1, np.nan, - 1, np.nan, np.nan, np.nan, 1]])
        result = spectre.parallel.pad_2d(x)

        expected = [pd.Series(x[0]).fillna(method='ffill'), pd.Series(x[1]).fillna(method='ffill')]
        assert_almost_equal(expected, result)
