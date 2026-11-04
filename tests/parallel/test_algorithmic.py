import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.testing import assert_almost_equal, assert_array_equal

from spectre import parallel

nan = np.nan


def tensor(data, device='cpu', dtype=torch.float32):
    return torch.tensor(data, dtype=dtype, device=device)


def to_np(t):
    return t.cpu().numpy()


# ------------------------------------ group by ------------------------------------


def test_groupby_split_and_revert():
    x = tensor([1, 2, 10, 3, 11, 20, 4, 21, 5, 12, 13, 14, 15])
    keys = torch.tensor([1, 1, 2, 1, 2, 3, 1, 3, 1, 2, 2, 2, 2])

    groupby = parallel.ParallelGroupBy(keys)
    groups = groupby.split(x)
    assert_array_equal(groups[0].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, nan])
    assert_array_equal(groups[1].tolist(), [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert_array_equal(groups[2].tolist(), [20.0, 21.0, nan, nan, nan, nan])
    assert_array_equal(
        groupby.padding_mask.tolist(), [[False] * 5 + [True], [False] * 6, [False] * 2 + [True] * 4]
    )
    assert_array_equal(groupby.revert(groups).tolist(), x.tolist())


def test_groupby_bool_padding_is_false():
    keys = torch.tensor([0, 0, 1])
    groups = parallel.ParallelGroupBy(keys).split(torch.tensor([True, True, True]))
    assert groups.tolist() == [[True, True], [True, False]]


def test_groupby_rejects_int():
    groupby = parallel.ParallelGroupBy(torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match='cannot be any type of int'):
        groupby.split(torch.tensor([1, 2, 3]))


def test_groupby_revert_shape_mismatch():
    groupby = parallel.ParallelGroupBy(torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match='must same as input'):
        groupby.revert(torch.zeros(3, 3), 'test')
    with pytest.raises(ValueError, match='multiple return values'):
        groupby.revert(torch.zeros(2, 2, 3), 'test')


# ------------------------------------ rolling ------------------------------------

PRICES = [[164.0000, 163.7100, 158.6100, 145.230], [104.6100, 104.4200, 101.3000, 102.280]]


def test_rolling_sum():
    x = tensor(PRICES)
    assert str(parallel.Rolling(x, 3)).startswith(
        'spectre.parallel.Rolling object contains:\ntensor('
    )

    s = parallel.Rolling(x, 3).sum()
    expected = [
        [nan, nan, 164 + 163.71 + 158.61, 163.71 + 158.61 + 145.23],
        [nan, nan, 104.61 + 104.42 + 101.3, 104.42 + 101.3 + 102.28],
    ]
    assert_almost_equal(s.numpy(), expected, decimal=4)


def test_rolling_adjustment():
    x = tensor(PRICES)
    # adjustment multipliers, window values are adjusted by `multiplier / last multiplier`
    adj = tensor([[0.25, 0.25, 0.5, 1], [0.6, 0.75, 0.75, 1]])
    s = parallel.Rolling(x, 3, adj).sum()
    expected = [
        [nan, nan, 164.0000 / 2 + 163.7100 / 2 + 158.6100, 163.7100 / 4 + 158.6100 / 2 + 145.230],
        [
            nan,
            nan,
            104.6100 * (0.6 / 0.75) + 104.4200 + 101.3000,
            104.4200 * 0.75 + 101.3000 * 0.75 + 102.280,
        ],
    ]
    assert_almost_equal(s.numpy(), expected, decimal=4)


def test_rolling_split_large_data(monkeypatch):
    """Rolling splits large data into chunks by memory usage, results should be the same."""
    x = torch.randn(4, 50, dtype=torch.float64)
    x[1, 10:15] = nan
    expected = parallel.Rolling(x, 5).nanmean()

    # pretend memory usage is huge, so every column becomes a chunk
    monkeypatch.setattr(parallel.Rolling, '_split_multi', 1e12)
    rolling = parallel.Rolling(x, 5)
    assert len(rolling.split) == 50
    assert_almost_equal(rolling.nanmean().numpy(), expected.numpy())
    assert_almost_equal(
        rolling.nanmean().numpy()[:, 4:],
        pd.DataFrame(x.numpy().T).rolling(5, min_periods=1).mean().values.T[:, 4:],
    )


def test_rolling_fill():
    x = torch.tensor([[True, False, True]])
    # nan is True for bool tensor
    assert parallel.Rolling(x, 3).values[0, 0].tolist() == [True, True, True]
    assert parallel.Rolling(x, 3, fill=False).values[0, 0].tolist() == [False, False, True]
    assert parallel.Rolling(x, 3, fill=False).values[0, 2].tolist() == [True, False, True]


def test_rolling_aggregations():
    x = tensor([[1, 2, nan, 4, 8, 3]], dtype=torch.float64)
    r = parallel.Rolling(x, 3)
    assert_almost_equal(r.first().numpy(), [[nan, nan, 1, 2, nan, 4]])
    assert_almost_equal(r.last().numpy(), x.numpy())
    assert_almost_equal(r.last_nonnan().numpy(), [[1, 2, 2, 4, 8, 3]])
    assert_almost_equal(r.nansum().numpy(), [[1, 3, 3, 6, 12, 15]])
    assert_almost_equal(r.nanprod().numpy(), [[1, 2, 2, 8, 32, 96]])
    assert_almost_equal(r.nanmean().numpy(), [[1, 1.5, 1.5, 3, 6, 5]])
    assert_almost_equal(r.nanmax().numpy(), [[1, 2, 2, 4, 8, 8]])
    assert_almost_equal(r.nanmin().numpy(), [[1, 1, 1, 2, 4, 3]])
    assert_almost_equal(
        r.nanstd().numpy()[:, 3:], [[np.std([2, 4]), np.std([4, 8]), np.std([4, 8, 3])]]
    )
    assert_almost_equal(r.nanstd(ddof=1).numpy()[:, 5:], [[np.std([4, 8, 3], ddof=1)]])


# ------------------------------------ nan functions ------------------------------------

DATA_2D = [[1, 2, 1], [4, nan, 2], [7, 8, 1]]
DATA_3D = [[[nan, 1, 2], [1, 2, 1]], [[nan, 4, nan], [4, nan, 2]], [[nan, 7, 8], [7, 8, 1]]]


@pytest.mark.parametrize('ddof', [0, 1])
def test_nanmean_nanstd(device, ddof):
    data = tensor(DATA_2D, device)
    assert_almost_equal(to_np(parallel.nanmean(data)), np.nanmean(DATA_2D, axis=1), decimal=6)
    assert_almost_equal(
        to_np(parallel.nanstd(data, ddof=ddof)), np.nanstd(DATA_2D, axis=1, ddof=ddof), decimal=6
    )
    assert_almost_equal(
        to_np(parallel.nanvar(data, ddof=ddof)), np.nanvar(DATA_2D, axis=1, ddof=ddof), decimal=6
    )
    assert_almost_equal(to_np(parallel.nansum(data)), np.nansum(DATA_2D, axis=1), decimal=6)


def test_nanmean_nanstd_3d(device):
    data = tensor(DATA_3D, device)
    assert_almost_equal(
        to_np(parallel.nanmean(data, dim=2)), np.nanmean(DATA_3D, axis=2), decimal=6
    )
    assert_almost_equal(to_np(parallel.nanstd(data, dim=2)), np.nanstd(DATA_3D, axis=2), decimal=6)


def test_nanmax_nanmin(device):
    data = [[1, 2, -14, nan, 2], [99999, 8, 1, nan, 2]]
    assert_almost_equal(to_np(parallel.nanmax(tensor(data, device))), np.nanmax(data, axis=1))
    assert_almost_equal(to_np(parallel.nanmin(tensor(data, device))), np.nanmin(data, axis=1))


def test_nanlast(device):
    data = [[1, 2, nan], [4, nan, 2], [7, 8, 1]]
    assert_almost_equal(to_np(parallel.nanlast(tensor(data, device))), [2.0, 2.0, 1.0])
    assert_almost_equal(to_np(parallel.nanlast(tensor([data], device), dim=2)), [[2.0, 2.0, 1.0]])
    # offset: ignore the first n data
    assert_almost_equal(to_np(parallel.nanlast(tensor([[1, nan, nan]], device), offset=1)), [nan])

    data = [1, 2, nan, 4, nan, 2, 7, 8, 1]
    assert_almost_equal(to_np(parallel.nanlast(tensor(data, device), dim=0)), 1.0)


def test_masked_first(device):
    data = tensor([1, 2, nan], device)
    mask = torch.tensor([False, True, True], device=device)
    assert_almost_equal(to_np(parallel.masked_first(data, mask, dim=0)), 2.0)

    mask = torch.tensor([False, False, True], device=device)
    assert_almost_equal(to_np(parallel.masked_first(data, mask, dim=0)), nan)

    mask = torch.tensor([False, False, False], device=device)
    assert_almost_equal(to_np(parallel.masked_first(data, mask, dim=0)), nan)


def test_pad_2d(device):
    x = tensor([[nan, 1, 1, nan, 1, nan, nan, 0, nan, 0, nan, nan, 0, nan, -1, nan, -1, nan, nan,
                 nan, 1],
                [nan, 1, 0, nan, 1, nan, nan, 1, nan, -1, nan, -1, 0, nan, -1, nan, -1, nan, nan,
                 nan, 1]], device)  # fmt: skip
    expected = pd.DataFrame(to_np(x).T).ffill().values.T
    assert_almost_equal(to_np(parallel.pad_2d(x)), expected)


def test_pad_2d_inf():
    x = tensor([[1, np.inf, nan, -np.inf, 2]])
    assert_almost_equal(parallel.pad_2d(x, including_inf=True), [[1, 1, 1, 1, 2]])
    assert_almost_equal(
        parallel.pad_2d(x, including_inf=True, including_nan=False), [[1, 1, nan, nan, 2]]
    )


# ------------------------------------ rank ------------------------------------


def test_rankdata_ties(device):
    x = tensor([[2, 3, 6, 8, 3, 6, 6], [2, 3, 6, nan, 3, 2, 9]], device)
    result = to_np(parallel.rankdata(x))
    assert_array_equal(result[0], scipy.stats.rankdata(to_np(x[0])))
    assert_array_equal(result[1], pd.Series(to_np(x[1])).rank(method='average'))


@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('method, pd_method', [('average', 'average'), ('dense', 'dense')])
def test_rankdata_methods(device, ascending, method, pd_method):
    x = tensor([[2, 3, 6, 8, 3, 6, 6], [2, 3, 6, nan, 3, nan, 9]], device)
    result = to_np(parallel.rankdata(x, ascending=ascending, method=method))
    for i in range(2):
        expected = pd.Series(to_np(x[i])).rank(method=pd_method, ascending=ascending)
        assert_array_equal(result[i], expected, err_msg=f'row {i}')


@pytest.mark.parametrize('ascending', [True, False])
def test_rankdata_ordinal(device, ascending):
    x = tensor([[2, 3, 6, 8, 1], [2, nan, 6, 7, 3]], device)
    result = to_np(parallel.rankdata(x, ascending=ascending, method='ordinal'))
    for i in range(2):
        expected = pd.Series(to_np(x[i])).rank(method='first', ascending=ascending)
        assert_array_equal(result[i], expected)


def test_rankdata_normalize():
    x = tensor([[2, 3, nan, 1]])
    # normalized by the length of data (nan included)
    assert_almost_equal(parallel.rankdata(x, normalize=True), [[2 / 4, 3 / 4, nan, 1 / 4]])


# ------------------------------------ statistics ------------------------------------

X = [[1., 2, 3, 4, 5], [10, 12, 13, 14, 16], [2, 2, 2, 2, 2, ]]  # fmt: skip
Y = [[-1., 2, 3, 4, -5], [11, 12, -13, 14, 15], [2, 2, 2, 2, 2, ]]  # fmt: skip


def test_covariance():
    result = parallel.covariance(tensor(X), tensor(Y), ddof=1)
    expected = [np.cov(x, y, ddof=1)[0, 1] for x, y in zip(X, Y)]
    assert_almost_equal(result, expected, decimal=5)


def test_linear_regression_1d():
    coef, intercept = parallel.linear_regression_1d(tensor(X), tensor(Y))
    for i in range(2):
        expected_coef, expected_intercept = np.polyfit(X[i], Y[i], 1)
        assert coef[i].item() == pytest.approx(expected_coef, abs=1e-5)
        assert intercept[i].item() == pytest.approx(expected_intercept, abs=1e-4)
    # constant x, coef is 0
    assert coef[2].item() == 0
    assert intercept[2].item() == 2


def test_linear_regression_1d_single_row():
    # squeeze bug
    coef, intercept = parallel.linear_regression_1d(tensor(X[:1]), tensor(Y[:1]))
    expected_coef, expected_intercept = np.polyfit(X[0], Y[0], 1)
    assert coef.shape == (1,)
    assert coef[0].item() == pytest.approx(expected_coef, abs=1e-6)
    assert intercept[0].item() == pytest.approx(expected_intercept, abs=1e-6)


def test_pearsonr():
    result = parallel.pearsonr(tensor(X), tensor(Y))
    for i in range(2):
        expected = scipy.stats.pearsonr(X[i], Y[i])[0]
        assert result[i].item() == pytest.approx(expected, abs=1e-6)
    # constant input
    assert np.isnan(result[2].item())


def test_spearman():
    rank_x = parallel.rankdata(tensor(X))
    rank_y = parallel.rankdata(tensor(Y))
    result = parallel.spearman(rank_x, rank_y)
    for i in range(2):
        expected = scipy.stats.spearmanr(X[i], Y[i])[0]
        assert result[i].item() == pytest.approx(expected, abs=1e-6)
    # constant input, all ranks are the same
    assert result[2].item() == 1


QUANTILE_X = [[1, 2, nan, 3, 4, 5, 6], [3, 4, 5, 1.01, nan, 1.02, 1.03]]


def test_quantile():
    result = parallel.quantile(tensor(QUANTILE_X), 5, dim=1)
    for i in range(2):
        assert_array_equal(result[i], pd.qcut(QUANTILE_X[i], 5, labels=False))


def test_quantile_3d():
    x = [
        [[1, 2, nan, 3, 4, 5, 6], [3, 4, 5, 1.01, nan, 1.02, 1.03]],
        [[1, 2, 2.1, 3, 4, 5, 6], [3, 4, 5, nan, nan, 1.02, 1.03]],
    ]
    result = parallel.quantile(tensor(x), 5, dim=2)
    for i in range(2):
        for j in range(2):
            assert_array_equal(result[i, j], pd.qcut(x[i][j], 5, labels=False))


def test_quantile_single_column():
    result = parallel.quantile(tensor([[1.0], [2.0]]), 5, dim=1)
    assert_array_equal(result, [[0], [0]])


def test_masked_kth_value_median():
    x = tensor([[1, 2, nan, 3, 4, 5, 6], [3, 4, 5, 1.01, nan, 1.02, 1.03]])
    mask = torch.tensor(
        [[True, False, True, True, True, True, True], [True, True, True, True, True, True, True]]
    )
    expected = np.nanmedian(torch.masked_fill(x, ~mask, nan).numpy(), axis=1)
    median, _ = parallel.masked_kth_value_1d(x, mask, [0.5], dim=1)
    assert_almost_equal(median[0].squeeze(-1), expected, decimal=6)


def test_clamp_1d():
    data = tensor([[1, 5, 10], [-5, 0, 5]])
    result = parallel.clamp_1d_(data, tensor([[2], [-1]]), tensor([[8], [1]]))
    assert_array_equal(result, [[2, 5, 8], [-1, 0, 1]])
