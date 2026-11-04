import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from spectre import parallel

nan = np.nan


def tensor(data, device="cpu", dtype=torch.float32):
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
    with pytest.raises(ValueError, match="cannot be any type of int"):
        groupby.split(torch.tensor([1, 2, 3]))


def test_groupby_revert_shape_mismatch():
    groupby = parallel.ParallelGroupBy(torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="must same as input"):
        groupby.revert(torch.zeros(3, 3), "test")
    with pytest.raises(ValueError, match="multiple return values"):
        groupby.revert(torch.zeros(2, 2, 3), "test")


# ------------------------------------ rolling ------------------------------------

PRICES = [[164.0000, 163.7100, 158.6100, 145.230], [104.6100, 104.4200, 101.3000, 102.280]]


def test_rolling_sum():
    x = tensor(PRICES)
    assert str(parallel.Rolling(x, 3)).startswith(
        "spectre.parallel.Rolling object contains:\ntensor("
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
    monkeypatch.setattr(parallel.Rolling, "_split_multi", 1e12)
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


@pytest.mark.parametrize("ddof", [0, 1])
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nanmax_nanmin_empty_group(device, dtype):
    data = tensor([[nan, nan, nan], [1, 2, 4]], device, dtype)
    assert_array_equal(to_np(parallel.nanmax(data)), [nan, 4])
    assert_array_equal(to_np(parallel.nanmin(data)), [nan, 1])

    # a real inf is a value, not an empty group
    data = tensor([[-np.inf, nan], [np.inf, nan]], device, dtype)
    assert_array_equal(to_np(parallel.nanmax(data)), [-np.inf, np.inf])
    assert_array_equal(to_np(parallel.nanmin(data)), [-np.inf, np.inf])


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nanvar_covariance_empty_group(device, dtype, ddof):
    data = tensor([[nan, nan, nan], [1, 2, 4]], device, dtype)
    var = np.var([1, 2, 4], ddof=ddof)
    assert_almost_equal(to_np(parallel.nanvar(data, ddof=ddof)), [nan, var], decimal=5)
    assert_almost_equal(to_np(parallel.nanstd(data, ddof=ddof)), [nan, var**0.5], decimal=5)
    assert_almost_equal(to_np(parallel.covariance(data, data, ddof=ddof)), [nan, var], decimal=5)


def test_covariance_without_common_samples(device):
    x = tensor([[1, 2, nan, nan]], device)
    y = tensor([[nan, nan, 3, 4]], device)
    for ddof in (0, 1):
        assert_array_equal(to_np(parallel.covariance(x, y, ddof=ddof)), [nan])
        assert_array_equal(to_np(parallel.pearsonr(x, y, ddof=ddof)), [nan])


@pytest.mark.parametrize("ddof", [1, 2])
def test_too_few_samples_is_nan(device, ddof):
    data = tensor([[nan, 5, nan]], device)
    assert_array_equal(to_np(parallel.nanvar(data, ddof=ddof)), [nan])
    assert_array_equal(to_np(parallel.nanstd(data, ddof=ddof)), [nan])
    assert_array_equal(to_np(parallel.covariance(data, data, ddof=ddof)), [nan])


def test_one_sample_with_ddof_0_has_no_variance(device):
    data = tensor([[nan, 5, nan]], device)
    assert_array_equal(to_np(parallel.nanvar(data)), [0])
    assert_array_equal(to_np(parallel.nanstd(data)), [0])
    assert_array_equal(to_np(parallel.covariance(data, data)), [0])


def test_rolling_empty_window(device):
    x = tensor([[nan, nan, nan, 1, 2, nan, nan, nan, 4]], device, torch.float64)
    r = parallel.Rolling(x, 3)
    expected = pd.Series(to_np(x)[0]).rolling(3, min_periods=1)
    assert_almost_equal(to_np(r.nanmax())[0], expected.max())
    assert_almost_equal(to_np(r.nanmin())[0], expected.min())
    assert_almost_equal(to_np(r.nanstd())[0], expected.std(ddof=0))
    assert_almost_equal(to_np(r.nanstd(ddof=1))[0], expected.std(ddof=1))
    assert_almost_equal(to_np(r.nanvar())[0], expected.var(ddof=0))


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
    assert_array_equal(result[1], pd.Series(to_np(x[1])).rank(method="average"))


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("method, pd_method", [("average", "average"), ("dense", "dense")])
def test_rankdata_methods(device, ascending, method, pd_method):
    x = tensor([[2, 3, 6, 8, 3, 6, 6], [2, 3, 6, nan, 3, nan, 9]], device)
    result = to_np(parallel.rankdata(x, ascending=ascending, method=method))
    for i in range(2):
        expected = pd.Series(to_np(x[i])).rank(method=pd_method, ascending=ascending)
        assert_array_equal(result[i], expected, err_msg=f"row {i}")


@pytest.mark.parametrize("ascending", [True, False])
def test_rankdata_ordinal(device, ascending):
    x = tensor([[2, 3, 6, 8, 1], [2, nan, 6, 7, 3]], device)
    result = to_np(parallel.rankdata(x, ascending=ascending, method="ordinal"))
    for i in range(2):
        expected = pd.Series(to_np(x[i])).rank(method="first", ascending=ascending)
        assert_array_equal(result[i], expected)


def test_rankdata_normalize():
    x = tensor([[2, 3, nan, 1]])
    # normalized by the length of data (nan included)
    assert_almost_equal(parallel.rankdata(x, normalize=True), [[2 / 4, 3 / 4, nan, 1 / 4]])


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8, 40])
def test_rankdata_same_as_scipy(device, width, method, ascending):
    rng = np.random.default_rng(width)
    x = rng.integers(0, 3, (300, width)).astype(np.float32)
    x[:100] = x[:100, :1]  # all values of these rows are tied
    x[200:][rng.random((100, width)) < 0.3] = nan
    expected = scipy.stats.rankdata(x if ascending else -x, method, axis=1, nan_policy="omit")
    result = parallel.rankdata(tensor(x, device), ascending=ascending, method=method)
    assert_array_equal(to_np(result), expected)


@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
def test_rankdata_along_last_dim_same_as_scipy(device, method):
    """The layout of `Rolling.agg`, [asset, chunk, win]"""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, (4, 30, 6)).astype(np.float64)
    x[:, :10] = x[:, :10, :1]  # flat windows
    x[:, 20:][rng.random((4, 10, 6)) < 0.3] = nan
    expected = scipy.stats.rankdata(x, method, axis=2, nan_policy="omit")
    result = parallel.rankdata(tensor(x, device, torch.float64), dim=2, method=method)
    assert_array_equal(to_np(result), expected)


def test_rankdata_normalize_tied_row(device):
    x = tensor([[2, 2, 2, 2]], device)
    assert_array_equal(to_np(parallel.rankdata(x, normalize=True)), [[2.5 / 4] * 4])


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
@pytest.mark.parametrize("width", [3, 8, 40, 300])
def test_rankdata_nan_does_not_tie_with_inf(device, width, method, ascending):
    rng = np.random.default_rng(width)
    x = rng.integers(0, 4, (200, width)).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.inf
    x[rng.random(x.shape) < 0.2] = -np.inf
    x[rng.random(x.shape) < 0.3] = nan
    expected = scipy.stats.rankdata(x if ascending else -x, method, axis=1, nan_policy="omit")
    result = parallel.rankdata(tensor(x, device), ascending=ascending, method=method)
    assert_array_equal(to_np(result), expected)


@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
def test_rankdata_nan_does_not_tie_with_inf_along_last_dim(device, method):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, (4, 30, 40)).astype(np.float64)
    x[rng.random(x.shape) < 0.2] = -np.inf
    x[rng.random(x.shape) < 0.2] = np.inf
    x[rng.random(x.shape) < 0.3] = nan
    data = tensor(x, device, torch.float64)
    for ascending in (True, False):
        expected = scipy.stats.rankdata(x if ascending else -x, method, axis=2, nan_policy="omit")
        result = parallel.rankdata(data, dim=2, ascending=ascending, method=method)
        assert_array_equal(to_np(result), expected)


@pytest.mark.parametrize(
    "row, ascending, method, expected",
    [
        ([1, -np.inf, nan], False, "average", [1, 2, nan]),
        ([nan, -np.inf, 1], False, "average", [nan, 2, 1]),
        ([nan, -np.inf, 1], False, "ordinal", [nan, 2, 1]),
        ([nan, nan, -np.inf, 1], False, "ordinal", [nan, nan, 2, 1]),
        ([nan, -np.inf, 1, -np.inf], False, "average", [nan, 2.5, 1, 2.5]),
        ([nan, -np.inf, 1, -np.inf], False, "ordinal", [nan, 2, 1, 3]),
        ([1, np.inf, nan], True, "average", [1, 2, nan]),
        ([nan, np.inf, 1], True, "ordinal", [nan, 2, 1]),
        ([np.inf, nan, np.inf, 1], True, "average", [2.5, nan, 2.5, 1]),
        ([np.inf, nan, np.inf, 1], True, "ordinal", [2, nan, 3, 1]),
    ],
)
def test_rankdata_inf_next_to_nan(device, row, ascending, method, expected):
    result = parallel.rankdata(tensor([row], device), ascending=ascending, method=method)
    assert_array_equal(to_np(result)[0], expected)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
def test_rankdata_bool(device, method, ascending):
    x = np.random.default_rng(0).random((50, 7)) < 0.5
    values = x.astype(np.float64)
    expected = scipy.stats.rankdata(values if ascending else -values, method, axis=1)
    result = parallel.rankdata(torch.tensor(x, device=device), ascending=ascending, method=method)
    assert_array_equal(to_np(result), expected)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "x, y",
    [
        ([0, 1, 2, 3, 4], [nan, nan, 1, 2, 3]),
        ([nan, nan, 2, 3, 4], [0, 1, 2, 4, 3]),
        ([nan, nan, 2, 3, 4], [nan, nan, 1, 2, 5]),
        ([nan, 1, 2, 3, 4], [0, 1, 2, nan, 7]),
        ([1, 2, 3, 4, nan], [nan, 5, 1, 7, 9]),
    ],
)
def test_linear_regression_1d_uses_valid_pairs(device, dtype, x, y):
    valid = ~np.isnan(np.array(x) * np.array(y))
    expected = np.polyfit(np.array(x)[valid], np.array(y)[valid], 1)
    coef, intercept = parallel.linear_regression_1d(
        tensor([x], device, dtype), tensor([y], device, dtype)
    )
    assert coef.item() == pytest.approx(expected[0], abs=1e-5)
    assert intercept.item() == pytest.approx(expected[1], abs=1e-5)


@pytest.mark.parametrize("win", [5, 10, 20, 60])
def test_linear_regression_1d_constant_x_float32(device, win):
    """The mean of a constant rounds in float32, so its variance is not exactly 0."""
    rng = np.random.default_rng(win)
    constant = rng.uniform(0.5, 2000, (2000, 1)).astype(np.float32).repeat(win, axis=1)
    y = rng.normal(100, 2, (2000, win)).astype(np.float32)
    coef, intercept = parallel.linear_regression_1d(
        torch.tensor(constant, device=device), torch.tensor(y, device=device)
    )
    assert_array_equal(to_np(coef), np.zeros(2000))
    assert_almost_equal(to_np(intercept), y.mean(axis=1, dtype=np.float64), decimal=4)


def test_linear_regression_1d_x_constant_on_valid_pairs(device):
    x = tensor([[1, 2, 3, 3, 3]], device)
    y = tensor([[nan, nan, 1, 2, 4]], device)
    coef, intercept = parallel.linear_regression_1d(x, y)
    assert coef.item() == 0
    assert intercept.item() == pytest.approx(7 / 3, abs=1e-6)


@pytest.mark.parametrize(
    "x, y",
    [
        ([nan, nan, nan], [1, 2, 3]),
        ([1, 2, 3], [nan, nan, nan]),
        ([1, 2, nan], [nan, nan, 3]),
        # a slope needs two samples
        ([1, 2, 3], [nan, nan, 3]),
        ([nan, 2, nan], [1, 2, 3]),
    ],
)
def test_linear_regression_1d_too_few_pairs(device, x, y):
    coef, intercept = parallel.linear_regression_1d(tensor([x], device), tensor([y], device))
    assert_array_equal(to_np(coef), [nan])
    assert_array_equal(to_np(intercept), [nan])


def test_pearsonr():
    result = parallel.pearsonr(tensor(X), tensor(Y))
    for i in range(2):
        expected = scipy.stats.pearsonr(X[i], Y[i])[0]
        assert result[i].item() == pytest.approx(expected, abs=1e-6)
    # constant input
    assert np.isnan(result[2].item())


def exact_corr(x, y):
    """Correlation of the valid pairs, by exact arithmetic on the values of the floats."""
    pairs = [(Fraction(float(a)), Fraction(float(b))) for a, b in zip(x, y) if a * b == a * b]
    n = len(pairs)
    if n < 2:
        return nan
    x_bar, y_bar = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    cov = sum((a - x_bar) * (b - y_bar) for a, b in pairs)
    x_var = sum((a - x_bar) ** 2 for a, _ in pairs)
    y_var = sum((b - y_bar) ** 2 for _, b in pairs)
    if x_var == 0 or y_var == 0:
        return nan
    return math.copysign(math.sqrt(float(cov * cov / (x_var * y_var))), float(cov))


DTYPES = [torch.float32, torch.float64]


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("dtype", DTYPES)
def test_pearsonr_of_constant_is_nan(device, dtype, ddof):
    """The mean of a constant is rounded or not depending on its value, so try a lot of them."""
    rng = np.random.default_rng(0)
    constant = tensor(np.repeat(rng.uniform(1, 300, (1000, 1)), 30, axis=1), device, dtype)
    other = tensor(rng.normal(size=(1000, 30)), device, dtype)
    for x, y in ((constant, other), (other, constant), (constant, constant)):
        assert torch.isnan(parallel.pearsonr(x, y, ddof=ddof)).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_pearsonr_of_constant_on_the_common_samples(device, dtype):
    """x varies, but only on the samples that y doesn't have."""
    rng = np.random.default_rng(0)
    x = np.repeat(rng.uniform(1, 300, (1000, 1)), 30, axis=1)
    x[:, -2:] += rng.uniform(1, 3, (1000, 2))
    y = rng.normal(size=x.shape)
    y[:, -2:] = nan
    result = parallel.pearsonr(tensor(x, device, dtype), tensor(y, device, dtype), ddof=1)
    assert torch.isnan(result).all()


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_pearsonr_in_rolling_windows(device, dtype, tol):
    """The windows inside the flat part of x have no correlation, all the others have one."""
    win = 5
    rng = np.random.default_rng(0)
    x = rng.normal(100, 1, (100, 30))
    x[:, 10:20] = rng.uniform(1, 300, (100, 1))
    y = rng.normal(size=x.shape)
    y[1, 5] = nan
    x, y = tensor(x, device, dtype), tensor(y, device, dtype)

    def corr(a, b):
        return parallel.pearsonr(a, b, dim=2, ddof=1)

    result = to_np(parallel.Rolling(x, win).agg(corr, parallel.Rolling(y, win)))
    x, y = to_np(x), to_np(y)

    def window_corr(row, bar):
        start = max(0, bar - win + 1)
        return exact_corr(x[row, start : bar + 1], y[row, start : bar + 1])

    expected = [[window_corr(row, bar) for bar in range(x.shape[1])] for row in range(x.shape[0])]
    assert_allclose(result, expected, rtol=tol, atol=tol)
    assert np.isnan(result[:, 14:20]).all()
    assert not np.isnan(result[:, 20:]).any()


def nearly_constant(rng, shape, spread, np_type):
    """Rows of a level, plus a whole number of ulps up to `spread`."""
    level = rng.uniform(1, 300, (shape[0], 1)).astype(np_type)
    ulp = np.nextafter(level, np.inf, dtype=np_type) - level
    steps = rng.integers(0, spread + 1, shape)
    return level + (steps * ulp).astype(np_type), steps


@pytest.mark.parametrize("spread", [1, 2, 16, 256])
@pytest.mark.parametrize(
    "dtype, np_type, tol", [(torch.float32, np.float32, 1e-6), (torch.float64, np.float64, 1e-14)]
)
def test_pearsonr_of_nearly_constant_is_real(device, dtype, np_type, tol, spread):
    """x only varies by a few ulps, and a mean that is rounded by that much isn't variance."""
    rng = np.random.default_rng(0)
    shape = (50, 30)
    x, steps = nearly_constant(rng, shape, spread, np_type)
    follows_x = (steps + rng.normal(0, spread * 0.5 + 0.5, shape)).astype(np_type)
    also_nearly_constant = nearly_constant(rng, shape, spread, np_type)[0]
    for y in (follows_x, also_nearly_constant):
        result = parallel.pearsonr(tensor(x, device, dtype), tensor(y, device, dtype), ddof=1)
        expected = [exact_corr(a, b) for a, b in zip(x, y)]
        assert_allclose(to_np(result), expected, rtol=tol, atol=tol)
        assert not np.isnan(expected).any()


@pytest.mark.parametrize("ddof", [0, 1])
def test_pearsonr_of_one_pair_is_nan(device, ddof):
    x = tensor([[1, nan, 3]], device)
    y = tensor([[2, 5, nan]], device)
    assert_array_equal(to_np(parallel.pearsonr(x, y, ddof=ddof)), [nan])


@pytest.mark.parametrize("ddof", [0, 1])
def test_pearsonr_of_two_pairs(device, ddof):
    x = tensor([[1, 2, nan], [1, 2, nan]], device)
    y = tensor([[5, 3, 7], [3, 5, 7]], device)
    assert_allclose(to_np(parallel.pearsonr(x, y, ddof=ddof)), [-1, 1], rtol=1e-6)


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


def test_quantile_single_column_nan(device):
    assert_array_equal(to_np(parallel.quantile(tensor([[nan], [2.0]], device), 5)), [[nan], [0]])


def test_quantile_single_row_of_windows(device):
    """`Rolling.agg` can pass a chunk of one bar, as [asset, 1, win]."""
    x = [[[3, 1, 2, 6, 5, 4]], [[1, nan, 2, 4, 3, nan]]]
    result = to_np(parallel.quantile(tensor(x, device), 5, dim=2))
    for i in range(2):
        assert_array_equal(result[i, 0], pd.qcut(x[i][0], 5, labels=False))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bins", [2, 3, 4, 5, 7, 10, 20])
def test_quantile_every_row_size(device, dtype, bins):
    rng = np.random.default_rng(bins)
    width = 64
    rows = np.full((width - 1, width), nan)
    for size in range(2, width + 1):
        rows[size - 2, :size] = rng.permutation(size)
    result = to_np(parallel.quantile(tensor(rows, device, dtype), bins))
    for row, labels in zip(rows, result):
        assert_array_equal(labels, pd.qcut(row, bins, labels=False), err_msg=str(row))


def test_quantile_exact_cuts():
    # Cuts fall on 3, 6, 9, 12 and 15, and a value on a cut belongs to the lower bin. pd.qcut has
    # a rounding error of its own here, its cut at 15 is 14.999999999999998.
    expected = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    assert_array_equal(parallel.quantile(tensor([list(range(19))]), 6), [expected])


@pytest.mark.parametrize("bins", [3, 5])
def test_quantile_float32_values(device, bins):
    rng = np.random.default_rng(bins)
    # every cut is exact, or falls between float32 neighbours
    normal = rng.standard_normal((50, 301)).astype(np.float32)
    neighbours = 1 + rng.integers(0, 40, (50, 301)).astype(np.float32) * 2.0**-23
    for rows in (normal, neighbours):
        result = to_np(parallel.quantile(torch.tensor(rows, device=device), bins))
        for row, labels in zip(rows.astype(np.float64), result):
            assert_array_equal(labels, pd.qcut(row, bins, labels=False))


@pytest.mark.parametrize("lowest", [2e7, 1e9, 1e30])
def test_quantile_large_float32(device, lowest):
    row = [lowest * i for i in (1, 5, 2, 3, 4)]
    result = to_np(parallel.quantile(tensor([row], device), 5))
    assert_array_equal(result[0], pd.qcut(row, 5, labels=False))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "row",
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf],
        [-np.inf, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [-np.inf, 1, 2, 3, 4, 5, 6, 7, 8, np.inf],
        [-np.inf, -np.inf, 1, 2, 3, 4, 5, 6, np.inf, np.inf],
        [1, np.inf, 2, 3, nan, 4, 5, 6],
    ],
)
def test_quantile_inf(device, dtype, row):
    # pd.qcut can't take -inf, its lowest cut becomes NaN. Bins only depend on the order of the
    # values, so give it finite values in the same order.
    finite = np.array(row, dtype=np.float64)
    finite[np.isinf(finite)] = 1e6 * np.sign(finite[np.isinf(finite)])
    result = to_np(parallel.quantile(tensor([row], device, dtype), 5))
    assert_array_equal(result[0], pd.qcut(finite, 5, labels=False))


def test_quantile_too_few_values(device):
    x = tensor([[5, nan, nan, nan], [nan, nan, nan, nan], [nan, 1, 2, nan]], device)
    expected = [[nan] * 4, [nan] * 4, [nan, 0, 4, nan]]
    assert_array_equal(to_np(parallel.quantile(x, 5)), expected)


def test_quantile_ties_stay_in_one_bin(device):
    # the cuts are 1, 1, 1, 1, 1.2, 2, so the second to fourth bins are empty
    x = tensor([[1, 1, 2, 1, 1]], device)
    assert_array_equal(to_np(parallel.quantile(x, 5)), [[0, 0, 4, 0, 0]])
    assert_array_equal(to_np(parallel.quantile(tensor([[2, 2, 2]], device), 5)), [[0, 0, 0]])


def test_quantile_bool(device):
    x = torch.tensor([[True, False, False, False, True]], device=device)
    assert_array_equal(to_np(parallel.quantile(x, 2)), [[1, 0, 0, 0, 1]])


def test_masked_kth_value_median():
    x = tensor([[1, 2, nan, 3, 4, 5, 6], [3, 4, 5, 1.01, nan, 1.02, 1.03]])
    mask = torch.tensor(
        [[True, False, True, True, True, True, True], [True, True, True, True, True, True, True]]
    )
    expected = np.nanmedian(torch.masked_fill(x, ~mask, nan).numpy(), axis=1)
    median, _ = parallel.masked_kth_value_1d(x, mask, [0.5], dim=1)
    assert_almost_equal(median[0].squeeze(-1), expected, decimal=6)


@pytest.mark.parametrize(
    "k, even_mean, last_row", [(0.5, True, 5), (0.5, False, 5), (0.25, False, 4), (-0.25, False, 6)]
)
def test_masked_kth_value_empty_group(device, k, even_mean, last_row):
    x = tensor([[1, 2, 3], [nan, nan, nan], [nan, 2, 3], [4, 5, 6]], device)
    # the first rows have nothing that is both valid and inside the universe
    mask = [[False] * 3, [True] * 3, [True, False, False], [True] * 3]
    mask = torch.tensor(mask, device=device)
    values, _ = parallel.masked_kth_value_1d(x, mask, [k], even_mean=even_mean)
    assert_array_equal(to_np(values[0]).ravel(), [nan, nan, nan, last_row])


def test_masked_kth_value_fill_is_not_empty(device):
    x = tensor([[1, 2, 3], [nan, nan, nan]], device)
    mask = torch.tensor([[False] * 3, [True] * 3], device=device)
    values, _ = parallel.masked_kth_value_1d(x, mask, [0.5], nan_policy=7.0)
    # NaNs are filled in the universe, so only the row with an empty universe has no value
    assert_array_equal(to_np(values[0]).ravel(), [nan, 7])


def test_clamp_1d():
    data = tensor([[1, 5, 10], [-5, 0, 5]])
    result = parallel.clamp_1d_(data, tensor([[2], [-1]]), tensor([[8], [1]]))
    assert_array_equal(result, [[2, 5, 8], [-1, 0, 1]])
