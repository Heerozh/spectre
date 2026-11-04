import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.testing import assert_allclose, assert_almost_equal

import spectre
from spectre import factors
from tests.conftest import DIRTY_DATES, LISTED_ON
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor

nan = np.nan
close = factors.OHLCV.close


def rolling_polyfit(y, win, deg, coef, x=None):
    """coef-th coefficient (in increasing powers) of the polynomial regression of y on x"""
    if x is None:
        return y.rolling(win).apply(
            lambda w: np.polyfit(np.arange(win), w, deg)[::-1][coef], raw=True
        )
    values = pd.concat([x, y], axis=1).values
    ret = [
        np.polyfit(values[i - win + 1 : i + 1, 0], values[i - win + 1 : i + 1, 1], deg)[::-1][coef]
        if i >= win - 1
        else nan
        for i in range(len(values))
    ]
    return pd.Series(ret, index=y.index)


@pytest.mark.parametrize("win", [2, 5])
def test_stddev(engine, win):
    assert_factor(
        engine, factors.STDDEV(win), *ref.expected(lambda df: df.close.rolling(win).std(ddof=0))
    )


def test_rolling_linear_regression(engine):
    f = factors.RollingLinearRegression(10, None, close)
    expected_slope = ref.expected(lambda df: rolling_polyfit(df.close, 10, 1, 1))
    expected_intercept = ref.expected(lambda df: rolling_polyfit(df.close, 10, 1, 0))
    assert_factor(engine, f.coef, *expected_slope, decimal=5)
    assert_factor(engine, f.intercept, *expected_intercept, decimal=3)
    assert_almost_equal(expected_slope[0], [-0.555879, -0.710545, -0.935697, -1.04103, -1.232,
                                            -1.704182, -0.873212, -0.640606, 0.046424],
                        decimal=5)  # fmt: skip

    engine.remove_all_factors()
    engine.add(f.coef, "slope")
    engine.add(f.intercept, "intcp")
    assert list(engine.run(ref.START, ref.END).columns) == ["slope", "intcp"]


def test_rolling_linear_regression_x(engine):
    f = factors.RollingLinearRegression(10, x=close, y=factors.OHLCV.open)
    expected = ref.expected(lambda df: rolling_polyfit(df.open, 10, 1, 1, x=df.close))
    assert_factor(engine, f.coef, *expected, decimal=5)


def test_rolling_linear_regression_of_new_listing(dirty_engine, dirty_panel):
    """A partial window fits the bars that exist, at their positions in the window."""
    win = 10
    closes = dirty_panel.xs("LISTED", level="asset").close.values
    expected = [[nan, nan]]  # one bar has no slope
    for i in range(1, len(closes)):
        y = closes[max(0, i - win + 1) : i + 1]
        expected.append(np.polyfit(np.arange(win)[-len(y) :], y, 1))

    f = factors.RollingLinearRegression(win, None, close)
    dirty_engine.add(f.coef, "slope")
    dirty_engine.add(f.intercept, "intercept")
    result = dirty_engine.run(DIRTY_DATES[LISTED_ON], DIRTY_DATES[-1], delay_factor=False)
    assert_allclose(
        result.xs("LISTED", level="asset")[["slope", "intercept"]].values, expected, rtol=1e-6
    )


def test_rolling_momentum(engine):
    f = factors.RollingMomentum(5)
    for coef, factor in enumerate([f.intercept, f.gain, f.accelerate]):
        expected = ref.expected(lambda df, coef=coef: rolling_polyfit(df.close, 5, 2, coef))
        assert_factor(engine, factor, *expected, decimal=5)


def weekday_and_day(df):
    return (
        pd.Series(df.index.day_of_week, index=df.index, dtype=float),
        pd.Series(df.index.day, index=df.index, dtype=float),
    )


def test_rolling_covariance(engine):
    factor = factors.RollingCovariance(
        win=3, inputs=[factors.WEEKDAY, factors.DatetimeDataFactor("day")]
    )

    def cov(df):
        x, y = weekday_and_day(df)
        return x.rolling(3).cov(y)

    expected = ref.expected(cov, delayed=False)
    assert_factor(engine, factor, *expected, decimal=5)
    assert_almost_equal(expected[0][:3], [-3.0, -23.5, 1.0])


def test_rolling_correlation(engine):
    factor = factors.RollingCorrelation(
        win=3, inputs=[factors.WEEKDAY, factors.DatetimeDataFactor("day")]
    )

    def corr(df):
        x, y = weekday_and_day(df)
        return x.rolling(3).corr(y)

    assert_factor(engine, factor, *ref.expected(corr, delayed=False), decimal=6)


def half_life(series, mean, win, keep=None):
    """Half life of each window, from a regression on the samples that have a lag and a diff."""
    lag = (series.shift(1) - mean).values
    diff = series.diff().values
    if keep is not None:
        lag, diff = np.where(keep, lag, nan), np.where(keep, diff, nan)
    ret = np.full(len(series), nan)
    for i in range(len(series)):
        x, y = lag[max(0, i - win + 1) : i + 1], diff[max(0, i - win + 1) : i + 1]
        valid = ~np.isnan(x * y)
        if valid.sum() > 1:
            ret[i] = -np.log(2) / np.polyfit(x[valid], y[valid], 1)[0]
    return pd.Series(ret, index=series.index)


def rolling_nanmean(series, win):
    return series.rolling(win, min_periods=1).mean()


# the adjustment multipliers are float32, a slope near 0 magnifies them
HALF_LIFE_RTOL = 1e-4


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("asset", ["SPLIT", "FULL", "HALTED", "NANBAR"])
def test_half_life_mean_reversion(dirty_engine, dirty_panel, asset):
    """A split or a dividend in the window does not change a regression in the adjusted scale."""
    win, ma_win, start = 10, 5, DIRTY_DATES[15]
    dirty_engine.add(factors.HalfLifeMeanReversion(win, close, factors.MA(ma_win)), "half_life")
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    clean = dirty_panel.xs(asset, level="asset").clean_close
    expected = half_life(clean, rolling_nanmean(clean, ma_win), win).loc[start:]
    got = result.xs(asset, level="asset").half_life
    assert_allclose(got.values, expected.values, rtol=HALF_LIFE_RTOL)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("asset", ["SPLIT", "FULL"])
def test_half_life_mean_reversion_of_a_number(dirty_engine, dirty_panel, asset):
    win, start = 10, DIRTY_DATES[15]
    dirty_engine.add(factors.HalfLifeMeanReversion(win, close, 0.0), "half_life")
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    clean = dirty_panel.xs(asset, level="asset").clean_close
    expected = half_life(clean, 0.0, win).loc[start:]
    got = result.xs(asset, level="asset").half_life
    assert_allclose(got.values, expected.values, rtol=HALF_LIFE_RTOL)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_half_life_mean_reversion_without_adjustments(ir_engine, ir_panel):
    win, ma_win, start = 10, 5, IR_DATES[15]
    ir_engine.add(factors.HalfLifeMeanReversion(win, close, factors.MA(ma_win)), "half_life")
    result = ir_engine.run(start, IR_DATES[-1], delay_factor=False)

    closes = ir_panel.xs("A", level="asset").close
    expected = half_life(closes, rolling_nanmean(closes, ma_win), win).loc[start:]
    assert_allclose(result.xs("A", level="asset").half_life, expected, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_half_life_mean_reversion_mask_drops_the_samples_of_masked_bars(dirty_engine, dirty_panel):
    win, ma_win, start = 10, 5, DIRTY_DATES[15]
    mask = factors.OHLCV.volume > 1400
    f = factors.HalfLifeMeanReversion(win, close, factors.MA(ma_win), mask=mask)
    dirty_engine.add(f, "half_life")
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    panel = dirty_panel.xs("FULL", level="asset")
    # a sample has data of 2 bars, only the last one decides if it stays
    keep = (panel.volume > 1400).values
    assert (keep[:-1] & ~keep[1:]).any() and (~keep[:-1] & keep[1:]).any()
    expected = half_life(panel.clean_close, rolling_nanmean(panel.clean_close, ma_win), win, keep)
    got = result.xs("FULL", level="asset").half_life
    assert_allclose(got.values, expected.loc[start:].values, rtol=1e-6)


def test_cross_section_r2():
    y = torch.tensor([[1.0, 3, 2, 4, 6, 5, 7, 9], [1.0, nan, nan, nan, nan, nan, nan, nan]])
    y_hat = torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8]).repeat(2, 1)
    result = factors.CrossSectionR2(None, None, None).compute(y, y_hat)

    y0, y_hat0 = y[0].numpy(), y_hat[0].numpy()
    expected = 1 - ((y0 - y_hat0) ** 2).sum() / ((y0 - y0.mean()) ** 2).sum()
    assert result[0, 0].item() == pytest.approx(expected, abs=1e-6)
    # r2 is undefined if less than 2 samples
    assert np.isnan(result[1, 0].item())


def test_xs_max_corr_coef():
    xs = [
        torch.tensor([[1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5]]),
        torch.tensor([[1.0, 2, 3, 3, 4], [2.0, 1, 0, 4, 5]]),
    ]
    corr = factors.XSMaxCorrCoef().compute(*xs)
    assert corr.shape == (2, 5, 2)
    for date in range(2):
        expected = np.corrcoef(torch.stack([xs[0][date], xs[1][date]]).numpy())[0, 1]
        assert_almost_equal(corr[date].numpy(), np.full((5, 2), expected), decimal=6)


def xs_max_corr(*columns):
    """Max correlation of each column with the others, on the rows both of them have."""
    corr = pd.DataFrame(np.stack(columns, axis=1)).corr().to_numpy(copy=True)
    np.fill_diagonal(corr, nan)
    return pd.DataFrame(corr).max(axis=1).values


def assert_max_corr_same_as_pandas(device, *xs, dtype=torch.float64, rtol=1e-9):
    tensors = [torch.tensor(x, device=device, dtype=dtype) for x in xs]
    result = factors.XSMaxCorrCoef().compute(*tensors).cpu().numpy()
    dates, assets = xs[0].shape
    assert result.shape == (dates, assets, len(xs))
    for date in range(dates):
        expected = np.tile(xs_max_corr(*[x[date] for x in xs]), (assets, 1))
        assert_allclose(result[date], expected, rtol=rtol, atol=rtol, err_msg=f"date {date}")


def test_xs_max_corr_coef_uses_the_samples_of_each_pair(device):
    a = np.array([[1.0, 2, 3, 4, nan, nan], [nan, 5, 3, 8, 1, 2]])
    b = np.array([[2.0, 4, 6, 8, 100, -100], [3, 1, nan, 7, 2, 2]])
    assert_max_corr_same_as_pandas(device, a, b)
    assert_almost_equal(xs_max_corr(a[0], b[0]), [1, 1])


def test_xs_max_corr_coef_of_pairs_without_samples(device):
    a = np.array([[1.0, 2, 3, nan, nan, nan]])
    b = np.array([[nan, nan, nan, 1.0, 2, 4]])
    c = np.array([[1.0, 3, 2, 5, 4, 7]])
    assert_max_corr_same_as_pandas(device, a, b, c)
    assert_max_corr_same_as_pandas(device, a, b)


def test_xs_max_corr_coef_of_negative_correlation(device):
    """The correlation of a factor with itself is not one of the others, even if it's the max."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 200))
    b = -a + rng.normal(size=a.shape) * 0.3
    assert_max_corr_same_as_pandas(device, a, b)
    assert xs_max_corr(a[0], b[0])[0] < -0.9


@pytest.mark.parametrize("dtype, rtol", [(torch.float64, 1e-9), (torch.float32, 1e-4)])
def test_xs_max_corr_coef_of_constant_factor(device, dtype, rtol):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 30))
    b = -a + rng.normal(size=a.shape) * 0.3
    # a different constant on each date, the rounding of the mean depends on the value
    constant = np.repeat(rng.uniform(1, 300, (100, 1)), 30, axis=1)
    assert_max_corr_same_as_pandas(device, a, b, constant, dtype=dtype, rtol=rtol)
    assert np.isnan(xs_max_corr(a[0], b[0], constant[0])[2])


@pytest.mark.parametrize("dtype, rtol", [(torch.float64, 1e-9), (torch.float32, 1e-4)])
def test_xs_max_corr_coef_of_constant_samples_of_a_pair(device, dtype, rtol):
    """A factor can be constant only on the samples it shares with another one."""
    rng = np.random.default_rng(0)
    b = rng.normal(size=(100, 9))
    c = -b + rng.normal(size=b.shape) * 0.3
    b[:, 7:] = nan
    a = np.repeat(rng.uniform(1, 300, (100, 1)), 9, axis=1)
    a[:, 7] -= rng.uniform(1, 3, 100)
    a[:, 8] += rng.uniform(1, 3, 100)
    assert_max_corr_same_as_pandas(device, a, b, c, dtype=dtype, rtol=rtol)
    assert xs_max_corr(a[0], b[0], c[0])[1] < -0.8


def test_xs_max_corr_coef_of_one_factor_has_no_others(device):
    x = torch.tensor([[1.0, 2, 3]], device=device)
    assert_allclose(factors.XSMaxCorrCoef().compute(x).cpu(), np.full((1, 3, 1), nan))


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_xs_max_corr_coef_on_dirty_data(dirty_engine, dirty_panel):
    columns = ["open", "close", "volume"]
    max_corr = factors.XSMaxCorrCoef(inputs=[getattr(factors.OHLCV, c) for c in columns])
    for i, column in enumerate(columns):
        dirty_engine.add(max_corr[i], column)
    result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1], delay_factor=False)

    # NANBAR has a volume but no prices, so the 3 pairs don't have the same samples
    for date in DIRTY_DATES:
        raw = dirty_panel.loc[date, columns].values
        expected = np.tile(xs_max_corr(*raw.T), (raw.shape[0], 1))
        assert_allclose(result.loc[date, columns].values, expected, rtol=1e-6, err_msg=str(date))


FACTOR_VALUES = [
    torch.tensor([[1.0, 2, 2, 4, 5], [1.0, 3, 3, 4, 5]]),
    torch.tensor([[1.0, 3, 3, 3, 4], [2.0, 1, 0, 5, 5]]),
    torch.tensor([[1.0, 1, 3, 3, 4], [2.0, 2, 0, 6, 6]]),
]


@pytest.mark.parametrize("kth", [1, 2, 3])
def test_factor_wise_kth_value(kth):
    result = factors.FactorWiseKthValue(kth, []).compute(*FACTOR_VALUES)
    # kth smallest of the cross-section means of each factor
    means = np.stack([x.numpy().mean(axis=1) for x in FACTOR_VALUES], axis=1)
    expected = np.sort(means, axis=1)[:, kth - 1]
    assert_almost_equal(np.sort(means, axis=1), [[2.4, 2.8, 2.8], [2.6, 3.2, 3.2]])
    assert_almost_equal(result.numpy(), np.repeat(expected[:, None], 5, axis=1), decimal=6)


@pytest.mark.parametrize("kth", [1, 2, 3])
def test_factor_wise_kth_value_of_missing_inputs(device, kth):
    """NaN is not one of the values, and there is no kth value if fewer inputs have one."""
    xs = [x.to(device=device, dtype=torch.float64).repeat(2, 1) for x in FACTOR_VALUES]
    xs[1][0] = nan  # one input is out on date 0
    xs[1][1] = xs[2][1] = nan  # two inputs on date 1
    for x in xs:
        x[2] = nan  # all on date 2
    xs[0][3] = np.inf  # a real value, not the same as a missing one
    result = factors.FactorWiseKthValue(kth, []).compute(*xs).cpu().numpy()

    expected = []
    for date in range(result.shape[0]):
        means = [x[date].cpu().numpy().mean() for x in xs]
        valid = np.sort([m for m in means if not np.isnan(m)])
        expected.append(valid[kth - 1] if kth <= len(valid) else nan)
    assert_allclose(result, np.repeat(np.array(expected)[:, None], 5, axis=1), rtol=1e-12)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.filterwarnings("ignore:Historical data seems insufficient")
def test_factor_wise_kth_value_on_engine(dirty_engine):
    """An input with no value on the first dates, like a long rolling window at the start."""
    inputs = [factors.Returns(win=20), factors.Returns(win=5), factors.Returns(win=2)]
    for kth in (1, 2, 3):
        dirty_engine.add(factors.FactorWiseKthValue(kth, inputs=inputs), f"k{kth}")
    for i, x in enumerate(inputs):
        dirty_engine.add(x, f"x{i}")
    result = dirty_engine.run(DIRTY_DATES[5], DIRTY_DATES[-1], delay_factor=False)

    means = result[["x0", "x1", "x2"]].groupby(level="date").mean().values
    for date, values in enumerate(means):
        valid = np.sort(values[~np.isnan(values)])
        row = result.loc[DIRTY_DATES[5 + date], ["k1", "k2", "k3"]].iloc[0].values
        expected = [valid[k] if k < len(valid) else nan for k in range(3)]
        assert_allclose(row, expected, rtol=1e-6, err_msg=str(DIRTY_DATES[5 + date]))
    assert np.isnan(means[:, 0]).any()


def test_factor_wise_zscore():
    result = factors.FactorWiseZScore().compute(*FACTOR_VALUES)
    means = np.stack([x.numpy().mean(axis=1) for x in FACTOR_VALUES], axis=1)
    zscore = (means - means.mean(axis=1, keepdims=True)) / means.std(axis=1, keepdims=True)
    assert_almost_equal(result.numpy(), np.repeat(zscore[:, None, :], 5, axis=1), decimal=5)
    assert_almost_equal(zscore[0], [0.7071068, 0.7071068, -1.4142136], decimal=6)


# ------------------------------------ IC ------------------------------------

IC_DATA = np.array(
    [
        [0.7, 0.3, 0, 0.1, 0.2, -0.1, -0.3, 0.4, -0.5, -0.3],
        [0.1, 0.4, 0.3, 0.4, -0.2, 0.2, 0.5, -0.3, -0.7, -0.9],
        [-0.1, -0.4, -0.3, -0.4, 0.2, -0.2, -0.5, 0.3, 0.7, 0.9],
        [5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
    ]
)
IC_DATE = pd.Timestamp("2020-01-02", tz="UTC")


def weighted_corr(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * x * y).sum() - mx * my
    return cov / np.sqrt(((w * x * x).sum() - mx**2) * ((w * y * y).sum() - my**2))


def rank_weight(y, half_life):
    alpha = np.exp(np.log(0.5) / half_life)
    return alpha ** (pd.Series(y).rank(ascending=False).values - 1)


@pytest.fixture
def ic_engine():
    # IC should not be related to the order of assets
    idx = np.random.default_rng(0).permutation(IC_DATA.shape[1])
    data = IC_DATA[:, idx]
    index = pd.MultiIndex.from_arrays(
        [[IC_DATE] * data.shape[1], list("abcdefghij")], names=("date", "asset")
    )
    df = pd.DataFrame(data.T, columns=["a", "b", "c", "r"], index=index)
    return spectre.factors.FactorEngine(spectre.data.MemoryLoader(df))


def ic_inputs():
    return [factors.ColumnDataFactor(inputs=[col]) for col in "abcr"]


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_information_coefficient(ic_engine):
    a, b, c, r = ic_inputs()
    ic_engine.add([factors.InformationCoefficient(x, r) for x in (a, b, c)], ["a", "b", "c"])
    ic_engine.add(
        [factors.RankWeightedInformationCoefficient(x, r, 5) for x in (a, b, c)],
        ["a_weighted", "b_weighted", "c_weighted"],
    )
    result = ic_engine.run(IC_DATE, IC_DATE, delay_factor=False)

    for i, col in enumerate("abc"):
        expected = np.corrcoef(IC_DATA[i], IC_DATA[3])[0, 1]
        assert result[col].iloc[0] == pytest.approx(expected, abs=1e-6)
        expected = weighted_corr(IC_DATA[i], IC_DATA[3], rank_weight(IC_DATA[3], 5))
        assert result[col + "_weighted"].iloc[0] == pytest.approx(expected, abs=1e-6)
    assert_almost_equal(
        result.iloc[0].values, [0.707, 0.716, -0.716, 0.747, 0.633, -0.633], decimal=3
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_information_coefficient_of_constant_factor_is_nan(device, dtype):
    rng = np.random.default_rng(0)
    x = torch.tensor(np.repeat(rng.uniform(1, 300, (500, 1)), 30, axis=1), device=device)
    y = torch.tensor(rng.normal(size=(500, 30)), device=device)
    ic = factors.InformationCoefficient(None, None).compute(x.to(dtype), y.to(dtype), None)
    assert torch.isnan(ic).all()


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_rank_weighted_ic_with_mask(ic_engine):
    a, _b, _c, r = ic_inputs()
    ic_engine.add(factors.RankWeightedInformationCoefficient(a, r, 3, mask=a > 0), "ic")
    result = ic_engine.run(IC_DATE, IC_DATE, delay_factor=False)

    masked = IC_DATA[0] > 0
    x, y = IC_DATA[0][masked], IC_DATA[3][masked]
    expected = weighted_corr(x, y, rank_weight(y, 3))
    assert result.ic.iloc[0] == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(0.518, abs=1e-3)


def test_weighted_ic_leaves_out_samples_without_a_weight(device):
    x, y = IC_DATA[0].copy(), IC_DATA[3].copy()
    w = rank_weight(y, 5)
    w[2] = nan
    x[5] = nan
    # a date without weights has no samples at all
    tensors = [
        torch.tensor(np.stack([v, v]), device=device, dtype=torch.float64) for v in (x, y, w)
    ]
    tensors[2][1] = nan
    result = factors.InformationCoefficient(None, None).compute(*tensors).cpu().numpy()

    valid = ~np.isnan(x * y * w)
    expected = weighted_corr(x[valid], y[valid], w[valid])
    assert_allclose(result[0], np.full(len(x), expected), rtol=1e-12)
    assert np.isnan(result[1]).all()


def test_ic_to_ir():
    ir = factors.InformationCoefficient(None, None).to_ir(3)
    ic = torch.tensor([1, 2, nan, 3, 1]).unsqueeze(-1).repeat(1, 5)
    result = ir.compute(ic)
    # rolling mean / std of ic, nan ignored
    expected = [
        nan,
        1.5 / np.std([1, 2], ddof=1),
        1.5 / np.std([1, 2], ddof=1),
        2.5 / np.std([2, 3], ddof=1),
        2 / np.std([3, 1], ddof=1),
    ]
    assert_almost_equal(
        result.numpy(), np.repeat(np.array(expected)[:, None], 5, axis=1), decimal=5
    )


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("asset", ["FULL", "NANBAR", "LISTED"])
def test_rolling_ic_to_ir(dirty_engine, dirty_panel, asset):
    win, ir_win, start = 5, 3, DIRTY_DATES[LISTED_ON]
    rolling_ic = factors.RollingInformationCoefficient(win, inputs=[factors.OHLCV.open, close])
    dirty_engine.add(rolling_ic.to_ir(ir_win), "ir")
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    panel = dirty_panel.xs(asset, level="asset")
    ic = panel.open.rolling(win, min_periods=2).corr(panel.close)
    # a missing IC is left out of the window, it does not count as 0
    mean = ic.rolling(ir_win, min_periods=1).mean()
    expected = (mean / ic.rolling(ir_win, min_periods=2).std(ddof=1)).loc[start:]
    assert_allclose(result.xs(asset, level="asset").ir, expected, rtol=1e-6)


IR_DATES = pd.date_range("2020-01-01", periods=45, freq="D", tz="UTC")


@pytest.fixture
def ir_panel():
    """Random walks with a noisy open, so the IC of open and close changes from date to date."""
    steps = np.random.default_rng(2).normal(0, 2, (len(IR_DATES), 6, 2))
    walks = 100 + steps[:, :, 0].cumsum(axis=0)
    index = pd.MultiIndex.from_product([IR_DATES, list("ABCDEF")], names=["date", "asset"])
    df = pd.DataFrame(
        {"close": walks.ravel().round(3), "open": (walks + steps[:, :, 1]).ravel().round(3)},
        index=index,
    )
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = 1.0
    return df


@pytest.fixture
def ir_engine(ir_panel, device):
    loader = spectre.data.MemoryLoader(ir_panel, ohlcv=("open", "high", "low", "close", "volume"))
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    return engine


def rolling_ir(ic, win):
    return ic.rolling(win).mean() / ic.rolling(win).std(ddof=1)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_ic_to_ir_loads_the_history_of_its_window(ir_engine, ir_panel):
    win, start = 10, IR_DATES[20]
    ic = factors.InformationCoefficient(factors.OHLCV.open, close)
    ir_engine.add(ic.to_ir(win), "ir")
    result = ir_engine.run(start, IR_DATES[-1], delay_factor=False)

    closes, opens = ir_panel.close.unstack(), ir_panel.open.unstack()
    expected = rolling_ir(opens.corrwith(closes, axis=1), win).loc[start:]
    assert_allclose(result.ir.xs("A", level="asset").values, expected.values, rtol=1e-6)
    assert_almost_equal(expected.values[:3], [14.32, 14.27, 13.93], decimal=2)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_ic_to_ir_adds_its_window_to_the_one_of_its_inputs(ir_engine, ir_panel):
    win, ma_win, start = 10, 5, IR_DATES[20]
    ma = factors.MA(ma_win, inputs=[factors.OHLCV.open])
    ir_engine.add(factors.InformationCoefficient(ma, close).to_ir(win), "ir")
    result = ir_engine.run(start, IR_DATES[-1], delay_factor=False)

    closes, opens = ir_panel.close.unstack(), ir_panel.open.unstack()
    ic = opens.rolling(ma_win).mean().corrwith(closes, axis=1)
    expected = rolling_ir(ic, win).loc[start:]
    assert_allclose(result.ir.xs("A", level="asset").values, expected.values, rtol=1e-6)


# ------------------------------------ t-test ------------------------------------


def rolling_ttest(values, win, pop_mean):
    """t-value of the valid values of each window, NaN if there are less than 2 of them."""
    ret = []
    for i in range(len(values)):
        window = values[max(0, i - win + 1) : i + 1]
        window = window[~np.isnan(window)]
        ret.append(scipy.stats.ttest_1samp(window, pop_mean).statistic if len(window) > 1 else nan)
    return np.array(ret)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.filterwarnings("ignore:Historical data seems insufficient")
@pytest.mark.parametrize("win", [3, 20])
@pytest.mark.parametrize(
    "pop_mean, assets",
    [
        (100.0, ["FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"]),
        # t of a window does not depend on its scale only if the mean is 0
        (0.0, ["FULL", "LISTED", "DELISTED", "HALTED", "NANBAR", "SPLIT"]),
    ],
)
def test_ttest_1samp_same_as_scipy(dirty_engine, dirty_panel, win, pop_mean, assets):
    dirty_engine.add(factors.TTest1Samp(win, inputs=[close, pop_mean]), "t")
    result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1], delay_factor=False)
    for asset in assets:
        got = result.xs(asset, level="asset").t.values
        closes = dirty_panel.xs(asset, level="asset").clean_close.values
        # the adjustment multipliers are float32, and t magnifies them by mean / std
        assert_allclose(got, rolling_ttest(closes, win, pop_mean), rtol=1e-5, err_msg=asset)


CDF_DOFS = [0.5, 1, 1.5, 2, 3, 5, 10, 30, 100, 341, 342, 343, 500, 1000]
CDF_PRECISIONS = [(factors.StudentCDF.DefaultPrecision, 1e-6), (1e-7, 1e-10)]
CDF_T = np.logspace(-6, 2, 200)
CDF_T = np.concatenate([-CDF_T[::-1], [0], CDF_T])
CDF_HUGE_T = np.logspace(2, 7, 200)
CDF_HUGE_T = np.concatenate([-CDF_HUGE_T[::-1], CDF_HUGE_T])


def student_cdf(t, dof, precision=factors.StudentCDF.DefaultPrecision, device="cpu"):
    """StudentCDF of each value in `t`, with one date for each value."""
    t = torch.tensor(t, device=device, dtype=torch.float64).unsqueeze(-1)
    return factors.StudentCDF().compute(t, dof, precision)[:, 0].cpu().numpy()


@pytest.mark.parametrize("dof", CDF_DOFS)
@pytest.mark.parametrize("precision, rtol", CDF_PRECISIONS)
def test_student_cdf_same_as_scipy(device, dof, precision, rtol):
    result = student_cdf(CDF_T, dof, precision, device)
    # the left tail is compared by relative error too, down to where scipy underflows
    assert_allclose(result, scipy.stats.t.cdf(CDF_T, dof), rtol=rtol, atol=1e-290)


@pytest.mark.parametrize("dof", CDF_DOFS)
@pytest.mark.parametrize("precision, rtol", CDF_PRECISIONS)
def test_student_cdf_huge_t_same_as_scipy(device, dof, precision, rtol):
    result = student_cdf(CDF_HUGE_T, dof, precision, device)
    assert_allclose(result, scipy.stats.t.cdf(CDF_HUGE_T, dof), rtol=rtol, atol=1e-290)


def test_student_cdf_closed_forms(device):
    """A tighter check than scipy allows, which is only exact to 3e-11 around t=0 when dof=1."""
    t = np.concatenate([CDF_T, CDF_HUGE_T])
    # both are written so that they don't cancel digits away in the left tail
    assert_allclose(student_cdf(t, 1, 1e-7, device), np.arctan2(1, -t) / np.pi, rtol=1e-14)
    root = np.sqrt(2 + t**2)
    tail = 1 / (root * (root + np.abs(t)))
    assert_allclose(student_cdf(t, 2, 1e-7, device), np.where(t < 0, tail, 1 - tail), rtol=1e-14)


def test_student_cdf_limits(device):
    result = student_cdf([-np.inf, np.inf, nan, 0.0], 5, device=device)
    assert_allclose(result, [0, 1, nan, 0.5])
    assert result[0] == 0


def test_student_cdf_uses_the_cross_section_mean(device):
    t = torch.tensor([[1.0, 3, nan], [-5.0, nan, -1], [nan, nan, nan]], device=device)
    result = factors.StudentCDF().compute(t, 5, factors.StudentCDF.DefaultPrecision).cpu()
    assert result.dtype == t.dtype
    expected = scipy.stats.t.cdf([[2.0], [-3], [nan]], 5).repeat(3, axis=1)
    assert_allclose(result.numpy(), expected, rtol=1e-5)


def test_student_cdf_precision_sets_the_error(device):
    t = CDF_T[np.abs(CDF_T) <= 10]
    expected = scipy.stats.t.cdf(t, 1000)
    errors = []
    for precision in (0.1, 0.01, 0.001):
        errors.append(np.abs(student_cdf(t, 1000, precision, device) / expected - 1).max())
        assert errors[-1] < precision**2
    assert errors[0] > errors[1] > errors[2]


@pytest.mark.cuda
@pytest.mark.parametrize("t", [1.0, 1e5, 1e7])
def test_student_cdf_memory_does_not_depend_on_t(t):
    values = torch.full((100, 5), t, device="cuda", dtype=torch.float64)
    torch.cuda.reset_peak_memory_stats()
    used = torch.cuda.memory_allocated()
    factors.StudentCDF().compute(values, 5, factors.StudentCDF.DefaultPrecision)
    assert torch.cuda.max_memory_allocated() - used < 2**20


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.filterwarnings("ignore:Historical data seems insufficient")
def test_student_cdf_of_ttest(dirty_engine):
    win = 5
    # `* 1` leaves the prices raw, `close` itself would be adjusted
    t = factors.TTest1Samp(win, inputs=[close * 1, 100.0])
    dirty_engine.add(t, "t")
    p = factors.StudentCDF(inputs=[t, win - 1, factors.StudentCDF.DefaultPrecision])
    dirty_engine.add(p, "p")
    result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1], delay_factor=False)

    dates = result.index.get_level_values("date")
    reduced_t = result.t.groupby(level="date").mean().reindex(dates)
    assert_allclose(result.p.values, scipy.stats.t.cdf(reduced_t.values, win - 1), rtol=1e-6)
