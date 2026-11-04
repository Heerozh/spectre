import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_almost_equal

import spectre
from spectre import factors
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor

nan = np.nan
close = factors.OHLCV.close


def rolling_polyfit(y, win, deg, coef, x=None):
    """coef-th coefficient (in increasing powers) of the polynomial regression of y on x"""
    if x is None:
        return y.rolling(win).apply(
            lambda w: np.polyfit(np.arange(win), w, deg)[::-1][coef], raw=True)
    values = pd.concat([x, y], axis=1).values
    ret = [np.polyfit(values[i - win + 1:i + 1, 0], values[i - win + 1:i + 1, 1], deg)[::-1][coef]
           if i >= win - 1 else nan for i in range(len(values))]
    return pd.Series(ret, index=y.index)


@pytest.mark.parametrize('win', [2, 5])
def test_stddev(engine, win):
    assert_factor(engine, factors.STDDEV(win), *ref.expected(
        lambda df: df.close.rolling(win).std(ddof=0)))


def test_rolling_linear_regression(engine):
    f = factors.RollingLinearRegression(10, None, close)
    expected_slope = ref.expected(lambda df: rolling_polyfit(df.close, 10, 1, 1))
    expected_intercept = ref.expected(lambda df: rolling_polyfit(df.close, 10, 1, 0))
    assert_factor(engine, f.coef, *expected_slope, decimal=5)
    assert_factor(engine, f.intercept, *expected_intercept, decimal=3)
    assert_almost_equal(expected_slope[0], [-0.555879, -0.710545, -0.935697, -1.04103, -1.232,
                                            -1.704182, -0.873212, -0.640606, 0.046424],
                        decimal=5)

    engine.remove_all_factors()
    engine.add(f.coef, 'slope')
    engine.add(f.intercept, 'intcp')
    assert list(engine.run(ref.START, ref.END).columns) == ['slope', 'intcp']


def test_rolling_linear_regression_x(engine):
    f = factors.RollingLinearRegression(10, x=close, y=factors.OHLCV.open)
    expected = ref.expected(lambda df: rolling_polyfit(df.open, 10, 1, 1, x=df.close))
    assert_factor(engine, f.coef, *expected, decimal=5)


def test_rolling_momentum(engine):
    f = factors.RollingMomentum(5)
    for coef, factor in enumerate([f.intercept, f.gain, f.accelerate]):
        expected = ref.expected(lambda df: rolling_polyfit(df.close, 5, 2, coef))
        assert_factor(engine, factor, *expected, decimal=5)


def weekday_and_day(df):
    return (pd.Series(df.index.day_of_week, index=df.index, dtype=float),
            pd.Series(df.index.day, index=df.index, dtype=float))


def test_rolling_covariance(engine):
    factor = factors.RollingCovariance(
        win=3, inputs=[factors.WEEKDAY, factors.DatetimeDataFactor('day')])

    def cov(df):
        x, y = weekday_and_day(df)
        return x.rolling(3).cov(y)

    expected = ref.expected(cov, delayed=False)
    assert_factor(engine, factor, *expected, decimal=5)
    assert_almost_equal(expected[0][:3], [-3., -23.5, 1.])


def test_rolling_correlation(engine):
    factor = factors.RollingCorrelation(
        win=3, inputs=[factors.WEEKDAY, factors.DatetimeDataFactor('day')])

    def corr(df):
        x, y = weekday_and_day(df)
        return x.rolling(3).corr(y)

    assert_factor(engine, factor, *ref.expected(corr, delayed=False), decimal=6)


def test_cross_section_r2():
    y = torch.tensor([[1., 3, 2, 4, 6, 5, 7, 9],
                      [1., nan, nan, nan, nan, nan, nan, nan]])
    y_hat = torch.tensor([1., 2, 3, 4, 5, 6, 7, 8]).repeat(2, 1)
    result = factors.CrossSectionR2(None, None, None).compute(y, y_hat)

    y0, y_hat0 = y[0].numpy(), y_hat[0].numpy()
    expected = 1 - ((y0 - y_hat0) ** 2).sum() / ((y0 - y0.mean()) ** 2).sum()
    assert result[0, 0].item() == pytest.approx(expected, abs=1e-6)
    # r2 is undefined if less than 2 samples
    assert np.isnan(result[1, 0].item())


def test_xs_max_corr_coef():
    xs = [torch.tensor([[1., 2, 3, 4, 5], [1., 2, 3, 4, 5]]),
          torch.tensor([[1., 2, 3, 3, 4], [2., 1, 0, 4, 5]])]
    corr = factors.XSMaxCorrCoef().compute(*xs)
    assert corr.shape == (2, 5, 2)
    for date in range(2):
        expected = np.corrcoef(torch.stack([xs[0][date], xs[1][date]]).numpy())[0, 1]
        assert_almost_equal(corr[date].numpy(), np.full((5, 2), expected), decimal=6)


FACTOR_VALUES = [torch.tensor([[1., 2, 2, 4, 5], [1., 3, 3, 4, 5]]),
                 torch.tensor([[1., 3, 3, 3, 4], [2., 1, 0, 5, 5]]),
                 torch.tensor([[1., 1, 3, 3, 4], [2., 2, 0, 6, 6]])]


@pytest.mark.parametrize('kth', [1, 2, 3])
def test_factor_wise_kth_value(kth):
    result = factors.FactorWiseKthValue(kth, []).compute(*FACTOR_VALUES)
    # kth smallest of the cross-section means of each factor
    means = np.stack([x.numpy().mean(axis=1) for x in FACTOR_VALUES], axis=1)
    expected = np.sort(means, axis=1)[:, kth - 1]
    assert_almost_equal(np.sort(means, axis=1), [[2.4, 2.8, 2.8], [2.6, 3.2, 3.2]])
    assert_almost_equal(result.numpy(), np.repeat(expected[:, None], 5, axis=1), decimal=6)


def test_factor_wise_zscore():
    result = factors.FactorWiseZScore().compute(*FACTOR_VALUES)
    means = np.stack([x.numpy().mean(axis=1) for x in FACTOR_VALUES], axis=1)
    zscore = (means - means.mean(axis=1, keepdims=True)) / means.std(axis=1, keepdims=True)
    assert_almost_equal(result.numpy(), np.repeat(zscore[:, None, :], 5, axis=1), decimal=5)
    assert_almost_equal(zscore[0], [0.7071068, 0.7071068, -1.4142136], decimal=6)


# ------------------------------------ IC ------------------------------------

IC_DATA = np.array([
    [0.7, 0.3, 0, 0.1, 0.2, -0.1, -0.3, 0.4, -0.5, -0.3],
    [0.1, 0.4, 0.3, 0.4, -0.2, 0.2, 0.5, -0.3, -0.7, -0.9],
    [-0.1, -0.4, -0.3, -0.4, 0.2, -0.2, -0.5, 0.3, 0.7, 0.9],
    [5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
])
IC_DATE = pd.Timestamp('2020-01-02', tz='UTC')


def weighted_corr(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * x * y).sum() - mx * my
    return cov / np.sqrt(((w * x * x).sum() - mx ** 2) * ((w * y * y).sum() - my ** 2))


def rank_weight(y, half_life):
    alpha = np.exp(np.log(0.5) / half_life)
    return alpha ** (pd.Series(y).rank(ascending=False).values - 1)


@pytest.fixture
def ic_engine():
    # IC should not be related to the order of assets
    idx = np.random.default_rng(0).permutation(IC_DATA.shape[1])
    data = IC_DATA[:, idx]
    index = pd.MultiIndex.from_arrays([[IC_DATE] * data.shape[1], list('abcdefghij')],
                                      names=('date', 'asset'))
    df = pd.DataFrame(data.T, columns=['a', 'b', 'c', 'r'], index=index)
    return spectre.factors.FactorEngine(spectre.data.MemoryLoader(df))


def ic_inputs():
    return [factors.ColumnDataFactor(inputs=[col]) for col in 'abcr']


@pytest.mark.filterwarnings('ignore:Warning!! delay_factor is set to False')
def test_information_coefficient(ic_engine):
    a, b, c, r = ic_inputs()
    ic_engine.add([factors.InformationCoefficient(x, r) for x in (a, b, c)], ['a', 'b', 'c'])
    ic_engine.add([factors.RankWeightedInformationCoefficient(x, r, 5) for x in (a, b, c)],
                  ['a_weighted', 'b_weighted', 'c_weighted'])
    result = ic_engine.run(IC_DATE, IC_DATE, delay_factor=False)

    for i, col in enumerate('abc'):
        expected = np.corrcoef(IC_DATA[i], IC_DATA[3])[0, 1]
        assert result[col].iloc[0] == pytest.approx(expected, abs=1e-6)
        expected = weighted_corr(IC_DATA[i], IC_DATA[3], rank_weight(IC_DATA[3], 5))
        assert result[col + '_weighted'].iloc[0] == pytest.approx(expected, abs=1e-6)
    assert_almost_equal(result.iloc[0].values, [0.707, 0.716, -0.716, 0.747, 0.633, -0.633],
                        decimal=3)


@pytest.mark.filterwarnings('ignore:Warning!! delay_factor is set to False')
def test_rank_weighted_ic_with_mask(ic_engine):
    a, b, c, r = ic_inputs()
    ic_engine.add(factors.RankWeightedInformationCoefficient(a, r, 3, mask=a > 0), 'ic')
    result = ic_engine.run(IC_DATE, IC_DATE, delay_factor=False)

    masked = IC_DATA[0] > 0
    x, y = IC_DATA[0][masked], IC_DATA[3][masked]
    expected = weighted_corr(x, y, rank_weight(y, 3))
    assert result.ic.iloc[0] == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(0.518, abs=1e-3)


def test_ic_to_ir():
    ir = factors.InformationCoefficient(None, None).to_ir(3)
    ic = torch.tensor([1, 2, nan, 3, 1]).unsqueeze(-1).repeat(1, 5)
    result = ir.compute(ic)
    # rolling mean / std of ic, nan ignored
    expected = [nan, 1.5 / np.std([1, 2], ddof=1), 1.5 / np.std([1, 2], ddof=1),
                2.5 / np.std([2, 3], ddof=1), 2 / np.std([3, 1], ddof=1)]
    assert_almost_equal(result.numpy(), np.repeat(np.array(expected)[:, None], 5, axis=1),
                        decimal=5)
