import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from spectre import factors, parallel
from tests.conftest import DATA_DIR, DIRTY_DATES, LISTED_ON
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, delete, run_dirty, run_factor

nan = np.nan
close = factors.OHLCV.close
volume = factors.OHLCV.volume

# On 2019-01-10 only AAPL has data, so the values of AAPL on 2019-01-11 (delayed) are calculated
# without MSFT (index 6 of delayed AAPL results).
SOLO = 6


def compute_with_mask(factor, *inputs):
    """Call factor.compute() without engine, the mask is non-nan of the first input."""
    factor._mask = factor
    factor._mask_out = ~torch.isnan(inputs[0])
    return factor.compute(*inputs)


# ------------------------------------ rank ------------------------------------


@pytest.mark.parametrize("ascending", [True, False])
def test_rank(engine, ascending):
    expected = ref.expected_xs(lambda df: df.rank(axis=1, ascending=ascending))
    assert_factor(engine, close.rank(ascending=ascending), *expected)
    # AAPL always has higher price than MSFT
    assert_array_equal(expected[0], ([2.0] * SOLO + [1.0] + [2.0] * 2) if ascending else [1.0] * 9)


def test_rank_single_day(engine):
    # bug #98a0bdc
    engine.add(close.rank(), "test")
    result = engine.run("2019-01-01", "2019-01-03")
    assert_array_equal(result.values, [[2.0], [1.0]])


@pytest.mark.parametrize("method", ["average", "dense", "ordinal"])
def test_rank_single_asset(engine, method):
    engine.set_filter(factors.StaticAssets({"AAPL"}))
    engine.add(close.rank(method=method), "test")
    assert_array_equal(engine.run(ref.START, ref.END).test.values, [1.0] * 9)


def test_rank_all_tied(engine):
    # all assets have the same weekday
    expected_aapl = [1.5] * 10
    expected_aapl[SOLO] = 1.0
    assert_factor(engine, factors.WEEKDAY.rank(), expected_aapl, [1.5] * 9)


def test_rank_with_mask(engine, device):
    if device == "cuda":
        engine.to_cuda()
    universe = factors.OHLCV.volume.top(1)
    engine.set_filter(universe)
    engine.add(close.rank(mask=universe), "test")
    result = engine.run("2019-01-01", "2019-01-15")
    assert_array_equal(result.test.values, [1.0] * 9)


def test_rank_normalize(engine):
    # divided by the max rank of each day
    def normalized_rank(df):
        rank = df.rank(axis=1)
        return rank.div(rank.max(axis=1), axis=0)

    expected = ref.expected_xs(normalized_rank)
    assert_factor(engine, close.rank(normalize=True), *expected)


# ------------------------------------ statistics ------------------------------------


def test_mean(engine):
    expected = ref.expected_xs(lambda df: pd.DataFrame({a: df.mean(axis=1) for a in df}))
    assert_factor(engine, close.mean(), *expected)
    assert expected[0][SOLO] == 161.0


def test_weighted_mean(engine):
    def weighted_mean(_):
        prices = {a: ref.daily_prices(a) for a in ref.ASSETS}
        value = sum(p.close * p.volume for p in prices.values())
        volume = sum(p.volume for p in prices.values())
        # 2019-01-10 only has AAPL
        solo = prices["AAPL"].close[value.isna()]
        mean = (value / volume).fillna(solo)
        return pd.DataFrame({a: mean for a in ref.ASSETS})

    expected = ref.expected_xs(weighted_mean)
    # mean is float32
    assert_factor(engine, close.mean(weight=factors.OHLCV.volume), *expected, decimal=4)


def xs_weighted_mean(data, weight):
    """Weighted mean of each date, over the rows valid in both `data` and `weight`."""
    total = (data * weight).groupby(level="date").transform("sum", min_count=1)
    return total / weight.where(data.notna()).groupby(level="date").transform("sum")


def test_weighted_mean_drops_the_weight_of_missing_data(dirty_engine, dirty_panel):
    expected = xs_weighted_mean(dirty_panel.close, dirty_panel.volume)
    assert_allclose(run_dirty(dirty_engine, close.mean(weight=volume)), expected, rtol=1e-5)


def test_weighted_mean_by_filter(dirty_engine, dirty_panel):
    expected = xs_weighted_mean(dirty_panel.close, (dirty_panel.close > 100).astype(float))
    assert_allclose(run_dirty(dirty_engine, close.mean(weight=close > 100)), expected, rtol=1e-5)


def test_weighted_mean_of_rows(device):
    data = torch.tensor([[1, nan, 3], [4, nan, 8], [nan, nan, nan], [1, 2, nan]], device=device)
    weight = torch.tensor([[1, 1, 1], [1, 8, 1], [1, 1, 1], [nan, 1, 1]], device=device)
    result = factors.factor.MeanFactor().compute(data, weight)
    assert_array_equal(result.cpu().numpy(), [[2] * 3, [6] * 3, [nan] * 3, [2] * 3])


def test_weighted_mean_of_rows_with_weight_sum(device):
    data = torch.tensor([[1, nan, 3], [nan, nan, nan]], device=device)
    result = factors.factor.MeanFactor().compute(data, torch.ones_like(data), 8)
    assert_array_equal(result.cpu().numpy(), [[0.5] * 3, [nan] * 3])


def test_std(engine):
    expected = ref.expected_xs(lambda df: pd.DataFrame({a: df.std(axis=1, ddof=0) for a in df}))
    assert_factor(engine, close.std(), *expected)
    assert expected[0][SOLO] == 0


def test_xs_sum(engine):
    log_returns = factors.LogReturns()
    aapl, msft = ref.expected(lambda df: ref.log_returns(df.close))
    msft_full = np.insert(msft, SOLO, 0)
    expected_aapl = aapl + msft_full
    assert_factor(engine, log_returns.xs_sum(), expected_aapl, delete(expected_aapl, SOLO))


def test_median(engine):
    expected = ref.expected_xs(lambda df: pd.DataFrame({a: df.median(axis=1) for a in df}))
    assert_factor(engine, close.median(), *expected)


def test_median_groupby_asset(engine):
    """groupby='asset' calculates the median of the whole loaded data of each asset."""
    factor = factors.Returns().round(3).median(groupby="asset")
    # closes from 2018-12-31 to 2019-01-15 are loaded, so returns are from 2019-01-02
    returns = {
        a: ref.returns(ref.daily_prices(a).close).loc["2019-01-02" : ref.END].round(3)
        for a in ref.ASSETS
    }
    aapl, msft = run_factor(engine, factor, check_bias=False)
    assert_almost_equal(aapl, [np.median(returns["AAPL"])] * 9)
    assert_almost_equal(msft, [np.median(returns["MSFT"])] * 8)
    assert_almost_equal([np.median(returns["AAPL"]), np.median(returns["MSFT"])], [0.0085, 0.007])


def test_xs_sum_of_a_date_without_values(dirty_engine, dirty_panel):
    weekdays = dirty_panel.index.get_level_values("date").dayofweek
    selected = dirty_panel.close.where(weekdays != 0)
    expected = selected.groupby(level="date").transform("sum", min_count=1)
    assert expected.isna().any()
    assert_allclose(run_dirty(dirty_engine, close.filter(factors.WEEKDAY != 0).xs_sum()), expected)


@pytest.mark.parametrize("helper", ["mean", "median", "max", "min", "std", "xs_sum"])
def test_empty_universe_is_nan(engine, helper):
    never = factors.WEEKDAY < 0
    aapl, msft = run_factor(engine, getattr(close, helper)(mask=never), check_bias=False)
    assert_array_equal(aapl, [nan] * 9)
    assert_array_equal(msft, [nan] * 8)


# ------------------------------------ zscore ------------------------------------


def test_zscore(engine):
    def zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0), axis=0)

    expected = ref.expected_xs(zscore)
    assert_factor(engine, close.zscore(), *expected)
    # only two assets, so the zscore is 1 or -1, nan if only one asset (0 / 0)
    assert_almost_equal(expected[1], [-1.0] * 8)
    assert np.isnan(expected[0][SOLO])


def test_zscore_zero_weight(engine):
    """weight=0 means mean = 0"""
    expected = ref.expected_xs(lambda df: df.div(df.std(axis=1, ddof=0), axis=0))
    assert_factor(engine, close.zscore(weight=0), *expected)
    assert expected[0][SOLO] == np.inf


def test_weighted_zscore(dirty_engine, dirty_panel):
    closes = dirty_panel.close
    std = closes.groupby(level="date").transform("std", ddof=0)
    expected = (closes - xs_weighted_mean(closes, dirty_panel.volume)) / std
    result = run_dirty(dirty_engine, close.zscore(weight=volume))
    assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_weighted_zscore_of_rows(device):
    data = torch.tensor([[4, nan, 8], [nan, nan, nan]], device=device)
    weight = torch.tensor([[1.0, 8, 1], [1, 1, 1]], device=device)
    result = factors.factor.ZScoreFactor(close).compute(data, weight)
    assert_array_equal(result.cpu().numpy(), [[-1, nan, 1], [nan, nan, nan]])


def test_zscore_shift_with_filter(engine):
    """mask bug of zscore and shift"""
    engine.set_filter(factors.OHLCV.open.top(2))
    expected_aapl = [nan, 1, 1, 1, 1, 1, 1, nan, 1, 1]
    expected_msft = [nan] + [-1.0] * 8
    assert_factor(
        engine, factors.OHLCV.open.zscore().shift(1), expected_aapl, expected_msft, delay=False
    )


def test_zscore_of_ts_max_on_listing_day(dirty_engine, dirty_panel):
    """A new listing has no returns in its window, that must not wipe the other assets."""
    returns = dirty_panel.clean_close.groupby(level="asset").pct_change(fill_method=None)
    ts_max = returns.groupby(level="asset").transform(lambda x: x.rolling(5, min_periods=1).max())
    expected = ts_max.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    factor = factors.Returns().ts_max(5)
    dirty_engine.add(factor, "ts_max")
    dirty_engine.add(factor.zscore(), "zscore")
    start, listed_on = DIRTY_DATES[10], DIRTY_DATES[LISTED_ON]
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    assert np.isnan(result.ts_max.loc[(listed_on, "LISTED")])
    assert result.zscore.loc[listed_on].notna().sum() == 5
    assert_allclose(result.ts_max, ts_max.loc[start:], rtol=1e-6)
    assert_allclose(result.zscore, expected.loc[start:], rtol=1e-6)


def test_bottom_skips_asset_without_returns(dirty_engine, dirty_panel):
    returns = dirty_panel.clean_close.groupby(level="asset").pct_change(fill_method=None)
    expected = returns.groupby(level="asset").transform(
        lambda x: x.rolling(3, min_periods=1).std(0)
    )

    factor = factors.STDDEV(3, inputs=[factors.Returns()])
    dirty_engine.add(factor, "std")
    dirty_engine.add(factor.bottom(1), "bottom")
    start, listed_on = DIRTY_DATES[10], DIRTY_DATES[LISTED_ON]
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    assert_allclose(result["std"], expected.loc[start:], rtol=1e-6, atol=1e-9)
    assert np.isnan(result["std"].loc[(listed_on, "LISTED")])
    assert not result.bottom.loc[(listed_on, "LISTED")]


# ------------------------------------ demean ------------------------------------


def demeaned(df):
    return df.sub(df.mean(axis=1), axis=0)


@pytest.mark.filterwarnings("ignore:group_dict")
def test_demean_group_dict(engine):
    expected = ref.expected_xs(demeaned)
    assert_factor(engine, close.demean(groupby={"AAPL": 1, "MSFT": 1}), *expected)
    # different groups
    assert_factor(engine, close.demean(groupby={"AAPL": 1, "MSFT": 2}), [0] * 9, [0] * 8)


def test_demean(engine):
    assert_factor(engine, close.demean(), *ref.expected_xs(demeaned))


def demean_by_asset(factor):
    demean = factor.demean()
    demean.groupby = "asset"
    return demean


def test_demean_leaves_its_input_alone(dirty_engine, dirty_panel):
    # float64 in the same group: `.to(torch.float64)` returns the tensor of the input itself
    closes = dirty_panel.close
    demedian = close.demedian()
    dirty_engine.add(demean_by_asset(close), "close_demeaned")
    dirty_engine.add(demedian.demean(), "demedian_demeaned")
    dirty_engine.add(close, "close")
    dirty_engine.add(demedian, "demedian")
    with pytest.warns(RuntimeWarning, match="delay_factor is set to False"):
        result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1], delay_factor=False)
    assert_allclose(result.close, closes)
    assert_allclose(result.demedian, closes - closes.groupby(level="date").transform("median"))


def test_demean_with_beta(dirty_engine, dirty_panel):
    closes = dirty_panel.close
    mean = closes.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, close.mean().demean(beta=close))
    assert_allclose(result, mean - closes * mean)


def test_filter_demean(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float)
    expected = selected - selected.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, (close > 100).demean())
    assert_allclose(result, expected, atol=1e-6)
    assert result.dtype == "float32"


def test_filter_demean_with_mask(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float).where(dirty_panel.volume > 1500)
    expected = selected - selected.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, (close > 100).demean(mask=volume > 1500))
    assert_allclose(result, expected, atol=1e-6)


def test_demedian(engine):
    expected = ref.expected_xs(lambda df: df.sub(df.median(axis=1), axis=0))
    assert_factor(engine, close.demedian(), *expected)


# ------------------------------------ quantile ------------------------------------


def test_quantile_classifier_ties():
    f = factors.QuantileClassifier()
    f.bins = 5
    result = f.compute(torch.tensor([[1, 1, nan, 1.01, 1.01, 2], [3, 4, 5, 1.01, nan, 1.01]]))
    assert_array_equal(result, [[0, 0, nan, 2, 2, 4], [2, 3, 4, 0, nan, 0]])

    result = f.compute(torch.tensor([[-1, 1, nan, 1.01, 1.02, 2], [3, -4, 5, 1.01, nan, 1.01]]))
    assert_array_equal(result, [[0, 1, nan, 2, 3, 4], [3, 0, 4, 1, nan, 1]])


def test_quantile_classifier_same_as_pandas():
    data = np.random.default_rng(0).standard_normal((2, 52)).round(3)
    f = factors.QuantileClassifier()
    f.bins = 5
    result = f.compute(torch.tensor(data)).numpy()
    for i in range(2):
        assert_array_equal(result[i], pd.qcut(data[i], 5, labels=False))


def test_quantile_classifier_real_data():
    # factor values of a real backtest, with some duplicates
    data = pd.read_csv(DATA_DIR / "quantile_values.csv").value.values
    f = factors.QuantileClassifier()
    f.bins = 5
    result = f.compute(torch.tensor(data[None, :])).numpy()
    assert_array_equal(result[0], pd.qcut(data, 5).codes)


def test_rolling_quantile():
    data = np.random.default_rng(1).standard_normal((2, 3))
    result = factors.RollingQuantile(3).compute(parallel.Rolling(torch.tensor(data), 3), 5)
    # the quantile of the last value in the rolling window
    windows = [data[0, :1], data[0, :2], data[0, :3]]
    expected = [
        nan,
        pd.qcut(windows[1], 5, labels=False)[-1],
        pd.qcut(windows[2], 5, labels=False)[-1],
    ]
    assert_array_equal(result[0].numpy(), expected)


def test_rolling_quantile_one_bar_chunks(monkeypatch, device):
    win, bars = 6, 12
    data = np.random.default_rng(2).standard_normal((3, bars))
    expected = np.full(data.shape, nan)
    for asset, row in enumerate(data):
        for bar in range(1, bars):
            window = row[max(0, bar - win + 1) : bar + 1]
            expected[asset, bar] = pd.qcut(window, 5, labels=False)[-1]

    # pretend memory usage is huge, so every bar becomes a chunk
    monkeypatch.setattr(parallel.Rolling, "_split_multi", 1e12)
    rolling = parallel.Rolling(torch.tensor(data, device=device), win)
    assert len(rolling.split) == bars
    result = factors.RollingQuantile(win).compute(rolling, 5)
    assert_array_equal(result.cpu().numpy(), expected)


def test_quantile_with_filter(engine):
    """quantile of factor got from engine._factors nan bug"""
    engine.set_filter(close.top(2))
    engine.add(close.zscore(), "pre")
    factor = engine.get_factor("pre")
    expected_aapl = [4.0] * 9
    expected_aapl[SOLO] = nan
    with pytest.warns(RuntimeWarning, match="mask parameter not set"):
        quantile = factor.quantile()
    assert_factor(engine, quantile, expected_aapl, [0.0] * 8)


# ------------------------------------ clamp ------------------------------------


def loaded_weekdays():
    return np.array([2.0, 3, 4, 0, 1, 2, 3, 4, 0, 1])


def test_mad_clamp_groupby_asset(engine):
    """groupby='asset' clamps by the median absolute deviation of each asset's loaded data."""
    weekdays = loaded_weekdays()
    expected = ref.mad_clamp(weekdays, 1)
    assert_array_equal(expected, [2, 3, 3, 1, 1, 2, 3, 3, 1, 1])
    assert_factor(
        engine,
        factors.WEEKDAY.mad_clamp(1, groupby="asset"),
        expected,
        ref.mad_clamp(delete(weekdays, SOLO), 1),
    )


@pytest.mark.parametrize("mean", [False, True])
def test_mad_clamp_close(engine, mean):
    # MAD of the closes from 2019-01-02 to 2019-01-15 (lookahead biased), then delayed
    expected = [ref.mad_clamp(ref.loaded_closes(a), 2, mean)[:-1] for a in ref.ASSETS]
    assert_factor(
        engine, close.mad_clamp(2, groupby="asset", mean=mean), *expected, check_bias=False
    )


def test_mad_clamp_nans(device):
    x = torch.tensor(
        [
            [-999, nan, nan, 2, nan, 3, nan, nan, 999, nan],
            [1, nan, nan, 2, nan, 3, nan, nan, 4, nan],
        ],
        dtype=torch.float64,
        device=device,
    )
    factor = factors.MADClampFactor()
    factor.z = 1.5
    result = compute_with_mask(factor, x, None).cpu().numpy()
    expected = np.array([ref.mad_clamp(row, 1.5) for row in x.cpu().numpy()])
    assert_almost_equal(result, expected)
    # median 2.5, MAD 498.5
    assert_almost_equal(expected[0, [0, -2]], [2.5 - 1.5 * 498.5, 2.5 + 1.5 * 498.5])


def test_mad_clamp_fill(engine):
    # values out of `center +- z * MAD` are filled, inclusive
    weekdays = loaded_weekdays()
    center, mad = np.median(weekdays), np.median(np.abs(weekdays - np.median(weekdays)))
    expected = np.where(np.abs(weekdays - center) >= mad, -1, weekdays)
    assert_array_equal(expected, [2, -1, -1, -1, -1, 2, -1, -1, -1, -1])
    weekdays_msft = delete(weekdays, SOLO)
    center_msft = np.median(weekdays_msft)
    mad_msft = np.median(np.abs(weekdays_msft - center_msft))
    expected_msft = np.where(np.abs(weekdays_msft - center_msft) >= mad_msft, -1, weekdays_msft)
    assert_factor(
        engine, factors.WEEKDAY.mad_clamp(1, groupby="asset", fill=-1.0), expected, expected_msft
    )


def test_mad_clamp_fill_center():
    x = torch.tensor([[1.0, 2, 3, 100], [10, 20, 30, 1000]], dtype=torch.float64)
    factor = factors.MADClampFactor()
    factor.z = 1
    result = compute_with_mask(factor, x, "center")
    # center 2.5 MAD 1, center 25 MAD 10
    assert_array_equal(result.numpy(), [[2.5, 2, 3, 2.5], [25, 20, 30, 25]])


@pytest.mark.parametrize("z", [0.2, 0.01, 0.001])
def test_winsorizing_groupby_asset(engine, z):
    weekdays = loaded_weekdays()
    factor = factors.WEEKDAY.winsorizing(z)
    factor.groupby = "asset"
    assert_factor(
        engine,
        factor,
        scipy.stats.mstats.winsorize(weekdays, [z, z]),
        scipy.stats.mstats.winsorize(delete(weekdays, SOLO), [z, z]),
    )


def test_winsorizing_close(engine):
    factor = close.winsorizing(0.2)
    factor.groupby = "asset"
    expected = [
        scipy.stats.mstats.winsorize(ref.loaded_closes(a), [0.2, 0.2])[:-1] for a in ref.ASSETS
    ]
    assert_factor(engine, factor, *expected, check_bias=False)


def test_winsorizing_few_values():
    """inf bug, few values in a row should not be clamped to inf"""
    data = torch.tensor(
        [[nan] * 7 + [5, 2, 1] + [nan] * 6, [nan] * 3 + [6, nan, nan, nan, 3, 2] + [nan] * 7]
    ).repeat(2, 1)
    factor = factors.WinsorizingFactor()
    factor.z = 0.2
    result = compute_with_mask(factor, data)
    assert_almost_equal(result.numpy(), data.numpy())
    assert not torch.isinf(result).any()


# ------------------------------------ padding ------------------------------------
# An input in the same group is not regrouped, so its padding slots hold whatever it left there.
# `fill_na` fills them, and a rolling factor goes on after the last bar of a delisted asset.


def by_asset(factor):
    factor.groupby = "asset"
    return factor


def weights_of(x):
    demeaned_ = x - x.mean()
    return demeaned_ / demeaned_.abs().sum()


@pytest.mark.parametrize(
    "build, reference",
    [
        (lambda f: f.mean(groupby="asset"), lambda x: x.mean()),
        (lambda f: f.std(groupby="asset"), lambda x: x.std(ddof=0)),
        (lambda f: f.max(groupby="asset"), lambda x: x.max()),
        (lambda f: f.min(groupby="asset"), lambda x: x.min()),
        (lambda f: f.xs_sum(groupby="asset"), lambda x: x.sum()),
        (lambda f: f.median(groupby="asset"), lambda x: x.median()),
        (lambda f: f.demedian(groupby="asset"), lambda x: x - x.median()),
        (lambda f: by_asset(f.demean()), lambda x: x - x.mean()),
        (lambda f: f.zscore(groupby="asset"), lambda x: (x - x.mean()) / x.std(ddof=0)),
        (lambda f: by_asset(f.rank()), lambda x: x.rank()),
        (lambda f: by_asset(f.to_weight()), weights_of),
        (lambda f: f.mad_clamp(1, groupby="asset"), lambda x: ref.mad_clamp(x, 1)),
        (lambda f: f.mad_clamp(1, groupby="asset", mean=True), lambda x: ref.mad_clamp(x, 1, True)),
    ],
    ids=[
        "mean",
        "std",
        "max",
        "min",
        "xs_sum",
        "median",
        "demedian",
        "demean",
        "zscore",
        "rank",
        "to_weight",
        "mad_clamp",
        "mad_clamp_mean",
    ],
)
@pytest.mark.parametrize(
    "fill, values",
    [
        (lambda: close.fill_na(0), lambda closes: closes.fillna(0)),
        (lambda: close.fill_na(ffill=True), lambda closes: closes.groupby(level="asset").ffill()),
    ],
    ids=["zero", "ffill"],
)
def test_same_group_input_is_read_without_padding(
    dirty_engine, dirty_panel, build, reference, fill, values
):
    expected = values(dirty_panel.close).groupby(level="asset").transform(reference)
    result = run_dirty(dirty_engine, build(fill()))
    assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_same_group_mask_is_read_without_padding(dirty_engine, dirty_panel):
    filled = close.fill_na(0)
    mask = filled >= 0  # True on every row, and in the padding too
    dirty_engine.add(filled.median(groupby="asset", mask=mask), "median")
    dirty_engine.add(by_asset(filled.winsorizing(0.2, mask=mask)), "winsorizing")
    dirty_engine.add(filled.quantile(3, mask=mask, groupby="asset"), "quantile")
    with pytest.warns(RuntimeWarning, match="delay_factor is set to False"):
        result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1], delay_factor=False)

    by_asset_ = dirty_panel.close.fillna(0).groupby(level="asset")
    assert_allclose(result["median"], by_asset_.transform("median"))
    assert_allclose(
        result["winsorizing"],
        by_asset_.transform(lambda x: np.asarray(scipy.stats.mstats.winsorize(x, [0.2, 0.2]))),
    )
    assert_array_equal(result["quantile"], by_asset_.transform(pd.qcut, 3, labels=False))


def test_cross_section_of_cross_section_is_read_without_padding(dirty_engine, dirty_panel):
    # a mean is expanded to every slot of the date, the padding too
    mean = dirty_panel.close.groupby(level="date").transform("mean")
    expected = mean.groupby(level="date").transform("sum")
    assert_allclose(run_dirty(dirty_engine, close.mean().xs_sum()), expected)


def test_filter_in_same_group_is_read_without_padding(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float)
    expected = selected - selected.groupby(level="asset").transform("mean")
    assert_allclose(run_dirty(dirty_engine, demean_by_asset(close > 100)), expected, atol=1e-6)


def test_multiple_outputs_are_read_without_padding(dirty_engine, dirty_panel):
    closes = dirty_panel.close
    volumes = dirty_panel.volume
    means = pd.DataFrame(
        {
            "close": closes.groupby(level="date").transform("mean"),
            "volume": volumes.groupby(level="date").transform("mean"),
        }
    )
    expected = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1, ddof=0), axis=0)
    factor = factors.FactorWiseZScore(inputs=[close, volume])[0]
    assert_allclose(run_dirty(dirty_engine, factor), expected["close"], rtol=1e-5)


def test_iqr_normality(engine):
    # interquartile range / std - 1.34898, if only two values, iqr / std = 2
    expected_aapl = [abs(2 - 1.34898)] * 9
    expected_aapl[SOLO] = nan
    factor = factors.IQRNormalityFactor(inputs=[factors.Returns()])
    assert_factor(engine, factor, expected_aapl, [abs(2 - 1.34898)] * 8)


def test_clamp(engine):
    expected = ref.expected(lambda df: df.close.clip(150, 155))
    assert_factor(engine, close.clamp(150, 155), *expected)
    expected = ref.expected(lambda df: df.close.mask((df.close < 150) | (df.close > 155), 0))
    assert_factor(engine, close.clamp(150, 155, fill=0), *expected)


def test_to_weight(engine):
    def to_weight(df):
        demeaned_ = df.sub(df.mean(axis=1), axis=0)
        return demeaned_.div(demeaned_.abs().sum(axis=1), axis=0)

    assert_factor(engine, close.to_weight(), *ref.expected_xs(to_weight))
    assert_factor(
        engine,
        close.to_weight(demean=False),
        *ref.expected_xs(lambda df: df.div(df.abs().sum(axis=1), axis=0)),
    )


def test_demean_of_to_weight(dirty_engine, dirty_panel):
    closes = dirty_panel.close
    by_date = closes.groupby(level="date")
    weight = closes / by_date.transform(lambda x: x.abs().sum())
    expected = weight - weight.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, close.to_weight(demean=False).demean())
    assert_allclose(result, expected, rtol=1e-6, atol=1e-9)
