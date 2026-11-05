import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import spectre
from spectre import factors, parallel
from tests.conftest import (
    DATA_DIR,
    DIRTY_ADJUSTMENTS,
    DIRTY_DATES,
    DIRTY_OHLCV,
    DIRTY_SECTORS,
    LISTED_ON,
)
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, delete, run_dirty, run_factor

nan = np.nan
close = factors.OHLCV.close
open_price = factors.OHLCV.open
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
    expected = ref.expected_xs(lambda df: df.rank(axis=1, pct=True))
    assert_factor(engine, close.rank(normalize=True), *expected)


def test_rank_normalize_of_ties_is_a_share_of_the_date(dirty_engine, dirty_panel):
    """Not a share of the highest rank: the first of [1, 2, 2, 2] is at .25, not at .33."""
    rounded = (dirty_panel.close / 50).round()
    expected = rounded.groupby(level="date").rank(pct=True)
    result = run_dirty(dirty_engine, (close / 50).round(0).rank(normalize=True))
    assert_allclose(result, expected, rtol=1e-6)
    assert result.groupby(level="date").max().min() < 1


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


def spread_of(df):
    """The std of each date, one asset has no spread to measure."""
    return df.std(axis=1, ddof=0).where(df.count(axis=1) > 1)


def test_std(engine):
    expected = ref.expected_xs(lambda df: pd.DataFrame({a: spread_of(df) for a in df}))
    assert_factor(engine, close.std(), *expected)
    assert np.isnan(expected[0][SOLO])


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
    # closes from 2018-12-28 to 2019-01-15 are loaded, so returns are from 2018-12-31
    returns = {
        a: ref.returns(ref.daily_prices(a).close).loc["2018-12-31" : ref.END].round(3)
        for a in ref.ASSETS
    }
    aapl, msft = run_factor(engine, factor, check_bias=False)
    assert_almost_equal(aapl, [np.median(returns["AAPL"])] * 9)
    assert_almost_equal(msft, [np.median(returns["MSFT"])] * 8)
    assert_almost_equal([np.median(returns["AAPL"]), np.median(returns["MSFT"])], [0.0, 0.0045])


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
    expected = ref.expected_xs(lambda df: df.div(spread_of(df), axis=0))
    assert_factor(engine, close.zscore(weight=0), *expected)
    assert np.isnan(expected[0][SOLO])


def test_zscore_zero_weight_of_equal_values(device):
    data = torch.full((2, 3), 33.3, device=device)
    result = factors.factor.ZScoreFactor(close).compute(data, 0)
    assert_array_equal(result.cpu().numpy(), np.full((2, 3), nan))


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


def test_weighted_zscore_with_nan_weight(dirty_engine, dirty_panel):
    """A row with data but no weight gets a score, it's just not a sample of the mean and std."""
    closes = dirty_panel.close
    weight = dirty_panel.volume.where(dirty_panel.volume > 1500)
    assert (closes.notna() & weight.isna()).any()
    samples = closes.where(weight.notna())
    std = samples.groupby(level="date").transform("std", ddof=0)
    expected = (closes - xs_weighted_mean(closes, weight)) / std.where(std > 0)
    result = run_dirty(dirty_engine, close.zscore(weight=volume.filter(volume > 1500)))
    assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_weighted_zscore_with_nan_weight_of_rows(device):
    data = torch.tensor([[1, 2, 3, 6.0]], device=device)
    weight = torch.tensor([[1, 1, nan, 2.0]], device=device)
    result = factors.factor.ZScoreFactor(close).compute(data, weight)
    # the samples are 1, 2 and 6, their weighted mean is (1 + 2 + 12) / 4
    expected = (np.array([1, 2, 3, 6]) - 3.75) / np.std([1, 2, 6])
    assert_allclose(result.cpu().numpy()[0], expected, rtol=1e-6)


@pytest.mark.parametrize(
    "data, weight",
    [
        ([[1, 2, 3]], [[1, nan, nan]]),  # one sample
        ([[5, 5, 7]], [[1, 1, nan]]),  # equal samples, and a row out of them
        ([[100.1, 100.1]], [[1, 1]]),  # the mean is float32, so it's not exactly the value
    ],
    ids=["one_sample", "equal_samples", "rounded_mean"],
)
def test_weighted_zscore_without_spread(device, data, weight):
    """Without spread there is no score, not an inf for each value that is off the mean."""
    data = torch.tensor(data, dtype=torch.float64, device=device)
    weight = torch.tensor(weight, dtype=torch.float64, device=device)
    result = factors.factor.ZScoreFactor(close).compute(data, weight)
    assert torch.isnan(result).all()


def constant_rows(rng, dtype, device):
    """Rows of different lengths, the values of a row are equal."""
    data = np.full((200, 40), nan)
    for row, value in enumerate(rng.uniform(0.01, 500, 200)):
        data[row, : 3 + row % 37] = value
    return torch.tensor(data, dtype=dtype, device=device)


def smallest_spread(dtype, device):
    """Equal values and one that is a step of the float type above them."""
    value = torch.tensor(123.456, dtype=dtype)
    next_value = torch.nextafter(value, torch.tensor(np.inf, dtype=dtype))
    return torch.stack([value, value, value, next_value]).view(1, 4).to(device)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zscore_of_constant_rows(device, dtype, weighted):
    """The rounding of the mean must not give scores to equal values."""
    rng = np.random.default_rng(0)
    data = constant_rows(rng, dtype, device)
    weight = torch.tensor(rng.uniform(1, 1000, data.shape), dtype=dtype, device=device)
    result = factors.factor.ZScoreFactor(close).compute(data, weight if weighted else None)
    assert torch.isnan(result).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zscore_of_the_smallest_spread(device, dtype):
    """Rounding must not hide a spread either, one step of the float type is still a spread."""
    result = factors.factor.ZScoreFactor(close).compute(smallest_spread(dtype, device))
    expected = np.array([-1, -1, -1, 3]) / 3**0.5
    assert_allclose(result.cpu().numpy()[0], expected, rtol=1e-6)


@pytest.mark.parametrize("dtype, atol", [(torch.float32, 2e-5), (torch.float64, 1e-12)])
def test_zscore_of_random_rows(device, dtype, atol):
    rng = np.random.default_rng(0)
    data = rng.normal(100, rng.uniform(0.01, 30, (500, 1)), (500, 60))
    data[rng.random(data.shape) < 0.2] = nan
    data = torch.tensor(data, dtype=dtype, device=device)
    # the same values, without the rounding of the float type
    exact = data.cpu().numpy().astype(np.float64)
    mean, std = np.nanmean(exact, axis=1, keepdims=True), np.nanstd(exact, axis=1, keepdims=True)
    result = factors.factor.ZScoreFactor(close).compute(data)
    assert_allclose(result.cpu().numpy(), (exact - mean) / std, rtol=0, atol=atol)


def typed_engine(dirty_panel, dtype, device):
    """The dirty panel with its prices in a float type, and an engine on it."""
    panel = dirty_panel.astype(dict.fromkeys(DIRTY_OHLCV, dtype))
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS))
    if device == "cuda":
        engine.to_cuda()
    return panel, engine


@pytest.mark.parametrize("groupby", ["date", "asset"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zscore_agrees_with_pandas(dirty_panel, device, dtype, groupby):
    panel, engine = typed_engine(dirty_panel, dtype, device)
    closes = panel.close.astype("float64")
    expected = closes.groupby(level=groupby).transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    result = run_dirty(engine, close.zscore(groupby=groupby))
    assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_weighted_zscore_agrees_with_pandas(dirty_panel, device, dtype):
    panel, engine = typed_engine(dirty_panel, dtype, device)
    closes, volumes = panel.close.astype("float64"), panel.volume.astype("float64")
    std = closes.groupby(level="date").transform("std", ddof=0)
    expected = (closes - xs_weighted_mean(closes, volumes)) / std
    result = run_dirty(engine, close.zscore(weight=volume))
    assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert result.dtype == dtype


def test_zscore_shift_with_filter(engine):
    """mask bug of zscore and shift"""
    engine.set_filter(factors.OHLCV.open.top(2))
    # the bar before the range is loaded, so the first row has a value too
    expected_aapl = [1, 1, 1, 1, 1, 1, 1, nan, 1, 1]
    expected_msft = [-1.0] * 9
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
        lambda x: x.rolling(3, min_periods=2).std(0)
    )

    factor = factors.STDDEV(3, inputs=[factors.Returns()])
    dirty_engine.add(factor, "std")
    dirty_engine.add(factor.bottom(1), "bottom")
    start, listed_on = DIRTY_DATES[10], DIRTY_DATES[LISTED_ON]
    result = dirty_engine.run(start, DIRTY_DATES[-1], delay_factor=False)

    assert_allclose(result["std"], expected.loc[start:], rtol=1e-6, atol=1e-9)
    assert np.isnan(result["std"].loc[(listed_on, "LISTED")])
    assert not result.bottom.loc[(listed_on, "LISTED")]
    # its first return is one sample, that is not the lowest risk of the date
    second_bar = DIRTY_DATES[LISTED_ON + 1]
    assert np.isnan(result["std"].loc[(second_bar, "LISTED")])
    assert not result.bottom.loc[(second_bar, "LISTED")]


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


def delayed_sector_demean(prices, sector):
    """Demean by sector on each date, then delay by a bar of each asset, by pandas."""
    dates = prices.index.get_level_values("date")
    result = prices - prices.groupby([dates, sector]).transform("mean")
    return result.groupby(level="asset").shift(1)


def test_demean_by_sector_is_cross_sectional(dirty_engine, dirty_panel):
    """The mean of a sector over all dates would read the prices of the future."""
    start, end = DIRTY_DATES[5], DIRTY_DATES[-1]
    dirty_engine.add(close.demean(groupby="sector"), "test")
    result = dirty_engine.run(start, end).test
    expected = delayed_sector_demean(dirty_panel.close, dirty_panel.sector)
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)
    assert dirty_engine.test_lookahead_bias(start, end) == "No assertion raised."


def test_demean_by_sector_with_a_mask(dirty_engine, dirty_panel):
    start, end = DIRTY_DATES[5], DIRTY_DATES[-1]
    dirty_engine.add(close.demean(groupby="sector", mask=close > 100), "test")
    result = dirty_engine.run(start, end).test
    masked = dirty_panel.close.where(dirty_panel.close > 100)
    expected = delayed_sector_demean(masked, dirty_panel.sector)
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)


def test_demean_by_sector_of_one_asset(dirty_engine):
    dirty_engine.set_filter(factors.StaticAssets({"FULL"}))
    dirty_engine.add(close.demean(groupby="sector"), "test")
    result = dirty_engine.run(DIRTY_DATES[5], DIRTY_DATES[-1]).test
    assert_almost_equal(result.values, np.zeros(len(result)))


@pytest.mark.parametrize(
    "groupby, expected",
    [
        ("sector", "date,sector"),
        ("sector,gics", "date,sector,gics"),
        ("candidate", "date,candidate"),
        ("asset", "asset"),
        ("date", "date"),
        ("date,sector", "date,sector"),
        ("sector,date", "sector,date"),
    ],
)
def test_demean_groupby(groupby, expected):
    assert close.demean(groupby=groupby).groupby == expected


COLUMN_GROUP_HELPERS = {
    "zscore": lambda f, g: f.zscore(groupby=g),
    "demean": lambda f, g: f.demean(groupby=g),
    "demedian": lambda f, g: f.demedian(groupby=g),
    "mean": lambda f, g: f.mean(groupby=g),
    "median": lambda f, g: f.median(groupby=g),
    "max": lambda f, g: f.max(groupby=g),
    "min": lambda f, g: f.min(groupby=g),
    "std": lambda f, g: f.std(groupby=g),
    "xs_sum": lambda f, g: f.xs_sum(groupby=g),
    "quantile": lambda f, g: f.quantile(2, mask=f > 0, groupby=g),
    "mad_clamp": lambda f, g: f.mad_clamp(3, groupby=g),
}
each_group_helper = pytest.mark.parametrize("helper", COLUMN_GROUP_HELPERS)


@each_group_helper
@pytest.mark.parametrize(
    "groupby, expected",
    [("sector", "date,sector"), ("date,sector", "date,sector"), ("asset", "asset")],
)
def test_group_of_helpers(helper, groupby, expected):
    assert COLUMN_GROUP_HELPERS[helper](close, groupby).groupby == expected


@each_group_helper
def test_helpers_by_a_column_are_cross_sections(dirty_engine, helper):
    """Pooling all dates would read the future, and take out nothing of the date."""
    start, end = DIRTY_DATES[5], DIRTY_DATES[-1]
    dirty_engine.add(COLUMN_GROUP_HELPERS[helper](close, "sector"), "test")
    assert dirty_engine.test_lookahead_bias(start, end) == "No assertion raised."


def test_zscore_by_sector_has_no_mean_in_a_sector(dirty_engine, dirty_panel):
    result = run_dirty(dirty_engine, close.zscore(groupby="sector"))
    dates = dirty_panel.index.get_level_values("date")
    groups = dirty_panel.close.groupby([dates, dirty_panel.sector])
    expected = (dirty_panel.close - groups.transform("mean")) / groups.transform("std", ddof=0)
    assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    assert result.groupby([dates, dirty_panel.sector]).mean().abs().max() < 1e-5


@pytest.mark.parametrize(
    "make",
    [
        lambda: factors.WinsorizingFactor(inputs=(close,)),
        lambda: factors.MADClampFactor(inputs=(close, None)),
        lambda: factors.ZScoreFactor(close),
    ],
    ids=["winsorizing", "mad_clamp", "zscore"],
)
def test_factors_of_helpers_are_cross_sections_by_default(make):
    assert make().groupby == "date"


def test_zscore_factor_adds_the_date_to_a_column():
    assert factors.ZScoreFactor(close, groupby="sector").groupby == "date,sector"


# more than 21 dates, a key of 8 digits doesn't fit in int32 after that
GROUP_DATES = pd.date_range("2020-01-01", periods=60, freq="B", tz="UTC")
GROUP_COLUMNS = ["gics", "decimal", "text", "category"]


def group_panel():
    """6 assets in 3 groups, the same groups in a column of each kind."""
    rng = np.random.default_rng(1)
    assets = list("ABCDEF")
    df = pd.DataFrame(
        rng.normal(100, 10, (len(GROUP_DATES), len(assets))), index=GROUP_DATES, columns=assets
    )
    df = df.stack().rename("close").rename_axis(["date", "asset"]).to_frame()
    df["open"] = df.close + 1
    df["high"], df["low"], df["volume"] = df.close, df.close, 1.0
    names = df.index.get_level_values("asset")
    group = names.map(dict(zip(assets, [0, 0, 1, 1, 2, 2])))
    df["gics"] = np.array([45102010, 45102020, 20106020])[group]
    df["decimal"] = np.array([1.25, 1.5, 1.75])[group]
    df["text"] = np.array(["tech", "bank", "food"])[group]
    df["category"] = pd.Categorical(df["text"], categories=["unused", "tech", "bank", "food"])
    # a different value on each asset, no space is left for the other columns of a key
    df["huge"] = 10**15 * (names.map(assets.index) + 1)
    return df


def group_engine(device):
    panel = group_panel()
    ohlcv = ("open", "high", "low", "close", "volume")
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, ohlcv=ohlcv))
    if device == "cuda":
        engine.to_cuda()
    return panel, engine


@pytest.mark.parametrize("column", GROUP_COLUMNS)
def test_group_by_a_column_of_any_type(device, column):
    """Set on a factor, a column is all the rows of the group, of any date."""
    panel, engine = group_engine(device)
    factor = open_price.mean()
    factor.groupby = column
    engine.add(factor, "test")
    result = engine.run(GROUP_DATES[0], GROUP_DATES[-1]).test
    expected = panel.open.groupby(panel[column], observed=True).transform("mean")
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)


@pytest.mark.parametrize("column", GROUP_COLUMNS)
def test_helper_by_a_column_of_any_type(device, column):
    panel, engine = group_engine(device)
    engine.add(open_price.mean(groupby=column), "test")
    result = engine.run(GROUP_DATES[0], GROUP_DATES[-1]).test
    dates = panel.index.get_level_values("date")
    expected = panel.open.groupby([dates, panel[column]], observed=True).transform("mean")
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)


@pytest.mark.parametrize("column", GROUP_COLUMNS)
@pytest.mark.parametrize("order", ["date,{}", "{},date"])
def test_group_by_date_and_a_column_of_any_type(device, column, order):
    panel, engine = group_engine(device)
    engine.add(open_price.mean(groupby=order.format(column)), "test")
    result = engine.run(GROUP_DATES[0], GROUP_DATES[-1]).test
    dates = panel.index.get_level_values("date")
    expected = panel.open.groupby([dates, panel[column]], observed=True).transform("mean")
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)


def test_group_keys_do_not_overflow(device):
    panel, engine = group_engine(device)
    engine.add(open_price.mean(groupby="date,huge,gics,text"), "test")
    result = engine.run(GROUP_DATES[0], GROUP_DATES[-1]).test
    # every asset is a group of its own on each date
    assert_allclose(result.values, panel.open.loc[result.index].values, rtol=1e-6)


def aligned_to_all_dates(panel):
    """What align_by_time gives the factors: a row for each asset on each date."""
    all_rows = pd.MultiIndex.from_product(panel.index.levels, names=panel.index.names)
    return panel.reindex(all_rows)


def demean_by_group(values, groups):
    """`groups` is the group of each asset, it doesn't depend on the row having data."""
    dates, assets = values.index.get_level_values("date"), values.index.get_level_values("asset")
    return values - values.groupby([dates, assets.map(groups)]).transform("mean")


def test_demean_by_sector_with_align_by_time(dirty_engine, dirty_panel):
    """A halt, a listing and a delisting all add rows, which have no sector."""
    dirty_engine.align_by_time = True
    dirty_engine.add(close.demean(groupby="sector"), "test")
    result = dirty_engine.run(DIRTY_DATES[5], DIRTY_DATES[-1]).test
    prices = aligned_to_all_dates(dirty_panel).close
    expected = demean_by_group(prices, DIRTY_SECTORS).groupby(level="asset").shift(1)
    assert result.isna().any()
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)


def test_demean_by_sector_of_rows_filled_after_align_by_time(dirty_engine, dirty_panel):
    """A filled row has a value, so it matters that it stays in the sector of its asset."""
    dirty_engine.align_by_time = True
    dirty_engine.add(close.fill_na(ffill=True).demean(groupby="sector"), "test")
    result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1]).test
    filled = aligned_to_all_dates(dirty_panel).close.groupby(level="asset").ffill()
    expected = demean_by_group(filled, DIRTY_SECTORS).groupby(level="asset").shift(1)
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6)


@pytest.mark.parametrize("column", GROUP_COLUMNS)
def test_group_column_of_any_type_with_align_by_time(device, column):
    panel = group_panel()
    panel = panel.drop(index=[(date, "A") for date in GROUP_DATES[10:20]])
    groups = panel.groupby(level="asset", observed=True)[column].first()
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, ohlcv=DIRTY_OHLCV))
    if device == "cuda":
        engine.to_cuda()
    engine.align_by_time = True
    engine.add(open_price.demean(groupby=column), "test")
    result = engine.run(GROUP_DATES[0], GROUP_DATES[-1]).test
    expected = demean_by_group(aligned_to_all_dates(panel).open, groups.to_dict())
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("align_by_time", [False, True], ids=["as_loaded", "aligned"])
def test_a_real_nan_in_a_group_column_raises(dirty_panel, device, align_by_time):
    """It would pool the assets of that row. An added row has none, this one has data."""
    panel = dirty_panel.astype({"sector": float})
    panel.loc[(DIRTY_DATES[10], "FULL"), "sector"] = nan
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS))
    if device == "cuda":
        engine.to_cuda()
    engine.align_by_time = align_by_time
    engine.add(close.demean(groupby="sector"), "test")
    with pytest.raises(AssertionError, match="sector"):
        engine.run(DIRTY_DATES[5], DIRTY_DATES[-1])


def test_demean_by_a_text_column(device):
    panel, engine = group_engine(device)
    engine.add(open_price.demean(groupby="text"), "test")
    result = engine.run(GROUP_DATES[0], GROUP_DATES[-1]).test
    dates = panel.index.get_level_values("date")
    expected = panel.open - panel.open.groupby([dates, panel.text]).transform("mean")
    assert_allclose(result.values, expected.loc[result.index].values, rtol=1e-6, atol=1e-9)


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


def test_demean_with_a_filter_as_beta(dirty_engine, dirty_panel):
    closes = dirty_panel.close
    mean = closes.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, close.demean(beta=close > 100))
    assert_allclose(result, closes - (closes > 100) * mean, rtol=1e-6)


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
    # MAD of the closes from 2018-12-31 to 2019-01-15 (lookahead biased), then delayed
    expected = [
        ref.mad_clamp(ref.loaded_closes(a, "2018-12-31"), 2, mean)[1:-1] for a in ref.ASSETS
    ]
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


@pytest.mark.parametrize("mean", [False, True])
def test_mad_clamp_fill_constant_rows(device, mean):
    """With MAD 0 both bounds are the center, which is not an outlier."""
    x = torch.tensor([[1.0, 1, 1, 1], [8, nan, 8, 8]], dtype=torch.float64, device=device)
    factor = factors.MADClampFactor()
    factor.z = 3
    factor.m_is_mean = mean
    result = compute_with_mask(factor, x, -1.0)
    assert_array_equal(result.cpu().numpy(), x.cpu().numpy())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mad_clamp_mean_fill_constant_rows(device, dtype):
    """The mean of equal values must not round away from them, a bound would then cut them."""
    values = np.random.default_rng(0).uniform(0.01, 500, 200)
    x = np.full((200, 40), nan)
    for row, value in enumerate(values):
        x[row, : 3 + row % 37] = value
    x = torch.tensor(x, dtype=dtype, device=device)
    factor = factors.MADClampFactor()
    factor.z = 1
    factor.m_is_mean = True
    result = compute_with_mask(factor, x, -1.0)
    assert_array_equal(result.cpu().numpy(), x.cpu().numpy())


def test_mad_clamp_fill_with_mad_of_zero(device):
    x = torch.tensor([[5.0, 5, 5, 5, 7], [2, nan, 2, 0, 2]], dtype=torch.float64, device=device)
    factor = factors.MADClampFactor()
    factor.z = 3
    result = compute_with_mask(factor, x, -1.0)
    # the values that are not the center are out of it, like the clamp without `fill` does
    assert_array_equal(result.cpu().numpy(), [[5, 5, 5, 5, -1], [2, nan, 2, -1, 2]])


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
        scipy.stats.mstats.winsorize(ref.loaded_closes(a, "2018-12-31"), [0.2, 0.2])[1:-1]
        for a in ref.ASSETS
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

    grouped = dirty_panel.close.fillna(0).groupby(level="asset")
    assert_allclose(result["median"], grouped.transform("median"))
    assert_allclose(
        result["winsorizing"],
        grouped.transform(lambda x: np.asarray(scipy.stats.mstats.winsorize(x, [0.2, 0.2]))),
    )
    assert_array_equal(result["quantile"], grouped.transform(pd.qcut, 3, labels=False))


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


def iqr_normality_of(x):
    """The quartiles are the values at n // 4 from each end."""
    values = np.sort(x.dropna().to_numpy())
    quarter = len(values) // 4
    iqr = values[len(values) - quarter - 1] - values[quarter]
    return abs(iqr / values.std() - 1.34898)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_iqr_normality_agrees_with_numpy(dirty_panel, device, dtype):
    panel, engine = typed_engine(dirty_panel, dtype, device)
    closes = panel.close.astype("float64")
    expected = closes.groupby(level="date").transform(iqr_normality_of)
    result = run_dirty(engine, factors.IQRNormalityFactor(inputs=[close]))
    assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_iqr_normality_of_equal_values(dirty_engine):
    # every close is above 0.3, so each date is a single value
    factor = factors.IQRNormalityFactor(inputs=[close.clamp(0, 0.3)])
    assert run_dirty(dirty_engine, factor).isna().all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_iqr_normality_of_constant_rows(device, dtype):
    data = constant_rows(np.random.default_rng(0), dtype, device)
    assert torch.isnan(compute_with_mask(factors.IQRNormalityFactor(), data)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_iqr_normality_of_the_smallest_spread(device, dtype):
    # two values a step apart, half of each: the iqr is a step, and the std half a step
    data = smallest_spread(dtype, device)[:, [0, 1, 3, 3]]
    result = compute_with_mask(factors.IQRNormalityFactor(), data)
    assert_allclose(result.cpu().numpy(), [[2 - 1.34898] * 4], rtol=1e-6)
    # no iqr, but there is a spread, it's as far from normal as it gets
    result = compute_with_mask(factors.IQRNormalityFactor(), smallest_spread(dtype, device))
    assert_allclose(result.cpu().numpy(), [[1.34898] * 4], rtol=1e-6)


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


def test_to_weight_of_equal_values(dirty_engine):
    """With demean, no asset is above or below the others, so there are no weights to give."""
    # every close is above 0.3, so each date is a single value
    assert run_dirty(dirty_engine, close.clamp(0, 0.3).to_weight()).isna().all()


def test_to_weight_of_equal_values_without_demean(dirty_engine, dirty_panel):
    clamped = dirty_panel.close.clip(0, 0.3)
    expected = clamped / clamped.groupby(level="date").transform("sum")
    assert_allclose(run_dirty(dirty_engine, close.clamp(0, 0.3).to_weight(demean=False)), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_weight_of_constant_rows(device, dtype):
    data = constant_rows(np.random.default_rng(0), dtype, device)
    assert torch.isnan(factors.factor.ToWeightFactor().compute(data)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_weight_of_the_smallest_spread(device, dtype):
    result = factors.factor.ToWeightFactor().compute(smallest_spread(dtype, device))
    # from their mean the values are -1/4 of a step three times, and 3/4 of a step
    assert_allclose(result.cpu().numpy()[0], [-1 / 6, -1 / 6, -1 / 6, 1 / 2], rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_to_weight_agrees_with_pandas(dirty_panel, device, dtype):
    panel, engine = typed_engine(dirty_panel, dtype, device)
    closes = panel.close.astype("float64")
    demeaned_ = closes - closes.groupby(level="date").transform("mean")
    expected = demeaned_ / demeaned_.abs().groupby(level="date").transform("sum")
    result = run_dirty(engine, close.to_weight())
    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    assert result.dtype == dtype


def test_demean_of_to_weight(dirty_engine, dirty_panel):
    closes = dirty_panel.close
    by_date = closes.groupby(level="date")
    weight = closes / by_date.transform(lambda x: x.abs().sum())
    expected = weight - weight.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, close.to_weight(demean=False).demean())
    assert_allclose(result, expected, rtol=1e-6, atol=1e-9)
