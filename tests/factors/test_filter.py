from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import spectre
from spectre import factors
from tests.factors import reference as ref
from tests.factors.helpers import run_dirty, run_late_listed

nan = np.nan
volume = factors.OHLCV.volume
close = factors.OHLCV.close


def assets_of(df):
    return list(df.index.get_level_values(1))


def test_filter_without_calendar(engine_no_calendar):
    """
    The asset with the highest volume of the previous trading day.
    MSFT's junk row on 2019-01-05 (nan volume) is never selected.
    """
    engine_no_calendar.add(volume, "not_used")
    engine_no_calendar.set_filter(volume.top(1))
    result = engine_no_calendar.run(ref.START, ref.END)
    assert assets_of(result) == [
        "MSFT",
        "AAPL",
        "AAPL",
        "AAPL",
        "AAPL",
        "AAPL",
        "AAPL",
        "MSFT",
        "AAPL",
    ]


def test_filter_does_not_affect_inputs(engine):
    engine.set_filter(volume.top(1))
    engine.add(factors.SMA(5), "ma5")
    result = engine.run(ref.START, ref.END)

    # dates of each asset when it has the highest volume (delayed)
    expected_dates = {
        "AAPL": [
            "2019-01-04",
            "2019-01-07",
            "2019-01-08",
            "2019-01-09",
            "2019-01-10",
            "2019-01-11",
            "2019-01-15",
        ],
        "MSFT": ["2019-01-03", "2019-01-14"],
    }
    for asset, dates in expected_dates.items():
        sma = ref.sma(ref.daily_prices(asset).close, 5).shift(1)
        selected = result.xs(asset, level=1).ma5
        assert [str(d.date()) for d in selected.index] == dates
        assert_almost_equal(selected.values, sma[dates].values)


@pytest.mark.parametrize(
    "method, n, top_aapl, solo",
    [
        ("top", 1, True, True),
        ("bottom", 1, False, True),
        # only AAPL on 2019-01-10, 50% of 1 asset is 0.5, nothing selected
        ("top", 0.5, True, False),
        ("bottom", 0.5, False, False),
        ("top", np.float64(0.5), True, False),
        ("bottom", np.float64(0.5), False, False),
        ("top", np.float32(0.5), True, False),
        ("bottom", np.float32(0.5), False, False),
    ],
)
def test_top_bottom(engine, method, n, top_aapl, solo):
    engine.add(getattr(close, method)(n), "test")
    result = engine.run(ref.START, ref.END)
    expected_aapl = [top_aapl] * 9
    expected_aapl[6] = solo
    assert_array_equal(result.xs("AAPL", level=1).test.values, expected_aapl)
    assert_array_equal(result.xs("MSFT", level=1).test.values, [not top_aapl] * 8)


def engine_of_counts(counts, device):
    """One date for each count, with that many assets."""
    dates = pd.date_range("2020-01-01", periods=len(counts), freq="D", tz="UTC")
    rows = [(date, f"A{i:04d}") for date, count in zip(dates, counts) for i in range(count)]
    index = pd.MultiIndex.from_tuples(rows, names=["date", "asset"])
    ohlcv = ("open", "high", "low", "close", "volume")
    df = pd.DataFrame({column: np.arange(1.0, len(rows) + 1) for column in ohlcv}, index=index)
    engine = factors.FactorEngine(spectre.data.MemoryLoader(df, ohlcv=ohlcv))
    if device == "cuda":
        engine.to_cuda()
    return engine, dates


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("method", ["top", "bottom"])
def test_top_bottom_of_every_count_and_fraction(device, method):
    """A fraction floors, and the rounding of the float type must not lose an asset."""
    counts = list(range(1, 301))
    engine, dates = engine_of_counts(counts, device)
    percents = list(range(1, 100))
    for percent in percents:
        engine.add(getattr(close, method)(percent / 100), str(percent))
    selected = engine.run(dates[0], dates[-1], delay_factor=False).groupby(level="date").sum()
    for percent in percents:
        expected = [count * percent // 100 for count in counts]
        assert_array_equal(selected[str(percent)], expected, err_msg=f"{percent}%")


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("method", ["top", "bottom"])
@pytest.mark.parametrize(
    "n, fraction",
    [
        # 100 * 0.53 is 52.999996 in float32
        (0.53, Fraction(53, 100)),
        # 100 * 0.29 is 28.999999999999996 in float64
        (0.29, Fraction(29, 100)),
        (1 / 3, Fraction(1, 3)),
        (2 / 3, Fraction(2, 3)),
        (3 / 11, Fraction(3, 11)),
        (0.009, Fraction(9, 1000)),
        (np.float64(0.29), Fraction(29, 100)),
        (np.float32(0.29), Fraction(29, 100)),
        (np.float32(0.53), Fraction(53, 100)),
    ],
)
def test_top_bottom_of_a_fraction_that_rounds(device, method, n, fraction):
    counts = [1, 2, 3, 6, 9, 11, 22, 55, 99, 100, 300, 1000, 3000]
    engine, dates = engine_of_counts(counts, device)
    engine.add(getattr(close, method)(n), "test")
    selected = engine.run(dates[0], dates[-1], delay_factor=False).groupby(level="date").sum()
    expected = [count * fraction.numerator // fraction.denominator for count in counts]
    assert_array_equal(selected.test, expected)


def test_static_assets(engine):
    engine.set_filter(factors.StaticAssets({"AAPL"}))
    engine.add(close, "close")
    df = engine.run("2018-01-01", ref.END)
    assert set(df.index.get_level_values(1)) == {"AAPL"}
    # pre-screened, only AAPL data loaded
    assert set(engine.dataframe_.index.get_level_values(1)) == {"AAPL"}


def test_static_assets_inverted(engine):
    # not pre-screened
    engine.set_filter(~factors.StaticAssets({"AAPL"}))
    engine.add(close, "close")
    df = engine.run("2018-01-01", ref.END)
    assert set(df.index.get_level_values(1)) == {"MSFT"}


def test_static_assets_not_found(engine):
    engine.set_filter(factors.StaticAssets({"AAPL", "NOT_EXISTS"}))
    engine.add(close, "close")
    assert set(engine.run(ref.START, ref.END).index.get_level_values(1)) == {"AAPL"}

    engine.set_filter(factors.StaticAssets({"NOT_EXISTS"}))
    with pytest.raises(ValueError, match="was not found in DataLoader"):
        engine.run(ref.START, ref.END)


def test_filter_filter_raises():
    mask = close > 177
    with pytest.raises(ValueError, match="does not support local filtering"):
        mask.filter(mask)
    with pytest.raises(ValueError, match="does not support `.sum\\(\\)`"):
        mask.sum(2)


def test_filter_as_input_of_cross_section_factor(engine):
    class AssertOnlyOneSelected(factors.CrossSectionFactor):
        def compute(self, x) -> torch.Tensor:
            assert (x.sum(dim=1) == 1).all()
            return x

    engine.add(AssertOnlyOneSelected(inputs=[factors.StaticAssets(["AAPL"])]), "aapl_filter")
    result = engine.run(ref.START, ref.END)
    assert result.aapl_filter.xs("AAPL", level=1).all()
    assert not result.aapl_filter.xs("MSFT", level=1).any()


def test_logic_ops(engine):
    a = close > 150
    b = factors.OHLCV.volume > 4e7
    engine.add([a, b, a & b, a | b, a ^ b, ~a], ["a", "b", "and", "or", "xor", "not"])
    result = engine.run(ref.START, ref.END)
    assert_array_equal(result["and"], result.a & result.b)
    assert_array_equal(result["or"], result.a | result.b)
    assert_array_equal(result["xor"], result.a ^ result.b)
    assert_array_equal(result["not"], ~result.a)
    # make sure the test is not trivial
    assert result["xor"].any() and not result["xor"].all()


def test_one_hot_many_classes_warns(engine):
    engine.add(factors.OHLCV.volume.one_hot()[0], "test")
    with pytest.warns(RuntimeWarning, match="One hot encoding with too many features"):
        # volume has more than 1000 unique values
        engine.run("2014-11-01", ref.END)
    assert np.issubdtype(engine.run(ref.START, ref.END).test.dtype, np.bool_)


def test_one_hot_has_no_class_from_padding(dirty_engine, dirty_panel):
    # -1 is only in the padding slots, the first class is Monday
    mondays = dirty_panel.index.get_level_values("date").dayofweek == 0
    assert_array_equal(run_dirty(dirty_engine, factors.WEEKDAY.fill_na(-1).one_hot()[0]), mondays)


# ------------------------------------ count of a filter ------------------------------------


def test_ts_sum_of_always_false_filter(dirty_engine):
    result = run_dirty(dirty_engine, (close < 0).ts_sum(3))
    assert_array_equal(result, 0)
    assert result.dtype == "float32"


def test_ts_sum_of_filter(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float)
    expected = selected.groupby(level="asset").transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    assert_array_equal(run_dirty(dirty_engine, (close > 100).ts_sum(3)), expected)


def test_ts_prod_of_filter(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float)
    expected = selected.groupby(level="asset").transform(
        lambda x: x.rolling(3, min_periods=1).apply(np.prod)
    )
    result = run_dirty(dirty_engine, (close > 100).ts_prod(3))
    assert_array_equal(result, expected)
    assert result.dtype == "float32"


def test_xs_sum_of_filter(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float)
    result = run_dirty(dirty_engine, (close > 100).xs_sum())
    assert_array_equal(result, selected.groupby(level="date").transform("sum"))
    assert result.dtype == "float32"


# ------------------------------------ statistics of a filter ------------------------------------
# A row that doesn't exist on a date is not a False of that date, the rows of the dirty panel
# change by date and by asset, so a helper that counts them gives other values.


def zscore_of(x):
    return (x - x.mean()) / x.std(ddof=0)


FILTER_HELPERS = [
    (lambda f, g: f.mean(groupby=g), lambda x: x.transform("mean")),
    (lambda f, g: f.std(groupby=g), lambda x: x.transform("std", ddof=0)),
    (lambda f, g: f.zscore(groupby=g), lambda x: x.transform(zscore_of)),
]
FILTER_HELPER_IDS = ["mean", "std", "zscore"]


@pytest.mark.parametrize("groupby", ["date", "asset"])
@pytest.mark.parametrize("build, reference", FILTER_HELPERS, ids=FILTER_HELPER_IDS)
def test_statistics_of_filter(dirty_engine, dirty_panel, build, reference, groupby):
    expected = reference((dirty_panel.close > 100).astype(float).groupby(level=groupby))
    result = run_dirty(dirty_engine, build(close > 100, groupby))
    assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("groupby", ["date", "asset"])
@pytest.mark.parametrize("build, reference", FILTER_HELPERS, ids=FILTER_HELPER_IDS)
def test_statistics_of_filter_as_float(dirty_engine, dirty_panel, build, reference, groupby):
    expected = reference((dirty_panel.close > 100).astype(float).groupby(level=groupby))
    result = run_dirty(dirty_engine, build((close > 100).float(), groupby))
    assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("helper, expected", [("mean", 1.0), ("std", 0.0), ("zscore", nan)])
def test_statistics_of_filter_skip_padding(dirty_engine, helper, expected):
    # True on every row, and a rolling factor goes on for two bars after the last one, so in
    # the padding it is True and then False
    filter_ = factors.MA(3) > 0
    result = run_dirty(dirty_engine, getattr(filter_, helper)(groupby="asset"))
    assert_array_equal(result, expected)


def test_mean_of_filter_with_mask(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float).where(dirty_panel.volume > 1500)
    expected = selected.groupby(level="date").transform("mean")
    result = run_dirty(dirty_engine, (close > 100).mean(mask=volume > 1500))
    assert_allclose(result, expected, rtol=1e-6)


def test_weighted_mean_of_filter(dirty_engine, dirty_panel):
    selected = (dirty_panel.close > 100).astype(float)
    total = (selected * dirty_panel.volume).groupby(level="date").transform("sum")
    expected = total / dirty_panel.volume.groupby(level="date").transform("sum")
    result = run_dirty(dirty_engine, (close > 100).mean(weight=volume))
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "build", [lambda f: factors.MA(3, inputs=[f]), lambda f: factors.STDDEV(3, inputs=[f])]
)
def test_rolling_statistics_of_filter_raise(dirty_engine, build):
    """The kernels reject bool on purpose: a padded False can't be told from a real False."""
    with pytest.raises(RuntimeError, match="bool tensors"):
        run_dirty(dirty_engine, build(close > 100))


# ------------------------------------ helpers of a filter ------------------------------------
# A helper on a filter gives the same as on filter.float(), and both agree with pandas. Padding
# shows on different filters: read as False, it moves the ranks of one that is always True, but
# not of one that is always False.

FILTERS = {
    "mixed": (lambda: close > 100, lambda panel: panel.close > 100),
    "always_true": (lambda: volume > 0, lambda panel: panel.volume > 0),
    "always_false": (lambda: close < 0, lambda panel: panel.close < 0),
}


@pytest.fixture(params=FILTERS)
def selected(request, dirty_panel):
    """A filter, and its values in pandas as floats."""
    factor, values = FILTERS[request.param]
    return factor(), values(dirty_panel).astype(float)


@pytest.fixture(params=[False, True], ids=["as_filter", "as_float"])
def cast(request):
    return (lambda factor: factor.float()) if request.param else (lambda factor: factor)


def assert_values(result, expected):
    assert_allclose(result.astype(float), expected.astype(float), rtol=1e-5, atol=1e-6)


def by_date(x):
    return x.groupby(level="date")


def by_asset(x):
    return x.groupby(level="asset")


def rank_of(x, **kwargs):
    return by_date(x).rank(**kwargs)


def share_of(rank, n):
    return rank <= by_date(rank).transform("max") * n


def quantile_of(x, bins):
    """Numpy interpolates the cuts, but no value of a filter is between two neighbours."""
    valid = x.dropna()
    if len(valid) < 2:
        return x * nan
    cuts = np.quantile(valid, np.linspace(0, 1, bins + 1))[1:]
    labels = np.digitize(valid, cuts, right=True)
    return pd.Series(labels, index=valid.index, dtype=float).reindex(x.index)


def weights_of(x):
    return x / x.abs().sum()


def mad_filled(x, z, fill):
    center = x.median()
    mad = (x - center).abs().median()
    return x.mask(((x <= center - z * mad) | (x >= center + z * mad)) & (x != center), fill)


def winsorized(x, z):
    valid = x.dropna()
    if valid.empty:
        return x
    clipped = np.asarray(scipy.stats.mstats.winsorize(valid.to_numpy(), [z, z]))
    return pd.Series(clipped, index=valid.index).reindex(x.index)


GROUP_HELPERS = {
    "rank": (lambda f, m: f.rank(mask=m), lambda x: rank_of(x)),
    "rank_descending": (
        lambda f, m: f.rank(ascending=False, mask=m),
        lambda x: rank_of(x, ascending=False),
    ),
    "rank_dense": (
        lambda f, m: f.rank(method="dense", mask=m),
        lambda x: rank_of(x, method="dense"),
    ),
    "rank_ordinal": (
        lambda f, m: f.rank(method="ordinal", mask=m),
        lambda x: rank_of(x, method="first"),
    ),
    "rank_normalized": (
        lambda f, m: f.rank(normalize=True, mask=m),
        lambda x: rank_of(x) / by_date(rank_of(x)).transform("max"),
    ),
    "top": (
        lambda f, m: f.top(2, mask=m),
        lambda x: rank_of(x, ascending=False, method="first") <= 2,
    ),
    "bottom": (lambda f, m: f.bottom(2, mask=m), lambda x: rank_of(x, method="first") <= 2),
    "top_percent": (
        lambda f, m: f.top(0.5, mask=m),
        lambda x: share_of(rank_of(x, ascending=False, method="first"), 0.5),
    ),
    "bottom_percent": (
        lambda f, m: f.bottom(0.5, mask=m),
        lambda x: share_of(rank_of(x, method="first"), 0.5),
    ),
    "max": (lambda f, m: f.max(mask=m), lambda x: by_date(x).transform("max")),
    "min": (lambda f, m: f.min(mask=m), lambda x: by_date(x).transform("min")),
    "median": (lambda f, m: f.median(mask=m), lambda x: by_date(x).transform("median")),
    "demedian": (lambda f, m: f.demedian(mask=m), lambda x: x - by_date(x).transform("median")),
    "max_by_asset": (
        lambda f, m: f.max(mask=m, groupby="asset"),
        lambda x: by_asset(x).transform("max"),
    ),
    "min_by_asset": (
        lambda f, m: f.min(mask=m, groupby="asset"),
        lambda x: by_asset(x).transform("min"),
    ),
    "median_by_asset": (
        lambda f, m: f.median(mask=m, groupby="asset"),
        lambda x: by_asset(x).transform("median"),
    ),
    "demedian_by_asset": (
        lambda f, m: f.demedian(mask=m, groupby="asset"),
        lambda x: x - by_asset(x).transform("median"),
    ),
    "quantile": (
        lambda f, m: f.quantile(3, mask=m if m is not None else volume > 0),
        lambda x: by_date(x).transform(quantile_of, 3),
    ),
    "to_weight": (
        lambda f, m: f.to_weight(mask=m),
        lambda x: by_date(x - by_date(x).transform("mean")).transform(weights_of),
    ),
    "to_weight_of_values": (
        lambda f, m: f.to_weight(demean=False, mask=m),
        lambda x: by_date(x).transform(weights_of),
    ),
    "mad_clamp": (
        lambda f, m: f.mad_clamp(2, mask=m),
        lambda x: by_date(x).transform(ref.mad_clamp, 2),
    ),
    "mad_clamp_fill": (
        lambda f, m: f.mad_clamp(2, mask=m, fill=-1.0),
        lambda x: by_date(x).transform(mad_filled, 2, -1.0),
    ),
    "winsorizing": (
        lambda f, m: f.winsorizing(0.2, mask=m),
        lambda x: by_date(x).transform(winsorized, 0.2),
    ),
}


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.parametrize("helper, reference", GROUP_HELPERS.values(), ids=GROUP_HELPERS)
def test_group_helpers_of_filter(dirty_engine, selected, cast, helper, reference):
    factor, values = selected
    assert_values(run_dirty(dirty_engine, helper(cast(factor), None)), reference(values))


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.parametrize("helper, reference", GROUP_HELPERS.values(), ids=GROUP_HELPERS)
def test_group_helpers_of_filter_with_mask(
    dirty_engine, dirty_panel, selected, cast, helper, reference
):
    factor, values = selected
    masked = values.where(dirty_panel.volume > 1500)
    result = run_dirty(dirty_engine, helper(cast(factor), volume > 1500))
    assert_values(result, reference(masked))


ROLLING_HELPERS = {
    "ts_max": (lambda f: f.ts_max(3), lambda x: x.rolling(3, min_periods=1).max()),
    "ts_min": (lambda f: f.ts_min(3), lambda x: x.rolling(3, min_periods=1).min()),
    "ts_count": (lambda f: f.ts_count(3), lambda x: x.rolling(3, min_periods=1).count()),
}


@pytest.mark.parametrize("helper, reference", ROLLING_HELPERS.values(), ids=ROLLING_HELPERS)
def test_rolling_helpers_of_filter(dirty_engine, selected, cast, helper, reference):
    factor, values = selected
    result = run_dirty(dirty_engine, helper(cast(factor)))
    assert_values(result, by_asset(values).transform(reference))
    assert result.dtype == "float32"


ELEMENT_HELPERS = {
    "abs": (lambda f: f.abs(), lambda x: x.abs()),
    "sign": (lambda f: f.sign(), np.sign),
    "clamp": (lambda f: f.clamp(0, 1), lambda x: x.clip(0, 1)),
    "clamp_inside": (lambda f: f.clamp(0.2, 0.8), lambda x: x.clip(0.2, 0.8)),
    "clamp_fill": (
        lambda f: f.clamp(0.2, 0.8, fill=0.5),
        lambda x: x.mask((x < 0.2) | (x > 0.8), 0.5),
    ),
}


@pytest.mark.parametrize("helper, reference", ELEMENT_HELPERS.values(), ids=ELEMENT_HELPERS)
def test_element_helpers_of_filter(dirty_engine, selected, cast, helper, reference):
    factor, values = selected
    result = run_dirty(dirty_engine, helper(cast(factor)))
    assert_values(result, reference(values))
    assert result.dtype == "float32"


def test_one_hot_of_filter(dirty_engine, selected, cast):
    """The first class is the lowest value of the rows, not the False of the padding."""
    factor, values = selected
    result = run_dirty(dirty_engine, cast(factor).one_hot()[0])
    assert_array_equal(result, values == values.min())


@pytest.mark.parametrize(
    "fill, values_of",
    [
        (0.5, lambda panel: 0.5),
        (2, lambda panel: 2),
        (lambda: close, lambda panel: panel.close),
    ],
    ids=["float", "int", "factor"],
)
def test_masked_fill_of_filter_with_number(
    dirty_engine, dirty_panel, selected, cast, fill, values_of
):
    factor, values = selected
    high_volume = dirty_panel.volume > 1500
    fill = fill() if callable(fill) else fill
    result = run_dirty(dirty_engine, cast(factor).masked_fill(volume > 1500, fill))
    assert_values(result, values.mask(high_volume, values_of(dirty_panel)))
    assert result.dtype in ("float32", "float64")


@pytest.mark.parametrize(
    "fill, values_of",
    [
        (True, lambda panel: True),
        (False, lambda panel: False),
        (lambda: close > 150, lambda panel: panel.close > 150),
    ],
    ids=["true", "false", "filter"],
)
def test_masked_fill_of_filter_with_bool_stays_bool(
    dirty_engine, dirty_panel, selected, fill, values_of
):
    factor, values = selected
    high_volume = dirty_panel.volume > 1500
    fill = fill() if callable(fill) else fill
    result = run_dirty(dirty_engine, factor.masked_fill(volume > 1500, fill))
    assert_array_equal(result, values.astype(bool).mask(high_volume, values_of(dirty_panel)))
    assert result.dtype == "bool"


# ------------------------------------ newly listed asset ------------------------------------
# B is listed on the first day of the results, so its rolling windows have no data at first.


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_ts_any_of_new_listed_asset(late_listing_engine):
    # no data before listing is not True
    result = run_late_listed(late_listing_engine, (close > 15).ts_any(3))
    assert_array_equal(result, [False, True, True, True, True, True, False])


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_ts_all_of_new_listed_asset(late_listing_engine):
    # days before listing are ignored
    result = run_late_listed(late_listing_engine, (close > 5).ts_all(3))
    assert_array_equal(result, [True] * 7)
