"""
The scale of a price: which factors have multipliers, the windows over them, and the delay.
On the dirty panel the prices of SPLIT without its split and dividend are known, so the value of
a calculation on them is known too, and so is that value in the scale of any bar.
"""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

import spectre
from spectre import factors
from tests.conftest import (
    DIRTY_ADJUSTMENTS,
    DIRTY_DATES,
    DIRTY_OHLCV,
    LISTED_ON,
    SPLIT_ON,
)
from tests.factors.helpers import aligned_bars, run_dirty

pytestmark = pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")

nan = np.nan
open_price, high, low = factors.OHLCV.open, factors.OHLCV.high, factors.OHLCV.low
close, volume = factors.OHLCV.close, factors.OHLCV.volume
START = 10
WIN = 5


def split_bars(panel):
    """The bars of SPLIT, the same bars without its events, and the scale of each bar."""
    bars = panel.xs("SPLIT", level="asset")
    scale = bars.close / bars.clean_close
    clean = bars[["open", "high", "low", "close"]].div(scale, axis=0)
    # the shares of the bars after the split are worth half of the ones before it
    shares = bars.volume * np.where(bars.index < DIRTY_DATES[SPLIT_ON], 1, 0.5)
    return bars, clean.assign(volume=shares), scale


def run_delayed(engine, factor):
    engine.remove_all_factors()
    engine.add(factor, "test")
    return engine.run(DIRTY_DATES[START], DIRTY_DATES[-1], delay_factor="all").test


def bollinger(c, sign):
    return c.close.rolling(3).mean() + sign * 2 * c.close.rolling(3).std(ddof=0)


def weighted_mean(c):
    return (c.close * c.volume).rolling(3).sum() / c.volume.rolling(3).sum()


# a factor that is a price, and the same calculation on the prices without events
PRICES = {
    "ma": (lambda: factors.MA(3), lambda c: c.close.rolling(3).mean()),
    "ema": (lambda: factors.EMA(3), lambda c: c.close.ewm(span=3, adjust=False).mean()),
    "lwma": (
        lambda: factors.LinearWeightedAverage(3, inputs=[close]),
        lambda c: c.close.rolling(3).apply(lambda x: np.average(x, weights=[1, 2, 3]), raw=True),
    ),
    "vwap": (lambda: factors.VWAP(3), weighted_mean),
    "high": (lambda: factors.RollingHigh(3), lambda c: c.close.rolling(3).max()),
    "low": (lambda: factors.RollingLow(3), lambda c: c.close.rolling(3).min()),
    "stddev": (lambda: factors.STDDEV(3), lambda c: c.close.rolling(3).std(ddof=0)),
    "ts_sum": (lambda: close.ts_sum(3), lambda c: c.close.rolling(3).sum()),
    "rolling_first": (
        lambda: factors.RollingFirst(3, close, volume > 0),
        lambda c: c.close.shift(2),
    ),
    "forward_data": (lambda: factors.ForwardSignalData(3, close, volume > 0), lambda c: c.close),
    "regression_intercept": (
        lambda: factors.RollingLinearRegression(3, None, close).intercept,
        lambda c: c.close.rolling(3).apply(lambda y: np.polyfit(range(3), y, 1)[1], raw=True),
    ),
    "bbands_upper": (lambda: factors.BBANDS(3)[0], lambda c: bollinger(c, 1)),
    "bbands_lower": (lambda: factors.BBANDS(3)[2], lambda c: bollinger(c, -1)),
    "filter": (lambda: close.filter(volume > 0), lambda c: c.close),
    "double": (lambda: close.double(), lambda c: c.close),
    "float": (lambda: close.float(), lambda c: c.close),
    "fill_na": (lambda: close.fill_na(0.0), lambda c: c.close),
    "ffill": (lambda: close.fill_na(ffill=True), lambda c: c.close),
    "abs": (lambda: close.abs(), lambda c: c.close),
    "negative": (lambda: -close, lambda c: -c.close),
    "masked_fill": (lambda: close.masked_fill(close < 0, 0.0), lambda c: c.close),
    "times_a_number": (lambda: close * 2, lambda c: c.close * 2),
    "number_times": (lambda: 2 * close, lambda c: c.close * 2),
    "over_a_number": (lambda: close / 2, lambda c: c.close / 2),
    "mid_price": (lambda: (high + low) / 2, lambda c: (c.high + c.low) / 2),
    "body": (lambda: close - open_price, lambda c: c.close - c.open),
    "max_of_prices": (
        lambda: factors.basic.ElementWiseMax(inputs=[open_price, close]),
        lambda c: c[["open", "close"]].max(axis=1),
    ),
    "min_of_prices": (
        lambda: factors.basic.ElementWiseMin(inputs=[open_price, close]),
        lambda c: c[["open", "close"]].min(axis=1),
    ),
}
each_price = pytest.mark.parametrize("name", PRICES)


@each_price
def test_window_over_a_price_is_in_the_scale_of_its_last_bar(dirty_engine, dirty_panel, name):
    make, reference = PRICES[name]
    _, clean, scale = split_bars(dirty_panel)
    result = run_dirty(dirty_engine, factors.MA(WIN, inputs=[make()]), START)
    expected = reference(clean).rolling(WIN).mean() * scale
    got = result.xs("SPLIT", level="asset")
    assert got.notna().all()
    assert_allclose(got, expected.loc[got.index], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "make, reference",
    [(PRICES["ma"][0], PRICES["ma"][1]), (PRICES["mid_price"][0], PRICES["mid_price"][1])],
    ids=["ma", "mid_price"],
)
def test_returns_of_a_price_go_across_an_event(dirty_engine, dirty_panel, make, reference):
    """A window of 2 bars over MA(5) read -50% on the split."""
    _, clean, _ = split_bars(dirty_panel)
    result = run_dirty(dirty_engine, factors.Returns(inputs=[make()]), START)
    got = result.xs("SPLIT", level="asset")
    expected = reference(clean).pct_change(fill_method=None)
    assert_allclose(got, expected.loc[got.index], rtol=1e-3, atol=1e-5)


def own_mean(bars):
    """The mean of 3 bars in the scale of each bar."""
    return bars.clean_close.rolling(3).mean() * (bars.close / bars.clean_close)


# what is not a price, or can't be told to be one, stays as it is in a window
NOT_PRICES = {
    "ratio_with_a_guard": (
        lambda: close / (factors.MA(3) + 1e-6),
        lambda b: b.close / (own_mean(b) + 1e-6),
    ),
    "range_with_a_guard": (
        lambda: (close - open_price) / (high - low + 0.001),
        lambda b: (b.close - b.open) / (b.high - b.low + 0.001),
    ),
    "price_over_a_column": (
        lambda: close / factors.ColumnDataFactor(inputs=("clean_close",)),
        lambda b: b.close / b.clean_close,
    ),
    "price_times_a_column": (
        lambda: close * factors.ColumnDataFactor(inputs=("clean_close",)),
        lambda b: b.close * b.clean_close,
    ),
    "price_times_a_ratio": (lambda: close * (open_price / close), lambda b: b.open),
    "price_times_a_filter": (lambda: close * (volume > 0), lambda b: b.close),
    "price_times_an_inverse": (
        lambda: close * (1 / factors.MA(3)),
        lambda b: b.close / own_mean(b),
    ),
    "products_and_ratios": (lambda: close * open_price / high, lambda b: b.close * b.open / b.high),
    "ratio_of_prices": (lambda: close / open_price, lambda b: b.close / b.open),
    "two_scales": (lambda: close - close.shift(1), lambda b: b.close - b.close.shift(1)),
    "price_and_number": (lambda: close + 1, lambda b: b.close + 1),
    "price_and_volume": (lambda: close + volume, lambda b: b.close + b.volume),
    "dollar_volume": (lambda: close * volume, lambda b: b.close * b.volume),
    "volume_per_price": (lambda: volume / close, lambda b: b.volume / b.close),
    "number_over_price": (lambda: 1 / close, lambda b: 1 / b.close),
    "log": (lambda: close.log(), lambda b: np.log(b.close)),
    "square": (lambda: close**2, lambda b: b.close**2),
}


each_not_price = pytest.mark.parametrize("name", NOT_PRICES)


@each_not_price
def test_window_over_what_is_not_a_price_stays_raw(dirty_engine, dirty_panel, name):
    make, reference = NOT_PRICES[name]
    bars, _, _ = split_bars(dirty_panel)
    result = run_dirty(dirty_engine, factors.MA(WIN, inputs=[make()]), START)
    got = result.xs("SPLIT", level="asset")
    expected = reference(bars).rolling(WIN).mean()
    assert_allclose(got, expected.loc[got.index], rtol=1e-4)


class RawClose(factors.CustomFactor):
    inputs = [close]

    def compute(self, data):
        return data


class ClosePrice(RawClose):
    _own_bar_scale = True


def test_custom_factor_has_no_scale_until_it_sets_one(dirty_engine, dirty_panel):
    bars, clean, scale = split_bars(dirty_panel)
    got = run_dirty(dirty_engine, factors.MA(WIN, inputs=[RawClose()]), START)
    got = got.xs("SPLIT", level="asset")
    assert_allclose(got, bars.close.rolling(WIN).mean().loc[got.index], rtol=1e-5)

    got = run_dirty(dirty_engine, factors.MA(WIN, inputs=[ClosePrice()]), START)
    got = got.xs("SPLIT", level="asset")
    expected = clean.close.rolling(WIN).mean() * scale
    assert_allclose(got, expected.loc[got.index], rtol=1e-4)


def test_a_shift_stays_raw_and_keeps_the_scale_of_its_bar(dirty_engine, dirty_panel):
    bars, clean, scale = split_bars(dirty_panel)
    got = run_dirty(dirty_engine, close.shift(2), START).xs("SPLIT", level="asset")
    assert_allclose(got, bars.close.shift(2).loc[got.index])

    # a window over it ends on the bar 2 bars ago, so that is its scale
    result = run_dirty(dirty_engine, factors.MA(WIN, inputs=[close.shift(2)]), START)
    got = result.xs("SPLIT", level="asset")
    expected = clean.close.shift(2).rolling(WIN).mean() * scale.shift(2)
    assert_allclose(got, expected.loc[got.index], rtol=1e-4)


def test_window_over_another_group_has_nothing_to_rescale(dirty_engine):
    class OverAssets(factors.CustomFactor):
        groupby = "date"
        win = 2
        inputs = [factors.MA(3)]

        def compute(self, data):
            return data.nanmean()

    assert run_dirty(dirty_engine, OverAssets(), START).notna().any()


def test_full_run_at_a_factor_price_goes_across_an_event(dirty_engine, dirty_panel):
    """The bar before a split read -51%, and all bars of a period over it."""
    dirty_engine.add(factors.MA(3), "alpha")
    data, _ = dirty_engine.full_run(
        DIRTY_DATES[START],
        DIRTY_DATES[-1],
        trade_at=(high + low) / 2,
        periods=(1, 4),
        quantiles=2,
        filter_zscore=None,
        preview=False,
    )
    _, clean, _ = split_bars(dirty_panel)
    mid = (clean.high + clean.low) / 2
    for bars, column in ((1, "1D"), (4, "4D")):
        got = data[("Returns", column)].xs("SPLIT", level="asset")
        expected = mid.shift(-bars) / mid - 1
        assert_allclose(got, expected.loc[got.index], rtol=1e-3, atol=1e-5)


# ------------------------------------ the multipliers ------------------------------------


def column_with(multipliers):
    column = factors.ColumnDataFactor(inputs=("close",))
    column._scale = (multipliers, 0)
    return column


def test_a_filled_price_is_in_the_scale_of_its_row(dirty_panel, device):
    """The prices of an ex-date are missing, which is what the csv loader leaves of a price of 0."""
    panel = dirty_panel.copy()
    ex_date = DIRTY_DATES[SPLIT_ON]
    panel.loc[(ex_date, "SPLIT"), list(DIRTY_OHLCV[:4])] = nan
    columns = list(DIRTY_OHLCV) + list(DIRTY_ADJUSTMENTS)
    loader = spectre.data.MemoryLoader(panel[columns], DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()

    _, clean, scale = split_bars(dirty_panel)
    filled = clean.close.mask(clean.index == ex_date).ffill()
    filled_close = close.fill_na(ffill=True)

    got = run_dirty(engine, filled_close, START).xs("SPLIT", level="asset")
    assert_allclose(got, (filled * scale).loc[got.index], rtol=1e-5)

    got = run_dirty(engine, factors.Returns(inputs=[filled_close]), START)
    got = got.xs("SPLIT", level="asset")
    assert_allclose(got, filled.pct_change().loc[got.index], rtol=1e-3, atol=1e-5)

    got = run_dirty(engine, factors.MA(WIN, inputs=[filled_close]), START)
    got = got.xs("SPLIT", level="asset")
    assert_allclose(got, (filled.rolling(WIN).mean() * scale).loc[got.index], rtol=1e-4)


def test_window_over_the_data_of_an_asset(dirty_engine, dirty_panel):
    dirty_engine.align_by_time = True
    _, clean, _ = split_bars(dirty_panel)
    factor = factors.Returns(inputs=[factors.AssetData("SPLIT", close)])
    got = run_dirty(dirty_engine, factor, START).xs("FULL", level="asset")
    expected = clean.close.pct_change(fill_method=None)
    assert_allclose(got, expected.loc[got.index], rtol=1e-3, atol=1e-5)


@pytest.mark.cuda
def test_adjusted_column_on_cpu_with_multipliers_on_cuda():
    """When CUDA runs out of memory `compute` runs again on cpu, the column stays on cuda."""
    column = column_with(torch.tensor([[0.5, 1]], device="cuda"))
    result = factors.AdjustedColumnDataFactor(column).compute(torch.tensor([[100, 60.0]]))
    assert_allclose(result.numpy(), [[50, 60]])


def test_a_deep_graph_fits_the_stack(dirty_engine):
    """A sum of hundreds of factors works, which a walk over its levels ran out of frames on."""
    total = sum(close.zscore() for _ in range(500))
    assert run_delayed(dirty_engine, total).notna().any()
    chain = close
    for _ in range(500):
        chain = chain * 1.0
    assert run_dirty(dirty_engine, factors.MA(2, inputs=[chain]), START).notna().any()


def test_rows_added_by_align_by_time_have_a_scale(added_rows_engine, added_rows_panel):
    """A MA keeps its value on them, in the scale of the bar before them."""
    bars, scale = aligned_bars(added_rows_panel, "NANSPLIT")
    inner = bars.clean_close.rolling(3, min_periods=1).mean()
    expected = inner.rolling(3, min_periods=1).mean() * scale
    result = run_dirty(added_rows_engine, factors.MA(3, inputs=[factors.MA(3)]), START)
    got = result.xs("NANSPLIT", level="asset")
    assert_allclose(got, expected.loc[got.index], rtol=1e-4)


def test_window_over_a_filled_price_before_a_listing(added_rows_engine):
    """A fill gives these rows a value, and they have no bar to take a multiplier from."""
    result = run_dirty(added_rows_engine, close.fill_na(0.0).ts_sum(3), 5)
    listed = result.xs("LISTED", level="asset")
    assert (listed.loc[: DIRTY_DATES[LISTED_ON - 1]] == 0).all()


def test_scale_of_a_deep_shared_graph_stays_flat(dirty_engine, monkeypatch):
    """Each level reads the one before it twice, a walk of every path would never end."""
    levels = 30
    factor = close
    for _ in range(levels):
        factor = (factor * 2 + factor) / 3
    calls = 0
    find = factors.factor.AddFactor._find_scale

    def counted(self):
        nonlocal calls
        calls += 1
        assert calls <= levels, "walked the paths of the graph"
        return find(self)

    monkeypatch.setattr(factors.factor.AddFactor, "_find_scale", counted)
    run_dirty(dirty_engine, factors.MA(2, inputs=[factor]), START)
    assert calls == levels


def test_scale_does_not_outlive_a_run(dirty_engine, dirty_panel):
    """Factors are shared by runs and engines, they must not keep a tensor, or use an old one."""
    columns = list(DIRTY_OHLCV) + list(DIRTY_ADJUSTMENTS)
    panel = dirty_panel.xs("SPLIT", level="asset", drop_level=False)[columns]
    other_engine = factors.FactorEngine(
        spectre.data.MemoryLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    )
    ma = factors.MA(3)
    window = factors.MA(3, inputs=[ma])
    run_dirty(dirty_engine, window, START)
    assert ma._scale is None
    assert run_dirty(other_engine, window, START).notna().all()


def test_adjustments_read_before_a_run_change_nothing(dirty_engine, dirty_panel):
    _, clean, scale = split_bars(dirty_panel)
    ma = factors.MA(3)
    assert ma.adjustments is None
    got = run_dirty(dirty_engine, factors.MA(WIN, inputs=[ma]), START).xs("SPLIT", level="asset")
    expected = clean.close.rolling(3).mean().rolling(WIN).mean() * scale
    assert_allclose(got, expected.loc[got.index], rtol=1e-4)


def test_columns_share_their_multipliers(dirty_engine):
    """The same tensor, so a sum of prices is one scale without comparing its values."""
    shared = []

    class Probe(factors.CustomFactor):
        inputs = [close, open_price]

        def compute(self, data, _):
            shared.append(self.inputs[0].adjustments is self.inputs[1].adjustments)
            return data

    run_dirty(dirty_engine, Probe(), START)
    assert shared == [True]


def test_same_scale():
    same_scale = factors.factor._same_scale
    multipliers = torch.tensor([[2, 2.0]])
    column = column_with(multipliers)
    assert same_scale(column, column_with(multipliers)) is column._scale
    assert same_scale(column, column_with(multipliers.clone())) is None
    assert same_scale(column, 1) is None
    assert same_scale(1, column) is None
    shifted = column.shift(1)
    shifted._scale = shifted._find_scale()
    assert same_scale(column, shifted) is None
