import numpy as np
import pandas as pd
import pytest
import torch
from numpy.lib.stride_tricks import sliding_window_view
from numpy.testing import assert_allclose, assert_almost_equal

import spectre
from spectre import factors, parallel
from tests.conftest import DIRTY_DATES, HALTED_BARS, NANSPLIT_ON, SPLIT_ON
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, run_dirty, run_factor, run_late_listed

nan = np.nan


def weekday(df):
    return pd.Series(df.index.day_of_week, index=df.index, dtype=float)


def day(df):
    return pd.Series(df.index.day, index=df.index, dtype=float)


# ------------------------------------ returns ------------------------------------


def test_returns(engine):
    expected = ref.expected(lambda df: ref.returns(df.close))
    assert_factor(engine, factors.Returns(), *expected)
    # close of 2019-01-02 / close of 2018-12-31 - 1, delayed to 2019-01-03
    assert_almost_equal(expected[0][:2], [158.61 / 163.71 - 1, 145.23 / 158.61 - 1])


def test_returns_window(engine):
    expected = ref.expected(lambda df: ref.returns(df.close, win=5))
    assert_factor(engine, factors.Returns(win=5), *expected)


def test_returns_with_nan(device):
    """missing data considered as delisted, returns are calculated by the last valid data."""
    data = torch.tensor([[1, 2, nan, 4, 8, nan], [3, 4, 4.5, 6, nan, nan]], device=device)
    result = factors.Returns().compute(parallel.Rolling(data, win=3)).cpu().numpy()
    expected = [[nan, nan, 1, 1, nan, 1], [nan, nan, 0.5, 0.5, 1 / 3, nan]]
    assert_almost_equal(result, expected)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_returns_over_a_split_after_a_nan_bar(dirty_engine, dirty_panel):
    """The split of NANSPLIT must be adjusted, even though the bar before it has no prices."""
    win = 5
    # the windows that start before the split, and end on or after it
    spanning = DIRTY_DATES[NANSPLIT_ON : NANSPLIT_ON + win - 1]
    dirty_engine.add(factors.Returns(win=win), "returns")
    result = dirty_engine.run(spanning[0], spanning[-1], delay_factor=False)

    clean_close = dirty_panel.clean_close.xs("NANSPLIT", level="asset")
    expected = (clean_close / clean_close.shift(win - 1) - 1)[spanning]
    assert expected.notna().sum() == win - 2  # one window starts on the NaN bar
    assert_allclose(result.returns.xs("NANSPLIT", level="asset"), expected, rtol=1e-5)


def test_log_returns(engine):
    assert_factor(engine, factors.LogReturns(), *ref.expected(lambda df: ref.log_returns(df.close)))


# ------------------------------------ moving averages ------------------------------------


@pytest.mark.parametrize("win", [3, 11])
def test_sma(engine, win):
    assert_factor(engine, factors.SMA(win), *ref.expected(lambda df: ref.sma(df.close, win)))


def test_sma_of_open_not_delayed(engine):
    # open price is available at market open, should not be delayed
    expected = ref.expected(lambda df: ref.sma(df.open, 11), delayed=False)
    assert_factor(engine, factors.SMA(11, inputs=[factors.OHLCV.open]), *expected, delay=False)
    assert len(expected[0]) == 10


def ema_truncation_bound(span, prices, missing=0):
    """
    spectre's EMA only uses the last `4.5 * (span + 1)` bars, the truncation error is
    (1 - alpha) ^ win * (EMA(t - win) - X(t - win + 1)), which is bounded by the price range.
    A window with `missing` bars has that many fewer bars to decay the error.
    """
    alpha = 2 / (span + 1)
    win = int(4.5 * (span + 1))
    return (1 - alpha) ** (win - missing) * (prices.max() - prices.min())


@pytest.mark.parametrize("span", [11, 50])
@pytest.mark.parametrize("adjust", [False, True])
def test_ema(engine, span, adjust):
    aapl, msft = run_factor(engine, factors.EMA(span, adjust=adjust))
    expected = ref.expected(lambda df: df.close.ewm(span=span, adjust=adjust).mean())
    for asset, result, exp in zip(ref.ASSETS, (aapl, msft), expected):
        bound = ema_truncation_bound(span, ref.daily_prices(asset).close)
        assert bound < 0.02
        assert np.abs(result - exp).max() <= bound, asset


def test_ema_half_life(engine):
    aapl, msft = run_factor(engine, factors.EMA(half_life=5))
    expected = ref.expected(lambda df: df.close.ewm(halflife=5, adjust=False).mean())
    # window is 15 half-lives, truncation error is 0.5^15 * price range
    assert_almost_equal(aapl, expected[0], decimal=2)
    assert_almost_equal(msft, expected[1], decimal=2)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("start", [0, 40])
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("span", [3, 5, 12, 26])
def test_ema_of_dirty_assets(dirty_engine, dirty_panel, span, adjust, start):
    """A new listing, a halt without rows and a NaN close. Only the window truncates it."""
    factor = factors.EMA(span, adjust=adjust)
    result = run_dirty(dirty_engine, factor, start)
    # history is loaded by dates, so after a halt fewer bars of that asset are loaded
    loaded = dirty_panel.loc[DIRTY_DATES[max(start - factor.win + 1, 0)] :]
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        close = loaded.xs(asset, level="asset").close
        got = result.xs(asset, level="asset")
        expected = close.ewm(span=span, adjust=adjust).mean().loc[got.index]
        # the weights of adjust=True are float32
        atol = ema_truncation_bound(span, close, missing=1) + 1e-5
        assert_allclose(got.values, expected.values, rtol=0, atol=atol, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("span", [3, 5, 12, 26])
def test_ema_across_split(dirty_engine, dirty_panel, span, adjust):
    """Each window is in the scale of its last bar, so the split bar must match too."""
    result = run_dirty(dirty_engine, factors.EMA(span, adjust=adjust)).xs("SPLIT", level="asset")
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    expected = split.clean_close.ewm(span=span, adjust=adjust).mean() * scale
    # the multipliers of the loader are float32
    atol = ema_truncation_bound(span, split.clean_close) + 1e-4
    assert_allclose(result.values, expected.values, rtol=0, atol=atol)
    assert scale.iloc[SPLIT_ON - 1] == pytest.approx(1)
    assert scale.iloc[SPLIT_ON] == pytest.approx(0.5)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("span", [1, 2, 3, 5, 12, 26])
def test_ema_of_missing_bars_in_the_window(dirty_engine, dirty_panel, span, adjust):
    """
    With align_by_time a halt is a run of NaN rows, and the older bars decay over it. This is
    `ewm` of the bars of each window, so the window is the only difference from `ewm`. A window
    with no valid bar left is NaN, where `ewm` would keep its last value.
    """
    dirty_engine.align_by_time = True
    result = run_dirty(dirty_engine, factors.EMA(span, adjust=adjust)).unstack("asset")
    closes = dirty_panel.close.unstack("asset")
    win = int(4.5 * (span + 1))

    def ewm_of_window(window):
        return pd.Series(window).ewm(span=span, adjust=adjust).mean().iloc[-1]

    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        expected = closes[asset].rolling(win, min_periods=1).apply(ewm_of_window, raw=True)
        # the weights are float32
        assert_allclose(result[asset], expected, rtol=0, atol=1e-5, err_msg=asset)
    halted = len(DIRTY_DATES[HALTED_BARS])
    assert result.HALTED.isna().sum() == max(halted - win + 1, 0)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
def test_ema_of_truncated_history(dirty_engine, dirty_panel, adjust):
    """With `max_backwards` the first loaded bar is the seed, as in `ewm` of the loaded rows."""
    span, start, backwards = 12, 30, 10
    dirty_engine.max_backwards = backwards
    result = run_dirty(dirty_engine, factors.EMA(span, adjust=adjust), start)
    loaded = dirty_panel.loc[DIRTY_DATES[start - backwards] :]
    for asset in ("FULL", "LISTED", "HALTED", "NANBAR"):
        close = loaded.xs(asset, level="asset").close
        got = result.xs(asset, level="asset")
        expected = close.ewm(span=span, adjust=adjust).mean().loc[got.index]
        assert_allclose(got.values, expected.values, rtol=0, atol=1e-5, err_msg=asset)


@pytest.mark.parametrize("span", [2, 3, 5])
def test_ema_of_nan_runs(device, span):
    """Leading NaNs are not 0, the first valid bar is the seed, and the last value is kept."""
    data = torch.tensor([[nan, nan, 4, 8, nan, nan, 2, nan]], device=device, dtype=torch.float64)
    expected = pd.Series(data[0].cpu().numpy()).ewm(span=span, adjust=False).mean().values
    factor = factors.EMA(span)
    factor.weight = torch.tensor(factor.weight, device=device)  # done by pre_compute_
    result = factor.compute(parallel.Rolling(data, factor.win))
    assert_allclose(result[0].cpu().numpy(), expected, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_new_listing_at_a_constant_price(device):
    dates = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    rows = [(date, "A", 100.0 + i) for i, date in enumerate(dates)]
    rows += [(date, "B", 50.0) for date in dates[12:]]
    df = pd.DataFrame(rows, columns=["date", "asset", "close"])
    df = df.set_index(["date", "asset"]).sort_index()
    for col in ("open", "high", "low"):
        df[col] = df.close
    df["volume"] = 1.0
    ohlcv = ("open", "high", "low", "close", "volume")
    engine = factors.FactorEngine(spectre.data.MemoryLoader(df[list(ohlcv)], ohlcv=ohlcv))
    if device == "cuda":
        engine.to_cuda()
    engine.add(factors.EMA(10), "ema")
    engine.add(factors.EMA(10, adjust=True), "adjusted")
    engine.add(factors.LinearWeightedAverage(10, inputs=[factors.OHLCV.close]), "lwa")
    result = engine.run(dates[0], dates[-1], delay_factor=False).xs("B", level=1)
    assert len(result) == 8
    assert_allclose(result.values, 50.0, rtol=1e-12)


def test_vwap(engine):
    def vwap(df):
        return (df.close * df.volume).rolling(3).sum() / df.volume.rolling(3).sum()

    expected = ref.expected(vwap)
    assert_factor(engine, factors.VWAP(3), *expected)
    assert_almost_equal(expected[1][:3], [103.3029256, 102.5066541, 102.5480467])


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_vwap_of_dirty_assets(dirty_engine, dirty_panel):
    """A bar with a price and no volume, or the other way around, is left out of both sums."""
    win = 3
    result = run_dirty(dirty_engine, factors.VWAP(win))
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        df = dirty_panel.xs(asset, level="asset")
        volume = df.volume.where(df.close.notna())
        amount = (df.close * volume).rolling(win, min_periods=1).sum()
        expected = amount / volume.rolling(win, min_periods=1).sum()
        got = result.xs(asset, level="asset")
        assert_allclose(got.values, expected.values, rtol=1e-6, err_msg=asset)


def test_weighted_average_uses_one_set_of_samples(device):
    base = torch.tensor([[100, nan, 107, 5, 8, 1, nan, nan]], device=device)
    weight = torch.tensor([[1000, 5000, 1000, nan, nan, nan, 1, 1]], device=device)
    rolling = parallel.Rolling(base, win=3), parallel.Rolling(weight, win=3)
    result = factors.WeightedAverageValue(3).compute(*rolling)
    # valid pairs: (100, 1000) and (107, 1000), so windows from the 6th on have none
    expected = [100, 100, 103.5, 107, 107, nan, nan, nan]
    assert_almost_equal(result[0].cpu().numpy(), expected, decimal=4)


def test_average_dollar_volume(engine):
    expected = ref.expected(lambda df: (df.close * df.volume).rolling(3).mean())
    assert_factor(engine, factors.AverageDollarVolume(3), *expected, decimal=2)
    # win=1 is close * volume
    assert_factor(
        engine,
        factors.AverageDollarVolume(1),
        *ref.expected(lambda df: df.close * df.volume),
        decimal=2,
    )


def test_annualized_volatility(engine):
    def volatility(df):
        return ref.returns(df.close).rolling(3).std(ddof=0) * np.sqrt(252)

    assert_factor(engine, factors.AnnualizedVolatility(3), *ref.expected(volatility))


def test_linear_weighted_average(engine):
    weights = np.arange(1, 6) / 15

    def lwa(df):
        return weekday(df).rolling(5).apply(lambda x: (x * weights).sum(), raw=True)

    expected = ref.expected(lwa, delayed=False)
    assert_factor(
        engine, factors.LinearWeightedAverage(5, inputs=[factors.WEEKDAY]), *expected, decimal=6
    )
    assert_almost_equal(expected[0][:3], [2.0, 2.2666667, 2.8])


def linear_weighted_average(close, win):
    """Weights 1 to `win`, on the valid bars of each window."""
    windows = sliding_window_view(np.concatenate((np.full(win - 1, nan), close)), win)
    weights = np.arange(1, win + 1.0)
    with np.errstate(invalid="ignore"):
        return np.nansum(windows * weights, axis=1) / (~np.isnan(windows) * weights).sum(axis=1)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_linear_weighted_average_of_dirty_assets(dirty_engine, dirty_panel):
    win = 5
    factor = factors.LinearWeightedAverage(win, inputs=[factors.OHLCV.close])
    result = run_dirty(dirty_engine, factor)
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        close = dirty_panel.xs(asset, level="asset").close.values
        got = result.xs(asset, level="asset").values
        assert_allclose(got, linear_weighted_average(close, win), rtol=1e-6, err_msg=asset)

    # a bar keeps the weight of its position, only the total is renormalized
    listed = dirty_panel.xs("LISTED", level="asset").close.values
    expected = [listed[0], (4 * listed[0] + 5 * listed[1]) / 9]
    assert_almost_equal(result.xs("LISTED", level="asset").values[:2], expected, decimal=6)

    split = dirty_panel.xs("SPLIT", level="asset")
    scale = (split.close / split.clean_close).values
    expected = linear_weighted_average(split.clean_close.values, win) * scale
    assert_allclose(result.xs("SPLIT", level="asset").values, expected, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_linear_weighted_average_of_window_without_valid_bars(dirty_engine, dirty_panel):
    win = 3
    dirty_engine.align_by_time = True
    factor = factors.LinearWeightedAverage(win, inputs=[factors.OHLCV.close])
    result = run_dirty(dirty_engine, factor).unstack("asset").HALTED.values
    close = dirty_panel.close.unstack("asset").HALTED.values
    assert_allclose(result, linear_weighted_average(close, win), rtol=1e-6)
    assert np.isnan(result[HALTED_BARS.start + win - 1 : HALTED_BARS.stop]).all()


CPU_SERIES = pd.Series([1.0, 3, nan, 4, 8, 2])


@pytest.mark.cuda
@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "make_factor, expected",
    [
        (
            lambda: factors.LinearWeightedAverage(5, inputs=[factors.OHLCV.close]),
            linear_weighted_average(CPU_SERIES.values, 5),
        ),
        (lambda: factors.EMA(5), CPU_SERIES.ewm(span=5, adjust=False).mean().values),
        (lambda: factors.EMA(5, adjust=True), CPU_SERIES.ewm(span=5, adjust=True).mean().values),
        (lambda: factors.MACD(sign=5), CPU_SERIES.ewm(span=5, adjust=False).mean().values),
    ],
    ids=["LinearWeightedAverage", "EMA", "EMA adjust", "MACD"],
)
def test_compute_on_cpu_after_a_run_on_cuda(dirty_loader, make_factor, expected):
    """When CUDA runs out of memory `compute` runs again on cpu, with the weights of the factor."""
    engine = factors.FactorEngine(dirty_loader)
    engine.to_cuda()
    factor = make_factor()
    run_dirty(engine, factor)

    data = torch.tensor(CPU_SERIES.values[None])
    result = factor.compute(parallel.Rolling(data, factor.win))
    assert result.device.type == "cpu"
    assert_allclose(result[0].numpy(), expected, rtol=1e-6)


# ------------------------------------ element wise ------------------------------------


def test_element_wise_max(engine):
    factor = factors.ElementWiseMax(inputs=[factors.WEEKDAY, factors.DatetimeDataFactor("day")])
    expected = ref.expected(lambda df: np.maximum(weekday(df), day(df)), delayed=False)
    assert_factor(engine, factor, *expected)
    assert_almost_equal(expected[0], [2.0, 3, 4, 7, 8, 9, 10, 11, 14, 15])


def test_element_wise_min(engine):
    factor = factors.ElementWiseMin(inputs=[factors.WEEKDAY, factors.DatetimeDataFactor("day") - 1])
    expected = ref.expected(lambda df: np.minimum(weekday(df), day(df) - 1), delayed=False)
    assert_factor(engine, factor, *expected)
    assert_almost_equal(expected[0], [1.0, 2, 3, 0, 1, 2, 3, 4, 0, 1])


def test_element_wise_max_nan():
    a = torch.tensor([[1.0, nan, nan, 4]])
    b = torch.tensor([[2.0, 1.0, nan, 3]])
    assert_almost_equal(factors.ElementWiseMax().compute(a, b), [[2, 1, nan, 4]])
    assert_almost_equal(factors.ElementWiseMin().compute(a, b), [[1, 1, nan, 3]])


# weekday has same values in the window, use day as a decimal point to break the ties
WEEKDAY_DAY = factors.WEEKDAY + factors.DatetimeDataFactor("day") / 100


def rolling_arg(func):
    def _arg(df):
        return (weekday(df) + day(df) / 100).rolling(5).apply(lambda x: (func(x) + 1) / 5, raw=True)

    return _arg


def test_rolling_argmax(engine):
    expected = ref.expected(rolling_arg(np.argmax), delayed=False)
    assert_factor(engine, factors.RollingArgMax(5, inputs=[WEEKDAY_DAY]), *expected)
    assert_almost_equal(
        expected[0], [3 / 5, 2 / 5, 1 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 5 / 5, 4 / 5, 3 / 5]
    )


def test_rolling_argmin(engine):
    expected = ref.expected(rolling_arg(np.argmin), delayed=False)
    assert_factor(engine, factors.RollingArgMin(5, inputs=[WEEKDAY_DAY]), *expected)
    assert_almost_equal(
        expected[1], [4 / 5, 3 / 5, 2 / 5, 5 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 4 / 5]
    )


def rolling_position(close, win, func):
    """Position of the extreme of each window, `win` is the last bar. No valid bar gives NaN."""
    windows = sliding_window_view(np.concatenate((np.full(win - 1, nan), close)), win)
    empty = np.isnan(windows).all(axis=1)
    windows = np.where(empty[:, None], 0, windows)
    return np.where(empty, nan, (func(windows, axis=1) + 1) / win)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("align_by_time", [False, True])
@pytest.mark.parametrize(
    "factor, func", [(factors.RollingArgMax, np.nanargmax), (factors.RollingArgMin, np.nanargmin)]
)
def test_rolling_arg_of_dirty_assets(dirty_engine, dirty_panel, factor, func, align_by_time):
    win = 5
    dirty_engine.align_by_time = align_by_time
    result = run_dirty(dirty_engine, factor(win, inputs=[factors.OHLCV.close]))
    closes = dirty_panel.close.unstack("asset")
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR", "SPLIT"):
        close = dirty_panel.xs(asset, level="asset").close
        if asset == "SPLIT":
            close = dirty_panel.xs(asset, level="asset").clean_close
        elif align_by_time:
            close = closes[asset]
        got = result.xs(asset, level="asset").values
        assert_allclose(got, rolling_position(close.values, win, func), rtol=1e-6, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "factor, positions",
    [
        # closes: [10, 20, 11, 30, 12, 13, 14]
        (factors.RollingArgMax, [3, 3, 2, 3, 2, 1, 3]),
        (factors.RollingArgMin, [3, 2, 1, 2, 1, 2, 1]),
    ],
)
def test_rolling_arg_of_new_listed_asset(late_listing_engine, factor, positions):
    result = run_late_listed(late_listing_engine, factor(3, inputs=[factors.OHLCV.close]))
    assert_almost_equal(result, np.array(positions) / 3, decimal=6)


def test_rolling_arg_of_infinite_values_and_ties(device):
    inf = np.inf
    data = torch.tensor([[nan, -inf, nan, nan, nan, 2, 5, 5, 1]], device=device)
    result = factors.RollingArgMax(3).compute(parallel.Rolling(data, win=3))
    expected = np.array([nan, 3, 2, 1, nan, 3, 3, 2, 1]) / 3
    assert_almost_equal(result[0].cpu().numpy(), expected, decimal=6)

    result = factors.RollingArgMin(3).compute(parallel.Rolling(-data, win=3))
    assert_almost_equal(result[0].cpu().numpy(), expected, decimal=6)


def test_constants_factor(engine):
    aapl, msft = run_factor(engine, factors.ConstantsFactor(3.5), delay=False)
    assert_almost_equal(aapl, [3.5] * 10)
    assert_almost_equal(msft, [3.5] * 9)


def test_min_win():
    with pytest.raises(ValueError, match="win"):
        factors.SMA(1)
