import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_almost_equal

import spectre
from spectre import factors, parallel
from tests.conftest import DELISTED_ON, DIRTY_ADJUSTMENTS, DIRTY_OHLCV
from tests.factors import reference as ref
from tests.factors.helpers import aligned_bars, assert_factor, nansplit_bars, run_dirty

nan = np.nan


# The EMAs used by MACD are truncated (see test_basic.test_ema), the error is about 1e-3
@pytest.mark.parametrize("params", [(12, 26, 9), (10, 15, 5)])
def test_macd_signal(engine, params):
    expected = ref.expected(lambda df: ref.macd(df.close, *params)[1])
    assert_factor(engine, factors.MACD(*params), *expected, decimal=3)


def test_macd_histogram(engine):
    expected = ref.expected(lambda df: ref.macd(df.close)[2])
    assert_factor(engine, factors.MACD().normalized(), *expected, decimal=3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_of_dirty_assets(dirty_engine, dirty_panel, params):
    signal = run_dirty(dirty_engine, factors.MACD(*params))
    histogram = run_dirty(dirty_engine, factors.MACD(*params).normalized())
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        close = dirty_panel.xs(asset, level="asset").close
        _, expected_signal, expected_histogram = ref.macd(close, *params)
        got = signal.xs(asset, level="asset").values
        assert_allclose(got, expected_signal.values, rtol=0, atol=2e-3, err_msg=asset)
        got = histogram.xs(asset, level="asset").values
        assert_allclose(got, expected_histogram.values, rtol=0, atol=2e-3, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_adjust(dirty_engine, dirty_panel, params):
    """`adjust` is for the fast and slow EMAs as well, all three match ewm(adjust=True)."""
    signal = run_dirty(dirty_engine, factors.MACD(*params, adjust=True))
    histogram = run_dirty(dirty_engine, factors.MACD(*params, adjust=True).normalized())
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        close = dirty_panel.xs(asset, level="asset").close
        _, expected_signal, expected_histogram = ref.macd(close, *params, adjust=True)
        got = signal.xs(asset, level="asset").values
        assert_allclose(got, expected_signal.values, rtol=0, atol=2e-3, err_msg=asset)
        got = histogram.xs(asset, level="asset").values
        assert_allclose(got, expected_histogram.values, rtol=0, atol=2e-3, err_msg=asset)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_across_split(dirty_engine, dirty_panel, params, adjust):
    """MACD is linear in the price, so it is the MACD of clean_close in the scale of each bar."""
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    _, signal, histogram = ref.macd(split.clean_close, *params, adjust=adjust)

    factor = factors.MACD(*params, adjust=adjust)
    result = run_dirty(dirty_engine, factor).xs("SPLIT", level="asset")
    assert_allclose(result.values, (signal * scale).values, rtol=0, atol=2e-3)
    result = run_dirty(dirty_engine, factors.MACD(*params, adjust=adjust).normalized())
    result = result.xs("SPLIT", level="asset")
    assert_allclose(result.values, (histogram * scale).values, rtol=0, atol=2e-3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_across_split_after_nan_bar(dirty_engine, dirty_panel, params, adjust):
    bars, scale = nansplit_bars(dirty_panel)
    _, signal, histogram = ref.macd(bars.clean_close, *params, adjust=adjust)

    result = run_dirty(dirty_engine, factors.MACD(*params, adjust=adjust))
    result = result.xs("NANSPLIT", level="asset")
    assert_allclose(result.values, (signal * scale).values, rtol=0, atol=2e-3)
    result = run_dirty(dirty_engine, factors.MACD(*params, adjust=adjust).normalized())
    result = result.xs("NANSPLIT", level="asset")
    assert_allclose(result.values, (histogram * scale).values, rtol=0, atol=2e-3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("params", [(12, 26, 9), (5, 10, 3)])
def test_macd_on_rows_added_by_align_by_time(added_rows_engine, added_rows_panel, params, adjust):
    """EMA keeps its value on these rows, so the line is valid there and stays in the window."""
    signal = run_dirty(added_rows_engine, factors.MACD(*params, adjust=adjust)).unstack("asset")
    for asset in ("FULL", "NANSPLIT", "LISTED", "DELISTED", "HALTED"):
        bars, scale = aligned_bars(added_rows_panel, asset)
        expected = ref.macd(bars.clean_close, *params, adjust=adjust)[1] * scale
        # later on, the windows of DELISTED run out of valid bars and leave `ewm` behind
        rows = DELISTED_ON + 10 if asset == "DELISTED" else None
        got = signal[asset].iloc[:rows]
        # a window with a gap has fewer bars to decay the truncation error
        assert_allclose(got, expected.iloc[:rows], rtol=0, atol=5e-3, err_msg=asset)


@pytest.mark.parametrize("band", [0, 1, 2])
def test_bbands(engine, band):
    expected = ref.expected(lambda df: ref.bbands(df.close)[band])
    assert_factor(engine, factors.BBANDS()[band], *expected)


def bbands_normalized(win, k):
    def _normalized(df):
        upper, middle, _ = ref.bbands(df.close, win, k)
        return (df.close - middle) / (upper - middle)

    return _normalized


def test_bbands_normalized(engine):
    assert_factor(engine, factors.BBANDS().normalized(), *ref.expected(bbands_normalized(20, 2)))
    factor = factors.BBANDS(win=50, inputs=(factors.OHLCV.close, 3)).normalized()
    assert_factor(engine, factor, *ref.expected(bbands_normalized(50, 3)))


def bbands_score(closes, win, k):
    """(last - mean) / (k * std) of the valid bars of each window, NaN if they are all equal."""
    ret = np.full(len(closes), nan)
    for i in range(len(closes)):
        window = closes[max(0, i - win + 1) : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) and valid.max() > valid.min():
            ret[i] = (window[-1] - valid.mean()) / (k * valid.std())
    return ret


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.filterwarnings("ignore:Historical data seems insufficient")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bbands_normalized_of_a_flat_run(device, dtype):
    """Whether a mean of equal values rounds depends on them, so try a lot of constants."""
    win, bars = 7, 30
    dates = pd.date_range("2020-01-01", periods=bars, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    closes = {f"FLAT{i}": np.full(bars, price) for i, price in enumerate(rng.uniform(1, 300, 30))}
    # a halted asset with carried prices
    carried = 100 + rng.normal(0, 1, bars).cumsum()
    carried[10:21] = carried[10]
    closes["CARRIED"] = carried
    df = pd.concat(
        {asset: pd.DataFrame(dict.fromkeys(DIRTY_OHLCV, c), dates) for asset, c in closes.items()},
        names=["asset", "date"],
    ).swaplevel()
    loader = spectre.data.MemoryLoader(df.sort_index().astype(dtype), DIRTY_OHLCV)
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(factors.BBANDS(win).normalized(), "score")
    result = engine.run(dates[0], dates[-1], delay_factor=False).score

    expected = bbands_score(carried.astype(dtype).astype(np.float64), win, 2)
    got = result.xs("CARRIED", level="asset").values
    assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert np.isnan(got[16:21]).all()
    assert result.drop("CARRIED", level="asset").isna().all()


@pytest.mark.filterwarnings("ignore:Historical data seems insufficient")
@pytest.mark.parametrize("asset", ["FULL", "LISTED", "HALTED", "NANBAR", "SPLIT"])
def test_bbands_normalized_of_dirty_assets(dirty_engine, dirty_panel, asset):
    """The score doesn't depend on the scale, so a split makes no difference."""
    result = run_dirty(dirty_engine, factors.BBANDS(5).normalized())
    closes = dirty_panel.xs(asset, level="asset").clean_close.values
    got = result.xs(asset, level="asset").values
    assert_allclose(got, bbands_score(closes, 5, 2), rtol=1e-4, atol=1e-5)


def test_bbands_needs_k():
    with pytest.raises(ValueError, match="needs 2 inputs"):
        factors.BBANDS(inputs=[factors.OHLCV.close])


def test_true_range(engine):
    assert_factor(engine, factors.TRANGE(), *ref.expected(ref.true_range))


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_of_dirty_assets(dirty_engine, dirty_panel):
    win = 14
    result = run_dirty(dirty_engine, factors.MA(win, inputs=(factors.TRANGE(),)))
    for asset in ("FULL", "LISTED", "DELISTED", "HALTED", "NANBAR"):
        df = dirty_panel.xs(asset, level="asset")
        expected = ref.true_range(df).rolling(win, min_periods=1).mean()
        got = result.xs(asset, level="asset").values
        assert_allclose(got, expected.values, rtol=1e-6, err_msg=asset)


def check_true_range_of_scaled_bars(engine, asset, bars, scale):
    """Range and ATR of `asset` are the ones of its clean prices, in the scale of each bar."""
    win = 14
    clean = pd.DataFrame(
        {"high": bars.high / scale, "low": bars.low / scale, "close": bars.clean_close}
    )
    true_range = ref.true_range(clean)

    result = run_dirty(engine, factors.TRANGE()).xs(asset, level="asset")
    assert_allclose(result.values, (true_range * scale).values, rtol=1e-5)

    atr = factors.MA(win, inputs=(factors.TRANGE(),))
    result = run_dirty(engine, atr).xs(asset, level="asset")
    expected = true_range.rolling(win, min_periods=1).mean() * scale
    assert_allclose(result.values, expected.values, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_across_split(dirty_engine, dirty_panel):
    """ATR from the docstring of TrueRange. Each bar's range is in the scale of that bar."""
    split = dirty_panel.xs("SPLIT", level="asset")
    check_true_range_of_scaled_bars(dirty_engine, "SPLIT", split, split.close / split.clean_close)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_across_split_after_nan_bar(dirty_engine, dirty_panel):
    bars, scale = nansplit_bars(dirty_panel)
    check_true_range_of_scaled_bars(dirty_engine, "NANSPLIT", bars, scale)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_true_range_on_rows_added_by_align_by_time(added_rows_engine, added_rows_panel):
    bars, scale = aligned_bars(added_rows_panel, "NANSPLIT")
    check_true_range_of_scaled_bars(added_rows_engine, "NANSPLIT", bars, scale)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "scale",
    [[1] * 7, [1] * 4 + [0.5] * 3],
    ids=["no split", "split on the next bar"],
)
def test_atr_keeps_a_bar_with_only_the_close_missing(device, scale):
    """
    The bar has a high, a low and a close before it, so it has a range. Clean prices by hand:
    true range [NaN, 3, 2, 5, NaN, 4, 2], the NaN after it has no close before it.
    """
    scale = np.array(scale)
    dates = pd.date_range("2020-01-01", periods=7, freq="B", tz="UTC")
    bars = pd.DataFrame(
        {
            "high": np.array([11.0, 13, 12, 16, 15, 18, 17]) * scale,
            "low": np.array([9.0, 10, 10, 12, 13, 14, 15]) * scale,
            "close": np.array([10.0, 12, 11, nan, 14, 17, 16]) * scale,
            "volume": 1000.0,
            "ex-dividend": 0.0,
            "split_ratio": np.where(np.diff(scale, prepend=1) == 0, 1.0, 0.5),
        },
        index=dates,
    )
    bars["open"] = bars.close
    panel = pd.concat({"A": bars, "B": bars}, names=["asset", "date"]).swaplevel().sort_index()
    loader = spectre.data.MemoryLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(factors.MA(3, inputs=(factors.TRANGE(),)), "atr")
    result = engine.run(dates[0], dates[-1], delay_factor=False).xs("A", level="asset")
    expected = np.array([nan, 3, 5 / 2, 10 / 3, 7 / 2, 9 / 2, 3]) * scale
    assert_allclose(result.atr.values, expected, rtol=1e-6)


# talib.RSI(close, 14) of the bars before the default range, 1000 bars after its first value
TALIB_RSI_AAPL = [39.1585967, 32.5935917, 35.8566833, 35.8566833, 38.2034011, 42.2040408,
                  46.3533033, 41.6204222, 44.3892683]  # fmt: skip
TALIB_RSI_MSFT = [43.3547561, 44.7610889, 47.7678806, 46.2391966, 49.2104855, 50.3515566,
                  46.327828, 46.6394416]  # fmt: skip


# The window of RSI is truncated like the one of EMA, the error is about 1e-2 on a scale of 100
def test_rsi(engine):
    expected = ref.expected(lambda df: ref.rsi(df.close))
    assert_factor(engine, factors.RSI(), *expected, decimal=1)
    assert_almost_equal(expected[0], TALIB_RSI_AAPL, decimal=6)
    assert_almost_equal(expected[1], TALIB_RSI_MSFT, decimal=6)


def test_rsi_normalized(engine):
    expected = ref.expected(lambda df: ref.rsi(df.close) / 50 - 1)
    assert_factor(engine, factors.RSI().normalized(), *expected, decimal=3)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize("asset", ["FULL", "LISTED", "HALTED", "NANBAR", "SPLIT"])
def test_rsi_of_missing_bars_and_splits(dirty_engine, dirty_panel, asset):
    """Until its window of 45 bars is full it's the series from the first bar of the asset."""
    result = run_dirty(dirty_engine, factors.RSI(5)).xs(asset, level="asset")
    clean = dirty_panel.xs(asset, level="asset").clean_close
    assert_allclose(result.values, ref.rsi(clean, 5).values, rtol=1e-3, atol=1e-4)


def test_atr_of_the_docstring_of_true_range_is_wilders(engine):
    def atr(df):
        return ref.true_range(df).ewm(alpha=1 / 14, adjust=False).mean()

    expected = ref.expected(atr)
    assert_factor(engine, factors.EMA(2 * 14 - 1, inputs=(factors.TRANGE(),)), *expected, decimal=2)


def test_stochf(engine):
    assert_factor(engine, factors.STOCHF(), *ref.expected(ref.stochf))


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_stochf_of_a_flat_window_is_nan(device):
    """
    A bar with one price and no trades, and multipliers that aren't 1 because of a later dividend.
    Rounding would leave the close above a low of the same price, which is inf over a range of 0,
    and one inf makes the zscore of the whole date NaN.
    """
    dates = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
    flat = pd.DataFrame(index=dates, data={"close": np.linspace(30, 33.3, 30)}, dtype=np.float32)
    flat.iloc[15:] = 33.3
    frames = {"FLAT": flat, "OTHER": flat + np.arange(30, dtype=np.float32).reshape(-1, 1) / 7}
    panel = pd.concat(frames, names=["asset", "date"]).swaplevel().sort_index()
    panel = panel.assign(open=panel.close, volume=1000.0, split_ratio=1.0)
    panel["high"], panel["low"] = panel.close * 1.0, panel.close * 1.0
    panel.loc[panel.index.get_level_values("asset") == "OTHER", "high"] *= 1.01
    panel.loc[panel.index.get_level_values("asset") == "OTHER", "low"] *= 0.99
    panel["ex-dividend"] = 0.0
    panel.loc[(dates[-1], "FLAT"), "ex-dividend"] = 0.17
    loader = spectre.data.MemoryLoader(panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    engine = factors.FactorEngine(loader)
    if device == "cuda":
        engine.to_cuda()
    engine.add(factors.STOCHF(5), "stochf")
    engine.add(factors.STOCHF(5).zscore(), "zscore")
    result = engine.run(dates[20], dates[-1], delay_factor=False)
    assert result.stochf.xs("FLAT", level="asset").iloc[:-1].isna().all()
    assert np.isfinite(result.stochf.xs("OTHER", level="asset")).all()


def test_stochf_of_a_close_out_of_an_empty_range(device):
    """A bar with only a close: the range is 0, and it's not a value of the oscillator."""
    zeros = torch.zeros(1, 4, device=device)
    highs, lows = parallel.Rolling(zeros, 2), parallel.Rolling(zeros, 2)
    closes = parallel.Rolling(torch.full((1, 4), 50.0, device=device), 2)
    result = factors.STOCHF(2).compute(highs, lows, closes)
    assert np.isnan(result.cpu().numpy()).all()


def test_stochf_normalized(engine):
    assert_factor(
        engine, factors.STOCHF().normalized(), *ref.expected(lambda df: ref.stochf(df) / 100 - 0.5)
    )
