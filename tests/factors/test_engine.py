import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import spectre
from spectre import factors
from tests.conftest import DELISTED_ON, DIRTY_DATES
from tests.factors import reference as ref

close = factors.OHLCV.close


def test_run_raw(engine):
    engine.add(factors.OHLCV.open, "open")
    df = engine.run("2018-01-01", ref.END, False)
    raw = engine.run_raw("2018-01-01", ref.END, False)
    assert_almost_equal(df.open.values, raw["open"].numpy())


def test_run_raw_with_filter(engine):
    engine.add(factors.SMA(5), "ma5")
    engine.set_filter(factors.OHLCV.volume.top(1))
    df = engine.run(ref.START, ref.END)
    raw, index = engine.run_raw(ref.START, ref.END, return_index=True)
    assert_almost_equal(df.ma5.values, raw["ma5"].numpy())
    assert_array_equal(df.index.get_level_values(0), index)


def test_run_raw_index_without_filter(engine):
    engine.add(factors.SMA(5), "ma5")
    df = engine.run(ref.START, ref.END)
    raw, index = engine.run_raw(ref.START, ref.END, return_index=True)
    assert_almost_equal(df.ma5.values, raw["ma5"].numpy())
    assert_array_equal(df.index.get_level_values(0), index)


def test_no_trading_day(engine):
    engine.add(factors.SMA(2), "test")
    with pytest.raises(AssertionError, match="There is no data between.*"):
        engine.run("2019-01-05", "2019-01-06")


def test_no_factors(engine):
    with pytest.raises(ValueError, match="Please add at least one factor"):
        engine.run(ref.START, ref.END)


def test_delay_factor(engine):
    engine.add(close, "close")
    engine.add(factors.OHLCV.open, "open")
    df = engine.run(ref.START, ref.END)
    # close is available after market closed, so it's delayed, open is not
    aapl = ref.daily_prices("AAPL").loc["2019-01-03" : ref.END]
    previous = ref.daily_prices("AAPL").loc["2019-01-02":"2019-01-14"]
    assert_array_equal(df.xs("AAPL", level=1).close.values, previous.close.values)
    assert_array_equal(df.xs("AAPL", level=1).open.values, aapl.open.values)

    with pytest.warns(RuntimeWarning, match="delay_factor is set to False"):
        df = engine.run(ref.START, ref.END, delay_factor=False)
    assert df.index.get_level_values(0)[0] == pd.Timestamp("2019-01-02", tz="UTC")

    df = engine.run(ref.START, ref.END, delay_factor="all")
    assert_array_equal(df.xs("AAPL", level=1).open.values, previous.open.values)


def test_insufficient_history_warning(engine):
    engine.add(factors.SMA(20), "test")
    with pytest.warns(RuntimeWarning, match="Historical data seems insufficient"):
        engine.run("2014-10-27", "2014-11-30")


def test_lookahead_bias(engine):
    engine.add(factors.SMA(5), "test")
    assert engine.test_lookahead_bias(ref.START, ref.END) == "No assertion raised."
    # shift(-1) is compensated by the delay, -2 uses tomorrow's data
    engine.add(close.shift(-2), "biased")
    with pytest.raises(RuntimeError, match="look-ahead bias"):
        engine.test_lookahead_bias(ref.START, ref.END)


@pytest.mark.parametrize("loader_aligned", [False, True])
def test_align_by_time(loader_aligned):
    if loader_aligned:
        loader = spectre.data.CsvDirLoader(
            str(ref.DATA_DIR / "daily"),
            calender_asset="AAPL",
            ohlcv=ref.DAILY_OHLCV,
            prices_index="date",
            parse_dates=True,
            align_by_time=True,
        )
    else:
        loader = spectre.data.CsvDirLoader(
            str(ref.DATA_DIR / "daily"),
            calender_asset="AAPL",
            ohlcv=ref.DAILY_OHLCV,
            prices_index="date",
            parse_dates=True,
        )
    engine = spectre.factors.FactorEngine(loader)
    engine.align_by_time = not loader_aligned
    engine.add(close, "close")
    engine.add(factors.SMA(2), "ma")
    df = engine.run(ref.START, ref.END)

    # MSFT has a nan row on 2019-01-10, so all assets have the same number of rows
    assert len(df.xs("MSFT", level=1)) == len(df.xs("AAPL", level=1)) == 9
    assert np.isnan(df.loc[("2019-01-11", "MSFT"), "close"])
    # mean of 2019-01-09 and nan
    assert df.loc[("2019-01-11", "MSFT"), "ma"] == df.loc[("2019-01-10", "MSFT"), "close"]


def test_same_factor_in_two_engines(daily_loader):
    engine = factors.FactorEngine(daily_loader)
    engine2 = factors.FactorEngine(daily_loader)
    f = factors.MA(5)
    universe = factors.OHLCV.volume.top(1)
    engine.add(f, "f")
    engine.add(universe, "mask")
    engine2.add(f, "f")
    engine2.set_filter(universe)

    result = engine.run(ref.START, ref.END)
    result2 = engine2.run(ref.START, ref.END)
    assert_array_equal(result.f[result["mask"]], result2.f)


def check_ref_count_cleared(factor):
    if isinstance(factor, factors.CustomFactor):
        assert factor._ref_count == 0, factor
        if factor._mask is not None:
            check_ref_count_cleared(factor._mask)
    if factor.inputs:
        for upstream in factor.inputs:
            if isinstance(upstream, factors.BaseFactor):
                check_ref_count_cleared(upstream)


def test_ref_count_after_run(engine_no_calendar):
    t = factors.OHLCV.volume.top(1)
    b = factors.OHLCV.volume.bottom(1)
    engine_no_calendar.add([t, t & b, t | b, ~t], ["t", "t&b", "t|b", "~t"])
    engine_no_calendar.run("2019-01-01", "2019-01-05")
    for f in engine_no_calendar.factors.values():
        check_ref_count_cleared(f)


def test_reused_factor_computed_once(engine):
    f1 = factors.BBANDS(win=20, inputs=[close, 2]).normalized()
    f2 = factors.EMA(span=10, inputs=[f1])
    fa = factors.STDDEV(win=15, inputs=[f2])
    fb = factors.MACD(12, 26, 9, inputs=[f2])
    engine.add([f2, fa, fb], ["f2", "fa", "fb"])

    engine.run("2019-01-11", ref.END)  # prepare data, so pre_compute_ can be called
    for f in engine.factors.values():
        f.pre_compute_(engine, "2019-01-11", ref.END)
    # referenced by engine, fa, and 2 EMAs of fb
    assert f2._ref_count == 4
    for f in engine.factors.values():
        f.clean_up_(force=True)

    result = engine.run(ref.START, ref.END)
    assert f2._ref_count == 0

    # nested windows should load enough data
    def expected_f2(df):
        upper, middle, _ = ref.bbands(df.close, 20, 2)
        return ((df.close - middle) / (upper - middle)).ewm(span=10, adjust=False).mean()

    aapl, msft = ref.expected(expected_f2)
    assert_almost_equal(result.xs("AAPL", level=1).f2.values, aapl, decimal=3)
    assert_almost_equal(result.xs("MSFT", level=1).f2.values, msft, decimal=3)
    aapl, msft = ref.expected(lambda df: expected_f2(df).rolling(15).std(ddof=0))
    assert_almost_equal(result.xs("AAPL", level=1).fa.values, aapl, decimal=3)


def test_keep_cache(engine_no_calendar):
    engine = engine_no_calendar
    ma5 = factors.MA(5)
    ma5._keep_cache = True

    engine.add(ma5, "a")
    engine.run("2019-01-01", "2019-01-05")
    shape1 = ma5._cache.shape

    # cache is kept when ma5 is not used
    engine.remove_all_factors()
    engine.add(factors.MA(10), "a")
    engine.run("2019-01-01", "2019-01-05")
    assert ma5._cache.shape == shape1

    # longer history loaded by MA(10), so cache invalidated and recalculated
    engine.remove_all_factors()
    engine.add(ma5, "a")
    engine.run("2019-01-01", "2019-01-05")
    assert ma5._cache.shape[0] == shape1[0]
    assert ma5._cache.shape[1] > shape1[1]
    assert tuple(ma5._cache.shape) == (2, 13)


def test_device(engine_no_calendar, device):
    engine = engine_no_calendar
    if device == "cuda":
        engine.to_cuda()
    assert engine.device.type == device
    f1 = factors.OHLCV.volume + 1
    f2 = f1 + 1
    engine.add(f2, "f2")
    engine.add(factors.OHLCV.volume + 2, "fv")
    engine.set_filter(factors.OHLCV.volume > 1)
    result = engine.run(ref.START, ref.END)
    assert_array_equal(result.f2, result.fv)
    engine.to_cpu()
    assert engine.device.type == "cpu"


def test_get_price_matrix(engine):
    engine.set_filter(factors.StaticAssets({"MSFT"}))
    prices = engine.get_price_matrix(ref.START, ref.END)
    assert list(prices.columns) == ["MSFT"]
    msft = ref.daily_prices("MSFT").close.loc["2019-01-02" : ref.END]
    assert_array_equal(prices.MSFT.dropna().values, msft.values)


def test_full_run(engine):
    engine.add(factors.MA(5) - factors.MA(10), "ma_cross")
    with pytest.warns(RuntimeWarning, match="mask parameter not set"):
        factor_data, mean_returns = engine.full_run(
            "2018-06-01", ref.END, periods=(1, 5), quantiles=2, filter_zscore=None, preview=False
        )

    assert set(factor_data.columns) == {
        ("Demeaned", "1D"),
        ("Demeaned", "5D"),
        ("Returns", "1D"),
        ("Returns", "5D"),
        ("ma_cross", "factor"),
        ("ma_cross", "factor_quantile"),
        ("ma_cross", "factor_weight"),
    }
    # forward returns at the close of the day
    aapl = ref.daily_prices("AAPL").close
    returns = factor_data.xs("AAPL", level=1)[("Returns", "5D")]
    expected = (aapl.shift(-5) / aapl - 1).loc[returns.index.tz_localize(None)]
    assert_almost_equal(returns.values, expected.values)
    # last 5 days have no forward returns, dropped
    assert returns.index[-1] == pd.Timestamp("2019-01-07", tz="UTC")

    assert list(mean_returns.index) == [0, 1]
    assert ("ma_cross", "1D", "mean") in mean_returns.columns


@pytest.mark.filterwarnings("ignore:quantile\\(\\) mask parameter not set")
@pytest.mark.parametrize("filter_zscore", [None, 20])
def test_full_run_keeps_forward_returns_of_delisted_asset(dirty_engine, dirty_panel, filter_zscore):
    """Near the delisting, the return is to the last close, it is read from the padding slots."""
    dirty_engine.add(factors.MA(2), "ma")
    factor_data, _ = dirty_engine.full_run(
        DIRTY_DATES[5],
        DIRTY_DATES[-1],
        periods=(3,),
        quantiles=2,
        filter_zscore=filter_zscore,
        preview=False,
    )
    returns = factor_data[("Returns", "3D")].xs("DELISTED", level="asset")

    closes = dirty_panel.close.xs("DELISTED", level="asset").to_numpy()
    last = len(closes) - 1
    assert last == DELISTED_ON
    forward = [closes[min(bar + 3, last)] / closes[bar] - 1 for bar in range(last)] + [np.nan]
    assert_allclose(returns, forward[-len(returns) :], rtol=1e-5, atol=1e-6)
    assert returns.iloc[-2] == pytest.approx(closes[last] / closes[last - 1] - 1, rel=1e-5)


def out_of_memory_once(monkeypatch, cls):
    """The next `cls.compute` raises a CUDA OOM, as a tensor too big for the GPU would."""
    real_compute = cls.compute
    calls = []

    def compute(self, *inputs):
        calls.append([t.device.type for t in inputs if hasattr(t, "device")])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
        return real_compute(self, *inputs)

    monkeypatch.setattr(cls, "compute", compute)
    return calls


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
@pytest.mark.parametrize(
    "cls, build",
    [
        (factors.factor.MedianFactor, lambda: close.median()),
        (factors.factor.MedianFactor, lambda: close.median(mask=close > 100)),
        (factors.factor.DemedianFactor, lambda: close.demedian()),
        (factors.MADClampFactor, lambda: close.mad_clamp(3)),
        (factors.WinsorizingFactor, lambda: close.winsorizing(0.1)),
        (factors.IQRNormalityFactor, lambda: factors.IQRNormalityFactor(inputs=[close])),
        (factors.factor.DemeanFactor, lambda: close.demean(groupby={"FULL": 1, "SPLIT": 2})),
        (factors.StaticAssets, lambda: ~factors.StaticAssets({"FULL"})),
        (factors.factor.RankFactor, lambda: close.rank()),
        (factors.MA, lambda: factors.MA(5)),
    ],
    ids=[
        "median",
        "median_mask",
        "demedian",
        "mad_clamp",
        "winsorizing",
        "iqr_normality",
        "demean_group_dict",
        "static_assets",
        "rank",
        "rolling",
    ],
)
def test_out_of_memory_falls_back_to_cpu(dirty_engine, monkeypatch, cls, build):
    dirty_engine.add(build(), "test")
    start, end = DIRTY_DATES[0], DIRTY_DATES[-1]
    expected = dirty_engine.run(start, end, delay_factor=False)

    calls = out_of_memory_once(monkeypatch, cls)
    result = dirty_engine.run(start, end, delay_factor=False)

    assert calls[-1] == ["cpu"] * len(calls[-1])
    assert len(calls) == 2
    assert_allclose(result.test, expected.test)


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_out_of_memory_fallback_does_not_stick(dirty_engine, monkeypatch):
    start, end = DIRTY_DATES[0], DIRTY_DATES[-1]
    # the mask is in another group, so it is reverted on the device of the engine first
    dirty_engine.add(close.median(mask=close.rank() > 2), "test")
    expected = dirty_engine.run(start, end, delay_factor=False).test

    calls = out_of_memory_once(monkeypatch, factors.factor.MedianFactor)
    assert_allclose(dirty_engine.run(start, end, delay_factor=False).test, expected)
    assert_allclose(dirty_engine.run(start, end, delay_factor=False).test, expected)
    assert calls[-1] == [dirty_engine.device.type]


def test_failed_out_of_memory_fallback_does_not_stick(dirty_engine, monkeypatch):
    factor = close.median()
    dirty_engine.add(factor, "test")
    start, end = DIRTY_DATES[0], DIRTY_DATES[-1]

    def compute(self, data):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")

    with monkeypatch.context() as patch:
        patch.setattr(factors.factor.MedianFactor, "compute", compute)
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            dirty_engine.run(start, end)
    assert factor._computing_on_cpu is False
    assert not dirty_engine.run(start, end).test.isna().all()


def test_factors_reset_after_error(engine):
    """Shared factors (like OHLCV.close) should not keep the data of a failed run."""

    class Fail(factors.CustomFactor):
        inputs = [factors.MarketReturn()]

        def compute(self, market_return):
            raise ZeroDivisionError()

    engine.add(Fail(), "fail")
    with pytest.raises(ZeroDivisionError):
        engine.run("2018-01-01", ref.END)
    assert factors.OHLCV.close._data is None

    # different data shape
    engine.remove_all_factors()
    engine.add(factors.MarketReturn(), "market_return")
    result = engine.run(ref.START, ref.END)
    expected = ref.daily_prices("AAPL").close / ref.daily_prices("AAPL").close.shift(1) - 1
    assert result.xs("AAPL", level=1).market_return.iloc[6] == pytest.approx(
        expected.loc["2019-01-10"]
    )


def test_settings_invalidate_cache(engine):
    engine.add(factors.WEEKDAY, "weekday")
    utc = engine.run(ref.START, ref.END).weekday.values

    # 00:00 UTC is the day before in New York
    engine.timezone = "America/New_York"
    new_york = engine.run(ref.START, ref.END).weekday.values
    assert_array_equal(new_york, (utc - 1) % 7)

    engine.timezone = "UTC"
    assert_array_equal(engine.run(ref.START, ref.END).weekday.values, utc)

    # MSFT gets a row on 2019-01-10
    engine.align_by_time = True
    assert len(engine.run(ref.START, ref.END)) == len(utc) + 1
