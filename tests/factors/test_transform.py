"""Time-series and element-wise transforms: shift, fill_na, masked_fill, ts_*, math ops..."""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_array_equal

from spectre import factors, parallel
from tests.factors import reference as ref
from tests.factors.helpers import assert_factor, delete, run_dirty, run_factor

nan = np.nan
close = factors.OHLCV.close
WEEKDAY = factors.WEEKDAY
# weekdays of the AAPL calendar from 2019-01-02 to 2019-01-15, MSFT has no data on 2019-01-10
WEEKDAYS = np.array([2.0, 3, 4, 0, 1, 2, 3, 4, 0, 1])
SOLO = 6


def weekday(df):
    return pd.Series(df.index.day_of_week, index=df.index, dtype=float)


# ------------------------------------ shift ------------------------------------


def test_shift(engine):
    # shift doesn't load more history data, so the first 2 rows of loaded data are nan
    aapl, msft = ref.expected(lambda df: df.close.shift(2))
    aapl[0:2] = nan
    msft[0:2] = nan
    assert_factor(engine, close.shift(2), aapl, msft)


def test_shift_backward_is_lookahead(engine):
    with pytest.raises(RuntimeError, match="look-ahead bias"):
        run_factor(engine, close.shift(-2))
    # shift(-2) then delayed by 1
    expected = ref.expected(lambda df: df.close.loc[: ref.END].shift(-2))
    assert_factor(engine, close.shift(-2), *expected, check_bias=False)
    assert_almost_equal(expected[1], [104.39, 103.2, 105.22, 106, 103.2, 103.39, 108.85, nan])


def test_shift_bool_raises(engine):
    with pytest.raises(ValueError, match="does not support `int` or `bool`"):
        run_factor(engine, factors.factor.RawShiftFactor(close > 150, 1))


def test_filter_shift(engine):
    # FilterFactor.shift supports bool
    aapl, msft = run_factor(engine, (close > 150).shift(1))
    expected = ref.expected(lambda df: (df.close > 150).shift(1, fill_value=False))
    # the first loaded row is filled with False
    expected[0][0] = False
    assert_array_equal(aapl, expected[0])
    assert_array_equal(msft, [False] * 8)


# ------------------------------------ fill ------------------------------------

LOCAL_FILTERED = np.array([nan, 3.0, 4.0, nan, nan, nan, 3.0, 4.0, nan, nan])


def filtered_weekday():
    """local filter, fills weekday < 3 with nan"""
    return WEEKDAY.filter(WEEKDAY >= 3)


def test_local_filter(engine):
    assert_factor(engine, filtered_weekday(), LOCAL_FILTERED, delete(LOCAL_FILTERED, SOLO))


def test_ffill(engine, device):
    if device == "cuda":
        engine.to_cuda()
    expected = np.array([nan, 3.0, 4.0, 4, 4, 4, 3.0, 4.0, 4, 4])
    assert_factor(engine, filtered_weekday().fill_na(ffill=True), expected, delete(expected, SOLO))


def test_fill_inf(engine, device):
    if device == "cuda":
        engine.to_cuda()
    # 1e10 ** 4 overflows float32 to inf, filled by nan; nan values are not filled
    factor = (1e10 ** filtered_weekday()).fill_na(nan, nan=False, inf=True) / 1e30
    expected = np.array([nan, 1.0, nan, nan, nan, nan, 1.0, nan, nan, nan])
    assert_factor(engine, factor, expected, delete(expected, SOLO), decimal=5)


def test_fill_value(engine, device):
    if device == "cuda":
        engine.to_cuda()
    expected = np.array([0, 3.0, 4.0, 0, 0, 0, 3.0, 4.0, 0, 0])
    assert_factor(
        engine, filtered_weekday().fill_na(0, nan=True, inf=True), expected, delete(expected, SOLO)
    )


def test_fill_factor(engine):
    expected = np.where(np.isnan(LOCAL_FILTERED), WEEKDAYS * 10, LOCAL_FILTERED)
    assert_factor(
        engine, filtered_weekday().fill_na(WEEKDAY * 10), expected, delete(expected, SOLO)
    )


def test_fill_na_needs_value():
    with pytest.raises(ValueError, match="must be specified"):
        WEEKDAY.fill_na()


def test_masked_fill(engine):
    mask = WEEKDAY >= 3
    expected = np.where(WEEKDAYS >= 3, WEEKDAYS * 2, WEEKDAYS)
    assert_factor(engine, WEEKDAY.masked_fill(mask, WEEKDAY * 2), expected, delete(expected, SOLO))
    expected = np.where(WEEKDAYS >= 3, -1, WEEKDAYS)
    assert_factor(engine, WEEKDAY.masked_fill(mask, -1), expected, delete(expected, SOLO))


# ------------------------------------ rolling ------------------------------------


def test_ts_prod(engine):
    expected = ref.expected(
        lambda df: weekday(df).rolling(3).apply(np.prod, raw=True), delayed=False
    )
    assert_factor(engine, WEEKDAY.ts_prod(3), *expected)
    assert_array_equal(expected[1], [0, 0, 24.0, 0, 0, 0, 8.0, 0, 0])


def test_ts_sum(engine):
    expected = ref.expected(lambda df: ref.log_returns(df.close).rolling(2).sum())
    assert_factor(engine, factors.LogReturns().ts_sum(2), *expected)


def test_ts_prod_log(engine):
    expected = ref.expected(lambda df: np.log(df.close) + np.log(df.close.shift(1)))
    assert_factor(engine, close.ts_prod(2).log(), *expected)


def test_ts_max_min(engine):
    assert_factor(engine, close.ts_max(3), *ref.expected(lambda df: df.close.rolling(3).max()))
    assert_factor(engine, close.ts_min(3), *ref.expected(lambda df: df.close.rolling(3).min()))


ROLLING_DATA = [[nan, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]


def test_ts_count():
    data = parallel.Rolling(torch.tensor(ROLLING_DATA), win=3)
    result = factors.CustomFactor().ts_count(3).compute(data)
    assert_almost_equal(result, [[0, 1, 2, 3, 3, 3], [1, 2, 3, 3, 3, 3]])


def test_ts_any_all():
    data = parallel.Rolling(torch.tensor(ROLLING_DATA), win=3)
    result = factors.CustomFactor().ts_any(3).compute(data)
    assert_array_equal(result, [[False] + [True] * 5, [True] * 6])
    result = factors.CustomFactor().ts_all(3).compute(data)
    assert_array_equal(result, [[False] * 3 + [True] * 3, [False] * 2 + [True] * 4])


def test_rolling_rank(engine):
    def rolling_rank(df):
        return weekday(df).rolling(5).apply(lambda x: pd.Series(x).rank().iloc[-1] / 5, raw=True)

    expected = ref.expected(rolling_rank, delayed=False)
    assert_factor(engine, factors.RollingRankFactor(5, inputs=[WEEKDAY]), *expected)


def test_rolling_rank_flat_window(device):
    prices = [1.0, 2, 2, 2, 2, 5, 5]
    rolling = parallel.Rolling(torch.tensor([prices], device=device), win=3)
    result = factors.RollingRankFactor(3).compute(rolling).cpu().numpy()
    expected = pd.Series(prices).rolling(3, min_periods=1).apply(lambda x: x.rank().iloc[-1] / 3)
    assert_almost_equal(result[0], expected, decimal=6)


def test_rolling_zscore(engine):
    def rolling_zscore(df):
        r = df.close.rolling(5)
        return (df.close - r.mean()) / r.std(ddof=0)

    assert_factor(
        engine, factors.RollingZScoreFactor(5, inputs=[close]), *ref.expected(rolling_zscore)
    )


# ------------------------------------ math ------------------------------------


def test_log(engine):
    with np.errstate(divide="ignore"):
        expected = np.log(WEEKDAYS)
    assert_factor(engine, WEEKDAY.log(), expected, delete(expected, SOLO))


@pytest.mark.parametrize("method, func", [("abs", np.abs), ("sign", np.sign), ("tanh", np.tanh)])
def test_math(engine, method, func):
    expected = ref.expected(lambda df: func(ref.returns(df.close)))
    assert_factor(engine, getattr(factors.Returns(), method)(), *expected, decimal=6)


def test_sign_keeps_nan(dirty_engine, dirty_panel):
    returns = dirty_panel.clean_close.groupby(level="asset").pct_change(fill_method=None)
    result = run_dirty(dirty_engine, factors.Returns().sign())
    assert_array_equal(result, np.sign(returns))


def test_round(engine):
    expected = np.round(np.array([2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 11.0, 14.0, 15.0]) / 1000, 2)
    factor = (factors.DatetimeDataFactor("day") / 1000).round(2)
    assert_factor(engine, factor, expected, delete(expected, SOLO))


@pytest.mark.parametrize(
    "method, dtype", [("float", "float32"), ("double", "float64"), ("half", "float16")]
)
def test_type_cast(engine, method, dtype):
    engine.add(getattr(close, method)(), "test")
    result = engine.run(ref.START, ref.END)
    assert result.test.dtype == dtype
