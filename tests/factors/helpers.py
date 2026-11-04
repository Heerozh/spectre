import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from tests.factors.reference import END, START


def run_factor(engine, factor, start=START, end=END, delay=True, check_bias=True):
    """
    Run `factor` from `start` to `end`, returns the values of (AAPL, MSFT).
    The default range has 10 trading days, AAPL has 9 values if the factor is delayed (the
    first day is dropped), and MSFT has one less day (no data on 2019-01-10).
    """
    engine.remove_all_factors()
    engine.add(factor, "test")
    result = engine.run(start, end, delay)
    if check_bias:
        engine.test_lookahead_bias(start, end)
    return (
        result.loc[(slice(None), "AAPL"), "test"].values,
        result.loc[(slice(None), "MSFT"), "test"].values,
    )


def assert_factor(engine, factor, expected_aapl, expected_msft, decimal=7, **kwargs):
    aapl, msft = run_factor(engine, factor, **kwargs)
    assert_almost_equal(aapl, np.asarray(expected_aapl), decimal=decimal, err_msg="AAPL")
    assert_almost_equal(msft, np.asarray(expected_msft), decimal=decimal, err_msg="MSFT")


def delete(values, index):
    """MSFT has no data on 2019-01-10, remove it from the expected values of AAPL."""
    return np.delete(np.asarray(values), index)


# data of the `late_listing_engine` fixture
LATE_DATES = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
# closes of B, which is listed on 2020-01-06
LATE_CLOSES = [10.0, 20, 11, 30, 12, 13, 14]


def run_late_listed(engine, factor):
    """Values of B from 2020-01-06 (listing day) to the end, not delayed."""
    engine.add(factor, "test")
    result = engine.run(LATE_DATES[4], LATE_DATES[-1], delay_factor=False)
    return result.xs("B", level=1).test.values
