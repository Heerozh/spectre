import numpy as np
from numpy.testing import assert_almost_equal

from tests.factors.reference import START, END


def run_factor(engine, factor, start=START, end=END, delay=True, check_bias=True):
    """
    Run `factor` from `start` to `end`, returns the values of (AAPL, MSFT).
    The default range has 10 trading days, AAPL has 9 values if the factor is delayed (the
    first day is dropped), and MSFT has one less day (no data on 2019-01-10).
    """
    engine.remove_all_factors()
    engine.add(factor, 'test')
    result = engine.run(start, end, delay)
    if check_bias:
        engine.test_lookahead_bias(start, end)
    return (result.loc[(slice(None), 'AAPL'), 'test'].values,
            result.loc[(slice(None), 'MSFT'), 'test'].values)


def assert_factor(engine, factor, expected_aapl, expected_msft, decimal=7, **kwargs):
    aapl, msft = run_factor(engine, factor, **kwargs)
    assert_almost_equal(aapl, np.asarray(expected_aapl), decimal=decimal, err_msg='AAPL')
    assert_almost_equal(msft, np.asarray(expected_msft), decimal=decimal, err_msg='MSFT')


def delete(values, index):
    """MSFT has no data on 2019-01-10, remove it from the expected values of AAPL."""
    return np.delete(np.asarray(values), index)
