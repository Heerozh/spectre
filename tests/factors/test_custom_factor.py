import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from spectre import factors


def test_total_backwards():
    a = factors.CustomFactor(win=2)
    b = factors.CustomFactor(win=3, inputs=(a,))
    c = factors.CustomFactor(win=3, inputs=(b,))
    # (2 - 1) + (3 - 1) + (3 - 1)
    assert c.get_total_backwards_() == 5
    # mask has a longer window
    c.set_mask(factors.CustomFactor(win=10))
    assert c.get_total_backwards_() == 9

    a1 = factors.CustomFactor(win=10)
    a2 = factors.CustomFactor(win=5)
    b1 = factors.CustomFactor(win=20, inputs=(a1, a2))
    b2 = factors.CustomFactor(win=100, inputs=(a2,))
    c1 = factors.CustomFactor(win=100, inputs=(b1,))
    assert a1.get_total_backwards_() == 9
    assert a2.get_total_backwards_() == 4
    assert b1.get_total_backwards_() == 28
    assert b2.get_total_backwards_() == 103
    assert c1.get_total_backwards_() == 127


def test_invalid_win():
    with pytest.raises(AssertionError, match='must be a integer'):
        factors.CustomFactor(win=1.5)
    with pytest.raises(AssertionError, match='must be a list'):
        factors.CustomFactor(inputs=factors.OHLCV.close)


class Arange(factors.CustomFactor):
    inputs = [factors.OHLCV.open]

    def compute(self, data):
        # 0, 1, 2, ... by the order of [asset, time]
        return torch.tensor(np.arange(data.nelement()).reshape(data.shape))


class WrongShape(factors.CustomFactor):
    inputs = []

    def compute(self):
        return torch.tensor([1])


def test_wrong_return_shape(engine_no_calendar):
    engine_no_calendar.add(WrongShape(), 'test2')
    with pytest.raises(ValueError, match="The return data shape.*test2.*"):
        engine_no_calendar.run('2019-01-11', '2019-01-15', False)


def test_duplicate_name(engine_no_calendar):
    engine_no_calendar.add(Arange(), 'test1')
    with pytest.raises(KeyError, match=".*exists.*"):
        engine_no_calendar.add(Arange(), 'test1')
    engine_no_calendar.add(Arange(), 'test1', replace=True)


def test_inheritance(engine_no_calendar):
    engine = engine_no_calendar
    arange = Arange()

    class CumSum(factors.CustomFactor):
        inputs = [arange]

        def compute(self, data):
            return torch.tensor(np.cumsum(data.numpy(), axis=1))

    engine.add(arange, 'test1')
    engine.add(CumSum(), 'test2')

    # need data to call pre_compute_
    engine.run('2019-01-11', '2019-01-15', delay_factor=False)
    for f in engine.factors.values():
        f.pre_compute_(engine, '2019-01-11', '2019-01-15')
    assert arange._ref_count == 2
    for f in engine.factors.values():
        f.clean_up_(force=True)

    df = engine.run('2019-01-11', '2019-01-15', delay_factor=False)
    assert arange._ref_count == 0
    # 3 days of AAPL (0, 1, 2) and MSFT (3, 4, 5), sorted by date
    assert_array_equal(df['test1'].values, [0, 3, 1, 4, 2, 5])
    assert_array_equal(df['test2'].values, [0, 3, 1, 7, 3, 12])


def test_multiple_returns_with_window(engine):
    bbands = factors.BBANDS()
    with pytest.raises(ValueError, match='has multiple outputs'):
        engine.add(factors.SMA(2, inputs=[bbands]), 'test')
        engine.run('2019-01-11', '2019-01-15')
    with pytest.raises(KeyError, match='only one return value'):
        engine.remove_all_factors()
        engine.add(factors.SMA(2)[0], 'test')
        engine.run('2019-01-11', '2019-01-15')
