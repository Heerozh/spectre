import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_array_equal

from spectre import factors
from tests.factors import reference as ref

volume = factors.OHLCV.volume
close = factors.OHLCV.close


def assets_of(df):
    return list(df.index.get_level_values(1))


def test_filter_without_calendar(engine_no_calendar):
    """
    The asset with the highest volume of the previous trading day.
    MSFT's junk row on 2019-01-05 (nan volume) is never selected.
    """
    engine_no_calendar.add(volume, 'not_used')
    engine_no_calendar.set_filter(volume.top(1))
    result = engine_no_calendar.run(ref.START, ref.END)
    assert assets_of(result) == ['MSFT', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL',
                                 'MSFT', 'AAPL']


def test_filter_does_not_affect_inputs(engine):
    engine.set_filter(volume.top(1))
    engine.add(factors.SMA(5), 'ma5')
    result = engine.run(ref.START, ref.END)

    # dates of each asset when it has the highest volume (delayed)
    expected_dates = {'AAPL': ['2019-01-04', '2019-01-07', '2019-01-08', '2019-01-09',
                               '2019-01-10', '2019-01-11', '2019-01-15'],
                      'MSFT': ['2019-01-03', '2019-01-14']}
    for asset, dates in expected_dates.items():
        sma = ref.sma(ref.daily_prices(asset).close, 5).shift(1)
        selected = result.xs(asset, level=1).ma5
        assert [str(d.date()) for d in selected.index] == dates
        assert_almost_equal(selected.values, sma[dates].values)


@pytest.mark.parametrize('method, n, top_aapl, solo', [
    ('top', 1, True, True), ('bottom', 1, False, True),
    # only AAPL on 2019-01-10, 50% of 1 asset is 0.5, nothing selected
    ('top', 0.5, True, False), ('bottom', 0.5, False, False),
])
def test_top_bottom(engine, method, n, top_aapl, solo):
    engine.add(getattr(close, method)(n), 'test')
    result = engine.run(ref.START, ref.END)
    expected_aapl = [top_aapl] * 9
    expected_aapl[6] = solo
    assert_array_equal(result.xs('AAPL', level=1).test.values, expected_aapl)
    assert_array_equal(result.xs('MSFT', level=1).test.values, [not top_aapl] * 8)


def test_static_assets(engine):
    engine.set_filter(factors.StaticAssets({'AAPL'}))
    engine.add(close, 'close')
    df = engine.run('2018-01-01', ref.END)
    assert set(df.index.get_level_values(1)) == {'AAPL'}
    # pre-screened, only AAPL data loaded
    assert set(engine.dataframe_.index.get_level_values(1)) == {'AAPL'}


def test_static_assets_inverted(engine):
    # not pre-screened
    engine.set_filter(~factors.StaticAssets({'AAPL'}))
    engine.add(close, 'close')
    df = engine.run('2018-01-01', ref.END)
    assert set(df.index.get_level_values(1)) == {'MSFT'}


def test_static_assets_not_found(engine):
    engine.set_filter(factors.StaticAssets({'AAPL', 'NOT_EXISTS'}))
    engine.add(close, 'close')
    assert set(engine.run(ref.START, ref.END).index.get_level_values(1)) == {'AAPL'}

    engine.set_filter(factors.StaticAssets({'NOT_EXISTS'}))
    with pytest.raises(ValueError, match='was not found in DataLoader'):
        engine.run(ref.START, ref.END)


def test_filter_filter_raises():
    mask = close > 177
    with pytest.raises(ValueError, match='does not support local filtering'):
        mask.filter(mask)
    with pytest.raises(ValueError, match='does not support `.sum\\(\\)`'):
        mask.sum(2)


def test_filter_as_input_of_cross_section_factor(engine):
    class AssertOnlyOneSelected(factors.CrossSectionFactor):
        def compute(self, x) -> torch.Tensor:
            assert (x.sum(dim=1) == 1).all()
            return x

    engine.add(AssertOnlyOneSelected(inputs=[factors.StaticAssets(['AAPL'])]), 'aapl_filter')
    result = engine.run(ref.START, ref.END)
    assert result.aapl_filter.xs('AAPL', level=1).all()
    assert not result.aapl_filter.xs('MSFT', level=1).any()


def test_logic_ops(engine):
    a = close > 150
    b = factors.OHLCV.volume > 4e7
    engine.add([a, b, a & b, a | b, a ^ b, ~a], ['a', 'b', 'and', 'or', 'xor', 'not'])
    result = engine.run(ref.START, ref.END)
    assert_array_equal(result['and'], result.a & result.b)
    assert_array_equal(result['or'], result.a | result.b)
    assert_array_equal(result['xor'], result.a ^ result.b)
    assert_array_equal(result['not'], ~result.a)
    # make sure the test is not trivial
    assert result['xor'].any() and not result['xor'].all()


def test_one_hot_many_classes_warns(engine):
    engine.add(factors.OHLCV.volume.one_hot()[0], 'test')
    with pytest.warns(RuntimeWarning, match='One hot encoding with too many features'):
        # volume has more than 1000 unique values
        engine.run('2014-11-01', ref.END)
    assert np.issubdtype(engine.run(ref.START, ref.END).test.dtype, np.bool_)
