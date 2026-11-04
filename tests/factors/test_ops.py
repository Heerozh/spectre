import operator

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from spectre import factors

# delayed close from 2019-01-01 to 2019-01-05 without calendar alignment:
# AAPL 01-03, MSFT 01-03, AAPL 01-04, MSFT 01-04, MSFT 01-05 (junk row)
F = np.array([158.61, 101.30, 145.23, 102.28, 104.39])
F2 = F**2


@pytest.fixture(scope='module')
def result(daily_loader_no_calendar):
    engine = factors.FactorEngine(daily_loader_no_calendar)
    f = factors.OHLCV.close
    f2 = f**2
    t = factors.OHLCV.volume.top(1)
    b = factors.OHLCV.volume.bottom(1)
    columns = {
        'f': f,
        'f^2': f2,
        '-f': -f,
        'f+f2': f + f2,
        'f-f2': f - f2,
        'f*f2': f * f2,
        'f/f2': f / f2,
        'f%f2': f % f2,
        'f>f2': f > f2,
        'f<f2': f < f2,
        'f>=f2': f >= f2,
        'f<=f2': f <= f2,
        'f==f2': f == f2,
        'f!=f2': f != f2,
        '2+f': 2 + f,
        '2-f': 2 - f,
        '2*f': 2 * f,
        '2/rf2': 2 / (2**f),
        '2>f': 2 > f,
        '2<f': 2 < f,
        '2>=f': 2 >= f,
        '2<=f': 2 <= f,
        '2==f': 2 == f,
        '2!=f': 2 != f,
        't': t,
        't&b': t & b,
        't|b': t | b,
        '~t': ~t,
    }
    engine.add(list(columns.values()), list(columns.keys()))
    return engine.run('2019-01-01', '2019-01-05')


@pytest.mark.parametrize(
    'name, expected',
    [
        ('f', F),
        ('f^2', F2),
        ('-f', -F),
        ('f+f2', F + F2),
        ('f-f2', F - F2),
        ('f*f2', F * F2),
        ('f/f2', F / F2),
        ('f%f2', F % F2),
        ('2+f', 2 + F),
        ('2-f', 2 - F),
        ('2*f', 2 * F),
    ],
)
def test_arithmetic(result, name, expected):
    assert_array_equal(result[name], expected)


def test_reverse_pow(result):
    assert_almost_equal(result['2/rf2'], 2 / (2**F))


@pytest.mark.parametrize(
    'name, expected',
    [
        ('f>f2', F > F2),
        ('f<f2', F < F2),
        ('f>=f2', F >= F2),
        ('f<=f2', F <= F2),
        ('f==f2', F == F2),
        ('f!=f2', F != F2),
        ('2>f', 2 > F),
        ('2<f', 2 < F),
        ('2>=f', 2 >= F),
        ('2<=f', 2 <= F),
        ('2==f', 2 == F),
        ('2!=f', 2 != F),
    ],
)
def test_compare(result, name, expected):
    assert_array_equal(result[name], expected)


def test_logic(result):
    # highest volume of the previous day: MSFT on 01-02, AAPL on 01-03, AAPL on 01-04
    t = np.array([False, True, True, False, False])
    b = ~t
    assert_array_equal(result['t'], t)
    assert_array_equal(result['t&b'], t & b)
    assert_array_equal(result['t|b'], t | b)
    assert_array_equal(result['~t'], b)


def test_unsupported_ops():
    f = factors.OHLCV.close
    with pytest.raises(TypeError):
        operator.mod(2.0, f)
    b = factors.OHLCV.volume.bottom(1)
    with pytest.raises(TypeError):
        operator.and_(2.0, b)
    with pytest.raises(TypeError):
        operator.or_(2.0, b)
