import operator

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from spectre import factors
from tests.factors.helpers import run_dirty

# delayed close from 2019-01-01 to 2019-01-05 without calendar alignment:
# AAPL 01-03, MSFT 01-03, AAPL 01-04, MSFT 01-04, MSFT 01-05 (junk row)
F = np.array([158.61, 101.30, 145.23, 102.28, 104.39])
F2 = F**2


@pytest.fixture(scope="module")
def result(daily_loader_no_calendar):
    engine = factors.FactorEngine(daily_loader_no_calendar)
    f = factors.OHLCV.close
    f2 = f**2
    t = factors.OHLCV.volume.top(1)
    b = factors.OHLCV.volume.bottom(1)
    columns = {
        "f": f,
        "f^2": f2,
        "-f": -f,
        "f+f2": f + f2,
        "f-f2": f - f2,
        "f*f2": f * f2,
        "f/f2": f / f2,
        "f%f2": f % f2,
        "f>f2": f > f2,
        "f<f2": f < f2,
        "f>=f2": f >= f2,
        "f<=f2": f <= f2,
        "f==f2": f == f2,
        "f!=f2": f != f2,
        "2+f": 2 + f,
        "2-f": 2 - f,
        "2*f": 2 * f,
        "2/rf2": 2 / (2**f),
        "2>f": 2 > f,
        "2<f": 2 < f,
        "2>=f": 2 >= f,
        "2<=f": 2 <= f,
        "2==f": 2 == f,
        "2!=f": 2 != f,
        "t": t,
        "t&b": t & b,
        "t|b": t | b,
        "~t": ~t,
    }
    engine.add(list(columns.values()), list(columns.keys()))
    return engine.run("2019-01-01", "2019-01-05")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("f", F),
        ("f^2", F2),
        ("-f", -F),
        ("f+f2", F + F2),
        ("f-f2", F - F2),
        ("f*f2", F * F2),
        ("f/f2", F / F2),
        ("f%f2", F % F2),
        ("2+f", 2 + F),
        ("2-f", 2 - F),
        ("2*f", 2 * F),
    ],
)
def test_arithmetic(result, name, expected):
    assert_array_equal(result[name], expected)


def test_reverse_pow(result):
    assert_almost_equal(result["2/rf2"], 2 / (2**F))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("f>f2", F > F2),
        ("f<f2", F < F2),
        ("f>=f2", F >= F2),
        ("f<=f2", F <= F2),
        ("f==f2", F == F2),
        ("f!=f2", F != F2),
        ("2>f", 2 > F),
        ("2<f", 2 < F),
        ("2>=f", 2 >= F),
        ("2<=f", 2 <= F),
        ("2==f", 2 == F),
        ("2!=f", 2 != F),
    ],
)
def test_compare(result, name, expected):
    assert_array_equal(result[name], expected)


def test_logic(result):
    # highest volume of the previous day: MSFT on 01-02, AAPL on 01-03, AAPL on 01-04
    t = np.array([False, True, True, False, False])
    b = ~t
    assert_array_equal(result["t"], t)
    assert_array_equal(result["t&b"], t & b)
    assert_array_equal(result["t|b"], t | b)
    assert_array_equal(result["~t"], b)


def test_unsupported_ops():
    f = factors.OHLCV.close
    with pytest.raises(TypeError):
        operator.mod(2.0, f)
    b = factors.OHLCV.volume.bottom(1)
    with pytest.raises(TypeError):
        operator.and_(2.0, b)
    with pytest.raises(TypeError):
        operator.or_(2.0, b)
    with pytest.raises(TypeError):
        operator.xor(2.0, b)
    with pytest.raises(TypeError):
        operator.mod(2.0, b)


# ------------------------------------ filters ------------------------------------
# In arithmetic a filter is 0 or 1, as a float. Each expression below runs on the filters, on
# filter.float() and on pandas, so all of them are the same expression.

close = factors.OHLCV.close
volume = factors.OHLCV.volume

FILTER_PAIRS = {
    "mixed": (lambda: (close > 100, volume > 1500), lambda df: (df.close > 100, df.volume > 1500)),
    "true_and_false": (lambda: (volume > 0, close < 0), lambda df: (df.volume > 0, df.close < 0)),
}


@pytest.fixture(params=FILTER_PAIRS)
def filters(request, dirty_panel):
    """Two filters, and their values in pandas."""
    factor, values = FILTER_PAIRS[request.param]
    return factor(), values(dirty_panel)


@pytest.fixture(params=[False, True], ids=["as_filter", "as_float"])
def cast(request):
    return (lambda factor: factor.float()) if request.param else (lambda factor: factor)


def run_operands(engine, panel, filters, cast, expression):
    """Returns the expression on the engine, and on pandas."""
    (f, g), (x, y) = filters
    result = run_dirty(engine, expression(cast(f), cast(g), close))
    return result, expression(x.astype(float), y.astype(float), panel.close)


ARITHMETIC = {
    "f + g": lambda f, g, c: f + g,
    "f + f": lambda f, g, c: f + f,
    "f + 1": lambda f, g, c: f + 1,
    "1 + f": lambda f, g, c: 1 + f,
    "f + 0.5": lambda f, g, c: f + 0.5,
    "f + c": lambda f, g, c: f + c,
    "c + f": lambda f, g, c: c + f,
    "f - g": lambda f, g, c: f - g,
    "f - f": lambda f, g, c: f - f,
    "f - 1": lambda f, g, c: f - 1,
    "1 - f": lambda f, g, c: 1 - f,
    "f - c": lambda f, g, c: f - c,
    "c - f": lambda f, g, c: c - f,
    "f * g": lambda f, g, c: f * g,
    "f * 2": lambda f, g, c: f * 2,
    "2 * f": lambda f, g, c: 2 * f,
    "f * c": lambda f, g, c: f * c,
    "c * f": lambda f, g, c: c * f,
    "f / g": lambda f, g, c: f / g,
    "f / 2": lambda f, g, c: f / 2,
    "1 / f": lambda f, g, c: 1 / f,
    "f / c": lambda f, g, c: f / c,
    "c / f": lambda f, g, c: c / f,
    "f % g": lambda f, g, c: f % g,
    "f % 2": lambda f, g, c: f % 2,
    "f % c": lambda f, g, c: f % c,
    "c % f": lambda f, g, c: c % f,
    "f ** g": lambda f, g, c: f**g,
    "f ** 2": lambda f, g, c: f**2,
    "2 ** f": lambda f, g, c: 2**f,
    "f ** c": lambda f, g, c: f**c,
    "c ** f": lambda f, g, c: c**f,
    "-f": lambda f, g, c: -f,
}


@pytest.mark.parametrize("expression", ARITHMETIC.values(), ids=ARITHMETIC)
def test_arithmetic_of_filters(dirty_engine, dirty_panel, filters, cast, expression):
    result, expected = run_operands(dirty_engine, dirty_panel, filters, cast, expression)
    assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    assert result.dtype.kind == "f"


def by_date(x):
    return x.groupby(level="date")


def by_asset(x):
    return x.groupby(level="asset")


# A missing row must stay out of what is calculated from an expression of filters.
OF_ARITHMETIC = {
    "mean": (lambda f, g: (f + g).mean(), lambda x, y: by_date(x + y).transform("mean")),
    "std": (lambda f, g: (f - g).std(), lambda x, y: by_date(x - y).transform("std", ddof=0)),
    "xs_sum": (lambda f, g: (f + g).xs_sum(), lambda x, y: by_date(x + y).transform("sum")),
    "min": (lambda f, g: (f + 1).min(), lambda x, y: by_date(x + 1).transform("min")),
    "max": (lambda f, g: (-f).max(), lambda x, y: by_date(-x).transform("max")),
    "rank": (lambda f, g: (f + g).rank(), lambda x, y: by_date(x + y).rank()),
    "mean_by_asset": (
        lambda f, g: (f + g).mean(groupby="asset"),
        lambda x, y: by_asset(x + y).transform("mean"),
    ),
    "min_by_asset": (
        lambda f, g: (f * 2).min(groupby="asset"),
        lambda x, y: by_asset(x * 2).transform("min"),
    ),
}


@pytest.mark.parametrize("helper, reference", OF_ARITHMETIC.values(), ids=OF_ARITHMETIC)
def test_helpers_of_filter_arithmetic(dirty_engine, filters, cast, helper, reference):
    (f, g), (x, y) = filters
    result = run_dirty(dirty_engine, helper(cast(f), cast(g)))
    assert_allclose(result, reference(x.astype(float), y.astype(float)), rtol=1e-5, atol=1e-6)


def test_sum_of_filters_counts_both(dirty_engine, filters, cast):
    (f, g), (x, y) = filters
    result = run_dirty(dirty_engine, (cast(f) + cast(g)) >= 2)
    assert_array_equal(result, x & y)


KEPT = {
    "f & g": (lambda f, g: f & g, lambda x, y: x & y),
    "f | g": (lambda f, g: f | g, lambda x, y: x | y),
    "f ^ g": (lambda f, g: f ^ g, lambda x, y: x ^ y),
    "~f": (lambda f, g: ~f, lambda x, y: ~x),
    "f == g": (lambda f, g: f == g, lambda x, y: x == y),
    "f != g": (lambda f, g: f != g, lambda x, y: x != y),
    "f > g": (lambda f, g: f > g, lambda x, y: x > y),
    "f <= g": (lambda f, g: f <= g, lambda x, y: x <= y),
    "f == 1": (lambda f, g: f == 1, lambda x, y: x),
    # reflected on purpose, see test_compare
    "1 == f": (lambda f, g: 1 == f, lambda x, y: x),
    "1 != f": (lambda f, g: 1 != f, lambda x, y: ~x),
    "1 > f": (lambda f, g: 1 > f, lambda x, y: ~x),
    "1 <= f": (lambda f, g: 1 <= f, lambda x, y: x),
}


@pytest.mark.parametrize("build, reference", KEPT.values(), ids=KEPT)
def test_logic_and_compare_of_filters_stay_bool(dirty_engine, filters, build, reference):
    (f, g), (x, y) = filters
    result = run_dirty(dirty_engine, build(f, g))
    assert_array_equal(result, reference(x, y))
    assert result.dtype == "bool"


def test_arithmetic_does_not_cast_other_factors():
    expression = factors.OHLCV.close + 1
    assert expression.inputs[0] is factors.OHLCV.close
