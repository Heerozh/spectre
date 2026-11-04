import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from spectre import factors

adjusted_open = factors.AdjustedColumnDataFactor(factors.OHLCV.open)
adjusted_close = factors.AdjustedColumnDataFactor(factors.OHLCV.close)


class MeanOfProduct(factors.CPUParallelFactor):
    """Must be defined at module level, so it can be pickled by multiprocessing."""

    @staticmethod
    def mp_compute(a, b) -> np.array:
        return (a * b).mean(axis=0).values


def test_column_data_not_allowed():
    with pytest.raises(ValueError, match="Cannot use ColumnDataFactor in CPUParallelFactor"):
        MeanOfProduct(inputs=[factors.OHLCV.open])


@pytest.mark.parametrize("multiprocess", [False, True])
def test_cpu_parallel_factor(engine_no_calendar, device, multiprocess):
    engine = engine_no_calendar
    if device == "cuda":
        engine.to_cuda()
    engine.add(
        MeanOfProduct(
            win=10, core=3, inputs=[adjusted_open, adjusted_close], multiprocess=multiprocess
        ),
        "f",
    )
    engine.add(factors.MA(win=10, inputs=[adjusted_open * adjusted_close]), "f2")
    result = engine.run("2018-12-25", "2019-01-05")
    assert_almost_equal(result.f.values, result.f2.values)
    assert not np.isnan(result.f.values).all()

    # one row
    result = engine.run("2019-01-04", "2019-01-05")
    assert_almost_equal(result.f.values, result.f2.values)


def test_nested_window(engine):
    # CPUParallelFactor rolls on dates (the inputs are unstacked), it's same as other factors
    # only when the data is aligned by time.
    engine.align_by_time = True
    engine.add(MeanOfProduct(win=10, core=3, inputs=[factors.MA(3), factors.MA(4)]), "f")
    engine.add(factors.MA(win=10, inputs=[factors.MA(3) * factors.MA(4)]), "f2")
    result = engine.run("2019-01-04", "2019-01-15")
    assert_almost_equal(result.f.values, result.f2.values)
