import pandas as pd
import pytest

import spectre
from tests.conftest import (
    DIRTY_ADJUSTMENTS,
    DIRTY_DATES,
    DIRTY_OHLCV,
    NANSPLIT_ON,
    make_dirty_panel,
)
from tests.factors.helpers import LATE_CLOSES, LATE_DATES


@pytest.fixture
def late_listing_engine():
    """A has 12 days of data, B is listed on the 6th day."""
    rows = [(date, "A", 100.0 + i) for i, date in enumerate(LATE_DATES)]
    rows += [(date, "B", close) for date, close in zip(LATE_DATES[5:], LATE_CLOSES)]
    df = pd.DataFrame(rows, columns=["date", "asset", "close"])
    df = df.set_index(["date", "asset"]).sort_index()
    for col in ("open", "high", "low"):
        df[col] = df.close
    df["volume"] = 1.0
    ohlcv = ("open", "high", "low", "close", "volume")
    loader = spectre.data.MemoryLoader(df[list(ohlcv)], ohlcv=ohlcv)
    return spectre.factors.FactorEngine(loader)


@pytest.fixture(scope="session")
def added_rows_panel():
    """The dirty panel, where NANSPLIT has no rows on the 3 bars before its split."""
    gap = [(date, "NANSPLIT") for date in DIRTY_DATES[NANSPLIT_ON - 3 : NANSPLIT_ON]]
    return make_dirty_panel().drop(index=gap)


@pytest.fixture
def added_rows_engine(added_rows_panel, device):
    """
    An engine with align_by_time on `added_rows_panel`. It adds the missing rows after the loader
    made the multipliers, so these rows have NaN multipliers: before the listing of LISTED, after
    the end of DELISTED, in the halt of HALTED, and between a bar of NANSPLIT and its split.
    """
    loader = spectre.data.MemoryLoader(added_rows_panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)
    engine = spectre.factors.FactorEngine(loader)
    engine.align_by_time = True
    if device == "cuda":
        engine.to_cuda()
    return engine
