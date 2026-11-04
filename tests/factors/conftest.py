import pandas as pd
import pytest

import spectre
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
