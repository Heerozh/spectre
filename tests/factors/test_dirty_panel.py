"""Checks the `dirty_panel` fixture itself, other tests rely on these defects being there."""

import numpy as np
from numpy.testing import assert_allclose

from spectre import factors
from tests.conftest import (
    DELISTED_ON,
    DIRTY_DATES,
    DIVIDEND_ON,
    HALTED_BARS,
    LISTED_ON,
    NAN_BAR,
    NANSPLIT_ON,
    SPLIT_ON,
)


def test_defects(dirty_panel):
    bars = len(DIRTY_DATES)
    counts = dirty_panel.groupby(level="asset").size()
    assert counts["FULL"] == bars
    assert counts["LISTED"] == bars - LISTED_ON
    assert counts["DELISTED"] == DELISTED_ON + 1
    assert counts["HALTED"] == bars - len(DIRTY_DATES[HALTED_BARS])

    closes = dirty_panel.close.unstack("asset")
    assert closes["FULL"].notna().all()
    assert closes["NANBAR"].isna().sum() == 1
    assert np.isnan(closes["NANBAR"].iloc[NAN_BAR])
    assert np.isnan(closes["NANSPLIT"].iloc[NANSPLIT_ON - 1])
    assert not np.isnan(dirty_panel.volume.unstack("asset")["NANBAR"].iloc[NAN_BAR])


def test_clean_close_gives_the_scale_of_each_bar(dirty_panel):
    split = dirty_panel.xs("SPLIT", level="asset")
    scale = split.close / split.clean_close
    last_close = split.close.iloc[DIVIDEND_ON - 1]
    assert_allclose(scale.iloc[:SPLIT_ON], 1)
    assert_allclose(scale.iloc[SPLIT_ON:DIVIDEND_ON], 0.5)
    assert_allclose(scale.iloc[DIVIDEND_ON:], 0.5 * (1 - 0.5 / last_close))


def test_engine_reads_every_row(dirty_engine, dirty_panel):
    dirty_engine.add(factors.OHLCV.close, "close")
    result = dirty_engine.run(DIRTY_DATES[0], DIRTY_DATES[-1], delay_factor=False)
    assert len(result) == len(dirty_panel)
    assert_allclose(result.close.values, dirty_panel.close.values, rtol=1e-6)
