import pandas as pd
import pytest

from spectre import trading

TZ = "America/New_York"
DAILY_EVENTS = {"Open": "9:00:00", "Close": "15:00:00"}


def ts(s):
    return pd.Timestamp(s, tz=TZ)


@pytest.fixture
def calendar():
    """Calendar built on Friday 2020-01-03 after the market closed."""
    cal = trading.Calendar()
    now = ts("2020-01-03 16:00")
    cal.hr_now = lambda: now
    cal.build(start="2020-01-03", end="2020-01-17", daily_events=DAILY_EVENTS, tz=TZ)
    return cal


def test_build_pops_passed_events(calendar):
    # Friday's events have passed, next is Monday
    assert calendar.events["Open"][0] == ts("2020-01-06 09:00")
    assert calendar.events["Close"][0] == ts("2020-01-06 15:00")
    # business days only
    assert calendar.events["Open"][1] == ts("2020-01-07 09:00")
    assert calendar.events["Open"][-1] == ts("2020-01-17 09:00")
    assert len(calendar.events["Open"]) == 10


def test_build_without_pop_passed():
    cal = trading.Calendar()
    cal.hr_now = lambda: ts("2020-01-03 16:00")
    cal.build(
        start="2020-01-03", end="2020-01-17", daily_events=DAILY_EVENTS, tz=TZ, pop_passed=False
    )
    assert cal.events["Open"][0] == ts("2020-01-03 09:00")


def test_holiday_and_pop_passed(calendar):
    calendar.set_as_holiday(ts("2020-01-08"))
    assert ts("2020-01-08 09:00") not in calendar.events["Open"]
    assert ts("2020-01-08 15:00") not in calendar.events["Close"]

    # Tuesday after open, Monday and Tuesday open passed, Wednesday is holiday
    calendar.hr_now = lambda: ts("2020-01-07 10:00")
    calendar.pop_passed("Open")
    assert calendar.events["Open"][0] == ts("2020-01-09 09:00")
    # other events are untouched
    assert calendar.events["Close"][0] == ts("2020-01-06 15:00")


def test_build_empty_range():
    cal = trading.Calendar()
    with pytest.raises(ValueError, match="Empty date range"):
        cal.build(start="2020-01-03", end="2019", daily_events=DAILY_EVENTS, tz=TZ)
