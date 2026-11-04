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


def test_pop_passed_with_no_events_left(calendar):
    calendar.hr_now = lambda: ts("2020-02-03 12:00")
    assert calendar.pop_passed("Open") is calendar
    assert calendar.events["Open"] == []
    # and when there was nothing to pop
    calendar.pop_passed("Open")
    assert calendar.events["Open"] == []
    # the other event is untouched
    assert len(calendar.events["Close"]) == 10


def test_today_next_with_no_events_left(calendar):
    calendar.hr_now = lambda: ts("2020-01-17 12:00")
    calendar.pop_passed("Open").pop_passed("Close")
    assert calendar.events["Open"] == []
    assert calendar.today_next() == {"Close": ts("2020-01-17 15:00")}


def test_build_with_every_event_passed():
    cal = trading.Calendar()
    cal.hr_now = lambda: ts("2020-02-03 12:00")
    cal.build(start="2020-01-03", end="2020-01-17", daily_events=DAILY_EVENTS, tz=TZ)
    assert cal.events == {"Open": [], "Close": []}
    assert cal.today_next() == {}


def test_cn_calendar_skips_2026_holidays():
    # `pd.date_range` is empty if the end is before the start, so a wrong year in the table of
    # `CNCalendar.closed` removes the holidays silently
    cal = trading.CNCalendar(start=pd.Timestamp("2026-09-01"), pop_passed=False)
    opens = cal.events["Open"]

    def open_on(date):
        return pd.Timestamp(f"{date} 09:30", tz="Asia/Shanghai")

    assert open_on("2026-09-30") in opens
    assert open_on("2026-10-01") not in opens
    assert open_on("2026-10-07") not in opens
    assert open_on("2026-10-08") in opens


def test_cn_calendar_says_when_its_table_needs_an_update():
    with pytest.raises(AssertionError, match="CNCalendar.closed ends in .* yearly update"):
        trading.CNCalendar(start=pd.Timestamp("2099-01-01"))


# fmt: off
# As listed by JPX: https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html
# It shows the current and the next year, so 2025 is from a copy of that page archived in 2025.
JPX_CLOSED = {
    2025: [
        "01-01", "01-02", "01-03", "01-13", "02-11", "02-23", "02-24", "03-20", "04-29", "05-03",
        "05-04", "05-05", "05-06", "07-21", "08-11", "09-15", "09-23", "10-13", "11-03", "11-23",
        "11-24", "12-31",
    ],
    2026: [
        "01-01", "01-02", "01-03", "01-12", "02-11", "02-23", "03-20", "04-29", "05-03", "05-04",
        "05-05", "05-06", "07-20", "08-11", "09-21", "09-22", "09-23", "10-12", "11-03", "11-23",
        "12-31",
    ],
    2027: [
        "01-01", "01-02", "01-03", "01-11", "02-11", "02-23", "03-21", "03-22", "04-29", "05-03",
        "05-04", "05-05", "07-19", "08-11", "09-20", "09-23", "10-11", "11-03", "11-23", "12-31",
    ],
}
# fmt: on


@pytest.fixture(scope="module")
def jp_calendar():
    """A fixed start, it doesn't depend on the year the tests run."""
    return trading.JPCalendar(start=pd.Timestamp("2025-01-01"), pop_passed=False)


@pytest.mark.parametrize("year", JPX_CLOSED)
def test_jp_calendar_is_open_on_every_weekday_jpx_does_not_close(jp_calendar, year):
    opens = pd.DatetimeIndex(jp_calendar.events["Open"]).tz_localize(None)

    closed = pd.to_datetime([f"{year}-{day}" for day in JPX_CLOSED[year]])
    expected = pd.bdate_range(f"{year}-01-01", f"{year}-12-31").difference(closed)
    assert list(opens[opens.year == year].normalize()) == list(expected)


def test_jp_calendar_removes_all_daily_events_of_a_closed_day(jp_calendar):
    for event, time in (("Open", "09:00"), ("Lunch", "11:30"), ("Close", "15:00")):
        assert pd.Timestamp(f"2026-11-05 {time}", tz="Asia/Tokyo") in jp_calendar.events[event]
        assert pd.Timestamp(f"2026-11-23 {time}", tz="Asia/Tokyo") not in jp_calendar.events[event]


def test_jp_calendar_says_when_its_table_needs_an_update():
    with pytest.raises(AssertionError, match="needs its yearly update"):
        trading.JPCalendar(start=pd.Timestamp("2099-01-01"))


@pytest.mark.parametrize("calendar_type", [trading.CNCalendar, trading.JPCalendar])
def test_closures_are_removed_in_one_pass(monkeypatch, calendar_type):
    """A pass goes over every event of every day, one for each closure took seconds."""
    passes = []
    remove_events = trading.Calendar.remove_events

    def counted(self, *dates):
        passes.append(dates)
        remove_events(self, *dates)

    monkeypatch.setattr(trading.Calendar, "remove_events", counted)
    calendar_type(start=pd.Timestamp("2026-11-02"), pop_passed=False)
    assert len(passes) == 1


@pytest.mark.parametrize(
    "calendar_type, start, tz",
    [
        (trading.CNCalendar, "2026-09-01", "Asia/Shanghai"),
        (trading.JPCalendar, "2027-09-01", "Asia/Tokyo"),
    ],
)
def test_closures_leave_what_one_holiday_at_a_time_leaves(calendar_type, start, tz):
    cal = calendar_type(start=pd.Timestamp(start), pop_passed=False)

    expected = trading.Calendar()
    expected.build(
        start=start,
        end=str(calendar_type.closed[-1].year + 1),
        daily_events={"Open": calendar_type.daily_events["Open"]},
        tz=tz,
        pop_passed=False,
    )
    for day in calendar_type.closed:
        expected.set_as_holiday(day.tz_localize(tz))

    assert cal.events["Open"] == expected.events["Open"]
    assert len(cal.events["Open"]) < 100


def test_removing_several_days_at_once(calendar):
    calendar.set_as_holiday(ts("2020-01-08"))
    calendar.remove_events(ts("2020-01-09"), ts("2020-01-10"))
    # the rest of the week is gone, and the next one is not
    opens = calendar.events["Open"]
    assert opens[:3] == [ts("2020-01-06 09:00"), ts("2020-01-07 09:00"), ts("2020-01-13 09:00")]
    assert len(calendar.events["Close"]) == 7
