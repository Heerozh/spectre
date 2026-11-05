from types import SimpleNamespace

import pandas as pd
import pytest

from spectre import trading


class Receiver(trading.EventReceiver):
    """Counts fired events, stops the event manager when EveryBarData received."""

    def __init__(self):
        super().__init__()
        self.fired = 0

    def on_run(self):
        self.schedule(trading.event.Always(self.on_always))
        self.schedule(trading.event.EveryBarData(self.on_every_bar))

    def on_always(self, _):
        self.fired += 1

    def on_every_bar(self, _):
        self.fired += 1
        if self.fired == 2:
            self.stop_event_manager()


class BarDataFirer(trading.EventReceiver):
    def on_run(self):
        self.schedule(trading.event.Always(self.on_always))

    def on_always(self, _):
        self.fire_event(trading.event.EveryBarData)


def test_unsubscribe():
    rcv = Receiver()
    evt_mgr = trading.EventManager()
    evt_mgr.subscribe(rcv)
    rcv.unsubscribe()
    assert len(evt_mgr._subscribers) == 0

    with pytest.raises(ValueError, match="At least one subscriber."):
        evt_mgr.run()
    with pytest.raises(AssertionError, match="Subscriber not exists"):
        evt_mgr.unsubscribe(rcv)


def test_duplicate_subscribe():
    rcv = Receiver()
    evt_mgr = trading.EventManager()
    evt_mgr.subscribe(rcv)
    with pytest.raises(AssertionError, match="Duplicate subscribe"):
        evt_mgr.subscribe(rcv)


def test_fire_event():
    rcv = Receiver()
    evt_mgr = trading.EventManager()
    evt_mgr.subscribe(rcv)
    evt_mgr.subscribe(BarDataFirer())

    evt_mgr.run()
    # 1. `Always` of rcv fired, 2. BarDataFirer fired EveryBarData to rcv, which stops the loop
    assert rcv.fired == 2


def test_run_again_after_a_stop():
    rcv = Receiver()
    evt_mgr = trading.EventManager()
    evt_mgr.subscribe(rcv)
    evt_mgr.subscribe(BarDataFirer())
    evt_mgr.run()

    rcv.fired = 0
    evt_mgr.run()
    assert rcv.fired == 2


def test_stop_while_a_run_starts():
    class StopsAtOnce(trading.EventReceiver):
        def on_run(self):
            self.stop_event_manager()
            self.schedule(trading.event.Always(self.on_always))

        def on_always(self, _):
            raise AssertionError("the loop should not have started")

    evt_mgr = trading.EventManager()
    evt_mgr.subscribe(StopsAtOnce())
    evt_mgr.run()


# ---------------------------------- live events, on a fake clock ----------------------------------

MILLISECOND = 1_000_000
SECOND = 1000 * MILLISECOND
MINUTE = 60 * SECOND
OFFSETS = [
    0,
    -10_000,
    -100 * MILLISECOND,
    -SECOND,
    -10 * MINUTE,
    1,
    10_000,
    100 * MILLISECOND,
    SECOND,
    10 * MINUTE,
]


def ts(time):
    return pd.Timestamp(time, tz="UTC")


def delta(ns):
    return pd.Timedelta(ns, unit="ns")


class FireLog(trading.EventReceiver):
    """Logs the time of each firing, by the offset of the event."""

    def __init__(self, event_type, offsets, calendar):
        super().__init__()
        self.event_type = event_type
        self.calendar = calendar
        self.fired = {offset: [] for offset in offsets}

    def on_run(self):
        for offset in self.fired:
            self.schedule(self.event_type(self.logger(offset), offset_ns=offset))

    def logger(self, offset):
        return lambda _: self.fired[offset].append(self.calendar.hr_now())


def run_live(monkeypatch, event_type, offsets, polls, start="2020-01-06 00:00"):
    """Runs the real loop of the event manager, and each sleep moves the clock to the next poll."""
    clock = SimpleNamespace(now=ts(start))
    calendar = trading.Calendar()
    calendar.hr_now = lambda: clock.now
    calendar.build("2020-01-06", "2020-01-17", {"Open": "9:00:00", "Close": "15:00:00"}, tz="UTC")
    evt_mgr = trading.MarketEventManager(calendar)
    log = FireLog(event_type, offsets, calendar)
    evt_mgr.subscribe(log)
    polls = iter(polls)

    def sleep(_):
        try:
            clock.now = next(polls)
        except StopIteration:
            evt_mgr.stop()

    monkeypatch.setattr(trading.event, "time", SimpleNamespace(sleep=sleep))
    evt_mgr.run()
    return log.fired


def polls_around(events, offsets):
    """Some polls before each trigger time, on it, and on until past the event itself."""
    polls = set()
    for event in events:
        for offset in offsets:
            # short of the lead, so a repeat would fire many times
            step = max(abs(offset) // 20, 1000)
            polls_in_lead = max(-offset, 0) // step
            polls.update(event + delta(offset + i * step) for i in range(-3, polls_in_lead + 4))
    return sorted(polls)


@pytest.mark.parametrize(
    "event_type, events",
    [
        (trading.event.MarketOpen, ["2020-01-06 09:00", "2020-01-07 09:00", "2020-01-08 09:00"]),
        (trading.event.MarketClose, ["2020-01-06 15:00", "2020-01-07 15:00", "2020-01-08 15:00"]),
    ],
    ids=["open", "close"],
)
@pytest.mark.parametrize("offset", OFFSETS)
def test_calendar_event_fires_once(monkeypatch, event_type, events, offset):
    events = [ts(event) for event in events]

    fired = run_live(monkeypatch, event_type, [offset], polls_around(events, [offset]))

    assert fired[offset] == [event + delta(offset) for event in events]


def test_calendar_events_sharing_a_calendar_fire_once(monkeypatch):
    """They pop passed events from the same list."""
    events = [ts("2020-01-06 15:00"), ts("2020-01-07 15:00"), ts("2020-01-08 15:00")]

    fired = run_live(monkeypatch, trading.event.MarketClose, OFFSETS, polls_around(events, OFFSETS))

    for offset in OFFSETS:
        assert fired[offset] == [event + delta(offset) for event in events]


@pytest.mark.parametrize("offset", [0, -SECOND, SECOND])
def test_calendar_event_skips_what_passed_between_polls(monkeypatch, offset):
    polls = [
        ts("2020-01-06 14:00"),
        # the loop was away for three closes
        ts("2020-01-08 16:00"),
        ts("2020-01-08 16:00:01"),
        ts("2020-01-09 15:00") + delta(offset) - delta(SECOND),
        ts("2020-01-09 15:00") + delta(offset),
        ts("2020-01-09 15:00") + delta(SECOND),
        ts("2020-01-09 15:00") + delta(2 * SECOND),
    ]

    fired = run_live(monkeypatch, trading.event.MarketClose, [offset], polls)

    assert fired[offset] == [polls[1], polls[4]]


def test_calendar_event_starting_inside_its_lead_waits_for_the_next_event(monkeypatch):
    start = ts("2020-01-06 14:55")
    polls = [start + delta(i * MINUTE) for i in range(1, 8)]
    polls += [ts("2020-01-07 14:49"), ts("2020-01-07 14:50"), ts("2020-01-07 14:51")]

    fired = run_live(
        monkeypatch, trading.event.MarketClose, [-10 * MINUTE], polls, start="2020-01-06 14:55"
    )

    # its time today had passed when it was scheduled
    assert fired[-10 * MINUTE] == [ts("2020-01-07 14:50")]


def test_holiday_on_a_passed_event_does_not_fire():
    calendar = trading.Calendar()
    calendar.hr_now = lambda: ts("2020-01-06 10:00")
    calendar.build("2020-01-06", "2020-01-10", {"Open": "9:00:00"}, tz=ts("2020-01-06").tz)
    calendar.set_as_holiday(ts("2020-01-06").normalize())
    event = trading.event.MarketOpen(lambda _: None, offset_ns=2 * 60 * MINUTE)

    event.on_schedule(SimpleNamespace(calendar=calendar))

    assert event.trigger_time == ts("2020-01-07 11:00")


def test_calendar_event_scheduled_between_its_event_and_its_offset_fires_today(monkeypatch):
    """The event has passed, and it is popped from the calendar, but its time is still to come."""
    days = ("06", "07", "08", "09", "10")
    events = [ts(f"2020-01-{day} 09:00") for day in days]
    polls = polls_around(events, [MINUTE])

    fired = run_live(
        monkeypatch,
        trading.event.MarketOpen,
        [MINUTE],
        [poll for poll in polls if poll > ts("2020-01-06 09:00:30")],
        start="2020-01-06 09:00:30",
    )

    assert fired[MINUTE] == [event + delta(MINUTE) for event in events]


def test_second_run_on_a_calendar_does_not_fire_an_event_again(monkeypatch):
    offset = -5 * 1000 * MILLISECOND
    clock = SimpleNamespace(now=ts("2020-01-06 08:59:50"))
    calendar = trading.Calendar()
    calendar.hr_now = lambda: clock.now
    calendar.build("2020-01-06", "2020-01-08", {"Open": "9:00:00"}, tz="UTC")
    evt_mgr = trading.MarketEventManager(calendar)
    log = FireLog(trading.event.MarketOpen, [offset], calendar)
    evt_mgr.subscribe(log)

    def run_until(stop):
        def sleep(_):
            clock.now += delta(SECOND)
            if clock.now >= stop:
                evt_mgr.stop()

        monkeypatch.setattr(trading.event, "time", SimpleNamespace(sleep=sleep))
        evt_mgr.run()

    # stops inside the lead, after it fired
    run_until(ts("2020-01-06 08:59:57"))
    assert log.fired[offset] == [ts("2020-01-06 08:59:55")]
    run_until(ts("2020-01-06 09:00:05"))
    assert log.fired[offset] == [ts("2020-01-06 08:59:55")]


def test_calendar_event_with_nothing_left_is_an_error_at_the_start_of_a_run():
    calendar = trading.Calendar()
    calendar.hr_now = lambda: ts("2020-02-03 12:00")
    calendar.build("2020-01-06", "2020-01-10", {"Close": "15:00:00"}, tz="UTC")
    evt_mgr = trading.MarketEventManager(calendar)
    evt_mgr.subscribe(FireLog(trading.event.MarketClose, [0], calendar))

    with pytest.raises(ValueError, match="no Close events left"):
        evt_mgr.run()


def test_calendar_event_fires_for_the_last_event_if_it_is_scheduled_inside_its_offset(monkeypatch):
    """The last event of the calendar has passed, its time with the offset has not."""
    clock = SimpleNamespace(now=ts("2020-01-10 15:00:30"))
    calendar = trading.Calendar()
    calendar.hr_now = lambda: clock.now
    calendar.build("2020-01-06", "2020-01-10", {"Close": "15:00:00"}, tz="UTC")
    assert calendar.events["Close"] == []
    evt_mgr = trading.MarketEventManager(calendar)
    log = FireLog(trading.event.MarketClose, [MINUTE], calendar)
    evt_mgr.subscribe(log)
    polls = iter([ts("2020-01-10 15:00:59"), ts("2020-01-10 15:01:00"), ts("2020-01-10 15:01:01")])

    def sleep(_):
        try:
            clock.now = next(polls)
        except StopIteration:
            evt_mgr.stop()

    monkeypatch.setattr(trading.event, "time", SimpleNamespace(sleep=sleep))
    with pytest.warns(RuntimeWarning, match="no Close events left"):
        evt_mgr.run()

    assert log.fired[MINUTE] == [ts("2020-01-10 15:01:00")]


@pytest.mark.parametrize("offset", [0, -SECOND, SECOND])
def test_calendar_event_at_the_end_of_the_calendar(monkeypatch, offset):
    events = [ts(f"2020-01-{day:02} 15:00") for day in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
    polls = polls_around(events, [offset])
    polls += [ts("2020-01-20 15:00"), ts("2020-01-21 15:00")]

    with pytest.warns(RuntimeWarning, match="no Close events left") as warned:
        fired = run_live(monkeypatch, trading.event.MarketClose, [offset], polls)

    # the last one fires too, and the loop goes on
    assert fired[offset] == [event + delta(offset) for event in events]
    assert len(warned) == 1


@pytest.mark.filterwarnings("ignore:The calendar has no Close events left")
@pytest.mark.parametrize("hours", [25, 49, 100])
@pytest.mark.parametrize("sibling_first", [True, False], ids=["sibling_first", "sibling_last"])
def test_positive_offset_longer_than_the_gap_with_a_sibling(monkeypatch, hours, sibling_first):
    """The sibling pops events this one still has to fire for."""
    offset = hours * 60 * MINUTE
    events = [ts(f"2020-01-{day:02} 15:00") for day in (6, 7, 8, 9, 10, 13, 14)]
    offsets = [0, offset] if sibling_first else [offset, 0]

    fired = run_live(monkeypatch, trading.event.MarketClose, offsets, polls_around(events, offsets))

    # the polls go on past `events`, so it fires for the next ones too
    assert fired[0][: len(events)] == events
    assert fired[offset] == [event + delta(offset) for event in events]


def test_positive_offset_fires_after_a_sibling_ran_out_of_events(monkeypatch):
    offset = 25 * 60 * MINUTE
    events = [ts(f"2020-01-{day:02} 15:00") for day in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
    polls = polls_around(events, [0, offset]) + [ts("2020-01-21 15:00")]

    with pytest.warns(RuntimeWarning, match="no Close events left") as warned:
        fired = run_live(monkeypatch, trading.event.MarketClose, [0, offset], polls)

    assert fired[0] == events
    assert fired[offset] == [event + delta(offset) for event in events]
    assert len(warned) == 2


def test_passed_events_stay_out_of_the_calendar_while_a_long_offset_uses_them(monkeypatch):
    """`events` keeps meaning what is still to come, `today_next` reads it."""
    offset = 25 * 60 * MINUTE
    clock = SimpleNamespace(now=ts("2020-01-06 00:00"))
    calendar = trading.Calendar()
    calendar.hr_now = lambda: clock.now
    calendar.build("2020-01-06", "2020-01-10", {"Close": "15:00:00"}, tz="UTC")
    event = trading.event.MarketClose(lambda _: None, offset_ns=offset)
    event.on_schedule(SimpleNamespace(calendar=calendar))
    sibling = trading.event.MarketClose(lambda _: None)
    sibling.on_schedule(SimpleNamespace(calendar=calendar))

    clock.now = ts("2020-01-07 15:00")
    assert sibling.should_trigger()
    assert calendar.events["Close"][0] == ts("2020-01-08 15:00")
    clock.now = ts("2020-01-07 16:00")
    assert event.should_trigger()
    assert event.trigger_time == ts("2020-01-08 16:00")
    assert calendar.events["Close"][0] == ts("2020-01-08 15:00")
