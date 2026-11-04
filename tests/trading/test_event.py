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


def test_calendar_event_starting_inside_its_lead_fires_once(monkeypatch):
    start = ts("2020-01-06 14:55")
    polls = [start + delta(i * MINUTE) for i in range(1, 8)]

    fired = run_live(
        monkeypatch, trading.event.MarketClose, [-10 * MINUTE], polls, start="2020-01-06 14:55"
    )

    # late, the first poll is past its time
    assert fired[-10 * MINUTE] == [polls[0]]


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
