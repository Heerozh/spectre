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
