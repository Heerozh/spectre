"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import time
import warnings

import pandas as pd

from .calendar import Calendar


class Event:
    def __init__(self, callback) -> None:
        self.callback = callback  # Callable[[Object], None]

    def on_schedule(self, evt_mgr):
        pass

    def should_trigger(self) -> bool:
        raise NotImplementedError("abstractmethod")


class EveryBarData(Event):
    """This event is triggered passively"""

    def should_trigger(self) -> bool:
        return False


class Always(Event):
    """Always event is useful for live date IO function like asio.read_until_complete()"""

    def should_trigger(self) -> bool:
        return True


class CalendarEvent(Event):
    """The following code is for live trading, will not work on back-testing"""

    def __init__(self, calendar_event_name, callback, offset_ns=0) -> None:
        super().__init__(callback)
        self.offset = offset_ns
        self.calendar = None
        self.event_name = calendar_event_name
        self.trigger_time = None

    def on_schedule(self, evt_mgr):
        try:
            self.calendar = evt_mgr.calendar
            self.calculate_range()
            if self.trigger_time is None:
                raise ValueError(
                    f"The calendar has no {self.event_name} events left, this event can't fire."
                )
        except AttributeError:
            pass

    def calculate_range(self):
        """Set `trigger_time` to the first time of the calendar that is still to come, or None."""
        # A negative offset fires before its event has passed, so popping alone would leave it
        # on the same event. A positive one can still need events that passed, and any event
        # sharing the calendar pops those.
        now = self.calendar.hr_now()
        offset = pd.Timedelta(self.offset, unit="ns")
        times = (dt + offset for dt in self.calendar._with_passed(self.event_name))
        self.trigger_time = next((t for t in times if t > now), None)

    def should_trigger(self) -> bool:
        if self.trigger_time is None or self.calendar.hr_now() < self.trigger_time:
            return False
        self.calculate_range()
        if self.trigger_time is None:
            warnings.warn(
                f"The calendar has no {self.event_name} events left, this event will not fire "
                "again.",
                RuntimeWarning,
            )
        self.calendar.pop_passed(self.event_name)
        return True


class MarketOpen(CalendarEvent):
    """Works on both live and backtest"""

    def __init__(self, callback, offset_ns=0) -> None:
        super().__init__("Open", callback, offset_ns)


class MarketClose(CalendarEvent):
    """Works on both live and backtest"""

    def __init__(self, callback, offset_ns=0) -> None:
        super().__init__("Close", callback, offset_ns)


# ----------------------------------------------------------------


class EventReceiver:
    def __init__(self) -> None:
        self._event_manager = None

    def unsubscribe(self):
        if self._event_manager is not None:
            self._event_manager.unsubscribe(self)

    def schedule(self, evt: Event):
        self._event_manager.schedule(self, evt)

    def stop_event_manager(self):
        self._event_manager.stop()

    def fire_event(self, evt_type: type[Event]):
        self._event_manager.fire_event(self, evt_type)

    def on_run(self):
        pass

    def on_end_of_run(self):
        pass


class EventManager:
    def __init__(self) -> None:
        self._subscribers = dict()
        self._stop = False

    def subscribe(self, receiver: EventReceiver):
        assert receiver not in self._subscribers, "Duplicate subscribe"
        self._subscribers[receiver] = []
        receiver._event_manager = self

    def unsubscribe(self, receiver: EventReceiver):
        assert receiver in self._subscribers, "Subscriber not exists"
        del self._subscribers[receiver]
        receiver._event_manager = None

    def schedule(self, receiver: EventReceiver, event: Event):
        self._subscribers[receiver].append(event)
        event.on_schedule(self)

    def fire_event(self, source, evt_type: type[Event]):
        for events in self._subscribers.values():
            for evt in events:
                if isinstance(evt, evt_type):
                    evt.callback(source)

    def stop(self):
        self._stop = True

    def _beg_run(self):
        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        self._stop = False
        for r, events in self._subscribers.items():
            # clear scheduled events
            events.clear()
            r.on_run()

    def _end_run(self):
        for r in self._subscribers:
            r.on_end_of_run()

    def _run_once(self):
        for events in self._subscribers.values():
            for event in events:
                if event.should_trigger():
                    event.callback(self)

    def run(self, *params):
        self._beg_run()

        while not self._stop:
            time.sleep(0.001)
            self._run_once()

        self._end_run()


# ----------------------------------------------------------------


class MarketEventManager(EventManager):
    def __init__(self, calendar: Calendar) -> None:
        self.calendar = calendar
        super().__init__()
