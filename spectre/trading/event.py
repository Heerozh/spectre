"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import time
from typing import Type
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
    """ The following code is for live trading, will not work on back-testing """
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
        except AttributeError:
            pass

    def calculate_range(self):
        self.trigger_time = self.calendar.events[self.event_name][0] + \
                            pd.Timedelta(self.offset, unit='ns')

    def should_trigger(self) -> bool:
        if self.calendar.hr_now() >= self.trigger_time:
            self.calendar.pop_passed(self.event_name)
            self.calculate_range()
            return True
        return False


class MarketOpen(CalendarEvent):
    """ Works on both live and backtest """
    def __init__(self, callback, offset_ns=0) -> None:
        super().__init__('Open', callback, offset_ns)


class MarketClose(CalendarEvent):
    """ Works on both live and backtest """
    def __init__(self, callback, offset_ns=0) -> None:
        super().__init__('Close', callback, offset_ns)


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

    def fire_event(self, evt_type: Type[Event]):
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
        assert receiver not in self._subscribers, 'Duplicate subscribe'
        self._subscribers[receiver] = []
        receiver._event_manager = self

    def unsubscribe(self, receiver: EventReceiver):
        assert receiver in self._subscribers, 'Subscriber not exists'
        del self._subscribers[receiver]
        receiver._event_manager = None

    def schedule(self, receiver: EventReceiver, event: Event):
        self._subscribers[receiver].append(event)
        event.on_schedule(self)

    def fire_event(self, source, evt_type: Type[Event]):
        for r, events in self._subscribers.items():
            for evt in events:
                if isinstance(evt, evt_type):
                    evt.callback(source)

    def stop(self):
        self._stop = True

    def _beg_run(self):
        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        for r, events in self._subscribers.items():
            # clear scheduled events
            events.clear()
            r.on_run()

    def _end_run(self):
        for r in self._subscribers.keys():
            r.on_end_of_run()

    def _run_once(self):
        for r, events in self._subscribers.items():
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

