"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import time
from typing import Type


class Event:
    def __init__(self, callback) -> None:
        self.callback = callback

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
    """TODO: The following code is prepared for live trading in future, no effect on back-testing"""
    def __init__(self, calendar_event_name, callback, offset_ns=0) -> None:
        super().__init__(callback)
        self.offset = offset_ns
        self.calendar = None
        self.event_name = calendar_event_name
        self.trigger_time = 0

    def on_schedule(self, evt_mgr):
        self.calendar = evt_mgr.calendar

    def calculate_range(self):
        self.trigger_time = self.calendar.events[self.event_name].first() + self.offset

    def should_trigger(self) -> bool:
        if self.calendar.hr_now() >= self.trigger_time:
            self.calculate_range()
            return True
        return False


class MarketOpen(CalendarEvent):
    def __init__(self, callback, offset_ns=0) -> None:
        super().__init__('open', callback, offset_ns)


class MarketClose(CalendarEvent):
    def __init__(self, callback, offset_ns=0) -> None:
        super().__init__('close', callback, offset_ns)

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

    def fire_event_type(self, evt_type: Type[Event]):
        self._event_manager.fire_event_type(evt_type)

    def on_subscribe(self):
        raise NotImplementedError("abstractmethod")

    def initialize(self):
        raise NotImplementedError("abstractmethod")


class EventManager:
    def __init__(self) -> None:
        self._subscribers = dict()
        self._stop = False

    def subscribe(self, receiver: EventReceiver):
        assert receiver not in self._subscribers, 'Duplicate subscribe'
        self._subscribers[receiver] = []
        receiver._event_manager = self
        receiver.on_subscribe()

    def unsubscribe(self, receiver: EventReceiver):
        assert receiver in self._subscribers, 'Subscriber not exists'
        del self._subscribers[receiver]
        receiver._event_manager = None

    def schedule(self, receiver: EventReceiver, event: Event):
        self._subscribers[receiver].append(event)
        event.on_schedule(self)

    def fire_event_type(self, evt_type: Type[Event]):
        for r, events in self._subscribers.items():
            for evt in events:
                if isinstance(evt, evt_type):
                    evt.callback()

    def stop(self):
        self._stop = True

    def run(self, *params):
        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        for r, events in self._subscribers.items():
            r.initialize()
        while not self._stop:
            time.sleep(0.001)
            for r, events in self._subscribers.items():
                for event in events:
                    if event.should_trigger():
                        event.callback()


# ----------------------------------------------------------------


class SimulationEventManager(EventManager):
    @classmethod
    def _get_most_granular(cls, data):
        freq = {k: min(v.index.levels[0][1:]-v.index.levels[0][:-1]) for k, v in data.items()}
        return data[min(freq, key=freq.get)]

    def fire_get_updated_factor(self, start, end):
        for r in self._subscribers:
            data = r.create_data(start, end)
            if isinstance(data, dict):
                main = self._get_most_granular(data)
                main = main[start:end]
            else:
                main = data
            ticks = main.index.get_level_values(0).unique()
            for t in ticks:
                # todo 判断如果是市场开门，fire 开门事件， 注意baseblotter也要注册关门事件，关门了交易就报错
                # 如果open offset>=9 就是after，不然就是before

                r.midnight_before_market_open()
                r.before_market_open()
                r.market_open()

                r.handle_factor({k: v[:t]for k, v in data.items()})

                r.before_market_close()
                r.market_close()
                # 每tick运行完后，记录时间，然后当天new_order存到每个时间的表里

    def run(self, start, end):
        self.fire_get_updated_factor(start, end)


# ----------------------------------------------------------------


# class MarketEventManager(EventManager):
# holiday calendar can found at https://iextrading.com/trading/
#     def __init__(self, calendar: MarketCalendar) -> None:
#         self.calendar = calendar






