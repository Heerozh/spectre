"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from collections import defaultdict
from typing import Dict
import pandas as pd


class Calendar:
    """
    Usage:
    call build() first, get business day calendar.
    and manually add holiday by calling set_as_holiday().
    if open half-day, use remove_events() remove all events that day, and add_event() manually.
    US holiday calendar can found at https://iextrading.com/trading/
    """

    def __init__(self, csv_file=None) -> None:
        # todo: read from file
        if csv_file is not None:
            pass
        else:
            self.events = defaultdict(list)

    def to_csv(self):
        # todo save to file
        pass

    def build(self, end, events: Dict[str, pd.Timestamp], tz='UTC', freq='B'):
        """ build("2020", {'Open': pd.Timestamp(9:00), 'Close': pd.Timestamp(15:00)}) """
        days = pd.date_range(pd.Timestamp.now(), end, tz=tz, freq=freq)
        self.events = {name: days + time for name, time in events.items()}

    def add_event(self, event: str, date_time: pd.Timestamp):
        self.events[event].append(date_time)
        self.events[event].sort()

    def remove_events(self, date):
        self.events = {
            event: [time for time in times if times.date != date]
            for event, times in self.events.items()
        }

    def set_as_holiday(self, date):
        # 要考虑下calendar设错的情况，比如下单到关闭还没成交的话，订单都默认会取消的，
        # 下个日期重新算就是了，加个测试用例
        return self.remove_events(date)

    def next(self, event_name):
        """return the next time of this event"""
        # todo, remove pasted times, and return next
        pass
