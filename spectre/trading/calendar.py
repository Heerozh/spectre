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

    def __init__(self) -> None:
        self.events = defaultdict(list)

    def build(self, end: str, daily_events: Dict[str, str], tz='UTC', freq='B'):
        """ build("2020", {'Open': '9:00:00', 'Close': '15:00:00'}) """
        days = pd.date_range(pd.Timestamp.now(tz=tz).normalize(), pd.Timestamp(end, tz=tz),
                             tz=tz, freq=freq)
        self.events = {name: [day + pd.Timedelta(time) for day in days]
                       for name, time in daily_events.items()}

    def add_event(self, event: str, datetime: pd.Timestamp):
        self.events[event].append(datetime)
        self.events[event].sort()

    def remove_events(self, date: pd.Timestamp):
        self.events = {
            event: [dt for dt in dts if dt.normalize() != date]
            for event, dts in self.events.items()
        }

    def set_as_holiday(self, date: pd.Timestamp):
        # 要考虑下calendar设错的情况，比如下单到关闭还没成交的话，订单都默认会取消的，
        # 下个日期重新算就是了，加个测试用例
        return self.remove_events(date)

    def next(self, event_name):
        """return the next time of this event"""
        # todo, remove pasted times, and return next
        pass


class CNCalendar(Calendar):
    """
    CN holiday calendar: http://www.sse.com.cn/disclosure/dealinstruc/closed/
    """
    timezone = 'Asia/Shanghai'
    # yearly manually update
    closed = [
        *pd.date_range('2020-06-25', '2020-06-28', freq='D'),
        *pd.date_range('2020-10-01', '2020-10-08', freq='D'),
    ]

    def __init__(self):
        super().__init__()
        self.build(
            end=str(pd.Timestamp.now(tz=CNCalendar.timezone).year + 1),
            daily_events={
                'Open': '9:30:00',
                'Lunch': '11:30:00',
                'LunchEnd': '13:00:00',
                'Close': '15:00:00'
            },
            tz=self.timezone)
        for d in CNCalendar.closed:
            self.set_as_holiday(d.tz_localize(CNCalendar.timezone))
