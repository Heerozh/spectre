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
        self.timezone = None

    def build(self, end: str, daily_events: Dict[str, str], tz='UTC', freq='B', pop_passed=True):
        """ build("2020", {'Open': '9:00:00', 'Close': '15:00:00'}) """
        self.timezone = tz
        days = pd.date_range(pd.Timestamp.now(tz=tz).normalize(), pd.Timestamp(end, tz=tz),
                             tz=tz, freq=freq)
        if len(days) == 0:
            raise ValueError("Empty date range between now({}) to end({})".format(
                pd.Timestamp.now(tz=tz).normalize(), end))

        self.events = {name: [day + pd.Timedelta(time) for day in days]
                       for name, time in daily_events.items()}

        if pop_passed:
            for k, _ in self.events.items():
                self.pop_passed(k)

    def add_event(self, event: str, datetime: pd.Timestamp):
        self.events[event].append(datetime)
        self.events[event].sort()

    def remove_events(self, date: pd.Timestamp):
        self.events = {
            event: [dt for dt in dts if dt.normalize() != date]
            for event, dts in self.events.items()
        }

    def set_as_holiday(self, date: pd.Timestamp):
        return self.remove_events(date)

    def hr_now(self):
        """ Return now time """
        # todo high res
        return pd.Timestamp.now(self.timezone)

    def pop_passed(self, event_name):
        """ Remove passed events """
        now = self.hr_now()
        # every event is daily, so will not be overkilled
        dts = self.events[event_name]
        while True:
            if dts[0] <= now:
                del dts[0]
            else:
                break
        return self

    def today_next(self):
        """ Return today next events """
        now = self.hr_now()
        return {
            event: dts[0]
            for event, dts in self.events.items()
            if dts[0].normalize() == now.normalize()
        }


class CNCalendar(Calendar):
    """
    CN holiday calendar: http://www.sse.com.cn/disclosure/dealinstruc/closed/
    """
    # yearly manually update
    closed = [
        *pd.date_range('2020-06-25', '2020-06-28', freq='D'),
        *pd.date_range('2020-10-01', '2020-10-08', freq='D'),
    ]

    def __init__(self, pop_passed=True):
        super().__init__()
        timezone = 'Asia/Shanghai'
        assert pd.Timestamp.now(self.timezone).year <= CNCalendar.closed[-1].year
        self.build(
            end=str(CNCalendar.closed[-1].year + 1),
            daily_events={
                'DayStart': '00:00:00',
                'Open': '9:30:00',
                'Lunch': '11:30:00',
                'LunchEnd': '13:00:00',
                'Close': '15:00:00',
                'DayEnd': '23:59:59'
            },
            tz=timezone, pop_passed=pop_passed)
        for d in CNCalendar.closed:
            self.set_as_holiday(d.tz_localize(timezone))
