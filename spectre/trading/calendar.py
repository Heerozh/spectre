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

    def build(self, start: str, end: str, daily_events: Dict[str, str], tz='UTC', freq='B',
              pop_passed=True):
        """ build("2020", {'Open': '9:00:00', 'Close': '15:00:00'}) """
        self.timezone = tz
        days = pd.date_range(pd.Timestamp(start, tz=tz).normalize(),
                             pd.Timestamp(end, tz=tz).normalize(),
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

        *pd.date_range('2021-01-01', '2021-01-03', freq='D'),
        *pd.date_range('2021-02-11', '2021-02-17', freq='D'),
        *pd.date_range('2021-04-03', '2021-04-05', freq='D'),
        *pd.date_range('2021-05-01', '2021-05-05', freq='D'),
        *pd.date_range('2021-06-12', '2021-06-14', freq='D'),
        *pd.date_range('2021-09-19', '2021-09-21', freq='D'),
        *pd.date_range('2021-10-01', '2021-10-07', freq='D'),

        *pd.date_range('2022-01-01', '2022-01-03', freq='D'),
        *pd.date_range('2022-01-31', '2022-02-06', freq='D'),
        *pd.date_range('2022-04-03', '2022-04-05', freq='D'),
        *pd.date_range('2022-04-30', '2022-05-04', freq='D'),
        *pd.date_range('2022-06-03', '2022-06-05', freq='D'),
        *pd.date_range('2022-09-10', '2022-09-12', freq='D'),
        *pd.date_range('2022-10-01', '2022-10-07', freq='D'),

        *pd.date_range('2023-01-01', '2023-01-02', freq='D'),
        *pd.date_range('2023-01-21', '2023-01-27', freq='D'),
        *pd.date_range('2023-04-05', '2023-04-05', freq='D'),
        *pd.date_range('2023-04-29', '2023-05-03', freq='D'),
        *pd.date_range('2023-06-22', '2023-06-24', freq='D'),
        *pd.date_range('2023-09-29', '2023-10-06', freq='D'),
    ]

    daily_events = {
        'DayStart': '00:00:00',
        'PreOpen': '9:15:00',
        'Open': '9:30:00',
        'Lunch': '11:30:00',
        'LunchEnd': '13:00:00',
        'Close': '15:00:00',
        'DayEnd': '23:59:59'
    }

    def __init__(self, start=None, pop_passed=True):
        super().__init__()
        timezone = 'Asia/Shanghai'
        if start is None:
            start = pd.Timestamp.now(self.timezone).normalize()
        assert start.year <= CNCalendar.closed[-1].year
        self.build(
            start=str(start),
            end=str(CNCalendar.closed[-1].year + 1),
            daily_events=self.daily_events,
            tz=timezone, pop_passed=pop_passed)
        for d in CNCalendar.closed:
            self.set_as_holiday(d.tz_localize(timezone))


class JPCalendar(Calendar):
    """
    JP holiday calendar: https://www.jpx.co.jp/corporate/about-jpx/calendar/index.html
    """
    closed = [
        *pd.date_range(f'{pd.Timestamp.now().year}-01-01',
                       f'{pd.Timestamp.now().year}-01-03', freq='D'),
        *pd.date_range(f'{pd.Timestamp.now().year+1}-01-01',
                       f'{pd.Timestamp.now().year+1}-01-03', freq='D'),
        *pd.date_range(f'{pd.Timestamp.now().year}-12-31',
                       f'{pd.Timestamp.now().year}-12-31', freq='D'),
        *pd.date_range(f'{pd.Timestamp.now().year+1}-12-31',
                       f'{pd.Timestamp.now().year+1}-12-31', freq='D'),

        # yearly manually updated
        *pd.date_range('2023-01-09', '2023-01-09', freq='D'),
        *pd.date_range('2023-02-11', '2023-02-11', freq='D'),
        *pd.date_range('2023-02-23', '2023-02-23', freq='D'),
        *pd.date_range('2023-03-21', '2023-03-21', freq='D'),
        *pd.date_range('2023-04-29', '2023-04-29', freq='D'),
        *pd.date_range('2023-05-03', '2023-05-05', freq='D'),
        *pd.date_range('2023-07-17', '2023-07-17', freq='D'),
        *pd.date_range('2023-08-11', '2023-08-11', freq='D'),
        *pd.date_range('2023-09-18', '2023-09-18', freq='D'),
        *pd.date_range('2023-09-23', '2023-09-23', freq='D'),
        *pd.date_range('2023-10-09', '2023-10-09', freq='D'),
        *pd.date_range('2023-11-03', '2023-11-03', freq='D'),
        *pd.date_range('2023-11-23', '2023-11-23', freq='D'),

        *pd.date_range('2024-01-08', '2024-01-08', freq='D'),
        *pd.date_range('2024-02-11', '2024-02-12', freq='D'),
        *pd.date_range('2024-02-23', '2024-02-23', freq='D'),
        *pd.date_range('2024-03-20', '2024-03-20', freq='D'),
        *pd.date_range('2024-04-29', '2024-04-29', freq='D'),
        *pd.date_range('2024-05-03', '2024-05-06', freq='D'),
        *pd.date_range('2024-07-15', '2024-07-15', freq='D'),
        *pd.date_range('2024-08-11', '2024-08-12', freq='D'),
        *pd.date_range('2024-09-16', '2024-09-16', freq='D'),
        *pd.date_range('2024-09-22', '2024-09-23', freq='D'),
        *pd.date_range('2024-10-14', '2024-10-14', freq='D'),
        *pd.date_range('2024-11-03', '2024-11-04', freq='D'),
        *pd.date_range('2024-11-23', '2024-11-23', freq='D'),
    ]

    daily_events = {
        'DayStart': '00:00:00',
        'PreOpen': '8:00:00',
        'Open': '9:00:00',
        'Lunch': '11:30:00',
        'LunchEnd': '12:30:00',
        'Close': '15:00:00',
        'DayEnd': '23:59:59'
    }

    def __init__(self, start=None, pop_passed=True):
        super().__init__()
        timezone = 'Asia/Tokyo'
        if start is None:
            start = pd.Timestamp.now(self.timezone).normalize()
        assert start.year <= self.closed[-1].year
        self.build(
            start=str(start),
            end=str(self.closed[-1].year + 1),
            daily_events=self.daily_events,
            tz=timezone, pop_passed=pop_passed)
        for d in self.closed:
            self.set_as_holiday(d.tz_localize(timezone))
