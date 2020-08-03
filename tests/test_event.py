import unittest
import spectre
from os.path import dirname
import pandas as pd

data_dir = dirname(__file__) + '/data/'


class TestTradingEvent(unittest.TestCase):

    def test_event_mgr(self):
        class TestEventReceiver(spectre.trading.EventReceiver):
            fired = 0

            def on_run(self):
                self.schedule(spectre.trading.event.Always(self.test_always))
                self.schedule(spectre.trading.event.EveryBarData(self.test_every_bar))

            def test_always(self, _):
                self.fired += 1

            def test_every_bar(self, _):
                self.fired += 1
                if self.fired == 2:
                    self.stop_event_manager()

        class StopFirer(spectre.trading.EventReceiver):
            def on_run(self):
                self.schedule(spectre.trading.event.Always(self.test))

            def test(self, _):
                self.fire_event(spectre.trading.event.EveryBarData)

        rcv = TestEventReceiver()

        evt_mgr = spectre.trading.EventManager()
        evt_mgr.subscribe(rcv)
        rcv.unsubscribe()
        self.assertEqual(0, len(evt_mgr._subscribers))
        self.assertRaisesRegex(ValueError, 'At least one subscriber.', evt_mgr.run)
        self.assertRaisesRegex(AssertionError, 'Subscriber not exists', evt_mgr.unsubscribe, rcv)

        evt_mgr.subscribe(rcv)
        evt_mgr.subscribe(StopFirer())
        self.assertRaisesRegex(AssertionError, 'Duplicate subscribe', evt_mgr.subscribe, rcv)

        evt_mgr.run()
        self.assertEqual(2, rcv.fired)

    def test_calendar(self):
        tz = 'America/New_York'
        end = pd.Timestamp.now(tz=tz) + pd.DateOffset(days=10)
        first = pd.date_range(pd.Timestamp.now(tz=tz).normalize(), end, freq='B')[0]
        if pd.Timestamp.now(tz=tz) > (first + pd.Timedelta("9:00:00")):
            first = first + pd.offsets.BDay(1)
        holiday = first + pd.offsets.BDay(2)
        test_now = first + pd.offsets.BDay(1) + pd.Timedelta("10:00:00")

        calendar = spectre.trading.Calendar()
        calendar.build(start=str(pd.Timestamp.now(tz=tz).normalize()), end=str(end.date()),
                       daily_events={'Open': '9:00:00', 'Close': '15:00:00'},
                       tz=tz)
        calendar.set_as_holiday(holiday)

        self.assertEqual(first + pd.Timedelta("9:00:00"),
                         calendar.events['Open'][0])

        calendar.hr_now = lambda: test_now

        calendar.pop_passed('Open')

        self.assertEqual(test_now.normalize() + pd.offsets.BDay(2) + pd.Timedelta("9:00:00"),
                         calendar.events['Open'][0])

        # test assert
        self.assertRaises(ValueError, calendar.build,
                          start=str(pd.Timestamp.now(tz=tz).normalize()), end='2019',
                          daily_events={'Open': '9:00:00', 'Close': '15:00:00'},
                          tz=tz)
