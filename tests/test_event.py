import unittest
import spectre
from os.path import dirname

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
