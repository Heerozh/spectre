import unittest
import spectre


class TestTradingEvent(unittest.TestCase):

    def test_event_mgr(self):
        class TestEventReceiver(spectre.trading.EventReceiver):
            fired = 0

            def initialize(self):
                self.schedule(spectre.trading.event.EveryBarData(self.test_every_bar))

            def on_subscribe(self):
                self.schedule(spectre.trading.event.Always(self.test_always))

            def test_always(self):
                self.fired += 1

            def test_every_bar(self):
                self.fired += 1
                if self.fired == 2:
                    self.stop_event_manager()

        class StopFirer(spectre.trading.EventReceiver):
            def initialize(self):
                self.schedule(spectre.trading.event.Always(self.test))

            def on_subscribe(self):
                pass

            def test(self):
                self.fire_event_type(spectre.trading.event.EveryBarData)

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

    def test_simulation_event_manager(self):
        pass
