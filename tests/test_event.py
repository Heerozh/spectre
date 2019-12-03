import unittest
import spectre


class TestTradingEvent(unittest.TestCase):

    def test_event_mgr(self):
        class TestEventReceiver(spectre.trading.EventReceiver):
            def initialize(self):
                pass

            def on_subscribe(self):
                pass

        rcv = TestEventReceiver()

        evt_mgr = spectre.trading.EventManager()
        evt_mgr.subscribe(rcv)
        rcv.unsubscribe()
        self.assertEqual(0, len(evt_mgr._subscribers))
        self.assertRaisesRegex(ValueError, 'At least one subscriber.', evt_mgr.run)
        self.assertRaisesRegex(AssertionError, 'Subscriber not exists', evt_mgr.unsubscribe, rcv)

        evt_mgr.subscribe(rcv)
        self.assertRaisesRegex(AssertionError, 'Duplicate subscribe', evt_mgr.subscribe, rcv)

        evt_mgr.run()

    def test_simulation_event_manager(self):
        pass
