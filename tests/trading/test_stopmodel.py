import pandas as pd
import pytest

from spectre import trading


def trailing_tracker(ratio):
    """Trailing stop tracker opened at 9, then the price falls to 6."""
    tracker = trading.TrailingStopModel(ratio, True).new_tracker(9, False)
    for price in (8, 7, 6):
        tracker.update_price(price)
    return tracker


def test_trailing_stop_price():
    # -10% of the highest price
    assert trailing_tracker(-0.1).stop_price == pytest.approx(9 * 0.9)
    # +10% of the lowest price
    assert trailing_tracker(0.1).stop_price == pytest.approx(6 * 1.1)


CHANGES = [0.8, 0.9, 0.91, 1, 1.09, 1.1, 1.2]


def test_trailing_stop_upper_bound():
    tracker = trailing_tracker(0.1)
    triggered = []
    for change in CHANGES:
        tracker.last_price = 6 * change
        triggered.append(tracker.check_trigger())
    assert triggered == [False, False, False, False, False, True, True]


def test_trailing_stop_lower_bound():
    tracker = trailing_tracker(-0.1)
    triggered = []
    for change in CHANGES:
        tracker.last_price = 9 * change
        triggered.append(tracker.check_trigger())
    assert triggered == [True, True, False, False, False, False, False]


def test_trailing_stop_callback():
    calls = []
    tracker = trading.TrailingStopModel(-0.1, lambda *args: calls.append(args) or 'fired') \
        .new_tracker(10, False)
    tracker.update_price(8)
    assert tracker.check_trigger('AAPL', -10) == 'fired'
    assert calls == [('AAPL', -10)]


def pnl_decay_tracker(ratio, pnl_target, position):
    tracker = trading.PnLDecayTrailingStopModel(ratio, pnl_target, True).new_tracker(
        position.last_price, False)
    tracker.tracking_position = position
    return tracker


def test_pnl_decay_short_stop_loss():
    pos = trading.Position(-10, 9, 0, None, None)
    tracker = pnl_decay_tracker(0.1, 0.1, pos)
    assert tracker.stop_price == pytest.approx(9 * 1.1)

    pos.last_price = 8
    tracker.update_price(pos.last_price)
    # PnL = 1/9, decayed ratio = ratio * decay_rate ^ (PnL / target)
    assert tracker.stop_price == pytest.approx(8 * (1 + 0.1 * 0.05 ** (1 / 9 / 0.1)))

    for price in (7, 6):
        pos.last_price = price
        tracker.update_price(price)
    # PnL is far above the target, stop price is very close to the lowest price
    tracker.last_price = 5.99
    assert not tracker.check_trigger()
    tracker.last_price = 6.01
    assert tracker.check_trigger()


def test_pnl_decay_long_stop_loss():
    pos = trading.Position(10, 9, 0, None, None)
    tracker = pnl_decay_tracker(-0.1, 0.1, pos)
    assert tracker.stop_price == pytest.approx(9 * 0.9)

    pos.last_price = 8
    tracker.update_price(pos.last_price)
    # PnL < 0 is not decayed for a stop loss model
    assert tracker.stop_price == pytest.approx(9 * 0.9)
    assert tracker.check_trigger()


def test_pnl_decay_long_stop_gain():
    pos = trading.Position(10, 9, 0, None, None)
    tracker = pnl_decay_tracker(0.1, -0.1, pos)
    assert tracker.stop_price == pytest.approx(9 * 1.1)

    pos.last_price = 8
    tracker.update_price(pos.last_price)
    assert tracker.stop_price == pytest.approx(8 * (1 + 0.1 * 0.05 ** (1 / 9 / 0.1)))

    for price in (7, 6):
        pos.last_price = price
        tracker.update_price(price)
    tracker.last_price = 5.99
    assert not tracker.check_trigger()
    tracker.last_price = 6.01
    assert tracker.check_trigger()


def test_time_decay_stop():
    pos = trading.Position(10, 9, 0, pd.Timestamp('2014-01-02', tz='UTC'), None)
    tracker = trading.TimeDecayTrailingStopModel(-0.1, pd.Timedelta(days=5), True) \
        .new_tracker(pos.last_price, False)
    tracker.tracking_position = pos
    assert tracker.stop_price == pytest.approx(9 * 0.9)

    pos.last_price = 8
    tracker.update_price(pos.last_price)
    assert tracker.check_trigger()

    # 3 of 5 days passed
    pos.current_dt = pd.Timestamp('2014-01-05', tz='UTC')
    assert tracker.stop_price == pytest.approx(9 * (1 - 0.1 * 0.05 ** 0.6))
