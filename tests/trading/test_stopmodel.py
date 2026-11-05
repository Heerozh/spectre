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
    tracker = trading.TrailingStopModel(
        -0.1, lambda *args: calls.append(args) or "fired"
    ).new_tracker(10, False)
    tracker.update_price(8)
    assert tracker.check_trigger("AAPL", -10) == "fired"
    assert calls == [("AAPL", -10)]


def pnl_decay_tracker(ratio, pnl_target, position):
    tracker = trading.PnLDecayTrailingStopModel(ratio, pnl_target, True).new_tracker(
        position.last_price, False
    )
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


@pytest.mark.parametrize(
    "shares, price, closed, stop_price",
    [
        # opened at 10, the best price is +20%, so ratio = 0.1 * 0.05 ^ (0.2 / 0.1) = 0.00025.
        # The stop is 0.00025 from the best price, on the side of a drawdown
        (100, 12, -50, 12 * (1 - 0.00025)),
        (100, 12, -90, 12 * (1 - 0.00025)),
        (-100, 8, 50, 8 * (1 + 0.00025)),
        (-100, 8, 90, 8 * (1 + 0.00025)),
    ],
)
def test_pnl_decay_after_reducing_position(shares, price, closed, stop_price):
    model = trading.PnLDecayTrailingStopModel(-0.1, 0.1, True)
    pos = trading.Position(shares, 10, 0, None, model)
    pos.last_price = price
    assert pos.stop_tracker.current == pytest.approx(0.2)
    assert pos.stop_tracker.stop_price == pytest.approx(stop_price)

    # PnL% is about the shares left, closing some of them at that price doesn't change it
    pos.update(closed, price, 0, None)
    assert pos.stop_tracker.current == pytest.approx(0.2)
    assert pos.stop_tracker.stop_price == pytest.approx(stop_price)


def test_pnl_decay_stop_gain_after_reducing_position():
    # long 100 at 10 with a stop gain model, the worst price is -20%
    model = trading.PnLDecayTrailingStopModel(0.1, -0.1, True)
    pos = trading.Position(100, 10, 0, None, model)
    pos.last_price = 8
    assert pos.stop_tracker.current == pytest.approx(-0.2)
    assert pos.stop_tracker.stop_price == pytest.approx(8 * (1 + 0.00025))

    pos.update(-90, 8, 0, None)
    assert pos.stop_tracker.current == pytest.approx(-0.2)
    assert pos.stop_tracker.stop_price == pytest.approx(8 * (1 + 0.00025))


def test_pnl_decay_with_zero_average_price():
    pos = trading.Position(100, 10, 0, None, trading.PnLDecayTrailingStopModel(-0.1, 0.1, True))
    # selling half at twice the price brings back all the cost
    pos.update(-50, 20, 0, None)
    assert pos.average_price == 0
    assert pos.stop_tracker.current == pytest.approx(1)
    assert pos.stop_tracker.stop_price == pytest.approx(20 * (1 - 0.1 * 0.05 ** (1 / 0.1)))


def test_pnl_decay_with_a_dividend():
    pos = trading.Position(100, 100, 0, None, trading.PnLDecayTrailingStopModel(-0.05, 0.1, True))

    # price and cost both 6 lower, there is still no PnL, so no decay, and the stop is -5%
    pos.process_dividend(6, tax=0)
    assert pos.stop_tracker.current == 0
    assert pos.stop_tracker.stop_price == pytest.approx(94 * 0.95)
    assert not pos.check_stop_trigger()


@pytest.mark.parametrize("amount, tax", [(5, 0), (4.5, 0.5)], ids=["no_tax", "taxed"])
def test_pnl_decay_in_profit_with_a_dividend(amount, tax):
    # bought at 50, high of 104, and just above the stop, where a decayed stop tends to be
    model = trading.PnLDecayTrailingStopModel(-0.1, 1.0, True)
    pos = trading.Position(100, 50, 0, None, model)
    pos.last_price = 104
    pos.last_price = 103.7
    stop_price = pos.stop_tracker.stop_price
    assert stop_price == pytest.approx(104 * (1 - 0.1 * 0.05 ** (104 / 50 - 1)))
    assert not pos.check_stop_trigger()

    # 103.7 to 98.7 is a total return of 0, so the same PnL% and the same stop in the new prices
    pos.process_dividend(amount, tax=tax)
    assert pos.stop_tracker.current == pytest.approx(104 / 50 - 1)
    assert pos.stop_tracker.stop_price == pytest.approx(stop_price * 98.7 / 103.7)
    assert not pos.check_stop_trigger()

    pos.last_price = 98
    assert pos.check_stop_trigger()


@pytest.mark.parametrize("amount, tax", [(5, 0), (4.5, 0.5)], ids=["no_tax", "taxed"])
def test_pnl_decay_of_short_in_profit_with_a_dividend(amount, tax):
    # sold at 150, low of 100
    model = trading.PnLDecayTrailingStopModel(-0.1, 1.0, True)
    pos = trading.Position(-100, 150, 0, None, model)
    pos.last_price = 100
    pos.last_price = 103
    pnl = 1 - 100 / 150
    stop_price = pos.stop_tracker.stop_price
    assert stop_price == pytest.approx(100 * (1 + 0.1 * 0.05**pnl))
    assert not pos.check_stop_trigger()

    pos.process_dividend(amount, tax=tax)
    assert pos.stop_tracker.current == pytest.approx(pnl)
    assert pos.stop_tracker.stop_price == pytest.approx(stop_price * 98 / 103)
    assert not pos.check_stop_trigger()

    pos.last_price = 99
    assert pos.check_stop_trigger()


def test_time_decay_with_a_dividend():
    dt = pd.Timestamp("2014-01-02", tz="UTC")
    model = trading.TimeDecayTrailingStopModel(-0.05, pd.Timedelta(days=5), True)
    pos = trading.Position(100, 100, 0, dt, model)

    pos.process_dividend(6, tax=0)
    assert pos.stop_tracker.stop_price == pytest.approx(94 * 0.95)
    assert not pos.check_stop_trigger()


def test_time_decay_stop():
    pos = trading.Position(10, 9, 0, pd.Timestamp("2014-01-02", tz="UTC"), None)
    tracker = trading.TimeDecayTrailingStopModel(-0.1, pd.Timedelta(days=5), True).new_tracker(
        pos.last_price, False
    )
    tracker.tracking_position = pos
    assert tracker.stop_price == pytest.approx(9 * 0.9)

    pos.last_price = 8
    tracker.update_price(pos.last_price)
    assert tracker.check_trigger()

    # 3 of 5 days passed
    pos.current_dt = pd.Timestamp("2014-01-05", tz="UTC")
    assert tracker.stop_price == pytest.approx(9 * (1 - 0.1 * 0.05**0.6))
