import pandas as pd
import pytest

from spectre import trading


def test_open_position():
    dt = pd.Timestamp("2019-01-02", tz="UTC")
    pos = trading.Position(10, 9, 10, dt)
    # commission is included in average price
    assert pos.average_price == 10
    assert pos.last_price == 9
    assert pos.value == 90
    assert pos.unrealized == -10
    assert pos.unrealized_percent == pytest.approx(-0.1)
    assert pos.open_dt == dt

    pos.current_dt = dt + pd.Timedelta(days=2)
    assert pos.period == pd.Timedelta(days=2)


def test_update_position():
    pos = trading.Position(10, 10, 0, None)
    empty, realized = pos.update(10, 12, 0, None)
    assert (empty, realized) == (False, 0)
    assert pos.average_price == 11

    empty, realized = pos.update(-5, 13, 0, None)
    assert not empty
    assert realized == pytest.approx(10)
    assert pos.shares == 15

    empty, realized = pos.update(-15, 10, 0, None)
    assert empty
    assert realized == pytest.approx(-15)
    assert pos.realized == pytest.approx(10 - 15)


def test_reverse_position():
    pos = trading.Position(10, 10, 0, None)
    # sell 15 shares with 15 commission, closes 10 shares and opens a short of 5
    empty, realized = pos.update(-15, 12, 15, None)
    assert not empty
    assert realized == pytest.approx((12 - 10) * 10 - 10)
    assert pos.shares == -5
    assert pos.average_price == pytest.approx(12 + 5 / -5)
    assert pos.realized == 0


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_realized_of_each_fill(side):
    # 150 shares at (100 * 10 + 50 * 8) / 150 = 9.3333 each, adding realizes nothing
    pos = trading.Position(side * 100, 10, 0, None)
    empty, realized = pos.update(side * 50, 8, 0, None)
    assert (empty, realized) == (False, 0)
    assert pos.realized == 0

    # close 60 at 9, which is 0.3333 under the entry: a long lost that, a short made it
    empty, realized = pos.update(-side * 60, 9, 0, None)
    assert not empty
    assert realized == pytest.approx(side * -20)
    assert pos.realized == pytest.approx(side * -20)

    # close the last 90 at 9
    empty, realized = pos.update(-side * 90, 9, 0, None)
    assert empty
    assert realized == pytest.approx(side * -30)
    # cash flow of the long: 150 * 9 - 1400 = -50, the short has the opposite flows
    assert pos.realized == pytest.approx(side * -50)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_realized_of_repeated_partial_closes(side):
    pos = trading.Position(side * 100, 10, 0, None)
    # each close is 2 above the entry of 10, so 25 shares are always side * 2 * 25. The profit of
    # the first close lowers the average price, and it must not be counted again by the second
    for shares, expected in ((25, 50), (25, 50), (50, 100)):
        _, realized = pos.update(-side * shares, 12, 0, None)
        assert realized == pytest.approx(side * expected)
    assert pos.realized == pytest.approx(side * 200)


def test_short_realized_with_commission():
    # short 100 at 10 paying 10, receives 990, so 9.9 per share
    pos = trading.Position(-100, 10, 10, None)
    assert pos.average_price == pytest.approx(9.9)

    # short 50 more at 10 paying 5, receives 495. The cost of adding goes to the average price
    empty, realized = pos.update(-50, 10, 5, None)
    assert (empty, realized) == (False, 0)
    assert pos.average_price == pytest.approx(9.9)

    # buy 50 back at 8 paying 5, each share made 9.9 - 8
    empty, realized = pos.update(50, 8, 5, None)
    assert not empty
    assert realized == pytest.approx(50 * (9.9 - 8) - 5)
    assert realized == pytest.approx(90)

    # buy the last 100 back at 12 paying 10, each share lost 12 - 9.9
    empty, realized = pos.update(100, 12, 10, None)
    assert empty
    assert realized == pytest.approx(100 * (9.9 - 12) - 10)
    assert realized == pytest.approx(-220)
    # cash flow: 990 + 495 - (400 + 5) - (1200 + 10) = -130
    assert pos.realized == pytest.approx(-130)


def test_short_realized_before_reversing():
    # short 100 at 10, buy 40 back at 8 for (10 - 8) * 40
    pos = trading.Position(-100, 10, 0, None)
    empty, realized = pos.update(40, 8, 0, None)
    assert not empty
    assert realized == pytest.approx(80)

    # buy 100 at 9, the first 60 close the short for (10 - 9) * 60, the rest open a long
    empty, realized = pos.update(100, 9, 0, None)
    assert not empty
    assert realized == pytest.approx(60)
    assert pos.shares == 40
    assert pos.average_price == 9
    assert pos.realized == 0


@pytest.mark.parametrize(
    "shares, closed, price, average_price, percent",
    [
        # long 100 at 10, the 20% move of the shares left doesn't depend on how many were sold
        (100, -50, 12, 8, 0.2),
        (100, -90, 12, -8, 0.2),
        (100, -50, 20, 0, 1.0),
        (100, -50, 8, 12, -0.2),
        # short 100 at 10, the price falling 20% is a gain of 20%
        (-100, 50, 8, 12, 0.2),
        (-100, 90, 12, -8, -0.2),
        (-100, 50, 20, 0, -1.0),
    ],
)
def test_unrealized_percent_after_reducing(shares, closed, price, average_price, percent):
    pos = trading.Position(shares, 10, 0, None)
    pos.update(closed, price, 0, None)
    # (cost of the position - proceeds) / shares left, the break-even price of the shares left
    assert pos.average_price == pytest.approx(average_price)
    assert pos.unrealized_percent == pytest.approx(percent)
    assert pos.cost_basis == pytest.approx(10)


def lifetime_pnl(pos):
    """Realized P&L + the shares left marked to market against what they cost."""
    return pos.realized + (pos.last_price - pos.cost_basis) * pos.shares


def test_cost_basis_of_long():
    # 100 at 10 paying 10, each share cost 10.1
    pos = trading.Position(100, 10, 10, None)
    assert pos.cost_basis == pytest.approx(10.1)
    assert pos.cost_basis == pos.average_price

    # selling doesn't change the cost of the other shares
    pos.update(-60, 12, 6, None)
    assert pos.cost_basis == pytest.approx(10.1)
    # cash flow is -1010 + 720 - 6 = -296, and the 40 shares left are worth 480
    assert lifetime_pnl(pos) == pytest.approx(-296 + 480)

    # buying 60 at 12 paying 6 averages in: (40 * 10.1 + 60 * 12 + 6) / 100
    pos.update(60, 12, 6, None)
    assert pos.cost_basis == pytest.approx(11.3)
    # -296 - 726 = -1022, and 100 shares worth 1200
    assert lifetime_pnl(pos) == pytest.approx(-1022 + 1200)

    # a dividend lowers the cost like it lowers average_price, then a 2-for-1 split halves it
    assert pos.process_dividend(0.3, tax=0) == pytest.approx(30)
    assert pos.cost_basis == pytest.approx(11)
    pos.process_split(2, 11.7)
    assert pos.cost_basis == pytest.approx(5.5)
    # 5.85 now, the unrealized move is the same as before the split
    assert pos.unrealized_percent == pytest.approx(11.7 / 11 - 1)

    # reversing starts a new position: 200 shares closed, short 100 at 7 paying 10
    pos.update(-300, 7, 30, None)
    assert pos.shares == -100
    assert pos.cost_basis == pytest.approx(7 + 10 / -100)


def test_cost_basis_of_short():
    # short 100 at 10 paying 10, each share brought in 9.9
    pos = trading.Position(-100, 10, 10, None)
    assert pos.cost_basis == pytest.approx(9.9)

    # buying back doesn't change it for the other shares
    pos.update(60, 8, 6, None)
    assert pos.cost_basis == pytest.approx(9.9)
    # cash flow is 990 - 486 = 504, and the short 40 shares left are worth -320
    assert lifetime_pnl(pos) == pytest.approx(504 - 320)

    # shorting 60 more at 12 paying 6 averages in: (40 * 9.9 + 60 * 12 - 6) / 100
    pos.update(-60, 12, 6, None)
    assert pos.cost_basis == pytest.approx(11.1)
    # 504 + 714 = 1218, and the short 100 shares are worth -1200
    assert lifetime_pnl(pos) == pytest.approx(1218 - 1200)

    # the short pays the dividend, a 2-for-1 split halves what's left
    assert pos.process_dividend(0.1, tax=0) == pytest.approx(-10)
    assert pos.cost_basis == pytest.approx(11)
    pos.process_split(2, 11.9)
    assert pos.cost_basis == pytest.approx(5.5)
    assert pos.shares == -200


def test_split():
    pos = trading.Position(10, 10, 0, None)
    assert pos.process_split(2, 12) == 0
    assert pos.shares == 20
    assert pos.average_price == 5
    assert pos.last_price == 6


def test_reverse_split_remaining_to_cash():
    pos = trading.Position(970, 10, 0, None)
    # 1-for-15 reverse split, 970 = 64 * 15 + 10, 10 shares are converted to cash
    cash = pos.process_split(1 / 15, 9)
    assert pos.shares == 64
    assert cash == pytest.approx(10 * 9)
    assert pos.average_price == pytest.approx(150)
    assert pos.last_price == pytest.approx(135)


def test_dividend():
    pos = trading.Position(10, 10, 0, None)
    assert pos.process_dividend(0.5, tax=0.1) == 5
    assert pos.average_price == 9.5
    assert pos.last_price == pytest.approx(10 - 0.6)
    assert pos.process_dividend(float("nan"), tax=0) == 0


def test_split_adjusts_trailing_stop():
    # long position, stop at 90% of the highest price
    pos = trading.Position(10, 10, 0, None, trading.TrailingStopModel(-0.1))
    pos.last_price = 12
    assert pos.stop_tracker.stop_price == pytest.approx(12 * 0.9)

    # 2-for-1 split, the highest price should be adjusted to 6
    pos.process_split(2, 11)
    assert pos.last_price == 5.5
    assert pos.stop_tracker.stop_price == pytest.approx(6 * 0.9)
    assert not pos.check_stop_trigger()


def test_reverse_split_adjusts_trailing_stop_of_short():
    # short position, stop at 110% of the lowest price
    pos = trading.Position(-15, 10, 0, None, trading.TrailingStopModel(-0.1, callback=True))
    pos.last_price = 8
    assert pos.stop_tracker.stop_price == pytest.approx(8 * 1.1)

    # 1-for-15 reverse split, the lowest price should be adjusted to 120
    pos.process_split(1 / 15, 8.5)
    assert pos.shares == -1
    assert pos.last_price == pytest.approx(127.5)
    assert pos.stop_tracker.stop_price == pytest.approx(120 * 1.1)
    assert not pos.check_stop_trigger()


def test_split_adjusts_fixed_stop():
    # stop loss at 90% of the open price
    pos = trading.Position(10, 10, 0, None, trading.StopModel(-0.1, callback=True))
    assert pos.stop_tracker.stop_price == pytest.approx(9)
    pos.process_split(2, 9.5)
    assert pos.stop_tracker.stop_price == pytest.approx(4.5)
    assert not pos.check_stop_trigger()
    pos.last_price = 4.4
    assert pos.check_stop_trigger()
