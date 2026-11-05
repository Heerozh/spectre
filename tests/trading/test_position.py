import pandas as pd
import pytest

from spectre import trading


@pytest.mark.parametrize(
    "stop_model",
    [
        None,
        trading.TrailingStopModel(-0.1),
        trading.PnLDecayTrailingStopModel(-0.1, 0.1, None),
    ],
    ids=["no_stop", "trailing", "pnl_decay"],
)
def test_dividend_on_a_position_with_no_price(stop_model):
    dt = pd.Timestamp("2019-01-02", tz="UTC")
    pos = trading.Position(10, 5.0, 0.0, dt, stop_model=stop_model)
    pos.last_price = 0.0
    assert pos.process_dividend(1.0, 0.0) == 10.0


def test_dividend_of_the_whole_price_with_a_pnl_stop():
    dt = pd.Timestamp("2019-01-02", tz="UTC")
    stop_model = trading.PnLDecayTrailingStopModel(-0.1, 0.1, None)
    pos = trading.Position(10, 5.0, 0.0, dt, stop_model=stop_model)
    assert pos.process_dividend(5.0, 0.0) == 50.0
    assert pos.last_price == 0
    pos.check_stop_trigger()  # it computes the PnL% from a basis of 0


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
    # the new side starts from what the closed side realized
    assert pos.realized == pytest.approx((12 - 10) * 10 - 10)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_realized_is_kept_through_a_reversal(side):
    # the long case: 100 at 10, sell 150 at 12. 100 are closed 2 above their cost, 50 sold short
    pos = trading.Position(side * 100, 10, 0, None)
    empty, realized = pos.update(-side * 150, 12, 0, None)
    assert not empty
    assert pos.shares == -side * 50
    assert realized == pytest.approx(side * 200)
    assert pos.realized == pytest.approx(side * 200)
    assert pos.unrealized == 0

    # cash flow -1000 + 1800, and owing 50 shares at 11 now: 800 - 550
    pos.last_price = 11
    assert pos.realized + pos.unrealized == pytest.approx(side * (800 - 550))


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_realized_adds_up_over_several_reversals(side):
    pos = trading.Position(side * 100, 10, 0, None)
    pos.update(-side * 30, 11, 0, None)  # 30 closed 1 above their cost
    assert pos.realized == pytest.approx(side * 30)

    _, realized = pos.update(-side * 120, 12, 0, None)  # 70 closed 2 above, 50 on the other side
    assert realized == pytest.approx(side * 140)
    assert pos.realized == pytest.approx(side * (30 + 140))

    _, realized = pos.update(side * 80, 13, 0, None)  # 50 closed 1 against them, 30 flipped back
    assert realized == pytest.approx(side * -50)
    assert pos.shares == side * 30
    assert pos.realized == pytest.approx(side * (30 + 140 - 50))
    assert pos.unrealized == 0


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
    assert pos.realized == pytest.approx(80 + 60)


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
    return pos.realized + pos.unrealized


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_realized_plus_unrealized_is_the_total_pnl(side):
    # P&L of a long, a short is the opposite: its cash flows are the opposite of the long's
    pos = trading.Position(side * 100, 10, 0, None)
    assert pos.unrealized == 0

    # close 50 at 12. 100 is realized, and the 50 left are 2 above their cost
    pos.update(-side * 50, 12, 0, None)
    assert pos.realized == pytest.approx(side * 100)
    assert pos.unrealized == pytest.approx(side * 100)
    assert lifetime_pnl(pos) == pytest.approx(side * 200)

    # a dividend of 1 moves money, the holder is not richer: price 11, cost 9
    pos.process_dividend(1, tax=0)
    assert pos.unrealized == pytest.approx(side * (11 - 9) * 50)
    assert lifetime_pnl(pos) == pytest.approx(side * 200)

    # a 2-for-1 split changes nothing either: 100 shares, price 5.5, cost 4.5
    pos.process_split(2, 11)
    assert pos.unrealized == pytest.approx(side * (5.5 - 4.5) * 100)
    assert lifetime_pnl(pos) == pytest.approx(side * 200)

    # and the price moves: 100 shares gain 7 - 5.5 each
    pos.last_price = 7
    assert pos.realized == pytest.approx(side * 100)
    assert lifetime_pnl(pos) == pytest.approx(side * (200 + 150))


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


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_reverse_split_cash_out_is_realized(side):
    # 970 = 64 * 15 + 10, the 10 left over are closed at 9, 1 under their cost of 10
    pos = trading.Position(side * 970, 10, 0, None)
    cash = pos.process_split(1 / 15, 9)
    assert cash == pytest.approx(side * 10 * 9)
    assert pos.realized == pytest.approx(side * -10)
    # the 64 shares left are at 135, and each of them cost 150
    assert pos.unrealized == pytest.approx(side * 64 * (135 - 150))

    empty, realized = pos.update(-side * 64, 135, 0, None)
    assert empty
    assert realized == pytest.approx(side * 64 * (135 - 150))
    # cash flow of the long: -970 * 10 + 10 * 9 + 64 * 135
    assert pos.realized == pytest.approx(side * -970)


@pytest.mark.parametrize("ratio", [49, 98, 103, 107, 161, 187, 196, 197])
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_reverse_split_of_whole_shares(side, ratio):
    # ratio * (1 / ratio) can be 0.9999999999999999 in floats, it is exactly 1 share
    pos = trading.Position(side * ratio, 10, 0, None)
    assert pos.process_split(1 / ratio, 10) == 0
    assert pos.shares == side
    assert pos.last_price == pytest.approx(10 * ratio)
    assert pos.value == pytest.approx(side * 10 * ratio)


@pytest.mark.parametrize(
    "shares, new, old, kept, cash",
    [
        # new shares for old shares, at a price of 10. What is left over is paid at the new price
        (50, 1, 49, 1, 1 / 49 * 490),  # 1 share and 1 old share, worth 10
        (99, 1, 98, 1, 1 / 98 * 980),
        (15, 1, 2, 7, 0.5 * 20),  # 7.5 shares at 20
        (3, 2, 5, 1, 0.2 * 25),  # 1.2 shares at 25
        (2, 3, 4, 1, 0.5 * 40 / 3),  # 1.5 shares at 13.33
        (2, 2, 3, 1, 1 / 3 * 15),  # 1.33 shares at 15
        (15, 3, 2, 22, 0.5 * 20 / 3),  # 22.5 shares at 6.67
    ],
    ids=["1:49", "1:98", "1:2", "2:5", "3:4", "2:3", "3:2"],
)
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_split_pays_what_is_left_over(side, shares, new, old, kept, cash):
    pos = trading.Position(side * shares, 10, 0, None)
    assert pos.process_split(1 / (old / new), 10) == pytest.approx(side * cash)
    assert pos.shares == side * kept
    assert pos.value + side * cash == pytest.approx(side * shares * 10)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_forward_split_cash_out_is_realized(side):
    # 15 at 10, 3-for-2 at 12. That is 22.5 shares at 8, each of them cost 6.67
    pos = trading.Position(side * 15, 10, 0, None)
    assert pos.process_split(1.5, 12) == pytest.approx(side * 0.5 * 8)
    assert pos.shares == side * 22
    assert pos.realized == pytest.approx(side * 0.5 * (8 - 10 / 1.5))
    # the 15 shares made 2 each before the split, and it doesn't matter how they are split up
    assert pos.realized + pos.unrealized == pytest.approx(side * 15 * 2)


def test_split_conserves_value():
    ratios = [(1, old) for old in (2, 3, 7, 15, 49, 50, 98, 100, 103, 107, 161, 187, 196, 197)]
    ratios += [(2, 5), (3, 4), (2, 3), (3, 2), (5, 4), (7, 3), (2, 1), (3, 1)]
    counts = [*range(1, 400), 970, 9999, 49000, 99999]
    for new, old in ratios:
        inverse_ratio = 1 / (old / new)  # what SimulationBlotter.market_close passes
        for shares in counts:
            for side in (1, -1):
                pos = trading.Position(side * shares, 10, 0, None)
                cash = pos.process_split(inverse_ratio, 9)
                where = (new, old, side * shares)
                # integers, nothing here depends on how the float was rounded
                assert pos.shares == side * (shares * new // old), where
                assert abs(pos.value + cash - side * shares * 9) < 1e-6, where
                assert abs(pos.realized + pos.unrealized + side * shares) < 1e-6, where


# not a price: no close on the bar, a close of 0, or that close less a dividend
NO_PRICE = pytest.mark.parametrize("no_price", [float("nan"), 0, -0.5], ids=["nan", "zero", "neg"])


@NO_PRICE
def test_split_without_a_price_uses_the_last_price(no_price):
    pos = trading.Position(10, 10, 0, None)
    assert pos.process_split(2, no_price) == 0
    assert pos.shares == 20
    assert pos.last_price == 5
    assert pos.value == 100


@NO_PRICE
def test_reverse_split_without_a_price_uses_the_last_price(no_price):
    pos = trading.Position(970, 10, 0, None)
    # 970 = 64 * 15 + 10, the 10 left over are sold at the last price of 10
    cash = pos.process_split(1 / 15, no_price)
    assert pos.shares == 64
    assert cash == pytest.approx(10 * 10)
    assert pos.last_price == pytest.approx(150)


def test_dividend():
    pos = trading.Position(10, 10, 0, None)
    assert pos.process_dividend(0.5, tax=0.1) == 5
    assert pos.average_price == 9.5
    assert pos.last_price == pytest.approx(10 - 0.6)
    assert pos.process_dividend(float("nan"), tax=0) == 0


@pytest.mark.parametrize("amount, tax", [(0.5, 0.1), (0, 0.6)], ids=["taxed", "all_tax"])
def test_short_pays_the_gross_dividend(amount, tax):
    # the tax comes off what the receiver gets, the short pays the whole dividend of 0.6
    pos = trading.Position(-10, 10, 0, None)
    assert pos.process_dividend(amount, tax=tax) == pytest.approx(-10 * 0.6)
    assert pos.average_price == pytest.approx(10 - 0.6)
    assert pos.cost_basis == pytest.approx(10 - 0.6)
    assert pos.last_price == pytest.approx(10 - 0.6)
    # the shares owed are 0.6 cheaper, and that is what it paid
    assert pos.unrealized == pytest.approx(0)


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


# a dividend of 6 with and without tax, the price drops by 6 either way
TAX = pytest.mark.parametrize("amount, tax", [(6, 0), (5.4, 0.6)], ids=["no_tax", "taxed"])


@TAX
def test_dividend_adjusts_trailing_stop(amount, tax):
    # long position, stop at 95% of the highest price
    pos = trading.Position(100, 100, 0, None, trading.TrailingStopModel(-0.05, callback=True))

    # the price goes to 94, below the stop of 95, and the holder got 6 for it
    pos.process_dividend(amount, tax=tax)
    assert pos.last_price == pytest.approx(94)
    assert pos.stop_tracker.stop_price == pytest.approx(94 * 0.95)
    assert not pos.check_stop_trigger()

    # a real fall of 6% still fires it
    pos.last_price = 94 * 0.94
    assert pos.check_stop_trigger()


@TAX
def test_dividend_adjusts_trailing_stop_of_short(amount, tax):
    # short position, stop at 105% of the lowest price
    pos = trading.Position(-100, 100, 0, None, trading.TrailingStopModel(-0.05, callback=True))
    pos.last_price = 96
    pos.last_price = 100
    assert pos.stop_tracker.stop_price == pytest.approx(96 * 1.05)

    # the fall to 94 is not a new low. The low falls 6% with the price, 4% below it like before
    pos.process_dividend(amount, tax=tax)
    assert pos.last_price == pytest.approx(94)
    assert pos.stop_tracker.recorded_price == pytest.approx(96 * 0.94)
    assert pos.stop_tracker.stop_price == pytest.approx(96 * 0.94 * 1.05)
    assert not pos.check_stop_trigger()


@TAX
def test_dividend_adjusts_fixed_stop(amount, tax):
    # stop loss at 90% of the open price, it stays 10% below the price after a dividend of 6%
    pos = trading.Position(100, 100, 0, None, trading.StopModel(-0.1, callback=True))
    pos.process_dividend(amount, tax=tax)
    assert pos.stop_tracker.stop_price == pytest.approx(90 * 0.94)
    assert not pos.check_stop_trigger()
    pos.last_price = 84.5
    assert pos.check_stop_trigger()


# a dividend of 20 with and without tax
TAX_20 = pytest.mark.parametrize("amount, tax", [(20, 0), (18, 2)], ids=["no_tax", "taxed"])


@TAX_20
def test_dividend_inside_the_gap_of_a_trailing_stop(amount, tax):
    # long, high of 110 and stop at 99, the price of 100 is just above it
    pos = trading.Position(100, 100, 0, None, trading.TrailingStopModel(-0.1, callback=True))
    pos.last_price = 110
    pos.last_price = 100
    assert pos.stop_tracker.stop_price == pytest.approx(99)
    assert not pos.check_stop_trigger()

    # 100 to 80 is a total return of 0. The dividend is 20% of the price, all of it moves by 20%
    pos.process_dividend(amount, tax=tax)
    assert pos.last_price == pytest.approx(80)
    assert pos.stop_tracker.recorded_price == pytest.approx(110 * 0.8)
    assert pos.stop_tracker.stop_price == pytest.approx(99 * 0.8)
    assert not pos.check_stop_trigger()

    # a bar that really falls through it
    pos.last_price = 79
    assert pos.check_stop_trigger()


@TAX_20
def test_dividend_inside_the_gap_of_a_trailing_stop_of_short(amount, tax):
    # short, low of 90 and stop at 99, the price of 98 is just below it
    pos = trading.Position(-100, 100, 0, None, trading.TrailingStopModel(-0.1, callback=True))
    pos.last_price = 90
    pos.last_price = 98
    assert pos.stop_tracker.stop_price == pytest.approx(99)
    assert not pos.check_stop_trigger()

    # 98 to 78, everything moves by that proportion and stays 1% from the stop
    pos.process_dividend(amount, tax=tax)
    assert pos.last_price == pytest.approx(78)
    assert pos.stop_tracker.recorded_price == pytest.approx(90 * 78 / 98)
    assert pos.stop_tracker.stop_price == pytest.approx(99 * 78 / 98)
    assert not pos.check_stop_trigger()

    pos.last_price = 79
    assert pos.check_stop_trigger()


def test_return_basis_follows_the_price_through_a_dividend():
    # 100 at 50 and 100 at 60 is a cost of 55, the price is 110
    pos = trading.Position(100, 50, 0, None)
    pos.update(100, 60, 0, None)
    assert pos.return_basis == pytest.approx(55)
    pos.last_price = 110

    # a dividend of 10% of the price takes 10% off, the cost basis loses the 11 that was paid
    pos.process_dividend(11, tax=0)
    assert pos.return_basis == pytest.approx(55 * 0.9)
    assert pos.cost_basis == pytest.approx(55 - 11)

    # sales don't touch it, and a split does what it does to the prices
    pos.update(-100, 99, 0, None)
    assert pos.return_basis == pytest.approx(55 * 0.9)
    pos.process_split(2, 99)
    assert pos.return_basis == pytest.approx(55 * 0.9 / 2)

    # the split made 200 shares of them, 100 more at 40 averages in
    pos.update(100, 40, 0, None)
    assert pos.return_basis == pytest.approx((200 * 24.75 + 100 * 40) / 300)

    # a reversal starts over
    pos.update(-500, 30, 0, None)
    assert pos.return_basis == 30


def test_dividend_and_split_adjust_trailing_stop():
    pos = trading.Position(100, 100, 0, None, trading.TrailingStopModel(-0.05, callback=True))
    # both on the same day, the highest price is (100 - 2) * 0.5 now
    pos.process_dividend(2, tax=0)
    pos.process_split(2, 98)
    assert pos.last_price == pytest.approx(49)
    assert pos.stop_tracker.stop_price == pytest.approx(49 * 0.95)
    assert not pos.check_stop_trigger()
