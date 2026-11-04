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
