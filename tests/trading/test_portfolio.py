import pandas as pd
import pytest
from numpy import nan
from numpy.testing import assert_almost_equal

from spectre import trading


def aapl_1_5_msft_2(asset):
    return 1.5 if asset == "AAPL" else 2


@pytest.fixture
def portfolio():
    """
    A portfolio with long AAPL, short MSFT, a split, dividends and a reversed position.
    Note: Portfolio.update doesn't touch cash, cash only changes by update_cash/dividends/splits.
    """
    pf = trading.Portfolio()
    pf.set_datetime("2019-01-01")
    pf.update_cash(10000, is_funds=True)

    pf.set_datetime("2019-01-03")
    pf.update("AAPL", 10, 9, 10)  # avg_px = 9 + 10 / 10 = 10
    pf.update("MSFT", -25, 9, 25)  # avg_px = 9 + 25 / -25 = 8
    pf.update_cash(-5000, is_funds=True)
    pf.process_split("AAPL", 2.0, 3)  # 20 shares, avg_px = 5, last price = 3 / 2

    pf.set_datetime("2019-01-04")
    pf.update("AAPL", -10, 12, 10)

    pf.set_datetime("2019-01-05 23:00:00")
    pf.update("MSFT", 30, 7, 30)  # cover 25 shares and then long 5 shares
    pf.update_cash(2000, is_funds=True)

    pf.process_dividend("AAPL", 2.0, tax=0)
    pf.process_dividend("MSFT", 2.0, tax=0)
    pf.update_value(aapl_1_5_msft_2)
    return pf


def test_realized(portfolio):
    # (12 - 5) * 10 - 10 commission
    assert portfolio.positions["AAPL"].realized == 60
    # Reversed position, keeps what the closed side realized: (8 - 7) * 25 - 25 commission
    assert portfolio.positions["MSFT"].realized == 0


def test_positions_and_cash(portfolio):
    assert {k: pos.shares for k, pos in portfolio.positions.items()} == {"AAPL": 10, "MSFT": 5}
    # funds 10000 - 5000 + 2000, dividends 2 * 10 + 2 * 5
    assert portfolio.cash == 7030


def test_history(portfolio):
    expected = pd.DataFrame(
        [
            [nan, nan, nan, nan, nan, nan, 10000.0],
            [5.0, 8.0, 20.0, -25, 30, -225, 5000.0],
            [-1.0, 8.0, 10.0, -25, 120, -225, 5000.0],
            [-3.0, 6.0, 10.0, 5, 15, 10, 7030.0],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("avg_px", "AAPL"),
                ("avg_px", "MSFT"),
                ("shares", "AAPL"),
                ("shares", "MSFT"),
                ("value", "AAPL"),
                ("value", "MSFT"),
                ("value", "cash"),
            ]
        ),
        index=pd.DatetimeIndex(["2019-01-01", "2019-01-03", "2019-01-04", "2019-01-05"]),
    )
    expected.index.name = "index"
    pd.testing.assert_frame_equal(expected, portfolio.history, check_index_type=False)


def test_repr(portfolio):
    expected = """<Portfolio>avg_px      shares        value
             AAPL MSFT   AAPL  MSFT   AAPL   MSFT     cash
index
2019-01-01    NaN  NaN    NaN   NaN    NaN    NaN  10000.0
2019-01-03    5.0  8.0   20.0 -25.0   30.0 -225.0   5000.0
2019-01-04   -1.0  8.0   10.0 -25.0  120.0 -225.0   5000.0
2019-01-05   -3.0  6.0   10.0   5.0   15.0   10.0   7030.0"""
    # ignore the trailing spaces of pandas' formatting
    assert [line.rstrip() for line in str(portfolio).splitlines()] == expected.splitlines()


def test_value_and_leverage(portfolio):
    assert portfolio.value == 7030 + 15 + 10
    assert portfolio.leverage == pytest.approx(25 / 7055)

    portfolio.set_datetime("2019-01-06")
    portfolio.update_cash(-6830, is_funds=True)
    portfolio.update("AAPL", -100, 1, 0)  # reverse to short 90
    portfolio.update("MSFT", 50, 1, 0)
    assert portfolio.cash == 200
    portfolio.update_value(aapl_1_5_msft_2)
    assert portfolio.value == 200 - 135 + 110
    assert portfolio.leverage == pytest.approx(245 / 175)

    portfolio.set_datetime("2019-01-07")
    portfolio.update_cash(-200, is_funds=True)
    portfolio.update("AAPL", 400, 1.5, 0)
    portfolio.update("MSFT", 50, 2, 0)
    portfolio.update_cash(-337.5, is_funds=True)
    portfolio.update_value(lambda x: None)  # None price is ignored, last fill price is used
    # (310 * 1.5 + 105 * 2) / (675 - 337.5)
    assert portfolio.leverage == pytest.approx(2)


def test_set_datetime_backwards():
    pf = trading.Portfolio()
    pf.set_datetime("2019-01-03")
    with pytest.raises(ValueError, match="Cannot set a date less than the current date"):
        pf.set_datetime("2019-01-02")


def test_cash_of_a_split():
    pf = trading.Portfolio()
    pf.set_datetime("2019-01-03")
    pf.update_cash(1000, is_funds=True)
    pf.update("AAPL", 15, 10, 0)
    pf.update_cash(-150)

    # 3-for-2 is 22.5 shares at 6.67, half a share is paid in cash
    pf.process_split("AAPL", 1.5, 10)
    assert pf.shares("AAPL") == 22
    assert pf.cash == pytest.approx(850 + 0.5 * 10 / 1.5)
    assert pf.value == pytest.approx(1000)
    # that is not a transfer of funds, the deposit is the only one
    assert len(pf.fund_history) == 1


@pytest.mark.parametrize("amount", [5, -5], ids=["long", "short"])
def test_split_down_to_no_shares_closes_the_position(amount):
    pf = trading.Portfolio(trading.TrailingStopModel(-0.05, True))
    pf.set_datetime("2019-01-03")
    pf.update_cash(1000, is_funds=True)
    pf.update("AAPL", amount, 10, 0)
    pf.update_cash(-amount * 10)

    # 1-for-10 is half a share at 100, which is paid in cash
    pf.process_split("AAPL", 0.1, 10)
    assert "AAPL" not in pf.positions
    assert pf.shares("AAPL") == 0
    assert pf.value == pytest.approx(1000)

    # the next order opens a position that knows nothing of the old one
    pf.set_datetime("2019-01-08")
    pf.update("AAPL", 3, 90, 0)
    pos = pf.positions["AAPL"]
    assert pos.open_dt == pd.Timestamp("2019-01-08")
    assert pos.stop_tracker.stop_price == pytest.approx(90 * 0.95)


def test_update_returns_realized():
    pf = trading.Portfolio()
    pf.set_datetime("2019-01-03")
    assert pf.update("AAPL", 10, 10, 0) == 0
    pos = pf.positions["AAPL"]
    assert pf.update("AAPL", -5, 11, 0) == 5
    assert pf.update("AAPL", -5, 9, 0) == -5
    # position closed, total realized is 5 - 5
    assert pos.realized == 0
    assert "AAPL" not in pf.positions


def test_returns():
    pf = trading.Portfolio()
    # funds transferred before the first datetime is set, count as the first day
    pf.update_cash(10000, is_funds=True)
    pf.set_datetime("2019-01-01")
    pf.update("AAPL", 100, 10, 0)
    pf.update_cash(-1000)
    pf.update_value({"AAPL": 10})

    pf.set_datetime("2019-01-02")
    pf.update_value({"AAPL": 11})
    pf.update_cash(5000, is_funds=True)
    pf.update_cash(-1000, is_funds=True)  # 2 transfers on the same day

    pf.set_datetime("2019-01-03")
    pf.update_value({"AAPL": 12})

    returns = pf.returns
    assert list(returns.index) == list(pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]))
    # funds transfers are not returns
    assert_almost_equal(returns.values, [nan, 100 / 10000, 100 / 14100])


def test_returns_without_funds():
    pf = trading.Portfolio()
    pf.set_datetime("2019-01-01")
    pf.update("AAPL", 100, 10, 0)
    pf.update_value({"AAPL": 10})
    pf.set_datetime("2019-01-02")
    pf.update_value({"AAPL": 11})
    assert_almost_equal(pf.returns.values, [nan, 0.1])
