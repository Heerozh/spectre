import numpy as np
import pandas as pd
import pytest
from numpy import nan

from spectre import trading

# AAPL and MSFT daily prices (uOpen, uClose, uVolume) used in the tests
AAPL_0102 = dict(open=155.92, close=158.61, volume=37932561)
MSFT_0109_CLOSE = 106.0
MSFT_DIVIDEND = 0.46 + 0.11  # ex-date 2019-01-11, paid at the close of 2019-01-09
AAPL_0110_CLOSE, AAPL_0109_CLOSE = 161.0, 156.03
AAPL_0111_CLOSE, MSFT_0111_CLOSE = 153.69, 103.2  # MSFT 1-for-15 reverse split on 2019-01-14
AAPL_0115_CLOSE, MSFT_0115_CLOSE = 156.94, 108.85


def ts(date):
    return pd.Timestamp(date, tz='UTC')


def per_share_commission(shares):
    """
    set_commission(0, 0.005, 1), rounded to cents. Prices are np.float64, so the commission is
    rounded by numpy, 1.925 is rounded to 1.92 (python's round gives 1.93).
    """
    return float(np.round(max(np.float64(abs(shares)) * 0.005, 1), 2))


# ------------------------------------ models ------------------------------------


def test_commission_model():
    model = trading.CommissionModel(0.001, 0.005, 1)
    assert model.calculate('AAPL', 100, 1000) == pytest.approx(100 * 1000 * 0.001 + 5)
    assert model.calculate('AAPL', 100, -1000) == pytest.approx(100 * 1000 * 0.001 + 5)
    # minimum
    assert model.calculate('AAPL', 1, 1) == 1
    assert trading.CommissionModel(0, 0, 0).is_empty()


def test_slippage_model():
    model = trading.blotter.SlippageModel(0.01, 0.1)
    # sigmoid of amount/volume, from 1 / (1 + 200^0.5) at 0 to 1 / (1 + 200^-0.5) at max ratio
    low, high = 1 / (1 + 200**0.5), 1 / (1 + 200**-0.5)
    assert model.calculate('AAPL', 100, 0) == pytest.approx(100 + 100 * 0.01 * low)
    assert model.calculate('AAPL', 100, 0.1) == pytest.approx(100 + 100 * 0.01 * high)
    # sell orders get lower price
    assert model.calculate('AAPL', 100, -0.1) == pytest.approx(100 - 100 * 0.01 * high)
    assert model.calculate('AAPL', 100, -0.05) == pytest.approx(
        200 - model.calculate('AAPL', 100, 0.05)
    )
    # no slippage
    assert trading.blotter.SlippageModel(0.01, 0).calculate('AAPL', 100, 0.1) == 100
    assert trading.blotter.SlippageModel(0, 0).is_empty()


def test_daily_curb_model():
    model = trading.DailyCurbModel(0.1)
    # last close is adjusted by dividend and split: (100 - 1) * 0.5
    assert model.calculate('AAPL', 54.4, 10, 0, 0, 100, 1, 0.5) == (54.4, 10)
    assert model.calculate('AAPL', 54.45, 10, 0, 0, 100, 1, 0.5) == (None, 0)
    assert model.calculate('AAPL', 44.55, 10, 0, 0, 100, 1, 0.5) == (None, 0)


# ------------------------------------ simulation blotter ------------------------------------


@pytest.fixture
def blotter(adjusted_daily_loader_no_calendar):
    blotter = trading.SimulationBlotter(adjusted_daily_loader_no_calendar, capital_base=200000)
    blotter.set_commission(0, 0.005, 1)
    blotter.borrow_money_interest_rate = 0
    blotter.borrow_stock_interest_rate = 0
    return blotter


def test_out_of_market_hours(blotter):
    blotter.set_datetime(ts('2019-01-02'))
    blotter.set_price('open')
    with pytest.raises(RuntimeError, match='Out of market hours'):
        blotter.order('AAPL', 1)


def test_order_checks(blotter):
    blotter.set_datetime(ts('2019-01-02'))
    blotter.market_open(None)
    blotter.set_price('open')
    with pytest.raises(OverflowError):
        blotter.order('AAPL', blotter.max_shares + 1)
    with pytest.raises(KeyError, match='must be a string'):
        blotter.order(1, 1)
    with pytest.raises(ValueError, match='must be float'):
        blotter.order_target_percent('AAPL', 1)
    blotter.long_only = True
    with pytest.raises(ValueError, match='Long only'):
        blotter.order('AAPL', -1)
    blotter.long_only = False
    blotter.order_multiplier = 100
    with pytest.raises(ValueError, match='multiples of 100'):
        blotter.order('AAPL', 10)


def test_t0_and_double_purchase(blotter):
    date = ts('2019-01-02')
    blotter.set_datetime(date)
    blotter.market_open(None)
    blotter.set_price('open')
    blotter.order_target_percent('AAPL', 0.3)
    # already reached the target, no order
    blotter.order_target_percent('AAPL', 0.3)
    blotter.set_price('close')
    blotter.order_target_percent('AAPL', 0.0)
    blotter.market_close(None)
    blotter.update_portfolio_value()

    shares = int(round(200000 * 0.3 / AAPL_0102['open']))
    assert shares == 385
    commission = per_share_commission(shares)
    realized = (AAPL_0102['close'] - AAPL_0102['open']) * shares - commission * 2
    open_, close = AAPL_0102['open'], AAPL_0102['close']
    expected = pd.DataFrame(
        [
            ['AAPL', shares, open_, open_, commission, 0.0],
            ['AAPL', -shares, close, close, commission, realized],
        ],
        columns=['symbol', 'amount', 'price', 'fill_price', 'commission', 'realized'],
        index=[date, date],
    )
    expected.index.name = 'index'
    pd.testing.assert_frame_equal(expected, blotter.get_transactions())

    expected = pd.DataFrame(
        [[200000 + realized]], columns=pd.MultiIndex.from_tuples([('value', 'cash')]), index=[date]
    )
    expected.index.name = 'index'
    pd.testing.assert_frame_equal(expected, blotter.get_historical_positions())


def test_slippage(blotter):
    blotter.set_slippage(0.005, 0.4)
    date = ts('2019-01-02')
    blotter.set_datetime(date)
    blotter.market_open(None)
    blotter.set_price('open')
    blotter.order('AAPL', 385)
    blotter.set_price('close')
    blotter.order('AAPL', -385)
    blotter.market_close(None)

    slippage = 0.005 / (1 + 200 ** (0.5 - 385 / AAPL_0102['volume'] / 0.4))
    buy_price = AAPL_0102['open'] * (1 + slippage)
    sell_price = AAPL_0102['close'] * (1 - slippage)
    txn = blotter.get_transactions()
    # `price` is the market price, `fill_price` includes the slippage
    assert list(txn.price) == [AAPL_0102['open'], AAPL_0102['close']]
    assert txn.fill_price.iloc[0] == pytest.approx(buy_price)
    assert txn.fill_price.iloc[1] == pytest.approx(sell_price)
    commission = per_share_commission(385)
    assert txn.realized.iloc[1] == pytest.approx((sell_price - buy_price) * 385 - commission * 2)
    assert blotter.portfolio.cash == pytest.approx(
        200000 - 385 * buy_price + 385 * sell_price - commission * 2
    )


def test_dividend_curb_and_split(blotter):
    """
    A sequence of trading days:
    01-09 buy MSFT before dividend, 01-10 MSFT has no data, AAPL hits daily curb,
    01-11 long/short before MSFT reverse split, 01-15 check positions.
    """
    cash = 200000

    # 01-09 dividend, paid at the close of the day before ex-date
    blotter.set_datetime(ts('2019-01-09'))
    blotter.market_open(None)
    blotter.set_price('close')
    blotter.order('MSFT', 2)
    cash -= 2 * MSFT_0109_CLOSE + per_share_commission(2)
    assert blotter.portfolio.cash == pytest.approx(cash)
    blotter.market_close(None)
    blotter.update_portfolio_value()
    cash += 2 * MSFT_DIVIDEND
    assert blotter.portfolio.cash == pytest.approx(cash)

    # 01-10 no data of MSFT
    blotter.set_datetime(ts('2019-01-10'))
    blotter.market_open(None)
    blotter.set_price('close')
    assert blotter.order_target_percent('MSFT', 0.5) is False
    # AAPL changed 3.2% from last close, curbed by 1%
    assert AAPL_0110_CLOSE / AAPL_0109_CLOSE - 1 == pytest.approx(0.0319, abs=1e-4)
    blotter.daily_curb = trading.DailyCurbModel(0.01)
    assert blotter.order('AAPL', 1) is False
    assert blotter.portfolio.shares('AAPL') == 0
    assert 'AAPL' not in blotter.positions
    blotter.daily_curb = trading.DailyCurbModel(0.033)
    blotter.order('AAPL', 1)
    assert blotter.positions['AAPL'].shares == 1
    blotter.order('AAPL', -1)
    blotter.market_close(None)
    blotter.update_portfolio_value()
    blotter.daily_curb = None
    cash -= per_share_commission(1) * 2
    assert blotter.portfolio.cash == pytest.approx(cash)

    # 01-11 overnight neutralized portfolio, MSFT valued by the close of 01-09
    blotter.set_datetime(ts('2019-01-11'))
    blotter.market_open(None)
    blotter.set_price('close')
    value = cash + 2 * MSFT_0109_CLOSE
    blotter.order_target_percent('AAPL', -0.5)
    blotter.order_target_percent('MSFT', 0.5)
    aapl_shares = int(round(value * -0.5 / AAPL_0111_CLOSE))
    aapl_commission = per_share_commission(aapl_shares)
    value -= aapl_commission
    msft_shares = int(round(value * 0.5 / MSFT_0111_CLOSE))
    msft_commission = per_share_commission(msft_shares - 2)
    assert (aapl_shares, msft_shares) == (-651, 969)
    blotter.market_close(None)
    blotter.update_portfolio_value()

    # reverse split, the remaining shares are converted to cash
    msft_split_shares = msft_shares // 15
    assert blotter.positions['MSFT'].shares == msft_split_shares
    remaining = msft_shares - msft_split_shares * 15
    cash += -aapl_shares * AAPL_0111_CLOSE - aapl_commission
    cash -= (msft_shares - 2) * MSFT_0111_CLOSE + msft_commission
    cash += remaining * MSFT_0111_CLOSE
    assert blotter.portfolio.cash == pytest.approx(cash)

    # 01-15, 01-14 is filled up automatically
    date = ts('2019-01-15')
    blotter.set_datetime(date)
    blotter.market_open(None)
    blotter.set_price('close')
    blotter.market_close(None)
    blotter.update_portfolio_value()

    aapl_basis = AAPL_0111_CLOSE + aapl_commission / aapl_shares
    msft_cost = (
        2 * MSFT_0109_CLOSE
        + per_share_commission(2)
        - 2 * MSFT_DIVIDEND
        + (msft_shares - 2) * MSFT_0111_CLOSE
        + msft_commission
    )
    msft_basis = msft_cost / msft_shares * 15
    expected = pd.Series(
        [
            aapl_basis,
            msft_basis,
            aapl_shares,
            msft_split_shares,
            aapl_shares * AAPL_0115_CLOSE,
            msft_split_shares * MSFT_0115_CLOSE,
            cash,
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ('avg_px', 'AAPL'),
                ('avg_px', 'MSFT'),
                ('shares', 'AAPL'),
                ('shares', 'MSFT'),
                ('value', 'AAPL'),
                ('value', 'MSFT'),
                ('value', 'cash'),
            ]
        ),
        name=date,
    )
    pd.testing.assert_series_equal(
        expected, blotter.get_historical_positions().iloc[-1], check_dtype=False
    )


def test_intraday(intraday_loader):
    blotter = trading.SimulationBlotter(intraday_loader, capital_base=200000)
    blotter.set_commission(0, 0.005, 1)
    blotter.set_slippage(0.005, 0.4)

    datetime1 = ts('2019-01-02 14:35:00')
    blotter.set_datetime(datetime1)
    blotter.market_open(None)
    blotter.set_price('open')
    blotter.set_price('close')
    blotter.order_target_percent('AAPL', 0.3)

    # intraday orders are not out of market hours
    datetime2 = ts('2019-01-02 14:40:00')
    blotter.set_datetime(datetime2)
    blotter.set_price('close')
    blotter.order_target_percent('AAPL', 0.0)
    blotter.market_close(None)
    blotter.update_portfolio_value()

    # close prices of the 5 minutes bars
    txn = blotter.get_transactions()
    assert txn.loc[datetime1].price == 154.65
    assert txn.loc[datetime2].price == 155.0
    assert txn.loc[datetime1].fill_price > 154.65
    assert txn.loc[datetime2].fill_price < 155.0


def test_intraday_stop_model(intraday_loader):
    blotter = trading.SimulationBlotter(intraday_loader, capital_base=200000)
    # close the position if the price raised 0.2% from the lowest price
    blotter.portfolio.set_stop_model(trading.TrailingStopModel(0.002, blotter.order))

    blotter.set_datetime(ts('2019-01-02 14:45:00'))
    blotter.market_open(None)
    blotter.set_price('close')
    blotter.order_target_percent('AAPL', 1.0)
    blotter.update_portfolio_value()
    assert blotter.portfolio.check_stop_trigger() == [False]
    open_price = blotter.positions['AAPL'].last_price

    blotter.set_datetime(ts('2019-01-02 14:50:00'))
    blotter.set_price('close')
    blotter.update_portfolio_value()
    assert blotter.positions['AAPL'].last_price >= open_price * 1.002
    assert blotter.portfolio.check_stop_trigger() == [True]
    blotter.market_close(None)
    assert len(blotter.portfolio.positions) == 0


# ------------------------------------ manual blotter ------------------------------------


def test_manual_blotter(tmp_path):
    working_dir = str(tmp_path) + '/'
    blotter = trading.ManualBlotter(working_dir, 'Asia/Shanghai')

    day1 = pd.Timestamp('2020-06-01', tz='Asia/Shanghai')
    blotter.set_datetime(day1)
    blotter.transfer_funds(10000)
    blotter.set_last_price({'test': 1, 'batch1': 1, 'batch2': 2})
    test_oid = blotter.order_target_percent('test', 0.3)
    batch_ids = blotter.batch_order_target_percent(['batch1', 'batch2'], [0.1, 0.2])
    assert len(blotter.pendings) == 3
    blotter.order_filled(batch_ids['batch2'], 10, 200, 5)
    blotter.order_filled(test_oid, 5, 200, 5)

    day2 = pd.Timestamp('2020-06-02', tz='Asia/Shanghai')
    blotter.set_datetime(day2)
    blotter.transfer_funds(20000)
    blotter.order_cancelled(batch_ids['batch1'])
    test_oid = blotter.order_target_percent('test', 0.0)
    blotter.order_filled(test_oid, -5, 300, 5)
    blotter.position_dividend('batch2', 1.5, tax=0, time_delta=pd.Timedelta(hours=23))
    blotter.position_split('batch2', 1.5, 200, time_delta=pd.Timedelta(hours=23))
    assert len(blotter.pendings) == 0

    # fill_price includes commission
    expected_txn = pd.DataFrame(
        [
            ['test', 5.0, 200.0, 201.0, 5.0, 0.0],
            ['batch2', 10.0, 200.0, 200.5, 5.0, 0.0],
            ['test', -5.0, 300.0, 299, 5.0, 490.0],
        ],
        columns=['symbol', 'amount', 'price', 'fill_price', 'commission', 'realized'],
        index=pd.DatetimeIndex([day1, day1, day2]),
    )
    expected_txn.index.name = 'index'
    pd.testing.assert_frame_equal(expected_txn, blotter.get_transactions())

    expected_pos = pd.DataFrame(
        [
            [200.5, 201.0, 10, 5.0, 2000.0, 6990.0, 1000.0],
            [132.666667, nan, 15, nan, 2000.0, 28500.0, nan],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ('avg_px', 'batch2'),
                ('avg_px', 'test'),
                ('shares', 'batch2'),
                ('shares', 'test'),
                ('value', 'batch2'),
                ('value', 'cash'),
                ('value', 'test'),
            ]
        ),
        index=pd.DatetimeIndex([day1, day2]),
    )
    expected_pos.index.name = 'index'
    pd.testing.assert_frame_equal(expected_pos, blotter.portfolio.history)

    # save and load
    blotter.save()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'orders_2020-06-01.csv',
        'orders_2020-06-02.csv',
    ]
    new_blotter = trading.ManualBlotter(working_dir, 'Asia/Shanghai')
    expected_pos[('shares', 'batch2')] = expected_pos[('shares', 'batch2')].astype('float')
    pd.testing.assert_frame_equal(expected_txn, new_blotter.get_transactions())
    pd.testing.assert_frame_equal(expected_pos, new_blotter.portfolio.history)

    # order_multiplier
    new_blotter.order_multiplier = 10
    new_blotter.set_datetime(pd.Timestamp('2020-06-03', tz='Asia/Shanghai'))
    new_blotter.set_last_price({'test': 1, 'batch1': 1, 'batch2': 2})
    new_blotter.transfer_funds(-30000)
    # target shares = 0.06 * 500 / 2 = 15, same as position
    new_blotter.order_target_percent('batch2', 0.06)
    assert new_blotter.orders.iloc[-1].amount == 0
    # target shares = 0.072 * 500 / 2 = 18, buy 3, not meet multiplier
    new_blotter.order_target_percent('batch2', 0.072)
    assert new_blotter.orders.iloc[-1].amount == 0
    # target shares = 0.084 * 500 / 2 = 21, buy 6, round to 10
    new_blotter.order_target_percent('batch2', 0.084)
    assert new_blotter.orders.iloc[-1].amount == 10


def test_manual_blotter_needs_last_price(tmp_path):
    blotter = trading.ManualBlotter(str(tmp_path) + '/', 'Asia/Shanghai')
    blotter.set_datetime(pd.Timestamp('2020-06-01', tz='Asia/Shanghai'))
    blotter.transfer_funds(10000)
    with pytest.raises(ValueError, match='set_last_price'):
        blotter.order_target_percent('test', 0.3)
    with pytest.raises(NotImplementedError):
        blotter.order('test', 1)
