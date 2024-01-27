"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Iterable
# import pandas as pd
import numpy as np
import os
import glob
from collections import namedtuple, defaultdict
from .event import *
from .portfolio import Portfolio
from ..data import DataLoaderFastGetter


class CommissionModel:
    def __init__(self, percentage: float, per_share: float, minimum: float):
        self.percentage = percentage
        self.per_share = per_share
        self.minimum = minimum

    def is_empty(self):
        return (self.percentage + self.per_share + self.minimum) == 0

    def calculate(self, asset: str, price: float, shares: int):
        commission = price * abs(shares) * self.percentage
        commission += abs(shares) * self.per_share
        return max(commission, self.minimum)


class SlippageModel(CommissionModel):
    def __init__(self, max_percentage: float, max_amount_to_volume_ratio: float = 2e-4):
        """
        When amount_to_volume_ratio reaches max, the slippage will be
            price * max_percentage, uses sigmod interpolate when ratio below max.
        """
        super().__init__(max_percentage, 0, 0)
        self.max_ratio = max_amount_to_volume_ratio

    def calculate(self, asset: str, price: float, amount_to_volume_ratio: int):
        def sigmoid(x, edge0, edge1):
            return (1 + 200 ** (-((x - edge0) / (edge1 - edge0)) + 0.5)) ** (-1)
        slippage = price * self.percentage * sigmoid(
            amount_to_volume_ratio, 0, self.max_ratio)
        return price + max(slippage, self.minimum)


class DailyCurbModel:
    def __init__(self, percentage: float):
        self.percentage = percentage

    def calculate(self, asset: str, current_price: float, shares: int,
                  current_high: float, current_low: float,
                  last_close: float, last_div: float, last_sp: float):
        """
        :param asset: asset
        :param current_price: current price
        :param shares: trading quantity
        :param current_high: current bar high price, may be lookahead biased.
        :param current_low: current bar low price, may be lookahead biased.
        :param last_close: previous day close price
        :param last_div: previous day dividend
        :param last_sp: previous day split ratio
        :return: float, int: filled price & amount
        """
        last_close = (round(last_close, 2) - last_div) * last_sp
        last_close = round(last_close, 2)
        if abs(current_price / last_close - 1) >= self.percentage:
            return None, 0
        else:
            return current_price, shares


class BaseBlotter:
    """
    Base class for Order Management System.
    """
    max_shares = int(1e+5)

    def __init__(self) -> None:
        super().__init__()
        self._portfolio = Portfolio()
        self._current_dt = None
        self.commission = CommissionModel(0, 0, 0)
        self.slippage = SlippageModel(0, 0)
        self.short_fee = CommissionModel(0, 0, 0)
        self.div_tax = CommissionModel(0, 0, 0)
        self.long_only = False
        self.order_multiplier = 1
        self.borrow_money_interest_rate = 0.06
        self.borrow_stock_interest_rate = 0.093

    @property
    def positions(self):
        return self._portfolio.positions

    @property
    def portfolio(self):
        return self._portfolio

    def set_datetime(self, dt: pd.Timestamp) -> None:
        self._current_dt = dt
        self._portfolio.set_datetime(dt)

    def set_commission(self, percentage: float, per_share: float, minimum: float):
        """
        <WORK IN BACKTEST ONLY>
        commission is sum of following:
        :param percentage: percentage part, calculated by percentage * price * shares
                           us: 0, china: 0.0005
        :param per_share: calculated by per_share * shares
                          us: 0.005, china: 0.0006
        :param minimum: minimum commission if above does not exceed
                        us: 1, china: 5
        """
        self.commission = CommissionModel(percentage, per_share, minimum)

    def set_slippage(self, max_percentage: float, max_volume_ratio: float):
        """
        <WORK IN BACKTEST ONLY>
        market impact add to price, calc by:
        slippage = price * max_percentage * sigmod()
        sigmod returns 1.0 when amount_to_volume_ratio hits max_volume_ratio
        """
        self.slippage = SlippageModel(max_percentage, max_volume_ratio)

    def set_short_fee(self, percentage: float):
        """
        <WORK IN BACKTEST ONLY>
        fee pay for short
        :param percentage: percentage * close_price * shares,  us: 0, china: 0.001
        """
        self.short_fee = CommissionModel(percentage, 0, 0)

    def get_price(self, asset: Union[str, Iterable]):
        raise NotImplementedError("abstractmethod")

    def _order(self, asset, amount, price=None):
        raise NotImplementedError("abstractmethod")

    def order(self, asset: str, amount: int, price: str = None):
        if abs(amount) > self.max_shares:
            raise OverflowError(
                'Cannot order more than ±{} shares: {}, set `blotter.max_shares` value '
                'to change this limit.'.format(self.max_shares, amount))

        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")

        opened = self._portfolio.shares(asset)
        if self.long_only and (amount + opened) < 0:
            raise ValueError("Long only blotter, order amount {}, opened {}.".format(
                amount, opened))
        if (amount % self.order_multiplier) != 0:
            raise ValueError("Order amount must be placed in multiples of {}".format(
                self.order_multiplier))
        return self._order(asset, amount, price)

    def _order_target(self, asset: str, target: Union[int, float]):
        opened = self._portfolio.shares(asset)
        amount = target - opened

        if abs(amount) > self.max_shares:
            raise OverflowError(
                'Cannot order more than ±{} shares: {}, set `blotter.max_shares` value'
                'to change this limit.'.format(self.max_shares, amount))

        return self._order(asset, amount)

    def order_target(self, asset: str, target: Union[int, float]):
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")
        if not isinstance(target, (int, float)):
            raise ValueError("`target` must be int or float")
        if self.long_only and target < 0:
            raise ValueError("Long only blotter, `target` must greater than 0.")

        return self._order_target(asset, target)

    def batch_order_target(self, assets: Iterable[str], targets: Iterable[float]):
        skipped = []
        if None in assets or np.nan in assets:
            raise ValueError('None/NaN in `assets: ' + str(assets))
        if None in targets or np.nan in targets:
            raise ValueError('None/NaN in `targets: ' + str(targets))
        assets = list(assets)  # copy for preventing del items in loop
        for asset, target in zip(assets, targets):
            if not self.order_target(asset, target):
                skipped.append([asset, self._portfolio.shares(asset)])
        return skipped

    def order_target_percent(self, asset: str, pct: float):
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")
        if not isinstance(pct, float):
            raise ValueError("`pct` must be float")
        if self.long_only and pct < 0:
            raise ValueError("Long only blotter, `pct` must greater than 0.")

        price = self.get_price(asset)
        if price is None:
            return False
        target = (self._portfolio.value * pct) / price
        target = int(round(target / self.order_multiplier)) * self.order_multiplier
        return self._order_target(asset, target)

    def batch_order_target_percent(self, assets: Iterable[str], weights: Iterable[float]):
        pf_value = self._portfolio.value
        prices = self.get_price(assets)
        skipped = []
        if None in assets or np.any([not (a == a) for a in assets]):
            raise ValueError('None/NaN in `assets: ' + str(assets))
        if None in weights or np.any([not (w == w) for w in weights]):
            raise ValueError('None/NaN in `weights: ' + str(weights))
        assets = list(assets)  # copy for preventing del items in loop
        for asset, pct in zip(assets, weights):
            try:
                price = prices[asset]
                if price != price:
                    raise KeyError("")
                if price == 0:
                    raise RuntimeError('{} price on {} is zero, you have a data error'.format(
                        asset, self._current_dt
                    ))
                target = (pf_value * pct) / price
                # todo 碎股情况，持有160股并不能再买40股，影响不大但最好弄掉, 不过现在交易股数方式就已经不一样了，所以也无所谓？
                target = int(round(target / self.order_multiplier)) * self.order_multiplier
            except KeyError:
                skipped.append([asset, self._portfolio.shares(asset)])
                continue
            if not self.order_target(asset, target):
                skipped.append([asset, self._portfolio.shares(asset)])
        return skipped

    def cancel_all_orders(self):
        raise NotImplementedError("abstractmethod")

    def get_historical_positions(self):
        return self._portfolio.history

    def get_returns(self):
        return self._portfolio.returns

    def get_transactions(self):
        raise NotImplementedError("abstractmethod")


class SimulationBlotter(BaseBlotter, EventReceiver):
    Order = namedtuple("Order", ['date', 'asset', 'amount', 'price',
                                 'fill_price', 'commission', 'realized'])

    def __init__(self, dataloader, capital_base=100000, daily_curb=None, start=None,
                 ohlcv=None):
        """
        :param dataloader: dataloader for get prices
        :param daily_curb: How many fluctuations to prohibit trading, in return.
        :param ohlcv: If None, blotter uses dataloader.ohlcv.
        """
        super().__init__()
        self.market_opened = False
        self.dataloader = dataloader
        self.orders = defaultdict(list)
        self.capital_base = capital_base
        self._portfolio.update_cash(capital_base, is_funds=True)

        if daily_curb is None:
            self.daily_curb = None
        elif type(daily_curb) is float:
            self.daily_curb = DailyCurbModel(daily_curb)
        else:
            self.daily_curb = daily_curb

        df = dataloader.load(start, None, 0).copy()
        # add previous day close price for daily curb
        ohlcv = ohlcv or dataloader.ohlcv
        if dataloader.adjustments is not None:
            sel_cols = [ohlcv[3], dataloader.adjustments[0], dataloader.adjustments[1]]
            lasts = df[sel_cols].groupby(level=1, group_keys=False, observed=False).apply(
                lambda x: x.ffill().shift(1))
            df['__last_close'] = lasts[sel_cols[0]]
            df['__last_div'] = lasts[sel_cols[1]]
            df['__last_sp'] = lasts[sel_cols[2]]
            curb_cols = ['__last_close', '__last_div', '__last_sp']
        else:
            df['__last_close'] = df[ohlcv[3]].groupby(level=1, group_keys=False, observed=False
                                                      ).apply(lambda x: x.ffill().shift(1))
            curb_cols = ['__last_close']
        self._data = df
        self._prices = DataLoaderFastGetter(df[list(ohlcv) + curb_cols])
        self._current_prices_col = None
        self._current_prices = None
        if dataloader.adjustments is not None:
            div_col = dataloader.adjustments[0]
            sp_col = dataloader.adjustments[1]
            adj = df[[div_col, sp_col, ohlcv[3]]]
            adj = adj[(adj[div_col] != 0) | (adj[sp_col] != 1)]
            self._adjustments = DataLoaderFastGetter(adj)
        else:
            self._adjustments = None

    def __repr__(self):
        return '<Transactions>' + str(self.get_transactions())[14:] + \
               '\n' + str(self.portfolio)

    def clear(self):
        self.orders = defaultdict(list)
        self._portfolio.clear()
        self._portfolio.update_cash(self.capital_base, is_funds=True)

    def set_datetime(self, dt: pd.Timestamp) -> None:
        self._current_prices_col = None
        self._current_prices = None
        if self._current_dt is not None:
            # make up events for skipped days for update splits and divs.
            current_date = self._current_dt.normalize()
            target_date = dt.normalize()
            if current_date != target_date:
                for date in pd.bdate_range(current_date, target_date):
                    if date == current_date or date == target_date:
                        continue

                    stop = date + pd.DateOffset(days=1, seconds=-1)
                    table = self._prices.get_as_dict(date, stop)
                    bar_times = table.get_datetime_index()
                    if len(bar_times) == 0:
                        continue

                    open_time = bar_times[0]
                    super().set_datetime(open_time)
                    self.set_price('open')
                    self.market_open(self)

                    close_time = bar_times[-1]
                    super().set_datetime(close_time)
                    self.set_price('close')
                    self.update_portfolio_value()
                    self.market_close(self)
        # get last datetime before portfolio datetime change
        last_dt = self._portfolio.current_dt
        if last_dt:
            self._portfolio.process_borrow_interest(
                (dt - last_dt).days,
                self.borrow_money_interest_rate,
                self.borrow_stock_interest_rate)
        super().set_datetime(dt)

    def set_price(self, name: str):
        if name == 'open':
            self._current_prices_col = 0  # ohlcv[0] = o
        else:
            self._current_prices_col = 3  # ohlcv[3] = c
        self._current_prices = None

    def _get_current_prices(self):
        if self._current_prices_col is None:
            raise ValueError("_current_prices is None, maybe set_price was not called.")

        if self._current_prices is None:
            self._current_prices = self._prices.get_as_dict(
                self._current_dt, column_id=self._current_prices_col)

        return self._current_prices

    def get_price(self, asset: Union[str, Iterable, None]):
        if not self.market_opened:
            raise RuntimeError('Out of market hours. Maybe you rebalance at AfterMarketClose; '
                               'or BeforeMarketOpen; or EveryBarData on daily data; '
                               'or you did not subscribe this class with SimulationEventManager')

        curr_prices = self._get_current_prices()
        if not isinstance(asset, str):
            return curr_prices

        try:
            price = curr_prices[asset]
            return price
        except KeyError:
            return None

    def _order(self, asset, amount, price=None):
        if not self.market_opened:
            raise RuntimeError('Out of market hours. Maybe you rebalance at AfterMarketClose; '
                               'or BeforeMarketOpen; or EveryBarData on daily data; '
                               'or you did not subscribe this class with SimulationEventManager')
        if amount == 0:
            return True

        # get price and change
        price = self.get_price(asset)
        if price is None:
            return False

        # trading curb for daily return
        if self.daily_curb is not None:
            curr_prices = self._get_current_prices()
            curr_high = self._prices.get_as_dict(curr_prices.row_slice, column_id=1)
            curr_low = self._prices.get_as_dict(curr_prices.row_slice, column_id=2)
            if self._adjustments is None:
                last_close = self._prices.get_as_dict(curr_prices.row_slice, column_id=-1)
                last_div = defaultdict(float)
                last_sp = defaultdict(lambda: 1.0)
            else:
                last_close = self._prices.get_as_dict(curr_prices.row_slice, column_id=-3)
                last_div = self._prices.get_as_dict(curr_prices.row_slice, column_id=-2)
                last_sp = self._prices.get_as_dict(curr_prices.row_slice, column_id=-1)
            # Detecting whether transactions are possible
            price, amount = self.daily_curb.calculate(
                asset, price, amount, curr_high[asset], curr_low[asset],
                last_close[asset], last_div[asset], last_sp[asset],)
            if amount == 0:
                return False

        # commission, slippage
        if not self.slippage.is_empty():
            bar_volume = self._prices.get_as_dict(self._current_dt, column_id=4)  # ohlcv=01234
            price = self.slippage.calculate(asset, price, amount/bar_volume[asset])
        commission = self.commission.calculate(asset, price, amount)
        if amount < 0:
            commission += self.short_fee.calculate(asset, price, amount)
            fill_price = price
        else:
            fill_price = price
        commission = round(commission, 2)

        # update portfolio, pay cash
        realized = self._portfolio.update(asset, amount, fill_price, commission)
        self._portfolio.update_cash(-amount * fill_price - commission)

        # make order
        order = SimulationBlotter.Order(
            self._current_dt, asset, amount, price, fill_price, commission, realized)
        self.orders[asset].append(order)
        return True

    def cancel_all_orders(self):
        # don't need
        pass

    def get_transactions(self):
        data = []
        for asset, orders in self.orders.items():
            for o in orders:
                data.append(dict(index=o.date, amount=o.amount, price=o.price, symbol=o.asset,
                                 fill_price=o.fill_price, commission=o.commission,
                                 realized=o.realized))
        ret = pd.DataFrame(data, columns=['index', 'symbol', 'amount', 'price',
                                          'fill_price', 'commission', 'realized'])
        ret = ret.set_index('index').sort_index()
        return ret

    def update_portfolio_value(self):
        if len(self._portfolio.positions) > 0:
            self._portfolio.update_value(self._get_current_prices().get)

    def market_open(self, _):
        self.market_opened = True

    def market_close(self, _):
        self.cancel_all_orders()
        self.market_opened = False

        # push dividend/split data to portfolio
        if self._adjustments is not None:
            try:
                div_col, sp_col, close_col = 0, 1, 2
                start = self._current_dt.normalize()
                stop = start + pd.DateOffset(days=1, seconds=-1)
                current_adj = self._adjustments.get_as_dict(start, stop)

                for asset, row in current_adj.items():
                    div = row[div_col] - self.div_tax.calculate(asset, row[div_col], 1)
                    tax = row[div_col] - div
                    self._portfolio.process_dividend(asset, div, tax=tax)
                    self._portfolio.process_split(asset, 1/row[sp_col], row[close_col])
            except KeyError:
                pass

    def new_bars_data(self, _):
        # update value if received intraday bars
        if self.market_opened:
            self.update_portfolio_value()

    def on_run(self):
        self.schedule(MarketOpen(self.market_open, -1))  # -1 for grab priority
        self.schedule(MarketClose(self.market_close))
        self.schedule(EveryBarData(self.new_bars_data))


class ManualBlotter(BaseBlotter):
    """
    This blotter will not actually place orders, but export orders to csv file.
    Not support intraday trading.

    Order columns:
    date:
        order created datetime
    status:
        PendingSubmit - placed, but has not been submitted yet, you need to manually send all orders
                        in this state to the broker.
        Submitted - When you submitted this order to broker, you can set to this status (skip able).
        Cancelled - When none filled of this order and cancelled, please set to this status.
        Filled - When your order has been completely/partially filled, please set to this status.
        Cash - Cash transfer log
    symbol:
        underlay asset
    target_percent:
        Order size, target percent unit. If zero, means close position.
    action_value:
        How many value need to be fill. Calculated from last daily portfolio value.
    amount:
        Calculated based on yesterday's closing price.
    limit_price:
        float: price.
        string: 'Market'
    filled_amount:
        If order filled, indicate the number of asset traded.
    filled_price:
        Average price of transactions.
    filled_percent:
        Percent of the target has been filled.
    commission:
        Commission
    realized:
        Realized gain/loss.
    """

    def __init__(self, working_dir, time_zone, loader=None):
        super().__init__()
        self.working_dir = working_dir
        self.time_zone = time_zone
        self.orders = None
        self.order_multiplier = 100
        self.last_price = None
        self.loader = loader
        self.load()

    def _rebuild_from_orders(self):
        self._portfolio.clear()
        if self.loader is not None:
            prices_df = self.loader.load()
        else:
            prices_df = None

        for date, df in self.orders.groupby(pd.Grouper(key='date', freq='D')):
            # skip holiday
            if prices_df is not None:
                if date.normalize() not in prices_df.index.levels[0] and df.empty:
                    continue

            self._portfolio.set_datetime(date.normalize())

            # update by open prices / for stop model recode high/low prices
            if prices_df is not None:
                try:
                    last_price_df = prices_df.xs(date.normalize())
                    price_dict = last_price_df.open.to_dict()
                    self.update_portfolio_value(price_dict)
                except KeyError:
                    pass

            for i, row in df.iterrows():
                if row.status == 'Cash':
                    self._portfolio.update_cash(row.filled_amount, is_funds=True)
                elif row.status == 'Filled':
                    self._portfolio.update(row.symbol, row.filled_amount, row.filled_price,
                                           row.commission)
                    self._portfolio.update_cash(-row.filled_amount * row.filled_price -
                                                row.commission)

            # update by close prices
            if prices_df is not None:
                try:
                    last_price_df = prices_df.xs(date.normalize())
                    price_dict = last_price_df.close.to_dict()
                    self.update_portfolio_value(price_dict)
                except KeyError:
                    pass

            # update div and split after close, otherwise last price won't be adjusted
            for i, row in df.iterrows():
                if row.status == 'Dividend':
                    self._portfolio.process_dividend(row.symbol, row.filled_amount,
                                                     tax=row.commission)
                elif row.status == 'Split':
                    self._portfolio.process_split(row.symbol, row.filled_amount,
                                                  row.filled_price)

    def load(self):
        """ Reload portfolio/orders from working_dir, for updating account and order status """
        pattern = os.path.join(self.working_dir, 'orders_*.csv')
        files = glob.glob(pattern)

        col_types = {
            'id': np.int64, 'target_percent': np.float64, 'action_value': np.float64,
            'amount': np.float64, 'limit_price': str,
            'filled_amount': np.float64, 'filled_price': np.float64, 'filled_percent': np.float64,
            'commission': np.float64, 'realized': np.float64
        }
        if len(files) == 0:
            self.orders = pd.DataFrame(columns=[
                'id', 'date', 'status', 'symbol', 'target_percent', 'action_value', 'amount',
                'limit_price', 'filled_amount', 'filled_price', 'filled_percent', 'commission',
                'realized'])
            print('Create ManualBlotter with empty orders.')
            for k, v in col_types.items():
                self.orders[k] = self.orders[k].astype(v)
            self.orders.set_index('id', inplace=True)
        else:
            self.orders = pd.concat([
                pd.read_csv(fn, index_col='id', dtype=col_types, parse_dates=False)
                for fn in files
            ])
            self.orders.date = pd.to_datetime(self.orders.date, utc=True)
            print('Orders loaded: {} - {}, total: {} orders'.format(
                self.orders.date.iloc[0], self.orders.date.iloc[-1], len(self.orders)))
            self.orders['date'] = self.orders['date'].dt.tz_convert(self.time_zone)
            self._rebuild_from_orders()

    def save(self):
        """
        Save orders and changes to working_dir.
        """
        self.orders.index.name = 'id'
        for n, g in self.orders.groupby(pd.Grouper(key='date', freq='D')):
            name = n.strftime('orders_%Y-%m-%d.csv')
            if not g.empty:
                g.round(4).to_csv(os.path.join(self.working_dir, name))

    def set_datetime(self, dt: pd.Timestamp) -> None:
        assert str(dt.tz) == self.time_zone
        super().set_datetime(dt)

    def transfer_funds(self, amount):
        """ Call this after you transfer funds to broker """
        assert self._current_dt is not None
        order = pd.Series(dict(
            date=self._current_dt, status='Cash', symbol='cash', target_percent=0.,
            action_value=0., amount=amount, limit_price='/',
            filled_amount=amount, filled_price=0., filled_percent=0.,
            commission=0., realized=0.))
        if len(self.orders.index) == 0:
            order.name = 1
        else:
            order.name = max(self.orders.index) + 1
        if len(self.orders) == 0:
            self.orders = pd.DataFrame([order])
        else:
            self.orders = pd.concat([self.orders, pd.DataFrame([order])])
        self._portfolio.update_cash(amount, is_funds=True)

    def _order(self, asset, amount, price=None):
        raise NotImplementedError("not support")

    def order(self, asset: str, amount: int, price: str = None):
        raise NotImplementedError("not support")

    def order_target(self, asset: str, target: Union[int, float]):
        raise NotImplementedError("not support")

    def batch_order_target(self, assets: Iterable[str], targets: Iterable[float]):
        raise NotImplementedError("not support")

    def set_last_price(self, price_dict):
        self.last_price = price_dict

    def order_target_percent(self, asset: str, pct: float):
        """
        Call by trading algorithm, only generate orders csv, you need to manually place real order
        to your broker, and call `order_filled` after broker tell you order filled.
        """
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")
        if not isinstance(pct, float):
            raise ValueError("`pct` must be float")
        if self.long_only and pct < 0:
            raise ValueError("Long only blotter, `pct` must greater than 0.")
        assert self._current_dt is not None

        pct = round(pct, 4)
        if pct == 0. and asset not in self.positions:
            return None

        opened_value = 0
        opened_shares = 0
        try:
            opened_value = self._portfolio.positions[asset].value
            opened_shares = self._portfolio.positions[asset].shares
        except KeyError:
            pass
        target_value = self._portfolio.value * pct
        action_value = target_value - opened_value

        if self.last_price is None:
            raise ValueError('call blotter.set_last_price(dict) first.')
        last_close_price = self.last_price[asset]
        target_amount = target_value / last_close_price

        multiplier = self.order_multiplier
        if target_amount == 0:
            # if close a position, allow odd lots
            action_amount = -opened_shares
        else:
            action_amount = int(round((target_amount - opened_shares) / multiplier)) * multiplier

        order = pd.Series(dict(
            date=self._current_dt, status='PendingSubmit',
            symbol=asset, target_percent=pct, action_value=action_value, amount=action_amount,
            limit_price='Market', filled_amount=0, filled_price=0, filled_percent=0, commission=0,
            realized=0))
        order.name = max(self.orders.index) + 1
        # order = order.to_frame().T.infer_objects().rename_axis(
        #       self.orders.index.names, copy=False)
        self.orders = pd.concat([self.orders, pd.DataFrame([order])])

        return self.orders.index[-1]

    def batch_order_target_percent(self, assets: Iterable[str], weights: Iterable[float]):
        """
        Call by trading algorithm, only generate orders csv, you need to manually place real order
        to your broker, and call `order_filled` after broker tell you order filled.
        """
        if None in assets or np.any([not (a == a) for a in assets]):
            raise ValueError('None/NaN in `assets: ' + str(assets))
        if None in weights or np.any([not (w == w) for w in weights]):
            raise ValueError('None/NaN in `weights: ' + str(weights))
        ret = dict()
        for asset, pct in zip(assets, weights):
            oid = self.order_target_percent(asset, pct)
            ret[asset] = oid
        return ret

    def order_filled(self, oid, filled_amount, filled_price, commission):
        """ Call this after you order filled by broker """
        assert self._current_dt is not None

        order = self.orders.loc[oid].copy()

        realized = self._portfolio.update(order.symbol, filled_amount, filled_price, commission)
        self._portfolio.update_cash(-filled_amount * filled_price - commission)

        if order.status == 'Filled':
            total_turnover = (abs(order.filled_amount * order.filled_price) +
                              abs(filled_amount * filled_price))

            filled_amount += order.filled_amount
            filled_price = total_turnover / abs(filled_amount)
            commission += order.commission
            realized += order.realized

        order.status = 'Filled'
        order.filled_amount = filled_amount
        order.filled_price = filled_price
        order.filled_percent = round(filled_price * filled_amount / order.action_value, 3)
        order.realized = realized
        order.commission = commission
        self.orders.loc[oid] = order

    def order_cancelled(self, oid):
        """ Call this after you cancelled one order """
        self.orders.loc[oid, 'status'] = 'Cancelled'

    def position_dividend(self, asset, amount, tax, time_delta):
        """ Call this if your position has dividends """
        assert self._current_dt is not None

        order = pd.Series(dict(
            date=self._current_dt + time_delta, status='Dividend',
            symbol=asset, target_percent=0, action_value=0, amount=amount, limit_price='/',
            filled_amount=amount, filled_price=0, filled_percent=0, commission=tax, realized=0))
        order.name = max(self.orders.index) + 1
        self.orders = pd.concat([self.orders, pd.DataFrame([order])])
        self._portfolio.process_dividend(asset, amount, tax)

    def position_split(self, asset, inverse_ratio: float, last_price, time_delta):
        """ Call this if your position has splits """
        assert self._current_dt is not None
        if last_price is None:
            last_price = self.positions[asset].last_price
        assert last_price is not None

        order = pd.Series(dict(
            date=self._current_dt + time_delta, status='Split',
            symbol=asset, target_percent=0, action_value=0, amount=inverse_ratio, limit_price='/',
            filled_amount=inverse_ratio, filled_price=last_price, filled_percent=0,
            commission=0, realized=0))
        order.name = max(self.orders.index) + 1
        self.orders = pd.concat([self.orders, pd.DataFrame([order])])
        self._portfolio.process_split(asset, inverse_ratio, last_price)

    def update_portfolio_value(self, prices):
        """ Call this nightly """
        if len(self._portfolio.positions) > 0:
            self._portfolio.update_value(prices)

    def get_price(self, asset: Union[str, Iterable]):
        raise NotImplementedError("not supported")

    def get_transactions(self):
        orders = self.orders[self.orders.status == 'Filled']
        ret = pd.DataFrame(columns=['index', 'symbol', 'amount', 'price',
                                    'fill_price', 'commission', 'realized'])
        ret['index'] = orders.date
        ret.symbol = orders.symbol
        ret.amount = orders.filled_amount
        ret.price = orders.filled_price
        ret.fill_price = orders.filled_price + orders.commission / orders.filled_amount
        ret.commission = orders.commission
        ret.realized = orders.realized
        ret = ret.set_index('index').sort_index()
        return ret

    @property
    def pendings(self):
        return self.orders[self.orders.status == 'PendingSubmit']

    def cancel_all_orders(self):
        raise NotImplementedError("not supported")
