"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Callable
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict, Iterable
from .event import *


class Portfolio:
    def __init__(self):
        self._history = []
        self._positions = dict()
        self._last_price = defaultdict(lambda: np.nan)
        self._cash = 0
        self._current_date = None

    @property
    def history(self):
        ret = pd.DataFrame(self._history + [self._get_today_record()])
        ret.columns = pd.MultiIndex.from_tuples(ret.columns)
        ret = ret.set_index('index').sort_index(axis=0).sort_index(axis=1)
        return ret

    @property
    def returns(self):
        return self.history.value.sum(axis=1).pct_change()

    @property
    def positions(self):
        return self._positions

    @property
    def cash(self):
        return self._cash

    @property
    def value(self):
        # for asset, shares in self.positions.items():
        #     if self._last_price[asset] != self._last_price[asset]:
        #         raise ValueError('{}({}) is nan in {}'.format(asset, shares, self._current_date))
        values = [self._last_price[asset] * shares
                  for asset, shares in self.positions.items() if shares != 0]
        return sum(values) + self._cash

    @property
    def leverage(self):
        values = [self._last_price[asset] * shares
                  for asset, shares in self.positions.items() if shares != 0]
        return sum(np.abs(values)) / (sum(values) + self._cash)

    def __repr__(self):
        return "<Portfolio>" + str(self.history)[11:]

    def clear(self):
        self.__init__()

    def shares(self, asset):
        try:
            return self._positions[asset]
        except KeyError:
            return 0

    def _get_today_record(self):
        record = {('index', ''): self._current_date, ('value', 'cash'): self._cash}
        for asset, shares in self._positions.items():
            record[('shares', asset)] = shares
            record[('value', asset)] = shares * self._last_price[asset]
        return record

    def set_date(self, date):
        if isinstance(date, str):
            date = pd.Timestamp(date)
        date = date.normalize()
        if self._current_date is not None:
            if date < self._current_date:
                raise ValueError('Cannot set a date less than the current date')
            elif date > self._current_date:
                # today add to history
                self._history.append(self._get_today_record())

        self._current_date = date

    def update(self, asset, amount, fill_price):
        assert self._current_date is not None
        self._positions[asset] = self.shares(asset) + amount
        if fill_price is not None:  # update last price for correct portfolio value
            self._last_price[asset] = fill_price
        if self._positions[asset] == 0:
            del self._positions[asset]

    def update_cash(self, amount):
        self._cash += amount

    def process_split(self, asset, inverse_ratio: float, last_price):
        if inverse_ratio != inverse_ratio or inverse_ratio == 1 or asset not in self._positions:
            return
        sp = self._positions[asset] * inverse_ratio
        if inverse_ratio < 1:  # reverse split remaining to cash
            remaining = int(self._positions[asset] - int(sp) / inverse_ratio)  # for more precise
            if remaining != 0:
                self.update_cash(remaining * last_price)
        change = int(sp) - self._positions[asset]
        self.update(asset, change, None)

    def process_dividends(self, asset, amount):
        if amount != amount or amount == 0 or asset not in self._positions:
            return
        div = self._positions[asset] * amount
        self.update_cash(div)

    def _update_value_func(self, func):
        for asset, shares in self._positions.items():
            price = func(asset)
            if price and price == price:
                self._last_price[asset] = price

    def _update_value_dict(self, prices):
        for asset, shares in self._positions.items():
            price = prices.get(asset, np.nan)
            if price == price:
                self._last_price[asset] = price

    def update_value(self, prices: Union[Callable, dict]):
        if callable(prices):
            self._update_value_func(prices)
        elif isinstance(prices, dict):
            self._update_value_dict(prices)
        else:
            raise ValueError('prices ether callable or dict')


class CommissionModel:
    def __init__(self, percentage: float, per_share: float, minimum: float):
        self.percentage = percentage
        self.per_share = per_share
        self.minimum = minimum

    def calculate(self, price: float, shares: int):
        commission = price * abs(shares) * self.percentage
        commission += abs(shares) * self.per_share
        if commission < self.minimum:
            commission = self.minimum
        return commission


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
        self.slippage = CommissionModel(0, 0, 0)
        self.short_fee = CommissionModel(0, 0, 0)
        self.long_only = False

    @property
    def positions(self):
        return self._portfolio.positions

    @property
    def portfolio(self):
        return self._portfolio

    def set_datetime(self, dt: pd.Timestamp) -> None:
        self._current_dt = dt
        self._portfolio.set_date(dt)

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

    def set_slippage(self, percentage: float, per_share: float):
        """
        <WORK IN BACKTEST ONLY>
        market impact add to price, sum of following:
        :param percentage: percentage * price * shares
        :param per_share: per_share * shares
        """
        self.slippage = CommissionModel(percentage, per_share, 0)

    def set_short_fee(self, percentage: float):
        """
        <WORK IN BACKTEST ONLY>
        fee pay for short
        :param percentage: percentage * close_price * shares,  us: 0, china: 0.001
        """
        self.short_fee = CommissionModel(percentage, 0, 0)

    def get_price(self, asset: Union[str, Iterable]):
        raise NotImplementedError("abstractmethod")

    def _order(self, asset, amount):
        raise NotImplementedError("abstractmethod")

    def order(self, asset: str, amount: int):
        if abs(amount) > self.max_shares:
            raise OverflowError('Cannot order more than ±{} shares: {}'.format(
                self.max_shares, amount))
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")

        return self._order(asset, amount)

    def batch_order_target_percent(self, assets: Iterable, weights: Iterable):
        pf_value = self._portfolio.value
        prices = self.get_price(assets)
        for asset, pct in zip(assets, weights):
            if self.long_only and pct < 0:
                raise ValueError("Long only blotter, `pct` must greater than 0.")
            try:
                amount = (pf_value * pct) / prices[asset]
            except KeyError:
                continue
            opened = self._portfolio.shares(asset)
            amount = int(amount - opened)
            if amount != 0:
                self._order(asset, amount)

    def order_target_percent(self, asset: Union[str, Iterable], pct: Union[float, Iterable]):
        if isinstance(asset, Iterable) and isinstance(pct, Iterable):
            return self.batch_order_target_percent(asset, pct)
        elif not isinstance(asset, str):
            raise KeyError("`asset` must be a string")
        if self.long_only and pct < 0:
            raise ValueError("Long only blotter, `pct` must greater than 0.")

        price = self.get_price(asset)
        if price is None:
            return
        amount = (self._portfolio.value * pct) / price
        opened = self._portfolio.shares(asset)
        amount = int(amount - opened)
        return self._order(asset, amount)

    def cancel_all_orders(self):
        raise NotImplementedError("abstractmethod")

    def get_history_positions(self):
        return self._portfolio.history

    def get_returns(self):
        return self._portfolio.returns

    def get_transactions(self):
        raise NotImplementedError("abstractmethod")


class SimulationBlotter(BaseBlotter, EventReceiver):
    Order = namedtuple("Order", ['date', 'asset', 'amount', 'price',
                                 'fill_price', 'commission'])

    def __init__(self, dataloader, capital_base=100000, daily_curb=None):
        """
        :param dataloader: dataloader for get prices
        :param daily_curb: How many fluctuations to prohibit trading, in return.
        """
        super().__init__()
        self.market_opened = False
        self.dataloader = dataloader
        self.daily_curb = daily_curb
        self.orders = defaultdict(list)
        self.capital_base = capital_base
        self._portfolio.update_cash(capital_base)

        df = dataloader.load(None, None, 0)
        self._data = df
        self._prices = df[list(dataloader.ohlcv)]
        self._current_row = None
        self._current_prices = None
        if dataloader.adjustments is not None:
            div_col = dataloader.adjustments[0]
            sp_col = dataloader.adjustments[1]
            adj = df[[div_col, sp_col, dataloader.ohlcv[3]]]
            self._adjustments = adj[(adj[div_col] != 0) | (adj[sp_col] != 1)]
        else:
            self._adjustments = None

    def __repr__(self):
        return '<Transactions>' + str(self.get_transactions())[14:] + \
               '\n' + str(self.portfolio)

    def clear(self):
        self.orders = defaultdict(list)
        self._portfolio.clear()
        self._portfolio.update_cash(self.capital_base)

    def _update_time(self) -> None:
        try:
            self._current_row = self._prices.loc[self._current_dt]
        except KeyError:
            self._current_row = None
        self._current_prices = None

    def set_datetime(self, dt: pd.Timestamp) -> None:
        if self._current_dt is not None:
            # make up events for skipped dates
            for day in pd.bdate_range(self._current_dt, dt):
                if day == self._current_dt or day == dt:
                    continue
                super().set_datetime(day)
                self._update_time()
                self.market_open(self)
                if self._current_row is not None:
                    self._current_prices = self._current_row[self.dataloader.ohlcv[3]].to_dict()
                    self.update_portfolio_value()
                self.market_close(self)
        super().set_datetime(dt)
        self._update_time()

    def set_price(self, name: str):
        if name == 'open':
            price_col = self.dataloader.ohlcv[0]
        else:
            price_col = self.dataloader.ohlcv[3]
        self._current_prices = self._current_row[price_col].to_dict()

    def get_price(self, asset: Union[str, Iterable]):
        if self._current_prices is None:
            return None

        if not isinstance(asset, str):
            return self._current_prices

        try:
            price = self._current_prices[asset]
            return price
        except KeyError:
            return None

    def _order(self, asset, amount):
        if not self.market_opened:
            raise RuntimeError('Out of market hours, or you did not subscribe this class '
                               'with SimulationEventManager')
        if amount == 0:
            return

        if abs(amount) > self.max_shares:
            raise OverflowError('Cannot order more than ±{} shares: {}'.format(
                self.max_shares, amount))

        opened = self._portfolio.shares(asset)
        if self.long_only and (amount + opened) < 0:
            raise ValueError("Long only blotter, order amount {}, opened {}.".format(
                amount, opened))

        # get price and change
        price = self.get_price(asset)
        if price is None:
            raise KeyError("`asset` is not tradable today.")

        # trading curb for daily return
        if self.daily_curb is not None:
            df = self._data.loc[(slice(None, self._current_dt), asset), :]
            if df.shape[0] < 2:
                return
            close_col = self.dataloader.ohlcv[3]
            previous_close = df[close_col].iloc[-2]
            change = price / previous_close - 1
            # Detecting whether transactions are possible
            if abs(change) > self.daily_curb:
                return

        # commission, slippage
        commission = self.commission.calculate(price, amount)
        slippage = self.slippage.calculate(price, 1)
        if amount < 0:
            commission += self.short_fee.calculate(price, amount)
            fill_price = price - slippage
        else:
            fill_price = price + slippage

        # make order
        order = SimulationBlotter.Order(
            self._current_dt, asset, amount, price, fill_price, commission)
        self.orders[asset].append(order)

        # update portfolio, pay cash
        self._portfolio.update(asset, amount, fill_price)
        self._portfolio.update_cash(-amount * fill_price - commission)

    def cancel_all_orders(self):
        # don't need
        pass

    def get_transactions(self):
        data = []
        for asset, orders in self.orders.items():
            for o in orders:
                data.append(dict(index=o.date, amount=o.amount, price=o.price, symbol=o.asset,
                                 fill_price=o.fill_price, commission=o.commission))
        ret = pd.DataFrame(data, columns=['index', 'symbol', 'amount', 'price',
                                          'fill_price', 'commission'])
        ret = ret.set_index('index').sort_index()
        return ret

    def update_portfolio_value(self):
        self._portfolio.update_value(self._current_prices)

    def market_open(self, _):
        self.market_opened = True

    def market_close(self, _):
        self.cancel_all_orders()
        self.market_opened = False

        # push dividend/split data to portfolio
        if self._adjustments is not None:
            try:
                current_adj = self._adjustments.loc[self._current_dt]
                div_col = self.dataloader.adjustments[0]
                sp_col = self.dataloader.adjustments[1]
                close_col = self.dataloader.ohlcv[3]

                for asset, div in current_adj[div_col].items():
                    self._portfolio.process_dividends(asset, div)
                for sr_row in current_adj[[sp_col, close_col]].itertuples():
                    self._portfolio.process_split(sr_row[0], 1/sr_row[1], sr_row[2])
            except KeyError:
                pass

    def on_run(self):
        self.schedule(MarketOpen(self.market_open, -1))  # -1 for grab priority
        self.schedule(MarketClose(self.market_close))
