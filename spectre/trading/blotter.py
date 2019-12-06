"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import pandas as pd
from collections import namedtuple, defaultdict
from .event import *


class Portfolio:
    def __init__(self):
        self.historical = pd.DataFrame()
        self.positions = defaultdict(int)
        self.cash = 0
        self.current_date = None

    def __repr__(self):
        return "<Portfolio>:\n" + str(self.get_history_positions())

    def clear(self):
        self.__init__()

    def set_date(self, date):
        if isinstance(date, str):
            date = pd.Timestamp(date)
        if self.current_date is not None and date < self.current_date:
            raise ValueError('Cannot set a date less than the current date')

        self.current_date = date

    def get_history_positions(self):
        return self.historical.fillna(method='ffill')

    def update(self, asset, amount):
        self.positions[asset] += amount
        self.historical.loc[self.current_date, asset] = self.positions[asset]

    def update_cash(self, amount):
        self.cash += amount
        self.positions['cash'] = self.cash
        self.historical.loc[self.current_date, 'cash'] = self.cash

    def process_split(self, asset, ratio):
        sp = self.positions[asset] * (ratio - 1)
        self.update(asset, sp)

    def process_dividends(self, asset, ratio):
        div = self.positions[asset] * ratio
        self.update_cash(div)

    def value(self, get_curr_price):
        value = sum([get_curr_price(asset) * amount
                     for asset, amount in self.positions.items()
                     if asset != 'cash'])
        return self.cash + value

    def leverage(self, get_curr_price):
        gross_exposure = [get_curr_price(asset) * abs(amount)
                          for asset, amount in self.positions.items()
                          if asset != 'cash']
        return sum(gross_exposure) / self.value(get_curr_price)


class BaseBlotter(EventReceiver):
    max_shares = int(1e+5)
    """
    Base class for Order Management System.
    """

    def __init__(self) -> None:
        super().__init__()
        self.portfolio = Portfolio()

    def set_commission(self, percentage, per_share, minimum):
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
        pass

    def set_slippage(self, percentage, minimum):
        """
        <WORK IN BACKTEST ONLY>
        market impact add to price, sum of following:
        :param percentage: percentage * price * shares
        :param minimum: minimum slippage
        """
        pass

    def set_close_fee(self, percentage):
        """
        <WORK IN BACKTEST ONLY>
        fee pay for close a position
        :param percentage: percentage * close_price * shares,  us: 0, china: 0.001
        """
        pass

    def get_price(self, asset):
        raise NotImplementedError("abstractmethod")

    def _order(self, asset, amount):
        raise NotImplementedError("abstractmethod")

    def order(self, asset, amount):
        if abs(amount) > self.max_shares:
            raise OverflowError('Cannot order more than ±%d shares'.format(self.max_shares))

        return self._order(asset, amount)

    def order_target_percent(self, asset, pct):
        amount = self.portfolio.value(self.get_price) * pct / self.get_price(asset)
        return self._order(asset, amount)

    def cancel_all_orders(self):
        raise NotImplementedError("abstractmethod")

    def process_split(self, asset, ratio):
        self.portfolio.process_split(asset, ratio)

    def process_dividends(self, asset, ratio):
        self.portfolio.process_dividends(asset, ratio)

    def get_transactions(self):
        raise NotImplementedError("abstractmethod")

    def get_portfolio(self):
        return self.portfolio.positions

    def get_positions(self):
        return self.portfolio.historical


class SimulationBlotter(BaseBlotter):
    Order = namedtuple("Order", ['date', 'asset', 'amount', 'price'])

    def __init__(self, dataloader):
        super().__init__()
        self.current_dt = None
        self.price_col = None
        self.market_opened = False
        self.dataloader = dataloader
        self.orders = defaultdict(list)

    def set_datetime(self, dt):
        self.current_dt = dt

    def set_price(self, name):
        if name == 'open':
            self.price_col = self.dataloader.get_ohlcv_names()[0]
        else:
            self.price_col = self.dataloader.get_ohlcv_names()[3]

    def get_price(self, asset):
        df = self.dataloader.load(self.current_dt, self.current_dt, 0)
        price = df.loc[(slice(None), asset), self.price_col]
        return price

    def _order(self, asset, amount):
        if not self.market_opened:
            raise RuntimeError('Out of market hours.')

        # 如果买入，要算佣金
        # 如果卖出要算税
        order = SimulationBlotter.Order(self.today, asset, amount, last_price)
        pos = SimulationBlotter.Position(self.today, asset, remined)
        self.orders[asset].append(order)
        self.positions[asset].append(pos)

    def cancel_all_orders(self):
        # don't need
        pass

    def market_open(self):
        self.market_opened = True

    def market_close(self):
        self.cancel_all_orders()
        self.market_opened = False

    def on_subscribe(self):
        self.schedule(MarketOpen(self.market_open, -100000))  # 100ms ahead for system preparation
        self.schedule(MarketClose(self.market_close))
