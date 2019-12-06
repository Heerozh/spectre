"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import pandas as pd
from numpy import nan
from collections import namedtuple, defaultdict
from .event import *


class Portfolio:
    def __init__(self):
        self.history = pd.DataFrame()
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
        date = date.normalize()
        if self.current_date is not None and date < self.current_date:
            raise ValueError('Cannot set a date less than the current date')

        self.current_date = date

    def get_history_positions(self):
        return self.history.fillna(method='ffill')

    def update(self, asset, amount):
        self.positions[asset] += amount
        self.history.loc[self.current_date, asset] = self.positions[asset]

    def update_cash(self, amount):
        self.cash += amount
        self.positions['cash'] = self.cash
        self.history.loc[self.current_date, 'cash'] = self.cash

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
        self.portfolio = Portfolio()
        self.current_dt = None
        self.commission = CommissionModel(0, 0, 0)
        self.slippage = CommissionModel(0, 0, 0)
        self.short_fee = CommissionModel(0, 0, 0)

    def set_datetime(self, dt):
        self.current_dt = dt
        self.portfolio.set_date(dt)

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

    def get_price(self, asset):
        raise NotImplementedError("abstractmethod")

    def _order(self, asset, amount):
        raise NotImplementedError("abstractmethod")

    def order(self, asset: str, amount: int):
        if abs(amount) > self.max_shares:
            raise OverflowError('Cannot order more than ±%d shares'.format(self.max_shares))
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")

        return self._order(asset, amount)

    def order_target_percent(self, asset: str, pct: float):
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")

        amount = self.portfolio.value(self.get_price) * pct / self.get_price(asset)
        held = self.portfolio.positions[asset]
        amount -= held
        return self._order(asset, int(amount))

    def cancel_all_orders(self):
        raise NotImplementedError("abstractmethod")

    def process_split(self, asset, ratio):
        self.portfolio.process_split(asset, ratio)

    def process_dividends(self, asset, ratio):
        self.portfolio.process_dividends(asset, ratio)

    def get_portfolio(self):
        return self.portfolio.positions

    def get_positions(self):
        return self.portfolio.get_history_positions()

    def get_transactions(self):
        raise NotImplementedError("abstractmethod")


class SimulationBlotter(BaseBlotter, EventReceiver):
    Order = namedtuple("Order", ['date', 'asset', 'amount', 'price', 'final_price', 'commission'])

    def __init__(self, dataloader, daily_curb=None):
        """
        :param dataloader: dataloader for get prices
        :param daily_curb: How many fluctuations to prohibit trading, in return.
        """
        super().__init__()
        self.price_col = None
        self.market_opened = False
        self.dataloader = dataloader
        self.daily_curb = daily_curb
        self.orders = defaultdict(list)

    def clear(self):
        self.orders = defaultdict(list)
        self.portfolio.clear()

    def set_price(self, name: str):
        if name == 'open':
            self.price_col = self.dataloader.get_ohlcv_names()[0]
        else:
            self.price_col = self.dataloader.get_ohlcv_names()[3]

    def get_price(self, asset: str):
        df = self.dataloader.load(self.current_dt, self.current_dt, 0)
        # todo 几个问题， 1是 get value时，没有的数据要用最后一个
        # 其次这个重复调用的次数太多了，有点浪费，考虑让portfolio自己记录下？
        if asset not in df.index.get_level_values(1):
            return None
        price = df.loc[(slice(None), asset), self.price_col]
        return price.values[-1]

    def _order(self, asset, amount):
        if not self.market_opened:
            raise RuntimeError('Out of market hours, or you did not subscribe this class '
                               'with SimulationEventManager')

        if amount == 0:
            return

        # get price and change
        df = self.dataloader.load(self.current_dt, self.current_dt, 1)
        df = df.loc[(slice(None), asset), :]
        price = df[self.price_col].iloc[-1]
        close_col = self.dataloader.get_ohlcv_names()[3]
        previous_close = df[close_col].iloc[-1]
        change = price / previous_close - 1
        # Detecting whether transactions are possible
        if self.daily_curb is not None and abs(change) > self.daily_curb:
            return

        # commission, slippage
        commission = self.commission.calculate(price, amount)
        if amount < 0:
            commission += self.short_fee.calculate(price, amount)
        slippage = self.slippage.calculate(price, amount)
        final_price = price + slippage

        # make order
        order = SimulationBlotter.Order(
            self.current_dt, asset, amount, price, final_price, commission)
        self.orders[asset].append(order)

        # update portfolio, pay cash
        self.portfolio.update(asset, amount)
        self.portfolio.update_cash(-amount * final_price + commission)

    def cancel_all_orders(self):
        # don't need
        pass

    def get_transactions(self):
        pass

    def market_open(self):
        self.market_opened = True

    def market_close(self):
        self.cancel_all_orders()
        self.market_opened = False

        df = self.dataloader.load(self.current_dt, self.current_dt, 0)
        if 'ex-dividend' not in df.columns:
            return
        for asset, shares in self.portfolio.positions.items():
            if shares == 0:
                continue
            row = df.loc[(slice(None), asset), :]
            div = row['ex-dividend']
            split = row['split_ratio']
            if div != nan:
                self.portfolio.process_dividends(asset, div)
            if split != nan:
                self.portfolio.process_split(asset, split)

    def on_subscribe(self):
        self.schedule(MarketOpen(self.market_open, -100000))  # 100ms ahead for system preparation
        self.schedule(MarketClose(self.market_close))
