"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
from .event import *


class Portfolio:
    def __init__(self):
        self._history = pd.DataFrame()
        self._positions = defaultdict(int)
        self._last_price = defaultdict(int)
        self._cash = 0
        self._current_date = None

    @property
    def history(self):
        return self._history.fillna(method='ffill')

    @property
    def positions(self):
        return self._positions

    @property
    def cash(self):
        return self._cash

    @property
    def value(self):
        values = [self._last_price[asset] * amount for asset, amount in self.positions.items()]
        return sum(values) + self._cash

    @property
    def leverage(self):
        values = [self._last_price[asset] * amount for asset, amount in self.positions.items()]
        return sum(np.abs(values)) / (sum(values) + self._cash)

    def __repr__(self):
        return "<Portfolio>:\n" + str(self.history)

    def clear(self):
        self.__init__()

    def set_date(self, date):
        if isinstance(date, str):
            date = pd.Timestamp(date)
        date = date.normalize()
        if self._current_date is not None and date < self._current_date:
            raise ValueError('Cannot set a date less than the current date')

        self._current_date = date

    def update(self, asset, amount):
        self._positions[asset] += amount
        self._history.loc[self._current_date, asset] = self._positions[asset]

    def update_cash(self, amount):
        self._cash += amount
        self._history.loc[self._current_date, 'cash'] = self._cash

    def process_split(self, asset, ratio):
        sp = self._positions[asset] * (ratio - 1)
        self.update(asset, sp)

    def process_dividends(self, asset, ratio):
        div = self._positions[asset] * ratio
        self.update_cash(div)

    def update_value(self, get_curr_price):
        for asset, shares in self._positions.items():
            price = get_curr_price(asset)
            if price:
                self._last_price[asset] = price
        self._history.loc[self._current_date, 'value'] = self.value


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

    @property
    def positions(self):
        return self._portfolio.positions

    def set_datetime(self, dt):
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

        amount = self._portfolio.value * pct / self.get_price(asset)
        held = self.positions[asset]
        amount -= held
        return self._order(asset, int(amount))

    def cancel_all_orders(self):
        raise NotImplementedError("abstractmethod")

    def process_split(self, asset, ratio):
        self._portfolio.process_split(asset, ratio)

    def process_dividends(self, asset, ratio):
        self._portfolio.process_dividends(asset, ratio)

    def get_history_positions(self):
        return self._portfolio.history

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
        self._portfolio.clear()

    def set_price(self, name: str):
        if name == 'open':
            self.price_col = self.dataloader.ohlcv[0]
        else:
            self.price_col = self.dataloader.ohlcv[3]

    def get_price(self, asset: str):
        df = self.dataloader.load(self._current_dt, self._current_dt, 0)
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
        df = self.dataloader.load(self._current_dt, self._current_dt, 1)
        df = df.loc[(slice(None), asset), :]
        price = df[self.price_col].iloc[-1]
        close_col = self.dataloader.ohlcv[3]
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
            self._current_dt, asset, amount, price, final_price, commission)
        self.orders[asset].append(order)

        # update portfolio, pay cash
        self._portfolio.update(asset, amount)
        self._portfolio.update_cash(-amount * final_price + commission)

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

        # push dividend/split data to portfolio
        df = self.dataloader.load(self._current_dt, self._current_dt, 0)
        if self.dataloader.adjustments is not None:
            div_col = self.dataloader.adjustments[0]
            sp_col = self.dataloader.adjustments[1]
            for asset, shares in self.positions.items():
                if shares == 0:
                    continue
                row = df.loc[(slice(None), asset), :]
                if div_col in df.columns:
                    div = row[div_col]
                    if div != np.nan:
                        self._portfolio.process_dividends(asset, div)
                if sp_col in df.columns:
                    split = row[sp_col]
                    if split != np.nan:
                        self._portfolio.process_split(asset, split)

        # push close price to portfolio
        self._portfolio.update_value(self.get_price)

    def on_subscribe(self):
        self.schedule(MarketOpen(self.market_open, -100000))  # 100ms ahead for system preparation
        self.schedule(MarketClose(self.market_close))
