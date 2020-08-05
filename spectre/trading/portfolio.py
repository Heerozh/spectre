"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Callable
import pandas as pd
import numpy as np
from .position import Position
from .stopmodel import StopModel


class Portfolio:
    def __init__(self, stop_model: StopModel = None):
        self._history = []
        self._positions = dict()
        self._cash = 0
        self._funds_change = []
        self._current_dt = None
        self.stop_model = stop_model

    def set_stop_model(self, stop_model: StopModel):
        """
        Set default portfolio stop model.
        Stop model can make more strategic triggers than stop orders.
        """
        self.stop_model = stop_model

    @property
    def history(self):
        ret = pd.DataFrame(self._history + [self._get_today_record()])
        ret.columns = pd.MultiIndex.from_tuples(ret.columns)
        ret = ret.set_index('index').sort_index(axis=0).sort_index(axis=1)
        return ret

    @property
    def fund_history(self):
        return pd.DataFrame(self._funds_change).set_index('index')

    @property
    def returns(self):
        if self._funds_change:
            value = self.history.value.sum(axis=1)
            funds = pd.DataFrame(self._funds_change)
            funds['index'] = funds['index'].fillna(value.index[0])
            funds = funds.set_index('index').amount
            return value.sub(funds, fill_value=0) / value.shift(1) - 1
        else:
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
        #         raise ValueError('{}({}) is nan in {}'.format(asset, shares, self._current_dt))
        values = [pos.value for asset, pos in self.positions.items() if pos.shares != 0]
        return sum(values) + self._cash

    @property
    def leverage(self):
        values = [pos.value for asset, pos in self.positions.items() if pos.shares != 0]
        return sum(np.abs(values)) / (sum(values) + self._cash)

    def __repr__(self):
        return "<Portfolio>" + str(self.history)[11:]

    def clear(self):
        self.__init__(self.stop_model)

    def shares(self, asset):
        try:
            return self._positions[asset].shares
        except KeyError:
            return 0

    def _get_today_record(self):
        current_date = self._current_dt.normalize()
        record = {('index', ''): current_date, ('value', 'cash'): self._cash}
        for asset, pos in self._positions.items():
            record[('avg_px', asset)] = pos.average_price
            record[('shares', asset)] = pos.shares
            record[('value', asset)] = pos.value
        return record

    def set_datetime(self, dt):
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        date = dt.normalize()
        if self._current_dt is not None:
            current_date = self._current_dt.normalize()
            if dt < self._current_dt:
                raise ValueError('Cannot set a date less than the current date')
            elif date > current_date:
                # today add to history
                self._history.append(self._get_today_record())

        self._current_dt = dt
        for pos in self._positions.values():
            pos.current_dt = dt

    def update(self, asset, amount, fill_price, commission) -> float:
        """asset position + amount, also calculation average_price and realized P&L"""
        assert self._current_dt is not None
        if amount == 0:
            return 0
        if asset in self._positions:
            empty, realized = self._positions[asset].update(
                amount, fill_price, commission, self._current_dt)
            if empty:
                del self._positions[asset]
            return realized
        else:
            self._positions[asset] = Position(
                amount, fill_price, commission, self._current_dt, self.stop_model)
            return 0

    def update_cash(self, amount, is_funds=False):
        """ is_funds: Is this cash update related to funds transfer, (deposits/withdraw) """
        self._cash += amount
        if is_funds:
            if self._current_dt is None:
                current_date = None
            else:
                current_date = self._current_dt.normalize()
            self._funds_change.append({'index': current_date, 'amount': amount})

    def process_split(self, asset, inverse_ratio: float, last_price):
        if asset not in self._positions:
            return
        pos = self._positions[asset]
        cash = pos.process_split(inverse_ratio, last_price)
        self.update_cash(cash)

    def process_dividend(self, asset, amount):
        if asset not in self._positions:
            return
        pos = self._positions[asset]
        cash = pos.process_dividend(amount)
        self.update_cash(cash)

    def _update_value_func(self, func):
        for asset, pos in self._positions.items():
            price = func(asset)
            if price and price == price:
                pos.last_price = price

    def _update_value_dict(self, prices):
        for asset, pos in self._positions.items():
            price = prices.get(asset, np.nan)
            if price == price:
                pos.last_price = price

    def update_value(self, prices: Union[Callable, dict]):
        if callable(prices):
            self._update_value_func(prices)
        elif isinstance(prices, dict):
            self._update_value_dict(prices)
        else:
            raise ValueError('prices either callable or dict')

    def check_stop_trigger(self, *args):
        ret = []
        for asset in list(self._positions.keys()):
            pos = self._positions[asset]
            ret.append(pos.check_stop_trigger(asset, -pos.shares, *args))
        return ret
