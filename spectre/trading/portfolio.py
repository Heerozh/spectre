"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Callable
import pandas as pd
import numpy as np
from collections import defaultdict


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
        self._last_price[asset] = last_price / inverse_ratio
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
