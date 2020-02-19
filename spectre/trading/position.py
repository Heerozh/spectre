"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import math
from .stopmodel import StopTracker


def sign(x):
    return math.copysign(1, x)


class Position:
    def __init__(self, shares: int, fill_price: float, commission: float,
                 stop_tracker: StopTracker = None):
        self._shares = shares
        self._average_price = fill_price + commission / shares
        self._last_price = fill_price
        self._realized = 0
        self.stop_tracker = stop_tracker
        if stop_tracker:
            self.stop_tracker.tracking_position = self

    @property
    def value(self):
        return self._shares * self._last_price

    @property
    def shares(self):
        return self._shares

    @property
    def average_price(self):
        return self._average_price

    @property
    def last_price(self):
        return self._last_price

    @last_price.setter
    def last_price(self, last_price: float):
        self._last_price = last_price
        if self.stop_tracker:
            self.stop_tracker.update_price(last_price)

    @property
    def realized(self):
        return self._realized

    @property
    def unrealized(self):
        return (self._last_price - self._average_price) * self._shares

    @property
    def unrealized_percent(self):
        return (self._last_price / self._average_price - 1) * sign(self._shares)

    def update(self, amount: int, fill_price: float, commission: float) -> bool:
        """
        position + amount, fill_price and commission is for calculation average_price and P&L
        return True when position is empty.
        """
        before_shares = self._shares
        before_avg_px = self._average_price
        after_shares = before_shares + amount

        # If the position is reversed, it will be filled in 2 steps
        if after_shares != 0 and sign(after_shares) != sign(before_shares):
            fill_1 = amount - after_shares
            fill_2 = amount - fill_1
            per_comm = commission / amount
            # close position, this class just save last state, so it can be skipped
            # self.update(fill_1, fill_price, per_comm * fill_1)
            # open a new position
            self.__init__(fill_2, fill_price, per_comm * fill_2)
            return False
        else:
            cum_cost = self._average_price * before_shares + amount * fill_price + commission
            self._shares = after_shares
            if after_shares == 0:
                self._average_price = 0
                self._realized += fill_price * amount - commission
                self.last_price = fill_price
                return True
            else:
                self._average_price = cum_cost / after_shares
                self._realized += (before_avg_px - self._average_price) * abs(after_shares)
                self.last_price = fill_price
                return False

    def process_split(self, inverse_ratio: float, last_price: float) -> float:
        if inverse_ratio != inverse_ratio or inverse_ratio == 1:
            return 0
        sp = self._shares * inverse_ratio
        cash = 0
        if inverse_ratio < 1:  # reverse split remaining to cash
            remaining = int(self._shares - int(sp) / inverse_ratio)  # for more precise
            if remaining != 0:
                cash = remaining * last_price
        self._shares = int(sp)
        self._average_price = self._average_price / inverse_ratio
        self.last_price = last_price / inverse_ratio

        if self.stop_tracker:
            self.stop_tracker.process_split(last_price)
        return cash

    def process_dividends(self, amount: float) -> float:
        if amount != amount or amount == 0:
            return 0
        self._average_price -= amount
        cash = self._shares * amount
        return cash

    def check_stop_trigger(self, *args):
        if self.stop_tracker:
            return self.stop_tracker.check_trigger(*args)
