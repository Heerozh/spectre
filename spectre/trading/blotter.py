"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Union, Iterable
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
from .event import *
from .portfolio import Portfolio
from ..data import DataLoaderFastGetter


class CommissionModel:
    def __init__(self, percentage: float, per_share: float, minimum: float):
        self.percentage = percentage
        self.per_share = per_share
        self.minimum = minimum

    def calculate(self, price: float, shares: int):
        commission = price * abs(shares) * self.percentage
        commission += abs(shares) * self.per_share
        return max(commission, self.minimum)


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
            raise OverflowError(
                'Cannot order more than ±{} shares: {}, set `blotter.max_shares` value '
                'to change this limit.'.format(self.max_shares, amount))

        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")

        opened = self._portfolio.shares(asset)
        if self.long_only and (amount + opened) < 0:
            raise ValueError("Long only blotter, order amount {}, opened {}.".format(
                amount, opened))

        return self._order(asset, amount)

    def _order_target(self, asset: str, target: Union[int, float]):
        opened = self._portfolio.shares(asset)
        amount = int(target - opened)

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
        return self._order_target(asset, target)

    def batch_order_target_percent(self, assets: Iterable[str], weights: Iterable[float]):
        pf_value = self._portfolio.value
        prices = self.get_price(assets)
        skipped = []
        if None in assets or np.nan in assets:
            raise ValueError('None/NaN in `assets: ' + str(assets))
        if None in weights or np.nan in weights:
            raise ValueError('None/NaN in `weights: ' + str(weights))
        assets = list(assets)  # copy for preventing del items in loop
        for asset, pct in zip(assets, weights):
            try:
                price = prices[asset]
                if price != price:
                    raise KeyError("")
                target = (pf_value * pct) / price
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

    def __init__(self, dataloader, capital_base=100000, daily_curb=None, start=None):
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

        df = dataloader.load(start, None, 0).copy()
        df['__pct_chg'] = df[dataloader.ohlcv[3]].groupby(level=1).pct_change()
        self._data = df
        self._prices = DataLoaderFastGetter(df[list(dataloader.ohlcv) + ['__pct_chg']])
        self._current_prices_col = None
        self._current_prices = None
        if dataloader.adjustments is not None:
            div_col = dataloader.adjustments[0]
            sp_col = dataloader.adjustments[1]
            adj = df[[div_col, sp_col, dataloader.ohlcv[3]]]
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
        self._portfolio.update_cash(self.capital_base)

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

    def _order(self, asset, amount):
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
            curr_changes = self._prices.get_as_dict(curr_prices.row_slice, column_id=-1)
            change = curr_changes[asset]
            # Detecting whether transactions are possible
            if abs(change) > self.daily_curb:
                return False

        # commission, slippage
        commission = self.commission.calculate(price, amount)
        slippage = self.slippage.calculate(price, 1)
        if amount < 0:
            commission += self.short_fee.calculate(price, amount)
            fill_price = price - slippage
        else:
            fill_price = price + slippage

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
                    self._portfolio.process_dividends(asset, row[div_col])
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
