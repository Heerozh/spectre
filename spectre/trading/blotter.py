"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import glob
import os
from collections import defaultdict, namedtuple
from collections.abc import Iterable

# import pandas as pd
import numpy as np
import pandas as pd

from ..data import DataLoaderFastGetter
from .event import EventReceiver, EveryBarData, MarketClose, MarketOpen
from .portfolio import Portfolio


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

    def calculate(self, asset: str, price: float, amount_to_volume_ratio: float):
        """
        Returns the fill price, slippage is added to the price of buy orders (positive ratio),
        and subtracted from the price of sell orders (negative ratio).
        """
        if self.max_ratio == 0:
            return price

        def sigmoid(x, edge0, edge1):
            return (1 + 200 ** (-((x - edge0) / (edge1 - edge0)) + 0.5)) ** (-1)

        slippage = price * self.percentage * sigmoid(abs(amount_to_volume_ratio), 0, self.max_ratio)
        slippage = max(slippage, self.minimum)
        if amount_to_volume_ratio < 0:
            return price - slippage
        return price + slippage


class DailyCurbModel:
    def __init__(self, percentage: float):
        self.percentage = percentage

    def calculate(
        self,
        asset: str,
        current_price: float,
        shares: int,
        current_high: float,
        current_low: float,
        last_close: float,
        last_div: float,
        last_sp: float,
    ):
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

    max_shares = int(1e5)

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

    def get_price(self, asset: str | Iterable):
        raise NotImplementedError("abstractmethod")

    def _order(self, asset, amount, price=None):
        raise NotImplementedError("abstractmethod")

    def order(self, asset: str, amount: int, price: str | None = None):
        if abs(amount) > self.max_shares:
            raise OverflowError(
                f"Cannot order more than ±{self.max_shares} shares: {amount}, "
                "set `blotter.max_shares` value to change this limit."
            )

        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")

        opened = self._portfolio.shares(asset)
        if self.long_only and (amount + opened) < 0:
            raise ValueError(f"Long only blotter, order amount {amount}, opened {opened}.")
        if (amount % self.order_multiplier) != 0:
            raise ValueError(f"Order amount must be placed in multiples of {self.order_multiplier}")
        return self._order(asset, amount, price)

    def _order_target(self, asset: str, target: float):
        opened = self._portfolio.shares(asset)
        amount = target - opened

        if abs(amount) > self.max_shares:
            raise OverflowError(
                f"Cannot order more than ±{self.max_shares} shares: {amount}, "
                "set `blotter.max_shares` value to change this limit."
            )

        return self._order(asset, amount)

    def order_target(self, asset: str, target: float):
        if not isinstance(asset, str):
            raise KeyError("`asset` must be a string")
        if not isinstance(target, (int, float)):
            raise ValueError("`target` must be int or float")
        if self.long_only and target < 0:
            raise ValueError("Long only blotter, `target` must greater than 0.")

        return self._order_target(asset, target)

    def batch_order_target(self, assets: Iterable[str], targets: Iterable[float]):
        skipped = []
        if None in assets or np.any([not (a == a) for a in assets]):
            raise ValueError("None/NaN in `assets: " + str(assets))
        if None in targets or np.any([not (t == t) for t in targets]):
            raise ValueError("None/NaN in `targets: " + str(targets))
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
        if price is None or price != price:
            return False
        target = (self._portfolio.value * pct) / price
        target = int(round(target / self.order_multiplier)) * self.order_multiplier
        return self._order_target(asset, target)

    def batch_order_target_percent(self, assets: Iterable[str], weights: Iterable[float]):
        pf_value = self._portfolio.value
        prices = self.get_price(assets)
        skipped = []
        if None in assets or np.any([not (a == a) for a in assets]):
            raise ValueError("None/NaN in `assets: " + str(assets))
        if None in weights or np.any([not (w == w) for w in weights]):
            raise ValueError("None/NaN in `weights: " + str(weights))
        assets = list(assets)  # copy for preventing del items in loop
        for asset, pct in zip(assets, weights):
            try:
                price = prices[asset]
                if price != price:
                    raise KeyError("")
                if price == 0:
                    raise RuntimeError(
                        f"{asset} price on {self._current_dt} is zero, you have a data error"
                    )
                target = (pf_value * pct) / price
                # todo 碎股情况，持有160股并不能再买40股，影响不大但最好弄掉,
                # 不过现在交易股数方式就已经不一样了，所以也无所谓？
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


def _last_of_previous_day(rows, tz):
    """
    `rows` are the bars of one asset. Returns, for every bar, the last valid values before the day
    of that bar in the timezone `tz`, which on daily bars is the previous bar.
    """
    days = rows.index.get_level_values(0).tz_convert(tz).normalize()
    return rows.ffill().shift(1).iloc[days.searchsorted(days)].set_axis(rows.index)


class SimulationBlotter(BaseBlotter, EventReceiver):
    Order = namedtuple(
        "Order", ["date", "asset", "amount", "price", "fill_price", "commission", "realized"]
    )

    def __init__(self, dataloader, capital_base=100000, daily_curb=None, start=None, ohlcv=None):
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

        # 1 bar before `start`, the first bar has to know what came before it
        df = dataloader.load(start, None, 1).copy()
        ohlcv = ohlcv or dataloader.ohlcv
        # add the last known volume for slippage at the open, in the shares of the bar reading it
        scale = 1
        if dataloader.adjustments is not None:
            scale = df[dataloader.adjustment_multipliers[1]]
        volume = (df[ohlcv[4]] * scale).groupby(level=1, observed=False).ffill()
        df["__last_volume"] = volume.groupby(level=1, observed=False).shift(1) / scale
        self._data = df
        self._curb_cols = [ohlcv[3]]
        if dataloader.adjustments is not None:
            self._curb_cols += list(dataloader.adjustments[:2])
        self._previous_day = None
        self._previous_day_tz = None
        self._prices = DataLoaderFastGetter(df[list(ohlcv) + ["__last_volume"]])
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
        return "<Transactions>" + str(self.get_transactions())[14:] + "\n" + str(self.portfolio)

    def clear(self):
        self.orders = defaultdict(list)
        self._portfolio.clear()
        self._portfolio.update_cash(self.capital_base, is_funds=True)
        self.market_opened = False
        self._current_dt = None
        self._current_prices_col = None
        self._current_prices = None

    def _advance_to(self, dt: pd.Timestamp) -> None:
        """Move the datetime forward, and charge the interest of the time that passed."""
        # no price of the new datetime is known until `set_price` picks one
        self._current_prices_col = None
        self._current_prices = None
        # get last datetime before portfolio datetime change
        last_dt = self._portfolio.current_dt
        if last_dt:
            self._portfolio.process_borrow_interest(
                (dt - last_dt) / pd.Timedelta(days=1),
                self.borrow_money_interest_rate,
                self.borrow_stock_interest_rate,
            )
        super().set_datetime(dt)

    def set_datetime(self, dt: pd.Timestamp) -> None:
        if self._current_dt is not None:
            # make up events for skipped days for update splits and divs.
            current_date = self._current_dt.normalize()
            target_date = dt.normalize()
            if current_date != target_date:
                dates = self._prices.indexes[0]
                rows = slice(dates.searchsorted(self._current_dt), dates.searchsorted(target_date))
                # `dt` is in the timezone of the market, the data is in UTC
                bar_times = dates[rows].unique().tz_convert(dt.tz)
                bar_dates = bar_times.normalize()
                for date in bar_dates.unique():
                    if date == current_date:
                        continue

                    day_times = bar_times[bar_dates == date]
                    open_time = day_times[0]
                    self._advance_to(open_time)
                    self.set_price("open")
                    self.market_open(self)

                    close_time = day_times[-1]
                    self._advance_to(close_time)
                    self.set_price("close")
                    self.update_portfolio_value()
                    self.market_close(self)
        self._advance_to(dt)

    def set_price(self, name: str):
        if name == "open":
            self._current_prices_col = 0  # ohlcv[0] = o
        else:
            self._current_prices_col = 3  # ohlcv[3] = c
        self._current_prices = None

    def _get_previous_day(self):
        """
        Close, dividend and split ratio of the day before each bar, for the daily curb. Only the
        datetimes the blotter is given tell the timezone of the market, which decides what a day
        is, so this is built on the first use.
        """
        tz = self._current_dt.tz
        if self._previous_day is None or self._previous_day_tz != tz:
            df = self._data[self._curb_cols]
            grouped = df.groupby(level=1, group_keys=False, observed=False)
            lasts = grouped.apply(_last_of_previous_day, tz).reindex(df.index)
            self._previous_day = DataLoaderFastGetter(lasts)
            self._previous_day_tz = tz
        return self._previous_day

    def _get_current_prices(self):
        if self._current_prices_col is None:
            raise ValueError("_current_prices is None, maybe set_price was not called.")

        if self._current_prices is None:
            self._current_prices = self._prices.get_as_dict(
                self._current_dt, column_id=self._current_prices_col
            )

        return self._current_prices

    def get_price(self, asset: str | Iterable | None):
        if not self.market_opened:
            raise RuntimeError(
                "Out of market hours. Maybe you rebalance at AfterMarketClose; "
                "or BeforeMarketOpen; or EveryBarData on daily data; "
                "or you did not subscribe this class with SimulationEventManager"
            )

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
            raise RuntimeError(
                "Out of market hours. Maybe you rebalance at AfterMarketClose; "
                "or BeforeMarketOpen; or EveryBarData on daily data; "
                "or you did not subscribe this class with SimulationEventManager"
            )
        if amount == 0:
            return True

        # get price and change
        price = self.get_price(asset)
        if price is None or price != price:
            return False

        # trading curb for daily return
        if self.daily_curb is not None:
            curr_prices = self._get_current_prices()
            curr_high = self._prices.get_as_dict(curr_prices.row_slice, column_id=1)
            curr_low = self._prices.get_as_dict(curr_prices.row_slice, column_id=2)
            previous_day = self._get_previous_day()
            last_close = previous_day.get_as_dict(curr_prices.row_slice, column_id=0)
            if self._adjustments is None:
                last_div = defaultdict(float)
                last_sp = defaultdict(lambda: 1.0)
            else:
                last_div = previous_day.get_as_dict(curr_prices.row_slice, column_id=1)
                last_sp = previous_day.get_as_dict(curr_prices.row_slice, column_id=2)
            # Detecting whether transactions are possible
            price, amount = self.daily_curb.calculate(
                asset,
                price,
                amount,
                curr_high[asset],
                curr_low[asset],
                last_close[asset],
                last_div[asset],
                last_sp[asset],
            )
            if amount == 0:
                return False

        # commission, slippage
        fill_price = price
        if not self.slippage.is_empty():
            # the volume of a bar is only known after it closed. ohlcv=01234, 5 is the last known
            volume_col = 5 if self._current_prices_col == 0 else 4
            volume = self._prices.get_as_dict(self._current_dt, column_id=volume_col)[asset]
            # a volume nobody knows is treated as no volume, which is the full slippage
            ratio = amount / volume if volume > 0 else np.copysign(np.inf, amount)
            fill_price = self.slippage.calculate(asset, price, ratio)
        commission = self.commission.calculate(asset, fill_price, amount)
        if amount < 0:
            commission += self.short_fee.calculate(asset, fill_price, amount)
        commission = round(commission, 2)

        # update portfolio, pay cash
        realized = self._portfolio.update(asset, amount, fill_price, commission)
        self._portfolio.update_cash(-amount * fill_price - commission)

        # make order
        order = SimulationBlotter.Order(
            self._current_dt, asset, amount, price, fill_price, commission, realized
        )
        self.orders[asset].append(order)
        return True

    def cancel_all_orders(self):
        # don't need
        pass

    def get_transactions(self):
        data = []
        for orders in self.orders.values():
            for o in orders:
                data.append(
                    dict(
                        index=o.date,
                        amount=o.amount,
                        price=o.price,
                        symbol=o.asset,
                        fill_price=o.fill_price,
                        commission=o.commission,
                        realized=o.realized,
                    )
                )
        ret = pd.DataFrame(
            data,
            columns=["index", "symbol", "amount", "price", "fill_price", "commission", "realized"],
        )
        ret = ret.set_index("index").sort_index()
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
                    # the close is still cum-dividend, the split applies to the ex-dividend price
                    ex_price = row[close_col] - row[div_col]
                    self._portfolio.process_split(asset, 1 / row[sp_col], ex_price)
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

        for date, df in self.orders.groupby(pd.Grouper(key="date", freq="D")):
            # skip holiday
            if (
                prices_df is not None
                and date.normalize() not in prices_df.index.levels[0]
                and df.empty
            ):
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
                if row.status == "Cash":
                    self._portfolio.update_cash(row.filled_amount, is_funds=True)
                elif row.status == "Filled":
                    self._portfolio.update(
                        row.symbol, row.filled_amount, row.filled_price, row.commission
                    )
                    self._portfolio.update_cash(
                        -row.filled_amount * row.filled_price - row.commission
                    )

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
                if row.status == "Dividend":
                    self._portfolio.process_dividend(
                        row.symbol, row.filled_amount, tax=row.commission
                    )
                elif row.status == "Split":
                    self._portfolio.process_split(row.symbol, row.filled_amount, row.filled_price)

    def load(self):
        """Reload portfolio/orders from working_dir, for updating account and order status"""
        pattern = os.path.join(self.working_dir, "orders_*.csv")
        files = glob.glob(pattern)

        col_types = {
            "id": np.int64,
            "target_percent": np.float64,
            "action_value": np.float64,
            "amount": np.float64,
            "limit_price": str,
            "filled_amount": np.float64,
            "filled_price": np.float64,
            "filled_percent": np.float64,
            "commission": np.float64,
            "realized": np.float64,
        }
        if len(files) == 0:
            self.orders = pd.DataFrame(
                columns=[
                    "id",
                    "date",
                    "status",
                    "symbol",
                    "target_percent",
                    "action_value",
                    "amount",
                    "limit_price",
                    "filled_amount",
                    "filled_price",
                    "filled_percent",
                    "commission",
                    "realized",
                ]
            )
            print("Create ManualBlotter with empty orders.")
            for k, v in col_types.items():
                self.orders[k] = self.orders[k].astype(v)
            self.orders = self.orders.set_index("id")
        else:
            self.orders = pd.concat(
                [
                    pd.read_csv(fn, index_col="id", dtype=col_types, parse_dates=False)
                    for fn in files
                ]
            )
            self.orders.date = pd.to_datetime(self.orders.date, utc=True)
            print(
                f"Orders loaded: {self.orders.date.iloc[0]} - {self.orders.date.iloc[-1]}, "
                f"total: {len(self.orders)} orders"
            )
            self.orders["date"] = self.orders["date"].dt.tz_convert(self.time_zone)
            self._rebuild_from_orders()

    def save(self):
        """
        Save orders and changes to working_dir.
        """
        self.orders.index.name = "id"
        for n, g in self.orders.groupby(pd.Grouper(key="date", freq="D")):
            name = n.strftime("orders_%Y-%m-%d.csv")
            if not g.empty:
                numbers = g.select_dtypes("number").columns
                g.round(dict.fromkeys(numbers, 4)).to_csv(os.path.join(self.working_dir, name))

    def set_datetime(self, dt: pd.Timestamp) -> None:
        assert str(dt.tz) == self.time_zone
        super().set_datetime(dt)

    def transfer_funds(self, amount):
        """Call this after you transfer funds to broker"""
        assert self._current_dt is not None
        order = pd.Series(
            dict(
                date=self._current_dt,
                status="Cash",
                symbol="cash",
                target_percent=0.0,
                action_value=0.0,
                amount=amount,
                limit_price="/",
                filled_amount=amount,
                filled_price=0.0,
                filled_percent=0.0,
                commission=0.0,
                realized=0.0,
            )
        )
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

    def order(self, asset: str, amount: int, price: str | None = None):
        raise NotImplementedError("not support")

    def order_target(self, asset: str, target: float):
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
        if pct == 0.0 and asset not in self.positions:
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
            raise ValueError("call blotter.set_last_price(dict) first.")
        if pct == 0:
            # if an asset removed from universe,
            # u don't want to get last_price because it not exists.
            target_amount = 0
        else:
            last_close_price = self.last_price[asset]
            target_amount = target_value / last_close_price

        multiplier = self.order_multiplier
        if target_amount == 0:
            # if close a position, allow odd lots
            action_amount = -opened_shares
        else:
            action_amount = int(round((target_amount - opened_shares) / multiplier)) * multiplier

        order = pd.Series(
            dict(
                date=self._current_dt,
                status="PendingSubmit",
                symbol=asset,
                target_percent=pct,
                action_value=action_value,
                amount=action_amount,
                limit_price="Market",
                filled_amount=0,
                filled_price=0,
                filled_percent=0,
                commission=0,
                realized=0,
            )
        )
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
            raise ValueError("None/NaN in `assets: " + str(assets))
        if None in weights or np.any([not (w == w) for w in weights]):
            raise ValueError("None/NaN in `weights: " + str(weights))
        ret = dict()
        for asset, pct in zip(assets, weights):
            oid = self.order_target_percent(asset, pct)
            ret[asset] = oid
        return ret

    def order_filled(self, oid, filled_amount, filled_price, commission):
        """Call this after you order filled by broker"""
        assert self._current_dt is not None

        order = self.orders.loc[oid].copy()

        realized = self._portfolio.update(order.symbol, filled_amount, filled_price, commission)
        self._portfolio.update_cash(-filled_amount * filled_price - commission)

        if order.status == "Filled":
            total_turnover = abs(order.filled_amount * order.filled_price) + abs(
                filled_amount * filled_price
            )

            filled_amount += order.filled_amount
            filled_price = total_turnover / abs(filled_amount)
            commission += order.commission
            realized += order.realized

        order.status = "Filled"
        order.filled_amount = filled_amount
        order.filled_price = filled_price
        order.filled_percent = round(filled_price * filled_amount / order.action_value, 3)
        order.realized = realized
        order.commission = commission
        self.orders.loc[oid] = order

    def order_cancelled(self, oid):
        """Call this after you cancelled one order"""
        self.orders.loc[oid, "status"] = "Cancelled"

    def position_dividend(self, asset, amount, tax, time_delta):
        """Call this if your position has dividends"""
        assert self._current_dt is not None

        order = pd.Series(
            dict(
                date=self._current_dt + time_delta,
                status="Dividend",
                symbol=asset,
                target_percent=0,
                action_value=0,
                amount=amount,
                limit_price="/",
                filled_amount=amount,
                filled_price=0,
                filled_percent=0,
                commission=tax,
                realized=0,
            )
        )
        order.name = max(self.orders.index) + 1
        self.orders = pd.concat([self.orders, pd.DataFrame([order])])
        self._portfolio.process_dividend(asset, amount, tax)

    def position_split(self, asset, inverse_ratio: float, last_price, time_delta):
        """Call this if your position has splits"""
        assert self._current_dt is not None
        if last_price is None:
            last_price = self.positions[asset].last_price
        assert last_price is not None

        order = pd.Series(
            dict(
                date=self._current_dt + time_delta,
                status="Split",
                symbol=asset,
                target_percent=0,
                action_value=0,
                amount=inverse_ratio,
                limit_price="/",
                filled_amount=inverse_ratio,
                filled_price=last_price,
                filled_percent=0,
                commission=0,
                realized=0,
            )
        )
        order.name = max(self.orders.index) + 1
        self.orders = pd.concat([self.orders, pd.DataFrame([order])])
        self._portfolio.process_split(asset, inverse_ratio, last_price)

    def update_portfolio_value(self, prices):
        """Call this nightly"""
        if len(self._portfolio.positions) > 0:
            self._portfolio.update_value(prices)

    def get_price(self, asset: str | Iterable):
        raise NotImplementedError("not supported")

    def get_transactions(self):
        orders = self.orders[self.orders.status == "Filled"]
        ret = pd.DataFrame(
            columns=["index", "symbol", "amount", "price", "fill_price", "commission", "realized"]
        )
        ret["index"] = orders.date
        ret.symbol = orders.symbol
        ret.amount = orders.filled_amount
        ret.price = orders.filled_price
        ret.fill_price = orders.filled_price + orders.commission / orders.filled_amount
        ret.commission = orders.commission
        ret.realized = orders.realized
        ret = ret.set_index("index").sort_index()
        return ret

    @property
    def pendings(self):
        return self.orders[self.orders.status == "PendingSubmit"]

    def cancel_all_orders(self):
        raise NotImplementedError("not supported")
