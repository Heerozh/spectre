"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import numpy as np
import pandas as pd


def drawdown(cumulative_returns):
    max_ret = cumulative_returns.cummax()
    dd = cumulative_returns / max_ret - 1
    dd_group = 0

    def drawdown_split(x):
        nonlocal dd_group
        if dd[x] == 0:
            dd_group += 1
        return dd_group

    dd_duration = dd.groupby(drawdown_split).cumcount()
    return dd, dd_duration


def sharpe_ratio(daily_returns: pd.Series, annual_risk_free_rate):
    risk_adj_ret = daily_returns.sub(annual_risk_free_rate/252)
    annual_factor = np.sqrt(252)
    return annual_factor * risk_adj_ret.mean() / risk_adj_ret.std(ddof=1)


def turnover(positions, transactions):
    if transactions.shape[0] == 0:
        return transactions.amount
    value_trades = (transactions.amount * transactions.fill_price).abs()
    value_trades = value_trades.groupby(value_trades.index.normalize()).sum()
    return value_trades / positions.value.sum(axis=1)


def annual_volatility(daily_returns: pd.Series):
    volatility = daily_returns.std(ddof=1)
    annual_factor = np.sqrt(252)
    return annual_factor * volatility
