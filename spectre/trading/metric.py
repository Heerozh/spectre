"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
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
    value_trades = (transactions.amount * transactions.fill_price).abs()
    value_trades = value_trades.groupby(value_trades.index.normalize()).sum()
    return value_trades / positions.value.sum(axis=1)


def annual_volatility(daily_returns: pd.Series):
    volatility = daily_returns.std(ddof=1)
    annual_factor = np.sqrt(252)
    return annual_factor * volatility


def plot_cumulative_returns(returns, positions, transactions, benchmark, annual_risk_free):
    import plotly.graph_objects as go
    import plotly.subplots as subplots

    fig = subplots.make_subplots(specs=[[{"secondary_y": True}]])

    cum_ret = (returns + 1).cumprod()
    fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values * 100 - 100, name='portfolio',
                             hovertemplate='<b>Date</b>:%{x}<br><b>Return</b>: %{y:.3f}%'))

    if benchmark is not None:
        cum_bench = (benchmark + 1).cumprod()
        fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values * 100 - 100,
                                 name='benchmark', line=dict(width=0.5)))

    fig.add_shape(go.layout.Shape(
        type="rect", xref="x", yref="paper", opacity=0.5, line_width=0,
        fillcolor="LightGoldenrodYellow", layer="below",
        y0=0, y1=1, x0=cum_ret.idxmax(), x1=cum_ret[cum_ret.idxmax():].idxmin(),
    ))

    to = turnover(positions, transactions) * 100
    resample = int(len(to) / 126)
    if resample > 0:
        to = to.fillna(0).rolling(resample).mean()[::resample]
    fig.add_trace(go.Bar(x=to.index, y=to.values, opacity=0.2, name='turnover'),
                  secondary_y=True)

    sr = sharpe_ratio(returns, annual_risk_free)
    bench_sr = sharpe_ratio(benchmark, annual_risk_free)
    dd, ddd = drawdown(cum_ret)
    mdd = abs(dd.min())
    mdd_dur = ddd.max()

    vol = annual_volatility(returns) * 100
    bench_vol = annual_volatility(benchmark) * 100

    ann = go.layout.Annotation(
        x=0.01, y=0.98, xref="paper", yref="paper",
        showarrow=False, borderwidth=1, bordercolor='black', align='left',
        text="<b>Overall</b> (portfolio/benchmark)<br>"
             "SharpeRatio:      {:.3f}/{:.3f}<br>"
             "MaxDrawDown:  {:.2f}%, {} Days<br>"
             "AnnualVolatility: {:.2f}%/{:.2f}%</b>"
            .format(sr, bench_sr, mdd * 100, mdd_dur, vol, bench_vol),
    )

    fig.update_layout(height=400, annotations=[ann], margin={'t': 50})
    fig.update_xaxes(tickformat='%Y-%m-%d')
    fig.update_yaxes(title_text='cumulative return', ticksuffix='%', secondary_y=False)
    fig.update_yaxes(title_text='turnover', ticksuffix='%', secondary_y=True)
    fig.show()
