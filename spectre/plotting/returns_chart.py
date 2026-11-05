"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import math
import sys
from itertools import cycle

import pandas as pd

DEFAULT_COLORS = [
    "rgb(99, 110, 250)",
    "rgb(239, 85, 59)",
    "rgb(0, 204, 150)",
    "rgb(171, 99, 250)",
    "rgb(255, 161, 90)",
    "rgb(25, 211, 243)",
]


def _period_bins(rows, dates, period):
    """
    The bin of each of `rows` on the line of `period`, a line compounds one return per bin.
    full_run labels a period as its number of bars of the data times the bar size, which is the
    smallest gap in `dates`. Bins start at the first row, `dates` can start earlier.
    """
    bars = 1
    if len(dates) > 1:
        # up, the rows alone can be further apart than a period
        bars = math.ceil(pd.Timedelta(period) / (dates[1:] - dates[:-1]).min())
    position = dates.get_indexer(rows)
    return pd.Index((position - position[0]) // bars)


def plot_quantile_and_cumulative_returns(factor_data, mean_ret, dates=None):
    """
    Plotly Installation:
    https://github.com/plotly/plotly.py#jupyterlab-support-python-35

    :param dates: All bars of the data, a filter leaves some of them out of `factor_data`.
                  If None, the dates of `factor_data`, which are all bars without a filter.
    """
    quantiles = mean_ret.index

    if "plotly.graph_objects" not in sys.modules:
        print("Importing plotly, it may take a while...")
    import plotly.graph_objects as go
    from plotly import subplots

    x = quantiles
    factors = mean_ret.columns.levels[0]
    periods = list(mean_ret.columns.levels[1])
    periods.sort(key=pd.Timedelta)
    rows = math.ceil(len(factors))

    colors = dict(zip(periods + ["to"], cycle(DEFAULT_COLORS)))
    quantile_styles = {
        period: {
            "name": period,
            "legendgroup": period,
            "hovertemplate": "<b>Quantile</b>:%{x}<br>"
            "<b>Return</b>: %{y:.2f}bps ±%{error_y.array:.2f}bps",
            "marker": {"color": colors[period]},
        }
        for period in periods
    }
    cumulative_styles = {
        period: {
            "name": period,
            "mode": "lines",
            "legendgroup": period,
            "showlegend": False,
            "hovertemplate": "<b>Date</b>:%{x}<br><b>Return</b>: %{y:.3f}%",
            "marker": {"color": colors[period]},
        }
        for period in periods
    }
    turnover_styles = {
        "opacity": 0.2,
        "name": "turnover",
        "legendgroup": "turnover",
        "marker": {"color": colors["to"]},
    }

    specs = [[{}, {"secondary_y": True}]] * rows
    fig = subplots.make_subplots(
        rows=rows,
        cols=2,
        vertical_spacing=0.06 / rows,
        horizontal_spacing=0.06,
        specs=specs,
        subplot_titles=["Quantile Return", "Portfolio cumulative returns"],
    )

    mean_ret = mean_ret * 10000
    for i, factor in enumerate(factors):
        row = i + 1
        weight_col = (factor, "factor_weight")
        weighted = factor_data["Returns"].multiply(factor_data[weight_col], axis=0)
        factor_return = weighted.groupby(level="date").sum()
        bars = factor_return.index if dates is None else dates
        for j, period in enumerate(periods):
            y = mean_ret.loc[:, (factor, period, "mean")]
            err_y = mean_ret.loc[:, (factor, period, "sem")]
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    error_y=dict(type="data", array=err_y, thickness=0.2),
                    yaxis="y1",
                    **quantile_styles[period],
                ),
                row=row,
                col=1,
            )
            quantile_styles[period]["showlegend"] = False

            returns = factor_return[period]
            bins = _period_bins(returns.index, bars, period)
            cum_ret = returns.groupby(bins).mean()
            cum_ret = (cum_ret + 1).cumprod() * 100 - 100
            fig.add_trace(
                go.Scatter(
                    x=returns.index[~bins.duplicated()],
                    y=cum_ret.values,
                    yaxis="y2",
                    **cumulative_styles[period],
                ),
                row=row,
                col=2,
            )

        fig.update_xaxes(type="category", row=row, col=1)

        fig.add_shape(
            go.layout.Shape(
                type="line",
                line=dict(width=1),
                y0=0,
                y1=0,
                x0=factor_return.index[0],
                x1=factor_return.index[-1],
            ),
            row=row,
            col=2,
        )

        # an asset without a row holds nothing, so entering or leaving the universe is a trade
        weight_diff = factor_data[weight_col].unstack(level=[1]).fillna(0).diff()
        to = weight_diff.abs().sum(axis=1) * 100
        resample = int(len(to) / 64)
        if resample > 0:
            to = to.fillna(0).rolling(resample).mean()[::resample]
        fig.add_trace(
            go.Bar(x=to.index, y=to.values, **turnover_styles), secondary_y=True, row=row, col=2
        )
        turnover_styles["showlegend"] = False

        fig.update_yaxes(title_text=factor, row=row, col=1, matches="y1")
        fig.update_yaxes(row=row, col=2, ticksuffix="%")
        fig.update_yaxes(row=row, col=2, secondary_y=False, matches="y2")

    fig.update_layout(height=300 * rows, barmode="group", bargap=0.5, margin={"t": 50})
    return fig


def _wealth(returns, start):
    """
    1 grown by `returns` from row `start` on. It starts on the row before, the value these returns
    grow from, or a loss on the first of them would have no high to fall from. The first row of
    the returns of a portfolio is NaN, so that row is the value it starts with.
    """
    first = max(start + len(returns), 0) if start < 0 else start
    rows = returns.iloc[max(first - 1, 0) :].copy()
    if first > 0:
        rows.iloc[0] = 0
    return (rows.fillna(0) + 1).cumprod()


def cumulative_returns_fig(returns, positions, transactions, benchmark, annual_risk_free, start=0):
    import plotly.graph_objects as go
    from plotly import subplots

    from ..trading import annual_volatility, drawdown, sharpe_ratio, turnover

    fig = subplots.make_subplots(specs=[[{"secondary_y": True}]])

    cum_ret = _wealth(returns, start)
    fig.add_trace(
        go.Scatter(
            x=cum_ret.index,
            y=cum_ret.values * 100 - 100,
            name="portfolio",
            hovertemplate="<b>Date</b>:%{x}<br><b>Return</b>: %{y:.3f}%",
        )
    )
    fig.add_shape(
        go.layout.Shape(
            y0=0, y1=0, x0=cum_ret.index[0], x1=cum_ret.index[-1], type="line", line=dict(width=1)
        )
    )

    if benchmark is not None:
        cum_bench = benchmark.iloc[start:]
        cum_bench = (cum_bench + 1).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum_bench.index,
                y=cum_bench.values * 100 - 100,
                name="benchmark",
                line=dict(width=0.5),
            )
        )

    fig.add_shape(
        go.layout.Shape(
            type="rect",
            xref="x",
            yref="paper",
            opacity=0.5,
            line_width=0,
            fillcolor="LightGoldenrodYellow",
            layer="below",
            y0=0,
            y1=1,
            x0=cum_ret.idxmax(),
            x1=cum_ret[cum_ret.idxmax() :].idxmin(),
        )
    )

    to = turnover(positions, transactions).iloc[start:] * 100
    resample = int(len(to) / 126)
    if resample > 0:
        to = to.fillna(0).rolling(resample).mean()[::resample]
    fig.add_trace(go.Bar(x=to.index, y=to.values, opacity=0.2, name="turnover"), secondary_y=True)

    sr = sharpe_ratio(returns, annual_risk_free)
    dd, ddd = drawdown(cum_ret)
    mdd = abs(dd.min())
    mdd_dur = ddd.max()
    vol = annual_volatility(returns) * 100

    if benchmark is not None:
        bench_sr = sharpe_ratio(benchmark, annual_risk_free)
        bench_vol = annual_volatility(benchmark) * 100
    else:
        bench_sr = 0
        bench_vol = 0

    ann = go.layout.Annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        borderwidth=1,
        bordercolor="black",
        align="left",
        text="<b>Overall</b> (portfolio/benchmark)<br>"
        f"SharpeRatio:      {sr:.3f}/{bench_sr:.3f}<br>"
        f"MaxDrawDown:  {mdd * 100:.2f}%, {mdd_dur} Days<br>"
        f"AnnualVolatility: {vol:.2f}%/{bench_vol:.2f}%</b>",
    )

    fig.update_layout(height=400, annotations=[ann], margin={"t": 50})
    fig.update_xaxes(tickformat="%Y-%m-%d")
    fig.update_yaxes(title_text="cumulative return", ticksuffix="%", secondary_y=False)
    fig.update_yaxes(title_text="turnover", ticksuffix="%", secondary_y=True)
    return fig
