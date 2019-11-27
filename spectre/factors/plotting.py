"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import math
from itertools import cycle


def plot_quantile_returns(mean_ret):
    quantiles = mean_ret.index

    import plotly.graph_objects as go
    import plotly.subplots as subplots

    x = quantiles
    factors = mean_ret.columns.levels[0]
    periods = mean_ret.columns.levels[1]
    rows = math.ceil(len(factors) / 2)
    cols = 2

    colors = dict(zip(periods, cycle([
        'rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)',
        'rgb(171, 99, 250)', 'rgb(255, 161, 90)', 'rgb(25, 211, 243)'
    ])))
    styles = {
        period: {'name': period, 'legendgroup': period,
                 'hovertemplate': '<b>Quantile</b>:%{x}<br>'
                                  '<b>Return</b>: %{y:.3f}%±%{error_y.array:.3f}%',
                 'marker': {'color': colors[period]}}
        for period in periods
    }

    fig = go.Figure()
    fig = subplots.make_subplots(
        rows=rows, cols=2,
        subplot_titles=factors,
    )

    for i, factor in enumerate(factors):
        row = int(i / cols) + 1
        col = i % cols + 1
        for j, period in enumerate(periods):
            y = mean_ret.loc[:, (factor, period, 'mean')] * 100
            err_y = mean_ret.loc[:, (factor, period, 'sem')] * 100
            fig.add_trace(go.Bar(
                x=x, y=y, error_y=dict(type='data', array=err_y, thickness=0.2),
                **styles[period]
            ), row=row, col=col)
            styles[period]['showlegend'] = False
            fig.update_xaxes(title_text="factor quantile", type="category", row=row, col=col)

    fig.update_layout(height=400*rows, barmode='group', bargap=0.5,
                      title_text="Mean return by quantile")
    fig.show()


def plot_cumulative_return(factor_data):
    pass