"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import math
from itertools import cycle
import sys

DEFAULT_COLORS = [
    'rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)',
    'rgb(171, 99, 250)', 'rgb(255, 161, 90)', 'rgb(25, 211, 243)'
]


def plot_quantile_returns(mean_ret):
    """
    install plotly extension first:
    https://plot.ly/python/getting-started/
    conda install -c conda-forge jupyterlab=1.2
    conda install "ipywidgets=7.5"
    """
    quantiles = mean_ret.index

    if 'plotly.graph_objects' not in sys.modules:
        print('Importing plotly...')
    import plotly.graph_objects as go
    import plotly.subplots as subplots

    x = quantiles
    factors = mean_ret.columns.levels[0]
    periods = mean_ret.columns.levels[1]
    rows = math.ceil(len(factors) / 2)
    cols = 2

    colors = dict(zip(periods, cycle(DEFAULT_COLORS)))
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

    fig.update_layout(height=400 * rows, barmode='group', bargap=0.5,
                      title_text="Mean return by quantile")
    fig.show()


def plot_cumulative_return(factor_data):
    import plotly.graph_objects as go
    import plotly.subplots as subplots

    factors = list(factor_data.columns.levels[0])
    factors.remove('Demeaned')
    factors.remove('Returns')
    periods = factor_data['Returns'].columns

    rows = math.ceil(len(factors) / 2)
    cols = 2

    fig = go.Figure()
    fig = subplots.make_subplots(
        rows=rows, cols=2,
        subplot_titles=factors,
    )

    colors = dict(zip(periods, cycle(DEFAULT_COLORS)))
    styles = {
        period: {'name': period, 'mode': 'lines', 'legendgroup': period,
                 'marker': {'color': colors[period]}}
        for period in periods
    }

    for i, factor in enumerate(factors):
        row = int(i / cols) + 1
        col = i % cols + 1
        weight_col = (factor, 'factor_weight')
        weighted = factor_data['Returns'].multiply(factor_data[weight_col], axis=0)
        factor_return = weighted.groupby(level='date').sum()
        for period in periods:
            cumret = factor_return[period].resample('b' + period).mean().dropna()
            cumret = (cumret + 1).cumprod()
            fig.add_trace(go.Scatter(x=cumret.index, y=cumret.values, **styles[period]),
                          row=row, col=col)
            styles[period]['showlegend'] = False

            fig.add_shape(go.layout.Shape(type="line", y0=1, y1=1,
                                          x0=cumret.index[0], x1=cumret.index[-1],
                                          line=dict(width=1)),
                          row=row, col=col)

    fig.update_layout(height=400 * rows, title_text="Portfolio cumulative return")
    fig.show()
