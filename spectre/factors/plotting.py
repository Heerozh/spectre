"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import math
from itertools import cycle, islice
import sys

DEFAULT_COLORS = [
    'rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)',
    'rgb(171, 99, 250)', 'rgb(255, 161, 90)', 'rgb(25, 211, 243)'
]


def plot_quantile_and_cumulative_returns(factor_data, mean_ret):
    """
    install plotly extension first:
    https://plot.ly/python/getting-started/
    conda install -c conda-forge jupyterlab=1.2
    conda install "ipywidgets=7.5"
    """
    quantiles = mean_ret.index

    if 'plotly.graph_objects' not in sys.modules:
        print('Importing plotly, it may take a while...')
    import plotly.graph_objects as go
    import plotly.subplots as subplots

    x = quantiles
    factors = mean_ret.columns.levels[0]
    periods = list(mean_ret.columns.levels[1])
    periods.sort(key=lambda cn: int(cn[:-1]))
    rows = math.ceil(len(factors))

    colors = dict(zip(periods + ['to'], cycle(DEFAULT_COLORS)))
    quantile_styles = {
        period: {'name': period, 'legendgroup': period,
                 'hovertemplate': '<b>Quantile</b>:%{x}<br>'
                                  '<b>Return</b>: %{y:.2f}bps ±%{error_y.array:.2f}bps',
                 'marker': {'color': colors[period]}}
        for period in periods
    }
    cumulative_styles = {
        period: {'name': period, 'mode': 'lines', 'legendgroup': period, 'showlegend': False,
                 'hovertemplate': '<b>Date</b>:%{x}<br>'
                                  '<b>Return</b>: %{y:.3f}%',
                 'marker': {'color': colors[period]}}
        for period in periods
    }
    turnover_styles = {'opacity': 0.2, 'name': 'turnover', 'legendgroup': 'turnover',
                       'marker': {'color': colors['to']}}

    specs = [[{}, {"secondary_y": True}]] * rows
    fig = subplots.make_subplots(
        rows=rows, cols=2,
        vertical_spacing=0.03,
        horizontal_spacing=0.06,
        specs=specs,
        subplot_titles=['Quantile Return', 'Portfolio cumulative returns'],
    )

    mean_ret = mean_ret * 10000
    for i, factor in enumerate(factors):
        row = i + 1
        weight_col = (factor, 'factor_weight')
        weighted = factor_data['Returns'].multiply(factor_data[weight_col], axis=0)
        factor_return = weighted.groupby(level='date').sum()
        for j, period in enumerate(periods):
            y = mean_ret.loc[:, (factor, period, 'mean')]
            err_y = mean_ret.loc[:, (factor, period, 'sem')]
            fig.add_trace(go.Bar(
                x=x, y=y, error_y=dict(type='data', array=err_y, thickness=0.2),
                yaxis='y1', **quantile_styles[period]
            ), row=row, col=1)
            quantile_styles[period]['showlegend'] = False

            cum_ret = factor_return[period].resample('b' + period).mean().dropna()
            cum_ret = (cum_ret + 1).cumprod() * 100 - 100
            fig.add_trace(go.Scatter(
                x=cum_ret.index, y=cum_ret.values, yaxis='y2', **cumulative_styles[period]
            ), row=row, col=2)

        fig.update_xaxes(type="category", row=row, col=1)

        fig.add_shape(go.layout.Shape(
            type="line", line=dict(width=1),
            y0=0, y1=0, x0=factor_return.index[0], x1=factor_return.index[-1],
        ), row=row, col=2)

        weight_diff = factor_data[weight_col].unstack(level=[1]).diff()
        to = weight_diff.abs().sum(axis=1) * 100
        resample = int(len(to) / 64)
        if resample > 0:
            to = to.fillna(0).rolling(resample).mean()[::resample]
        fig.add_trace(go.Bar(x=to.index, y=to.values, **turnover_styles),
                      secondary_y=True, row=row, col=2)
        turnover_styles['showlegend'] = False

        fig.update_yaxes(title_text=factor, row=row, col=1, matches='y1')
        fig.update_yaxes(row=row, col=2, ticksuffix='%')
        fig.update_yaxes(row=row, col=2, secondary_y=False, matches='y2')

    fig.update_layout(height=300 * rows, barmode='group', bargap=0.5, margin={'t': 50})
    fig.show()


def plot_factor_diagram(factor):
    import plotly.graph_objects as go
    from .factor import BaseFactor, CustomFactor, DataFactor

    color = [
        "rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)", "rgba(44, 160, 44, 0.8)",
        "rgba(214, 39, 40, 0.8)", "rgba(148, 103, 189, 0.8)", "rgba(140, 86, 75, 0.8)",
        "rgba(227, 119, 194, 0.8)", "rgba(127, 127, 127, 0.8)", "rgba(188, 189, 34, 0.8)",
        "rgba(23, 190, 207, 0.8)", "rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)",
        "rgba(44, 160, 44, 0.8)", "rgba(214, 39, 40, 0.8)", "rgba(148, 103, 189, 0.8)",
        "rgba(140, 86, 75, 0.8)", "rgba(227, 119, 194, 0.8)", "rgba(127, 127, 127, 0.8)",
        "rgba(188, 189, 34, 0.8)", "rgba(23, 190, 207, 0.8)", "rgba(31, 119, 180, 0.8)",
        "rgba(255, 127, 14, 0.8)", "rgba(44, 160, 44, 0.8)", "rgba(214, 39, 40, 0.8)",
        "rgba(148, 103, 189, 0.8)", "rgba(140, 86, 75, 0.8)", "rgba(227, 119, 194, 0.8)",
        "rgba(127, 127, 127, 0.8)", "rgba(188, 189, 34, 0.8)", "rgba(23, 190, 207, 0.8)",
        "rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)", "rgba(44, 160, 44, 0.8)",
        "rgba(214, 39, 40, 0.8)", "rgba(148, 103, 189, 0.8)", "magenta",
        "rgba(227, 119, 194, 0.8)", "rgba(127, 127, 127, 0.8)", "rgba(188, 189, 34, 0.8)",
        "rgba(23, 190, 207, 0.8)", "rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)",
        "rgba(44, 160, 44, 0.8)", "rgba(214, 39, 40, 0.8)", "rgba(148, 103, 189, 0.8)",
        "rgba(140, 86, 75, 0.8)", "rgba(227, 119, 194, 0.8)", "rgba(127, 127, 127, 0.8)"
    ]

    factor_id = dict()
    label = []
    source = []
    target = []
    value = []
    line_label = []

    def add_node(this, parent_label_id, parent_label):
        class_id = id(this)

        if class_id in factor_id:
            this_label_id = factor_id[class_id]
        else:
            this_label_id = len(label)
            if isinstance(this, DataFactor):
                label.append(this.inputs[0])
            else:
                label.append(type(this).__name__)

        if parent_label_id is not None:
            source.append(parent_label_id)
            target.append(this_label_id)
            if isinstance(this, CustomFactor):
                value.append(this.win)
            else:
                value.append(1)
            line_label.append(parent_label)

        if class_id in factor_id:
            return

        factor_id[class_id] = this_label_id
        if isinstance(this, CustomFactor):
            if this.inputs:
                for upstream in this.inputs:
                    if isinstance(upstream, BaseFactor):
                        add_node(upstream, this_label_id, 'inputs')

            if this._mask is not None:
                add_node(this._mask, this_label_id, 'mask')

    add_node(factor, None, None)

    fig = go.Figure(data=[go.Sankey(
        valueformat=".0f",
        valuesuffix="win",
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=label,
            color=list(islice(cycle(color), len(label)))
        ),
        # Add links
        link=dict(
            source=source,
            target=target,
            value=value,
            label=line_label
        ))])

    fig.update_layout(title_text="Factor Diagram")
    fig.show()
