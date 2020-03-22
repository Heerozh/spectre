"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import warnings


def plot_chart(df_prices, ohlcv, df_factor, trace_types=None, styles=None, inline=True):
    import plotly.graph_objects as go

    # group df by asset
    asset_group = df_factor.index.get_level_values(1).remove_unused_categories()
    dfs = list(df_factor.groupby(asset_group))
    if len(dfs) > 5:
        warnings.warn("Warning!! Too many assets {}, only plotting top 5.".format(len(dfs)),
                      RuntimeWarning)
        dfs = dfs[:5]
    dfs = [(asset, factors) for (asset, factors) in dfs if factors.shape[0] > 0]
    trace_types = trace_types and trace_types or {}

    # init styles
    styles = styles and styles or {}
    styles['price'] = styles.get('price', {})
    styles['volume'] = styles.get('volume', {})

    # default styles
    styles['height'] = styles.get('height', 500)
    styles['price']['line'] = styles['price'].get('line', dict(width=1))
    styles['price']['name'] = styles['price'].get('name', 'price')
    styles['volume']['opacity'] = styles['volume'].get('opacity', 0.2)
    styles['volume']['yaxis'] = styles['volume'].get('yaxis', 'y2')
    styles['volume']['name'] = styles['volume'].get('name', 'volume')

    # get y-axises
    y_axises = set()
    for k, v in styles.items():
        if not isinstance(v, dict):
            continue
        if 'yaxis' in v:
            y_axises.add('yaxis' + v['yaxis'][1:])
        if 'yref' in v:
            y_axises.add('yaxis' + v['yref'][1:])

    figs = {}
    # plotting
    for i, (asset, factors) in enumerate(dfs):
        fig = go.Figure()
        figs[asset] = fig

        factors = factors.droplevel(level=1)
        start, end = factors.index[0], factors.index[-1]

        prices = df_prices.loc[(slice(start, end), asset), :].droplevel(level=1)
        index = prices.index.strftime("%y-%m-%d %H%M")
        # add candlestick
        fig.add_trace(
            go.Candlestick(x=index, open=prices[ohlcv[0]], high=prices[ohlcv[1]],
                           low=prices[ohlcv[2]], close=prices[ohlcv[3]], **styles['price']))
        fig.add_trace(
            go.Bar(x=index, y=prices[ohlcv[4]], **styles['volume']))

        # add factors
        for col in factors.columns:
            trace_type = trace_types.get(col, 'Scatter')
            if trace_type is None:
                continue
            style = styles.get(col, {})
            style['name'] = style.get('name', col)
            fig.add_trace(go.__dict__[trace_type](x=index, y=factors[col], **style))

        new_axis = dict(anchor="free", overlaying="y", side="right", position=1)
        alpha_ordered_axises = list(y_axises)
        alpha_ordered_axises.sort()
        for y_axis in alpha_ordered_axises:
            fig.update_layout(**{y_axis: new_axis})
            new_axis['position'] -= 0.03
        x_right = new_axis['position'] + 0.03

        fig.update_layout(xaxis=dict(domain=[0, x_right]))
        fig.update_xaxes(rangeslider=dict(visible=False))
        fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)
        fig.update_layout(legend=dict(xanchor='right', x=x_right, y=1, bgcolor='rgba(0,0,0,0)'))
        fig.update_layout(height=styles['height'], barmode='group', bargap=0.5, margin={'t': 50},
                          title=asset)

        if inline:
            fig.show()

    return figs
