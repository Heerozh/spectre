"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from itertools import cycle, islice


def plot_factor_diagram(factor):
    import plotly.graph_objects as go
    from ..factors import BaseFactor, CustomFactor
    from ..factors import DataFactor

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

    def add_node(this, parent_label_id, parent_label, parent_win):
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
            value.append(parent_win)
            line_label.append(parent_label)

        if class_id in factor_id:
            return

        if isinstance(this, CustomFactor):
            this_win = this.win
        else:
            this_win = 1

        factor_id[class_id] = this_label_id
        if isinstance(this, CustomFactor):
            if this.inputs:
                for upstream in this.inputs:
                    if isinstance(upstream, BaseFactor):
                        add_node(upstream, this_label_id, 'inputs', this_win)

            if this._mask is not None:
                add_node(this._mask, this_label_id, 'mask', this_win)

    add_node(factor, None, None, None)

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
