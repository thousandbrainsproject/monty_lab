"""
Plotting utilities for the Monty capabilities analysis.
"""

# TBP colors. Violin plots use blue.
from numbers import Number
from typing import List

import matplotlib.axes

TBP_COLORS = {
    "blue": "#00A0DF",
    "pink": "#F737BD",
    "purple": "#5D11BF",
    "green": "#008E42",
    "yellow": "#FFBE31",
}


def violinplot(
    ax: matplotlib.axes.Axes,
    data: List,
    conditions: List[str],
    rotation: Number = 0,
) -> None:
    """Add violin plot with TBP colors.

    Args:
        ax (matplotlib.axes.Axes): Axes on which to plot.
        data (List): List of arrays.
        conditions (List[str]): List of conditions (e.g., ['base', 'noise', ...])
            associated with each array in `data`.
        rotation (Number, optional): Label rotation. Defaults to 0.
    """
    vp = ax.violinplot(
        data,
        showextrema=False,
        showmedians=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor(TBP_COLORS["blue"])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax.set_xticks(list(range(1, len(conditions) + 1)))
    ax.set_xticklabels(conditions, rotation=rotation, ha="right")
