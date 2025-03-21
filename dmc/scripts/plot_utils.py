# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""
Plotting utilities for the Monty capabilities analysis.
"""

# TBP colors. Violin plots use blue.
from numbers import Number
from typing import (
    Container,
    Mapping,
    Optional,
    Sequence,
)

import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D

TBP_COLORS = {
    "blue": "#00A0DF",
    "pink": "#F737BD",
    "purple": "#5D11BF",
    "green": "#008E42",
    "yellow": "#FFBE31",
}


def axes3d_clean(
    ax: Axes3D,
    grid: bool = True,
    ticks: bool = False,
    label_axes: bool = False,
) -> None:
    """Remove clutter from 3D axes.

    Args:
        grid (bool): Whether to show the background grid. Default is True.
        ticks (bool): Whether to show the ticks that stick out the side of the
            axes. Setting this to `False` does not remove grid lines. Default is False.
        label_axes (bool): Whether to show the x, y, z axis labels. Default is False.
    """

    # # Turn grid on or off.
    # ax.grid(grid)

    # Remove dark spines that outline the plot.
    ax.w_xaxis.line.set_color((1, 1, 1, 0))  # Hide X-axis line
    ax.w_yaxis.line.set_color((1, 1, 1, 0))  # Hide Y-axis line
    ax.w_zaxis.line.set_color((1, 1, 1, 0))  # Hide Z-axis line

    # Optionally remove ticks. This method keeps the grid lines visible while
    # making the stubs that stick out of the plot invisible.
    if not ticks:
        ax.tick_params(axis="x", colors=(0, 0, 0, 0))  # Make X-axis ticks invisible
        ax.tick_params(axis="y", colors=(0, 0, 0, 0))  # Make Y-axis ticks invisible
        ax.tick_params(axis="z", colors=(0, 0, 0, 0))  # Make Z-axis ticks invisible

    # Optionally remove or add axis labels.
    if label_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    # Turn grid on or off.
    ax.grid(grid)


def axes3d_set_aspect_equal(ax: Axes3D) -> None:
    """Set equal aspect ratio for 3D axes."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Get the max range
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    half_max_range = max(x_range, y_range, z_range) / 2

    # Find midpoints
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set new limits
    ax.set_xlim([x_middle - half_max_range, x_middle + half_max_range])
    ax.set_ylim([y_middle - half_max_range, y_middle + half_max_range])
    ax.set_zlim([z_middle - half_max_range, z_middle + half_max_range])

    # Set aspect ratio.
    ax.set_box_aspect([1, 1, 1])


def add_legend(
    ax: plt.Axes,
    colors: Container[str],
    labels: Container[str],
    loc: Optional[str] = None,
    lw: int = 4,
    fontsize: int = 8,
) -> matplotlib.legend.Legend:
    # Create custom legend handles (axes.legend() doesn't work when multiple
    # violin plots are on the same axes.
    legend_handles = []
    for i in range(len(colors)):
        handle = Line2D([0], [0], color=colors[i], lw=lw, label=labels[i])
        legend_handles.append(handle)

    return ax.legend(handles=legend_handles, loc=loc, fontsize=fontsize)


def violinplot(
    dataset: Sequence,
    positions: Sequence,
    width: Number = 0.8,
    color: Optional[str] = None,
    alpha: Optional[Number] = 1,
    edgecolor: Optional[str] = None,
    showextrema: bool = False,
    showmeans: bool = False,
    showmedians: bool = False,
    percentiles: Optional[Sequence] = None,
    side: str = "both",
    gap: float = 0.0,
    percentile_style: Optional[Mapping] = None,
    median_style: Optional[Mapping] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Create a violin plot with customizable styling.

    Args:
        dataset (Sequence): Data to plot, where each element is a sequence of values.
        positions (Sequence): Positions on x-axis where to center each violin.
        width (Number, optional): Width of each violin. Defaults to 0.8.
        color (Optional[str], optional): Fill color of violins. Defaults to None.
        alpha (Optional[Number], optional): Transparency of violins. Defaults to 1.
        edgecolor (Optional[str], optional): Color of violin edges. Defaults to None.
        showextrema (bool, optional): Whether to show min/max lines. Defaults to False.
        showmeans (bool, optional): Whether to show mean lines. Defaults to False.
        showmedians (bool, optional): Whether to show median lines. Defaults to False.
        percentiles (Optional[Sequence], optional): Percentiles to show as lines.
          Defaults to None.
        side (str, optional): Which side of violin to show - "both", "left" or "right".
          Defaults to "both".
        gap (float, optional): Gap between violins when using half violins.
          Defaults to 0.0.
        percentile_style (Optional[Mapping], optional): Style dict for percentile lines.
          Defaults to None.
        median_style (Optional[Mapping], optional): Style dict for median lines.
          Defaults to None.
        ax (Optional[plt.Axes], optional): Axes to plot on. If None, creates new figure.
          Defaults to None.

    Raises:
        ValueError: If side is not one of "both", "left", or "right"

    Returns:
        plt.Axes: The axes containing the violin plot
    """

    # Move positions and shrink widths if we're doing half violins.
    if side == "both":
        offset = 0
    elif side == "left":
        width = width * 2
        offset = -gap / 2
        width = width - gap
    elif side == "right":
        width = width * 2
        offset = gap / 2
        width = width - gap
    else:
        raise ValueError(f"Invalid side: {side}")

    # Handle style info.
    default_median_style = dict(lw=1, color="black", ls="-")
    if median_style:
        default_median_style.update(median_style)
    median_style = default_median_style

    default_percentile_style = dict(lw=1, color="black", ls="--")
    if percentile_style:
        default_percentile_style.update(percentile_style)
    percentile_style = default_percentile_style

    # Handle style info.
    percentiles = [] if percentiles is None else percentiles

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    positions = np.asarray(positions)
    vp = ax.violinplot(
        dataset,
        positions=positions + offset,
        showextrema=showextrema,
        showmeans=showmeans,
        showmedians=False,
        widths=width,
    )

    for i, body in enumerate(vp["bodies"]):
        # Set face- and edge- colors for violins.
        if color is not None:
            body.set_facecolor(color)
            if alpha is not None:
                body.set_alpha(alpha)
        if edgecolor is not None:
            body.set_edgecolor(edgecolor)

        # If half-violins, mask out not-shown half of the violin.
        p = body.get_paths()[0]
        center = positions[i]
        if side == "both":
            limit = center
            half_curve = p.vertices[p.vertices[:, 0] < limit]
        elif side == "left":
            # Mask the right side of the violin.
            limit = center - gap / 2
            p.vertices[:, 0] = np.clip(p.vertices[:, 0], -np.inf, limit)
            half_curve = p.vertices[p.vertices[:, 0] < limit]
        elif side == "right":
            # Mask the left side of the violin.
            limit = center + gap / 2
            p.vertices[:, 0] = np.clip(p.vertices[:, 0], limit, np.inf)
            half_curve = p.vertices[p.vertices[:, 0] > limit]

        # compensation for line width. depends on points-to-data coordinate ratio.
        line_info = [(percentiles, percentile_style)]
        if showmedians:
            line_info.append(([50], median_style))

        lw_factor = 0.01
        for ptiles, style in line_info:
            for q in ptiles:
                y = np.percentile(dataset[i], q)
                if side == "both":
                    x_left = half_curve[np.argmin(np.abs(y - half_curve[:, 1])), 0]
                    x_right = center + abs(center - x_left)
                elif side == "left":
                    x_left = half_curve[np.argmin(np.abs(y - half_curve[:, 1])), 0]
                    x_right = limit
                elif side == "right":
                    x_right = half_curve[np.argmin(np.abs(y - half_curve[:, 1])), 0]
                    x_left = limit
                ln = Line2D([x_left + lw_factor, x_right - lw_factor], [y, y], **style)
                ax.add_line(ln)
    return ax
