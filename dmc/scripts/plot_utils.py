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
from typing import Any, List, Optional

import matplotlib.axes
import numpy as np
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


def violinplot(
    ax: matplotlib.axes.Axes,
    data: List,
    conditions: List[str],
    rotation: Number = 0,
    color: Optional[Any] = None,
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
    if color is not None:
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax.set_xticks(list(range(1, len(conditions) + 1)))
    ax.set_xticklabels(conditions, rotation=rotation, ha="right")
