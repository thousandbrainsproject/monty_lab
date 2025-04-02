# Copyright 2025 Thousand Brains Project
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
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Color palette for TBP.
TBP_COLORS = {
    "black": "#000000",
    "blue": "#00A0DF",
    "pink": "#F737BD",
    "purple": "#5D11BF",
    "green": "#008E42",
    "yellow": "#FFBE31",
}

"""
Style and Formatting Utilities
-------------------------------------------------------------------------------
"""


def init_matplotlib_style():
    """Initialize matplotlib plotting style."""
    style = {
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes3d.grid": True,
        "axes3d.xaxis.panecolor": (0.9, 0.9, 0.9),
        "axes3d.yaxis.panecolor": (0.875, 0.875, 0.875),
        "axes3d.zaxis.panecolor": (0.85, 0.85, 0.85),
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "font.size": 8,
        "grid.color": "white",
        "legend.fontsize": 8,
        "legend.framealpha": 1.0,
        "legend.handlelength": 0.75,
        "legend.title_fontsize": 10,
        "svg.fonttype": "none",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    mpl.rcParams.update(style)


def extract_style(dct: Dict[str, Any], prefix: str, strip: bool = True) -> Mapping:
    """Extract a subset of a dictionary with keys that start with a given prefix."""
    prefix = prefix + "." if not prefix.endswith(".") else prefix
    if strip:
        return {
            k.replace(prefix, ""): v for k, v in dct.items() if k.startswith(prefix)
        }
    else:
        return {k: v for k, v in dct.items() if k.startswith(prefix)}


def update_style(
    base: Optional[Dict[str, Any]],
    new: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Join two dictionaries of style properties.

    If a key is present in both dictionaries, the value from the new dictionary is used.
    If a key is present in only one dictionary, the value from that dictionary is used.

    """
    base = {} if base is None else base
    new = {} if new is None else new
    return {**base, **new}


"""
3D Plotting Utilities
-------------------------------------------------------------------------------
"""


def axes3d_clean(ax: Axes3D, grid: bool = True) -> None:
    """Remove clutter from 3D axes.

    Args:
        ax: The 3D axes to clean.
        grid: Whether to show the background grid. Default is True.
    """

    # Remove dark spines that outline the plot.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((1, 1, 1, 0))

    # Remove axis labels.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_label(None)

    if grid:
        # Remove tick marks. This method keeps the grid lines visible while
        # making the little nubs that stick out invisible. (Setting xticks=[] removes
        # grid lines).
        for axis in ("x", "y", "z"):
            ax.tick_params(axis=axis, colors=(0, 0, 0, 0))

    else:
        # Remove tick marks.
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([])

        ax.grid(False)


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


"""
Image Utilities
-------------------------------------------------------------------------------
"""


def add_solid_background(
    image: np.ndarray,
    color: str,
) -> np.ndarray:
    """
    Add a solid background to an RGBA image.
    """
    width, height = image.shape[0], image.shape[1]
    c = np.array(to_rgba(color))
    bg = np.zeros([width, height, 4])
    bg[:, :] = c
    return blend_rgba_images(bg, image)


def add_gradient_background(
    image: np.ndarray,
    vmin: float = 0.7,
    vmax: float = 0.9,
) -> np.ndarray:
    """Add a grayscale gradient background to an RGBA image."""

    width, height = image.shape[0], image.shape[1]

    # First, create the gradient background.
    # - Make pixel coordinates.
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    # - Compute the randomly oriented gradient.
    theta = np.random.uniform(0, 2 * np.pi)
    gradient = (X * np.cos(theta) + Y * np.sin(theta)) / np.sqrt(width**2 + height**2)

    # Scale gradient to desired range, and put in an RGBA array.
    gradient = vmin + (vmax - vmin) * gradient[..., np.newaxis]
    bg = np.clip(gradient, vmin, vmax)
    bg = np.dstack((bg, bg, bg, np.ones((width, height))))

    # - Finally, blend the image with the background.
    return blend_rgba_images(bg, image)


def blend_rgba_images(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Blends two RGBA image arrays using alpha compositing.

    Args:
        background: An RGBA array.
        foreground: An RGBA array.

    Returns:
    - Blended image as an RGBA NumPy array.
    """
    assert background.shape == foreground.shape, "Images must have the same shape"
    assert background.shape[2] == 4, "Images must be RGBA (H, W, 4)"
    image_1, image_2 = foreground, background

    # Ensure 0-1 floats.
    if image_1.max() > 1:
        image_1 = image_1 / 255.0
    if image_2.max() > 1:
        image_2 = image_2 / 255.0

    # Extract RGB and Alpha channels
    rgb_1, alpha_1 = image_1[..., :3], image_1[..., 3:]
    rgb_2, alpha_2 = image_2[..., :3], image_2[..., 3:]

    # Compute blended alpha
    alpha_out = alpha_1 + alpha_2 * (1 - alpha_1)

    # Compute blended RGB
    rgb_out = (rgb_1 * alpha_1 + rgb_2 * alpha_2 * (1 - alpha_1)) / np.maximum(
        alpha_out, 1e-8
    )

    # Stack RGB and alpha back together
    return np.dstack((rgb_out, alpha_out))


"""
Plotting Utilities
-------------------------------------------------------------------------------
"""


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
    **kw,
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
        **kw: Additional keyword arguments to pass to the matplotlib's violinplot
          function.
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
        **kw,
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

        line_info = [(percentiles, percentile_style)]
        if showmedians:
            line_info.append(([50], median_style))

        lw_factor = 0.01  # compensation for line width.
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


class SensorModuleData:
    """Class for plotting sensor module data on 3D axes.

    Args:
        sm_dict (Mapping): Dictionary containing sensor module data. Comes from
          detailed JSON stats dictionary.
        style (Optional[Mapping], optional): Style dictionary in the same format as
          `_default_style`. Key/Value pairs will override the default style.

    """
    _default_style = {
        # for "plot_raw_observation"
        "raw_observation.contour.color": "black",
        "raw_observation.contour.alpha": 1,
        "raw_observation.contour.linewidth": 1,
        "raw_observation.contour.zorder": 20,
        "raw_observation.scatter.color": "rgba",  # 'rgba' mean use patch color
        "raw_observation.scatter.alpha": 1,
        "raw_observation.scatter.edgecolor": "none",
        "raw_observation.scatter.s": 1,
        "raw_observation.scatter.zorder": 10,
        # for "plot_sensor_path"
        "sensor_path.start.color": "black",
        "sensor_path.start.alpha": 1,
        "sensor_path.start.marker": "x",
        "sensor_path.start.s": 15,
        "sensor_path.start.zorder": 10,
        "sensor_path.scatter.color": "black",
        "sensor_path.scatter.alpha": 1,
        "sensor_path.scatter.marker": "v",
        "sensor_path.scatter.s": 10,
        "sensor_path.scatter.zorder": 10,
        "sensor_path.scatter.edgecolor": "none",
        "sensor_path.line.color": TBP_COLORS["blue"],
        "sensor_path.line.alpha": 1,
        "sensor_path.line.linewidth": 1,
        "sensor_path.line.zorder": 10,
    }

    def __init__(
        self,
        sm_dict: Mapping,
        style: Optional[Mapping] = None,
    ):
        self.sm_dict = sm_dict
        self.raw_observations = sm_dict.get("raw_observations", None)
        self.processed_observations = sm_dict.get("processed_observations", None)
        self.sm_properties = sm_dict.get("sm_properties", None)

        self.style = self._default_style.copy()
        self.update_style(style)

    def update_style(self, style: Optional[Mapping]) -> None:
        self.style = update_style(self.style, style)

    def get_processed_observation(self, step: int) -> Mapping:
        obs = self.processed_observations[step]
        obs["location"] = np.array(obs["location"])
        return obs

    def get_raw_observation(self, step: int) -> Mapping:
        rgba = np.array(self.raw_observations[step]["rgba"]) / 255.0
        n_rows, n_cols = rgba.shape[0], rgba.shape[1]

        # Extract locations and on-object filter.
        semantic_3d = np.array(self.sm_dict["raw_observations"][step]["semantic_3d"])
        pos_1d = semantic_3d[:, 0:3]
        pos = pos_1d.reshape(n_rows, n_cols, 3)
        on_object_1d = semantic_3d[:, 3].astype(int) > 0
        on_object = on_object_1d.reshape(n_rows, n_cols)
        return {
            "rgba": rgba,
            "pos": pos,
            "on_object": on_object,
        }

    def plot_raw_observation(
        self,
        ax: plt.Axes,
        step: int,
        scatter: bool = True,
        contour: bool = True,
        style: Optional[Mapping] = None,
    ):
        """Plot the raw observation (i.e, an RGBA patch).

        Args:
            ax (plt.Axes): The axes to plot on.
            step (int): The step to plot.
            scatter (bool): Whether to plot the scatter.
            contour (bool): Whether to plot the contour.
        """
        style = update_style(self.style, style)
        obs = self.get_raw_observation(step)
        rgba = obs["rgba"]
        pos = obs["pos"]
        on_object = obs["on_object"]
        pos_valid_1d = pos[on_object]
        rgba_valid_1d = rgba[on_object]

        if scatter:
            scatter_style = extract_style(style, "raw_observation.scatter")
            if scatter_style["color"] == "rgba":
                scatter_style["color"] = rgba_valid_1d
            print(scatter_style)
            ax.scatter(
                pos_valid_1d[:, 0],
                pos_valid_1d[:, 1],
                pos_valid_1d[:, 2],
                **scatter_style,
            )

        if contour:
            contour_style = extract_style(style, "raw_observation.contour")
            contours = self._find_patch_contours(pos, on_object)
            for xyz in contours:
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], **contour_style)

    def plot_sensor_path(
        self,
        ax: plt.Axes,
        steps: Optional[Union[int, slice]] = None,
        start: bool = True,
        scatter: bool = True,
        line: bool = True,
        style: Optional[Mapping] = None,
    ):
        """Plot the raw observation.

        Args:
            ax (plt.Axes): The axes to plot on.
            step (int): The step to plot.
            scatter (bool): Whether to plot the scatter.
            contour (bool): Whether to plot the contour.
        """
        style = update_style(self.style, style)
        all_steps = np.arange(len(self))
        if steps is None:
            steps = slice(None)
        elif isinstance(steps, (int, np.integer)):
            steps = slice(0, steps)
        assert isinstance(steps, slice)
        steps = all_steps[steps]

        locations = []
        for step in steps:
            locations.append(self.get_processed_observation(step)["location"])
        locations = np.array(locations)
        scatter_locations = line_locations = locations
        if start:
            start_style = extract_style(style, "sensor_path.start")
            ax.scatter(
                scatter_locations[0, 0],
                scatter_locations[0, 1],
                scatter_locations[0, 2],
                **start_style,
            )
            scatter_locations = scatter_locations[1:]

        if scatter:
            scatter_style = extract_style(style, "sensor_path.scatter")
            ax.scatter(
                scatter_locations[:, 0],
                scatter_locations[:, 1],
                scatter_locations[:, 2],
                **scatter_style,
            )
        if line:
            line_style = extract_style(style, "sensor_path.line")
            ax.plot(
                line_locations[:, 0],
                line_locations[:, 1],
                line_locations[:, 2],
                **line_style,
            )

    def _find_patch_contours(
        self, pos: np.ndarray, on_object: np.ndarray
    ) -> List[np.ndarray]:
        n_rows, n_cols = on_object.shape
        row_mid, col_mid = n_rows // 2, n_cols // 2
        n_pix_on_object = on_object.sum()
        if n_pix_on_object == 0:
            contours = []
        elif n_pix_on_object == on_object.size:
            temp = np.zeros((n_rows, n_cols), dtype=bool)
            temp[0, :] = True
            temp[-1, :] = True
            temp[:, 0] = True
            temp[:, -1] = True
            contours = [np.argwhere(temp)]
        else:
            contours = skimage.measure.find_contours(
                on_object, level=0.5, positive_orientation="low"
            )
            contours = [] if contours is None else contours

        xyz_list = []
        for ct in contours:
            row_mid, col_mid = n_rows // 2, n_cols // 2

            # Contour may be floating point (fractional indices from scipy). If so,
            # round rows/columns towards the center of the patch.
            if not np.issubdtype(ct.dtype, np.integer):
                # Round towards the center.
                rows, cols = ct[:, 0], ct[:, 1]
                rows_new, cols_new = np.zeros_like(rows), np.zeros_like(cols)
                rows_new[rows >= row_mid] = np.floor(rows[rows >= row_mid])
                rows_new[rows < row_mid] = np.ceil(rows[rows < row_mid])
                cols_new[cols >= col_mid] = np.floor(cols[cols >= col_mid])
                cols_new[cols < col_mid] = np.ceil(cols[cols < col_mid])
                ct_new = np.zeros_like(ct, dtype=int)
                ct_new[:, 0] = rows_new.astype(int)
                ct_new[:, 1] = cols_new.astype(int)
                ct = ct_new

            # Drop any points that happen to be off-object (it's possible that
            # some boundary points got rounded off-object).
            points_on_object = on_object[ct[:, 0], ct[:, 1]]
            ct = ct[points_on_object]

            # In order to plot the boundary as a line, we need the points to
            # be in order. We can order them by associating each point with its
            # angle from the center of the patch. This isn't a general solution,
            # but it works here.
            Y, X = row_mid - ct[:, 0], ct[:, 1] - col_mid  # pixel to X/Y coords.
            theta = np.arctan2(Y, X)
            sort_order = np.argsort(theta)
            ct = ct[sort_order]

            # Finally, plot the contour.
            xyz = pos[ct[:, 0], ct[:, 1]]
            xyz_list.append(xyz)
        return xyz_list

    def __len__(self) -> int:
        if self.raw_observations:
            return len(self.raw_observations)
        elif self.processed_observations:
            return len(self.processed_observations)
        elif self.sm_properties:
            return len(self.sm_properties)
        else:
            return 0
