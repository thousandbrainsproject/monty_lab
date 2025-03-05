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
Figure 4: Visualize 8-patch view finder
"""
import copy
import fnmatch
import functools
import os
import shutil
from collections import UserList
from numbers import Number
from pathlib import Path
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_frequency,
    load_eval_stats,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot_utils import TBP_COLORS, axes3d_set_aspect_equal

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

OUT_DIR = DMC_ANALYSIS_DIR / "fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERFORMANCE_OPTIONS = (
    "correct",
    "confused",
    "no_match",
    "correct_mlh",
    "confused_mlh",
    "time_out",
    "pose_time_out",
    "no_label",
    "patch_off_object",
)

def plot_8lm_patches():
    """Plot the 8-SM + view_finder visualization for figure 4.

    Uses data from the experiment `fig4_visualize_8lm_patches` defined in
    `configs/visualizations.py`. This function renders the sensor module
    RGBA data in the scene (in 3D) and overlays the sensor module's patch
    boundaries.

    Creates:
     - $DMC_ANALYSIS_DIR/fig4/8lm_patches.png
     - $DMC_ANALYSIS_DIR/fig4/8lm_patches.svg

    """

    # Load the detailed stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_visualize_8lm_patches"
    detailed_stats_path = experiment_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    def pull(sm_num: int):
        """Helper function for extracting necessary sensor data."""
        sm_dict = stats[f"SM_{sm_num}"]
        # Extract RGBA sensor patch.
        rgba_2d = np.array(sm_dict["raw_observations"][0]["rgba"]) / 255.0
        n_rows, n_cols = rgba_2d.shape[0], rgba_2d.shape[1]

        # Extract locations and on-object filter.
        semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
        pos_1d = semantic_3d[:, 0:3]
        pos_2d = pos_1d.reshape(n_rows, n_cols, 3)
        on_object_1d = semantic_3d[:, 3].astype(int) > 0
        on_object_2d = on_object_1d.reshape(n_rows, n_cols)

        # Filter out points that aren't on-object. Yields a flat list of points/colors.
        return rgba_2d, pos_2d, on_object_2d

    # Create a 3D plot of the semantic point cloud
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=90, azim=-90, roll=0)
    ax.set_proj_type("persp", focal_length=0.125)
    ax.dist = 4.55

    # Render the view finder's RGBA data in the scene.
    rgba_2d, pos_2d, on_object_2d = pull(8)
    rows, cols = np.where(on_object_2d)
    pos_valid_1d = pos_2d[on_object_2d]
    rgba_valid_1d = rgba_2d[on_object_2d]
    ax.scatter(
        pos_valid_1d[:, 0],
        pos_valid_1d[:, 1],
        pos_valid_1d[:, 2],
        c=rgba_valid_1d,
        marker="o",
        alpha=0.3,
        zorder=5,
        s=10,
        edgecolors="none",
    )

    # Render patches and patch boundaries for all sensors.
    for i in range(8):
        # Load sensor data.
        rgba_2d, pos_2d, on_object_2d = pull(i)
        rows, cols = np.where(on_object_2d)
        pos_valid_1d = pos_2d[on_object_2d]
        rgba_valid_1d = rgba_2d[on_object_2d]

        # Render the patch.
        ax.scatter(
            pos_valid_1d[:, 0],
            pos_valid_1d[:, 1],
            pos_valid_1d[:, 2],
            c=rgba_valid_1d,
            marker="o",
            alpha=1,
            zorder=10,
            edgecolors="none",
            s=1,
        )

        # Draw the patch boundaries (complicated).
        n_rows, n_cols = on_object_2d.shape
        row_mid, col_mid = n_rows // 2, n_cols // 2
        n_pix_on_object = on_object_2d.sum()

        if n_pix_on_object == 0:
            contours = []
        elif n_pix_on_object == on_object_2d.size:
            temp = np.zeros((n_rows, n_cols), dtype=bool)
            temp[0, :] = True
            temp[-1, :] = True
            temp[:, 0] = True
            temp[:, -1] = True
            contours = [np.argwhere(temp)]
        else:
            contours = skimage.measure.find_contours(
                on_object_2d, level=0.5, positive_orientation="low"
            )
            contours = [] if contours is None else contours

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
            points_on_object = on_object_2d[ct[:, 0], ct[:, 1]]
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
            xyz = pos_2d[ct[:, 0], ct[:, 1]]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="k", linewidth=3, zorder=20)

    axes3d_set_aspect_equal(ax)
    ax.axis("off")
    plt.show()

    fig.savefig(OUT_DIR / "8lm_patches.png", dpi=300)
    fig.savefig(OUT_DIR / "8lm_patches.svg")

"""
-------------------------------------------------------------------------------
Analysis
"""


class Experiment:
    _eval_stats: Optional[pd.DataFrame] = None
    _reduced_stats: Optional[pd.DataFrame] = None
    _detailed_stats: Optional[DetailedJSONStatsInterface] = None

    def __init__(self, name: os.PathLike, **attrs):
        path = Path(name).expanduser()
        if path.is_dir():
            # Case 1: Given a path to an experiment directory.
            self.path = path
            self.name = path.name
        else:
            # Given a run name. Assume results in DMC results folder.
            self.path = DMC_RESULTS_DIR / name
            self.name = self.path.name
            assert self.path.exists()
        for key, val in attrs.items():
            setattr(self, key, val)

    @property
    def eval_stats(self) -> pd.DataFrame:
        if self._eval_stats is None:
            csv_path = self.path / "eval_stats.csv"
            self._eval_stats = load_eval_stats(csv_path)
        return self._eval_stats

    @eval_stats.setter
    def eval_stats(self, eval_stats: pd.DataFrame):
        self._eval_stats = eval_stats

    @property
    def reduced_stats(self) -> pd.DataFrame:
        if self._reduced_stats is None:
            self._reduced_stats = reduce_eval_stats(self.eval_stats)
        return self._reduced_stats

    @reduced_stats.setter
    def reduced_stats(self, reduced_stats: pd.DataFrame):
        self._reduced_stats = reduced_stats

    @property
    def detailed_stats(self) -> DetailedJSONStatsInterface:
        if self._detailed_stats is None:
            json_path = self.path / "detailed_run_stats.json"
            self._detailed_stats = DetailedJSONStatsInterface(json_path)
        return self._detailed_stats

    @detailed_stats.setter
    def detailed_stats(self, detailed_stats: DetailedJSONStatsInterface):
        self._detailed_stats = detailed_stats

    def copy(self, deep: bool = False) -> "Experiment":
        return copy.deepcopy(self) if deep else copy.copy(self)

    def get_accuracy(
        self,
        primary_performance: Optional[Union[str, Container[str]]] = [
            "correct",
            "correct_mlh",
        ],
    ) -> float:
        return 100 * get_frequency(
            self.reduced_stats["primary_performance"], primary_performance
        )

    def get_n_steps(
        self,
        step_mode: str = "num_steps",
    ) -> np.ndarray:
        if step_mode == "monty_matching_steps":
            return (
                self.eval_stats.groupby("episode").monty_matching_steps.first().values
            )
        elif step_mode == "num_steps":
            return self.eval_stats.num_steps.values
        elif step_mode == "num_steps_terminal":
            # just num_steps for terminal LMs
            terminated = self.eval_stats.primary_performance.isin(
                ["correct", "confused"]
            )
            return self.eval_stats.num_steps[terminated].values
        else:
            raise ValueError(f"Invalid step mode: {step_mode}")

    def __repr__(self):
        return f"Experiment('{self.name}')"


all_experiments = [
    Experiment(
        name="dist_agent_1lm_randrot_noise",
        group="half_lms_match",
        min_lms_match=1,
        n_lms=1,
    ),
    Experiment(
        name="dist_agent_2lm_half_lms_match_randrot_noise",
        group="half_lms_match",
        min_lms_match=1,
        n_lms=2,
    ),
    Experiment(
        name="dist_agent_4lm_half_lms_match_randrot_noise",
        group="half_lms_match",
        min_lms_match=2,
        n_lms=4,
    ),
    Experiment(
        name="dist_agent_8lm_half_lms_match_randrot_noise",
        group="half_lms_match",
        min_lms_match=4,
        n_lms=8,
    ),
    Experiment(
        name="dist_agent_16lm_half_lms_match_randrot_noise",
        group="half_lms_match",
        min_lms_match=8,
        n_lms=16,
    ),
    Experiment(
        name="dist_agent_1lm_randrot_noise",
        group="fixed_min_lms_match",
        min_lms_match=1,
        n_lms=1,
    ),
    Experiment(
        name="dist_agent_2lm_fixed_min_lms_match_randrot_noise",
        group="fixed_min_lms_match",
        min_lms_match=2,
        n_lms=2,
    ),
    Experiment(
        name="dist_agent_4lm_fixed_min_lms_match_randrot_noise",
        group="fixed_min_lms_match",
        min_lms_match=2,
        n_lms=4,
    ),
    Experiment(
        name="dist_agent_8lm_fixed_min_lms_match_randrot_noise",
        group="fixed_min_lms_match",
        min_lms_match=4,
        n_lms=8,
    ),
    Experiment(
        name="dist_agent_16lm_fixed_min_lms_match_randrot_noise",
        group="fixed_min_lms_match",
        min_lms_match=8,
        n_lms=16,
    ),
]


def get_experiments(**filters) -> List[Experiment]:
    experiments = copy.deepcopy(all_experiments)
    for key, val in filters.items():
        experiments = [exp for exp in experiments if getattr(exp, key, None) == val]
    return experiments


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


def reduce_eval_stats(eval_stats: pd.DataFrame, require_majority: bool = True):
    """Reduce the eval stats dataframe to a single row per episode.

    The main purpose of this function is to classify an episode as either "correct"
    or "confused" based on the number of correct and confused performances (or
    "correct_mlh" and "confused_mlh" for timed-out episodes).

    Args:
        eval_stats (pd.DataFrame): The eval stats dataframe.
        require_majority (bool): Whether to require a majority of correct performances
            for 'correct' classification.
    Returns:
        pd.DataFrame: A dataframe with a single row per episode.
    """
    episodes = np.arange(eval_stats.episode.max() + 1)
    assert np.array_equal(eval_stats.episode.unique(), episodes)  # sanity check
    n_episodes = len(episodes)

    # Columns of output dataframe. More are added later.
    output_data = {
        "primary_performance": np.zeros(n_episodes, dtype=object),
    }
    for name in PERFORMANCE_OPTIONS:
        output_data[f"n_{name}"] = np.zeros(n_episodes, dtype=int)

    episode_groups = eval_stats.groupby("episode")
    for episode, df in episode_groups:
        # Find one result given many LM results.
        row = {}

        perf_counts = {key: 0 for key in PERFORMANCE_OPTIONS}
        perf_counts.update(df.primary_performance.value_counts())
        found = []
        for name in PERFORMANCE_OPTIONS:
            row[f"n_{name}"] = perf_counts[name]
            if perf_counts[name] > 0:
                found.append(name)
        performance = found[0]

        # Require a majority of correct performances for 'correct' classification.
        if require_majority:
            if performance == "correct":
                if row["n_confused"] > row["n_correct"]:
                    performance = "confused"
                elif row["n_confused"] < row["n_correct"]:
                    performance = "correct"
                else:
                    # Ties go to "confused" by default, but the tie can be broken
                    # in favor of "correct" if the number of LMs with "correct_mlh"
                    # exceeds the number of LMs with "confused_mlh".
                    performance = "confused"
                    if row["n_correct_mlh"] > row["n_confused_mlh"]:
                        performance = "correct"

            elif performance == "correct_mlh":
                if row["n_confused_mlh"] >= row["n_correct_mlh"]:
                    performance = "confused_mlh"

        row["primary_performance"] = performance

        for key, val in row.items():
            output_data[key][episode] = val

    # Add episode data not specific to the LM.
    output_data["monty_matching_steps"] = (
        episode_groups.monty_matching_steps.first().values
    )
    output_data["primary_target_object"] = (
        episode_groups.primary_target_object.first().values
    )
    output_data["primary_target_rotation"] = (
        episode_groups.primary_target_object.first().values
    )
    output_data["episode"] = episode_groups.episode.first().values
    output_data["epoch"] = episode_groups.epoch.first().values

    out = pd.DataFrame(output_data)
    return out


"""
-------------------------------------------------------------------------------
Plotting
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
) -> plt.Axes:
    """_summary_

    Args:
        dataset (_type_): _description_
        positions (Optional[Sequence], optional): _description_. Defaults to None.
        color (Optional[str], optional): _description_. Defaults to None.
        widths (Optional[Sequence], optional): _description_. Defaults to None.
        side (str, optional): _description_. Defaults to "both".
        showmedians (bool, optional): _description_. Defaults to False.
        percentiles (Optional[Sequence], optional): _description_. Defaults to None.
        edgecolor (Optional[str], optional): _description_. Defaults to None.
        ax (Optional[plt.Axes], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        plt.Axes: _description_
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
        # Modify appearance.
        if color is not None:
            body.set_facecolor(color)
            if alpha is not None:
                body.set_alpha(alpha)
        if edgecolor is not None:
            body.set_edgecolor(edgecolor)

        # If half-violins, mask out not-shown half of the violin.
        # get the center
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


def double_accuracy_plot(
    groups: List[List[Experiment]],
    colors: Container[str] = (TBP_COLORS["blue"], TBP_COLORS["purple"]),
    labels: Optional[Container[str]] = None,
    title: str = "Accuracy",
    xlabel: str = "Num. LMs",
    ylabel: str = "% Correct",
    ylim: Container[Number] = (0, 100),
    gap: float = 0.02,
    width: float = 0.4,
    legend: bool = False,
    ax: Optional[plt.Axes] = None,
    **kw,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=kw.get("figsize", (3, 3)))

    xticks = np.arange(len(groups[0]))
    left_positions = xticks - width / 2 - gap / 2
    right_positions = xticks + width / 2 + gap / 2
    positions = [left_positions, right_positions]
    if not labels:
        labels = ("a", "b")

    for i, g in enumerate(groups):
        accuracies = [exp.get_accuracy(["correct", "correct_mlh"]) for exp in g]
        ax.bar(
            positions[i],
            accuracies,
            color=colors[i],
            width=width,
            label=labels[i],
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels([exp.n_lms for exp in groups[0]])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    if legend:
        legend_kw = {name: kw.get(name) for name in ("loc", "fontsize")}
        ax.legend(labels=labels, **legend_kw)

    return ax


def double_n_steps_plot(
    groups: List[List[Experiment]],
    colors: Container[str] = (TBP_COLORS["blue"], TBP_COLORS["purple"]),
    labels: Optional[Container[str]] = None,
    title: str = "Steps",
    xlabel: str = "Num. LMs",
    ylabel: str = "Steps",
    ylim: Container[Number] = (0, 500),
    gap: float = 0.02,
    width: float = 0.4,
    showmeans: bool = True,
    legend: bool = False,
    loc: Optional[str] = "upper right",
    ax: Optional[plt.Axes] = None,
    **kw,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=kw.get("figsize", (3, 3)))

    xticks = np.arange(len(groups[0]))
    n_steps = []
    for g in groups:
        n_steps.append([exp.get_n_steps("num_steps") for exp in g])

    sides = ["left", "right"]
    alpha = 0.75 if showmeans else 1
    for i, g in enumerate(groups):
        violinplot(
            n_steps[i],
            xticks,
            width=width,
            gap=gap,
            color=colors[i],
            alpha=alpha,
            showmedians=True,
            median_style=dict(color="lightgray"),
            side=sides[i],
            ax=ax,
        )

    # Plot means.
    if showmeans:
        for i, g in enumerate(groups):
            means = [np.mean(arr) for arr in n_steps[i]]
            if sides[i] == "left":
                x_pos = xticks - width / 2 - gap / 2
            else:
                x_pos = xticks + width / 2 + gap / 2
            ax.scatter(x_pos, means, color=colors[i], marker="o", edgecolor="black")
            ax.plot(x_pos, means, color=colors[i], linestyle="-", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels([exp.n_lms for exp in half])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    if legend:
        if labels is None:
            labels = ("a", "b")
        legend_handles = []
        for i in range(len(colors)):
            handle = Line2D([0], [0], color=colors[i], lw=4, label=labels[i])
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc=loc, fontsize=8)

    return ax


def double_accuracy_and_n_steps_plot(
    group: List[Experiment],
    colors: Container[str] = (TBP_COLORS["blue"], TBP_COLORS["purple"]),
    title: Optional[str] = None,
    xlabel: str = "Num. LMs",
    left_ylabel: str = "% Correct",
    left_ylim: Container[Number] = (0, 100),
    right_ylabel: str = "Steps",
    right_ylim: Container[Number] = (0, 500),
    gap: float = 0.02,
    width: float = 0.4,
    ax: Optional[plt.Axes] = None,
    **kw,
) -> plt.Axes:
    """Make figure where accuracies and num_steps are on the same axes."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=kw.get("figsize", (3, 3)))
    ax_1 = ax
    ax_2 = ax_1.twinx()
    xticks = np.arange(len(group))

    # amount of white space between violins
    xticks = np.arange(len(group))
    bar_positions = xticks - width / 2 - gap / 2
    violin_positions = xticks + width / 2 + gap / 2

    # Plot accuracy.
    accuracies = [exp.get_accuracy(["correct", "correct_mlh"]) for exp in group]
    ax_1.bar(
        bar_positions,
        accuracies,
        color=colors[0],
        width=width,
    )

    # Plot num steps.
    n_steps = [exp.get_n_steps("num_steps") for exp in group]
    violinplot(
        n_steps,
        violin_positions,
        width=width,
        color=colors[1],
        showmedians=True,
        median_style=dict(color="lightgray"),
        ax=ax_2,
    )

    ax_1.set_title(title)
    ax_1.set_xlabel(xlabel)
    ax_1.set_xticks(xticks)
    ax_1.set_xticklabels([exp.n_lms for exp in group])
    ax_1.set_ylabel(left_ylabel)
    ax_1.set_ylim(left_ylim)
    ax_2.set_ylabel(right_ylabel)
    ax_2.set_ylim(right_ylim)
    for ax in [ax_1, ax_2]:
        ax.spines["top"].set_visible(False)

    return ax


"""
-------------------------------------------------------------------------------
Figure Functions
"""


def plot_1_and_2_lms():
    lm1 = Experiment(
        name="dist_agent_1lm_randrot_noise",
        group="half_lms_match",
        min_lms_match=1,
        n_lms=1,
    )
    lm2_1 = Experiment(
        name="dist_agent_2lm_half_lms_match_randrot_noise",
        group="half_lms_match",
        min_lms_match=1,
        n_lms=2,
    )
    lm2_2 = Experiment(
        name="dist_agent_2lm_fixed_min_lms_match_randrot_noise",
        group="fixed_min_lms_match",
        min_lms_match=2,
        n_lms=2,
    )

    group = [lm1, lm2_1, lm2_2]
    colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
    labels = ["match: n_lms / 2", "match: 2"]
    sides = ["left", "right"]

    fig, ax_1 = plt.subplots(1, 1, figsize=(6, 4))
    ax_2 = ax_1.twinx()

    # amount of white space between violins
    gap = 0.02
    item_width = 0.4
    xticks = np.arange(3)
    bar_positions = xticks - item_width / 2 - gap / 2
    violin_positions = xticks + item_width / 2 + gap / 2

    # Plot accuracy.
    accuracies_correct = [exp.get_accuracy("correct") for exp in group]
    ax_1.bar(
        bar_positions,
        accuracies_correct,
        color=colors[0],
        width=item_width,
        label="correct",
    )
    bottom = np.array(accuracies_correct)

    accuracies_correct_mlh = [exp.get_accuracy("correct_mlh") for exp in group]
    ax_1.bar(
        bar_positions,
        accuracies_correct_mlh,
        color=colors[0],
        width=item_width,
        bottom=bottom,
        hatch="///",
        label="correct_mlh",
    )
    bottom += accuracies_correct_mlh

    accuracies_confused = [exp.get_accuracy("confused") for exp in group]
    ax_1.bar(
        bar_positions,
        accuracies_confused,
        color="red",
        width=item_width,
        bottom=bottom,
        label="confused",
    )
    bottom += accuracies_confused

    accuracies_confused_mlh = [exp.get_accuracy("confused_mlh") for exp in group]
    ax_1.bar(
        bar_positions,
        accuracies_confused_mlh,
        color="red",
        width=item_width,
        bottom=bottom,
        hatch="///",
        label="confused_mlh",
    )

    # Plot num steps.
    n_steps = [exp.get_n_steps("num_steps") for exp in group]
    violinplot(
        n_steps,
        violin_positions,
        width=item_width,
        color=colors[1],
        alpha=1,
        showmedians=True,
        median_style=dict(lw=1, color="lightgray", ls="-"),
        ax=ax_2,
    )

    ax_1.set_xlabel("Num. LMs : min_lms_match")
    ax_1.set_xticks(xticks)
    ax_1.set_xticklabels(["1 : 1", "2 : 1", "2 : 2"])
    ax_1.set_ylabel("% Correct")
    ax_1.set_ylim(50, 100)
    ax_2.set_ylabel("Steps")
    ax_2.set_ylim(0, 500)

    axes = [ax_1, ax_2]
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.set_xlim([-0.55, 3])
    ax_1.legend()

    # ------------------------------------------------------------------------------
    # Plot num steps for confused  vs correct, all cases.

    correct_group = []
    confused_group = []
    for exp in group:
        eval_stats = exp.eval_stats
        x = eval_stats[eval_stats.primary_performance == "correct"].num_steps
        correct_group.append(x)
        x = eval_stats[eval_stats.primary_performance == "confused"].num_steps
        confused_group.append(x)

    xticks = np.arange(3)
    item_width = 0.4
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = [TBP_COLORS["blue"], "red"]
    violinplot(
        correct_group,
        xticks,
        width=0.4,
        color=colors[0],
        showmedians=True,
        side="left",
        gap=0.01,
        ax=ax,
    )
    violinplot(
        confused_group,
        xticks,
        width=0.4,
        color=colors[1],
        showmedians=True,
        side="right",
        gap=0.01,
        ax=ax,
    )
    ax.set_ylim([0, 500])
    add_legend(ax, colors, labels=["correct", "confused"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1 : 1", "2 : 1", "2 : 2"])
    ax.set_xlabel("Num. LMs : min_lms_match")
    ax.set_ylabel("Steps")
    plt.show()

    # Plot num steps for confused  vs correct, all cases.

    # Let's look at symmetry evidence for confused vs correct.
    correct_group = []
    confused_group = []
    for exp in group:
        eval_stats = exp.eval_stats
        x = eval_stats[eval_stats.primary_performance == "correct"].symmetry_evidence
        correct_group.append(x)
        x = eval_stats[eval_stats.primary_performance == "confused"].symmetry_evidence
        confused_group.append(x)

    xticks = np.arange(3)
    item_width = 0.4
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = [TBP_COLORS["blue"], "red"]
    violinplot(
        correct_group,
        xticks,
        width=0.4,
        color=colors[0],
        showmedians=True,
        showextrema=True,
        side="left",
        gap=0.01,
        ax=ax,
    )
    violinplot(
        confused_group,
        xticks,
        width=0.4,
        color=colors[1],
        showmedians=True,
        showextrema=True,
        side="right",
        gap=0.01,
        ax=ax,
    )
    ax.set_ylim([0, 100])
    add_legend(ax, colors, labels=["correct", "confused"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(["1 : 1", "2 : 1", "2 : 2"])
    ax.set_xlabel("Num. LMs : min_lms_match")
    ax.set_ylabel("Steps")
    plt.show()

    exp = group[2]
    df = exp.reduced_stats[exp.reduced_stats.primary_performance == "confused"]
    n_correct = df.n_correct.values
    n_confused = df.n_confused.values
    confused_eps_with_1_correct = (n_correct == 1).sum() / len(n_correct)
    print(f"confused eps with 1 correct: {100*confused_eps_with_1_correct:.2f}%")

def plot_performance_1lm():
    exp = get_experiments(name="dist_agent_1lm_randrot_noise")[0]
    colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
    fn_kw = {
        "left_ylim": (50, 100),
        "right_ylim": (0, 500),
    }

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    double_accuracy_and_n_steps_plot([exp], colors, ax=ax, **fn_kw)

    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    plt.show()
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "performance_1lm.png", dpi=300)
    fig.savefig(out_dir / "performance_1lm.svg")


def plot_performance_multi_lm_hom():
    """Make figure where accuracies and num_steps are on separate axes."""
    half = get_experiments(group="half_lms_match")[1:]
    fixed = get_experiments(group="fixed_min_lms_match")[1:]
    groups = [half, fixed]
    colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    double_accuracy_plot(groups, colors, ylim=(50, 100), ax=axes[0])
    double_n_steps_plot(groups, colors, ylim=(0, 100), legend=True, ax=axes[1])

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "performance_multi_lm_hom.png", dpi=300)
    fig.savefig(out_dir / "performance_multi_lm_hom.svg")


def plot_performance_multi_lm_het():
    """Make figure where accuracies and num_steps are on separate axes."""
    half = get_experiments(group="half_lms_match")[1:]
    fixed = get_experiments(group="fixed_min_lms_match")[1:]

    colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
    fn_kw = {
        "left_ylim": (50, 100),
        "right_ylim": (0, 100),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    double_accuracy_and_n_steps_plot(
        half, colors, title="Half LMs Match", ax=axes[0], **fn_kw
    )
    double_accuracy_and_n_steps_plot(
        fixed, colors, title="2 LMs Match", ax=axes[1], **fn_kw
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)

    fig.tight_layout()
    plt.show()
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "performance_multi_lm_het.png", dpi=300)
    fig.savefig(out_dir / "performance_multi_lm_het.svg")
