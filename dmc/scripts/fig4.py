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
import matplotlib.patches as patches
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
from plot_utils import TBP_COLORS, axes3d_set_aspect_equal, violinplot

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


def plot_accuracy_and_num_steps():
    # Plot accuracy and num steps on separate axes.
    # - Prepare data

    half = get_experiments(group="half_lms_match")
    fixed = get_experiments(group="fixed_min_lms_match")
    groups = [half, fixed]
    group_names = ["half_match", "fixed_match"]
    group_colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Plot accuracy
    ax = axes[0]
    big_spacing = 2
    small_spacing = 0.85
    x_positions_0 = big_spacing * np.arange(len(groups[0]))
    x_positions_1 = x_positions_0 + small_spacing
    x_positions = np.vstack([x_positions_0, x_positions_1])
    for i, grp in enumerate(groups):
        x_pos = x_positions[i].tolist()
        accuracy = [exp.get_accuracy() for exp in grp]
        ax.bar(
            x_pos,
            accuracy,
            color=group_colors[i],
            width=0.8,
        )
    xticks = np.mean(x_positions, axis=0)
    xticklabels = [str(dct["n_lms"]) for dct in groups[0]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha="center")
    ax.set_ylim(50, 100)
    ax.set_ylabel("% Correct")

    # Plot num steps
    ax = axes[1]

    big_spacing = 2
    small_spacing = 0.6
    x_positions_0 = big_spacing * np.arange(len(groups[0]))
    x_positions_1 = x_positions_0 + small_spacing
    x_positions = np.vstack([x_positions_0, x_positions_1])
    for i, grp in enumerate(groups):
        x_pos = x_positions[i].tolist()
        num_steps = [exp.get_n_steps() for exp in grp]

        vp = ax.violinplot(
            num_steps,
            positions=x_pos,
            showextrema=False,
            showmedians=True,
        )
        for body in vp["bodies"]:
            body.set_facecolor(group_colors[i])
            body.set_alpha(1.0)
        vp["cmedians"].set_color("black")
    xticks = np.mean(x_positions, axis=0)
    xticklabels = [str(dct["n_lms"]) for dct in groups[0]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha="center")
    ax.set_ylim([0, 200])
    ax.set_ylabel("Matching Steps")

    for ax in axes:
        ax.set_xlabel("Number of LMs")

    # Create custom legend handles (regular matplotlib legend doesn't work with
    # two violin plots -- both patches end up with the same color).
    legend_handles = [
        Line2D([0], [0], color=color, lw=4, label=label)
        for label, color in zip(group_names, group_colors)
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    plt.show()


def plot_acc():
    correct_result = ["correct", "correct_mlh"]

    half = get_experiments(group="half_lms_match")
    half.set_attrs(
        name="half_lms_match",
        label="match: n_lms / 2",
        color=TBP_COLORS["blue"],
        correct_result=correct_result,
    )

    fixed = get_experiments(group="fixed_min_lms_match")
    fixed.set_attrs(
        name="fixed_min_lms_match",
        label="match: 2",
        color=TBP_COLORS["purple"],
        correct_result=correct_result,
    )

    exp = half[0]

    groups = [half, fixed]

    # num steps conditions
    correct_result = ["correct", "correct_mlh"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Plot accuracy
    ax = axes[0]
    big_spacing = 2
    small_spacing = 0.85
    x_positions_0 = big_spacing * np.arange(len(groups[0]))
    x_positions_1 = x_positions_0 + small_spacing
    x_positions = np.vstack([x_positions_0, x_positions_1])
    for i, grp in enumerate(groups):
        x_pos = x_positions[i].tolist()
        accuracy = [exp.get_accuracy() for exp in grp]
        ax.bar(
            x_pos,
            accuracy,
            color=grp.color,
            width=0.8,
        )
    xticks = np.mean(x_positions, axis=0)
    xticklabels = [str(exp.n_lms) for exp in groups[0]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha="center")
    ax.set_ylim(50, 100)
    ax.set_ylabel("% Correct")

    # Plot num steps
    ax = axes[1]

    big_spacing = 2
    small_spacing = 0.6
    x_positions_0 = big_spacing * np.arange(len(groups[0]))
    x_positions_1 = x_positions_0 + small_spacing
    x_positions = np.vstack([x_positions_0, x_positions_1])
    for i, g in enumerate(groups):
        x_pos = x_positions[i].tolist()
        num_steps = [exp.get_n_steps() for exp in g]

        vp = ax.violinplot(
            num_steps,
            positions=x_pos,
            showextrema=False,
            showmedians=True,
        )
        for body in vp["bodies"]:
            body.set_facecolor(g.color)
            body.set_alpha(1.0)
        vp["cmedians"].set_color("black")
    xticks = np.mean(x_positions, axis=0)
    xticklabels = [str(exp.n_lms) for exp in groups[0]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha="center")
    ax.set_ylim([0, 200])
    ax.set_ylabel("Matching Steps")

    for ax in axes:
        ax.set_xlabel("Number of LMs")

    # Create custom legend handles (regular matplotlib legend doesn't work with
    # two violin plots -- both patches end up with the same color).
    group_names = [g.label for g in groups]
    group_colors = [g.color for g in groups]
    legend_handles = [
        Line2D([0], [0], color=color, lw=4, label=label)
        for label, color in zip(group_names, group_colors)
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    plt.show()


def plot_accuracy(
    groups: Sequence[Experiment],
    colors: Optional[Sequence] = None,
    labels: Optional[Sequence[str]] = None,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    **kw,
) -> plt.Axes:
    """Make a double-bar plot of accuracy for two groups of experiments.

    Args:
        groups (_type_): _description_
        colors (_type_, optional): _description_. Defaults to None.
        labels (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        plt.Axes: _description_
    """
    if ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1, figsize=kw.get("figsize", None))

    if colors is None:
        colors = [g.color for g in groups]
    if labels is None:
        labels = [g.label for g in groups]

    # Plot accuracy
    big_spacing = 2
    small_spacing = 0.85
    x_positions_0 = big_spacing * np.arange(len(groups[0]))
    x_positions_1 = x_positions_0 + small_spacing
    x_positions = np.vstack([x_positions_0, x_positions_1])
    for i, g in enumerate(groups):
        x_pos = x_positions[i].tolist()
        accuracy = [exp.get_accuracy() for exp in g]
        ax.bar(
            x_pos,
            accuracy,
            color=colors[i],
            width=0.8,
        )
    # x-axis
    defaults = {
        # x-axis
        "xlabel": "Number of LMs",
        "xlim": None,
        "xticks": np.mean(x_positions, axis=0),
        "xticklabels": [str(exp.n_lms) for exp in groups[0]],
        # y-axis
        "ylabel": "% Correct",
        "ylim": (0, 100),
        "yticks": None,
        "yticklabels": None,
        # etc.
        "title": None,
    }
    for key in defaults.keys():
        if key in kw:
            val = kw[key]
        else:
            val = defaults.get(key)
        if val is not None:
            getattr(ax, f"set_{key}")(val)

    if legend:
        legend_kw = {name: kw.get(name) for name in ("loc", "lw") if name in kw}
        add_legend(ax, groups, colors=colors, labels=labels, **legend_kw)

    return ax


def plot_double_violin(step_mode: str = "num_steps_terminal"):
    """
    Num Steps

    step_mode: one of
    - "monty_matching_steps"
    - "num_steps"
    - "num_steps_terminal"
    """

    half = get_experiments(group="half_lms_match")
    fixed = get_experiments(group="fixed_min_lms_match")

    groups = [half, fixed]
    colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
    labels = ["match: n_lms / 2", "match: 2"]
    sides = ["left", "right"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # amount of white space between violins
    inter_width = 0.02

    # middle positions for each half-violin pair
    x_positions = np.arange(len(groups[0]))
    for group_num, g in enumerate(groups):
        n_steps = []
        for exp in g:
            if step_mode == "monty_matching_steps":
                n_steps_j = exp.eval_stats.groupby(
                    "episode"
                ).monty_matching_steps.first()
                n_steps.append(n_steps_j)
            elif step_mode == "num_steps":
                n_steps_j = exp.eval_stats.num_steps
                n_steps.append(n_steps_j)
            elif step_mode == "num_steps_terminal":
                # just num_steps for terminal LMs
                terminated = exp.eval_stats.primary_performance.isin(
                    ["correct", "confused"]
                )
                n_steps_j = exp.eval_stats.num_steps[terminated]
                n_steps.append(n_steps_j)
            else:
                raise ValueError(f"Invalid step mode: {step_mode}")

        # Plot num steps
        vp = ax.violinplot(
            n_steps,
            positions=x_positions,
            showextrema=False,
            showmedians=False,
            widths=0.8,
        )
        for j, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[group_num])
            body.set_alpha(1.0)

            # 1. Mask out not-shown half of the violin to make a half-violin.
            # 2. Draw a line for the median that fits within the half-violin.

            # get the center
            p = body.get_paths()[0]
            center_x = x_positions[j]
            median = np.median(n_steps[j])
            if sides[group_num] == "left":
                # Mask the right side of the violin.
                right_max = center_x - inter_width / 2
                p.vertices[:, 0] = np.clip(p.vertices[:, 0], -np.inf, right_max)
                # find leftmost x-value of violin curve where y is the median
                curve_verts = p.vertices[p.vertices[:, 0] < right_max]
                imin = np.argmin(np.abs(median - curve_verts[:, 1]))
                left_max = curve_verts[imin, 0]
            elif sides[group_num] == "right":
                # Mask the left side of the violin.
                left_max = center_x + inter_width / 2
                p.vertices[:, 0] = np.clip(p.vertices[:, 0], left_max, np.inf)
                # find rightmost n curve where y is the median
                curve_verts = p.vertices[p.vertices[:, 0] > left_max]
                imin = np.argmin(np.abs(median - curve_verts[:, 1]))
                right_max = curve_verts[imin, 0]
            else:
                raise ValueError(f"Invalid side: {sides[group_num]}")

            # compensation for line width. depends on points-to-data coordinate ratio.
            lw_factor = 0.01
            ax.plot(
                [left_max + lw_factor, right_max - lw_factor],
                [median, median],
                color="black",
            )

    ax.set_title(step_mode)
    ax.set_xlabel("Number of LMs")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["1", "2", "4", "8", "16"])

    ax.set_ylabel("Steps")
    ax.set_ylim([0, 150])

    add_legend(ax, groups, colors=colors, labels=labels)
    # ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


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
inter_width = 0.02
item_width = 0.4
xticks = np.arange(3)
bar_positions = xticks - item_width / 2 - inter_width / 2
violin_positions = xticks + item_width / 2 + inter_width / 2

# Plot accuracy.
accuracies_correct = [exp.get_accuracy("correct") for exp in group]
accuracies_correct_mlh = [exp.get_accuracy("correct_mlh") for exp in group]
ax_1.bar(
    bar_positions,
    accuracies_correct,
    color=colors[0],
    width=item_width,
)
ax_1.bar(
    bar_positions,
    accuracies_correct_mlh,
    color=colors[0],
    width=item_width,
    bottom=accuracies_correct,
    hatch="///",
)

# Plot num steps.
n_steps = [exp.get_n_steps("num_steps") for exp in group]
vp = ax_2.violinplot(
    n_steps,
    positions=violin_positions,
    showextrema=False,
    showmedians=True,
    widths=item_width,
)
for j, body in enumerate(vp["bodies"]):
    body.set_facecolor(colors[1])
    body.set_alpha(1.0)

ax_1.set_xlabel("Min. LMs Match / Num. LMs")
ax_1.set_xticks(xticks)
ax_1.set_xticklabels(["1 / 1", "1 / 2", "2 / 2"])
ax_1.set_ylabel("% Correct")
ax_1.set_ylim(50, 100)
ax_2.set_ylabel("Steps")
ax_2.set_ylim(0, 500)

axes = [ax_1, ax_2]
for ax in axes:
    ax.spines["top"].set_visible(False)


plt.show()
# groups = [half, fixed]
# colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
# labels = ["match: n_lms / 2", "match: 2"]
# sides = ["left", "right"]

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# # amount of white space between violins
# inter_width = 0.02

# # middle positions for each half-violin pair
# x_positions = np.arange(len(groups[0]))
# for group_num, g in enumerate(groups):
#     n_steps = []
#     for exp in g:
#         if step_mode == "monty_matching_steps":
#             n_steps_j = exp.eval_stats.groupby("episode").monty_matching_steps.first()
#             n_steps.append(n_steps_j)
#         elif step_mode == "num_steps":
#             n_steps_j = exp.eval_stats.num_steps
#             n_steps.append(n_steps_j)
#         elif step_mode == "num_steps_terminal":
#             # just num_steps for terminal LMs
#             terminated = exp.eval_stats.primary_performance.isin(
#                 ["correct", "confused"]
#             )
#             n_steps_j = exp.eval_stats.num_steps[terminated]
#             n_steps.append(n_steps_j)
#         else:
#             raise ValueError(f"Invalid step mode: {step_mode}")

#     # Plot num steps
#     vp = ax.violinplot(
#         n_steps,
#         positions=x_positions,
#         showextrema=False,
#         showmedians=False,
#         widths=0.8,
#     )
#     for j, body in enumerate(vp["bodies"]):
#         body.set_facecolor(colors[group_num])
#         body.set_alpha(1.0)

#         # 1. Mask out not-shown half of the violin to make a half-violin.
#         # 2. Draw a line for the median that fits within the half-violin.

#         # get the center
#         p = body.get_paths()[0]
#         center_x = x_positions[j]
#         median = np.median(n_steps[j])
#         if sides[group_num] == "left":
#             # Mask the right side of the violin.
#             right_max = center_x - inter_width / 2
#             p.vertices[:, 0] = np.clip(p.vertices[:, 0], -np.inf, right_max)
#             # find leftmost x-value of violin curve where y is the median
#             curve_verts = p.vertices[p.vertices[:, 0] < right_max]
#             imin = np.argmin(np.abs(median - curve_verts[:, 1]))
#             left_max = curve_verts[imin, 0]
#         elif sides[group_num] == "right":
#             # Mask the left side of the violin.
#             left_max = center_x + inter_width / 2
#             p.vertices[:, 0] = np.clip(p.vertices[:, 0], left_max, np.inf)
#             # find rightmost n curve where y is the median
#             curve_verts = p.vertices[p.vertices[:, 0] > left_max]
#             imin = np.argmin(np.abs(median - curve_verts[:, 1]))
#             right_max = curve_verts[imin, 0]
#         else:
#             raise ValueError(f"Invalid side: {sides[group_num]}")

#         # compensation for line width. depends on points-to-data coordinate ratio.
#         lw_factor = 0.01
#         ax.plot(
#             [left_max + lw_factor, right_max - lw_factor],
#             [median, median],
#             color="black",
#         )

# ax.set_title(step_mode)
# ax.set_xlabel("Number of LMs")
# ax.set_xticks(x_positions)
# ax.set_xticklabels(["1", "2", "4", "8", "16"])

# ax.set_ylabel("Steps")
# ax.set_ylim([0, 150])

# legend = add_legend(ax, groups, colors=colors, labels=labels)
# # ax.legend(loc="upper left")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.show()
