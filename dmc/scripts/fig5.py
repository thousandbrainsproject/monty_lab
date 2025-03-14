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
import os
from numbers import Number
from pathlib import Path
from typing import (
    Container,
    List,
    Optional,
    Union,
)

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

OUT_DIR = DMC_ANALYSIS_DIR / "fig5"
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
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig5_visualize_8lm_patches"
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
        # Plot percent correct.
        accuracies = [exp.get_accuracy() for exp in g]
        ax.bar(
            positions[i],
            accuracies,
            color=colors[i],
            width=width,
            label=labels[i],
        )
        # Plot percent confused but confused and correct were tied.
        percent_ties = []
        for exp in g:
            df = exp.reduced_stats
            is_confused = df.primary_performance == "confused"
            is_tied = is_confused & (df.n_correct == df.n_confused)
            percent_ties.append(100 * is_tied.sum() / len(is_tied))
        ax.bar(
            positions[i],
            percent_ties,
            bottom=accuracies,
            # color="gold",
            color=colors[i],
            alpha=0.3,
            width=width,
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
        n_steps.append([exp.get_n_steps() for exp in g])

    sides = ["left", "right"]
    for i, g in enumerate(groups):
        violinplot(
            n_steps[i],
            xticks,
            width=width,
            gap=gap,
            color=colors[i],
            showmedians=True,
            median_style=dict(color="lightgray"),
            side=sides[i],
            ax=ax,
        )

    # Plot means.
    if showmeans:
        for i, g in enumerate(groups):
            means = [np.mean(arr) for arr in n_steps[i]]
            ax.scatter(
                xticks,
                means,
                color=colors[i],
                marker="o",
                edgecolor="black",
                facecolor="none",
            )
            ax.plot(xticks, means, color="k", linestyle="-", linewidth=3)
            ax.plot(xticks, means, color=colors[i], linestyle="-", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels([exp.n_lms for exp in groups[0]])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    if legend:
        if labels is None:
            labels = ("a", "b")
        legend_handles = []
        for i in range(len(colors)):
            handle = Line2D([0], [0], color=colors[i], lw=4, label=labels[i])
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc=loc, fontsize=8, title="LMs")

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
    accuracies = [exp.get_accuracy() for exp in group]
    ax_1.bar(
        bar_positions,
        accuracies,
        color=colors[0],
        width=width,
    )

    # Plot num steps.
    n_steps = [exp.get_n_steps() for exp in group]
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


def plot_performance_1lm(n_steps_ylim=(0, 500)):
    exp = get_experiments(name="dist_agent_1lm_randrot_noise")[0]
    color = TBP_COLORS["green"]

    fig, axes = plt.subplots(1, 2, figsize=(2, 3))

    # Plot accuracy.
    ax = axes[0]
    accuracies = [exp.get_accuracy()]
    ax.bar(
        [1],
        accuracies,
        color=color,
        width=0.8,
    )
    ax.set_title("Accuracy")
    ax.set_xlabel("Num. LMs")
    ax.set_xlim([0.5, 1.5])
    ax.set_xticks([1])
    ax.set_xticklabels(["1"])
    ax.set_ylabel("% Correct")
    ax.set_ylim([50, 100])

    # Plot num steps.
    ax = axes[1]
    n_steps = [exp.get_n_steps()]
    violinplot(
        n_steps,
        [1],
        width=0.8,
        color=color,
        showmedians=True,
        median_style=dict(color="lightgray"),
        ax=ax,
    )
    ax.scatter(
        [1],
        [n_steps[0].mean()],
        color="black",
        marker="o",
        edgecolor="black",
        facecolor="none",
    )
    ax.set_title("Steps")
    ax.set_xlabel("Num. LMs")
    ax.set_xlim([0.5, 1.5])
    ax.set_xticks([1])
    ax.set_xticklabels(["1"])
    ax.set_ylabel("Steps")
    ax.set_ylim(n_steps_ylim)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "performance_1lm.png", dpi=300)
    fig.savefig(out_dir / "performance_1lm.svg")


def plot_performance_multi_lm(n_steps_ylim=(0, 100)):
    """Make figure where accuracies and num_steps are on separate axes."""
    exp_1lm = get_experiments(name="dist_agent_1lm_randrot_noise")[0]
    half = get_experiments(group="half_lms_match")[1:]
    fixed = get_experiments(group="fixed_min_lms_match")[1:]
    groups = [half, fixed]
    colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # Plot accuracy with horizontal line for 1-LM accuracy.
    ax = axes[0]
    accuracy_1lm = exp_1lm.get_accuracy()
    ax.axhline(
        accuracy_1lm,
        color=TBP_COLORS["green"],
        alpha=1,
        linestyle="--",
        linewidth=1,
    )
    double_accuracy_plot(groups, colors, ylim=(50, 100), ax=ax)

    # Plot num steps with horizontal lines for 1-LM median and mean number of steps.
    ax = axes[1]
    n_steps_1lm = exp_1lm.get_n_steps()
    ax.axhline(
        n_steps_1lm.mean(),
        color=TBP_COLORS["green"],
        alpha=0.75,
        linestyle="--",
        linewidth=1,
    )
    ax.axhline(
        np.median(n_steps_1lm),
        color=TBP_COLORS["green"],
        alpha=0.75,
        linestyle=":",
        linewidth=1,
    )
    double_n_steps_plot(
        groups,
        colors,
        ylim=(0, 100),
        ax=ax,
    )
    ax.set_ylim(n_steps_ylim)

    # Add a legend.
    legend_handles = []
    legend_labels = ["num. LMs / 2", "2"]
    for i in range(len(colors)):
        handle = Line2D([0], [0], color=colors[i], lw=4, label=legend_labels[i])
        legend_handles.append(handle)
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=8, title="Num. LMs Converge"
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "performance_multi_lm.png", dpi=300)
    fig.savefig(out_dir / "performance_multi_lm.svg")

# plot_8lm_patches()
# plot_performance_1lm()
# plot_performance_multi_lm()
