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


# all_experiments = [
#     {
#         "name": "dist_agent_1lm_randrot_noise",
#         "group": "half_lms_match",
#         "min_lms_match": 1,
#         "n_lms": 1,
#     },
#     {
#         "name": "dist_agent_2lm_half_lms_match_randrot_noise",
#         "group": "half_lms_match",
#         "min_lms_match": 1,
#         "n_lms": 2,
#     },
#     {
#         "name": "dist_agent_4lm_half_lms_match_randrot_noise",
#         "group": "half_lms_match",
#         "min_lms_match": 2,
#         "n_lms": 4,
#     },
#     {
#         "name": "dist_agent_8lm_half_lms_match_randrot_noise",
#         "group": "half_lms_match",
#         "min_lms_match": 4,
#         "n_lms": 8,
#     },
#     {
#         "name": "dist_agent_16lm_half_lms_match_randrot_noise",
#         "group": "half_lms_match",
#         "min_lms_match": 8,
#         "n_lms": 16,
#     },
#     {
#         "name": "dist_agent_1lm_randrot_noise",
#         "group": "fixed_min_lms_match",
#         "min_lms_match": 1,
#         "n_lms": 1,
#     },
#     {
#         "name": "dist_agent_2lm_fixed_min_lms_match_randrot_noise",
#         "group": "fixed_min_lms_match",
#         "min_lms_match": 2,
#         "n_lms": 2,
#     },
#     {
#         "name": "dist_agent_4lm_fixed_min_lms_match_randrot_noise",
#         "group": "fixed_min_lms_match",
#         "min_lms_match": 2,
#         "n_lms": 4,
#     },
#     {
#         "name": "dist_agent_8lm_fixed_min_lms_match_randrot_noise",
#         "group": "fixed_min_lms_match",
#         "min_lms_match": 4,
#         "n_lms": 8,
#     },
#     {
#         "name": "dist_agent_16lm_fixed_min_lms_match_randrot_noise",
#         "group": "fixed_min_lms_match",
#         "min_lms_match": 8,
#         "n_lms": 16,
#     },
# ]


class Experiment:
    _eval_stats: Optional[pd.DataFrame] = None
    _reduced_stats: Optional[pd.DataFrame] = None
    _detailed_stats: Optional[DetailedJSONStatsInterface] = None
    correct_result: Union[str, Container[str]] = "correct"

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

    def set_attrs(self, **attrs):
        for key, val in attrs.items():
            setattr(self, key, val)

    def get_accuracy(
        self, primary_performance: Optional[Union[str, Container[str]]] = None
    ) -> float:
        result = (
            self.correct_result if primary_performance is None else primary_performance
        )
        return 100 * get_frequency(self.reduced_stats["primary_performance"], result)

    def get_n_steps(
        self, result: Optional[Union[str, Container[str]]] = None
    ) -> pd.Series:
        result = self.correct_result if result is None else result
        result = np.atleast_1d(result)
        df = self.eval_stats[self.eval_stats.primary_performance.isin(result)]
        return df.num_steps

    def __repr__(self):
        return f"Experiment('{self.name}')"


class ExperimentGroup(UserList):
    def __init__(
        self, experiments: List[Experiment], name: Optional[str] = None, **attrs
    ):
        super().__init__(experiments)
        for key, val in attrs.items():
            setattr(self, key, val)
        self.name = name

    def copy(self, deep: bool = False) -> "ExperimentGroup":
        return copy.deepcopy(self) if deep else copy.copy(self)

    def set_attrs(self, **attrs):
        for key, val in attrs.items():
            setattr(self, key, val)

    def map(self, fn: Union[Callable, str], *args, **kw) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        if isinstance(fn, (str, np.str_)):
            out = [getattr(exp, fn)(*args, **kw) for exp in self]
        else:
            out = [fn(exp, *args, **kw) for exp in self]
        try:
            return np.array(out)
        except ValueError:
            return np.array(out, dtype=object)

    def __getitem__(self, key: int) -> Experiment:
        out = self.data[key]
        if isinstance(out, list):
            return ExperimentGroup(out)
        return out

    def __repr__(self):
        if self.name:
            s = f"ExperimentGroup('{self.name}')"
        else:
            s = f"ExperimentGroup"
        for exp in self:
            s += f"\n  - {exp}"
        return s


all_experiments = ExperimentGroup(
    [
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
)


def get_experiments(**filters) -> ExperimentGroup:
    experiments = copy.deepcopy(all_experiments)
    for key, val in filters.items():
        experiments = [exp for exp in experiments if getattr(exp, key, None) == val]
    return ExperimentGroup(experiments)


def reduce_eval_stats(eval_stats: pd.DataFrame, require_majority: bool = True):
    """_summary_

    Args:
        eval_stats (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    episodes = np.arange(eval_stats.episode.max() + 1)
    assert np.array_equal(eval_stats.episode.unique(), episodes)  # sanity check
    n_episodes = len(episodes)

    # Columns of output dataframe.
    output_data = {
        "primary_performance": np.zeros(n_episodes, dtype=object),
        "n_steps": np.zeros(n_episodes, dtype=int),
    }
    for name in PERFORMANCE_OPTIONS:
        output_data[f"n_{name}"] = np.zeros(n_episodes, dtype=int)

    # temporary
    output_data["eval_stats_start"] = np.zeros(n_episodes, dtype=int)
    output_data["eval_stats_end"] = np.zeros(n_episodes, dtype=int)

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
                if row["n_confused"] >= row["n_correct"]:
                    performance = "confused"
            elif performance == "correct_mlh":
                if row["n_confused_mlh"] >= row["n_correct_mlh"]:
                    performance = "confused_mlh"

        row["primary_performance"] = performance
        # Choose number of steps taken.
        lm_inds = np.where(df.primary_performance == performance)[0]
        n_steps = df.num_steps.iloc[lm_inds].mean()
        row["n_steps"] = n_steps

        # temporary
        row["eval_stats_start"] = df.index[0]
        row["eval_stats_end"] = df.index[-1] + 1

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
    n_correct, n_confused = output_data["n_correct"], output_data["n_confused"]
    single_hypothesis_correct = (n_correct > 0) & (n_confused == 0)
    single_hypothesis_confused = (n_correct == 0) & (n_confused > 0)
    single_hypothesis = single_hypothesis_correct | single_hypothesis_confused
    output_data["single_hypothesis"] = single_hypothesis
    output_data["episode"] = episode_groups.episode.first()

    out = pd.DataFrame(output_data)
    return out


def get_accuracy(
    reduced_stats: pd.DataFrame, primary_performance: Union[str, List[str]] = "correct"
) -> float:
    """Get the percentage of correct performances.

    Args:
        reduced_stats (pd.DataFrame): The dataframe containing the `result` column.
        result: (str or list of str): One or more result types (e.g., `"correct"` or
            `["correct", "correct_mlh"]`).
    Returns:
        float: The percentage of correct performances (between 0 and 100).
    """
    return 100 * get_frequency(
        reduced_stats["primary_performance"], primary_performance
    )


def get_num_steps(
    reduced_stats: pd.DataFrame, primary_performance: Union[str, List[str]] = "correct"
) -> pd.Series:
    """Get the percentage of correct performances.

    Args:
        df (pd.DataFrame): The reduced eval stats dataframe containing. Must contain
         columns `result` and `n_steps`.
        result: (str or list of str): One or more result types (e.g., `"correct"` or
            `["correct", "correct_mlh"]`).
    Returns:
        pd.Series: The number of steps taken for each episode.
    """
    sub_df = reduced_stats[
        reduced_stats["primary_performance"].isin(primary_performance)
    ]
    return sub_df.n_steps

"""
-------------------------------------------------------------------------------
TEMPORARY / EXPLORATORY
"""


def temp_fn():
    pass
    # for i, grp in enumerate(groups):
    #     num_steps = [get_num_steps(dct["summary"], correct_result) for dct in grp]
    #     mean_steps = [np.mean(arr) for arr in num_steps]
    #     median_steps = [np.median(arr) for arr in num_steps]
    #     ax.plot(median_steps, color=group_colors[i], label=group_names[i] + " Median")
    #     ax.plot(mean_steps, color=group_colors[i], label=group_names[i] + " Mean", ls="--")
    # xticklabels = [str(dct["n_lms"]) for dct in groups[0]]
    # ax.set_xticks(np.arange(len(xticklabels)))
    # ax.set_xticklabels(xticklabels, ha="center")
    # ax.set_xlabel("Number of LMs")
    # ax.set_ylabel("Matching Steps")
    # ax.legend(loc="upper right")
    # plt.show()

    # ax.set_yticks([0, 100, 200, 300, 400, 500])
    # ax.set_ylim(0, 500)
    # ax.set_ylabel("Steps")

    # for ax in axes:
    #     xticks = np.mean(
    #         np.vstack([data[0]["x_positions"], data[1]["x_positions"]]), axis=0
    #     )
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(data[0]["conditions"], ha="center")
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["top"].set_visible(False)

    # fig.tight_layout()
    # plt.show()

    # # Plot accuracy and num steps on the same axes
    # fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    # for i, (group_name, group_dict) in enumerate(groups.items()):
    #     # Aggregate data
    #     dataframes = [dct["eval_stats"] for dct in group_dict.values()]
    #     n_lms_list = [dct["n_lms"] for dct in group_dict.values()]
    #     percent_correct_arrays = [get_percent_correct(df) for df in dataframes]
    #     num_steps_arrays = [df.num_steps for df in dataframes]

    #     x_positions = np.arange(len(group_dict) * 2)
    #     x_positions_1 = x_positions[::2]
    #     x_positions_2 = x_positions[1::2]

    #     ax_1 = axes[i]
    #     ax_2 = ax_1.twinx()
    #     # Plot accuracy bars
    #     ax_1.bar(
    #         x_positions_1,
    #         percent_correct_arrays,
    #         color=TBP_COLORS["blue"],
    #         width=0.8,
    #     )
    #     ax_1.set_ylim(0, 100)
    #     ax_1.set_ylabel("% Correct")

    #     # Plot num steps
    #     vp = ax_2.violinplot(
    #         num_steps_arrays,
    #         positions=x_positions_2,
    #         showextrema=False,
    #         showmedians=True,
    #     )
    #     for body in vp["bodies"]:
    #         body.set_facecolor(TBP_COLORS["purple"])
    #         body.set_alpha(1.0)
    #     vp["cmedians"].set_color("black")
    #     ax_2.set_yticks([0, 100, 200, 300, 400, 500])
    #     ax_2.set_ylim(0, 500)
    #     ax_2.set_ylabel("Steps")

    #     xticks = np.mean(np.vstack([x_positions_1, x_positions_2]), axis=0)
    #     ax_1.set_xticks(xticks)
    #     ax_1.set_xticklabels(n_lms_list, ha="center")

    #     ax_1.spines["top"].set_visible(False)
    #     ax_2.spines["top"].set_visible(False)
    #     ax_1.set_title(group_name)
    #     fig.tight_layout()

    # plt.show()


def plot_accuracy_and_num_steps():
    # Plot accuracy and num steps on separate axes.
    # - Prepare data

    half = get_experiments(group="half_lms_match")
    fixed = get_experiments(group="fixed_min_lms_match")
    groups = [half, fixed]
    group_names = ["half_match", "fixed_match"]
    group_colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

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
        accuracy = [get_accuracy(dct["reduced_stats"], correct_result) for dct in grp]
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
        num_steps = [get_num_steps(dct["reduced_stats"], correct_result) for dct in grp]

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


def save_flipped_eval_stats():
    group = get_experiments()

    for exp in group:
        eval_stats_raw = pd.read_csv(exp.path / "eval_stats.csv")
        eval_stats, reduced_stats = exp.eval_stats, exp.reduced_stats
        n_correct = reduced_stats.n_correct
        n_confused = reduced_stats.n_confused
        flipped = (n_correct > 0) & (n_confused >= n_correct)
        flipped_episodes = reduced_stats[flipped].episode
        n_flipped = len(flipped_episodes)

        n_correct_mlh = reduced_stats.n_correct_mlh
        n_confused_mlh = reduced_stats.n_confused_mlh
        flipped_mlh = (n_correct_mlh > 0) & (n_confused_mlh >= n_correct_mlh)
        flipped_mlh_episodes = reduced_stats[flipped_mlh].episode
        n_flipped_mlh = len(flipped_mlh_episodes)

        all_flipped = np.concatenate([flipped_episodes, flipped_mlh_episodes])
        n_episodes = len(reduced_stats)

        print(f"{exp.name}: {n_episodes} episodes")
        print(f" - flipped: {n_flipped} / {n_episodes}")
        print(f" - flipped_mlh: {n_flipped_mlh} / {n_episodes}")
        print(f" - pct flipped: {100*len(all_flipped) / n_episodes:.2f}%")
        lst = []
        if len(all_flipped):
            empty = pd.DataFrame({col: [""] for col in eval_stats_raw.columns})
            for episode in all_flipped:
                df = eval_stats[eval_stats.episode == episode]
                start, stop = df.index[0], df.index[-1] + 1
                chunk = eval_stats_raw.iloc[start:stop]
                lst.append(chunk)
                lst.append(empty)

            lst = lst[:-1]
            out_dir = OUT_DIR / "flipped/eval_stats_chunks"
            out_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_dir / f"{exp.name}_chunks.csv"
            combined = pd.concat(lst)
            combined.to_csv(out_path, index=False)

            out_dir = OUT_DIR / "flipped/eval_stats"
            out_dir.mkdir(exist_ok=True, parents=True)
            src = exp.path / "eval_stats.csv"
            dst = out_dir / f"{exp.name}.csv"
            shutil.copy(src, dst)


def plot_flipped_eval_stats():
    out_dir = OUT_DIR / "flipped/plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Initialize groups
    correct_result = ["correct", "correct_mlh"]
    for correct_result in ["correct", "correct_mlh"]:
        half = get_experiments(group="half_lms_match")
        fixed = get_experiments(group="fixed_min_lms_match")
        for exp in half + fixed:
            exp.correct_result = correct_result
            exp.reduced_stats = reduce_eval_stats(exp.eval_stats, require_majority=True)

        half_no_majority = get_experiments(group="half_lms_match")
        fixed_no_majority = get_experiments(group="fixed_min_lms_match")
        for exp in half_no_majority + fixed_no_majority:
            exp.correct_result = correct_result
            exp.reduced_stats = reduce_eval_stats(
                exp.eval_stats, require_majority=False
            )

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
        colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
        kw = {
            "colors": colors,
            "ylim": (50, 100),
        }

        groups = [half_no_majority, half]
        labels = ["corr > 0", "corr > conf"]
        title = "Half LMs Match"
        plot_accuracy(groups, labels=labels, title=title, ax=axes[0], **kw)

        groups = [fixed_no_majority, fixed]
        title = "Fixed LMs Match"
        plot_accuracy(
            groups, labels=labels, title=title, legend=False, ax=axes[1], **kw
        )
        if "correct_mlh" in correct_result:
            fig.savefig(out_dir / "majority_vs_no_majority_mlh.png", dpi=300)
        else:
            fig.savefig(out_dir / "majority_vs_no_majority.png", dpi=300)


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


def add_legend(
    ax: plt.Axes,
    groups: Iterable[Experiment],
    colors: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    loc: Optional[str] = None,
    lw: int = 4,
) -> matplotlib.legend.Legend:
    # Create custom legend handles (regular matplotlib legend doesn't work with
    # two violin plots -- both patches end up with the same color).
    colors = [g.color for g in groups] if colors is None else colors
    labels = [g.label for g in groups] if labels is None else labels
    legend_handles = []
    for i, g in enumerate(groups):
        handle = Line2D([0], [0], color=colors[i], lw=lw, label=labels[i])
        legend_handles.append(handle)

    return ax.legend(handles=legend_handles, loc=loc, fontsize=8)


def plot_accuracy(
    groups: Sequence[ExperimentGroup],
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

# half = get_experiments(group="half_lms_match")
# fixed = get_experiments(group="fixed_min_lms_match")
# common = [half[0], half[2]]
# for ind in [2, 0]:
#     for g in [half, fixed]:
#         g.pop(ind)

# groups = [common, half, fixed]
# colors = [TBP_COLORS["green"], TBP_COLORS["blue"], TBP_COLORS["purple"]]
# labels = ["half=fixed", "match: n_lms / 2", "match: 2"]

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# # Plot accuracy
# big_spacing = 2
# small_spacing = 0.85
# x_positions_common = np.array([0, 4])
# x_positions_half = np.array([2, 6, 8])
# x_positions_fixed = x_positions_half + small_spacing
# x_positions = [x_positions_common, x_positions_half, x_positions_fixed]
# widths = [1.6, 0.8, 0.8]
# for i, g in enumerate(groups):
#     x_pos = x_positions[i].tolist()
#     accuracy = [exp.get_accuracy() for exp in g]
#     ax.bar(
#         x_pos,
#         accuracy,
#         color=colors[i],
#         width=widths[i],
#         align="edge",
#     )

# ax.set_ylim([50, 100])
# ax.set_xlabel("Number of LMs")
# ax.set_ylabel("% Correct")
# ax.legend()
# plt.show()

"""
Accuracy / Bar plot
"""
# half = get_experiments(group="half_lms_match")
# fixed = get_experiments(group="fixed_min_lms_match")

# groups = [half, fixed]
# colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]
# labels = ["match: n_lms / 2", "match: 2"]

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# # Plot accuracy
# n_groups = len(groups)
# width = 0.4
# gap = 0.025
# x_positions_0 = np.arange(len(groups[0])) + 1
# x_positions_1 = x_positions_0 + width + gap
# x_positions = np.vstack([x_positions_0, x_positions_1])
# for i, g in enumerate(groups):
#     x_pos = x_positions[i].tolist()
#     accuracy = [exp.get_accuracy() for exp in g]
#     ax.bar(
#         x_pos,
#         accuracy,
#         color=colors[i],
#         width=width,
#         align="edge",
#         label=labels[i],
#     )

# ax.set_xlabel("Number of LMs")
# ax.set_xticks(x_positions[0])
# ax.set_xticklabels(["1", "2", "4", "8", "16"])

# ax.set_ylabel("% Correct")
# ax.set_ylim([50, 100])

# # legend = add_legend(ax, groups, colors=colors, labels=labels)
# ax.legend(loc="upper left")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)


# plt.show()
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
    # ax.set_xticks(x_positions[0])
    # ax.set_xticklabels(["1", "2", "4", "8", "16"])

    ax.set_ylabel("% Correct")
    ax.set_ylim([0, 150])

    legend = add_legend(ax, groups, colors=colors, labels=labels)
    # ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


plot_double_violin(step_mode="monty_matching_steps")
plot_double_violin(step_mode="num_steps")
plot_double_violin(step_mode="num_steps_terminal")
