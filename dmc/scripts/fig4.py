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

import fnmatch
import functools
from typing import Any, Iterable, List, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
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


# half_lms_match = [
#     "dist_agent_1lm_randrot_noise",
#     "dist_agent_2lm_half_lms_match_randrot_noise",
#     "dist_agent_4lm_half_lms_match_randrot_noise",
#     "dist_agent_8lm_half_lms_match_randrot_noise",
#     "dist_agent_16lm_half_lms_match_randrot_noise",
# ]

# fixed_min_lms_match = [
#     "dist_agent_1lm_randrot_noise",
#     "dist_agent_2lm_fixed_min_lms_match_randrot_noise",
#     "dist_agent_4lm_fixed_min_lms_match_randrot_noise",
#     "dist_agent_8lm_fixed_min_lms_match_randrot_noise",
#     "dist_agent_16lm_fixed_min_lms_match_randrot_noise",
# ]
# experiment_names = {
#     "half_lms_match": half_lms_match,
#     "fixed_min_lms_match": fixed_min_lms_match,
# }

# experiments = {}
# for name in half_lms_match:
#     df = load_eval_stats(name)
#     df.attrs["name"] = name

#     experiments[name] = load_eval_stats(name)


all_experiments = [
    {
        "name": "dist_agent_1lm_randrot_noise",
        "group": "half_lms_match",
        "min_lms_match": 1,
        "n_lms": 1,
    },
    {
        "name": "dist_agent_2lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
        "min_lms_match": 1,
        "n_lms": 2,
    },
    {
        "name": "dist_agent_4lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
        "min_lms_match": 2,
        "n_lms": 4,
    },
    {
        "name": "dist_agent_8lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
        "min_lms_match": 4,
        "n_lms": 8,
    },
    {
        "name": "dist_agent_16lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
        "min_lms_match": 8,
        "n_lms": 16,
    },
    {
        "name": "dist_agent_1lm_randrot_noise",
        "group": "fixed_min_lms_match",
        "min_lms_match": 1,
        "n_lms": 1,
    },
    {
        "name": "dist_agent_2lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
        "min_lms_match": 2,
        "n_lms": 2,
    },
    {
        "name": "dist_agent_4lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
        "min_lms_match": 2,
        "n_lms": 4,
    },
    {
        "name": "dist_agent_8lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
        "min_lms_match": 4,
        "n_lms": 8,
    },
    {
        "name": "dist_agent_16lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
        "min_lms_match": 8,
        "n_lms": 16,
    },
]


def get_experiments(load: bool = True, **filters) -> List[Mapping]:
    experiments = all_experiments
    for key, val in filters.items():
        experiments = [dct for dct in experiments if dct.get(key, None) == val]
    if load:
        for dct in experiments:
            dct["eval_stats"] = load_eval_stats(dct["name"])
            dct["summary"] = reduce_eval_stats(dct["eval_stats"])
    return experiments


def reduce_eval_stats(eval_stats: pd.DataFrame):
    """_summary_

    Args:
        eval_stats (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    n_episodes = eval_stats.episode.max() + 1
    episodes = np.arange(n_episodes)
    assert np.array_equal(eval_stats.episode.unique(), episodes)  # sanity check

    # Initialize output data.
    summary = {
        "result": np.zeros(n_episodes, dtype=object),
        "monty_matching_steps": np.zeros(n_episodes, dtype=int),
        "time_out": np.zeros(n_episodes, dtype=bool),
        "n_steps": np.zeros(n_episodes, dtype=int),
    }
    performance_options = [
        "correct",
        "confused",
        "no_match",
        "correct_mlh",
        "confused_mlh",
        "time_out",
        "pose_time_out",
        "no_label",
        "patch_off_object",
    ]
    for name in performance_options:
        summary[f"n_{name}"] = np.zeros(n_episodes, dtype=int)

    for episode in episodes:
        df = eval_stats[eval_stats.episode == episode]

        # Find one result given many LM results.
        perf_counts = {key: 0 for key in performance_options}
        perf_counts.update(df.primary_performance.value_counts())
        found = []
        for name in performance_options:
            summary[f"n_{name}"][episode] = perf_counts[name]
            if perf_counts[name] > 0:
                found.append(name)

        result = found[0]

        # Require a majority of correct performances for 'correct' classification.
        if result == "correct":
            if perf_counts["confused"] >= perf_counts["correct"]:
                result = "confused"

        # Choose number of steps taken.
        lm_inds = np.where(df.primary_performance == result)[0]
        n_steps = df.num_steps.iloc[lm_inds].mean()
        summary["n_steps"][episode] = n_steps

        summary["result"][episode] = result
        summary["result"][episode] = result
        summary["time_out"][episode] = (
            perf_counts["correct"] + perf_counts["confused"]
        ) == 0

    # Add episode data not specific to the LM.
    groups = eval_stats.groupby("episode")
    summary["monty_matching_steps"] = groups.monty_matching_steps.first().values
    summary["primary_target_object"] = groups.primary_target_object.first().values

    n_correct, n_confused = summary["n_correct"], summary["n_confused"]
    summary["mixed"] = (n_correct > 0) & (n_confused > 0)

    return pd.DataFrame(summary)


def get_num_steps(
    df: pd.DataFrame,
    result: Optional[Union[str, List[str]]] = None,
) -> pd.Series:
    """Get the monty steps"""
    if result is None:
        return df.monty_matching_steps
    result = [result] if isinstance(result, (str, np.str_)) else result
    matches = df.result.isin(result)
    sub_df = df[matches]
    return sub_df.monty_matching_steps


def get_frequency(items: Iterable, match: Union[Any, List[Any]]) -> float:
    """Get the fraction of values that belong to a collection of values.

    Args:
        items (iterable): The list of items.
        match: (scalar or list of scalars): One or more values to match against
          (e.g., `"correct"` or `["correct", "correct_mlh"]`).
    Returns:
        float: The frequency that values in `items` belong to `match`.
    """
    s = items if isinstance(items, pd.Series) else pd.Series(items)
    match = np.atleast_1d(match)
    value_counts = dict(s.value_counts())
    n_matching = sum([value_counts.get(val, 0) for val in match])
    return n_matching / len(s)


def get_accuracy(df: pd.DataFrame, result: Union[str, List[str]] = "correct") -> float:
    """Get the percentage of correct performances.

    Args:
        df (pd.DataFrame): The dataframe containing the `result` column.
        result: (str or list of str): One or more result types (e.g., `"correct"` or
            `["correct", "correct_mlh"]`).
    Returns:
        float: The percentage of correct performances (between 0 and 100).
    """
    n_rows = len(df)
    value_counts = df["result"].value_counts()
    result = [result] if isinstance(result, (str, np.str_)) else result
    count = sum([value_counts[res] for res in result])
    return 100 * count / n_rows


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
    accuracy = [get_accuracy(dct["summary"], correct_result) for dct in grp]
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
    # num_steps = [get_num_steps(dct["summary"], correct_result) for dct in grp]
    num_steps = []
    for dct in grp:
        _df = dct["summary"]
        _df = _df[_df.result.isin(correct_result)]
        num_steps.append(_df.n_steps)

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
# plt.show()

lst1 = [get_num_steps(dct["summary"], ["correct"]) for dct in grp]
lst2 = [get_num_steps(dct["summary"], ["correct", "correct_mlh"]) for dct in grp]
for i in range(len(lst1)):
    print(np.median(lst1[i]), np.median(lst2[i]))
    # print(lst1[i].mean(), lst2[i].mean())
# fig, ax = plt.subplots(1, 1, figsize=(6, 4))

"""
NOTES

For deciding num steps:
 - Should I find the smallest number of steps taken by a terminating LM?


"""

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
pass