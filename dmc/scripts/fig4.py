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
<<<<<<< Updated upstream
import fnmatch
import functools
from typing import Mapping, Optional
=======

import fnmatch
import functools
from typing import Iterable, List, Mapping, Optional
>>>>>>> Stashed changes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_percent_correct,
    load_eval_stats,
)
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


<<<<<<< Updated upstream
specs = [
    {
        "name": "dist_agent_1lm_randrot_noise",
        "group": "half_lms_match",
        "min_n_lms_match": 1,
=======
all_experiments = [
    {
        "name": "dist_agent_1lm_randrot_noise",
        "group": "half_lms_match",
        "min_lms_match": 1,
>>>>>>> Stashed changes
        "n_lms": 1,
    },
    {
        "name": "dist_agent_2lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 1,
=======
        "min_lms_match": 1,
>>>>>>> Stashed changes
        "n_lms": 2,
    },
    {
        "name": "dist_agent_4lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 2,
=======
        "min_lms_match": 2,
>>>>>>> Stashed changes
        "n_lms": 4,
    },
    {
        "name": "dist_agent_8lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 4,
=======
        "min_lms_match": 4,
>>>>>>> Stashed changes
        "n_lms": 8,
    },
    {
        "name": "dist_agent_16lm_half_lms_match_randrot_noise",
        "group": "half_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 8,
=======
        "min_lms_match": 8,
>>>>>>> Stashed changes
        "n_lms": 16,
    },
    {
        "name": "dist_agent_1lm_randrot_noise",
        "group": "fixed_min_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 1,
=======
        "min_lms_match": 1,
>>>>>>> Stashed changes
        "n_lms": 1,
    },
    {
        "name": "dist_agent_2lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 2,
=======
        "min_lms_match": 2,
>>>>>>> Stashed changes
        "n_lms": 2,
    },
    {
        "name": "dist_agent_4lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 2,
=======
        "min_lms_match": 2,
>>>>>>> Stashed changes
        "n_lms": 4,
    },
    {
        "name": "dist_agent_8lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 2,
=======
        "min_lms_match": 4,
>>>>>>> Stashed changes
        "n_lms": 8,
    },
    {
        "name": "dist_agent_16lm_fixed_min_lms_match_randrot_noise",
        "group": "fixed_min_lms_match",
<<<<<<< Updated upstream
        "min_n_lms_match": 2,
        "n_lms": 16,
    },
]
=======
        "min_lms_match": 8,
        "n_lms": 16,
    },
]


def query(
    experiments: Optional[Iterable[Mapping]] = None, get=None, apply=None, **filters
):
    out = experiments if experiments is not None else all_experiments
    for key, val in filters.items():
        out = [obj for obj in out if obj.get(key, None) == val]
    if get:
        out = [obj.get(get) for obj in out]
    if apply:
        out = [apply(obj) for obj in out]
    return out


>>>>>>> Stashed changes
for entry in specs:
    entry["eval_stats"] = load_eval_stats(entry["name"])

db = specs


<<<<<<< Updated upstream
import functools
import operator
from functools import partial
from typing import KeysView, ValuesView


=======
>>>>>>> Stashed changes
def multi(fn):
    """
    Decorator to run a function multiple times and return the results.
    """

    def wrapper(obj, *args, **kwargs):
        if isinstance(obj, dict):
            out = {key: fn(val, *args, **kwargs) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            out = [fn(val, *args, **kwargs) for val in obj]
        else:
            out = fn(obj, *args, **kwargs)
        return out

    return wrapper


<<<<<<< Updated upstream
def query(*args, get=None, apply=None, **kw):
    def filt(entries, key, val):
        if callable(val):
            return [d for d in entries if val(d.get(key, None))]
        return [d for d in entries if d.get(key, None) == val]

    out = args if args else db
    for key, val in kw.items():
        out = filt(out, key, val)
    if get:
        out = [obj.get(get) for obj in out]
    if apply:
        out = [apply(obj) for obj in out]
    return out


=======
>>>>>>> Stashed changes
@multi
def get_attr(obj: Mapping, key: str, default=None):
    return getattr(obj, key, default)


@multi
def get_item(obj: Mapping, key: str, default=None):
    return obj.get(key, default)


@multi
def get_num_steps(df, performance: Optional[str] = None):
    # if performance is None:
    #     sub_df = df
    # else:
    #     tf = [fnmatch.fnmatch(val, performance) for val in df.primary_performance]
    #     sub_df = df[np.array(tf)]
    n_lms = len(df.index.unique())
    obj = df.monty_matching_steps[::n_lms]
    return obj
    # return sub_df.monty_matching_steps


<<<<<<< Updated upstream
@multi
def get_percent_correct(df, performance: str = "correct*"):
    n_matches = len(fnmatch.filter(df.primary_performance, performance))
    return 100 * n_matches / len(df)


=======
def get_num_steps(df, performance: Optional[str] = None):
    # if performance is None:
    #     sub_df = df
    # else:
    #     tf = [fnmatch.fnmatch(val, performance) for val in df.primary_performance]
    #     sub_df = df[np.array(tf)]
    n_lms = len(df.index.unique())
    obj = df.monty_matching_steps[::n_lms]
    return obj


@multi
def get_percent_correct(df: pd.DataFrame, primary_performance: str = "correct*"):
    """Get the percentage of correct performances.

    Args:
        df (pd.DataFrame): The dataframe containing the `primary_performance` column.
        primary_performance (str): Which primary_performance values to count as correct.
            Should be one of:
         - "correct": primary performance must be "correct"
         - "correct_mlh": primary performance must be "correct_mlh"
         - "correct*": primary performance may be "correct" or "correct_mlh".

    Returns:
        float: The percentage of correct performances (between 0 and 100).
    """
    n_rows = len(df)
    value_counts = df.primary_performance.value_counts()
    if primary_performance == "correct":
        return 100 * value_counts["correct"] / n_rows
    elif primary_performance == "correct_mlh":
        return 100 * value_counts["correct_mlh"] / n_rows
    elif primary_performance == "correct*":
        return 100 * (value_counts["correct"] + value_counts["correct_mlh"]) / n_rows
    else:
        raise ValueError(f"Invalid primary_performance: {primary_performance}")


performance_options = [
    "patch_off_object",
    "no_label",
    "pose_time_out",
    "time_out",
    "confused_mlh",
    "correct_mlh",
    "no_match",
    "confused",
    "correct",
]
>>>>>>> Stashed changes
eval_stats = load_eval_stats("dist_agent_8lm_half_lms_match_randrot_noise")

episode = 0
# df = eval_stats[eval_stats.episode == episode]
# ts_step = df["individual_ts_reached_at_step"]
# ts_performance = df["individual_ts_performance"]
df = eval_stats
<<<<<<< Updated upstream
g = df.groupby("episode")
n_episodes = len(g)

primary_perf = g.primary_performance.unique()
ts_perf = g.individual_ts_performance.unique()
# timed_out =
arr = np.array(["time_out"], dtype=object)
=======
groups = eval_stats.groupby("episode")
n_episodes = len(groups)

primary_perf = groups.primary_performance.unique()
ts_perf = groups.individual_ts_performance.unique()
>>>>>>> Stashed changes


result = np.zeros(n_episodes, dtype=object)
time_out = np.zeros(n_episodes, dtype=bool)
<<<<<<< Updated upstream
=======
# Columns for the result dataframe.
columns = {
    "primary_performance": np.zeros(n_episodes, dtype=object),
    "monty_matching_steps": np.zeros(n_episodes, dtype=int),
    "time_out": np.zeros(n_episodes, dtype=bool),
}
use_first = ["monty_matching_steps"]

>>>>>>> Stashed changes
for i in range(n_episodes):
    primary_perf_set = set(primary_perf[i])
    ts_perf_set = set(ts_perf[i])

    if "patch_off_object" in primary_perf_set:
        primary_perf_set.remove("patch_off_object")
    if "patch_off_object" in ts_perf_set:
        ts_perf_set.remove("patch_off_object")

    time_out_i = len(ts_perf_set) == 1 and list(ts_perf_set)[0] == "time_out"
    time_out[i] = time_out_i
    if "correct" in primary_perf_set:
        result[i] = "correct"
        assert not time_out_i
        continue
    if "confused" in primary_perf_set:
        result[i] = "confused"
        assert not time_out_i
        continue
    if len(primary_perf_set) == 1:
        assert time_out_i
        result[i] = list(primary_perf_set)[0]
    elif len(primary_perf_set) == 2:
        assert time_out_i
<<<<<<< Updated upstream
        result[i] = "mixed"  # majority rule?
=======
        result[i] = "mixed"  # majority rule? use min_lms_match?
>>>>>>> Stashed changes
    else:
        raise ValueError(
            f"Unexpected number of primary performances: {len(primary_perf_set)}"
        )

result_df = pd.DataFrame({"result": result, "time_out": time_out})

# a = get_item(lst, "eval_stats")
# df2 = db[]
# names = get_item(query(n_lms=2), "name")


# # Plot accuracy and num steps on the same axes.
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


# Plot accuracy and num steps on separate axes.
# - Prepare data

# groups = [query(group="half_lms_match"), query(group="fixed_min_lms_match")]
<<<<<<< Updated upstream
group_a = [d for d in db if d["group"] == "half_lms_match"]
group_b = [d for d in db if d["group"] == "fixed_min_lms_match"]
groups = [group_a, group_b]
names = ["half_match", "fixed_match"]
colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

data = []
for i, g in enumerate(groups):
    d = {}
    d["name"] = names[i]
    d["eval_stats"] = eval_stats = list(map(lambda obj: obj["eval_stats"], g))
    d["percent_correct"] = [get_percent_correct(df, "correct*") for df in eval_stats]
    d["num_steps"] = [get_num_steps(df, "correct") for df in eval_stats]
    d["conditions"] = list(map(lambda obj: obj["n_lms"], g))
    d["x_positions"] = np.arange(len(g) * 2)[::2] + i
    d["color"] = colors[i]
    data.append(d)


fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Plot accuracy bars
ax = axes[0]
for i, d in enumerate(data):
    ax.bar(
        d["x_positions"],
        d["percent_correct"],
        color=d["color"],
        width=0.8,
    )
ax.set_ylim(0, 100)
ax.set_ylabel("% Correct")
# Put a legend on with labels "half_match" and "fixed_match" and colors
# 'blue' and 'purple'
ax.legend(names, loc="upper right")

# Plot num steps
ax = axes[1]
for i, d in enumerate(data):
    vp = ax.violinplot(
        d["num_steps"],
        positions=d["x_positions"],
        showextrema=False,
        showmedians=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor(d["color"])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")

ax.set_yticks([0, 100, 200, 300, 400, 500])
ax.set_ylim(0, 500)
ax.set_ylabel("Steps")

for ax in axes:
    xticks = np.mean(
        np.vstack([data[0]["x_positions"], data[1]["x_positions"]]), axis=0
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(data[0]["conditions"], ha="center")
    ax.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)

fig.tight_layout()
plt.show()

=======
"""_summary_
"""
pass
# group_a = [d for d in db if d["group"] == "half_lms_match"]
# group_b = [d for d in db if d["group"] == "fixed_min_lms_match"]
# groups = [group_a, group_b]
# names = ["half_match", "fixed_match"]
# colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

# data = []
# for i, g in enumerate(groups):
#     d = {}
#     d["name"] = names[i]
#     d["eval_stats"] = eval_stats = list(map(lambda obj: obj["eval_stats"], g))
#     d["percent_correct"] = [get_percent_correct(df, "correct*") for df in eval_stats]
#     d["num_steps"] = [get_num_steps(df, "correct") for df in eval_stats]
#     d["conditions"] = list(map(lambda obj: obj["n_lms"], g))
#     d["x_positions"] = np.arange(len(g) * 2)[::2] + i
#     d["color"] = colors[i]
#     data.append(d)


# fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# # Plot accuracy bars
# ax = axes[0]
# for i, d in enumerate(data):
#     ax.bar(
#         d["x_positions"],
#         d["percent_correct"],
#         color=d["color"],
#         width=0.8,
#     )
# ax.set_ylim(0, 100)
# ax.set_ylabel("% Correct")
# # Put a legend on with labels "half_match" and "fixed_match" and colors
# # 'blue' and 'purple'
# ax.legend(names, loc="upper right")

# # Plot num steps
# ax = axes[1]
# for i, d in enumerate(data):
#     vp = ax.violinplot(
#         d["num_steps"],
#         positions=d["x_positions"],
#         showextrema=False,
#         showmedians=True,
#     )
#     for body in vp["bodies"]:
#         body.set_facecolor(d["color"])
#         body.set_alpha(1.0)
#     vp["cmedians"].set_color("black")

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
>>>>>>> Stashed changes
