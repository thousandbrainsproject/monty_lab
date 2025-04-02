# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 5.

Panel B: 8-patch view finder
 - `plot_8lm_patches()`

Panel C: Accuracy
 - `plot_accuracy()`

Panel C: Steps
 - `plot_steps()`

Running the above functions requires that the following experiments have been run:
 - `fig5_visualize_8lm_patches`
 - `dist_agent_1lm_randrot_noise`
 - `dist_agent_2lm_half_lms_match_randrot_noise`
 - `dist_agent_4lm_half_lms_match_randrot_noise`
 - `dist_agent_8lm_half_lms_match_randrot_noise`
 - `dist_agent_16lm_half_lms_match_randrot_noise`
 - `dist_agent_2lm_fixed_min_lms_match_randrot_noise`
 - `dist_agent_4lm_fixed_min_lms_match_randrot_noise`
 - `dist_agent_8lm_fixed_min_lms_match_randrot_noise`
 - `dist_agent_16lm_fixed_min_lms_match_randrot_noise`
"""

from typing import Tuple

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
from plot_utils import (
    TBP_COLORS,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
    violinplot,
)

init_matplotlib_style()

OUT_DIR = DMC_ANALYSIS_DIR / "fig5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment names
ONE_LM_EXPERIMENT = "dist_agent_1lm_randrot_noise"
HALF_LMS_MATCH_EXPERIMENTS = (
    "dist_agent_2lm_half_lms_match_randrot_noise",
    "dist_agent_4lm_half_lms_match_randrot_noise",
    "dist_agent_8lm_half_lms_match_randrot_noise",
    "dist_agent_16lm_half_lms_match_randrot_noise",
)

FIXED_MIN_LMS_MATCH_EXPERIMENTS = (
    "dist_agent_2lm_fixed_min_lms_match_randrot_noise",
    "dist_agent_4lm_fixed_min_lms_match_randrot_noise",
    "dist_agent_8lm_fixed_min_lms_match_randrot_noise",
    "dist_agent_16lm_fixed_min_lms_match_randrot_noise",
)

"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""


def reduce_eval_stats(eval_stats: pd.DataFrame) -> pd.DataFrame:
    """Reduce the eval stats dataframe to a single row per episode.

    The main purpose of this function is to classify an episode as either "correct"
    or "confused" based on the number of correct and confused performances (or
    "correct_mlh" and "confused_mlh" for timed-out episodes).

    Args:
        eval_stats: The eval stats dataframe.

    Returns:
        pd.DataFrame: A dataframe with a single row per episode.
    """
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


def get_accuracy(experiment: str) -> Tuple[float, float]:
    """Get the percent correct and percent LMS tied for an experiment."""
    eval_stats = load_eval_stats(experiment)
    reduced_stats = reduce_eval_stats(eval_stats)
    percent_correct = 100 * get_frequency(
        reduced_stats["primary_performance"], ("correct", "correct_mlh")
    )
    is_confused = reduced_stats.primary_performance == "confused"
    is_tied = is_confused & (reduced_stats.n_correct == reduced_stats.n_confused)
    percent_tied = 100 * is_tied.sum() / len(is_tied)
    return percent_correct, percent_tied


def get_n_steps(experiment: str) -> np.ndarray:
    """Get the number of steps taken across all LMs for an experiment."""
    eval_stats = load_eval_stats(experiment)
    return eval_stats.num_steps.values


"""
--------------------------------------------------------------------------------
Panel B: 8-patch view finder
--------------------------------------------------------------------------------
"""


def plot_8lm_patches():
    """Plot the 8-SM + view_finder visualization for figure 4.

    Uses data from the experiment `fig4_visualize_8lm_patches` defined in
    `configs/visualizations.py`. This function renders the sensor module
    RGBA data in the scene (in 3D) and overlays the sensor module's patch
    boundaries.

    Output is saved to `DMC_ANALYSIS_DIR/fig5/8lm_patches`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "8lm_patches"
    out_dir.mkdir(exist_ok=True, parents=True)

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

    fig.savefig(out_dir / "8lm_patches.png")
    fig.savefig(out_dir / "8lm_patches.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel C: Accuracy
--------------------------------------------------------------------------------
"""


def plot_accuracy():
    """Plot accuracy of 1-LM and multi-LM experiments.

    Requires the following experiments to have been run:
    - `dist_agent_1lm_randrot_noise`
    - `dist_agent_2lm_half_lms_match_randrot_noise`
    - `dist_agent_4lm_half_lms_match_randrot_noise`
    - `dist_agent_8lm_half_lms_match_randrot_noise`
    - `dist_agent_16lm_half_lms_match_randrot_noise`
    - `dist_agent_2lm_fixed_min_lms_match_randrot_noise`
    - `dist_agent_4lm_fixed_min_lms_match_randrot_noise`
    - `dist_agent_8lm_fixed_min_lms_match_randrot_noise`
    - `dist_agent_16lm_fixed_min_lms_match_randrot_noise`

    Output is saved to `DMC_ANALYSIS_DIR/fig5/performance`.

    """

    # Initialize output directory.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)

    groups = [HALF_LMS_MATCH_EXPERIMENTS, FIXED_MIN_LMS_MATCH_EXPERIMENTS]
    one_lm_color = TBP_COLORS["green"]
    group_colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3), sharex=True)
    top_ax, bottom_ax = axes
    fig.subplots_adjust(hspace=0.05)

    # Plot params.
    ylims = [(0, 25), (75, 100)]
    half_bar_width = 0.4
    gap = 0.02
    xticks = np.arange(5)
    left_positions = xticks[1:] - half_bar_width / 2 - gap / 2
    right_positions = xticks[1:] + half_bar_width / 2 + gap / 2
    positions = [left_positions, right_positions]

    # 1-LM first.
    percent_correct_1lm, _ = get_accuracy(ONE_LM_EXPERIMENT)
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.bar(
            xticks[0],
            [percent_correct_1lm],
            color=one_lm_color,
            width=2 * half_bar_width,
        )

    # Multi-LM.
    for i, g in enumerate(groups):
        # Plot percent correct.
        accuracies = [get_accuracy(exp) for exp in g]
        percent_correct = [acc[0] for acc in accuracies]
        percent_tied = [acc[1] for acc in accuracies]

        for ax_num, ax in enumerate([bottom_ax, top_ax]):
            # Plot percent correct.
            ax.bar(
                positions[i],
                percent_correct,
                color=group_colors[i],
                width=half_bar_width,
            )
            # Plot percent confused but confused and correct were tied.
            ax.bar(
                positions[i],
                percent_tied,
                bottom=percent_correct,
                color=group_colors[i],
                alpha=0.3,
                width=half_bar_width,
            )

    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.set_ylim(ylims[ax_num])

    # Sets parameters for both x-axes (they're shared, so removing ticks for the
    # top plot removes ticks for the bottom plot).
    bottom_ax.set_xlabel("Num. LMs")
    bottom_ax.set_xticks(xticks)
    bottom_ax.set_xticklabels(["1", "2", "4", "8", "16"])

    bottom_ax.set_ylabel("% Correct")
    bottom_ax.set_yticks([0, 10, 20])
    top_ax.spines.bottom.set_visible(False)
    top_ax.set_yticks([80, 90, 100])

    # Draw y-axis divider markers.
    marker_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    top_ax.plot([0], [0], transform=top_ax.transAxes, **marker_kwargs)
    bottom_ax.plot([0], [1], transform=bottom_ax.transAxes, **marker_kwargs)

    fig.savefig(out_dir / "accuracy.png")
    fig.savefig(out_dir / "accuracy.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel C: Steps
--------------------------------------------------------------------------------
"""


def plot_steps():
    """Plot the number of steps taken by 1-LM and multi-LM experiments.

    Requires the following experiments to have been run:
    - `dist_agent_1lm_randrot_noise`
    - `dist_agent_2lm_half_lms_match_randrot_noise`
    - `dist_agent_4lm_half_lms_match_randrot_noise`
    - `dist_agent_8lm_half_lms_match_randrot_noise`
    - `dist_agent_16lm_half_lms_match_randrot_noise`
    - `dist_agent_2lm_fixed_min_lms_match_randrot_noise`
    - `dist_agent_4lm_fixed_min_lms_match_randrot_noise`
    - `dist_agent_8lm_fixed_min_lms_match_randrot_noise`
    - `dist_agent_16lm_fixed_min_lms_match_randrot_noise`

    Output is saved to `DMC_ANALYSIS_DIR/fig5/performance`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)

    groups = [HALF_LMS_MATCH_EXPERIMENTS, FIXED_MIN_LMS_MATCH_EXPERIMENTS]
    one_lm_color = TBP_COLORS["green"]
    group_colors = [TBP_COLORS["blue"], TBP_COLORS["purple"]]

    fig = plt.figure(figsize=(3.4, 3))
    gs = fig.add_gridspec(3, 1)  # 3 rows, bottom plot will take 2 rows

    # Create the two subplots with shared x-axis
    top_ax = fig.add_subplot(gs[0, 0])  # Top subplot takes 1/3
    bottom_ax = fig.add_subplot(gs[1:, 0], sharex=top_ax)  # Bottom subplot takes 2/3

    # fig, axes = plt.subplots(2, 1, figsize=(3.4, 3), sharex=True)
    # top_ax, bottom_ax = axes
    fig.subplots_adjust(hspace=0.05)

    # Plot params.
    ylims = [(0, 110), (440, 500)]
    yticks = [
        [0, 25, 50, 75, 100],
        [450, 475, 500],
    ]
    half_bar_width = 0.4
    xticks = np.arange(5)
    sides = ["left", "right"]
    bw_method = 0.1

    # 1-LM
    n_steps_1lm = get_n_steps(ONE_LM_EXPERIMENT)
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        violinplot(
            [n_steps_1lm],
            [xticks[0]],
            color=one_lm_color,
            width=2 * half_bar_width,
            showmedians=True,
            median_style=dict(color="lightgray"),
            bw_method=bw_method,
            ax=ax,
        )
        ax.scatter(
            xticks[0],
            np.mean(n_steps_1lm),
            color=one_lm_color,
            marker="o",
            edgecolor="black",
            facecolor="black",
            s=20,
        )

    # Multi-LM
    for i, g in enumerate(groups):
        # Plot percent correct.
        n_steps = [get_n_steps(exp) for exp in g]

        for ax_num, ax in enumerate([bottom_ax, top_ax]):
            # Plot percent correct.
            violinplot(
                n_steps,
                xticks[1:],
                color=group_colors[i],
                width=half_bar_width,
                gap=0.02,
                side=sides[i],
                showmedians=True,
                median_style=dict(color="lightgray"),
                bw_method=bw_method,
                ax=ax,
            )

            means = [np.mean(arr) for arr in n_steps]
            ax.scatter(
                xticks[1:],
                means,
                color=group_colors[i],
                marker="o",
                edgecolor="black",
                facecolor="black",
                s=20,
            )
            ax.plot(xticks[1:], means, color="k", linestyle="-", linewidth=2, zorder=10)
            ax.plot(
                xticks[1:],
                means,
                color=group_colors[i],
                linestyle="-",
                linewidth=1,
                zorder=15,
            )

    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.set_ylim(ylims[ax_num])
        ax.set_yticks(yticks[ax_num])

    # Sets parameters for both x-axes (they're shared, so removing ticks for the
    # top plot removes ticks for the bottom plot).
    bottom_ax.set_xlabel("Num. LMs")
    bottom_ax.set_xticks(xticks)
    bottom_ax.set_xticklabels(["1", "2", "4", "8", "16"])

    bottom_ax.set_ylabel("Steps")
    top_ax.spines.bottom.set_visible(False)

    # Draw y-axis divider markers.
    marker_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    top_ax.plot([0], [0], transform=top_ax.transAxes, **marker_kwargs)
    bottom_ax.plot([0], [1], transform=bottom_ax.transAxes, **marker_kwargs)

    fig.savefig(out_dir / "steps.png")
    fig.savefig(out_dir / "steps.svg")
    plt.show()


if __name__ == "__main__":
    plot_8lm_patches()
    plot_accuracy()
    plot_steps()
