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
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_frequency,
    load_eval_stats,
    load_object_model,
)
from matplotlib.lines import Line2D
from plot_utils import (
    TBP_COLORS,
    add_legend,
    axes3d_clean,
    axes3d_set_aspect_equal,
    violinplot,
)
from scipy.spatial.transform import Rotation as R

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

OUT_DIR = DMC_ANALYSIS_DIR / "fig6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def euler(r):
    return r.as_euler("xyz", degrees=True)


def get_relative_rotation(
    rot_a: R,
    rot_b: R,
    degrees: bool = True,
) -> Tuple[float, np.ndarray]:
    """Computes the angle and axis of rotation between two rotation matrices.

    Args:
        rot_a (scipy.spatial.transform.Rotation): The first rotation.
        rot_b (scipy.spatial.transform.Rotation): The second rotation.

    Returns:
        Tuple[float, np.ndarray]: The rotational difference and the relative rotation matrix.
    """
    # Compute rotation angle
    rel = rot_a * rot_b.inv()
    mat = rel.as_matrix()
    trace = np.trace(mat)
    theta = np.arccos((trace - 1) / 2)

    if np.isclose(theta, 0):  # No rotation
        return 0.0, np.array([0.0, 0.0, 0.0])

    # Compute rotation axis
    axis = np.array(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ]
    )
    axis = axis / (2 * np.sin(theta))  # Normalize
    if degrees:
        theta, axis = np.degrees(theta), np.degrees(axis)

    return theta, axis


def plot_curvature_guided_policy():
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_curvature_guided_policy"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    locations = [obs["location"] for obs in stats["SM_0"]["processed_observations"]]
    locations = np.array(locations)[:14]

    # %matplotlib qt
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw={"projection": "3d"})

    model = load_object_model("dist_agent_1lm", "mug")
    ax.scatter(
        model.x,
        model.y,
        model.z,
        color=model.rgba,
        alpha=0.35,
        s=3,
        edgecolor="none",
    )

    ax.scatter(
        locations[:, 0],
        locations[:, 1],
        locations[:, 2],
        color=TBP_COLORS["blue"],
        alpha=1,
        s=20,
        marker="v",
        zorder=10,
    )
    ax.plot(
        locations[:, 0],
        locations[:, 1],
        locations[:, 2],
        color=TBP_COLORS["blue"],
        alpha=1,
        zorder=10,
        lw=2,
    )
    ax.set_proj_type("persp", focal_length=0.5)
    axes3d_clean(ax, grid_color=(1, 1, 1, 1))
    axes3d_set_aspect_equal(ax)
    ax.view_init(elev=54, azim=-36, roll=60)
    plt.show()
    fig.savefig(OUT_DIR / "curvature_guided_policy.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "curvature_guided_policy.svg", bbox_inches="tight")


def plot_evidence_over_time():
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_surf_mismatch"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    evidences = stats["LM_0"]["evidences"]
    evidences_max = stats["LM_0"]["evidences_max"]
    goal_states = stats["LM_0"]["goal_states"]
    goal_state_achieved = stats["LM_0"]["goal_state_achieved"]
    possible_matches = stats["LM_0"]["possible_matches"]
    n_steps = len(evidences)

    # Summarize goal states and evidence counts as each attempted goal state.
    print("Goal States")
    for i, step in enumerate(np.where(goal_states)[0]):
        gs = goal_states[step]
        gs["step"] = step
        pos_match_ids = np.array(possible_matches[step], dtype=object)
        pos_match_evs = np.array(
            [evidences_max[step][match] for match in pos_match_ids]
        )
        sorting_order = np.argsort(pos_match_evs)[::-1]
        pos_match_ids = pos_match_ids[sorting_order]
        pos_match_evs = pos_match_evs[sorting_order]

        gs["step"] = step
        gs["achieved"] = goal_state_achieved[i]
        gs["possible_matches"] = {
            pos_match_ids[i]: pos_match_evs[i] for i in range(len(pos_match_ids))
        }
        s = f" - Step {step} (achieved: {gs['achieved']}): "
        lst = []
        for i in range(len(pos_match_ids)):
            lst.append(f"{pos_match_ids[i]} ({pos_match_evs[i]:.2f})")
        s += ", ".join(lst)
        print(s)

    # Plot evidence values over time for a handful of objects.
    all_graph_ids = list(evidences_max[0].keys())
    evs_per_step = {}
    for graph_id in all_graph_ids:
        evs_per_step[graph_id] = np.array([dct[graph_id] for dct in evidences_max])

    # Sort evidence values by maximum over time.
    ev_maxs = {graph_id: np.max(arr) for graph_id, arr in evs_per_step.items()}
    ev_maxs_names = np.array(list(ev_maxs.keys()), dtype=object)
    ev_maxs_arr = np.array(list(ev_maxs.values()))
    sorting_order = np.argsort(ev_maxs_arr)[::-1]
    sorted_names = ev_maxs_names[sorting_order]
    top_3 = sorted_names[:3]
    bottom_2 = sorted_names[-2:]

    colors = [
        "black",
        TBP_COLORS["blue"],
        TBP_COLORS["purple"],
        TBP_COLORS["green"],
        TBP_COLORS["yellow"],
        "red",
    ]
    graph_ids = list(top_3) + list(bottom_2)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i in range(len(graph_ids)):
        graph_id = graph_ids[i]
        color = colors[i]
        ax.plot(evs_per_step[graph_id], label=graph_id, color=color)

    for gs in goal_states:
        if gs:
            # if gs["achieved"]:
            ax.axvline(gs["step"], color="black", linestyle="--", alpha=0.5)

    ax.legend(framealpha=1)
    ax.set_xlabel("Step")
    ax.set_xlim(0, n_steps)
    ax.set_ylabel("Evidence")
    ax.set_ylim(0, 55)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()
    fig.savefig(OUT_DIR / "evidence_over_time.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "evidence_over_time.svg", bbox_inches="tight")


def get_mlh_dict(object_name: str, stats: Mapping, step: int) -> dict:
    """Get the most likely hypothesis for a given graph id."""
    evidences = stats["LM_0"]["evidences"]
    locations = stats["LM_0"]["possible_locations"]
    rotations = stats["LM_0"]["possible_rotations"][0]
    mlh_id = np.argmax(evidences[step][object_name])
    evidence = evidences[step][object_name][mlh_id]
    location = np.array(locations[step][object_name][mlh_id])
    rotation = R.from_matrix(rotations[object_name][mlh_id])
    return {
        "object_name": object_name,
        "mlh_id": mlh_id,
        "location": location,
        "rotation": rotation,
        "evidence": evidence,
    }


def plot_performance():
    experiments = [
        "dist_agent_1lm_randrot_noise_nohyp",
        "surf_agent_1lm_randrot_noise_nohyp",
        "surf_agent_1lm_randrot_noise",
    ]
    xticks = np.arange(len(experiments))
    xticklabels = [
        "None",
        "Model-Free",
        "Model-Based",
    ]
    eval_stats = []
    for exp in experiments:
        eval_stats.append(load_eval_stats(exp))

    fig, axes = plt.subplots(1, 2, figsize=(4, 3))

    ax = axes[0]
    accuracies, accuracies_mlh = [], []
    for df in eval_stats:
        accuracies.append(100 * get_frequency(df["primary_performance"], "correct"))
        accuracies_mlh.append(
            100 * get_frequency(df["primary_performance"], "correct_mlh")
        )
    ax.bar(xticks, accuracies, width=0.8, color=TBP_COLORS["blue"])
    ax.bar(
        xticks, accuracies_mlh, bottom=accuracies, width=0.8, color=TBP_COLORS["yellow"]
    )

    ax.set_title("Accuracy")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel("% Correct")
    ax.set_ylim(0, 100)
    ax.legend(["Correct", "Correct MLH"], loc="lower right", framealpha=1)
    sns.despine(ax=ax)

    ax = axes[1]
    n_steps = []
    for df in eval_stats:
        n_steps.append(df["num_steps"])

    violinplot(
        n_steps,
        xticks,
        color=TBP_COLORS["blue"],
        showmedians=True,
        median_style=dict(color="lightgray"),
        ax=ax,
    )
    ax.set_title("Steps")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 500)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    fig.savefig(OUT_DIR / "performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "performance.svg", bbox_inches="tight")


def plot_sensor_path(
    ax,
    sensor_locations: np.ndarray,
    markerstyle: Optional[Mapping] = None,
    linestyle: Optional[Mapping] = None,
):
    if markerstyle is None:
        markerstyle = dict(
            color=TBP_COLORS["blue"],
            alpha=1,
            s=20,
            marker="v",
            zorder=5,
        )
    if linestyle is None:
        linestyle = dict(
            color=TBP_COLORS["blue"],
            alpha=1,
        )
    ax.scatter(
        sensor_locations[:, 0],
        sensor_locations[:, 1],
        sensor_locations[:, 2],
        color=TBP_COLORS["blue"],
        alpha=1,
        s=20,
        marker="v",
        zorder=5,
    )
    ax.plot(
        sensor_locations[:, 0],
        sensor_locations[:, 1],
        sensor_locations[:, 2],
        color=TBP_COLORS["blue"],
        alpha=1,
        zorder=10,
        lw=2,
    )


def get_mlh_graph(
    object_name: str,
    stats: Mapping,
    step: int,
    pretrained_model: str = "surf_agent_1lm",
) -> Tuple[np.ndarray, R]:
    """
    Get the graph of the MLH object at the given step.
    """
    mlh = get_mlh_dict(object_name, stats, step)
    rotated_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
    sensor_location = np.array(
        stats["SM_0"]["processed_observations"][step]["location"]
    )
    learned_graph = load_object_model(pretrained_model, object_name)
    graph = learned_graph.rotated(mlh["rotation"].inv())
    graph -= rotated_mlh_location
    graph += sensor_location
    return graph


def plot_mismatch_at_step(
    stats: Mapping,
    step: int,
    mlh_objects: Iterable[str],
    colors: Mapping,
    pretrained_model: str = "surf_agent_1lm",
):
    """ """

    # Get ground-truth object and its pose.
    target_object = stats["target"]["primary_target_object"]
    target_position = np.array(stats["target"]["primary_target_position"])
    target_rotation = R.from_euler(
        "xyz", stats["target"]["primary_target_rotation_euler"], degrees=True
    )
    learned_position = np.array([0, 1.5, 0])

    # Load sensor locations.
    sensor_locations = np.array(
        [obs["location"] for obs in stats["SM_0"]["processed_observations"]]
    )
    sensor_locations = sensor_locations[: step + 1]

    # Load MLH objects.
    top_mlh_object = mlh_objects[0]
    second_mlh_object = mlh_objects[1]

    # Load goal state.
    goal_state = stats["LM_0"]["goal_states"][step]

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), subplot_kw={"projection": "3d"})

    """
    First plot has ground-truth object and sensor path.
    """
    ax = axes[0]

    # Plot ground-truth object.
    learned_graph = load_object_model(pretrained_model, target_object)
    graph = learned_graph - learned_position
    graph = graph.rotated(target_rotation)  # ground-truth location
    graph += target_position  # ground-truth location
    ax.scatter(
        graph.x,
        graph.y,
        graph.z,
        color=graph.rgba,
        alpha=0.5,
        s=5,
        edgecolor="none",
    )

    # Plot sensor path.
    c = colors.get("sensor_path", TBP_COLORS["purple"])
    ax.scatter(
        sensor_locations[:, 0],
        sensor_locations[:, 1],
        sensor_locations[:, 2],
        color=c,
        alpha=1,
        s=10,
        marker="v",
        zorder=10,
    )
    ax.plot(
        sensor_locations[:, 0],
        sensor_locations[:, 1],
        sensor_locations[:, 2],
        color=c,
        alpha=1,
        zorder=5,
        lw=1,
    )

    """
    Second plot has first and second MLHs.
    """
    ax = axes[1]

    # Plot first and second MLHs.
    top_mlh_graph = get_mlh_graph(
        top_mlh_object, stats, step, pretrained_model=pretrained_model
    )
    second_mlh_graph = get_mlh_graph(
        second_mlh_object, stats, step, pretrained_model=pretrained_model
    )
    colors = {} if colors is None else colors
    mlh_colors = [
        colors.get("top_mlh", TBP_COLORS["blue"]),
        colors.get("second_mlh", TBP_COLORS["green"]),
    ]
    for i, graph in enumerate([top_mlh_graph, second_mlh_graph]):
        ax.scatter(
            graph.x,
            graph.y,
            graph.z,
            color=mlh_colors[i],
            alpha=0.20,
            s=2,
            edgecolor="none",
            label=mlh_objects[i],
        )

    # Plot the goal state's target if possible.
    if goal_state:
        proposed_surface_loc = goal_state["info"]["proposed_surface_loc"]
        c = colors.get("proposed_point", TBP_COLORS["yellow"])
        for ax in axes:
            ax.scatter(
                proposed_surface_loc[0],
                proposed_surface_loc[1],
                proposed_surface_loc[2],
                color=c,
                alpha=1,
                s=20,
                marker="o",
                zorder=10,
                label="Proposed Point",
            )

    return fig, axes


def plot_object_mismatch():
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_surf_mismatch"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    mlh_objects = ["spoon", "fork"]
    colors = {
        "sensor_path": TBP_COLORS["blue"],
        "top_mlh": TBP_COLORS["blue"],
        "second_mlh": TBP_COLORS["green"],
        "proposed_point": TBP_COLORS["yellow"],
    }

    step = 10
    fig, axes = plot_mismatch_at_step(
        stats, step, mlh_objects, colors=colors, pretrained_model="surf_agent_1lm"
    )
    width = 0.08
    axis_limits = [[-width, width], [1.5 - width, 1.5 + width], [-width, width]]
    view_angles = [(-50, -180, 0), (-80, 180, 0)]

    axes[0].set_title("Ground Truth + Sensor Path")
    axes[1].set_title("First + Second MLHs")
    for i, ax in enumerate(axes):
        ax.set_proj_type("persp", focal_length=0.8)
        axes3d_set_aspect_equal(ax)
        axes3d_clean(ax)
        ax.view_init(*view_angles[i])
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_zlim(axis_limits[2])

    # Add legend to the second plot.
    colors = list(colors.values())
    labels = ["Sensor Path", "MLH 1 (spoon)", "MLH 2 (fork)", "Proposed Point"]
    legend_handles = []
    for i in range(1, 4):
        h = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=8,
            label=labels[i],
        )
        legend_handles.append(h)
    axes[1].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.2, 0.8),
        framealpha=1,
        fontsize=8,
    )

    plt.show()

    out_dir = OUT_DIR / "mismatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"object_mismatch.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"object_mismatch.svg", bbox_inches="tight")


plot_object_mismatch()
