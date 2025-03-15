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


# %matplotlib qt


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


def get_mlh_dict(episode_stats: Mapping, object_name: str, step: int) -> dict:
    """Get the most likely hypothesis for a given graph id."""
    evidences = episode_stats["LM_0"]["evidences"]
    locations = episode_stats["LM_0"]["possible_locations"]
    rotations = episode_stats["LM_0"]["possible_rotations"][0]
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


def plot_goal_state_points():
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_surf_mismatch"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    # Get ground-truth object and its pose.
    target_object = stats["target"]["primary_target_object"]
    target_position = np.array(stats["target"]["primary_target_position"])
    target_rotation = R.from_euler(
        "xyz", stats["target"]["primary_target_rotation_euler"], degrees=True
    )

    goal_states = stats["LM_0"]["goal_states"]
    goal_state_achieved = stats["LM_0"]["goal_state_achieved"]
    sensor_locations = [
        obs["location"] for obs in stats["SM_0"]["processed_observations"]
    ]
    sensor_locations = np.array(sensor_locations)
    n_steps = len(sensor_locations)

    out_dir = OUT_DIR / "steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlim = [-0.10, 0.10]
    ylim = [1.40, 1.60]
    zlim = [-0.10, 0.10]
    view_init = (-50, -180, 0)

    top_mlh_object = "spoon"
    second_mlh_object = "fork"

    for step in range(n_steps):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})

        # Plot ground truth object and sensor path locations.
        ax = axes[0]
        model = load_object_model("surf_agent_1lm", target_object)
        model -= target_position
        model = model.rotated(target_rotation)
        model += target_position
        ax.scatter(
            model.x,
            model.y,
            model.z,
            color="black",
            alpha=0.1,
            s=2,
            edgecolor="none",
        )

        sensor_locs = sensor_locations[: step + 1]
        ax.scatter(
            sensor_locs[:, 0],
            sensor_locs[:, 1],
            sensor_locs[:, 2],
            color="red",
            alpha=1,
            s=20,
            marker="v",
            zorder=10,
        )
        ax.set_proj_type("persp", focal_length=0.8)
        ax.set_title("Sensor Path")
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # Plot top MLH object.
        ax = axes[1]
        top_mlh = get_mlh_dict(stats, top_mlh_object, step)

        model = load_object_model("surf_agent_1lm", top_mlh_object)
        model = model.rotated(top_mlh["rotation"].inv())
        current_mlh_location = top_mlh["rotation"].inv().apply(top_mlh["location"])
        model -= current_mlh_location
        # model -= top_mlh_loc
        # model -= mlh["location"]
        # model = model.rotated(mlh["rotation"].inv())
        # model += target_position
        ax.scatter(
            model.x,
            model.y,
            model.z,
            color=TBP_COLORS["blue"],
            alpha=0.10,
            s=2,
            edgecolor="none",
        )

        # Plot second MLH object.
        # mlh = get_mlh_dict(stats, second_mlh_object, step)
        # model = load_object_model("surf_agent_1lm", second_mlh_object)
        # model -= mlh["location"]
        # model = model.rotated(mlh["rotation"].inv())
        # model += target_position
        # ax.scatter(
        #     model.x,
        #     model.y,
        #     model.z,
        #     color=TBP_COLORS["green"],
        #     alpha=0.10,
        #     s=2,
        #     edgecolor="none",
        # )

        ax.set_proj_type("persp", focal_length=0.8)
        ax.set_title("First/Second MLH")
        # axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        # ax.set_zlim(zlim)

        # Plot goal state, if there is one.
        gs = goal_states[step]
        if gs:
            for ax in axes:
                ax.scatter(
                    gs["location"][0],
                    gs["location"][1],
                    gs["location"][2],
                    color=TBP_COLORS["yellow"],
                    alpha=1,
                    s=20,
                    marker="s",
                    zorder=10,
                )
        for ax in axes:
            ax.view_init(*view_init)

        fig.suptitle(f"Step {step}")
        plt.show()
        fig.savefig(out_dir / f"step_{step}.png", dpi=300, bbox_inches="tight")


def plot_overlay():
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_surf_mismatch"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    # Get ground-truth object and its pose.
    target_object = stats["target"]["primary_target_object"]
    target_position = np.array(stats["target"]["primary_target_position"])
    target_rotation = R.from_euler(
        "xyz", stats["target"]["primary_target_rotation_euler"], degrees=True
    )

    goal_states = stats["LM_0"]["goal_states"]
    goal_state_achieved = stats["LM_0"]["goal_state_achieved"]
    sensor_locations = [
        obs["location"] for obs in stats["SM_0"]["processed_observations"]
    ]
    sensor_locations = np.array(sensor_locations)
    n_steps = len(sensor_locations)

    out_dir = OUT_DIR / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlim = [-0.10, 0.10]
    ylim = [1.40, 1.60]
    zlim = [-0.10, 0.10]
    view_init = (-50, -180, 0)

    top_mlh_object = "spoon"
    second_mlh_object = "fork"

    for step in range(n_steps):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})

        # Plot ground truth object, sensor path, and mlh object together.
        """
        Plot ground truth object, sensor path, and mlh object together.
        """
        ax = axes[0]

        # Plot ground truth object.
        model = load_object_model("surf_agent_1lm", target_object)
        model -= target_position
        model = model.rotated(target_rotation)
        model += target_position
        ax.scatter(
            model.x,
            model.y,
            model.z,
            color="black",
            alpha=0.1,
            s=2,
            edgecolor="none",
        )

        # Plot sensor path.
        sensor_locs = sensor_locations[: step + 1]
        alphas = np.exp(-np.arange(len(sensor_locs)) / 10)[::-1]
        ax.scatter(
            sensor_locs[:, 0],
            sensor_locs[:, 1],
            sensor_locs[:, 2],
            color="red",
            alpha=alphas,
            s=20,
            marker="v",
            zorder=10,
        )

        # Plot top MLH object.
        top_mlh = get_mlh_dict(stats, top_mlh_object, step)
        model = load_object_model("surf_agent_1lm", top_mlh_object)
        model = model.rotated(top_mlh["rotation"].inv())
        current_mlh_location = top_mlh["rotation"].inv().apply(top_mlh["location"])
        model -= current_mlh_location
        model += target_position
        ax.scatter(
            model.x,
            model.y,
            model.z,
            color=TBP_COLORS["blue"],
            alpha=0.10,
            s=2,
            edgecolor="none",
        )

        ax.set_proj_type("persp", focal_length=0.8)
        ax.set_title("Sensor Path")
        axes3d_clean(ax)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        # ax.set_zlim(zlim)

        # Plot top MLH objects.
        ax = axes[1]

        # Plot second MLH object.
        colors = [TBP_COLORS["blue"], TBP_COLORS["green"]]
        for i, mlh_object in enumerate([top_mlh_object, second_mlh_object]):
            mlh = get_mlh_dict(stats, mlh_object, step)
            model = load_object_model("surf_agent_1lm", mlh_object)
            model = model.rotated(mlh["rotation"].inv())
            current_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
            model -= current_mlh_location
            ax.scatter(
                model.x,
                model.y,
                model.z,
                color=colors[i],
                alpha=0.10,
                s=2,
                edgecolor="none",
            )

        ax.set_title("First/Second MLH")
        axes3d_clean(ax)

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        # ax.set_zlim(zlim)

        # Plot goal state, if there is one.
        gs = goal_states[step]
        if gs:
            for ax in axes:
                ax.scatter(
                    gs["location"][0],
                    gs["location"][1],
                    gs["location"][2],
                    color=TBP_COLORS["yellow"],
                    alpha=1,
                    s=20,
                    marker="s",
                    zorder=10,
                )

        for ax in axes:
            ax.set_proj_type("persp", focal_length=0.8)
            axes3d_set_aspect_equal(ax)
            ax.view_init(*view_init)

        fig.suptitle(f"Step {step}")
        plt.show()
        fig.savefig(out_dir / f"step_{step}.png", dpi=300, bbox_inches="tight")


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


exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_surf_mismatch"
detailed_stats_path = exp_dir / "detailed_run_stats.json"
detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
stats = detailed_stats_interface[0]

# Get ground-truth object and its pose.
target_object = stats["target"]["primary_target_object"]
target_position = np.array(stats["target"]["primary_target_position"])
target_rotation = R.from_euler(
    "xyz", stats["target"]["primary_target_rotation_euler"], degrees=True
)
learned_position = np.array([0, 1.5, 0])

goal_states = stats["LM_0"]["goal_states"]
goal_state_achieved = stats["LM_0"]["goal_state_achieved"]
sensor_locations = [obs["location"] for obs in stats["SM_0"]["processed_observations"]]
sensor_locations = np.array(sensor_locations)
n_steps = len(sensor_locations)

out_dir = OUT_DIR / "overlay"
out_dir.mkdir(parents=True, exist_ok=True)
xlim = [-0.10, 0.10]
ylim = [1.40, 1.60]
zlim = [-0.10, 0.10]
view_init = (-50, -180, 0)

top_mlh_object = "spoon"
second_mlh_object = "fork"

# %matplotlib qt

# target_graph = learned_graph.rotated(mlh["rotation"].inv())
# current_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
# target_graph = target_graph - current_mlh_location
# target_graph += target_position

mlh_locs = [get_mlh_dict(stats, top_mlh_object, i)["location"] for i in range(n_steps)]
mlh_locs = np.array(mlh_locs)

for step in range(9, 12):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})

    """
    Plot ground truth object, sensor path, and mlh object together.
    
    mlh_rot: mlh_rot.inv() ~= target_rotation

    """
    _mlh = get_mlh_dict(stats, top_mlh_object, step)
    _mlh_loc = _mlh["location"]  # where we are
    _mlh_rot = _mlh["rotation"]  # object's rotation

    def learned_to_target(pos):
        _current_mlh_loc = _mlh_rot.inv().apply(_mlh_loc)
        p = _mlh_rot.inv().apply(pos) - _current_mlh_loc + sensor_locations[step]
        return p

    learned_graph = load_object_model("surf_agent_1lm", top_mlh_object)
    target_graph = learned_graph.rotated(mlh["rotation"].inv())
    current_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
    target_graph -= current_mlh_location
    target_graph += sensor_locations[step]

    ax = axes[0]

    # Plot ground truth object.
    learned_graph = load_object_model("surf_agent_1lm", target_object)
    target_graph = learned_graph - learned_position
    target_graph = target_graph.rotated(target_rotation)  # ground-truth location
    target_graph = target_graph + target_position  # ground-truth location
    ax.scatter(
        target_graph.x,
        target_graph.y,
        target_graph.z,
        color="black",
        alpha=0.1,
        s=2,
        edgecolor="none",
    )

    # Plot sensor path.
    sensor_locs = sensor_locations[: step + 1]
    alphas = np.exp(-np.arange(len(sensor_locs)) / 1)[::-1]
    ax.scatter(
        sensor_locs[:, 0][-1],
        sensor_locs[:, 1][-1],
        sensor_locs[:, 2][-1],
        color="red",
        alpha=alphas[-1],
        s=20,
        marker="v",
        zorder=10,
    )
    # ax.scatter(
    #     mlh_locs[: step + 1, 0][-1],
    #     mlh_locs[: step + 1, 1][-1],
    #     mlh_locs[: step + 1, 2][-1],
    #     color=TBP_COLORS["purple"],
    #     alpha=alphas[-1],
    #     s=20,
    #     marker="s",
    # )

    # Overlay top MLH object.
    mlh = get_mlh_dict(stats, top_mlh_object, step)
    learned_graph = load_object_model("surf_agent_1lm", top_mlh_object)
    target_graph = learned_graph.rotated(mlh["rotation"].inv())
    current_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
    target_graph -= current_mlh_location
    target_graph += sensor_locations[step]
    ax.scatter(
        target_graph.x,
        target_graph.y,
        target_graph.z,
        color=TBP_COLORS["blue"],
        alpha=0.10,
        s=2,
        edgecolor="none",
    )

    # Overlay learned graph.
    mlh = get_mlh_dict(stats, top_mlh_object, step)
    learned_graph = load_object_model("surf_agent_1lm", top_mlh_object)
    ax.scatter(
        learned_graph.x,
        learned_graph.y,
        learned_graph.z,
        color="red",
        alpha=0.10,
        s=2,
        edgecolor="none",
    )

    # mlh_location = mlh["location"]
    # ax.scatter(
    #     mlh_location[0],
    #     mlh_location[1],
    #     mlh_location[2],
    #     color=TBP_COLORS["purple"],
    #     alpha=1,
    #     s=20,
    #     marker="s",
    # )

    ax.set_proj_type("persp", focal_length=0.8)
    ax.set_title("Sensor Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.set_zlim(zlim)

    """
    Plot first and second MLHs
    """

    ax = axes[1]

    # Plot second MLH object.
    colors = [TBP_COLORS["blue"], TBP_COLORS["green"]]
    for i, mlh_object in enumerate([top_mlh_object, second_mlh_object]):
        mlh = get_mlh_dict(stats, mlh_object, step)
        learned_graph = load_object_model("surf_agent_1lm", mlh_object)
        target_model = learned_graph.rotated(mlh["rotation"].inv())
        current_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
        target_model -= current_mlh_location
        ax.scatter(
            target_model.x,
            target_model.y,
            target_model.z,
            color=colors[i],
            alpha=0.10,
            s=2,
            edgecolor="none",
        )

    ax.set_title("First/Second MLH")
    # axes3d_clean(ax)

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.set_zlim(zlim)

    # Plot goal state, if there is one.
    gs = goal_states[step]
    if gs:
        mlh = get_mlh_dict(stats, top_mlh_object, step)
        loc_0 = gs["location"]

        # Same transformation as MLH object.
        loc_a = mlh["rotation"].inv().apply(loc_0)
        loc -= current_mlh_location
        loc += sensor_locations[step]

        for ax in axes:
            ax.scatter(
                loc[0],
                loc[1],
                loc[2],
                color=TBP_COLORS["yellow"],
                alpha=1,
                s=20,
                marker="s",
                zorder=10,
            )

    for ax in axes:
        ax.set_proj_type("persp", focal_length=0.8)
        axes3d_set_aspect_equal(ax)
        ax.view_init(*view_init)

    fig.suptitle(f"Step {step}")
    plt.show()
    fig.savefig(out_dir / f"step_{step}.png", dpi=300, bbox_inches="tight")
