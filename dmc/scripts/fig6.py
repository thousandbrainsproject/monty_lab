# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""
Figure 6: Rapid Inference With Model-Free and Model-Based Policies
"""

from typing import (
    List,
    Mapping,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    ObjectModel,
    get_frequency,
    load_eval_stats,
    load_object_model,
)
from matplotlib.lines import Line2D
from plot_utils import (
    TBP_COLORS,
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


def plot_curvature_guided_policy():
    """Plot the curvature guided policy.

    Plots the ground truth object and the sensor path over the course of the episode.

    """
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_curvature_guided_policy"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    locations = [obs["location"] for obs in stats["SM_0"]["processed_observations"]]
    locations = np.array(locations)[:14]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), subplot_kw={"projection": "3d"})

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

    plot_sensor_path(
        ax,
        locations,
        color=TBP_COLORS["purple"],
        alpha=1,
        lw=2,
        size=20,
        start_marker_size=20,
    )
    # ax.scatter(
    #     locations[:, 0],
    #     locations[:, 1],
    #     locations[:, 2],
    #     color=TBP_COLORS["blue"],
    #     alpha=1,
    #     s=20,
    #     marker="v",
    #     zorder=10,
    # )
    # ax.plot(
    #     locations[:, 0],
    #     locations[:, 1],
    #     locations[:, 2],
    #     color=TBP_COLORS["blue"],
    #     alpha=1,
    #     zorder=10,
    #     lw=2,
    # )
    ax.set_proj_type("persp", focal_length=0.5)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)
    ax.view_init(elev=54, azim=-36, roll=60)
    plt.show()
    fig.savefig(OUT_DIR / "curvature_guided_policy.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "curvature_guided_policy.svg", bbox_inches="tight")


def plot_evidence_over_time(episode: int):
    """Plot the evidence over time for a given episode.

    Plots the maximum evidence at each step for the top 3 and bottom 2 objects (in
    terms of maximum evidence value) over the course of the episode.
    Args:
        episode (int): The episode to plot the evidence over time for.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes.
    """
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_surf_mismatch"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[episode]

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
        try:
            gs["achieved"] = goal_state_achieved[i]
        except IndexError:
            gs["achieved"] = None
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
            c = "red" if gs["achieved"] else "black"
            ax.axvline(gs["step"], color=c, linestyle="--", alpha=0.5)

    ax.legend(framealpha=1)
    ax.set_xlabel("Step")
    ax.set_xlim(0, n_steps)
    ax.set_ylabel("Evidence")
    ax.set_ylim(0, 55)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()
    out_dir = OUT_DIR / "evidence_over_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"evidence_over_time_{episode}.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(out_dir / f"evidence_over_time_{episode}.svg", bbox_inches="tight")


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


def get_mlh_for_object(object_name: str, stats: Mapping, step: int) -> Mapping:
    """Get the most likely hypothesis for a given object.

    Args:
        object_name (str): The object name.
        stats (Mapping): Detailed stats for an episode.
        step (int): The step to get the MLH for.

    Returns:
        Mapping: The MLH.
    """
    evidences = stats["LM_0"]["evidences"][step]
    locations = stats["LM_0"]["possible_locations"][step]
    rotations = stats["LM_0"]["possible_rotations"][0]
    mlh_id = np.argmax(evidences[object_name])
    return {
        "object_name": object_name,
        "mlh_id": mlh_id,
        "evidence": evidences[object_name][mlh_id],
        "location": np.array(locations[object_name][mlh_id]),
        "rotation": R.from_matrix(rotations[object_name][mlh_id]),
    }


def get_top_two_mlhs(stats, step) -> Tuple[Mapping, Mapping]:
    """Get the top two MLHs for a given step.

    Args:
        stats (Mapping): Detailed stats for an episode.
        step (int): The step to get the MLHs for.

    Returns:
        Tuple[Mapping, Mapping]: The top two MLHs in descending order of evidence.
    """
    evidences = stats["LM_0"]["evidences"][step]
    mlh_info = []
    for object_name in evidences.keys():
        mlh_info.append(get_mlh_for_object(object_name, stats, step))
    lst = sorted(mlh_info, key=lambda x: x["evidence"], reverse=True)
    return lst[0], lst[1]


def get_top_two_mlhs_for_object(
    object_name: str, stats: Mapping, step: int
) -> Tuple[Mapping, Mapping]:
    """Get the top two pose hypotheses for a given object.

    Args:
        object_name (str): The object/graph id.
        stats (Mapping): Detailed stats for an episode.
        step (int): The step to get the MLHs for.

    Returns:
        Tuple[Mapping, Mapping]: An object's top two pose hypothesis MLHs in
        descending order of evidence.
    """
    evidences = stats["LM_0"]["evidences"][step]
    locations = stats["LM_0"]["possible_locations"][step]
    rotations = stats["LM_0"]["possible_rotations"][0]
    sort_order = np.argsort(evidences[object_name])[::-1]
    mlhs = []
    for mlh_id in sort_order[:2]:
        mlhs.append(
            {
                "object_name": object_name,
                "mlh_id": mlh_id,
                "evidence": evidences[object_name][mlh_id],
                "location": np.array(locations[object_name][mlh_id]),
                "rotation": R.from_matrix(rotations[object_name][mlh_id]),
            }
        )
    return mlhs


def get_graph_for_mlh(
    mlh: Mapping,
    stats: Mapping,
    step: int,
    pretrained_model: str = "surf_agent_1lm",
) -> ObjectModel:
    """Get the graph of the MLH in the same reference frame as the sensed target object.

    Args:
        mlh (Mapping): The MLH dictionary.
        stats (Mapping): Detailed stats for an episode.
        step (int): The step to get the MLH for.
        pretrained_model (str): The pretrained model to use.

    Returns:
        ObjectModel: The graph of the MLH in the same reference frame as
          the sensed target object.
    """
    rotated_mlh_location = mlh["rotation"].inv().apply(mlh["location"])
    sensor_location = np.array(
        stats["SM_0"]["processed_observations"][step]["location"]
    )
    learned_graph = load_object_model(pretrained_model, mlh["object_name"])
    graph = learned_graph.rotated(mlh["rotation"].inv())
    graph -= rotated_mlh_location
    graph += sensor_location
    return graph


def get_goal_states(stats: Mapping) -> List[Mapping]:

    evidences_max = stats["LM_0"]["evidences_max"]
    goal_states = stats["LM_0"]["goal_states"]
    goal_state_achieved = stats["LM_0"]["goal_state_achieved"]
    possible_matches = stats["LM_0"]["possible_matches"]

    # Collect info about goal states.
    goal_state_episodes = np.argwhere(goal_states).squeeze()
    out = []
    for i, step in enumerate(goal_state_episodes):
        gs = goal_states[step]
        match_ids = np.array(possible_matches[step], dtype=object)
        match_evs = np.array([evidences_max[step][match] for match in match_ids])
        match_evs, match_ids = zip(*sorted(zip(match_evs, match_ids), reverse=True))
        gs["step"] = step
        try:
            gs["achieved"] = goal_state_achieved[i]
        except IndexError:
            gs["achieved"] = None
        gs["possible_matches"] = dict(zip(match_ids, match_evs))
        out.append(gs)

    return out

def plot_sensor_path(
    ax: plt.Axes,
    sensor_locations: np.ndarray,
    color: str = TBP_COLORS["purple"],
    alpha: float = 1,
    # Line style
    lw: float = 1,
    # Scatter style
    size: float = 10,
    marker: str = "v",
    start_marker: Optional[str] = "x",
    start_marker_size: float = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the sensor path on the ground-truth object.

    Args:
        ax (plt.Axes): The axes to plot on.
        sensor_locations (np.ndarray): The sensor locations.
    """
    scatter_locations = line_locations = sensor_locations
    if start_marker:
        ax.scatter(
            [scatter_locations[0, 0]],
            [scatter_locations[0, 1]],
            [scatter_locations[0, 2]],
            marker=start_marker,
            color=color,
            s=start_marker_size,
            alpha=alpha,
            zorder=10,
        )
        scatter_locations = scatter_locations[1:]

    ax.scatter(
        scatter_locations[:, 0],
        scatter_locations[:, 1],
        scatter_locations[:, 2],
        marker=marker,
        color=color,
        s=size,
        alpha=alpha,
        zorder=10,
    )
    ax.plot(
        line_locations[:, 0],
        line_locations[:, 1],
        line_locations[:, 2],
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=20,
    )

def plot_hypotheses_for_step(
    stats: Mapping,
    step: int,
    top_mlh: Mapping,
    second_mlh: Mapping,
    style: Optional[Mapping] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the hypotheses for a given step.

    Args:
        stats (Mapping): Detailed stats for an episode.
        step (int): The step to plot the hypotheses for.
        top_mlh (Mapping): The MLH with the highest evidence value.
        second_mlh (Mapping): The MLH with the second highest evidence value.
        style (Optional[Mapping]): The style for the plot items.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes.
    """
    default_style = {
        "target": {
            "alpha": 0.2,
            "s": 2,
            "edgecolor": "none",
        },
        "sensor_path_scatter": {
            "color": TBP_COLORS["blue"],
            "s": 10,
            "marker": "v",
            "zorder": 10,
        },
        "sensor_path_line": {
            "color": TBP_COLORS["blue"],
            "alpha": 1,
            "lw": 1,
            "zorder": 5,
        },
        "top_mlh": {
            "color": TBP_COLORS["blue"],
            "alpha": 0.20,
            "s": 2,
            "edgecolor": "none",
        },
        "second_mlh": {
            "color": TBP_COLORS["green"],
            "alpha": 0.20,
            "s": 2,
            "edgecolor": "none",
        },
        "proposed_point": {
            "color": TBP_COLORS["yellow"],
            "alpha": 1,
            "s": 20,
            "marker": "o",
            "zorder": 20,
            "edgecolor": "black",
        },
    }
    if style:
        for key, val in style.items():
            default_style[key].update(val.copy())
    style = default_style

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

    fig, axes = plt.subplots(1, 2, figsize=(5, 4), subplot_kw={"projection": "3d"})

    """
    First plot has ground-truth object and sensor path.
    """
    ax = axes[0]

    # Plot ground-truth object.
    learned_graph = load_object_model("surf_agent_1lm", target_object)
    target_graph = learned_graph - learned_position
    target_graph = target_graph.rotated(target_rotation)
    target_graph += target_position
    ax.scatter(
        target_graph.x,
        target_graph.y,
        target_graph.z,
        color=style["target"].pop("color", target_graph.rgba),
        **style["target"],
    )

    # Plot sensor path on ground-truth object.
    plot_sensor_path(
        ax,
        sensor_locations,
    )
    """
    Second plot has first and second MLHs.
    """
    ax = axes[1]

    # Plot first and second MLHs.
    top_mlh["graph"] = get_graph_for_mlh(
        top_mlh, stats, step, pretrained_model="surf_agent_1lm"
    )
    ax.scatter(
        top_mlh["graph"].x,
        top_mlh["graph"].y,
        top_mlh["graph"].z,
        **style["top_mlh"],
    )

    second_mlh["graph"] = get_graph_for_mlh(
        second_mlh, stats, step, pretrained_model="surf_agent_1lm"
    )
    ax.scatter(
        second_mlh["graph"].x,
        second_mlh["graph"].y,
        second_mlh["graph"].z,
        **style["second_mlh"],
    )

    # Plot the goal state's target if possible.
    goal_state = stats["LM_0"]["goal_states"][step]
    if goal_state:
        proposed_surface_loc = goal_state["info"]["proposed_surface_loc"]
        for ax in axes:
            ax.scatter(
                proposed_surface_loc[0],
                proposed_surface_loc[1],
                proposed_surface_loc[2],
                **style["proposed_point"],
            )
    return fig, axes


def plot_object_hypothesis():
    experiment = "fig6_surf_mismatch"
    episode = 0

    exp_dir = VISUALIZATION_RESULTS_DIR / experiment
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[episode]

    # Select a step where there's a goal state was achieved, and it was
    # for distuinguishing between different objects.
    goal_states = get_goal_states(stats)
    lst = filter(lambda gs: gs["achieved"] is True, goal_states)
    lst = list(filter(lambda gs: len(gs["possible_matches"]) > 1, lst))
    gs = lst[0]
    step = gs["step"]

    # Get the pose MLHs.
    top_mlh, second_mlh = get_top_two_mlhs(stats, step)

    style = {}
    fig, axes = plot_hypotheses_for_step(
        stats,
        step,
        top_mlh,
        second_mlh,
        style=style,
    )

    # Add label, legends, etc.
    width = 0.08
    axis_limits = [[-width, width], [1.5 - width, 1.5 + width], [-width, width]]
    view_angles = [(-50, -180, 0), (-80, 180, 0)]
    for i, ax in enumerate(axes):
        ax.set_proj_type("persp", focal_length=0.8)
        axes3d_set_aspect_equal(ax)
        axes3d_clean(ax)
        ax.view_init(*view_angles[i])
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_zlim(axis_limits[2])

    legend_handles = get_legend_handles(["sensor", "spoon", "fork", "goal"])
    axes[1].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.1, 0.8),
        framealpha=1,
        fontsize=8,
    )

    plt.show()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "object_hypothesis.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "object_hypothesis.svg", bbox_inches="tight")


def plot_pose_hypothesis():
    experiment = "fig6_surf_mismatch"
    episode = 1

    exp_dir = VISUALIZATION_RESULTS_DIR / experiment
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[episode]

    # Select a step where there's a goal state was achieved, and it was
    # for pose estimation (object was already determined).
    goal_states = get_goal_states(stats)
    lst = filter(lambda gs: gs["achieved"] is True, goal_states)
    lst = list(filter(lambda gs: len(gs["possible_matches"]) == 1, lst))
    gs = lst[0]
    step = gs["step"]

    # Get the pose MLHs.
    mlh_graph_id = stats["LM_0"]["current_mlh"][step]["graph_id"]
    top_mlh, second_mlh = get_top_two_mlhs_for_object(mlh_graph_id, stats, step)
    style = {}
    fig, axes = plot_hypotheses_for_step(
        stats,
        step,
        top_mlh,
        second_mlh,
        style=style,
    )

    # Add label, legends, etc.
    width = 0.08
    axis_limits = [[-width, width], [1.5 - width, 1.5 + width], [-width, width]]
    view_angles = [
        (-37, 26, 0),
        (-7.498398271623369, 34.260448483945446, 0),
    ]
    for i, ax in enumerate(axes):
        ax.set_proj_type("persp", focal_length=0.8)
        axes3d_set_aspect_equal(ax)
        axes3d_clean(ax)
        ax.view_init(*view_angles[i])
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_zlim(axis_limits[2])

    # Add legend to the second plot.
    legend_handles = get_legend_handles(["sensor", "pose 1", "pose 2", "goal"])
    axes[1].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.1, 0.8),
        framealpha=1,
        fontsize=8,
    )

    plt.show()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pose_hypothesis.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "pose_hypothesis.svg", bbox_inches="tight")

    return fig, axes


def get_legend_handles(
    labels: List[str],
) -> List[Line2D]:
    legend_handles = []
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=TBP_COLORS["purple"],
            markeredgecolor=TBP_COLORS["purple"],
            markersize=7,
            label=labels[0],
        )
    )
    colors = [TBP_COLORS["blue"], TBP_COLORS["green"]]
    for i in range(2):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                markersize=6,
                label=labels[i + 1],
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TBP_COLORS["yellow"],
            markeredgecolor="black",
            markersize=6,
            label=labels[3],
        )
    )
    return legend_handles


plot_curvature_guided_policy()
plot_pose_hypothesis()
plot_object_hypothesis()
