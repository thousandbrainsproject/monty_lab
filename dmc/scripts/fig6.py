# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 6.

Panel A: Curvature Guided Policy
 - `plot_curvature_guided_policy()`

Panel B: Performance
 - `plot_performance()`

Panel C and E: Hypotheses Visualizations
 - `plot_object_hypotheses_visualization()`
 - `plot_pose_hypotheses_visualization()`

Panels D and F: Hypotheses Evidence
 - `plot_object_hypotheses_evidence()`
 - `plot_pose_hypotheses_evidence()`

Running the above functions requires that the following experiments have been run:
 - `fig6_curvature_guided_policy`
 - `fig6_hypothesis_driven_policy`
 - `dist_agent_1lm_randrot_noise_nohyp
 - `surf_agent_1lm_randrot_noise_nohyp`
 - `surf_agent_1lm_randrot_noise`
"""

from numbers import Number
from typing import (
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
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
from matplotlib.patches import Polygon
from plot_utils import (
    TBP_COLORS,
    SensorModuleData,
    axes3d_clean,
    axes3d_set_aspect_equal,
    extract_style,
    init_matplotlib_style,
    update_style,
    violinplot,
)
from scipy.spatial.transform import Rotation as R

init_matplotlib_style()

OUT_DIR = DMC_ANALYSIS_DIR / "fig6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Style utilities. Can be hard-coded later when we decide on colors, etc.
"""


# Plotting styles.
HYPOTHESIS_COLORS = [
    TBP_COLORS["purple"],
    TBP_COLORS["green"],
    TBP_COLORS["blue"],
    TBP_COLORS["pink"],
    TBP_COLORS["yellow"],
]

STYLE = {
    "target.color": "gray",
    "target.alpha": 0.5,
    "target.marker": "o",
    "target.s": 1,
    "target.edgecolor": "none",
    "top_mlh.color": HYPOTHESIS_COLORS[0],
    "top_mlh.alpha": 0.20,
    "top_mlh.marker": "o",
    "top_mlh.s": 1,
    "top_mlh.edgecolor": "none",
    "second_mlh.color": HYPOTHESIS_COLORS[1],
    "second_mlh.alpha": 0.20,
    "second_mlh.marker": "o",
    "second_mlh.s": 1,
    "second_mlh.edgecolor": "none",
    "goal.color": TBP_COLORS["yellow"],
    "goal.alpha": 1,
    "goal.marker": "v",
    "goal.s": 20,
    "goal.edgecolor": "black",
    "goal.lw": 0.5,
    "goal.zorder": 20,
}

"""
--------------------------------------------------------------------------------
Panel A: Curvature Guided Policy
--------------------------------------------------------------------------------
"""


def plot_curvature_guided_policy():
    """Plot the object being sensed and the sensor path, including observations.

    Requires the experiment `fig6_curvature_guided_policy` to have been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig6/curvature_guided_policy`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "curvature_guided_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the stats.
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_curvature_guided_policy"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    # Initialize figure.
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), subplot_kw={"projection": "3d"})

    # Plot the ground truth object.
    model = load_object_model("dist_agent_1lm", "mug")
    ax.scatter(
        model.x,
        model.y,
        model.z,
        color=model.rgba,
        alpha=0.35,
        s=4,
        edgecolor="none",
    )

    # Plot the patches.
    sm = SensorModuleData(stats["SM_0"])
    for step in range(14):
        sm.plot_raw_observation(ax, step, scatter=True, contour=True)

    # Plot the sensor path.
    sm.plot_sensor_path(ax, steps=14, start=False, scatter=False)

    # Set axes properties.
    ax.set_proj_type("persp", focal_length=0.5)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)
    ax.view_init(elev=54, azim=-36, roll=60)

    # Save the figure.
    fig.savefig(out_dir / "curvature_guided_policy.png", bbox_inches="tight")
    fig.savefig(out_dir / "curvature_guided_policy.svg", bbox_inches="tight")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel B: Performance
--------------------------------------------------------------------------------
"""


def plot_performance():
    """Plot performance as a function of motor policy.

    Requires the experiments `dist_agent_1lm_randrot_noise_nohyp`,
    `surf_agent_1lm_randrot_noise_nohyp`, and `surf_agent_1lm_randrot_noise` to
    have been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig6/performance`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        "dist_agent_1lm_randrot_noise_nohyp",
        "surf_agent_1lm_randrot_noise_nohyp",
        "surf_agent_1lm_randrot_noise",
    ]
    xticks = np.arange(len(experiments))
    xticklabels = [
        "Random Walk",
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

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="center")
    ax.set_xlabel("Motor Policy")
    ax.set_ylabel("% Correct")
    ax.set_ylim(0, 100)
    ax.legend(["Correct", "Correct MLH"], loc="lower right", framealpha=1)

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
        bw_method=0.1,
        ax=ax,
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="center")
    ax.set_xlabel("Motor Policy")
    ax.set_ylabel("Steps")
    ax.set_ylim(0, 500)
    fig.tight_layout()

    fig.savefig(out_dir / "performance.png", bbox_inches="tight")
    fig.savefig(out_dir / "performance.svg", bbox_inches="tight")
    plt.show()


"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""


def get_goal_states(stats: Mapping, achieved: Optional[bool] = True) -> List[Mapping]:
    """Get the goal states for an episode.

    Args:
        stats: Detailed stats for an episode.
        achieved: Whether to filter goal states by whether they were achieved. Default
          is True.

    Returns:
        List[Mapping]: The goal states for an episode.
    """
    goal_states = stats["LM_0"]["goal_states"]
    if achieved is not None:
        goal_states = [gs for gs in goal_states if gs["info"]["achieved"] == achieved]
    possible_matches = stats["LM_0"]["possible_matches"]
    for gs in goal_states:
        step = gs["info"]["matching_step_when_output_goal_set"]
        gs["info"]["is_pose_hypothesis"] = len(possible_matches[step]) == 1
        gs["info"]["possible_matches"] = possible_matches[step]
    return goal_states


def draw_triangle(
    ax: plt.Axes,
    x: Number,
    y: Number,
    theta: Number = 0,
    degrees: bool = True,
    radius: Number = 1,
    align: str = "center",
    **style,
) -> Polygon:
    """Draw a triangle, properly sized and located.

    Matplotlib wasn't playing nicely when trying to put triangles on the figure
    that went outside the axes limits. This function converts the triangle to
    normalized axes coordinates, so that it can be drawn on any axes.

    Args:
        ax: Axes to plot in.
        x: x position in data coordinates.
        y: y position in data coordinates.
        theta: Counter-clockwise rotation of triangle. Default is 0, which has the
            triangle pointing up.
        degrees: Whether theta is in degrees. Defaults to True.
        radius: Radius of the circle that inscribes the triangle in millimeters.
        align: Alignment of triangle. Defaults to "center".
        style: Additional style arguments for the matplotlib polygon.
    Raises:
        ValueError: If align is not one of "center", "bottom", "top", "left", or
        "right".

    Returns:
        np.ndarray: Triangle vertices in normalized axes coordinates.
    """
    t = np.arange(0, 1, 1 / 3)
    x_coords = radius * np.cos(t * 2 * np.pi)
    y_coords = radius * np.sin(t * 2 * np.pi)
    verts_mm = np.stack([x_coords, y_coords], axis=1)

    # Get triangle pointing up.
    d_theta = 2 * np.pi / 3 - np.pi / 2
    c, s = np.cos(d_theta), np.sin(d_theta)
    rot = np.array([[c, -s], [s, c]])
    verts_mm = verts_mm @ rot

    # Apply counter-clockwise rotation.
    if degrees:
        theta = np.deg2rad(-theta)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    verts_mm = verts_mm @ rot

    # Get points in axis-length coordinates.
    verts_pix = verts_mm * ax.figure.dpi / 25.4
    bbox = ax.get_window_extent()
    pix_per_ax_length = np.array([bbox.width, bbox.height])
    verts_ax = verts_pix / pix_per_ax_length[None, :]

    # Align triangle.
    if align == "bottom":
        offset = np.array([0, verts_ax[:, 1].min()])
    elif align == "top":
        offset = np.array([0, verts_ax[:, 1].max()])
    elif align == "left":
        offset = np.array([verts_ax[:, 0].min(), 0])
    elif align == "right":
        offset = np.array([verts_ax[:, 0].max(), 0])
    elif align == "center":
        offset = np.array([0, 0])
    else:
        raise ValueError(f"Invalid align: {align}")
    verts_ax = verts_ax - offset[None, :]

    # Move vertices to location on axes.
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_ax = (x - x_lim[0]) / (x_lim[1] - x_lim[0])
    y_ax = (y - y_lim[0]) / (y_lim[1] - y_lim[0])
    offset = np.array([x_ax, y_ax])
    verts_ax = verts_ax + offset[None, :]

    kwargs = {
        "closed": True,
        "transform": ax.transAxes,
        "clip_on": False,
    }
    kwargs.update(style)
    triangle = Polygon(verts_ax, **kwargs)
    ax.add_patch(triangle)

    return triangle


"""
--------------------------------------------------------------------------------
Panels C and E: Hypothesis Visualizations
--------------------------------------------------------------------------------
"""

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
    """Get the top two MLHs for a given step (different objects).

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


def get_graph_for_hypothesis(
    mlh: Mapping,
    stats: Mapping,
    step: int,
    pretrained_model: str = "surf_agent_1lm",
) -> ObjectModel:
    """Get the graph of the MLH in the same reference frame as the sensed target object.

    Args:
        mlh: The MLH dictionary.
        stats: Detailed stats for an episode.
        step: The step to get the MLH for.
        pretrained_model: The pretrained model to use. Default is "surf_agent_1lm".

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


def plot_hypotheses_visualization(
    stats: Mapping,
    step: int,
    top_mlh: Mapping,
    second_mlh: Mapping,
    style: Optional[Mapping] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the hypotheses for a given step.

    Args:
        stats: Detailed stats for an episode.
        step: The step to plot the hypotheses for.
        top_mlh: The MLH with the highest evidence value.
        second_mlh: The MLH with the second highest evidence value.
        style: Optional style overrides for the plot items.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes.
    """
    style = update_style(STYLE, style)

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
    target_style = extract_style(style, "target")
    if target_style.get("color") == "rgba":
        target_style["color"] = target_graph.rgba
    ax.scatter(
        target_graph.x,
        target_graph.y,
        target_graph.z,
        **target_style,
    )

    # Plot sensor path on ground-truth object.
    sm = SensorModuleData(stats["SM_0"], style=style)
    sm.plot_sensor_path(ax, steps=step + 1)

    """
    Second plot has first and second MLHs.
    """
    ax = axes[1]

    # Plot first and second MLHs.
    top_mlh["graph"] = get_graph_for_hypothesis(
        top_mlh, stats, step, pretrained_model="surf_agent_1lm"
    )
    top_mlh_style = extract_style(style, "top_mlh")
    if top_mlh_style.get("color") == "rgba":
        top_mlh_style["color"] = top_mlh["graph"].rgba
    ax.scatter(
        top_mlh["graph"].x,
        top_mlh["graph"].y,
        top_mlh["graph"].z,
        **top_mlh_style,
    )

    second_mlh["graph"] = get_graph_for_hypothesis(
        second_mlh, stats, step, pretrained_model="surf_agent_1lm"
    )
    second_mlh_style = extract_style(style, "second_mlh")
    if second_mlh_style.get("color") == "rgba":
        second_mlh_style["color"] = second_mlh["graph"].rgba
    ax.scatter(
        second_mlh["graph"].x,
        second_mlh["graph"].y,
        second_mlh["graph"].z,
        **second_mlh_style,
    )

    # Plot the goal state's target if possible.
    goal_states = get_goal_states(stats)
    gs = None
    for g in goal_states:
        if g["info"]["matching_step_when_output_goal_set"] == step:
            gs = g
    if gs:
        proposed_surface_loc = gs["info"]["proposed_surface_loc"]
        goal_style = extract_style(style, "goal")
        for ax in axes:
            ax.scatter(
                proposed_surface_loc[0],
                proposed_surface_loc[1],
                proposed_surface_loc[2],
                **goal_style,
            )
    return fig, axes


def get_visualization_legend_handles(
    labels: List[str],
    style: Mapping,
) -> List[Line2D]:
    style = update_style(STYLE, style)
    legend_handles = []
    for i, name in enumerate(("sensor_path.scatter", "top_mlh", "second_mlh", "goal")):
        st = extract_style(style, name)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=st["marker"],
                color="w",
                markerfacecolor=st["color"],
                markeredgecolor=st["color"],
                markersize=6,
                label=labels[i],
            )
        )

    return legend_handles


def plot_object_hypotheses_visualization():
    """Draw top two object MLHs, sensor paths, and goal the state point.

    Requires having run `fig6_hypothesis_driven_policy`.

    Output is saved to `DMC_ANALYSIS_DIR/fig6/object_hypotheses/visualization`.
    """
    out_dir = OUT_DIR / "object_hypotheses/visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    style = STYLE.copy()
    experiment = "fig6_hypothesis_driven_policy"
    episode = 0

    exp_dir = VISUALIZATION_RESULTS_DIR / experiment
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[episode]

    # Select a step where there's a goal state was achieved, and it was
    # for distuinguishing between different objects.
    goal_states = get_goal_states(stats, achieved=True)
    goal_states = [g for g in goal_states if not g["info"]["is_pose_hypothesis"]]
    gs = goal_states[0]
    step = gs["info"]["matching_step_when_output_goal_set"]

    # Get the pose MLHs.
    top_mlh, second_mlh = get_top_two_mlhs(stats, step)

    fig, axes = plot_hypotheses_visualization(
        stats,
        step,
        top_mlh,
        second_mlh,
        style=style,
    )

    # Add label, legends, etc.
    view_angles = [(-50, -180, 0), (-80, 180, 0)]
    for i, ax in enumerate(axes):
        ax.set_proj_type("persp", focal_length=0.8)
        axes3d_set_aspect_equal(ax)
        axes3d_clean(ax)
        ax.view_init(*view_angles[i])

    sm = SensorModuleData(stats["SM_0"], style=style)
    legend_handles = get_visualization_legend_handles(
        ["sensor", "spoon", "fork", "goal"], sm.style
    )
    axes[1].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.1, 0.8),
    )

    fig.savefig(out_dir / "object_hypothesis.png", bbox_inches="tight")
    fig.savefig(out_dir / "object_hypothesis.svg", bbox_inches="tight")
    plt.show()

    return fig, axes


def plot_pose_hypotheses_visualization():
    """Draw top two pose MLHs, sensor paths, and goal the state point.

    Requires having run `fig6_hypothesis_driven_policy`.

    Output is saved to `DMC_ANALYSIS_DIR/fig6/pose_hypotheses/visualization`.
    """
    out_dir = OUT_DIR / "pose_hypotheses/visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    style = update_style(STYLE, {"target.s": 4, "top_mlh.s": 4, "second_mlh.s": 4})
    experiment = "fig6_hypothesis_driven_policy"
    episode = 1

    exp_dir = VISUALIZATION_RESULTS_DIR / experiment
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[episode]

    # Select a step where there's a goal state was achieved, and it was
    # for pose estimation (object was already determined).
    goal_states = get_goal_states(stats, achieved=True)
    goal_states = [g for g in goal_states if g["info"]["is_pose_hypothesis"]]
    gs = goal_states[0]
    step = gs["info"]["matching_step_when_output_goal_set"]

    # Get the pose MLHs.
    mlh_graph_id = stats["LM_0"]["current_mlh"][step]["graph_id"]
    top_mlh, second_mlh = get_top_two_mlhs_for_object(mlh_graph_id, stats, step)
    fig, axes = plot_hypotheses_visualization(
        stats,
        step,
        top_mlh,
        second_mlh,
        style=style,
    )

    # Add label, legends, etc.
    view_angles = [(-37, 26, 0), (-7.50, 34.26, 0)]
    for i, ax in enumerate(axes):
        ax.set_proj_type("persp", focal_length=0.8)
        axes3d_set_aspect_equal(ax)
        axes3d_clean(ax)
        ax.view_init(*view_angles[i])

    # Add legend to the second plot.
    sm = SensorModuleData(stats["SM_0"], style=style)
    legend_handles = get_visualization_legend_handles(
        ["sensor", "pose 1", "pose 2", "goal"], sm.style
    )
    axes[1].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.1, 0.8),
        framealpha=1,
        fontsize=8,
    )

    fig.savefig(out_dir / "pose_hypothesis.png", bbox_inches="tight")
    fig.savefig(out_dir / "pose_hypothesis.svg", bbox_inches="tight")
    plt.show()

    return fig, axes


"""
--------------------------------------------------------------------------------
Panels D and F: Hypotheses Evidence
--------------------------------------------------------------------------------
"""


def set_steps_above_threshold(
    hypotheses: Iterable[Mapping],
    x_percent_threshold: float = 20.0,
) -> np.ndarray:
    """Set the `steps_above_threshold` field for each hypothesis.

    Helper function for `init_object_hypotheses` and `init_pose_hypotheses`.

    Args:
        hypotheses: Hypotheses to find steps above threshold for.
        x_percent_threshold: Percentage threshold for evidence. Default is 20.

    Returns:
        np.ndarray: The step-wise evidence threshold.
    """
    # Find highest evidence at each step to compute the evidence threshold.
    n_steps = hypotheses[0]["evidence"].shape[0]
    highest_evidences = np.zeros(n_steps)
    for step in range(n_steps):
        highest_evidences[step] = np.max([h["evidence"][step] for h in hypotheses])
    thresholds = highest_evidences * (1 - x_percent_threshold / 100)

    # Find the last step where evidence was above threshold for each hypothesis.
    for h in hypotheses:
        where_above_threshold = np.where(h["evidence"] >= thresholds)[0]
        if where_above_threshold.size == 0:
            h["steps_above_threshold"] = 0
        else:
            h["steps_above_threshold"] = where_above_threshold[-1] + 1

    return thresholds


def init_object_hypotheses(stats: Mapping) -> np.ndarray:
    """Initialize object hypotheses.

    Args:
        stats: Detailed stats for an episode.

    Returns:
        np.ndarray: An array of hypotheses, where each is a dictionary with the
          following keys:
            - `object_name`: The name of the object.
            - `label`: The label for the hypothesis.
            - `evidence`: The evidence for the hypothesis.
    """
    max_evidence_dicts: List[Mapping[str, float]] = stats["LM_0"]["evidences_max"]
    hypotheses = []
    for key in max_evidence_dicts[0].keys():
        h = {
            "object_name": key,
            "label": key,
            "evidence": np.array([dct[key] for dct in max_evidence_dicts]),
        }
        hypotheses.append(h)
    hypotheses = np.array(hypotheses, dtype=object)
    set_steps_above_threshold(hypotheses)
    return hypotheses


def init_pose_hypotheses(stats: Mapping, object_name: str) -> np.ndarray:
    """Initialize pose hypotheses.

    Args:
        stats: Detailed stats for an episode.
        object_name: The name of the object to initialize pose hypotheses for.

    Returns:
        np.ndarray: An array of hypotheses, where each is a dictionary with the
          following keys:
            - `rotation_matrix`: The rotation matrix for the hypothesis.
            - `evidence`: The evidence for the hypothesis.
            - `label`: The label for the hypothesis.
    """
    evidences = np.array([dct[object_name] for dct in stats["LM_0"]["evidences"]]).T
    rotations = stats["LM_0"]["possible_rotations"][0][object_name]
    hypotheses = []
    for i in range(len(rotations)):
        angles = R.from_matrix(rotations[i]).as_euler("xyz", degrees=True)
        if np.any(angles < 0):
            angles = (angles + 180) % 360
        label = f"({int(angles[0])}, {int(angles[1])}, {int(angles[2])})"
        h = {
            "rotation_matrix": rotations[i],
            "evidence": evidences[i],
            "label": label,
        }
        hypotheses.append(h)
    hypotheses = np.array(hypotheses, dtype=object)
    set_steps_above_threshold(hypotheses)
    return hypotheses


def plot_goal_state_steps(ax: plt.Axes, goal_states: List[Mapping]):
    for gs in goal_states:
        step = gs["info"]["matching_step_when_output_goal_set"]
        if gs["info"]["is_pose_hypothesis"]:
            color = "gray"
            ls = "--"
        else:
            color = "gray"
            ls = "-"
        ax.axvline(step, color=color, lw=1, linestyle=ls, alpha=1)


def plot_evidence_for_one_hypothesis(
    ax: plt.Axes,
    h: Mapping,
    skip_if_empty: bool = False,
) -> None:
    """Plot the evidence over time for a single hypothesis.

    Helper function for `plot_hypotheses_evidence` and
    """
    last_step = h["steps_above_threshold"]
    if last_step == 0 and skip_if_empty:
        return
    arr = h["evidence"][:last_step]
    ax.plot(arr, **h["style"])


def plot_hypotheses_evidence(
    ax: plt.Axes,
    hypotheses: Iterable[Mapping],
    n_top_hypotheses: int = 4,
    n_bottom_hypotheses: Optional[Number] = np.inf,
    bottom_hypothesis_color: str = "gray",
    bottom_hypothesis_alpha: float = 0.5,
    legend_kwargs: Optional[Mapping] = None,
) -> None:
    # Split hypotheses into two groups.
    top_hypotheses = hypotheses[:n_top_hypotheses]
    if n_bottom_hypotheses is None:
        bottom_hypotheses = []
    elif n_bottom_hypotheses is np.inf:
        bottom_hypotheses = hypotheses[n_top_hypotheses:]
    else:
        h_start = n_top_hypotheses
        h_stop = n_top_hypotheses + n_bottom_hypotheses
        bottom_hypotheses = hypotheses[h_start:h_stop]

    # Add style to hypotheses.
    for i, h in enumerate(top_hypotheses):
        h["style"] = {
            "color": HYPOTHESIS_COLORS[i],
            "alpha": 1,
            "label": h["label"],
        }
    for i, h in enumerate(bottom_hypotheses):
        h["style"] = {
            "color": bottom_hypothesis_color,
            "alpha": bottom_hypothesis_alpha,
            "label": None,
        }
        if i == 0:
            h["style"]["label"] = "others"

    # Plot in reverse order because zorder argument doesn't work.
    for i, h in enumerate(bottom_hypotheses[::-1]):
        can_skip = h["style"]["label"] is None
        plot_evidence_for_one_hypothesis(ax, h, skip_if_empty=can_skip)
    for h in top_hypotheses[::-1]:
        plot_evidence_for_one_hypothesis(ax, h, skip_if_empty=False)

    # Hack to get legend items in order since we had to plot in reverse order.
    legend_order = [h["style"]["label"] for h in top_hypotheses]
    if len(bottom_hypotheses) > 0:
        legend_order.append("others")
    cur_handles, cur_labels = ax.get_legend_handles_labels()
    handles, labels = [], []
    for lbl in legend_order:
        try:
            ind = cur_labels.index(lbl)
            handles.append(cur_handles[ind])
            labels.append(cur_labels[ind])
        except ValueError:
            pass
    legend_kwargs = legend_kwargs or {}
    ax.legend(handles, labels, **legend_kwargs)


def finalize_evidence_axes(
    ax: plt.Axes,
    jump_step: Optional[int] = None,
    **kw,
):
    # Set axes properties.
    for key, val in kw.items():
        getattr(ax, f"set_{key}")(val)

    # Add a triangle indicating where the jump was. Has to be done last.
    if jump_step is not None:
        draw_triangle(
            ax,
            jump_step,
            ax.get_ylim()[1],
            theta=180,
            radius=2,
            align="bottom",
            facecolor=TBP_COLORS["yellow"],
            edgecolor="black",
        )


def plot_object_hypotheses_evidence():
    """Plot evidence over time for object hypotheses.

    Requires having run `fig6_hypothesis_driven_policy`.

    Output is saved to `DMC_ANALYSIS_DIR/fig6/object_hypotheses/evidence`.
    """
    # Initialize ouput directory.
    out_dir = OUT_DIR / "object_hypotheses/evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize stats.
    episode = 0
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_hypothesis_driven_policy"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats[episode]

    # Get all object hypotheses.
    hypotheses = init_object_hypotheses(stats)
    n_steps = hypotheses[0]["evidence"].shape[0]

    # Collect match threshold info.
    # Sort by evidence at step of hypothesis-driven jump.
    goal_states = get_goal_states(stats)
    obj_goal_states = [g for g in goal_states if not g["info"]["is_pose_hypothesis"]]
    if len(obj_goal_states) == 0:
        order_step = n_steps - 1
    else:
        order_step = obj_goal_states[0]["info"]["matching_step_when_output_goal_set"]
    sort_by = [h["evidence"][order_step] for h in hypotheses]
    hypotheses = hypotheses[np.argsort(sort_by)[::-1]]

    fig, ax = plt.subplots(1, 1, figsize=(3.75, 2.5), dpi=300)
    plot_goal_state_steps(ax, goal_states)
    plot_hypotheses_evidence(
        ax,
        hypotheses,
        4,
        np.inf,
        legend_kwargs={"title": "Hypothesis", "fontsize": 8, "handlelength": 0.75},
    )
    # Finalize axes.
    finalize_evidence_axes(
        ax,
        jump_step=order_step,
        xlim=(0, 15),
        ylim=(0, 30),
        xticks=[0, 5, 10, 15],
        yticks=[0, 10, 20, 30],
        xlabel="Step",
        ylabel="Evidence",
    )

    fig.savefig(out_dir / "evidence.png", bbox_inches="tight")
    fig.savefig(out_dir / "evidence.svg", bbox_inches="tight")
    plt.show()


def plot_pose_hypotheses_evidence():
    """Plot evidence over time for pose hypotheses.

    Requires having run `fig6_hypothesis_driven_policy`.

    Output is saved to `DMC_ANALYSIS_DIR/fig6/pose_hypotheses/evidence`.
    """
    # Initialize ouput directory.
    out_dir = OUT_DIR / "pose_hypotheses/evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize stats.
    episode = 1
    object_name = "mug"
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_hypothesis_driven_policy"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats[episode]

    # Get all object hypotheses.
    hypotheses = init_pose_hypotheses(stats, object_name)
    n_steps = hypotheses[0]["evidence"].shape[0]

    # Collect match threshold info.
    # Sort by evidence at step of hypothesis-driven jump.
    goal_states = get_goal_states(stats)
    pose_goal_states = [g for g in goal_states if g["info"]["is_pose_hypothesis"]]
    if len(pose_goal_states) == 0:
        order_step = n_steps - 1
    else:
        order_step = pose_goal_states[0]["info"]["matching_step_when_output_goal_set"]
    sort_by = [h["evidence"][order_step] for h in hypotheses]
    hypotheses = hypotheses[np.argsort(sort_by)[::-1]]

    fig, ax = plt.subplots(1, 1, figsize=(3.75, 2.5), dpi=300)
    plot_goal_state_steps(ax, goal_states)
    plot_hypotheses_evidence(
        ax,
        hypotheses,
        4,
        np.inf,
        legend_kwargs={"title": "Hypothesis", "fontsize": 8, "handlelength": 0.75},
    )

    # Finalize axes.
    finalize_evidence_axes(
        ax,
        jump_step=order_step,
        xlim=(0, 20),
        xticks=[0, 5, 10, 15, 20],
        ylim=(0, 30),
        yticks=[0, 10, 20, 30],
        xlabel="Step",
        ylabel="Evidence",
    )
    fig.savefig(out_dir / "evidence.png", bbox_inches="tight")
    fig.savefig(out_dir / "evidence.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_curvature_guided_policy()
    plot_performance()
    plot_object_hypotheses_visualization()
    plot_object_hypotheses_evidence()
    plot_pose_hypotheses_visualization()
    plot_pose_hypotheses_evidence()
