# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Get overview plots for DMC experiments.

This script generates basic figures for each set of experiments displaying number
of monty matching steps, accuracy, and rotation error. If functions are called with
`save=True`, figures and tables are saved under `DMC_ANALYSIS_DIR / overview`.
"""

import os
from numbers import Number
from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    ObjectModel,
    describe_dict,
    get_percent_correct,
    load_eval_stats,
    load_object_model,
)
from plot_utils import TBP_COLORS, axes3d_clean, axes3d_set_aspect_equal
from scipy.spatial.transform import Rotation as R
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.utils.logging_utils import get_pose_error

# from tbp.monty.frameworks.models.object_model import GraphObjectModel

plt.rcParams["font.size"] = 8

# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

"""
--------------------------------------------------------------------------------
Panel A: Sensor path and known objects
--------------------------------------------------------------------------------
"""


def plot_sensor_path():
    """Plot the sensor path for panel A."""
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    mug = load_object_model("dist_agent_1lm_10distinctobj", "mug")
    raw_observations = stats["SM_0"]["raw_observations"]  # a list of dicts
    n_steps = 36

    # Extract the (central) locations of each observation.
    n_rows = n_cols = 64
    center_loc = n_rows // 2 * n_cols + n_cols // 2
    centers = np.zeros((n_steps, 3))
    for i in range(n_steps):
        arr = np.array(raw_observations[i]["semantic_3d"])
        centers[i, 0] = arr[center_loc, 0]
        centers[i, 1] = arr[center_loc, 1]
        centers[i, 2] = arr[center_loc, 2]

    out_dir = OUT_DIR / "sensor_path"
    out_dir.mkdir(parents=True, exist_ok=True)

    def init_plot(
        observed_points: bool = False,
        path: bool = False,
        path_labels: bool = False,
    ):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection="3d")
        linewidths = np.zeros(len(mug.x))
        ax.scatter(
            mug.x,
            mug.y,
            mug.z,
            color=mug.rgba,
            alpha=0.5,
            linewidths=linewidths,
            zorder=1,
            s=5,
        )
        ax.view_init(115, -90, 0)
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        blue = TBP_COLORS["blue"]
        x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
        if observed_points:
            # z += 0.005
            ax.scatter(
                x,
                y,
                z,
                color="k",
                edgecolors="k",
                alpha=1,
                zorder=5,
                marker="s",
                s=8,
                linewidths=1,
            )
        if path:
            ax.plot(x, y, z, color=blue, alpha=1, ls="--", lw=1)
        if path_labels:
            for i in range(len(x)):
                if i % 10 == 0:
                    ax.text(x[i], y[i], z[i], f"{i}", color="k", alpha=1, zorder=5)
        return fig, ax

    fig, ax = init_plot()
    fig.savefig(out_dir / "mug.png", dpi=300)
    fig.savefig(out_dir / "mug.svg")
    plt.show()

    fig, ax = init_plot(observed_points=True)
    fig.savefig(out_dir / "mug_with_points.png", dpi=300)
    fig.savefig(out_dir / "mug_with_points.svg")
    plt.show()

    fig, ax = init_plot(observed_points=True, path=True)
    fig.savefig(out_dir / "mug_with_path.png", dpi=300)
    fig.savefig(out_dir / "mug_with_path.svg")
    plt.show()

    fig, ax = init_plot(observed_points=True, path=True, path_labels=True)
    fig.savefig(out_dir / "mug_with_labels.png", dpi=300)
    fig.savefig(out_dir / "mug_with_labels.svg")
    plt.show()


def plot_known_objects():
    """Plot the "known objects" for panel A."""
    fig, axes = plt.subplots(1, 3, figsize=(5, 4), subplot_kw={"projection": "3d"})
    mug = load_object_model("dist_agent_1lm", "mug")
    bowl = load_object_model("dist_agent_1lm", "bowl")
    golf_ball = load_object_model("dist_agent_1lm", "golf_ball")

    axes[0].scatter(mug.x, mug.y, mug.z, color=mug.rgba, alpha=0.5, s=5, linewidth=0)
    axes[1].scatter(
        bowl.x, bowl.y, bowl.z, color=bowl.rgba, alpha=0.5, s=5, linewidth=0
    )
    axes[2].scatter(
        golf_ball.x,
        golf_ball.y,
        golf_ball.z,
        color=golf_ball.rgba,
        alpha=0.5,
        s=5,
        linewidth=0,
    )

    for ax in axes:
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        ax.view_init(115, -90, 0)

    fig.savefig(OUT_DIR / "known_objects.png", dpi=300)
    fig.savefig(OUT_DIR / "known_objects.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel B: Evidence graphs
--------------------------------------------------------------------------------
"""


def plot_evidence_graphs_and_patches():
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    object_names = ["mug", "bowl", "golf_ball"]
    steps = np.array([0, 10, 20, 39, 40])
    steps = np.arange(41)
    n_steps = len(steps)

    # steps = np.arange(0, 40)
    # steps = np.arange(0, 41, 20)
    n_rows = n_cols = 64
    center_loc = n_rows // 2 * n_cols + n_cols // 2
    centers = np.zeros((n_steps, 3))
    for i in range(n_steps):
        arr = np.array(stats["SM_0"]["raw_observations"][i]["semantic_3d"])
        centers[i, 0] = arr[center_loc, 0]
        centers[i, 1] = arr[center_loc, 1]
        centers[i, 2] = arr[center_loc, 2]

    # Extract evidence values for all objects.
    all_evidences = stats["LM_0"]["evidences"]
    all_possible_locations = stats["LM_0"]["possible_locations"]
    all_possible_rotations = stats["LM_0"]["possible_rotations"]
    objects = {
        name: load_object_model("dist_agent_1lm_10distinctobj", name)
        for name in object_names
    }
    for name, obj in objects.items():
        obj.evidences, obj.locations = [], []
        for i in range(n_steps):
            obj.evidences.append(all_evidences[i][name])
            obj.locations.append(all_possible_locations[i][name])
        obj.evidences = np.array(obj.evidences)
        obj.locations = np.array(obj.locations)
        obj.rotation = np.array(all_possible_rotations[0][name])

    # Plot evidence graphs for each object and step individually.
    out_dir = OUT_DIR / "evidence_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(steps):
        fig, axes = plt.subplots(
            1, len(objects) + 1, figsize=(12, 6), subplot_kw=dict(projection="3d")
        )

        # Plot mug with current observation location.
        ax = axes[0]
        mug = objects["mug"]
        ax.scatter(
            mug.x,
            mug.y,
            mug.z,
            color="gray",
            alpha=0.1,
            s=1,
            linewidths=0,
        )
        center = centers[step]
        ax.scatter(center[0], center[1], center[2], color="red", s=50)
        ax.view_init(100, -100, -10)
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        ax.set_title(f"step {step}")
        ax.set_xlim(-0.119, 0.119)
        ax.set_ylim(1.5 - 0.119, 1.5 + 0.119)
        ax.set_zlim(-0.119, 0.119)

        # Make colormap for this step.
        all_evidences = [obj.evidences[step].flatten() for obj in objects.values()]
        all_evidences = np.concatenate(all_evidences)
        evidences_min = np.percentile(all_evidences, 2.5)
        evidences_max = np.percentile(all_evidences, 99.99)
        scalar_map = plt.cm.ScalarMappable(
            cmap="inferno", norm=plt.Normalize(vmin=evidences_min, vmax=evidences_max)
        )

        for j, obj in enumerate(objects.values()):
            ax = axes[j + 1]
            # ax.scatter(
            #     obj.x,
            #     obj.y,
            #     obj.z,
            #     color="gray",
            #     alpha=0.1,
            #     s=1,
            #     linewidths=0,
            # )

            locations = obj.locations[step]
            n_points = locations.shape[0] // 2
            locations = locations[:n_points]
            evidences = obj.evidences[step]
            ev1 = evidences[:n_points]
            ev2 = evidences[n_points:]
            stacked = np.hstack([ev1[:, np.newaxis], ev2[:, np.newaxis]])
            evidences = stacked.max(axis=1)

            colors = evidences
            sizes = np.log(evidences - evidences.min() + 1) * 10
            alphas = np.array(evidences)
            alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
            x, y, z = locations[:, 0], locations[:, 1], locations[:, 2]
            ax.scatter(
                x,
                y,
                z,
                c=colors,
                cmap="inferno",
                alpha=alphas,
                vmin=evidences_min,
                vmax=evidences_max,
                s=sizes,
                linewidths=0,
            )

            # Plot highest evidence location separately.
            ind_ev_max = evidences.argmax()
            ev = evidences[ind_ev_max]
            x, y, z = x[ind_ev_max], y[ind_ev_max], z[ind_ev_max]
            sizes = sizes[ind_ev_max]
            ax.scatter(
                x,
                y,
                z,
                c=ev,
                cmap="inferno",
                alpha=1,
                vmin=evidences_min,
                vmax=evidences_max,
                s=sizes,
                linewidths=0,
            )

            ax.view_init(100, -100, -10)
            axes3d_clean(ax, grid=False)
            axes3d_set_aspect_equal(ax)
            ax.set_title(f"step {step}")
            ax.set_xlim(-0.119, 0.119)
            ax.set_ylim(1.5 - 0.119, 1.5 + 0.119)
            ax.set_zlim(-0.119, 0.119)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
        fig.savefig(png_dir / f"evidence_graphs_{step}.png", dpi=300)
        fig.savefig(svg_dir / f"evidence_graphs_{step}.svg")
        plt.show()
        plt.close(fig)

    # Plot the colorbar.
    fig, ax = plt.subplots(1, 1, figsize=(1, 2))
    cbar = plt.colorbar(scalar_map, ax=ax, orientation="vertical", label="Evidence")
    ax.remove()  # Remove the empty axes, we just want the colorbar
    cbar.set_ticks([])
    cbar.set_label("")
    fig.tight_layout()
    fig.savefig(out_dir / "colorbar.png", dpi=300)
    fig.savefig(out_dir / "colorbar.svg")

    # Extract RGBA patches for sensor module 0.
    rgba_patches = []
    for ind in range(n_steps):
        rgba_patches.append(np.array(stats["SM_0"]["raw_observations"][ind]["rgba"]))
    rgba_patches = np.array(rgba_patches)

    # Save the RGBA patches.
    out_dir = OUT_DIR / "patches"
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        patch = rgba_patches[step]
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(patch)
        ax.axis("off")
        fig.tight_layout(pad=0.0)
        fig.savefig(png_dir / f"patch_step_{step}.png", dpi=300)
        fig.savefig(svg_dir / f"patch_step_{step}.svg")
        plt.close(fig)


"""
--------------------------------------------------------------------------------
Panel C: ?
--------------------------------------------------------------------------------
"""


def plot_accuracy_and_steps():
    out_dir = OUT_DIR / "accuracy_and_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataframes = [
        load_eval_stats("dist_agent_1lm"),
        load_eval_stats("dist_agent_1lm_noise"),
        load_eval_stats("dist_agent_1lm_randrot_all"),
        load_eval_stats("dist_agent_1lm_randrot_all_noise"),
    ]
    conditions = ["base", "noise", "RR", "noise + RR"]

    percent_correct = [get_percent_correct(df) for df in dataframes]
    rotation_errors = [np.rad2deg(df.rotation_error.dropna()) for df in dataframes]

    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2))

    # Plot accuracy bars
    ax1.bar(
        [0, 2, 4, 6],
        percent_correct,
        color=TBP_COLORS["blue"],
        width=0.8,
    )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("% Correct")

    # Plot rotation error violins
    ax2 = ax1.twinx()
    vp = ax2.violinplot(
        rotation_errors,
        positions=[1, 3, 5, 7],
        showextrema=False,
        showmedians=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor(TBP_COLORS["pink"])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylim(0, 180)
    ax2.set_ylabel("Error (deg)")

    ax1.set_xticks([0.5, 2.5, 4.5, 6.5])
    ax1.set_xticklabels(conditions, rotation=0, ha="center")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_and_steps.png", dpi=300)
    fig.savefig(out_dir / "accuracy_and_steps.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel D: Symmetrical rotations
--------------------------------------------------------------------------------
"""


def get_l2_distance(
    pc1: Union[np.ndarray, ObjectModel],
    pc2: Union[np.ndarray, ObjectModel],
) -> float:
    """
    Computes the L2 Distance between two point clouds.

    Parameters:
    pc1, pc2 : np.ndarray of shape (N, 3) - Two point clouds with the same number of points.

    Returns:
    float : Chamfer distance.
    """
    pc1 = pc1.pos if isinstance(pc1, ObjectModel) else pc1
    pc2 = pc2.pos if isinstance(pc2, ObjectModel) else pc2

    return np.mean(np.linalg.norm(pc1 - pc2, axis=1))


def get_emd_distance(
    pc1: Union[np.ndarray, ObjectModel],
    pc2: Union[np.ndarray, ObjectModel],
) -> float:
    """
    Computes the Earth Mover's Distance (EMD) between two point clouds.

    Parameters:
    pc1, pc2 : np.ndarray of shape (N, 3) - Two point clouds with the same number of points.

    Returns:
    float : EMD distance.
    """
    pc1 = pc1.pos if isinstance(pc1, ObjectModel) else pc1
    pc2 = pc2.pos if isinstance(pc2, ObjectModel) else pc2
    # Compute pairwise distances
    cost_matrix = scipy.spatial.distance.cdist(pc1, pc2)
    # Solve transport problem
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(pc1)


def get_chamfer_distance(
    pc1: Union[np.ndarray, ObjectModel],
    pc2: Union[np.ndarray, ObjectModel],
) -> float:
    """
    Computes the Chamfer Distance between two point clouds.

    Parameters:
    pc1, pc2 : np.ndarray of shape (N, 3) - Two point clouds with the same number of points.

    Returns:
    float : Chamfer distance.
    """
    pc1 = pc1.pos if isinstance(pc1, ObjectModel) else pc1
    pc2 = pc2.pos if isinstance(pc2, ObjectModel) else pc2

    dists1 = np.min(scipy.spatial.distance.cdist(pc1, pc2), axis=1)
    dists2 = np.min(scipy.spatial.distance.cdist(pc2, pc1), axis=1)
    return np.mean(dists1) + np.mean(dists2)


def load_symmetry_rotations(episode_stats: Mapping) -> List[SimpleNamespace]:
    """Load symmetric rotations.

    Returns:
        List[SimpleNamespace]: A list of SimpleNamespace objects, each containing
        the id, rotation, and evidence value.
    """
    # Get all evidence vals.
    evidences_ls = episode_stats["LM_0"]["evidences_ls"]
    mlh_object = list(evidences_ls.keys())[0]
    evidences_ls = np.array(evidences_ls[mlh_object])

    # Get all symmetric rotations.
    symmetric_rotations = np.array(episode_stats["LM_0"]["symmetric_rotations"])
    n_rotations = len(symmetric_rotations)

    # Find evidence values associated with each rotation.
    sorting_inds = np.argsort(evidences_ls)[::-1]
    evidences_sorted = evidences_ls[sorting_inds]
    ev_threshold = np.mean(evidences_sorted[n_rotations - 1 : n_rotations + 1])
    evidences_ls = evidences_ls[evidences_ls >= ev_threshold]
    assert len(evidences_ls) == n_rotations
    rotations = []
    for i in range(len(symmetric_rotations)):
        rotations.append(
            SimpleNamespace(
                id=i,
                rot=R.from_matrix(symmetric_rotations[i]).inv(),
                evidence=evidences_ls[i],
            )
        )

    return rotations


def get_relative_rotation(
    rot_a: scipy.spatial.transform.Rotation,
    rot_b: scipy.spatial.transform.Rotation,
    degrees: bool = False,
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


def plot_symmetrical_rotations_qualitative(episode: int):
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    episode_params = {
        6: {
            "other_index": 5,
            "random_rotation": np.array([98, 86, 134]),
            "elev": 30,
            "azim": -90,
        },
        130: {
            "other_index": 5,
            "random_rotation": np.array([98, 86, 134]),
            "elev": 30,
            "azim": -10,
        },
        309: {
            "other_index": 17,
            "random_rotation": np.array([172, 68, -25]),
            "elev": 30,
            "azim": -90,
        },
    }

    params = episode_params.get(episode, {})

    row = eval_stats.iloc[episode]
    primary_target_object = row.primary_target_object
    target = SimpleNamespace(
        rot=R.from_euler("xyz", row.primary_target_rotation_euler, degrees=True)
    )

    # Load rotations, compute rotation error for each, and sort them by error.
    episode_stats = detailed_stats[episode]
    rotations = load_symmetry_rotations(episode_stats)
    for r in rotations:
        theta, axis = get_relative_rotation(r.rot, target.rot, degrees=True)
        r.theta = theta
        r.axis = axis
    rotations = sorted(rotations, key=lambda x: x.theta)

    # Get rotation with lowest error and any other symmetrical rotation.
    best = rotations[0]
    other_index = params.get("other_index", np.random.randint(1, len(rotations)))
    other = rotations[other_index]

    # Get a random rotation, and compute its error.
    random_rotation = params.get(
        "random_rotation", np.random.randint(0, 360, size=(3,))
    )
    rot_random = R.from_euler("xyz", random_rotation, degrees=True)
    random = SimpleNamespace(rot=rot_random)
    theta, axis = get_relative_rotation(random.rot, target.rot, degrees=True)
    random.theta = theta
    random.axis = axis

    base_model = load_object_model("dist_agent_1lm", primary_target_object)
    base_model = base_model.centered()

    target.model = base_model.rotated(target.rot)
    best.model = base_model.rotated(best.rot)
    other.model = base_model.rotated(other.rot)
    random.model = base_model.rotated(random.rot)

    fig, axes = plt.subplots(2, 3, figsize=(5, 4), subplot_kw={"projection": "3d"})
    objects = [best, other, random]
    elev, azim = params.get("elev", 30), params.get("azim", -90)
    for i in range(3):
        obj = objects[i]

        # Plot object.
        ax = axes[0, i]
        model = obj.model
        ax.scatter(
            model.x,
            model.y,
            model.z,
            color=model.rgba,
            alpha=0.5,
            edgecolors="none",
            s=10,
        )
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(elev, azim)

        # Plot basis vectors.
        ax = axes[1, i]
        mat = obj.rot.as_matrix()
        origin = np.array([0, 0, 0])
        colors = ["red", "green", "blue"]
        axis_names = ["x", "y", "z"]
        for i in range(3):
            ax.quiver(
                *origin,
                *mat[:, i],
                color=colors[i],
                length=1,
                arrow_length_ratio=0.2,
                normalize=True,
            )
            getattr(ax, f"set_{axis_names[i]}lim")([-1, 1])
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(elev, azim)
        ax.axis("off")

    plt.show()
    out_dir = OUT_DIR / "symmetrical_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{primary_target_object}_{episode}.png", dpi=300)
    fig.savefig(out_dir / f"{primary_target_object}_{episode}.svg")
    plt.close()


def plot_symmetrical_rotations_qualitative_all() -> None:
    for episode in (6, 130, 309):
        plot_symmetrical_rotations_qualitative(episode)


def plot_symmetrical_distances():
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    stat_arrays = {
        "L2": {"best": [], "other": [], "random": []},
        "EMD": {"best": [], "other": [], "random": []},
        "Chamfer": {"best": [], "other": [], "random": []},
    }

    for episode, episode_stats in enumerate(detailed_stats):
        sym_rots = episode_stats["LM_0"]["symmetric_rotations"]
        if sym_rots is None or len(sym_rots) < 2:
            continue
        print(f"Episode {episode}")
        experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"
        detailed_stats = DetailedJSONStatsInterface(
            experiment_dir / "detailed_run_stats.json"
        )
        eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

        row = eval_stats.iloc[episode]
        primary_target_object = row.primary_target_object
        target = SimpleNamespace(
            rot=R.from_euler("xyz", row.primary_target_rotation_euler, degrees=True)
        )

        # Load rotations, compute rotation error for each, and sort them by error.
        rotations = load_symmetry_rotations(episode_stats)
        for r in rotations:
            theta, axis = get_relative_rotation(r.rot, target.rot, degrees=True)
            r.theta = theta
            r.axis = axis
        rotations = sorted(rotations, key=lambda x: x.theta)

        # Get rotation with lowest error and any other symmetrical rotation.
        best = rotations[0]
        other_index = np.random.randint(1, len(rotations))
        other = rotations[other_index]

        # Get a random rotation, and compute its error.
        random_rotation = np.random.randint(0, 360, size=(3,))
        random = R.from_euler("xyz", random_rotation, degrees=True)
        random = SimpleNamespace(rot=random)
        theta, axis = get_relative_rotation(random.rot, target.rot, degrees=True)
        random.theta = theta
        random.axis = axis

        # Get object models, rotated accordingly.
        base_model = load_object_model("dist_agent_1lm", primary_target_object)
        base_model = base_model.centered()
        target.model = base_model.rotated(target.rot)
        best.model = base_model.rotated(best.rot)
        other.model = base_model.rotated(other.rot)
        random.model = base_model.rotated(random.rot)

        objects = {"best": best, "other": other, "random": random}
        metrics = {
            "L2": get_l2_distance,
            "EMD": get_emd_distance,
            "Chamfer": get_chamfer_distance,
        }
        for metric_name, metric_func in metrics.items():
            for obj_name, obj in objects.items():
                stat_arrays[metric_name][obj_name].append(
                    metric_func(obj.model, target.model)
                )

    for metric_name in stat_arrays:
        for obj_name in stat_arrays[metric_name]:
            arr = np.array(stat_arrays[metric_name][obj_name])
            print(
                f"{metric_name} {obj_name}: min={arr.min():.4f}, max={arr.max():.4f}, "
                + f"mean={arr.mean():.4f}, median={np.median(arr):.4f}"
            )
            stat_arrays[metric_name][obj_name] = np.array(arr)

    metric_names = ["L2", "EMD", "Chamfer"]
    object_names = ["best", "other", "random"]
    colors = [TBP_COLORS["blue"], TBP_COLORS["pink"], TBP_COLORS["green"]]

    fig, ax = plt.subplots(1, 3, figsize=(4, 2))
    for i, ax in enumerate(ax):
        array_dict = stat_arrays[metric_names[i]]
        arrays = [array_dict[name] for name in object_names]
        ymax = max(np.percentile(arr, 95) for arr in arrays)
        vp = ax.violinplot(
            arrays,
            showextrema=False,
            showmedians=True,
        )
        for j, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[j])
            body.set_alpha(1.0)
        vp["cmedians"].set_color("black")
        ax.set_title(metric_names[i])
        ax.set_xticks(list(range(1, len(object_names) + 1)))
        ax.set_xticklabels(object_names, rotation=45)
        ax.set_ylim(0, ymax)
    fig.tight_layout()
    plt.show()
    fig.savefig(OUT_DIR / "distances.png", dpi=300)
    fig.savefig(OUT_DIR / "distances.svg")


"""
--------------------------------------------------------------------------------
Exploratory (OK to delete or archive later)
--------------------------------------------------------------------------------
"""


def get_pairwise_relative_rotations(
    rotations: List[SimpleNamespace],
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the relative rotations between all pairs of rotations.

    Args:
        rotations (List[SimpleNamespace]): A list of SimpleNamespace objects,
        each containing the id, rotation, and evidence value.
    """
    n_rotations = len(rotations)
    theta_matrix = np.zeros((n_rotations, n_rotations))
    axis_matrix = np.zeros((n_rotations, n_rotations, 3))
    for i in range(n_rotations - 1):
        for j in range(i + 1, n_rotations):
            rot_a, rot_b = rotations[i].rot, rotations[j].rot
            theta, axis = get_relative_rotation(rot_a, rot_b, degrees=True)
            theta_matrix[i, j] = theta
            axis_matrix[i, j] = axis
    return theta_matrix, axis_matrix


def group_rotations_by_symmetry(
    rotations: List[SimpleNamespace],
    group_threshold: float = 20.0,
) -> Tuple[List[SimpleNamespace], List[SimpleNamespace]]:
    """Groups the rotations into two groups based on the rotation difference.

    Args:
        rotations (List[SimpleNamespace]): A list of SimpleNamespace objects,
        each containing the id, rotation, and evidence value.


    """
    # Group rotations
    theta_matrix, _ = get_pairwise_relative_rotations(rotations)
    group_a_inds = np.where(abs(theta_matrix[0]) <= group_threshold)[0]
    group_b_inds = np.where(abs(theta_matrix[0] - 180) <= group_threshold)[0]

    # Replace indices with rotation objects.
    group_a = []
    for ind in group_a_inds:
        r = rotations[ind]
        r.group = "a"
        group_a.append(r)

    group_b = []
    for ind in group_b_inds:
        r = rotations[ind]
        r.group = "b"
        group_b.append(r)

    return group_a, group_b


def plot_symmetrical_rotations_overview(
    episode: int,
    max_rotations: int = 100,
    group_threshold: Number = 20,
    show: bool = False,
    episode_stats: Optional[Mapping] = None,
    eval_stats: Optional[pd.DataFrame] = None,
) -> "matplotlib.figure.Figure":
    """Plots the rotations of the object for a given episode."""

    # Load rotations, evidence, and episode info.
    # ------------------------------------------

    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"

    # - Extract info from 'eval_stats.csv'.
    if eval_stats is None:
        eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")
    row = eval_stats.iloc[episode]
    primary_target_object = row.primary_target_object
    primary_target_rotation = row.primary_target_rotation_euler
    detected_rotation = row.detected_rotation
    most_likely_rotation = row.most_likely_rotation

    # - Load rotations. We place the scipy rotation objects inside a SimpleNamespace
    # object to bind it to an evidence value and its insex.

    if episode_stats is None:
        detailed_stats = DetailedJSONStatsInterface(
            experiment_dir / "detailed_run_stats.json"
        )
        episode_stats = detailed_stats[episode]
    rotations = load_symmetry_rotations(episode_stats)

    # - Optionally, only look at the top `n` rotations, ranked by evidence.
    if max_rotations and len(rotations) > max_rotations:
        evidence_values = np.array([r.evidence for r in rotations])
        sorting_inds = np.argsort(evidence_values)[::-1]
        rotations = [rotations[ind] for ind in sorting_inds][:max_rotations]

    print(f"\nEpisode {episode}\n-----------")
    print(f" - primary target object: {primary_target_object}")
    print(f" - primary target rotation: {primary_target_rotation}")
    print(f" - detected rotation: {detected_rotation}")
    print(f" - most likely rotation: {most_likely_rotation}")
    print(f" - num. rotations: {len(rotations)}")

    # Partition rotations into two symmetrical groups.
    # ------------------------------------------------
    # - Find relative rotations between all rotations.
    group_a, group_b = group_rotations_by_symmetry(rotations, group_threshold)
    if len(group_b) == 0:
        print(" - No 180 degree rotations found. Returning.")
        return None
    print(f" - Num. rotations per group: a={len(group_a)}, b={len(group_b)}")

    # - Sort rotations within each group by evidence.
    group_a = sorted(group_a, key=lambda x: x.evidence, reverse=True)
    group_b = sorted(group_b, key=lambda x: x.evidence, reverse=True)

    # - Have group_a contain the rotation with the highest evidence/mlh.
    if max([r.evidence for r in group_b]) > max([r.evidence for r in group_a]):
        group_a, group_b = group_b, group_a

    # Draw objects models
    # --------------------------------
    init_elev, init_azim = 30, -90

    # Load the object model.
    base_obj = load_object_model("dist_agent_1lm", primary_target_object)
    base_obj = base_obj.centered()

    # Get target rotation (no inversion needed), and rotate an object model with it.
    true_rotation = R.from_euler("xyz", primary_target_rotation, degrees=True)
    true_obj = base_obj.rotated(true_rotation)

    # Initialize figure. The top row is for true rotation and the axes of relative
    # rotation between group_a and group_b. The next three rows are for objects
    # from group_a, group_b, and random rotations, respectively.
    n_rows, n_cols = 4, min([4, len(group_a), len(group_b)])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        subplot_kw={"projection": "3d"},
    )
    axes = axes[:, np.newaxis] if n_cols == 1 else axes

    # - Draw the object at the target rotation.
    ax = axes[0, 0]
    ax.scatter(true_obj.x, true_obj.y, true_obj.z, color=true_obj.rgba, alpha=0.5)
    ax.set_title("True Rotation")
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)
    ax.view_init(init_elev, init_azim)

    # - Draw the rotation axes between group_a and group_b (if we have room).
    if n_cols > 1:
        ax = axes[0, 1]
        a, b = group_a[0], group_b[0]
        rel_rot = a.rot * b.rot.inv()
        rel_rot = rel_rot * true_rotation
        rel_mat = rel_rot.as_matrix()
        origin = np.array([0, 0, 0])
        colors = ["red", "green", "blue"]
        axis_names = ["x", "y", "z"]
        for i in range(3):
            ax.quiver(
                *origin,
                *rel_mat[:, i],
                color=colors[i],
                length=1,
                arrow_length_ratio=0.2,
            )
            getattr(ax, f"set_{axis_names[i]}lim")([-1, 1])
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(init_elev, init_azim)
        ax.set_title("Relative Rotation Axis")
        for ax in axes[0, 2:]:
            ax.remove()

    # - Draw object models with group_a and group_b rotations.
    for i, group in enumerate([group_a, group_b]):
        for j in range(n_cols):
            r = group[j]
            obj = base_obj.rotated(r.rot)
            ax = axes[i + 1, j]
            ax.scatter(obj.x, obj.y, obj.z, color=obj.rgba, alpha=0.5, marker="o")
            l2 = get_l2_distance(obj, true_obj)
            emd = get_emd_distance(obj, true_obj)
            chamfer = get_chamfer_distance(obj, true_obj)
            ax.set_title(
                f"ID={r.id}: Evidence: {r.evidence:.2f}\nL2: {l2:.4f}\n"
                + f"EMD: {emd:.4f}\nChamfer: {chamfer:.4f}"
            )
            axes3d_clean(ax)
            axes3d_set_aspect_equal(ax)
            ax.view_init(init_elev, init_azim)

    # - Draw object models with random rotations.
    random_rots = [
        R.from_euler("xyz", np.random.randint(0, 360, size=(3,)), degrees=True)
        for _ in range(n_cols)
    ]
    for j in range(len(random_rots)):
        obj = base_obj.rotated(random_rots[j])
        ax = axes[3, j]
        ax.scatter(obj.x, obj.y, obj.z, color=obj.rgba, alpha=0.5)
        l2 = get_l2_distance(obj, true_obj)
        emd = get_emd_distance(obj, true_obj)
        chamfer = get_chamfer_distance(obj, true_obj)
        ax.set_title(f"L2: {l2:.4f}\nEMD: {emd:.4f}\nChamfer: {chamfer:.4f}")
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(init_elev, init_azim)

    title = f"Episode {episode}: '{primary_target_object}' at {primary_target_rotation}"
    fig.suptitle(title)

    if show:
        plt.show()
    return fig


def run_plot_symmetrical_rotations_overview():
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")
    maybe_usable_episodes = []
    for i, stats in enumerate(detailed_stats):
        if "last_hypotheses_evidence" not in stats["LM_0"]:
            continue
        maybe_usable_episodes.append(i)

    maybe_usable_episodes = np.array(maybe_usable_episodes)
    unusable_episodes = [9]
    highest_completed_episode = 0

    episodes_to_plot = np.setdiff1d(maybe_usable_episodes, unusable_episodes)
    episodes_to_plot = episodes_to_plot[episodes_to_plot > highest_completed_episode]

    out_dir = OUT_DIR / "symmetrical_rotations_overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode, episode_stats in enumerate(detailed_stats):
        if episode not in episodes_to_plot:
            continue
        try:
            fig = plot_symmetrical_rotations_overview(
                episode, episode_stats=episode_stats, eval_stats=eval_stats
            )
        except Exception as e:
            print(f"Error plotting episode {episode}: {e}")
            unusable_episodes.append(episode)
            raise
        if fig is None:
            unusable_episodes.append(episode)
            continue

        fig.savefig(out_dir / f"rotations_{episode}.png", dpi=300)
        plt.close()


def plot_mlh_vs_min_error():
    """Exploratory plotting to visualize rotation error and symmetry."""
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    mlh_rotations = []
    min_error_rotations = []
    target_names = []
    target_rotations = []
    differences = []
    episode_nums = []
    # - Extract info from 'eval_stats.csv'.
    # episode = 2
    # episode_stats = detailed_stats[episode]

    for episode, episode_stats in enumerate(detailed_stats):
        if "last_hypotheses_evidence" not in episode_stats["LM_0"]:
            continue
        rotations = load_symmetry_rotations(episode_stats)
        row = eval_stats.iloc[episode]
        primary_target_object = row.primary_target_object
        primary_target_rotation = row.primary_target_rotation_euler
        primary_target_rotation_r = R.from_euler(
            "xyz", primary_target_rotation, degrees=True
        )

        # - Load rotations. We place the scipy rotation objects inside a SimpleNamespace
        # object to bind it to an evidence value and its index.
        # rotations = load_symmetry_rotations(episode_stats)
        # group_a, group_b = group_rotations_by_symmetry(rotations)
        # rotations = group_a + group_b
        rotations = sorted(rotations, key=lambda x: x.evidence, reverse=True)

        for r in rotations:
            error_1 = get_pose_error(
                r.rot.as_quat(), primary_target_rotation_r.as_quat()
            )
            r.error_1 = np.degrees(error_1)
            error_2, _ = get_relative_rotation(
                r.rot, primary_target_rotation_r, degrees=True
            )
            r.error_2 = error_2
            r.error = error_1

        # compute the difference between the mlh and min error rotations.
        errors = np.array([r.error for r in rotations])
        mlh_r = rotations[0].rot
        min_error_r = rotations[np.argmin(errors)].rot
        theta, _ = get_relative_rotation(mlh_r, min_error_r, degrees=True)
        differences.append(theta)

        episode_nums.append(episode)
        target_names.append(primary_target_object)
        mlh_rotations.append(mlh_r)
        min_error_rotations.append(min_error_r)
        target_rotations.append(primary_target_rotation_r)

        print(f"Episode {episode}: '{primary_target_object}'")
        print(f" delta = {theta:.2f} deg")
        if theta > 20:
            base_obj = load_object_model("dist_agent_1lm", primary_target_object)
            base_obj = base_obj.centered()

            true_obj = base_obj.rotated(primary_target_rotation_r)
            mlh_obj = base_obj.rotated(mlh_r)
            min_error_obj = base_obj.rotated(min_error_r)

            fig, axes = plt.subplots(
                1, 3, figsize=(5, 2), subplot_kw={"projection": "3d"}
            )
            titles, objects = (
                ["True", "MLH", "Min Error"],
                [true_obj, mlh_obj, min_error_obj],
            )
            for j, ax in enumerate(axes):
                ax.scatter(
                    objects[j].x,
                    objects[j].y,
                    objects[j].z,
                    color=objects[j].rgba,
                    alpha=0.5,
                    s=5,
                )
                ax.set_title(titles[j])
                axes3d_clean(ax)
                axes3d_set_aspect_equal(ax)
                ax.view_init(125, -100, -10)
                fig_title = f"Episode {episode} ('{primary_target_object}'): theta = {theta:.2f} deg"
                fig.suptitle(fig_title)
            out_dir = OUT_DIR / "mlh_vs_min_error"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"episode_{episode}.png", dpi=300)
            plt.close()
