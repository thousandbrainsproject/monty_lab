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
from typing import Iterable, List, Mapping, Optional, Tuple, Type, Union

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
from tbp.monty.frameworks.utils.logging_utils import get_pose_error

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"


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


def plot_performance():
    out_dir = OUT_DIR / "performance"
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
        body.set_facecolor(TBP_COLORS["purple"])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylim(0, 180)
    ax2.set_ylabel("Rotation Error (deg)")

    ax1.set_xticks([0.5, 2.5, 4.5, 6.5])
    ax1.set_xticklabels(conditions, rotation=0, ha="center")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=300)
    fig.savefig(out_dir / "performance.svg")
    plt.show()


def draw_randrot_noise_icons():
    model = load_object_model("dist_agent_1lm", "mug")
    model = model - [0, 1.5, 0]
    fig, axes = plt.subplots(1, 4, figsize=(8, 4), subplot_kw={"projection": "3d"})

    # Params
    params = {
        "alpha": 0.45,
        "edgecolors": "none",
        "s": 2.5,
    }

    # Use bluementa as base color.
    hex = TBP_COLORS["blue"].lstrip("#")
    rgba = np.array([int(hex[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0])
    rgba = np.tile(rgba, (len(model.x), 1))

    # Make noisy color.
    rgba_noise = rgba.copy()
    rgba_noise = rgba_noise + 0.3 * np.random.randn(len(rgba_noise), 4)
    rgba_noise = np.clip(rgba_noise, 0, 1)
    rgba_noise[:, 3] = 1

    # - draw base model
    ax = axes[0]
    ax.scatter(model.x, model.y, model.z, color=rgba, **params)

    # - draw noise
    rot = R.from_euler("xyz", [45, 10, 30], degrees=True)
    rot_model = model.rotated(rot)
    ax = axes[1]
    ax.scatter(
        model.x,
        model.y,
        model.z,
        color=rgba_noise,
        **params,
    )

    # - draw random rotation
    rot = R.from_euler("xyz", [45, 10, 30], degrees=True)
    rot_model = model.rotated(rot)
    ax = axes[2]
    ax.scatter(
        rot_model.x,
        rot_model.y,
        rot_model.z,
        color=rgba,
        **params,
    )

    # - draw radom rotation + noise
    rot = R.from_euler("xyz", [25, 30, -135], degrees=True)
    rot_model = model.rotated(rot)
    ax = axes[3]
    ax.scatter(
        rot_model.x,
        rot_model.y,
        rot_model.z,
        color=rgba_noise,
        **params,
    )

    # Clean up
    for ax in axes:
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(90, -90)
        ax.axis("off")

    plt.show()
    fig.savefig(OUT_DIR / "randrot_noise_icons.png", dpi=300)
    fig.savefig(OUT_DIR / "randrot_noise_icons.svg")


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
    """Computes the Earth Mover's Distance (EMD) between two point clouds.

    Args:
        pc1: A numpy array of shape (N, 3) representing the first point cloud.
        pc2: A numpy array of shape (N, 3) representing the second point cloud.

    Returns:
        float: The EMD distance between the two point clouds.
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
    """Compute the Chamfer Distance between two point clouds.

    Args:
        pc1: A numpy array of shape (N, 3) representing the first point cloud.
        pc2: A numpy array of shape (N, 3) representing the second point cloud.

    Returns:
        The Chamfer distance between the two point clouds as a float.
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

    # Get MLH object name -- symmetric rotations are only computed for the MLH object.
    object_name = episode_stats["LM_0"]["current_mlh"][-1]["graph_id"]

    # Load evidence values and possible locations.
    evidences = np.array(episode_stats["LM_0"]["evidences_ls"][object_name])
    # TODO: delete this once confident. It's only used as a sanity check below.
    possible_rotations = np.array(
        episode_stats["LM_0"]["possible_rotations_ls"][object_name]
    )

    # Load symmetric rotations.
    symmetric_rotations = np.array(episode_stats["LM_0"]["symmetric_rotations"])
    symmetric_locations = np.array(episode_stats["LM_0"]["symmetric_locations"])

    # To get evidence values for each rotation, we need to find hypothesis IDs for the
    # symmetric rotations. To do this, we find the evidence threshold that would yield
    # the number of symmetric rotations given.
    n_hypotheses = len(symmetric_rotations)
    sorting_inds = np.argsort(evidences)[::-1]
    evidences_sorted = evidences[sorting_inds]
    evidence_threshold = np.mean(evidences_sorted[n_hypotheses - 1 : n_hypotheses + 1])
    above_threshold = evidences >= evidence_threshold
    hypothesis_ids = np.arange(len(evidences))[above_threshold]
    symmetric_evidences = evidences[hypothesis_ids]

    # Sanity checks.
    assert len(hypothesis_ids) == n_hypotheses
    assert np.allclose(symmetric_rotations, possible_rotations[hypothesis_ids])

    rotations = []
    for i in range(n_hypotheses):
        rotations.append(
            SimpleNamespace(
                id=i,
                hypothesis_id=hypothesis_ids[i],
                rot=R.from_matrix(symmetric_rotations[i]).inv(),
                location=symmetric_locations[i],
                evidence=symmetric_evidences[i],
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


def get_symmetry_stats(
    experiment: os.PathLike = "fig3_symmetry_run",
) -> Mapping:
    """Compute pose errors and Chamfer distances for symmetric rotations.

    Used to generate data by `plot_symmetry_stats`.

    Computes the pose errors and Chamfer distances between symmetric rotations
    and the target rotation. The following rotations are considered:
     - "best": the rotation with the lowest pose error.
     - "mlh": the rotation of the MLH.
     - "other": another rotation from the same group of symmetric rotation.
     - "random": a random rotation.

    Returns:
        The computed pose errors and Chamfer distances for the best, MLH, and
        random rotations, as well as another random rotation from the same group
        of symmetric rotations. Has the items "pose_error" and "Chamfer", each of
        which is a dict with "best", "mlh", "other", and "random" (all numpy arrays)

    """
    experiment_dir = VISUALIZATION_RESULTS_DIR / experiment
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    # Preload models that we'll be rotating.
    models = {
        name: load_object_model("dist_agent_1lm", name)
        for name in eval_stats.primary_target_object.unique()
    }

    # Initialize dict that we'll be returning.
    stat_arrays = {
        "pose_error": {"best": [], "mlh": [], "other": [], "random": []},
        "Chamfer": {"best": [], "mlh": [], "other": [], "random": []},
    }
    for episode, stats in enumerate(detailed_stats):
        # print(f"Episode {episode}/{len(detailed_stats)}")
        # Check valid symmetry rotations.
        # - Must be correct performance
        row = eval_stats.iloc[episode]
        if not row.primary_performance.startswith("correct"):
            continue
        # - Must have at least two symmetry rotations.
        sym_rots = stats["LM_0"]["symmetric_rotations"]
        if sym_rots is None or len(sym_rots) < 2:
            continue

        # - Load the target rotation.
        target = SimpleNamespace(
            rot=R.from_euler("xyz", row.primary_target_rotation_euler, degrees=True),
            location=row.primary_target_position,
        )

        # - Create a random rotation.
        random = SimpleNamespace(
            rot=R.from_euler("xyz", np.random.randint(0, 360, size=(3,)), degrees=True),
            location=np.array([0, 1.5, 0]),
        )

        # - Load symmetry rotations, and computed pose error.
        rotations = load_symmetry_rotations(stats)
        for r in rotations + [target, random]:
            r.pose_error = np.degrees(
                get_pose_error(r.rot.as_quat(), target.rot.as_quat())
            )

        # - Find mlh, best, and some other symmetric.
        rotations = sorted(rotations, key=lambda x: x.pose_error)
        best = sorted(rotations, key=lambda x: x.pose_error)[0]
        other = rotations[np.random.randint(1, len(rotations))]
        mlh = sorted(rotations, key=lambda x: x.evidence)[-1]

        # - Compute chamfer distances, and store the stats.
        rotations = dict(best=best, mlh=mlh, other=other, random=random)
        model = models[row.primary_target_object] - [0, 1.5, 0]
        target_obj = model.rotated(target.rot)
        for name, r in rotations.items():
            obj = model.rotated(r.rot)
            stat_arrays["Chamfer"][name].append(get_chamfer_distance(obj, target_obj))
            stat_arrays["pose_error"][name].append(r.pose_error)

    # - Convert lists to arrays, and return the data.
    for key_1, dct_1 in stat_arrays.items():
        for key_2 in dct_1.keys():
            stat_arrays[key_1][key_2] = np.array(stat_arrays[key_1][key_2])

    return stat_arrays

def plot_symmetry_stats():
    """Plot the symmetry stats."""
    stat_arrays = get_symmetry_stats()

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.3))

    rotation_types = ["best", "mlh", "other", "random"]
    colors = [TBP_COLORS["blue"]] * len(rotation_types)
    xticks = list(range(1, len(rotation_types) + 1))
    xticklabels = ["min. error", "MLH", "other", "random"]

    # Pose Error
    ax = axes[0]
    arrays = [stat_arrays["pose_error"][name] for name in rotation_types]
    vp = ax.violinplot(arrays, showextrema=False, showmedians=True)
    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[j])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_ylim(0, 180)
    ax.set_ylabel("degrees")
    ax.set_title("Pose Error")

    # Chamfer
    ax = axes[1]
    arrays = [stat_arrays["Chamfer"][name] for name in rotation_types]
    vp = ax.violinplot(arrays, showextrema=False, showmedians=True)
    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[j])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("meters")
    ymax = max(np.percentile(arr, 95) for arr in arrays)
    ax.set_ylim(0, ymax)
    ax.set_title("Chamfer Distance")

    fig.tight_layout()

    plt.show()
    out_dir = OUT_DIR / "symmetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "symmetry_stats.png", dpi=300)
    fig.savefig(out_dir / "symmetry_stats.svg")


def plot_symmetry_objects():
    """Render symmetric objects and their rotations."""
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    episode_params = {
        # clamp - god. 180 degree rotation.
        130: {
            "other_index": 1,
            "random_rotation": np.array([98, 86, 134]),
            "elev": 55,
            "azim": 0,
        },
        # bowl - good
        309: {
            "other_index": 1,
            "random_rotation": np.array([172, 68, -25]),
            "elev": 30,
            "azim": -90,
        },
        # mug - not totally symmetric, mixed messages
        154: {
            "other_index": 1,
            "random_rotation": np.array([172, 68, -25]),
            "elev": -70,
            "azim": -80,
        },
    }
    for episode in episode_params.keys():
        params = episode_params.get(episode, {})

        row = eval_stats.iloc[episode]
        primary_target_object = row.primary_target_object
        target = SimpleNamespace(
            rot=R.from_euler("xyz", row.primary_target_rotation_euler, degrees=True)
        )

        # Load rotations, compute rotation error for each, and sort them by error.
        stats = detailed_stats[episode]
        rotations = load_symmetry_rotations(stats)
        for r in rotations:
            theta, axis = get_relative_rotation(r.rot, target.rot, degrees=True)
            r.theta = theta
            r.axis = axis
        rotations = sorted(rotations, key=lambda x: x.theta)

        # Get rotation with lowest error and any other symmetrical rotation.
        best = rotations[0]
        mlh = sorted(rotations, key=lambda x: x.evidence)[-1]
        other_index = params.get("other_index", np.random.randint(1, len(rotations)))
        other = rotations[other_index]

        # Get a random rotation, and compute its error.
        random_euler = params.get(
            "random_rotation", np.random.randint(0, 360, size=(3,))
        )
        random = SimpleNamespace(rot=R.from_euler("xyz", random_euler, degrees=True))
        theta, axis = get_relative_rotation(random.rot, target.rot, degrees=True)
        random.theta = theta
        random.axis = axis

        base_model = load_object_model("dist_agent_1lm", primary_target_object)
        base_model = base_model - [0, 1.5, 0]

        poses = dict(target=target, best=best, mlh=mlh, other=other, random=random)
        for name, obj in poses.items():
            obj.model = base_model.rotated(obj.rot)

        fig, axes = plt.subplots(2, 5, figsize=(8, 4), subplot_kw={"projection": "3d"})
        elev, azim = params.get("elev", 30), params.get("azim", -90)

        for i, (name, obj) in enumerate(poses.items()):
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
            axes3d_clean(ax, grid=False)
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
            ax.set_title(name)

        plt.show()
        out_dir = OUT_DIR / "symmetry"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{primary_target_object}_{episode}.png", dpi=300)
        fig.savefig(out_dir / f"{primary_target_object}_{episode}.svg")
        plt.close()


def patch_affinity_svg(path: os.PathLike):
    with open(path, "r") as f:
        svg_text = f.read()
    import re

    """Patch Matplotlib SVG so that it can be read by Affinity Designer."""
    matches = [x for x in re.finditer("font: ([0-9.]+)px ([^;]+);", svg_text)]
    svg_pieces = [svg_text[: matches[0].start()]]
    for i, match in enumerate(matches):
        # Change "font" style property to separate "font-size" and
        # "font-family" properties because Affinity ignores "font".
        font_size_px, font_family = match.groups()
        new_font_style = (
            f"font-size: {float(font_size_px):.1f}px; " f"font-family: {font_family};"
        )
        svg_pieces.append(new_font_style)
        if i < len(matches) - 1:
            svg_pieces.append(svg_text[match.end() : matches[i + 1].start()])
        else:
            svg_pieces.append(svg_text[match.end() :])
    text = "".join(svg_pieces)
    with open(path, "w") as f:
        f.write(text)


plot_performance()
# patch_affinity_svg(OUT_DIR / "performance/performance.svg")
