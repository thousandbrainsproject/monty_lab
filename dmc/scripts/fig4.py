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
from types import SimpleNamespace
from typing import Iterable, List, Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    ObjectModel,
    load_eval_stats,
    load_object_model,
)
from plot_utils import TBP_COLORS, axes3d_clean, axes3d_set_aspect_equal
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.transform import Rotation as R
from tbp.monty.frameworks.environments.ycb import SIMILAR_OBJECTS
from tbp.monty.frameworks.utils.logging_utils import get_pose_error

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


"""
--------------------------------------------------------------------------------
Symmetrical rotations
--------------------------------------------------------------------------------
"""


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
    """Load symmetric rotations for the MLH.

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
    experiment: os.PathLike = "fig4_symmetry_run",
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
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_symmetry_run"
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


"""
Dendrogram and confusion matrix.
"""


def get_relative_evidence_matrices(modality: str) -> np.ndarray:
    """Get relative evidence matrices for a set of objects.

    Args:
        objects (Iterable[str]): The objects to get the relative evidence matrices for.

    Returns:
        np.ndarray: A 3D array of shape (n_epochs, n_objects, n_objects), where each
        matrix is the relative evidence matrix for the corresponding epoch.

    TODO: Maybe hard-code `objects` to `SIMILAR_OBJECTS` when done developing.
    """

    assert modality in ["dist", "surf"]
    exp_dir = DMC_RESULTS_DIR / f"{modality}_agent_1lm_randrot_noise_10simobj"
    eval_stats = load_eval_stats(exp_dir / "eval_stats.csv")
    detailed_stats = DetailedJSONStatsInterface(exp_dir / "detailed_run_stats.json")

    # Generate a relative evidence matrix for each epoch.
    all_objects = SIMILAR_OBJECTS
    id_to_object = {i: obj for i, obj in enumerate(all_objects)}
    object_to_id = {obj: i for i, obj in id_to_object.items()}
    n_epochs = eval_stats.epoch.max() + 1

    rel_evidence_matrices = np.zeros((n_epochs, len(all_objects), len(all_objects)))
    for episode, stats in enumerate(detailed_stats):
        max_evidences = stats["LM_0"]["max_evidences_ls"]
        row = eval_stats.iloc[episode]
        target_object = row.primary_target_object
        if target_object not in all_objects:
            continue
        target_object_id = object_to_id[target_object]
        target_evidence = max_evidences[target_object]
        for object_id, object_name in id_to_object.items():
            rel_evidence_matrices[row.epoch, target_object_id, object_id] = np.abs(
                max_evidences[object_name] - target_evidence
            )

    return rel_evidence_matrices


def plot_dendrogram_and_confusion_matrix(modality: str):
    """Plot the dendrogram and confusion matrix."""
    assert modality in ["dist", "surf"]
    out_dir = OUT_DIR / "object_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_objects = SIMILAR_OBJECTS
    id_to_object = {i: obj for i, obj in enumerate(all_objects)}
    object_to_id = {obj: i for i, obj in id_to_object.items()}

    rel_evidence_matrices = get_relative_evidence_matrices(modality)
    rel_evidence_matrix = rel_evidence_matrices.mean(axis=0)

    # Normalize the rows of the evidence matrix.
    sums = rel_evidence_matrix.sum(axis=1, keepdims=True)
    rel_evidence_matrix_normed = rel_evidence_matrix / sums

    # Average off-diagonal/symmetric pairs.
    for i in range(rel_evidence_matrix_normed.shape[0]):
        for j in range(i + 1, rel_evidence_matrix_normed.shape[1]):
            upper = rel_evidence_matrix_normed[i, j]
            lower = rel_evidence_matrix_normed[j, i]
            avg_value = (upper + lower) / 2
            rel_evidence_matrix_normed[i, j] = avg_value
            rel_evidence_matrix_normed[j, i] = avg_value

    # Plot dendrogram.
    if modality == "dist":
        permuted_names = [
            "fork",
            "knife",
            "spoon",
            "cracker_box",
            "sugar_box",
            "pudding_box",
            "mug",
            "e_cups",
            "d_cups",
            "c_cups",
        ]
    else:
        permuted_names = [
            "knife",
            "fork",
            "spoon",
            "pudding_box",
            "sugar_box",
            "cracker_box",
            "mug",
            "e_cups",
            "d_cups",
            "c_cups",
        ]

    fig, ax = plt.subplots(figsize=(6, 4))
    Z = linkage(rel_evidence_matrix_normed, optimal_ordering=True)
    link_color_palette = [TBP_COLORS["blue"], TBP_COLORS["green"], TBP_COLORS["purple"]]
    set_link_color_palette(link_color_palette)
    dendrogram(
        Z,
        labels=all_objects,
        color_threshold=0.150,
        above_threshold_color="black",
        ax=ax,
    )
    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45, fontsize=8, ha="right")
    ax.set_ylabel("Cluster Distance", fontsize=10)
    sns.despine(left=False, bottom=False, right=True)
    plt.tight_layout()
    plt.show()
    fig.savefig(out_dir / f"dendrogram_{modality}.png", dpi=300)
    fig.savefig(out_dir / f"dendrogram_{modality}.svg")

    # Plot confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 4))

    # Rearrange the evidence matrix to match the dendrogram. We want to
    # keep clustered objects close to each other.
    permuted_inds = [object_to_id[obj] for obj in permuted_names]
    permuted_matrix = rel_evidence_matrix_normed[np.ix_(permuted_inds, permuted_inds)]
    rel_evidence_df = pd.DataFrame(permuted_matrix, columns=permuted_names)
    sns.heatmap(rel_evidence_df, cmap="inferno", ax=ax)
    ax.set_xticks(np.arange(permuted_matrix.shape[0]) + 0.5)
    ax.set_yticks(np.arange(permuted_matrix.shape[1]) + 0.5)
    ax.set_yticklabels(permuted_names, rotation=0)
    ax.set_xticklabels(permuted_names, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_aspect("equal")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(
        "Evidence rel. Target (Normalized)", rotation=270, labelpad=20, fontsize=10
    )
    plt.tight_layout()
    plt.show()
    fig.savefig(out_dir / f"confusion_matrix_{modality}.png", dpi=300)
    fig.savefig(out_dir / f"confusion_matrix_{modality}.svg")


def plot_similar_object_models(modality: str):
    assert modality in ["dist", "surf"]
    """Plot the models of similar objects."""
    all_objects = SIMILAR_OBJECTS
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 4), subplot_kw={"projection": "3d"})
    plot_params = {
        "mug": dict(elev=30, azim=-60, roll=0),
        "e_cups": dict(elev=30, azim=-60, roll=0),
        "fork": dict(elev=60, azim=-40, roll=0),
        "knife": dict(elev=60, azim=-40, roll=0),
        "spoon": dict(elev=60, azim=-40, roll=0),
        "c_cups": dict(elev=30, azim=-30, roll=0),
        "d_cups": dict(elev=30, azim=-30, roll=0),
        "cracker_box": dict(elev=30, azim=-30, roll=0),
        "sugar_box": dict(elev=30, azim=-30, roll=0),
        "pudding_box": dict(elev=35, azim=-30, roll=0),
    }
    for i, ax in enumerate(axes.flatten()):
        object_name = all_objects[i]
        model = load_object_model(f"{modality}_agent_1lm", object_name)
        model = model.rotated(R.from_euler("xyz", [90, 0, 0], degrees=True))
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
        ax.set_proj_type("persp", focal_length=1)
        params = plot_params[object_name]
        ax.set_title(object_name)
        ax.view_init(elev=params["elev"], azim=params["azim"], roll=params["roll"])
    plt.show()
    fig.savefig(OUT_DIR / f"object_models_{modality}.png", dpi=300)
    fig.savefig(OUT_DIR / f"object_models_{modality}.svg")


plot_dendrogram_and_confusion_matrix("dist")
plot_dendrogram_and_confusion_matrix("surf")
plot_similar_object_models("dist")
plot_similar_object_models("surf")
