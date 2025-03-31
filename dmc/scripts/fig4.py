# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 4.

Panel A: Dendrogram and Similar Object Models
 - `plot_dendrogram()`
 - `plot_similar_object_models()`

Panel B: Example Symmetry Episode
 - `plot_icup_symmetry_episode()`

Panels C and D: Rotation Error and Chamfer Distance for All Episodes
 - `plot_symmetry_stats()`

The functions above require the following experiments to have been run:
 - `pretrain_surf_agent_1lm`: for plotting object models.
 - `surf_agent_1lm_randrot_noise_10simobj`: for generating the dendrogram and
   confusion matrix.
 - `fig4_symmetry_run`: for plotting symmetry stats.

"""

from types import SimpleNamespace
from typing import List, Mapping, Tuple, Union

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
from plot_utils import (
    TBP_COLORS,
    axes3d_clean,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
)
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.transform import Rotation
from tbp.monty.frameworks.environments.ycb import SIMILAR_OBJECTS
from tbp.monty.frameworks.utils.logging_utils import get_pose_error

init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""


def draw_basis_vectors(ax: plt.Axes, rot: Rotation):
    """Draw the basis vectors of a rotation."""
    mat = rot.as_matrix()
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
    ax.axis("off")


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
        Tuple[float, np.ndarray]: The rotational difference and the relative rotation
        matrix.
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


def get_symmetry_stats() -> Mapping:
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
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_symmetry_run"
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
        "pose_error": {"min": [], "MLH": [], "sym": [], "rand": []},
        "Chamfer": {"min": [], "MLH": [], "sym": [], "rand": []},
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
            rot=Rotation.from_euler(
                "xyz", row.primary_target_rotation_euler, degrees=True
            ),
            location=row.primary_target_position,
        )

        # - Create a random rotation.
        rand = SimpleNamespace(
            rot=Rotation.from_euler(
                "xyz", np.random.randint(0, 360, size=(3,)), degrees=True
            ),
            location=np.array([0, 1.5, 0]),
        )

        # - Load symmetry rotations, and computed pose error.
        rotations = load_symmetry_rotations(stats)
        for r in rotations + [target, rand]:
            r.pose_error = np.degrees(
                get_pose_error(r.rot.as_quat(), target.rot.as_quat())
            )

        # - Find mlh, best, and some other symmetric.
        rotations = sorted(rotations, key=lambda x: x.pose_error)
        min_ = sorted(rotations, key=lambda x: x.pose_error)[0]
        sym = rotations[np.random.randint(1, len(rotations))]
        mlh = sorted(rotations, key=lambda x: x.evidence)[-1]

        # - Compute chamfer distances, and store the stats.
        rotations = dict(min=min_, MLH=mlh, sym=sym, rand=rand)
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


def get_relative_evidence_matrices() -> np.ndarray:
    """Get relative evidence matrices for a set of objects.

    Used for generating dendrograms and confusion matrices.

    Requires the experiment `surf_agent_1lm_randrot_noise_10simobj` has been run.

    Args:
        modality: The objects to get the relative evidence matrices for.

    Returns:
        np.ndarray: A 3D array of shape (n_epochs, n_objects, n_objects), where each
        matrix is the relative evidence matrix for the corresponding epoch.

    """

    exp_dir = DMC_RESULTS_DIR / "surf_agent_1lm_randrot_noise_10simobj"
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


def load_symmetry_rotations(stats: Mapping) -> List[SimpleNamespace]:
    """Load symmetric rotations for the MLH.

    Returns:
       A list of SimpleNamespace objects, each representing a symmetric rotation.

    """

    # Get MLH object name -- symmetric rotations are only computed for the MLH object.
    object_name = stats["LM_0"]["current_mlh"][-1]["graph_id"]

    # Load evidence values and possible locations.
    evidences = np.array(stats["LM_0"]["evidences_ls"][object_name])
    # TODO: delete this once confident. It's only used as a sanity check below.
    possible_rotations = np.array(stats["LM_0"]["possible_rotations_ls"][object_name])

    # Load symmetric rotations.
    symmetric_rotations = np.array(stats["LM_0"]["symmetric_rotations"])
    symmetric_locations = np.array(stats["LM_0"]["symmetric_locations"])

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
                rot=Rotation.from_matrix(symmetric_rotations[i]).inv(),
                location=symmetric_locations[i],
                evidence=symmetric_evidences[i],
            )
        )

    return rotations


"""
--------------------------------------------------------------------------------
Panel A: Dendrogram and Similar Object Models
--------------------------------------------------------------------------------
"""


def plot_dendrogram():
    """Plot the dendrogram used in panel A.

    Requires the experiment `surf_agent_1lm_randrot_noise_10simobj` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig4/dendrogram`.

    NOTE: Also plots a confusion matrix, which is not used in the paper.
    """

    out_dir = OUT_DIR / "dendrogram"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_objects = SIMILAR_OBJECTS
    id_to_object = {i: obj for i, obj in enumerate(all_objects)}
    object_to_id = {obj: i for i, obj in id_to_object.items()}

    rel_evidence_matrices = get_relative_evidence_matrices()
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
    rel_evidence_matrix_normed = 1 - rel_evidence_matrix_normed

    # Plot dendrogram.
    fig, ax = plt.subplots(figsize=(7, 3))
    Z = linkage(rel_evidence_matrix_normed, optimal_ordering=True)
    link_color_palette = [
        TBP_COLORS["blue"],
        TBP_COLORS["yellow"],
        TBP_COLORS["purple"],
    ]
    set_link_color_palette(link_color_palette)
    dendrogram(
        Z,
        labels=all_objects,
        color_threshold=0.150,
        above_threshold_color="black",
        ax=ax,
    )
    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=8, ha="center")
    ax.set_ylabel("Cluster Distance", fontsize=10)
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])

    plt.tight_layout()
    plt.show()
    fig.savefig(out_dir / "dendrogram.png")
    fig.savefig(out_dir / "dendrogram.svg")

    # Plot confusion matrix.
    fig, ax = plt.subplots(figsize=(3.5, 3))
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
        "Evidence Similarity (1 - cluster dist.)",
        rotation=270,
        labelpad=20,
        fontsize=10,
    )
    fig.savefig(out_dir / "confusion_matrix.png")
    fig.savefig(out_dir / "confusion_matrix.svg")
    plt.show()


def plot_similar_object_models():
    """Plot the object models for the 10 similar objects used in panel A.

    Requires the experiment `pretrain_surf_agent_1lm` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig4/similar_object_models`.
    """
    # Initialize output directory.
    out_dir = OUT_DIR / "similar_object_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot object models.
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
        model = load_object_model("surf_agent_1lm", object_name)
        model = model.rotated(Rotation.from_euler("xyz", [90, 0, 0], degrees=True))
        ax.scatter(
            model.x,
            model.y,
            model.z,
            color=model.rgba,
            alpha=0.5,
            edgecolors="none",
            s=10,
        )
        ax.set_proj_type("persp", focal_length=1)
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        params = plot_params[object_name]
        ax.set_title(object_name)
        ax.view_init(elev=params["elev"], azim=params["azim"], roll=params["roll"])

    fig.savefig(out_dir / "similar_object_models.png")
    fig.savefig(out_dir / "similar_object_models.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel B: Example Symmetry Episode
--------------------------------------------------------------------------------
"""


def plot_all_symmetry_episodes():
    """Plot object models, rotations, and stats for individual episodes.

    This function was used to find an example for panel B. The example, along with
    hardcoded parameters for reproducibility, can be generated with
    `plot_icup_symmetry_episode()`.

    Requires the experiment `fig4_symmetry_run` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig4/symmetry_episodes`.
    """
    # Initialize output directory.
    out_dir = OUT_DIR / "symmetry_episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    # Preload models that we'll be rotating.
    models = {
        name: load_object_model("dist_agent_1lm", name)
        for name in eval_stats.primary_target_object.unique()
    }
    view_init = dict(elev=90, azim=-80, roll=40)
    for episode, stats in enumerate(detailed_stats):
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
            rot=Rotation.from_euler(
                "xyz", row.primary_target_rotation_euler, degrees=True
            ),
            location=row.primary_target_position,
        )

        # - Create a random rotation.
        rand = SimpleNamespace(
            rot=Rotation.from_euler(
                "xyz", np.random.randint(0, 360, size=(3,)), degrees=True
            ),
            location=np.array([0, 1.5, 0]),
        )

        # - Load symmetry rotations, and computed pose error.
        sym_rotations = load_symmetry_rotations(stats)
        for r in sym_rotations + [target, rand]:
            r.pose_error = np.degrees(
                get_pose_error(r.rot.as_quat(), target.rot.as_quat())
            )

        # - Find mlh, best, and some other symmetric.
        sym_rotations = sorted(sym_rotations, key=lambda x: x.pose_error)
        min_ = sym_rotations[0]
        sym_id = np.random.randint(1, len(sym_rotations))
        sym = sym_rotations[sym_id]
        mlh = sorted(sym_rotations, key=lambda x: x.evidence)[-1]

        # - Compute chamfer distances, and store the stats.
        rotations = dict(target=target, min=min_, MLH=mlh, sym=sym, rand=rand)
        model = models[row.primary_target_object] - [0, 1.5, 0]
        target_obj = model.rotated(target.rot)
        rotation_errors = []
        chamfer_distances = []
        for name, r in rotations.items():
            r.obj = model.rotated(r.rot)
            r.chamfer_distance = get_chamfer_distance(r.obj, target_obj)
            rotation_errors.append(r.pose_error)
            chamfer_distances.append(r.chamfer_distance)

        # Create figure with gridspec
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=5)
        object_axes = []
        pose_axes = []
        labels = ["target", "min", "MLH", "sym", "rand"]
        # plot objects and poses
        for i, lbl in enumerate(labels):
            r = rotations[lbl]
            ax = fig.add_subplot(gs[0, i], projection="3d")
            object_axes.append(ax)
            ax.scatter(
                r.obj.x,
                r.obj.y,
                r.obj.z,
                color=r.obj.rgba,
                alpha=0.5,
            )
            axes3d_set_aspect_equal(ax)
            ax.grid(False)
            ax.axis("off")
            ax.view_init(**view_init)
            ax = fig.add_subplot(gs[1, i], projection="3d")
            pose_axes.append(ax)
            draw_basis_vectors(ax, r.rot)
            axes3d_set_aspect_equal(ax)
            ax.view_init(**view_init)

        ax1 = fig.add_subplot(gs[2, :2])
        ax2 = ax1.twinx()
        labels = ["min", "MLH", "sym", "rand"]
        xticks = np.arange(len(labels))
        width = 0.4
        gap = 0.02
        left_positions = xticks - width / 2 - gap / 2
        right_positions = xticks + width / 2 + gap / 2
        rotation_errors, chamfer_distances = [], []
        for lbl in labels:
            r = rotations[lbl]
            rotation_errors.append(r.pose_error)
            chamfer_distances.append(r.chamfer_distance)
        ax1.bar(left_positions, rotation_errors, width, color=TBP_COLORS["blue"])
        ax2.bar(right_positions, chamfer_distances, width, color=TBP_COLORS["purple"])
        ax1.set_ylabel("Rotation Error (deg)")

        ax1.set_xticks(xticks)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 180)
        ax1.set_yticks([0, 45, 90, 135, 180])
        ax1.spines["right"].set_visible(True)

        ax2.set_ylabel("Chamfer Distance (m)")
        ax2.set_ylim(0, 0.04)
        ax2.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
        ax2.spines["right"].set_visible(True)

        title = f"Episode {episode} - {row.primary_target_object}"
        fig.suptitle(title)

        fig.tight_layout()
        plt.show()
        fig.savefig(png_dir / f"{episode}_{row.primary_target_object}.png")
        fig.savefig(svg_dir / f"{episode}_{row.primary_target_object}.svg")


def plot_icup_symmetry_episode():
    """Plot the icup object models, rotations, and stats used in panel B.

    Requires the experiment `fig4_symmetry_run` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig4/icup_symmetry_episode`.
    """
    out_dir = OUT_DIR / "icup_symmetry_episode"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_symmetry_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")

    # Preload models that we'll be rotating.
    models = {
        name: load_object_model("dist_agent_1lm", name)
        for name in eval_stats.primary_target_object.unique()
    }

    episode = 4
    sym_id = 105
    randrot_euler = (10, 20, 70)
    view_init = dict(elev=40, azim=-90, roll=0)

    stats = detailed_stats[episode]
    row = eval_stats.iloc[episode]

    # - Load the target rotation.
    target = SimpleNamespace(
        rot=Rotation.from_euler("xyz", row.primary_target_rotation_euler, degrees=True),
        location=row.primary_target_position,
    )

    # - Load a random rotation.
    rand = SimpleNamespace(
        rot=Rotation.from_euler("xyz", randrot_euler, degrees=True),
        location=np.array([0, 1.5, 0]),
    )

    # - Load symmetry rotations, and computed pose error.
    sym_rotations = load_symmetry_rotations(stats)
    for r in sym_rotations + [target, rand]:
        r.pose_error = np.degrees(get_pose_error(r.rot.as_quat(), target.rot.as_quat()))

    # - Find mlh, best, and some other symmetric.
    sym_rotations = sorted(sym_rotations, key=lambda x: x.pose_error)
    min_ = sym_rotations[0]
    sym = sym_rotations[sym_id]
    mlh = sorted(sym_rotations, key=lambda x: x.evidence)[-1]

    # - Compute chamfer distances, and store the stats.
    rotations = dict(target=target, min=min_, MLH=mlh, sym=sym, rand=rand)
    model = models[row.primary_target_object] - [0, 1.5, 0]
    target_obj = model.rotated(target.rot)
    rotation_errors = []
    chamfer_distances = []
    for name, r in rotations.items():
        r.obj = model.rotated(r.rot)
        r.chamfer_distance = get_chamfer_distance(r.obj, target_obj)
        rotation_errors.append(r.pose_error)
        chamfer_distances.append(r.chamfer_distance)

    # Create figure with gridspec
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=5)
    object_axes = []
    pose_axes = []
    labels = ["target", "min", "MLH", "sym", "rand"]
    # plot objects and poses
    for i, lbl in enumerate(labels):
        r = rotations[lbl]
        ax = fig.add_subplot(gs[0, i], projection="3d")
        object_axes.append(ax)
        ax.scatter(
            r.obj.x,
            r.obj.y,
            r.obj.z,
            color=r.obj.rgba,
            alpha=0.5,
            s=1,
        )
        axes3d_set_aspect_equal(ax)
        ax.grid(False)
        ax.axis("off")
        ax.view_init(**view_init)
        ax = fig.add_subplot(gs[1, i], projection="3d")
        pose_axes.append(ax)
        draw_basis_vectors(ax, r.rot)
        axes3d_set_aspect_equal(ax)
        ax.view_init(**view_init)

    ax1 = fig.add_subplot(gs[2, :2])
    ax2 = ax1.twinx()
    labels = ["min", "MLH", "sym", "rand"]
    xticks = np.arange(len(labels))
    width = 0.4
    gap = 0.02
    left_positions = xticks - width / 2 - gap / 2
    right_positions = xticks + width / 2 + gap / 2
    rotation_errors, chamfer_distances = [], []
    for lbl in labels:
        r = rotations[lbl]
        rotation_errors.append(r.pose_error)
        chamfer_distances.append(r.chamfer_distance)
    ax1.bar(left_positions, rotation_errors, width, color=TBP_COLORS["blue"])
    ax2.bar(right_positions, chamfer_distances, width, color=TBP_COLORS["purple"])
    ax1.set_ylabel("Rotation Error (deg)")

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 180)
    ax1.set_yticks([0, 45, 90, 135, 180])
    ax1.spines["right"].set_visible(True)

    ax2.set_ylabel("Chamfer Distance (m)")
    ax2.set_ylim(0, 0.04)
    ax2.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    ax2.spines["right"].set_visible(True)

    title = f"Episode {episode} - {row.primary_target_object} - {sym_id}"
    fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_dir / "icup_symmetry_episode.png")
    fig.savefig(out_dir / "icup_symmetry_episode.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panels C and D: Symmetry Stats
--------------------------------------------------------------------------------
"""


def plot_symmetry_stats():
    """Plot the symmetry stats.

    Requires the experiment `fig4_symmetry_run` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig4/symmetry_stats`.
    """
    # Initialize output directory.
    out_dir = OUT_DIR / "symmetry_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load stats.
    stat_arrays = get_symmetry_stats()

    # Create figure.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    rotation_types = ["min", "MLH", "sym", "rand"]
    colors = [TBP_COLORS["blue"]] * len(rotation_types)
    xticks = list(range(1, len(rotation_types) + 1))
    xticklabels = ["min", "MLH", "sym", "rand"]

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
    ax.set_ylabel("Rotation Error (deg)")

    # Chamfer Distance
    ax = axes[1]
    arrays = [stat_arrays["Chamfer"][name] for name in rotation_types]
    vp = ax.violinplot(arrays, showextrema=False, showmedians=True)
    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[j])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Chamfer Distance (m)")
    ymax = max(np.percentile(arr, 95) for arr in arrays)
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(out_dir / "symmetry_stats.png")
    fig.savefig(out_dir / "symmetry_stats.svg")
    plt.show()


if __name__ == "__main__":
    plot_dendrogram()
    plot_similar_object_models()
    plot_icup_symmetry_episode()
    plot_symmetry_stats()
