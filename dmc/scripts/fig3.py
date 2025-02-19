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
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    ObjectModel,
    get_percent_correct,
    load_eval_stats,
    load_object_model,
)
from plot_utils import TBP_COLORS, axes3d_clean, axes3d_set_aspect_equal
from scipy.spatial.transform import Rotation as R
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS

# from tbp.monty.frameworks.models.object_model import GraphObjectModel

plt.rcParams["font.size"] = 8

# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    fig.savefig(out_dir / "accuracy_and_steps.pdf")
    plt.show()



def mug_plot_top_left():
    mug = load_object_model("dist_agent_1lm", "mug")
    mug.translation = np.array([-0.012628763, 1.4593439, 0.00026388466])
    mug -= mug.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw mug at detected_rotation.
    mug_rot_detected = mug.rotated(0, 0, 0)
    ax.scatter(
        mug_rot_detected.x,
        mug_rot_detected.y,
        mug_rot_detected.z,
        color=blue,
        alpha=0.3,
    )

    # Draw mug at most_likely_rotation.
    mug_r = mug.rotated(0, 0, 180)
    ax.scatter(mug_r.x, mug_r.y, mug_r.z, color=green, alpha=0.3)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()

    out_path = OUT_DIR / "mug_top_left.png"
    fig.savefig(out_path, dpi=300)
    return fig


def mug_plot_top_right():
    mug = load_object_model("dist_agent_1lm", "mug")
    mug.translation = np.array([-0.012628763, 1.4593439, 0.00026388466])
    mug -= mug.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw original mug.
    ax.scatter(mug.x, mug.y, mug.z, color=blue, alpha=0.3)

    # Draw rotated mug.
    mug_r = mug.rotated(15, 70, 45)
    ax.scatter(mug_r.x, mug_r.y, mug_r.z, color=green, alpha=0.3)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()
    out_path = OUT_DIR / "mug_top_right.png"
    fig.savefig(out_path, dpi=300)

    return fig


def spoon_plot_bottom_left():
    spoon = load_object_model("surf_agent_1lm", "spoon")
    spoon -= spoon.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw original spoon.
    spoon = spoon.rotated(0, 0, 20)
    ax.scatter(spoon.x, spoon.y, spoon.z, color=blue, alpha=0.8)

    # Draw rotated spoon.
    spoon_r = spoon.rotated(180, 0, 0)
    ax.scatter(spoon_r.x, spoon_r.y, spoon_r.z, color=green, alpha=0.8)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()
    out_path = OUT_DIR / "spoon_bottom_left.png"
    fig.savefig(out_path, dpi=300)


def spoon_plot_bottom_right():
    spoon = load_object_model("surf_agent_1lm", "spoon")
    spoon -= spoon.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw original spoon.
    ax.scatter(spoon.x, spoon.y, spoon.z, color=blue, alpha=0.75)

    # Draw rotated spoon.
    spoon_r = spoon.rotated(80, 70, 180)
    # spoon_r -= ()
    ax.scatter(spoon_r.x, spoon_r.y, spoon_r.z, color=green, alpha=0.75)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax, label_axes=True)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()
    out_path = OUT_DIR / "spoon_bottom_right.png"
    fig.savefig(out_path, dpi=300)


def plot_evidence_graphs_and_patches():
    # Load detailed stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    # Find the steps where the LM has processed data, but limit to 21 steps total.
    lm_processed_steps = np.array(stats["LM_0"]["lm_processed_steps"])
    lm_processed_steps = np.argwhere(lm_processed_steps).flatten()
    lm_processed_steps = lm_processed_steps[:21]
    n_steps = len(lm_processed_steps)

    # Extract evidence values for all objects.
    evidences = stats["LM_0"]["evidences_ls"]
    possible_locations = stats["LM_0"]["possible_locations_ls"]
    possible_rotations = stats["LM_0"]["possible_rotations_ls"]
    objects = {}
    for object_name in DISTINCT_OBJECTS:
        obj_evidences, obj_locations = [], []
        for i in range(n_steps):
            obj_evidences.append(evidences[i][object_name])
            obj_locations.append(possible_locations[i][object_name])
        obj_evidences = np.array(obj_evidences)
        obj_locations = np.array(obj_locations)
        obj_rotation = np.array(possible_rotations[0][object_name])
        objects[object_name] = {
            "evidences": obj_evidences,
            "locations": obj_locations,
            "rotation": obj_rotation,
        }

    # Define which objects and steps we're going to plot, and where to save the plots.
    object_names = ["mug", "bowl", "golf_ball"]
    steps = np.array([1, 10, 20])
    out_dir = OUT_DIR / "evidence_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a color map that spans the range of all evidence values.
    all_evidences = [objects[name]["evidences"].flatten() for name in DISTINCT_OBJECTS]
    all_evidences = np.concatenate(all_evidences)
    evidence_min = np.percentile(all_evidences, 2.5)
    evidence_max = np.percentile(all_evidences, 97.5)
    scalar_map = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=evidence_min, vmax=evidence_max)
    )

    # Plot evidence graphs for each object and step individually.
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for i, step in enumerate(steps):
        for j, obj_name in enumerate(object_names):
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(projection="3d")

            dct = objects[obj_name]
            locations = dct["locations"][step]
            evidences = dct["evidences"][step]
            x, y, z = locations[:, 0], locations[:, 1], locations[:, 2]
            color = scalar_map.to_rgba(evidences)

            obj = ObjectModel(locations, translation=np.array([0, 1.5, 0]), rgba=color)
            obj = obj.rotated(0, 0, -20)
            x, y, z = obj.x, obj.y, obj.z

            sizes = evidences * 10
            sizes[sizes <= 0] = 0.1
            linewidths = [0] * sizes.shape[0]
            ax.scatter(x, y, z, color=color, alpha=0.5, s=sizes, linewidths=linewidths)
            ax.view_init(100, -100, -10)
            axes3d_clean(ax, grid=False)
            axes3d_set_aspect_equal(ax)
            ax.margins(0)
            fig.tight_layout(pad=0.1)
            fig.savefig(png_dir / f"{obj_name}_step_{step}.png", dpi=300)
            fig.savefig(pdf_dir / f"{obj_name}_step_{step}.pdf")
            plt.close(fig)

    # Plot the colorbar.
    fig, ax = plt.subplots(1, 1, figsize=(1, 2))
    cbar = plt.colorbar(scalar_map, ax=ax, orientation="vertical", label="Evidence")
    ax.remove()  # Remove the empty axes, we just want the colorbar
    cbar.set_ticks([])
    cbar.set_label("")
    fig.tight_layout()
    fig.savefig(out_dir / "colorbar.png", dpi=300)
    fig.savefig(out_dir / "colorbar.pdf")

    # Extract RGBA patches for sensor module 0.
    rgba_patches = []
    for ind in lm_processed_steps:
        rgba_patches.append(np.array(stats["SM_0"][ind]["rgba"]))
    rgba_patches = np.array(rgba_patches)

    # Save the RGBA patches.
    out_dir = OUT_DIR / "patches"
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        patch = rgba_patches[step]
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(patch)
        ax.axis("off")
        fig.tight_layout(pad=0.0)
        fig.savefig(png_dir / f"patch_step_{step}.png", dpi=300)
        fig.savefig(pdf_dir / f"patch_step_{step}.pdf")
        plt.close(fig)

def plot_trajectory():
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    mug = load_object_model("dist_agent_1lm_10distinctobj", "mug")

    # Find the steps where the LM has processed data, but limit to 21 steps total.
    lm_processed_steps = np.array(stats["LM_0"]["lm_processed_steps"])
    lm_processed_steps = np.argwhere(lm_processed_steps).flatten()
    lm_processed_steps = lm_processed_steps[:21]
    n_steps = len(lm_processed_steps)

    sm = stats["SM_0"]  # a list of dicts

    # Extract the (central) locations of each observation.
    n_rows = n_cols = 64
    center_loc = n_rows // 2 * n_cols + n_cols // 2
    centers = np.zeros((n_steps, 3))
    for i, step in enumerate(lm_processed_steps):
        arr = np.array(sm[step]["semantic_3d"])
        centers[i, 0] = arr[center_loc, 0]
        centers[i, 1] = arr[center_loc, 1]
        centers[i, 2] = arr[center_loc, 2]

    out_dir = OUT_DIR / "trajectory_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(projection="3d")

    # Draw the mug.
    linewidths = np.zeros(len(mug.x))
    ax.scatter(mug.x, mug.y, mug.z, color=mug.rgba, alpha=0.5, linewidths=linewidths)
    ax.view_init(115, -90, 0)
    axes3d_clean(ax, grid=False)
    axes3d_set_aspect_equal(ax)
    fig.savefig(out_dir / "mug.png", dpi=300)
    fig.savefig(out_dir / "mug.pdf")

    # Draw the observation locations.
    blue = TBP_COLORS["blue"]
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    z += 0.0025
    edgecolors = np.zeros([len(x), 4])
    # edgecolors[:, 3] = 1
    ax.scatter(x, y, z, color=blue, edgecolors="none")
    fig.savefig(out_dir / "mug_with_points.png", dpi=300)
    fig.savefig(out_dir / "mug_with_points.pdf")

    # Draw points connecting the observation locations.
    ax.plot(x, y, z, color=blue, alpha=1, ls="--")
    fig.savefig(out_dir / "mug_with_path.png", dpi=300)
    fig.savefig(out_dir / "mug_with_path.pdf")

    for i in range(len(x)):
        # if i in [1, 10, 20]:
        ax.text(x[i], y[i], z[i], f"{i}", color=blue, alpha=1)

    fig.savefig(out_dir / "mug_with_path_and_labels.png", dpi=300)
    fig.savefig(out_dir / "mug_with_path_and_labels.pdf")

    # axes3d_clean(ax, grid=False)
    # axes3d_set_aspect_equal(ax)
    plt.show()


def rotation_difference(
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
        return 0, np.array([0.0, 0.0, 0.0])

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
        theta = np.degrees(theta)
        axis = np.degrees(axis)

    return theta, axis


def l2_distance(
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
    pc1 = pc1.points if isinstance(pc1, ObjectModel) else pc1
    pc2 = pc2.points if isinstance(pc2, ObjectModel) else pc2

    return np.mean(np.linalg.norm(pc1 - pc2, axis=1))


def emd_distance(
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
    pc1 = pc1.points if isinstance(pc1, ObjectModel) else pc1
    pc2 = pc2.points if isinstance(pc2, ObjectModel) else pc2
    # Compute pairwise distances
    cost_matrix = scipy.spatial.distance.cdist(pc1, pc2)
    # Solve transport problem
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(pc1)


def chamfer_distance(
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
    pc1 = pc1.points if isinstance(pc1, ObjectModel) else pc1
    pc2 = pc2.points if isinstance(pc2, ObjectModel) else pc2

    dists1 = np.min(scipy.spatial.distance.cdist(pc1, pc2), axis=1)
    dists2 = np.min(scipy.spatial.distance.cdist(pc2, pc1), axis=1)
    return np.mean(dists1) + np.mean(dists2)


class RotationUtility:
    def __init__(self, experiment_dir: os.PathLike):
        self.experiment_dir = Path(experiment_dir)
        self.detailed_stats = DetailedJSONStatsInterface(
            experiment_dir / "detailed_run_stats.json"
        )
        self.eval_stats = load_eval_stats(experiment_dir / "eval_stats.csv")
        self._episode = None
        self.stats = None

    @property
    def episode(self) -> int:
        return self._episode

    @episode.setter
    def episode(self, episode: int):
        self._episode = episode
        self.stats = self.detailed_stats[episode]

    def get_csv_info(
        self, episode: int
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        df_row = self.eval_stats.iloc[episode]
        primary_target_object = df_row.primary_target_object
        primary_target_rotation = df_row.primary_target_rotation_euler
        detected_rotation = df_row.detected_rotation
        most_likely_rotation = df_row.most_likely_rotation
        primary_target_rotation = np.array(
            [float(x) for x in primary_target_rotation[1:-1].split(",")]
        )
        detected_rotation = np.array(
            [float(x) for x in detected_rotation[1:-1].split()]
        )
        most_likely_rotation = np.array(
            [float(x) for x in most_likely_rotation[1:-1].split()]
        )
        return (
            primary_target_object,
            primary_target_rotation,
            detected_rotation,
            most_likely_rotation,
        )

    def load_rotations(self) -> List[SimpleNamespace]:
        """Loads the rotations and evidence values.

        Returns:
            List[SimpleNamespace]: A list of SimpleNamespace objects, each containing
            the id, rotation, and evidence value.
        """
        last_hypotheses_evidence = np.array(
            self.stats["LM_0"]["last_hypotheses_evidence"]
        )
        possible_rotations = np.array(self.stats["LM_0"]["symmetric_rotations"])
        rotations = []
        for i in range(len(possible_rotations)):
            rotations.append(
                SimpleNamespace(
                    id=i,
                    rot=R.from_matrix(possible_rotations[i]).inv(),
                    evidence=last_hypotheses_evidence[i],
                )
            )
        return rotations

    def filter_rotations(
        self, rotations: List[SimpleNamespace], max_rotations: int = 100
    ) -> List[SimpleNamespace]:
        """Filters the rotations to the top `max_rotations` rotations.

        Args:
            rotations (List[SimpleNamespace]): A list of SimpleNamespace objects,
            each containing the id, rotation, and evidence value.
            max_rotations (int): The maximum number of rotations to return.

        Returns:
            List[SimpleNamespace]: A list of SimpleNamespace objects, each containing
            the id, rotation, and evidence value.
        """
        n_rotations = len(rotations)
        if n_rotations > max_rotations:
            last_hypotheses_evidence = np.array([obj.evidence for obj in rotations])
            sorting_inds = np.argsort(last_hypotheses_evidence)[::-1]
            rotations = [rotations[ind] for ind in sorting_inds][:max_rotations]
        return rotations

    def compute_relative_rotations(
        self, rotations: List[SimpleNamespace]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the relative rotations between all rotations.

        Args:
            rotations (List[SimpleNamespace]): A list of SimpleNamespace objects,
            each containing the id, rotation, and evidence value.
        """
        # Find relative rotations between all rotations.
        n_rotations = len(rotations)
        theta_matrix = np.zeros((n_rotations, n_rotations))
        axis_matrix = np.zeros((n_rotations, n_rotations, 3))
        for i in range(n_rotations - 1):
            for j in range(i + 1, n_rotations):
                rot_a, rot_b = rotations[i].rot, rotations[j].rot
                theta, axis = rotation_difference(rot_a, rot_b, degrees=True)
                theta_matrix[i, j] = theta
                axis_matrix[i, j] = axis
        return theta_matrix, axis_matrix

    def group_rotations(
        self,
        rotations: List[SimpleNamespace],
        theta_threshold: float = 20.0,
    ) -> Tuple[List[SimpleNamespace]]:
        """Groups the rotations into two groups based on the rotation difference.

        Args:
            rotations (List[SimpleNamespace]): A list of SimpleNamespace objects,
            each containing the id, rotation, and evidence value.
        """
        # Group rotations
        theta_matrix, axis_matrix = self.compute_relative_rotations(rotations)
        row = theta_matrix[0]
        group_a = np.where(abs(row - 0) < theta_threshold)[0]
        group_b = np.where(abs(row - 180) < theta_threshold)[0]

        if len(group_b) == 0:
            print("No 180 degree rotations found. Returning.")
            return None
        if len(group_a) + len(group_b) != len(rotations):
            print(
                f" - WARNING: episode {self.episode} has more than two rotation groups"
            )
        print(f" - Num. rotations per group: a={len(group_a)}, b={len(group_b)}")

        # Replace indices with rotation objects.
        group_a = [rotations[i] for i in group_a]
        group_b = [rotations[i] for i in group_b]

        # Sort rotations within each group by evidence.
        group_a = sorted(group_a, key=lambda x: x.evidence, reverse=True)
        group_b = sorted(group_b, key=lambda x: x.evidence, reverse=True)
        groups = [group_a, group_b]

        # Find best rotation, make its group group a.
        if max([r.evidence for r in group_b]) > max([r.evidence for r in group_a]):
            groups = [group_b, group_a]

        return groups


def plot_rotations(episode: int):
    """Plots the rotations of the object for a given episode."""

    from types import SimpleNamespace

    # Load detailed stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_rotations"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[episode]

    # Load some info from 'eval_stats.csv'.
    df = load_eval_stats(experiment_dir / "eval_stats.csv")
    df_row = df.iloc[episode]
    primary_target_object = df_row.primary_target_object
    primary_target_rotation = df_row.primary_target_rotation_euler
    detected_rotation = df_row.detected_rotation
    most_likely_rotation = df_row.most_likely_rotation
    primary_target_rotation = np.array(
        [float(x) for x in primary_target_rotation[1:-1].split(",")]
    )
    detected_rotation = np.array([float(x) for x in detected_rotation[1:-1].split()])
    most_likely_rotation = np.array(
        [float(x) for x in most_likely_rotation[1:-1].split()]
    )
    # Load the rotations and evidence values.
    last_hypotheses_evidence = np.array(stats["LM_0"]["last_hypotheses_evidence"])
    possible_rotations = np.array(stats["LM_0"]["symmetric_rotations"])
    n_rotations = len(possible_rotations)
    rotations = []
    for i in range(n_rotations):
        rot = R.from_matrix(possible_rotations[i]).inv()
        ev = last_hypotheses_evidence[i]
        rotations.append(SimpleNamespace(id=i, rot=rot, evidence=ev))

    print(f"\nEpisode {episode}\n-----------")
    print(f" - primary target object: {primary_target_object}")
    print(f" - primary target rotation: {primary_target_rotation}")
    print(f" - detected rotation: {detected_rotation}")
    print(f" - most likely rotation: {most_likely_rotation}")
    print(f" - num. rotations: {n_rotations}")
    max_rotations = 100
    if n_rotations > max_rotations:
        print(f" - Using top {max_rotations} rotations.")
        sorting_inds = np.argsort(last_hypotheses_evidence)[::-1]
        rotations = [rotations[ind] for ind in sorting_inds][:max_rotations]
        n_rotations = len(rotations)

    # Find relative rotations between all rotations.
    theta_matrix = np.zeros((n_rotations, n_rotations))
    axis_matrix = np.zeros((n_rotations, n_rotations, 3))
    for i in range(n_rotations - 1):
        for j in range(i + 1, n_rotations):
            rot_a, rot_b = rotations[i].rot, rotations[j].rot
            theta, axes = rotation_difference(rot_a, rot_b, degrees=True)
            theta_matrix[i, j] = theta
            axis_matrix[i, j] = axes

    # Group rotations
    theta_threshold = 20
    row = theta_matrix[0]
    group_a = np.where(abs(row - 0) < theta_threshold)[0]
    group_b = np.where(abs(row - 180) < theta_threshold)[0]

    if len(group_b) == 0:
        print("No 180 degree rotations found. Returning.")
        return None
    if len(group_a) + len(group_b) != n_rotations:
        print(f" - WARNING: episode {episode} has more than two rotation groups")
    print(f" - Num. rotations per group: a={len(group_a)}, b={len(group_b)}")

    # Replace indices with rotation objects.
    group_a = [rotations[i] for i in group_a]
    group_b = [rotations[i] for i in group_b]

    # Sort rotations within each group by evidence.
    group_a = sorted(group_a, key=lambda x: x.evidence, reverse=True)
    group_b = sorted(group_b, key=lambda x: x.evidence, reverse=True)
    groups = [group_a, group_b]

    # Find best rotation, make its group group a.
    if max([r.evidence for r in group_b]) > max([r.evidence for r in group_a]):
        group_a, group_b = group_b, group_a

    # Plot objects.
    init_elev, init_azim = 30, -90

    # Plot true object first.
    base_obj = load_object_model("dist_agent_1lm", primary_target_object)
    base_obj = base_obj.centered()
    true_rotation = R.from_euler("xyz", primary_target_rotation, degrees=True)
    true_obj = base_obj.rotated(true_rotation)

    # Plot same-group, different-group, and random rotations.
    n_rows = 4
    n_cols = min([4, len(group_a), len(group_b)])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        subplot_kw={"projection": "3d"},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Plot the true rotation top-left.
    ax = axes[0, 0]
    ax.scatter(true_obj.x, true_obj.y, true_obj.z, color=true_obj.rgba, alpha=0.5)
    ax.set_title("True Rotation")
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)
    ax.view_init(init_elev, init_azim)

    # Draw rotation axes between group_a and group_b.
    if n_cols > 1:
        ax = axes[0, 1]
        a, b = groups[0][0], groups[1][0]
        rel_rot = a.rot * b.rot.inv()
        rel_mat = rel_rot.as_matrix()
        origin = np.array([0, 0, 0])
        ax.quiver(*origin, *rel_mat[0], color="red", length=1, arrow_length_ratio=0.1)
        ax.quiver(*origin, *rel_mat[1], color="green", length=1, arrow_length_ratio=0.1)
        ax.quiver(*origin, *rel_mat[2], color="blue", length=1, arrow_length_ratio=0.1)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(init_elev, init_azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Relative Rotation Axis")
        for ax in axes[0, 2:]:
            ax.remove()

    # Plot group_a and group_b rotations.
    for i, group in enumerate(groups):
        for j in range(n_cols):
            r = group[j]
            rot = r.rot
            obj = base_obj.rotated(rot)
            ax = axes[i + 1, j]
            ax.scatter(obj.x, obj.y, obj.z, color=obj.rgba, alpha=0.5)
            evidence = r.evidence
            l2 = l2_distance(obj, true_obj)
            emd = emd_distance(obj, true_obj)
            chamfer = chamfer_distance(obj, true_obj)
            ax.set_title(
                f"ID={r.id}: Evidence: {evidence:.2f}\nL2: {l2:.4f}\nEMD: {emd:.4f}\nChamfer: {chamfer:.4f}"
            )
            axes3d_clean(ax)
            axes3d_set_aspect_equal(ax)
            ax.view_init(init_elev, init_azim)

    # Plot random rotations.
    random_rots = [
        R.from_euler("xyz", np.random.randint(0, 360, size=(3,)), degrees=True)
        for _ in range(n_cols)
    ]
    for j in range(len(random_rots)):
        obj = base_obj.rotated(random_rots[j])
        ax = axes[3, j]
        ax.scatter(obj.x, obj.y, obj.z, color=obj.rgba, alpha=0.5)
        l2 = l2_distance(obj, true_obj)
        emd = emd_distance(obj, true_obj)
        chamfer = chamfer_distance(obj, true_obj)
        ax.set_title(f"L2: {l2:.4f}\nEMD: {emd:.4f}\nChamfer: {chamfer:.4f}")
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        ax.view_init(init_elev, init_azim)

    return fig


def run_plots():
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_rotations"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    maybe_usable_episodes = []
    for i, stats in enumerate(detailed_stats):
        if "last_hypotheses_evidence" not in stats["LM_0"]:
            continue
        last_hypotheses_evidence = np.array(stats["LM_0"]["last_hypotheses_evidence"])
        n_rotations = len(last_hypotheses_evidence)
        if n_rotations < 2:
            continue
        maybe_usable_episodes.append(i)

    maybe_usable_episodes = np.array(maybe_usable_episodes)
    unusable_episodes = [9]
    highest_completed_episode = 0

    episodes = np.setdiff1d(maybe_usable_episodes, unusable_episodes)
    episodes = episodes[episodes > highest_completed_episode]

    out_dir = OUT_DIR / "rotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        try:
            fig = plot_rotations(episode)
        except Exception as e:
            print(f"Error plotting episode {episode}: {e}")
            unusable_episodes.append(episode)
            continue
        if fig is None:
            unusable_episodes.append(episode)
            continue

        fig.savefig(out_dir / f"rotations_{episode}.png", dpi=300)
        plt.close()


run_plots()

# util = RotationUtility(VISUALIZATION_RESULTS_DIR / "fig3_rotations")
# util.episode = 4
# rotations = util.load_rotations()
# rotations = util.filter_rotations(rotations, max_rotations=100)
# n_rotations = len(rotations)

# groups = util.group_rotations(rotations)
# a, b = groups[0][0], groups[1][0]
# theta, rot_axes = rotation_difference(a.rot, b.rot, degrees=True)
# rel_rot = a.rot * b.rot.inv()
# rel_mat = rel_rot.as_matrix()

# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# origin = np.array([0, 0, 0])
# i_hat = rel_mat[0]
# j_hat = rel_mat[1]
# k_hat = rel_mat[2]
# ax.quiver(*origin, *rel_mat[0], color="red", length=1, arrow_length_ratio=0.1)
# ax.quiver(*origin, *rel_mat[1], color="green", length=1, arrow_length_ratio=0.1)
# ax.quiver(*origin, *rel_mat[2], color="blue", length=1, arrow_length_ratio=0.1)
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# axes3d_clean(ax)
# axes3d_set_aspect_equal(ax)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")


# plt.show()
