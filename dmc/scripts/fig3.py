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

from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_PRETRAIN_DIR,
    DMC_RESULTS_DIR,
    DMC_ROOT_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    ObjectModel,
    describe_dict,
    get_percent_correct,
    load_eval_stats,
    load_object_model,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import ArrayLike
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


experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_rotations"
detailed_stats = DetailedJSONStatsInterface(experiment_dir / "detailed_run_stats.json")
df = load_eval_stats(experiment_dir / "eval_stats.csv")
df = df[df["primary_performance"].str.startswith("correct")]
has_symmetry = df[df["symmetry_evidence"] > 1]["episode"].values
# 77 308
# for episode in has_symmetry:
#     stats = detailed_stats[episode]

#     # Extract evidence values for all objects.
#     symmetric_locations = stats["LM_0"]["symmetric_locations"]
#     if symmetric_locations is None or len(symmetric_locations) == 0:
#         n_sym_loc = 0
#     else:
#         n_sym_loc = len(symmetric_locations[-1])
#     symmetric_rotations = stats["LM_0"]["symmetric_rotations"]
#     if symmetric_rotations is None or len(symmetric_rotations) == 0:
#         n_sym_rot = 0
#     else:
#         n_sym_rot = len(symmetric_rotations[-1])

#     print(f"Episode {episode}: {n_sym_loc} {n_sym_rot}")

episode = 2
stats = detailed_stats[episode]
matches = stats["LM_0"]["possible_matches"]
symmetric_locations = np.array(stats["LM_0"]["symmetric_locations"][1])
symmetric_rotations = np.array(stats["LM_0"]["symmetric_rotations"][-1])


obj_name = df[df.episode == episode].primary_target_object.values[0]
df_row = df[df.episode == episode]
obj_name = df_row.primary_target_object.values[0]
primary_target_rotation = df_row.primary_target_rotation_euler.values[0]
detected_rotation = df_row.detected_rotation.values[0]
most_likely_rotation = df_row.most_likely_rotation.values[0]

detected_rotation = np.array([float(x) for x in detected_rotation[1:-1].split()])
most_likely_rotation = np.array([float(x) for x in most_likely_rotation[1:-1].split()])
print(f"Episode {episode}: {obj_name} ")
print(f"target rotation: {primary_target_rotation}")
print(f"detected rotation: {detected_rotation}")
print(f"most likely rotation: {most_likely_rotation}")

sym = np.array(stats["LM_0"]["symmetric_rotations"])
th = 2
for i, s in enumerate(sym):
    r = R.from_matrix(s)
    euler = r.inv().as_euler("xyz", degrees=True)
    euler_mod = euler % 360
    if np.all(np.abs(euler - detected_rotation) < th):
        print(f"{i} - close - detected: {euler}")
    if np.all(np.abs(euler_mod - detected_rotation) < th):
        print(f"{i} - close - detected (euler_mod): {euler_mod}")

    if np.all(np.abs(euler - most_likely_rotation) < th):
        print(f"{i} - close - most likely: {euler}")
    if np.all(np.abs(euler_mod - most_likely_rotation) < th):
        print(f"{i} - close - most likely (mod): {euler_mod}")


sym_rot = symmetric_rotations
# sym_rot = np.degrees(sym_rot)
print(f"symmetric rotations: {sym_rot}")
for i, angles in enumerate(sym_rot):
    rot = R.from_euler("xyz", angles, degrees=True)
    # rot = rot.inv()
    angles_2 = rot.as_euler("xyz", degrees=True) % 360
    print(f"symmetric rotation {i}: {angles_2}")

obj = load_object_model("dist_agent_1lm", obj_name)
center = np.array([obj.x.mean(), obj.y.mean(), obj.z.mean()])

obj = obj - center
obj.translation = np.array([0, 0, 0], dtype=float)

# need to rotate object by symmetric_rotations
rot_a = R.from_euler("xyz", sym_rot[0], degrees=True)
locs_a = rot_a.apply(obj.points)
obj_a = deepcopy(obj)
obj_a.points = locs_a
# obj_a = obj_a.rotated(90, 0, 0)

rot_b = R.from_euler("xyz", sym_rot[1], degrees=True)
locs_b = rot_b.apply(obj.points)
obj_b = deepcopy(obj)
obj_b.points = locs_b
# obj_b = obj_b.rotated(90, 0, 0)

rot_c = R.from_euler("xyz", np.random.randint(0, 360, size=(3,)), degrees=True)
locs_c = rot_c.apply(obj.points)
obj_c = deepcopy(obj)
obj_c.points = locs_c

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
blue = TBP_COLORS["blue"]
green = TBP_COLORS["green"]
ax.scatter(obj_a.x, obj_a.y, obj_a.z, color=blue, alpha=0.5)
ax.scatter(obj_b.x, obj_b.y, obj_b.z, color=green, alpha=0.5)
axes3d_set_aspect_equal(ax)
plt.show()

dist = np.linalg.norm(obj_a.points - obj_b.points, axis=1).mean()
print(f"Distance between symmetric rotations: {dist}")

dist = np.linalg.norm(obj_a.points - obj_c.points, axis=1).mean()
print(f"Distance between random rotations: {dist}")

rots = stats["LM_0"]["possible_rotations_ls"][0]
sp = rots[obj_name]
a = sp[-2]
b = R.from_matrix(a)