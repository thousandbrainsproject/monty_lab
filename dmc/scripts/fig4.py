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

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
)
from plot_utils import axes3d_clean, axes3d_set_aspect_equal

OUT_DIR = DMC_ANALYSIS_DIR / "fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_8lm_patches():
    """
    Plot the 8-patch view finder on the object.
    """
    json_path = os.path.join(
        VISUALIZATION_RESULTS_DIR,
        "fig4_visualize_8lm_patches",
        "detailed_run_stats.json",
    )
    detailed_stats_interface = DetailedJSONStatsInterface(json_path)
    stats = detailed_stats_interface[0]

    sensor_module_id = "SM_8"
    ep = stats["0"]
    SM = ep[sensor_module_id]
    obs = SM["raw_observations"][0]
    rgba = np.array(obs["rgba"])
    semantic_3d = np.array(obs["semantic_3d"])

    def pull(sm_num: int):
        SM = ep[f"SM_{sm_num}"]
        obs = SM["raw_observations"][0]
        rgba = np.array(obs["rgba"])
        semantic_3d = np.array(obs["semantic_3d"])
        x = semantic_3d[:, 0].flatten()
        y = semantic_3d[:, 1].flatten()
        z = semantic_3d[:, 2].flatten()
        sem = semantic_3d[:, 3].flatten().astype(int)
        x = x[sem == 1]
        y = y[sem == 1]
        z = z[sem == 1]
        c = rgba.reshape(-1, 4)
        c = c[sem == 1] / 255
        return x, y, z, c

    # Create a 3D plot of the semantic point cloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis="y")
    # Extract x,y,z coordinates from semantic_3d

    # view finder
    x, y, z, c = pull(8)
    ax.scatter(x, y, z, c=c, marker="o", alpha=0.05)

    mat = np.zeros((64, 64), dtype=bool)
    mat[0, :] = True
    mat[:, 0] = True
    mat[-1, :] = True
    mat[:, -1] = True
    border = mat.flatten()

    # patches
    for i in range(8):
        x, y, z, c = pull(i)
        # c[border] = np.array([0, 0, 1, 1])
        # c[border] = np.array([0, 0, 1, 1])
        ax.scatter(x, y, z, c=c, marker="o")

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    ax.set_title("3D Semantic Point Cloud")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.06, -0.06 + 0.12)
    ax.set_ylim(1.44, 1.44 + 0.12)
    ax.set_zlim(-0.06, 0.06)
    ax.axis("off")
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, "8lm_patches.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "8lm_patches.pdf"))


experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_visualize_8lm_patches"
detailed_stats_path = experiment_dir / "detailed_run_stats.json"
detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
stats = detailed_stats_interface[0]

tempdir = experiment_dir / "temp"
tempdir.mkdir(parents=True, exist_ok=True)
for key, val in stats.items():
    with open(tempdir / f"{key}.json", "w") as f:
        json.dump(val, f)

# Get view-finder RGBA and semantic 3D data for first observation.
sm_dict = stats["SM_8"]
rgba = np.array(sm_dict["raw_observations"][0]["rgba"])
semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])


# Get semantic 3D points that are on the object.
def pull(sm_num: int):
    sm_dict = stats[f"SM_{sm_num}"]

    # Extract x,y,z coordinates and an array indicating which points are on-object.
    semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
    pos = semantic_3d[:, 0:3]
    on_object = semantic_3d[:, 3].astype(int) > 0

    # Extract the RGBA patch, and shape it to be a flat list of points/colors. We need
    # it to have the same format/shape as semantic_3d so we can use the `on_object`
    # array to filter out points.
    rgba = np.array(sm_dict["raw_observations"][0]["rgba"])
    rgba = rgba.reshape(-1, 4)

    # Filter out colors and points that aren't on-object.
    pos = pos[on_object]
    rgba = rgba[on_object] / 255.0
    return pos, rgba


# # Create a 3D plot of the semantic point cloud
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.view_init(elev=90, azim=-90, roll=0)
ax.set_proj_type("persp", focal_length=0.125)
ax.dist = 4.55


def pull(sm_num: int):
    """Extract the RGBA sensor patch, locations, and an on-object mask."""
    sm_dict = stats[f"SM_{sm_num}"]
    # Extract RGBA sensor patch.
    rgba_2d = np.array(sm_dict["raw_observations"][0]["rgba"]) / 255.0
    n_rows, n_cols = rgba_2d.shape[0], rgba_2d.shape[1]

    # Extract locations and on-object filter.
    semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
    pos_1d = semantic_3d[:, 0:3]
    pos_2d = pos_1d.reshape(n_rows, n_cols, 3)
    on_object_1d = semantic_3d[:, 3].astype(int) > 0
    on_object_2d = on_object_1d.reshape(n_rows, n_cols)

    # Filter out points that aren't on-object. Yields a flat list of points/colors.
    return rgba_2d, pos_2d, on_object_2d


# Do view finder first.
rgba_2d, pos_2d, on_object_2d = pull(8)
rows, cols = np.where(on_object_2d)
pos_valid_1d = pos_2d[on_object_2d]
rgba_valid_1d = rgba_2d[on_object_2d]
# Plot the patch.
ax.scatter(
    pos_valid_1d[:, 0],
    pos_valid_1d[:, 1],
    pos_valid_1d[:, 2],
    c=rgba_valid_1d,
    marker="o",
    alpha=0.3,
    zorder=5,
    s=10,
    edgecolors="none",
)

# Do other sensor patches.
for i in range(8):
    rgba_2d, pos_2d, on_object_2d = pull(i)
    rows, cols = np.where(on_object_2d)
    pos_valid_1d = pos_2d[on_object_2d]
    rgba_valid_1d = rgba_2d[on_object_2d]

    # Plot the patch.
    ax.scatter(
        pos_valid_1d[:, 0],
        pos_valid_1d[:, 1],
        pos_valid_1d[:, 2],
        c=rgba_valid_1d,
        marker="o",
        alpha=1,
        zorder=10,
        edgecolors="none",
        s=1,
    )

    # Plot the contours.
    n_pix_on_object = on_object_2d.sum()
    if n_pix_on_object == 0:
        contours = []
    elif n_pix_on_object == on_object_2d.size:
        n_rows, n_cols = on_object_2d.shape
        temp = np.zeros((n_rows, n_cols), dtype=bool)
        temp[0, :] = True
        temp[-1, :] = True
        temp[:, 0] = True
        temp[:, -1] = True
        contours = [np.argwhere(temp)]
    else:
        contours = skimage.measure.find_contours(
            on_object_2d, level=0.5, positive_orientation="low"
        )
        contours = [] if contours is None else contours

    for ct in contours:
        n_rows, n_cols = on_object_2d.shape
        row_mid, col_mid = n_rows // 2, n_cols // 2

        # Contour may be float point (fractional indices). Round
        # row/column indices towards the center of the patch.
        if not np.issubdtype(ct.dtype, np.integer):
            # Round towards the center.
            rows, cols = ct[:, 0], ct[:, 1]
            rows_new, cols_new = np.zeros_like(rows), np.zeros_like(cols)
            rows_new[rows >= row_mid] = np.floor(rows[rows >= row_mid])
            rows_new[rows < row_mid] = np.ceil(rows[rows < row_mid])
            cols_new[cols >= col_mid] = np.floor(cols[cols >= col_mid])
            cols_new[cols < col_mid] = np.ceil(cols[cols < col_mid])
            ct_new = np.zeros_like(ct, dtype=int)
            ct_new[:, 0] = rows_new.astype(int)
            ct_new[:, 1] = cols_new.astype(int)
            ct = ct_new

        # Remove points that are off object.
        points_on_object = on_object_2d[ct[:, 0], ct[:, 1]]
        ct = ct[points_on_object]

        # Order points by their angle from the center.
        Y, X = row_mid - ct[:, 0], ct[:, 1] - col_mid  # pixel to X/Y coords.
        theta = np.arctan2(Y, X)
        sort_order = np.argsort(theta)
        ct = ct[sort_order]

        # Finally, Plot the contour.
        xyz = pos_2d[ct[:, 0], ct[:, 1]]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="k", linewidth=3, zorder=20)


axes3d_clean(ax)
axes3d_set_aspect_equal(ax)
ax.axis("off")
plt.show()

out_dir = OUT_DIR
fig.savefig(out_dir / "8lm_patches.png", dpi=300)
fig.savefig(out_dir / "8lm_patches.svg")

# mat = np.zeros((64, 64), dtype=bool)
# mat[0, :] = True
# mat[:, 0] = True
# mat[-1, :] = True
# mat[:, -1] = True
# border = mat.flatten()

# # patches
# for i in range(8):
#     x, y, z, c = pull(i)
#     # c[border] = np.array([0, 0, 1, 1])
#     # c[border] = np.array([0, 0, 1, 1])
#     ax.scatter(x, y, z, c=c, marker="o")

# # Set labels
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_aspect("equal")
# ax.set_title("3D Semantic Point Cloud")

# # Set equal aspect ratio
# ax.set_box_aspect([1, 1, 1])
# ax.set_xlim(-0.06, -0.06 + 0.12)
# ax.set_ylim(1.44, 1.44 + 0.12)
# ax.set_zlim(-0.06, 0.06)
# ax.axis("off")
# plt.show()
# fig.savefig(os.path.join(OUT_DIR, "8lm_patches.png"), dpi=300)
# fig.savefig(os.path.join(OUT_DIR, "8lm_patches.pdf"))
