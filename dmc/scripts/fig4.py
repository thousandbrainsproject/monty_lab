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
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
)

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
    rgba = np.array(sm_dict["raw_observations"][0]["rgba"])
    semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
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


sm_num = 0
sm_dict = stats[f"SM_{sm_num}"]
rgba = np.array(sm_dict["raw_observations"][0]["rgba"])
semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
x = semantic_3d[:, 0].flatten()
y = semantic_3d[:, 1].flatten()
z = semantic_3d[:, 2].flatten()
sem = semantic_3d[:, 3].flatten().astype(int)
x = x[sem == 1]
y = y[sem == 1]
z = z[sem == 1]
c = rgba.reshape(-1, 4)
c = c[sem == 1] / 255

# # Create a 3D plot of the semantic point cloud
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="3d")
# ax.view_init(elev=0, azim=0, roll=0, vertical_axis="y")
# # Extract x,y,z coordinates from semantic_3d

# # view finder
# x, y, z, c = pull(8)
# ax.scatter(x, y, z, c=c, marker="o", alpha=0.05)

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
