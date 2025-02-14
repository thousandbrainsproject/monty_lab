import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    DMC_ANALYSIS_DIR,
)

DMC_ROOT_DIR = Path(os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc")).expanduser()
VISUALIZATIONS_DIR = os.path.join(DMC_ROOT_DIR, "visualizations")
OUT_DIR = DMC_ANALYSIS_DIR / "multilm_patches"
OUT_DIR.mkdir(parents=True, exist_ok=True)

json_path = os.path.join(
    VISUALIZATIONS_DIR, "visualize_8lm_patches", "detailed_run_stats.json"
)
with open(json_path, "r") as f:
    stats = json.load(f)


sensor_module_id = "SM_8"
ep = stats["0"]
SM = ep[sensor_module_id]
obs = SM["raw_observations"][0]
rgba = np.array(obs["rgba"])
semantic_3d = np.array(obs["semantic_3d"])

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.imshow(rgba)
# ax.set_title(f"{sensor_module_id}")
# plt.show()

# %%


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
scatter = ax.scatter(x, y, z, c=c, marker="o", alpha=0.05)

width = 0.0003952849842607975

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
    scatter = ax.scatter(x, y, z, c=c, marker="o")


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
fig.savefig(os.path.join(OUT_DIR, "multilm_patches.png"), dpi=300)
fig.savefig(os.path.join(OUT_DIR, "multilm_patches.pdf"))

# %%
