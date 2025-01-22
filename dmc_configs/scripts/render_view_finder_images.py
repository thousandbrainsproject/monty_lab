# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Render images from `view_finder_images` experiments.

Three configs are defined:
- view_finder_base: 14 standard training rotations
- view_finder_randrot_all: 14 randomly generated rotations
- view_finder_randrot: 5 pre-defined "random" rotations

All use 77 objects.

To visualize the images, run the script
`monty_lab/dmc_config/scripts/render_view_finder_images.py`.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm

experiment = "view_finder_randrot"

figure_settings = {
    "view_finder_base": {
        "n_rotations": 14,
        "n_rows": 2,
        "n_cols": 7,
        "figsize": (8, 4),
    },
    "view_finder_randrot_all": {
        "n_rotations": 14,
        "n_rows": 2,
        "n_cols": 7,
        "figsize": (8, 4),
    },
    "view_finder_randrot": {
        "n_rotations": 5,
        "n_rows": 1,
        "n_cols": 5,
        "figsize": (4, 4),
    },
}
view_finder_dir = Path("~/tbp/results/dmc").expanduser() / "view_finder_images"
data_dir = view_finder_dir / f"{experiment}/view_finder_rgbd"
arrays_dir = data_dir / "arrays"
visualization_dir = data_dir / "visualizations"
visualization_dir.mkdir(parents=True, exist_ok=True)

# Load episodes.json to get object ids and rotations.
# Items are tuples of (episode_num, object_name, rotation)
episodes = []
with open(os.path.join(data_dir, "episodes.jsonl"), "r") as f:
    for line in f:
        episode = json.loads(line)
        episode_num = episode["episode"]
        object_name = episode["object"]
        rotation = episode["rotation"]
        episodes.append((episode_num, object_name, rotation))

# Let's get unique rotations
rotations = [tuple(episode[2]) for episode in episodes]
unique_rotations = set(rotations)

# Let's get unique objects so we can create 4 x 8 grids for each object.
objects = [episode[1] for episode in episodes]
unique_objects = set(objects)


def _get_episodes_for_object(object_name, episodes):
    # get all episodes for the given object sorted be episode number
    episodes = [episode for episode in episodes if episode[1] == object_name]
    # sort the episodes by rotations
    episodes = sorted(episodes, key=lambda x: x[2])
    return episodes


def _plot_all_rotations_for_object(
    object_name, episodes, visualization_dir, fig_params
):
    figsize = fig_params["figsize"]
    n_rows, n_cols = fig_params["n_rows"], fig_params["n_cols"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, episode in enumerate(episodes):
        episode_number = episode[0]
        rotation = episode[2]
        image_rgbd = np.load(os.path.join(data_dir, f"arrays/{episode_number}.npy"))
        image_rgb = image_rgbd[:, :, :3]
        ax = axes.flatten()[i]
        ax.imshow(image_rgb)
        # Title with the rotation
        title = "[{:d}, {:d}, {:d}]".format(*map(round, rotation))
        ax.set_title(title, fontsize=6)
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(
        os.path.join(visualization_dir, f"{object_name}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

for object_name in tqdm.tqdm(unique_objects):
    object_episodes = _get_episodes_for_object(object_name, episodes)
    fig_params = figure_settings[experiment]
    _plot_all_rotations_for_object(
        object_name, object_episodes, visualization_dir, fig_params
    )

