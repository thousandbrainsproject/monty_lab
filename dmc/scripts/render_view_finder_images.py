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

This script generates and saves a set of figures from arrays saved by
`view_finder_images` experiments. These aren't "true" experiments part of the
Demonstrating Monty's Capabilities paper, but rather they were used to generating
views of images used to train and evaluate a ViT model against which we compare
Monty. To run this script, call

```
$ python render_view_finder_images.py -e EXPERIMENT_NAME
```

Running this script creates the directory
`EXPERIMENT_DIR/view_finder_rgbd/visualizations` where `EXPERIMENT_DIR` is one of
 - `view_finder_base`: 14 standard training rotations
 - `view_finder_randrot_all`: 14 randomly generated rotations
 - `view_finder_randrot`: 5 pre-defined "random" rotations
 - `view_finder_32`: 32 rotations used for rapid learning experiments.

77 figures will be created, one for each object, showing all its rotations.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm

# The path to the directory where all view_finder_images experiments are stored.
DMC_ROOT_DIR = Path(os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc")).expanduser()
VIEW_FINDER_DIR = DMC_ROOT_DIR / "view_finder_images"

# Settings for each experiment.
figure_settings = {
    "view_finder_base": {
        "n_rotations": 14,
        "n_rows": 2,
        "n_cols": 7,
    },
    "view_finder_randrot_all": {
        "n_rotations": 14,
        "n_rows": 2,
        "n_cols": 7,
    },
    "view_finder_randrot": {
        "n_rotations": 5,
        "n_rows": 1,
        "n_cols": 5,
    },
    "view_finder_32": {
        "n_rotations": 32,
        "n_rows": 4,
        "n_cols": 8,
    },
}
inches_per_subplot = 1.5
h_pad = 0.25  # Height padding between rows and left/right margins.
w_pad = 0.20  # Width padding between columns and top/bottom margins.
for fig_params in figure_settings.values():
    n_rows, n_cols = fig_params["n_rows"], fig_params["n_cols"]
    width = n_cols * inches_per_subplot + (n_cols + 1) * w_pad
    height = n_rows * inches_per_subplot + (n_rows + 1) * h_pad
    fig_params["figsize"] = (width, height)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render images from view_finder_images experiments"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        choices=[
            "view_finder_base",
            "view_finder_randrot_all",
            "view_finder_randrot",
            "view_finder_32",
        ],
        help="Name of experiment to render images from",
    )
    return parser.parse_args()


def render_figures(experiment: str) -> None:
    """Render figures for each object showing its rotations"""

    # Initialize storage directory.
    data_dir = VIEW_FINDER_DIR / f"{experiment}/view_finder_rgbd"
    visualization_dir = data_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # Load 'episodes.jsonl' to get object ids and rotations.
    # Items are tuples of (episode_num, object_name, rotation)
    episodes = []
    with open(os.path.join(data_dir, "episodes.jsonl"), "r") as f:
        for line in f:
            episode = json.loads(line)
            episode_num = episode["episode"]
            object_name = episode["object"]
            rotation = episode["rotation"]
            episodes.append((episode_num, object_name, rotation))

    # Iterate through all objects, making a figure for each.
    unique_objects = list(sorted(set([episode[1] for episode in episodes])))
    fig_params = figure_settings[experiment]
    print(fig_params)
    for object_name in tqdm.tqdm(unique_objects):
        object_episodes = _get_episodes_for_object(object_name, episodes)
        fig = _plot_all_rotations_for_object(object_episodes, fig_params, data_dir)
        fig_path = visualization_dir / f"{object_name}.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()


def _get_episodes_for_object(object_name: str, episodes: List[Tuple]) -> List[Tuple]:
    # get all episodes for the given object sorted be episode number
    episodes = [episode for episode in episodes if episode[1] == object_name]
    # sort the episodes by rotations
    episodes = sorted(episodes, key=lambda x: x[2])
    return episodes


def _plot_all_rotations_for_object(
    episodes: List[Tuple],
    fig_params: Mapping,
    data_dir: os.PathLike,
) -> None:
    """Create and save a figure for a given object showing all its rotations."""
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
        ax.set_title(title, fontsize=10)
        # remove ticks
        ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args()
    experiment = args.experiment
    render_figures(experiment)
