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
This module defines functions used to generate images for figure 2.

- `plot_potted_meat_can_object_models`: Plots the potted meat can (i.e., Spam)
object models for the distant and touch agents.
- `plot_potted_meat_can_views`: Plots the view finder images of the potted meat can
at 14 training rotations.



"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    DMC_ANALYSIS_DIR,
    load_object_model,
)
from PIL import Image
from plot_utils import axes3d_clean
from render_view_finder_images import VIEW_FINDER_DIR

OUT_DIR = DMC_ANALYSIS_DIR / "fig2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_potted_meat_can_object_models():
    """Plot the potted meat can (i.e., Spam) object model.

    Plots 2 object models for the potted meat can -- one with color as learned
    by the distant agent, and one without color as learned by the touch agent.
    """

    out_dir = OUT_DIR / "object_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot the distant agent's object model using stored colors.
    obj = load_object_model("dist_agent_1lm_10distinctobj", "potted_meat_can")
    obj -= obj.translation
    obj = obj.rotated(90, 260, 0)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(obj.x, obj.y, obj.z, c=obj.rgba, marker="o", s=10, alpha=1)
    axes3d_clean(ax)
    ax.view_init(elev=10, azim=10, roll=0)
    fig.tight_layout()
    plt.show()
    fig.savefig(
        out_dir / "potted_meat_can_dist_agent.png", bbox_inches="tight", dpi=300
    )
    fig.savefig(out_dir / "potted_meat_can_dist_agent.svg", pad_inches=0)

    # Plot the touch agent's object model. Generate colors.
    obj = load_object_model("touch_agent_1lm", "potted_meat_can")
    obj -= obj.translation
    obj = obj.rotated(90, 260, 0)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(projection="3d")
    # Generate colors. Use the middle third of the colormap, otherwise its
    # pretty busy.
    values = obj.z
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.cm.magma
    rgba = cmap(norm(values) * 0.33 + 0.33)
    ax.scatter(obj.x, obj.y, obj.z, c=rgba, marker="o", s=10, alpha=1)
    axes3d_clean(ax)
    ax.view_init(elev=10, azim=10, roll=0)
    fig.tight_layout()
    plt.show()
    fig.savefig(
        out_dir / "potted_meat_can_touch_agent.png", bbox_inches="tight", dpi=300
    )
    fig.savefig(out_dir / "potted_meat_can_touch_agent.svg", pad_inches=0)


def plot_potted_meat_can_views():
    """
    Loads view finder images of the potted meat can at 14 training rotations,
    and saves them as individual PNG and SVG files.
    """

    # Initialize input and output paths.
    data_dir = VIEW_FINDER_DIR / "view_finder_base/view_finder_rgbd"
    png_dir = OUT_DIR / "potted_meat_can_views/png"
    svg_dir = OUT_DIR / "potted_meat_can_views/svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    # Load 'episodes.jsonl' to get info about potted_meat_can episodes.
    episodes = []
    with open(os.path.join(data_dir, "episodes.jsonl"), "r") as f:
        for line in f:
            episode = json.loads(line)
            episode_num = episode["episode"]
            object_name = episode["object"]
            if object_name != "potted_meat_can":
                continue
            rotation = episode["rotation"]
            episodes.append((episode_num, object_name, rotation))

    # Plot each image as its own figure.
    for i, episode in enumerate(episodes):
        episode_number = episode[0]

        # Load the rgbd image, and alpha mask out any pixels that have a depth
        # greater than 0.9. We do this because we want to place the objects over
        # a neater (gray) background to match other plots in the figure.
        rgbd = np.load(os.path.join(data_dir, f"arrays/{episode_number}.npy"))
        depth = rgbd[:, :, 3]
        rgba = rgbd.copy()
        rgba[:, :, 3] = 1
        masked = np.argwhere(depth > 0.9)
        rgba[masked[:, 0], masked[:, 1], 3] = 0

        # Put the image on the gray background, and plot it.
        image = put_image_on_gray_background(rgba)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(image)
        ax.axis("off")
        fig.tight_layout()

        fig.savefig(png_dir / f"{i}.png", bbox_inches="tight", dpi=300)
        fig.savefig(svg_dir / f"{i}.svg", bbox_inches="tight", pad_inches=0)

        plt.show()


def blend_rgba_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Blends two RGBA image arrays using alpha compositing.

    Parameters:
    - img1: NumPy array (H, W, 4) - First image (background)
    - img2: NumPy array (H, W, 4) - Second image (foreground)

    Returns:
    - Blended image as an RGBA NumPy array.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    assert img1.shape[2] == 4, "Images must be RGBA (H, W, 4)"

    # Extract RGB and Alpha channels
    rgb1, alpha1 = img1[..., :3], img1[..., 3:]
    rgb2, alpha2 = img2[..., :3], img2[..., 3:]

    # Compute blended alpha
    alpha_out = alpha1 + alpha2 * (1 - alpha1)

    # Compute blended RGB
    rgb_out = (rgb1 * alpha1 + rgb2 * alpha2 * (1 - alpha1)) / np.maximum(
        alpha_out, 1e-8
    )

    # Stack RGB and alpha back together
    blended = np.dstack((rgb_out, alpha_out)) * 255  # Convert back to 0-255 range
    return blended.astype(np.uint8)


def put_image_on_gray_background(image: np.ndarray) -> np.ndarray:
    """
    Puts an image on a random (grayscale) gradient background.

    Args:
        image: The image to put on the background.
    """

    width, height = image.shape[0], image.shape[1]

    # First, create the gradient background.
    # - Make pixel coordinates.
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    # - Compute the randomly oriented gradient.
    theta = np.random.uniform(0, 2 * np.pi)
    gradient = (X * np.cos(theta) + Y * np.sin(theta)) / np.sqrt(width**2 + height**2)

    # Scale gradient to desired range, and put in an RGBA array.
    vmin, vmax = 0.2, 0.5
    gradient = vmin + (vmax - vmin) * gradient[..., np.newaxis]
    bg = np.clip(gradient, vmin, vmax)
    bg = np.dstack((bg, bg, bg, np.ones((width, height))))

    # - Finally, blend the image with the background.
    blended = blend_rgba_images(image, bg)

    return blended


