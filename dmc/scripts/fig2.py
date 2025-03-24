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


"""

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    DMC_ANALYSIS_DIR,
    load_object_model,
)
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from plot_utils import (
    TBP_COLORS,
    axes3d_clean,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
)
from render_view_finder_images import VIEW_FINDER_DIR
from scipy.spatial.transform import Rotation as R

init_matplotlib_style()

OUT_DIR = DMC_ANALYSIS_DIR / "fig2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_agent_models_mug():
    out_dir = OUT_DIR / "agent_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), subplot_kw={"projection": "3d"})

    dist_mug = load_object_model("dist_agent_1lm", "mug")
    dist_mug -= np.array([0.0, 1.5, 0.0])

    touch_mug = load_object_model("touch_agent_1lm", "mug")
    touch_mug -= np.array([0.0, 1.5, 0.0])

    ax = axes[0]
    obj = dist_mug
    color = obj.rgba
    ax.scatter(obj.x, obj.y, obj.z, c=color, s=15, alpha=0.5, linewidths=0)
    ax.set_proj_type("persp", focal_length=1)
    axes3d_clean(ax, grid=False)
    axes3d_set_aspect_equal(ax)
    ax.axis("off")
    ax.view_init(120, -45, 48)
    ax.set_xlim(-0.055, 0.055)
    ax.set_ylim(-0.055, 0.055)
    ax.set_zlim(-0.055, 0.055)

    ax = axes[1]
    obj = touch_mug
    color = TBP_COLORS["blue"]
    values = obj.z
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.cm.gray
    color = cmap(norm(values) * 0.33 + 0.33)
    ax.scatter(obj.x, obj.y, obj.z, c=color, s=5, alpha=0.5)
    ax.set_proj_type("persp", focal_length=1)
    axes3d_clean(ax, grid=False)
    axes3d_set_aspect_equal(ax)
    ax.axis("off")
    ax.view_init(120, -45, 48)
    ax.set_xlim(-0.055, 0.055)
    ax.set_ylim(-0.055, 0.055)
    ax.set_zlim(-0.055, 0.055)

    fig.tight_layout()
    fig.savefig(out_dir / "mug.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "mug.svg", bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_agent_models_potted_meat_can():
    """Plot the potted meat can (i.e., Spam) object model.

    Plots 2 object models for the potted meat can -- one with color as learned
    by the distant agent, and one without color as learned by the touch agent.

    TODO: Delete me, probably.
    """

    out_dir = OUT_DIR / "agent_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot the distant agent's object model using stored colors.
    obj = load_object_model("dist_agent_1lm_10distinctobj", "potted_meat_can")
    obj -= np.array([0.0, 1.5, 0.0])
    obj = obj.rotated(R.from_euler("xyz", [90, 260, 0], degrees=True))

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(obj.x, obj.y, obj.z, c=obj.rgba, marker="o", s=10, alpha=1)
    ax.set_proj_type("persp", focal_length=1)
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
    obj -= np.array([0.0, 1.5, 0.0])
    obj = obj.rotated(R.from_euler("xyz", [90, 260, 0], degrees=True))

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(projection="3d")
    # Generate colors. Use the middle third of the colormap, otherwise its
    # pretty busy.
    values = obj.z
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.cm.magma
    rgba = cmap(norm(values) * 0.33 + 0.33)
    ax.scatter(obj.x, obj.y, obj.z, c=rgba, marker="o", s=10, alpha=1)
    ax.set_proj_type("persp", focal_length=1)
    axes3d_clean(ax)
    ax.view_init(elev=10, azim=10, roll=0)
    fig.tight_layout()
    plt.show()
    fig.savefig(
        out_dir / "potted_meat_can_touch_agent.png", bbox_inches="tight", dpi=300
    )
    fig.savefig(out_dir / "potted_meat_can_touch_agent.svg", pad_inches=0)


def plot_object_views(
    object_name: str,
    background: str = "gradient",
    vmin: float = 0.6,
    vmax: float = 0.8,
) -> None:
    """
    Loads view finder images of the potted meat can at 14 training rotations,
    and saves them as individual PNG and SVG files.
    """

    # Initialize input and output paths.
    data_dir = VIEW_FINDER_DIR / "view_finder_base_highres/view_finder_rgbd"
    png_dir = OUT_DIR / f"object_views/{object_name}/png"
    svg_dir = OUT_DIR / f"object_views/{object_name}/svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    # Load 'episodes.jsonl' to get info about potted_meat_can episodes.
    episodes = []
    with open(os.path.join(data_dir, "episodes.jsonl"), "r") as f:
        for line in f:
            episode = json.loads(line)
            episode_num = episode["episode"]
            name = episode["object"]
            if name != object_name:
                continue
            rotation = episode["rotation"]
            episodes.append((episode_num, object_name, rotation))

    # Plot each image as its own figure.
    for i, episode in enumerate(episodes):
        episode_number = episode[0]

        # Load the rgbd image, and alpha mask out any pixels that have a depth
        # greater than 0.9. We do this because we want to replace the black
        # background with something more visually appealing.
        rgbd = np.load(os.path.join(data_dir, f"arrays/{episode_number}.npy"))
        depth = rgbd[:, :, 3]
        rgba = rgbd.copy()
        rgba[:, :, 3] = 1
        masked = np.argwhere(depth > 0.9)
        rgba[masked[:, 0], masked[:, 1], 3] = 0

        # Put the image on a background.
        if background == "gradient":
            image = add_gradient_background(rgba, vmin=vmin, vmax=vmax)
        else:
            image = add_solid_background(rgba, background)

        fig = Figure(figsize=(1, 1))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(png_dir / f"{i}.png")
        fig.savefig(svg_dir / f"{i}.svg", bbox_inches="tight", pad_inches=0)


def add_solid_background(
    image: np.ndarray,
    color: str,
) -> np.ndarray:
    """
    Returns a solid background image of the given color.
    """
    width, height = image.shape[0], image.shape[1]
    c = np.array(to_rgba(color))
    bg = np.zeros([width, height, 4])
    bg[:, :] = c
    return blend_rgba_images(bg, image)


def add_gradient_background(
    image: np.ndarray, vmin: float = 0.7, vmax: float = 0.9
) -> np.ndarray:
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
    gradient = vmin + (vmax - vmin) * gradient[..., np.newaxis]
    bg = np.clip(gradient, vmin, vmax)
    bg = np.dstack((bg, bg, bg, np.ones((width, height))))

    # - Finally, blend the image with the background.
    return blend_rgba_images(bg, image)


def blend_rgba_images(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """
    Blends two RGBA image arrays using alpha compositing.

    Args:
        img1: NumPy array (H, W, 4) - First image (background)
        img2: NumPy array (H, W, 4) - Second image (foreground)

    Returns:
    - Blended image as an RGBA NumPy array.
    """
    assert background.shape == foreground.shape, "Images must have the same shape"
    assert background.shape[2] == 4, "Images must be RGBA (H, W, 4)"
    image_1, image_2 = foreground, background

    # Ensure 0-1 floats.
    if image_1.max() > 1:
        image_1 = image_1 / 255.0
    if image_2.max() > 1:
        image_2 = image_2 / 255.0

    # Extract RGB and Alpha channels
    rgb_1, alpha_1 = image_1[..., :3], image_1[..., 3:]
    rgb_2, alpha_2 = image_2[..., :3], image_2[..., 3:]

    # Compute blended alpha
    alpha_out = alpha_1 + alpha_2 * (1 - alpha_1)

    # Compute blended RGB
    rgb_out = (rgb_1 * alpha_1 + rgb_2 * alpha_2 * (1 - alpha_1)) / np.maximum(
        alpha_out, 1e-8
    )

    # Stack RGB and alpha back together
    return np.dstack((rgb_out, alpha_out))


def remove_svg_groups(
    input_svg: os.PathLike,
    output_svg: Optional[os.PathLike] = None,
    group_prefix: str = "axis3d",
):
    """Removes <g> elements with an id starting with `group_prefix`."""
    import xml.etree.ElementTree as ET

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
    ET.register_namespace("cc", "http://creativecommons.org/ns#")
    ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    # Parse the SVG file
    tree = ET.parse(input_svg)
    root = tree.getroot()

    # Define the SVG namespace
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Store elements to remove (to avoid modifying the tree while iterating)
    to_remove = []

    for parent in root.findall(".//svg:g/..", namespaces=ns):
        for g in parent.findall("svg:g", namespaces=ns):
            group_id = g.get("id", "")

            # Ensure <defs> elements are NOT removed
            if g.tag.endswith("defs"):
                continue  # Skip <defs> elements

            # Remove only groups that start with the given prefix
            if group_id.startswith(group_prefix):
                to_remove.append((parent, g))

    # Safely remove elements
    for parent, g in to_remove:
        parent.remove(g)

    # Write the modified SVG while preserving formatting
    tree.write(output_svg, encoding="utf-8", xml_declaration=True, method="xml")


def plot_pretraining_epochs():
    out_dir = OUT_DIR / "pretraining_epochs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 7, figsize=(15, 5), subplot_kw={"projection": "3d"})

    axes = axes.flatten()
    for i, ax in enumerate(axes.flatten()):
        obj = load_object_model(
            "dist_agent_1lm_checkpoints", "potted_meat_can", checkpoint=i + 1
        )
        obj -= np.array([0.0, 1.5, 0.0])
        # color = TBP_COLORS["blue"]
        ax.scatter(obj.x, obj.y, obj.z, c=obj.rgba, s=5, alpha=0.5)
        ax.set_proj_type("persp", focal_length=1)
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        ax.view_init(115, -50, 40)
        ax.set_xlim(-0.055, 0.055)
        ax.set_ylim(-0.055, 0.055)
        ax.set_zlim(-0.055, 0.055)
    fig.tight_layout()
    fig.savefig(out_dir / "pretraining_epochs.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "pretraining_epochs.svg", bbox_inches="tight", pad_inches=0)
    plt.show()

    input_file = out_dir / "pretraining_epochs.svg"
    output_file = out_dir / "pretraining_epochs.svg"
    remove_svg_groups(input_file, output_file, group_prefix="axis3d_")


# plot_agent_models_mug()
# plot_agent_models_potted_meat_can()
plot_object_views("potted_meat_can", background="gradient")
plot_object_views("mug", background="white")
# plot_pretraining_epochs()
