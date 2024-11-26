"""WIP script to extract partial or full graph object models.

The ultimate goal of the script is to extract points observed by Monty at
various steps (or at least when it terminates) during evaluation experiments.

We also want to quantitatively calculate the proportion of surface area uncovered
by Monty during evaluation experiments.

For Phase I, I will:
1. Extract points for full object models from pretrained models.
2. Write a function to calculate the surface area of the object models.

Phase I will help me identify exactly what will need to be extracted from
evaluation experiments, as well have the surface area calculation function for
full or partial object models.

For Phase II, I will:
1. Need to modify `tbp.monty` to save point clouds at evaluation steps.

-------------------------------------------------------------------------------

Note: Please install `tbp.monty` in editable mode to run this script.
e.g. `pip install -e ~/tbp/tbp.monty`
"""

from pathlib import Path
from typing import List, Optional
import argparse
import numpy as np
import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from calculate_surface_area import estimate_surface_area_convex_hull


def extract_points(
    pretrained_model_path: Path, object_id: List[str] or str = "all"
) -> None:
    """Extract points from a pretrained model.

    Args:
        pretrained_model_path (Path): Path to pretrained model.
        object_id (List[str] or str): ID of object to extract, or "all" to extract
            all objects.

    Returns:
        dict: Dictionary containing points for each object.
    """
    if "model.pt" not in pretrained_model_path.name:
        pretrained_model_path = pretrained_model_path / "model.pt"

    # Load state dict.
    state_dict = torch.load(pretrained_model_path)

    # Extract the graph memory which contains the object models.
    graph_memory = state_dict["lm_dict"][0]["graph_memory"]

    pt_dict = {}

    if object_id == "all":
        for obj_id, obj_data in tqdm.tqdm(graph_memory.items()):
            # Data(
            #   x=[2264, 30],
            #   pos=[2264, 3],
            #   norm=[2264, 3],
            #   feature_mapping={
            #     node_ids=[2],
            #     pose_vectors=[2],
            #     pose_fully_defined=[2],
            #     on_object=[2],
            #     object_coverage=[2],
            #     rgba=[2],
            #     min_depth=[2],
            #     mean_depth=[2],
            #     hsv=[2],
            #     principal_curvatures=[2],
            #     principal_curvatures_log=[2],
            #     gaussian_curvature=[2],
            #     mean_curvature=[2],
            #     gaussian_curvature_sc=[2],
            #     mean_curvature_sc=[2]
            #   },
            #   edge_index=[2, 13584],
            #   edge_attr=[13584, 3]
            # )

            pt_dict[obj_id] = obj_data["patch"].pos.numpy()
    else:
        for obj_id in tqdm.tqdm(object_id):
            pt_dict[obj_id] = graph_memory[obj_id]["patch"].pos.numpy()

    return pt_dict

def plot_pointcloud(
    points: np.ndarray,
    show_axticks: bool = True,
    rotation: float = -80,
    ax: Optional[Axes3D] = None,
) -> matplotlib.figure.Figure:
    """Plot a 3D graph of an object model.

    TODO: add color_by option

    Args:
        points (np.ndarray): Points to plot.
        show_axticks: Whether to show axis ticks.
        rotation: Rotation of the 3D plot (moving camera up or down).
        ax: Axes3D instance to plot on. If not supplied, a figure and Axes3D
            instance will be created.

    Returns:
        Figure: the figure on which the graph was plotted.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.figure

    ax.scatter(points[:, 1], points[:, 0], points[:, 2], c=points[:, 2])

    if not show_axticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("x", labelpad=-10)
        ax.set_zlabel("z", labelpad=-15)
        ax.set_ylabel("y", labelpad=-15)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_aspect("equal")
    ax.view_init(rotation, 180)
    fig.tight_layout()
    plt.close(fig)
    return fig


def main(args):
    """Extract points from a pretrained model, save, and plot and save plots."""
    pt_dict = extract_points(args.pretrained_model, args.object_id)

    # Save each object's points as npy files.
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for obj_id, points in tqdm.tqdm(pt_dict.items()):
            np.save(args.save_dir / f"{obj_id}.npy", points)

    # Plot points, save in same directory but with _plots suffix.
    if args.save_dir:
        plots_dir = args.save_dir.with_name(args.save_dir.name + "_plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        for obj_id, points in tqdm.tqdm(pt_dict.items()):
            fig = plot_pointcloud(points)
            fig.savefig(plots_dir / f"{obj_id}.png")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to pretrained model.
    parser.add_argument("--pretrained_model", type=Path, required=True)
    parser.add_argument("--object_id", type=str, default="all")
    parser.add_argument(
        "--save_dir", type=Path, default=None, help="Directory to save points."
    )
    args = parser.parse_args()

    main(args)