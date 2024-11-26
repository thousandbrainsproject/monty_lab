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

import argparse

import torch

from calculate_surface_area import estimate_surface_area_convex_hull


def main(pretrained_model_path: Path, object_id: str = "all") -> None:
    """Extract points from a pretrained model.

    Args:
        pretrained_model_path: Path to pretrained model.
        object_id: ID of object to extract, or "all" to extract all objects.

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
        for obj_id, obj_data in graph_memory.items():
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

            pt_dict[obj_id] = obj_data.pos.numpy()
    else:
        pt_dict[object_id] = graph_memory[object_id].pos.numpy()

    return pt_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to pretrained model.
    parser.add_argument("--pretrained_model", type=Path, required=True)
    parser.add_argument("--object_id", type=str, default="all")
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model.expanduser()
    pt_dict = main(pretrained_model_path, args.object_id)

    print(pt_dict)
    print()
