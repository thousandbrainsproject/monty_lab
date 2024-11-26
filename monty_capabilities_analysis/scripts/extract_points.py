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
# Calculating Surface Area from Point Clouds

There are several ways to calculate the surface area of a point cloud, each with
their own sub-variations. I will roughly sketch out the methods I am considering
below.

1. Using a convex hull to approximate the surface area.
- This is the simplest and fastest method, but will not be accurate for
non-convex shapes. Still, it may be a good sanity checker for convex shapes.

2. Using local density estimates to approximate the surface area.
- This works directly with points without any surface reconstruction. It is
related to mesh methods in that it approximates local surface patches. A key
parameter is the radius of the local neighborhood used to estimate the density.
- If we incorporate surface normals, it could be more accurate.

3. Using a alpha shape to approximate the surface area.
- This is a generalization of the convex hull that works well for both convex
and non-convex shapes.
- It makes use of the Delaunay triangulation whose circumsphere radius is below
specific threshold (alpha). Surface area is calculated as sum of all triangular
faces.

4. Mesh-based Reconstruction
- This is the most accurate method, but also the most computationally expensive.
- It involves reconstructing a mesh from the point cloud using a variety of
methods including Ball Pivoting, Alpha Shape, and others.
- Surface area is calculated as sum of all triangular faces of the mesh.

5. Voxel-based Reconstruction
- This is a middle-ground between point cloud and mesh methods.
- Surfaces are discretized into a voxel grid.
- Surface area is approximated by the sum of all surface voxels.

Workflows:
- Points -> Local Density (quick estimate)
- Points -> Alpha Shape -> Mesh (accurate surface)
- Points -> Voxels -> Marching Cubes -> Mesh (alternative path)

-------------------------------------------------------------------------------
"""

from pathlib import Path

import argparse

import torch


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
    graph_memory = state_dict["lm_dict"]["0"]["graph_memory"]

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to pretrained model.
    parser.add_argument("--pretrained_model", type=Path, required=True)
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model.expanduser()
    main(pretrained_model_path)
