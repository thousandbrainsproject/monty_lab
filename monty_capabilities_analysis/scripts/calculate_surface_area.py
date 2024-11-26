"""Calculate the surface area of a point cloud.

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
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def estimate_sa_local_density(points: np.ndarray, radius: float, k: int = 30) -> float:
    """Estimate the surface area of a point cloud using local density.

    This implementation uses the relationship between point density and surface area:
    Surface Area ≈ N / (ρ * h)
    where:
    - N is number of points
    - ρ is the local point density per unit area
    - h is the thickness of the sampling shell

    Args:
        points: (N, 3) array of points.
        radius: Radius of the local neighborhood. If None, estimated from k-nearest neighbors.
        k: Number of neighbors to use for radius estimation if radius is None.

    Returns:
        float: Estimated surface area
    """
    # Estimate radius if not provided
    if radius is None:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(points)
        distances, _ = nbrs.kneighbors(points)
        radius = np.median(distances[:, 1:]) * 1.5

    # Find all points within radius of each point
    nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(points)
    distances, indices = nbrs.radius_neighbors(points)

    # Calculate local densities (points per unit volume)
    local_densities = np.array([len(idx) for idx in indices]) / (
        4 / 3 * np.pi * radius**3
    )

    # Estimate the effective thickness of the point cloud
    # (assuming points are distributed in a thin shell around the true surface)
    thickness = radius * 0.1  # typically about 10% of the radius

    # Calculate surface area using the density-area relationship
    # SA = N / (ρ * h) where ρ is points per unit volume
    avg_density = np.mean(local_densities)
    surface_area = len(points) / (avg_density * thickness)

    return surface_area

def estimate_surface_area_convex_hull(points: np.ndarray) -> float:
    """Estimate the surface area of a point cloud using a convex hull."""
    hull = ConvexHull(points)
    return hull.area

def main(args):
    """Calculate the surface area of a point cloud."""
    results = []
    for points_path in args.points_dir.glob("*.npy"):
        points = np.load(points_path)
        convex_hull_area = estimate_surface_area_convex_hull(points)
        print(f"Surface area (Convex Hull): {convex_hull_area}")

        local_density_area = estimate_sa_local_density(points, radius=None)
        print(f"Surface area (Local Density): {local_density_area}")

        results.append(
            {
                "object_id": points_path.stem,
                "surface_area_convex_hull": convex_hull_area,
                "surface_area_local_density": local_density_area,
            }
        )
    df = pd.DataFrame(results)
    df.to_csv(args.points_dir.parent / "surface_area.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
