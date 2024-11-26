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

import numpy as np
from scipy.spatial import ConvexHull


def estimate_surface_area_convex_hull(points: np.ndarray) -> float:
    """Estimate the surface area of a point cloud using a convex hull."""
    hull = ConvexHull(points)
    return hull.area
