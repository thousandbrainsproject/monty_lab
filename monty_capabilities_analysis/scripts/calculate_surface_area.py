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
from scipy.spatial import Delaunay
import itertools
import open3d as o3d


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

def estimate_sa_alpha_shape(points: np.ndarray, alpha: float = None) -> float:
    """Estimate the surface area using alpha shapes.

    Args:
        points: (N, 3) array of points
        alpha: Alpha value for filtering simplices. If None, estimated from point density.
            Smaller alpha creates a more detailed shape, larger alpha approaches convex hull.

    Returns:
        float: Estimated surface area
    """
    # Estimate alpha if not provided
    if alpha is None:
        # Use average distance to k nearest neighbors as a heuristic
        k = min(30, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(points)
        distances, _ = nbrs.kneighbors(points)
        alpha = np.median(distances[:, 1:]) * 2.0

    # Compute Delaunay triangulation
    tri = Delaunay(points)

    # For each tetrahedron, compute circumradius
    tetras = points[tri.simplices]
    circum_radii = np.zeros(len(tetras))

    for i, tetra in enumerate(tetras):
        # Get circumcenter and radius
        a, b, c, d = tetra
        circum_radii[i] = _get_circumradius(a, b, c, d)

    # Filter tetrahedra based on alpha criterion
    valid_tetras = tri.simplices[circum_radii <= alpha]

    # Get surface triangles (those that appear only once)
    faces = _get_surface_triangles(valid_tetras)

    # Calculate total surface area
    surface_area = 0.0
    for face in faces:
        triangle = points[face]
        surface_area += _triangle_area(triangle)

    return surface_area


def _get_circumradius(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    """Calculate the circumradius of a tetrahedron."""
    # Matrix of squared distances
    m = np.zeros((4, 4))
    vertices = [a, b, c, d]

    for i, j in itertools.combinations(range(4), 2):
        dist = np.sum((vertices[i] - vertices[j]) ** 2)
        m[i, j] = m[j, i] = dist

    # Calculate circumradius using determinant method
    M = np.ones((5, 5))
    M[0:4, 0:4] = m
    M[4, 0:4] = 1
    M[0:4, 4] = 1
    M[4, 4] = 0

    det = np.linalg.det(M)
    if abs(det) < 1e-10:  # Handle degenerate case
        return float("inf")

    return np.sqrt(det / (288.0))


def _get_surface_triangles(tetras: np.ndarray) -> np.ndarray:
    """Find triangular faces that appear only once (surface triangles)."""
    # Get all triangular faces
    faces = []
    for tetra in tetras:
        for face in itertools.combinations(tetra, 3):
            face = tuple(sorted(face))
            faces.append(face)

    # Count occurrences of each face
    from collections import Counter

    face_counts = Counter(faces)

    # Return faces that appear exactly once
    surface_faces = np.array(
        [face for face, count in face_counts.items() if count == 1]
    )
    return surface_faces


def _triangle_area(triangle: np.ndarray) -> float:
    """Calculate the area of a triangle using cross product."""
    a, b, c = triangle
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def estimate_sa_mesh(points: np.ndarray, method: str = "poisson", **kwargs) -> float:
    """Estimate surface area using mesh reconstruction.

    Args:
        points: (N, 3) array of points
        method: Reconstruction method ('poisson' or 'ball_pivot')
        **kwargs: Additional parameters for reconstruction
            For Poisson:
                depth: int = 8  # Octree depth
                scale: float = 1.1  # Greater than 1, scales the reconstruction
                linear_fit: bool = False  # Use linear interpolation
            For Ball Pivoting:
                radii: List[float] = None  # Radii for ball pivoting. If None, estimated.

    Returns:
        float: Estimated surface area
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate appropriate radius for normal estimation
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist * 3  # Use 3x average nearest neighbor distance

    # Estimate normals if not present
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    if method == "poisson":
        # Poisson surface reconstruction
        depth = kwargs.get("depth", 8)
        scale = kwargs.get("scale", 1.1)
        linear_fit = kwargs.get("linear_fit", False)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale, linear_fit=linear_fit
        )

        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    elif method == "ball_pivot":
        # Ball Pivoting surface reconstruction
        radii = kwargs.get("radii", None)
        if radii is None:
            # Estimate radius from point cloud density
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clean up the mesh
    mesh = clean_mesh(mesh)

    # Calculate surface area
    return np.sum(mesh.get_surface_area())


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Clean up the mesh by removing degenerate triangles and filling holes."""
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()

    # Remove duplicated triangles
    mesh.remove_duplicated_triangles()

    # Remove duplicated vertices
    mesh.remove_duplicated_vertices()

    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    return mesh


def estimate_sa_voxel(
    points: np.ndarray, voxel_size: float = None, padding: float = 1.1
) -> float:
    """Estimate surface area using voxelization.

    Args:
        points: (N, 3) array of points
        voxel_size: Size of each voxel. If None, estimated from point density.
        padding: Padding factor for voxel size estimation (> 1.0)

    Returns:
        float: Estimated surface area
    """
    # Estimate voxel size if not provided
    if voxel_size is None:
        # Use average distance to nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(points)
        distances, _ = nbrs.kneighbors(points)
        voxel_size = np.median(distances[:, 1]) * padding

    # Create voxel grid
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)

    # Calculate grid dimensions
    dims = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int) + 1

    # Initialize voxel grid
    grid = np.zeros(dims, dtype=bool)

    # Convert points to voxel coordinates
    voxel_coords = np.floor((points - min_bounds) / voxel_size).astype(int)

    # Mark occupied voxels
    grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True

    # Find surface voxels using 6-connectivity
    surface_voxels = _find_surface_voxels(grid)

    # Calculate surface area
    # Each surface voxel contributes its exposed faces to the total area
    exposed_faces = _count_exposed_faces(grid, surface_voxels)
    surface_area = exposed_faces * (voxel_size**2)

    return surface_area


def _find_surface_voxels(grid: np.ndarray) -> np.ndarray:
    """Find surface voxels using 6-connectivity."""
    # Pad grid to handle boundaries
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=False)

    # Initialize surface voxels array
    surface = np.zeros_like(grid, dtype=bool)

    # Check 6-connectivity
    for dx, dy, dz in [
        (-1, 0, 0),
        (1, 0, 0),  # left, right
        (0, -1, 0),
        (0, 1, 0),  # front, back
        (0, 0, -1),
        (0, 0, 1),  # bottom, top
    ]:
        # A voxel is on the surface if it's occupied and has at least
        # one empty neighbor in 6-connectivity
        neighbor_empty = ~padded[
            1 + dx : padded.shape[0] - 1 + dx,
            1 + dy : padded.shape[1] - 1 + dy,
            1 + dz : padded.shape[2] - 1 + dz,
        ]
        surface |= grid & neighbor_empty

    return surface


def _count_exposed_faces(grid: np.ndarray, surface: np.ndarray) -> int:
    """Count number of exposed faces for surface voxels."""
    # Pad grid to handle boundaries
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=False)

    total_faces = 0
    x, y, z = np.where(surface)

    # For each surface voxel, count empty neighbors
    for i, j, k in zip(x, y, z):
        # Check all 6 directions
        for di, dj, dk in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            if not padded[i + di + 1, j + dj + 1, k + dk + 1]:
                total_faces += 1

    return total_faces


def calculate_accurate_surface_area(points: np.ndarray) -> float:
    """Calculate accurate surface area using Open3D's Poisson reconstruction."""
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate appropriate radius for normal estimation
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist * 3  # Use 3x average nearest neighbor distance

    # Estimate normals with careful parameters
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=50,  # Increased for better normal estimation
        )
    )

    # ... rest of the function remains the same ...


def main(args):
    """Calculate the surface area of a point cloud."""
    results = []
    for points_path in args.points_dir.glob("*.npy"):
        points = np.load(points_path)
        convex_hull_area = estimate_surface_area_convex_hull(points)
        print(f"Surface area (Convex Hull): {convex_hull_area}")

        local_density_area = estimate_sa_local_density(points, radius=None)
        print(f"Surface area (Local Density): {local_density_area}")

        alpha_shape_area = estimate_sa_alpha_shape(points, alpha=None)
        print(f"Surface area (Alpha Shape): {alpha_shape_area}")

        # Calculate mesh-based surface areas
        poisson_area = estimate_sa_mesh(points, method="poisson")
        ball_pivot_area = estimate_sa_mesh(points, method="ball_pivot")

        print(f"Surface area (Poisson): {poisson_area}")
        print(f"Surface area (Ball Pivot): {ball_pivot_area}")

        # Calculate voxel-based surface area
        voxel_area = estimate_sa_voxel(points)
        print(f"Surface area (Voxel): {voxel_area}")

        results.append(
            {
                "object_id": points_path.stem,
                "surface_area_convex_hull": convex_hull_area,
                "surface_area_local_density": local_density_area,
                "surface_area_alpha_shape": alpha_shape_area,
                "surface_area_poisson": poisson_area,
                "surface_area_ball_pivot": ball_pivot_area,
                "surface_area_voxel": voxel_area,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(args.points_dir.parent / "surface_area.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
