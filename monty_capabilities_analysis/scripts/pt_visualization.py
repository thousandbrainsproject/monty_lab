import numpy as np
import open3d as o3d
from pathlib import Path


def main():
    points_dir = Path(
        "/Users/hlee/tbp/results/monty/pretrained_ycb_dmc/surf_agent_1lm/points"
    )
    for points_path in points_dir.glob("*.npy"):
        points = np.load(points_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Visualize the point cloud and wait for user input to close the window
        # o3d.visualization.draw_geometries_with_editing([pcd])

        # # Convert to mesh
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        #     pcd, alpha=0.01
        # )
        # # Visualize and wait for user input to close the window
        # o3d.visualization.draw_geometries([mesh])

        # # Convert to mesh using ball-pivot
        # # Calculate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(10)

        # 3. Create a denser point cloud using Poisson disk sampling
        # First create a preliminary mesh for better surface approximation
        mesh_init = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )[0]
        mesh_init.compute_vertex_normals()
        dense_pcd = mesh_init.sample_points_poisson_disk(
            number_of_points=len(pcd.points) * 4,  # Increase point count
            init_factor=5,
        )

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            dense_pcd, o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04])
        )
        # # fill holes
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.compute_vertex_normals()
        # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # mesh = tmesh.fill_holes(100000).to_legacy()
        # # Visualize and wait for user input to close the window
        o3d.visualization.draw_geometries([mesh])

        # Poisson surface reconstruction
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     pcd, depth=11
        # )
        # # Visualize and wait for user input to close the window
        # o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
