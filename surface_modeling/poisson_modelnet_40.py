# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from modelnet import ModelNet40
from torch_geometric import transforms as T

from experiments.transforms import RandomRotate

N_POINTS = [100, 1_000, 10_000]
DEPTHS = [3, 5, 7, 9, 11]

# NOTE: to run this script install open3d!


class RemoveFaces(T.BaseTransform):
    """Borrowed from Karan in preprocess modelnet 40"""
    def __call__(self, data):
        return data.pos


def get_transform(num):
    """Borrowed from Karan in preprocess modelnet 40"""
    transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(num=num),
        RemoveFaces()
    ])
    return transform


def get_rotation_transform(num, axes=("x")):
    """Borrowed from Karan in preprocess modelnet 40"""
    rotation = RandomRotate(axes=axes)
    transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(num=num),
        RemoveFaces(),
        rotation,
    ])
    return transform, rotation.rotation_matrix


def modelnet40_to_poisson_surface(obj, depth=9, **kwargs):
    """
    Take in a torch tensor and
        1) convert to o3d point cloud
        2) estimate point normals
        3) do poisson surface reconstruction

    :param obj: tensor[n x 3]
    :param depth: int how deep the octree; default 9 based on tutorial
    :param kwargs: other stuf!
    :return mesh: o3d triangle mesh reconstruction
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # clear existing data
    pcd.estimate_normals()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as _:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            **kwargs)

    return mesh


def get_single_modelnet40_sample(idx):
    """
    Instantiance ModelNet40 dataset and return a dataset[idx]
    """
    dataset = ModelNet40(
        root=os.path.expanduser("~/tbp/datasets/ModelNet40/raw"),
        transform=None,  # raw torch geometric object
        train=True,
        num_samples_train=2,
    )

    return dataset.data[idx]


def scan_n_points(n_points, **kwargs):
    """
    Create modelnet 40 dataset, sample an object, try reconstructing with different
    numbers of sampled points. kwargs are passed to poisson_reconstruct. Use this to
    get a feel for how number of points affects reconstruction quality.
    """
    sample = get_single_modelnet40_sample(idx=6)

    for n in n_points:
        x = copy.deepcopy(sample)  # transforms mutate original data
        transform = get_transform(n)
        transformed_x = transform(x)
        mesh = modelnet40_to_poisson_surface(transformed_x, **kwargs)
        o3d.visualization.draw_geometries([mesh])


def scan_depth(depths, n, **kwargs):
    """
    Like scan_n_points, but fix n and scan depth of octree. Use this to get a feel for
    how depth affects reconstruction quality.
    """
    sample = get_single_modelnet40_sample(idx=6)
    transform = get_transform(n)
    transformed_x = transform(sample)  # point cloud with n points

    for depth in depths:
        mesh = modelnet40_to_poisson_surface(transformed_x, depth=depth, **kwargs)
        o3d.visualization.draw_geometries([mesh])


def mesh_to_implicit(mesh):
    """
    Take an open3d triangle mesh and create a ray casting scene from whih we can compute
    distances of query points.
    """
    mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_)
    return scene


def compute_dist_and_occopancy(scene, query):
    """
    Given an open3d scene (output of mesh_to_implicit) and a query point ([,3] tensor)
    measure 3 types of distances and return in a single array.
    """
    typed_query = o3d.core.Tensor([query.numpy()], dtype=o3d.core.Dtype.Float32)
    unsigned_distance = scene.compute_distance(typed_query)
    signed_distance = scene.compute_signed_distance(typed_query)
    occupancy = scene.compute_occupancy(typed_query)
    return np.squeeze(
        np.array([
            unsigned_distance.numpy(),
            signed_distance.numpy(),
            occupancy.numpy()
        ])
    )


def get_distances_of_points_to_mesh(reconstructed_mesh, surface_points):
    """
    Given a mesh and a set of query points, calculate distances of each point to mesh.
    Basically a loop that wraps compute_dist_and_occupancy.
    """
    # 3 dims are unsigned, signed, occupancy
    n_points = surface_points.size()[0]
    estimated_distances = np.zeros((n_points, 3))
    scene = mesh_to_implicit(reconstructed_mesh)
    for i in range(n_points):
        query = surface_points[i, :]
        estimated_distances[i, :] = compute_dist_and_occopancy(scene, query)

    return estimated_distances


def reconstruction_experiment(n_train_samples, n_eval_samples, **kwargs):
    """
    1) Sample n_train_samples from a mesh to make a pointcloud.
    2) Reconstruct with Poisson surface reconstruction.
    3) Sample n_eval_samples from the ground truth mesh.
    4) Evaluate reconstruction on eval points.
    """

    model_net_sample = get_single_modelnet40_sample(idx=6)
    train_transform = get_transform(n_train_samples)
    eval_transform = get_transform(n_eval_samples)

    x_train = train_transform(copy.deepcopy(model_net_sample))
    x_eval = eval_transform(copy.deepcopy(model_net_sample))
    mesh = modelnet40_to_poisson_surface(x_train, **kwargs)
    distances = get_distances_of_points_to_mesh(mesh, x_eval)

    return distances


def get_distance_statistics(distances):
    """
    Distances is assumed to be output of reconstruction_experiment. Compute measures
    assuming ground truth distance is always 0.
    """
    return distances.mean(dim=1)  # quick and easy for now


if __name__ == "__main__":

    print("*" * 20)
    print("Running Poisson surface reconstruction experiments")
    # scan_n_points(N_POINTS, linear_fit=True)
    # scan_depth(DEPTHS, 1024, linear_fit=True)

    n_eval_points = 500
    n_train_points = [100, 10_000]
    distances_1 = reconstruction_experiment(n_train_points[0], n_eval_points, depth=9)
    distances_2 = reconstruction_experiment(n_train_points[1], n_eval_points, depth=9)
    distances = [distances_1, distances_2]

    n_histogram_bins = 50
    idx_to_dist_type = {
        0: "unsigned distance", 1: "signed distance", 2: "occupancy guess"
    }

    fig, ax = plt.subplots(2, 2, sharex=True)
    for i in range(2):
        for j in range(2):
            ax[i, j].hist(distances[j][:, i + 1], n_histogram_bins)
            ax[i, j].set_xlabel(idx_to_dist_type[i + 1])
            ax[i, j].set_ylabel("Frequency")
            ax[i, j].set_title(
                f"Error distribution with {n_train_points[j]}training points"
            )

    plt.tight_layout()
    plt.show()
