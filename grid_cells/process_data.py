# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import os
import pickle

import numpy as np
import trimesh
from tbp.monty.frameworks.environments.ycb import YCBMeshDataset
from trimesh.curvature import discrete_gaussian_curvature_measure as gaussian_curvature


def float_to_int(values, decimals):
    values = np.round(values, decimals)

    return (values * (10**decimals)).astype(np.int64)


if __name__ == "__main__":
    """"
    args parser
    """
    parser = argparse.ArgumentParser(
        description="Generate processed grid cell data from YCBMeshDataset."
    )

    parser.add_argument(
        "-sdr_p",
        type=str,
        default="~/tbp/tbp.monty/projects/grid_cells/grid_dataset",
        help="Enter SDR_YCBMeshDataset relative path in the form of: ~/path/of/dataset",
    )
    parser.add_argument(
        "-ycb_p",
        type=str,
        default="~/tbp/data/habitat/objects/ycb",
        help="Enter relative path of YCB objects in the form of: ~/path/of/dataset",
    )

    args = parser.parse_args()

    sdr_dataset_path = os.path.expanduser(args.sdr_p)
    ycbmesh_dataset_path = os.path.expanduser(args.ycb_p)

    coordinate_data_dir = os.path.join(sdr_dataset_path, "coordinate_data")
    curvature_data_dir = os.path.join(sdr_dataset_path, "curvature_data")
    os.makedirs(coordinate_data_dir, exist_ok=True)
    os.makedirs(curvature_data_dir, exist_ok=True)

    coordinate_pkl_path = os.path.join(
        coordinate_data_dir,
        "processed_coordinate_data.pkl"
    )
    curvature_pkl_path = os.path.join(
        curvature_data_dir,
        "processed_curvature_data.pkl"
    )

    # if YCBMesh dataset has not already been processed, process it
    if not os.path.exists(curvature_pkl_path):
        dataset = YCBMeshDataset(ycbmesh_dataset_path)

        # scale the dataset to the same range (0 to 100)
        scale_a, scale_b = 0, 100
        curvature_radius = int(0.15 * (scale_b - scale_a))

        scaled_trimesh = []

        coordinates = []
        curvatures = []

        for cloud in dataset[:][0]:
            points = cloud.vertices
            a, b = cloud.bounds[0, :], cloud.bounds[1, :]

            points = (points - a) * (scale_b - scale_a) / (b - a) + scale_a

            scaled_trimesh.append(points)

        for i, cloud in enumerate(scaled_trimesh):
            scaled_trimesh[i] = trimesh.Trimesh(
                float_to_int(cloud, decimals=0),
                dataset[i][0].faces
            )

            # scaled coordinates
            coordinates.append(scaled_trimesh[i].vertices)

        for cloud in scaled_trimesh:
            # scaled curvatures
            c = gaussian_curvature(cloud, cloud.vertices, radius=curvature_radius)
            c = (c - c.min()) * (scale_b - scale_a) / (c.max() - c.min()) + scale_a

            curvatures.append(float_to_int(c, decimals=0))

        with open(coordinate_pkl_path, "wb") as f:
            coordinate_data = pickle.dump(coordinates, f)
        with open(curvature_pkl_path, "wb") as f:
            curvature_data = pickle.dump(curvatures, f)
