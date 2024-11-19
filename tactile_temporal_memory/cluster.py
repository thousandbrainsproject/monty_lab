# Copyright 2023 Numenta Inc.
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
from scipy.cluster.hierarchy import fcluster, linkage


def cluster_type(num_clusters, cluster_by_coord, cluster_by_curve):
    cluster_dir = ""
    cluster_split_fn = None

    def coord_curve_split(coord, curve):
        return np.hstack((curve.reshape(-1, 1), coord))

    def coord_split(coord, curve):
        return coord

    def curve_split(coord, curve):
        return curve.reshape(-1, 1)

    if cluster_by_coord and cluster_by_curve:
        cluster_dir = "coord_curve_k={0}".format(num_clusters)
        cluster_split_fn = coord_curve_split
    elif cluster_by_coord:
        cluster_dir = "coord_k={0}".format(num_clusters)
        cluster_split_fn = coord_split
    elif cluster_by_curve:
        cluster_dir = "curve_k={0}".format(num_clusters)
        cluster_split_fn = curve_split
    else:
        raise Exception("Need to cluster by either coordinates or curvatures or both!")

    return cluster_dir, cluster_split_fn


if __name__ == "__main__":
    """ "
    args parser
    """

    parser = argparse.ArgumentParser(description="Generate clusters for training.")

    parser.add_argument(
        "-sdr_p",
        type=str,
        default=f"~/tbp/tbp.monty/projects/tactile_temporal_memory/tm_dataset",
        help="Enter SDR_YCBMeshDataset relative path in the form of: ~/path/of/dataset",
    )
    parser.add_argument(
        "-n", type=int, nargs="?", default=100, help="Number of clusters to generate."
    )
    parser.add_argument(
        "-coord",
        type=str,
        nargs="?",
        choices=("True", "False"),
        default="True",
        help="If True, then (also) cluster by coordinates.",
    )
    parser.add_argument(
        "-curve",
        type=str,
        nargs="?",
        choices=("True", "False"),
        default="True",
        help="If True, then (also) cluster by curvatures.",
    )

    args = parser.parse_args()

    sdr_dataset_path = os.path.expanduser(args.sdr_p)
    num_clusters = args.n
    cluster_by_coord = eval(args.coord)
    cluster_by_curve = eval(args.curve)

    coordinate_dir = os.path.join(sdr_dataset_path, "coordinate_data")
    curvature_dir = os.path.join(sdr_dataset_path, "curvature_data")

    # get processed coordinates and curvatures
    processed_coord_file = os.path.join(coordinate_dir, "processed_coordinate_data.pkl")
    processed_curve_file = os.path.join(curvature_dir, "processed_curvature_data.pkl")

    if not os.path.exists(processed_coord_file) or not os.path.exists(
        processed_curve_file
    ):
        raise Exception(
            "Missing files. Please run `python process_data.py -sdr_p "
            "{0} -ycb_p <YCB objects relative path>`".format(args.sdr_p)
        )

    with open(processed_coord_file, "rb") as f:
        coordinates = pickle.load(f)
    with open(processed_curve_file, "rb") as f:
        curvatures = pickle.load(f)

    cluster_dir, cluster_split_fn = cluster_type(
        num_clusters=num_clusters,
        cluster_by_coord=cluster_by_coord,
        cluster_by_curve=cluster_by_curve,
    )

    cluster_dir = os.path.join(curvature_dir, cluster_dir)
    os.makedirs(cluster_dir, exist_ok=True)

    for d in range(len(coordinates)):
        train_path = os.path.join(cluster_dir, "train{0}.npy".format(d))
        test_path = os.path.join(cluster_dir, "test{0}.npy".format(d))

        int_cloud = coordinates[d]
        int_curvatures = curvatures[d]

        cluster_data = cluster_split_fn(int_cloud, int_curvatures)

        train_curvatures_ind = []

        # cluster curvatures into clusters. pick the curvature closest to the mean
        # of each cluster.
        z = linkage(cluster_data, "ward")
        clusters = fcluster(z, num_clusters, criterion="maxclust")

        # find nearest point to the mean of each cluster and use those points for
        # training
        for k in range(1, num_clusters + 1):
            mask = clusters == k

            curvature_values = int_curvatures.astype(np.float64)
            curvature_values[~mask] = np.inf
            curvature_values[mask] = abs(
                curvature_values[mask] - curvature_values[mask].mean()
            )

            train_curvatures_ind.append(curvature_values.argmin())

        # curvatures for training
        train_curvatures_ind = np.array(sorted(train_curvatures_ind))

        # curvatures for testing
        test_curvatures_ind = np.setdiff1d(
            np.arange(int_curvatures.shape[0]), train_curvatures_ind
        )

        np.save(train_path, train_curvatures_ind)
        np.save(test_path, test_curvatures_ind)
        print("saved")
