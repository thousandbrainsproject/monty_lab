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
from sklearn.neighbors import KDTree

if __name__ == "__main__":
    """"
    args parser
    """
    parser = argparse.ArgumentParser(
        description="Generate points along somewhat continuous paths for training."
    )

    parser.add_argument(
        "-sdr_p",
        type=str,
        default="~/tbp/tbp.monty/projects/grid_cells/grid_dataset",
        help="Enter SDR_YCBMeshDataset relative path in the form of: ~/path/of/dataset",
    )
    parser.add_argument(
        "-objects", type=int, nargs="+",
        help="Which objects to generate paths for."
    )
    parser.add_argument(
        "-num_paths", type=int, nargs="?", default=50,
        help="Number of paths to generate to generate for training and testing."
    )
    parser.add_argument(
        "-path_size", type=int, nargs="?", default=10,
        help="Length of each path to generate for training and testing."
    )

    args = parser.parse_args()

    sdr_dataset_path = os.path.expanduser(args.sdr_p)
    objects = list(args.objects)
    num_paths = args.num_paths
    path_size = args.path_size

    coordinate_dir = os.path.join(sdr_dataset_path, "coordinate_data")
    curvature_dir = os.path.join(sdr_dataset_path, "curvature_data")

    # get processed coordinates and curvatures
    processed_coord_file = os.path.join(coordinate_dir, "processed_coordinate_data.pkl")
    processed_curve_file = os.path.join(curvature_dir, "processed_curvature_data.pkl")

    if not os.path.exists(
        processed_coord_file
    ) or not os.path.exists(
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

    paths_dir = os.path.join(
        curvature_dir,
        "num_paths={0},path_size={1}".format(num_paths, path_size)
    )
    os.makedirs(paths_dir, exist_ok=True)

    for mode in ["train", "test"]:
        for object_id in objects:
            mode_path = os.path.join(paths_dir, "{0}{1}.npy".format(mode, object_id))

            if not os.path.exists(mode_path):
                int_object = np.array(coordinates[object_id])

                path_indices = []

                # create KD Tree of scaled coordinates of object
                tree = KDTree(int_object, metric="l2")

                path_counter = 0
                while path_counter < num_paths:
                    start_point_index = np.random.choice(int_object.shape[0])

                    seen_points = [start_point_index]

                    for _ in range(path_size - 1):
                        closest_dists, closest_inds = tree.query(
                            int_object[seen_points[-1]].reshape(1, -1),
                            k=50
                        )

                        closest_inds = closest_inds.squeeze()[
                            (closest_dists.squeeze() > 0)
                            & (closest_dists.squeeze() <= 5)
                        ]

                        closest_inds = np.setdiff1d(list(closest_inds), seen_points)

                        if len(closest_inds):
                            next_point_index = closest_inds[0]

                            seen_points.append(next_point_index)

                    # if you have collected the requested path size, continue to next
                    # path.
                    # otherwise, restart the path collection process.
                    if len(seen_points) == path_size:
                        path_counter += 1

                        path_indices.append(seen_points)

                np.save(mode_path, path_indices)
