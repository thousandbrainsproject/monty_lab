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

if __name__ == "__main__":
    """"
    args parser
    """
    parser = argparse.ArgumentParser(
        description="Generate uniformly random points for training."
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
        "-num_points", type=int, nargs="?", default=500,
        help="Number of uniformly random points to generate for training and testing."
    )

    args = parser.parse_args()

    sdr_dataset_path = os.path.expanduser(args.sdr_p)
    objects = list(args.objects)
    num_points = args.num_points

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

    points_dir = os.path.join(
        curvature_dir,
        "num_points={0}".format(num_points)
    )
    os.makedirs(points_dir, exist_ok=True)

    for mode in ["train", "test"]:
        for object_id in objects:
            mode_path = os.path.join(points_dir, "{0}{1}.npy".format(mode, object_id))

            if not os.path.exists(mode_path):
                int_object = np.array(coordinates[object_id])

                point_indices = np.random.choice(
                    int_object.shape[0],
                    size=num_points,
                    replace=False
                )

                np.save(mode_path, point_indices)
