# Copyright 2025 Thousand Brains Project
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
import torch


def float_to_int(values, decimals):
    values = np.round(values, decimals)

    return (values * (10**decimals)).astype(np.int64)


if __name__ == "__main__":
    """ "
    args parser
    """

    parser = argparse.ArgumentParser(
        description="Generate processed data from YCBMeshDataset."
    )

    parser.add_argument(
        "-sdr_p",
        type=str,
        default="~/tbp/tbp.monty/projects/tactile_temporal_memory/tm_dataset",
        help="Enter SDR_YCBMeshDataset relative path in the form of: ~/path/of/dataset",
    )
    parser.add_argument(
        "-ycb_p",
        type=str,
        default="~/tbp/data/habitat/objects/ycb",
        help="Enter relative path of YCB objects in the form of: ~/path/of/dataset",
    )

    parser.add_argument(
        "-num_objects",
        type=int,
        default=10,
        help="Enter the number of objects to process their data",
    )

    args = parser.parse_args()

    sdr_dataset_path = os.path.expanduser(args.sdr_p)
    ycbmesh_dataset_path = os.path.expanduser(args.ycb_p)
    num_objects = args.num_objects

    coordinate_data_dir = os.path.join(sdr_dataset_path, "coordinate_data")
    curvature_data_dir = os.path.join(sdr_dataset_path, "curvature_data")
    os.makedirs(coordinate_data_dir, exist_ok=True)
    os.makedirs(curvature_data_dir, exist_ok=True)

    coordinate_pkl_path = os.path.join(
        coordinate_data_dir, "processed_coordinate_data.pkl"
    )
    curvature_pkl_path = os.path.join(
        curvature_data_dir, "processed_curvature_data.pkl"
    )

    # if YCBMesh dataset has not already been processed, process it
    if not (os.path.exists(coordinate_pkl_path) and os.path.exists(curvature_pkl_path)):
        print("Overwriting previous processed data!")

    # scale the dataset to the same range (0 to 100)
    min_value, max_value = 0, 100

    coordinates = []
    curvatures = []
    point_array = []
    curv_array = []

    for object_id in range(num_objects):
        object_dataset = torch.load(
            os.path.expanduser(
                f"~/tbp/results/monty/projects/feature_eval_runs/logs/explore_touch/observations{object_id}.pt"  # noqa: E501
            )
        )
        print(object_dataset[0]["object"])
        for i in range(len(object_dataset)):
            if object_dataset[i]["features"]["on_object"] == 0:
                continue
            point_array.append(object_dataset[i]["location"])
            curv_array.append(object_dataset[i]["features"]["gaussian_curvature"])

    point_array = np.array(point_array)
    curv_array = np.array(curv_array)

    points_per_object = len(point_array) / num_objects

    lower_bound = np.min(point_array, axis=0)
    upper_bound = np.max(point_array, axis=0)

    lower_bound_curv = np.percentile(curv_array, 90)
    upper_bound_curv = np.percentile(curv_array, 10)

    for object_id in range(num_objects):
        object_dataset = torch.load(
            os.path.expanduser(
                f"~/tbp/results/monty/projects/feature_eval_runs/logs/explore_touch/observations{object_id}.pt"  # noqa: E501
            )
        )

        # make an array out of the locations {every fourth one}
        point_array = []
        curv_array = []
        for i in range(len(object_dataset)):
            if object_dataset[i]["features"]["on_object"] == 0:
                continue
            point_array.append(object_dataset[i]["location"])
            curv_array.append(object_dataset[i]["features"]["gaussian_curvature"])

        point_array = np.array(point_array)
        curv_array = np.array(curv_array)

        # scale the arrays
        point_array = (point_array - lower_bound) * (max_value - min_value) / (
            upper_bound - lower_bound
        ) + min_value
        curv_array = (curv_array - lower_bound_curv) * (max_value - min_value) / (
            upper_bound_curv - lower_bound_curv
        ) + min_value
        curv_array[curv_array < min_value] = min_value
        curv_array[curv_array > max_value] = max_value

        # add the arrays to the list
        coordinates.append(float_to_int(point_array, decimals=0))
        curvatures.append(float_to_int(curv_array, decimals=0))

    with open(coordinate_pkl_path, "wb") as f:
        pickle.dump(coordinates, f)
    with open(curvature_pkl_path, "wb") as f:
        pickle.dump(curvatures, f)

    norm_parameters_path = os.path.join(sdr_dataset_path, "norm_parameters.pkl")
    norm_parameters = {
        "upper_bound_loc": upper_bound,
        "lower_bound_loc": lower_bound,
        "upper_bound_curv": upper_bound_curv,
        "lower_bound_curv": lower_bound_curv,
        "min_value": min_value,
        "max_value": max_value,
    }
    with open(norm_parameters_path, "wb") as f:
        pickle.dump(norm_parameters, f)

    print(
        f"Processed data, for {num_objects} objects,"
        + f"with {points_per_object} points per object"
    )
