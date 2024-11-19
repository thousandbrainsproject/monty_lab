# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Add this to frameworks/datasets/ycb.py

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class YCBMeshSDRDataset(Dataset):
    def __init__(
        self,
        root,
        root_test,
        curve_hash_radius,
        coord_hash_radius,
        num_clusters,
        cluster_by_coord,
        cluster_by_curve,
        test_data_size,
        occluded,
        deterministic_inference,
        train,
        curve_n="total num of bits",
        coord_n="total num of bits",
        curve_w="num of on bits",
        coord_w="num of on bits",
    ):
        """A YCBMeshSDRDataset super class that has been converted to SDRs.

        Use the sub-classes that are defined over a specific set of objects.

        What is the curve_hash_radius and coord_hash_radius? In order to use
        coordinates and curvatures, each point (a 3D coordinate or a 1D
        curvature) needs to be translated into SDRs that have overlap if the original
        points are close by. The hash_radius defines a cube around each point -- this
        cube defines all the "neighbors" of each point. Then, a random but deterministic
        process determines which "neighbors" to consider for a particular point. These
        neighbors are then hashed into bins/buckets/indices of an SDR.
        Thus, the hash_radius determines how many possible neighbors of a point to
        consider.

        Args:
            root: path to YCBMeshSDRDataset
            root_test: in case the training dataset and test dataset are at different
                file locs, this should be the path to the test set. If not, set same as
                root.
            curve_hash_radius: hash radius for curvatures
            coord_hash_radius: hash radius for coordinates
            num_clusters: number of clusters for training
            cluster_by_coord: whether to cluster by coordinates
            cluster_by_curve: whether to cluster by curvatures
            test_data_size: how many test samples to retrieve
            occluded: whether the test set is an occluded subset of the object
            deterministic_inference: whether the testing set data is seeded
            train: training set or testing set
            curve_n: "number of bits"
            coord_n: "number of bits"
            curve_w: "number of on bits"
            coord_w: "number of on bits"
        """
        self.root_train = os.path.expanduser(root)
        self.root_test = os.path.expanduser(root_test)
        self.root = self.root_train
        self.curve_hash_radius = curve_hash_radius
        self.coord_hash_radius = coord_hash_radius
        self.coord_n = coord_n
        self.curve_n = curve_n
        self.coord_w = coord_w
        self.curve_w = curve_w
        self.num_clusters = num_clusters
        self.cluster_by_coord = cluster_by_coord
        self.cluster_by_curve = cluster_by_curve
        self.test_data_size = test_data_size
        self.occluded = occluded
        self.deterministic_inference = deterministic_inference
        self.train = train

        self.validate_data()

    def validate_data(self):
        # ----------------------check existence of processed data----------------------#
        self.processed_coord_file = os.path.join(
            self.root, "coordinate_data", "processed_coordinate_data.pkl"
        )
        self.processed_curve_file = os.path.join(
            self.root, "curvature_data", "processed_curvature_data.pkl"
        )

        if not os.path.exists(self.processed_coord_file) or not os.path.exists(
            self.processed_curve_file
        ):
            raise Exception(
                "Missing files. Please run "
                "`python ~/tbp/tbp.monty/projects/temporal_memory/process_data.py "
                "-sdr_p {0} -ycb_p <YCB objects relative path>`".format(self.root)
            )

        with open(self.processed_coord_file, "rb") as f:
            self.processed_coord_data = pickle.load(f)
        with open(self.processed_curve_file, "rb") as f:
            self.processed_curve_data = pickle.load(f)

        # -----------check existence of coordinate/curvature hashes and SDRs-----------#
        self.coordinate_dir = os.path.join(
            self.root,
            "coordinate_data",
            "hash_radius={0}".format(self.coord_hash_radius),
        )
        self.curvature_dir = os.path.join(
            self.root,
            "curvature_data",
            "hash_radius={0}".format(self.curve_hash_radius),
        )

        self.coordinate_files = []
        for _, _, files in os.walk(self.coordinate_dir):
            self.coordinate_files += files

        self.curvature_files = []
        for _, _, files in os.walk(self.curvature_dir):
            self.curvature_files += files

        for object_id in self.object_ids:
            if ("sdr{0}.npy".format(object_id) not in self.coordinate_files) or (
                "orders{0}.npy".format(object_id) not in self.coordinate_files
            ):
                raise Exception(
                    "Missing files. Please run `python "
                    "~/tbp/tbp.monty/projects/temporal_memory/coordinate_encoder.py "
                    "-sdr_p {0} -r {1} -d1 {2} -d2 {3} "
                    "-n {4} -w {5}`".format(
                        self.root,
                        self.coord_hash_radius,
                        min(self.object_ids),
                        max(self.object_ids) + 1,
                        self.coord_n,
                        self.coord_w,
                    )
                )

            if ("sdr{0}.npy".format(object_id) not in self.curvature_files) or (
                "orders{0}.npy".format(object_id) not in self.curvature_files
            ):
                raise Exception(
                    "Missing files. Please run `python "
                    "~/tbp/tbp.monty/projects/temporal_memory/curvature_encoder.py "
                    "-sdr_p {0} -r {1} -d1 {2} -d2 {3} "
                    "-n {4} -w {5}`".format(
                        self.root,
                        self.curve_hash_radius,
                        min(self.object_ids),
                        max(self.object_ids) + 1,
                        self.curve_n,
                        self.curve_w,
                    )
                )

        # -----------------check existence of cluster train/test masks-----------------#
        cluster_dir = ""
        if self.cluster_by_coord:
            cluster_dir += "coord_"
        if self.cluster_by_curve:
            cluster_dir += "curve_"
        cluster_dir += "k={0}".format(self.num_clusters)

        self.cluster_dir = os.path.join(self.root, "curvature_data", cluster_dir)

        self.cluster_files = []
        for _, _, files in os.walk(self.cluster_dir):
            self.cluster_files += files

        for object_id in self.object_ids:
            if ("train{0}.npy".format(object_id) not in self.cluster_files) or (
                "test{0}.npy".format(object_id) not in self.cluster_files
            ):
                raise Exception(
                    "Missing files. Please run `python "
                    "~/tbp/tbp.monty/projects/temporal_memory/cluster.py -sdr_p "
                    "{0} -n {1} -coord {2} -curve {3}`".format(
                        self.root,
                        self.num_clusters,
                        self.cluster_by_coord,
                        self.cluster_by_curve,
                    )
                )

    def __getitem__(self, idx):

        curvatures_list = []
        coordinates_list = []

        if self.train:
            curve_sdr, coord_sdr, train_mask, test_mask = self.get_data(idx=idx)

            curvatures = curve_sdr[train_mask, :]
            coordinates = coord_sdr[train_mask, :]

            for ind in range(curvatures.shape[0]):
                curvatures_list.append(np.where(curvatures[ind] != 0)[0])
                coordinates_list.append(np.where(coordinates[ind] != 0)[0])
        else:
            # Switch to the test data set
            self.root = self.root_test
            self.validate_data()

            curve_sdr, coord_sdr, train_mask, test_mask = self.get_data(idx=idx)

            curvatures = curve_sdr[test_mask, :]
            coordinates = coord_sdr[test_mask, :]

            assert self.test_data_size <= curvatures.shape[0]

            test_samples = self.calculate_test_points(idx=idx, test_mask=test_mask)

            for ind in sorted(test_samples):
                curvatures_list.append(np.where(curvatures[ind] != 0)[0])
                coordinates_list.append(np.where(coordinates[ind] != 0)[0])

        return idx, coordinates_list, curvatures_list

    def get_data(self, idx):
        curve_sdr_file = "sdr{0}.npy".format(idx)
        coord_sdr_file = "sdr{0}.npy".format(idx)
        train_mask_file = "train{0}.npy".format(idx)
        test_mask_file = "test{0}.npy".format(idx)

        if not (
            (curve_sdr_file in self.curvature_files)
            and (coord_sdr_file in self.coordinate_files)
            and (train_mask_file in self.cluster_files)
            and (test_mask_file in self.cluster_files)
        ):
            raise IndexError(
                "Index out of range. "
                "Enter an index in the range [{0}, {1}].".format(
                    min(self.object_ids), max(self.object_ids)
                )
            )

        curve_sdr = np.load(os.path.join(self.curvature_dir, curve_sdr_file))
        coord_sdr = np.load(os.path.join(self.coordinate_dir, coord_sdr_file))
        train_mask = np.load(os.path.join(self.cluster_dir, train_mask_file))
        test_mask = np.load(os.path.join(self.cluster_dir, test_mask_file))

        assert curve_sdr.shape[0] == coord_sdr.shape[0]

        return curve_sdr, coord_sdr, train_mask, test_mask

    def calculate_test_points(self, idx, test_mask):
        point_cloud = self.processed_coord_data[idx][test_mask, :]

        if self.deterministic_inference:
            seed = idx
        else:
            seed = torch.random.seed()

        if self.occluded:
            point_cloud = torch.from_numpy(point_cloud)

            coeffs = torch.rand(3, generator=torch.manual_seed(seed))
            coeffs = coeffs * 2.0 - 1.0
            inner_prods = (point_cloud * coeffs.unsqueeze(0)).sum(dim=1)

            _, inds = torch.topk(inner_prods, k=self.test_data_size)

            return inds.tolist()
        else:
            every_other = 4
            return list(
                np.arange(
                    len(test_mask) - self.test_data_size * every_other, len(test_mask)
                )[::every_other]
            )
            return (
                np.random.default_rng(seed)
                .choice(point_cloud.shape[0], self.test_data_size, replace=False)
                .tolist()
            )

    def show_object(self, idx):
        self.root = self.root_train
        self.validate_data()
        train_coordinates_all = self.processed_coord_data[idx]
        _, _, train_mask, _ = self.get_data(idx)
        train_coordinates = train_coordinates_all[train_mask, :]

        self.root = self.root_test
        self.validate_data()
        test_coordinates_all = self.processed_coord_data[idx]
        _, _, _, test_mask = self.get_data(idx)
        test_coordinates = test_coordinates_all[test_mask, :][
            self.calculate_test_points(idx=idx, test_mask=test_mask)
        ]

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")

        fig.suptitle(
            "Object: {0}\n\n# of Training Clusters: {1}\nCluster By Location: {2}\n"
            "Cluster by Curvature: {3}\n\n# of Test Samples: {4}\n"
            "Occluded Test Samples: {5}".format(
                idx,
                self.num_clusters,
                self.cluster_by_coord,
                self.cluster_by_curve,
                self.test_data_size,
                self.occluded,
            )
        )
        ax.scatter3D(
            train_coordinates_all[:, 0],
            train_coordinates_all[:, 1],
            train_coordinates_all[:, 2],
            color="lightskyblue",
            alpha=0.2,
            label="all points",
        )
        ax.scatter3D(
            train_coordinates[:, 0],
            train_coordinates[:, 1],
            train_coordinates[:, 2],
            color="darkmagenta",
            alpha=1.0,
            label="training points",
        )
        ax.scatter3D(
            test_coordinates[:, 0],
            test_coordinates[:, 1],
            test_coordinates[:, 2],
            color="darkgreen",
            alpha=1.0,
            label="testing points",
        )
        ax.legend()
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.set_zlim([0, 100])

        fig.savefig(
            os.path.join(os.path.expanduser("~/tbp/results"), f"pointcloud-{idx}.png")
        )

    def __len__(self):
        return len(self.object_ids)


class YCBMeshSDRDataset10(YCBMeshSDRDataset):
    """Small version of YCBMeshSDRDataset."""

    def __init__(self, *args, **kwargs):
        self.object_ids = list(range(0, 10))

        super().__init__(*args, **kwargs)


class YCBMeshSDRDataset30(YCBMeshSDRDataset):
    """Medium version of YCBMeshSDRDataset."""

    def __init__(self, *args, **kwargs):
        self.object_ids = list(range(0, 30))

        super().__init__(*args, **kwargs)


class YCBMeshSDRDataset50(YCBMeshSDRDataset):
    """Large version of YCBMeshSDRDataset."""

    def __init__(self, *args, **kwargs):
        self.object_ids = list(range(0, 50))

        super().__init__(*args, **kwargs)


class YCBMeshSDRDataset70(YCBMeshSDRDataset):
    """Extra large version of YCBMeshSDRDataset."""

    def __init__(self, *args, **kwargs):
        self.object_ids = list(range(0, 70))

        super().__init__(*args, **kwargs)


class YCBMeshGridCellDataset(Dataset):
    def __init__(
        self,
        root,
        train,
        objects,
        curve_hash_radius,
        path_info=None,
        num_points=None,
    ):
        """A YCBMeshGridCellDataset class that has been converted to SDRs.

        Use the sub-classes that are defined over a specific set of objects.

        What is the curve_hash_radius? In order to use
        coordinates and curvatures, each point (a 3D coordinate or a 1D
        curvature) needs to be translated into SDRs that have overlap if the original
        points are close by. The hash_radius defines a cube around each point -- this
        cube defines all the "neighbors" of each point. Then, a random but deterministic
        process determines which "neighbors" to consider for a particular point. These
        neighbors are then hashed into bins/buckets/indices of an SDR.
        Thus, the hash_radius determines how many possible neighbors of a point to
        consider.

        Args:
            root: path to YCBMeshGridCellDataset
            train: training set or testing set
            objects: list of objects
            curve_hash_radius: hash radius for curvatures
            path_info: use somewhat continuous points along a path for training and
                testing. This parameter is a tuple specified as (num_paths, path_size)
            num_points: use uniformly but randomly distributed points for training

        Raises:
            Exception: If neither path_info nor num_points is given
        """
        self.root = os.path.expanduser(root)

        # need to use either continuous path data or uniform random sampling of points
        if path_info is not None:
            self.data_mode = "path"
            assert isinstance(path_info, tuple) and len(path_info) == 2

            self.num_paths, self.path_size = path_info
        elif num_points is not None:
            self.data_mode = "random"

            self.num_points = num_points
        else:
            raise Exception(
                "Must specify either path_info=(num_paths, path_size) or num_points."
            )

        # objects must a list or tuple of object IDs
        assert isinstance(objects, list) or isinstance(objects, tuple)
        assert min(objects) >= 0 and max(objects) <= 77

        self.objects = objects
        self.curve_hash_radius = curve_hash_radius
        self.train = train

        self.validate_data()

    def validate_data(self):
        # ----------------------check existence of processed data----------------------#
        self.processed_coord_file = os.path.join(
            self.root, "coordinate_data", "processed_coordinate_data.pkl"
        )
        self.processed_curve_file = os.path.join(
            self.root, "curvature_data", "processed_curvature_data.pkl"
        )

        if not os.path.exists(self.processed_coord_file) or not os.path.exists(
            self.processed_curve_file
        ):
            raise Exception(
                "Missing files. Please run "
                "`python ~/tbp/tbp.monty/projects/grid_cells/process_data.py "
                "-sdr_p {0} -ycb_p <YCB objects relative path>`".format(self.root)
            )

        with open(self.processed_coord_file, "rb") as f:
            self.processed_coord_data = pickle.load(f)
        with open(self.processed_curve_file, "rb") as f:
            self.processed_curve_data = pickle.load(f)

        # ----------------check existence of curvature hashes and SDRs-----------------#
        self.curvature_dir = os.path.join(
            self.root,
            "curvature_data",
            "hash_radius={0}".format(self.curve_hash_radius),
        )

        self.curvature_files = []
        for _, _, files in os.walk(self.curvature_dir):
            self.curvature_files += files

        for object_id in self.objects:
            if ("sdr{0}.npy".format(object_id) not in self.curvature_files) or (
                "orders{0}.npy".format(object_id) not in self.curvature_files
            ):
                raise Exception(
                    "Missing files. Please run `python "
                    "~/tbp/tbp.monty/projects/grid_cells/curvature_encoder.py "
                    "-sdr_p {0} -r {1} -objects {2} "
                    "-n <Size of SDR> -w <# of on bits>`".format(
                        self.root,
                        self.curve_hash_radius,
                        " ".join(list(map(str, self.objects))),
                    )
                )

        # -----------------check existence of data train/test masks--------------------#
        data_dir = ""
        if self.data_mode == "path":
            data_dir += "num_paths={0},path_size={1}".format(
                self.num_paths, self.path_size
            )
        else:
            data_dir += "num_points={0}".format(self.num_points)

        self.data_dir = os.path.join(self.root, "curvature_data", data_dir)

        self.data_files = []
        for _, _, files in os.walk(self.data_dir):
            self.data_files += files

        for object_id in self.objects:
            if ("train{0}.npy".format(object_id) not in self.data_files) or (
                "test{0}.npy".format(object_id) not in self.data_files
            ):
                if self.data_mode == "path":
                    raise Exception(
                        "Missing files. Please run `python "
                        "~/tbp/tbp.monty/projects/grid_cells/generate_paths.py "
                        "-sdr_p {0} -objects {1} -num_paths {2} -path_size {3}".format(
                            self.root,
                            " ".join(list(map(str, self.objects))),
                            self.num_paths,
                            self.path_size,
                        )
                    )
                elif self.data_mode == "random":
                    raise Exception(
                        "Missing files. Please run `python "
                        "~/tbp/tbp.monty/projects/grid_cells/generate_random.py "
                        "-sdr_p {0} -objects {1} -num_points {2}".format(
                            self.root,
                            " ".join(list(map(str, self.objects))),
                            self.num_points,
                        )
                    )

    def __getitem__(self, idx):
        curve_sdr, train_mask, test_mask = self.get_data(idx)

        mask = train_mask if self.train else test_mask

        curvatures = curve_sdr[mask, :]
        coordinates = np.array(self.processed_coord_data[idx][mask, :])

        curvatures_list = []
        for ind in range(curvatures.shape[0]):
            curvatures_list.append(np.where(curvatures[ind] != 0)[0])

        return idx, curvatures_list, coordinates

    def get_data(self, idx):
        curve_sdr_file = "sdr{0}.npy".format(idx)
        train_mask_file = "train{0}.npy".format(idx)
        test_mask_file = "test{0}.npy".format(idx)

        if not (
            (curve_sdr_file in self.curvature_files)
            and (train_mask_file in self.data_files)
            and (test_mask_file in self.data_files)
        ):
            raise IndexError(
                "Index out of range. "
                "Enter an index in the list [{0}]".format(
                    " ".join(list(map(str, self.objects)))
                )
            )

        curve_sdr = np.load(os.path.join(self.curvature_dir, curve_sdr_file))
        train_mask = np.load(os.path.join(self.data_dir, train_mask_file))
        test_mask = np.load(os.path.join(self.data_dir, test_mask_file))

        return curve_sdr, train_mask.ravel(), test_mask.ravel()

    def show_object(self, idx):
        _, train_mask, test_mask = self.get_data(idx)

        coordinates = self.processed_coord_data[idx]
        train_coordinates = coordinates[train_mask, :]
        test_coordinates = coordinates[test_mask, :]

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")

        if self.data_mode == "path":
            fig.suptitle(
                "Object: {0}\n\n# of Paths: {1}\n\nPath Size: {2}".format(
                    idx, self.num_paths, self.path_size
                )
            )
        elif self.data_mode == "random":
            fig.suptitle("Object: {0}\n\n# of Points: {1}".format(idx, self.num_points))

        ax.scatter3D(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color="lightskyblue",
            alpha=0.2,
            label="all points",
        )
        ax.scatter3D(
            train_coordinates[:, 0],
            train_coordinates[:, 1],
            train_coordinates[:, 2],
            color="darkgreen",
            alpha=1.0,
            label="training points",
        )
        ax.scatter3D(
            test_coordinates[:, 0],
            test_coordinates[:, 1],
            test_coordinates[:, 2],
            color="darkred",
            alpha=1.0,
            label="testing points",
        )
        ax.legend()

    def __len__(self):
        return len(self.objects)
        return len(self.objects)
